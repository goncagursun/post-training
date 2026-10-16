"""Scenario 3: one frozen base, isolated LoRA optimizers, round-robin updates."""
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

from post_training.concurrent_scenario import load_scenario
from post_training.telemetry import RunTelemetry
from post_training.training.sft import format_prompt


def adapter_parameters(model, name):
    parameters = [p for key, p in model.named_parameters()
                  if '.lora_' in key and f'.{name}.' in key]
    if not parameters:
        raise ValueError(f'No LoRA parameters for {name}')
    return parameters


def frozen_digest(model):
    """Hash the single base allocation; adapters are deliberately excluded."""
    import torch
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if '.lora_' not in name:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tokenize_pairs(tokenizer, pairs, max_length):
    rows = []
    for question, answer in pairs:
        prompt = format_prompt(question)
        prompt_ids = tokenizer(prompt)['input_ids']
        ids = tokenizer(prompt + ' ' + answer + tokenizer.eos_token)['input_ids']
        if ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError('Prompt/completion boundary retokenizes unexpectedly')
        if len(ids) > max_length or len(ids) <= len(prompt_ids):
            raise ValueError('Example exceeds max_length or has no completion')
        rows.append({'input_ids': ids,
                     'completion_mask': [0]*len(prompt_ids) + [1]*(len(ids)-len(prompt_ids))})
    if not rows:
        raise ValueError('Empty dataset')
    return rows


def isolated_update(model, name, parameters, optimizer, batch, precision):
    import torch
    model.zero_grad(set_to_none=True)
    model.set_adapter(name)
    expected = {id(p) for p in parameters}
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if actual != expected:
        raise RuntimeError('Active trainable parameters do not match the tenant optimizer')
    model.train()
    context = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
    with context:
        loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite training loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach()), float(norm)


def export_adapter(model, name, destination, tokenizer=None):
    destination = Path(destination)
    temporary = destination.parent / ('.' + destination.name + '.export')
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        model.save_pretrained(str(temporary), selected_adapters=[name])
        # PEFT stores non-default named adapters in a nested directory.
        (temporary / name).rename(destination)
        if tokenizer is not None:
            tokenizer.save_pretrained(str(destination))
    finally:
        shutil.rmtree(temporary)


def run_shared_training(spec_path, output_dir):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    tenants = load_scenario(spec_path)
    shared_keys = ('model_name', 'model_revision', 'device', 'precision')
    config = tenants[0]['config']
    if any(any(t['config'][k] != config[k] for k in shared_keys) for t in tenants):
        raise ValueError('Both tenants must use the same base, revision, device, and precision')
    if any(t['config']['batch_size'] != 1 or t['config']['gradient_accumulation_steps'] != 1
           or t['config']['lora_dropout'] != 0 for t in tenants):
        raise ValueError('Initial shared-base experiment supports batch=1, accumulation=1, LoRA dropout=0')
    if not torch.cuda.is_available() or (config['precision'] == 'bf16' and not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Configured CUDA/BF16 is unavailable')
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    telemetry = RunTelemetry(output, {'scenario': 'shared-base-interleaved', 'pid': os.getpid()})
    torch.cuda.reset_peak_memory_stats()
    report = {'schema_version': 1, 'scenario': 'shared-base-interleaved', 'status': 'RUNNING',
              'pid': os.getpid(), 'base_model_copies': 1, 'scheduler': 'round-robin, complete optimizer updates',
              'simultaneous_kernel_execution_claimed': False, 'database_service_used': False,
              'limitations': ['custom PyTorch loop; different sampler and Trainer overhead from scenarios 1/2',
                             'inspection checkpoints only; coordinated resume is not implemented'], 'jobs': []}
    def save():
        temporary = output / '.report.tmp'
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(output/'report.json')
    save()
    started = time.perf_counter()
    status = 'FAILED'
    try:
        set_seed(config['seed'])
        with telemetry.phase('shared_model_loading', torch.cuda.synchronize):
            tokenizer = AutoTokenizer.from_pretrained(config['model_name'], revision=config['model_revision'])
            if tokenizer.eos_token is None:
                raise ValueError('EOS token required')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'right'
            base = AutoModelForCausalLM.from_pretrained(config['model_name'], revision=config['model_revision'],
                dtype=torch.bfloat16 if config['precision'] == 'bf16' else torch.float32)
            report['resolved_model_revision'] = getattr(base.config, '_commit_hash', None)
            initial_rng = torch.get_rng_state()
            model = None
            for tenant in tenants:
                c = tenant['config']
                torch.set_rng_state(initial_rng)
                lora = LoraConfig(r=c['lora_rank'], lora_alpha=c['lora_alpha'], target_modules=['q_proj','v_proj'],
                                  lora_dropout=0, bias='none', task_type='CAUSAL_LM')
                if model is None:
                    model = get_peft_model(base, lora, adapter_name=tenant['tenant_id'])
                else:
                    model.add_adapter(tenant['tenant_id'], lora)
            model.config.use_cache = False
            model.to('cuda')
            before = frozen_digest(model)
        collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id, completion_only_loss=True)
        states = []
        with telemetry.phase('tenant_setup'):
            for tenant in tenants:
                name, c = tenant['tenant_id'], tenant['config']
                run_dir = output/'customers'/name/'attempts'/'001'
                run_dir.mkdir(parents=True, exist_ok=False)
                (run_dir/'config.json').write_text(json.dumps(c,indent=2)+'\n')
                (run_dir/'dataset.json').write_text(json.dumps(tenant['manifest'],indent=2)+'\n')
                parameters = adapter_parameters(model, name)
                optimizer = torch.optim.AdamW(parameters, lr=c['learning_rate'], weight_decay=0.0,
                                               betas=(0.9,0.999), eps=1e-8)
                rows = tokenize_pairs(tokenizer, tenant['examples'], c['max_length'])
                validation = tokenize_pairs(tokenizer, tenant['validation'], c['max_length'])
                generator = torch.Generator().manual_seed(c['seed'])
                orders = [torch.randperm(len(rows), generator=generator).tolist() for _ in range(c['num_train_epochs'])]
                order = [i for epoch in orders for i in epoch]
                job = {'tenant_id':name, 'run_dir':str(run_dir),'manifest_path':tenant['manifest_path'],
                       'baseline_adapter_dir':tenant['baseline_adapter_dir'],'config':c,'status':'TRAINING',
                       'steps':0,'input_tokens':0,'supervised_tokens':0,'loss_sum':0.0,'active_update_seconds':0.0,
                       'isolation_first_update_verified':False}
                report['jobs'].append(job)
                states.append({'job':job,'rows':rows,'validation':validation,'order':order,'parameters':parameters,
                               'optimizer':optimizer,'checkpoints':[]})
            if {id(p) for p in states[0]['parameters']} & {id(p) for p in states[1]['parameters']}:
                raise RuntimeError('Optimizers share adapter parameters')
        report['optimizer_parameter_sets_disjoint'] = True
        save()
        training_started = time.perf_counter()
        with telemetry.phase('training', torch.cuda.synchronize):
            while any(s['job']['steps'] < len(s['order']) for s in states):
                for state in states:
                    job = state['job']
                    step = job['steps']
                    if step >= len(state['order']):
                        continue
                    other = next(s for s in states if s is not state)
                    snapshot = [p.detach().clone() for p in other['parameters']] if step == 0 else None
                    batch = {k:v.to('cuda') for k,v in collator([state['rows'][state['order'][step]]]).items()}
                    torch.cuda.synchronize()
                    update_started = time.perf_counter()
                    loss, norm = isolated_update(model, job['tenant_id'], state['parameters'],state['optimizer'],batch,config['precision'])
                    torch.cuda.synchronize()
                    job['active_update_seconds'] += time.perf_counter()-update_started
                    if snapshot is not None:
                        if not all(torch.equal(p, saved) for p,saved in zip(other['parameters'],snapshot)):
                            raise RuntimeError('Inactive adapter changed')
                        job['isolation_first_update_verified'] = True
                    job['steps'] += 1
                    job['input_tokens'] += int(batch['attention_mask'].sum())
                    job['supervised_tokens'] += int((batch['labels'][:,1:] != -100).sum())
                    job['loss_sum'] += loss
                    telemetry.event('tenant_update',tenant_id=job['tenant_id'],step=job['steps'],loss=loss,grad_norm=norm)
                    cadence = max(job['config']['save_steps'],100) if len(state['rows']) > 4 else job['config']['save_steps']
                    if job['steps'] % cadence == 0:
                        with telemetry.phase('checkpoint_writing',torch.cuda.synchronize):
                            checkpoint = Path(job['run_dir'])/'checkpoints'/f"checkpoint-{job['steps']}"
                            checkpoint.mkdir(parents=True)
                            export_adapter(model,job['tenant_id'],checkpoint/'adapter')
                            torch.save(state['optimizer'].state_dict(),checkpoint/'optimizer.pt')
                            (checkpoint/'state.json').write_text(json.dumps({'step':job['steps'],'resume_supported':False})+'\n')
                            state['checkpoints'].append(checkpoint)
                            if len(state['checkpoints']) > 2:
                                shutil.rmtree(state['checkpoints'].pop(0))
                    if job['steps'] % 100 == 0:
                        save()
                        print(job['tenant_id'],'step',job['steps'],flush=True)
        report['training_seconds'] = time.perf_counter()-training_started
        with telemetry.phase('base_integrity_check',torch.cuda.synchronize):
            report['frozen_base_unchanged'] = frozen_digest(model) == before
            if not report['frozen_base_unchanged']:
                raise RuntimeError('Frozen base weights changed')
        for state in states:
            job = state['job']
            model.set_adapter(job['tenant_id'])
            model.eval()
            loss_sum, targets = 0.0, 0
            with telemetry.phase('validation',torch.cuda.synchronize), torch.inference_mode():
                for row in state['validation']:
                    batch = {k:v.to('cuda') for k,v in collator([row]).items()}
                    count = int((batch['labels'][:,1:] != -100).sum())
                    context = torch.autocast('cuda',dtype=torch.bfloat16) if config['precision']=='bf16' else nullcontext()
                    with context:
                        loss_sum += float(model(**batch).loss)*count
                    targets += count
            job['validation_loss'] = loss_sum/targets
            with telemetry.phase('adapter_export',torch.cuda.synchronize):
                destination = Path(job['run_dir'])/'adapter'
                export_adapter(model,job['tenant_id'],destination,tokenizer)
            job['adapter_dir'] = str(destination)
            job['training_loss'] = job.pop('loss_sum')/job['steps']
            job['input_tokens_per_shared_training_second'] = job['input_tokens']/report['training_seconds']
            job['samples_per_shared_training_second'] = job['steps']/report['training_seconds']
            job['status'] = 'SUCCEEDED'
            (Path(job['run_dir'])/'result.json').write_text(json.dumps(job,indent=2)+'\n')
        elapsed = time.perf_counter()-started
        report['execution_seconds'] = elapsed
        report['aggregate_input_tokens_per_second'] = sum(j['input_tokens'] for j in report['jobs'])/elapsed
        report['aggregate_samples_per_second'] = sum(j['steps'] for j in report['jobs'])/elapsed
        report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['base_parameter_bytes'] = sum(p.numel()*p.element_size() for name,p in model.named_parameters() if '.lora_' not in name)
        report['status'] = status = 'SUCCEEDED'
    finally:
        report['status'] = status
        report['telemetry'] = telemetry.finish(status)
        save()
    return report
