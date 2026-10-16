"""Batch-size counterfactual engine, derived from the preserved controlled engine.

Only batching, sample-based progress/checkpoint accounting, and padding metrics differ.
The historical controlled engine is intentionally unchanged.
"""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from post_training.shared_base_scenario import adapter_parameters, frozen_digest, tokenize_pairs, isolated_update, export_adapter
from post_training.telemetry import RunTelemetry

def run_batch_worker(tenants, output_dir, before_training=None):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    shared_keys = ('model_name', 'model_revision', 'device', 'precision')
    config = tenants[0]['config']
    if any(any(t['config'][k] != config[k] for k in shared_keys) for t in tenants):
        raise ValueError('Both tenants must use the same base, revision, device, and precision')
    if len(tenants) != 1:
        raise ValueError('Batch counterfactual requires exactly one adapter/job')
    if any(type(t['config']['batch_size']) is not int or t['config']['batch_size'] < 1
           or t['config']['gradient_accumulation_steps'] != 1 or t['config']['lora_dropout'] != 0
           for t in tenants):
        raise ValueError('Positive batch size, accumulation=1, and zero dropout required')
    if not torch.cuda.is_available() or (config['precision'] == 'bf16' and not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Configured CUDA/BF16 is unavailable')
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    telemetry = RunTelemetry(output, {'scenario': 'batch-size-counterfactual', 'pid': os.getpid()})
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    report = {'schema_version': 1, 'scenario': 'batch-size-counterfactual', 'status': 'RUNNING',
              'pid': os.getpid(), 'base_model_copies': 1, 'scheduler': 'one job; contiguous batches in the same seeded sample order',
              'simultaneous_kernel_execution_claimed': False, 'database_service_used': False,
              'limitations': ['inspection checkpoints only; coordinated resume is not implemented'], 'jobs': []}
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
                dtype=torch.bfloat16 if config['precision'] == 'bf16' else torch.float32, attn_implementation='eager')
            report['resolved_model_revision'] = getattr(base.config, '_commit_hash', None)
            if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in base.modules()):
                raise ValueError('Controlled benchmark requires zero model dropout')
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
                initial_adapter_digest = hashlib.sha256(b''.join(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() for p in parameters)).hexdigest()
                rows = tokenize_pairs(tokenizer, tenant['examples'], c['max_length'])
                validation = tokenize_pairs(tokenizer, tenant['validation'], c['max_length'])
                generator = torch.Generator().manual_seed(c['seed'])
                orders = [torch.randperm(len(rows), generator=generator).tolist() for _ in range(c['num_train_epochs'])]
                order = [i for epoch in orders for i in epoch]
                job = {'tenant_id':name, 'run_dir':str(run_dir),'manifest_path':tenant['manifest_path'],
                       'baseline_adapter_dir':tenant['baseline_adapter_dir'],'config':c,'status':'TRAINING','initial_adapter_sha256':initial_adapter_digest,
                       'steps':0,'samples':0,'padded_tokens':0,'input_tokens':0,'supervised_tokens':0,'loss_sum':0.0,'active_update_seconds':0.0,
                       'order_sha256':hashlib.sha256(json.dumps(order).encode()).hexdigest()}
                report['jobs'].append(job)
                states.append({'job':job,'rows':rows,'validation':validation,'order':order,'parameters':parameters,
                               'optimizer':optimizer,'checkpoints':[]})
            parameter_sets = [{id(p) for p in s['parameters']} for s in states]
            if any(a & b for i,a in enumerate(parameter_sets) for b in parameter_sets[i+1:]):
                raise RuntimeError('Optimizers share adapter parameters')
        report['optimizer_parameter_sets_disjoint'] = True
        save()
        if before_training is not None:
            with telemetry.phase('training_start_barrier', torch.cuda.synchronize):
                before_training()
        training_started = time.perf_counter()
        with telemetry.phase('training', torch.cuda.synchronize):
            while any(s['job']['samples'] < len(s['order']) for s in states):
                for state in states:
                    job = state['job']
                    cursor = job['samples']
                    if cursor >= len(state['order']):
                        continue
                    indices = state['order'][cursor:cursor+job['config']['batch_size']]
                    batch = {k:v.to('cuda') for k,v in collator([state['rows'][i] for i in indices]).items()}
                    torch.cuda.synchronize()
                    update_started = time.perf_counter()
                    loss, norm = isolated_update(model, job['tenant_id'], state['parameters'],state['optimizer'],batch,config['precision'])
                    torch.cuda.synchronize()
                    job['active_update_seconds'] += time.perf_counter()-update_started
                    job['steps'] += 1
                    job['samples'] += len(indices)
                    job['padded_tokens'] += batch['input_ids'].numel()
                    job['input_tokens'] += int(batch['attention_mask'].sum())
                    job['supervised_tokens'] += int((batch['labels'][:,1:] != -100).sum())
                    job['loss_sum'] += loss
                    telemetry.event('tenant_update',tenant_id=job['tenant_id'],step=job['steps'],samples=job['samples'],batch_samples=len(indices),loss=loss,grad_norm=norm)
                    cadence = max(job['config']['save_steps'],100) if len(state['rows']) > 4 else job['config']['save_steps']
                    if job['samples']//cadence > cursor//cadence:
                        with telemetry.phase('checkpoint_writing',torch.cuda.synchronize):
                            checkpoint = Path(job['run_dir'])/'checkpoints'/f"checkpoint-{job['steps']}"
                            checkpoint.mkdir(parents=True)
                            export_adapter(model,job['tenant_id'],checkpoint/'adapter')
                            torch.save(state['optimizer'].state_dict(),checkpoint/'optimizer.pt')
                            (checkpoint/'state.json').write_text(json.dumps({'step':job['steps'],'resume_supported':False})+'\n')
                            state['checkpoints'].append(checkpoint)
                            if len(state['checkpoints']) > 2:
                                shutil.rmtree(state['checkpoints'].pop(0))
                    if job['samples']//100 > cursor//100:
                        save()
                        print(job['tenant_id'],'samples',job['samples'],'updates',job['steps'],flush=True)
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
            job['mean_update_training_loss'] = job.pop('loss_sum')/job['steps']
            job['input_tokens_per_shared_training_second'] = job['input_tokens']/report['training_seconds']
            job['samples_per_shared_training_second'] = job['samples']/report['training_seconds']
            job['status'] = 'SUCCEEDED'
            (Path(job['run_dir'])/'result.json').write_text(json.dumps(job,indent=2)+'\n')
        elapsed = time.perf_counter()-started
        report['execution_seconds'] = elapsed
        report['aggregate_input_tokens_per_second'] = sum(j['input_tokens'] for j in report['jobs'])/elapsed
        report['aggregate_samples_per_second'] = sum(j['samples'] for j in report['jobs'])/elapsed
        report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['base_parameter_bytes'] = sum(p.numel()*p.element_size() for name,p in model.named_parameters() if '.lora_' not in name)
        report['status'] = status = 'SUCCEEDED'
    finally:
        report['status'] = status
        report['telemetry'] = telemetry.finish(status)
        save()
    return report
