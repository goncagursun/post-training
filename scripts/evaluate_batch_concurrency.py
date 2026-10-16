"""Untimed full held-out evaluation of every batch/tenant solo reference."""
import argparse
import json
from pathlib import Path

from post_training.bitext import load_manifest, intent_question
from post_training.intent_evaluation import score_predictions
from post_training.training.sft import format_prompt
from run_concurrency_sweep import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', required=True)
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    report = json.loads((root/'report.json').read_text())
    if report['status'] != 'TRAINING_COMPLETE':
        raise ValueError('Timing trials must finish first')
    destination = root/'quality_report.json'
    if destination.exists():
        raise FileExistsError(destination)
    jobs = [(batch, name, job) for batch, group in report['references'].items()
            for name, job in sorted(group.items())]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftConfig, PeftModel
    torch.set_num_threads(1)
    first = jobs[0][2]
    config = first['config']
    tokenizer = AutoTokenizer.from_pretrained(first['adapter_dir'], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(config['model_name'],
        revision=config['model_revision'], dtype=torch.float32)
    model = None
    quality = {'status': 'RUNNING', 'expected_adapters': len(jobs), 'completed_adapters': 0,
        'model_revision': report['model_revision'], 'device': 'cuda', 'dtype': 'float32',
        'decoding': {'do_sample': False, 'max_new_tokens': 16, 'num_beams': 1},
        'scoring': 'exact label after stripping whitespace and lowercasing', 'batches': {},
        'scope': 'Solo references, inherited by concurrent adapters only when exact weight checks pass'}
    atomic_json(destination, quality)
    output = root/'quality'
    output.mkdir(exist_ok=False)
    try:
        for batch, name, job in jobs:
            path = Path(job['adapter_dir'])
            manifest, splits = load_manifest(job['manifest_path'], name)
            if json.loads((path.parent/'dataset.json').read_text()) != manifest:
                raise ValueError('Adapter/evaluation dataset lineage mismatch')
            saved = json.loads((path.parent/'config.json').read_text())
            if saved != job['config'] or saved['model_revision'] != report['model_revision']:
                raise ValueError('Saved recipe or pinned model revision mismatch')
            if PeftConfig.from_pretrained(str(path), local_files_only=True).base_model_name_or_path != config['model_name']:
                raise ValueError('Adapter base model mismatch')
            adapter = f'b{batch}-{name}'
            if model is None:
                model = PeftModel.from_pretrained(base, str(path), adapter_name=adapter,
                    is_trainable=False, local_files_only=True).to('cuda')
            else:
                model.load_adapter(str(path), adapter_name=adapter, is_trainable=False, local_files_only=True)
            model.set_adapter(adapter)
            model.eval()
            model.config.use_cache = True
            predictions = []
            for row in splits['test']:
                inputs = tokenizer(format_prompt(intent_question(row['instruction'])),
                    add_special_tokens=False, return_tensors='pt').to('cuda')
                length = inputs['input_ids'].shape[1]
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                        num_beams=1, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id)
                predictions.append(tokenizer.decode(generated[0, length:], skip_special_tokens=True).strip())
            metrics = score_predictions(splits['test'], predictions)
            result = {'batch_size': int(batch), 'tenant_id': name, 'adapter_dir': str(path),
                'manifest': manifest, 'split': 'test', 'partial': False, 'metrics': metrics,
                'predictions': [{**row, 'adapter_prediction': prediction}
                    for row, prediction in zip(splits['test'], predictions)]}
            if name == 'sweep-tenant-01':
                historical = json.loads((Path(report['prior_batch_counterfactual'])/f'batch-{batch}-test.json').read_text())
                result['historical_prediction_equivalence'] = predictions == [
                    row['adapter_prediction'] for row in historical['predictions']]
            atomic_json(output/f'b{batch}-{name}.json', result)
            if result.get('historical_prediction_equivalence') is False:
                raise RuntimeError('Shared evaluation-model optimization changed historical predictions')
            quality['batches'].setdefault(batch, {})[name] = metrics
            quality['completed_adapters'] += 1
            atomic_json(destination, quality)
            print('QUALITY', batch, name, metrics['accuracy'], flush=True)
        quality['status'] = 'COMPLETE'
    except BaseException as exc:
        quality.update(status='FAILED', error_type=type(exc).__name__, error=str(exc)[:500])
        raise
    finally:
        atomic_json(destination, quality)


if __name__ == '__main__':
    main()
