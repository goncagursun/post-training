"""One-job physical batch-size sweep, preserving the historical concurrency runs."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from post_training.batch_engine import run_batch_worker
from post_training.concurrency_sweep import randomized_plan, SweepSampler, summarize_resources
from post_training.monitoring import query_gpus
from post_training.telemetry import timestamp
from run_concurrency_sweep import atomic_json, prepare_profiles
from verify_controlled_isolation import compare_weights


def worker(profile, output):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = run_batch_worker([profile], output)
    result['torch_intraop_threads'] = torch.get_num_threads()
    result['torch_interop_threads'] = torch.get_num_interop_threads()
    gpu = str(torch.cuda.get_device_properties(0).uuid)
    result['gpu_uuid'] = gpu if gpu.startswith('GPU-') else 'GPU-' + gpu
    return result


def wait_for_idle_gpu(gpu, timeout=20):
    """Exclude prior worker context teardown from the next measurement window."""
    started = time.perf_counter()
    while True:
        query = subprocess.run(['nvidia-smi', '--id', gpu, '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        if not any(line.strip().isdigit() for line in query.stdout.splitlines()):
            return time.perf_counter() - started
        if time.perf_counter() - started >= timeout:
            raise RuntimeError('GPU compute processes remain active; measurement was not started')
        time.sleep(.25)


def run_trial(profile, directory, ray, gpu):
    directory.mkdir(parents=True, exist_ok=False)
    sampler = SweepSampler(directory / 'resource_samples.jsonl')
    result = {'status': 'RUNNING', 'batch_size': profile['config']['batch_size']}
    ref = None
    try:
        result['excluded_gpu_settle_seconds'] = wait_for_idle_gpu(gpu)
        sampler.start()
        task = ray.remote(num_cpus=.5, num_gpus=1, max_calls=1, max_retries=0)(worker)
        start = timestamp()
        tick = time.perf_counter()
        ref = task.remote(profile, str(directory / 'worker'))
        returned = ray.get(ref)
        elapsed = time.perf_counter() - tick
        end = timestamp()
        sampler.stop()
        job = returned['jobs'][0]
        if not returned['frozen_base_unchanged'] or returned['gpu_uuid'] != gpu:
            raise RuntimeError('Base integrity or GPU identity failed')
        if returned['torch_intraop_threads'] != 1 or returned['torch_interop_threads'] != 1:
            raise RuntimeError('Worker CPU thread settings changed')
        if job['samples'] != len(profile['examples']) * profile['config']['num_train_epochs']:
            raise RuntimeError('Sample budget changed')
        if job['steps'] != math.ceil(job['samples'] / profile['config']['batch_size']):
            raise RuntimeError('Optimizer update count does not match physical batch size')
        resources = summarize_resources(directory / 'resource_samples.jsonl', start, end, gpu)
        if resources['gpu_query_errors'] or not resources['gpu']['sample_count']:
            raise RuntimeError('GPU monitoring failed')
        result.update(status='SUCCEEDED', started_at=start, finished_at=end,
            execution_seconds=elapsed, training_seconds=returned['training_seconds'],
            end_to_end_input_tokens_per_second=job['input_tokens']/elapsed,
            training_input_tokens_per_second=job['input_tokens']/returned['training_seconds'],
            padding_efficiency=job['input_tokens']/job['padded_tokens'], job=job,
            frozen_base_unchanged=returned['frozen_base_unchanged'], resources=resources,
            cuda_peak_allocated_bytes=returned['cuda_peak_allocated_bytes'],
            cuda_peak_reserved_bytes=returned['cuda_peak_reserved_bytes'])
    except BaseException as exc:
        result.update(status='FAILED', error_type=type(exc).__name__, error=str(exc)[:500])
        if ref is not None:
            ray.cancel(ref, force=True)
        raise
    finally:
        if sampler.thread.is_alive():
            sampler.stop()
        atomic_json(directory / 'report.json', result)
    return result


def summarize(trials, batches):
    def spread(values):
        return {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
    rows = []
    for batch in batches:
        group = [t for t in trials if t['batch_size'] == batch]
        if not group:
            continue
        row = {'batch_size': batch, 'repeats': len(group),
            'optimizer_updates': sorted({t['job']['steps'] for t in group}),
            'samples': sorted({t['job']['samples'] for t in group}),
            'input_tokens': sorted({t['job']['input_tokens'] for t in group})}
        for key in ('execution_seconds', 'training_seconds', 'end_to_end_input_tokens_per_second',
                    'training_input_tokens_per_second', 'padding_efficiency'):
            row[key] = spread([t[key] for t in group])
        row['validation_loss'] = spread([t['job']['validation_loss'] for t in group])
        row['peak_sampled_vram_gib'] = spread([
            t['resources']['gpu']['maximum']['device_memory_used_bytes']/1024**3 for t in group])
        row['mean_cpu_quota_percent'] = spread([
            t['resources']['host']['mean_cpu_quota_percent'] for t in group])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-sweep', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    plan = randomized_plan(args.batches, args.repeats, 42)
    if 1 not in args.batches:
        parser.error('Include a fresh batch-one control')
    prior_root = Path(args.reference_sweep).resolve()
    prior = json.loads((prior_root / 'report.json').read_text())
    if prior['status'] != 'COMPLETE':
        raise ValueError('Prior sweep must be complete')
    prior_profiles = json.loads((prior_root / 'profiles.json').read_text())
    source = prior_profiles[0]
    historical = prior['references'][source['tenant_id']]
    import post_training.controlled_engine as historical_engine
    if hashlib.sha256(Path(historical_engine.__file__).read_bytes()).hexdigest() != prior['engine_sha256']:
        raise ValueError('Historical controlled engine changed')
    inventory = query_gpus()
    if len(inventory) != 1:
        raise ValueError('Exactly one visible GPU required')
    gpu = inventory[0]['gpu_uuid']
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    profiles, _ = prepare_profiles(source, root, 1, prior['epochs_per_job'], prior['model_revision'])
    profile = profiles[0]
    if profile['config'] != source['config'] or profile['manifest'] != source['manifest']:
        raise ValueError('Counterfactual workload differs from sweep baseline')
    atomic_json(root / 'profiles.json', profiles)
    repo = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), repo/'src/post_training/batch_engine.py',
        repo/'src/post_training/controlled_engine.py', repo/'src/post_training/shared_base_scenario.py',
        repo/'src/post_training/concurrency_sweep.py', repo/'scripts/run_concurrency_sweep.py',
        repo/'src/post_training/intent_evaluation.py']
    snapshot = root / 'source_snapshots'
    snapshot.mkdir()
    hashes = {}
    for path in sources:
        shutil.copyfile(path, snapshot/path.name)
        hashes[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {'status': 'RUNNING', 'plan': plan, 'batches': args.batches, 'repeats': args.repeats,
        'reference_sweep': str(prior_root), 'historical_report_sha256': hashlib.sha256(
            (prior_root/'report.json').read_bytes()).hexdigest(), 'source_sha256': hashes,
        'model_revision': prior['model_revision'], 'thread_settings': prior['thread_settings'],
        'controls': {'sample_budget': len(profile['examples']), 'epochs': prior['epochs_per_job'],
            'seed': profile['config']['seed'], 'learning_rate': profile['config']['learning_rate'],
            'gradient_accumulation_steps': 1, 'checkpoint_cadence':
                'each 100 processed examples, rounded up to the next complete batch',
            'checkpoint_retention': 2, 'test_split_used_for_batch_selection': False},
        'limitations': ['same examples and learning rate, fewer updates at larger batches; not equal optimization trajectories',
            'one adapter is not equivalent to N distinct tenant adapters',
            'historical concurrency timings come from an earlier session',
            'three timing repeats at one fixed seed; not independent training seeds',
            'no learning-rate or epoch tuning; quality results do not establish the best achievable larger-batch model'],
        'trials': [], 'quality': {}}
    atomic_json(root/'protocol.json', {k:v for k,v in report.items() if k not in ('trials','quality','status')})
    atomic_json(root/'report.json', report)
    import ray
    try:
        ray.init(address='local', include_dashboard=False)
        first = {}
        for index, entry in enumerate(plan, 1):
            batch = entry['n']
            current = copy.deepcopy(profile)
            current['config']['batch_size'] = batch
            trial = run_trial(current, root/'trials'/f'{index:02d}-b{batch}-r{entry["repeat"]}', ray, gpu)
            job = trial['job']
            if (job['initial_adapter_sha256'] != historical['initial_adapter_sha256'] or
                job['order_sha256'] != historical['order_sha256'] or
                job['input_tokens'] != historical['input_tokens'] or
                job['supervised_tokens'] != historical['supervised_tokens']):
                raise RuntimeError('Initial state, sample order, or useful token budget changed')
            if batch == 1:
                trial['historical_batch_one_equivalence'] = compare_weights(historical['adapter_dir'], job['adapter_dir'])
                if not trial['historical_batch_one_equivalence']['exact']:
                    raise RuntimeError('Batch-one engine does not reproduce the historical adapter exactly')
            if batch in first:
                trial['repeat_equivalence'] = compare_weights(first[batch]['adapter_dir'], job['adapter_dir'])
                if not trial['repeat_equivalence']['exact']:
                    raise RuntimeError('Repeat adapter weights differ; investigate determinism')
            else:
                first[batch] = job
            trial['repeat'] = entry['repeat']
            report['trials'].append(trial)
            report['summary'] = summarize(report['trials'], args.batches)
            atomic_json(root/'report.json', report)
            print('TRIAL', index, 'batch', batch, 'repeat', entry['repeat'],
                  'seconds', round(trial['execution_seconds'], 2), flush=True)
        report['status'] = 'TRAINING_COMPLETE'
    except BaseException as exc:
        report.update(status='FAILED', error_type=type(exc).__name__, error=str(exc)[:500])
        raise
    finally:
        ray.shutdown()
        atomic_json(root/'report.json', report)
    # Full generated-label test accuracy is measured separately, after every timing run.
    try:
        for batch, job in sorted(first.items()):
            output = root/f'batch-{batch}-test.json'
            with (root/f'batch-{batch}-evaluation.log').open('x') as log:
                subprocess.run([sys.executable, str(repo/'scripts/evaluate_intents.py'),
                    '--manifest', job['manifest_path'], '--split', 'test', '--adapter-dir', job['adapter_dir'],
                    '--device', 'cuda', '--output', str(output)], cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
            evaluation = json.loads(output.read_text())
            report['quality'][str(batch)] = evaluation['adapter']
            report['base_test_quality'] = evaluation['base']
            atomic_json(root/'report.json', report)
            print('QUALITY batch', batch, report['quality'][str(batch)]['accuracy'], flush=True)
        report['status'] = 'COMPLETE'
    except BaseException as exc:
        report.update(status='QUALITY_FAILED', error_type=type(exc).__name__, error=str(exc)[:500])
        raise
    finally:
        atomic_json(root/'report.json', report)
    atomic_json(root/'summary.json', {'batches': report['summary'], 'quality': report['quality']})


if __name__ == '__main__':
    main()
