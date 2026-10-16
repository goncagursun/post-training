"""Controlled physical batch x independent-concurrency experiment, with solo gates."""
import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from post_training.batch_engine import run_batch_worker
from post_training.batch_concurrency import matrix_plan, update_token_counts
from post_training.concurrency_sweep import SweepSampler, summarize_resources, steady_window, summarize_sweep, host_snapshot
from post_training.monitoring import query_gpus
from post_training.telemetry import timestamp
from run_batch_counterfactual import wait_for_idle_gpu
from run_concurrency_sweep import atomic_json, prepare_profiles
from verify_controlled_isolation import compare_weights


class StartGate:
    def __init__(self, count):
        self.count = count
        self.arrivals = 0
        self.event = asyncio.Event()

    async def health(self):
        return True

    async def arrive(self):
        self.arrivals += 1
        if self.arrivals == self.count:
            self.event.set()
        await self.event.wait()
        return self.arrivals


def worker(profile, directory, gate):
    import ray
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = run_batch_worker([profile], directory,
        before_training=lambda: ray.get(gate.arrive.remote(), timeout=180))
    result['torch_intraop_threads'] = torch.get_num_threads()
    result['torch_interop_threads'] = torch.get_num_interop_threads()
    gpu = str(torch.cuda.get_device_properties(0).uuid)
    result['gpu_uuid'] = gpu if gpu.startswith('GPU-') else 'GPU-' + gpu
    return result


def run_trial(profiles, output, ray, gpu, sample_tokens, references=None):
    output.mkdir(parents=True, exist_ok=False)
    sampler = SweepSampler(output/'resource_samples.jsonl')
    batch = profiles[0]['config']['batch_size']
    report = {'n': len(profiles), 'batch_size': batch, 'status': 'RUNNING', 'jobs': []}
    refs = {}
    gate = None
    try:
        report['excluded_gpu_settle_seconds'] = wait_for_idle_gpu(gpu)
        gate = ray.remote(num_cpus=0)(StartGate).remote(len(profiles))
        ray.get(gate.health.remote(), timeout=60)
        task = ray.remote(num_cpus=.5, num_gpus=1/len(profiles), max_calls=1, max_retries=0)(worker)
        sampler.start()
        start_wall = timestamp()
        started = time.perf_counter()
        for index, profile in enumerate(profiles):
            submitted = time.perf_counter()
            directory = output/f'worker-{index:02d}'
            ref = task.remote(profile, str(directory), gate)
            refs[ref] = (profile, directory, submitted)
        while refs:
            if time.perf_counter()-started > 600:
                raise RuntimeError('Trial exceeded the 600-second timeout')
            ready, _ = ray.wait(list(refs), timeout=.1)
            for ref in ready:
                profile, directory, submitted = refs.pop(ref)
                result = ray.get(ref)
                completed = time.perf_counter()
                job = result['jobs'][0]
                expected_samples = len(profile['examples'])*profile['config']['num_train_epochs']
                if job['samples'] != expected_samples or job['steps'] != math.ceil(expected_samples/batch):
                    raise RuntimeError('Sample or optimizer-update budget changed')
                if not result['frozen_base_unchanged'] or result['gpu_uuid'] != gpu:
                    raise RuntimeError('INTEGRITY_FAILED: base or GPU identity')
                if result['torch_intraop_threads'] != 1 or result['torch_interop_threads'] != 1:
                    raise RuntimeError('INTEGRITY_FAILED: worker CPU threads')
                report['jobs'].append({**job, 'worker_dir': str(directory), 'pid': result['pid'],
                    'gpu_uuid': gpu, 'completion_seconds': completed-submitted,
                    'submission_offset_seconds': submitted-started,
                    'training_seconds': result['training_seconds'],
                    'training_tokens_per_second': job['input_tokens']/result['training_seconds'],
                    'frozen_base_unchanged': result['frozen_base_unchanged'],
                    'cuda_peak_allocated_bytes': result['cuda_peak_allocated_bytes'],
                    'cuda_peak_reserved_bytes': result['cuda_peak_reserved_bytes']})
        elapsed = time.perf_counter()-started
        end_wall = timestamp()
        sampler.stop()
        window = steady_window(report['jobs'], update_token_counts(sample_tokens, batch))
        # A batch-eight job has just 63 total updates. Prespecify a minimum of ten
        # completed updates/worker and ten seconds of common training overlap.
        window['adequacy_rule'] = 'at least 10 seconds and 10 completed updates per worker'
        window['adequate_overlap'] = (window['seconds'] >= 10 and
            len(window['completed_updates_per_tenant']) == len(profiles) and
            min(window['completed_updates_per_tenant'].values(), default=0) >= 10)
        resources = summarize_resources(output/'resource_samples.jsonl', start_wall, end_wall, gpu)
        training_resources = summarize_resources(output/'resource_samples.jsonl',
            window['started_at'], window['finished_at'], gpu)
        report.update(started_at=start_wall, finished_at=end_wall, execution_seconds=elapsed,
            aggregate_input_tokens_per_second=sum(j['input_tokens'] for j in report['jobs'])/elapsed,
            steady_window=window, resources=resources, training_resources=training_resources)
        if len({j['pid'] for j in report['jobs']}) != len(profiles):
            raise RuntimeError('Requested number of independent worker processes was not realized')
        if not window['adequate_overlap']:
            raise RuntimeError('Insufficient all-worker training overlap')
        if (resources['gpu_query_errors'] or not resources['gpu']['sample_count'] or
            training_resources['gpu_query_errors'] or training_resources['gpu']['sample_count'] < 5):
            raise RuntimeError('GPU monitoring missing or failed')
        if references is not None:
            for job in report['jobs']:
                solo = references[job['tenant_id']]
                check = compare_weights(solo['adapter_dir'], job['adapter_dir'])
                check['initial_adapter_equal'] = solo['initial_adapter_sha256'] == job['initial_adapter_sha256']
                check['order_equal'] = solo['order_sha256'] == job['order_sha256']
                check['token_budget_equal'] = (solo['input_tokens'] == job['input_tokens'] and
                    solo['supervised_tokens'] == job['supervised_tokens'])
                job['solo_equivalence'] = check
                job['completion_slowdown'] = job['completion_seconds']/solo['completion_seconds']
                job['training_slowdown'] = job['training_seconds']/solo['training_seconds']
                if not (check['exact'] and check['initial_adapter_equal'] and check['order_equal'] and check['token_budget_equal']):
                    raise RuntimeError('INTEGRITY_FAILED: exact solo equivalence')
        report['status'] = 'SUCCEEDED'
    except Exception as exc:
        report.update(status='FAILED', error_type=type(exc).__name__, error=str(exc)[:500])
    finally:
        for ref in refs:
            ray.cancel(ref, force=True)
            try:
                ray.get(ref)
            except ray.exceptions.RayError:
                pass
        if sampler.thread.is_alive():
            sampler.stop()
        if gate is not None:
            ray.kill(gate, no_restart=True)
        atomic_json(output/'report.json', report)
    return report


def summarize(trials, plan, repeats):
    result = {}
    for batch in sorted({p['batch_size'] for p in plan}):
        levels = sorted({p['n'] for p in plan if p['batch_size'] == batch})
        subset = [t for t in trials if t['batch_size'] == batch]
        summary = summarize_sweep(subset, levels)
        # Partial cells cannot qualify as completed capacity results.
        for row in summary['levels']:
            row['complete'] = row['successful_repeats'] == repeats and row['attempted_repeats'] == repeats
        passing = [r['n'] for r in summary['levels'] if r['eligible'] and r['complete'] and r['meets_slowdown_target']]
        summary['largest_tested_n_meeting_slowdown_target'] = max(passing, default=None)
        summary['scope'] = 'This batch size, workload and host; batch-one cells are endpoint controls only'
        result[str(batch)] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-sweep', required=True)
    parser.add_argument('--batch-counterfactual', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    prior_root = Path(args.reference_sweep).resolve()
    prior = json.loads((prior_root/'report.json').read_text())
    prior_batch_root = Path(args.batch_counterfactual).resolve()
    prior_batch = json.loads((prior_batch_root/'report.json').read_text())
    if prior['status'] != 'COMPLETE' or prior_batch['status'] != 'COMPLETE':
        raise ValueError('Historical reference experiments must be complete')
    if prior['epochs_per_job'] != 1 or prior['model_revision'] != prior_batch['model_revision']:
        raise ValueError('Expected matching one-epoch historical experiments')
    inventory = query_gpus()
    if len(inventory) != 1:
        raise ValueError('Exactly one GPU required')
    gpu = inventory[0]['gpu_uuid']
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = json.loads((prior_root/'profiles.json').read_text())[0]
    if len(source['examples']) != 500:
        raise ValueError('Expected the fixed 500-example screening workload')
    profiles, tokens = prepare_profiles(source, root, 8, prior['epochs_per_job'], prior['model_revision'])
    if any(p['config']['batch_size'] != 1 for p in profiles):
        raise ValueError('Historical profiles must use batch one')
    atomic_json(root/'profiles.json', profiles)
    atomic_json(root/'sample_token_counts.json', tokens)
    plan = matrix_plan(repeats=args.repeats)
    reference_plan = [(b, i) for b in (4, 8) for i in range(8)]
    random.Random(43).shuffle(reference_plan)
    repo = Path(__file__).resolve().parents[1]
    files = [Path(__file__), repo/'scripts/evaluate_batch_concurrency.py',
        repo/'scripts/run_batch_counterfactual.py', repo/'scripts/run_concurrency_sweep.py',
        repo/'src/post_training/batch_engine.py', repo/'src/post_training/batch_concurrency.py',
        repo/'src/post_training/shared_base_scenario.py', repo/'src/post_training/concurrency_sweep.py',
        repo/'src/post_training/intent_evaluation.py']
    snapshots = root/'source_snapshots'
    snapshots.mkdir()
    hashes = {}
    for path in files:
        shutil.copyfile(path, snapshots/path.name)
        hashes[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {'status': 'RUNNING', 'plan': plan, 'reference_plan': reference_plan, 'repeats': args.repeats,
        'model_revision': prior['model_revision'], 'source_sha256': hashes,
        'prior_sweep': str(prior_root), 'prior_batch_counterfactual': str(prior_batch_root),
        'historical_report_hashes': {str(p): hashlib.sha256((p/'report.json').read_bytes()).hexdigest()
            for p in (prior_root, prior_batch_root)},
        'host_before_run': host_snapshot(), 'gpu_inventory': inventory,
        'references': {'1': prior['references'], '4': {}, '8': {}},
        'trials': [], 'controls': {'examples_per_job': 500, 'epochs': 1, 'seeds': [p['config']['seed'] for p in profiles],
            'physical_batches': [4, 8], 'levels': [1, 2, 4, 8], 'fresh_batch_one_controls': [1, 8],
            'cpu_threads_per_worker': 1, 'logical_ray_cpus_per_worker': .5, 'base_copies_per_job': 1,
            'training_start_barrier': True, 'barrier_actor_ready_before_timer': True,
            'gpu_idle_before_each_trial': True, 'slowdown_limit': 1.2,
            'overlap_minimum_seconds': 10, 'overlap_minimum_updates_per_job': 10,
            'checkpoint_cadence': '100 examples rounded to next complete batch, retain latest two'},
        'limitations': ['One fresh solo timing reference per tenant/batch for batch four/eight; timing noise near target matters',
            'Batch-one timing references reused from earlier sweep; fresh endpoint cells check consistency',
            'Homogeneous dataset clones with distinct fixed seeds, not independent customer datasets',
            'Three repeats are exploratory; synchronized training start differs from the historical unbarriered sweep',
            'Fixed learning rate and equal examples; optimizer updates differ across physical batch sizes']}
    atomic_json(root/'protocol.json', {k:v for k,v in report.items() if k not in ('references','trials','status')})
    atomic_json(root/'report.json', report)
    import ray
    try:
        ray.init(address='local', include_dashboard=False)
        capacity = ray.cluster_resources()
        if capacity.get('GPU', 0) != 1 or capacity.get('CPU', 0) < 4:
            raise RuntimeError('Insufficient Ray logical resources for eight workers')
        report['ray_resources'] = capacity
        for batch, index in reference_plan:
            profile = copy.deepcopy(profiles[index])
            profile['config']['batch_size'] = batch
            result = run_trial([profile], root/'references'/f'b{batch}'/profile['tenant_id'], ray, gpu, tokens)
            if result['status'] != 'SUCCEEDED':
                raise RuntimeError(f'Solo reference failed: {result.get("error")}')
            job = result['jobs'][0]
            if index == 0:
                historical = next(t['job'] for t in prior_batch['trials'] if t['batch_size'] == batch)
                result['historical_batch_equivalence'] = compare_weights(historical['adapter_dir'], job['adapter_dir'])
                if not result['historical_batch_equivalence']['exact']:
                    raise RuntimeError('Fresh reference differs from historical batch adapter')
            report['references'][str(batch)][profile['tenant_id']] = job
            atomic_json(root/'report.json', report)
            print('REFERENCE', batch, profile['tenant_id'], round(result['execution_seconds'], 2), flush=True)
        for index, entry in enumerate(plan, 1):
            profiles_now = copy.deepcopy(profiles[:entry['n']])
            for profile in profiles_now:
                profile['config']['batch_size'] = entry['batch_size']
            result = run_trial(profiles_now, root/'trials'/f'{index:02d}-b{entry["batch_size"]}-n{entry["n"]}-r{entry["repeat"]}',
                ray, gpu, tokens, report['references'][str(entry['batch_size'])])
            result.update(entry)
            report['trials'].append(result)
            report['summary'] = summarize(report['trials'], plan, args.repeats)
            atomic_json(root/'report.json', report)
            print('TRIAL', index, entry, result['status'], round(result.get('execution_seconds', 0), 2), flush=True)
            if 'INTEGRITY_FAILED' in result.get('error', ''):
                raise RuntimeError(result['error'])
        report['status'] = 'TRAINING_COMPLETE'
    except BaseException as exc:
        report.update(status='FAILED', error_type=type(exc).__name__, error=str(exc)[:500])
        raise
    finally:
        ray.shutdown()
        atomic_json(root/'report.json', report)
    try:
        with (root/'quality.log').open('x') as log:
            subprocess.run([sys.executable, str(repo/'scripts/evaluate_batch_concurrency.py'),
                '--experiment-dir', str(root)], stdout=log, stderr=subprocess.STDOUT, check=True)
        report['quality'] = json.loads((root/'quality_report.json').read_text())
        report['status'] = 'COMPLETE'
    except BaseException as exc:
        report.update(status='QUALITY_FAILED', error_type=type(exc).__name__, error=str(exc)[:500])
        raise
    finally:
        atomic_json(root/'report.json', report)
    atomic_json(root/'summary.json', report['summary'])


if __name__ == '__main__':
    main()
