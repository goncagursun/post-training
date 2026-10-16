"""Independent two-worker GPU benchmark; deliberately separate from the FIFO service."""
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from post_training.bitext import intent_pairs, load_manifest
from post_training.config import SFTJobConfig
from post_training.monitoring import query_gpus
from post_training.recovery import fence_attempt
from post_training.storage import job_run_dir
from post_training.telemetry import timestamp


def load_scenario(path):
    path = Path(path).expanduser().resolve()
    spec = json.loads(path.read_text())
    tenants = spec['tenants']
    if len(tenants) != 2 or len({t['tenant_id'] for t in tenants}) != 2:
        raise ValueError('Exactly two distinct tenants are required')
    prepared = []
    for tenant in tenants:
        # Validate identifiers before creating any experiment directories.
        job_run_dir('/tmp', tenant['tenant_id'], 'validation')
        def resolve(value):
            return (path.parent / value).expanduser().resolve()
        config = SFTJobConfig.from_json(resolve(tenant['config_path']))
        if config.device != 'cuda':
            raise ValueError('Independent concurrency requires explicit CUDA configs')
        manifest_path = resolve(tenant['manifest_path'])
        manifest, splits = load_manifest(manifest_path, tenant['tenant_id'])
        baseline = tenant.get('baseline_adapter_dir')
        if baseline:
            baseline = resolve(baseline)
            saved = json.loads((baseline.parent / 'dataset.json').read_text())
            if saved != manifest:
                raise ValueError('Dataset differs from the baseline manifest')
            if json.loads((baseline.parent / 'config.json').read_text()) != asdict(config):
                raise ValueError('Configuration differs from the baseline')
        prepared.append({'tenant_id': tenant['tenant_id'], 'config': asdict(config),
                         'manifest_path': str(manifest_path), 'manifest': manifest,
                         'examples': intent_pairs(splits['train']),
                         'validation': intent_pairs(splits['validation']),
                         'baseline_adapter_dir': str(baseline) if baseline else None})
    return prepared


def training_interval(run_dir):
    events = [json.loads(line) for line in (Path(run_dir) / 'events.jsonl').read_text().splitlines()]
    bounds = {}
    for event in events:
        if event.get('phase') == 'training' and event['event'] in ('phase_started', 'phase_finished'):
            bounds[event['event']] = datetime.fromisoformat(event['timestamp']).timestamp()
    return bounds['phase_started'], bounds['phase_finished']


def summarize_jobs(jobs, elapsed, hourly_rate):
    succeeded = [job for job in jobs if job['status'] == 'SUCCEEDED']
    intervals = [training_interval(job['run_dir']) for job in succeeded]
    overlap = (max(0.0, min(end for _, end in intervals) - max(start for start, _ in intervals))
               if len(intervals) == 2 else None)
    pids = [job['result']['telemetry']['pid'] for job in succeeded]
    gpu_ids = [job['result']['telemetry'].get('gpu_activity', {}).get('gpu_uuid') for job in succeeded]
    verified = (len(succeeded) == 2 and len(set(pids)) == 2 and
                all(gpu_ids) and len(set(gpu_ids)) == 1 and overlap > 0)
    tokens = sum(j['result']['telemetry']['training']['input_tokens'] for j in succeeded)
    samples = sum(j['result']['telemetry']['training'].get('samples', 0) for j in succeeded)
    return {'jobs_succeeded': len(succeeded), 'training_overlap_seconds': overlap,
            'distinct_worker_processes': len(pids) == 2 and len(set(pids)) == 2,
            'same_gpu_verified': len(gpu_ids) == 2 and all(gpu_ids) and len(set(gpu_ids)) == 1,
            'concurrency_verified': verified,
            'submission_to_all_results_seconds': elapsed,
            'aggregate_input_tokens_per_second': tokens / elapsed if elapsed > 0 else None,
            'aggregate_samples_per_second': samples / elapsed if elapsed > 0 else None,
            'estimated_execution_cost_usd': elapsed / 3600 * hourly_rate if hourly_rate is not None else None,
            'cost_scope': 'one GPU counted once, first submission to all observed results; includes startup, validation, checkpoints, export',
            'tenant_cost_policy': 'no per-tenant dollar allocation; fractional Ray resources are scheduling declarations'}


def run_scenario(spec_path, output_dir, hourly_rate=None, gpu_uuid=None):
    if hourly_rate is not None and (not math.isfinite(hourly_rate) or hourly_rate < 0):
        raise ValueError('Hourly rate must be finite and nonnegative')
    tenants = load_scenario(spec_path)
    inventory = query_gpus()
    if gpu_uuid is None:
        if len(inventory) != 1:
            raise ValueError('Select --gpu-uuid on a multi-GPU host')
        gpu_uuid = inventory[0]['gpu_uuid']
    if gpu_uuid not in {g['gpu_uuid'] for g in inventory}:
        raise ValueError('Selected GPU is not visible to nvidia-smi')
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'scenario': 'independent-concurrency',
              'started_at': timestamp(), 'status': 'STARTING', 'selected_gpu_uuid': gpu_uuid,
              'gpu_hourly_rate_usd': hourly_rate, 'gpu_fraction_per_worker': 0.5,
              'base_model_copies': 2, 'database_service_used': False,
              'gpu_activity_scope': 'whole device; worker samples include both tenants', 'jobs': []}
    def save():
        temporary = output / '.report.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(output / 'report.json')
    for tenant in tenants:
        job_id = str(uuid.uuid4())
        run_dir = job_run_dir(output, tenant['tenant_id'], job_id)
        fence_attempt(run_dir)
        report['jobs'].append({'tenant_id': tenant['tenant_id'], 'job_id': job_id,
            'run_dir': str(run_dir), 'manifest_path': tenant['manifest_path'],
            'config': tenant['config'], 'baseline_adapter_dir': tenant['baseline_adapter_dir'],
            'status': 'PENDING'})
    save()
    # Set before Ray starts so both logical half-GPU tasks see the same physical device.
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    import ray
    from post_training.worker import train_sft_worker
    # Reuse the existing standalone monitor without its FIFO-only DB attribution.
    monitor_script = Path(__file__).resolve().parents[2] / 'scripts' / 'monitor_gpus.py'
    monitor_command = [sys.executable, str(monitor_script), '--output-dir', str(output / 'monitoring')]
    if hourly_rate is not None:
        monitor_command += ['--gpu-hourly-rate-usd', str(hourly_rate)]
    refs = {}
    execution_start = None
    monitor = None
    try:
        with (output / 'monitor.log').open('x') as log:
            monitor = subprocess.Popen(monitor_command, stdout=log, stderr=subprocess.STDOUT)
        ray.init(address='local', include_dashboard=False)
        resources = ray.cluster_resources()
        if resources.get('CPU', 0) < 2 or resources.get('GPU', 0) != 1:
            raise RuntimeError(f'Requires at least two CPUs and exactly one visible Ray GPU: {resources}')
        report['status'] = 'RUNNING'
        report['ray_resources'] = resources
        execution_start = time.perf_counter()
        for tenant, job in zip(tenants, report['jobs']):
            job['submitted_at'] = timestamp()
            ref = train_sft_worker.options(num_gpus=0.5).remote(
                tenant['config'], job['run_dir'], examples=tenant['examples'],
                validation_examples=tenant['validation'], dataset_metadata=tenant['manifest'],
                context={'tenant_id': job['tenant_id'], 'job_id': job['job_id'],
                         'attempt': '001', 'scenario': 'independent-concurrency',
                         'ray_submission_requested_at': job['submitted_at']})
            refs[ref] = job
            job['status'] = 'SUBMITTED'
        save()
        while refs:
            ready, _ = ray.wait(list(refs), timeout=1)
            for ref in ready:
                job = refs.pop(ref)
                try:
                    job['result'] = ray.get(ref)
                    job['status'] = 'SUCCEEDED'
                except Exception as exc:
                    job['status'] = 'FAILED'
                    job['error_type'] = type(exc).__name__
                job['finished_at'] = timestamp()
                save()
        elapsed = time.perf_counter() - execution_start
        report['measurements'] = summarize_jobs(report['jobs'], elapsed, hourly_rate)
        report['status'] = 'SUCCEEDED' if report['measurements']['concurrency_verified'] else 'FAILED'
    except BaseException as exc:
        report['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED'
        report['error_type'] = type(exc).__name__
        raise
    finally:
        for ref, job in refs.items():
            ray.cancel(ref, force=True)
            try:
                ray.get(ref)
            except ray.exceptions.RayError:
                pass
            job['status'] = 'CANCELLED'
        ray.shutdown()
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        monitor_summary = output / 'monitoring' / 'summary.json'
        if monitor_summary.exists():
            report['platform_monitor'] = json.loads(monitor_summary.read_text())
        report['monitor_exit_code'] = monitor.returncode if monitor is not None else None
        report['finished_at'] = timestamp()
        save()
    if report['status'] != 'SUCCEEDED':
        raise RuntimeError('Concurrent scenario failed or training overlap was not verified; inspect report.json')
    return report
