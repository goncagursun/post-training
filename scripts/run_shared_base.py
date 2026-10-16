"""Scenario 3: one Ray worker, one frozen base, two interleaved tenant adapters."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from post_training.concurrent_scenario import load_scenario
from post_training.monitoring import query_gpus
from post_training.shared_base_scenario import run_shared_training


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--gpu-hourly-rate-usd',type=float)
    parser.add_argument('--gpu-uuid')
    args=parser.parse_args()
    rate=args.gpu_hourly_rate_usd
    if rate is not None and (not math.isfinite(rate) or rate<0):
        parser.error('Hourly rate must be finite and nonnegative')
    load_scenario(args.spec)
    inventory=query_gpus()
    gpu=args.gpu_uuid
    if gpu is None:
        if len(inventory)!=1:parser.error('Select --gpu-uuid on a multi-GPU host')
        gpu=inventory[0]['gpu_uuid']
    if gpu not in {g['gpu_uuid'] for g in inventory}:parser.error('Unknown GPU UUID')
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    output=Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=False)
    import ray
    command=[sys.executable,str(Path(__file__).with_name('monitor_gpus.py')),'--output-dir',str(output/'monitoring')]
    if rate is not None:command+=['--gpu-hourly-rate-usd',str(rate)]
    report={'scenario':'shared-base-interleaved','status':'STARTING','selected_gpu_uuid':gpu}
    ref=None
    with (output/'monitor.log').open('x') as log:
        monitor=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    try:
        ray.init(address='local',include_dashboard=False)
        resources=ray.cluster_resources()
        if resources.get('GPU',0)!=1 or resources.get('CPU',0)<1:
            raise RuntimeError(f'Expected one visible GPU and at least one CPU: {resources}')
        worker=ray.remote(num_cpus=1,num_gpus=1,max_calls=1,max_retries=0)(run_shared_training)
        started=time.perf_counter()
        ref=worker.remote(str(Path(args.spec).resolve()),str(output/'worker'))
        report=ray.get(ref)
        elapsed=time.perf_counter()-started
        report['worker_execution_seconds']=report.pop('execution_seconds')
        report['worker_aggregate_input_tokens_per_second']=report['aggregate_input_tokens_per_second']
        report['submission_to_result_seconds']=elapsed
        report['aggregate_input_tokens_per_second']=sum(j['input_tokens'] for j in report['jobs'])/elapsed
        report['aggregate_samples_per_second']=sum(j['steps'] for j in report['jobs'])/elapsed
        report['gpu_hourly_rate_usd']=rate
        report['estimated_execution_cost_usd']=elapsed/3600*rate if rate is not None else None
        report['cost_scope']='one GPU, task submission to observed result; includes worker startup, loading, training, validation, export'
        report['selected_gpu_uuid']=gpu
    except BaseException as exc:
        report['status']='FAILED'
        report['error_type']=type(exc).__name__
        if ref is not None:
            ray.cancel(ref,force=True)
            try:ray.get(ref)
            except ray.exceptions.RayError:pass
        raise
    finally:
        ray.shutdown()
        monitor.terminate()
        try:monitor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()
        summary=output/'monitoring/summary.json'
        if summary.exists():report['platform_monitor']=json.loads(summary.read_text())
        report['monitor_exit_code']=monitor.returncode
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','base_model_copies','frozen_base_unchanged',
        'submission_to_result_seconds','aggregate_input_tokens_per_second','estimated_execution_cost_usd')},indent=2))


if __name__=='__main__':main()
