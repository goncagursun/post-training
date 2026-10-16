"""Run all three scenarios and the solo-A isolation control with one training engine."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from post_training.concurrent_scenario import load_scenario
from post_training.controlled_engine import run_controlled_worker
from post_training.monitoring import query_gpus
from post_training.telemetry import timestamp


def run_mode(mode, tenants, output, ray, gpu, rate):
    output.mkdir(exist_ok=False)
    # Keep monitor ready before starting the common task-submission clock.
    monitor_script=Path(__file__).with_name('monitor_gpus.py')
    command=[sys.executable,str(monitor_script),'--output-dir',str(output/'monitoring')]
    if rate is not None:command+=['--gpu-hourly-rate-usd',str(rate)]
    with (output/'monitor.log').open('x') as log:
        monitor=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    reports=[]
    refs=[]
    summary={'mode':mode,'status':'RUNNING','worker_reports':reports}
    try:
        deadline=time.monotonic()+60
        while not (output/'monitoring/summary.json').exists():
            if monitor.poll() is not None:raise RuntimeError('Monitor failed to start')
            if time.monotonic()>deadline:raise TimeoutError('Monitor startup')
            time.sleep(.1)
        worker=ray.remote(num_cpus=1,num_gpus=.5 if mode=='independent' else 1,
                          max_calls=1,max_retries=0)(run_controlled_worker)
        start_wall=timestamp()
        started=time.perf_counter()
        if mode=='fifo':
            for index,tenant in enumerate(tenants):
                ref=worker.remote([tenant],str(output/f'worker-{index}'))
                refs.append(ref)
                reports.append(ray.get(ref))
        else:
            groups=[[t] for t in tenants] if mode=='independent' else [tenants[:1] if mode=='solo-a' else tenants]
            for index,group in enumerate(groups):
                refs.append(worker.remote(group,str(output/f'worker-{index}')))
            # All independent tasks are submitted before awaiting either.
            reports.extend(ray.get(refs))
        elapsed=time.perf_counter()-started
        end_wall=timestamp()
        jobs=[job for report in reports for job in report['jobs']]
        summary.update(status='SUCCEEDED',started_at=start_wall,finished_at=end_wall,
            execution_seconds=elapsed,jobs=jobs,
            aggregate_input_tokens_per_second=sum(j['input_tokens'] for j in jobs)/elapsed,
            estimated_cost_usd=elapsed/3600*rate if rate is not None else None,
            timing_scope='first task submission to last observed result; same boundary for every mode',
            selected_gpu_uuid=gpu)
    except BaseException as exc:
        summary['status']='FAILED'
        summary['error_type']=type(exc).__name__
        for ref in refs:
            ray.cancel(ref,force=True)
            try:ray.get(ref)
            except ray.exceptions.RayError:pass
        raise
    finally:
        monitor.terminate()
        try:monitor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor.kill(); monitor.wait()
        monitor_path=output/'monitoring/gpu_samples.jsonl'
        # Restrict GPU samples to exactly the same wall-clock execution interval.
        if summary['status']=='SUCCEEDED':
            from post_training.monitoring import Measurements
            measurements=Measurements()
            count=0
            for line in monitor_path.read_text().splitlines():
                record=json.loads(line)
                if record.get('event')=='gpu_sample' and record['gpu_uuid']==gpu and start_wall<=record['timestamp']<=end_wall:
                    measurements.add(record)
                    count+=1
            summary['execution_window_gpu_measurements']=measurements.report()
            summary['gpu_sample_count']=count
        (output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(mode,round(summary['execution_seconds'],2),'seconds',flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--model-revision',required=True,help='Pinned base-model commit shared by all modes')
    parser.add_argument('--gpu-hourly-rate-usd',type=float,default=.59)
    args=parser.parse_args()
    if not math.isfinite(args.gpu_hourly_rate_usd) or args.gpu_hourly_rate_usd<0:parser.error('Invalid hourly rate')
    tenants=load_scenario(args.spec)
    for tenant in tenants:tenant['config']['model_revision']=args.model_revision
    inventory=query_gpus()
    if len(inventory)!=1:parser.error('Controlled benchmark currently requires exactly one visible GPU')
    gpu=inventory[0]['gpu_uuid']
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=False)
    # Save the exact common engine and input definition with the experiment.
    import post_training.controlled_engine as engine
    (output/'engine_snapshot.py').write_text(Path(engine.__file__).read_text())
    (output/'inputs.json').write_text(json.dumps(tenants,indent=2)+'\n')
    report={'status':'RUNNING','repetitions':1,'model_revision':args.model_revision,
            'modes':{},'limitations':['one measured repetition per mode','warm shared disk cache; mode order is fixed'],
            'engine_sha256':hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest()}
    import ray
    try:
        ray.init(address='local',include_dashboard=False)
        if ray.cluster_resources().get('GPU',0)!=1 or ray.cluster_resources().get('CPU',0)<2:
            raise RuntimeError('Need two CPUs and one GPU')
        for mode in ('solo-a','shared','fifo','independent'):
            report['modes'][mode]=run_mode(mode,tenants,output/mode,ray,gpu,args.gpu_hourly_rate_usd)
            (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            if mode=='shared':
                subprocess.run([sys.executable,str(Path(__file__).with_name('verify_controlled_isolation.py')),
                    '--output-dir',str(output)],check=True)
                report['isolation']=json.loads((output/'isolation.json').read_text())
        report['status']='TRAINING_COMPLETE_ISOLATION_PASSED'
    except BaseException as exc:
        report['status']='FAILED'
        report['error_type']=type(exc).__name__
        raise
    finally:
        ray.shutdown()
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
