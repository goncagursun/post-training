"""Repeated N-job independent-concurrency sweep, using the unchanged controlled engine."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from post_training.concurrent_scenario import load_scenario
from post_training.controlled_engine import run_controlled_worker
from post_training.concurrency_sweep import randomized_plan,SweepSampler,summarize_resources,steady_window,summarize_sweep
from post_training.monitoring import query_gpus
from post_training.telemetry import timestamp


def atomic_json(path,value):
    temporary=path.with_name('.'+path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def single_thread_worker(tenant,output):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    props=torch.cuda.get_device_properties(0)
    gpu=str(props.uuid)
    if not gpu.startswith('GPU-'):gpu='GPU-'+gpu
    result=run_controlled_worker([tenant],output)
    result['gpu_uuid']=gpu
    result['torch_intraop_threads']=torch.get_num_threads()
    result['torch_interop_threads']=torch.get_num_interop_threads()
    return result


def prepare_profiles(source,output,count,epochs,revision):
    import torch
    from transformers import AutoTokenizer
    from post_training.shared_base_scenario import tokenize_pairs
    from post_training.bitext import load_manifest
    profiles=[]
    tokens={}
    tokenizer=AutoTokenizer.from_pretrained(source['config']['model_name'],revision=revision)
    for index in range(count):
        name=f'sweep-tenant-{index+1:02d}'
        directory=output/'inputs'/name
        directory.mkdir(parents=True)
        manifest=copy.deepcopy(source['manifest'])
        manifest['tenant_id']=name
        for split,info in manifest['splits'].items():
            shutil.copyfile(Path(source['manifest_path']).parent/info['file'],directory/info['file'])
        atomic_json(directory/'manifest.json',manifest)
        load_manifest(directory/'manifest.json',name)
        config={**source['config'],'model_revision':revision,'num_train_epochs':epochs,
                'seed':source['config']['seed']+index}
        profile={'tenant_id':name,'config':config,'manifest':manifest,
            'manifest_path':str(directory/'manifest.json'),'examples':source['examples'],
            'validation':source['validation'],'baseline_adapter_dir':None,
            'source_profile_tenant_id':source['tenant_id']}
        profiles.append(profile)
        rows=tokenize_pairs(tokenizer,profile['examples'],config['max_length'])
        generator=torch.Generator().manual_seed(config['seed'])
        order=[i for _ in range(epochs) for i in torch.randperm(len(rows),generator=generator).tolist()]
        tokens[name]=[len(rows[i]['input_ids']) for i in order]
        atomic_json(directory/'config.json',config)
    return profiles,tokens


def run_trial(profiles,output,ray,gpu,token_counts,references=None):
    from verify_controlled_isolation import compare_weights
    output.mkdir(parents=True,exist_ok=False)
    sampler=SweepSampler(output/'resource_samples.jsonl')
    report={'n':len(profiles),'status':'RUNNING','jobs':[]}
    refs={}
    start_wall=end_wall=None
    try:
        sampler.start()
        worker=ray.remote(num_cpus=.5,num_gpus=1/len(profiles),max_calls=1,max_retries=0)(single_thread_worker)
        start_wall=timestamp()
        started=time.perf_counter()
        for index,profile in enumerate(profiles):
            submitted=time.perf_counter()
            directory=output/f'worker-{index:02d}'
            ref=worker.remote(profile,str(directory))
            refs[ref]=(profile,directory,submitted)
        while refs:
            ready,_=ray.wait(list(refs),timeout=.1)
            for ref in ready:
                profile,directory,submitted=refs.pop(ref)
                result=ray.get(ref)
                completed=time.perf_counter()
                job=result['jobs'][0]
                report['jobs'].append({**job,'worker_dir':str(directory),'pid':result['pid'],
                    'gpu_uuid':result['gpu_uuid'],'completion_seconds':completed-submitted,
                    'submission_offset_seconds':submitted-started,
                    'training_seconds':result['training_seconds'],
                    'training_tokens_per_second':job['input_tokens']/result['training_seconds'],
                    'frozen_base_unchanged':result['frozen_base_unchanged'],
                    'torch_intraop_threads':result['torch_intraop_threads'],
                    'torch_interop_threads':result['torch_interop_threads']})
        elapsed=time.perf_counter()-started
        end_wall=timestamp()
        sampler.stop()
        report.update(started_at=start_wall,finished_at=end_wall,execution_seconds=elapsed,
            aggregate_input_tokens_per_second=sum(j['input_tokens'] for j in report['jobs'])/elapsed,
            steady_window=steady_window(report['jobs'],token_counts),
            resources=summarize_resources(output/'resource_samples.jsonl',start_wall,end_wall,gpu))
        if len({j['pid'] for j in report['jobs']})!=len(profiles) or any(j['gpu_uuid']!=gpu for j in report['jobs']):
            raise RuntimeError('Requested concurrency was not realized on one GPU')
        if not report['steady_window']['adequate_overlap']:
            raise RuntimeError('Insufficient all-job training overlap; increase workload length')
        for job in report['jobs']:
            if not job['frozen_base_unchanged'] or job['torch_intraop_threads']!=1 or job['torch_interop_threads']!=1:
                raise RuntimeError('Worker control settings failed')
            if references is not None:
                reference=references[job['tenant_id']]
                check=compare_weights(reference['adapter_dir'],job['adapter_dir'])
                check['initial_adapter_equal']=reference['initial_adapter_sha256']==job['initial_adapter_sha256']
                check['order_equal']=reference['order_sha256']==job['order_sha256']
                job['solo_equivalence']=check
                job['completion_slowdown']=job['completion_seconds']/reference['completion_seconds']
                job['training_slowdown']=job['training_seconds']/reference['training_seconds']
                if not check['passed'] or not check['initial_adapter_equal'] or not check['order_equal']:
                    raise RuntimeError('SOLO_EQUIVALENCE_FAILED')
        report['status']='SUCCEEDED'
    except Exception as exc:
        report['status']='FAILED'
        report['error_type']=type(exc).__name__
        report['error']=str(exc)[:500]
    finally:
        for ref in refs:
            ray.cancel(ref,force=True)
            try:ray.get(ref)
            except ray.exceptions.RayError:pass
        if sampler.thread.is_alive():sampler.stop()
        atomic_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--model-revision',required=True)
    parser.add_argument('--levels',type=int,nargs='+',default=[1,2,4,8])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--references-from',help='Completed compatible sweep whose solo references can be reused')
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--slowdown-limit',type=float,default=1.2)
    parser.add_argument('--plateau-gain',type=float,default=.10)
    args=parser.parse_args()
    plan=randomized_plan(args.levels,args.repeats,args.seed)
    if args.epochs<1 or not math.isfinite(args.slowdown_limit) or args.slowdown_limit<1 or not 0<=args.plateau_gain<1:
        parser.error('Invalid epochs or decision thresholds')
    source=load_scenario(args.spec)[0]
    inventory=query_gpus()
    if len(inventory)!=1:parser.error('Exactly one visible GPU is required')
    gpu=inventory[0]['gpu_uuid']
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    root=Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=False)
    profiles,tokens=prepare_profiles(source,root,max(args.levels),args.epochs,args.model_revision)
    atomic_json(root/'profiles.json',profiles)
    atomic_json(root/'per_update_token_counts.json',tokens)
    import post_training.controlled_engine as engine
    (root/'engine_snapshot.py').write_text(Path(engine.__file__).read_text())
    report={'status':'RUNNING','plan':plan,'levels':args.levels,'repeats':args.repeats,
        'epochs_per_job':args.epochs,'updates_per_job':len(tokens[profiles[0]['tenant_id']]),
        'workload':'homogeneous clones of tenant A dataset, distinct fixed seeds and logical tenant IDs',
        'thread_settings':{'intraop':1,'interop':1,'logical_ray_cpus_per_job':.5},
        'model_revision':args.model_revision,'engine_sha256':hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
        'references':{},'trials':[],'limitations':['one solo timing reference per tenant; repeated N=1 cells estimate profile-level variability',
            'homogeneous synthetic tenant workload; not independent customer datasets','container CPU/RAM includes other container processes',
            'no coordinated checkpoint resume; historical experiments are preserved']}
    atomic_json(root/'report.json',report)
    import ray
    try:
        ray.init(address='local',include_dashboard=False)
        capacity=ray.cluster_resources()
        if capacity.get('GPU',0)!=1 or capacity.get('CPU',0)<max(args.levels)*.5:
            raise RuntimeError(f'Cannot admit all requested jobs: {capacity}')
        report['ray_resources']=capacity
        if args.references_from:
            prior_root=Path(args.references_from).expanduser().resolve()
            prior=json.loads((prior_root/'report.json').read_text())
            prior_profiles={p['tenant_id']:p for p in json.loads((prior_root/'profiles.json').read_text())}
            if prior['status']!='COMPLETE' or prior['engine_sha256']!=report['engine_sha256'] or prior['thread_settings']!=report['thread_settings']:
                raise ValueError('Reference sweep is incomplete or uses a different engine/thread configuration')
            for profile in profiles:
                old_profile=prior_profiles[profile['tenant_id']]
                if old_profile['config']!=profile['config'] or old_profile['manifest']!=profile['manifest']:
                    raise ValueError('Reference profile configuration or dataset differs')
                reference=prior['references'][profile['tenant_id']]
                if not (Path(reference['adapter_dir'])/'adapter_model.safetensors').is_file():
                    raise ValueError('Reference adapter is missing')
                report['references'][profile['tenant_id']]=reference
            report['references_source']=str(prior_root)
            report['references_report_sha256']=hashlib.sha256((prior_root/'report.json').read_bytes()).hexdigest()
            atomic_json(root/'report.json',report)
        else:
            order=list(range(len(profiles)))
            random.Random(args.seed+1).shuffle(order)
            for index in order:
                profile=profiles[index]
                result=run_trial([profile],root/'references'/profile['tenant_id'],ray,gpu,tokens)
                if result['status']!='SUCCEEDED':raise RuntimeError(f'Solo reference failed: {result}')
                report['references'][profile['tenant_id']]=result['jobs'][0]
                atomic_json(root/'report.json',report)
                print('REFERENCE',profile['tenant_id'],round(result['execution_seconds'],2),flush=True)
        for index,entry in enumerate(plan,1):
            result=run_trial(profiles[:entry['n']],root/'trials'/f"{index:02d}-n{entry['n']}-r{entry['repeat']}",
                ray,gpu,tokens,report['references'])
            result.update(entry)
            report['trials'].append(result)
            report['summary']=summarize_sweep(report['trials'],args.levels,args.slowdown_limit,args.plateau_gain)
            atomic_json(root/'report.json',report)
            print('TRIAL',index,'N',entry['n'],'repeat',entry['repeat'],result['status'],
                  round(result.get('execution_seconds',0),2),flush=True)
            if result.get('error')=='SOLO_EQUIVALENCE_FAILED':raise RuntimeError('Solo-equivalence gate failed; stopping')
        report['status']='COMPLETE'
    except BaseException as exc:
        report['status']='FAILED'
        report['error_type']=type(exc).__name__
        raise
    finally:
        ray.shutdown()
        atomic_json(root/'report.json',report)
    atomic_json(root/'summary.json',report['summary'])
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':main()
