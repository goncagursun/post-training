"""Independent-concurrency sweep measurements and decision rules."""
from datetime import datetime
import json
from pathlib import Path
import random
import statistics
import threading
import time

from post_training.monitoring import Measurements, query_gpus
from post_training.telemetry import timestamp


def randomized_plan(levels, repeats, seed):
    if not levels or len(set(levels)) != len(levels) or min(levels) < 1 or repeats < 1:
        raise ValueError('Unique positive concurrency levels and positive repeats are required')
    rng=random.Random(seed)
    plan=[]
    for repeat in range(1,repeats+1):
        block=list(levels)
        rng.shuffle(block)
        plan.extend({'repeat':repeat,'n':n} for n in block)
    return plan


def host_snapshot(root=Path('/sys/fs/cgroup')):
    def integer(path):
        try:return int(path.read_text().strip())
        except (OSError,ValueError):return None
    def fields(path):
        try:return {key:int(value) for key,value in map(str.split,path.read_text().splitlines())}
        except (OSError,ValueError):return {}
    if (root/'cpu.max').exists():
        quota,period=(root/'cpu.max').read_text().split()
        stats=fields(root/'cpu.stat')
        memory=fields(root/'memory.stat')
        return {'scope':'container cgroup v2, includes other container processes',
            'cpu_quota_cores':int(quota)/int(period) if quota!='max' else None,
            'cpu_usage_ns':stats.get('usage_usec',0)*1000 if 'usage_usec' in stats else None,
            'cpu_throttled_ns':stats.get('throttled_usec',0)*1000 if 'throttled_usec' in stats else None,
            'memory_usage_bytes':integer(root/'memory.current'),'memory_limit_bytes':integer(root/'memory.max'),
            'memory_rss_bytes':memory.get('anon')}
    quota,period=integer(root/'cpu/cpu.cfs_quota_us'),integer(root/'cpu/cpu.cfs_period_us')
    stats=fields(root/'cpu/cpu.stat')
    memory=fields(root/'memory/memory.stat')
    return {'scope':'container cgroup v1, includes other container processes',
        'cpu_quota_cores':quota/period if quota is not None and quota>0 and period else None,
        'cpu_usage_ns':integer(root/'cpuacct/cpuacct.usage'),
        'cpu_throttled_ns':stats.get('throttled_time'),
        'memory_usage_bytes':integer(root/'memory/memory.usage_in_bytes'),
        'memory_limit_bytes':integer(root/'memory/memory.limit_in_bytes'),
        'memory_rss_bytes':memory.get('total_rss',memory.get('rss'))}


class SweepSampler:
    def __init__(self,path):
        self.path=Path(path)
        self.stop_event=threading.Event()
        self.ready=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.errors=0
    def start(self):
        self.thread.start()
        if not self.ready.wait(15):raise RuntimeError('Resource sampler did not become ready')
    def _run(self):
        while not self.stop_event.is_set():
            record={'timestamp':timestamp(),'monotonic':time.monotonic(),'host':host_snapshot()}
            try:record['gpus']=query_gpus()
            except Exception as exc:
                self.errors+=1
                record['gpu_error_type']=type(exc).__name__
            with self.path.open('a') as file:file.write(json.dumps(record)+'\n')
            self.ready.set()
            self.stop_event.wait(1)
    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():raise RuntimeError('Resource sampler did not stop')


def summarize_resources(path,start,end,gpu_uuid):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    rows=[r for r in rows if start<=r['timestamp']<=end]
    gpu=Measurements()
    errors=0
    for row in rows:
        errors+='gpu_error_type' in row
        for sample in row.get('gpus',[]):
            if sample['gpu_uuid']==gpu_uuid:gpu.add(sample)
    host={'sample_count':len(rows),'scope':'container, not per-tenant; memory usage can include filesystem cache'}
    for key in ('memory_usage_bytes','memory_rss_bytes'):
        values=[r['host'][key] for r in rows if r['host'].get(key) is not None]
        host['peak_'+key]=max(values,default=None)
    if len(rows)>=2:
        first,last=rows[0],rows[-1]
        elapsed=last['monotonic']-first['monotonic']
        a,b=first['host'],last['host']
        quota=b.get('cpu_quota_cores')
        host['cpu_quota_cores']=quota
        host['covered_seconds']=elapsed
        if a.get('cpu_usage_ns') is not None and b.get('cpu_usage_ns') is not None and elapsed>0:
            cores=(b['cpu_usage_ns']-a['cpu_usage_ns'])/1e9/elapsed
            host['mean_cpu_cores_used']=cores
            host['mean_cpu_quota_percent']=cores/quota*100 if quota else None
        if a.get('cpu_throttled_ns') is not None and b.get('cpu_throttled_ns') is not None:
            host['cpu_throttled_seconds']=(b['cpu_throttled_ns']-a['cpu_throttled_ns'])/1e9
    return {'gpu':gpu.report(),'host':host,'gpu_query_errors':errors}


def training_events(worker_dir):
    records=[json.loads(line) for line in (Path(worker_dir)/'events.jsonl').read_text().splitlines()]
    starts=[e['timestamp'] for e in records if e.get('event')=='phase_started' and e.get('phase')=='training']
    ends=[e['timestamp'] for e in records if e.get('event')=='phase_finished' and e.get('phase')=='training']
    return starts[0],ends[-1],[e for e in records if e.get('event')=='tenant_update']


def steady_window(jobs,token_counts):
    phases=[training_events(job['worker_dir']) for job in jobs]
    start=max(p[0] for p in phases)
    end=min(p[1] for p in phases)
    seconds=max(0,(datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds())
    counts={}
    tokens=0
    if seconds>0:
        for job,(_,_,events) in zip(jobs,phases):
            selected=[e for e in events if start<=e['timestamp']<=end]
            counts[job['tenant_id']]=len(selected)
            tokens+=sum(token_counts[job['tenant_id']][e['step']-1] for e in selected)
    return {'started_at':start,'finished_at':end,'seconds':seconds,
        'completed_updates_per_tenant':counts,'input_tokens':tokens,
        'aggregate_input_tokens_per_second':tokens/seconds if seconds>0 else None,
        'adequate_overlap':seconds>=10 and len(counts)==len(jobs) and min(counts.values(),default=0)>=50,
        'scope':'completed update tokens during common training-loop overlap, includes checkpoint/logging time; boundary uncertainty <=1 update per tenant'}


def summarize_sweep(trials,levels,slowdown_limit=1.2,plateau_gain=.10):
    rows=[]
    for n in sorted(levels):
        attempted=[t for t in trials if t['n']==n]
        successful=[t for t in attempted if t['status']=='SUCCEEDED']
        if not successful:
            rows.append({'n':n,'successful_repeats':0,'attempted_repeats':len(attempted),'eligible':False})
            continue
        def spread(values):
            return {'median':statistics.median(values),'min':min(values),'max':max(values)}
        per_tenant={}
        for name in {j['tenant_id'] for t in successful for j in t['jobs']}:
            jobs=[j for t in successful for j in t['jobs'] if j['tenant_id']==name]
            per_tenant[name]={k:spread([j[k] for j in jobs]) for k in
                ('completion_seconds','training_tokens_per_second','completion_slowdown','training_slowdown')}
        all_good=all(t['steady_window']['adequate_overlap'] and t['resources']['gpu_query_errors']==0 and
                     t['resources']['gpu']['sample_count']>0 for t in successful)
        worst=max(v['completion_slowdown']['median'] for v in per_tenant.values())
        rows.append({'n':n,'successful_repeats':len(successful),'attempted_repeats':len(attempted),
            'eligible':len(successful)==len(attempted) and all_good,
            'execution_seconds':spread([t['execution_seconds'] for t in successful]),
            'aggregate_input_tokens_per_second':spread([t['aggregate_input_tokens_per_second'] for t in successful]),
            'steady_input_tokens_per_second':spread([t['steady_window']['aggregate_input_tokens_per_second'] for t in successful]),
            'peak_vram_gib':spread([t['resources']['gpu']['maximum']['device_memory_used_bytes']/1024**3 for t in successful]),
            'worst_tenant_median_completion_slowdown':worst,
            'meets_slowdown_target':worst<=slowdown_limit,'tenants':per_tenant})
    eligible=[r for r in rows if r['eligible']]
    passing=[r['n'] for r in eligible if r['meets_slowdown_target']]
    plateau=[]
    for low,high in zip(rows,rows[1:]):
        if low['eligible'] and high['eligible']:
            gain=high['steady_input_tokens_per_second']['median']/low['steady_input_tokens_per_second']['median']-1
            high['steady_throughput_gain_vs_previous']=gain
            if gain<=plateau_gain:plateau.append({'lower_n':low['n'],'higher_n':high['n'],'gain':gain})
    return {'levels':rows,'slowdown_limit':slowdown_limit,'plateau_gain_threshold':plateau_gain,
        'largest_tested_n_meeting_slowdown_target':max(passing,default=None),
        'throughput_maximizing_tested_n':max(eligible,key=lambda r:r['steady_input_tokens_per_second']['median'])['n'] if eligible else None,
        'candidate_plateau_intervals':plateau,
        'scope':'this workload, host, thread settings and tested levels only; three repeats are exploratory'}
