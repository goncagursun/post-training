import json
from pathlib import Path
import pytest
from post_training.concurrency_sweep import randomized_plan,host_snapshot,steady_window,summarize_sweep


def test_randomized_blocks_include_every_level_once():
    plan=randomized_plan([1,2,4,8],3,42)
    assert plan==randomized_plan([1,2,4,8],3,42)
    for repeat in (1,2,3):
        assert sorted(p['n'] for p in plan if p['repeat']==repeat)==[1,2,4,8]
    with pytest.raises(ValueError):randomized_plan([1,1],3,42)


def test_cgroup_v1_resource_units(tmp_path):
    files={'cpu/cpu.cfs_quota_us':'765000','cpu/cpu.cfs_period_us':'100000',
        'cpu/cpu.stat':'nr_throttled 2\nthrottled_time 123456\n','cpuacct/cpuacct.usage':'987654',
        'memory/memory.stat':'total_rss 1234\n','memory/memory.usage_in_bytes':'5000',
        'memory/memory.limit_in_bytes':'50000'}
    for name,value in files.items():
        path=tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(value)
    sample=host_snapshot(tmp_path)
    assert sample['cpu_quota_cores']==7.65
    assert sample['cpu_usage_ns']==987654
    assert sample['cpu_throttled_ns']==123456
    assert sample['memory_rss_bytes']==1234


def test_cgroup_v2_missing_memory_limit_is_unknown(tmp_path):
    (tmp_path/'cpu.max').write_text('200000 100000')
    (tmp_path/'cpu.stat').write_text('usage_usec 50\nthrottled_usec 3\n')
    (tmp_path/'memory.stat').write_text('anon 100\n')
    (tmp_path/'memory.current').write_text('200')
    (tmp_path/'memory.max').write_text('max')
    sample=host_snapshot(tmp_path)
    assert sample['cpu_quota_cores']==2
    assert sample['cpu_usage_ns']==50000
    assert sample['memory_limit_bytes'] is None


def write_events(root,name,start,end,updates):
    path=root/name
    path.mkdir()
    def t(second):return f'2026-01-01T00:00:{second:02d}+00:00'
    events=[{'timestamp':t(start),'event':'phase_started','phase':'training'}]
    events.extend({'timestamp':t(second),'event':'tenant_update','step':step,'tenant_id':name} for second,step in updates)
    events.append({'timestamp':t(end),'event':'phase_finished','phase':'training'})
    (path/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    return {'tenant_id':name,'worker_dir':str(path)}


def test_common_window_counts_actual_completed_tokens(tmp_path):
    a=write_events(tmp_path,'a',10,30,[(12,1),(20,2),(28,3)])
    b=write_events(tmp_path,'b',15,35,[(16,1),(25,2),(32,3)])
    result=steady_window([a,b],{'a':[10,20,30],'b':[100,200,300]})
    assert result['seconds']==15
    assert result['input_tokens']==350
    assert result['aggregate_input_tokens_per_second']==pytest.approx(350/15)
    assert not result['adequate_overlap']


def test_capacity_target_and_throughput_plateau_are_distinct():
    trials=[]
    for n,speed,slowdown in [(1,100,1),(2,200,1.1),(4,208,2)]:
        for repeat in (1,2,3):
            trials.append({'n':n,'status':'SUCCEEDED','execution_seconds':100,
                'aggregate_input_tokens_per_second':speed,
                'steady_window':{'adequate_overlap':True,'aggregate_input_tokens_per_second':speed},
                'resources':{'gpu_query_errors':0,'gpu':{'sample_count':10,'maximum':{'device_memory_used_bytes':1024**3}}},
                'jobs':[{'tenant_id':'a','completion_seconds':100,'training_tokens_per_second':speed/n,
                         'completion_slowdown':slowdown,'training_slowdown':slowdown}]})
    trials.append({'n':8,'status':'FAILED'})
    result=summarize_sweep(trials,[1,2,4,8])
    assert result['largest_tested_n_meeting_slowdown_target']==2
    assert result['throughput_maximizing_tested_n']==4
    assert result['candidate_plateau_intervals'][0]['lower_n']==2
    assert not result['levels'][-1]['eligible']
