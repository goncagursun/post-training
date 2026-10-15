import json

import pytest

from post_training.telemetry import RunTelemetry, NvidiaSampler, cost_estimate


def test_phases_and_failure_summary(tmp_path):
    recorder=RunTelemetry(tmp_path,{'tenant_id':'a','job_id':'job'})
    with pytest.raises(ValueError):
        with recorder.phase('loading'):
            raise ValueError('sensitive internal detail')
    summary=recorder.finish('FAILED','ValueError')
    assert summary['phase_seconds']['loading'] >= 0
    assert summary['process_lifetime_peak_rss_bytes'] > 0
    assert summary['error_type']=='ValueError'
    events=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert all(row['tenant_id']=='a' and row['job_id']=='job' for row in events)
    assert 'sensitive internal detail' not in (tmp_path/'telemetry.json').read_text()


def test_cost_scope():
    assert cost_estimate(1800,'cuda',2)['estimated_cost_usd']==1
    assert cost_estimate(1800,'cuda',None)['estimated_cost_usd'] is None
    assert cost_estimate(1800,'cpu',2)['estimated_cost_usd']==0
    assert cost_estimate(1800,'cpu',2)['gpu_seconds']==0


def test_unavailable_gpu_sampler_is_explicit(tmp_path):
    sampler=NvidiaSampler(RunTelemetry(tmp_path),None)
    sampler.start()
    metrics=sampler.stop()
    assert metrics['status']=='unavailable'
    assert metrics['sample_count']==0
    assert metrics['mean_utilization_percent'] is None


def test_gpu_sampler_parses_device_metrics(tmp_path,monkeypatch):
    import subprocess
    recorder=RunTelemetry(tmp_path)
    sampler=NvidiaSampler(recorder,'GPU-test')
    monkeypatch.setattr('post_training.telemetry.shutil.which',lambda _: '/nvidia-smi')
    def query(command,**kwargs):
        assert '--id=GPU-test' in command
        sampler.stop_event.set()
        return subprocess.CompletedProcess(command,0,'70, 256\n','')
    monkeypatch.setattr('post_training.telemetry.subprocess.run',query)
    sampler.start()
    sampler.thread.join()
    metrics=sampler.stop()
    assert metrics['mean_utilization_percent']==70
    assert metrics['max_device_memory_used_bytes']==256*1024**2
