import json
import subprocess

import pytest

from post_training import monitoring
from post_training.monitoring import PlatformMonitor, query_gpus


def gpu(uuid='GPU-a', utilization=0):
    return {'gpu_uuid': uuid, 'name': 'test', 'gpu_utilization_percent': utilization,
            'device_memory_used_bytes': 1024, 'device_memory_total_bytes': 4096,
            'temperature_celsius': 42, 'power_watts': 50}


def test_query_handles_optional_unsupported_fields(monkeypatch):
    monkeypatch.setattr(monitoring.subprocess, 'run', lambda *a, **k:
        subprocess.CompletedProcess(a, 0, 'GPU-a, Test GPU, 0, 2, 10, [N/A], [Not Supported]\n'))
    sample = query_gpus()[0]
    assert sample['gpu_utilization_percent'] == 0
    assert sample['device_memory_used_bytes'] == 2 * 1024**2
    assert sample['temperature_celsius'] is sample['power_watts'] is None


def test_tenant_transitions_and_idle_cost_reconcile(tmp_path, monkeypatch):
    clock = [0.0]
    owner = [None]
    monkeypatch.setattr(monitoring.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(monitoring, 'query_gpus', lambda: [gpu()])
    monkeypatch.setattr(monitoring, 'active_reservation', lambda _: owner[0])
    monitor = PlatformMonitor(tmp_path/'monitor', 3.6, sessions=object())
    monitor.poll()
    clock[0] = 10
    owner[0] = {'tenant_id': 'a', 'job_id': 'job-a', 'run_dir': '/a/001', 'status': 'RUNNING'}
    monitor.poll()
    clock[0] = 30
    owner[0] = {'tenant_id': 'b', 'job_id': 'job-b', 'run_dir': '/b/001', 'status': 'RUNNING'}
    monitor.poll()
    clock[0] = 40
    result = monitor.stop()
    assert result['observed_allocation_gpu_seconds'] == 40
    assert result['reserved_gpu_seconds'] == 30
    assert result['unreserved_gpu_seconds'] == 10
    assert result['unattributed_gpu_seconds'] == 0
    assert result['tenants']['a']['reserved_gpu_seconds'] == 20
    assert result['tenants']['b']['reserved_gpu_seconds'] == 10
    assert result['estimated_cost_usd'] == pytest.approx(.04)
    assert result['estimated_unreserved_cost_usd'] == pytest.approx(.01)
    assert json.loads((monitor.directory/'summary.json').read_text())['status'] == 'STOPPED'


def test_failed_query_gap_is_not_zero_activity(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(monitoring.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(monitoring, 'query_gpus', lambda: [gpu(utilization=None)])
    monitor = PlatformMonitor(tmp_path/'monitor')
    monitor.poll()
    clock[0] = 2
    def fail():
        raise subprocess.TimeoutExpired('nvidia-smi', 3)
    monkeypatch.setattr(monitoring, 'query_gpus', fail)
    monitor.poll()
    clock[0] = 10
    result = monitor.stop()
    assert result['observed_allocation_gpu_seconds'] == 2
    assert result['query_error_count'] == 1
    assert result['estimated_cost_usd'] is None
    measurements = result['gpus']['GPU-a']['measurements']
    assert measurements['sample_count'] == 1
    assert measurements['zero_utilization_sample_fraction'] is None


def test_multiple_gpus_require_explicit_attribution(tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring, 'query_gpus', lambda: [gpu(), gpu('GPU-b')])
    monkeypatch.setattr(monitoring, 'active_reservation', lambda _: {
        'tenant_id': 'a', 'job_id': 'j', 'run_dir': '/a/001', 'status': 'RUNNING'})
    monitor = PlatformMonitor(tmp_path/'unknown', sessions=object())
    monitor.poll()
    result = monitor.stop()
    assert result['tenants'] == {}
    assert result['unattributed_gpu_seconds'] == result['observed_allocation_gpu_seconds']
    selected = PlatformMonitor(tmp_path/'selected', sessions=object(), gpu_uuid='GPU-b')
    selected.poll()
    result = selected.stop()
    assert result['tenants']['a']['measurements']['sample_count'] == 1
    records = [json.loads(line) for line in (selected.directory/'gpu_samples.jsonl').read_text().splitlines()]
    assert records[0]['reservation'] is None
    assert records[1]['reservation']['tenant_id'] == 'a'


def test_database_outage_does_not_stop_gpu_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring, 'query_gpus', lambda: [gpu()])
    def fail(_):
        raise RuntimeError('private connection information')
    monkeypatch.setattr(monitoring, 'active_reservation', fail)
    monitor = PlatformMonitor(tmp_path/'monitor', sessions=object())
    monitor.poll()
    result = monitor.stop()
    assert result['query_error_count'] == 1
    assert result['unattributed_gpu_seconds'] == result['observed_allocation_gpu_seconds']
    assert 'private' not in (monitor.directory/'gpu_samples.jsonl').read_text()


@pytest.mark.parametrize('rate', [-1, float('nan'), float('inf')])
def test_invalid_rates_rejected(tmp_path, rate):
    with pytest.raises(ValueError):
        PlatformMonitor(tmp_path/'monitor', rate)


from test_platform_api import platform


def test_database_reservation_tracks_cuda_and_rejects_multiple_slots(platform):
    from post_training.database import Job, JobExecution
    from post_training.monitoring import active_reservation
    _, sessions, _ = platform
    with sessions.begin() as session:
        session.add(Job(id='j1', tenant_id='a', dataset_id='dataset_a', base_model_id='smollm2-135m',
                        config={}, status='RUNNING'))
        session.flush()
        session.add(JobExecution(job_id='j1', run_dir='/a/001', dataset_uri='demo://a', config={'device':'cuda'}))
    assert active_reservation(sessions)['tenant_id'] == 'a'
    with sessions.begin() as session:
        session.get(Job, 'j1').status = 'CANCEL_REQUESTED'
    assert active_reservation(sessions)['status'] == 'CANCEL_REQUESTED'
    with sessions.begin() as session:
        session.add(Job(id='j2', tenant_id='b', dataset_id='dataset_b', base_model_id='smollm2-135m',
                        config={}, status='CLAIMED'))
        session.flush()
        session.add(JobExecution(job_id='j2', run_dir='/b/001', dataset_uri='demo://b', config={'device':'cuda'}))
    with pytest.raises(ValueError, match='Multiple active'):
        active_reservation(sessions)
    with sessions.begin() as session:
        session.get(Job, 'j2').status = 'SUCCEEDED'
        session.get(JobExecution, 'j1').config = {'device':'cpu'}
    assert active_reservation(sessions) is None
