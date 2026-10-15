import json
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_platform_api import platform,body,headers
from post_training.database import Job


def test_cancel_auth_ownership_queue_and_repeats(platform):
    client,sessions,keys=platform
    job=client.post('/training/jobs',json=body(),headers=headers(keys)).json()
    url=f"/training/jobs/{job['id']}/cancel"
    assert client.post(url).status_code==401
    assert client.post(url,headers=headers(keys,'b')).status_code==404
    response=client.post(url,headers=headers(keys))
    assert response.status_code==202 and response.json()['status']=='CANCELLED'
    assert client.post(url,headers=headers(keys)).json()['status']=='CANCELLED'
    assert client.get(f"/training/jobs/{job['id']}",headers=headers(keys)).json()['execution'] is None


def test_running_request_and_terminal_preservation(platform):
    client,sessions,keys=platform
    job=client.post('/training/jobs',json=body(),headers=headers(keys)).json()
    with sessions.begin() as session:session.get(Job,job['id']).status='RUNNING'
    url=f"/training/jobs/{job['id']}/cancel"
    assert client.post(url,headers=headers(keys)).json()['status']=='CANCEL_REQUESTED'
    assert client.post(url,headers=headers(keys)).json()['status']=='CANCEL_REQUESTED'
    with sessions.begin() as session:session.get(Job,job['id']).status='SUCCEEDED'
    assert client.post(url,headers=headers(keys)).json()['status']=='SUCCEEDED'


@pytest.mark.skipif(not os.environ.get('RUN_RAY_SMOKE'),reason='Opt-in Ray cancellation')
def test_cancel_live_ray_training(platform,tmp_path):
    import ray
    from post_training.database import Tenant,DatasetRecord,JobExecution
    from post_training.auth import issue_api_key
    from post_training.orchestration import claim_next_job,execute_claimed_job
    from post_training.recovery import job_lock
    client,sessions,_=platform
    with sessions() as session:
        if session.bind.dialect.name!='postgresql':pytest.skip('Requires PostgreSQL')
    with sessions.begin() as session:
        session.add(Tenant(id='tenant-a'));session.flush()
        session.add(DatasetRecord(id='cancel-data',tenant_id='tenant-a',uri='demo://tenant-a/mini-qa-v1'))
        key=issue_api_key(session,'tenant-a')
    h={'X-API-Key':key}
    job=client.post('/training/jobs',headers=h,json=body(dataset_id='cancel-data',epochs=10)).json()
    ray.init(address='local',include_dashboard=False)
    try:
        claimed=claim_next_job(sessions,tmp_path,'cpu')
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(execute_claimed_job,sessions,claimed)
            deadline=time.monotonic()+60
            events=Path(claimed['run_dir'])/'events.jsonl'
            while not events.exists() or 'trainer_metrics' not in events.read_text():
                if future.done():raise AssertionError(f'Job ended before cancellation: {future.result()}')
                if time.monotonic()>deadline:raise AssertionError('Worker did not start training')
                time.sleep(0.05)
            response=client.post(f"/training/jobs/{job['id']}/cancel",headers=h)
            assert response.json()['status']=='CANCEL_REQUESTED'
            assert future.result(timeout=30)=='CANCELLED'
        with job_lock(claimed['run_dir']):pass
        result=client.get(f"/training/jobs/{job['id']}",headers=h).json()
        assert result['status']=='CANCELLED'
        assert result['execution']['finished_at'] is not None
        assert not (Path(claimed['run_dir'])/'adapter').exists()
    finally:ray.shutdown()


def test_cancel_claimed_job_preserves_request_and_skips_dispatch(platform,tmp_path):
    from post_training.orchestration import claim_next_job,mark_dispatched,execute_claimed_job
    client,sessions,keys=platform
    with sessions() as session:
        if session.bind.dialect.name!='postgresql':pytest.skip('Requires PostgreSQL')
    job=client.post('/training/jobs',json=body(),headers=headers(keys)).json()
    claimed=claim_next_job(sessions,tmp_path,'cpu')
    assert client.post(f"/training/jobs/{job['id']}/cancel",headers=headers(keys)).json()['status']=='CANCEL_REQUESTED'
    mark_dispatched(sessions,job['id'],'test-no-task',claimed['run_dir'])
    assert execute_claimed_job(sessions,claimed)=='CANCELLED'
    assert not Path(claimed['run_dir']).exists()
    assert claim_next_job(sessions,tmp_path,'cpu') is None
