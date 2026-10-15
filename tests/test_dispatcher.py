from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from test_platform_api import platform
from post_training.database import DatasetRecord, Job, JobExecution, utcnow
from post_training.datasets import load_demo_dataset
from post_training.orchestration import claim_next_job, finish_job, mark_dispatched
from post_training.storage import job_run_dir


def enqueue(sessions):
    with sessions.begin() as session:
        for i, letter in enumerate(('a', 'b')):
            dataset=session.get(DatasetRecord, f'dataset_{letter}')
            dataset.uri=f'demo://tenant-{letter}/mini-qa-v1'
            # Fixture tenants are a/b, so use matching IDs below only for claim tests.
            session.add(Job(id=f'job-{letter}', tenant_id=letter, dataset_id=dataset.id,
                base_model_id='smollm2-135m', status='QUEUED',
                created_at=utcnow()+timedelta(seconds=i),
                config={'model_name':'HuggingFaceTB/SmolLM2-135M','learning_rate':0.001,
                        'num_train_epochs':1,'lora_rank':8}))


def require_postgres(sessions):
    with sessions() as session:
        if session.bind.dialect.name != 'postgresql':
            pytest.skip('Claim locking requires TEST_DATABASE_URL PostgreSQL')


def test_fifo_slot_and_terminal_results(platform, tmp_path):
    client,sessions,keys=platform
    require_postgres(sessions)
    enqueue(sessions)
    claimed=claim_next_job(sessions,tmp_path,'cpu')
    assert claimed['job_id']=='job-a'
    assert claimed['config']['precision']=='fp32'
    with pytest.raises(RuntimeError,match='reserved'):
        claim_next_job(sessions,tmp_path,'cpu')
    mark_dispatched(sessions,'job-a','task-1')
    finish_job(sessions,'job-a',result={'adapter_dir':'adapter-a','global_step':2,'training_loss':4.0})
    claimed_b=claim_next_job(sessions,tmp_path,'cpu')
    assert claimed_b['job_id']=='job-b'
    assert claimed_b['run_dir'] != claimed['run_dir']
    finish_job(sessions,'job-b',error='Worker failed: TestError')
    assert claim_next_job(sessions,tmp_path,'cpu') is None
    with sessions() as session:
        assert session.get(Job,'job-a').status=='SUCCEEDED'
        assert session.get(Job,'job-b').status=='FAILED'
        assert session.get(JobExecution,'job-a').result['global_step']==2
        assert session.get(JobExecution,'job-b').finished_at is not None
    response=client.get('/training/jobs/job-a',headers={'X-API-Key':keys['a']})
    assert response.json()['execution']['result']['global_step']==2
    assert client.get('/training/jobs/job-a',headers={'X-API-Key':keys['b']}).status_code==404


def test_competing_claims_reserve_only_one_job(platform,tmp_path):
    _,sessions,_=platform
    require_postgres(sessions)
    enqueue(sessions)
    def claim():
        try:
            return claim_next_job(sessions,tmp_path,'cpu')
        except RuntimeError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:claim(),range(2)))
    assert sum(item is not None for item in results)==1
    with sessions() as session:
        assert len(session.scalars(select(JobExecution)).all())==1


def test_dataset_resolution_and_path_safety(tmp_path):
    a=load_demo_dataset('demo://tenant-a/mini-qa-v1','tenant-a')
    b=load_demo_dataset('demo://tenant-b/mini-qa-v1','tenant-b')
    assert len(a)==len(b)==2 and a!=b
    with pytest.raises(ValueError):
        load_demo_dataset('demo://tenant-a/mini-qa-v1','tenant-b')
    with pytest.raises(ValueError):
        job_run_dir(tmp_path,'../../escape','job')
    assert job_run_dir(tmp_path,'tenant-a','job-a').is_relative_to(tmp_path)


@pytest.mark.skipif(not __import__('os').environ.get('RUN_RAY_SMOKE'), reason='Opt-in real CPU training')
def test_real_ray_two_tenants(platform,tmp_path):
    import ray
    from post_training.database import Tenant
    from post_training.orchestration import execute_claimed_job
    _,sessions,_=platform
    require_postgres(sessions)
    with sessions.begin() as session:
        for letter in ('a','b'):
            tenant=f'tenant-{letter}'
            session.add(Tenant(id=tenant))
            session.flush()
            session.add(DatasetRecord(id=f'real_{letter}',tenant_id=tenant,uri=f'demo://{tenant}/mini-qa-v1'))
            session.flush()
            session.add(Job(id=f'real-{letter}',tenant_id=tenant,dataset_id=f'real_{letter}',
                base_model_id='smollm2-135m',config={'model_name':'HuggingFaceTB/SmolLM2-135M',
                'learning_rate':0.001,'num_train_epochs':1,'lora_rank':8},status='QUEUED'))
    ray.init(address='local',include_dashboard=False)
    try:
        for _ in range(2):
            claimed=claim_next_job(sessions,tmp_path,'cpu')
            assert execute_claimed_job(sessions,claimed)=='SUCCEEDED'
            with sessions() as session:
                execution=session.get(JobExecution,claimed['job_id'])
                assert execution.result['global_step']==2
                telemetry=execution.result['telemetry']
                assert telemetry['status']=='SUCCEEDED'
                assert telemetry['context']['tenant_id']==claimed['tenant_id']
                assert telemetry['training']['input_tokens']>0
                assert telemetry['cuda_peak_allocated_bytes'] is None
                assert execution.result['platform_metrics']['cost']['gpu_seconds']==0
                assert (Path(execution.run_dir)/'events.jsonl').exists()
                adapter=Path(execution.result['adapter_dir'])
                assert (adapter/'adapter_model.safetensors').is_file()
                assert claimed['tenant_id'] in adapter.parts
        assert claim_next_job(sessions,tmp_path,'cpu') is None
    finally:
        ray.shutdown()


def test_dispatch_failure_releases_slot(platform,tmp_path):
    from post_training.orchestration import execute_claimed_job
    _,sessions,_=platform
    require_postgres(sessions)
    enqueue(sessions)
    claimed=claim_next_job(sessions,tmp_path,'cpu')
    claimed['dataset_uri']='demo://unsupported'
    assert execute_claimed_job(sessions,claimed)=='FAILED'
    with sessions() as session:
        assert session.get(Job,'job-a').status=='FAILED'
        assert session.get(JobExecution,'job-a').error=='Dispatch failed: ValueError'
    assert claim_next_job(sessions,tmp_path,'cpu')['job_id']=='job-b'
