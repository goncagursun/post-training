import json
from pathlib import Path

import pytest
from post_training.recovery import (job_lock,worker_guard,fence_attempt,latest_checkpoint,
                                   publish_checkpoint,validate_checkpoint)


def fake_checkpoint(root,step=4):
    path=root/'checkpoints'/f'checkpoint-{step}'
    path.mkdir(parents=True)
    for name in ['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','rng_state.pth']:
        (path/name).write_text('dummy state')
    (path/'trainer_state.json').write_text(json.dumps({'global_step':step}))
    publish_checkpoint(path,'fingerprint')
    return path


def test_checkpoint_publication_validation_and_fallback(tmp_path):
    checkpoint=fake_checkpoint(tmp_path)
    assert latest_checkpoint(tmp_path)==checkpoint
    newer=fake_checkpoint(tmp_path,8)
    (newer/'optimizer.pt').write_text('corrupted')
    assert latest_checkpoint(tmp_path)==checkpoint
    with pytest.raises(ValueError,match='differs'):
        validate_checkpoint(checkpoint,'different')
    (checkpoint/'complete.json').unlink()
    with pytest.raises(ValueError,match='No completed'):
        latest_checkpoint(tmp_path)


def test_live_worker_and_stale_attempt_refused(tmp_path):
    first=tmp_path/'job/attempts/001'
    second=tmp_path/'job/attempts/002'
    fence_attempt(first)
    with worker_guard(first):
        with pytest.raises(RuntimeError,match='still holds'):
            fence_attempt(second)
    fence_attempt(second)
    with pytest.raises(RuntimeError,match='Stale'):
        with worker_guard(first):pass
    with worker_guard(second):pass


@pytest.mark.skipif(not __import__('os').environ.get('RUN_RAY_SMOKE'),reason='Opt-in PostgreSQL/Ray CPU recovery')
def test_ray_interrupt_and_resume_preserves_history(platform,tmp_path):
    import ray
    from post_training.database import Job,Tenant,DatasetRecord,JobExecution,JobRecovery
    from post_training.orchestration import claim_next_job,execute_claimed_job,prepare_recovery
    client,sessions,keys=platform
    with sessions() as session:
        if session.bind.dialect.name!='postgresql':pytest.skip('Requires PostgreSQL')
    with sessions.begin() as session:
        session.add(Tenant(id='tenant-a'));session.flush()
        session.add(DatasetRecord(id='recovery-data',tenant_id='tenant-a',uri='demo://tenant-a/mini-qa-v1'));session.flush()
        session.add(Job(id='recovery-job',tenant_id='tenant-a',dataset_id='recovery-data',
            base_model_id='smollm2-135m',status='QUEUED',config={'model_name':'HuggingFaceTB/SmolLM2-135M',
            'learning_rate':0.001,'num_train_epochs':4,'lora_rank':8}))
    ray.init(address='local',include_dashboard=False)
    try:
        first=claim_next_job(sessions,tmp_path,'cpu')
        assert execute_claimed_job(sessions,first,interrupt_after_checkpoint=4)=='INTERRUPTED'
        with sessions() as session:
            assert session.get(Job,'recovery-job').status=='INTERRUPTED'
        plan=prepare_recovery(sessions,'recovery-job')
        assert plan['attempt']==2
        with sessions() as session:
            assert len(session.get(JobRecovery,'recovery-job').history)==1
        second=claim_next_job(sessions,tmp_path,'cpu')
        assert second['run_dir'].endswith('/attempts/002')
        assert second['resume_checkpoint'].endswith('checkpoint-4')
        assert execute_claimed_job(sessions,second)=='SUCCEEDED'
        with sessions() as session:
            result=session.get(JobExecution,'recovery-job').result
            assert result['global_step']==8
            assert result['telemetry']['training']['observed_optimizer_steps']==4
            assert result['telemetry']['resume']['global_step']==4
    finally:ray.shutdown()


from test_platform_api import platform


@pytest.mark.skipif(not __import__('os').environ.get('RUN_RAY_SMOKE'),reason='Opt-in actual Trainer resume')
def test_resumed_weights_match_uninterrupted_and_finalization(tmp_path):
    from safetensors.torch import load_file
    from post_training.training.sft import run_sft
    from post_training.config import SFTJobConfig
    from post_training.recovery import TrainingInterrupted
    config=SFTJobConfig(device='cpu',precision='fp32',num_train_epochs=2)
    baseline=run_sft(config,tmp_path/'baseline')
    with pytest.raises(TrainingInterrupted):
        run_sft(config,tmp_path/'first',interrupt_after_checkpoint=4)
    checkpoint=latest_checkpoint(tmp_path/'first')
    resumed=run_sft(config,tmp_path/'second',resume_checkpoint=str(checkpoint))
    a=load_file(str(Path(baseline.adapter_dir)/'adapter_model.safetensors'))
    b=load_file(str(Path(resumed.adapter_dir)/'adapter_model.safetensors'))
    assert all((a[k]-b[k]).abs().max().item()<1e-6 for k in a)
    assert resumed.telemetry['training']['observed_optimizer_steps']==4
    final=run_sft(config,tmp_path/'finalize',resume_checkpoint=str(latest_checkpoint(tmp_path/'second')))
    c=load_file(str(Path(final.adapter_dir)/'adapter_model.safetensors'))
    assert all((b[k]-c[k]).abs().max().item()==0 for k in b)
    assert final.telemetry['training']['observed_optimizer_steps']==0
    assert final.training_loss is None
