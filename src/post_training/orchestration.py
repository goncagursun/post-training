"""Layer 4: atomically claim FIFO work, dispatch through Ray, persist outcomes."""
from dataclasses import asdict

from sqlalchemy import select, text

from post_training.config import SFTJobConfig
from post_training.database import DatasetRecord, Job, JobExecution, utcnow
from post_training.datasets import load_registered_dataset
from post_training.storage import job_run_dir

# All dispatcher instances using this database share one managed training slot.
SLOT_LOCK_ID = 70143001
ACTIVE_STATES = ("CLAIMED", "RUNNING")


def claim_next_job(sessions, data_root, device):
    """A short transaction reserves both the next job and the single platform slot."""
    with sessions.begin() as session:
        if session.bind.dialect.name != "postgresql":
            raise RuntimeError("Dispatcher claims require PostgreSQL locking")
        # Serializes the capacity check across competing dispatchers, not just a row.
        session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": SLOT_LOCK_ID})
        active = session.scalar(select(Job.id).where(Job.status.in_(ACTIVE_STATES)).limit(1))
        if active is not None:
            raise RuntimeError(f"Training slot is reserved by job {active}. Inspect its execution before restarting; automatic recovery is not implemented.")
        job = session.scalar(select(Job).where(Job.status == "QUEUED")
                             .order_by(Job.created_at, Job.id).limit(1)
                             .with_for_update(skip_locked=True))
        if job is None:
            return None
        dataset = session.scalar(select(DatasetRecord).where(
            DatasetRecord.id == job.dataset_id, DatasetRecord.tenant_id == job.tenant_id))
        if dataset is None:
            raise RuntimeError("Job dataset ownership is inconsistent")
        config = SFTJobConfig(
            model_name=job.config["model_name"],
            learning_rate=job.config["learning_rate"],
            num_train_epochs=job.config["num_train_epochs"],
            lora_rank=job.config["lora_rank"],
            device=device, precision="fp32" if device == "cpu" else "bf16")
        execution = JobExecution(job_id=job.id,
            run_dir=str(job_run_dir(data_root, job.tenant_id, job.id)),
            dataset_uri=dataset.uri, config=asdict(config))
        session.add(execution)
        job.status = "CLAIMED"
        return {"job_id": job.id, "tenant_id": job.tenant_id, "dataset_uri": dataset.uri,
                "config": asdict(config), "run_dir": execution.run_dir}


def mark_dispatched(sessions, job_id, task_id):
    with sessions.begin() as session:
        job = session.get(Job, job_id, with_for_update=True)
        if job.status != "CLAIMED":
            raise RuntimeError("Expected CLAIMED job")
        job.status = "RUNNING"
        execution = session.get(JobExecution, job_id)
        execution.ray_task_id = task_id
        execution.dispatched_at = utcnow()


def finish_job(sessions, job_id, result=None, error=None):
    with sessions.begin() as session:
        job = session.get(Job, job_id, with_for_update=True)
        if job.status not in ACTIVE_STATES:
            raise RuntimeError("Cannot finish an inactive job")
        execution = session.get(JobExecution, job_id)
        execution.finished_at = utcnow()
        execution.result = result
        execution.error = error
        job.status = "FAILED" if error is not None else "SUCCEEDED"


def execute_claimed_job(sessions, claimed):
    import ray
    from post_training.worker import train_sft_worker

    job_id = claimed["job_id"]
    try:
        examples, validation, metadata = load_registered_dataset(
            claimed["dataset_uri"], claimed["tenant_id"])
        num_gpus = int(claimed["config"]["device"] == "cuda")
        ref = train_sft_worker.options(num_gpus=num_gpus).remote(
            claimed["config"], claimed["run_dir"], examples=examples,
            validation_examples=validation, dataset_metadata=metadata)
    except Exception as exc:
        finish_job(sessions, job_id, error=f"Dispatch failed: {type(exc).__name__}")
        return "FAILED"
    # If a DB write fails after submission, do not report a training failure or
    # release the slot: execution may still be active. Stop for manual inspection.
    mark_dispatched(sessions, job_id, ref.task_id().hex())
    try:
        result = ray.get(ref)
    except Exception as exc:
        finish_job(sessions, job_id, error=f"Worker failed: {type(exc).__name__}")
        return "FAILED"
    finish_job(sessions, job_id, result=result)
    return "SUCCEEDED"
