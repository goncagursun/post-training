"""Layer 4: atomically claim FIFO work, dispatch through Ray, persist outcomes."""
from dataclasses import asdict
from datetime import timezone
import json
import time
from pathlib import Path

from post_training.telemetry import cost_estimate

from sqlalchemy import select, text

from post_training.config import SFTJobConfig
from post_training.database import DatasetRecord, Job, JobExecution, JobRecovery, utcnow
from post_training.datasets import load_registered_dataset
from post_training.storage import job_run_dir
from post_training.recovery import (TrainingInterrupted, fence_attempt, job_lock,
                                    latest_checkpoint, validate_checkpoint)

# All dispatcher instances using this database share one managed training slot.
SLOT_LOCK_ID = 70143001
ACTIVE_STATES = ("CLAIMED", "RUNNING", "CANCEL_REQUESTED")


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
        recovery = session.get(JobRecovery, job.id)
        config = SFTJobConfig(
            model_name=job.config["model_name"],
            learning_rate=job.config["learning_rate"],
            num_train_epochs=job.config["num_train_epochs"],
            lora_rank=job.config["lora_rank"],
            device=device, precision="fp32" if device == "cpu" else "bf16")
        attempt = recovery.attempt if recovery else 1
        if recovery:
            original_config = recovery.history[-1]["config"]
            if device != original_config["device"]:
                raise ValueError("Recovery requires the original device and precision")
            config = SFTJobConfig(**original_config)
        run_dir = job_run_dir(data_root, job.tenant_id, job.id, attempt)
        fence_attempt(run_dir)
        execution = JobExecution(job_id=job.id,
            run_dir=str(run_dir),
            dataset_uri=dataset.uri, config=asdict(config))
        session.add(execution)
        job.status = "CLAIMED"
        return {"job_id": job.id, "tenant_id": job.tenant_id, "dataset_uri": dataset.uri,
                "config": asdict(config), "run_dir": execution.run_dir,
                "attempt": attempt, "resume_checkpoint": recovery.checkpoint if recovery else None}


def mark_dispatched(sessions, job_id, task_id, expected_run_dir=None):
    with sessions.begin() as session:
        job = session.get(Job, job_id, with_for_update=True)
        if expected_run_dir and (job.execution is None or job.execution.run_dir != expected_run_dir):
            raise RuntimeError("Stale dispatcher update refused")
        if job.status not in {"CLAIMED", "CANCEL_REQUESTED"}:
            raise RuntimeError("Expected CLAIMED or CANCEL_REQUESTED job")
        if job.status == "CLAIMED":
            job.status = "RUNNING"
        execution = session.get(JobExecution, job_id)
        execution.ray_task_id = task_id
        execution.dispatched_at = utcnow()


def finish_job(sessions, job_id, result=None, error=None, gpu_hourly_rate_usd=None,
               terminal_status=None, expected_run_dir=None):
    with sessions.begin() as session:
        job = session.get(Job, job_id, with_for_update=True)
        if expected_run_dir and (job.execution is None or job.execution.run_dir != expected_run_dir):
            raise RuntimeError("Stale dispatcher update refused")
        if job.status not in ACTIVE_STATES:
            raise RuntimeError("Cannot finish an inactive job")
        execution = session.get(JobExecution, job_id)
        execution.finished_at = utcnow()
        def aware(value):
            return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
        slot_seconds = (aware(execution.finished_at) - aware(execution.claimed_at)).total_seconds()
        platform = {
            "queue_to_claim_seconds": (aware(execution.claimed_at) - aware(job.created_at)).total_seconds(),
            "slot_reserved_seconds": slot_seconds,
            "submission_to_finished_seconds": (aware(execution.finished_at) - aware(job.created_at)).total_seconds(),
            "cost": {"scope": "single platform slot from claim to recorded completion; includes dispatch/wait",
                     **cost_estimate(slot_seconds, execution.config["device"], gpu_hourly_rate_usd)},
        }
        result = dict(result or {})
        result["platform_metrics"] = platform
        execution.result = result
        execution.error = error
        job.status = terminal_status or ("FAILED" if error is not None else "SUCCEEDED")


def cancellation_requested(sessions, job_id):
    with sessions() as session:
        return session.scalar(select(Job.status).where(Job.id == job_id)) == "CANCEL_REQUESTED"


class WorkerStopUnconfirmed(RuntimeError):
    pass


def confirm_stopped_and_fence(run_dir, timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        try:
            with job_lock(run_dir) as job_dir:
                (job_dir / ".current-attempt").write_text("cancelled")
                return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise WorkerStopUnconfirmed("Worker stop not confirmed; keeping cancellation pending") from None
            time.sleep(0.1)


def execute_claimed_job(sessions, claimed, gpu_hourly_rate_usd=None,
                        interrupt_after_checkpoint=None):
    import ray
    from post_training.worker import train_sft_worker

    job_id = claimed["job_id"]
    if cancellation_requested(sessions, job_id):
        confirm_stopped_and_fence(claimed["run_dir"])
        finish_job(sessions, job_id, terminal_status="CANCELLED", expected_run_dir=claimed["run_dir"],
                   gpu_hourly_rate_usd=gpu_hourly_rate_usd)
        return "CANCELLED"
    try:
        examples, validation, metadata = load_registered_dataset(
            claimed["dataset_uri"], claimed["tenant_id"])
        num_gpus = int(claimed["config"]["device"] == "cuda")
        ref = train_sft_worker.options(num_gpus=num_gpus).remote(
            claimed["config"], claimed["run_dir"], examples=examples,
            validation_examples=validation, dataset_metadata=metadata,
            context={"tenant_id": claimed["tenant_id"], "job_id": job_id,
                     "attempt": f"{claimed.get('attempt', 1):03d}", "ray_submission_requested_at": utcnow().isoformat()},
            resume_checkpoint=claimed.get("resume_checkpoint"),
            interrupt_after_checkpoint=interrupt_after_checkpoint)
    except Exception as exc:
        finish_job(sessions, job_id, error=f"Dispatch failed: {type(exc).__name__}",
                   expected_run_dir=claimed["run_dir"])
        return "FAILED"
    # If a DB write fails after submission, do not report a training failure or
    # release the slot: execution may still be active. Stop for manual inspection.
    mark_dispatched(sessions, job_id, ref.task_id().hex(), claimed["run_dir"])
    try:
        # Poll rather than block in ray.get, so API cancellation can reach live work.
        while True:
            ready, _ = ray.wait([ref], timeout=0.25)
            if ready:
                break  # A completed result wins a concurrent cancellation request.
            if cancellation_requested(sessions, job_id):
                ray.cancel(ref, force=True)
                try:
                    ray.get(ref)
                except ray.exceptions.RayError:
                    pass
                confirm_stopped_and_fence(claimed["run_dir"])
                finish_job(sessions, job_id, terminal_status="CANCELLED", expected_run_dir=claimed["run_dir"],
                           gpu_hourly_rate_usd=gpu_hourly_rate_usd)
                return "CANCELLED"
        result = ray.get(ref)
    except KeyboardInterrupt:
        ray.cancel(ref, force=True)
        try:
            ray.get(ref)
        except ray.exceptions.RayError:
            pass
        finish_job(sessions, job_id, error="Dispatcher interrupted; worker cancellation requested",
                   terminal_status="INTERRUPTED", expected_run_dir=claimed["run_dir"],
                   gpu_hourly_rate_usd=gpu_hourly_rate_usd)
        raise
    except WorkerStopUnconfirmed:
        raise
    except Exception as exc:
        if cancellation_requested(sessions, job_id):
            confirm_stopped_and_fence(claimed["run_dir"])
            finish_job(sessions, job_id, terminal_status="CANCELLED", expected_run_dir=claimed["run_dir"],
                       gpu_hourly_rate_usd=gpu_hourly_rate_usd)
            return "CANCELLED"
        failure_result = None
        path = Path(claimed["run_dir"]) / "telemetry.json"
        if path.exists():
            failure_result = {"telemetry": json.loads(path.read_text())}
        interrupted = isinstance(exc, TrainingInterrupted) or isinstance(getattr(exc, "cause", None), TrainingInterrupted)
        status = "INTERRUPTED" if interrupted else "FAILED"
        finish_job(sessions, job_id, result=failure_result, error=f"Worker stopped: {type(exc).__name__}",
                   gpu_hourly_rate_usd=gpu_hourly_rate_usd, terminal_status=status,
                   expected_run_dir=claimed["run_dir"])
        return status
    finish_job(sessions, job_id, result=result, gpu_hourly_rate_usd=gpu_hourly_rate_usd,
               expected_run_dir=claimed["run_dir"])
    return "SUCCEEDED"


def prepare_recovery(sessions, job_id):
    """Admin operation: verify no live worker, archive the old attempt, requeue."""
    with sessions.begin() as session:
        if session.bind.dialect.name != "postgresql":
            raise RuntimeError("Recovery requires PostgreSQL")
        session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": SLOT_LOCK_ID})
        job = session.get(Job, job_id, with_for_update=True)
        if job is None or job.status not in ("CLAIMED", "RUNNING", "FAILED", "INTERRUPTED"):
            raise ValueError("Job must be interrupted, failed, or left active by a stopped dispatcher")
        execution = session.get(JobExecution, job_id)
        if execution is None:
            raise ValueError("No execution to recover")
        recovery = session.get(JobRecovery, job_id)
        with job_lock(execution.run_dir) as job_dir:
            # The lock proves no guarded worker is training. Invalidate pending old
            # tasks before releasing it; those tasks will refuse a stale marker.
            try:
                checkpoint = latest_checkpoint(execution.run_dir)
            except ValueError:
                if recovery is None:
                    raise
                checkpoint = Path(recovery.checkpoint)
                validate_checkpoint(checkpoint)
            if (Path(execution.run_dir) / "result.json").exists():
                raise ValueError("Run has a final result; reconcile it rather than retraining")
            (job_dir / ".current-attempt").write_text("recovery-pending")
            archive = {column.name: getattr(execution, column.name)
                       for column in JobExecution.__table__.columns}
            for key, value in list(archive.items()):
                if hasattr(value, "isoformat"):
                    archive[key] = value.isoformat()
            archive["status"] = job.status
            if recovery:
                recovery.attempt += 1
                recovery.checkpoint = str(checkpoint)
                recovery.history = [*recovery.history, archive]
            else:
                recovery = JobRecovery(job_id=job_id, attempt=2, checkpoint=str(checkpoint), history=[archive])
                session.add(recovery)
            session.delete(execution)
            session.flush()
            job.status = "QUEUED"
            return {"job_id": job_id, "attempt": recovery.attempt, "checkpoint": str(checkpoint)}
