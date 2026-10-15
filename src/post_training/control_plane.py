"""Submission contract and tenant-scoped job operations."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from post_training.database import DatasetRecord, Job

SUPPORTED_MODELS = {"smollm2-135m": "HuggingFaceTB/SmolLM2-135M"}


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    dataset_id: str = Field(min_length=1, max_length=64)
    base_model_id: Literal["smollm2-135m"]
    learning_rate: float = Field(default=0.001, gt=0, le=0.01, allow_inf_nan=False)
    epochs: int = Field(default=1, ge=1, le=10)
    lora_rank: Literal[4, 8, 16] = 8


class ExecutionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    claimed_at: datetime
    dispatched_at: datetime | None
    finished_at: datetime | None
    ray_task_id: str | None
    result: dict | None
    error: str | None


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    dataset_id: str
    base_model_id: str
    config: dict
    status: str
    created_at: datetime
    execution: ExecutionResponse | None = None


def create_queued_job(session, tenant_id, request):
    dataset = session.scalar(select(DatasetRecord).where(
        DatasetRecord.id == request.dataset_id, DatasetRecord.tenant_id == tenant_id))
    if dataset is None:
        return None
    job = Job(tenant_id=tenant_id, dataset_id=dataset.id,
              base_model_id=request.base_model_id,
              config={"method": "sft", "model_name": SUPPORTED_MODELS[request.base_model_id],
                      "learning_rate": request.learning_rate, "num_train_epochs": request.epochs,
                      "lora_rank": request.lora_rank}, status="QUEUED")
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def get_owned_job(session, tenant_id, job_id):
    return session.scalar(select(Job).where(Job.id == job_id, Job.tenant_id == tenant_id))
