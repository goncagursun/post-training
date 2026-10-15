"""Durable platform metadata. Training files live outside the database."""
from datetime import datetime, timezone
import os
import uuid

from sqlalchemy import JSON, DateTime, ForeignKey, ForeignKeyConstraint, String, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, relationship


def utcnow():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)


class APIKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DatasetRecord(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("id", "tenant_id"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    # An administrator registers this reference. Customers submit IDs, never paths.
    uri: Mapped[str] = mapped_column(String(2048))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (ForeignKeyConstraint(
        ["dataset_id", "tenant_id"], ["datasets.id", "datasets.tenant_id"]),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    dataset_id: Mapped[str] = mapped_column(String(64))
    base_model_id: Mapped[str] = mapped_column(String(128))
    config: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="QUEUED", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    execution: Mapped["JobExecution | None"] = relationship(lazy="selectin", uselist=False)


class JobExecution(Base):
    """One attempt per job for this increment; stored separately to preserve existing jobs."""
    __tablename__ = "job_executions"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), primary_key=True)
    run_dir: Mapped[str] = mapped_column(String(2048))
    dataset_uri: Mapped[str] = mapped_column(String(2048))
    config: Mapped[dict] = mapped_column(JSON)
    claimed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ray_task_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(String(1024), nullable=True)


def make_engine(url=None):
    url = url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("Set DATABASE_URL; see docs/api_quickstart.md")
    return create_engine(url, pool_pre_ping=True)


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
