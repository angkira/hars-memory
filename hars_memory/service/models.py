"""SQLAlchemy persistence models for indexes, immutable versions, and jobs."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index as SqlIndex,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Declarative base for the service database."""


class JobKind(str, enum.Enum):
    CREATE = "create"
    EXTEND = "extend"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


class Index(Base):
    __tablename__ = "service_indexes"
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_index_tenant_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    active_version_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("service_index_versions.id", use_alter=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    versions: Mapped[list[IndexVersion]] = relationship(
        back_populates="index", foreign_keys="IndexVersion.index_id"
    )


class IndexVersion(Base):
    """Published index snapshot.

    Rows are insert-only. The repository exposes no update operation; publication
    inserts this row and switches ``Index.active_version_id`` in one transaction.
    """

    __tablename__ = "service_index_versions"
    __table_args__ = (
        UniqueConstraint("index_id", "version_number", name="uq_index_version_number"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    index_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("service_indexes.id"), nullable=False, index=True
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_version_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("service_index_versions.id"), nullable=True
    )
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    index: Mapped[Index] = relationship(back_populates="versions", foreign_keys=[index_id])


class IndexJob(Base):
    __tablename__ = "service_index_jobs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_job_tenant_idempotency"),
        SqlIndex("ix_jobs_claim", "status", "lease_expires_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    index_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("service_indexes.id"), nullable=False, index=True
    )
    kind: Mapped[JobKind] = mapped_column(Enum(JobKind, native_enum=False), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False), nullable=False, default=JobStatus.QUEUED
    )
    base_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    target_version_id: Mapped[str] = mapped_column(String(36), nullable=False, default=new_id)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    request_json: Mapped[str] = mapped_column(Text, nullable=False)
    input_artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    input_artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    input_artifact_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    worker_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


__all__ = ["Base", "Index", "IndexJob", "IndexVersion", "JobKind", "JobStatus"]
