from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import update

from hars_memory.service.database import (
    IdempotencyConflictError,
    LeaseLostError,
    ServiceDatabase,
)
from hars_memory.service.models import IndexJob, JobKind, JobStatus


@pytest.fixture
def database(tmp_path: Path) -> ServiceDatabase:
    db = ServiceDatabase(f"sqlite:///{tmp_path / 'service.sqlite3'}")
    db.create_schema()
    yield db
    db.close()


def _job(database: ServiceDatabase, tenant: str = "tenant-a") -> IndexJob:
    index = database.create_index(tenant_id=tenant, name="docs")
    return database.create_job(
        tenant_id=tenant,
        index_id=index.id,
        kind=JobKind.CREATE,
        idempotency_key="request-1",
        request={"files": ["a.md"]},
        input_artifact_uri="file:///input.tar.gz",
        input_artifact_sha256="a" * 64,
        input_artifact_size_bytes=12,
    )


def test_job_creation_is_tenant_scoped_and_idempotent(database: ServiceDatabase) -> None:
    first = _job(database)
    again = database.create_job(
        tenant_id="tenant-a",
        index_id=first.index_id,
        kind=JobKind.CREATE,
        idempotency_key="request-1",
        request={"files": ["a.md"]},
        input_artifact_uri="file:///input.tar.gz",
        input_artifact_sha256="a" * 64,
    )
    assert again.id == first.id
    assert database.get_job(tenant_id="tenant-b", job_id=first.id) is None
    with pytest.raises(IdempotencyConflictError):
        database.create_job(
            tenant_id="tenant-a",
            index_id=first.index_id,
            kind=JobKind.CREATE,
            idempotency_key="request-1",
            request={"files": ["different.md"]},
            input_artifact_uri="file:///other.tar.gz",
            input_artifact_sha256="b" * 64,
        )


def test_expired_job_is_reclaimed_and_old_worker_loses_lease(database: ServiceDatabase) -> None:
    created = _job(database)
    claimed = database.claim_next_job(worker_id="worker-1", lease_seconds=60)
    assert claimed is not None and claimed.id == created.id and claimed.attempt_count == 1
    with database._sessions.begin() as session:
        session.execute(
            update(IndexJob)
            .where(IndexJob.id == created.id)
            .values(lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        )
    reclaimed = database.claim_next_job(worker_id="worker-2", lease_seconds=60)
    assert reclaimed is not None and reclaimed.id == created.id and reclaimed.attempt_count == 2
    with pytest.raises(LeaseLostError):
        database.renew_lease(job_id=created.id, worker_id="worker-1")


def test_publish_is_atomic_and_version_rows_are_snapshots(database: ServiceDatabase) -> None:
    created = _job(database)
    claimed = database.claim_next_job(worker_id="worker")
    assert claimed is not None
    version = database.publish_version(
        job_id=created.id,
        worker_id="worker",
        artifact_uri="file:///result.tar.gz",
        artifact_sha256="c" * 64,
        artifact_size_bytes=42,
        manifest={"format": "hars-index-bundle.v1"},
    )
    index = database.get_index(tenant_id="tenant-a", index_id=created.index_id)
    job = database.get_job(tenant_id="tenant-a", job_id=created.id)
    assert index is not None and index.active_version_id == version.id
    assert job is not None and job.status == JobStatus.SUCCEEDED
    assert version.version_number == 1


def test_queued_cancel_is_terminal_and_never_claimed(database: ServiceDatabase) -> None:
    created = _job(database)
    cancelled = database.request_cancel(tenant_id="tenant-a", job_id=created.id)
    assert cancelled.status == JobStatus.CANCELLED
    assert database.claim_next_job(worker_id="worker") is None


def test_postgresql_url_is_accepted_without_connecting() -> None:
    database = ServiceDatabase("postgresql+psycopg://user:password@db/hars_memory")
    try:
        assert database.engine.dialect.name == "postgresql"
    finally:
        database.close()
