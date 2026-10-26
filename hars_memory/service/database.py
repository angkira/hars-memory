"""Portable SQLite/PostgreSQL repository with lease-based job recovery."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Engine, and_, create_engine, event, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from hars_memory.service.models import Base, Index, IndexJob, IndexVersion, JobKind, JobStatus


class ServiceDatabaseError(RuntimeError):
    """Base persistence error."""


class IdempotencyConflictError(ServiceDatabaseError):
    pass


class InvalidTransitionError(ServiceDatabaseError):
    pass


class LeaseLostError(ServiceDatabaseError):
    pass


class StaleBaseVersionError(ServiceDatabaseError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_request(request: dict[str, Any]) -> tuple[str, str]:
    value = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return value, hashlib.sha256(value.encode("utf-8")).hexdigest()


class ServiceDatabase:
    """Synchronous repository safe to call from API/worker thread pools."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        kwargs: dict[str, Any] = {"pool_pre_ping": True, "echo": echo}
        if url.startswith("sqlite:"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        self.engine: Engine = create_engine(url, **kwargs)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", self._configure_sqlite)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    @staticmethod
    def _configure_sqlite(connection: Any, _record: Any) -> None:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    def create_index(self, *, tenant_id: str, name: str) -> Index:
        with self._sessions.begin() as session:
            existing = session.scalar(
                select(Index).where(Index.tenant_id == tenant_id, Index.name == name)
            )
            if existing is not None:
                return existing
            index = Index(tenant_id=tenant_id, name=name)
            session.add(index)
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                with self._sessions() as retry_session:
                    found = retry_session.scalar(
                        select(Index).where(Index.tenant_id == tenant_id, Index.name == name)
                    )
                    if found is None:
                        raise
                    return found
            return index

    def get_index(self, *, tenant_id: str, index_id: str) -> Index | None:
        with self._sessions() as session:
            return session.scalar(
                select(Index).where(Index.id == index_id, Index.tenant_id == tenant_id)
            )

    def get_version(self, *, tenant_id: str, version_id: str) -> IndexVersion | None:
        with self._sessions() as session:
            return session.scalar(
                select(IndexVersion)
                .join(Index, Index.id == IndexVersion.index_id)
                .where(IndexVersion.id == version_id, Index.tenant_id == tenant_id)
            )

    def get_version_by_number(
        self, *, tenant_id: str, index_id: str, version_number: int
    ) -> IndexVersion | None:
        with self._sessions() as session:
            return session.scalar(
                select(IndexVersion)
                .join(Index, Index.id == IndexVersion.index_id)
                .where(
                    IndexVersion.index_id == index_id,
                    IndexVersion.version_number == version_number,
                    Index.tenant_id == tenant_id,
                )
            )

    def create_job(
        self,
        *,
        tenant_id: str,
        index_id: str,
        kind: JobKind,
        idempotency_key: str,
        request: dict[str, Any],
        input_artifact_uri: str,
        input_artifact_sha256: str,
        input_artifact_size_bytes: int = 0,
        base_version_id: str | None = None,
    ) -> IndexJob:
        request_json, request_sha256 = _canonical_request(request)
        with self._sessions.begin() as session:
            # Idempotency wins over current index state. A client may retry a
            # create request after its original job has already published v1;
            # validating "create against an active index" first would turn a
            # successful retry into a false 409 instead of returning the
            # original terminal job.
            existing = session.scalar(
                select(IndexJob).where(
                    IndexJob.tenant_id == tenant_id,
                    IndexJob.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise IdempotencyConflictError(
                        "Idempotency key was already used with a different request"
                    )
                return existing

            index = session.scalar(
                select(Index).where(Index.id == index_id, Index.tenant_id == tenant_id)
            )
            if index is None:
                raise ServiceDatabaseError(f"Index {index_id!r} does not exist for tenant")
            if kind is JobKind.EXTEND and base_version_id != index.active_version_id:
                raise StaleBaseVersionError(
                    f"Base version {base_version_id!r} is not active version {index.active_version_id!r}"
                )
            if kind is JobKind.CREATE and index.active_version_id is not None:
                raise InvalidTransitionError("Cannot create an already-published index; use extend")
            job = IndexJob(
                tenant_id=tenant_id,
                index_id=index_id,
                kind=kind,
                base_version_id=base_version_id,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                request_json=request_json,
                input_artifact_uri=input_artifact_uri,
                input_artifact_sha256=input_artifact_sha256,
                input_artifact_size_bytes=input_artifact_size_bytes,
            )
            session.add(job)
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                with self._sessions() as retry_session:
                    raced = retry_session.scalar(
                        select(IndexJob).where(
                            IndexJob.tenant_id == tenant_id,
                            IndexJob.idempotency_key == idempotency_key,
                        )
                    )
                    if raced is None:
                        raise
                    if raced.request_sha256 != request_sha256:
                        raise IdempotencyConflictError(
                            "Idempotency key was already used with a different request"
                        )
                    return raced
            return job

    def get_job(self, *, tenant_id: str, job_id: str) -> IndexJob | None:
        with self._sessions() as session:
            return session.scalar(
                select(IndexJob).where(IndexJob.id == job_id, IndexJob.tenant_id == tenant_id)
            )

    def get_job_by_idempotency(
        self, *, tenant_id: str, idempotency_key: str
    ) -> IndexJob | None:
        """Return a prior request before mutable index-state validation."""
        with self._sessions() as session:
            return session.scalar(
                select(IndexJob).where(
                    IndexJob.tenant_id == tenant_id,
                    IndexJob.idempotency_key == idempotency_key,
                )
            )

    def claim_next_job(self, *, worker_id: str, lease_seconds: int = 60) -> IndexJob | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = _utc_now()
        expires = now + timedelta(seconds=lease_seconds)
        claimable = or_(
            IndexJob.status == JobStatus.QUEUED,
            and_(IndexJob.status == JobStatus.RUNNING, IndexJob.lease_expires_at < now),
        )
        # Conditional UPDATE makes the claim race-safe on SQLite too. PostgreSQL's
        # SKIP LOCKED avoids contenders waiting behind the same candidate.
        for _ in range(8):
            with self._sessions.begin() as session:
                query = (
                    select(IndexJob.id)
                    .where(claimable, IndexJob.cancel_requested.is_(False))
                    .order_by(IndexJob.created_at, IndexJob.id)
                    .limit(1)
                )
                if self.engine.dialect.name == "postgresql":
                    query = query.with_for_update(skip_locked=True)
                job_id = session.scalar(query)
                if job_id is None:
                    return None
                values: dict[str, Any] = {
                    "status": JobStatus.RUNNING,
                    "worker_id": worker_id,
                    "lease_expires_at": expires,
                    "attempt_count": IndexJob.attempt_count + 1,
                }
                job = session.get(IndexJob, job_id)
                if job is not None and job.started_at is None:
                    values["started_at"] = now
                result = session.execute(
                    update(IndexJob)
                    .where(IndexJob.id == job_id, claimable)
                    .values(**values)
                    .execution_options(synchronize_session=False)
                )
                if result.rowcount == 1:
                    session.flush()
                    session.expire_all()
                    return session.get(IndexJob, job_id)
        return None

    def renew_lease(self, *, job_id: str, worker_id: str, lease_seconds: int = 60) -> None:
        now = _utc_now()
        with self._sessions.begin() as session:
            result = session.execute(
                update(IndexJob)
                .where(
                    IndexJob.id == job_id,
                    IndexJob.status == JobStatus.RUNNING,
                    IndexJob.worker_id == worker_id,
                    IndexJob.lease_expires_at >= now,
                )
                .values(lease_expires_at=now + timedelta(seconds=lease_seconds))
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                raise LeaseLostError(f"Worker {worker_id!r} no longer owns job {job_id}")

    def request_cancel(self, *, tenant_id: str, job_id: str) -> IndexJob:
        now = _utc_now()
        with self._sessions.begin() as session:
            job = session.scalar(
                select(IndexJob).where(IndexJob.id == job_id, IndexJob.tenant_id == tenant_id)
            )
            if job is None:
                raise ServiceDatabaseError(f"Job {job_id!r} does not exist for tenant")
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.cancel_requested = True
                job.finished_at = now
            elif job.status == JobStatus.RUNNING:
                job.cancel_requested = True
            return job

    def finish_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        status: JobStatus,
        error_message: str | None = None,
    ) -> IndexJob:
        if status not in {JobStatus.FAILED, JobStatus.CANCELLED}:
            raise InvalidTransitionError("finish_job only accepts failed or cancelled")
        now = _utc_now()
        with self._sessions.begin() as session:
            job = session.get(IndexJob, job_id)
            if job is None or job.status != JobStatus.RUNNING or job.worker_id != worker_id:
                raise LeaseLostError(f"Worker {worker_id!r} no longer owns job {job_id}")
            job.status = status
            job.error_message = error_message
            job.finished_at = now
            job.lease_expires_at = None
            return job

    def publish_version(
        self,
        *,
        job_id: str,
        worker_id: str,
        artifact_uri: str,
        artifact_sha256: str,
        artifact_size_bytes: int,
        manifest: dict[str, Any],
    ) -> IndexVersion:
        """Atomically insert an immutable version, activate it, and succeed its job."""
        now = _utc_now()
        with self._sessions.begin() as session:
            job = session.get(IndexJob, job_id)
            if (
                job is None
                or job.status != JobStatus.RUNNING
                or job.worker_id != worker_id
                or job.lease_expires_at is None
                or _timestamp_before(job.lease_expires_at, now)
                or job.cancel_requested
            ):
                raise LeaseLostError(f"Worker {worker_id!r} no longer owns job {job_id}")
            index = session.get(Index, job.index_id)
            if index is None:
                raise ServiceDatabaseError(f"Index {job.index_id!r} disappeared")
            if index.active_version_id != job.base_version_id:
                raise StaleBaseVersionError("Active index version changed before publication")
            latest = session.scalar(
                select(IndexVersion.version_number)
                .where(IndexVersion.index_id == index.id)
                .order_by(IndexVersion.version_number.desc())
                .limit(1)
            )
            version = IndexVersion(
                id=job.target_version_id,
                index_id=index.id,
                version_number=(latest or 0) + 1,
                parent_version_id=job.base_version_id,
                artifact_uri=artifact_uri,
                artifact_sha256=artifact_sha256,
                artifact_size_bytes=artifact_size_bytes,
                manifest_json=json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            )
            session.add(version)
            session.flush()
            index.active_version_id = version.id
            job.status = JobStatus.SUCCEEDED
            job.finished_at = now
            job.lease_expires_at = None
            return version


def _timestamp_before(left: datetime, right: datetime) -> bool:
    """Compare SQLite-naive and PostgreSQL-aware timestamps consistently."""
    if left.tzinfo is None:
        left = left.replace(tzinfo=timezone.utc)
    if right.tzinfo is None:
        right = right.replace(tzinfo=timezone.utc)
    return left < right


__all__ = [
    "IdempotencyConflictError",
    "InvalidTransitionError",
    "LeaseLostError",
    "ServiceDatabase",
    "ServiceDatabaseError",
    "StaleBaseVersionError",
]
