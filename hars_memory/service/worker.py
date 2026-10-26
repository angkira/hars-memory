"""Durable polling worker for index jobs.

The repository owns state transitions and leases; this module owns scratch
space, engine dispatch, artifact publication ordering, and restart recovery.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from hars_memory.service.database import LeaseLostError
from hars_memory.service.engines import (
    CorpusEngine,
    EngineRequest,
    IndexEngine,
    JobCancelled,
    LightRAGEngine,
)

logger = logging.getLogger(__name__)


class Repository(Protocol):
    def claim_next_job(self) -> object | None: ...
    def requeue_running_jobs(self) -> int: ...
    def mark_job_succeeded(
        self, job_id: str, result_version: int, result_uri: str, manifest: dict[str, object]
    ) -> object: ...
    def mark_job_failed(self, job_id: str, error: str) -> object: ...
    def get_latest_index_version(self, tenant_id: str, index_id: str) -> object | None: ...


class ArtifactStore(Protocol):
    def materialize_input(self, uri: str, destination: Path) -> object: ...
    def materialize_result(self, uri: str, destination: Path) -> object: ...
    def put_result(self, index_id: str, version: int, path: Path) -> object: ...


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    scratch_root: Path | None = None
    poll_seconds: float = 1.0
    recover_on_start: bool = True
    worker_id: str = ""
    lease_seconds: int = 60
    heartbeat_seconds: float = 20.0


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_path(result: object, fallback: Path) -> Path:
    if isinstance(result, (str, Path)):
        return Path(result)
    path = _field(result, "path")
    return Path(path) if path else fallback


def _as_uri(result: object) -> str:
    if isinstance(result, str):
        return result
    for name in ("uri", "artifact_uri", "result_uri", "key"):
        value = _field(result, name)
        if value:
            return str(value)
    raise RuntimeError("artifact store put_result() returned no URI")


class DurableWorker:
    def __init__(
        self,
        repository: Repository,
        artifact_store: ArtifactStore,
        *,
        engines: Mapping[str, IndexEngine] | None = None,
        config: WorkerConfig | None = None,
    ) -> None:
        self.repository = repository
        self.artifact_store = artifact_store
        self.engines = dict(engines or {"corpus": CorpusEngine(), "lightrag": LightRAGEngine()})
        self.config = config or WorkerConfig()
        self.worker_id = self.config.worker_id or f"worker-{uuid.uuid4()}"
        self._stop = threading.Event()
        self._index_locks: dict[str, threading.Lock] = {}
        self._index_locks_guard = threading.Lock()
        self._started = False

    def stop(self) -> None:
        self._stop.set()

    def recover(self) -> int:
        """Return expired/running jobs to the queue after a process restart."""
        requeue = getattr(self.repository, "requeue_running_jobs", None)
        # ServiceDatabase claims expired RUNNING leases directly, so it needs
        # no eager state rewrite. Simpler repositories may provide an explicit
        # recovery hook.
        count = int(requeue()) if callable(requeue) else 0
        logger.info("Requeued %d interrupted index job(s)", count)
        self._started = True
        return count

    def _lock_for(self, tenant_id: str, index_id: str) -> threading.Lock:
        key = f"{tenant_id}:{index_id}"
        with self._index_locks_guard:
            return self._index_locks.setdefault(key, threading.Lock())

    def _cancel_requested(self, job: object) -> bool:
        # Repositories may expose a live lookup; otherwise the claimed object's
        # flag still gives correct cancellation before publication.
        getter = getattr(self.repository, "is_cancel_requested", None)
        if callable(getter):
            return bool(getter(str(_field(job, "id"))))
        get_job = getattr(self.repository, "get_job", None)
        if callable(get_job):
            try:
                current = get_job(
                    tenant_id=str(_field(job, "tenant_id")), job_id=str(_field(job, "id"))
                )
            except TypeError:
                current = None
            if current is not None:
                return bool(_field(current, "cancel_requested", False))
        return bool(_field(job, "cancel_requested", False))

    def run_once(self) -> bool:
        if not self._started:
            if self.config.recover_on_start:
                self.recover()
            else:
                self._started = True
        try:
            job = self.repository.claim_next_job(
                worker_id=self.worker_id, lease_seconds=self.config.lease_seconds
            )
        except TypeError:  # structural compatibility for small/local adapters
            job = self.repository.claim_next_job()
        if job is None:
            return False
        self._execute(job)
        return True

    def run_forever(self) -> None:
        if not self._started and self.config.recover_on_start:
            self.recover()
        while not self._stop.is_set():
            try:
                progressed = self.run_once()
            except Exception:  # noqa: BLE001 -- keep the service alive between jobs
                logger.exception("Unhandled index worker error")
                progressed = True
            if not progressed:
                self._stop.wait(self.config.poll_seconds)

    def _execute(self, job: object) -> None:
        job_id = str(_field(job, "id"))
        heartbeat_stop = threading.Event()
        lease_lost = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat,
            args=(job_id, heartbeat_stop, lease_lost),
            daemon=True,
        )
        heartbeat.start()
        try:
            self._execute_under_lease(job, lease_lost)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=max(self.config.heartbeat_seconds * 2, 1.0))

    def _execute_under_lease(self, job: object, lease_lost: threading.Event) -> None:
        job_id = str(_field(job, "id"))
        tenant_id = str(_field(job, "tenant_id"))
        index_id = str(_field(job, "index_id"))
        operation_value = _field(job, "operation", _field(job, "kind"))
        operation = str(getattr(operation_value, "value", operation_value))
        request_json = _field(job, "request_json", "{}")
        try:
            request_data = json.loads(str(request_json))
        except json.JSONDecodeError:
            request_data = {}
        engine_name = str(_field(job, "engine", request_data.get("engine", "corpus")))
        strategy_data = request_data.get("index_strategy", {})
        strategy_options: dict[str, object] = {}
        if isinstance(strategy_data, dict):
            raw_options = strategy_data.get("options", {})
            if isinstance(raw_options, dict):
                strategy_options = dict(raw_options)
            strategy_options["name"] = str(strategy_data.get("name", "default"))
        input_uri = str(_field(job, "input_uri", _field(job, "input_artifact_uri", "")))
        engine = self.engines.get(engine_name)
        if engine is None:
            self.repository.mark_job_failed(job_id, f"unknown index engine: {engine_name}")
            return

        lock = self._lock_for(tenant_id, index_id)
        with lock:
            try:
                if lease_lost.is_set():
                    raise LeaseLostError(f"lease lost for job {job_id}")
                if self._cancel_requested(job):
                    raise JobCancelled("job cancelled before execution")
                base_version_id = _field(job, "base_version_id")
                latest_getter = getattr(self.repository, "get_latest_index_version", None)
                if callable(latest_getter):
                    latest = latest_getter(tenant_id, index_id)
                elif base_version_id is not None:
                    latest = self.repository.get_version(
                        tenant_id=tenant_id, version_id=str(base_version_id)
                    )
                else:
                    latest = None
                base_artifact_uri: str | None = None
                base_number = 0
                if latest is not None:
                    base_number = int(_field(latest, "version", _field(latest, "version_number", 0)))
                    base_artifact_uri = str(
                        _field(latest, "artifact_uri", _field(latest, "result_uri", ""))
                    ) or None
                requested_base = _field(job, "base_version")
                if operation == "create" and latest is not None:
                    raise RuntimeError("create job targets an index that already has a version")
                if operation == "extend":
                    if latest is None or base_artifact_uri is None:
                        raise RuntimeError("extend job has no published base version")
                    if requested_base is not None and int(requested_base) != base_number:
                        raise RuntimeError(
                            f"base version conflict: requested {requested_base}, latest is {base_number}"
                        )
                version = base_number + 1

                scratch_parent = self.config.scratch_root
                if scratch_parent is not None:
                    scratch_parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=f"hars-job-{job_id}-", dir=scratch_parent
                ) as temporary:
                    workspace = Path(temporary)
                    input_target = workspace / "input"
                    input_sha = _field(job, "input_artifact_sha256")
                    try:
                        input_result = self.artifact_store.materialize_input(
                            input_uri, input_target, str(input_sha) if input_sha else None
                        )
                    except TypeError:
                        input_result = self.artifact_store.materialize_input(input_uri, input_target)
                    input_path = _as_path(input_result, input_target)
                    if lease_lost.is_set():
                        raise LeaseLostError(f"lease lost for job {job_id}")
                    base_path: Path | None = None
                    if base_artifact_uri is not None:
                        base_target = workspace / "base.tar.gz"
                        base_sha = _field(latest, "artifact_sha256")
                        try:
                            base_result = self.artifact_store.materialize_result(
                                base_artifact_uri, base_target,
                                str(base_sha) if base_sha else None,
                            )
                        except TypeError:
                            base_result = self.artifact_store.materialize_result(
                                base_artifact_uri, base_target
                            )
                        base_path = _as_path(base_result, base_target)
                    if lease_lost.is_set():
                        raise LeaseLostError(f"lease lost for job {job_id}")
                    output_path = workspace / f"{index_id}-v{version}.tar.gz"
                    request = EngineRequest(
                        job_id=job_id,
                        tenant_id=tenant_id,
                        index_id=index_id,
                        operation=operation,
                        input_path=input_path,
                        output_path=output_path,
                        workspace=workspace / "engine",
                        base_artifact=base_path,
                        version=version,
                        strategy=strategy_options,  # validated by the selected engine
                        cancel_requested=lambda: lease_lost.is_set()
                        or self._cancel_requested(job),
                    )
                    request.workspace.mkdir(parents=True, exist_ok=False)
                    result = engine.build(request)
                    if lease_lost.is_set():
                        raise LeaseLostError(f"lease lost for job {job_id}")
                    if self._cancel_requested(job):
                        raise JobCancelled("job cancelled before artifact publication")
                    target_version_id = str(_field(job, "target_version_id", job_id))
                    put_file = getattr(self.artifact_store, "put_file", None)
                    if callable(put_file):
                        stored = put_file(
                            f"indexes/{index_id}/candidates/{target_version_id}.tar.gz",
                            result.artifact_path,
                        )
                    else:
                        stored = self.artifact_store.put_result(
                            index_id, version, result.artifact_path
                        )
                    result_uri = _as_uri(stored)
                    if lease_lost.is_set():
                        raise LeaseLostError(f"lease lost for job {job_id}")
                    if self._cancel_requested(job):
                        # Object may be orphaned, but no immutable version is
                        # published. Store lifecycle cleanup can reap it safely.
                        raise JobCancelled("job cancelled before version commit")
                    publisher = getattr(self.repository, "publish_version", None)
                    if callable(publisher):
                        publisher(
                            job_id=job_id,
                            worker_id=self.worker_id,
                            artifact_uri=result_uri,
                            artifact_sha256=str(_field(stored, "sha256", result.manifest.get("artifact_sha256", ""))),
                            artifact_size_bytes=int(_field(stored, "size_bytes", result.artifact_path.stat().st_size)),
                            manifest=result.manifest,
                        )
                    else:
                        self.repository.mark_job_succeeded(
                            job_id, version, result_uri, result.manifest
                        )
            except LeaseLostError:
                # A different worker may now own the job. Never write a terminal
                # state after ownership has been lost.
                logger.warning("Lease lost for index job %s; leaving recovery to its owner", job_id)
            except JobCancelled as exc:
                marker = getattr(self.repository, "mark_job_cancelled", None)
                if callable(marker):
                    marker(job_id, str(exc))
                elif callable(getattr(self.repository, "finish_job", None)):
                    from hars_memory.service.models import JobStatus

                    try:
                        self.repository.finish_job(
                            job_id=job_id, worker_id=self.worker_id,
                            status=JobStatus.CANCELLED, error_message=str(exc),
                        )
                    except LeaseLostError:
                        logger.warning("Lease lost while cancelling index job %s", job_id)
                else:
                    self.repository.mark_job_failed(job_id, str(exc))
            except Exception as exc:  # noqa: BLE001 -- terminal job record is the boundary
                logger.exception("Index job %s failed", job_id)
                finisher = getattr(self.repository, "finish_job", None)
                if callable(finisher):
                    from hars_memory.service.models import JobStatus

                    try:
                        finisher(
                            job_id=job_id, worker_id=self.worker_id,
                            status=JobStatus.FAILED, error_message=str(exc)[:4000],
                        )
                    except LeaseLostError:
                        logger.warning("Lease lost while failing index job %s", job_id)
                else:
                    self.repository.mark_job_failed(job_id, str(exc)[:4000])

    def _heartbeat(
        self, job_id: str, stop: threading.Event, lease_lost: threading.Event
    ) -> None:
        renew = getattr(self.repository, "renew_lease", None)
        if not callable(renew):
            return
        while not stop.wait(self.config.heartbeat_seconds):
            try:
                renew(
                    job_id=job_id,
                    worker_id=self.worker_id,
                    lease_seconds=self.config.lease_seconds,
                )
            except Exception:  # engine cancellation/publish will observe lease loss
                logger.exception("Could not renew lease for index job %s", job_id)
                lease_lost.set()
                return


# Short alias for API/CLI callers that prefer the generic name.
Worker = DurableWorker

__all__ = ["ArtifactStore", "DurableWorker", "Repository", "Worker", "WorkerConfig"]
