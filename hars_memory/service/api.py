"""FastAPI routes for durable, tenant-scoped index jobs."""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import shutil
import tarfile
import tempfile
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Mapping, Protocol

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from hars_memory.service.artifacts import ArtifactRef, ArtifactStore
from hars_memory.service.auth import APIKeyAuthenticator, Principal
from hars_memory.service.database import (
    IdempotencyConflictError,
    InvalidTransitionError,
    ServiceDatabaseError,
    StaleBaseVersionError,
)
from hars_memory.service.models import JobKind
from hars_memory.service.validation import (
    UploadLimits,
    UploadValidationError,
    read_uploads,
)
from hars_memory.strategies import IndexStrategy, StrategyConfigurationError


class Repository(Protocol):
    def create_index(self, *, tenant_id: str, name: str) -> object: ...
    def get_index(self, *, tenant_id: str, index_id: str) -> object | None: ...
    def get_version(self, *, tenant_id: str, version_id: str) -> object | None: ...
    def create_job(self, **kwargs: object) -> object: ...
    def get_job(self, *, tenant_id: str, job_id: str) -> object | None: ...
    def request_cancel(self, *, tenant_id: str, job_id: str) -> object: ...


def _value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _job_json(job: object) -> dict[str, object]:
    return {
        "id": str(getattr(job, "id")),
        "status": _value(getattr(job, "status")),
        "operation": _value(getattr(job, "kind", getattr(job, "operation", ""))),
        "engine": _job_request(job).get("engine", ""),
        "index_id": str(getattr(job, "index_id")),
        "target_version_id": getattr(job, "target_version_id", None),
        "cancel_requested": bool(getattr(job, "cancel_requested", False)),
        "attempt_count": int(getattr(job, "attempt_count", 0)),
        "error": getattr(job, "error_message", None),
        "created_at": _iso(getattr(job, "created_at", None)),
        "started_at": _iso(getattr(job, "started_at", None)),
        "finished_at": _iso(getattr(job, "finished_at", None)),
    }


def _job_request(job: object) -> Mapping[str, Any]:
    raw = getattr(job, "request_json", "{}")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _iso(value: object) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None


def _version_json(version: object) -> dict[str, object]:
    raw_manifest = getattr(version, "manifest_json", "{}")
    try:
        manifest = json.loads(raw_manifest) if isinstance(raw_manifest, str) else raw_manifest
    except json.JSONDecodeError:
        manifest = {}
    return {
        "id": str(getattr(version, "id")),
        "index_id": str(getattr(version, "index_id")),
        "version": int(getattr(version, "version_number", getattr(version, "version", 0))),
        "artifact_sha256": str(getattr(version, "artifact_sha256")),
        "artifact_size_bytes": int(getattr(version, "artifact_size_bytes")),
        "manifest": manifest if isinstance(manifest, dict) else {},
        "created_at": _iso(getattr(version, "created_at", None)),
    }


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _write_input_bundle(files: Mapping[str, bytes], destination: Path) -> None:
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, content in sorted(files.items()):
                    info = tarfile.TarInfo(f"sources/{name}")
                    info.size = len(content)
                    info.mode = 0o600
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(content))


def create_app(
    repository: Repository,
    artifact_store: ArtifactStore,
    worker: object,
    authenticator: APIKeyAuthenticator,
    *,
    upload_limits: UploadLimits | None = None,
) -> FastAPI:
    """Build an app with injected stateful adapters for local/cloud parity."""

    limits = upload_limits or UploadLimits.from_env()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        start = getattr(worker, "start", None)
        if callable(start):
            start()
        try:
            yield
        finally:
            stop = getattr(worker, "stop", None)
            if callable(stop):
                stop()

    app = FastAPI(title="HARS Long-Term Memory Index Service", version="1", lifespan=lifespan)
    principal_dependency: Callable[..., Principal] = authenticator.dependency

    @app.get("/health")
    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/index-jobs", status_code=status.HTTP_202_ACCEPTED)
    async def submit_job(
        operation: str = Form(...),
        engine: str = Form("corpus"),
        strategy_name: str = Form("default"),
        strategy_options: str = Form("{}"),
        index_id: str | None = Form(None),
        index_name: str | None = Form(None),
        expected_version: int | None = Form(None),
        files: list[UploadFile] = File(...),
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
        principal: Principal = Depends(principal_dependency),
    ) -> dict[str, object]:
        if not idempotency_key or len(idempotency_key) > 255:
            raise _error(400, "invalid_idempotency_key", "A bounded Idempotency-Key is required")
        if operation not in {"create", "extend"}:
            raise _error(422, "invalid_operation", "operation must be create or extend")
        if engine not in {"corpus", "lightrag"}:
            raise _error(422, "invalid_engine", "engine must be corpus or lightrag")
        try:
            parsed_options = json.loads(strategy_options)
            if not isinstance(parsed_options, dict):
                raise StrategyConfigurationError("strategy_options must be a JSON object")
            index_strategy = IndexStrategy(
                name=strategy_name, engine=engine, options=parsed_options
            )
        except (json.JSONDecodeError, StrategyConfigurationError) as exc:
            raise _error(422, "invalid_index_strategy", str(exc)) from exc
        try:
            uploaded = await read_uploads(files, limits)
        except UploadValidationError as exc:
            raise _error(422, exc.code, str(exc)) from exc

        file_digests = {
            name: hashlib.sha256(content).hexdigest() for name, content in uploaded.items()
        }
        request: dict[str, object] = {
            "operation": operation,
            "engine": engine,
            "files": file_digests,
            "expected_version": expected_version,
            "index_strategy": index_strategy.to_dict(),
        }
        if operation == "create":
            if index_id is not None:
                raise _error(422, "unexpected_index_id", "index_id is not allowed for create")
            stable_name = index_name or (
                "index-"
                + hashlib.sha256(
                    f"{principal.tenant_id}\0{idempotency_key}".encode("utf-8")
                ).hexdigest()[:16]
            )
            request["index_name"] = stable_name
        else:
            if not index_id:
                raise _error(422, "missing_index_id", "index_id is required for extend")
            request["index_id"] = index_id

        # Resolve idempotency before reading mutable index state or creating an
        # index row. This makes terminal retries stable and prevents a
        # conflicting create retry from leaving an orphan index behind.
        canonical = json.dumps(
            request, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        request_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        prior_getter = getattr(repository, "get_job_by_idempotency", None)
        if callable(prior_getter):
            prior = prior_getter(
                tenant_id=principal.tenant_id, idempotency_key=idempotency_key
            )
            if prior is not None:
                if getattr(prior, "request_sha256", "") != request_sha:
                    raise _error(
                        409,
                        "idempotency_conflict",
                        "Idempotency key was already used with a different request",
                    )
                return _job_json(prior)

        if operation == "create":
            index = repository.create_index(tenant_id=principal.tenant_id, name=stable_name)
            index_id = str(getattr(index, "id"))
            base_version_id = None
            kind = JobKind.CREATE
        else:
            assert index_id is not None
            index = repository.get_index(tenant_id=principal.tenant_id, index_id=index_id)
            if index is None:
                raise _error(404, "index_not_found", "Index not found")
            base_version_id = getattr(index, "active_version_id", None)
            if base_version_id is None:
                raise _error(409, "index_has_no_version", "Index has no published version")
            base = repository.get_version(
                tenant_id=principal.tenant_id, version_id=str(base_version_id)
            )
            if base is None:
                raise _error(409, "index_version_missing", "Active index version is missing")
            active_number = int(getattr(base, "version_number", 0))
            if expected_version is not None and expected_version != active_number:
                raise _error(
                    409,
                    "stale_base_version",
                    f"Expected version {expected_version}, active version is {active_number}",
                )
            kind = JobKind.EXTEND
        with tempfile.TemporaryDirectory(prefix="hars-upload-") as temporary:
            bundle = Path(temporary) / "input.tar.gz"
            _write_input_bundle(uploaded, bundle)
            ref = artifact_store.put_file(
                f"inputs/{principal.tenant_id}/{request_sha}.tar.gz", bundle
            )
        try:
            job = repository.create_job(
                tenant_id=principal.tenant_id,
                index_id=index_id,
                kind=kind,
                idempotency_key=idempotency_key,
                request=request,
                input_artifact_uri=ref.uri,
                input_artifact_sha256=ref.sha256,
                input_artifact_size_bytes=ref.size_bytes,
                base_version_id=base_version_id,
            )
        except IdempotencyConflictError as exc:
            raise _error(409, "idempotency_conflict", str(exc)) from exc
        except StaleBaseVersionError as exc:
            raise _error(409, "stale_base_version", str(exc)) from exc
        except InvalidTransitionError as exc:
            raise _error(409, "invalid_transition", str(exc)) from exc
        wake = getattr(worker, "wake", None)
        if callable(wake):
            wake()
        return _job_json(job)

    @app.get("/v1/index-jobs/{job_id}")
    def get_job(job_id: str, principal: Principal = Depends(principal_dependency)) -> dict[str, object]:
        job = repository.get_job(tenant_id=principal.tenant_id, job_id=job_id)
        if job is None:
            raise _error(404, "job_not_found", "Job not found")
        return _job_json(job)

    @app.post("/v1/index-jobs/{job_id}/cancel")
    def cancel_job(
        job_id: str, principal: Principal = Depends(principal_dependency)
    ) -> dict[str, object]:
        if repository.get_job(tenant_id=principal.tenant_id, job_id=job_id) is None:
            raise _error(404, "job_not_found", "Job not found")
        try:
            job = repository.request_cancel(tenant_id=principal.tenant_id, job_id=job_id)
        except ServiceDatabaseError as exc:
            raise _error(409, "job_not_cancellable", str(exc)) from exc
        wake = getattr(worker, "wake", None)
        if callable(wake):
            wake()
        return _job_json(job)

    def resolve_version(tenant_id: str, index_id: str, version_number: int | None) -> object:
        index = repository.get_index(tenant_id=tenant_id, index_id=index_id)
        if index is None:
            raise _error(404, "index_not_found", "Index not found")
        if version_number is None:
            version_id = getattr(index, "active_version_id", None)
            version = (
                repository.get_version(tenant_id=tenant_id, version_id=str(version_id))
                if version_id
                else None
            )
        else:
            getter = getattr(repository, "get_version_by_number", None)
            if not callable(getter):
                raise _error(501, "version_lookup_unavailable", "Version lookup is unavailable")
            version = getter(
                tenant_id=tenant_id, index_id=index_id, version_number=version_number
            )
        if version is None:
            raise _error(404, "index_version_not_found", "Index version not found")
        return version

    @app.get("/v1/indexes/{index_id}/latest")
    def latest_index(
        index_id: str, principal: Principal = Depends(principal_dependency)
    ) -> dict[str, object]:
        return _version_json(resolve_version(principal.tenant_id, index_id, None))

    @app.get("/v1/indexes/{index_id}/versions/{version_number}")
    def get_index_version(
        index_id: str,
        version_number: int,
        principal: Principal = Depends(principal_dependency),
    ) -> dict[str, object]:
        return _version_json(resolve_version(principal.tenant_id, index_id, version_number))

    @app.get("/v1/indexes/{index_id}/versions/{version_number}/artifact")
    def download_artifact(
        index_id: str,
        version_number: int,
        principal: Principal = Depends(principal_dependency),
    ) -> FileResponse:
        version = resolve_version(principal.tenant_id, index_id, version_number)
        ref = ArtifactRef(
            uri=str(getattr(version, "artifact_uri")),
            sha256=str(getattr(version, "artifact_sha256")),
            size_bytes=int(getattr(version, "artifact_size_bytes")),
        )
        temporary_dir = Path(tempfile.mkdtemp(prefix="hars-download-"))
        destination = temporary_dir / f"{index_id}-v{version_number}.tar.gz"
        try:
            artifact_store.materialize(ref, destination)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise
        return FileResponse(
            destination,
            media_type="application/gzip",
            filename=destination.name,
            headers={"X-Artifact-SHA256": ref.sha256, "ETag": f'"{ref.sha256}"'},
            background=BackgroundTask(shutil.rmtree, temporary_dir, ignore_errors=True),
        )

    return app


__all__ = ["Repository", "create_app"]
