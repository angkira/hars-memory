"""Synchronous Python SDK for the HARS memory index-job HTTP API."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx

from hars_memory.strategies import IndexStrategy

TERMINAL_JOB_STATES: Final[frozenset[str]] = frozenset(
    {"succeeded", "failed", "cancelled"}
)


class HarsMemoryAPIError(RuntimeError):
    """An HTTP API request failed."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(f"HTTP {status_code} {code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message


class JobTimeoutError(TimeoutError):
    def __init__(self, job_id: str, timeout_seconds: float) -> None:
        super().__init__(f"Job {job_id} did not finish within {timeout_seconds:g}s")
        self.job_id = job_id
        self.timeout_seconds = timeout_seconds


class ArtifactChecksumError(IOError):
    pass


@dataclass(frozen=True, slots=True)
class JobInfo:
    id: str
    status: str
    operation: str
    engine: str
    index_id: str | None
    version: int | None
    error: str | None
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JobInfo:
        return cls(
            id=str(value.get("id") or value.get("job_id")),
            status=str(value["status"]),
            operation=str(value.get("operation", "")),
            engine=str(value.get("engine", "")),
            index_id=_optional_str(value.get("index_id")),
            version=_optional_int(value.get("version") or value.get("target_version")),
            error=_optional_str(value.get("error") or value.get("error_message")),
            raw=dict(value),
        )


@dataclass(frozen=True, slots=True)
class IndexVersionInfo:
    index_id: str
    version: int
    artifact_sha256: str
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> IndexVersionInfo:
        checksum = value.get("artifact_sha256") or value.get("checksum_sha256") or ""
        return cls(
            index_id=str(value["index_id"]),
            version=int(value["version"]),
            artifact_sha256=str(checksum),
            raw=dict(value),
        )


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[arg-type]


class HarsMemoryClient:
    """Blocking client. Credentials are sent only in ``X-API-Key`` headers."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key},
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> HarsMemoryClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def submit_job(
        self,
        files: Sequence[Path] | Mapping[str, bytes | str],
        *,
        operation: str,
        engine: str = "corpus",
        index_id: str | None = None,
        index_name: str | None = None,
        expected_version: int | None = None,
        idempotency_key: str | None = None,
        strategy: IndexStrategy | None = None,
    ) -> JobInfo:
        if operation not in {"create", "extend"}:
            raise ValueError("operation must be 'create' or 'extend'")
        if operation == "extend" and not index_id:
            raise ValueError("index_id is required for extend")
        multipart = _multipart_files(files)
        data: dict[str, str] = {"operation": operation, "engine": engine}
        if strategy is not None:
            if strategy.engine != engine:
                raise ValueError("strategy engine must match the submitted engine")
            data["strategy_name"] = strategy.name
            data["strategy_options"] = json.dumps(
                dict(strategy.options), sort_keys=True, separators=(",", ":")
            )
        if index_id is not None:
            data["index_id"] = index_id
        if index_name is not None:
            data["index_name"] = index_name
        if expected_version is not None:
            data["expected_version"] = str(expected_version)
        response = self._client.post(
            "/v1/index-jobs",
            data=data,
            files=multipart,
            headers={"Idempotency-Key": idempotency_key or str(uuid.uuid4())},
        )
        return JobInfo.from_dict(_json(response))

    def create_index(
        self,
        files: Sequence[Path] | Mapping[str, bytes | str],
        *,
        engine: str = "corpus",
        name: str | None = None,
        idempotency_key: str | None = None,
        strategy: IndexStrategy | None = None,
    ) -> JobInfo:
        return self.submit_job(
            files,
            operation="create",
            engine=engine,
            index_name=name,
            idempotency_key=idempotency_key,
            strategy=strategy,
        )

    def extend_index(
        self,
        index_id: str,
        files: Sequence[Path] | Mapping[str, bytes | str],
        *,
        engine: str = "corpus",
        expected_version: int | None = None,
        idempotency_key: str | None = None,
        strategy: IndexStrategy | None = None,
    ) -> JobInfo:
        return self.submit_job(
            files,
            operation="extend",
            engine=engine,
            index_id=index_id,
            expected_version=expected_version,
            idempotency_key=idempotency_key,
            strategy=strategy,
        )

    def get_job(self, job_id: str) -> JobInfo:
        return JobInfo.from_dict(_json(self._client.get(f"/v1/index-jobs/{job_id}")))

    def wait_job(
        self,
        job_id: str,
        *,
        timeout: float = 600.0,
        poll_interval: float = 0.5,
    ) -> JobInfo:
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("timeout and poll_interval must be positive")
        deadline = time.monotonic() + timeout
        while True:
            job = self.get_job(job_id)
            if job.status in TERMINAL_JOB_STATES:
                return job
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise JobTimeoutError(job_id, timeout)
            time.sleep(min(poll_interval, remaining))

    def cancel_job(self, job_id: str) -> JobInfo:
        return JobInfo.from_dict(
            _json(self._client.post(f"/v1/index-jobs/{job_id}/cancel"))
        )

    def get_latest_index(self, index_id: str) -> IndexVersionInfo:
        response = self._client.get(f"/v1/indexes/{index_id}/latest")
        return IndexVersionInfo.from_dict(_json(response))

    def get_index_version(self, index_id: str, version: int) -> IndexVersionInfo:
        response = self._client.get(f"/v1/indexes/{index_id}/versions/{version}")
        return IndexVersionInfo.from_dict(_json(response))

    def download_artifact(
        self,
        index_id: str,
        destination: Path,
        *,
        version: int | None = None,
        expected_sha256: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        if version is None:
            descriptor = self.get_latest_index(index_id)
            version = descriptor.version
            expected_sha256 = expected_sha256 or descriptor.artifact_sha256
        route = f"/v1/indexes/{index_id}/versions/{version}/artifact"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        digest = hashlib.sha256()
        try:
            with self._client.stream("GET", route) as response:
                _raise_for_error(response)
                header_checksum = response.headers.get("X-Artifact-SHA256")
                expected = expected_sha256 or header_checksum
                with temporary.open("wb") as output:
                    for chunk in response.iter_bytes():
                        output.write(chunk)
                        digest.update(chunk)
            if expected and not hmac_compare(digest.hexdigest(), expected):
                raise ArtifactChecksumError(
                    f"Artifact checksum mismatch: expected {expected}, got {digest.hexdigest()}"
                )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination


def hmac_compare(left: str, right: str) -> bool:
    import hmac

    return hmac.compare_digest(left.lower(), right.lower())


def _multipart_files(
    files: Sequence[Path] | Mapping[str, bytes | str],
) -> list[tuple[str, tuple[str, bytes, str]]]:
    items: list[tuple[str, bytes]] = []
    if isinstance(files, Mapping):
        for name, value in files.items():
            items.append((name, value.encode("utf-8") if isinstance(value, str) else value))
    else:
        for path in files:
            items.append((path.name, path.read_bytes()))
    if not items:
        raise ValueError("at least one file is required")
    return [("files", (name, content, "text/plain")) for name, content in items]


def _json(response: httpx.Response) -> Mapping[str, Any]:
    _raise_for_error(response)
    value = response.json()
    if not isinstance(value, dict):
        raise HarsMemoryAPIError(response.status_code, "invalid_response", "Expected object")
    return value


def _raise_for_error(response: httpx.Response) -> None:
    if not response.is_error:
        return
    try:
        body = response.json()
    except ValueError:
        body = {}
    detail = body.get("detail", body) if isinstance(body, dict) else {}
    if not isinstance(detail, dict):
        detail = {"message": str(detail)}
    raise HarsMemoryAPIError(
        response.status_code,
        str(detail.get("code", "http_error")),
        str(detail.get("message", response.reason_phrase)),
    )


__all__ = [
    "ArtifactChecksumError",
    "HarsMemoryAPIError",
    "HarsMemoryClient",
    "IndexVersionInfo",
    "JobInfo",
    "JobTimeoutError",
]
