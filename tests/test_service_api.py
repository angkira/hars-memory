from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from hars_memory.service.api import create_app
from hars_memory.service.artifacts import LocalArtifactStore
from hars_memory.service.auth import APIKeyAuthenticator, AuthConfigurationError
from hars_memory.service.database import ServiceDatabase
from hars_memory.service.models import Index
from hars_memory.service.validation import UploadLimits


class _Worker:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def wake(self) -> None:
        pass


def _client(tmp_path: Path) -> tuple[TestClient, ServiceDatabase]:
    database = ServiceDatabase(f"sqlite:///{tmp_path / 'service.sqlite3'}")
    database.create_schema()
    app = create_app(
        database,
        LocalArtifactStore(tmp_path / "artifacts"),
        _Worker(),
        APIKeyAuthenticator({"key-a": "tenant-a", "key-b": "tenant-b"}),
        upload_limits=UploadLimits(max_files=2, max_file_bytes=64, max_total_bytes=100),
    )
    return TestClient(app), database


def test_authenticator_fails_closed() -> None:
    try:
        APIKeyAuthenticator({})
    except AuthConfigurationError:
        pass
    else:
        raise AssertionError("empty API-key mapping must fail")


def test_submit_is_authenticated_tenant_scoped_and_idempotent(tmp_path: Path) -> None:
    client, database = _client(tmp_path)
    files = [("files", ("note.md", b"hello memory", "text/markdown"))]
    data = {"operation": "create", "engine": "corpus"}
    headers = {"X-API-Key": "key-a", "Idempotency-Key": "same-request"}

    assert client.post("/v1/index-jobs", data=data, files=files).status_code == 401
    first = client.post("/v1/index-jobs", data=data, files=files, headers=headers)
    second = client.post("/v1/index-jobs", data=data, files=files, headers=headers)
    assert first.status_code == second.status_code == 202
    assert first.json()["id"] == second.json()["id"]

    job_id = first.json()["id"]
    assert client.get(f"/v1/index-jobs/{job_id}", headers={"X-API-Key": "key-b"}).status_code == 404
    assert client.get(f"/v1/index-jobs/{job_id}", headers={"X-API-Key": "key-a"}).status_code == 200
    database.close()


def test_upload_validation_rejects_traversal_and_binary(tmp_path: Path) -> None:
    client, database = _client(tmp_path)
    headers = {"X-API-Key": "key-a", "Idempotency-Key": "unsafe"}
    data = {"operation": "create", "engine": "corpus"}
    traversal = client.post(
        "/v1/index-jobs",
        data=data,
        files=[("files", ("../note.md", b"hello", "text/plain"))],
        headers=headers,
    )
    assert traversal.status_code == 422
    assert traversal.json()["detail"]["code"] == "unsafe_filename"
    binary = client.post(
        "/v1/index-jobs",
        data=data,
        files=[("files", ("note.txt", b"hello\x00world", "text/plain"))],
        headers={**headers, "Idempotency-Key": "binary"},
    )
    assert binary.status_code == 422
    assert binary.json()["detail"]["code"] == "binary_file"
    database.close()


@pytest.mark.parametrize(
    ("files", "code"),
    [
        ([("files", ("blank.md", b" \n\t", "text/plain"))], "empty_file"),
        (
            [("files", ("image.exe", b"not really an image", "text/plain"))],
            "unsupported_file_type",
        ),
        (
            [
                ("files", ("same.md", b"one", "text/plain")),
                ("files", ("same.md", b"two", "text/plain")),
            ],
            "duplicate_filename",
        ),
        (
            [
                ("files", ("one.md", b"one", "text/plain")),
                ("files", ("two.md", b"two", "text/plain")),
                ("files", ("three.md", b"three", "text/plain")),
            ],
            "too_many_files",
        ),
        ([("files", ("large.md", b"x" * 65, "text/plain"))], "file_too_large"),
        (
            [
                ("files", ("one.md", b"x" * 60, "text/plain")),
                ("files", ("two.md", b"y" * 41, "text/plain")),
            ],
            "upload_too_large",
        ),
    ],
)
def test_upload_validation_bounds(
    tmp_path: Path,
    files: list[tuple[str, tuple[str, bytes, str]]],
    code: str,
) -> None:
    client, database = _client(tmp_path)
    response = client.post(
        "/v1/index-jobs",
        data={"operation": "create", "engine": "corpus"},
        files=files,
        headers={"X-API-Key": "key-a", "Idempotency-Key": code},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == code
    database.close()


def test_terminal_retries_are_stable_and_conflicts_do_not_create_indexes(
    tmp_path: Path,
) -> None:
    client, database = _client(tmp_path)
    headers = {"X-API-Key": "key-a", "Idempotency-Key": "terminal-create"}
    files = [("files", ("note.md", "héllo".encode(), "text/markdown"))]
    first = client.post(
        "/v1/index-jobs",
        data={"operation": "create", "engine": "corpus", "index_name": "docs"},
        files=files,
        headers=headers,
    )
    assert first.status_code == 202
    claimed = database.claim_next_job(worker_id="test-worker")
    assert claimed is not None
    database.publish_version(
        job_id=claimed.id,
        worker_id="test-worker",
        artifact_uri="file:///version-1.tar.gz",
        artifact_sha256="a" * 64,
        artifact_size_bytes=1,
        manifest={"document_count": 1},
    )

    retry = client.post(
        "/v1/index-jobs",
        data={"operation": "create", "engine": "corpus", "index_name": "docs"},
        files=files,
        headers=headers,
    )
    assert retry.status_code == 202
    assert retry.json()["id"] == first.json()["id"]
    assert retry.json()["status"] == "succeeded"

    conflict = client.post(
        "/v1/index-jobs",
        data={"operation": "create", "engine": "corpus", "index_name": "orphan"},
        files=files,
        headers=headers,
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_conflict"
    with database._sessions() as session:
        assert session.scalar(select(func.count()).select_from(Index)) == 1
    database.close()


def test_api_persists_strategy_snapshot_and_rejects_unknown_options(tmp_path: Path) -> None:
    client, database = _client(tmp_path)
    response = client.post(
        "/v1/index-jobs",
        data={
            "operation": "create",
            "engine": "lightrag",
            "strategy_name": "small-graph",
            "strategy_options": json.dumps(
                {"chunk_token_size": 256, "max_gleaning": 0}
            ),
        },
        files=[("files", ("note.md", b"hello", "text/plain"))],
        headers={"X-API-Key": "key-a", "Idempotency-Key": "strategy"},
    )
    assert response.status_code == 202
    job = database.get_job(tenant_id="tenant-a", job_id=response.json()["id"])
    request = json.loads(job.request_json)  # type: ignore[union-attr]
    assert request["index_strategy"] == {
        "name": "small-graph",
        "engine": "lightrag",
        "options": {"chunk_token_size": 256, "max_gleaning": 0},
    }
    rejected = client.post(
        "/v1/index-jobs",
        data={
            "operation": "create",
            "engine": "lightrag",
            "strategy_options": '{"extractor_base_url":"http://untrusted"}',
        },
        files=[("files", ("note.md", b"hello", "text/plain"))],
        headers={"X-API-Key": "key-a", "Idempotency-Key": "unsafe-strategy"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["detail"]["code"] == "invalid_index_strategy"
    database.close()
