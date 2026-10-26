from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from hars_memory.sdk import ArtifactChecksumError, HarsMemoryClient
from hars_memory.strategies import IndexStrategy


def test_sdk_create_get_cancel_and_auth_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        payload = {
            "id": "job-1",
            "status": "queued",
            "operation": "create",
            "engine": "corpus",
            "index_id": "index-1",
        }
        return httpx.Response(202 if request.method == "POST" else 200, json=payload)

    with HarsMemoryClient(
        "https://memory.example", "secret-key", transport=httpx.MockTransport(handler)
    ) as client:
        job = client.create_index({"note.md": "hello"}, idempotency_key="request-1")
        assert job.id == "job-1"
        client.get_job(job.id)
        client.cancel_job(job.id)

    assert all(request.headers["X-API-Key"] == "secret-key" for request in seen)
    assert seen[0].headers["Idempotency-Key"] == "request-1"
    assert b"hello" in seen[0].content


def test_sdk_download_verifies_checksum(tmp_path: Path) -> None:
    content = b"artifact bytes"
    checksum = hashlib.sha256(content).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/latest"):
            return httpx.Response(
                200,
                json={"index_id": "idx", "version": 1, "artifact_sha256": checksum},
            )
        return httpx.Response(200, content=content, headers={"X-Artifact-SHA256": checksum})

    target = tmp_path / "index.tar.gz"
    with HarsMemoryClient(
        "https://memory.example", "key", transport=httpx.MockTransport(handler)
    ) as client:
        assert client.download_artifact("idx", target) == target
    assert target.read_bytes() == content

    def corrupt(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"corrupt")

    with HarsMemoryClient(
        "https://memory.example", "key", transport=httpx.MockTransport(corrupt)
    ) as client:
        with pytest.raises(ArtifactChecksumError):
            client.download_artifact(
                "idx", tmp_path / "bad.tar.gz", version=1, expected_sha256=checksum
            )


def test_sdk_serializes_validated_index_strategy() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            202,
            json={
                "id": "job-1",
                "status": "queued",
                "operation": "create",
                "engine": "lightrag",
                "index_id": "index-1",
            },
        )

    strategy = IndexStrategy(
        "small-graph", "lightrag", {"chunk_token_size": 256, "max_gleaning": 0}
    )
    with HarsMemoryClient(
        "https://memory.example", "key", transport=httpx.MockTransport(handler)
    ) as client:
        client.create_index(
            {"note.md": "hello"},
            engine="lightrag",
            strategy=strategy,
            idempotency_key="strategy",
        )
    assert b'small-graph' in seen[0].content
    assert b'chunk_token_size' in seen[0].content
    with HarsMemoryClient(
        "https://memory.example", "key", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(ValueError, match="must match"):
            client.create_index({"note.md": "hello"}, strategy=strategy)
