from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from hars_memory.service.artifacts import (
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactRef,
    LocalArtifactStore,
    S3ArtifactStore,
    create_artifact_store,
)


def test_local_store_round_trip_and_idempotent_put(tmp_path: Path) -> None:
    source = tmp_path / "source.tar.gz"
    source.write_bytes(b"index bundle")
    store = LocalArtifactStore(tmp_path / "artifacts")
    first = store.put_result("index-1", 1, source)
    second = store.put_result("index-1", 1, source)
    assert first == second
    destination = tmp_path / "download.tar.gz"
    store.materialize_result(first.uri, destination, first.sha256)
    assert destination.read_bytes() == b"index bundle"
    with store.open(first) as stream:
        assert stream.read() == b"index bundle"


def test_local_store_rejects_traversal_and_checksum_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"safe")
    store = LocalArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ArtifactError):
        store.put_file("../escape", source)
    ref = store.put_file("inputs/source", source)
    with pytest.raises(ArtifactIntegrityError):
        store.materialize(ArtifactRef(ref.uri, "0" * 64, ref.size_bytes), tmp_path / "bad")


def test_existing_local_key_is_immutable(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    store.put_file("same", first)
    with pytest.raises(ArtifactIntegrityError):
        store.put_file("same", second)


class _Missing(Exception):
    response = {"Error": {"Code": "404"}}


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        try:
            data, metadata = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise _Missing from exc
        return {"ContentLength": len(data), "Metadata": metadata}

    def put_object(
        self, *, Bucket: str, Key: str, Body: object, Metadata: dict[str, str], IfNoneMatch: str
    ) -> None:
        assert IfNoneMatch == "*"
        if (Bucket, Key) in self.objects:
            error = _Missing()
            error.response = {"Error": {"Code": "PreconditionFailed"}}
            raise error
        self.objects[(Bucket, Key)] = (Body.read(), Metadata)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        data, metadata = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(data), "ContentLength": len(data), "Metadata": metadata}


def test_s3_adapter_preserves_checksum_metadata(tmp_path: Path) -> None:
    fake = _FakeS3()
    source = tmp_path / "bundle"
    source.write_bytes(b"cloud index")
    store = S3ArtifactStore("bucket", prefix="tenant", client=fake)
    ref = store.put_result("index", 2, source)
    assert ref.sha256 == hashlib.sha256(b"cloud index").hexdigest()
    destination = tmp_path / "materialized"
    store.materialize_result(ref.uri, destination, ref.sha256)
    assert destination.read_bytes() == b"cloud index"


def test_store_factory_supports_paths_and_file_urls(tmp_path: Path) -> None:
    assert isinstance(create_artifact_store(str(tmp_path / "one")), LocalArtifactStore)
    assert isinstance(create_artifact_store((tmp_path / "two").as_uri()), LocalArtifactStore)
