from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

from hars_memory.service.engines import (
    BUNDLE_MANIFEST,
    CorpusEngine,
    EngineError,
    EngineRequest,
    LightRAGEngine,
    UnsafeBundleError,
    _extract_tar_safely,
)


def _request(tmp_path: Path, input_path: Path, **overrides: object) -> EngineRequest:
    values: dict[str, object] = {
        "job_id": "job-1",
        "tenant_id": "tenant-1",
        "index_id": "index-1",
        "operation": "create",
        "input_path": input_path,
        "output_path": tmp_path / "result.tar.gz",
        "workspace": tmp_path / "work",
        "version": 1,
    }
    values.update(overrides)
    Path(values["workspace"]).mkdir(parents=True)
    return EngineRequest(**values)  # type: ignore[arg-type]


def _unpack(archive: Path, destination: Path) -> dict[str, object]:
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination, filter="data")
    return json.loads((destination / BUNDLE_MANIFEST).read_text(encoding="utf-8"))


def test_corpus_engine_builds_portable_checksum_bundle(tmp_path: Path) -> None:
    uploaded = tmp_path / "uploaded"
    uploaded.mkdir()
    (uploaded / "alpha.md").write_text("alpha searchable marker", encoding="utf-8")
    (uploaded / "nested").mkdir()
    (uploaded / "nested" / "beta.txt").write_text("beta content", encoding="utf-8")

    result = CorpusEngine().build(_request(tmp_path, uploaded))

    assert result.manifest["engine"] == "corpus"
    assert result.manifest["document_count"] == 2
    assert result.manifest["chunk_count"] == 2
    assert result.manifest["portable"] is True
    assert result.manifest["artifact_sha256"] == hashlib.sha256(
        result.artifact_path.read_bytes()
    ).hexdigest()
    unpacked = tmp_path / "unpacked"
    bundle = _unpack(result.artifact_path, unpacked)
    assert bundle["content_sha256"] == result.manifest["content_sha256"]
    chunks = json.loads(
        (unpacked / "index" / "kv_store_text_chunks.json").read_text(encoding="utf-8")
    )
    assert {entry["file_path"] for entry in chunks.values()} == {
        "upload://alpha.md",
        "upload://nested/beta.txt",
    }


def test_extend_merges_sources_without_mutating_base(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.mkdir()
    (first / "a.md").write_text("version one", encoding="utf-8")
    create = CorpusEngine().build(_request(tmp_path, first))
    base_bytes = create.artifact_path.read_bytes()

    update = tmp_path / "update"
    update.mkdir()
    (update / "a.md").write_text("version two", encoding="utf-8")
    (update / "b.py").write_text("VALUE = 'new'", encoding="utf-8")
    extend_root = tmp_path / "extend"
    extend_root.mkdir()
    request = _request(
        extend_root,
        update,
        operation="extend",
        base_artifact=create.artifact_path,
        version=2,
    )

    extended = CorpusEngine().build(request)

    assert create.artifact_path.read_bytes() == base_bytes
    assert extended.manifest["version"] == 2
    assert extended.manifest["document_count"] == 2
    unpacked = tmp_path / "extended-unpacked"
    _unpack(extended.artifact_path, unpacked)
    assert (unpacked / "sources" / "a.md").read_text() == "version two"
    assert (unpacked / "sources" / "b.py").is_file()


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tar:
        payload = b"escape"
        info = tarfile.TarInfo("../escaped.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with pytest.raises(UnsafeBundleError):
        _extract_tar_safely(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_rejects_non_text_upload(tmp_path: Path) -> None:
    uploaded = tmp_path / "uploaded"
    uploaded.mkdir()
    (uploaded / "weights.bin").write_bytes(b"\x00\x01")
    with pytest.raises(Exception, match="unsupported uploaded file"):
        CorpusEngine().build(_request(tmp_path, uploaded))


class _CompletedIndexer:
    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.command = command
        self.env = kwargs["env"]
        self.returncode = 0
        self.stdout = io.StringIO("index complete")
        index = Path(self.env["HARS_MEMORY_INDEX_DIR"])
        index.mkdir(parents=True, exist_ok=True)
        (index / "kv_store_text_chunks.json").write_text(
            json.dumps({"chunk-1": {"content": "ok", "file_path": "upload://a.md"}}),
            encoding="utf-8",
        )

    def poll(self) -> int:
        return self.returncode


def test_lightrag_qdrant_versions_get_immutable_remote_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploaded = tmp_path / "uploaded"
    uploaded.mkdir()
    (uploaded / "a.md").write_text("graph content", encoding="utf-8")
    captured: list[_CompletedIndexer] = []

    def spawn(command: list[str], **kwargs: Any) -> _CompletedIndexer:
        process = _CompletedIndexer(command, **kwargs)
        captured.append(process)
        return process

    monkeypatch.setattr("hars_memory.service.engines.subprocess.Popen", spawn)
    monkeypatch.setenv("HARS_MEMORY_VECTOR_STORAGE", "QdrantVectorDBStorage")
    monkeypatch.setenv("HARS_MEMORY_QDRANT_COLLECTION_PREFIX", "memory")
    monkeypatch.setenv("HARS_MEMORY_QDRANT_COLLECTION", "tenant")

    result = LightRAGEngine().build(_request(tmp_path, uploaded, version=3))

    env = captured[0].env
    assert captured[0].command[:3] == [sys.executable, "-m", "hars_memory.server.index"]
    assert captured[0].command[3:5] == [
        "--paths",
        str(Path(env["HARS_MEMORY_INDEX_DIR"]).parent / "sources"),
    ]
    assert captured[0].command[-1] == "--refresh-changed"
    assert env["HARS_MEMORY_QDRANT_COLLECTION_PREFIX"] == "memory"
    assert env["HARS_MEMORY_QDRANT_COLLECTION"].startswith("tenant_")
    assert env["QDRANT_WORKSPACE"] == env["HARS_MEMORY_QDRANT_COLLECTION"]
    assert result.manifest["portable"] is False
    external = result.manifest["external_state"]
    assert external["workspace"] == env["HARS_MEMORY_QDRANT_COLLECTION"]
    assert external["immutable_version_scope"] is True


def test_lightrag_qdrant_refuses_unscoped_collections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploaded = tmp_path / "uploaded"
    uploaded.mkdir()
    (uploaded / "a.md").write_text("graph content", encoding="utf-8")
    monkeypatch.setenv("HARS_MEMORY_VECTOR_STORAGE", "QdrantVectorDBStorage")
    monkeypatch.delenv("HARS_MEMORY_QDRANT_COLLECTION_PREFIX", raising=False)
    with pytest.raises(EngineError, match="COLLECTION_PREFIX"):
        LightRAGEngine().build(_request(tmp_path, uploaded))


def test_lightrag_applies_and_records_job_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploaded = tmp_path / "uploaded"
    uploaded.mkdir()
    (uploaded / "a.md").write_text("graph content", encoding="utf-8")
    captured: list[_CompletedIndexer] = []

    def spawn(command: list[str], **kwargs: Any) -> _CompletedIndexer:
        process = _CompletedIndexer(command, **kwargs)
        captured.append(process)
        return process

    monkeypatch.setattr("hars_memory.service.engines.subprocess.Popen", spawn)
    monkeypatch.setenv("HARS_MEMORY_VECTOR_STORAGE", "NanoVectorDBStorage")
    result = LightRAGEngine().build(
        _request(
            tmp_path,
            uploaded,
            strategy={"name": "small", "chunk_token_size": 256, "max_gleaning": 0},
        )
    )
    assert captured[0].env["HARS_MEMORY_CHUNK_TOKEN_SIZE"] == "256"
    assert captured[0].env["HARS_MEMORY_MAX_GLEANING"] == "0"
    assert result.manifest["index_strategy"]["name"] == "small"
    assert len(result.manifest["index_strategy_sha256"]) == 64
