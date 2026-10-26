"""Unit tests for tools/memory/corpus/build.py — LLM-free corpus build.

GPU-free, no LLM calls, no network — pure filesystem + Python.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hars_memory.corpus.build import (
    CorpusPathNotFoundError,
    EmptyCorpusError,
    LiveGraphIndexGuardError,
    MANIFEST_FILENAME,
    NoPathsProvidedError,
    UnsafeOverwriteTargetError,
    build_corpus,
)
from hars_memory.retrieval.bm25_index import CHUNKS_FILENAME


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestChunkStoreContract:
    def test_build_emits_exact_chunk_store_shape(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(src / "a.md", "Hello world, this is a small test document about zebras.")
        index_dir = tmp_path / "index"

        result = build_corpus([src], index_dir, chunk_size=1200, chunk_overlap=200)

        chunk_store = json.loads((index_dir / CHUNKS_FILENAME).read_text(encoding="utf-8"))
        assert isinstance(chunk_store, dict)
        assert len(chunk_store) == result.chunk_count
        for chunk_id, entry in chunk_store.items():
            assert isinstance(chunk_id, str)
            assert isinstance(entry, dict)
            assert "content" in entry and isinstance(entry["content"], str)
            assert "file_path" in entry and isinstance(entry["file_path"], str)
            assert entry["file_path"] == str((src / "a.md").resolve())

    def test_manifest_records_provenance(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(src / "a.md", "content one " * 50)
        index_dir = tmp_path / "index"

        result = build_corpus([src], index_dir, chunk_size=200, chunk_overlap=20)

        manifest = json.loads((index_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert "build" in manifest and "documents" in manifest
        build = manifest["build"]
        assert build["corpus_fingerprint"] == result.corpus_fingerprint
        assert build["chunk_store_sha256"] == result.chunk_store_sha256
        assert build["chunk_size"] == 200
        assert build["chunk_overlap"] == 20
        assert "tool_version" in build
        assert "created_at" in build

        documents = manifest["documents"]
        assert len(documents) == 1
        (doc_record,) = documents.values()
        assert doc_record["source_path"] == str((src / "a.md").resolve())
        assert "content_sha256" in doc_record
        assert doc_record["chunk_count"] == len(doc_record["chunk_ids"])
        assert doc_record["chunk_count"] > 0


class TestDeterminism:
    def test_chunk_ids_and_fingerprint_deterministic_across_rebuilds(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(src / "a.md", "Deterministic content. " * 30)
        _write(src / "b.md", "Second document content. " * 10)

        index_dir_1 = tmp_path / "index1"
        index_dir_2 = tmp_path / "index2"
        r1 = build_corpus([src], index_dir_1, chunk_size=300, chunk_overlap=40)
        r2 = build_corpus([src], index_dir_2, chunk_size=300, chunk_overlap=40)

        assert r1.corpus_fingerprint == r2.corpus_fingerprint
        assert r1.chunk_count == r2.chunk_count

        store1 = json.loads((index_dir_1 / CHUNKS_FILENAME).read_text(encoding="utf-8"))
        store2 = json.loads((index_dir_2 / CHUNKS_FILENAME).read_text(encoding="utf-8"))
        assert set(store1.keys()) == set(store2.keys())
        for chunk_id in store1:
            assert store1[chunk_id]["content"] == store2[chunk_id]["content"]

    def test_rebuild_in_place_is_byte_identical_fingerprint(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(src / "a.md", "Stable content that never changes across builds.")
        index_dir = tmp_path / "index"

        r1 = build_corpus([src], index_dir, chunk_size=1200, chunk_overlap=200)
        r2 = build_corpus([src], index_dir, chunk_size=1200, chunk_overlap=200)

        assert r1.corpus_fingerprint == r2.corpus_fingerprint
        assert r2.added == 0
        assert r2.changed == 0
        assert r2.deleted == 0
        assert r2.unchanged == r1.document_count


class TestIncremental:
    def test_unchanged_file_reported_unchanged(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(src / "a.md", "unchanged content")
        index_dir = tmp_path / "index"
        build_corpus([src], index_dir)

        result = build_corpus([src], index_dir)
        assert result.unchanged == 1
        assert result.added == 0
        assert result.changed == 0
        assert result.deleted == 0

    def test_changed_file_reported_changed(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        target = src / "a.md"
        _write(target, "version one")
        index_dir = tmp_path / "index"
        build_corpus([src], index_dir)

        _write(target, "version two, materially different content")
        result = build_corpus([src], index_dir)
        assert result.changed == 1
        assert result.added == 0
        assert result.unchanged == 0
        assert result.deleted == 0

    def test_added_file_reported_added(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(src / "a.md", "first document")
        index_dir = tmp_path / "index"
        build_corpus([src], index_dir)

        _write(src / "b.md", "second document, brand new")
        result = build_corpus([src], index_dir)
        assert result.added == 1
        assert result.unchanged == 1

    def test_deleted_file_chunks_removed_from_chunk_store(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        keep = src / "keep.md"
        gone = src / "gone.md"
        _write(keep, "this file stays around")
        _write(gone, "this file will be deleted before the next build")
        index_dir = tmp_path / "index"
        r1 = build_corpus([src], index_dir)
        assert r1.document_count == 2

        gone.unlink()
        r2 = build_corpus([src], index_dir)
        assert r2.deleted == 1
        assert r2.document_count == 1

        chunk_store = json.loads((index_dir / CHUNKS_FILENAME).read_text(encoding="utf-8"))
        for entry in chunk_store.values():
            assert "gone.md" not in entry["file_path"]

    def test_rename_leaves_no_orphan_chunks(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        old_path = src / "old_name.md"
        _write(old_path, "content that survives a rename unchanged")
        index_dir = tmp_path / "index"
        r1 = build_corpus([src], index_dir)
        assert r1.document_count == 1

        new_path = src / "new_name.md"
        old_path.rename(new_path)
        r2 = build_corpus([src], index_dir)

        assert r2.document_count == 1
        assert r2.added == 1  # new path = new doc_id (path-derived identity)
        assert r2.deleted == 1  # old path's doc_id no longer present

        chunk_store = json.loads((index_dir / CHUNKS_FILENAME).read_text(encoding="utf-8"))
        file_paths = {entry["file_path"] for entry in chunk_store.values()}
        assert all("old_name.md" not in fp for fp in file_paths)
        assert any("new_name.md" in fp for fp in file_paths)


class TestFailFastGuards:
    def test_no_paths_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NoPathsProvidedError):
            build_corpus([], tmp_path / "index")

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusPathNotFoundError):
            build_corpus([tmp_path / "does_not_exist"], tmp_path / "index")

    def test_zero_documents_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _write(src / "binary.bin", "irrelevant")  # not in default include globs
        with pytest.raises(EmptyCorpusError):
            build_corpus([src], tmp_path / "index")

    def test_refuses_live_graph_index_directory(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(src / "a.md", "some content")
        index_dir = tmp_path / "graph_index"
        index_dir.mkdir()
        (index_dir / "graph_chunk_entity_relation.graphml").write_text("<graphml/>")

        with pytest.raises(LiveGraphIndexGuardError):
            build_corpus([src], index_dir)

        # Guard fires for either graph-artifact filename independently.
        index_dir_2 = tmp_path / "graph_index_2"
        index_dir_2.mkdir()
        (index_dir_2 / "kv_store_full_entities.json").write_text("{}")
        with pytest.raises(LiveGraphIndexGuardError):
            build_corpus([src], index_dir_2)

    def test_refuses_unsafe_nonempty_target_without_force(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(src / "a.md", "some content")
        index_dir = tmp_path / "unrelated"
        index_dir.mkdir()
        (index_dir / "some_unrelated_file.txt").write_text("not ours")

        with pytest.raises(UnsafeOverwriteTargetError):
            build_corpus([src], index_dir)

        # force=True bypasses this guard.
        result = build_corpus([src], index_dir, force=True)
        assert result.document_count == 1


class TestAtomicSwap:
    def test_failed_build_leaves_existing_index_intact(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        src = tmp_path / "src"
        _write(src / "a.md", "original content that must survive a failed rebuild")
        index_dir = tmp_path / "index"
        build_corpus([src], index_dir)

        original_chunk_store = (index_dir / CHUNKS_FILENAME).read_text(encoding="utf-8")
        original_manifest = (index_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")

        import hars_memory.corpus.build as build_module

        def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated mid-build failure")

        monkeypatch.setattr(build_module, "_build_chunk_store_and_records", _boom)

        _write(src / "b.md", "a new document that would have been added")
        with pytest.raises(RuntimeError, match="simulated mid-build failure"):
            build_corpus([src], index_dir)

        # index_dir must be byte-for-byte untouched.
        assert (index_dir / CHUNKS_FILENAME).read_text(encoding="utf-8") == original_chunk_store
        assert (index_dir / MANIFEST_FILENAME).read_text(encoding="utf-8") == original_manifest
        # No leftover temp/backup directories.
        siblings = {p.name for p in index_dir.parent.iterdir()}
        assert siblings == {"src", "index"}
