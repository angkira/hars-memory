"""Tests for garbage collection of documents whose source file is gone.

Covers both halves:
  * `ingest/change_detection.py::detect_changed_documents` reporting
    `deleted_doc_ids` (fingerprinted before, absent from this walk).
  * `server/index.py::_apply_refresh_changed` issuing one
    `adelete_by_doc_id` per such doc_id, with no reinsert, and dropping the
    stale BM25 cache afterwards.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from hars_memory.ingest.change_detection import (
    FingerprintStore,
    compute_fingerprint,
    detect_changed_documents,
)
from hars_memory.ingest.document import Document, SourceKind
from hars_memory.server import index as index_module


def _doc(doc_id: str, content: str) -> Document:
    return Document(
        doc_id=doc_id,
        content=content,
        source_kind=SourceKind.MARKDOWN,
        source_path=f"/kb/{doc_id}.md",
    )


class _FakeRag:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        self.deleted.append(doc_id)


class TestDetectDeletedDocuments:
    def test_fingerprinted_doc_absent_from_walk_is_reported_deleted(
        self, tmp_path: Path
    ) -> None:
        store = FingerprintStore(path=tmp_path / "fp.json")
        store.set("file:kept", compute_fingerprint("kept body"))
        store.set("file:gone", compute_fingerprint("gone body"))

        report = detect_changed_documents([_doc("file:kept", "kept body")], store)

        assert report.deleted_doc_ids == ("file:gone",)
        assert report.changed_doc_ids == ()
        assert report.unchanged_doc_ids == ("file:kept",)

    def test_deleted_doc_fingerprint_is_dropped_from_the_store(
        self, tmp_path: Path
    ) -> None:
        store = FingerprintStore(path=tmp_path / "fp.json")
        store.set("file:gone", compute_fingerprint("gone body"))

        detect_changed_documents([], store)

        assert store.get("file:gone") is None
        assert store.keys() == frozenset()

    def test_returning_file_is_treated_as_no_fingerprint_not_changed(
        self, tmp_path: Path
    ) -> None:
        """A file deleted and later restored must land in the safe class."""
        path = tmp_path / "fp.json"
        store = FingerprintStore(path=path)
        store.set("file:x", compute_fingerprint("old body"))

        detect_changed_documents([], store)
        store.save()

        reloaded = FingerprintStore.load(path)
        report = detect_changed_documents([_doc("file:x", "new body")], reloaded)

        assert report.changed_doc_ids == ()
        assert report.no_fingerprint_count == 1
        assert report.deleted_doc_ids == ()

    def test_no_deletions_when_every_doc_is_present(self, tmp_path: Path) -> None:
        store = FingerprintStore(path=tmp_path / "fp.json")
        store.set("file:a", compute_fingerprint("a"))

        report = detect_changed_documents(
            [_doc("file:a", "a"), _doc("file:b", "b")], store
        )

        assert report.deleted_doc_ids == ()
        assert report.no_fingerprint_count == 1

    def test_deleted_ids_are_sorted_for_deterministic_logs(
        self, tmp_path: Path
    ) -> None:
        store = FingerprintStore(path=tmp_path / "fp.json")
        for doc_id in ("file:c", "file:a", "file:b"):
            store.set(doc_id, compute_fingerprint(doc_id))

        report = detect_changed_documents([], store)

        assert report.deleted_doc_ids == ("file:a", "file:b", "file:c")

    def test_persisted_store_no_longer_lists_deleted_doc(self, tmp_path: Path) -> None:
        path = tmp_path / "fp.json"
        store = FingerprintStore(path=path)
        store.set("file:gone", compute_fingerprint("x"))
        store.set("file:kept", compute_fingerprint("kept"))

        detect_changed_documents([_doc("file:kept", "kept")], store)
        store.save()

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert list(raw) == ["file:kept"]


class TestApplyRefreshChangedGarbageCollection:
    def test_partial_walk_skips_gc_and_preserves_fingerprints(
        self, tmp_path: Path, caplog
    ) -> None:
        fp_path = tmp_path / "fp.json"
        seed = FingerprintStore(path=fp_path)
        for i in range(10):
            seed.set(f"file:{i}", compute_fingerprint(f"body {i}"))
        seed.save()

        rag = _FakeRag()
        store = asyncio.run(
            index_module._apply_refresh_changed(
                rag,
                [_doc("file:0", "body 0")],
                refresh_changed=True,
                fingerprint_store_path=fp_path,
            )
        )

        assert rag.deleted == []
        assert store is not None
        assert store.keys() == frozenset(f"file:{i}" for i in range(10))
        assert "Skipping deleted-document GC" in caplog.text
        assert "--paths may be narrowed" in caplog.text

    def test_partial_walk_still_refreshes_changed_documents(
        self, tmp_path: Path, caplog
    ) -> None:
        fp_path = tmp_path / "fp.json"
        seed = FingerprintStore(path=fp_path)
        seed.set("file:changed", compute_fingerprint("old"))
        for i in range(9):
            seed.set(f"file:omitted-{i}", compute_fingerprint(f"body {i}"))
        seed.save()

        rag = _FakeRag()
        store = asyncio.run(
            index_module._apply_refresh_changed(
                rag,
                [_doc("file:changed", "new")],
                refresh_changed=True,
                fingerprint_store_path=fp_path,
            )
        )

        assert rag.deleted == ["file:changed"]
        assert store is not None
        assert store.get("file:omitted-0") == compute_fingerprint("body 0")
        assert "Skipping deleted-document GC" in caplog.text

    def test_deleted_docs_are_deleted_from_the_index(self, tmp_path: Path) -> None:
        fp_path = tmp_path / "fp.json"
        seed = FingerprintStore(path=fp_path)
        seed.set("file:gone", compute_fingerprint("gone"))
        seed.set("file:kept", compute_fingerprint("kept"))
        seed.save()

        rag = _FakeRag()
        store = asyncio.run(
            index_module._apply_refresh_changed(
                rag,
                [_doc("file:kept", "kept")],
                refresh_changed=True,
                fingerprint_store_path=fp_path,
            )
        )

        assert rag.deleted == ["file:gone"]
        assert store is not None
        assert store.get("file:gone") is None

    def test_changed_and_deleted_are_both_handled(self, tmp_path: Path) -> None:
        fp_path = tmp_path / "fp.json"
        seed = FingerprintStore(path=fp_path)
        seed.set("file:changed", compute_fingerprint("old"))
        seed.set("file:gone", compute_fingerprint("gone"))
        seed.save()

        rag = _FakeRag()
        asyncio.run(
            index_module._apply_refresh_changed(
                rag,
                [_doc("file:changed", "new")],
                refresh_changed=True,
                fingerprint_store_path=fp_path,
            )
        )

        assert sorted(rag.deleted) == ["file:changed", "file:gone"]

    def test_opt_out_never_deletes_anything(self, tmp_path: Path) -> None:
        fp_path = tmp_path / "fp.json"
        seed = FingerprintStore(path=fp_path)
        seed.set("file:gone", compute_fingerprint("gone"))
        seed.save()

        rag = _FakeRag()
        store = asyncio.run(
            index_module._apply_refresh_changed(
                rag,
                [],
                refresh_changed=False,
                fingerprint_store_path=fp_path,
            )
        )

        assert rag.deleted == []
        assert store is None
        # Sidecar untouched.
        assert "file:gone" in json.loads(fp_path.read_text(encoding="utf-8"))

    def test_bm25_cache_is_invalidated_after_gc(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        fp_path = tmp_path / "fp.json"
        seed = FingerprintStore(path=fp_path)
        seed.set("file:gone", compute_fingerprint("gone"))
        seed.set("file:kept", compute_fingerprint("kept"))
        seed.save()

        cache_dir = tmp_path / "bm25_cache"
        cache_dir.mkdir()
        (cache_dir / "bm25_cache_meta.json").write_text("{}", encoding="utf-8")
        monkeypatch.setenv("HARS_MEMORY_BM25_CACHE_DIR", str(cache_dir))

        asyncio.run(
            index_module._apply_refresh_changed(
                _FakeRag(),
                [_doc("file:kept", "kept")],
                refresh_changed=True,
                fingerprint_store_path=fp_path,
            )
        )

        assert not cache_dir.exists()

    def test_bm25_cache_invalidation_failure_does_not_break_indexing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        fp_path = tmp_path / "fp.json"
        seed = FingerprintStore(path=fp_path)
        seed.set("file:gone", compute_fingerprint("gone"))
        seed.set("file:kept", compute_fingerprint("kept"))
        seed.save()

        # Relative path -> BM25CacheDirNotAbsoluteError inside invalidate_cache.
        monkeypatch.setenv("HARS_MEMORY_BM25_CACHE_DIR", "relative/cache")

        rag = _FakeRag()
        asyncio.run(
            index_module._apply_refresh_changed(
                rag,
                [_doc("file:kept", "kept")],
                refresh_changed=True,
                fingerprint_store_path=fp_path,
            )
        )

        assert rag.deleted == ["file:gone"]

    def test_no_cache_touch_when_nothing_was_deleted(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        fp_path = tmp_path / "fp.json"
        seed = FingerprintStore(path=fp_path)
        seed.set("file:kept", compute_fingerprint("kept"))
        seed.save()

        cache_dir = tmp_path / "bm25_cache"
        cache_dir.mkdir()
        monkeypatch.setenv("HARS_MEMORY_BM25_CACHE_DIR", str(cache_dir))

        asyncio.run(
            index_module._apply_refresh_changed(
                _FakeRag(),
                [_doc("file:kept", "kept")],
                refresh_changed=True,
                fingerprint_store_path=fp_path,
            )
        )

        assert cache_dir.exists()


class TestInvalidateBm25Cache:
    def test_removes_an_existing_cache_directory(self, tmp_path: Path) -> None:
        from hars_memory.retrieval.bm25_index import invalidate_cache

        cache_dir = tmp_path / "bm25_cache"
        cache_dir.mkdir()
        (cache_dir / "params.index.json").write_text("{}", encoding="utf-8")

        assert invalidate_cache(str(cache_dir)) is True
        assert not cache_dir.exists()
        # No staging leftovers next to it.
        assert list(tmp_path.iterdir()) == []

    def test_missing_cache_is_a_noop(self, tmp_path: Path) -> None:
        from hars_memory.retrieval.bm25_index import invalidate_cache

        assert invalidate_cache(str(tmp_path / "absent")) is False

    def test_relative_cache_dir_is_refused(self) -> None:
        from hars_memory.retrieval.bm25_index import (
            BM25CacheDirNotAbsoluteError,
            invalidate_cache,
        )

        try:
            invalidate_cache("relative/cache")
        except BM25CacheDirNotAbsoluteError:
            return
        raise AssertionError("expected BM25CacheDirNotAbsoluteError")
