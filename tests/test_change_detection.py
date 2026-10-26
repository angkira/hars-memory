"""Unit tests for ingest/change_detection.py (fingerprint-based change detection).

All tests are GPU-free and use only tmp_path — no LightRAG instance, no
network calls.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from hars_memory.ingest.change_detection import (
    FingerprintStore,
    FingerprintStoreError,
    compute_fingerprint,
    default_fingerprint_store_path,
    detect_changed_documents,
)
from hars_memory.ingest.document import Document, SourceKind


def _doc(doc_id: str, content: str) -> Document:
    return Document(
        doc_id=doc_id,
        content=content,
        source_kind=SourceKind.MARKDOWN,
        source_path=f"/fake/{doc_id}.md",
    )


# ---------------------------------------------------------------------------
# compute_fingerprint
# ---------------------------------------------------------------------------


class TestComputeFingerprint:
    def test_stable_for_same_content(self) -> None:
        assert compute_fingerprint("hello") == compute_fingerprint("hello")

    def test_differs_for_different_content(self) -> None:
        assert compute_fingerprint("hello") != compute_fingerprint("hello!")


# ---------------------------------------------------------------------------
# FingerprintStore
# ---------------------------------------------------------------------------


class TestFingerprintStore:
    def test_load_missing_file_is_empty_store(self, tmp_path: Path) -> None:
        store = FingerprintStore.load(tmp_path / "missing.json")
        assert store.get("file:abc") is None

    def test_save_then_load_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "fp.json"
        store = FingerprintStore.load(path)
        store.set("file:abc", "deadbeef")
        store.save()

        reloaded = FingerprintStore.load(path)
        assert reloaded.get("file:abc") == "deadbeef"

    def test_save_is_atomic_no_leftover_tmp_file(self, tmp_path: Path) -> None:
        path = tmp_path / "fp.json"
        store = FingerprintStore.load(path)
        store.set("file:abc", "deadbeef")
        store.save()
        assert path.exists()
        assert not (tmp_path / "fp.json.tmp").exists()

    def test_load_malformed_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "fp.json"
        path.write_text("{not valid json")
        with pytest.raises(FingerprintStoreError):
            FingerprintStore.load(path)

    def test_load_non_object_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "fp.json"
        path.write_text(json.dumps(["not", "a", "dict"]))
        with pytest.raises(FingerprintStoreError):
            FingerprintStore.load(path)


class TestDefaultFingerprintStorePath:
    def test_lives_alongside_working_dir(self, tmp_path: Path) -> None:
        result = default_fingerprint_store_path(tmp_path)
        assert result.parent == tmp_path
        assert result.name == "doc_fingerprints.json"


# ---------------------------------------------------------------------------
# detect_changed_documents — pure classification logic
# ---------------------------------------------------------------------------


class TestDetectChangedDocuments:
    def test_no_stored_fingerprint_is_never_changed(self, tmp_path: Path) -> None:
        """Documents with no prior fingerprint on record must NEVER be
        reported as changed — 'no fingerprint on record' == 'do not touch'."""
        store = FingerprintStore.load(tmp_path / "fp.json")
        docs = [_doc("file:new1", "content A"), _doc("file:new2", "content B")]

        report = detect_changed_documents(docs, store)

        assert report.changed_doc_ids == ()
        assert report.no_fingerprint_count == 2
        assert report.unchanged_count == 0
        # Fingerprints ARE recorded going forward.
        assert store.get("file:new1") == compute_fingerprint("content A")
        assert store.get("file:new2") == compute_fingerprint("content B")

    def test_unchanged_content_is_a_noop(self, tmp_path: Path) -> None:
        store = FingerprintStore.load(tmp_path / "fp.json")
        store.set("file:same", compute_fingerprint("stable content"))
        docs = [_doc("file:same", "stable content")]

        report = detect_changed_documents(docs, store)

        assert report.changed_doc_ids == ()
        assert report.unchanged_count == 1
        assert report.no_fingerprint_count == 0

    def test_changed_content_is_reported(self, tmp_path: Path) -> None:
        store = FingerprintStore.load(tmp_path / "fp.json")
        store.set("file:edited", compute_fingerprint("old content"))
        docs = [_doc("file:edited", "new content")]

        report = detect_changed_documents(docs, store)

        assert report.changed_doc_ids == ("file:edited",)
        assert report.unchanged_count == 0
        assert report.no_fingerprint_count == 0
        # Fingerprint is updated to the new value.
        assert store.get("file:edited") == compute_fingerprint("new content")

    def test_mixed_batch_classifies_each_independently(self, tmp_path: Path) -> None:
        store = FingerprintStore.load(tmp_path / "fp.json")
        store.set("file:unchanged", compute_fingerprint("same"))
        store.set("file:changed", compute_fingerprint("old"))
        docs = [
            _doc("file:unchanged", "same"),
            _doc("file:changed", "new"),
            _doc("file:brand_new", "brand new content"),
        ]

        report = detect_changed_documents(docs, store)

        assert report.changed_doc_ids == ("file:changed",)
        assert report.unchanged_count == 1
        assert report.no_fingerprint_count == 1


# ---------------------------------------------------------------------------
# _apply_refresh_changed — the index.py-level safety gate
# ---------------------------------------------------------------------------


def _make_rag() -> MagicMock:
    rag = MagicMock()
    rag.adelete_by_doc_id = AsyncMock()
    return rag


class TestApplyRefreshChangedDefaultSafety:
    """Prove the default path (--refresh-changed NOT passed) never deletes
    and never touches the fingerprint sidecar at all."""

    def test_refresh_changed_false_never_deletes_or_touches_store(
        self, tmp_path: Path
    ) -> None:
        from hars_memory.server.index import _apply_refresh_changed

        store_path = tmp_path / "fp.json"
        # Pre-existing store with a DIFFERENT fingerprint — if the safety gate
        # were broken, this would look "changed" and trigger a delete.
        FingerprintStore.load(store_path)  # does not create the file
        rag = _make_rag()
        docs = [_doc("file:x", "content")]

        async def run() -> None:
            await _apply_refresh_changed(
                rag,
                docs,
                refresh_changed=False,
                fingerprint_store_path=store_path,
            )

        asyncio.run(run())

        rag.adelete_by_doc_id.assert_not_awaited()
        assert not store_path.exists()  # store was never read OR written

    def test_refresh_changed_true_changed_doc_triggers_delete(
        self, tmp_path: Path
    ) -> None:
        from hars_memory.server.index import _apply_refresh_changed

        store_path = tmp_path / "fp.json"
        store_path.write_text(
            json.dumps({"file:x": compute_fingerprint("old content")})
        )
        rag = _make_rag()
        docs = [_doc("file:x", "new content")]

        async def run() -> FingerprintStore | None:
            return await _apply_refresh_changed(
                rag,
                docs,
                refresh_changed=True,
                fingerprint_store_path=store_path,
            )

        store = asyncio.run(run())

        rag.adelete_by_doc_id.assert_awaited_once_with("file:x")
        # The helper itself must NOT persist: the caller saves only after a
        # successful insert, otherwise an interrupted run would skip the doc.
        assert json.loads(store_path.read_text()) == {
            "file:x": compute_fingerprint("old content")
        }
        assert store is not None
        store.save()
        persisted = json.loads(store_path.read_text())
        assert persisted["file:x"] == compute_fingerprint("new content")

    def test_refresh_changed_true_no_prior_fingerprint_never_deletes(
        self, tmp_path: Path
    ) -> None:
        from hars_memory.server.index import _apply_refresh_changed

        store_path = tmp_path / "fp.json"  # no prior store at all
        rag = _make_rag()
        docs = [_doc("file:brand_new", "content")]

        async def run() -> FingerprintStore | None:
            return await _apply_refresh_changed(
                rag,
                docs,
                refresh_changed=True,
                fingerprint_store_path=store_path,
            )

        store = asyncio.run(run())

        rag.adelete_by_doc_id.assert_not_awaited()
        # Nothing on disk yet — the caller persists after a successful insert.
        assert not store_path.exists()
        assert store is not None
        # Fingerprint IS recorded going forward even though nothing deleted.
        store.save()
        persisted = json.loads(store_path.read_text())
        assert persisted["file:brand_new"] == compute_fingerprint("content")

    def test_refresh_changed_true_unchanged_doc_no_delete(self, tmp_path: Path) -> None:
        from hars_memory.server.index import _apply_refresh_changed

        store_path = tmp_path / "fp.json"
        store_path.write_text(
            json.dumps({"file:x": compute_fingerprint("same content")})
        )
        rag = _make_rag()
        docs = [_doc("file:x", "same content")]

        async def run() -> None:
            await _apply_refresh_changed(
                rag,
                docs,
                refresh_changed=True,
                fingerprint_store_path=store_path,
            )

        asyncio.run(run())

        rag.adelete_by_doc_id.assert_not_awaited()
