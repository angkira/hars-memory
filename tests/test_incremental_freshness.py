"""Incremental reindex places the updated document content into the index.

After a document is edited and the indexer runs with ``--refresh-changed``,
the new content (not the old content) must be what the retrieval layer
sees.  Uses a real walker on a tiny temporary corpus and a recording
LightRAG mock to capture actual inserted content, so this test is
GPU-free and does not require a live LLM endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from hars_memory.ingest.change_detection import FingerprintStore


# ---------------------------------------------------------------------------
# Recording mock for LightRAG
# ---------------------------------------------------------------------------


@dataclass
class _InsertCall:
    """Record of a single ``ainsert`` invocation."""
    texts: list[str]
    ids: list[str]
    file_paths: list[str]


class _RecordingRag:
    """Mock LightRAG that records every ``ainsert`` / ``adelete_by_doc_id``.

    No real LLM, embedding, or storage backends are touched.
    """

    def __init__(self) -> None:
        self.inserts: list[_InsertCall] = []
        self.deleted: list[str] = []

    async def initialize_storages(self) -> None:
        pass

    async def finalize_storages(self) -> None:
        pass

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        self.deleted.append(doc_id)

    async def ainsert(self, texts: list, ids: list, file_paths: list) -> None:
        self.inserts.append(
            _InsertCall(
                texts=list(texts),
                ids=list(ids),
                file_paths=list(file_paths),
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _args(paths: list[str]) -> argparse.Namespace:
    """Minimal ``argparse.Namespace`` for ``index._run_indexing``."""
    return argparse.Namespace(
        paths=paths,
        full=False,
        refresh_changed=True,
        dry_run=False,
    )


def _all_text(inserts: list[_InsertCall]) -> str:
    """Concatenate all text content across all insert calls."""
    return " ".join(t for ins in inserts for t in ins.texts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIncrementalReindexContentFreshness:
    """After changing a file and re-running the indexer with
    ``--refresh-changed``, the indexed content reflects the new version,
    not the old one.
    """

    # ------------------------------------------------------------------ #
    # Core freshness: one file changed, one stable
    # ------------------------------------------------------------------ #

    def test_incremental_reindex_places_updated_content(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        import hars_memory.server.index as index_mod
        import hars_memory.server.lightrag_init as lightrag_mod

        # ---- Setup: a tiny KB corpus on disk ----
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        alpha = kb_dir / "alpha.md"
        alpha.write_text("# Alpha\n\nAlpha content v1")
        beta = kb_dir / "beta.md"
        beta.write_text("# Beta\n\nBeta content stable")

        # ---- Setup: env vars for the test index dir ----
        monkeypatch.setenv("HARS_MEMORY_INDEX_DIR", str(tmp_path / "index"))

        # ---- Mock LightRAG to record instead of calling LLM ----
        rag = _RecordingRag()
        monkeypatch.setattr(lightrag_mod, "create_lightrag", lambda **kw: rag)

        # ============================================================ #
        # First pass: full indexing (no prior fingerprints)
        # ============================================================ #
        asyncio.run(index_mod._run_indexing(_args([str(kb_dir)])))

        # Both documents must be inserted.
        assert len(rag.inserts) > 0, "First pass must insert documents"
        text1 = _all_text(rag.inserts)
        assert "Alpha content v1" in text1, (
            "First pass must insert the old content"
        )
        assert "Beta content stable" in text1, (
            "First pass must insert the stable content"
        )
        assert rag.deleted == [], (
            "First pass should not delete any documents "
            "(no prior fingerprints)"
        )

        # ---- Modify one file ----
        alpha.write_text("# Alpha\n\nAlpha content v2")

        # Reset recording so the second pass is clean.
        rag.inserts.clear()
        rag.deleted.clear()

        # ============================================================ #
        # Second pass: incremental refresh (--refresh-changed)
        # ============================================================ #
        asyncio.run(index_mod._run_indexing(_args([str(kb_dir)])))

        # The changed document must be deleted first.
        assert len(rag.deleted) > 0, (
            "Second pass must delete the changed document"
        )

        # The new content must be inserted instead of the old one.
        text2 = _all_text(rag.inserts)
        assert "Alpha content v2" in text2, (
            "Second pass must insert the new content"
        )
        assert "Alpha content v1" not in text2, (
            "Second pass must not reinsert the old content"
        )
        # The stable document must still be reinserted (unchanged).
        assert "Beta content stable" in text2, (
            "Second pass must keep the unchanged content"
        )

        # ---- Fingerprint store must reflect the new content ----
        fp_path = tmp_path / "index" / "doc_fingerprints.json"
        assert fp_path.exists(), (
            "Fingerprint store must exist after indexing"
        )
        store = FingerprintStore.load(fp_path)
        assert len(store.keys()) == 2, (
            "Fingerprint store must have 2 entries (one per file)"
        )
        # Every stored fingerprint must be a non-empty sha256 hex digest.
        for doc_id, fp in [
            (k, store.get(k)) for k in sorted(store.keys())
        ]:
            assert fp is not None, (
                f"Fingerprint for {doc_id} must not be None"
            )
            assert len(fp) == 64, (
                f"Fingerprint for {doc_id} must be a 64-char sha256 hex "
                f"digest, got {len(fp)} chars"
            )

    # ------------------------------------------------------------------ #
    # Edge case: no changes between runs
    # ------------------------------------------------------------------ #

    def test_no_changes_does_not_delete_anything(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When no file has changed, the second pass must not delete
        any document — only reinsert unchanged content."""
        import hars_memory.server.index as index_mod
        import hars_memory.server.lightrag_init as lightrag_mod

        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        (kb_dir / "stable.md").write_text("# Stable\n\nNever changes.")

        monkeypatch.setenv("HARS_MEMORY_INDEX_DIR", str(tmp_path / "index"))
        rag = _RecordingRag()
        monkeypatch.setattr(lightrag_mod, "create_lightrag", lambda **kw: rag)

        # First pass
        asyncio.run(index_mod._run_indexing(_args([str(kb_dir)])))
        rag.inserts.clear()
        rag.deleted.clear()

        # Second pass — no file changed
        asyncio.run(index_mod._run_indexing(_args([str(kb_dir)])))

        assert rag.deleted == [], (
            "No changes: no document should be deleted"
        )
        # The unchanged document is still reinserted (that is normal
        # LightRAG behavior — the test only asserts nothing is deleted).

    # ------------------------------------------------------------------ #
    # Edge case: stale fingerprint is not kept after full reindex
    # ------------------------------------------------------------------ #

    def test_full_reindex_ignores_existing_fingerprints(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When ``--full`` is passed, the existing fingerprint store must
        be wiped and all documents treated as new."""
        import hars_memory.server.index as index_mod
        import hars_memory.server.lightrag_init as lightrag_mod

        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        (kb_dir / "alpha.md").write_text("# Alpha\n\nAlpha content v1")

        monkeypatch.setenv("HARS_MEMORY_INDEX_DIR", str(tmp_path / "index"))
        rag = _RecordingRag()
        monkeypatch.setattr(lightrag_mod, "create_lightrag", lambda **kw: rag)

        # First pass (refresh-changed)
        args = _args([str(kb_dir)])
        asyncio.run(index_mod._run_indexing(args))

        rag.inserts.clear()
        rag.deleted.clear()

        # Modify the file
        (kb_dir / "alpha.md").write_text("# Alpha\n\nAlpha content v2")

        # Second pass with --full (no fingerprint check)
        full_args = argparse.Namespace(
            paths=[str(kb_dir)],
            full=True,
            refresh_changed=False,
            dry_run=False,
        )
        asyncio.run(index_mod._run_indexing(full_args))

        # With --full, no delete is issued (refresh_changed is off).
        assert rag.deleted == [], (
            "--full with refresh_changed=False: no deletes"
        )
        # The new content is still inserted (--full wipes the working
        # dir, so everything is inserted fresh).
        text = _all_text(rag.inserts)
        assert "Alpha content v2" in text, (
            "--full must insert the new content"
        )