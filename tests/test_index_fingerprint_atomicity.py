"""Fingerprints must be persisted only AFTER a successful insert.

Regression guard for the interrupted-indexing data-loss bug: `--refresh-changed`
deletes a changed document's stale content and then reinserts it.  If the
fingerprint sidecar were saved before/independently of the insert, an insert
that raised (or was cancelled by a signal) would leave the document deleted
from the index yet marked "unchanged" forever — silently lost.

GPU-free: LightRAG is a mock, the walker and the insert helper are patched.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hars_memory.ingest.change_detection import compute_fingerprint
from hars_memory.ingest.document import Document, SourceKind


def _doc(doc_id: str, content: str) -> Document:
    return Document(
        doc_id=doc_id,
        content=content,
        source_kind=SourceKind.MARKDOWN,
        source_path=f"/fake/{doc_id}.md",
        metadata={"relative_path": f"{doc_id}.md"},
    )


def _args(paths: list[str]) -> argparse.Namespace:
    return argparse.Namespace(
        paths=paths,
        full=False,
        refresh_changed=True,
        dry_run=False,
    )


def _make_rag() -> MagicMock:
    rag = MagicMock()
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.adelete_by_doc_id = AsyncMock()
    rag.ainsert = AsyncMock()
    return rag


@pytest.fixture()
def wired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Patch the walker + LightRAG factory; return (store_path, rag, docs)."""
    import hars_memory.server.index as index_mod
    import hars_memory.server.lightrag_init as lightrag_mod

    store_path = tmp_path / "fingerprints.json"
    store_path.write_text(json.dumps({"file:x": compute_fingerprint("old content")}))
    monkeypatch.setenv("HARS_MEMORY_FINGERPRINT_STORE", str(store_path))
    monkeypatch.setenv("HARS_MEMORY_INDEX_DIR", str(tmp_path / "index"))

    docs = [_doc("file:x", "new content")]
    stats = SimpleNamespace(files_accepted=len(docs), per_kind={"markdown": len(docs)})
    monkeypatch.setattr(index_mod, "walk", lambda *_a, **_kw: (docs, stats))

    rag = _make_rag()
    monkeypatch.setattr(lightrag_mod, "create_lightrag", lambda *_a, **_kw: rag)

    return SimpleNamespace(store_path=store_path, rag=rag, docs=docs, mod=index_mod)


def _persisted(store_path: Path) -> dict[str, str]:
    return json.loads(store_path.read_text())


class TestFingerprintSavedOnlyAfterInsert:
    def test_insert_failure_leaves_stale_fingerprint(
        self, wired, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def failing_insert(*_a: object, **_kw: object) -> None:
            raise RuntimeError("extractor LLM died mid-batch")

        monkeypatch.setattr(wired.mod, "_insert_all_batches", failing_insert)

        with pytest.raises(RuntimeError):
            asyncio.run(wired.mod._run_indexing(_args([str(tmp_path)])))

        # Stale content WAS deleted, but the fingerprint must still be the OLD
        # one so the next run detects the document as changed again.
        wired.rag.adelete_by_doc_id.assert_awaited_once_with("file:x")
        assert _persisted(wired.store_path) == {
            "file:x": compute_fingerprint("old content")
        }

    def test_rerun_after_failure_reindexes_the_document(
        self, wired, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def failing_insert(*_a: object, **_kw: object) -> None:
            raise RuntimeError("interrupted")

        monkeypatch.setattr(wired.mod, "_insert_all_batches", failing_insert)
        with pytest.raises(RuntimeError):
            asyncio.run(wired.mod._run_indexing(_args([str(tmp_path)])))

        inserted: list[str] = []

        async def recording_insert(_rag: object, docs: list, _batch: int) -> None:
            inserted.extend(d.doc_id for d in docs)

        monkeypatch.setattr(wired.mod, "_insert_all_batches", recording_insert)
        asyncio.run(wired.mod._run_indexing(_args([str(tmp_path)])))

        assert inserted == ["file:x"]
        assert _persisted(wired.store_path) == {
            "file:x": compute_fingerprint("new content")
        }

    def test_successful_insert_persists_new_fingerprint(
        self, wired, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def ok_insert(*_a: object, **_kw: object) -> None:
            return None

        monkeypatch.setattr(wired.mod, "_insert_all_batches", ok_insert)

        asyncio.run(wired.mod._run_indexing(_args([str(tmp_path)])))

        assert _persisted(wired.store_path) == {
            "file:x": compute_fingerprint("new content")
        }

    def test_cancelled_insert_does_not_persist_fingerprint(
        self, wired, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def cancelled_insert(*_a: object, **_kw: object) -> None:
            raise asyncio.CancelledError

        monkeypatch.setattr(wired.mod, "_insert_all_batches", cancelled_insert)

        # A cancelled run exits 130 (graceful shutdown) — the sidecar must be
        # untouched so the resumed run reinserts the document.
        with pytest.raises(SystemExit) as excinfo:
            asyncio.run(wired.mod._run_indexing(_args([str(tmp_path)])))

        assert excinfo.value.code == 130
        assert _persisted(wired.store_path) == {
            "file:x": compute_fingerprint("old content")
        }

    def test_refresh_changed_off_never_writes_sidecar(
        self, wired, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def ok_insert(*_a: object, **_kw: object) -> None:
            return None

        monkeypatch.setattr(wired.mod, "_insert_all_batches", ok_insert)
        args = _args([str(tmp_path)])
        args.refresh_changed = False

        asyncio.run(wired.mod._run_indexing(args))

        wired.rag.adelete_by_doc_id.assert_not_awaited()
        assert _persisted(wired.store_path) == {
            "file:x": compute_fingerprint("old content")
        }
