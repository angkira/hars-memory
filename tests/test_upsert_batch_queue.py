"""MCP-level tests for ``HARS_MEMORY_UPSERT_BATCH_MODE=queue`` (memory_upsert_document).

Queue mode turns the synchronous insert into an inbox write: no LLM, no purge,
no fingerprint recorded — the document is indexed later by
``memory index-batch --source upsert-inbox`` (see tests/test_upsert_inbox.py).
Default (flag unset) must stay byte-for-byte on the synchronous path.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from hars_memory.ingest.change_detection import (
    FingerprintStore,
    compute_fingerprint,
    default_fingerprint_store_path,
)
from hars_memory.ingest.document import file_stable_id
from hars_memory.ingest.upsert_inbox import UpsertInboxStore, default_upsert_inbox_path

# helper shared with the content-dedup tests: the fingerprint upsert computes
from tests.test_mcp_knowledge_tools import _upsert_fingerprint
from tests.test_mcp_server import _load_mcp_module_with_env

CONTENT = "# Session Notes\nVerbatim transcript payload for the queue mode."


def _final_content(doc_file: Path, content: str) -> str:
    """The header-applied content memory_upsert_document builds for (path, content)."""
    from hars_memory.ingest.walker import (
        HEADER_DATE_UNKNOWN,
        _file_date,
        _infer_section,
        apply_source_header,
        build_source_header,
        infer_source_kind,
    )

    kind = infer_source_kind(doc_file)
    mtime = doc_file.stat().st_mtime if doc_file.exists() else 0.0
    date_str = _file_date(doc_file, mtime) if doc_file.exists() else HEADER_DATE_UNKNOWN
    header = build_source_header(
        document_name=doc_file.name,
        section=_infer_section(doc_file.parent, kind),
        date=date_str,
    )
    return apply_source_header(content, header)


class TestUpsertBatchQueueMode:
    def _load(self, tmp_path: Path, extra: dict[str, str] | None = None) -> Any:
        env = {
            "HARS_MEMORY_INDEX_DIR": str(tmp_path / "index"),
            "HARS_MEMORY_STAGING_DIR": str(tmp_path / "staging"),
            "HARS_MEMORY_UPSERT_BATCH_MODE": "queue",
        }
        env.update(extra or {})
        return _load_mcp_module_with_env(env)

    def test_default_without_flag_still_inserts_synchronously(self, tmp_path: Path) -> None:
        """Flag unset: exactly the old path — insert now, no inbox sidecar."""
        index_dir = tmp_path / "index"
        doc_file = tmp_path / "doc.md"
        doc_file.write_text("# Doc\nContent.", encoding="utf-8")

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(tmp_path / "staging"),
        })
        mock_rag = AsyncMock()
        mock_rag.adelete_by_doc_id = AsyncMock()
        with patch.object(mod, "_get_rag", new=AsyncMock(return_value=mock_rag)), \
             patch("hars_memory.server.index._insert_all_batches", new=AsyncMock()) as mock_insert:
            res = asyncio.run(mod.call_tool("memory_upsert_document", {"file_path": str(doc_file)}))
        data = json.loads(res[0].text)
        assert data["ok"] is True
        assert data["action"] == "inserted"
        mock_insert.assert_called_once()
        assert not default_upsert_inbox_path(str(index_dir)).exists()

    def test_queue_mode_answers_queued_for_batch_without_touching_the_index(
        self, tmp_path: Path
    ) -> None:
        index_dir = tmp_path / "index"
        doc_file = tmp_path / "notes" / "session.md"
        doc_file.parent.mkdir(parents=True)
        doc_file.write_text("irrelevant on-disk body", encoding="utf-8")

        mod: Any = self._load(tmp_path)
        get_rag = AsyncMock()
        with patch.object(mod, "_get_rag", new=get_rag), \
             patch("hars_memory.server.index._insert_all_batches", new=AsyncMock()) as mock_insert:
            res = asyncio.run(
                mod.call_tool("memory_upsert_document", {
                    "file_path": str(doc_file),
                    "content": CONTENT,
                })
            )
        data = json.loads(res[0].text)

        assert data["ok"] is True
        assert data["action"] == "queued_for_batch"
        assert data["doc_id"] == file_stable_id(doc_file)
        assert data["entry_id"] == data["doc_id"]
        assert data["fingerprint"] == _upsert_fingerprint(doc_file, CONTENT)
        get_rag.assert_not_called()  # no LightRAG init, no purge, no insert
        mock_insert.assert_not_called()

        # the inbox entry holds the FINAL (header-applied) content — the
        # byte-for-byte prompt invariant the whole mode stands on
        inbox = UpsertInboxStore.load(default_upsert_inbox_path(str(index_dir)))
        assert inbox.doc_ids() == [data["doc_id"]]
        entry = inbox.get(data["doc_id"])
        assert entry["content"] == _final_content(doc_file, CONTENT)
        assert entry["fingerprint"] == compute_fingerprint(entry["content"])
        assert entry["source_path"] == str(doc_file)

        # fingerprint-after-the-fact contract: nothing recorded while only queued
        fp_path = default_fingerprint_store_path(str(index_dir))
        assert not fp_path.exists() or FingerprintStore.load(fp_path).get(data["doc_id"]) is None

    def test_queue_mode_unchanged_fast_path_still_wins(self, tmp_path: Path) -> None:
        """Already indexed identical content answers "unchanged" even in queue
        mode — queueing it again would be pointless work."""
        index_dir = tmp_path / "index"
        doc_file = tmp_path / "session.md"

        fp = _upsert_fingerprint(doc_file, CONTENT)
        store = FingerprintStore.load(default_fingerprint_store_path(str(index_dir)))
        store.set(file_stable_id(doc_file), fp)
        store.save()

        mod: Any = self._load(tmp_path)
        res = asyncio.run(
            mod.call_tool("memory_upsert_document", {
                "file_path": str(doc_file),
                "content": CONTENT,
            })
        )
        data = json.loads(res[0].text)
        assert data["ok"] is True
        assert data["action"] == "unchanged"
        assert not default_upsert_inbox_path(str(index_dir)).exists()

    def test_queue_mode_is_idempotent_for_the_same_content(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "index"
        doc_file = tmp_path / "session.md"

        mod: Any = self._load(tmp_path)
        args = {"file_path": str(doc_file), "content": CONTENT}
        with patch.object(mod, "_get_rag", new=AsyncMock()), \
             patch("hars_memory.server.index._insert_all_batches", new=AsyncMock()):
            first = json.loads(asyncio.run(mod.call_tool("memory_upsert_document", args))[0].text)
            second = json.loads(asyncio.run(mod.call_tool("memory_upsert_document", args))[0].text)

        assert first["action"] == second["action"] == "queued_for_batch"
        inbox = UpsertInboxStore.load(default_upsert_inbox_path(str(index_dir)))
        assert inbox.doc_ids() == [first["doc_id"]]  # one entry, overwritten in place

    def test_queue_mode_file_source_queues_the_walked_document(self, tmp_path: Path) -> None:
        """No content override: the walked doc (header applied by walk()) is queued."""
        index_dir = tmp_path / "index"
        doc_file = tmp_path / "walked.md"
        doc_file.write_text("# Walked\nReal file content.", encoding="utf-8")

        mod: Any = self._load(tmp_path)
        with patch.object(mod, "_get_rag", new=AsyncMock()), \
             patch("hars_memory.server.index._insert_all_batches", new=AsyncMock()):
            res = asyncio.run(mod.call_tool("memory_upsert_document", {"file_path": str(doc_file)}))
        data = json.loads(res[0].text)
        assert data["action"] == "queued_for_batch"

        inbox = UpsertInboxStore.load(default_upsert_inbox_path(str(index_dir)))
        entry = inbox.get(data["doc_id"])
        assert entry is not None
        # the walker applies the attribution header itself; the queued content is
        # the final walked content (starts with the same header as the override path)
        assert entry["content"].startswith("[Document:")
        assert "Real file content." in entry["content"]

    def test_invalid_flag_value_falls_back_to_synchronous(self, tmp_path: Path) -> None:
        """Only the exact value `queue` enables the mode; anything else (typo,
        empty) keeps the synchronous default."""
        index_dir = tmp_path / "index"
        doc_file = tmp_path / "doc.md"
        doc_file.write_text("# Doc\nContent.", encoding="utf-8")

        mod: Any = self._load(tmp_path, {"HARS_MEMORY_UPSERT_BATCH_MODE": "queues"})
        mock_rag = AsyncMock()
        mock_rag.adelete_by_doc_id = AsyncMock()
        with patch.object(mod, "_get_rag", new=AsyncMock(return_value=mock_rag)), \
             patch("hars_memory.server.index._insert_all_batches", new=AsyncMock()) as mock_insert:
            res = asyncio.run(mod.call_tool("memory_upsert_document", {"file_path": str(doc_file)}))
        data = json.loads(res[0].text)
        assert data["action"] == "inserted"
        mock_insert.assert_called_once()
        assert not default_upsert_inbox_path(str(index_dir)).exists()

    def teardown_method(self) -> None:
        """Reload the module with the ambient env so the queue flag does not
        leak into later test modules through the shared module object."""
        _load_mcp_module_with_env({})
