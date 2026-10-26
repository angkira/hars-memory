"""Hermetic tests for the session-compaction MCP tools and recall channel.

Reloads ``hars_memory.mcp_server`` with per-test env (the same
``_load_mcp_module_with_env`` helper tests/test_mcp_server.py uses), replaces
the ingest pipeline with an in-memory recorder, and never touches a real
LightRAG index, embedder or LLM.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import hars_memory.session_memory as session_memory_module
from hars_memory.ingest.api import IngestResult
from tests.test_mcp_server import (
    _FakeRag,
    _load_mcp_module_with_env,
    asyncio_run_list_tools,
)

SESSION_ID = "sess-test"


# ---------------------------------------------------------------------------
# Fixtures / doubles
# ---------------------------------------------------------------------------


class _StoreStub:
    """Records store calls; returns/raises whatever the test configured."""

    def __init__(
        self,
        *,
        compaction_payload: dict[str, Any] | None = None,
        merge_payload: dict[str, Any] | None = None,
        status_payload: dict[str, Any] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.compaction_payload = compaction_payload or {"ok": True, "seq": 1}
        self.merge_payload = merge_payload or {"ok": True, "merged": True}
        self.status_payload = status_payload or {"sessions": []}
        self.raises = raises
        self.compaction_calls: list[dict[str, Any]] = []
        self.merge_calls: list[dict[str, Any]] = []
        self.status_calls: list[Any] = []

    def store_compaction(self, session_id: str, summary: str, **kwargs: Any) -> dict[str, Any]:
        self.compaction_calls.append({"session_id": session_id, "summary": summary, **kwargs})
        if self.raises is not None:
            raise self.raises
        return self.compaction_payload

    def merge_session(self, session_id: str, delete_source: bool = False) -> dict[str, Any]:
        self.merge_calls.append({"session_id": session_id, "delete_source": delete_source})
        if self.raises is not None:
            raise self.raises
        return self.merge_payload

    def session_status(self, session_id: str | None = None) -> dict[str, Any]:
        self.status_calls.append(session_id)
        if self.raises is not None:
            raise self.raises
        if session_id is None:
            return self.status_payload
        return {"session_id": session_id, "exists": True, "compactions": 3, "merged": False}


class _SimIngest:
    """In-memory ingest stand-in that ALSO persists full docs for merge."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.batch_sizes: list[int | None] = []

    def __call__(
        self,
        docs: list[Any],
        *,
        index_dir: Path,
        vector_storage: str | None = None,
        graph_storage: str | None = None,
        batch_size: int | None = None,
        embed_batch_size: int | None = None,
    ) -> IngestResult:
        self.calls.append({
            "docs": list(docs),
            "index_dir": Path(index_dir),
            "vector_storage": vector_storage,
            "graph_storage": graph_storage,
        })
        self.batch_sizes.append(batch_size)
        full_docs_path = Path(index_dir) / session_memory_module.FULL_DOCS_FILENAME
        existing: dict[str, Any] = {}
        if full_docs_path.is_file():
            existing = json.loads(full_docs_path.read_text(encoding="utf-8"))
        for document in docs:
            existing[document.doc_id] = {"content": document.content}
        full_docs_path.parent.mkdir(parents=True, exist_ok=True)
        full_docs_path.write_text(json.dumps(existing), encoding="utf-8")
        return IngestResult(documents_written=len(docs), documents_skipped=0, errors=[])


def _write_session_fixture(
    tmp_path: Path,
    session_id: str = SESSION_ID,
    *,
    merged: bool = False,
) -> Path:
    """Create a split-mode session dir with chunks + a known 3-d vector store."""
    session_dir = tmp_path / "index" / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    chunks = {
        "chunk-1": {
            "content": "Q7X42 stabilized the sidecar gate at step 80",
            "file_path": f"sessions/{session_id}/compaction-1.md",
        },
        "chunk-2": {
            "content": "pasta cooking recipes and grocery lists",
            "file_path": f"sessions/{session_id}/compaction-2.md",
        },
    }
    (session_dir / session_memory_module.FULL_DOCS_FILENAME).write_text(
        json.dumps({}), encoding="utf-8"
    )
    (session_dir / "kv_store_text_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    vdb = {
        "embedding_dim": 3,
        "data": [
            {"__id__": "chunk-1", "file_path": chunks["chunk-1"]["file_path"]},
            {"__id__": "chunk-2", "file_path": chunks["chunk-2"]["file_path"]},
        ],
        "matrix": base64.b64encode(matrix.tobytes()).decode(),
    }
    (session_dir / "vdb_chunks.json").write_text(json.dumps(vdb), encoding="utf-8")
    if merged:
        (session_dir / "merged.json").write_text(
            json.dumps({"merged_at": "2026-10-22T00:00:00", "compactions": 3}),
            encoding="utf-8",
        )
    return session_dir


def _hybrid_fixture() -> dict[str, Any]:
    """Minimal hybrid block with one ordinary fused chunk."""
    return {
        "enabled": True,
        "alpha": 0.5,
        "candidate_pool_size": 1,
        "fused_chunks": [
            {
                "chunk_id": "base-1",
                "fused_score": 0.5,
                "dense_score": 0.5,
                "sparse_score": 0.5,
                "file_path": "docs/base.md",
                "content": "baseline knowledge about the Phase C rollout",
                "snippet": "baseline knowledge",
            }
        ],
        "confidence": {
            "top_dense_score": 0.5,
            "low_confidence": False,
            "threshold": 0.28,
            "note": "stub",
        },
        "rerank": {
            "backend": "off",
            "applied": False,
            "latency_ms": None,
            "fallback_reason": "disabled_by_default",
            "pool": 0,
        },
        "latency_ms": {"dense_channel": 1.0, "sparse_channel": 0.5},
        "bm25_index": {"chunks_indexed": 1},
    }


def _recall_mod(tmp_path: Path, **env: str) -> Any:
    base = {
        "HARS_MEMORY_INDEX_DIR": str(tmp_path / "index"),
        "HARS_MEMORY_SESSIONS_DIR": str(tmp_path / "index" / "sessions"),
        "HARS_MEMORY_SESSION_MODE": "split",
    }
    base.update(env)
    return _load_mcp_module_with_env(base)


class _RecallRag(_FakeRag):
    async def aquery(self, _question: str, *, param: object) -> str:  # noqa: ARG002
        return "[no-context]"


async def _embed_query(texts: list[str], context: str | None = None) -> np.ndarray:  # noqa: ARG001
    return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


def _call_recall(
    mod: Any, monkeypatch: pytest.MonkeyPatch, **args: Any
) -> dict[str, Any]:
    hybrid = _hybrid_fixture()

    async def fake_hybrid(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return copy.deepcopy(hybrid)

    monkeypatch.setattr(mod, "_compute_hybrid_block", fake_hybrid)
    mod._rag_instance = _RecallRag([], embedding_func=_embed_query)
    result = asyncio.run(mod.call_tool(
        "memory_recall",
        {"question": "Q7X42 sidecar gate", "ll_keywords": ["Q7X42"], "context_only": True, **args},
    ))
    return json.loads(result[0].text)


# ---------------------------------------------------------------------------
# (i) tool registration
# ---------------------------------------------------------------------------


def test_three_session_tools_are_registered_with_expected_schema(tmp_path: Path) -> None:
    mod = _recall_mod(tmp_path)
    tools = {tool.name: tool for tool in asyncio_run_list_tools(mod)}

    assert {"memory_compact", "memory_merge_session", "memory_session_status"} <= set(tools)

    compact_schema = tools["memory_compact"].inputSchema
    assert compact_schema["required"] == ["session_id", "summary"]
    assert compact_schema["properties"]["seq"]["minimum"] == 1
    assert compact_schema["properties"]["dropped_context"]["type"] == ["array", "string"]

    merge_schema = tools["memory_merge_session"].inputSchema
    assert merge_schema["required"] == ["session_id"]
    assert merge_schema["properties"]["delete_source"]["default"] is False

    status_schema = tools["memory_session_status"].inputSchema
    assert not status_schema.get("required")


def test_session_tool_scopes(tmp_path: Path) -> None:
    from hars_memory.mcp_oauth import SCOPE_READ, SCOPE_WRITE, required_scope_for_tool

    mod = _recall_mod(tmp_path)
    assert required_scope_for_tool("memory_compact") == SCOPE_WRITE
    assert required_scope_for_tool("memory_merge_session") == SCOPE_WRITE
    assert required_scope_for_tool("memory_session_status") == SCOPE_READ
    assert mod is not None


# ---------------------------------------------------------------------------
# (ii) memory_compact
# ---------------------------------------------------------------------------


def test_memory_compact_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _recall_mod(tmp_path)
    store = _StoreStub(compaction_payload={
        "ok": True, "session_id": SESSION_ID, "seq": 4,
        "doc_id": f"session:{SESSION_ID}:c4", "mode": "split", "index_dir": "x",
    })
    monkeypatch.setattr(mod, "_get_session_store", lambda project_id="default": store)

    result = asyncio.run(mod.call_tool("memory_compact", {
        "session_id": SESSION_ID,
        "summary": "the compaction summary",
        "dropped_context": ["fact one", "fact two"],
        "metadata": {"attempt": 1},
        "seq": 4,
    }))
    data = json.loads(result[0].text)

    assert data["ok"] is True
    assert data["seq"] == 4
    call = store.compaction_calls[0]
    assert call["session_id"] == SESSION_ID
    assert call["summary"] == "the compaction summary"
    assert call["dropped_context"] == ["fact one", "fact two"]
    assert call["metadata"] == {"attempt": 1}
    assert call["seq"] == 4


def test_memory_compact_error_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _recall_mod(tmp_path)

    missing_session = json.loads(asyncio.run(mod.call_tool(
        "memory_compact", {"summary": "s"}
    ))[0].text)
    assert missing_session["ok"] is False
    assert "session_id" in missing_session["error"]

    missing_summary = json.loads(asyncio.run(mod.call_tool(
        "memory_compact", {"session_id": SESSION_ID}
    ))[0].text)
    assert missing_summary["ok"] is False
    assert "summary" in missing_summary["error"]

    bad_metadata = json.loads(asyncio.run(mod.call_tool("memory_compact", {
        "session_id": SESSION_ID, "summary": "s", "metadata": "not-an-object",
    }))[0].text)
    assert bad_metadata["ok"] is False

    store = _StoreStub(raises=ValueError("invalid session_id: expected 1-128 characters"))
    monkeypatch.setattr(mod, "_get_session_store", lambda project_id="default": store)
    failure = json.loads(asyncio.run(mod.call_tool("memory_compact", {
        "session_id": "../etc", "summary": "s",
    }))[0].text)
    assert failure["ok"] is False
    assert failure["error_type"] == "ValueError"
    assert "invalid session_id" in failure["error"]


# ---------------------------------------------------------------------------
# (iii) memory_session_status
# ---------------------------------------------------------------------------


def test_memory_session_status_list_and_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _recall_mod(tmp_path)
    store = _StoreStub(status_payload={
        "mode": "split",
        "sessions_dir": "/tmp/sessions",
        "sessions": [{"session_id": SESSION_ID, "compactions": 3}],
    })
    monkeypatch.setattr(mod, "_get_session_store", lambda project_id="default": store)

    listing = json.loads(asyncio.run(mod.call_tool("memory_session_status", {}))[0].text)
    assert listing["ok"] is True
    assert listing["mode"] == "split"
    assert listing["sessions"][0]["session_id"] == SESSION_ID

    detail = json.loads(asyncio.run(mod.call_tool(
        "memory_session_status", {"session_id": SESSION_ID}
    ))[0].text)
    assert detail["ok"] is True
    assert detail["session_id"] == SESSION_ID
    assert detail["compactions"] == 3
    assert store.status_calls == [None, SESSION_ID]


def test_memory_session_status_error_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _recall_mod(tmp_path)
    monkeypatch.setattr(
        mod, "_get_session_store",
        lambda project_id="default": _StoreStub(raises=ValueError("bad id")),
    )
    data = json.loads(asyncio.run(mod.call_tool(
        "memory_session_status", {"session_id": "bad id"}
    ))[0].text)
    assert data["ok"] is False
    assert data["error_type"] == "ValueError"


# ---------------------------------------------------------------------------
# (iv) recall session channel
# ---------------------------------------------------------------------------


def test_recall_appends_session_channel_for_unmerged_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)

    data = _call_recall(mod, monkeypatch, session_id=SESSION_ID)

    assert data["ok"] is True
    assert data["session_id"] == SESSION_ID
    assert data["session"]["session_id"] == SESSION_ID
    assert data["session"]["merged"] is False
    assert data["session"]["hits"] == 1  # chunk-2 (0.0 score) is below the cutoff
    assert data["hybrid"]["session"] == data["session"]

    fused = data["hybrid"]["fused_chunks"]
    session_chunks = [chunk for chunk in fused if chunk.get("channel") == "session"]
    assert len(session_chunks) == data["session"]["hits"]
    first = session_chunks[0]
    assert first["chunk_id"].startswith(f"session:{SESSION_ID}:")
    assert first["session_id"] == SESSION_ID
    assert "sidecar" in first["content"]
    # The pre-existing fused chunk is preserved (append, never evict).
    assert any(chunk["chunk_id"] == "base-1" for chunk in fused)

    # The appended session facts also reach the assembled context.
    assert "sidecar" in data["context"]


def test_recall_merged_session_skips_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path, merged=True)
    mod = _recall_mod(tmp_path)

    data = _call_recall(mod, monkeypatch, session_id=SESSION_ID)

    assert data["ok"] is True
    assert data["session"]["merged"] is True
    assert data["session"]["hits"] == 0
    assert "already merged" in data["session"]["skipped"]
    assert not any(
        str(chunk.get("chunk_id", "")).startswith(f"session:{SESSION_ID}:")
        for chunk in data["hybrid"]["fused_chunks"]
    )


def test_recall_unknown_session_errors_but_recall_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _recall_mod(tmp_path)

    data = _call_recall(mod, monkeypatch, session_id="never-seen")

    assert data["ok"] is True
    assert data["session"]["error"] == "unknown session"
    assert data["session"]["hits"] == 0


def test_recall_session_min_score_env_drops_all_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HARS_MEMORY_SESSION_MIN_SCORE is the cut-off applied to max(cosine, F1)."""
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path, HARS_MEMORY_SESSION_MIN_SCORE="1.5")

    data = _call_recall(mod, monkeypatch, session_id=SESSION_ID)

    assert data["ok"] is True
    assert data["session"]["hits"] == 0
    assert data["session"]["below_threshold"] >= 1


def test_recall_without_session_id_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)

    data = _call_recall(mod, monkeypatch)

    assert "session" not in data
    assert "session_id" not in data
    assert not any(
        chunk.get("channel") == "session" for chunk in data["hybrid"]["fused_chunks"]
    )


def test_recall_subgraph_mode_skips_session_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path, HARS_MEMORY_SESSION_MODE="subgraph")

    data = _call_recall(mod, monkeypatch, session_id=SESSION_ID)

    assert data["ok"] is True
    assert data["session"]["hits"] == 0
    assert "subgraph" in data["session"]["skipped"]


def test_recall_session_channel_survives_corrupt_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = _write_session_fixture(tmp_path)
    (session_dir / "vdb_chunks.json").write_text("{not json", encoding="utf-8")
    mod = _recall_mod(tmp_path)

    data = _call_recall(mod, monkeypatch, session_id=SESSION_ID)

    # Lexical F1 still finds the chunk even with an unusable vector store.
    assert data["ok"] is True
    assert data["session"]["hits"] >= 1
    assert any(
        chunk.get("channel") == "session" for chunk in data["hybrid"]["fused_chunks"]
    )


def test_recall_session_channel_lean_view_includes_session_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)

    data = _call_recall(mod, monkeypatch, session_id=SESSION_ID, view="lean")

    assert data["view"] == "lean"
    assert data["session"]["session_id"] == SESSION_ID
    assert data["session_id"] == SESSION_ID


def test_recall_session_channel_compact_view_does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`compact=true` de-duplicates fused chunk text; new session fields must
    not break it (they are kept; snippet/content handling is unchanged)."""
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)

    data = _call_recall(mod, monkeypatch, session_id=SESSION_ID, compact=True)

    assert data["ok"] is True
    session_chunks = [
        chunk for chunk in data["hybrid"]["fused_chunks"]
        if chunk.get("channel") == "session"
    ]
    assert session_chunks
    assert all("snippet" not in chunk for chunk in session_chunks)


# ---------------------------------------------------------------------------
# (v) 30-compaction simulation with batch-size-25 capture + single merge ingest
# ---------------------------------------------------------------------------


def test_simulation_30_compactions_batches_and_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _SimIngest()
    monkeypatch.setattr(session_memory_module, "ingest_documents", recorder)
    store = session_memory_module.SessionMemoryStore(
        index_dir=tmp_path / "index", mode="split"
    )

    seqs = [
        store.store_compaction(SESSION_ID, f"compaction summary {index}")["seq"]
        for index in range(30)
    ]

    assert seqs == list(range(1, 31))
    assert len(recorder.calls) == 30
    assert recorder.batch_sizes == [25] * 30
    assert all(call["index_dir"] == tmp_path / "index" / "sessions" / SESSION_ID
               for call in recorder.calls)

    merged = store.merge_session(SESSION_ID)

    assert merged["ok"] is True
    assert merged["merged"] is True
    assert merged["compactions"] == 30
    assert len(recorder.calls) == 31  # exactly ONE extra ingest, into MAIN
    merge_call = recorder.calls[-1]
    assert merge_call["index_dir"] == tmp_path / "index"
    assert len(merge_call["docs"]) == 1
    assert merge_call["docs"][0].doc_id == f"session:{SESSION_ID}:merged"
    assert "## Compaction 30" in merge_call["docs"][0].content

    status = store.session_status(SESSION_ID)
    assert status["merged"] is True
    assert status["compactions"] == 30
