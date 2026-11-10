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
        timeout_s: float | None = None,
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


# ---------------------------------------------------------------------------
# (vi) degraded / failure reporting of the session channel
# ---------------------------------------------------------------------------


async def _embed_raises(texts: list[str], context: str | None = None) -> np.ndarray:  # noqa: ARG001
    raise RuntimeError("embedder unreachable at 127.0.0.1:8791 (HTTP 530)")


async def _embed_hangs(texts: list[str], context: str | None = None) -> np.ndarray:  # noqa: ARG001
    await asyncio.sleep(5)
    return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


async def _embed_wrong_dim(texts: list[str], context: str | None = None) -> np.ndarray:  # noqa: ARG001
    return np.array([[1.0, 0.0]], dtype=np.float32)


class _FailingGraphRag(_RecallRag):
    async def aquery(self, _question: str, *, param: object) -> str:  # noqa: ARG002
        raise RuntimeError("graph backend returned HTTP 530")


def _call_recall_with(
    mod: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    embedding_func: Any = _embed_query,
    rag_cls: type = _RecallRag,
    base_chunks: bool = True,
    **args: Any,
) -> dict[str, Any]:
    """`_call_recall` with a configurable embedder, RAG and base fused chunks."""
    hybrid = _hybrid_fixture()
    if not base_chunks:
        hybrid["fused_chunks"] = []

    async def fake_hybrid(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return copy.deepcopy(hybrid)

    monkeypatch.setattr(mod, "_compute_hybrid_block", fake_hybrid)
    mod._rag_instance = rag_cls([], embedding_func=embedding_func)
    result = asyncio.run(mod.call_tool(
        "memory_recall",
        {"question": "Q7X42 sidecar gate", "ll_keywords": ["Q7X42"], "context_only": True, **args},
    ))
    return json.loads(result[0].text)


def test_recall_session_channel_falls_back_to_lexical_when_embedding_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)

    data = _call_recall_with(
        mod, monkeypatch, embedding_func=_embed_raises, base_chunks=False,
        session_id=SESSION_ID,
    )

    session = data["session"]
    assert session["degraded"] == "lexical_only"
    assert session["status"] == "degraded"
    assert session["error_kind"] == "embedding"
    assert "query embedding failed" in session["error"]
    assert "HTTP 530" in session["error"]
    # Lexical F1 alone still finds the matching compaction chunk.
    assert session["hits"] == 1
    session_chunks = [
        chunk for chunk in data["hybrid"]["fused_chunks"] if chunk.get("channel") == "session"
    ]
    assert session_chunks[0]["dense_score"] == 0.0
    assert session_chunks[0]["sparse_score"] > 0.2
    assert data["ok"] is True
    assert "sidecar" in data["context"]


def test_recall_session_channel_embedding_timeout_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)
    monkeypatch.setenv("HARS_MEMORY_QUERY_TIMEOUT_SECONDS", "0.05")

    data = _call_recall_with(
        mod, monkeypatch, embedding_func=_embed_hangs, session_id=SESSION_ID,
    )

    assert data["session"]["degraded"] == "lexical_only"
    assert "TimeoutError" in data["session"]["error"]
    assert data["session"]["hits"] == 1


def test_recall_session_channel_dimension_mismatch_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)

    data = _call_recall_with(
        mod, monkeypatch, embedding_func=_embed_wrong_dim, session_id=SESSION_ID,
    )

    assert data["session"]["degraded"] == "lexical_only"
    assert data["session"]["error_kind"] == "vector_scoring"
    assert data["session"]["hits"] == 1


def test_recall_session_block_exposes_index_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import networkx as nx

    session_dir = _write_session_fixture(tmp_path)
    graph = nx.Graph()
    graph.add_edge("Q7X42", "sidecar gate")
    nx.write_graphml(graph, str(session_dir / session_memory_module.GRAPH_FILE_NAME))
    mod = _recall_mod(tmp_path)

    data = _call_recall(mod, monkeypatch, session_id=SESSION_ID)

    session = data["session"]
    assert session["status"] == "ok"
    assert "degraded" not in session
    assert "error" not in session
    assert session["entities"] == 2
    assert session["relations"] == 1
    assert session["vectors_indexed"] == 2
    assert session["chunks_indexed"] == 2


def test_recall_session_channel_hard_failure_is_ok_false_with_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken session channel with nothing else to return must never look
    like an empty success: ok=false, explicit error, error_kind set."""
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)

    def broken_vdb(_vdb: dict[str, Any]) -> tuple[list[str], Any]:
        raise RuntimeError("vdb decoder exploded")

    monkeypatch.setattr(mod, "_session_vectors_from_vdb", broken_vdb)

    data = _call_recall_with(mod, monkeypatch, base_chunks=False, session_id=SESSION_ID)

    assert data["ok"] is False
    assert not data["context"]
    assert data["error_kind"] == "session_channel"
    assert "session channel" in data["error"]
    assert "vdb decoder exploded" in data["error"]
    assert data["session"]["status"] == "error"
    assert data["session"]["hits"] == 0


def test_recall_session_legitimate_no_hits_is_distinguishable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path, HARS_MEMORY_SESSION_MIN_SCORE="1.5")

    data = _call_recall_with(mod, monkeypatch, base_chunks=False, session_id=SESSION_ID)

    assert data["ok"] is False
    assert data["error"]
    assert data["error_kind"] == "no_hits"
    assert "session channel" not in data["error"]
    assert data["session"]["status"] == "no_hits"
    assert "error" not in data["session"]

    lean = _call_recall_with(
        mod, monkeypatch, base_chunks=False, session_id=SESSION_ID, view="lean"
    )
    assert lean["error_kind"] == "no_hits"


def test_recall_exception_path_keeps_session_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)

    data = _call_recall_with(
        mod, monkeypatch, rag_cls=_FailingGraphRag, session_id=SESSION_ID,
    )

    assert data["ok"] is False
    assert "HTTP 530" in data["error"]
    assert data["error_kind"] == "recall_exception"
    assert data["session"]["status"] == "ok"
    assert data["session"]["hits"] == 1


def test_memory_session_status_reports_vectors_indexed_from_real_store(
    tmp_path: Path,
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)

    detail = json.loads(asyncio.run(mod.call_tool(
        "memory_session_status", {"session_id": SESSION_ID}
    ))[0].text)

    assert detail["ok"] is True
    assert detail["vectors_indexed"] == 2
    assert detail["chunks_indexed"] == 2
    assert detail["entities"] == 0
    assert detail["relations"] == 0


# ---------------------------------------------------------------------------
# Session-graph recall (opt-in, .plans/2026-10-26_session-graph-recall.md)
# ---------------------------------------------------------------------------

_KG_CONTEXT = (
    "\nKnowledge Graph Data (Entity):\n\n```json\n"
    '{"entity": "Q7X42", "type": "experiment", "description": "stabilized the sidecar gate"}\n'
    '{"entity": "sidecar gate", "type": "component", "description": "gate at step 80"}\n'
    "```\n\nKnowledge Graph Data (Relationship):\n\n```json\n"
    '{"entity1": "Q7X42", "entity2": "sidecar gate", "description": "stabilized"}\n'
    "```\n\nDocument Chunks (Each entry has a reference_id refer to the `Reference Document List`):"
    "\n\n```json\n{\"reference_id\": \"1\", \"content\": \"IGNORED chunk from the session rag\"}\n"
    "```\n\nReference Document List (Each entry starts with a [reference_id] that corresponds "
    "to entries in the Document Chunks):\n\n```\n[1] ignored.md\n```\n"
)


class _SessionGraphRag:
    def __init__(
        self,
        context: Any = _KG_CONTEXT,
        *,
        llm_reply: str | Exception = (
            '{"high_level_keywords": ["gate stabilisation"], "low_level_keywords": ["Q7X42"]}'
        ),
        raises: Exception | None = None,
    ) -> None:
        self.context = context
        self.llm_reply = llm_reply
        self.raises = raises
        self.queries: list[Any] = []
        self.llm_calls = 0

    async def aquery(self, question: str, param: Any) -> Any:
        self.queries.append((question, param))
        if self.raises is not None:
            raise self.raises
        return self.context

    async def llm_model_func(self, prompt: str, system_prompt: str | None = None, **_kw: Any) -> str:  # noqa: ARG002
        self.llm_calls += 1
        if isinstance(self.llm_reply, Exception):
            raise self.llm_reply
        return self.llm_reply


def _patch_sg_rag(mod: Any, monkeypatch: pytest.MonkeyPatch, sg_rag: _SessionGraphRag) -> None:
    async def fake_get(_sid: str, _dir: Path) -> _SessionGraphRag:
        return sg_rag

    monkeypatch.setattr(mod, "_get_session_graph_rag", fake_get)
    mod._session_graph_keyword_cache.clear()


def _stable(data: Any) -> str:
    """Canonical JSON with wall-clock fields removed (everything else byte-exact)."""
    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k != "latency_ms"}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node
    return json.dumps(strip(data), sort_keys=True)


def _chunk_section(context: str) -> str:
    return context[context.index("Document Chunks"):]


def test_session_graph_off_payload_is_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)
    sg_rag = _SessionGraphRag()
    _patch_sg_rag(mod, monkeypatch, sg_rag)

    golden = _stable(_call_recall(mod, monkeypatch, session_id=SESSION_ID))
    off = _stable(_call_recall(mod, monkeypatch, session_id=SESSION_ID, session_graph=False))
    assert off == golden
    assert "session_graph" not in golden
    assert sg_rag.queries == []

    monkeypatch.setenv("HARS_MEMORY_SESSION_GRAPH", "0")
    assert _stable(_call_recall(mod, monkeypatch, session_id=SESSION_ID)) == golden


def test_session_graph_on_splices_kg_above_unchanged_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)
    sg_rag = _SessionGraphRag()
    _patch_sg_rag(mod, monkeypatch, sg_rag)

    off = _call_recall(mod, monkeypatch, session_id=SESSION_ID)
    on = _call_recall(mod, monkeypatch, session_id=SESSION_ID, session_graph=True)

    ctx = on["context"]
    assert ctx.index("Knowledge Graph Data (Entity)") < ctx.index("Knowledge Graph Data (Relationship)")
    assert ctx.index("Knowledge Graph Data (Relationship)") < ctx.index("Document Chunks")
    assert '"entity": "Q7X42"' in ctx
    assert "IGNORED chunk" not in ctx  # session rag's own chunk section is discarded
    assert _chunk_section(ctx) == _chunk_section(off["context"])  # chunk side unchanged
    sg = on["session_graph"]
    assert (sg["status"], sg["entities"], sg["relations"], sg["empty"]) == ("ok", 2, 1, False)
    assert sg["keyword_source"] == "caller"
    assert sg["ll_keywords"] == ["Q7X42"]
    assert sg["mode"] == "hybrid"
    assert sg_rag.llm_calls == 0
    (_q, param), = sg_rag.queries
    assert param.mode == "hybrid" and param.only_need_context is True and param.enable_rerank is False
    # everything else in the payload is unchanged
    for key in ("session", "hybrid", "graph_branch"):
        assert _stable(on[key]) == _stable(off[key])


def test_session_graph_env_default_on_and_arg_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)
    monkeypatch.setenv("HARS_MEMORY_SESSION_GRAPH", "on")
    _patch_sg_rag(mod, monkeypatch, _SessionGraphRag())
    assert _call_recall(mod, monkeypatch, session_id=SESSION_ID)["session_graph"]["status"] == "ok"
    assert "session_graph" not in _call_recall(
        mod, monkeypatch, session_id=SESSION_ID, session_graph=False
    )


def test_session_graph_keyword_order_caller_llm_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)

    # llm: no caller keywords -> exactly ONE extractor call, then cached per question+session
    sg_rag = _SessionGraphRag()
    _patch_sg_rag(mod, monkeypatch, sg_rag)
    a = _call_recall(mod, monkeypatch, session_id=SESSION_ID, session_graph=True, ll_keywords=[])
    b = _call_recall(mod, monkeypatch, session_id=SESSION_ID, session_graph=True, ll_keywords=[])
    assert a["session_graph"]["keyword_source"] == "llm"
    assert a["session_graph"]["ll_keywords"] == ["Q7X42"]
    assert a["session_graph"]["hl_keywords"] == ["gate stabilisation"]
    assert b["session_graph"]["keyword_cached"] is True
    assert sg_rag.llm_calls == 1
    assert sg_rag.queries[0][1].ll_keywords == ["Q7X42"]
    assert sg_rag.queries[0][1].hl_keywords == ["gate stabilisation"]

    # question fallback: llm raises / returns garbage; not cached
    for reply in (RuntimeError("boom"), "not json"):
        sg_rag = _SessionGraphRag(llm_reply=reply)
        _patch_sg_rag(mod, monkeypatch, sg_rag)
        c = _call_recall(mod, monkeypatch, session_id=SESSION_ID, session_graph=True, ll_keywords=[])
        sg = c["session_graph"]
        assert sg["keyword_source"] == "question_fallback"
        assert sg["ll_keywords"] == sg["hl_keywords"] == ["Q7X42 sidecar gate"]
        assert sg["keyword_error"]
        assert sg["status"] == "ok"  # still queried with the fallback keywords


def test_session_graph_empty_is_reported_not_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)
    _patch_sg_rag(mod, monkeypatch, _SessionGraphRag(context="[no-context]"))
    off = _call_recall(mod, monkeypatch, session_id=SESSION_ID)
    on = _call_recall(mod, monkeypatch, session_id=SESSION_ID, session_graph=True)
    sg = on["session_graph"]
    assert sg["empty"] is True and sg["status"] == "empty" and "error" not in sg
    assert on["context"] == off["context"]


def test_session_graph_error_is_explicit_and_chunks_unaffected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)
    off = _call_recall(mod, monkeypatch, session_id=SESSION_ID)
    _patch_sg_rag(mod, monkeypatch, _SessionGraphRag(raises=RuntimeError("graph exploded")))
    on = _call_recall(mod, monkeypatch, session_id=SESSION_ID, session_graph=True)
    assert on["ok"] is True
    assert on["session_graph"]["status"] == "error"
    assert "graph exploded" in on["session_graph"]["error"]
    assert on["session_graph"]["keyword_source"] == "caller"
    assert on["context"] == off["context"]

    async def build_fails(_sid: str, _dir: Path) -> Any:
        raise OSError("cannot load storages")

    monkeypatch.setattr(mod, "_get_session_graph_rag", build_fails)
    on = _call_recall(mod, monkeypatch, session_id=SESSION_ID, session_graph=True)
    assert on["session_graph"]["status"] == "error" and "cannot load" in on["session_graph"]["error"]
    assert on["context"] == off["context"]


def test_session_graph_skips_merged_unknown_subgraph_and_no_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path, merged=True)
    mod = _recall_mod(tmp_path)
    sg_rag = _SessionGraphRag()
    _patch_sg_rag(mod, monkeypatch, sg_rag)
    assert _call_recall(mod, monkeypatch, session_id=SESSION_ID, session_graph=True)[
        "session_graph"]["status"] == "skipped"
    unk = _call_recall(mod, monkeypatch, session_id="nope", session_graph=True)
    assert unk["session_graph"]["status"] == "unknown_session"
    assert _call_recall(mod, monkeypatch, session_graph=True)["session_graph"]["status"] == "skipped"
    sub = _recall_mod(tmp_path, HARS_MEMORY_SESSION_MODE="subgraph")
    _patch_sg_rag(sub, monkeypatch, sg_rag)
    assert _call_recall(sub, monkeypatch, session_id=SESSION_ID, session_graph=True)[
        "session_graph"]["status"] == "skipped"
    assert sg_rag.queries == []


def test_session_graph_lean_view_keeps_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)
    _patch_sg_rag(mod, monkeypatch, _SessionGraphRag())
    lean = _call_recall(mod, monkeypatch, session_id=SESSION_ID, session_graph=True, view="lean")
    assert lean["session_graph"]["status"] == "ok"


def test_splice_session_kg_merges_into_existing_main_sections(tmp_path: Path) -> None:
    mod = _recall_mod(tmp_path)
    main = (
        "\nKnowledge Graph Data (Entity):\n\n```json\n{\"entity\": \"MAIN\"}\n```\n\n"
        "Knowledge Graph Data (Relationship):\n\n```json\n{\"entity1\": \"MAIN\"}\n```\n\n"
        "Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):"
        "\n\n```json\n{\"c\": 1}\n```\n"
    )
    out = mod._splice_session_kg(main, ['{"entity": "SESS"}'], ['{"entity1": "SESS"}'])
    assert out.index('"SESS"') < out.index('"MAIN"')  # session first inside the same section
    assert out.count("Knowledge Graph Data (Entity)") == 1
    assert _chunk_section(out) == _chunk_section(main)
    only_kg = mod._splice_session_kg(None, ['{"entity": "S"}'], [])
    assert "Knowledge Graph Data (Entity)" in only_kg and "Document Chunks" not in only_kg
    assert mod._splice_session_kg(main, [], []) == main


def test_session_graph_rag_uses_file_backed_storages_even_with_qdrant_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = _write_session_fixture(tmp_path)
    mod = _recall_mod(
        tmp_path, HARS_MEMORY_VECTOR_STORAGE="QdrantVectorDBStorage", HARS_MEMORY_GRAPH_STORAGE="Neo4JStorage"
    )
    captured: dict[str, Any] = {}

    class _Rag:
        async def initialize_storages(self) -> None:
            captured["init"] = True

    def fake_create(**kwargs: Any) -> _Rag:
        captured.update(kwargs)
        return _Rag()

    import hars_memory.server.lightrag_init as li

    monkeypatch.setattr(li, "create_lightrag", fake_create)
    mod._session_graph_rags.clear()
    rag = asyncio.run(mod._get_session_graph_rag(SESSION_ID, session_dir))
    assert captured["vector_storage"] == "NanoVectorDBStorage"
    assert captured["graph_storage"] == "NetworkXStorage"
    assert captured["working_dir"] == str(session_dir) and captured["init"]
    # cached while the files are unchanged, rebuilt when a compaction rewrites them
    assert asyncio.run(mod._get_session_graph_rag(SESSION_ID, session_dir)) is rag
    (session_dir / "vdb_entities.json").write_text("{}", encoding="utf-8")
    assert asyncio.run(mod._get_session_graph_rag(SESSION_ID, session_dir)) is not rag


def test_session_graph_with_main_branch_skipped_runs_only_session_aquery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_session_fixture(tmp_path)
    mod = _recall_mod(tmp_path)
    sg_rag = _SessionGraphRag()
    _patch_sg_rag(mod, monkeypatch, sg_rag)
    main_calls: list[Any] = []

    class _CountingRag(_RecallRag):
        async def aquery(self, question: str, *, param: object) -> str:  # noqa: ARG002
            main_calls.append(question)
            return "[no-context]"

    hybrid = _hybrid_fixture()

    async def fake_hybrid(*_a: Any, **_k: Any) -> dict[str, Any]:
        return copy.deepcopy(hybrid)

    monkeypatch.setattr(mod, "_compute_hybrid_block", fake_hybrid)
    mod._rag_instance = _CountingRag([], embedding_func=_embed_query)
    out = asyncio.run(mod.call_tool("memory_recall", {
        "question": "Q7X42 sidecar gate", "context_only": True,
        "session_id": SESSION_ID, "session_graph": True,
    }))
    data = json.loads(out[0].text)
    assert data["graph_branch"]["enabled"] is False  # no caller keywords -> main graph skipped
    assert main_calls == []
    assert len(sg_rag.queries) == 1
    assert data["session_graph"]["keyword_source"] == "llm"
    assert "Knowledge Graph Data (Entity)" in data["context"]
