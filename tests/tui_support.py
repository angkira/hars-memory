"""Fake HARS backend for the TUI tests: an ``httpx.MockTransport`` handler, no sockets.

Serves the HARS HTTP server (``/healthz``, ``/mcp`` JSON-RPC), the trace SSE listener
(``/healthz``, ``/trace``) and two model endpoints, with per-test knobs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from hars_memory.trace import format_sse
from hars_memory.tui.config import TuiConfig

BASE = "http://hars.test:8099"
TRACE = "http://hars.test:8098"
EMBEDDER = "http://emb.test/v1"
EXTRACTOR = "http://llm.test/v1"

TOOL_NAMES = ["memory_recall", "memory_status", "memory_session_status", "memory_list_projects",
              "memory_remember", "memory_compact", "memory_merge_session"]


def make_config(**overrides: Any) -> TuiConfig:
    trace_url = overrides.pop("trace_url", TRACE)  # `with_overrides` skips None, but None means "listener disabled"
    cfg = TuiConfig(base_url=BASE, mcp_url=BASE + "/mcp", trace_url=trace_url, refresh_s=3600.0)
    return cfg.with_overrides(**overrides)


STATUS = {
    "ok": True, "project": "default", "project_name": "Default Project", "index_exists": True,
    "node_count": 120, "edge_count": 340, "last_ingest": "2026-10-26T10:00:00",
    "working_dir": "/srv/index",
    "storage": {"vector": {"backend": "NanoVectorDBStorage", "file_count": 3}, "graph": {"backend": "NetworkXStorage"}},
    "configured_models": {
        "embedder": f"embeddinggemma @ {EMBEDDER} (http)",
        "extractor": f"extractor-m @ {EXTRACTOR}",
        "query": f"query-m @ {EXTRACTOR}",
    },
    "staging_backlog": {"pending_notes": 2, "oldest_note": "x.md", "oldest_note_age_days": 1, "staging_dir": "/srv/staging"},
}

HEALTH = {"status": "ok", "mode": "combined", "uptime_seconds": 12.5, "index_warm": True, "oauth": False,
          "engine_router": None}

RECALL_RESULT = {
    "ok": True, "question": "retry limit", "mode": "hybrid", "lightrag_mode": "mix", "top_k": 20, "fetch_top_k": 40,
    "fetch_top_k_source": "multiplier", "context_priority_applied": "merged", "stale_days": None, "last_ingest": None,
    "context": (
        "Knowledge Graph Data (Entity):\n\n```json\n"
        '{"entity": "Payments Queue", "type": "artifact", "description": "retry limit 7"}\n```\n\n'
        "Knowledge Graph Data (Relationship):\n\n```json\n"
        '{"entity1": "Maria", "entity2": "Payments Queue", "description": "Maria set the limit"}\n```\n\n'
        "Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):\n\n```json\n"
        '{"reference_id": "1", "chunk_id": "chunk-a", "content": "The retry limit is 7 attempts."}\n```\n\n'
        "Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):\n\n"
        "```\n[1] docs/queues.md\n```\n"
    ),
    "hybrid": {
        "enabled": True,
        "rerank": {"backend": "off", "applied": False, "fallback_reason": "disabled_by_default", "latency_ms": None, "pool": 0},
        "latency_ms": {"total": 41.0},
        "fused_chunks": [
            {"chunk_id": "chunk-a", "fused_score": 0.91, "dense_score": 0.8, "sparse_score": 3.2,
             "file_path": "docs/queues.md", "start_line": 12, "heading_path": "Queues > Retry", "content": "The retry limit is 7 attempts."},
            {"chunk_id": "chunk-b", "fused_score": 0.42, "dense_score": 0.4, "sparse_score": 1.1,
             "file_path": "docs/other.md", "start_line": 3, "content": "unrelated"},
        ],
    },
}

TRACE_EVENTS: list[tuple[str, dict[str, Any]]] = [
    ("trace_start", {"seq": 0, "t_ms": 1.0, "mode": "hybrid", "lightrag_mode": "mix", "top_k": 20, "ll_keywords": ["a"], "hl_keywords": []}),
    ("bm25_hits", {"seq": 1, "t_ms": 5.0, "latency_ms": 4.0, "pool_size": 40, "hits": [{"chunk_id": "chunk-a"}]}),
    ("vector_hits", {"seq": 2, "t_ms": 9.0, "latency_ms": 8.0, "pool_size": 40, "embedder": "gemma", "hits": [{"chunk_id": "chunk-a"}, {"chunk_id": "chunk-b"}]}),
    ("fusion", {"seq": 3, "t_ms": 10.0, "alpha": 0.15, "pool_size": 2, "candidate_pool_size": 40}),
    ("rerank", {"seq": 4, "t_ms": 10.5, "applied": False, "backend": "off", "fallback_reason": "disabled_by_default"}),
    ("graph_expand", {"seq": 5, "t_ms": 20.0, "latency_ms": 9.0, "skipped": False, "mode": "mix", "entity_count": 1, "relation_count": 1}),
    ("context", {"seq": 6, "t_ms": 21.0, "chars": 400, "source_count": 1, "entity_count": 1, "relation_count": 1}),
    ("result", RECALL_RESULT),
    ("done", {"trace_id": "t1", "ok": True, "total_ms": 22.0, "has_context": True}),
]


def _tool_text(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}


class FakeHars:
    """Configurable fake. ``calls`` records every request as ``(method, url, json-body-or-None)``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.headers_seen: list[tuple[str, dict[str, str]]] = []
        self.health: dict[str, Any] | None = dict(HEALTH)
        self.trace_up = True
        self.trace_events = list(TRACE_EVENTS)
        self.trace_truncate_after: int | None = None
        self.mcp_up = True
        self.embedder_up = True
        self.extractor_up = False
        self.tools: dict[str, Any] = {
            "memory_status": dict(STATUS),
            "memory_recall": dict(RECALL_RESULT),
            "memory_list_projects": {"ok": True, "projects": [
                {"project_id": "default", "name": "Default Project", "visibility": "public", "description": "shared KB",
                 "shared_departments": [], "owner_user_id": "system", "actions": ["read", "write", "admin"]},
                {"project_id": "lab", "name": "Lab", "visibility": "private", "description": "", "shared_departments": ["r&d"],
                 "owner_user_id": "u1", "actions": ["read"]},
            ]},
            "memory_session_status": {"ok": True, "mode": "split", "sessions_dir": "/nonexistent/sessions", "sessions": [
                {"session_id": "sg1", "mode": "split", "exists": True, "compactions": 2, "merged": False, "merged_at": None,
                 "entities": 3, "relations": 2, "chunks_indexed": 2, "vectors_indexed": 2},
                {"session_id": "other-9", "mode": "split", "exists": True, "compactions": 1, "merged": True, "merged_at": "x",
                 "entities": 1, "relations": 0, "chunks_indexed": 1, "vectors_indexed": 1},
            ]},
            "memory_merge_session": {"ok": True, "merged": True},
            "memory_remember": {"ok": True, "saved": "/srv/staging/n.md", "pending_notes": 3},
            "memory_compact": {"ok": True, "seq": 3},
        }
        self.tool_errors: dict[str, str] = {}
        self.on_tool: Callable[[str, dict[str, Any]], None] | None = None

    # -- helpers ------------------------------------------------------------

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def tool_calls(self, name: str | None = None) -> list[tuple[str, dict[str, Any]]]:
        out = []
        for method, url, body in self.calls:
            if url.endswith("/mcp") and isinstance(body, dict) and body.get("method") == "tools/call":
                tool = body["params"]["name"]
                if name is None or tool == name:
                    out.append((tool, body["params"]["arguments"]))
        return out

    # -- transport ------------------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body: Any = None
        if request.content:
            try:
                body = json.loads(request.content)
            except ValueError:
                body = None
        self.calls.append((request.method, url, body))
        self.headers_seen.append((url, dict(request.headers)))
        base = url.split("?")[0]

        if base == BASE + "/healthz":
            if self.health is None:
                raise httpx.ConnectError("connection refused")
            return httpx.Response(200, json=self.health)
        if base == BASE + "/mcp":
            if not self.mcp_up:
                raise httpx.ConnectError("connection refused")
            return self._mcp(body)
        if base == TRACE + "/healthz":
            if not self.trace_up:
                raise httpx.ConnectError("connection refused")
            return httpx.Response(200, json={"ok": True, "service": "fake", "trace": True})
        if base == TRACE + "/trace":
            if not self.trace_up:
                raise httpx.ConnectError("connection refused")
            events = self.trace_events if self.trace_truncate_after is None else self.trace_events[: self.trace_truncate_after]
            payload = b": ping\n\n" + b"".join(format_sse(e, p) for e, p in events)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload)
        if base == EMBEDDER + "/models":
            if not self.embedder_up:
                raise httpx.ConnectError("refused")
            return httpx.Response(404)
        if base == EXTRACTOR + "/models":
            if not self.extractor_up:
                raise httpx.ConnectError("refused")
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404, json={"error": f"no fake route for {base}"})

    def _mcp(self, body: Any) -> httpx.Response:
        method = (body or {}).get("method")
        if method == "tools/list":
            result: Any = {"tools": [{"name": n} for n in TOOL_NAMES]}
        elif method == "tools/call":
            name = body["params"]["name"]
            args = body["params"]["arguments"]
            if self.on_tool:
                self.on_tool(name, args)
            if name in self.tool_errors:
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {
                    "content": [{"type": "text", "text": json.dumps({"ok": False, "error": self.tool_errors[name]})}], "isError": False}})
            if name not in self.tools:
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": f"unknown tool {name}"}})
            payload = self.tools[name]
            if callable(payload):
                payload = payload(args)
            result = _tool_text(payload)
        else:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "method not found"}})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": result})


def write_session(root, sid: str = "sg1") -> None:
    """Create a minimal session working dir (full docs + graphml) like SessionMemoryStore writes."""
    d = root / sid
    d.mkdir(parents=True)
    docs = {
        f"session:{sid}:c1": {"content": f"[Document: session-{sid}-c1 | Section: session | Date: 2026-10-26]\n\nSession: {sid}\nSeq: 1\n"
                              "Tags: session, compaction\n\n## Summary\nBilling migration notes.\n\n## Dropped context facts\n"
                              "- The retry limit is 7.\n- Backoff starts at 250 ms.\n"},
        f"session:{sid}:c2": {"content": f"[Document: session-{sid}-c2 | Section: session | Date: 2026-10-26]\n\nSession: {sid}\nSeq: 2\n"
                              "Tags: session, compaction\n\n## Summary\nSecond pass.\n\n## Dropped context facts\n- Maria approved.\n"},
    }
    (d / "kv_store_full_docs.json").write_text(json.dumps(docs))
    (d / "graph_chunk_entity_relation.graphml").write_text(GRAPHML)


GRAPHML = """<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d8" for="edge" attr.name="description" attr.type="string" />
  <key id="d7" for="edge" attr.name="weight" attr.type="double" />
  <key id="d9" for="edge" attr.name="keywords" attr.type="string" />
  <key id="d2" for="node" attr.name="description" attr.type="string" />
  <key id="d1" for="node" attr.name="entity_type" attr.type="string" />
  <graph edgedefault="undirected">
    <node id="Payments Queue"><data key="d1">artifact</data><data key="d2">queue with retry limit 7</data></node>
    <node id="Maria"><data key="d1">person</data><data key="d2">engineer</data></node>
    <node id="Exponential Backoff"><data key="d1">concept</data><data key="d2">starts at 250 ms</data></node>
    <edge source="Maria" target="Payments Queue"><data key="d7">1.0</data><data key="d8">Maria set the retry limit</data><data key="d9">owner</data></edge>
    <edge source="Payments Queue" target="Exponential Backoff"><data key="d7">2.0</data><data key="d8">uses backoff</data></edge>
  </graph>
</graphml>
"""
