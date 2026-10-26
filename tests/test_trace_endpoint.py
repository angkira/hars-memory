"""Live recall trace (SSE) — emitter ordering, non-trace isolation, HTTP E2E.

Hermetic tests use synthetic dense hits + a tiny synthetic BM25 index and a fake
LightRAG object; nothing here touches the real index, network daemons or GPU.
The live test (last class) is skipped unless `HARS_MEMORY_TRACE_URL` points at
the running service's trace endpoint.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

import hars_memory.trace as trace

EXPECTED_STAGES = [
    "trace_start",
    "bm25_hits",
    "vector_hits",
    "fusion",
    "rerank",
    "graph_expand",
    "context",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_mcp(env: dict[str, str] | None = None):
    """Import (reload) hars_memory.mcp_server with `env` applied transiently."""
    env = env or {}
    mod = importlib.import_module("hars_memory.mcp_server")
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        importlib.reload(mod)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return mod


def _parse_sse(lines: list[str]) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    event: str | None = None
    data: list[str] = []
    for line in lines:
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif line == "":
            if event is not None and data:
                events.append((event, json.loads("\n".join(data))))
            event, data = None, []
    return events


async def _read_sse(client: httpx.AsyncClient, url: str, **kwargs: Any) -> list[tuple[str, dict[str, Any]]]:
    """GET `url` and read SSE frames until the terminal `done`/`error`."""
    lines: list[str] = []
    event: str | None = None
    async with client.stream("GET", url, **kwargs) as response:
        assert response.status_code == 200, await response.aread()
        async for line in response.aiter_lines():
            lines.append(line)
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line == "":
                if event in ("done", "error"):
                    break
                event = None
    return _parse_sse(lines)


def _write_chunks(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks = {
        "chunk-1": {
            "content": "[Document: hyp_q7x42.md | Section: gate | Date: 2026-05-01]\n\nQ7X42 sidecar gate.",
            "file_path": "hyp_q7x42.md",
        },
        "chunk-2": {
            "content": "phase_x1_adapter_safe ran 3000 steps.",
            "file_path": "exp_phase_c1.md",
        },
        "chunk-3": {
            "content": "vma_native encodes video tokens.",
            "file_path": "feedback_vma_native.md",
        },
    }
    (index_dir / "kv_store_text_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")


def _sample_context() -> str:
    entity = {"entity": "Q7X42", "type": "experiment", "description": "sidecar gate"}
    relation = {"entity1": "Q7X42", "entity2": "Sidecar Gate", "description": "uses"}
    chunk = {"reference_id": "1", "content": "[Document: hyp_q7x42.md | Section: gate]\n\nbody", "section": "gate"}
    return (
        "\nKnowledge Graph Data (Entity):\n\n```json\n"
        + json.dumps(entity)
        + "\n```\n\nKnowledge Graph Data (Relationship):\n\n```json\n"
        + json.dumps(relation)
        + "\n```\n\nDocument Chunks (Each entry has a reference_id):\n\n```json\n"
        + json.dumps(chunk)
        + "\n```\n\nReference Document List (Each entry starts with a [reference_id]):\n\n```\n[1] hyp_q7x42.md\n```\n"
    )


class _FakeChunksVdb:
    def __init__(self, hits: list[dict], calls: list[str] | None = None) -> None:
        self._hits = hits
        self._calls = calls

    async def query(self, query: str, top_k: int) -> list[dict]:
        if self._calls is not None:
            self._calls.append("chunks_vdb.query")
        return self._hits[:top_k]


class _TraceRag:
    """Fake LightRAG exposing exactly what memory_recall touches.

    `calls` records the order of the two passes memory_recall performs — the
    dense chunk query that feeds `fusion` and the LightRAG `aquery` that IS the
    graph pass — so a test can assert their real dependency order instead of
    trusting the emitted frame order.
    """

    def __init__(self, dense_hits: list[dict], context: str = "[no-context]") -> None:
        self.calls: list[str] = []
        self.chunks_vdb = _FakeChunksVdb(dense_hits, self.calls)
        self._context = context

    async def aquery(self, question: str, param: object) -> str:
        self.calls.append("aquery")
        return self._context


def _real_pipeline_env(tmp_path: Path) -> dict[str, str]:
    return {
        "HARS_MEMORY_INDEX_DIR": str(tmp_path),
        "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        "HARS_MEMORY_RIPGREP_CHANNEL": "0",
        "HARS_MEMORY_FLAT_CHANNEL": "0",
        "HARS_MEMORY_HYBRID_ENABLED": "1",
        "HARS_MEMORY_TRACE_HTTP_PORT": "0",
    }


def _dense_hits() -> list[dict]:
    return [
        {"id": "chunk-1", "distance": 0.91, "content": "[Document: hyp_q7x42.md | Section: gate]\n\nQ7X42 gate.", "file_path": "hyp_q7x42.md"},
        {"id": "chunk-2", "distance": 0.55, "content": "phase_x1_adapter_safe ran.", "file_path": "exp_phase_c1.md"},
        {"id": "chunk-3", "distance": 0.22, "content": "vma_native.", "file_path": "feedback_vma_native.md"},
    ]


# ---------------------------------------------------------------------------
# parsing / framing
# ---------------------------------------------------------------------------

class TestFramingAndParsing:
    def test_format_sse_is_one_event_frame(self) -> None:
        frame = trace.format_sse("done", {"ok": True, "n": 3})
        assert frame.startswith(b"event: done\n")
        assert frame.endswith(b"\n\n")
        assert json.loads(frame.decode().split("data: ", 1)[1]) == {"ok": True, "n": 3}

    def test_parse_context_blocks_classifies_lines(self) -> None:
        blocks = trace.parse_context_blocks(_sample_context())
        assert [e["entity"] for e in blocks["entities"]] == ["Q7X42"]
        assert blocks["relations"][0]["entity2"] == "Sidecar Gate"
        assert [s["reference_id"] for s in blocks["sources"]] == ["1"]
        assert blocks["references"] == {"1": "hyp_q7x42.md"}

    def test_hit_doc_fields_prefers_document_header(self) -> None:
        fields = trace.hit_doc_fields("[Document: d.md | Section: s | Date: 2026-01-01]\n\nbody", "x.md")
        assert fields == {"doc": "d.md", "section": "s"}

    def test_hit_doc_fields_falls_back_to_basename(self) -> None:
        assert trace.hit_doc_fields("plain body", "a/b/c.md") == {"doc": "c.md", "section": ""}


# ---------------------------------------------------------------------------
# emitter: ordering, isolation, error path
# ---------------------------------------------------------------------------

class TestEmitter:
    async def test_emit_is_noop_without_a_trace(self) -> None:
        assert trace._trace_emitter.get() is None
        assert await trace.emit("vector_hits", hits=[]) is None

    async def test_emitter_is_context_scoped(self) -> None:
        """An emitter installed inside one task must not leak into another —
        this is what keeps the normal MCP path free of trace side effects."""
        collected: list[str] = []

        async def emitter(event: str, payload: dict) -> None:
            collected.append(event)

        async def traced() -> None:
            token = trace._trace_emitter.set(emitter)
            try:
                await trace.emit("vector_hits", hits=[])
            finally:
                trace._trace_emitter.reset(token)

        await traced()
        await trace.emit("vector_hits", hits=[])  # no emitter in THIS context
        assert collected == ["vector_hits"]

    async def test_fake_recall_event_order_is_one_per_stage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_mcp({"HARS_MEMORY_TRACE_HTTP_PORT": "0"})

        async def fake_call_tool(name: str, arguments: dict) -> list:
            assert name == "memory_recall"
            await trace.emit("trace_start", question=arguments["question"], mode="hybrid")
            await trace.emit("bm25_hits", latency_ms=1.5, hits=[{"chunk_id": "c-s"}])
            await trace.emit("vector_hits", latency_ms=2.5, hits=[{"chunk_id": "c-d"}])
            await trace.emit("fusion", alpha=0.5, order=["c-d", "c-s"])
            await trace.emit("rerank", applied=False, fallback_reason="fusion_confident", latency_ms=None)
            await trace.emit("graph_expand", latency_ms=10.0, entities=[{"name": "A", "type": "t"}])
            await trace.emit("context", chars=42, has_context=True)
            return [mod.TextContent(type="text", text=json.dumps({"ok": True, "context": "ctx", "hybrid": {"enabled": True}}))]

        monkeypatch.setattr(mod, "call_tool", fake_call_tool)
        events = [event async for event in trace.stream_recall_events({"question": "q"})]
        names = [name for name, _ in events]
        assert names == EXPECTED_STAGES + ["result", "done"]
        stage_events = [(name, payload) for name, payload in events if name in EXPECTED_STAGES]
        ids = {payload["trace_id"] for _, payload in stage_events}
        ids.add(dict(events)["done"]["trace_id"])
        assert len(ids) == 1
        assert [payload["seq"] for _, payload in stage_events] == list(range(len(stage_events)))
        t_ms = [payload["t_ms"] for _, payload in stage_events]
        assert t_ms == sorted(t_ms)
        # `result` is the raw, unchanged memory_recall payload — no trace fields
        assert "hybrid" in dict(events)["result"]

    async def test_error_path_emits_terminal_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_mcp({"HARS_MEMORY_TRACE_HTTP_PORT": "0"})

        async def boom(name: str, arguments: dict) -> list:
            raise RuntimeError("kaboom")

        monkeypatch.setattr(mod, "call_tool", boom)
        events = [event async for event in trace.stream_recall_events({"question": "q"})]
        assert [name for name, _ in events] == ["error"]
        payload = events[0][1]
        assert payload["ok"] is False and payload["error"] == "kaboom"
        assert payload["error_type"] == "RuntimeError" and payload["total_ms"] >= 0

    async def test_soft_failure_dict_becomes_error_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """memory_recall catches its own failures and returns a bare
        {ok:false,error...} (no `hybrid`); the trace must not fake a `result`."""
        mod = _load_mcp({"HARS_MEMORY_TRACE_HTTP_PORT": "0"})

        async def softfail(name: str, arguments: dict) -> list:
            body = {"ok": False, "error_type": "TimeoutError", "error": "no context timer", "hint": "retry"}
            return [mod.TextContent(type="text", text=json.dumps(body))]

        monkeypatch.setattr(mod, "call_tool", softfail)
        events = [event async for event in trace.stream_recall_events({"question": "q"})]
        assert [name for name, _ in events] == ["error"]
        assert events[0][1]["error"] == "no context timer"


# ---------------------------------------------------------------------------
# real pipeline wiring (fake dense/BM25, real _compute_hybrid_block + recall)
# ---------------------------------------------------------------------------

class TestRealPipelineTrace:
    def _install(self, tmp_path: Path, mod):
        _write_chunks(tmp_path)
        mod._rag_instance = _TraceRag(_dense_hits(), _sample_context())

    async def test_all_stages_with_measured_timings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "0")
        monkeypatch.setenv("HARS_MEMORY_FLAT_CHANNEL", "0")
        mod = _load_mcp(_real_pipeline_env(tmp_path))
        self._install(tmp_path, mod)
        events = [event async for event in trace.stream_recall_events({
            "question": "Q7X42 sidecar gate", "ll_keywords": ["Q7X42"], "top_k": 3,
        })]
        by_name = {name: payload for name, payload in events}
        names = [name for name, _ in events]
        for stage in EXPECTED_STAGES:
            assert stage in names, f"missing {stage} in {names}"
        # order: sparse measured before dense; graph/context after rerank
        assert names.index("bm25_hits") < names.index("vector_hits") < names.index("fusion") < names.index("rerank")
        assert names.index("rerank") < names.index("graph_expand") < names.index("context")
        assert names[-2:] == ["result", "done"]
        # measured values present and plausible
        assert by_name["bm25_hits"]["latency_ms"] >= 0
        assert by_name["vector_hits"]["latency_ms"] >= 0
        assert by_name["vector_hits"]["embedder"]
        assert any(h["chunk_id"] == "chunk-1" for h in by_name["vector_hits"]["hits"])
        assert by_name["fusion"]["alpha"] == mod.HARS_MEMORY_HYBRID_ALPHA
        assert by_name["fusion"]["order"]  # true pre-rerank order, measured
        assert by_name["rerank"]["before"] == by_name["fusion"]["order"]
        assert by_name["graph_expand"]["entity_count"] == 1
        assert by_name["context"]["chars"] > 0
        assert by_name["done"]["total_ms"] >= 0
        # the final result is the untouched memory_recall response
        assert by_name["result"]["ok"] is True
        assert by_name["result"]["hybrid"]["enabled"] is True

    async def test_fusion_is_independent_of_the_later_graph_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dependency guard (verdict b): `fusion` is a dense+sparse blend emitted
        by `_compute_hybrid_block`, which completes BEFORE LightRAG's graph pass
        (`aquery`) runs. The later `graph_expand` frame only PARSES the rendered
        context string; it feeds nothing back into fusion.

        If a future change makes fusion consume graph output, the recorded call
        order (`chunks_vdb.query` -> `aquery`) and/or the graph-free `fusion`
        payload below will fail, which is exactly the intent: this ordering must
        never be "fixed" by moving the graph pass first to match a label.
        """
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "0")
        monkeypatch.setenv("HARS_MEMORY_FLAT_CHANNEL", "0")
        mod = _load_mcp(_real_pipeline_env(tmp_path))
        _write_chunks(tmp_path)
        rag = _TraceRag(_dense_hits(), _sample_context())
        mod._rag_instance = rag
        events = [
            event
            async for event in trace.stream_recall_events({
                "question": "Q7X42 sidecar gate", "ll_keywords": ["Q7X42"], "top_k": 3,
            })
        ]
        by_name = dict(events)
        names = [name for name, _ in events]
        # The graph pass is invoked strictly after the dense channel it does not
        # feed; there is no read of graph state before fusion.
        assert rag.calls == ["chunks_vdb.query", "aquery"]
        assert names.index("fusion") < names.index("graph_expand")
        # The fusion frame is a pure channel blend: no entity/relation/edge field
        # (i.e. no graph-derived score) is among its inputs.
        graph_keys = {
            "entities", "relations", "entity_count", "relation_count",
            "traversed_edge_count", "graph", "graph_scores",
        }
        assert set(by_name["fusion"]) & graph_keys == set()
        assert {"alpha", "candidate_pool_size", "pool_size", "order", "scores"} <= set(
            by_name["fusion"]
        )
        # graph_expand is derived from the rendered context, and reports that.
        assert by_name["graph_expand"]["derived_from"] == "context_string"

    async def test_non_trace_path_is_not_decorated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same recall with no emitter installed must be byte-identical to
        the traced one apart from volatile timings/cache status — no trace
        fields, no extra keys."""
        # ripgrep/flat re-read env at call time, so pin them via monkeypatch
        # (which persists through the test body) to keep this deterministic.
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "0")
        monkeypatch.setenv("HARS_MEMORY_FLAT_CHANNEL", "0")
        mod = _load_mcp(_real_pipeline_env(tmp_path))
        self._install(tmp_path, mod)
        args = {"question": "Q7X42 sidecar gate", "ll_keywords": ["Q7X42"], "top_k": 3}

        traced = [
            event async for event in trace.stream_recall_events(dict(args))
        ]
        assert trace._trace_emitter.get() is None  # reset after the run
        plain = await mod.call_tool("memory_recall", dict(args))
        plain_resp = json.loads(plain[0].text)
        traced_resp = dict(traced)[ "result"]

        def strip(resp: dict) -> dict:
            out = json.loads(json.dumps(resp))
            hybrid = out.get("hybrid", {})
            hybrid.pop("latency_ms", None)
            hybrid.get("rerank", {}).pop("latency_ms", None)
            hybrid.get("bm25_index", {}).pop("cache_status", None)
            hybrid.get("ripgrep", {}).pop("latency_ms", None)
            hybrid.get("flat_dense", {}).pop("latency_ms", None)
            out.pop("debug", None)
            return out

        assert strip(traced_resp) == strip(plain_resp)
        assert not any(key.startswith("_trace") for key in plain_resp)

    async def test_ripgrep_hits_frame_reports_the_ranked_channel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`ripgrep_hits` mirrors `bm25_hits`: measured latency, matched chunk
        ids and scores. It is a first-class frame, not a footnote in `result`."""
        from hars_memory.retrieval import ripgrep_channel
        from hars_memory.retrieval.ripgrep_channel import RipgrepSearchHit, RipgrepSearchResult

        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "1")
        monkeypatch.setenv("HARS_MEMORY_FLAT_CHANNEL", "0")
        mod = _load_mcp(_real_pipeline_env(tmp_path))
        self._install(tmp_path, mod)

        async def fake_asearch(question, **_kwargs):  # noqa: ANN001
            return RipgrepSearchResult(
                hits=[
                    RipgrepSearchHit(
                        chunk_id="file-stable-q7x42",
                        score=3.5,
                        content="Q7X42 sidecar gate",
                        file_path="/corpus/hyp_q7x42.md",
                        match_count=1,
                        term_hits={"Q7X42": 1},
                    )
                ],
                query_terms=("Q7X42",),
                available=True,
                unavailable_reason=None,
                latency_seconds=0.004,
                timed_out=False,
            )

        monkeypatch.setattr(ripgrep_channel, "asearch", fake_asearch)
        events = [
            event
            async for event in trace.stream_recall_events({
                "question": "Q7X42 sidecar gate", "ll_keywords": ["Q7X42"], "top_k": 3,
            })
        ]
        by_name = dict(events)
        names = [name for name, _ in events]
        assert "ripgrep_hits" in names
        # channel frame precedes the fusion it feeds, like bm25/vector
        assert names.index("vector_hits") < names.index("ripgrep_hits") < names.index("fusion")
        frame = by_name["ripgrep_hits"]
        assert frame["enabled"] is True and frame["skipped"] is False
        assert frame["available"] is True and frame["mode"] == "ranked"
        assert frame["latency_ms"] == 4.0
        assert [h["chunk_id"] for h in frame["hits"]] == ["file-stable-q7x42"]
        assert frame["hits"][0]["ripgrep_score"] == 3.5
        assert frame["hits"][0]["matched_terms"] == {"Q7X42": 1}
        # the pool chunk whose (flattened) file_path matches the basename
        assert frame["matched_chunk_ids"] == ["chunk-1"]

    async def test_ripgrep_hits_frame_when_channel_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "0")
        monkeypatch.setenv("HARS_MEMORY_FLAT_CHANNEL", "0")
        mod = _load_mcp(_real_pipeline_env(tmp_path))
        self._install(tmp_path, mod)
        events = [
            event
            async for event in trace.stream_recall_events({
                "question": "Q7X42 sidecar gate", "ll_keywords": ["Q7X42"], "top_k": 3,
            })
        ]
        frame = dict(events)["ripgrep_hits"]
        assert frame["enabled"] is False
        assert frame["skipped"] is True
        assert "HARS_MEMORY_RIPGREP_CHANNEL" in frame["reason"]
        assert frame["latency_ms"] is None
        assert frame["hits"] == [] and frame["matched_chunk_ids"] == []


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------

class TestTraceHTTP:
    async def test_healthz(self) -> None:
        server = await trace.start_trace_server("127.0.0.1", 0, keepalive_s=1.0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://127.0.0.1:{port}/healthz")
            assert response.status_code == 200 and response.json()["ok"] is True
        finally:
            server.close()
            await server.wait_closed()

    async def test_missing_question_is_400(self) -> None:
        server = await trace.start_trace_server("127.0.0.1", 0, keepalive_s=1.0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://127.0.0.1:{port}/trace")
            assert response.status_code == 400 and "question" in response.json()["error"]
        finally:
            server.close()
            await server.wait_closed()

    async def test_sse_stream_orders_stages_and_terminates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _load_mcp({"HARS_MEMORY_TRACE_HTTP_PORT": "0"})

        async def fake_call_tool(name: str, arguments: dict) -> list:
            await trace.emit("trace_start", question=arguments["question"])
            await trace.emit("bm25_hits", latency_ms=1.0, hits=[{"chunk_id": "s"}])
            await trace.emit("vector_hits", latency_ms=2.0, hits=[{"chunk_id": "d"}])
            await trace.emit("fusion", alpha=0.5, order=["d", "s"])
            await trace.emit("rerank", applied=True, latency_ms=7.0)
            await trace.emit("graph_expand", latency_ms=5.0)
            await trace.emit("context", chars=10, has_context=True)
            return [mod.TextContent(type="text", text=json.dumps({"ok": True, "context": "c", "hybrid": {"enabled": True}}))]

        monkeypatch.setattr(mod, "call_tool", fake_call_tool)
        server = await trace.start_trace_server("127.0.0.1", 0, keepalive_s=1.0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                events = await _read_sse(
                    client,
                    f"http://127.0.0.1:{port}/trace",
                    params={"question": "hello", "ll_keywords": "a,b", "top_k": "3"},
                )
        finally:
            server.close()
            await server.wait_closed()
        names = [name for name, _ in events]
        assert names == EXPECTED_STAGES + ["result", "done"]
        assert events[-1][1]["ok"] is True

    async def test_format_json_collects_events(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_mcp({"HARS_MEMORY_TRACE_HTTP_PORT": "0"})

        async def fake_call_tool(name: str, arguments: dict) -> list:
            await trace.emit("trace_start", question=arguments["question"])
            return [mod.TextContent(type="text", text=json.dumps({"ok": True, "context": None, "hybrid": {"enabled": True}}))]

        monkeypatch.setattr(mod, "call_tool", fake_call_tool)
        server = await trace.start_trace_server("127.0.0.1", 0, keepalive_s=1.0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://127.0.0.1:{port}/trace", params={"question": "q", "format": "json"}
                )
        finally:
            server.close()
            await server.wait_closed()
        body = response.json()
        assert [e["event"] for e in body["events"]] == ["trace_start", "result", "done"]
        assert body["trace_id"]


# ---------------------------------------------------------------------------
# Live, read-only E2E against the running service (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("HARS_MEMORY_TRACE_URL"),
    reason="set HARS_MEMORY_TRACE_URL to the running service's /trace endpoint",
)
class TestLiveTraceEndpoint:
    async def test_live_endpoint_streams_all_stages(self) -> None:
        url = os.environ["HARS_MEMORY_TRACE_URL"]
        async with httpx.AsyncClient(timeout=180) as client:
            events = await _read_sse(
                client,
                url,
                params={"question": "Q7X42 sidecar gate", "ll_keywords": "Q7X42", "top_k": "3"},
            )
        names = [name for name, _ in events]
        for stage in EXPECTED_STAGES:
            assert stage in names, f"missing {stage}: {names}"
        assert names[-1] in ("done", "error")
        by_name = {name: payload for name, payload in events}
        for stage in ("bm25_hits", "vector_hits", "graph_expand"):
            assert by_name[stage]["latency_ms"] is not None
        assert by_name["done"]["total_ms"] > 0
