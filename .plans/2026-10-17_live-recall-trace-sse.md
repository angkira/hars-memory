# Live recall trace (SSE) for the Memory Travel UI

## Goal
Emit per-stage `memory_recall` telemetry **while** a recall executes, over a
streaming HTTP endpoint, so `memory-travel` can animate a recall live instead of
replaying saved fixtures. Contract source of truth:
`memory-travel/docs/trace-api.md` (draft). Implementation doc:
`docs/trace-api.md` in this repo.

## Transport decision (deviation from the draft, documented)
The service is a **stdio MCP server wrapped by supergateway** (`--stateful`,
`memory-host:8099/mcp`). supergateway is Node and exposes no hook to add a plain
HTTP route, so the draft's `GET /mcp/trace` cannot be hosted there. Instead the
MCP process itself starts a **sibling read-only HTTP/1.1 + SSE listener** on
`HARS_MEMORY_TRACE_HTTP_PORT` (default `8098`; host defaults to
`HARS_MEMORY_MCP_BIND_HOST`, i.e. WireGuard-only, else `127.0.0.1`). Implemented
with `asyncio.start_server` — **no new runtime dependency** (fastapi/uvicorn are
an optional extra and would be a silent failure mode after `uv run` prunes it).
Runs on the same event loop, so the trace recall shares `_rag_instance` and the
existing caches with the MCP path.

## Emitter design (zero impact on the default path)
- `hars_memory/trace.py` owns a `ContextVar[EmitFn | None]` (`_trace_emitter`),
  `emit(event, **payload)` (no-op when unset), context parsing, SSE framing and
  the HTTP server.
- `mcp_server.py` calls `await trace_emit(...)` at stage boundaries. The
  ContextVar defaults to `None`, so the MCP/CLI path emits nothing and its
  behaviour is byte-identical (the SDK decorator calls `call_tool(name, args)`
  with 2 positional args; no signature change at all).
- The trace HTTP handler sets the ContextVar, then `await call_tool("memory_recall", args)`
  — the **same** code path, same auth, same project resolution, same response.

## Events (execution order on the wire)
`trace_start` (keywords/mode) → `bm25_hits` (sparse) → `vector_hits` (dense) →
`fusion` (pre-rerank order) → `rerank` (before/after, skip reason) →
`graph_expand` → `context` → `result` (full, unchanged JSON) → `done`
(terminal) or `error` (terminal).

Real order differs from the draft's UI order (dense→graph→sparse) because the
code runs sparse before dense, and the LightRAG graph pass after the hybrid
block; a live stream must reflect actual execution. Events carry `seq` and
`t_ms`, so the client animates by arrival.

## Files
- `hars_memory/trace.py` (new): ContextVar, emit, parsing, SSE, HTTP server.
- `hars_memory/mcp_server.py`: emit hooks + start the trace listener in `_serve`.
- `tests/test_trace_endpoint.py` (new): emitter ordering, one-per-stage, error
  path, no-leak non-trace path, SSE framing, HTTP end-to-end with a fake recall,
  opt-in live test against the running service (`HARS_MEMORY_TRACE_URL`).
- `docs/trace-api.md` (new) + `memory-travel/docs/trace-api.md` (update).
- `tests/conftest.py`: disable the listener by default in tests.

## Measured vs derived (honesty)
Measured: per-channel latency, hit ids/scores/doc/section/file, pool sizes,
alpha, rerank status, graph latency, context char count, total ms, stale_days.
Derived: graph entities/relations/counts and context source list are parsed from
the returned LightRAG context string (the same parse `compile_trace.py` uses);
`seq`/`t_ms`/`trace_id` are synthetic. Not emitted: a separate `embed` event
(embedding latency is not separable from the dense call).

## Verification
1. `uv run pytest -q` (hermetic incl. new tests).
2. Restart `hars-longterm-memory-mcp.service` **once** at the very end.
3. `memory_recall` still green via the bridge; `curl -N .../trace` streams the
   raw event sequence; live test asserts all stages with plausible timings.
