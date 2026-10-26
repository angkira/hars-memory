# `memory_trace` — live recall telemetry over SSE (implemented)

Status: **implemented** (2026-10-17). This is the implementation-side contract;
the consumer-side draft lives in `memory-travel/docs/trace-api.md` and is kept in
sync (see "Deviations from the draft" below). Everything here was verified
against the running service (see "Verification").

Memory Travel v1 animates **saved** recall fixtures. v2 animates a recall
**while it executes** by subscribing to this endpoint instead of replaying
fixtures.

## Endpoint

```
GET /trace?question=...&mode=hybrid&top_k=10&fetch_top_k=...&context_only=true&ll_keywords=a,b&hl_keywords=x&project=default
Accept: text/event-stream          (add ?format=json for a buffered JSON body)
```

Behaves exactly like `tools/call memory_recall` with `context_only=true`
(default), except that it emits one SSE frame per pipeline stage **as that stage
finishes**, then a final `result` frame with the full, unchanged `memory_recall`
JSON, then a tiny terminal `done` frame. `format=json` buffers the same events
into `{"ok": true, "trace_id": ..., "events": [{"event": ..., "data": ...}]}`
(handy for non-SSE clients and debugging).

### Transport — why a sibling port, not `/mcp`

The service is a **stdio MCP server** wrapped by **supergateway**
(`--stateful`, `127.0.0.1:8099/mcp`). supergateway is a Node process and
exposes no hook to add an extra HTTP route, so the draft's `GET /mcp/trace`
cannot be hosted there. Instead the MCP process itself starts a small
**read-only HTTP/1.1 + SSE listener** on a sibling port, on the **same event
loop**, sharing `_rag_instance` and the existing caches with the MCP path.

The listener is implemented with `asyncio.start_server` — **no new runtime
dependency**. (`fastapi`/`uvicorn` are an optional extra here; depending on them
in the stdio server would be a silent failure mode after `uv run` prunes extras.)

### Configuration (all optional)

| Env var | Default | Meaning |
| --- | --- | --- |
| `HARS_MEMORY_TRACE_HTTP_PORT` | `8098` | Listener port. `0` or empty disables the listener entirely. |
| `HARS_MEMORY_TRACE_HTTP_HOST` | `HARS_MEMORY_MCP_BIND_HOST`, else `127.0.0.1` | Bind address. In the systemd service `HARS_MEMORY_MCP_BIND_HOST=127.0.0.1`, so the trace endpoint is WireGuard-only like the MCP bridge. |
| `HARS_MEMORY_TRACE_KEEPALIVE_S` | `10` | `: ping` comment interval (rerank can take 7+ s). |
| `HARS_MEMORY_TRACE_HITS_CAP` | `30` | Rows per `*_hits` event. Payload-size only; the full pool still travels in `result`. |

Bind failure (port taken) is logged and non-fatal: the MCP server always starts.

Other routes: `GET /healthz` → `{"ok": true, "service": ..., "trace": true}`;
`OPTIONS` → CORS preflight (`Access-Control-Allow-Origin: *`, so the Vite dev
UI at `:5174` can `EventSource` the endpoint); anything else → 404; a missing
`question` → 400.

## Events (wire order = real execution order)

| # | event | when | payload |
| --- | --- | --- | --- |
| 0 | `trace_start` | recall accepted | `{trace_id, seq, t_ms, question, mode, lightrag_mode, mode_fallback, top_k, fetch_top_k, fetch_top_k_source, alpha, index_id, index_dir, stale_days, last_ingest, context_only, context_priority, ll_keywords, hl_keywords}` |
| 1 | `bm25_hits` | sparse channel done | `{trace_id, seq, t_ms, latency_ms, chunks_indexed, pool_size, hits:[{chunk_id, sparse_score, doc, section}]}` |
| 2 | `vector_hits` | dense channel done | `{trace_id, seq, t_ms, latency_ms, embedder, pool_size, hits:[{chunk_id, dense_score, doc, section}]}` |
| 3 | `ripgrep_hits` | ripgrep channel done — or disabled/unavailable | `{trace_id, seq, t_ms, enabled, skipped, reason, latency_ms, mode, available, query_terms, hits:[{chunk_id, ripgrep_score, match_count, matched_terms, doc, section}], matched_chunk_ids:[chunk_id…]}`. Emitted on **every** path: `enabled:false, skipped:true, reason:"HARS_MEMORY_RIPGREP_CHANNEL=0"` when off; `enabled:true, skipped:true, reason:<unavailable_reason>` when `rg` is missing; `enabled:true, skipped:false` + hits when it ran. `matched_chunk_ids` are the pool chunks whose file the literal match joined to (the channel's ranked contribution), not the pre-join file-level hits. |
| 4 | `fusion` | RRF/alpha blend + diversity done | `{trace_id, seq, t_ms, alpha, candidate_pool_size, pool_size, order:[chunk_id…], scores:{chunk_id: fused}}` |
| 5 | `rerank` | cross-encoder done / skipped / disabled | `{trace_id, seq, t_ms, applied, backend, latency_ms, pool, fallback_reason, max_fusion_score, skip_threshold, blocked_inversions, fusion_margin, before:[ids], after:[ids]}`. With the default `HARS_MEMORY_RERANK_BACKEND` (off) this is emitted as `applied:false, backend:"off", latency_ms:null, fallback_reason:"disabled_by_default"` — never a fabricated timing. |
| 6 | `graph_expand` | LightRAG entity/relation pass done — or skipped | `{trace_id, seq, t_ms, skipped, mode, latency_ms, entity_count, relation_count, traversed_edge_count, entities:[{name, type}], relations:[{entity1, entity2}], derived_from}`; when `skipped: true`: `{…, skipped: true, reason, mode: "skipped", setting, derived_from: null}` and **no** entity/relation counters at all |
| 7 | `context` | prompt context assembled | `{trace_id, seq, t_ms, has_context, chars, context_priority_applied, source_count, entity_count, relation_count, sources:[{reference_id, doc, section}], caps:{max_chars, max_entity_tokens, max_relation_tokens}}` |
| 8 | `result` | final | **today's full `memory_recall` JSON, unchanged** (no trace fields added) |
| 9 | `done` | terminal (success) | `{trace_id, ok, total_ms, has_context}` |
| 9 | `error` | terminal (failure) | `{trace_id, ok:false, total_ms, error_type, error, hint?}` |

The pipeline is **parallel channels: vector + BM25 + ripgrep + graph**.

Notes:

- **Dependency DAG vs wire order.** The wire order above is the real execution
  order, but it is a **linearisation of a two-branch DAG**, not a chain:

  ```
  keywords/mode ─┬─ bm25_hits (sparse) ────┐
                 ├─ vector_hits (dense) ────┤
                 ├─ ripgrep_hits (literal) ─┴─▶ fusion ─▶ rerank ─┐
                 │        (flat / qwen gates feed fusion, too)    │
                 └─ graph_expand (LightRAG aquery: entity/relation pass) ─┴─▶ context ─▶ result
  ```

  `fusion` fuses **dense+sparse** (`retrieval/fusion.py::fuse`) plus the
  env-gated additive ripgrep/flat/3way gates (supersession is a
  content-marker re-score, not a graph signal); the ripgrep channel is default
  ON in `ranked` mode and re-ranks chunks INSIDE the top-k window. It does **not** consume
  `graph_expand` output: LightRAG's graph pass runs *after* the whole hybrid
  block has finished (`mcp_server.py`: `_compute_hybrid_block` at the
  `memory_recall` body, then `rag.aquery`), and `graph_expand` is emitted
  afterwards by **parsing the rendered context string** (`derived_from:
  "context_string"`). The two branches are independent and converge only in
  `context`: the assembly orders both branches' CHUNK candidates by score
  (diversity is a soft tiebreaker) under the global/per-document char budgets.
  A consumer must therefore not draw a
  `graph_expand → fusion` edge; see
  `tests/test_trace_endpoint.py::TestRealPipelineTrace::
  test_fusion_is_independent_of_the_later_graph_pass`.
- With `mode: hybrid` + `lightrag_mode: naive` (the `mode_fallback` case, or an
  explicit `mode=naive`) there is no graph traversal, but the trace still emits
  `graph_expand` — with `mode: naive` and `entity_count/relation_count/
  traversed_edge_count` all `0` (parsed from a context string that contains no
  entity/relation JSON lines). The frame is **not** skipped; a zeroed frame is
  the honest signal that the branch ran and yielded nothing.
- **Graph-branch routing (2026-10-17 review rec. #1).** `trace_start` reports the
  RESOLVED mode (`lightrag_mode`) plus a `graph_branch` `{enabled, setting,
  reason}` block. Under the default `HARS_MEMORY_GRAPH_BRANCH=auto` the graph pass
  is skipped when the resolved mode is LightRAG `naive` (no keywords supplied, or
  an explicit `mode=naive`): with no keywords LightRAG could only re-run a dense
  vector search over the same `chunks_vdb` branch 1 already queried (measured
  +7.6 s/query for no quality gain), so the context is assembled from the fusion
  branch alone. `trace_start.lightrag_mode` then reads `"skipped"` — never a fake
  `naive` — and `graph_expand` is emitted with `skipped: true` + `reason` and NO
  entity/relation counters (zeros would pretend the pass ran and found nothing).
  `HARS_MEMORY_GRAPH_BRANCH=on` restores the always-call behaviour, `=off` never
  calls it, and an explicit `context_priority=lightrag` or the answer path keep
  it on. When it runs, the frame carries `skipped: false` and the measured counts
  as before.
- `order`/`scores`/`before`/`after` are the **true pre-rerank order**, measured

  where `_compute_diverse` runs, not reconstructed from dense scores — this
  removes the only reconstructed pixel in the v1 demo.
- A skipped/disabled rerank still emits event 5 (`applied=false`) with its
  recorded `fallback_reason` (`disabled_by_default` on the default path;
  `fusion_confident`, `empty_pool`, `timeout`, … on the opt-in http path);
  `latency_ms` is `null` when the backend was never called, never a fabricated
  `0`.
- A disabled/unavailable ripgrep channel still emits event 3 with
  `enabled`/`skipped`/`reason` — a missing frame would be indistinguishable
  from a channel that ran and found nothing.
- `result` carries no `trace_id`/`seq`/`t_ms`: it is byte-for-byte the normal
  response (the client can correlate it via the terminal `done.trace_id`).
- A recall whose implementation returns a bare `{ok:false, error…}` (no
  `hybrid`) is surfaced as a terminal `error`, never a fake `result`. A genuine
  "no context" answer still carries `hybrid` and is reported as a normal result.

## Measured vs derived (honesty)

**Measured** (read straight off the pipeline):

- every `latency_ms` (`bm25_hits`, `vector_hits`, `ripgrep_hits`, `rerank`,
  `graph_expand`) and the terminal `total_ms`;
- hit ids/scores/doc/section/file for every channel; `chunks_indexed`,
  `candidate_pool_size`, `pool_size`; `alpha`; the ripgrep channel's
  `enabled`/`available`/`skipped`/`reason`/`mode`/`query_terms`/
  `matched_chunk_ids`; the whole rerank status
  (`applied`, `backend`, `pool`, `fallback_reason`, `max_fusion_score`,
  `skip_threshold`, `blocked_inversions`, `fusion_margin`);
- `order`/`before`/`after` (pre-rerank is captured before rerank mutates the
  pool); `chars`, `has_context`, `context_priority_applied`, the caps;
  `stale_days`, `last_ingest`, `index_id`; `embedder`.

**Derived** (documented, never faked):

- `graph_expand.{entity_count, relation_count, traversed_edge_count, entities,
  relations}` and `context.{source_count, entity_count, relation_count,
  sources}` are **parsed from the rendered LightRAG context string** (the same
  parse `memory-travel/scripts/compile_trace.py::parse_context` uses). On the
  `context_only` path that string is the only place these names exist. The
  `derived_from` field states the source (`"context_string"` /
  `"answer_result"`); the answer path has no rendered context, so it reports
  `relation_count: null` rather than guessing.
- `seq`, `t_ms`, `trace_id` are synthetic (ordering/correlation only).

**Omitted** (cannot be measured at that point):

- a separate `embed` event: embedding latency is not separable from the dense
  `chunks_vdb.query` call and is included in `vector_hits.latency_ms`. The
  draft's `embed`/`query_xy_hint` are dropped (the latter needs a persisted UMAP
  reducer the service does not have).

## Read-only and isolation

- Zero index writes; no new auth surface (same project/`access_token` handling
  as `memory_recall` through the same code path).
- The MCP/CLI path is untouched: an unset `ContextVar` makes every `emit()` a
  no-op. Verified by a test that runs the identical recall with and without an
  active trace and compares the responses.

## Verification

```bash
# after restarting hars-longterm-memory-mcp.service
curl -N 'http://127.0.0.1:8098/trace?question=Q7X42%20sidecar%20gate&ll_keywords=Q7X42&top_k=3'
curl -s 'http://127.0.0.1:8098/healthz'
HARS_MEMORY_TRACE_URL=http://127.0.0.1:8098/trace uv run pytest -q tests/test_trace_endpoint.py::TestLiveTraceEndpoint
```

`memory-travel` v2 client: append events by name into the existing `TravelTrace`
shape — stage panels consume `vector_hits`/`bm25_hits`/`fusion`/`rerank`/
`graph_expand`/`context`, the answer/grounding panel consumes `result`. No
component change is needed for the fixture schema itself.

## Cosmos export endpoints (2026-10-18)

Same listener, read-only, for the Memory Travel UI's data-integrity fix (the UI
used to load baked fixtures unconditionally and could show OUR index on a
FOREIGN deployment). Full consumer-facing contract + deployment recipes:
`memory-travel/docs/CONNECTION.md`.

| Route | Purpose | Notes |
|---|---|---|
| `GET /stats` | identity + counts | `{index_id, docs, chunks, entities, relations, nodes, edges, last_ingest, generated_at}`; `index_id` = basename of `HARS_MEMORY_INDEX_DIR` (source of truth for the UI badge); 503 `index_dir_unconfigured` without the env |
| `GET /graph?limit_nodes=N&limit_edges=M` | sampled starfield | defaults 8000/16000, caps 50000/200000; top-N **by undirected degree** (ties in graphml order); `edges` are index pairs into the returned `nodes` array; `meta.sampled/full_nodes/full_edges` say exactly what was cut; `desc_head` truncated to 200 chars |
| `GET /coords?dim=3` | UMAP projection over stored vectors | served from `<index_dir>/coords_cache.json` when fresh (stored `source_mtimes` == current `vdb_entities.json`/`vdb_chunks.json` mtimes, `dim` covered); otherwise computed inline if umap is importable (minutes; documented) else 503 `coords_not_cached` with the offline hint: `uv run --with numpy --with umap-learn python scripts/export_cosmos.py --index-dir <dir>` |

- Read-only wrt the index: the ONLY file ever written is `coords_cache.json`
  (a new file LightRAG ignores). The 325 MB `vdb_entities` matrix is never
  serialised; graph/stats are cached in memory keyed by file mtime.
- No new auth; same CORS posture as `/trace`. POST to these routes → 405.
- Tests: `tests/test_graph_export.py` (hermetic synthetic index through the
  real listener).
