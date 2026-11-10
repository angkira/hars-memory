# Terminal UI (`memory tui` / `memory-tui`)

A [Textual](https://textual.textualize.io/) front-end for a running HARS service. It is a pure
out-of-process client: it talks HTTP to the service and reads a few local directories. It never imports
`hars_memory.mcp_server`, LightRAG or torch, so it starts fast and can run on a laptop against a remote
box. Design record: `.plans/2026-10-26_tui.md` (Phase 1 MVP is implemented; later phases are stubs).

## Install and run

```bash
pip install 'hars-longterm-memory[tui]'      # or: uv sync --extra tui
memory tui                                   # or: memory-tui
memory tui --url http://memory-host:8099 --screen recall
```

Without the extra, `memory tui` prints the install hint and exits with status 2. Textual is pinned
`>=8.2,<9` (8.2.8 on PyPI as of 2026-10-26; 8.x has had breaking changes between majors).

## Screens and keys

| Key | Screen | What it shows |
| --- | --- | --- |
| `1` / `F1` | **Health** | HARS API (`/healthz`), MCP endpoint (`tools/list`), trace listener, index (`memory_status`), embedder / extractor / query LLM (the `configured_models` strings from `memory_status`, plus a reachability probe `GET <base>/models` from the TUI host), Qdrant (only if the vector backend is Qdrant), engine shards, index-job service. Polls every 5 s; `r` refreshes. A dead service is a red row with the last error, never a traceback. |
| `2` / `F2` | **Recall** | question, project, mode, top_k, `ll_keywords` / `hl_keywords`, `session_id` + `session_graph`, stage-trace toggle. Stage events stream over SSE into the left table (select a row for its payload). Right: *Packed context* (KG entities, KG relations, document chunks with their fused score), *Scored hits* (fused / dense / sparse, `file:start_line`), *Meta* (rerank block, routing, latencies, `session` and `session_graph` blocks, `keyword_source`). `Ctrl+R`/Enter runs, `Ctrl+P`/`Ctrl+N` walk the history, `o` opens the selected hit at `start_line` in `$EDITOR`/`$PAGER`, `y` copies the chunk id. |
| `3` / `F3` | **Sessions** | session inventory (`memory_session_status`), per-session compactions (summary + dropped-context facts) and the session graph as an entity -> relations tree. `/` filters sessions, the graph tab has its own filter. Polls and re-reads when the session files change. `M` merge into the main index, `C` manual compaction (both confirm-gated). |
| `4` / `F4` | **Projects** | `memory_list_projects` and, per project, `memory_status` (index existence, node/edge counts, last ingest, storage, models, staging backlog). Read-only. |
| `5` / `F5` | **Eval results** | point it at a runs / artifacts directory. Per run: verdict, `run_manifest.json` / `MANIFEST.json` (profile, models, embedder, spend, redacted command line), per-arm table, paired deltas with CIs, per-qtype strata and slices, WIN/FAIL gate rules, invalid chains with their reasons. |
| `?` / `F10` | **Help** | keys, the stubbed later phases, the open API gaps. |
| `Ctrl+T` | Remember | opens the note form (title, tags, importance, body) -> confirm modal -> `memory_remember`. |
| `q` | Quit | |

Number keys are typed into text fields, so the `F` keys always work; `Esc` returns from a filter to the list.

### Recall: SSE trace vs MCP

The trace listener (`HARS_MEMORY_TRACE_HTTP_PORT`, default 8098, see `docs/trace-api.md`) emits one event per
pipeline stage as it finishes. The TUI uses it when the query can be expressed on `/trace`
(question, mode, top_k, project, keywords). Otherwise it calls `memory_recall` over MCP and shows a single
result row: no stage events, but the full result. That happens for

* anything with `session_id` / `session_graph` (the listener ignores them, plan **GAP-1**),
* a disabled or unreachable trace listener (the status line says which).

Over `/trace` a multi-word keyword is split on spaces by the server; over MCP it is kept intact.

### Sessions: where the data comes from

The inventory comes from MCP. Compactions and the session graph have **no API yet** (plan **GAP-2**), so the
TUI reads the session working dirs directly: `<sessions dir>/<sid>/kv_store_full_docs.json` and
`graph_chunk_entity_relation.graphml`, read-only, with retry on a torn write. That works on the machine
that runs the service, or with the directory mounted. The root is `--sessions-dir` /
`HARS_MEMORY_TUI_SESSIONS_DIR` / `HARS_MEMORY_SESSIONS_DIR`; if none is set, the `sessions_dir` the server reports is
used when it exists on this host. Otherwise those panes say so (the inventory still works).

### Eval viewer: what is read

No metric is ever recomputed. The viewer shows what the tools wrote: a `metrics_v2.py --json` file
(`meta`, `summary`, `deltas`, `gates`, `slices`, `longsession.strata`), `run_manifest.json` /
`MANIFEST.json`, the long-session runner's `summary.json` (`by_arm_budget`, `by_qtype`),
`chain_budget.jsonl`, `done.jsonl`, `*run_budget.json` and (fallback) the `valid` /
`chain_invalid_reasons` fields of `records.jsonl`. Judge text and arm-blinding files are not loaded.
It scans the given directory and two levels below it.

## Writes are gated

Writes available in the MVP: `memory_remember`, `memory_compact`, `memory_merge_session`. Every one opens a
modal that prints tool, project and arguments (merge also warns that it re-extracts into the **main**
index). Writes into project `default` (the shared KB) need a second, typed confirmation. `--read-only`
refuses all writes. Admin tools (`memory_forget`, `memory_delete_document`, `memory_consolidate`) are
refused by the client unless `--allow-admin` is given, and the MVP has no screen that offers them.
Tokens are never printed, stored in history, or sent to embedder / LLM probe URLs.

## Configuration

Flags override environment variables. The server's own variables are honoured as defaults, so a TUI next to
the service needs no configuration.

| Variable | Flag | Default |
| --- | --- | --- |
| `HARS_MEMORY_TUI_URL` | `--url` | `http://<HARS_MEMORY_MCP_HTTP_HOST or 127.0.0.1>:<HARS_MEMORY_MCP_HTTP_PORT or 8099>` (`https` when `HARS_MEMORY_MCP_TLS_CERTFILE` is set) |
| `HARS_MEMORY_TUI_MCP_URL` | `--mcp-url` | `<url><HARS_MEMORY_MCP_HTTP_PATH or /mcp>` |
| `HARS_MEMORY_TUI_TRACE_URL` (or `HARS_MEMORY_TRACE_URL`, a trailing `/trace` is stripped) | `--trace-url` (`off` disables) | same host, `HARS_MEMORY_TRACE_HTTP_PORT` or 8098; `HARS_MEMORY_TRACE_HTTP_PORT=0` disables |
| `HARS_MEMORY_TUI_JOBS_URL` | `--jobs-url` | unset (health probe only) |
| `HARS_MEMORY_TUI_TOKEN`, else `HARS_MEMORY_ACCESS_TOKEN` | `--token-env NAME` (names the variable; there is no `--token VALUE`) | none |
| `HARS_MEMORY_TUI_SESSIONS_DIR`, else `HARS_MEMORY_SESSIONS_DIR` | `--sessions-dir` | none (server-reported dir if it exists locally) |
| `HARS_MEMORY_TUI_EVAL_ROOT` | `--eval-root` | none (type a path on the Eval screen) |
| `HARS_MEMORY_TUI_PROJECT` | `--project` | `default` |
| `HARS_MEMORY_TUI_STATE_DIR` | `--state-dir` | `$XDG_STATE_HOME/hars-memory-tui` (recall history) |
| `HARS_MEMORY_TUI_REFRESH_S` | `--refresh` | 5 |
| `HARS_MEMORY_TUI_ALLOW_ADMIN=1` | `--allow-admin` | off |
| `HARS_MEMORY_TUI_READ_ONLY=1` | `--read-only` | off |
| | `--no-probe` | probe embedder/LLM URLs from this host |
| | `--screen NAME` | `health` |

The token is sent as `Authorization: Bearer` and as the `access_token` tool argument to the configured HARS
host only (and as a query parameter to the trace listener, which has no header auth).

## Not in the MVP (stubs, see the Help screen)

Phase 2 memory ops (upsert-document form, entities/related/inspect browser, sync-status, artifact browser),
Phase 3 indexing jobs, Phase 4 eval launcher, Phase 5 themes/publish. Never planned: graph canvas, project
CRUD, token management, metric recomputation.

Open API gaps the TUI works around: GAP-1 (`/trace` ignores session fields), GAP-2 (no session
compaction/graph read tools), GAP-3 (no deep health: embedder/extractor are probed from the TUI host),
GAP-4/6 (no job list / progress).

## Tests

`tests/test_tui_core.py` (config, adapters, SSE parser, health probes, file readers, eval reader; no Textual
needed) and `tests/test_tui_app.py` (Textual `Pilot`, one or more tests per screen against the fake backend in
`tests/tui_support.py`, an `httpx.MockTransport`; no sockets, no live server). SVG snapshot tests are not used:
`pytest-textual-snapshot` 1.1.0 forces `pytest<9`, which conflicts with the project's pytest.
