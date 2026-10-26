# Configuration reference

All configuration is via environment variables. No hardcoded defaults for
machine-specific paths — this package does not assume any particular filesystem
layout. The canonical source of truth is `config/.env.example` (176 lines with
inline rationale).

## Required variables

| Variable | Description |
|---|---|
| `HARS_MEMORY_INDEX_DIR` | Path to LightRAG working directory (NetworkX graph + KV store + Qdrant workspace config). No default — must be set explicitly |
| `HARS_MEMORY_STAGING_DIR` | Directory for `memory_remember` note files. Must exist and be writable |

## MCP server

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_EXTRACTOR_BASE_URL` | — | OpenAI-compatible endpoint for entity extraction (GPU). Required for indexing |
| `HARS_MEMORY_EXTRACTOR_MODEL` | — | Model name for extraction, e.g. `Qwen3.6-27B-Q4_K_M` |
| `HARS_MEMORY_QUERY_BASE_URL` | — | OpenAI-compatible endpoint for query LLM. Optional if `context_only=true` (default) |
| `HARS_MEMORY_QUERY_MODEL` | — | Model name for query LLM, e.g. `Qwen3.5-4B` |
| `HARS_MEMORY_EMBED_MODEL` | `embeddinggemma-300m` | Embedding model. Changing requires `--full` reindex |
| `HARS_MEMORY_QUERY_TIMEOUT_SECONDS` | `60` | Timeout for LightRAG `aquery` call |
| `HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER` | `1.0` | Multiplier for `fetch_top_k` when not explicitly passed. Measured optimum 1.0 at `top_k=20` |
| `HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT` | `merged` | Server-wide default for `context_priority` parameter |
| `HARS_MEMORY_GRAPH_BRANCH` | `auto` | Whether the LightRAG graph pass runs on the `context_only` path. `auto` = only when the resolved mode is a real graph mode (`mix`/`local`/`global`, i.e. keywords supplied); with no keywords it would degrade to `naive` — a second dense search over the same chunk store the fusion channel already queried (measured +7.6 s/query) — so it is skipped and reported as `lightrag_mode: "skipped"`. `on` = historical always-call, `off` = never call |
| `HARS_MEMORY_CONTEXT_MAX_CHARS` | `15000` | Hard global budget (emitted chunk content chars) for the assembled `context`; `0` disables. Best-ranked chunks win the space |
| `HARS_MEMORY_CONTEXT_MAX_DOC_CHARS` | `6000` | Per-document cap on emitted content (line-boundary truncation with a `…[truncated]` marker); stops one mega-note from starving the budget. `0` disables |
| `HARS_MEMORY_CONTEXT_DIVERSITY_PENALTY` | `0.05` | Soft source-diversity tiebreaker: score penalty per already-selected chunk of the same document during assembly |
| `HARS_MEMORY_SUPERSESSION_SCORING` | `1` | Enable supersession-aware rescoring (`0` to disable) |
| `HARS_MEMORY_HYBRID_ALPHA` | `0.5` | Alpha weight for dense+sparse fusion. Measured optimum 0.5 |
| `HARS_MEMORY_EXTRACTOR_EXTRA_HEADERS` | — | JSON object of extra HTTP headers for the extractor endpoint |
| `HARS_MEMORY_QUERY_EXTRA_HEADERS` | — | JSON object of extra HTTP headers for the query endpoint |

## Extractor provider (local or hosted)

Selects the entity-extraction LLM backend (`hars_memory/server/extractor_config.py`). The default `local` provider keeps the behaviour of the two `HARS_MEMORY_EXTRACTOR_*` variables above. No model id or key is baked into the code.

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_EXTRACTOR_PROVIDER` | `local` | `local` (llama.cpp / any OpenAI-compatible server on the GPU) or `openrouter` (hosted, no GPU). An empty value means `local`; anything else fails fast |
| `HARS_MEMORY_OPENROUTER_MODEL` | — | Required for `openrouter`: the hosted model id. The generic `HARS_MEMORY_EXTRACTOR_BASE_URL` / `_MODEL` are ignored for this provider |
| `HARS_MEMORY_OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Endpoint of the hosted provider |
| `HARS_MEMORY_EXTRACTOR_API_KEY_ENV` | `OPENROUTER_API_KEY` | NAME of the environment variable that holds the API key (the key itself is never logged) |
| `HARS_MEMORY_EXTRACTOR_API_KEY_FILE` | — | Optional dotenv-style file (`NAME=value` or `export NAME=value`) read only when the variable named above is unset or empty; lets an MCP server started without exported secrets authenticate |
| `HARS_MEMORY_EXTRACTOR_REASONING_EFFORT` | `low` for `openrouter`, not sent for `local` | Sent as `reasoning.effort`; `unset` disables sending, `none` turns reasoning off (measured worse on the hosted model) |
| `HARS_MEMORY_EXTRACTOR_EXTRA_BODY_JSON` | — | JSON object merged into every extractor request body |

Timeout, retries and concurrency reuse `HARS_MEMORY_LLM_TIMEOUT_SECONDS`, `HARS_MEMORY_LLM_RETRY_*` and `HARS_MEMORY_LLM_MAX_ASYNC`. With a hosted provider `memory_consolidate` and `cortex-scripts/update_kb.sh` skip the local GPU guard and do not start a local extractor. `update_kb.sh` archives a staged note only after its LightRAG `doc_status` is `processed` (`hars_memory.scripts.archive_ingested`); failed notes stay in staging and the script exits non-zero.

## Incremental refresh

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_ID_PATH_MAP` | — | `ACTUAL=LOGICAL` path-prefix mappings (`os.pathsep`-separated) applied before document ids are hashed, so a refresh run from another checkout location reproduces the ids stored in the index. See [incremental refresh](incremental-refresh.md) |
| `HARS_MEMORY_FINGERPRINT_STORE` | `<index dir>/doc_fingerprints.json` | Fingerprint sidecar used by `incremental_refresh` |

## Vector storage

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_VECTOR_STORAGE` | `NanoVectorDBStorage` | Vector backend: `NanoVectorDBStorage` (file-based, default), `QdrantVectorDBStorage` (recommended for production) |
| `HARS_MEMORY_QDRANT_URL` | `http://localhost:6333` | Qdrant gRPC/HTTP endpoint |
| `HARS_MEMORY_QDRANT_COLLECTION` | — | NOT a Qdrant collection name — LightRAG uses it as the `workspace_id` tenant id written into every payload. Actual collections are `{prefix}_lightrag_vdb_{chunks,entities,relationships}` when a prefix is set, otherwise `lightrag_vdb_{chunks,entities,relationships}` |
| `HARS_MEMORY_QDRANT_COLLECTION_PREFIX` | — | Per-tenant prefix for the Qdrant collection **names**. Set it on a shared container so each index gets its own collections (otherwise every tenant shares `lightrag_vdb_*` and only `workspace_id` separates them). Must match `qdrant_transplant.py --collection-prefix`; see `docs/http-mcp-scaling.md` §5 |
| `HARS_MEMORY_QDRANT_API_KEY` | — | Qdrant API key for authentication |

## Retrieval channels

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_RIPGREP_CHANNEL` | `1` | Enable the ripgrep live-worktree channel (`0` to disable). Searches the live filesystem for exact identifiers |
| `HARS_MEMORY_RIPGREP_ROOTS` | — | Colon-separated paths for ripgrep search roots. Defaults to paths registered in `knowledge-sources.yaml` |
| `HARS_MEMORY_FLAT_CHANNEL` | `0` | Enable the flat-dense channel. Off by default — pays CPU embed cost per query |
| `HARS_MEMORY_BM25_INDEX_PATH` | — | Path to BM25 sparse index. Auto-detected under `HARS_MEMORY_INDEX_DIR` if unset |
| `HARS_MEMORY_BM25_INDEX_ID` | — | BM25 index identifier for multi-tenant BM25. Defaults to `default` |

## Session compaction memory

Agent sessions persist every compaction (summary + dropped-context facts) through
`memory_compact`. In `split` mode each session writes to its own LightRAG working
directory and is consolidated into the main index later by `memory_merge_session`;
`subgraph` mode indexes compactions straight into the main graph. See
`docs/mcp-reference.md` for the tools.

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_SESSION_MODE` | `split` | `split` (per-session isolated working dir, merged on demand) or `subgraph` (ingest compactions directly into the main graph) |
| `HARS_MEMORY_SESSIONS_DIR` | `<HARS_MEMORY_INDEX_DIR>/sessions` | Root directory holding one sub-directory per session id (split mode). Session ids are validated against `[A-Za-z0-9._-]{1,128}` |
| `HARS_MEMORY_SESSION_INDEX_BS` | `25` | Documents per ingest batch for compaction indexing, clamped to `1..256`. Also used as the session embedding batch size |
| `HARS_MEMORY_SESSION_MIN_SCORE` | `0.2` | `memory_recall` session-channel cutoff: chunks scoring below this (max of cosine and lexical F1) are dropped |

Split-mode session ingests always force `NanoVectorDBStorage` + `NetworkXStorage`
for the session directory, even when the main index uses Qdrant, so session data
can never collide with the main collections.

## Indexing

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_CHUNK_TOKEN_SIZE` | `512` | Token budget per chunk (tiktoken `cl100k_base`; the chunker's unit is tokens). With `HARS_MEMORY_CHUNKER=md_struct` this token budget is converted to the section chunker's char budget via the canonical `CHARS_PER_TOKEN=4` ratio (`target_chars = budget * 4`) — see `hars_memory/ingest/md_structure.py`. Overridden by `CHUNKER_CHUNK_SIZE` for legacy compat |
| `HARS_MEMORY_CHUNK_OVERLAP_TOKENS` | `64` | Token overlap for the legacy token/markdown chunkers. `md_struct` is section-aligned and applies **no** overlap (its breadcrumb prefix + section boundaries replace it) |
| `HARS_MEMORY_GPU_GUARD_SCRIPT_PATH` | — | Path to a Python file exposing `assert_gpu_free(api_base_url)`. Loaded dynamically; if unset, no GPU-concurrency check is performed |
| `HARS_MEMORY_ENTITY_SCHEMA_PATH` | — | Path to custom entity/relation schema YAML. Default uses `hars_memory/schema/default_schema.yaml` |
| `HARS_MEMORY_LEGACY_ENV_PREFIXES` | — | Comma-separated forbidden env prefixes (e.g. `GRAPHRAG_`). Fails closed at startup if any env var with that prefix is set |

## Index service (HTTP server)

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_API_KEYS_JSON` | — | JSON object mapping API keys to tenant IDs. Required for the HTTP service. Example: `{"key1":"tenant-a","key2":"tenant-b"}` |
| `HARS_MEMORY_SERVICE_HOST` | `0.0.0.0` | HTTP service bind address |
| `HARS_MEMORY_SERVICE_PORT` | `8787` | HTTP service port |
| `HARS_MEMORY_SERVICE_DATABASE_URL` | `sqlite:///./hars-memory-service.sqlite3` | Job database URL. Postgres: `postgresql+psycopg://user:pass@host/db` |
| `HARS_MEMORY_ARTIFACT_STORE_URL` | `file:///./hars-memory-artifacts` | Artifact store URL. S3: `s3://bucket/prefix` |
| `HARS_MEMORY_S3_ENDPOINT_URL` | — | Custom S3 endpoint (for MinIO, etc.) |
| `HARS_MEMORY_S3_REGION` | — | S3 region for artifact store |
| `HARS_MEMORY_WORKER_SCRATCH_DIR` | `/tmp/hars-memory-worker-scratch` | Scratch directory for index workers |
| `HARS_MEMORY_WORKER_POLL_SECONDS` | `5` | Worker poll interval for new jobs |
| `HARS_MEMORY_WORKER_LEASE_SECONDS` | `30` | Job lease duration before another worker can claim it |

## Logging

| Variable | Default | Description |
|---|---|---|
| `HARS_MEMORY_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `HARS_MEMORY_LOG_FILE` | — | Optional log file path. If unset, logs to stderr |
| `HARS_MEMORY_LOG_QUERY_EVENTS` | `1` | Enable structured JSON query event logging |
| `HARS_MEMORY_LOG_QUERY_EVENTS_DIR` | `/tmp/hars-memory-query-events` | Directory for query event log files |

## Deprecated / legacy

| Variable | Status | Description |
|---|---|---|
| `CHUNKER_CHUNK_SIZE` | Deprecated | Use `HARS_MEMORY_CHUNK_TOKEN_SIZE` |
| `CHUNKER_OVERLAP_SIZE` | Deprecated | Use `HARS_MEMORY_CHUNK_OVERLAP_TOKENS` |
| `HARS_EMBED_MODEL` | Deprecated | Use `HARS_MEMORY_EMBED_MODEL` |
| `GRAPHRAG_*` | Forbidden | Legacy prefix — server fails closed if detected. Set `HARS_MEMORY_LEGACY_ENV_PREFIXES=GRAPHRAG_` to enforce |

## Variable naming convention

- `HARS_MEMORY_*` — current prefix for all env vars
- `HARS_*` (without `MEMORY`) — deprecated fallback, read if `HARS_MEMORY_*` is unset
- `CHUNKER_*` — legacy chunker config, deprecated in favour of `HARS_MEMORY_CHUNK_*`
- `GRAPHRAG_*` — legacy prefix, forbidden — the server refuses to start if any
  env var with this prefix is set (see `HARS_MEMORY_LEGACY_ENV_PREFIXES`)