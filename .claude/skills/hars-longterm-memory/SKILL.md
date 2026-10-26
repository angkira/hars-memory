---
name: hars-longterm-memory
description: Use when the user asks to set up, query, or maintain a persistent LightRAG/Qdrant-backed long-term memory (knowledge base) for a project via the hars-longterm-memory MCP server or CLI — covers installation, configuration, the memory_* MCP tools, and the `memory` CLI subcommands.
---

# hars-longterm-memory

Read this whole file before doing anything — it assumes you have never seen
the `hars-longterm-memory` repo and are working in a completely different
project that wants to add persistent memory to it.

## What it is

`hars-longterm-memory` is a standalone Python package providing a graph+vector
knowledge base built on LightRAG, with Qdrant as the vector backend (NetworkX
graph storage, CPU-only embeddings, pluggable LLM/embedding backends via
plain HTTP env-var config — no code changes needed to swap models). It is
exposed two ways: an **MCP server** (`hars-longterm-memory-mcp` /
`memory-mcp`, stdio transport) for direct use by an AI agent inside a coding
session, and a **CLI** (`memory`, plus the standalone `memory-index` /
`memory-eval-battle` scripts) for scripting, CI, and one-off maintenance.
Published to a private GitLab Package Registry — package name
`hars-longterm-memory`, GitLab project id `14`
(`https://code.knotgate.com/hars/hars-longterm-memory`).

**Full documentation** is available in the `docs/` directory of the source
repo: MCP tool reference, configuration reference, architecture, integration
guide, and release/distribution process. This skill is a quick-start; the
docs are the canonical source.

## Installing into a new project

Create a small companion config project (its own directory, its own
`pyproject.toml` + `uv.lock` + `.venv`) that depends on the published
package. Do not vendor the package's source into your own project.

```toml
# your-project/tools/memory-config/pyproject.toml
[project]
name = "your-project-memory-config"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
  "hars-longterm-memory==0.1.3",
  # torch MUST be listed here as an explicit DIRECT dependency, even though
  # hars-longterm-memory already depends on it — a built wheel does not
  # propagate its own [tool.uv.sources] table to a consumer, so without this
  # explicit line + the [tool.uv.sources] override below, torch resolves to
  # the full CUDA/PyPI build instead of the CPU-only one.
  "torch>=2.0.0",
]

[[tool.uv.index]]
name = "hars-longterm-memory"
url = "https://code.knotgate.com/api/v4/projects/14/packages/pypi/simple"
explicit = true   # only used when a package explicitly requests this index —
                   # keeps every OTHER dependency resolving off plain PyPI

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
hars-longterm-memory = { index = "hars-longterm-memory" }
torch = { index = "pytorch-cpu" }
```

```bash
cd your-project/tools/memory-config
uv sync
```

## Registering the MCP server

Drop this into your project's `.mcp.json` (adjust paths/models to your own
deployment). Every env var below is documented inline: what it controls,
whether it's required, and an example value.

```json
{
  "mcpServers": {
    "hars-longterm-memory": {
      "command": "uv",
      "args": ["run", "--project", "tools/memory-config", "hars-longterm-memory-mcp"],
      "env": {
        "HARS_MEMORY_INDEX_DIR": "/home/you/.local/share/your-project-memory/index",
        "HARS_MEMORY_STAGING_DIR": "/home/you/.local/share/your-project-memory/staging",

        "HARS_MEMORY_VECTOR_STORAGE": "QdrantVectorDBStorage",
        "HARS_MEMORY_GRAPH_STORAGE": "NetworkXStorage",
        "HARS_MEMORY_QDRANT_URL": "http://localhost:6335",
        "HARS_MEMORY_QDRANT_COLLECTION": "your_project_memory",
        "HARS_MEMORY_QDRANT_COLLECTION_PREFIX": "your_project",

        "HARS_MEMORY_EMBED_MODEL": "unsloth/embeddinggemma-300m",
        "HARS_MEMORY_EMBED_LOCAL_FILES_ONLY": "1",
        "HF_HOME": "/home/you/.hf_home",

        "HARS_MEMORY_EXTRACTOR_BASE_URL": "http://localhost:8080/v1",
        "HARS_MEMORY_EXTRACTOR_MODEL": "Qwen3.6-27B-Q4_K_M",
        "HARS_MEMORY_QUERY_BASE_URL": "http://localhost:8081/v1",
        "HARS_MEMORY_QUERY_MODEL": "Qwen3.5-4B-Q4_K_M",

        "HARS_MEMORY_ENTITY_SCHEMA_PATH": "",
        "HARS_MEMORY_GPU_GUARD_SCRIPT_PATH": "",
        "HARS_MEMORY_LEGACY_ENV_PREFIXES": "",
        "HARS_MEMORY_RIPGREP_ROOTS": "/home/you/your-project",

        "HARS_API_BASE_URL": "http://localhost:8765"
      },
      "timeout": 60000
    }
  }
}
```

| Env var | Required? | What it controls |
|---|---|---|
| `HARS_MEMORY_INDEX_DIR` | **Required, no default** | Where the LightRAG working dir (GraphML graph + KV JSON stores) lives. The server refuses to start without it — no machine-specific default is assumed. |
| `HARS_MEMORY_STAGING_DIR` | **Required, no default** | Where `memory_remember` writes notes before the next `memory-index` / `memory_consolidate` run merges them into the graph. |
| `HARS_MEMORY_VECTOR_STORAGE` | Optional (default `NanoVectorDBStorage`) | LightRAG vector backend class name. Set to `QdrantVectorDBStorage` to use Qdrant. |
| `HARS_MEMORY_GRAPH_STORAGE` | Optional (default `NetworkXStorage`) | LightRAG graph backend class name. |
| `HARS_MEMORY_QDRANT_URL` | Optional (default `http://localhost:6335`) | Qdrant endpoint. Only relevant when `HARS_MEMORY_VECTOR_STORAGE` contains `qdrant`. |
| `HARS_MEMORY_QDRANT_COLLECTION` | Optional (default `hars_longterm_memory`) | **Not a Qdrant collection name** — LightRAG uses this as the tenant id (`workspace_id`) written into every payload/query filter. The real Qdrant collections are always fixed to `lightrag_vdb_{chunks,entities,relationships}`. |
| `HARS_MEMORY_QDRANT_COLLECTION_PREFIX` | Optional, but effectively **required if using Qdrant** | Namespaces those three fixed collection names per project. Without it, `memory_status`'s Qdrant branch fails closed with an explicit `config_error` (not a silent fallback to unprefixed, collision-prone names). |
| `HARS_MEMORY_EMBED_MODEL` | Optional (default `unsloth/embeddinggemma-300m`) | CPU embedding model (HF). Changing it changes the vector dimension — reindex with `memory-index --full` after switching. |
| `HARS_MEMORY_EMBED_LOCAL_FILES_ONLY` | Optional (default `1`) | Set `0` to allow the embedder to download from the HF Hub instead of requiring a local cache hit. |
| `HF_HOME` | Optional (default `/mnt/datasets/models/.hf_home` — a machine-specific value from the original deployment; **set your own**) | HF cache root for the embedder (and reranker, if enabled). |
| `HARS_MEMORY_EXTRACTOR_BASE_URL` / `HARS_MEMORY_EXTRACTOR_MODEL` | Optional (defaults point at `localhost:8080` / `Qwen3.6-27B-Q4_K_M`) | OpenAI-compatible endpoint + model used for entity/relation extraction during indexing. GPU-exclusive in the reference deployment; only touched by `memory-index` / `memory_consolidate` (non-dry-run), never by query tools. |
| `HARS_MEMORY_QUERY_BASE_URL` / `HARS_MEMORY_QUERY_MODEL` | Optional (defaults point at `localhost:8081` / `Qwen3.5-4B-Q4_K_M`) | Endpoint + model used for LLM-mediated query paths (`memory_recall` with `context_only=false`). Can be a much smaller/CPU model. |
| `HARS_MEMORY_ENTITY_SCHEMA_PATH` | Optional, **has a package default** | Path to a YAML file of `entity_types` / `relation_types` / extraction `guidance`. If unset, the package's own generic `hars_memory/schema/default_schema.yaml` is used automatically — you only need to set this for domain-specific entity typing. |
| `HARS_MEMORY_GPU_GUARD_SCRIPT_PATH` | Optional, **unset = no-op** | Path to a Python file exposing `assert_gpu_free(api_base_url)`, called before every non-dry-run `memory_consolidate` and (by convention) any consumer indexing wrapper. If unset, **no GPU-concurrency check is performed at all** — indexing proceeds unconditionally. Query tools (`memory_recall`, `memory_status`, etc.) never call this guard regardless — CPU embeddings keep them available even mid-training. |
| `HARS_MEMORY_LEGACY_ENV_PREFIXES` | Optional, unset = no-op | Comma-separated env-var prefixes that make the MCP server refuse to start if present (fail-closed guard against a stale rename). |
| `HARS_MEMORY_RIPGREP_ROOTS` | Optional, unset = channel unavailable (skipped, not an error) | Comma-separated absolute paths `rg` searches live (not the index) for exact identifiers — lets `memory_recall` surface files edited since the last consolidation. |
| `HARS_API_BASE_URL` | Optional (default `http://localhost:8765`) | Backend URL a GPU-guard script would query; only meaningful if you supply one via `HARS_MEMORY_GPU_GUARD_SCRIPT_PATH`. |

There are additional retrieval-tuning env vars (hybrid alpha, fetch-width
multiplier, supersession scoring, etc.) with measured defaults — see
`config/.env.example` in the `hars-longterm-memory` source repo for the full
list with the measurements behind each default. None of them are required.

## The MCP tools

All seven tools, grounded in `hars_memory/mcp_server.py`'s `list_tools()`.

### `memory_recall`

**Query language rule:** write `question`, `ll_keywords` and `hl_keywords` in the language of the documents you expect to match, not the language the user asked in. English corpus: English (a Russian question over English docs finds the fact in the top-5 ~49% of the time vs ~78% for English). Russian corpus: Russian (a Russian question was never worse than English there). Mixed: add keywords in both languages or repeat the query. Keep identifiers, field names and error codes verbatim.
Primary query workhorse. Params: `question` (string, **required**), `mode`
(`local`|`global`|`hybrid`|`naive`, default `hybrid`), `top_k` (int, default
20, 1-50 — result-count knob), `fetch_top_k` (int, 1-200, optional —
fetch-width knob, must be ≥ `top_k`), `ll_keywords` (array of strings —
specific entities/codes extracted from the question), `hl_keywords` (array
of strings — themes/concepts), `context_only` (bool, default `true` — return
raw retrieved context for the calling agent to synthesize, skipping the
local answer LLM entirely), `context_priority` (`lightrag`|`merged`, default
`merged`). Supply `ll_keywords`/`hl_keywords` yourself (you are the LLM) —
omitting both silently falls back to `naive` mode. The response always
includes a top-level `hybrid` block (additive dense+BM25 fusion,
`identifier_matches` for exact codes/ids, and a `confidence.low_confidence`
marker — check that before treating results as reliable).

```json
{"question": "What broke the Qdrant migration?",
 "ll_keywords": ["Qdrant", "migration"], "hl_keywords": ["vector storage cutover"],
 "mode": "hybrid", "top_k": 12}
```

### `memory_remember`
Save a note into the staging area (instant file write, no LLM/GPU). Params:
`title` (string, **required** — kebab-case slug), `content` (string,
**required** — self-contained markdown prose, use full entity names/codes,
not abbreviations), `importance` (`critical`|`normal`|`low`, default
`normal`), `tags` (array of strings).

```json
{"title": "qdrant-migration-cutover-bug",
 "content": "The 2026-07-30 vector transplant left workspace_id unset...",
 "importance": "critical", "tags": ["qdrant", "migration"]}
```

### `memory_entities`
Search graph entities by name/alias. Params: `name` (string, **required**),
`limit` (int, default 10, 1-50). Returns matching nodes plus their 1-hop
neighbourhood.

### `memory_related`
Compact node/edge subgraph from one entity. Params: `entity_id` (string,
**required**, e.g. `hyp:abc-123`), `hops` (int, default 1 — **1 is the only
supported value**; 2 hops measured at 160 KB, 3 hops at 2.1 MB, unusable for
any context budget). Response is capped at ~150 nodes / ~300 edges even at 1
hop on a hub node.

### `memory_status`
No params. Index health: existence, node/edge counts, last-ingest age,
vector-backend detail, configured models, staging backlog. GPU-free, always
available — call this first to confirm the index exists before querying.

### `memory_consolidate`
Trigger incremental (re)ingest. Params: `paths` (array of strings, optional
— defaults to `[".plans", "docs"]`), `since` (ISO8601 string, optional —
only reindex files newer than this), `dry_run` (bool, default `true`).
Relative `paths` resolve against the **MCP server process's own current
working directory** at invocation time (fixed in v0.1.3 — see the parent
repo's `AGENTS.md` "Known parked bug" section for the history). Blocked by
the GPU guard (if configured) unless `dry_run=true`.

```json
{"paths": [".plans", "docs", ".session"], "dry_run": false}
```

### `memory_forget`
Purge stale documents by date, with mandatory keyword protection. Params:
exactly one of `before` (ISO date `YYYY-MM-DD`) or `older_than_days` (int);
`protect` (array of regex patterns — filename OR content match protects a
doc from deletion); `sections` (array of strings, optional filter);
`apply` (bool, default `false` — dry-run report only); `confirm_unprotected`
(bool, default `false` — required to set `apply=true` with **no** `protect`
patterns, an explicit "yes, I really mean delete everything in range" flag).
Docs with no recognised header date are never deleted, unconditionally.

## The CLI

Console scripts (`[project.scripts]` in `pyproject.toml`):

- **`memory`** — unified subcommand CLI for the separate, LLM-free, CPU-only
  corpus pipeline (distinct from the LightRAG/MCP pipeline above):
  - `memory build --paths <dirs...> --index-dir <dir> [--chunk-size N] [--chunk-overlap N] [--force]`
  - `memory query --index-dir <dir> --question "..." [--top-k N] [--mode sparse|dense|fusion]`
  - `memory eval --index-dir <dir> --queries <yaml> --report <path> [--mode ...] [--top-k N]`
  - `memory regress --baseline <report.json> --candidate <report.json> [--report <path>]` — CI regression gate, exits 1 on regression.
  - `memory status --index-dir <dir>` — prints the corpus manifest summary.
- **`memory-index`** — the LightRAG/graph pipeline's reindex entrypoint:
  `memory-index --paths .plans docs [--full] [--refresh-changed] [--dry-run]`.
  This is what `memory_consolidate` shells out to as a subprocess.
- **`memory-mcp`** / **`hars-longterm-memory-mcp`** — start the MCP server
  (stdio transport; no CLI flags, all config via env vars above). Both point
  at the same server; `hars-longterm-memory-mcp` is the more explicit name.
- **`memory-eval-battle`** — generated retrieval battle test:
  `memory-eval-battle --paths <dirs...> --cases 100 --context-only --report <path>`.
  **Pass `--paths` as absolute paths** — relative paths hit the known parked
  bug below.

## Credentials

Consumers authenticate to the registry index with a **read-only deploy
token**, obtained from the GitLab project's Settings → Repository → Deploy
tokens (scope: `read_package_registry`). Configure it as environment
variables matching the named index above:

```bash
export UV_INDEX_HARS_LONGTERM_MEMORY_USERNAME=<deploy-token-username>
export UV_INDEX_HARS_LONGTERM_MEMORY_PASSWORD=<deploy-token>
```

(uv derives the env var name from the index's `name =` in `[[tool.uv.index]]`,
uppercased with non-alphanumerics turned to `_` — `hars-longterm-memory` →
`HARS_LONGTERM_MEMORY`.) Put these in a **gitignored** secrets file on your
own machine (e.g. `.env.secrets`, sourced by your shell profile or CI
secrets store) — never commit them. `explicit = true` on both
`[[tool.uv.index]]` blocks in the snippet above protects every other
dependency in your project from accidentally resolving off this private
index (or off the CPU-only torch index) — only packages that explicitly name
that index via `[tool.uv.sources]` are ever fetched from it.

## Known limitations

- **`battle.py` parked path bug**: `hars_memory/eval/battle.py` resolves a
  monorepo-depth-assumption `_PROJECT_ROOT` from `__file__` that does not
  correspond to anything once genuinely `pip`/`uv`-installed. It only
  affects `memory-eval-battle` invoked with **relative** `--paths` — pass
  absolute paths as a workaround. Not load-bearing for the MCP server or any
  other CLI path. Tracked for a v0.1.4 fix; not fixed as of v0.1.3.
- **GPU-guard and entity-schema are optional, consumer-supplied hooks**:
  - No `HARS_MEMORY_GPU_GUARD_SCRIPT_PATH` set → `memory_consolidate` and
    `memory-index` perform **zero** GPU-concurrency checking and always
    proceed. If you're sharing a GPU with training/inference workloads and
    want indexing to defer to them, you must write and point at your own
    `assert_gpu_free(api_base_url)` script.
  - No `HARS_MEMORY_ENTITY_SCHEMA_PATH` set → the package's own generic
    default schema is used automatically (7 entity types, 5 relation types —
    see `hars_memory/schema/default_schema.yaml` in the source repo). Set
    this only if you need domain-specific entity typing.
