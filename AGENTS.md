# AGENTS.md — hars-longterm-memory

Onboarding doc for any AI coding agent picking this repo up cold. If you have
never seen this repo before, read this file in full before touching code.

## What this is

`hars-longterm-memory` is a **standalone, reusable Python package**: a
LightRAG (graph+vector) knowledge-base service exposed as both an MCP server
and a CLI. It was extracted out of the `cortex` monorepo's `tools/memory/` on
2026-08-25 (history preserved via `git filter-repo`) and is now published as
an independent package to a private GitLab Package Registry
(`code.knotgate.com`, project id `14`, path `hars/hars-longterm-memory`).

**What this repo is NOT**: it is not Cortex-specific. It carries no opinion
about which GPU-guard mechanism, entity-typing schema, or downstream export
pipeline a consumer needs — those are consumer-supplied hook points (see
"Consumer hook points" below). No cortex-specific code should be added here.
`cortex-scripts/` is the one deliberate exception: it holds Cortex's own
operational wrapper scripts, explicitly **not** part of the installed
package (excluded from the wheel — see `[tool.hatch.build.targets.wheel]` in
`pyproject.toml`, which lists only `packages = ["hars_memory"]`). Any other
consuming project owns its own equivalent of these scripts; this repo does
not ship them.

## Architecture — module map

```
hars_memory/
  cli.py             — console-script entrypoints (memory-index, memory-mcp,
                        hars-longterm-memory-mcp, memory-eval-battle, memory)
  mcp_server.py       — the MCP server: 7 tools (memory_recall/remember/
                        entities/related/status/consolidate/forget), ~2600
                        lines, the primary read surface for an AI agent
  ingest/             — walker (.memoryignore-aware glob), chunker, change
                        detection (content-fingerprint sidecar), and the
                        public ingest_documents() API a consumer calls
                        directly for non-filesystem sources (e.g. Postgres);
                        migrate.py backs `memory migrate-index` (zero-LLM
                        back-fill of chunk location metadata + opt-in dedupe)
                        batch.py backs `memory index-batch` (extraction through a
                        provider Batch API by priming LightRAG's LLM cache; phases
                        collect/submit/status/apply/cancel, state in <index>/batch_state.json;
                        stall detection + cancel + abandon -> sync fallback in wait())
  server/             — index.py (GPU-guarded CLI reindex entrypoint),
                        lightrag_init.py (wires LightRAG to the configured
                        embedder/extractor/query LLM/vector+graph storage),
                        embedder.py, reranker.py, legacy_env_guard.py,
                        logging_setup.py
  retrieval/          — BM25 sparse index, flat-dense fallback index,
                        ripgrep live-worktree channel, dense/sparse fusion,
                        supersession-aware rescoring — all additive channels
                        layered onto LightRAG's own retrieval, surfaced via
                        memory_recall's `hybrid` response field
  corpus/              — a second, LLM-free, CPU-only build/query pipeline
                        (`memory build|query|status` subcommands) — separate
                        from the LightRAG graph pipeline, no LLM calls at all
  schema/              — entity/relation type schema loader; ships its own
                        generic `default_schema.yaml`, overridable via
                        HARS_MEMORY_ENTITY_SCHEMA_PATH
  eval/                — gold-question harness (check.py), generated
                        retrieval battle test (battle.py — has the one known
                        parked bug, see below), regression gate (regression.py,
                        `memory regress`), retrieval metrics (metrics.py)
  service/             — tenant-scoped HTTP index-job API, durable SQL jobs,
                        artifact stores, index engines, and lease worker
  sdk.py               — synchronous HTTP client for create/extend/poll/download
  strategies.py        — validated index/search strategies and YAML matrices
  scripts/              — cleanup_kb.py (index-only maintenance),
                        qdrant_transplant.py (one-off migration tool)
cortex-scripts/         — Cortex's own consumer scripts (update_kb.sh,
                        start_query_llm.sh) — NOT part of the installed
                        package; kept in-repo only as a working reference
config/.env.example     — full env-var reference with inline rationale
.gitlab-ci.yml          — Ruff, pytest, package artifacts, container publishing
Dockerfile.service      — production HTTP index-service image
tests/                  — 583 passing tests plus 6 opt-in/environment skips, pytest
```

`memory_recall` defaults to its existing full response. Set `HARS_MEMORY_RECALL_VIEW=lean` or pass `view="lean"` for top_k fused chunks, compact ripgrep hits, and metadata without the LightRAG context blob; `debug=true` always returns full detail.

## Chunking mode (`HARS_MEMORY_CHUNKER`)

`token` (default) uses LightRAG's own token splitter. `markdown` plugs
`ingest/chunker.py::lightrag_chunking_func` into `LightRAG(chunking_func=...)`
(wired in `server/lightrag_init.py`): H1-H6 sections with full breadcrumbs,
atomic tables/fences/lists/blockquotes, oversized ones split by row/line
(table header repeated, fences re-balanced); non-Markdown files fall back to
the token splitter. Chunks also carry `heading_path`/`section`/`start_line`/
`end_line` (+ `part` for split pieces) in `kv_store_text_chunks.json`.
Undersized chunks (< `HARS_MEMORY_CHUNK_MIN_TOKENS`, default 200, 0 = off) are merged
into a neighbour of the same top-level section (heading below a sole H1), never across
an oversized-block cut and never past `HARS_MEMORY_CHUNK_TOKEN_SIZE`; the record keeps
each part in `merged_from` (`heading_path`, `start_line`, `end_line`). The walker's
`[Document: ...]` header is no longer emitted as a standalone chunk.
**Baseline parity** (measured read-only on the live index: 5533 gleaning prompts for
5534 chunk ids in `kv_store_llm_response_cache.json`, 1726 of 3111 neighbouring chunks
share exactly 64 tokens, max chunk 512 tokens): the baseline is token chunker, 512/64,
`HARS_MEMORY_MAX_GLEANING=1` -- identical to the code defaults, so v2 only changes the
chunker (markdown mode applies no overlap) unless env overrides.
**Chunk id = md5(chunk text)**: turning `markdown` on for an existing index
changes the ids of re-chunked docs, so those docs need re-extraction (LLM
re-index); do not flip it on a live index without that plan.

## Building / comparing the markdown "v2" index (Batch API)

```bash
export HARS_MEMORY_SOURCES_MANIFEST=/path/to/knowledge-sources.yaml
scripts/build_index_md_v2.sh estimate        # no network, no cost: chunk + price round 1
MAX_COST=20 scripts/build_index_md_v2.sh run # PAID. Resumable: re-run the same command
scripts/build_index_md_v2.sh cancel          # give up on unfinished batches (manual)
scripts/build_index_md_v2.sh status          # REFRESH=1 to poll the provider
```
The script pins the build config (chunker=markdown, 512 tokens, min 200, gleaning 1,
`openai/gpt-6-luna`, temperature 1.0), reads `HARS_MEMORY_LLM_API_KEY` from
`$HARS_ENV_FILE` (default `config/live-llm-e2e.env`) and refuses the live index. Target
default: `~/.local/share/hars-longterm-memory/index-<VARIANT>` (`VARIANT=md|md-merge`, default `md-merge`; `INDEX_DIR=` to change).
`memory index-batch` itself also refuses `~/.local/share/hars-longterm-memory/index`,
`$HARS_MEMORY_LIVE_INDEX_DIR` and any dir holding an index it did not create.
`MAX_COST` is cumulative over both rounds (gleaning is a second batch); set
`MAX_INFLIGHT_TOKENS` if the provider has an enqueued-token quota. Evals (no LLM):
`docs/index-v2-eval.md`.

**Stragglers (batches stuck at N-1 of N).** `run` tracks `completed+failed` per part
(`last_progress_at` in `batch_state.json`). No growth for `--stall-minutes` (default 20; env
`HARS_MEMORY_BATCH_STALL_MINUTES`, script `STALL_MINUTES`; 0 disables) while at least
`--stall-min-done-fraction` (0.5) is done => POST cancel. Then wait up to `--cancel-wait-minutes`
(default 15, script `CANCEL_WAIT_MINUTES`): `cancelled` + output file => partial answers are applied;
anything else (still `cancelling`, or cancelled without output) => part marked `abandoned`.
Every request without an answer then goes through LightRAG's normal synchronous LLM call
(logged: count, estimated cost at 2x batch price). `apply`/`run` check batch spend + that
fallback estimate against `--max-cost` first and exit 3 without indexing if it would exceed it.
`memory index-batch cancel --index-dir DIR` cancels all unfinished parts manually (the next `run` handles
them the same way). `status` shows per part `done=/total`, last-progress age and
`STALLED`/`CANCELLING`/`CANCELLED`/`ABANDONED`. Exit codes of `run`: 0 applied, 1 error, 3 cost guard,
**4 = TIMEOUT** (`--timeout-minutes`, script default 600; nothing applied, re-run to resume).

## Consumer hook points (this package ships no defaults for these)

- **GPU guard** (`HARS_MEMORY_GPU_GUARD_SCRIPT_PATH`): optional. Unset (the
  default) means `memory_consolidate` and `server/index.py` perform **no**
  GPU-concurrency check at all and proceed unconditionally. If set, it must
  point at a Python file exposing `assert_gpu_free(api_base_url)`, loaded
  dynamically via `importlib.util.spec_from_file_location` (not a normal
  package import — the guard script is expected to live in a separate,
  consumer-owned project with its own venv). Cortex supplies its own at
  `tools/memory-config/scripts/gpu_guard.py`.
- **Entity schema** (`HARS_MEMORY_ENTITY_SCHEMA_PATH`): optional. This
  package DOES ship a generic default at `hars_memory/schema/default_schema.yaml`
  (entity types: Concept/Document/Decision/Event/Person/Tool/Task; relation
  types: relates_to/produces/depends_on/supersedes/documented_in) and uses it
  automatically if the env var is unset — a consumer only needs to set this
  if it wants domain-specific entity typing.
- **Legacy env-prefix guard** (`HARS_MEMORY_LEGACY_ENV_PREFIXES`): optional,
  no-op when unset. A consumer that renamed its own env vars in the past can
  list forbidden prefixes (comma-separated) here to fail closed at startup
  instead of silently reading stale values.
- **Recall reranking** (`HARS_MEMORY_RERANK_BACKEND`): **OFF by default**
  (2026-10-17 product decision). Unset, `off` and `none` make **no reranker
  call at all** — the cross-encoder (Qwen3-Reranker-0.6B on the iGPU) measured
  7.4–14 s/query (97% of end-to-end latency) for +6.3 pts recall@1 on
  saturated scores and is being replaced by a learned channel-fusion ranker.
  `local` (LightRAG's `rerank_model_func`) and `http` (the external
  cross-encoder) remain explicit opt-ins; both, plus `off`, also disable
  LightRAG's native rerank pass for the recall call. Every hybrid block
  includes `rerank: {backend, applied, latency_ms, fallback_reason, pool}`;
  the default path reports `backend:"off", fallback_reason:"disabled_by_default"`.
  Opt-in http options: `HARS_MEMORY_RERANK_HTTP_URL` (default
  `http://127.0.0.1:8081/v1/rerank`; a host or `/v1` base URL is accepted),
  `HARS_MEMORY_RERANK_HTTP_MODEL` (default `x`), `HARS_MEMORY_RERANK_POOL`
  (default 8 candidates; the remaining fused chunks retain their order),
  `HARS_MEMORY_RERANK_TIMEOUT_S` (default 4 s hard budget — on expiry the
  fusion order is kept, `fallback_reason:"timeout"`), `HARS_MEMORY_RERANK_HTTP_DOC_CHARS`
  (per-document cap sent to the reranker, default 2000), and the two guards
  `HARS_MEMORY_RERANK_SKIP_FUSION_SCORE` (skip when the top fused score is
  already confident, default 0.9) / `HARS_MEMORY_RERANK_FUSION_MARGIN`
  (0.15 — the reranker may only swap pairs inside this fusion-score margin).
  Failed/timed-out HTTP reranks keep the original order.

### Multilingual retrieval routing (opt-in)

Caveat: the query twin is produced with `rag.llm_model_func`, i.e. the LightRAG extractor LLM (`HARS_MEMORY_LLM_*` extractor model), not the query model; point the extractor model at a fast cheap model or accept its latency (measured 2.2 s median / $0.00006 per Russian query with gpt-6-luna). Evidence: docs/multilingual-retrieval-2026-10-02.md.

Unset or `HARS_MEMORY_ROUTING=off` leaves recall output unchanged and omits
`routing`. `lang` detects Russian by Cyrillic-letter share / word count and
selects the language profile.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `HARS_MEMORY_ROUTING` | `off` | `off` or `lang` |
| `HARS_MEMORY_ROUTING_PROFILE_EN` | `3way` | `base` or `3way` |
| `HARS_MEMORY_ROUTING_PROFILE_RU` | `3way_twin` | `base`, `3way`, or `3way_twin` |
| `HARS_MEMORY_ROUTING_WEIGHTS` | `0.2,0.4,0.4` | Gemma dense, Qwen sidecar, BM25 weights |
| `HARS_MEMORY_ROUTING_SIDECAR_DIR` | newest complete sidecar | Optional explicit sidecar directory |
| `HARS_MEMORY_ROUTING_SIDECAR_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Local query embedding model |
| `HARS_MEMORY_ROUTING_TWIN_TIMEOUT_S` | `6` | Query-twin timeout in seconds |
| `HARS_MEMORY_ROUTING_MIN_COVERAGE` | `0.95` | Minimum current-chunk sidecar coverage |

Three-way fusion combines existing Gemma dense, local Qwen dense sidecar, and
BM25. The twin profile asks the configured query LLM for an English twin and
appends it to all three channel queries. Build a sidecar with
`uv run memory build-dense-sidecar --index-dir DIR`. Rebuild/re-run after
ingesting new chunks: stale coverage falls back to base. Missing/unloadable
sidecar, model/dimension errors, and Qwen failures also fall back to base;
twin timeout/error uses the original question. See
`docs/multilingual-retrieval-2026-10-02.md` for experiment evidence and
trade-offs.

## Build / test locally

```bash
uv sync --extra dev     # NOT `uv sync --group dev` — dev is declared under
                         # [project.optional-dependencies], not a dependency-group
uv run pytest -q        # 583 passed, 6 skipped as of v0.2.0
```

`.python-version` pins Python 3.13. `uv.lock` is the pinned, reproducible
lock file — commit it alongside any `pyproject.toml` dependency change.

## Publishing a new version

There is no CI pipeline in this repo (no `.gitlab-ci.yml`) — publishing is a
manual, local operation:

1. Bump `version` in `pyproject.toml` (and `uv.lock` will pick it up on the
   next `uv sync`/`uv lock`).
2. Run the test suite; it must pass.
3. Build: `uv build` (produces `dist/hars_longterm_memory-<version>-py3-none-any.whl`
   and the matching `.tar.gz` sdist).
4. Publish via HTTP PUT to the GitLab Package Registry:
   ```bash
   uv publish --publish-url https://code.knotgate.com/api/v4/projects/14/packages/pypi \
       --username __token__ --password "$GITLAB_TOKEN_WITH_write_package_registry_SCOPE"
   ```
   (or `twine upload --repository-url ... dist/*` with the same credentials —
   either works against a GitLab PyPI-compatible registry endpoint). The
   token needs the `write_package_registry` scope; get a real one from that
   machine's own gitignored secrets file (e.g. cortex's `.env` /
   `.env.secrets` — never hardcode one here).
5. **Never trust the publish command's own success message.** Re-query the
   registry directly to confirm the new version landed:
   ```bash
   curl -s "https://code.knotgate.com/api/v4/projects/14/packages" \
       -H "PRIVATE-TOKEN: $TOKEN" | python3 -c "import json,sys; [print(p['name'], p['version']) for p in json.load(sys.stdin)]"
   ```

## Credential model

- **Consumers** (installing/resolving the package) authenticate to the
  registry index with a **deploy token**, supplied as
  `UV_INDEX_<NAME>_USERNAME` / `UV_INDEX_<NAME>_PASSWORD` env vars (uv's
  convention for a named `[[tool.uv.index]]` — `<NAME>` is that index's
  uppercased `name`). Store these in a gitignored secrets file on the
  consumer's own machine, never committed.
- **Publishers** need a personal/project access token with the
  `write_package_registry` scope — same rule: kept in a gitignored secrets
  file, never in this repo, never in a commit message, never in CI config
  checked into git.

## Known parked bug (do not fix casually)

`hars_memory/eval/battle.py` line 23:
```python
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
```
This assumes a monorepo-style directory depth (`tools/memory/eval/battle.py`
→ 3 parents up = project root) that does not exist once this package is
genuinely `pip`/`uv`-installed into `site-packages` — a relative `--paths`
argument to `memory-eval-battle` / `memory_eval_battle_main` will resolve
against a nonsense path. It is **not load-bearing** for the MCP server or the
core CLI paths (`memory_recall`, `memory_consolidate`, `memory build/query`,
etc. never touch it) — it only affects `memory-eval-battle` invoked with
relative `--paths`. Workaround today: always pass absolute `--paths` to
`memory-eval-battle`. Flagged here as a known issue for a future v0.1.4, not
fixed as part of this doc pass.

Note: a near-identical instance of this exact pattern in
`hars_memory/server/index.py` **was** fixed in v0.1.3 (commit `8f5016f6`) —
relative `--paths` there now resolve against `Path.cwd()` at invocation time,
which is what `memory_consolidate`'s subprocess spawn (no `cwd=` override)
and `cortex-scripts/update_kb.sh`'s relative-extra-paths convention both rely
on. `battle.py` was not part of that fix wave.

Also worth knowing (found but out of scope to fix here): `hars_memory/cli.py`'s
`_project_root()` / `_ensure_project_root()` (used by `index_main`,
`eval_battle_main`, and the unified `memory` command) still computes
`Path(__file__).resolve().parents[3]` and inserts it into `sys.path`. In
practice this looks harmless once the package is installed (the actual
`from hars_memory.X import Y` imports resolve via `site-packages` regardless
of what garbage this inserts at `sys.path[0]`), but it is the same
anti-pattern and worth cleaning up alongside `battle.py` in v0.1.4.

## torch is CPU-only by design

Confirmed in `pyproject.toml`:
```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```
`[tool.uv.sources]` does **not** propagate to a consumer through a built
wheel — a project that depends on `hars-longterm-memory` from the registry
must repeat this override itself (see the skill doc for the exact snippet),
or `torch` will resolve to the full CUDA build from PyPI.

## Deeper conventions

- `CLAUDE.md` — Claude-Code-specific operating notes for this repo.
- `.session/` — one file per significant work session (see `.session/README.md`).
- `.plans/` — forward-looking implementation plans for non-trivial work (see
  `.plans/README.md`).
- `.claude/skills/hars-longterm-memory/SKILL.md` — the portable skill meant
  to be copied into any OTHER project's `.claude/skills/` so an agent there
  can install/configure/use this package without ever having seen this repo.
