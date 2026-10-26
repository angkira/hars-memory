# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- **Native HTTP(S) MCP endpoint with OAuth 2.1** (`hars-longterm-memory-mcp-https`,
  `hars_memory/http_server.py`, `hars_memory/mcp_oauth.py`). One process holds
  one LightRAG index shared by every client — the `stdio + supergateway
  --stateful` bridge spawned one child per MCP session, so each new client paid
  a ~24 s cold start. Includes RFC 9728/RFC 8414 discovery, RFC 7591 dynamic
  client registration, authorization-code + PKCE S256, refresh-token rotation,
  revocation, audience-bound Ed25519 access tokens, hierarchical scopes
  (`memory:read`/`write`/`admin`) enforced at `tools/list` and `tools/call`,
  optional TLS with mutual-TLS support, and `scripts/mcp_http_smoke.py`.
  See `docs/http-mcp.md`.

## [0.8.1] - 2026-10-21

### Fixed
- **Retrieval improvements**: source-diverse window selection on full pool, chunk-local ripgrep scoring, graph-bridge provenance seeding and degree damping.

### Added
- **Visualizer**: query-driven live presentation deck without hardcoded fixture traces.

### Changed

- **Retrieval defaults retuned from measurements, not guesses.** Every query
  now improves; no gold set regresses:
  - `top_k` 6 → **20** (it is a *chunk* window; at 7.65 chunks/doc the old
    default left the answering document outside the window 87% of the time),
  - fusion `alpha` 0.5 → **0.15** (0.5 was an untuned midpoint; BM25 holds the
    gold in its top-3 for 45/48 while dense misses 7/48),
  - fetch width ×2, `MAX_CHUNKS_PER_SOURCE` 3 → 2 (applied before slicing).
  - 48q: recall@1 0.7083 → **0.8333**, recall@10 0.9792 → **1.0000**,
    NDCG@10 0.8563 → **0.9196**, MRR 0.8149 → **0.8932**.
  - 28q: recall@1 0.2857 → **0.3393**, recall@10 0.6964 → **0.7679**,
    NDCG@10 0.6144 → **0.6988**, MRR 0.6845 → **0.7562**.
  - hard_1k (300 adversarial cases): recall@1 0.3561 → **0.3600**,
    recall@10 0.6506 → **0.6772**, NDCG 0.6119 → **0.6336**, MRR 0.7965 → **0.8096**.

### Added

- **Graph/citation bridge channel (`G_chunks`, opt-in).** Hub-damped 1-hop
  expansion seeded from the question plus the top-3 anchor chunks (entity
  names, `.md` citations, rare UUIDs, run ids), provenance-scored and merged
  before diversity. Measured on hard_1k: answer-in-window 0.4200 → **0.4500**
  (+3.0 pts, CI [+0.67, +5.67]); neutral on the 48q/28q gold sets. Off by
  default (`HARS_MEMORY_GRAPH_BRIDGE=1` enables), ~16 ms mean latency.
- **OpenRouter inline Batch API backend** for `memory index-batch`
  (`HARS_MEMORY_BATCH_PROVIDER=openrouter`): inline `requests` submit + poll,
  `--append` to extend a restored index, identity guard on `apply`.
- **`hard_1k` adversarial retrieval set generator** (deterministic, $0):
  1000 multi-hop/relational/temporal/negation/numeric/lexical-mismatch/
  disambiguation/implicit-reference cases. The generated data is not shipped.
- Fine-tuning scaffolding and runbooks for a 9B extraction student.

### Fixed

- Batch `apply` now refuses to run when the cache identity (model/chunker/
  gleaning) differs from `collect`, which would silently miss every primed key
  and re-extract synchronously.

## [0.7.1] - 2026-10-18

### Security

- **Sanitized the published artifacts.** The 0.7.0 wheel and sdist carried
  confidential deployment data: internal experiment identifiers in
  docstrings/example strings and package-data eval gold sets (real research
  questions, internal run ids, internal corpus filenames), plus internal host
  addresses and operator home paths in sdist-shipped operator scripts and ops
  docs. The 0.7.0 release has been **deleted from the GitLab PyPI registry**
  (project 14) — do not use it; pin `>=0.7.1`. Changes:
  - Example identifiers in docstrings/tool descriptions are now generic
    placeholders (`EXAMPLE-ID-1`, `demo-alignment-task`-style names, generic
    model codenames) across `mcp_server.py`, `retrieval/*` and the docs.
  - Eval gold sets no longer ship as package data: the old 46-query corpus
    gold set is removed from the repo, and the in-package
    `eval/retrieval_queries.yaml` / `eval/gold_questions.yaml` are now small
    SYNTHETIC schema samples. Real gold sets are operator-local (gitignored
    `eval_agentic-data/gold_sets/`), passed via the existing CLI arguments.
  - The sdist no longer ships operator/local files: `cortex-scripts/`,
    `AGENTS.md`, `CLAUDE.md`, `.claude/`, `.junie/`, `docs/OPERATIONS_HANDOFF.md`,
    `docs/SWAP_CHECKLIST.md`, `docs/VERSION_PIN.md`, and one-off verification
    notes. All stay in the repository.
  - Internal host addresses and `/home/<user>` paths in shipped code, tests,
    artifacts and docs are replaced by generic defaults (`127.0.0.1`,
    `/home/demo`, generic index directory names); recorded provenance JSON in
    `ranking/artifacts/` is scrubbed the same way (feature data untouched —
    only the recorded `index_dir` provenance strings changed).
  - Deployment-specific duplicate-entity alias spellings were removed from
    `mcp_server.py`; operators now register their own via the new
    `HARS_MEMORY_EXTRA_ENTITY_ALIASES` env var (YAML file mapping canonical
    name -> alias list, merged into `ENTITY_ALIASES` at startup; missing or
    invalid files are ignored with a warning and can never break startup).
  - New regression guard `tests/test_package_hygiene.py` scans the shipped
    package tree for the leaked identifier/host patterns.

## [0.7.0] - 2026-10-17

### Added

- Learned channel-fusion ranker (learning-to-rank) for the recall path,
  replacing the hand-tuned additive blend with a LightGBM LambdaRank model over
  20 deterministic rank-time features (dense/BM25/ripgrep scores + norms, fused
  score/gap/rank, channel count, identifier/term overlap, chunk/doc length,
  doc recency, section depth, code-heaviness, supersession flag, graph
  entity fan-in/match). New package `hars_memory/ranking/` (features, artifact
  loader with a feature-manifest guard, fail-soft `apply_ltr_ranker`, training
  lib) plus `hars_memory/eval/ltr_{dataset,train,eval}.py` and a scaled
  synthetic-query generator. Inference ≈0.20 ms/candidate on CPU, no GPU/LLM.
  Offline gate on the held-out 48-query GC-fixed and 28-query relational sets:
  recall@1 0.7083→0.7708 / 0.2857→0.3214, NDCG@10 0.8563→0.8848 / 0.6144→0.6288,
  MRR 0.8149→0.8479 / 0.6845→0.7055 (all ≥ baseline). Artifact
  `hars_memory/ranking/artifacts/ranker.json` (+ manifest, committed). New
  optional extra `ltr` (`uv sync --extra ltr`) provides LightGBM for
  training and the LightGBM artifact backend; without it the loader fails soft
  to the hand-tuned fusion. Additive integration point in
  `retrieval/fusion.py` (`apply_ltr_ranker`, `ltr_ranker_enabled`) — no
  call-site change is made in `mcp_server.py`; see `docs/ltr-ranker.md` for the
  exact wiring spec.

## [0.6.0] - 2026-10-14

### Added

- Remote/shared embedder mode: `HARS_MEMORY_EMBED_PROVIDER=http` sends texts to
  an OpenAI-compatible `/v1/embeddings` endpoint instead of loading
  sentence-transformers in-process (~300–600 MB RAM saved by sharing one
  embedder, e.g. a llama.cpp `--embeddings` server or a hosted API). Query
  prompts / doc prefixes are applied client-side for vector parity;
  batching (`HARS_MEMORY_EMBED_BATCH_SIZE`,
  `HARS_MEMORY_EMBED_HTTP_MAX_INPUTS_PER_REQUEST` with automatic halving on
  size-limit 400s), 429 `Retry-After` + 5xx retries, optional pacing
  (`HARS_MEMORY_EMBED_HTTP_MIN_INTERVAL_MS`), Bearer auth
  (`HARS_MEMORY_EMBED_API_KEY`), dimension validation against the index, and
  token-usage cost logging. Default (no env vars) behaviour is unchanged.
  See README "Shared & hosted remote embedders".
- Remote/shared reranker mode: `HARS_MEMORY_RERANK_PROVIDER=http` sends rerank
  calls to a Cohere-style `POST {BASE_URL}/rerank` endpoint (verified against
  llama.cpp `/v1/rerank`; same code-diver endpoint the MCP fusion rerank uses)
  instead of loading a cross-encoder in-process. Chunked requests, client-side
  `top_n` after merge, 429 `Retry-After` + 5xx retries, optional pacing, Bearer
  auth, run-summary logs. `auto` (default) selects http when
  `HARS_MEMORY_RERANK_BASE_URL` is set, local when `HARS_MEMORY_RERANK_MODEL`
  is set, disabled otherwise. See README "Shared & hosted remote rerankers".
- Structure-aware Markdown chunking: `HARS_MEMORY_CHUNKER_MODE=md_struct`
  (or `index --chunker-mode md_struct`, new `ingest/md_structure.py`) chunks
  notes on section boundaries with breadcrumb prefixes, frontmatter parsing,
  fence-aware headings, table row-blocks and per-note manifests for future
  section routing. Default `legacy` is byte-identical to before. Adopting
  `md_struct` changes chunk texts/ids and needs a full re-embed, not an
  incremental update.
- Extractor benchmark tool: `memory extractor-bench run/score/report` over
  fixed fixtures (`eval/fixtures/extractor_bench_v1/`) with a thinking-leak
  guard, for comparing extractor LLMs.
- `memory_recall` accepts an optional `max_chars` argument sizing the returned
  context; context assembly caps each document block
  (`HARS_MEMORY_CONTEXT_MAX_DOC_CHARS`, default 6000, truncated at line
  boundaries) and admits whole documents within the total budget
  (`HARS_MEMORY_CONTEXT_MAX_CHARS`) — no more mid-buffer amputations.
- Rerank quality guards in the MCP fusion path: per-request document truncation
  (`HARS_MEMORY_RERANK_HTTP_DOC_CHARS`, default 2000), margin-guarded blending
  (`HARS_MEMORY_RERANK_FUSION_MARGIN`, default 0.15 — a saturated reranker can
  no longer promote a worst-in-pool doc past a confident fusion top-1), and
  rerank-skip when fusion is already confident
  (`HARS_MEMORY_RERANK_SKIP_FUSION_SCORE`, default 0.9).

### Fixed

- `memory recall` no longer crashes with `TypeError` when
  `hybrid.identifier_matches` is `None` (treat `None` as empty).
- Bridge servability: supergateway now runs `--stateful` (one stdio child per
  bridge instead of one per request — removes the 13–15 s per-call LightRAG
  re-init; see `cortex/scripts/hars-longterm-memory-mcp-http.sh`).

## [0.5.0] - 2026-10-13

### Added

- Incremental refresh of a LightRAG index: `python -m hars_memory.scripts.incremental_refresh`
  with `plan`, `apply`, `audit` and `repair-orphans` subcommands, see
  [docs/incremental-refresh.md](docs/incremental-refresh.md).
  - `plan` is read-only and classifies every walked document against the fingerprint
    sidecar and the content LightRAG actually stored (new, changed, retry, adopt,
    unchanged, gone, kept).
  - `apply` deletes many documents in one batched pass (entity/relation rebuild and
    storage flushes run once for the union) and inserts in verified batches with a
    fingerprint checkpoint after each batch. An unchanged index is a strict no-op.
  - `audit` cross-checks the KV stores, the graph and Qdrant; `repair-orphans` removes
    Qdrant points left behind by a killed run.
  - `HARS_MEMORY_ID_PATH_MAP=ACTUAL=LOGICAL` lets a refresh run from another checkout
    location reproduce the document ids stored in an existing index.
- Config-driven extractor provider: `HARS_MEMORY_EXTRACTOR_PROVIDER=local|openrouter`
  with `HARS_MEMORY_OPENROUTER_MODEL`, `HARS_MEMORY_OPENROUTER_BASE_URL`,
  `HARS_MEMORY_EXTRACTOR_API_KEY_ENV`, `HARS_MEMORY_EXTRACTOR_REASONING_EFFORT` and
  `HARS_MEMORY_EXTRACTOR_EXTRA_BODY_JSON`. The default `local` provider is unchanged.
  Credit: Angkira.
- Optional dotenv-style key file `HARS_MEMORY_EXTRACTOR_API_KEY_FILE`, consulted only
  when the key variable is unset. Credit: Angkira.
- CHANGELOG.md and a configuration reference section for the extractor provider.

### Changed

- `memory_consolidate` and `cortex-scripts/update_kb.sh` skip the GPU guard (and do not
  start a local extractor) when the extractor provider is hosted. An empty
  `HARS_MEMORY_EXTRACTOR_PROVIDER` is treated as `local`, so the guard still runs.
  Credit: Angkira.
- `cortex-scripts/update_kb.sh` archives a staged note only after its LightRAG
  `doc_status` is `processed` (`hars_memory.scripts.archive_ingested`). Failed notes stay
  in staging and the script exits non-zero instead of silently losing them.
  Credit: Angkira.

### Fixed

- `audit` tolerates LightRAG's `md5(src + tgt)` relationship point-id clashes: edges
  that share a point id are reported under `relationship_id_collisions` instead of
  failing the audit, and self-loop edges no longer crash the clash lookup.

## [0.4.0] - 2026-10-03

- Release commit `a69ae4b`; session, plan and experiment files are no longer tracked
  and are excluded from the sdist. No detailed notes were kept for earlier versions,
  see the git history.
