# Live LLM E2E and configurable strategy benchmarks

Status: complete (prepared and offline-verified; live model run intentionally pending)

## Goal

Prepare an opt-in, bounded real-model E2E journey and reusable abstractions for
comparing indexing and retrieval strategies across repeatable runs.

## Contract

- Index jobs accept a validated, named strategy snapshot; arbitrary environment
  variables, paths, URLs, and secrets are not accepted from API clients.
- Every published artifact records the exact strategy and a stable hash.
- Search strategies are immutable value objects shared by benchmark runners,
  reports, and live tests.
- A YAML matrix expands named index x search combinations, validates backend
  compatibility, and produces a report with corpus/query hashes, per-run
  timings, quality metrics, and aggregate statistics.
- Matrix evaluation normalizes portable gold basenames/suffixes against the
  absolute or URI paths stored by per-run indexes; legacy A/B runs remain
  exact-string by default.
- Corpus matrices run without LLM/GPU. LightRAG matrices use the existing real
  indexer and retrieval-only `ab_bench` in isolated subprocesses.
- The real-model E2E is opt-in, skipped by default, checks endpoint readiness,
  indexes a tiny deterministic corpus through SDK/API, downloads immutable v1
  and v2 artifacts, and reads them through real MCP retrieval and synthesis.

## Verification

1. Unit tests for strict strategy parsing, hashing, API/SDK transport, engine
   application, matrix expansion, aggregation, and invalid combinations.
2. Offline strategy-matrix E2E using two corpus indexing strategies and sparse
   search; no model/network/GPU.
3. Existing CLI/MCP/HTTP E2E remains green.
4. Live test collects diagnostics but is only runnable with
   `HARS_RUN_LIVE_LLM_E2E=1`.
5. Full suite, Ruff, and `git diff --check`.
