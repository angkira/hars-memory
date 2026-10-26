# Plan: opt-in language routing + Qwen3 dense sidecar + query twin (2026-10-02)
Evidence: docs/multilingual-retrieval-2026-10-02.md (experiments/search-2026-10/ml/).
Feature (all opt-in, defaults unchanged): HARS_MEMORY_ROUTING=off|lang.
- detect query language (Cyrillic ratio, no model); en -> 3-way fusion gemma+qwen3+bm25 (weights .2/.4/.4); ru/mixed -> LLM English twin concatenated to the query, then the same 3-way fusion.
- qwen3 dense lives in a sidecar dir inside the index dir built by `memory build-dense-sidecar` (never touches vdb_chunks.json); in-process sentence-transformers.
- graceful fallbacks: sidecar/model missing or stale -> current hybrid; twin LLM error/timeout -> original query. Reranker stays the separate existing opt-in stage.
- response field `routing`; tests with fakes; AGENTS.md + config/.env.example docs.
