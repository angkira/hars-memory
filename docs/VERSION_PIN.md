# LightRAG version pin (`lightrag-hku`) — the index-cutover blocker

Status: **migrated and landed 2026-10-17** (channel-hygiene + index-cutover audit).
Applies to the serving venv (`/home/demo/Project/software/hars-longterm-memory/.venv`).

## Decision

**`lightrag-hku==1.5.7`** — the builder's exact version.

The deployment target is the **builder's version, 1.5.7**, and the serving venv
must be able to faithfully *read the store it serves*; the LightRAG
`kv_store_doc_status.json` schema is version-sensitive. The repo now imports
cleanly under 1.5.7 (see "Landed import migration"), so the pin is exact:
reader == builder, with no version drift possible.

The previous interim pin (`>=1.4.16,<1.5`) was a fail-safe while the repo still
imported two symbols that 1.5 removed/moved. That migration has now landed.

## Why this was a cutover blocker (measured, not inferred)

The serving venv must be able to read the store it serves. Even the *current*
live index was built by a newer LightRAG than 1.4.16 reads.

Two specimens, loaded with the **1.4.16** reader (`DocProcessingStatus(**row)`,
the exact call `JsonDocStatusStorage` makes):

| Specimen | Rows | Status values | 1.4.16-constructible | Failure |
| --- | ---: | --- | ---: | --- |
| `index_full_mdstruct_20261014` (new, 1.5.7 Colab build) | 1340 | `processed 958, failed 290, pending 17, **parsing 75**` | 290 / 1340 | `TypeError: unexpected keyword argument 'content_hash'` on **1050** rows |
| `index_compact` (live) | 3158 | `processed 1604, failed 1554` | 2769 / 3158 | `content_hash` on **389** rows |

1.5.x `DocProcessingStatus` gained `content_hash` (`lightrag/base.py`); 1.4.16
has no such field. 1.5.x `DocStatus` also added the enum values `parsing` and
`analyzing`; 1.4.16's `get_status_counts()` does `counts[doc["status"]] += 1`
over a dict keyed only by the old enum, so on the new store it raised:

```
KeyError: 'parsing'      # 75 parsing rows are unknown to 1.4.16
```

Consequences of serving the new store on 1.4.16:

- `get_status_counts()` / `memory_status` → **KeyError** (crash).
- `get_docs_by_status*()` / `get_by_id()` → the 1050 `content_hash` rows are
  **silently dropped** (per-row `except (KeyError, TypeError): continue`), so
  status and delete surfaces under-report and can miss documents.
- The **dense/sparse recall path is unaffected**: `vdb_chunks.json` (7442
  vectors) and `kv_store_text_chunks.json` load and query fine under 1.4.16
  (verified). Only the management/status surfaces break.

So a 1.4.16 reader can *answer queries* but cannot faithfully *manage* a
1.5.7-built store. Since the cutover swaps in exactly such a store, reader and
builder are now aligned at **1.5.7**.

## Landed import migration

Confirmed moved/removed between 1.4.16 and 1.5.7, and migrated in this commit:

| File:line | 1.4.16 import | 1.5.7 import now in use |
| --- | --- | --- |
| `hars_memory/ingest/chunker.py` (`lightrag_chunking_func`) | `from lightrag.operate import chunking_by_token_size` | `from lightrag.chunker import chunking_by_token_size` |
| `hars_memory/ingest/md_structure.py` (`md_struct_chunking_func`) | `from lightrag.operate import chunking_by_token_size` | `from lightrag.chunker import chunking_by_token_size` |
| `hars_memory/eval/extractor_bench.py` | `from lightrag.constants import DEFAULT_ENTITY_TYPES` | **removed in 1.5** — module-local `DEFAULT_ENTITY_TYPES` (the 1.4.16 list) in `extractor_bench.py` |
| `tests/test_chunker_structured.py` (×2), `tests/test_md_structure.py` | stubbed/imported `lightrag.operate.chunking_by_token_size` | updated to the `lightrag.chunker` home |

`DEFAULT_SUMMARY_LANGUAGE` still exists in `lightrag.constants` (1.5.7), so the
same line's second import is unchanged.

No compatibility shim was needed for `chunking_by_token_size`: the 1.5.7
signature is identical to 1.4.16's and, on every valid input, the output is
**byte-identical** (see below). It now lives at `lightrag.chunker.token_size`
and is re-exported from `lightrag.chunker`.

## Behaviour differences the upgrade forces

1. **Chunking output — unchanged for the current pipeline.** A 1.4.16↔1.5.7
   differential over 7 representative inputs (plain prose, zero overlap,
   `split_by_character`, `split_by_character_only`, oversized split-char,
   Cyrillic, tiny) produced **0 mismatches** with the repo's tiktoken
   tokenizer. The only difference is on *invalid* configs, where 1.5.7 fails
   fast with `ValueError` (`chunk_overlap_token_size` negative, or `>=`
   `chunk_token_size`) instead of 1.4.16's opaque `range()` error / silent empty
   result. The pipeline uses overlap 64 < size 512 (and default 100 < 1200), so
   no chunk output changes. The `md_struct`/token fixtures
   (`test_md_structure.py`, `test_chunker*.py`) pass unchanged.
2. **Extraction-prompt contract changed.** 1.5 removed the `{entity_types}`
   template slot and `DEFAULT_ENTITY_TYPES`; the system/user prompts now take
   `{entity_types_guidance}` plus `{max_total_records}` /
   `{max_entity_records}`, and the user prompt has a
   `{heading_context_block}` slot. `eval/extractor_bench.py::build_prompts` was
   ported to build prompts exactly as `operate._process_single_content` does in
   1.5.7 (via `get_default_entity_extraction_prompt_profile()`); it still uses
   the REAL `PROMPTS` dict and appends the schema guidance. The `meta` block
   still reports the 11-type `DEFAULT_ENTITY_TYPES` list for cross-version
   comparability.
3. **Extraction cache-key derivation changed.** 1.5 folds the serialized
   `response_format` and a per-role `llm_cache_identity` into the hashed input
   of `use_llm_func_with_cache`. `ingest/batch.py::request_key` (the batch
   collector's mirror, which fails loudly if it drifts) was updated to mirror
   the 1.5.7 derivation, and `_collect_async` now builds its `global_config`
   via `rag._build_global_config()` and computes
   `get_llm_cache_identity(global_config, "extract")`.
4. **Per-role LLM wrappers.** 1.5 routes every LLM call through
   `global_config["role_llm_funcs"][<role>]` (`extract`, `query`, `keyword`,
   `vlm`); the role wrappers are built **once** at `LightRAG.__post_init__` from
   the base `llm_model_func`. Consequences:
   - `ingest/batch.py::_collect_async` must set
     `role_llm_funcs["extract"]` (done).
   - ⚠️ **`rag.llm_model_func = ...` post-init swaps no longer affect which
     function a role calls.** `mcp_server.py`'s `memory_recall` synthesis path
     (~line 4024), `cli.py`, and `grpc/server.py` all use this swap pattern and
     are therefore **no-ops under 1.5.7** — `aquery_llm` will use the extractor
     model for synthesis instead of the query model. Empirically confirmed:
     after `rag.llm_model_func = new`, `rag._role_llm_states["query"].raw_func`
     is still the original function. The 1.5.7-supported replacement is
     `await rag.aupdate_llm_role_config("query", model_func=...)` (or
     `role_llm_configs={"query": ...}` at construction);
     `QueryParam` still has no `model_func`. **Not fixed here** (mcp_server.py
     is owned by another agent; cli/grpc share the pattern) — flagged for a
     dedicated follow-up. No test covers real role-model selection, so the
     suite is green either way.

## The decisive proof: 1.5.7 reads the new store

`tests/test_lightrag_157_store_read.py` loads the 2026-10-17 snapshot
(`index_full_mdstruct_20261014`) with the installed 1.5.7 reader in a subprocess
and asserts, on the real data:

```
reader version: 1.5.7
rows: 1340 | DocProcessingStatus failures: 0
get_status_counts(): {'pending': 17, 'parsing': 75, 'analyzing': 0,
                      'processing': 0, 'preprocessed': 0,
                      'processed': 958, 'failed': 290}
DocStatus enum known: analyzing, failed, parsing, pending, preprocessed,
                      processed, processing
sum: 1340
```

- `DocProcessingStatus(**row)` for **all 1340 rows → 0 failures** (was 1050
  `TypeError`s on 1.4.16).
- `get_status_counts()` → no `KeyError`, counts complete
  (`processed 958 / failed 290 / parsing 75 / pending 17`, sum 1340).

The test skips when the external snapshot is absent (it is not repo content), so
CI stays green; override the location with
`HARS_LIGHTRAG_157_SNAPSHOT=<dir containing kv_store_doc_status.json>`.

## Gate results (2026-10-17)

- `uv run pytest -q` → **1 failed, 1180 passed, 27 skipped**; the only failure is
  the known pre-existing `experiments/search-2026-10/ml/test_tokenize_russian.py`
  environment failure.
- `uvx --from ruff==0.14.13 ruff check .` → **All checks passed** (the seven
  pre-existing findings were fixed: `experiments/search-2026-10/{gen_cases.py,
  h1_rerank.py,show.py}`, `hars_memory/server/lightrag_init.py`,
  `tests/test_context_merge_budget.py`, `tests/test_embedder_http_mode.py` ×2).
- The 37 test failures measured on an unpatched `==1.5.7` are all resolved.

## Upgrade procedure (serving venv → 1.5.7) — DONE

```bash
cd /home/demo/Project/software/hars-longterm-memory
cp pyproject.toml /tmp/pyproject.toml.bak && cp uv.lock /tmp/uv.lock.bak
# import migration landed (chunker.py, md_structure.py, extractor_bench.py)
$EDITOR pyproject.toml     # "lightrag-hku==1.5.7"
uv lock
uv sync --all-extras --all-groups
uv run python -c "import importlib.metadata as m; assert m.version('lightrag-hku')=='1.5.7'"
uv run pytest -q
```

The MCP service consumes this venv via `cortex/.mcp.json`
(`uv run --project ../hars-longterm-memory memory mcp`), so it picks up 1.5.7 on
its **next restart** — no service restart is performed by this migration.

## Rollback procedure (back to 1.4.16)

```bash
cd /home/demo/Project/software/hars-longterm-memory
cp /tmp/pyproject.toml.bak pyproject.toml
cp /tmp/uv.lock.bak uv.lock
uv lock && uv sync --all-extras --all-groups
uv run python -c "import importlib.metadata as m; print('lightrag-hku', m.version('lightrag-hku'))"   # 1.4.16
uv run pytest -q
```

Rollback is pure dependency state — no index mutation, so it can be done at any
time without touching the store. If the pip cache is cold, `uv` re-downloads
1.4.16 from PyPI (network required). Note that the 1.5-era import migration is
backward-compatible: `lightrag.chunker.chunking_by_token_size` does not exist in
1.4.16, so reverting the *dependency* alone leaves the code unable to import the
chunker — revert the code commit too, or keep the pin forward.

## Historical: interim mitigation (no longer needed)

While the repo was on 1.4.16, serving a 1.5.7-built store required either
keeping the 1.4.16-built live store, or normalising the new store's
`kv_store_doc_status.json` (strip `content_hash`; drop/relabel
`parsing`/`analyzing` rows). Both were inferior to aligning the version; the
normalisation was lossy and had to be redone on every rebuild. Not needed now.
