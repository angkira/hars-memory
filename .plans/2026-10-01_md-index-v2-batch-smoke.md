# Index v2 (markdown chunker) built via gpt-6-luna Batch — smoke first

## Goal
Build a SEPARATE index (new version) with `HARS_MEMORY_CHUNKER=markdown` and extraction by
`openai/gpt-6-luna` through the OpenAI Batch API (50% off, 0.05/0.25 per 1M), so token-vs-markdown
chunking can be compared by evals against the existing index. The live index must not be touched.

- Old (baseline, read-only): `~/.local/share/hars-longterm-memory/index` (chunker=token, gpt-5.6-luna).
  The running MCP server uses it: never write there, never delete it.
- New: `~/.local/share/hars-longterm-memory/index-md-v2` (override via `--index-dir` / env).
- Sources: `HARS_MEMORY_SOURCES_MANIFEST=/Users/iurii.medvedev/Work/sre-support-pier/knowledge-sources.yaml`
  (kb/ has ~621 md files; repos/sre-docs/docs has H4-H5 nesting).
- LiteLLM proxy `https://litellm.labs.jb.gg/v1`, key `HARS_MEMORY_LLM_API_KEY` from
  `config/live-llm-e2e.env` (never print it). gpt-6-luna requires temperature 1.0. Proxy exposes
  `/v1/batches` and `/v1/files`.

## Batch mechanism (verify before building)
LightRAG calls the LLM synchronously per chunk. Idea: prime LightRAG's LLM response cache.
1. Read installed lightrag 1.4.16 (`operate.extract_entities`, `utils.handle_cache`, the
   `llm_response_cache` KV, `enable_llm_cache_for_entity_extract`) and establish EXACTLY how the cache
   key is derived (hash of prompt + mode/type) and which prompts the extraction issues (incl.
   gleaning rounds, which depend on earlier answers -> may need a second batch round or be disabled
   via entity_extract_max_gleaning=0 for v2; note this and keep it identical for the baseline comparison caveat).
2. Phase A "collect": run chunking + the extraction prompt build with a recording LLM func that
   returns a stub and records (cache_key, prompt) — no network.
3. Phase B "submit": write JSONL (custom_id = cache_key), upload to /v1/files, create batch against
   the endpoint the proxy supports (check which of /v1/chat/completions or /v1/responses works for
   gpt-6-luna with a 2-request test first), persist batch id + mapping in `<index>/batch_state.json`.
4. Phase C "poll/apply": poll; on completion download output, write responses into the cache
   (error lines retried via sync fallback, counted), then run normal indexing so LightRAG gets
   cache hits for extraction. Resumable and idempotent at every phase.
Expose as `memory index-batch {collect|submit|status|apply|run}` (or flags on the existing
indexer — pick what fits cli.py), plus `--dry-run` cost estimate (tokens x batch prices) printed
BEFORE any paid call.
If the cache-priming idea is unworkable, stop and report why with the alternative; do not
hack around LightRAG internals by monkeypatching more than a thin, documented adapter.

## Smoke scope (this task)
Hard cap: 10 documents total (mix: 4 plain kb md, 3 with tables, 3 from repos/sre-docs/docs with
H4+/duplicate headings, include the T4/T7/T2 failure documents if findable). Cost guard: print the
estimate and abort if > $0.50.
1. Sync smoke: index those 10 docs into a scratch dir with chunker=markdown and gpt-6-luna
   sync. Verify end-to-end: chunks have heading_path/line fields in kv_store_text_chunks, entities
   created, recall via `memory recall`/MCP-level function returns hits with location fields.
2. Batch smoke: same 10 docs into another scratch dir via the batch path; wait up to ~30 min
   for completion; if not finished leave state resumable and report the batch id/status.
3. Compare the two scratch indexes (entity/relation counts, cache hit rate, any failures) and
   report. No full-KB run, no evals yet.

## Out of scope
Full KB indexing, comparison evals (next step: `memory regress` / gold_questions / battle-style
queries on old vs v2 — plan it in the report, don't run it), eval/battle.py bug, mcp bump,
commits, version bump.

## Done when
`uv run pytest -q` green, new tests for collect/submit/apply with a fake batch backend, a
.session note, and a report with measured numbers and the exact commands to run the full build.
