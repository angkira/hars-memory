# Index migration command (`memory migrate-index`) + chunk location metadata

## Motivation
Eval of `memory_recall` (tmp/memory-eval/results.md, 9 facts / 7 docs) showed:
- no location fields in hits: `heading_path`, `start_line`, `end_line` absent, `section` always empty,
  file name is flattened (`/` -> `-`) instead of the real path;
- chunks cut table rows / headings; text orphaned under the next section's heading;
- duplicate docs (`internal/how-to-articles/` vs `platform/internal/how-to-articles/`);
- bloated responses (12-61k chars) — same text in context, fused_chunks, snippet, identifier_matches.

Already true: `ingest/chunker.py` is Markdown-aware and prepends a breadcrumb. What is missing is
persisting location metadata and a way to apply it to an EXISTING index without paying for LLM
entity extraction again.

## Goal
A CLI command, runnable inside a repo / KB root, that migrates an existing index to the new
chunk layout with **zero LLM calls** (local embedder only; no extraction, no query LLM).
Graph entities/relations stay untouched.

## Scope
1. Chunker/ingest: `Chunk` gains `heading_path` (list[str]), `start_line`, `end_line` (1-based,
   source file), `section` (innermost heading). Table rows and fenced blocks never split mid-row/fence.
   Chunk ids stay stable where text span unchanged. Keep the plain-window path byte-for-byte.
2. Persist these in chunk payload (vector store + LightRAG chunk KV) and real source path
   (`source_path`, repo-relative) alongside the flattened name.
3. `memory migrate-index [--root PATH] [--dry-run] [--no-embed-reuse] [--dedupe]`:
   - walks sources with the existing walker/`.memoryignore`, re-chunks, diffs against stored chunks;
   - re-embeds only chunks whose text changed (reuse vectors otherwise); local embedder only;
   - back-fills metadata on unchanged chunks without re-embedding;
   - removes duplicate documents (same content fingerprint, different path) keeping the canonical one;
   - `--dry-run` prints counts (chunks kept / re-embedded / metadata-only / deleted / dup docs) and
     refuses nothing; real run takes a backup/tmp copy first and is resumable/idempotent;
   - fails loudly if any LLM env/config would be touched.
4. Recall output: expose the new fields in `memory_recall` hits; add compact mode (drop duplicated
   `context`/`snippet`/`identifier_matches` text) — only if it can be done without breaking
   existing MCP response consumers (additive flag, default unchanged).

## Out of scope
- Fixing `eval/battle.py` `_PROJECT_ROOT` bug (parked, see AGENTS.md).
- Bumping `mcp` upper bound.
- Re-running graph extraction.

## Acceptance
- `uv run pytest -q` green; new tests: chunker line ranges/heading_path (nested to H5, tables,
  fences), migration dry-run + idempotency + no-LLM guard (LLM client patched to raise), dedupe.
- Fixture-based end-to-end: build old-layout index, migrate, recall returns location fields.

## Open questions for the implementer (investigate first, report before big changes)
- Where exactly chunks live (LightRAG `kv_store_text_chunks`, Qdrant payload, BM25 sidecar) and
  whether payload can be updated in place without re-upsert of vectors.
- How flattened names are produced and whether the real path can be recovered from the sidecar.
