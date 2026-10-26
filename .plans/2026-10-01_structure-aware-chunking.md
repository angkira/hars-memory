# Structure-aware chunking for the LightRAG index

## Problem
`memory_recall` eval: chunks cut table rows/headings (T4), text orphaned under the next
section's heading (T7), duplicate headings without parents (T2).
Root cause (found while building migrate-index): the LightRAG index chunks with LightRAG's own
token splitter (`chunking_by_token_size`), NOT `ingest/chunker.py`. Markdown-awareness exists
only on the corpus/service paths.

## Hook
LightRAG 1.4.16 exposes `LightRAG(chunking_func=...)` with signature
`(tokenizer, content, split_by_character, split_by_character_only, chunk_overlap_token_size,
chunk_token_size) -> list[{tokens, content, chunk_order_index}]`. Wire a markdown-aware function
in `server/lightrag_init.py` behind `HARS_MEMORY_CHUNKER=markdown|token` (default `token` until
validated, so existing indexes stay consistent; chunk ids = md5(text) change under a new chunker
and require re-extraction, see the LLM migration plan).

## Chunking rules
1. Section = text under one heading, any level H1-H6 (today only H2/H3 split). Every chunk gets the
   full breadcrumb prefix `# Doc > ## A > ### B > #### C`. Text never crosses a heading boundary,
   so it can't be orphaned under a neighbour heading. Tiny sibling sections may merge up into the
   parent only if the breadcrumb is preserved for each merged part.
2. Atomic blocks, never cut when they fit chunk_token_size: pipe tables, fenced code, list
   (with its lead-in sentence), blockquote/admonition, YAML/front-matter.
3. Oversized atomic block -> "nested level", split by its own rows/lines:
   - table: split between rows; the header row + separator repeated at the top of every piece;
   - code fence: split at blank lines / line boundaries; re-open and close the fence in each piece
     with the same info string;
   - list: split between top-level items, lead-in repeated.
   Pieces carry `part i/n` in metadata and the same heading_path; never cut mid-row/mid-line.
4. Prose: paragraph boundaries, then sentence boundaries, token-window only as last resort.
5. Output chunks carry heading_path/start_line/end_line/section (fields already exist on
   `ingest.chunker.Chunk`); the new function must record them so index stamping persists them.
6. Plain / non-markdown input: delegate to existing token splitter unchanged.

## Reuse
Extend `ingest/chunker.py` (`_split_blocks`, `_chunk_markdown`) rather than writing a second
chunker; keep `chunk_text` public behaviour for corpus/service backwards-compatible (flag if a
behaviour change is unavoidable). Use LightRAG's tokenizer for sizing in the adapter.

## Tests
Fixtures from the failures: table whose row would straddle a boundary; Deploy-style section where
a command precedes the next heading; repeated "What happens when you Abandon" under different
H2/H3/H4 parents; H4-H5 nesting; oversized table (header repeated, no mid-row cut); oversized fence
(balanced fences in every piece); idempotent/deterministic output; token budget respected;
`token` mode byte-identical to today.

## Out of scope
Re-extraction/migration of existing indexes (separate LLM-migration plan), eval/battle.py bug,
mcp bump, commits/version.
