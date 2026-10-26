# Session-graph recall (M2) + full dropped-span persistence (M3) for the nc-eval

Status: design + plan only, nothing implemented. Written 2026-10-26.
Inputs: `.plans/2026-10-22_compaction-session-graph.md` (session memory spec),
`experiments/neurocompact-eval/HANDOFF.md` (OPEN PROTOCOL ISSUES M2, M3, B4, M7),
`REPORT-PROTOCOL.md`, `RUNBOOK-gate.md`, live smoke test in the dev container
(Haiku extractor, `HARS_MEMORY_SESSION_MODE=split`).

## 0. TL;DR

- M2 is CONFIRMED from code: in split mode nothing reads the per-session graph at
  recall time. The `nc` arm is chunk-RAG over compaction documents (cosine + token F1),
  not graph memory. The graph is built (extraction cost paid) and only counted.
- Recommended fix: an opt-in **session KG branch** — a LightRAG instance scoped to the
  session working dir, queried with `only_need_context=True` in `hybrid` (local+global)
  mode; its Entity/Relationship sections are spliced on top of the UNCHANGED fusion-only
  chunk section. Keywords: caller-supplied, else one LightRAG keyword-extraction LLM call,
  else question-as-keyword fallback. Off by default: the existing `nc` call shape stays
  byte-identical.
- Ablation: a 4th arm `nc-graph` forked from the `nc` session at C3 (same note, same
  HARS session dir, same packer, same cap); only the recall flag differs. `nc` becomes
  `nc-chunks` in reporting. The contrast `nc-graph - nc-chunks` isolates the graph.
- M3: send the whole serialized dropped span (<= RAW_CAP) to `memory_compact` in
  ~8k-char pieces. Measure the current loss first ($0, offline). Cost at realistic span
  size: +2-4 chunks per compaction (~+15-25% extraction calls).
- The experiment as designed is weakly suited to show a graph benefit (single-hop
  verbatim-evidence questions, leak-proof questions, ~300 graph tokens after the packer
  split, C=3 tiny graphs, equal budgets so no token-saving claim). See section 5.
- Everything in section 6 must be written into REPORT-PROTOCOL.md as Amendment C
  BEFORE any run under the new code is looked at.

## 1. Finding (M2) — confirmed from code

Call shape: `scripts/run_chainreplay.py:539-540` sends
`{"session_id", "question", "top_k": 8}` — no `ll_keywords`/`hl_keywords`, no `mode`.

Recall path, `hars_memory/mcp_server.py`:

1. `_resolve_query_mode` (1184-1200): no keywords -> `naive` + `mode_fallback`.
2. `_resolve_graph_branch` (1203-1243): `HARS_MEMORY_GRAPH_BRANCH=auto` (655) and
   resolved mode `naive` -> `enabled=False`, reported mode `"skipped"`, reason
   "no graph signal ...". Applied at 4819-4834.
3. `rag = await _get_rag(project_id)` (4875) -> `ProjectRegistry.get_rag`
   (`hars_memory/projects/registry.py:204-231`): `working_dir=project.index_dir` = the
   MAIN index. There is no other LightRAG instance on the recall path.
4. Session channel `_compute_session_channel` (2773-3039, called at 4895): reads only
   `kv_store_text_chunks.json` + `vdb_chunks.json` (2899-2900), scores
   `max(cosine, token-F1)` (2935-2945), appends chunks. The graph is touched only by
   `session_store._graph_counts` (2966-2970) for the `entities`/`relations` stats.
5. With keywords, `rag.aquery(...)` (4984) runs `mix` against the main rag. In the eval the
   main index is empty -> `[no-context]` -> `has_lightrag_context=False` -> fusion-only
   render (`_render_fusion_only_context`, 4010) of main fused chunks (none) + session
   chunks. Matches the smoke test exactly.

Existing code that touches a session graph (none of it serves recall):

- `session_memory.py:506-532` `_graph_entity_names` / `_graph_counts` (status + merge
  alias hints).
- `memory_merge_session` re-ingests the compaction TEXT into the main index
  (re-extraction); the session graph contributes only alias hints. After a merge the
  session channel is skipped (2845-2857) and a keyworded recall would read the MAIN graph.
- `HARS_MEMORY_SESSION_MODE=subgraph` ingests compactions straight into the main graph, so a
  keyworded recall DOES read graph data there — but the main index is shared by every
  chain/arm (cross-chain contamination) and the eval runs split mode.
- `scripts/funnel_chains.py` (offline) reads graphml for `anchor_in_graph`.

Consequence for the record: the "NeuroCompact graph memory" arm has so far tested
externalized chunk memory. Every existing `nc` number (gates, batch 1) is a chunk-RAG
number. The funnel's `anchor_in_graph` and the "kill graph" rule (HARS recall vs BM25)
measure chunk retrieval, not the graph.

## 2. Design options for recall-time session-graph use

Budget context for all options: mem_budget 1536; nc note realized 417-635 tokens ->
recall cap `clamp(1536 - note, 256, 1024)` = roughly 900-1024 tokens. Packer v2
(`run_chainreplay.py:873ff`) parses LightRAG's `Knowledge Graph Data (Entity)` /
`(Relationship)` fenced JSON-lines and, when any graph section is present, gives chunks
70% and relations+entities 30% (relations first, entities last). So whatever the graph
branch returns, it gets ~270-310 tokens: about 4-8 relation descriptions plus a few
entity lines, and the chunk side loses the same amount versus today.

| # | Option | LLM calls / recall | Added latency (est.) | What reaches the packer | Notes |
|---|---|---|---|---|---|
| A | Session-scoped LightRAG `aquery(hybrid, only_need_context)` with keywords (caller -> LLM extraction -> question fallback); splice KG sections over the unchanged chunk section | 0 if keywords given; 1 keyword call otherwise (extractor model; LightRAG caches it in the session's `kv_store_llm_response_cache.json`) | instance build + storage load (session dir is small: ~50-300 ms), 2-3 query embeddings, keyword call ~1-3 s on Haiku | KG entity + relation sections (graph-ranked, 1-hop via local/global) + the SAME chunk section as today | Least new retrieval code; real LightRAG semantics; packer already parses the format. |
| A-mix | Same, but `mix` and keep LightRAG's own chunk section (graph-reached chunks merged via `_merge_context_with_fusion`) | same | same | KG + graph-guided chunk selection | Changes the chunk side too: confounds "graph facts" with "different chunk retrieval". Keep as exploratory arm only. |
| D | In-house session KG channel: cosine + token-F1 over `vdb_entities.json` / `vdb_relationships.json` (reuse `_session_vectors_from_vdb`), +1-hop edges from graphml, render LightRAG-format sections | 0 | ~10-50 ms | KG sections (flat ranking, no keyword step) | Deterministic, hermetically testable, cheapest. Re-implements part of LightRAG local/global. Fallback if A's latency/determinism is unacceptable. |
| C | Graph-guided chunk selection: `retrieval/graph_bridge.py` `augment_fused_pool` with `load_store(session_dir)`, seeded from session-channel hits | 0 | ~ms | chunks only (reordered / injected) | Tests "graph helps retrieval", not "graph as a denser memory representation". With 15-50 chunks per session and top_k 8 the headroom is small. Not the product claim. |
| B | Merge-then-recall: `memory_merge_session` after C3, keyworded recall on main | 0-1 per recall, but a full RE-EXTRACTION of all compactions per session (~2x extraction spend, minutes) | merge 1-3 min once per session | main-graph KG + main chunks | Defeats the "temporary per-session graph" idea; shared main index -> cross-chain contamination unless one project per chain. Reject for the eval. |
| E | Zero-code: `subgraph` mode + one project per chain + runner-generated keywords | 1 (runner side) | same as A | main-graph KG + main fused chunks | Needs project registration per chain and changes how compactions are stored for the whole nc arm (not a fork-level ablation). Reject. |
| K | Agent-supplied keywords (agent calls recall as a tool) | agent turn(s) | seconds | as A | Changes the ask protocol (tool loop in one arm only). Product option, not for this eval. |

Packer interaction summary:
- A and D add KG sections -> 70/30 split activates; chunk side shrinks by ~30%. That IS the
  treatment ("graph facts instead of the last ~300 chunk tokens") and must be pre-registered as
  such; do not retune the split for the graph arm.
- A-mix also changes which chunks arrive; C changes only chunk order. Both bypass the 70/30
  split question (C) or mix two effects (A-mix).
- LightRAG's KG JSON lines carry key/quote overhead (~15-25% of tokens); entity
  descriptions are extractor paraphrases and can drop exact identifiers (the dataset's
  anchors are identifiers). The reader (qwen flash) must parse JSON lines — reader risk is
  measured by P(anchor_hit | anchor_in_recall) per section.

### Recommendation: Option A (KG-only splice), D as fallback

Why: it is the actual product claim (a LightRAG graph per session), it reuses LightRAG's
local/global retrieval and the existing context-splice helpers, it keeps the chunk side
byte-comparable with today's `nc` so the ablation is clean, and it costs at most one small
LLM call per recall. D is the drop-in if the keyword call turns out slow/nondeterministic.

### Minimal implementation plan

Server (`hars_memory/`):

1. `mcp_server.py`, `memory_recall` input schema (near 4297): new optional
   `session_graph: bool` (default from env `HARS_MEMORY_SESSION_GRAPH`, default `off`), and
   optional `session_graph_mode: "hybrid"|"local"|"global"|"mix"` (default `hybrid`).
   Absent/off -> response byte-identical to today (regression test).
2. New `async def _compute_session_graph_context(question, sid, *, ll, hl, mode, top_k,
   project_id) -> tuple[dict, str | None]`, next to `_compute_session_channel`:
   - same guards as the session channel (split mode, dir exists, not merged), fail-soft;
   - build a LightRAG instance via `create_lightrag(working_dir=str(session_dir),
     vector_storage="NanoVectorDBStorage", graph_storage="NetworkXStorage")` — the same
     storages `session_memory._storage_kwargs` uses for ingest (never the main Qdrant
     collections); cache per (sid, mtime of graphml + vdb files), invalidate on change
     (a compaction rewrites them); hold a per-session asyncio lock vs a concurrent
     `memory_compact` (product; the eval asks only after C3);
   - keywords: caller `ll/hl` if any -> else LightRAG extraction (`QueryParam` without
     keywords; record `keyword_source="llm"` + the extracted lists from the debug log /
     cache) -> on error/timeout `ll=hl=[question]` (`keyword_source="question"`);
   - `aquery(question, QueryParam(mode, only_need_context=True, top_k, chunk_top_k,
     max_entity_tokens=DEFAULT_MAX_ENTITY_CONTEXT_BYTES,
     max_relation_tokens=DEFAULT_MAX_RELATION_CONTEXT_BYTES, enable_rerank=False))`;
   - returns block `{status, keyword_source, ll_keywords, hl_keywords, entities, relations,
     latency_ms, error?}` and the raw context.
3. Context assembly (context_only branch, ~5000-5100): when the session graph context is
   non-empty, keep the chunk lines exactly as `_render_fusion_only_context` would produce
   them and emit `KG(entity) + KG(relationship) + chunk section`; implement as a helper
   `_splice_session_kg(session_ctx, fusion_chunk_lines, ref_lines)` built on
   `_splice_context_sections` (3956). If the main graph branch also produced KG sections
   (product case), concatenate main + session JSON lines in one section each (session
   first), each capped by the same byte caps.
4. Response: `session_graph` block; `graph_branch` gains `session: {enabled, reason}`;
   trace frame `session_graph` (same style as `session_hits`).
5. Docs: `docs/mcp-reference.md`, `docs/configuration.md` (new env var).

Eval (`experiments/neurocompact-eval/`):

6. `run_chainreplay.py`: arm `nc-graph` = fork of the `nc` post-C3 session (like
   `nc-note-only`), recall `{session_id, question, top_k: 8, session_graph: true}`; same cap
   (`recall_cap_for` with the nc C3 note), same packer. Record per ask:
   `recall_sections` (kg_entities, kg_relations, chunks kept), `keyword_source`,
   `session_graph.status`, recall latency. RECALL FAIL for nc-graph also on
   `session_graph.status == "error"`.
7. `metrics_v2.py`: add pairs `nc-graph - nc` and `nc-graph - baseline`; triplet ->
   quadruplet completeness; anchor_in_recall split by section.
8. `funnel_chains.py`: `anchor_in_graph` = anchor substring in session entity names /
   descriptions / relation descriptions (graphml + kv stores); `anchor_in_kg_recall`.

Tests (hermetic, fake RAG, tmp dirs):

- `tests/test_mcp_session_tools.py`: `session_graph` absent/off -> payload identical to
  today (golden compare); on -> fake session LightRAG (monkeypatch `create_lightrag`)
  returning a context with KG + chunk sections -> response context has the KG sections
  and the SAME chunk section as the off case; merged/subgraph/unknown session -> block
  status skipped/unknown, recall ok; aquery raises -> status error, chunks unaffected;
  keyword fallback order (caller > llm > question).
- `tests/test_graph_branch_routing.py`: main branch skipped + session graph on -> no main
  `aquery`, one session `aquery`; storage kwargs are Nano/NetworkX even with Qdrant env.
- Eval selftests (`--pack-selftest`): packer on a KG+chunks payload keeps relations within
  the 30% side budget and the chunk section identical to the chunk-only pack up to the 70%
  cut.
- `uv run pytest -q` green; ruff 0.14.13 clean.

Smoke (before any paid run): one live recall on the dev container session used for the
M2 finding, `session_graph: true`, check `session_graph.entities > 0`, KG sections in
`context`, latency; archive per CLAUDE.md run-artifact rules.

## 3. M3 — full dropped span to `memory_compact`

Today (`pi-agent/extensions/neurocompact.ts:72-73, 793, 1000-1004`): `convText` is cut to
`RAW_CAP=60000`; if longer than 24000 chars only `[0:12000]` and `[-12000:]` are sent. The
baseline (and the nc note) summarizer see up to 60k. Dataset spans are ~24.1-25.2k chars
plus replay wrappers ("SESSION LOG replay - part i/n ... received"), so the lost middle is
probably a few k chars per span — but it sits exactly at the HEAD/TAIL boundary from which
`make_chains.py` draws items, so it can hit anchors.

Step 0 ($0, offline): for every answerable item, check whether its evidence span / anchor
is in the persisted compaction doc (`kv_store_full_docs.json` of the archived session
dirs) vs in the full span. Report `anchor_in_compaction_doc` per chain. This sizes M3 on
the existing data.

Fix: send ALL of `convText` (still `<= RAW_CAP`, identical input to both summarizers) as
`dropped_context` pieces of ~8000 chars cut at message boundaries (fallback: whitespace).
Keep `RAW_CAP` unchanged (raising it would change the baseline arm too). Optional server
hardening, separate change: `memory_compact` input limit / warning when `dropped_context`
exceeds N chars.

Cost (LightRAG chunk 512 tokens, overlap 64, gleaning `HARS_MEMORY_MAX_GLEANING=1` -> 2
extraction calls per chunk; ~3.5-4 chars/token on log/code text):

| dropped payload | tokens (est.) | chunks | extraction calls | wall time at llm_max_async 4, ~5-10 s/call (Haiku, est.) |
|---|---|---|---|---|
| today, 24k chars | ~6-7k | ~15-16 (+summary) | ~32 | ~40-80 s |
| full span, ~28-30k chars (realistic here) | ~7.5-8.5k | ~18-20 | ~38-40 | ~50-100 s |
| worst case RAW_CAP 60k | ~15-17k | ~35-38 | ~72-76 | ~1.5-3 min |

Per call ~2.5k input (prompt + schema guidance + chunk; gleaning call ~4k) and ~0.5-1.5k
output. Merge-summary LLM calls are rare at C=3. Halving via `HARS_MEMORY_MAX_GLEANING=0`
for session ingest is possible but changes extraction quality — if used, pre-register it.
On the glm :8307 service (~60 s/compaction today) expect ~+15-25%, inside the 600 s queue
timeout. Under the M3 fix all nc compactions change, so every decision chain (incl.
batch 1) must be re-run.

## 4. Ablation design (graph contribution)

- Arms (per chain, after C3): `baseline`, `nc` (reported as `nc-chunks`), `nc-note-only`
  (fork), `nc-graph` (fork of the nc session; identical note, compactions, session dir,
  packer, cap; only `session_graph: true`). Extra spend: one recall + one ask per question;
  no extra compactions.
- Primary graph contrast: `nc-graph - nc-chunks`, paired per qid, cluster bootstrap by
  session, Wilcoxon over cluster means (metrics_v2 machinery).
- Mechanism readouts: anchor_in_recall split into kg vs chunk section; P(anchor_hit |
  anchor only in kg section); chunk tokens displaced by KG; keyword_source distribution;
  `session_graph` empty rate.
- Optional exploratory arm (not decision-bearing): `nc-graph-mix` (A-mix) or D, only if
  budget allows; label exploratory in the amendment.
- Determinism: run the KG keyword extraction with LightRAG's LLM cache on; record the
  extracted keywords per ask so offline refetches are not needed.

## 5. Can the experiment show the claimed benefit at all? (confounds)

1. **Mislabelled treatment.** Until this lands, `nc` is chunk-RAG; no existing number
   speaks to "graph memory". The protocol's claim (section 1, "one graph per session") is
   currently not under test.
2. **Equal budget cannot show token savings.** Both arms are pinned to ~1536 realized
   tokens; the design can only show "equal or better at equal cost". "Cut tokens" needs an
   iso-quality curve (e.g. mem_budget 768/1024/1536 for nc vs baseline) — a separate,
   exploratory experiment.
3. **Graph gets ~300 tokens.** After the note (~420-635) and the 70/30 packer split, KG
   gets ~270-310 tokens and costs the chunk side the same. Any graph gain must beat the
   last ~300 chunk tokens it displaces; JSON-lines overhead and extractor paraphrase reduce
   density further.
4. **Question types favour chunks.** Every answerable item is single-hop, grounded in ONE
   verbatim evidence span (EVID/EWORD gates), with an identifier anchor (ANCH). That is the
   regime where chunk retrieval is strongest and a graph adds least. Graph value is in
   multi-hop, cross-compaction state ("value of X after the change in C3"), entity
   aggregation and supersession — none of which the dataset contains. Expect a small or
   null graph effect even if the graph is useful in the product.
5. **Leak-proof questions** (LEAK gate: no anchor in the question) defeat lexical matching
   and entity-name matching; graph retrieval then depends on keyword extraction bridging a
   paraphrase to an entity. That tests the keyword LLM as much as the graph.
6. **Tiny graphs at C=3.** 3 compactions -> tens of entities. The product hypothesis is
   about a graph that grows over 10-50 compactions; C=3 under-represents it (and lag effects
   that hurt the baseline summary are weak at lag <= 3).
7. **Baseline headroom.** The baseline summary sees the whole span and is rewritten 3 times;
   the drift test (lag-3 vs lag-1) decides whether there is headroom. If
   anchor_in_summary >= 85% the kill-architecture rule fires regardless of the graph.
8. **Asymmetric inputs (M3)** disadvantage nc on mid-span anchors until fixed.
9. **Extractor confound.** Haiku (smoke) vs glm (:8307) produce different graphs; the
   extractor and keyword model ids must be fixed and recorded per run.
10. **Reader confound.** qwen flash reads fragmented, packed text and JSON lines; a graph
    fact present in recall may still be missed. Report P(anchor_hit | in kg section).
11. **Power (B4).** 5 independent sessions, MIN_CHAINS=8 sessions -> verdict forced
    INCONCLUSIVE. A 4th arm does not fix power; the dataset rebuild does.
12. **Results already seen.** Batch 1 / gate numbers for chunk-`nc` have been looked at;
    they cannot enter a decision set under the amended protocol (also M7: tuning chains).
13. **Nondeterministic retrieval** (5/12 payloads differed on refetch) — record payloads at
    run time; never reconstruct offline.

## 6. Protocol Amendment C — must be committed BEFORE any new result is looked at

Write into REPORT-PROTOCOL.md section 4 (amendment log) and RUNBOOK-gate.md, commit with a
timestamp, then run:

1. **Claim correction:** state that all prior `nc` results are chunk-RAG over compaction
   documents (M2), exploratory only; batch 1, fullB and gates are excluded from the decision
   set.
2. **Arms:** add `nc-graph` (fork of nc at C3, `session_graph: true`, mode `hybrid`, KG-only
   splice); rename `nc` -> `nc-chunks` in reports. Freeze the full recall call shape per arm
   (top_k 8, fetch_top_k, session_graph, mode, byte caps 500/4500, rerank off).
3. **Keyword policy** for nc-graph: LLM extraction with fallback to question-as-keyword;
   freeze the keyword model id (= extractor id) and the LLM-cache setting; record
   `keyword_source` per ask.
4. **Hypotheses and primary endpoints:**
   - H-graph (primary for the graph claim): `nc-graph - nc-chunks` on clean_anchor_hit and
     judge score. Pre-register thresholds, e.g. graph helps = Δanchor >= +5 pp with CI95
     lower > 0 OR Δscore >= +0.10 with CI95 lower > 0; graph dead = |Δanchor| < 3 pp AND
     |Δscore| < 0.05; graph harms = Δanchor CI95 upper < 0.
   - H-product: which arm enters the existing WIN/FAIL rules vs baseline — designate
     `nc-graph` as the primary treatment (nc-chunks secondary), with Holm correction across
     the two treatment-vs-baseline tests, or declare nc-chunks results descriptive only.
   - The nc-vs-note-only "dead memory" rule applies to both treatments.
5. **Packer frozen:** packer v2 unchanged incl. the 70/30 graph split, PACK_FACTS_BONUS,
   PACK_QUERY_BONUS=1.0; no per-arm tuning.
6. **Budget parity:** same recall-cap formula for nc-graph; parity gate (run-mean ratio in
   [0.80, 1.05]) evaluated separately for each treatment vs baseline.
7. **Validity gates for nc-graph:** `session_graph.status == "error"` on any ask -> RECALL
   FAIL (chain invalid); an empty KG result is NOT invalidating but is reported
   (`kg_empty_rate`), threshold for a "graph not exercised" diagnostic (e.g. > 30% of asks).
8. **M3 fix in force:** full dropped span (<= RAW_CAP, ~8k-char pieces) for every nc
   compaction; baseline unchanged; Step-0 offline measurement reported alongside.
9. **Extraction config frozen:** extractor model id, `HARS_MEMORY_MAX_GLEANING`, chunk
   size/overlap, `HARS_MEMORY_SESSION_INDEX_BS`, embedder id.
10. **Funnel redefinitions:** `anchor_in_graph` = anchor in session KG (entity names,
    entity/relation descriptions); `anchor_in_recall` split by section; "kill graph" rule
    restated as nc-graph vs nc-chunks (H-graph), BM25 comparison kept for the chunk channel.
11. **Dataset (B4) and tuning chains (M7):** rebuild to >= 8 independent sessions with
    distinct probes, OR amend MIN_CHAINS explicitly (owner decision); exclude
    `qq8Vqbow_c0` / `y062eGZR_c0` from the decision set or report with/without.
12. **Strata (diagnostic only):** lag 1/2/3, HEAD/TAIL, anchor type; if a multi-hop /
    cross-compaction item type is added to the dataset, declare it now as a secondary
    endpoint (it is where the graph should matter).
13. **Cost/latency:** nc-graph median ask <= 20 s (WIN rule 4) includes the keyword call;
    cost/Q includes it.

## 7. Order of work

1. Step-0 M3 measurement ($0) and owner decision on B4 + Amendment C text.
2. Server: session KG branch + tests (section 2) -> `uv run pytest -q`.
3. Eval: M3 fix in `neurocompact.ts`, `nc-graph` arm, metrics/funnel changes, selftests.
4. Live smoke (1 session, Haiku), archived under `/mnt/datasets/ft/artifacts/`.
5. Commit Amendment C. Only then: `--dry-run`, gate run, decisive run.

Open questions for the owner: keyword policy (LLM vs question-only) as the frozen choice;
whether `nc-graph` replaces `nc` as the primary treatment; dataset rebuild vs MIN_CHAINS
amendment; whether to add a multi-hop item type now (requires re-validating the dataset).
