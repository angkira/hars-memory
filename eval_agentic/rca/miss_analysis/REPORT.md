# Where memory retrieval loses accuracy — and where extra power actually buys it

**Investment memo.** Read-only investigation, 2026-10-19. Repo
`hars-longterm-memory`; both indexes untouched; no services started; CPU-only
retrieval (the already-running local embed server on `:8791` was queried, no GPU
work launched).

Evidence base
* `docs/EVAL_NEW_VS_BASELINE.md` + `/mnt/datasets/ft/eval_new_vs_base/*` (the
  2026-10-19 gate: 48-query single-hop set, 28-query relational set, production
  index vs the `md_struct` index).
* Re-derived per-query traces for all 76 questions × 2 indexes
  (`traces/trace_*.json`, `traces/candidates_*.jsonl`) — these reproduce the
  published gate **exactly** (see §0), so every miss below is attributable.
* Existing RCA docs (`eval_agentic/rca/RCA.md`, `rerank_off_vs_on.json`,
  `retrieval_architecture_dossier.md`).
* Graph reachability audit of all 28 relational questions per index.

Deliverables in this directory: `REPORT.md`, `miss_table.csv` (208 gold-entry
rows), `taxonomy.json`, `headroom.json`, and the four runnable scripts
(`trace_misses.py`, `headroom_analysis.py`, `graph_reachability.py`,
`build_report_data.py`). Commands in §8.

---

## TL;DR

1. **On the production index, retrieval loses accuracy in ranking, not reach.**
   48/48 gold docs are retrievable and 48/48 are in the candidate pool; 47/48 are
   already in the top-10 chunk window. Yet only 34/48 are rank 1. Every one of
   the 14 rank-1 losses is a ranking loss; zero are coverage losses.
2. **The single biggest *available* lever is already built and not shipped.**
   The LightGBM ranker (`hars_memory/ranking/artifacts/ranker.json`) has a wiring
   spec but no call site in `mcp_server.py`. It is +10.4 pts recall@1
   (0.7083 → 0.8125) on the 48-query set, for ~29 ms/query of CPU scoring — and
   6 of its 7 fixed queries are also fixed by changing one constant (`alpha`
   0.5 → 0.15). Its honest marginal value over a retuned alpha is
   **+1 query (+2.1 pts)**.
3. **`alpha=0.5` is an untuned default that costs 4 rank-1 hits** (LOO-validated
   +8.3 pts, 0.7083 → 0.7917). But it is a *legacy-corpus* effect: on
   `md_struct` chunks, no alpha in 0.00–1.00 moves a single query. Do not build
   a roadmap on it.
4. **The `md_struct` index is coverage-limited, not ranking-limited.** 15/48 and
   25/56 gold docs are absent. On the docs it does hold it gets 29/33 rank 1
   (0.8788) — better than production's 0.7083. Its biggest need is the missing
   635 documents, then a ranker retrained on fine chunks (the current ranker is
   **inert** on `md_struct`: +0.0 pts r@1, 47/48 queries unchanged).
5. **The graph is the largest unmeasured lever.** The offline gate scores
   dense+BM25 only; it never invokes the graph branch that production runs for
   keyword queries. On the production graph, a 1-hop expansion seeded with the
   question's named entities (the gold set's `expected_entities`, which the
   question text names) reaches **both** gold docs for **28/28** relational
   questions (median 18 docs, max 260) while fusion alone gets 39/56 gold
   entries into the top 10. The relational deficit is a missing channel +
   ranking problem, not missing graph knowledge.
6. **The reranker is a measured bad buy** (+2 rank-1, −1 top-10, 5.9 s/query)
   and **query expansion has no demonstrated headroom** on these sets: the gold
   document's content-term overlap is strictly higher than the winning
   competitor's in 8 of 14 rank-1 misses and equal in the other 6 — never lower.
7. **Before any of that: the metric is the weakest link.** 3 of 14 hand-checked
   rank-1 misses are label problems (two where the top-1 document contains the
   same answer, one where two documents report conflicting values for the same
   question), 17/48 gold documents contain UUIDs and in 13 of those 17 at least
   one of those UUIDs also appears in another document (median 2 other docs), and
   n=48 gives a 95 % CI of ±12.5 pts.
   Every candidate improvement below is smaller than the noise floor of the
   instrument used to judge it.

---

## 0. Harness validation

`trace_misses.py` rebuilds the exact eval path (`build_pool` + `ltr_eval`
variants) and reproduces the published numbers to the fourth decimal:

| set | variant | recall@1 | recall@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| 48q | fusion | 0.7083 | 0.9792 | 0.8563 | 0.8149 |
| 48q | ltr_full | 0.8125 | 1.0000 | 0.9085 | 0.8785 |
| 28q | fusion | 0.2857 | 0.6964 | 0.6144 | 0.6845 |
| 28q | ltr_full | 0.3036 | 0.7143 | 0.6183 | 0.6887 |

Two facts the published report does not state, both verified here:

* **The ranked-ripgrep channel contributed exactly zero offline.**
  `ripgrep.default_roots()` is `[]` in this environment (no
  `HARS_MEMORY_RIPGREP_ROOTS`, no sources manifest), so the channel reports
  `available=false` for all 76 queries. `fusion` and `fusion_rog` produce
  byte-identical rankings on 76/76 queries in both saved per-query reports.
  Production *does* enable it (`cortex/.mcp.json`: `HARS_MEMORY_RIPGREP_CHANNEL=1`,
  `MODE=ranked`). The offline gate therefore measures neither of the two
  channels production actually relies on for identifiers (ripgrep) and
  multi-hop (graph).
* **The LTR ranker is not wired into serving.** `apply_ltr_ranker` is called
  only from `hars_memory/eval/ltr_eval.py`; `docs/ltr-ranker.md` §"Wiring spec
  (for the mcp_server.py owner)" confirms it is a proposal. Production serves
  `fusion` (0.7083/0.9792).

---

## 1. Where accuracy is lost — the miss table

`miss_table.csv` has one row per (index, set, query, gold doc) with the rank
achieved by fusion and LTR, the per-channel rank of the gold document
(BM25 top-30 and dense top-30, independently), and whether the document is in
the index at all. Failure classes are mechanical (see `taxonomy.json` for the
exact rules); only 3 rows were hand-adjudicated, and they are marked.

### 1.1 Production index, 48q (n=48, one gold doc each)

| class | n | share | what it means |
|---|---|---|---|
| hit (rank 1) | 34 | 70.8 % | — |
| `fusion_demoted` | 13 | 27.1 % | in the top-10 chunk window, not rank 1 |
| `window_truncation` | 1 | 2.1 % | out of the 10-chunk window, inside a 20-chunk window |
| coverage / channel miss | **0** | **0 %** | — |

Sub-causes of the 13 demotions: `rank_competition` 8 (both channels found the
gold; another doc scored higher), `single_channel_only` 4 (only BM25 returned the
gold, and the missing dense score halves its fused value at α=0.5),
`dense_gap` 1 (dense missed, BM25 rank ≤ 5).

Channel presence for the same 48 gold docs: **BM25 top-30: 48/48; BM25 top-3:
45/48; dense top-30: 41/48; dense top-3: 29/48.** BM25 is the strong channel on
this corpus; the losses happen in the blend.

Widening the doc window from 10 to 20 chunks takes top-10 from **47/48 to
48/48** and changes rank-1 not at all.

### 1.2 Production index, 28q relational (n=28, two gold docs each, 56 entries)

| class | n | share |
|---|---|---|
| hit (rank 1) | 16 | 28.6 % |
| `fusion_demoted` | 23 | 41.1 % |
| `channel_miss` (indexed, not in either channel's top-30) | 9 | 16.1 % |
| `fusion_buried` (in pool, below a 20-chunk window) | 6 | 10.7 % |
| `window_truncation` | 2 | 3.6 % |
| coverage | 0 | 0 % |

The asymmetry is the story: **the first-hop document reaches rank 1 in 13/28
questions (46.4 %), the bridged second document in 3/28 (10.7 %).** Top-10 rates
are 78.6 % vs 60.7 %. No single chunk (and no single query) contains the answer
to a 2-hop question; the retriever has no mechanism that *combines* the two
endpoints, because the graph branch is not part of the scored path.

### 1.3 `md_struct` index (NEW)

| set | class | n | share |
|---|---|---|---|
| 48q | hit | 29 | 60.4 % |
| 48q | **coverage** | **15** | **31.3 %** |
| 48q | fusion_demoted | 4 | 8.3 % |
| 28q | hit | 12 | 21.4 % |
| 28q | **coverage** | **25** | **44.6 %** |
| 28q | fusion_demoted | 14 | 25.0 % |
| 28q | fusion_buried | 5 | 8.9 % |

There is no coverage problem in the *production* index and no ranking problem in
the *new* one that its own corpus can expose: NEW gets 29/33 = 0.8788 rank 1 on
the documents it holds. The two indexes fail for opposite reasons.

### 1.4 What the ranker does per class

| | base 48q | new 48q | base 28q | new 28q |
|---|---|---|---|---|
| rank-1 queries fixed by LTR | 7 | 0 | 4 entries | 0 |
| rank-1 queries broken by LTR | 2 | 0 | 3 entries | 1 entry |
| net recall@1 | +5 | 0 | +1 entry | −1 entry |

On the production index the ranker is a real +10.4 pts. On `md_struct` it is
inert — the artifact was trained on legacy 3.7k-char chunk statistics.

---

## 2. The measurement problem (why some of the above is phantom error)

This is not a side note; it changes the ranking of every investment below.

1. **Document identity is the wrong target.** The corpus replicates facts across
   planning notes and launch reports. 17/48 gold documents contain UUIDs; in
   **13 of those 17 at least one of those UUIDs also appears in another document
   (median 2, max 27)**. A retriever that returns the launch note instead of the
   plan is scored as a miss although the answer is delivered. Hand-checked
   examples:
   * *"Which ports does the Isaac Sim ZMQ bridge use?"* — gold says 5570/5572;
     the rank-1 `docs-runtime_operations.md` contains **both** 5570 and 5572.
   * *"What PipelineRun ID was recorded for the fixed-T1 execution program?"* —
     gold plan and rank-1 launch note both contain
     `64b3aa63-735e-4b38-8265-8d52cf54ec62`.
   * *PACO DINOv3 support-bank pytest* — gold (newer note) says "16 tests
     passed"; the rank-1 sibling (older note) says "10 passed" for the same test
     file. Two answers, one arbitrary gold.
2. **Only 3 of 14 rank-1 misses were adjudicated**, so the verified non-error
   share is 21 % (3/14). The 76 % UUID-redundancy rate above is an upper-bound
   signal, not a measured miss share — but even the verified 21 % is comparable
   to the biggest model-side gains in §4.
3. **Answer delivery ≠ document hit.** For the relational set, 39/56 gold docs
   reach the top-10 chunk window, but only **31/56 gold *evidence chunks* do**
   (51/56 evidence chunks exist in the index, 41/56 are in the pool). The
   doc-level metric overstates answer delivery by ~14 %.
4. **Resolution.** n=48 with r@1≈0.71 gives a Wilson 95 % CI of
   **[0.568, 0.818]**; the smallest detectable difference at p≈0.7 is
   **±12.5 pts**. n=28 relational is ±15.9 pts. The 48q set is 36 identifier /
   11 fact / 1 decision questions, generated *from* the documents: BM25 finds the
   gold in its top-3 for 45/48. It cannot measure ambiguous, short, or real
   agent queries. All 1,013 training queries are single-gold, so the label
   format itself bakes in the identity assumption.

---

## 3. Headroom per class (upper bounds, computed)

All bounds below are computed by re-fusing the **saved per-channel scores** of
the actual candidate pools (no new retrieval, no guessing), or read from the
existing measurements.

### 3.1 Fusion weights (oracle α re-fusion)

| index / set | α=0.5 (deployed) | best fixed α | leave-one-out α | per-query oracle α |
|---|---|---|---|---|
| base / 48q | 0.7083 | **0.7917** @ 0.15 | **0.7917** | 0.8125 |
| base / 28q | 0.2857 | 0.3393 @ 0.10 | 0.3036 | 0.3571 |
| new / 48q | 0.6042 | 0.6042 @ 0.05 | 0.5833 | 0.6042 |
| new / 28q | 0.2143 | 0.2143 @ 0.45 | 0.1607 | 0.2679 |

Recall@10 moves 0.6964 → 0.75 on base/28q (α=0.10); nothing on either 48q set.
**Upper bound from perfect per-query weighting: +10.4 pts r@1 (48q), +7.1 pts
(28q), and exactly 0.0 on `md_struct`.**

### 3.2 Perfect ranker inside the existing pool

| index / set | gold in pool | r@1 ceiling | r@10 ceiling |
|---|---|---|---|
| base / 48q | 48/48 | 1.000 | 1.000 |
| base / 28q | 47/56 | 0.500 (two gold docs; 0.5 is structural) | 0.839 |
| new / 48q | 33/48 | 0.688 | 0.688 |
| new / 28q | 31/56 | 0.500 | 0.554 |

So a perfect ranker is worth at most +19 pts r@1 on base/48q, +21 pts on
base/28q — but §1.4 shows the shipped 23-feature model captures +5 of those 14,
and a single scalar captures +4 of them. The remaining 9 are near-identical
documents competing for the same query; no monotone reweighting of the two
channels separates them.

### 3.3 Graph traversal

| | base | new |
|---|---|---|
| questions whose 3 expected entities exist in the graph | 28/28 | 2/28 |
| bridge entity is attached to **both** gold docs | 28/28 | 3/28 |
| 1-hop expansion seeded with `expected_entities` contains both gold docs | **28/28** | 5/28 |
| 1-hop fan-out (distinct docs) | median 18, max 260 | median 0 (unresolvable seeds) |

Seeding is oracle-assisted: the gold `expected_entities` are taken as the
linked entities, so this bounds what a correct entity linker could do, not what
the current linker does. NEW's absolute number is dominated by its missing
documents (21/28 questions lack a gold doc). Conditional on both gold docs
existing (7 questions), bridge linkage is 4/7 — directional evidence of
graph/entity-extraction drift under `md_struct`, on n=7, not proof.

**Base headroom: r@10 0.6964 → up to 1.000; r@1 0.2857 → up to 0.500.**
The graph already contains the path for every question; what is missing is a
scored channel that traverses it and a ranker that can place two documents from
an 18-document fan-out.

### 3.4 Chunking

* Measured on the matched 33-question subset (published report): fine
  `md_struct` chunks give **+15.2 pts r@1** for fusion (0.7273 → 0.8788) and only
  +6.1 pts for LTR — the ranker trained on legacy chunks gives back most of the
  gain.
* Structural cost of fine chunks under the published metric: **8.83 distinct
  docs per 10-chunk window on base vs 7.33 on NEW** (relational: 7.54 vs 6.68).
  At full `md_struct` coverage (~7.65 chunks/doc) the top-10-chunk convention
  silently truncates ~1.5 documents per query.
* Answer-level: 31/56 evidence chunks delivered vs 39/56 docs (base).

### 3.5 Reranker (existing measurement, `rerank_off_vs_on.json`)

| | off | on | Δ |
|---|---|---|---|
| recall@1 | 0.7083 | 0.7500 | **+0.0417 (2 queries)** |
| recall@10 | 0.9792 | 0.9583 | **−0.0209 (1 query)** |
| NDCG@10 | 0.8563 | 0.8605 | +0.0042 |
| latency | 36.7 ms | 5 919 ms | **×161** |

---

## 4. Candidate investments

Each item: measured/lower bound, ceiling, cost, risk, first experiment.

### A. Ship the ranker that already exists (highest value / lowest cost)
* **Expected gain:** +10.4 pts r@1, +2.1 pts r@10, +5.2 pts NDCG@10, +6.4 pts
  MRR on 48q; +1.8 pts r@1 / +1.8 r@10 on 28q. Measured now, artifact committed.
* **Ceiling:** +19 pts r@1 on 48q (perfect ranking inside the pool); no new
  compute needed.
* **Cost:** 1–2 days engineering (wire `apply_ltr_ranker` into
  `_compute_hybrid_block`; 29 ms/query of ranker scoring measured, fail-soft already
  implemented). No GPU, no data.
* **Risk:** the ranker was trained on *synthetic* questions from the same
  generator family; its held-out MRR gain on that distribution is +4.3 pts
  (0.881 → 0.925), much smaller than the +6.4 pts MRR on 48q. It demotes 2
  queries to fix 7. It is inert on `md_struct` and must be retrained if chunking
  changes. Mitigation: gate on `r@10` non-decreasing, keep α tuned, and retrain
  before the index switch.
* **First experiment:** feature-flag the call in `_compute_hybrid_block`, A/B
  `fusion` vs `ltr_full` on both gold sets with the existing harness; require
  r@10 ≥ fusion and no per-type regression. Then run a 30-query sample of real
  agent queries to check the synthetic-distribution risk.

### B. Repair the labels and the metric (highest enabling value)
* **Expected gain:** removes 3 confirmed false negatives on 48q (**+6.3 pts
  r@1** measured); more may follow from the multi-gold rewrite. Shrinks the CI so
  3–5 pt deltas become visible.
* **Cost:** 1–2 days human adjudication + a script to auto-add alternative gold
  docs (all docs containing the answer UUID/ID), plus adding an
  "evidence-chunk in window" metric alongside doc recall. No compute.
* **Risk:** none to production; the risk is *not* doing it — you will ship
  changes that move the metric but not the user experience.
* **First experiment:** re-score the current production run with multi-gold
  labels generated from answer-token containment and see how many of the 14
  rank-1 "misses" disappear; publish the delta before and after.
### C. Complete `md_struct` coverage + adopt fine chunks
* **Expected gain:** for the NEW index, 48q r@10 0.6875 → up to 1.000, r@1
  0.6042 → ~0.8788 at the matched-subset rate (**+27.5 / +31.3 pts**); vs
  today's *production+ranker* the honest increment is **+6.6 pts r@1** (0.8788
  vs 0.8125), r@10 equal. Relational availability-normalised r@10 0.839 vs 0.696
  (+14 pts), but the matched 28q subset is n=7 and cannot support that claim.
* **Ceiling:** 1.0 r@10 (all gold present) on 48q; relational r@10 ≤ 1.0.
* **Cost:** batch-extract the 635 missing docs (already in flight by another
  agent), re-run entity/relation extraction, rebuild both indexes. Days of
  compute, no GPU contention if batched.
* **Risk:** the matched subset is not random (it is the docs NEW happens to
  hold) and is therefore optimistic; entity-graph drift (4/7 bridge coverage vs
  28/28) may hurt the relational class; the 10-chunk window penalises 7.65
  chunks/doc.
* **First experiment:** run the identical gate on the full-corpus `md_struct`
  index **before** touching the ranker, with the metric fixed to dedupe-then-
  top-10-docs; require the matched-subset gain to survive at full coverage.

### D. Retrain / ablate the ranker on `md_struct` (and on all 1,013 labels)
* **Expected gain:** restores the +9 pts that the legacy ranker provides but is
  currently **0.0** on NEW. Honest incremental over a retuned α on the legacy
  index is **+2.1 pts r@1** (1 query) — so the case for this investment rests on
  the index switch, not on the current corpus.
* **Ceiling:** +19 pts r@1 on 48q with perfect in-pool ranking; realistically
  +3–6 pts from feature work (answer locality, hop distance, supersession),
  since in 8/14 rank-1 misses the gold already has a higher `content_term_overlap` than
  gold than for the winner — more of the same feature cannot help.
* **Cost:** CPU-only LightGBM, ~1 hour per training run. The labels already
  exist (1,013 queries: 673 identifier / 274 fact / 66 decision) but only
  223 train + 74 valid were used for the shipped artifact.
* **Risk:** synthetic-label overfit; feature distribution shift every time
  chunking changes; adding features without ablation repeats the current
  situation where the model is worth +1 query over a scalar.
* **First experiment:** (1) export features from `md_struct`, retrain on the
  existing 1,013 labels, run the 48q/28q gate; (2) leave-one-feature-out
  ablation of the 23 columns on the 48q set to find harmful features (doc age,
  doc length, section depth are the suspects for the 5 unfixable
  "gold matches better" misses).

### E. A real graph channel (depth-1/2, scored, fused) for multi-hop
* **Expected gain:** base 28q r@10 0.6964 → up to 1.000 (28/28 reachable);
  realistic 0.85 (+15 pts) once hub damping and a ranker are in place. r@1
  0.2857 → up to 0.500, realistic 0.35–0.40 (+6–11 pts). Zero effect expected on
  48q (single-hop, already 48/48 in pool) — that is the regression test.
* **Ceiling:** the two ceilings above; both computed from the actual graph, not
  estimated.
* **Cost:** 1–2 days engineering, <100 ms/query, no GPU, no new data. The
  dossier and `antigravity_review_real.md` already specify the design
  (`G_chunks`: entity/relation vector search → 1-hop NetworkX → provenance
  chunks → WRRF with `w_g≈0.10`, hub damping by `1/ln(1+|chunks(e)|)`).
* **Risk:** 28 questions is a small, synthetic, chain-shaped set; hub entities
  (median fan-out 18, max 260) can flood the top-10; entity linking on the
  question text is the real failure mode (the gold set supplies `ll_keywords`,
  production clients must supply them too).
* **First experiment:** offline only, on the base index: seed with the
  question's identifiers → entities, expand 1 hop, score chunks, fuse as a third
  RRF channel; measure 28q r@10/r@1 **and** 48q regression. No service, no GPU.

### F. Fusion tuning (`alpha`)
* **Expected gain:** +8.3 pts r@1 on base/48q (LOO-validated), +5.4 pts r@1 /
  +5.4 pts r@10 on base/28q at α=0.10. **0.0 pts on `md_struct`.**
* **Cost:** one constant + a re-run. Hours.
* **Risk:** it is a corpus artefact (dense dilutes long legacy chunks); it is
  redundant with the ranker (6 of 7 fixes overlap) and would be wiped out by
  the index switch. Do not let it substitute for B/C/D.
* **First experiment:** A/B α ∈ {0.10, 0.15, 0.20} × ranker on/off on both gold
  sets; ship only if the ranker is not shipped first.

### G. More labelled queries (which strata)
* **Expected gain:** n=300 shrinks the 48q half-width from ±12.5 to ±5.0 pts —
  the minimum needed to adjudicate the 3–6 pt changes above. The strata that
  matter, in order: (1) **multi-gold / answer-level** questions (currently 0 of
  1,013 training and 0 of 48 eval questions have more than one gold doc);
  (2) **fact and decision** questions (11 + 1 of 48 today; identifier questions
  are the easy, lexically-saturated majority); (3) **real agent queries and
  judged answers** — the current sets are generated from documents and cannot
  measure query understanding; (4) **multi-hop** beyond the 28 synthetic chains.
* **Cost:** low for generation (the generator already produced 1,013 in one
  session), medium for real-query harvesting + judging.
* **Risk:** more of the same generator adds confidence in the wrong distribution
  — the shipped ranker's +4.3 pt held-out MRR vs +6.4 pt on 48q is the warning.
* **First experiment:** generate 250 fact/decision questions with 2–3 gold docs
  each from answer-token containment, hold out 100, and re-measure the ranker's
  marginal value over α.

### H. Reranker — **reject**
* +2 rank-1 hits, −1 top-10 hit, +0.004 NDCG, 5.9 s/query (161× the fusion
  path). The RCA documents the 0.9996-vs-0.9998 saturation flipping a correct
  #1 to a 7.5k-char wrong note and amputating the gold from the context window.
  Even a perfectly optimised version can recover at most the 2 queries that
  remain after α/ranker work, at 100–1000× the latency.
* Conditional salvage only: a CPU cross-encoder over the top-20 with a
  margin guard, measured on the 5 residual "rank competition" misses.

### I. Query expansion / rewriting — **reject for this corpus (revisit with real queries)**
* In 8 of the 14 rank-1 misses the **gold document has strictly higher
  content-term overlap with the query than the winner**; in the other 6 it is
  equal. The winner is never the better match (0/14). BM25 already returns the gold in its top-3 for 45/48. The query
  is not under-specified relative to the corpus; expansion would amplify the
  gold and its near-duplicate competitor together.
* The caveat is a sampling one: a document-generated query set cannot contain
  the vague, referential, or temporal queries where rewriting pays. Fund (G)
  before (I).

### J. Deduplication — **no work needed**
0 exact-content duplicate documents in both indexes (1,624 and 989 docs).
Title-level repetition exists (20 vs 121 docs sharing a heading) but it is
section-structure, not duplicated content.

### K. Recency / supersession priors — **not a global lever**
Among the 9 date-resolvable rank-1 misses on base/48q: 3 have a *newer* gold,
2 an *older* gold, 4 same-day. A global recency discount would net +1 query and
`fusion.py` documents a reproducible regression from exactly this (identifier
recall@1 0.602 → 0.560). Supersession is worth it only for explicit
same-fact-conflict documents (the PACO "10 vs 16 passed" case).

### L. A better embedding model — **no evidence, defer**
The dense channel is the weaker one (absent from top-30 for 7/48 golds vs 0/48
for BM25) but the failures are not paraphrase failures: 45/48 golds are in
BM25's top-3 and the gold is never a *worse* lexical match than the winner (8 better, 6 tied)
winner. A stronger embedder does not separate near-identical project notes. Do
not spend GPU on this before (B) and (E).

---

## 5. Ranked plan (expected lift, effort, confidence)

| # | Investment | Expected lift (measured / bound) | Effort | Confidence |
|---|---|---|---|---|
| 1 | **Ship the existing LTR ranker** (A) | +10.4 pts r@1, +2.1 r@10 on 48q; +1.8/ +1.8 on 28q; 29 ms/query scoring | 1–2 d eng, 0 GPU | High (artifact measured); medium on real-query distribution |
| 2 | **Repair labels/metric** (B): multi-gold, answer-level, adjudicate | ≥6.3 pts of the 48q gap is phantom; CI ±12.5 → ±5 with (G) | 1–2 d human | High (3 verified cases, 13/17 redundancy) |
| 3 | **Complete `md_struct` + fine chunks** (C) | +27.5 r@1 / +31.3 r@10 for NEW; +6.6 r@1 over shipped production+ranker | batch ingest (in flight) + rebuild | Medium (matched subset optimistic) |
| 4 | **Graph channel, depth-1/2, WRRF** (E) | 28q r@10 +15 pts (ceiling +30), r@1 +6–11 pts (ceiling +21) | 1–2 d eng, CPU, 0 GPU | Medium (28 reachable, n=28) |
| 5 | **Retrain/ablate ranker on `md_struct` + all 1,013 labels** (D) | restores the +9 pts lost on NEW; +2.1 over tuned α today | ~1 h/run, CPU | Medium (synthetic labels) |
| — | *Opportunistic, only if 1 slips:* α 0.5→0.15 (F) | +8.3 pts r@1 on legacy 48q | hours | High, but legacy-only |
| — | *Enabler for 2/5:* 250 stratified queries (G) | makes 3–5 pt deltas measurable | 1 d generate + judge | High |

Sequencing rationale: (1) is a finished asset with a measured 10-point gain and
no data/compute cost. (2) is next because it is the *instrument* used to accept
or reject 3–5, and it is currently biased by at least 6 pts. (3) is the only
path that beats today's production number rather than merely closing the
`md_struct` gap. (4) is the only large headroom that needs no new corpus. (5)
must ride with (3), otherwise the switch loses the ranker's +9 pts.

**Do (1) and (2) in parallel, then (3), then (4), then (5).** If the goal is
strictly "accuracy on the current production index", stop after (1) + (2): the
remaining headroom on 48q is 9 queries, 3 of which are label artifacts, and no
channel reweighting can separate the other 6.

---

## 6. What NOT to invest in (with the evidence)

| Not this | Why (measured) |
|---|---|
| **Cross-encoder reranker** | +2 rank-1 / −1 top-10 for 5.9 s/query (161×); saturated scores; documented catastrophic flip (RCA Q05). |
| **Query expansion / rewriting on these sets** | Gold is never a worse lexical match than the winner (8 better, 6 tied, 0 worse); BM25 gold-in-top-3 45/48. No headroom. Revisit only with real agent queries. |
| **Deduplication / corpus GC** | 0 exact-content duplicate docs in either index. |
| **Global recency prior** | 3 newer vs 2 older vs 4 same-day among decidable misses; already measured to regress identifier recall in `fusion.py`. |
| **A bigger/better embedding model** | Dense is the weaker channel but the failures are near-duplicate discrimination, not paraphrase; 0/8 misses are lexical-gap cases. |
| **Parent-document retrieval on the legacy index** | Chunks are already 3.7k chars (median 2.2k); the parent is the whole note (10–150k chars) and would recreate the dilution that `md_struct` fixes. |
| **Tuning the metric window alone** | Widening 10→20 chunks recovers 1 query on base/48q and 0 on NEW. Useful hygiene at full `md_struct` coverage, not an investment. |
| **Serving `rag.aquery` / LightRAG mix** | 7.6–79 s/query at 0.688–0.792 recall@10 vs 12.7 ms at 0.979 for fusion (dossier §6.1, review #1). |

---

## 7. Caveats / what this analysis does not measure

1. The offline gate scores **documents**, not answers, and not context-window
   delivery. The RCA showed a case where retrieval was perfect and the answer
   was still lost to assembly/truncation. This memo cannot see that class.
2. The graph and ripgrep channels are **absent from the offline numbers**
   (§0); production behaviour on identifier and multi-hop queries is therefore
   *better* than these numbers in an unmeasured way, and no claim here about
   graph usefulness is a production measurement.
3. The relational set's evidence chunk ids are base-index-specific, so
   evidence-chunk reachability could not be computed on `md_struct`.
4. Only 3 of 14 rank-1 misses were hand-adjudicated; the rest of the taxonomy is
   mechanical from channel ranks.
5. `headroom.json` per-query-oracle α is an in-sample upper bound; the LOO
   column is the honest one.
6. Reranker numbers are the 2026-10-17 measurement (the only one available);
   the reranker is opt-in and off by default in the current code.
7. The `md_struct` index is being actively extended by the batch-ingest work
   (another agent owns it). All NEW numbers here are a snapshot at 2026-10-19
   23:20 of a 990-document / 989-processed index; re-run `trace_misses.py` after
   the batch append before quoting them.

---

## 8. Commands and artifacts

All commands are CPU-only, read-only on the indexes, and run with
`.venv/bin/python` from the repo root.

```bash
# environment used by the published gate
set -a; source eval_agentic-data/bench_env.sh; set +a

# 1. per-query trace (base) — reproduces 0.7083/0.8125 on 48q, 0.2857/0.3036 on 28q
export HARS_MEMORY_INDEX_DIR=$HOME/.local/share/hars-graphrag/index_compact_20260903_imported
export HARS_MEMORY_BM25_CACHE_DIR=/mnt/datasets/ft/eval_new_vs_base/bm25_base
.venv/bin/python eval_agentic/rca/miss_analysis/trace_misses.py \
  --label base --out eval_agentic/rca/miss_analysis/traces/trace_base.json \
  --candidates eval_agentic/rca/miss_analysis/traces/candidates_base.jsonl

# 2. same for md_struct
export HARS_MEMORY_INDEX_DIR=/mnt/datasets/ft/index_local/index_full_mdstruct_20261014
export HARS_MEMORY_BM25_CACHE_DIR=/mnt/datasets/ft/eval_new_vs_base/bm25_new
.venv/bin/python eval_agentic/rca/miss_analysis/trace_misses.py \
  --label new --out eval_agentic/rca/miss_analysis/traces/trace_new.json \
  --candidates eval_agentic/rca/miss_analysis/traces/candidates_new.jsonl

# 3. oracle-alpha re-fusion over the saved candidates
for L in base new; do
  .venv/bin/python eval_agentic/rca/miss_analysis/headroom_analysis.py --label $L \
    --trace eval_agentic/rca/miss_analysis/traces/trace_$L.json \
    --candidates eval_agentic/rca/miss_analysis/traces/candidates_$L.jsonl \
    --out eval_agentic/rca/miss_analysis/traces/headroom_$L.json
done

# 4. graph reachability of the 28 relational questions
.venv/bin/python eval_agentic/rca/miss_analysis/graph_reachability.py --label base \
  --index-dir $HOME/.local/share/hars-graphrag/index_compact_20260903_imported \
  --out eval_agentic/rca/miss_analysis/traces/graph_base.json
.venv/bin/python eval_agentic/rca/miss_analysis/graph_reachability.py --label new \
  --index-dir /mnt/datasets/ft/index_local/index_full_mdstruct_20261014 \
  --out eval_agentic/rca/miss_analysis/traces/graph_new.json

# 5. miss table + taxonomy + headroom
.venv/bin/python eval_agentic/rca/miss_analysis/build_report_data.py
```

| file | contents |
|---|---|
| `miss_table.csv` | 208 rows = (index × set × query × gold doc): ranks, per-channel ranks, class, sub-cause, ranker effect, adjudication |
| `taxonomy.json` | class/sub-cause definitions, distributions at gold-entry and query level, adjudicated rows |
| `headroom.json` | α grid, LOO and oracle α, in-pool ceilings, graph reachability, reranker trade-off, Wilson intervals |
| `traces/trace_{base,new}.json` | per-query channel hit lists, doc orders, LTR orders, index stats |
| `traces/candidates_{base,new}.jsonl` | every candidate with 23 features, fused and LTR scores (gitignored, 5.5 MB; regenerate with steps 1–2) |
| `traces/headroom_*.json`, `traces/graph_*.json` | intermediate outputs |
