# `G_chunks` — hub-damped graph/citation bridge channel: measurement report

2026-10-20. Repo `hars-longterm-memory`, CPU only, production index
`~/.local/share/hars-graphrag/index_compact_20260903_imported`, read-only (no
`apply`, nothing written to either index). Every number below is produced by the
three scripts in this directory and stored in
[`metrics.json`](metrics.json) / [`diagnostics.json`](diagnostics.json) /
[`rank_experiment.json`](rank_experiment.json); the command for each table is in
§10.

---

## 0. Verdict

**It does not beat the tuned baseline on the metrics the project gates on, and
it is not worth its latency. Leave it off (the default).**

One number does move in the channel's favour: on `hard_1k` it adds **+3.0 pts of
answer-in-window** with a paired 95 % CI that excludes zero. Everything else is
flat: recall@10/NDCG@10 have CIs that straddle zero, MRR is 0.07 pts lower, and
on 48q/28q the rankings are byte-identical to the baseline. That is not a win —
it is 3 points out of the 53.7-point availability bound, bought with 16 ms mean /
43 ms p95 per query.

The channel was built exactly as specified — hub-damped 1-hop expansion seeded
by the top-3 anchor chunks, provenance scoring over `kv_store_entity_chunks`,
flat citation/rare-UUID bonus — and measured against the current tuned config
(`top_k=12`, `alpha=0.15`, `pool multiplier 2`).

| set | metric | baseline | `fusion_graph` | Δ | paired 95 % CI |
|---|---|---:|---:|---:|---|
| 48q (n=48) | recall@1 / @10 | .8125 / 1.0000 | .8125 / 1.0000 | **0.00 / 0.00** | — (identical ranking) |
| 28q (n=28) | recall@1 / @10 | .3393 / .7500 | .3393 / .7500 | **0.00 / 0.00** | — (identical ranking) |
| hard_1k (n=300) | recall@10 | .6772 | **.6883** | **+1.11 pts** | [−0.17, +2.44] *includes 0* |
| hard_1k | NDCG@10 | .6336 | **.6389** | +0.53 | [−0.05, +1.16] *includes 0* |
| hard_1k | **answer in window** | .4200 | **.4500** | **+3.00 pts** | [+0.67, +5.67] *excludes 0* |
| hard_1k | all@10 (both gold docs) | .4000 | .4267 | +2.67 | — |
| hard_1k | MRR | .8096 | .8089 | −0.07 | [−0.18, +0.01] |

Read it plainly:

* On the two single-hop / relational sets the channel changes **nothing** — the
  rankings are byte-identical to the baseline. It cannot help where the gold
  document already shares the question's vocabulary.
* On `hard_1k` it buys **+3.0 pts of answer-in-window** (the product metric from
  `root_cause/REPORT.md` §10) and +1.1 pts recall@10, at **−0.07 pts MRR**. Only
  the answer-in-window interval excludes zero; recall@10 and NDCG@10 do not.
* Against the **+53.7 pts** availability bound measured in §9 of the root-cause
  report, this delivers **3 points — 5.6 % of the headroom**. §6 explains
  exactly where the other 50.7 points are lost.
* Cost: **16.5 ms mean / 42.6 ms p95** marginal latency per query (in-situ, warm
  process), against a <15 ms budget. The budget holds on the mean and is missed
  at p95.

Per the brief, the channel ships **OFF** (`HARS_MEMORY_GRAPH_BRIDGE=0` is the
default and stays that way) with the code, tests and this measurement in place
for a later ranker to build on.

---

## 1. What was built

`hars_memory/retrieval/graph_bridge.py` — a pure-CPU, read-only channel over the
stores the graph branch already writes, in the shape the review's Rank 1
(`antigravity_tuning_review_v1.md` §2) proposed:

| stage | implementation |
|---|---|
| **seeds** | (a) question: exact case-folded, word-boundary entity-name matches against the `kv_store_full_entities.json` vocabulary; (b) anchors (top-3 of the already-fused pool): the anchor **document's extracted entity roster**, plus regex-extracted `.md` filenames, rare UUIDs (document frequency ≤ 5) and run/experiment ids (`A2S124A`) that resolve to a known entity |
| **expansion** | `graph_chunk_entity_relation.graphml` via NetworkX; hub cap `fanout ≤ 80` (skip schema super-hubs), decay `0.6` per hop, 2-hop only through nodes with graph `degree ≤ 12` |
| **scoring** | `Σ_e weight(e) / ln(1 + fanout(e))` over `kv_store_entity_chunks`, weight `1.0` for question-named seeds / `0.9` for anchor-roster seeds / `0.6^hop` for neighbours, `+0.50` for a chunk whose document is cited by filename inside an anchor or that shares a rare UUID with one |
| **output** | up to 20 `FusedChunk`-shaped candidates (`GraphBridgeChunk`, `channel="graph_bridge"`), min-max normalised and scaled by `HARS_MEMORY_GRAPH_BRIDGE_WEIGHT` (default **0.2**) so they merge into a fused pool whose scores live in `[0, 1]` |

Integration (`mcp_server._compute_hybrid_block`): the channel runs **after
fusion and before the ripgrep gate / diversity / truncation**, merges with
`max(existing_score, bridge_score)` (existing entries are never replaced,
provenance fields preserved), and is **fail-soft**: any exception contributes
nothing, is logged once per process, and never breaks the query. 31 tests are in
`tests/test_graph_bridge.py` (synthetic graph, no network; 29 unit tests plus two
end-to-end calls through the real `_compute_hybrid_block`, where the flag-OFF
baseline cannot see the one-hop document in a 2-chunk window and the flag-ON
channel can).

### Three deviations from the sketch, and why

Each is a measured, documented decision — not an unrecorded re-tune.

1. **Anchor seeding uses the document roster, not a re-scan of the chunk text.**
   The 85.7 % availability measurement (§9) seeded with the *document's* entity
   roster; extraction is document-level, so the roster is the union over the
   document's chunks. Matching every globally known entity name in the anchor
   text additionally pulls in one-off entity names that happen to be common
   English words (`before`, `Action`, `staging`) — measured on 30 random chunks:
   84 % of text matches are NOT in that document's roster. The text scan is
   kept as an automatic fallback for a document with no extraction roster (and
   is available behind `include_anchor_text_entities=True`).
2. **The citation/rare-UUID bonus is applied on a normalised scale.**
   The entity sum on this corpus is not O(1): 200–400 seeds accumulate, so raw
   scores reach 10–30 and a flat `+0.50` is invisible — measured: **0 of 300**
   `hard_1k` queries had a cited chunk inside the 20-candidate cut with the
   literal formula. Dividing by the query's best entity score first puts chunks
   in `[0, 1]`, where 0.50 is the decisive boost the sketch intends
   (`rank_experiment.json`: answer-in-top-20 for actionable queries 15.1 % →
   21.5 %).
3. **The merge sorts stably on score only.** Re-sorting on `(-score, chunk_id)`
   silently swapped two equal-scored (0.85) duplicate documents on 48q and cost
   2 rank-1 queries — a merge artefact that initially looked like a −4.2 pt
   channel regression. Stability keeps `fuse()`'s own tie-break intact; with it,
   48q/28q are exactly baseline.

---

## 2. Method

* Baseline = current tuned config, as specified: `--top-k 12 --alpha 0.15
  --pool-multiplier 2` (pool 24 per channel). Production index, production
  `fuse()`, production `apply_ripgrep_gate` with `HARS_MEMORY_RIPGREP_ROOTS`
  pointed at the corpus (so `fusion_rog` is a live channel, not a no-op).
* Variants per set: `fusion` (baseline ranking), `fusion_rog` (production
  default), `fusion_graph` (baseline + bridge), `fusion_rog_graph` (bridge then
  the ripgrep gate).
* Metrics: `recall@1/@5/@10`, `ndcg@10`, `MRR` over the first `top_k=12`
  **chunks**, deduped to documents (the published convention); plus `any@10`,
  `all@10` and `answer_in_window` (literal, whitespace-normalised,
  case-folded substring containment of the `answer` field in the returned
  chunks) for `hard_1k`.
* Intervals: **paired bootstrap** (10 000 resamples, fixed seed, 95 %), which is
  the only honest interval on per-query paired deltas at n=48/300 (`root_cause`
  §11.7: Wilson half-widths are ±12.8 / ±5.4 pts).
* Cross-check: the repository's own gate (`hars_memory.eval.ltr_eval`, now
  carrying `fusion_graph` / `fusion_rog_graph`) reproduces **every**
  `fusion` and `fusion_graph` number in §3 to the digit on all three sets —
  `ltr_gate.json` vs `metrics.json`. The harness is not a parallel
  implementation with its own definition of the metric.

---

## 3. Headline results (all four variants)

### 48q — `retrieval_queries_current_gcfixed.yaml` (n=48)

| variant | r@1 | r@5 | r@10 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|
| fusion (baseline) | **.8125** | .9792 | 1.0000 | **.9119** | **.8828** |
| fusion_rog | .8125 | .9792 | 1.0000 | .9119 | .8828 |
| fusion_graph | .8125 | .9792 | 1.0000 | .9119 | .8828 |
| fusion_rog_graph | .8125 | .9792 | 1.0000 | .9119 | .8828 |

Identical. The bridge fires on 100 % of queries and still changes no ranking —
its 4.0 injected chunks per query all land below the window's content. (At
`weight=0.5` it starts costing r@10; see §5.)

### 28q — `relational_gold.yaml` (n=28)

| variant | r@1 | r@5 | r@10 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|
| fusion (baseline) | .3393 | .6964 | .7500 | .6899 | .7512 |
| fusion_rog | .3393 | **.7143** | .7500 | .6886 | **.7571** |
| fusion_graph | .3393 | .6964 | .7500 | .6899 | .7512 |
| fusion_rog_graph | .3393 | **.7143** | .7500 | .6886 | .7571 |

`fusion_graph` ≡ `fusion`; `fusion_rog_graph` ≡ `fusion_rog`. Nothing gained,
nothing lost.

### hard_1k — `hard_1k_eval.yaml` (n=300)

| variant | r@1 | r@5 | r@10 | NDCG@10 | MRR | any@10 | all@10 | **answer in window** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fusion (baseline) | **.3600** | .6044 | .6772 | .6336 | **.8096** | .9600 | .4000 | .4200 |
| fusion_rog | .3350 | .6028 | .6783 | .6219 | .7827 | .9500 | .4100 | .4367 |
| **fusion_graph** | **.3600** | **.6094** | **.6883** | **.6389** | .8089 | .9533 | **.4267** | **.4500** |
| fusion_rog_graph | .3383 | .6011 | .6861 | .6268 | .7857 | .9500 | .4233 | **.4500** |

Paired deltas vs the matched baseline (bootstrap, 10 000 resamples):

| comparison | metric | Δ pts | 95 % CI |
|---|---|---:|---|
| `fusion_graph` − `fusion` | recall@1 | 0.00 | [0.00, 0.00] |
| | recall@10 | +1.11 | [−0.17, +2.44] |
| | NDCG@10 | +0.53 | [−0.05, +1.16] |
| | MRR | −0.07 | [−0.18, +0.01] |
| | **answer in window** | **+3.00** | **[+0.67, +5.67]** |
| `fusion_rog_graph` − `fusion_rog` | recall@1 | +0.33 | [0.00, +0.83] |
| | recall@10 | +0.78 | [−0.56, +2.11] |
| | answer in window | +1.33 | [−1.33, +4.00] |

---

## 4. Per-stratum delta on hard_1k

`fusion` → `fusion_graph`, chunk window 12 (Δ in points):

| stratum | r@1 | r@5 | r@10 | NDCG@10 | MRR | answer in window |
|---|---:|---:|---:|---:|---:|---:|
| **multi_hop** | 0.0 | **+2.2** | **+5.3** | **+2.6** | 0.0 | **+10.6** |
| **lexical_mismatch** | 0.0 | +1.6 | +2.6 | +1.3 | −0.3 | **+9.4** |
| **numeric** | 0.0 | +1.2 | +2.3 | +1.0 | 0.0 | **+4.7** |
| disambiguation | 0.0 | −1.3 | −1.3 | −0.5 | 0.0 | 0.0 |
| temporal | 0.0 | 0.0 | −1.7 | −0.6 | 0.0 | 0.0 |
| relational | 0.0 | 0.0 | 0.0 | 0.0 | −0.3 | −2.3 |
| implicit_reference | 0.0 | 0.0 | 0.0 | −0.1 | 0.0 | 0.0 |
| negation | 0.0 | 0.0 | 0.0 | −0.1 | 0.0 | 0.0 |

This is the shape the mechanism predicts: the three strata whose second hop is a
*bridge* (multi_hop, lexical_mismatch, numeric — the same strata `root_cause` §1
identified as carrying the second-hop failure) gain; the two strata whose
failure is a within-document disambiguation/ordering problem (disambiguation,
temporal) lose a little r@10 and gain nothing. Even the winning strata gain
+5 pts of r@10, not the +40–50 the availability bound suggested.

---

## 5. Weight / limit sweep (the channel is placement-sensitive)

`fusion_graph`, hard_1k, n=300 and 48q, n=48 (full grid in `metrics.json → sets.*.sweep`):

| injection weight | limit | hard_1k r@10 | hard_1k answer | 48q r@10 | 48q NDCG@10 |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 20 | .6839 | .4333 | 1.0000 | .9119 |
| 0.15 | 20 | .6889 | .4333 | 1.0000 | .9119 |
| **0.20** | 20 | **.6883** | **.4500** | 1.0000 | .9119 |
| 0.25 | 20 | .6767 | .4533 | 1.0000 | .9119 |
| 0.50 | 20 | .6672 | .4267 | .9792 | .9029 |
| 1.00 | 20 | .5306 | .2500 | .9792 | .6667 |

(`limit` 5 vs 20 changes nothing at weight ≤ 0.25: at a low weight the 5th–20th
candidates fall below the window anyway.) The channel has a **narrow working
band**: below 0.15 it is too weak to enter the window, above 0.25 it starts
evicting correct chunks faster than it adds the answering one, and at 1.0 it
destroys the ranking. 0.2 is the measured best and the shipped default.

---

## 6. Why the +53.7 pt bound is not reached — measured

`diagnose_bridge.py` on all 300 `hard_1k` queries, per query:

| quantity | value |
|---|---:|
| answer document has at least one scored provenance chunk | **.7467** |
| answer document lands inside the **20-candidate** cut | **.2467** |
| answer document inside the candidates but **not** in the baseline window | **.1267** |
| median rank of the answer document among scored chunks | **49.5** |
| mean scored chunks per query | **1509.1** |
| answer chunks carrying the citation/rare-UUID bonus (when found) | .5270 |

So the funnel is: **75 % reachable → 25 % inside the candidate cut → 12.7 %
actionable → 3 pts delivered.** The channel finds the answering document in
three quarters of the queries, then loses it against ~1 500 competitors that are
connected to the anchor through *more* (generically named) entities. This is
precisely the failure the root-cause report predicted: "the graph channel needs
a ranker that can place 2 docs out of hundreds" (§10).

`rank_experiment.py` holds the traversal fixed and tries six aggregations and an
expansion grid; the best candidate-ranking it finds puts the answer in the top 20
for **24.4 %** of the actionable queries (`hub cap 10, 2 hops, bonus-first`
ordering) vs **15.1 %** for the review's raw `sum` formula (note: that
experiment scores the *raw* formula before the normalised-bonus fix, which is
why the shipped channel now reaches 24.7 % in `diagnostics.json`):

| aggregation (default expansion) | answer in top-20, actionable queries | median rank |
|---|---:|---:|
| `sum` (review formula) | .1512 | 71.0 |
| `max` | .1512 | 88.0 |
| `mean` | .1047 | 104.5 |
| `sqrt` (sum / √n) | .1395 | 62.0 |
| `bonus_first` | **.2151** | 54.0 |
| `max_bonus` | .2209 | 71.5 |

| expansion (`bonus_first`) | answer in top-20, actionable | median rank | median scored chunks |
|---|---:|---:|---:|
| hub 10, hop 2 | **.2442** | 46.5 | 1 010 |
| hub 10, hop 1 | .2326 | 40.0 | 925 |
| hub 20, hop 2 | .2151 | 51.5 | 1 165 |
| hub 40, hop 2 | .2035 | 51.0 | 1 377 |
| hub 80, hop 2 (shipped) | .2151 | 54.0 | 1 540 |

The ceiling of this family of scoring functions is therefore ≈ **+4 pts of
answer coverage** (24 % of the 57 % actionable queries) — consistent with the
+3 pts measured end-to-end. Turning the remaining ~50 points of availability
into retrieval requires a *learned* ranker over candidate features
(`graph_bridge_score`, `anchor_shared_uuid_or_cite`, hop, fan-out, document
size), not another hand-tuned scalar. That is the honest next step, and this
channel now supplies the features and the candidate generator for it.

---

## 7. Bridge telemetry (shipped config: weight 0.2, limit 20, hops 2)

| | 48q | 28q | hard_1k |
|---|---:|---:|---:|
| fire rate (≥1 candidate) | 100 % | 100 % | 100 % |
| candidates injected into the pool | 20/query | 20/query | 20/query |
| **candidates landing in the final top-12** | 4.04 | 2.93 | 4.07 |
| of those, chunks in a gold document | 0 | 0 | 75 total |
| queries gaining a gold document | 0 | 0 | **14** |
| queries losing a gold document | 0 | 0 | **5** |
| queries where the *answering* document is added | 0 | 0 | **14** |
| mean seed entities (question + anchors) | 196 | 321 | 212 |
| mean scored chunks | 1 320 | 1 427 | 1 509 |

The channel fires on every query and places ~4 of its 20 candidates inside the
window; on `hard_1k` that nets 14 answer documents gained against 5 gold
documents evicted. On 48q/28q it fires, injects four chunks, and moves nothing.

---

## 8. Latency

Measured in the eval loop (process warm, retrieval for the same query just
completed — the realistic in-situ marginal cost):

| | 48q | 28q | hard_1k |
|---|---:|---:|---:|
| **bridge channel** mean | 12.0 ms | 16.9 ms | **16.5 ms** |
| bridge p50 | 7.6 ms | 11.2 ms | 11.0 ms |
| bridge p95 | 30.7 ms | 39.8 ms | **42.6 ms** |
| full pool build (dense + BM25 + ripgrep) mean | 66.7 ms | 89.9 ms | 128.9 ms |

Isolated (no retrieval in flight, 48q pools, 3 warm-up passes): **5.8 ms mean /
13.7 ms p95** with 2 hops, 5.0 / 13.4 with 1 hop.

The <15 ms budget is met on the mean and missed at p95. The tail is the
fan-out: a document with ~1 000 extracted entities (e.g.
`vea_positive_solution_search_2026-05-19.md`) seeds ~1 200 entities and forces
~2 200 scored chunks. Store loading (GraphML + 4 JSON stores, 3–4 s) happens
once per process and is excluded from these numbers; it is cached and
mtime-invalidated so a concurrent ingest is picked up without a restart.

---

## 9. Caveats

1. `answer_in_window` is literal substring containment of the `answer` field;
   it is a lower bound where the answer is paraphrased and an upper bound where
   the answer string is generic (`completed`, `3`). Same definition as
   `root_cause/REPORT.md` §11.1.
2. The 3.0 pt gain is statistically detectable but small: the paired 95 % CI is
   [+0.67, +5.67]. recall@10's CI includes zero. The correct reading is "a small
   real improvement in answer delivery, no measurable change in ranking
   quality".
3. Graph availability is measured on the production index only. The md_struct
   index was excluded by the brief (a concurrent process is writing it) and its
   entity bridge is known to be broken (4/7 vs 28/28 in
   `miss_analysis/REPORT.md` §3.3), so none of this transfers to it.
4. Seed extraction uses the anchor document's extraction roster, which is
   produced by the same LLM extraction as the graph; an index with partial
   extraction falls back to text matching (slower, noisier) rather than failing.
5. The hub cap (`80`), hop decay (`0.6`), degree cap (`12`) and `+0.50` bonus
   are the review's constants, kept unchanged so this measures the proposal.
   Only the *placement* (weight 0.2) and the *bonus scale* were chosen from the
   data, and both are reported above.

---

## 10. Reproduction

All commands from the repo root with `.venv/bin/python`, CPU only, read-only.

```bash
set -a; source eval_agentic-data/bench_env.sh; set +a
export HARS_MEMORY_BM25_CACHE_DIR="$HOME/.local/share/hars-graphrag/bm25_cache_compact_20260903"
export HARS_MEMORY_RIPGREP_ROOTS=/home/angkira/mem-recovery/r7_stage/corpus_full

# 1. unit tests (synthetic graph, no network)
.venv/bin/python -m pytest tests/test_graph_bridge.py -q

# 2. the headline table: 3 sets x 4 variants + weight/limit grid (~4 min)
.venv/bin/python eval_agentic/rca/graph_bridge/eval_graph_bridge.py \
  --top-k 12 --alpha 0.15 --pool-multiplier 2 --max-hops 2 \
  --weights 0.1 0.15 0.2 0.25 0.5 --limits 5 20 --shipped-weight 0.2 \
  --out eval_agentic/rca/graph_bridge/metrics.json

# 3. where the answer document ranks (funnel)
.venv/bin/python eval_agentic/rca/graph_bridge/diagnose_bridge.py

# 4. aggregation / expansion search
.venv/bin/python eval_agentic/rca/graph_bridge/rank_experiment.py

# 5. the offline gate now carries fusion_graph / fusion_rog_graph variants
.venv/bin/python -m hars_memory.eval.ltr_eval \
  --queries eval_agentic-data/retrieval_queries_current_gcfixed.yaml \
            eval_agentic-data/relational_gold.yaml \
            eval_agentic-data/hard_1k/hard_1k_eval.yaml \
  --top-k 12 --alpha 0.15 --pool-multiplier 2 \
  --report eval_agentic/rca/graph_bridge/ltr_gate.json
```

| file | contents |
|---|---|
| `metrics.json` | config, all variants, paired bootstrap, per-stratum, sweep grid, bridge telemetry, latency |
| `diagnostics.json` | per-query funnel: answer-doc scored / candidate rank / bonus / baseline-window, plus summary |
| `rank_experiment.json` | six aggregations + hub-cap/hop grid, ranked by "answer in candidate top-20 on actionable queries" |
| `ltr_gate.json` | the same two graph variants as produced by the repository's own offline gate (byte-identical metrics to `metrics.json` for `fusion`/`fusion_graph`) |

---

## 11. Recommendation

1. **Do not enable** `HARS_MEMORY_GRAPH_BRIDGE` on the strength of this
   measurement: +3 pts answer-in-window and +1.1 pts r@10 (CI includes 0) do not
   justify 16 ms mean / 43 ms p95 on every query.
2. **Keep the module and its features.** It is the candidate generator a bridge
   ranker needs: `graph_bridge_score`, candidate hop, fan-out, the
   citation/rare-UUID flag. The measured ceiling for hand-scored candidates is
   ~24 % of actionable queries (`rank_experiment.json`); a learned ranker over
   those features is the only path from there to the remaining headroom.
3. **If it is enabled anyway, use weight 0.2 and limit ≤ 5.** Above 0.25 it
   starts evicting correct chunks; the sweep in §5 is the guard rail.
4. **The cheaper lever stays the window.** `root_cause §10` measured +20.7 pts
   answer-in-window from `top_k 6→20` at zero latency; this channel's +3.0 is
   not a substitute for it.
