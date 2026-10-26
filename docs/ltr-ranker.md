# Learned channel-fusion ranker (H-91a playbook, ported)

**Status: shipped (offline gate PASSED on both held-out sets).**

The recall path used to decide the final ordering with a hand-tuned additive
blend, `fused = alpha*dense_norm + (1-alpha)*sparse_norm` (plus small
additive gates). This adds a SMALL learned model over the per-candidate channel
features that **assigns those weights from data** instead of from a sweep — a
LightGBM LambdaRank ranker, trained on our own synthetic queries, scoring in
**≈0.20 ms per candidate on CPU** with no GPU, no LLM and no network.

It replaces the cross-encoder reranker on the default path: the cross-encoder
cost ~8.5 s/query for +6.3 pts recall@1 on 3 of 48 queries; the learned ranker
costs a few ms/query for a larger, measured gain on both held-out sets.

The recipe is code-diver's H-91a (LightGBM LambdaRank over rank-time features,
feature-manifest guard that refuses a mismatched artifact, fail-soft loader).
The implementations are independent; the shared idea is the contract.

## Module map

| Path | Role |
| --- | --- |
| `hars_memory/ranking/features.py` | `LTR_FEATURE_NAMES`, deterministic feature extraction, corpus context |
| `hars_memory/ranking/artifact.py` | manifest-guarded, fail-soft artifact loader (`LtrRankerModel`) |
| `hars_memory/ranking/rerank.py` | `apply_ltr_ranker` / `load_default_ranker` (the call-site API) |
| `hars_memory/ranking/train.py` | training library (LightGBM LambdaRank + tiny MLP) |
| `hars_memory/ranking/artifacts/` | shipped artifact + the gate reports (committed) |
| `hars_memory/eval/ltr_dataset.py` | live-index feature/label export (read-only) |
| `hars_memory/eval/ltr_train.py` | train both backends, pick the winner, write the manifest |
| `hars_memory/eval/ltr_eval.py` | the mandatory offline gate |
| `hars_memory/eval/gen_synthetic_queries.py` | scaled, concurrent, resumable training-query generator |
| `hars_memory/retrieval/fusion.py` | additive integration point (`apply_ltr_ranker`, `ltr_ranker_enabled`) |

## Features (`LTR_FEATURE_NAMES`, 20 columns, ordered)

The order **is** part of the artifact contract: `LtrRankerModel.load` refuses a
manifest whose `feature_names` differ from this checkout's tuple rather than
scoring shifted columns.

| # | Feature | Meaning | Availability |
| --- | --- | --- | --- |
| 0 | `dense_norm` | min-max normalized dense score within the pool [0,1] | fusions-only |
| 1 | `sparse_norm` | min-max normalized BM25 score within the pool [0,1] | fusions-only |
| 2 | `ripgrep_norm` | min-max normalized raw ripgrep score within the pool [0,1] | fusions-only |
| 3 | `dense_score` | raw dense cosine (confidence, not rank) | fusions-only |
| 4 | `sparse_score` | raw BM25 (confidence, not rank) | fusions-only |
| 5 | `ripgrep_score` | raw ripgrep score for the chunk's file | fusions-only |
| 6 | `fused_score` | the hand-tuned blend the ranker replaces | fusions-only |
| 7 | `fused_gap_to_top` | `max(pool fused) - fused` (0.0 for the top) | fusions-only |
| 8 | `base_rank_reciprocal` | `1/rank` in the hand-tuned order | fusions-only |
| 9 | `channel_count` | how many of {dense,BM25,ripgrep,flat} hit this chunk | fusions-only |
| 10 | `exact_identifier_match` | query identifier term appears verbatim in the chunk | fusions-only |
| 11 | `content_term_overlap` | fraction of query content words present in the chunk | fusions-only |
| 12 | `chunk_len_log` | `log1p(len(content))` | fusions-only |
| 13 | `doc_len_log` | `log1p(total doc chars)` | corpus context |
| 14 | `doc_age_days_log` | `log1p(days since doc date)`; **-1.0 if undated** | corpus context |
| 15 | `section_depth` | markdown heading depth (`#` count) | index field, else content |
| 16 | `is_code_heavy` | chunk dominated by fenced/inline code | index field, else content |
| 17 | `self_declared_deprecated` | supersession flag (chunk declares itself stale) | content |
| 18 | `graph_entity_fanin_log` | `log1p(max entity fan-in in the chunk)` | graph context only |
| 19 | `graph_entity_match` | fraction of query identifiers that are entity names in the chunk | graph context only |

**Missing values are encoded, never faked.** `doc_age_days_log = -1.0` keeps
`0.0` meaning "dated today". Without a corpus context (features 13–14) the
values are `0.0`/`-1.0`; without a graph context (18–19) they are `0.0`. The
shipped model was trained WITH the corpus/graph context populated (the live
index stores are loaded once and cached by mtime), and the wiring spec below
passes `index_dir` so the same context is available at inference.

## Datasets

### Ours (primary) — `eval_agentic-data/synthetic_queries_scaled.yaml` *(gitignored)*

* **300 queries**, LLM-written one per note, span-validated (the `answer_span`
  appears verbatim in the note). Same prompt + validation rule as the original
  48-query generator, scaled up via `hars_memory/eval/gen_synthetic_queries.py`
  (concurrent, resumable, excludes the 83 held-out documents from sampling).
* Feature/label export: **15,506 rows** over **297 queries with a positive**
  (**453 positives**), from `hars_memory/eval/ltr_dataset.py` running the real
  dense+BM25+ripgrep path over the live index (read-only). Mean pool ≈ 52.6
  candidates/query.

**Graded labels (deterministic, no LLM):**

* `2` — chunk is from a gold document AND contains the query's `answer_span`
  (for span-less queries, every gold-doc chunk is `2`).
* `1` — "same source document" chunk: from a gold document but not the evidence
  chunk (only produced for span-bearing queries).
* `0` — irrelevant.

### Open (attempted, cached, NOT used)

* **BEIR SciFact** was fetched and cached at
  `eval_agentic-data/open_datasets/scifact.zip` (2.8 MB) — network allows.
* **MS MARCO passage-ranking** collection is reachable but is the full ~3 GB
  `collection.tar.gz` (8.8 M passages).
* Neither is used for training. Two reasons, stated honestly rather than
  worked around: (1) MS MARCO passage is impractical to index here, and any
  BEIR corpus would need a fresh index built with our embedder; (2) more
  fundamentally, the feature space is defined by **our** channel stack — the
  `dense_norm`/`sparse_norm`/`ripgrep_norm` columns are per-query min-max
  scores from our dense embedder and our BM25 corpus statistics. A ranker
  trained on a different stack's score distributions would not transfer to
  these columns. The task explicitly allows proceeding with our own data; we
  did.

### Held-out evaluation (never trained on, never used for model selection)

* `eval_agentic-data/retrieval_queries_current_gcfixed.yaml` — **48 queries**
  (GC-fixed gold).
* `eval_agentic-data/relational_gold.yaml` — **28 multi-hop queries**.

## Training

```bash
# 1. generate the training queries (OpenRouter; ~2 min at concurrency 14)
python -m hars_memory.eval.gen_synthetic_queries \
    --count 300 --concurrency 14 \
    --out eval_agentic-data/synthetic_queries_scaled.yaml \
    --state eval_agentic-data/synthetic_queries_scaled.state.json

# 2. export features + labels from the live index (read-only)
set -a; source eval_agentic-data/bench_env.sh; set +a
export HARS_MEMORY_EMBED_PROVIDER=http
python -m hars_memory.eval.ltr_dataset \
    --queries eval_agentic-data/synthetic_queries_scaled.yaml \
    --out eval_agentic-data/ltr_train_300.jsonl

# 3. train both backends, pick the winner by held-out VALID NDCG@10, write the artifact
uv run --extra ltr python -m hars_memory.eval.ltr_train \
    --features eval_agentic-data/ltr_train_300.jsonl \
    --out hars_memory/ranking/artifacts/ranker.json \
    --valid-fraction 0.25 --seed 42 \
    --metrics-json eval_agentic-data/ltr_train_metrics.json
```

The split is **by query**, never by row (candidates of one query are correlated;
a row split leaks). The `--test-features` sets, if supplied, are reported only —
never used for early stopping or model selection.

## Model choice

Both backends were trained on the same query-level split (223 train / 74 valid
queries). Selection signal = **valid split** (not the held-out sets):

| backend | valid NDCG@10 | valid MRR | (base) NDCG@10 |
| --- | --- | --- | --- |
| **LightGBM LambdaRank** | **0.8973** | **0.9077** | 0.8813 |
| tiny MLP (16×8, RankNet, torch-CPU) | 0.8748 | 0.8868 | 0.8813 |

**LightGBM ships** because:

1. It wins the valid split (the legitimate selection signal).
2. It wins on **both** held-out sets (see the gate), while the MLP regresses
   *below the baseline* on the harder relational set — i.e. the MLP would FAIL
   the "at least as good as baseline on both sets" ship gate
   (relational recall@1 0.2143 < 0.2857, NDCG@10 0.5627 < 0.6144,
   MRR 0.5964 < 0.6845). On the 48-set the MLP happens to be a little better
   than LightGBM, which is exactly the small-data instability a single
   held-out set would have hidden.
3. LambdaRank is list-aware (optimises NDCG over a query's candidates); the MLP
   uses a RankNet pairwise loss that is noisier on this few-hundred-query set.

The MLP artifact is kept next to the winner (`ranker_mlp.json`) plus its own
gate report, purely so the comparison is reproducible.

## Offline gate (mandatory) — held-out results

`fusion` = hand-tuned fused order truncated to the first 10 chunks then deduped
to documents (this reproduces the published `hybrid_bm25` baseline **exactly**,
which validates the harness). `fusion_rog` = the same pool after the ranked
ripgrep additive gate (the current production default path, "rerank=off").
`ltr_full` = the ranker over the full pre-truncation pool (recommended wiring).
`ltr_gated` = the ranker over the already top-k-truncated gated pool.

### 48-query GC-fixed (`retrieval_queries_current_gcfixed.yaml`)

| variant | recall@1 | recall@10 | NDCG@10 | MRR |
| --- | --- | --- | --- | --- |
| `fusion` (baseline / rerank-off) | 0.7083 | 0.9792 | 0.8563 | 0.8149 |
| `fusion_rog` (default path) | 0.7083 | 0.9792 | 0.8563 | 0.8149 |
| **`ltr_full` (LightGBM)** | **0.7708** | **1.0000** | **0.8848** | **0.8479** |
| `ltr_gated` (LightGBM) | 0.7708 | 0.9792 | 0.8780 | 0.8450 |
| `ltr_full` (MLP) | 0.8125 | 0.9792 | 0.8998 | 0.8733 |

Delta `ltr_full` vs `fusion`: **recall@1 +0.0625, recall@10 +0.0208,
NDCG@10 +0.0285, MRR +0.0330**.

### 28-query relational (`relational_gold.yaml`)

| variant | recall@1 | recall@10 | NDCG@10 | MRR |
| --- | --- | --- | --- | --- |
| `fusion` (baseline / rerank-off) | 0.2857 | 0.6964 | 0.6144 | 0.6845 |
| `fusion_rog` (default path) | 0.2857 | 0.6964 | 0.6144 | 0.6845 |
| **`ltr_full` (LightGBM)** | **0.3214** | **0.7321** | **0.6288** | **0.7055** |
| `ltr_gated` (LightGBM) | 0.3214 | 0.6964 | 0.6223 | 0.7113 |
| `ltr_full` (MLP) | 0.2143 | 0.6964 | 0.5627 | 0.5964 |

Delta `ltr_full` vs `fusion`: **recall@1 +0.0357, recall@10 +0.0357,
NDCG@10 +0.0144, MRR +0.0210**.

**Verdict: PASS.** The learned ranker is at least as good as the hand-tuned
fusion on every metric on BOTH held-out sets, and improves all four. Raw reports
are committed at `hars_memory/ranking/artifacts/offline_gate_lgbm.json` and
`offline_gate_mlp.json`.

> Note: the `ripgrep` gate was a no-op on these two sets (`fusion_rog` ≡
> `fusion`) — consistent with the earlier finding that the basename join only
> connects to 4/48 gold documents on this corpus. The ranker learns a nonzero
> `ripgrep_norm` weight from the training data regardless.

### Latency (clean microbenchmark, 8 queries, 40 repeats)

| stage | ms/query (median) | ms/candidate |
| --- | --- | --- |
| feature extraction | 3.60 | 0.068 |
| LightGBM scoring | 7.03 | 0.134 |
| **total** | **10.6** | **0.202** |

Well under the 1 ms/candidate budget. (The `ms/query` figures in the first gate
run were inflated by host contention — load ~80; the same code measured 7–9 ms
once the host settled, which is the number above.)

## Wiring spec (for the `mcp_server.py` owner)

**No `mcp_server.py` change is made here.** The ranker reorders the FULL
pre-truncation candidate pool, which is the only way it can improve recall@10.
Recommended 3-line hook in `_compute_hybrid_block`, right after the additive
gates and before the pre-rerank trace capture:

```python
# after: fused = _select_diverse_fused_chunks(fused, len(fused))
from hars_memory.retrieval.fusion import apply_ltr_ranker  # (module already imported as fusion)

_fused_pool_ids = {c.chunk_id for c in fused_pool}
_additive_tail = [c for c in fused if c.chunk_id not in _fused_pool_ids]
fused = apply_ltr_ranker(  # 1 line; fail-soft: no artifact -> fused_pool[:top_k] unchanged
    list(fused_pool) + _additive_tail,
    question,
    top_k=top_k,
    index_dir=chunk_wdir,     # populates doc_len/doc_date/graph features
)
```

* **Inputs:** the full `fuse()` output (`fused_pool`), the query text, `top_k`,
  and the index working dir. `ll_keywords` is *not* needed (the ranker reads
  score features only; the ripgrep feature is attached to the pool separately).
* **Fail-soft:** if there is no artifact, or the artifact's feature list
  disagrees, or scoring raises, `apply_ltr_ranker` returns `fused[:top_k]`
  **unchanged** — wiring it in can never break a search.
* **Env:** `HARS_MEMORY_LTR_RANKER` points at the manifest (default:
  `hars_memory/ranking/artifacts/ranker.json`); set it to `0` to disable.
* **Dependency note:** the shipped artifact is `format: lightgbm_txt`, so the
  serving venv must include the `ltr` extra (`uv sync --extra ltr`). Without
  `lightgbm`, the loader fails soft and the ranker silently no-ops — so this is
  a deploy requirement, not an optional nicety, if you want it active.
* **Minimal alternative** (1 line, if you do not want to widen the pool): call
  `fused = apply_ltr_ranker(fused, question, top_k=top_k, index_dir=chunk_wdir)`
  on the post-gate `fused`. This is the `ltr_gated` row above — still ≥ baseline
  on both sets, but gives up the recall@10 gain that needs the wider pool.

A thin delegating wrapper exists in `hars_memory/retrieval/fusion.py`
(`apply_ltr_ranker`, `ltr_ranker_enabled`), kept in its own section at the end
of the file with lazy imports so it cannot collide with the graph-channel work
and does not change fusion's import graph for existing callers.

## Reproduce the gate

```bash
set -a; source eval_agentic-data/bench_env.sh; set +a
export HARS_MEMORY_EMBED_PROVIDER=http
python -m hars_memory.eval.ltr_eval \
    --queries eval_agentic-data/retrieval_queries_current_gcfixed.yaml \
              eval_agentic-data/relational_gold.yaml \
    --model hars_memory/ranking/artifacts/ranker.json \
    --report eval_agentic-data/ltr_offline_gate.json
```
