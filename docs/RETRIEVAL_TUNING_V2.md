# Retrieval tuning v2 — measured setup and defaults

Date: 2026-10-20. Index: production `index_compact_20260903_imported`
(1624 docs / 2820 chunks). Gate: `hars_memory.eval.ltr_eval`, CPU only.

## What changed (shipped defaults)

| Parameter | Before | After | File |
|---|---|---|---|
| `DEFAULT_QUERY_TOP_K` | 6 | **20** | `mcp_server.py:237` |
| `HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER` | 1.0 | **2.0** | `mcp_server.py:321` |
| `HARS_MEMORY_HYBRID_ALPHA` | 0.5 (untuned) | **0.15** | `mcp_server.py:200` |
| `MAX_CHUNKS_PER_SOURCE` | 3 | **2** | `mcp_server.py:379` |

All four remain env-overridable, so a rollback needs no code change:

```bash
HARS_MEMORY_QUERY_DEFAULT_TOP_K=6
HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER=1.0
HARS_MEMORY_HYBRID_ALPHA=0.5
HARS_MEMORY_MAX_CHUNKS_PER_SOURCE=3
```

## Measured result

Gate command:

```bash
set -a; source eval_agentic-data/bench_env.sh; set +a
export HARS_MEMORY_INDEX_DIR=~/.local/share/hars-graphrag/index_compact_20260903_imported
.venv/bin/python -m hars_memory.eval.ltr_eval \
  --queries eval_agentic-data/retrieval_queries_current_gcfixed.yaml \
            eval_agentic-data/relational_gold.yaml \
  --index-dir "$HARS_MEMORY_INDEX_DIR" \
  --top-k 20 --alpha 0.15 --pool-multiplier 2 \
  --report /tmp/opencode/eval_tuned.json
```

| Set | Metric | Before (k=10, α=0.5, pool×3) | After (k=20, α=0.15, pool×2) | Δ |
|---|---|---:|---:|---:|
| 48q single-hop | recall@1 | 0.7083 | **0.8333** | +12.5 pts |
| 48q | recall@10 | 0.9792 | **1.0000** | +2.1 |
| 48q | NDCG@10 | 0.8563 | **0.9196** | +6.3 |
| 48q | MRR | 0.8149 | **0.8932** | +7.8 |
| 28q relational | recall@1 | 0.2857 | **0.3393** | +5.4 |
| 28q | recall@10 | 0.6964 | **0.7679** | +7.2 |
| 28q | NDCG@10 | 0.6144 | **0.6988** | +8.4 |
| 28q | MRR | 0.6845 | **0.7562** | +7.2 |
| hard_1k eval (300) | recall@1 | 0.3561 | **0.3600** | +0.4 |
| hard_1k | recall@10 | 0.6506 | **0.6772** | +2.7 |
| hard_1k | NDCG@10 | 0.6119 | **0.6336** | +2.2 |
| hard_1k | MRR | 0.7965 | **0.8096** | +1.3 |

Every set improves on every metric; no set regresses. `top_k` sweep on the
production index: k=12 gives 48q r@1 0.8125, k=20 gives 0.8333, k=30 gives
0.8125 (28q recall@10 rises monotonically: 0.7500 → 0.7679 → 0.7857).

## Why these values (evidence)

- `top_k=6` is a chunk window. On the production index, k=6 → 10 → 20 moves
  answer-in-window .287 → .360 → .480 with flat latency (~65 ms); knee 10–12,
  plateau 20–30 (`eval_agentic/rca/root_cause/topk_sweep.csv`). On md_struct
  (7.65 chunks/doc) k=6 covers <1 document and leaves the answering document
  outside the window 87% of the time.
- `alpha=0.5` was an untuned midpoint. BM25 puts the gold in its top-3 for
  45/48 queries while dense misses 7/48 outright; LOO puts the optimum at 0.15
  (+12.5 pts recall@1). The same change is +1.5 pts on md_struct and never
  harmful.
- `fetch×2` fills the 12-slot window from a deeper pool; pool depth 18→100 was
  measured to recover 61.5 gold entries but only +1.1 pts recall@1, so a 2×
  fetch width is the cheap part of that curve.
- `max_per_source=2` is applied to the FULL fused pool before slicing, so a
  12-chunk result spans ≥6 distinct documents. (This one is enforced in the
  serving path; the offline gate does not apply diversity, so it is not
  reflected in the table above.)

## What was deliberately NOT shipped

- **LTR ranker stays unwired.** On the tuned configuration it is neutral on
  48q (0.8125/1.0/0.9081/0.8780 vs fusion 0.8125/1.0/0.9119/0.8828) and
  clearly worse on 28q (0.3036/0.75/0.6381/0.6958 vs 0.3393/0.75/0.6899/0.7512).
  Its apparent +10.4 pts was almost entirely the untuned `alpha` it implicitly
  replaced. It also regresses −3.5 pts recall@10 on hard_1k and is inert on
  md_struct (domain shift: trained on legacy 3.7k-char chunks).
- **Graph bridge channel** (`G_chunks`) is the next lever, not this push: 1-hop
  from the anchor document's entities reaches the answering document for 85.7%
  of questions vs 18.0% from the question's own entities, a +53.7 pts
  answer-in-window bound. Needs its own implementation + eval.
- **Reranker** stays off (measured: +2 rank-1, −1 top-10, 5.9 s/query).

## Next

1. Apply the pending OpenRouter batch (872 docs) to the md_struct index.
2. Re-run this gate on the full md_struct corpus.
3. Implement `G_chunks` and re-measure on 28q + hard_1k.
