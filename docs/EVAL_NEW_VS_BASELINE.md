# Eval: new md_struct index vs production index

Measured 2026-10-19, read-only, no GPU. Raw artifacts:
`/mnt/datasets/ft/eval_new_vs_base/` (see `REPORT.md`, `ltr_base.json`,
`ltr_new.json`, `comparison.json`, `per_query_*.json`).

## Indexes

| | BASELINE (production) | NEW (Colab, restored) |
|---|---|---|
| Path | `~/.local/share/hars-graphrag/index_compact_20260903_imported` | `/mnt/datasets/ft/index_local/index_full_mdstruct_20261014` |
| Docs with chunks | 1624 | 989 (strict subset) |
| Chunks | 2820 | 7561 |
| Chunks / doc | 1.74 | 7.65 |
| mean / median chunk chars | 3740 / 2233 | 866 / 706 |

Harness sanity: baseline `fusion` reproduces the published gate exactly
(0.7083 / 0.9792 / 0.8563 / 0.8149 on 48q). The shipped ranker artifact is
newer than `docs/ltr-ranker.md`; today's baseline `ltr_full` is
0.8125 / 1.0000 / 0.9085 / 0.8785. Same artifact used for both indexes.

## Headline (all questions — misleading, coverage-limited)

48q `fusion`: r@1 0.7083 → 0.6042, r@5/@10 0.9792 → 0.6875, NDCG 0.8563 →
0.6484, MRR 0.8149 → 0.6354.
New r@10 = 0.6875 = exactly 33/48 because only 33 of 48 gold docs exist in NEW.

## Matched subset (gold doc present in both), 48q, n=33

| metric | fusion BASE | fusion NEW | Δ | ltr BASE | ltr NEW | Δ |
|---|---|---|---|---|---|---|
| recall@1 | 0.7273 | **0.8788** | +0.1515 (+20.8%) | 0.8182 | **0.8788** | +0.0606 (+7.4%) |
| recall@5 | 1.0000 | 1.0000 | 0 | 1.0000 | 1.0000 | 0 |
| recall@10 | 1.0000 | 1.0000 | 0 | 1.0000 | 1.0000 | 0 |
| NDCG@10 | 0.8740 | **0.9431** | +0.0691 (+7.9%) | 0.9128 | **0.9452** | +0.0324 (+3.5%) |
| MRR | 0.8318 | **0.9242** | +0.0924 (+11.1%) | 0.8838 | **0.9268** | +0.0430 (+4.9%) |

Availability-normalised (all questions, gold held by the index):
48q fusion r@1 0.7083 → 0.8788, r@10 0.9792 → 1.0000;
28q fusion r@1 0.2857 → 0.3871, r@10 0.6964 → 0.8387.

28q matched subset is only 7 queries (mixed signs) — noise, not evidence.

## Why the gain exists

Pruning the baseline ranking to NEW's document set (isolates the effect of
having fewer docs):

| set | variant | BASE rank-1 | BASE pruned to NEW corpus | NEW |
|---|---|---|---|---|
| 48q | fusion | 24/33 | 25/33 | **29/33** |
| 48q | ltr_full | 27/33 | 29/33 | 29/33 |
| 28q | fusion | 5/14 | 7/14 | 6/14 |

Corpus pruning explains ~1 of fusion's +5 rank-1 hits; the rest is
**`md_struct` chunking**. For the LTR ranker, pruning explains the entire
gain and the ranker adds nothing on NEW (domain shift: trained on legacy
3.7k-char chunks).

## Fairness flags

- Different doc sets (989 ⊂ 1624); 31% (48q) and 45% (28q) of gold docs absent.
- 2.7x more chunks for 39% fewer docs; a fixed top-10 *chunk* window penalises
  fine chunking.
- Ranker trained on baseline chunk statistics.
- Matched subset is not random (docs NEW happens to hold) — gains may be
  optimistic.
- Paired sign test on 48q recall@1: 6 wins / 1 loss, p = 0.125 (directionally
  consistent, not significant).

## Recommendation

Do **not** switch production yet. To ship:
1. Finish the corpus (full ~1624 docs; remainder via batch extraction; recover
   `hypotheses/`, `experiments/`, `hypothesis_links/` from cortex git history).
2. Retrain the LTR ranker on `md_struct` chunks.
3. Re-run this exact gate; require significance and no coverage gap.
