# Chained-compaction metrics (v2)

records: 42 | chains: 2 | questions: 14 | seed: 1337 | bootstrap: 10000 resamples over chains | memory-token ratio pair: nc / nc-note-only

clean_anchor_hit = anchor_hit AND NOT primed_hit, where primed_hit marks a raw anchor hit whose gold anchor is absent from the arm's own visible_context (question-primed reconstruction). defined on 42/42 records (needs --judge-input / visible_context); raw anchor_hit is kept alongside, labeled (raw).

## Per-arm means

| arm | n | n_ans | n_unans | score | anchor_hit (raw) | clean_anchor_hit | primed | halluc | abstained | abstain@UNANS | lat_med_s | mem_tok_med | persisted | cost/q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 14 | 12 | 2 | 1.042 | 0.417 | 0.333 | 1 | 0.250 | 0.429 | 1.000 | 4.12 | n/a | n/a | n/a |
| nc | 14 | 12 | 2 | 0.917 | 0.500 | 0.417 | 1 | 0.179 | 0.571 | 1.000 | 5.62 | n/a | n/a | n/a |
| nc-note-only | 14 | 12 | 2 | 0.333 | 0.167 | 0.000 | 2 | 0.571 | 0.536 | 0.250 | 7.67 | n/a | n/a | n/a |

score / anchor_hit (raw) / clean_anchor_hit are answerable items only; hallucination and abstained cover all items (abstain@UNANS is the correct-abstention rate on the adversarial items). primed = answerable records with primed_hit (raw hits not backed by the arm's visible context).

## Paired per-question deltas (cluster bootstrap over chains)

| pair (A-B) | metric | delta | CI2.5 | CI97.5 | chains | wilcoxon p | W | method | paired q |
|---|---|---|---|---|---|---|---|---|---|
| nc-baseline | score | -0.125 | -0.500 | 0.250 | 2 | 1.000 | 1.0 | exact | 14 |
| nc-baseline | anchor_hit | 0.083 | -0.167 | 0.333 | 2 | 1.000 | 1.0 | exact | 14 |
| nc-baseline | clean_anchor_hit | 0.083 | -0.167 | 0.333 | 2 | 1.000 | 1.0 | exact | 14 |
| nc-baseline | hallucination | -0.071 | -0.143 | 0.000 | 2 | 1.000 | 0 | exact | 14 |
| nc-baseline | abstained | 0.143 | -0.143 | 0.429 | 2 | 1.000 | 1.0 | exact | 14 |
| nc-nc-note-only | score | 0.583 | 0.500 | 0.667 | 2 | 0.500 | 0 | exact | 14 |
| nc-nc-note-only | anchor_hit | 0.333 | 0.333 | 0.333 | 2 | 0.500 | 0 | exact | 14 |
| nc-nc-note-only | clean_anchor_hit | 0.417 | 0.333 | 0.500 | 2 | 0.500 | 0 | exact | 14 |
| nc-nc-note-only | hallucination | -0.393 | -0.500 | -0.286 | 2 | 0.500 | 0 | exact | 14 |
| nc-nc-note-only | abstained | 0.036 | 0.000 | 0.071 | 2 | 1.000 | 0 | exact | 14 |
| baseline-nc-note-only | score | 0.708 | 0.417 | 1.000 | 2 | 0.500 | 0 | exact | 14 |
| baseline-nc-note-only | anchor_hit | 0.250 | 0.000 | 0.500 | 2 | 1.000 | 0 | exact | 14 |
| baseline-nc-note-only | clean_anchor_hit | 0.333 | 0.167 | 0.500 | 2 | 0.500 | 0 | exact | 14 |
| baseline-nc-note-only | hallucination | -0.321 | -0.357 | -0.286 | 2 | 0.500 | 0 | exact | 14 |
| baseline-nc-note-only | abstained | -0.107 | -0.357 | 0.143 | 2 | 1.000 | 1.0 | exact | 14 |

## Slices

| arm | slice | n | score | anchor_hit (raw) | clean_anchor_hit |
|---|---|---|---|---|---|
| baseline | lag=1 | 4 | 1.125 | 0.500 | 0.500 |
| baseline | lag=2 | 4 | 0.500 | 0.250 | 0.250 |
| baseline | lag=3 | 4 | 1.500 | 0.500 | 0.250 |
| baseline | HEAD | 6 | 0.750 | 0.333 | 0.167 |
| baseline | TAIL | 6 | 1.333 | 0.500 | 0.500 |
| nc | lag=1 | 4 | 0.000 | 0.000 | 0.000 |
| nc | lag=2 | 4 | 1.000 | 0.500 | 0.500 |
| nc | lag=3 | 4 | 1.750 | 1.000 | 0.750 |
| nc | HEAD | 6 | 0.667 | 0.333 | 0.167 |
| nc | TAIL | 6 | 1.167 | 0.667 | 0.667 |
| nc-note-only | lag=1 | 4 | 0.000 | 0.000 | 0.000 |
| nc-note-only | lag=2 | 4 | 0.000 | 0.000 | 0.000 |
| nc-note-only | lag=3 | 4 | 1.000 | 0.500 | 0.000 |
| nc-note-only | HEAD | 6 | 0.333 | 0.167 | 0.000 |
| nc-note-only | TAIL | 6 | 0.333 | 0.167 | 0.000 |

## Suspicious reconstruction (primed_hit, not hallucinated)

| qid | arm | chain | gold_anchor | hallucinated |
|---|---|---|---|---|
| chain_qq8Vqbow_c0_1_1 | baseline | chain_qq8Vqbow_c0 | importlib.metadata | 0.000 |
| chain_qq8Vqbow_c0_1_1 | nc | chain_qq8Vqbow_c0 | importlib.metadata | 0.000 |
| chain_qq8Vqbow_c0_1_1 | nc-note-only | chain_qq8Vqbow_c0 | importlib.metadata | 0.000 |

These raw anchor hits are flagged primed_hit: the gold anchor is absent from the arm's visible_context (C3 summary/note + injected recall + post-compaction messages), i.e. plausibly reconstructed from the question's own phrasing rather than recalled from memory.

## Numeric gates

**verdict: INCONCLUSIVE** (underpowered: 2 chain(s) (< 8) -> no WIN/FAIL verdict claimed; the rule tables below are diagnostics only)

note: 2 chains is below 8 -> the result is underpowered; no directional WIN/FAIL verdict is claimed (INCONCLUSIVE; the rule tables above are diagnostics only).

| WIN rule | status | rule | observed |
|---|---|---|---|
| win_score_ci | FAIL | CI lower of paired d_score(nc-baseline) >= -0.05 | ci_lo=-0.500 point=-0.125 |
| win_anchor_ci | FAIL | CI lower of paired d_anchor(nc-baseline) >= -0.02 | ci_lo=-0.167 point=0.083 |
| win_halluc_ci | FAIL | d_halluc(nc-baseline) <= -0.10 with CI upper < 0 | point=-0.071 ci_hi=0.000 |
| win_vs_note_only | PASS | (d_score(nc-nc-note-only) >= +0.15 or d_anchor >= +0.10) with CI lower > 0 | score point=0.583 ci_lo=0.500; anchor point=0.333 ci_lo=0.333 |
| win_mem_token_ratio | FAIL | real mem-token ratio (nc/nc-note-only) within [0.80, 1.05] | ratio=n/a (n/a/n/a) |
| win_persisted_rate | FAIL | persisted_compaction_rate(nc) >= 0.95 | rate=n/a |
| win_latency | PASS | median ask latency(nc) <= 20 s | median=5.6 s |

| FAIL rule | status | rule | observed |
|---|---|---|---|
| fail_score_bad | FIRED | d_score(nc-baseline) < -0.10 | point=-0.125 |
| fail_anchor_bad | clear | d_anchor(nc-baseline) < -0.05 | point=0.083 |
| fail_halluc_bad | clear | d_halluc(nc-baseline) > -0.05 | point=-0.071 |
| fail_dead_memory | clear | dead memory: |d_score(nc-nc-note-only)| < 0.05 AND |d_anchor| < 0.03 | score=0.583 anchor=0.333 |
| fail_latency_bad | clear | median ask latency(nc) > 30 s | median=5.6 s |
| fail_cost | clear | cost per question(nc) > 3x baseline | ratio=n/a (n/a/n/a) |

Verdict precedence: underpowered (< 8 chains) -> INCONCLUSIVE, no directional label; else any FAIL rule -> FAIL; else all WIN rules -> WIN; otherwise INCONCLUSIVE (never reported as a directional win).
