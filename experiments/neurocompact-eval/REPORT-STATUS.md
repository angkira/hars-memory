# Interim status — chained long-replay A/B (HARS graph memory vs in-context summary)

Date: 2026-10-26. Status: decisive run in progress, batches. NO OUTCOME CLAIM IS MADE YET.

## Protocol (one paragraph)

Ten real OpenCode sessions are cut into 10 chains x 3 spans (~10k tokens each); each chain is
replayed verbatim through 3 forced compactions, and questions are asked only after C3. Three
arms: `baseline` (in-context structured summary, hard-trimmed to 1536 realized tokens),
`nc` (short 640-token note + HARS recall packed to the remaining budget,
`clamp(1536 - realized nc note, 256, 1024)`; the baseline's realized size is only reported), and `nc-note-only` (fork of the nc session, note
only — isolates the recall channel). Agent and compactors: `qwen/qwen3.8-flash`; judge:
`openai/gpt-6-luna`, blind (no arm labels, shuffled order with logged mapping) and
evidence-grounded. 70 QA items (60 answerable with verbatim-verified evidence and unique gold
anchors, 10 adversarial unanswerables); hallucination is cross-checked against the model's
actually visible context (`clean_anchor_hit`, primed-hit exclusion). Statistics: paired deltas
with cluster bootstrap (10,000 resamples, clustered by source SESSION) plus Wilcoxon over
per-cluster means; MIN_CHAINS = 8 counts distinct sessions; decisions
follow pre-registered WIN/FAIL thresholds (RUNBOOK-gate.md); a kill/blame funnel
(anchor_in_summary / anchor_in_graph / anchor_in_recall vs BM25 over the same dropped context)
runs offline on archived artifacts. Design follows the external review (argon_answer_r1.md);
protocol frozen in REPORT-PROTOCOL.md; realized-token parity is enforced by construction
(section trim + derived recall cap), not by caps alone.

## Current sample

- Decisive run, batch 1 (`runs/full-20261026T021613`): chains 1-3 complete and valid
  (63 records; per-chain parity ratios 0.987 / 0.992 / 0.982; recall channel healthy). Run
  killed by host OOM at 04:17 before chains 4-10.
- Batch A (`runs/fullB-075703`, chains 4-7): completed but INVALID — the HARS recall channel
  returned error 530 with empty payloads on all 28 nc asks, run parity ratio 0.579 (window
  [0.80, 1.05]), 2/4 chains CLIP FAIL. Excluded from any decision set pending diagnosis and
  re-run. Batch B (chains 8-10) not started.
- Gate runs (1-2 chains each) plus their judge/calibration outputs; dataset:
  10 chains, 70 items, independently re-validated at pass rate 1.000 (verify_chains.report.md).

## Gate results so far

| Run | Change under test | Run-mean parity ratio [0.80, 1.05] | Verdict |
|---|---|---|---|
| gate1 `gate-212839` (cap 1575) | ceiling-only baseline cap | chain2 1.261 | FAIL: baseline C3 CLIP + ratio |
| gate2 `gate2-220255` (cap 2240) | higher ceiling | 0.714 | FAIL: fill lottery (1183 vs 2047 same chain) |
| gate3 `gate3-230010` (1536 + section trim) | Amendment A | 1.163 | FAIL: baseline under-fill 0.688 |
| gate4 `gate4-233211` | trim stable | 0.989 | PASS ratio; 1 chain baseline CLIP FAIL |
| gate5 `gate5-001652` | repair path | 0.991 | PASS ratio; 1 chain nc CLIP FAIL |
| gate6 `gate6-005402` / `gate6-013036` | final config + judge calibration | 0.991 / 0.991 | PASS; calibration agree 0.976, kappa 0.956, P(unsup AND score=2)=0.024 |
| full run partial (`full-20261026T021613`, chains 1-3) | final config | 0.982-0.987 per chain | valid; killed by host OOM (host, not protocol) |
| batch A (`fullB-075703`, chains 4-7) | final config | 0.579 | INVALID: recall_error=530 28/28 nc asks, 2/4 CLIP FAIL — re-run |

Gate history led to two pre-decisive amendments (RUNBOOK-gate.md): Amendment A (deterministic
section trim to mem_budget replaces ceiling caps) and Amendment B (recall packer v2,
query boost off). Forensic baseline: gate-212839 showed the recall channel delivers —
P(anchor_hit | anchor_in_recall) = 3/3 — and both unanswerable probes were cleanly abstained
by nc while baseline and nc-note-only each hallucinated 1/2 (n=2 chains, diagnostic only).

## Known caveats (HANDOFF.md "OPEN PROTOCOL ISSUES")

The 10 chains are 5 independent sessions, so the metrics_v2 power check (8 sessions) cannot be
met with this dataset and the verdict will be INCONCLUSIVE unless the dataset is rebuilt or
the protocol amended. Also open: whether `nc` recall exercises the graph or only chunk
retrieval (M2, unverified), the 12k+12k payload limit to HARS (M3), the undiagnosed batch-A
error 530 (M5), tuning chains inside the decision set and batch 1 run under the old
prompt/cap (M7).

## What the decisive numbers will be

Per arm (baseline / nc / nc-note-only) on the merged 10-chain set: judge score mean,
hallucination rate (answerable + unanswerable false-answer rate), clean_anchor_hit, median ask
latency, realized memory tokens and parity ratio, persisted-compaction rate; per-lag (3/2/1)
slices; chain-cluster bootstrap 95% CIs for paired deltas (nc - baseline, nc - nc-note-only);
the pre-registered WIN/FAIL/INCONCLUSIVE verdict (thresholds verbatim in RUNBOOK-gate.md);
kill/blame funnel table with the BM25 comparison (kill graph if HARS anchor_in_recall
<= BM25 + 5 pp). Chains invalid under the current gates (CLIP FAIL, RECALL FAIL, aborts) and
incomplete arm triplets are dropped before any number is computed.

## No outcome claim

With 3 valid chains (plus diagnostic gates at n<=2), the experiment is underpowered and the
pre-registered gates are not evaluable. No directional claim — WIN, FAIL, or trend — is made
until the full 10-chain dataset (batches re-run and merged) has been judged and scored.
All interim numbers above are infrastructure/parity diagnostics, not effect estimates.
