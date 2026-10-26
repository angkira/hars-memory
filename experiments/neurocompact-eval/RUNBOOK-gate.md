# RUNBOOK — chained long-replay A/B (pre-freeze gate + full run)

Protocol owner notes, 2026-10-25. Design follows the external review (argon_answer_r1.md):
10 chains x C=3 chained compactions, ask only after C3, arms `baseline` / `nc` / `nc-note-only`,
realized-token budget parity B=1536, blind evidence-grounded judge, chain-cluster bootstrap gates.

## Invariants (checked by the runner; never bypass)

1. Exactly 3 forced compactions per chain per live arm; `memory_session_status.compactions == 3`
   for `nc` before any ask; baseline has 0 HARS compactions.
2. `finish_reason` recorded for every compaction; a summary starting with
   `[unsummarized` invalidates the chain (`!!!` line, `valid=false`).
3. Ask-time memory budget: nc = note + recall where recall cap = clamp(1536 − realized_note, 256, 1024);
   baseline = single summary, cap calibrated so realized ≈ 1450–1600.
   Gate: realized ratio nc/baseline ∈ [0.80, 1.05] per chain, else `!!! BUDGET FAIL`, chain invalid.
4. Clipping gate: >1/6 compactions with `stop=length` in a chain → `!!! CLIP FAIL`, chain invalid.
5. Every question asked on a fresh `cp -a` fork of the post-C3 session dir (no cross-question
   contamination). `nc-note-only` forks the `nc` session and never calls recall (0 recall bytes).
6. QA evidence re-verified verbatim-in-span at runtime; anchors regex-verified (no bare digits,
   unique across the chain's other spans).
7. Judge is blind: no arm labels in prompts, presentation order shuffled with a logged mapping.

## Pre-freeze gate (MANDATORY before the full run)

Cost ceiling $0.60, wall ~35–50 min (dominated by 6 nc compactions on :8305).

```
cd /tmp/opencode/nc-eval
export OPENROUTER_API_KEY=$(grep -m1 '^OPENROUTER_API_KEY=' /home/angkira/Project/software/cortex/.env | cut -d= -f2- | tr -d '"'"'"'[:space:]')
python3 run_chainreplay.py --chains chains.jsonl --arms baseline,nc,nc-note-only \
  --limit 2 --out runs/gate-$(date -u +%H%M%S) \
  --agent-model qwen/qwen3.8-flash \
  --mem-budget 1536 --require-no-clip --mcp http://127.0.0.1:8307/mcp
python3 export_judge_input.py --records "runs/gate-*/records.jsonl" --chains chains.jsonl \
  --out "$(ls -dt runs/gate-* | head -1)/judge-input.jsonl"
python3 judge_v2.py --in "$(ls -dt runs/gate-* | head -1)/judge-input.jsonl" --calibrate \
  --model openai/gpt-6-luna --out "$(ls -dt runs/gate-* | head -1)/judged.jsonl" \
  --mapping "$(ls -dt runs/gate-* | head -1)/judge-mapping.json"
python3 metrics_v2.py --records runs/gate-*/records.jsonl --judge runs/gate-*/judged.jsonl
```

Calibrated/decided values (gate-212839 + gate2-220255 + calib2-4, 2026-10-25/26):
- nc note cap **640** (0/6 clips, realized 417-635); memory service **:8307**
  (`nc-hars-chain-glm`, glm extractor, ~13-73 s/compact).
- baseline **cap = mem_budget = 1536 with NC_SUM_SECTION_TRIM=1** (hard budget: the
  section trimmer cuts an over-long summary at boundaries; baseline natural fill
  measured 1183-2135 tokens across runs of the SAME chain, so ceiling-only caps
  landed the run-mean ratio anywhere in 0.66-1.26 — trim-to-cap makes parity a
  protocol property, not sampling luck).
- Recall packer v2 (rank-aware, boilerplate strip, unitized salience knapsack):
  in-raw anchor retention 6/8 at the ~950-token cap (query-blind; the query boost
  is disabled — leak-proof questions share no tokens with anchor lines and the
  boost displaced anchor units, 5/8). anchor_in_recall is the memory channel's
  honest delivery: 3/12 -> 6/12 answerable items; when the anchor IS delivered the
  agent answered 3/3 (gate-212839 forensics).

## Amendment log (pre-decisive-run)

1. gate1 (cap 1575): chain1 baseline C3 clipped (CLIP FAIL), chain2 ratio 1.261.
2. gate2 (cap 2240): run ratio 0.714; same-chain fill varies run-to-run
   (1183 vs 2047) -> static-ceiling parity is a lottery.
3. AMENDMENT A (before any decisive run, after 2 gate cycles): baseline budget is
   enforced by deterministic section trim to mem_budget; CLI default
   --baseline-sum-max-tokens changed 3072 -> 1536; NC_SUM_SECTION_TRIM=1 for all
   summarizing arms.
4. AMENDMENT B: recall packer v2 replaces the rank-prefix packer (retention 3/8 ->
   6/8 in-raw); PACK_QUERY_BONUS kept but set 1.0 (off) for the run.
The per-chain ratio window is a warning; the gate is the run-mean (see
RUN BUDGET / run_budget.json). CLIP FAIL stays per-chain.

Gate pass criteria (ALL must hold; numbers from the external review):
- 0 `[unsummarized` across all compactions; ≤1/6 `stop=length`.
- baseline anchor_hit after C3 ∈ [0.40, 0.85] (comparator neither dead nor saturated).
- nc recall injected non-empty (≥100 chars packed) on 100% of asks; `recall_packed_tokens ≤ 1024`.
- Realized memory-token ratio ∈ [0.80, 1.05].
- Judge calibration on duplicates: exact agreement ≥ 90% (kappa ≥ 0.80);
  P(unsupported_claim AND score=2) ≤ 5%.
- `metrics_v2.py` runs clean and reports per-lag slices.

If the gate fails: fix the named defect, re-run the gate. Do NOT start the full run on a partial pass.

## Full run

```
python3 run_chainreplay.py --chains chains.jsonl --arms baseline,nc,nc-note-only \
  --limit 10 --out runs/full-$(date -u +%Y%m%dT%H%M%S) \
  --agent-model qwen/qwen3.8-flash \
  --mem-budget 1536 --require-no-clip --mcp http://127.0.0.1:8307/mcp
python3 export_judge_input.py --records "runs/full-*/records.jsonl" --chains chains.jsonl \
  --out "$(ls -dt runs/full-* | head -1)/judge-input.jsonl"
python3 judge_v2.py --in "$(ls -dt runs/full-* | head -1)/judge-input.jsonl" \
  --model openai/gpt-6-luna --out "$(ls -dt runs/full-* | head -1)/judged.jsonl" \
  --mapping "$(ls -dt runs/full-* | head -1)/judge-mapping.json"
python3 metrics_v2.py --records runs/full-*/records.jsonl \
  --judge runs/full-*/judged.jsonl --judge-input "$(ls -dt runs/full-* | head -1)/judge-input.jsonl" \
  --out REPORT-chains.md
```
(`--judge-input` is REQUIRED for clean_anchor_hit/primed_hit; without it those columns are n/a.
Multi-batch runs: point --records/--judge/--judge-input at ALL batch dirs' files (globs) and merge
records/compactions/chain_budget by chain; sessions/ dirs merge by relative path.)

# Budget ceiling and balance

OpenRouter balance was **$6.52** before the full run (runs/full-spend.txt, 2026-10-26T03:16;
an earlier "~$4.0" figure is stale). Remaining-run ceiling: **$2.50**.
Spend check before launch: `python3 -c` balance probe or OpenRouter dashboard.
Abort criteria mid-run (runner or operator):
- 2 consecutive chains with HARS persist failure (`memory_compact` not ok or compactions < 3) → stop, infra blame.
- ANY recall_error run (`recall_raw_chars=0`, e.g. HTTP 530 provider outage): the nc arm is
  dead for those chains (note-only in disguise) — mark the batch invalid, re-run it after the
  provider recovers. Preflight: one `memory_recall` against :8307 before every batch.
- Any single chain's spend pacing projects past $2.50 total → stop after current chain.
- Wall time: nc compactions ~55-75 s each on :8307 glm extractor. 10 chains ≈ 2-2.5 h total.
  Serialize nc compactions (single worker); do not run two chains' nc arms concurrently on :8307.

## Decision rules (external review, verbatim thresholds)

WIN = ALL four:
1. paired Δscore CI95 lower ≥ −0.05 AND Δanchor CI95 lower ≥ −2 pp (point ≥ 0);
2. Δhalluc ≤ −10 pp with CI95 upper < 0; unanswerable false-answer Δ ≤ −10 pp
   (or ≤ 25% absolute if baseline ≤ 30%);
3. nc − nc-note-only: Δscore ≥ +0.15 or Δanchor ≥ +10 pp with CI lower > 0;
4. realized mem-token ratio ∈ [0.80, 1.05], persisted ≥ 95%, median ask ≤ 20 s.

FAIL = ANY: Δscore < −0.10; Δanchor < −5 pp; Δhalluc > −5 pp;
dead memory (|Δ| vs note-only: score < 0.05 AND anchor < 3 pp); median ask > 30 s; cost/Q > 3x baseline.
Everything else: INCONCLUSIVE / UNDERPOWERED — never report as a directional win.

Kill/blame funnel (offline, $0, on archived artifacts):
- `anchor_in_summary` (baseline C3 summary), `anchor_in_graph` (HARS graphml/KV),
  `anchor_in_recall` (packed recall string), plus BM25/FTS5 over the same dropped_context.
- Kill architecture: anchor_in_summary ≥ 85% after C3 (no headroom), OR
  P(anchor_hit | anchor_in_recall) < 70% (reader broken).
- Kill graph specifically: HARS anchor_in_recall ≤ BM25 + 5 pp.
- Blame extractor only if: anchor_in_summary ≤ 60% AND P(anchor_hit|recall) ≥ 85%
  AND anchor_in_graph < 35% AND chunk-only recall ≥ 75%.

## After the run

1. Archive to `/mnt/datasets/ft/artifacts/<ts>-neurocompact-chains/`: chains.jsonl, runs/full-*,
   judged outputs, metrics, HARS session dirs (graphs), REPORT-chains.md.
2. Append results + verdict to the artifact REPORT.md (the raw-tail correction is already there).
3. If INCONCLUSIVE: check the drift test first — baseline anchor on lag-3 spans vs lag-1 spans;
   if baseline loses < 10 pp across two re-summarizations, 3 compactions are below the
   information-loss cliff (say so; do not spin).
