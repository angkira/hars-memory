# RUNBOOK — chained long-replay A/B (pre-freeze gate + full run)

Protocol owner notes, 2026-10-25. Design follows the external review (argon_answer_r1.md):
10 chains x C=3 chained compactions, ask only after C3, arms `baseline` / `nc` / `nc-note-only`,
realized-token budget parity B=1536, blind evidence-grounded judge, chain-cluster bootstrap gates.

## Invariants (checked by the runner; never bypass)

1. Exactly 3 forced compactions per chain per live arm; `memory_session_status.compactions == 3`
   for `nc` before any ask; baseline has 0 HARS compactions.
   Persistence gate (nc, after C3): `memory_session_status.entities > 0` (an empty graph from a
   dead extractor fails) and `vectors_indexed > 0` when the server reports it; failure aborts
   the chain (written with `valid=false`).
2. `finish_reason` recorded for every compaction; an empty summary or one starting with
   `[unsummarized` aborts the chain (`!!!` line; the chain's rows are written with
   `valid=false`, reason `chain aborted: ...`).
3. Ask-time memory budget: nc = realized nc C3 note + packed recall, where recall cap =
   clamp(mem_budget(1536) − realized nc note, 256, 1024) (`recall_cap_for`; the baseline's
   realized C3 is only REPORTED, it never enters the cap). Note cap 640. Baseline = single
   summary hard-trimmed to mem_budget 1536 (`NC_SUM_SECTION_TRIM=1`); its realized size is
   whatever the model writes under that cap.
   Gate = RUN-MEAN ratio nc/baseline realized memory tokens ∈ [0.80, 1.05] (runner:
   `RUN BUDGET ... PASS/FAIL` + run_budget.json; metrics_v2: `win_mem_token_ratio` on the
   kept records). Per-chain ratio outside the window is a `[budget warn]` only (Amendment A);
   it does NOT invalidate the chain. A chain is invalidated by budget only if the nc note alone
   leaves less than the recall floor (recall cap clamped low → `BUDGET ...`).
4. Clipping gate: >1/6 unrepaired compactions with `stop=length` in an arm of a chain →
   `!!! CLIP FAIL`, chain invalid (stop=length that the extension boundary-repaired is recorded
   as `clipped_repaired` and does not count).
4a. Recall gate: ANY nc recall that is empty (raw 0 chars), has a payload `error` / `ok=false`
   / session-channel error / `vectors_indexed=0`, or packs 0 content units → `!!! RECALL FAIL`,
   chain invalid (nc without recall is note-only in disguise). Gate is on the raw payload and
   packed content units, not on packed chars (the wrapper header alone is ~100 chars).
4b. Preflight: one `memory_recall` before any replay/compaction spend; transport/tool error →
   `!!! PREFLIGHT FAIL`, exit 1. Pass `--preflight-session <sid>` (a session with persisted
   compactions) to also abort on an empty recall.
4c. Records are buffered per chain and written once at chain end with the FINAL `valid` flag
   and `chain_invalid_reasons` (also on chain_budget.jsonl / compactions.jsonl): an abort never
   leaves valid=true rows. The runner refuses a non-empty `--out`.
4d. All arms use the identical ask prompt (`ASK_PROMPT`), including the abstention sentence
   ("If you do not have the information needed to answer, say that you do not know."); the nc
   recall block adds no arm-specific abstention text.
5. Every question asked on a fresh `cp -a` fork of the post-C3 session dir (no cross-question
   contamination). `nc-note-only` forks the `nc` session and never calls recall (0 recall bytes).
6. QA evidence re-verified verbatim-in-span at runtime; anchors regex-verified (no bare digits,
   unique across the chain's other spans).
7. Judge is blind: no arm labels in prompts, presentation order shuffled with a logged mapping;
   arm-revealing strings inside answers (`-nc-`/`-baseline-` session-id parts, recall-wrapper
   headers, "recalled memory", HARS) are masked as `[masked]` in the judge-visible answer and
   logged (`answer_masked`, `masking` in the mapping file).
8. Metrics input validity (metrics_v2.py and export_judge_input.py, same filter): only rows
   with `valid is True` are used; a chain with any invalid row or any `recall_error` is dropped
   whole; a qid without the full baseline/nc/nc-note-only triplet is dropped; duplicate
   (qid, arm) or duplicate rid across all loaded rows is a hard error (exit 2).

## Pre-freeze gate (MANDATORY before the full run)

Cost ceiling $0.60, wall ~35–50 min (dominated by 6 nc compactions on :8307; :8305 is the
dead-extractor trap, see INFRA-NOTES.md).

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
The per-chain ratio window is a warning only; the gate is the run-mean ratio in [0.80, 1.05]
(runner `RUN BUDGET` / run_budget.json; metrics_v2 recomputes the same run-mean ratio from
the kept records' `memory_tokens`). Invariant 3 and Amendment A agree: per-chain = warning,
run mean = gate. CLIP FAIL and RECALL FAIL stay per-chain.

Gate pass criteria (ALL must hold; numbers from the external review):
- 0 `[unsummarized` across all compactions; ≤1/6 `stop=length`.
- baseline anchor_hit after C3 ∈ [0.40, 0.85] (comparator neither dead nor saturated).
- nc recall usable on 100% of asks (no RECALL FAIL: raw payload non-empty, no error, ≥1 packed
  content unit); `recall_packed_tokens ≤ recall cap ≤ 1024`.
- Realized memory-token ratio, RUN MEAN, ∈ [0.80, 1.05] (per-chain values are warnings).
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
Multi-batch runs: merge records/compactions/chain_budget by chain into ONE dir first (see
HANDOFF.md merge procedure; the same qid/arm must not appear twice, an invalid batch must not be
concatenated with its re-run); sessions/ dirs merge by relative path. run_budget.json is not
recomputed by metrics_v2.)

# Budget ceiling and balance

OpenRouter balance was **$6.52** before the full run (runs/full-spend.txt, 2026-10-26T03:16;
an earlier "~$4.0" figure is stale). Remaining-run ceiling: **$2.50**.
Spend check before launch: `python3 -c` balance probe or OpenRouter dashboard.
Abort criteria mid-run (runner or operator):
- 2 consecutive chains with HARS persist failure (`memory_compact` not ok or compactions < 3) → stop, infra blame.
- ANY recall_error run (`recall_raw_chars=0`, e.g. HTTP 530 provider outage): the nc arm is
  dead for those chains (note-only in disguise) — the runner marks the chain valid=false
  (RECALL FAIL) and metrics drop it; re-run the batch after the channel recovers. The runner
  performs a `memory_recall` preflight automatically (invariant 4b); also check by hand
  against :8307 before every batch.
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

How metrics_v2.py implements the rules (code state, verify against the script before citing):
- Pairs: nc−baseline, nc−nc-note-only, baseline−nc-note-only; deltas are first minus second.
  Bootstrap (`--boot` 10000, `--seed` 1337) clusters by SESSION (chain_id minus `_cN`;
  `--cluster chain` only for comparison); Wilcoxon over per-cluster means.
- Anchor rules use `clean_anchor_hit` (primed hits excluded) when defined on every answerable
  record with an anchor verdict (needs `--judge-input`); otherwise they fall back to raw
  `anchor_hit` with a printed WARNING.
- WIN anchor rule: Δanchor CI95 lower ≥ −0.02 AND point ≥ 0 (rule 1). Unanswerable false-answer
  rule (rule 2): Δ(nc−baseline) ≤ −0.10, or nc ≤ 0.25 absolute when baseline ≤ 0.30.
- Memory ratio (rule 4) = run-mean nc `memory_tokens` / run-mean baseline `memory_tokens`.
- Missing input ⇒ rule is n/a: a WIN rule that is n/a is not met; an n/a FAIL rule blocks WIN;
  result INCONCLUSIVE, never FAIL. FAIL wins over WIN when both evaluable.
- `MIN_CHAINS = 8` counts distinct SESSIONS. Below it the verdict is INCONCLUSIVE regardless of
  the rules (they are printed as diagnostics). The current dataset has 5 sessions (HANDOFF.md,
  OPEN PROTOCOL ISSUES B4), so the decisive verdict is not reachable as built.

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
