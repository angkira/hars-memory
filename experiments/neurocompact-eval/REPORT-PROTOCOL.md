# Protocol — chained long-replay A/B: externalized graph memory vs in-context compaction

Status: protocol frozen for the decisive run; this document describes design and measurement
only. All outcome numbers are in the marked placeholder section at the end.

## 1. Claim under test

For long coding-agent sessions, a short handoff note plus externalized searchable memory
(HARS, one graph per session) answers factual questions about earlier context at least as
well as an in-context summary, with fewer hallucinations, at comparable or lower realized
ask-time context cost (brief.md).

The decisive experiment implements the external review's design (argon_answer_r1.md): the
three-arm structure with `nc-note-only` as a zero-compaction fork, C=3 chained compactions
with questions only after C3, realized-token budget parity, blind evidence-grounded judging,
unique-anchor metrics, adversarial unanswerables, cluster-bootstrap gates (clustered by source
session, see section 5), and the
kill/blame funnel are all the reviewer's design choices (argon_answer_r1.md, Q1-Q5).

## 2. Data

| Property | Value | Source |
|---|---|---|
| Corpus | 41 real OpenCode sessions, 99 compaction boundaries | brief.md 2.4 |
| Chains | 10 chains x 3 spans; span = content between consecutive boundaries | chains-manifest.json |
| Window | tail 8 segments per span, replayed verbatim, never truncated | chains-manifest.json (`tail_segments: 8`), brief.md 2.4 |
| Span size | ~24.1-25.2k chars, ~10.1-10.5k est tokens per span | dataset_stats.md |
| QA volume | 70 items: 2 answerable per span (HEAD/TAIL, 60 total) + 1 unanswerable per chain (10 total) | dataset_stats.md, verify_chains.report.md |
| QA generator | `z-ai/glm-5.3-flash` — different family from the agent `qwen/qwen3.8-flash` | chains-manifest.json (`qa_model`); make_chains.py refuses a generator from the agent family |
| Evidence gate | generator must copy an evidence span verbatim; evidence words/item min 10, mean 20.4, max 42 | dataset_stats.md |

Deterministic validation gates (verify_chains.report.md): every chain re-validated by
independent rebuild from the source sessions; gates — evidence verbatim-in-span (EVID),
evidence-word (EWORD), anchor format (no bare digits, ANCH), anchor uniqueness across the
chain's other spans (ANCHOR-UNIQ), question-discrepancy (DISC), question leak (LEAK), format
(FORM), half-membership (HALF), and for unanswerables: probe zero-hit (ZERO) and empty
evidence (EMPTY). All gates passed at rate 1.000 on all 70 items; hard-gate failures: 0
(verify_chains.report.md).

## 3. Arms

| Arm | Post-compaction artifact | Ask-time memory | Compactions |
|---|---|---|---|
| `baseline` | structured summary, hard budget 1536 tokens, section-trimmer enforces the cap (`NC_SUM_SECTION_TRIM=1`); realized size is only reported | the summary alone | 3 (no HARS) |
| `nc` | short schema-matched note, cap 640 tokens | note + injected `memory_recall` packed to the remainder: `clamp(mem_budget 1536 - realized nc C3 note, 256, 1024)` | 3, persisted to HARS |
| `nc-note-only` | fork of the `nc` session at C3; identical note | note only, 0 recall bytes | none extra (fork) |

- Memory service: HARS per-session graphs on :8307 (`nc-hars-chain-glm`, glm extractor,
  ~13-73 s/compact) (RUNBOOK-gate.md). Instance details: INFRA-NOTES.md.
- Agent and compactors: `qwen/qwen3.8-flash` everywhere, reasoning off (RUNBOOK-gate.md;
  run_chainreplay.py `NC_SUM_REASONING_EFFORT=off`).
- Judge: `openai/gpt-6-luna`, one fixed judge for all arms. Blind: no arm labels, shuffled
  presentation order with a logged mapping file; arm-revealing strings inside answers
  (`-nc-`/`-baseline-` session-id parts, recall-wrapper headers, "recalled memory", HARS) are
  masked as `[masked]` in the judge-visible answer only and logged in the mapping.
  Evidence-grounded: the judge sees a
  deterministic source excerpt around the verbatim evidence span, and its hallucination flag
  is split into `unsupported_or_contradicted_claim` vs `abstained`. Unsupported answers are
  re-verified against the model's actually visible context (visible_context.py) before
  counting (argon_answer_r1.md Q2; REPORT-forensics-gate212839.md post-pass).
- Runner invariants: exactly 3 forced compactions per chain per live arm; `[unsummarized`
  fallback aborts the chain (rows written `valid=false`); nc persistence gate after C3
  (`compactions == 3`, `entities > 0`, `vectors_indexed > 0` when reported); every question
  answered on a fresh `cp -a` fork of the post-C3 session (no cross-question contamination);
  QA evidence re-verified at runtime (RUNBOOK-gate.md).
- Chain validity: a chain is `valid=false` on CLIP FAIL (>1/6 unrepaired `stop=length` per
  arm), RECALL FAIL (any nc recall empty/errored/0 packed content units), nc note alone
  busting the budget, baseline writing to HARS, or an abort. The per-chain parity ratio is a
  warning, not a validity criterion. Rows are written once per chain at chain end with the
  final flag; a recall preflight runs before any spend. Metrics and judge export drop invalid
  chains and incomplete arm triplets and hard-fail on duplicate (qid, arm) / rid.
- Ask prompt: identical for all arms, including the abstention sentence ("If you do not have
  the information needed to answer, say that you do not know"); the nc recall block is
  appended after the question and carries no arm-specific abstention instruction.
- Cost ceilings: pre-freeze gate $0.60; full run $2.50 hard (RUNBOOK-gate.md).

## 4. Budget parity protocol, and why (amendment log)

The original caps (baseline 1000 vs nc note 2048 + unbounded recall) granted nc roughly
3.5-6x more ask-time context and were called a fatal confound by the external review
(argon_answer_r1.md Q3). The target became realized-token parity at B=1536: baseline summary
+ nc (note + recall) both land at ~1536 realized ask-time memory tokens, ratio window
[0.80, 1.05] (RUNBOOK-gate.md).

Getting there took two failed gate cycles, recorded in RUNBOOK-gate.md "Amendment log":

1. **gate1, ceiling 1575**: chain1 baseline C3 clipped (`stop=length`, CLIP FAIL) and chain2
   realized ratio 1.261 — a ceiling-only cap both amputates the summary and does not control
   what the model actually writes (RUNBOOK-gate.md).
2. **gate2, ceiling 2240**: run ratio 0.714; the same chain's baseline fill varied 1183 vs
   2047 tokens across runs — static-ceiling parity is a lottery, because the summary cap is
   a ceiling the model under-fills unpredictably (RUNBOOK-gate.md).
3. **AMENDMENT A (final design)**: the baseline budget is enforced by deterministic section
   trim to `mem_budget`; `--baseline-sum-max-tokens` default changed 3072 -> 1536;
   `NC_SUM_SECTION_TRIM=1` for all summarizing arms. Parity is now a protocol property, not
   sampling luck (RUNBOOK-gate.md). The nc recall cap is derived from the realized C3 note
   (`clamp(1536 - realized_nc_note, 256, 1024)`; the baseline's realized C3 is reported only and
   never enters the cap), so nc cannot exceed the budget by
   construction; 256/1024 are guard clamps against starving or busting the channel
   (RUNBOOK-gate.md, run_chainreplay.py). nc note cap calibrated at 640 (0/6 clips, realized
   417-635) on gate-212839 + gate2 + calib2-4 (RUNBOOK-gate.md); the gate-212839 nc-note-only
   arm had been starved (584-600 note tokens) under the old design, which is why the note cap
   and parity window are now gate-checked (REPORT-forensics-gate212839.md).
4. **AMENDMENT B**: recall packer v2 replaces the rank-prefix packer (in-raw anchor
   retention 3/8 -> 6/8); `PACK_QUERY_BONUS` kept but set to 1.0 (off) for the run
   (RUNBOOK-gate.md).

The per-chain ratio window is a warning only; the gate is the run-mean ratio, printed as
`RUN BUDGET ... PASS/FAIL` and written to run_budget.json (metrics_v2 computes the same
run-mean nc/baseline `memory_tokens` ratio from the kept records; it does not read
run_budget.json). CLIP FAIL and RECALL FAIL stay per-chain (RUNBOOK-gate.md).

Gate pass criteria (all must hold before the full run; thresholds from the external review,
RUNBOOK-gate.md): 0 `[unsummarized` across all compactions; <=1/6 `stop=length`; baseline
anchor_hit after C3 in [0.40, 0.85]; nc recall usable (no RECALL FAIL: non-empty raw payload, no error,
>=1 packed content unit) on 100% of asks with `recall_packed_tokens <= 1024`; realized memory-token ratio in [0.80, 1.05]; judge
calibration on duplicates exact agreement >= 90% (kappa >= 0.80) and P(unsupported_claim AND
score=2) <= 5%.

## 5. Measurement stack

**Recall packer v2** (run_chainreplay.py, frozen after the gate-212839 offline eval):
strips per-chunk boilerplate (replay preamble, `received` lines, section banners — ~12-15%
of every budget), unitizes chunks, then fills the budget with a unitized idf-salience
knapsack in HARS rank order (whole-chunk pass capped at 0.15 of the chunk bucket; chunks +
direct edges get >=70% of budget whenever a graph section exists; entities last). In-raw
anchor retention 6/8 at the ~950-token cap (RUNBOOK-gate.md). The query-aware boost was
measured at 5/8 vs 6/8 without — on leak-proof questions it fires on 0-3 units and displaces
anchor-bearing units — and is disabled (`PACK_QUERY_BONUS=1.0`) (RUNBOOK-gate.md;
run_chainreplay.py).

**anchor_hit vs clean_anchor_hit**: raw `anchor_hit` flatters an agent that reconstructs an
anchor from the question's own description. Forensics on gate-212839 found 2 of 3 baseline
anchor_hits were question-primed reconstructions, with the anchor provably absent from every
summary and post-compaction message (REPORT-forensics-gate212839.md). `clean_anchor_hit`
excludes primed hits by checking the anchor and answer specifics against the model's actually
visible context (final summary/note + recall + post-compaction messages; visible_context.py).

**Unanswerable probes as judge-independent hallucination detector**: 1 probe per chain asks
for a plausible detail never present in the session; the ZERO gate requires 0 hits across the
transcript at build time (verify_chains.report.md). Any non-abstaining factual answer is an
objective hallucination that does not depend on judge calibration (argon_answer_r1.md Q2).
On gate-212839: nc abstained 2/2 cleanly; baseline and nc-note-only each hallucinated 1/2 on
the never-mentioned probes (REPORT-forensics-gate212839.md).

**Statistics**: paired per-question deltas with 95% CIs via cluster bootstrap (`--boot`
default 10,000, `--seed` default 1337), plus paired Wilcoxon over per-cluster means. The
cluster unit is the source SESSION (chain_id minus `_cN`), not the chain: c0 and c1 of one
session share 2 of 3 spans (`--cluster chain` exists for comparison only). Questions within a
cluster share summary/graph fate and are not i.i.d. (argon_answer_r1.md Q2 asked for chain
clusters; session clustering is the stricter implementation after the code review). The power
check `MIN_CHAINS = 8` counts distinct sessions; below it metrics_v2 returns INCONCLUSIVE with
no WIN/FAIL verdict. The dataset has 5 sessions (HANDOFF.md OPEN PROTOCOL ISSUES B4).

**Decision gates (verbatim thresholds, RUNBOOK-gate.md "Decision rules", from the external
review)**:

- WIN = ALL four:
  1. paired Δscore CI95 lower >= −0.05 AND Δanchor CI95 lower >= −2 pp (point >= 0);
  2. Δhalluc <= −10 pp with CI95 upper < 0; unanswerable false-answer Δ <= −10 pp
     (or <= 25% absolute if baseline <= 30%);
  3. nc − nc-note-only: Δscore >= +0.15 or Δanchor >= +10 pp with CI lower > 0;
  4. realized mem-token ratio ∈ [0.80, 1.05], persisted >= 95%, median ask <= 20 s.
- FAIL = ANY: Δscore < −0.10; Δanchor < −5 pp; Δhalluc > −5 pp; dead memory (|Δ| vs
  note-only: score < 0.05 AND anchor < 3 pp); median ask > 30 s; cost/Q > 3x baseline.
- Everything else: INCONCLUSIVE / UNDERPOWERED — never reported as a directional win.
- Implementation notes (metrics_v2.py): anchor rules use `clean_anchor_hit` (fallback to raw
  `anchor_hit` with a warning when visible_context is missing); the unanswerable false-answer
  rule is `Δ <= -0.10` or `nc <= 0.25` when `baseline <= 0.30`; the memory ratio is
  run-mean nc / run-mean baseline `memory_tokens`; a missing gate input is n/a and yields
  INCONCLUSIVE, never FAIL; invalid chains and incomplete arm triplets are dropped before any
  statistic.

**Kill/blame funnel** (offline, $0, on archived artifacts; RUNBOOK-gate.md):

- Stage booleans: `anchor_in_summary` (baseline C3 summary), `anchor_in_graph` (HARS
  graphml/KV), `anchor_in_recall` (packed recall string), plus BM25/FTS5 over the same
  dropped_context.
- Kill architecture: `anchor_in_summary` >= 85% after C3 (no headroom), OR
  P(anchor_hit | anchor_in_recall) < 70% (reader broken).
- Kill graph specifically: HARS anchor_in_recall <= BM25 + 5 pp.
- Blame extractor only if: anchor_in_summary <= 60% AND P(anchor_hit|recall) >= 85% AND
  anchor_in_graph < 35% AND chunk-only recall >= 75%.

## 6. Known limitations

1. **Tool-call markup leakage in answers.** Free-tier models emit `<tool_call>`/`<invoke>`
   text through the completions adapter instead of tool calls; such pseudo-calls have
   appeared as submitted answers in gate3, including in `nc-note-only` and also in baseline
   and nc answers (gate3-run.log; same failure class documented in invest/report.md).
2. **HARS retrieval nondeterminism across repeated `memory_recall` calls.** Refetching raw
   recall for the same 12 gate-212839 items returned different payloads for 5/12 items
   (refetched 13,687-16,862 chars vs the run's payloads), and the refetch surfaced 8/12
   anchors where the run's own records delivered 3/12; no anchor delivered in-run was absent
   on refetch (blame_pack212839.json). Consequence: offline refetches overstate the memory
   channel's delivered content; run-time records are authoritative for anchor_in_recall.
3. **Packer budget ceiling.** The pack cap is <=1024 tokens (protocol ceiling) against raw
   recall payloads an order of magnitude larger in chars (ranges above,
   blame_pack212839.json); the packed share of deduped raw content is
   [NUMBERS: pending full run].
4. **Repo-default extractor trap.** A HARS instance launched without extractor env silently
   falls back to the repo default `localhost:8080` — a dead 405-nginx — and produces
   `ok=true` compactions with empty graphs. The as-found :8305 instance is in this state
   (INFRA-NOTES.md).
5. **Open protocol issues** (not resolved by the harness fixes; details and owner questions in
   HANDOFF.md "OPEN PROTOCOL ISSUES"): B4 (10 chains = 5 independent sessions; shared probe in
   7/10 unanswerables; MIN_CHAINS over sessions unreachable), M2 (recall called without
   `ll_keywords`: graph branch probably skipped, nc may be chunk-RAG; unverified), M3
   (`neurocompact.ts` sends only the first and last 12k chars of each dropped span to HARS,
   baseline summarizer sees up to 60k), M5 (batch-A recall_error=530 undiagnosed), M7
   (packer/cap tuning used decision chains `qq8Vqbow_c0`, `y062eGZR_c0`; batch 1 ran under the
   old prompt/cap; judge calibration = two temp-0 passes of one model = repeatability only).
6. **Single host network namespace.** All containers run `--network host`; every HARS
   instance is port-pinned on the one host namespace, so a port collision or a stray listener
   takes out an arm's memory service (hars-api.sh; INFRA-NOTES.md).

## 7. Results

[NUMBERS: pending full run]

This section is intentionally empty until the decisive run completes. It will contain:
per-arm (baseline / nc / nc-note-only) score mean, hallucination rate, clean anchor_hit,
unanswerable false-answer rate, median ask latency, realized memory tokens and parity ratio,
persisted-compaction rate; per-lag (3/2/1) slices; cluster-bootstrap CIs for the paired
deltas; the WIN/FAIL/INCONCLUSIVE verdict per the gates in section 5; and the kill/blame
funnel table (anchor_in_summary / anchor_in_graph / anchor_in_recall / BM25).
