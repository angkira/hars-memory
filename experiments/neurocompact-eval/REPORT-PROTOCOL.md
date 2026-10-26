# Protocol — chained long-replay A/B: externalized graph memory vs in-context compaction

Status: protocol frozen for the decisive run; this document describes design and measurement
only. All outcome numbers are in the marked placeholder section at the end.

**AMENDMENT C (2026-10-26, section 4a) SUPERSEDES sections 2, 3 and 5 wherever they differ**
(arms, dataset, treatment, hypotheses, budget, decision set). Sections 2-3 and 5 below describe
the pre-amendment design and are kept as the audit trail of what the exploratory runs were. It
was written and committed before any result of the amended design exists.

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

## 4a. PROTOCOL AMENDMENT C (frozen 2026-10-26, before any result under the new code)

Provenance: plan `.plans/2026-10-26_session-graph-recall.md` (commit 8ffb91d, section 6 items
1-13) and the owner decisions of 2026-10-26, encoded here verbatim as frozen. No number, rule or
threshold below may be changed after the first amended run produces a record; a deviation is a
new amendment (D) written before the next run, never an edit of this one.

**C1. Claim correction (M2).** In split mode `memory_recall` never read the session graph: with no
`ll_keywords`/`hl_keywords` the mode resolved to naive and the graph branch was skipped, so the
old `nc` arm is chunk-RAG over the compaction documents (cosine + token F1), not graph memory.
Every prior nc number is therefore a chunk-RAG number and EXPLORATORY ONLY: batch 1
(`runs/full-20261026T021613`), `fullB-075703`, gate1-gate7, calib runs. They are EXCLUDED from the
decision set; metrics_v2 reads their `nc` records as `nc-chunks`, labels the report
"LEGACY / EXPLORATORY" and never combines them with amended-design records.

**C2. Arms (four) and frozen recall call shapes.**

| Arm | Post-compaction artifact | Ask-time memory | memory_recall args |
|---|---|---|---|
| `baseline` | structured summary, hard-trimmed to B (`NC_SUM_SECTION_TRIM=1`) | summary alone | none |
| `nc-chunks` (was `nc`) | note, cap `note_cap(B)`, persisted to HARS | note + packed recall | `{session_id, question, top_k: 8}` (byte-identical to the pre-amendment call) |
| `nc-graph` | fork (`cp -a`) of the nc-chunks session after C3: same note, same compactions, same HARS session dir | note + packed recall, cap identical to nc-chunks | `{session_id, question, top_k: 8, session_graph: true, session_graph_mode: "hybrid"}`; no caller keywords |
| `nc-note-only` | fork of nc-chunks after C3 | note alone, 0 recall bytes | none |

`nc-graph` differs from `nc-chunks` ONLY in `session_graph: true` (and the mode string). Server
side parameters not named in the call (fetch_top_k, KG byte caps 500/4500, rerank off) are the
server defaults of the code revision under test; the server git sha is recorded in the run
MANIFEST. The nc-graph recall is made on the same `nc-chunks` HARS session, read-only, right
after C3, before any ask.

**C3. Keyword policy (nc-graph), frozen.** LLM keyword extraction by the extractor model
(LightRAG LLM cache ON, keywords cached per question), falling back to the question text as
keywords if the LLM call errors. The runner supplies no keywords. `keyword_source`
(`llm` | `question_fallback` | `caller`), the extracted `ll_keywords`/`hl_keywords` and the keyword
model id are logged per ask. The keyword call is part of the ask: its time is inside the recall
call and is counted in the median-ask-latency rules (metrics_v2 ask latency = `latency_s` +
`recall_seconds`), and its tokens are counted in cost/question (server-reported usage; if the
server reports none, a flagged estimate of 450 in / 80 out tokens per LLM-keyword ask is added
and the number of estimated rows is printed).

**C4. Hypotheses, in decreasing authority.**

- *PRIMARY (the only decision):* `nc-graph` vs `baseline` at B = 1536, paired per question,
  cluster bootstrap over source SESSIONS, evaluated with the unchanged section-5 WIN / FAIL
  rules (verbatim thresholds; every "nc" in them reads "nc-graph"; rule 3 is
  `nc-graph - nc-note-only`; the dead-memory FAIL rule is `nc-graph` vs `nc-note-only`). One
  primary test, no multiplicity correction. Verdict INCONCLUSIVE below MIN_CHAINS = 8 distinct
  sessions (unchanged).
- *Descriptive ablation:* `nc-chunks` (the old nc arm). Reported in every table, never tested
  against baseline, never enters the verdict.
- *SECONDARY family (fixed, m = 5), Holm step-down at one-sided alpha = 0.025* (= the two-sided
  95% CIs used everywhere). Tests use the cluster bootstrap over sessions; one-sided p = (1 +
  #{bootstrap mean <= theta0}) / (N + 1) on the same resamples as the CIs; a cell with no data
  counts as p = 1 (the family never shrinks); an underpowered run (< 8 sessions) labels every
  secondary INCONCLUSIVE regardless of p. Metric: `clean_anchor_hit` (raw `anchor_hit` only on the
  loud fallback). Decision is on the Holm-adjusted p.
  - S1 H-graph: `nc-graph - nc-chunks`, B = 1536; H0 delta <= 0. Descriptive label from the plan's
    thresholds: *graph helps* = (Δanchor >= +5 pp and CI95 lower > 0) or (Δscore >= +0.10 and CI95
    lower > 0); *graph dead* = |Δanchor| < 3 pp and |Δscore| < 0.05; *graph harms* = Δanchor CI95
    upper < 0. (Score is reported, the test is on anchor.)
  - S2 token saving @1024: `nc-graph@1024` non-inferior to `baseline@1536`.
  - S3 token saving @512: `nc-graph@512` non-inferior to `baseline@1536`.
  - S4 multi-hop: `nc-graph - baseline`, B = 1536, multi-hop stratum only (`qtype` multihop).
  - S5 multi-hop graph: `nc-graph - nc-chunks`, B = 1536, multi-hop stratum only. **[Superseded by C.2-1: S4 and S5 are
    DESCRIPTIVE on the `xspan_locator` stratum; the Holm family is S1-S3, m = 3.]**
- *Exploratory (no inference claimed):* everything else, in particular lag 1/2/3, HEAD/TAIL,
  anchor type, per-budget dose-response, nc-chunks vs baseline, mechanism readouts.

**C4a. Non-inferiority margins (S2, S3), frozen.** Margin Δanchor = 0.10 (10 pp of
clean_anchor_hit) AND margin Δscore = 0.15 (judge score, 0-2 scale); non-inferior iff BOTH
one-sided cluster-bootstrap tests reject H0 delta <= -margin (intersection-union: p = max of the
two, no extra multiplicity cost). Justification: (i) the pairing +0.15 score <-> +0.10 anchor is
the protocol's own equivalence (WIN rule 3), so the margins are not new magnitudes; (ii) it is twice
the equal-cost FAIL line (Δanchor < -5 pp): at B = 1024 / 512 the arm carries 1/3 / 2/3 fewer
memory tokens, so a larger loss than at equal cost is tolerated, but not one that erases the
+10 pp the memory channel must show over the note alone; (iii) precision: with ~14 sessions the
paired session-cluster CI half-width on anchor is of the order of 0.08-0.12 (batch 1: ±0.25 on 3
clusters, scaled by sqrt(3/14)), so a margin much below 0.10 could not be demonstrated even for a
zero true loss. The same margin applies at 512 (it is NOT widened for the larger saving). Both
tests are expected to be power-limited; an interval wider than the margin is INCONCLUSIVE, not a
failure of the claim. The margin was fixed before any amended record existed and is not tuned.

**C5. Packer v2 frozen**, including the 70/30 chunk/graph split when a KG section is present,
`PACK_FACTS_BONUS`, `PACK_QUERY_BONUS = 1.0`; no per-arm tuning. The graph arm therefore trades its
KG share for the last ~30% of chunk tokens, and that trade IS the treatment.

**C6. Budget sweep (same run).** `mem_budget` B in {1536, 1024, 512} for EVERY memory arm; every
B-dependent quantity is a frozen function of B (Python `round`; no value of this table sits on a .5 tie):

| B | baseline summary trim | note cap `round(640*B/1536)` | recall floor `round(256*B/1536)` | recall ceiling `round(1024*B/1536)` |
|---|---|---|---|---|
| 1536 | 1536 | 640 | 256 | 1024 |
| 1024 | 1024 | 427 | 171 | 683 |
| 512 | 512 | 213 | 85 | 341 |

recall cap = `clamp(B - realized_C3_note_tokens, floor(B), ceil(B))` (realized note, never the
configured cap; the baseline's realized size is reported only). At B = 1536 these are exactly the
pre-amendment constants. Each B is an independent set of sessions (replays, compactions, HARS
sessions `cr-...-b<B>-...`); the primary B runs first. The PRIMARY decision is taken at B = 1536
only; B = 1024 / 512 feed S2 / S3 and descriptive dose-response tables. Cost multiplier (explicit,
vs one 3-arm run at one B): replays and summarizer/persist compactions x3; ask turns x4 (4 arms x 3
B vs 3); memory_recall calls x6 (2 recall arms x 3 B), half of them with one extra keyword LLM
call; HARS extraction x3 plus ~20% from M3 (more chunks per compaction); judge calls x4.
Combined with 14 sessions instead of 10 chains and Haiku list prices (≈7x qwen-flash input, ≈10x
output; haiku-5-5 price unverified, haiku-4-5 used as proxy) the run costs of the order of 30x
the old $2.50 ceiling. This is an order-of-magnitude estimate, not a quote; `--dry-run
--mem-budgets 1536,1024,512` prints the projection and the owner sets the ceiling before launch.

**C7. Parity.** The run-mean realized-memory ratio gate [0.80, 1.05] is evaluated separately per B
and per treatment (nc-graph and nc-chunks vs the baseline of the same B); only the B = 1536
nc-graph ratio enters the primary WIN rule 4. Per-chain deviations remain warnings.

**C8. Validity gates for nc-graph.** A chain is `valid = false` (RECALL FAIL (nc-graph)) when ANY
nc-graph ask has `session_graph.status` in {error, skipped, unknown_session}, a missing
`session_graph` block (a server that ignored the flag would silently make the arm nc-chunks), or
any generic recall failure. An empty KG result (`status = empty`, 0 entities and 0 relations) is NOT
invalidating; `kg_empty_rate` is reported per budget and > 30% of asks raises the diagnostic
"graph not exercised". Validity is per (chain, B) cell; incomplete arm quadruplets are dropped.

**C9. M3 fix in force for every nc compaction.** The whole serialized dropped span (still capped
at `RAW_CAP = 60000`, identical to the baseline summarizer's input) is cut at message boundaries
into pieces of at most 8000 chars (oversized single messages split at whitespace) and sent in
order as the `dropped_context` items of the compaction's `memory_compact` call. Deviation from the
lead brief ("one memory_compact call per piece"): ONE call per compaction, because each call
allocates a new compaction seq and re-indexes the summary, which would break the
`compactions == 3` persistence gate and duplicate the note in the store; the pieces are still
indexed separately, in order. `NC_DROP_MODE=headtail` reproduces the old design for measurement
only. Step 0 ($0, `scripts/m3_anchor_report.py`): anchors present in the persisted compaction
documents vs the span, reported alongside the results.

**C10. Frozen extraction / model configuration (this run).** Agent, replay, compactor and answer
model: `claude-haiku-5-5` via `pi --provider anthropic`, reasoning off. HARS extractor and
session-graph keyword model: `claude-haiku-5-5` via Anthropic. This DIFFERS from the pre-amendment
glm extractor (`z-ai/glm-5.3-flash`), so graphs are not comparable with any earlier run.
Gleaning (`HARS_MEMORY_MAX_GLEANING`), chunk size/overlap, `HARS_MEMORY_SESSION_INDEX_BS` and the
server revision are recorded in the MANIFEST at launch and may not change within the run.
**[Superseded by C.1-3: the owner's real embedder is used for the decision runs.] Embedder: only the hash bag-of-words shim exists in this environment (HuggingFace is blocked);
it is a frozen configuration and a limitation: retrieval quality of both recall arms (and the
vector part of the graph arm) is lexical, a real embedding model may change every recall number.**

**C11. Funnel redefinitions.** `anchor_in_graph` = anchor in the session KG (entity names,
entity/relation descriptions); `anchor_in_recall` is split into KG section vs chunk section (and
the per-ask records carry `anchor_in_kg_recall` / `anchor_in_chunk_recall`); the "kill graph" rule
becomes `nc-graph` vs `nc-chunks` (S1), the BM25 comparison is kept for the chunk channel.
(Implementation of the funnel changes is outside this amendment's commit and is listed as an open
item.)

**C12. Dataset (B4) and tuning chains (M7).** The dataset is rebuilt from public OpenHands
SWE-bench Verified sessions: 14 independent sessions, at most 3 per repository.
`MIN_CHAINS = 8` distinct sessions stays. Questions target TRANSCRIPT specifics (file paths,
command outputs, intermediate errors), never the known SWE-bench fix (the agent model may have
seen the benchmark; a question answerable from the fix alone would measure contamination). A new
item type `qtype = multihop` (cross-compaction: facts spanning two compactions, e.g. "the value
after the change") is declared a pre-registered SECONDARY stratum (S4, S5); **[Superseded by C.2-1: relabelled `xspan_locator`, descriptive.]** such items carry an
`evidence_spans` list, every leg of which the runner re-verifies verbatim in its span. The
old tuning chains (`qq8Vqbow_c0`, `y062eGZR_c0`) belong to the retired dataset and are not in the
decision set; the packer was tuned on them only.

**C13. Strata (diagnostic only):** lag 1/2/3, HEAD/TAIL, anchor type, budget; the multi-hop stratum
is the only stratum with inferential endpoints (S4, S5).

**C14. Judge.** One fixed judge for all arms, blind and evidence-grounded as in section 3, with
the runner's family-refusal intact: the judge must be of a different model family than the agent
(`claude-haiku-5-5`, family anthropic). **[Resolved by C.1-2: same-family, different-model judge via the explicit flag, limitation stated.] OPEN ISSUE, not resolved here:** in the build
environment only `api.anthropic.com` is reachable (`openrouter.ai` is blocked), so every reachable
judge is the agent's family and `run_chainreplay.py` refuses it by design; `judge_v2.py` also only
speaks to OpenRouter. The check is not weakened. A non-anthropic judge route (or an explicit
owner amendment with bias controls) is a precondition for judging; generation runs may proceed
and be judged later from the archived artifacts.

**C15. Hygiene.** Run artifacts follow CLAUDE.md ("never lose a run"): per-ask raw outputs,
recall payload summaries, keyword sources, usage and the exact command line are archived under
`/mnt/datasets/ft/artifacts/<UTC>-<name>/`. Retrieval is nondeterministic across calls (section
6): run-time records are authoritative, never offline refetches.

### 4a.1 PROTOCOL AMENDMENT C.1 (2026-10-26, written before any result of the amended design was seen)

Provenance: owner decisions of 2026-10-26 (home runs, vendor-agnostic tooling). Nothing here changes a
metric, threshold, arm, budget formula or decision rule of C1-C13; it fixes the run configuration
(C10, C14), the judge-independence exception and the cost rule. Like C, it may not be edited after the
first amended record exists; a deviation is Amendment D.

**C.1-1. Home-run configuration (supersedes the model lines of C10 and the "only Anthropic is
reachable" premise of C14).** Profile `anthropic` (`experiments/neurocompact-eval/profiles/anthropic.json`):
agent / replay / compactor / answer model `claude-haiku-5-5`; HARS extractor and session-graph keyword
(query) model `claude-haiku-5-5`; judge `claude-sonnet-5-5`. The runs are made by the owner at home on
the owner's machine. Run with `run_chainreplay.py --profile anthropic --allow-same-family-judge
--budget-usd <cap>`; the HARS server is started from the output of `providers.py hars-env --profile
anthropic` (extractor + query endpoint `https://api.anthropic.com/v1`, `HARS_MEMORY_LLM_OMIT_SAMPLING_PARAMS=1`
because Anthropic 5.x rejects temperature/top_p).

**C.1-2. Judge: same family, different model, limitation stated.** The judge `claude-sonnet-5-5` is
a different MODEL from the agent `claude-haiku-5-5` but the SAME family. The C14 rule (judge family !=
agent family, refused by default) is NOT weakened: it still refuses by default in `run_chainreplay.py`
and `judge_v2.py`. The exception is the explicit flag `--allow-same-family-judge`, which (a) requires
judge model != every agent-side model (agent, answer, summarizer; checked across router id spellings,
e.g. `claude-haiku-5-5` == `anthropic/claude-haiku-5.5`), (b) prints a loud warning, (c) records
`judge_same_family: true` and the reason in `run_manifest.json`, the judge `mapping` file and every judged row,
and (d) makes `metrics_v2.py` print "LIMITATION - SAME-FAMILY JUDGE" in the report header. Consequence,
to be repeated in the final report: self-preference / shared-bias of the judge toward Haiku-written
answers is NOT controlled; the evidence-grounded blind design (C14, section 3) and the deterministic
`clean_anchor_hit` (not judge-dependent) are the mitigations, and the anchor-based endpoints (primary
rules on `clean_anchor_hit`, S1-S5) are the headline; judge-scored endpoints (score, hallucination) are
read as "same-family judged". The judge is fixed for the whole decision set.

**C.1-3. Embedder (supersedes the hash-shim line of C10).** The owner's real embedder is used; the
hash bag-of-words shim was this build container's debugging setup only and is not a valid decision
configuration. The embedder is independent of the chat profile (Anthropic has no embeddings API) and
is configured on the HARS server (`HARS_MEMORY_EMBED_*`, see `config/.env.example`). `run_chainreplay.py`
records the ACTUAL embedder id in `run_manifest.json`: the id the server reports
(`memory_status.configured_models.embedder`) wins, else `--embedder-id`, else the runner's own env
(flagged "not verified against the server"); an unknown or hash/shim embedder is flagged loudly and a
run whose manifest has no real embedder id is not a decision run. The embedder is frozen for the
decision set (a different id in two merged runs is refused by `merge_runs.py`).

**C.1-4. Vendor-switch rule.** A profile is FROZEN per decision set. Changing the profile (any
role's vendor or model: agent, extractor, keyword, judge) or the embedder in the middle of a decision
set is forbidden. A decision set = all runs merged into one `metrics_v2` input; `merge_runs.py` refuses
to merge runs whose `run_manifest.json` disagree on profile, role models or embedder id, or whose
`chain_budget` rows disagree on agent model/provider/judge model/tokenizer/per-B caps. Switching to
`openrouter` or `gemini` (`--profile`, `--model-<role>`; no code change) means a NEW decision set from
scratch; results across profiles are separate descriptive replications, never pooled. Mixing legacy
(`nc`) rows with amended rows stays forbidden (C1).

**C.1-5. Cost ceiling.** Every paid script (`run_chainreplay.py`, `judge_v2.py`, and any script that
adopts `providers.chat`) requires a hard `--budget-usd` cap and refuses to start a live run without
it. The runner charges measured agent-side usage per assistant turn (profile price table) plus flagged
ESTIMATES for the server-side HARS extraction and keyword calls (constants `HARS_EXTRACT_IN/OUT`,
`KEYWORD_IN/OUT`), checks the cap before every pi call and stops at the cap ("BUDGET STOP": no new chain or
budget cell starts; the aborted chain is written valid=false). B-major order (C6) means a stop still
leaves the primary B = 1536 cell first. The judge stops before the call that would exceed the cap. The
cap values are **TBD by the owner** (placeholders: generation run `<GEN_CAP_USD>`, judge run
`<JUDGE_CAP_USD>`; to be written into the MANIFEST at launch, not into this document after the fact).
`--dry-run` prints the projection and a BUDGET CHECK against `--budget-usd`. Prices come from the profile
tables; each entry carries `status` (`confirmed` = web sources and the pi-ai 1.4.0 catalog agree,
`catalog_only`, `unverified`), source and date, and the status is recorded in the manifest.

### 4a.2 PROTOCOL AMENDMENT C.2 (2026-10-26, written before any decision-set result was seen)

Provenance: the dataset agent's spot check of the OpenHands multi-hop items (artifact
`20261026T144511Z-oh-multihop-qa`) and the owner's decision to add the long-session track
(`.plans/2026-10-26_long-session-eval.md`). No record of any decision set (OpenHands chains, BEAM,
Claude Code) exists at the time of writing: only offline fixtures and cents-level smoke runs on the
hash-shim embedder, which are not decision data. Like C and C.1, nothing below may be edited after
the first decision-set record exists; a deviation is Amendment D. C.2 changes the secondary family
and the meaning of one stratum; it does not touch any metric, threshold, arm, budget formula,
parity gate or the section-5 WIN / FAIL rules.

**C.2-1. The OpenHands "multihop" items are relabelled `xspan_locator` (supersedes the multi-hop
wording of C4 S4/S5, C12 and C13).** The spot check found that only 1 of 10 inspected items
genuinely needs both hops. The rest contain a LOCATOR in an old span (a file, a name, a step) and a
FACT in a newer span that is answerable once the locator is known: they test whether a locator that
lived in an old span and a fact that lives in a newer span can both be retained, NOT reasoning
across a compaction. They are therefore analysed under the label `xspan_locator`. The data files are
NOT rewritten (the stored `qtype` stays `"multihop"`, `verify_chains.py` keeps its gates, the runner
keeps writing the stored label); the mapping happens at LOAD time in every analysis script through
`scripts/qtypes.py::canon_qtype` (metrics_v2 `normalize`, funnel_chains), and
`metrics_v2.py --legacy-qtypes` restores the pre-C.2 reading for old artifacts. Consequence: S4 and S5
on the OpenHands set are DESCRIPTIVE (reported as S4d / S5d on the `xspan_locator` stratum, no p-value
enters any correction, no inference is claimed). The chain-track secondary family becomes
**S1, S2, S3 (m = 3)**, Holm step-down at one-sided alpha 0.025 exactly as in C4 (a cell without
data still counts as p = 1; the family does not shrink further). The change makes the correction
LESS conservative (m 5 -> 3) and is declared here, before any decision data, for that reason.

**C.2-2. Genuine multi-hop moves to BEAM `multi_session_reasoning` on the long-session track.** The
declared multi-hop stratum is the BEAM question type `multi_session_reasoning` (stored with
`qtype = "multihop"` and `source_qtype = "multi_session_reasoning"`; the analysis label is the
`source_qtype`). It is the only stratum with inferential endpoints besides the primary (L4, L5 below).
`knowledge_update`, `temporal_reasoning` and `contradiction_resolution` are DECLARED strata: they are
reported with paired nc-graph - baseline contrasts and cluster-bootstrap CIs, as descriptive
(D_<stratum>, no p-value in the family). `information_extraction` (stored `single`) and the remaining
BEAM types are exploratory.

**C.2-3. Long-session track design.**

- *Data and units.* BEAM size 1M, ALL chats the import finds in the BEAM repository (35 per the BEAM
  dataset description; `import_beam.py` prints the count and `inputs.json` of the run records the sid
  list - if the repo holds another number, that number is the sample and is stated in the report; no
  chat is dropped after seeing results). The cluster unit is the chat (sid); `MIN_CHAINS = 8`
  distinct sessions stays. BEAM sizes are never pooled: the analysis set is selected with
  `metrics_v2.py --include-sid '^beam-1M-'`; BEAM-10M, 100K and 500K runs, if made, are separate
  descriptive replications. Question types imported: information_extraction, knowledge_update,
  temporal_reasoning, multi_session_reasoning, contradiction_resolution (`import_beam.py` defaults).
- *Frozen run configuration.* Profile `anthropic` as in C.1 (agent / summariser / answer
  `claude-haiku-5-5`, HARS extractor and keyword model `claude-haiku-5-5`, judge `claude-sonnet-5-5`
  via `--allow-same-family-judge`, real embedder); boundaries window 150000 / keep-tail 20000 / reserve
  2048 (`run_longsession.py` defaults, arm-independent, forced final boundary); the C2 recall call
  shapes, C3 keyword policy, C5 packer, C6 caps and C9 persistence apply unchanged to the nc arms;
  `--checkpoints end`. The primary analysis uses the END-checkpoint rows only (`metrics_v2.py
  --end-only`, every question once per session); mid-checkpoint rows (retention curves) are
  exploratory.
- *PRIMARY (the only decision of the track):* `nc-graph` vs `baseline` at B = 1536 on BEAM-1M, paired per
  question, cluster bootstrap over chats, evaluated with the unchanged section-5 WIN / FAIL rules
  (verbatim thresholds, every "nc" reads "nc-graph", exactly as in C4). Judge: C.1-2, the BEAM
  rubric is passed to the judge for every item that carries one (`judge_v2.py` rubric block), so the
  score legs are rubric-graded. The anchor legs are computed on the items that carry a deterministic
  `gold_anchor` (BEAM answers are mostly sentences, so this is a minority subset: its size and the
  number of chats contributing are printed with the verdict); a rule whose input is missing is n/a and
  cannot be met, exactly as in C4 (verdict INCONCLUSIVE, never FAIL or WIN by default).
- *SECONDARY family L1-L5, fixed (m = 5), Holm step-down at one-sided alpha 0.025.* Test metric: the
  rubric-graded judge SCORE (0-2, answerable items), NOT `clean_anchor_hit`: the BEAM import shows that
  only 34 of 350 BEAM-1M items carry a deterministic `gold_anchor` (31 of them `knowledge_update`; 1 of
  70 `multi_session_reasoning` items), so an anchor test of L4 / L5 would be empty by construction.
  Non-inferiority (L2, L3) therefore has ONE leg, score margin 0.15 (the anchor leg of C4a is dropped
  for this track only, for that reason). Everything else as in C4: cluster bootstrap over chats, one-sided
  p on the same resamples as the CIs, cells without data count as p = 1, underpowered (< 8 chats) =
  INCONCLUSIVE: L1 `nc-graph - nc-chunks` @1536; L2 `nc-graph@1024` non-inferior to `baseline@1536`; L3
  `nc-graph@512` non-inferior to `baseline@1536`; L4 `nc-graph - baseline` @1536 on
  `multi_session_reasoning`; L5 `nc-graph - nc-chunks` @1536 on `multi_session_reasoning`. L2 and L3
  need the B sweep (`--mem-budgets 1536,1024,512`); if the sweep is not run they are n/a (p = 1) and
  the family stays at five. `nc-chunks`, `nc-note-only`, the anchor-based readouts and every other
  contrast are descriptive.
- *Claude Code sessions (owner's transcripts): DESCRIPTIVE ONLY.* Their questions are generated
  (`gen_longsession_qa.py`, verbatim-evidence gates, generator `claude-sonnet-5-5`), the sample is
  small and selected by the owner, and the real summary (`orig` arm) exists only at real boundaries
  and at the reference B. No hypothesis test, family or verdict is attached to them; the `orig` arm is
  a descriptive side arm in every table (never part of the required arm set, never in an inference);
  generated `multihop` items there are `xspan_locator` by C.2-1. Transcripts are redacted on import
  (`ls_redact.py`) and raw transcripts / session graph stores are not part of any result bundle.
- *Holm families declared, in one place:* chain track {S1, S2, S3}; long-session track {L1, L2, L3,
  L4, L5}. The two tracks are separate decision sets (separate datasets, separate primaries) and
  are not pooled.

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
5. **Open protocol issues** (status after Amendment C, section 4a: B4 -> dataset rebuild (C12),
   M2 -> nc-graph arm (C1-C3), M3 -> full-span persistence (C9), M7 -> retired dataset; the text
   below is the pre-amendment record; still open: judge family reachability (C14), the funnel
   changes (C11), M5): B4 (10 chains = 5 independent sessions; shared probe in
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
per-arm (baseline / nc-chunks / nc-graph / nc-note-only) and per mem_budget (1536 primary;
1024, 512 secondary) score mean, hallucination rate, clean anchor_hit,
unanswerable false-answer rate, median ask latency, realized memory tokens and parity ratio,
persisted-compaction rate; per-lag (3/2/1) slices; cluster-bootstrap CIs for the paired
deltas; the WIN/FAIL/INCONCLUSIVE verdict per the gates in section 5; and the kill/blame
funnel table (anchor_in_summary / anchor_in_graph / anchor_in_recall / BM25).
