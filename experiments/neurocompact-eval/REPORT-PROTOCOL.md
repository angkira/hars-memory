# Protocol — chained long-replay A/B: externalized graph memory vs in-context compaction

Status: protocol frozen for the decisive run; this document describes design and measurement
only. All outcome numbers are in the marked placeholder section at the end.

**AMENDMENT C (2026-10-26, section 4a) SUPERSEDES sections 2, 3 and 5 wherever they differ**
(arms, dataset, treatment, hypotheses, budget, decision set). Sections 2-3 and 5 below describe
the pre-amendment design and are kept as the audit trail of what the exploratory runs were. It
was written and committed before any result of the amended design exists.

**AMENDMENT C.6 (2026-10-27, section 4a.6)**: the OpenHands track runs on the streaming long-session runner
(`chains_to_longsession.py` + `run_longsession.py`), not on the pi agent replay; chain-replay numbers are not comparable.

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

> **Amendment C.8 (2026-10-28), section 4a.8:** the only memory configuration is the PROJECT layout (the session's dropped context lives in its own project on the owner's server and is read with the global memory's own call); every split-session result (C.6 nc arms, any C.7 run on split) measured the server's session side channel and is diagnostic only.
> **Amendment C.7 (2026-10-28), section 4a.7:** the primary treatment is now `nc-enriched` (memory recalled at compaction time, none at ask time); `nc-chunks` / `nc-graph` are exploratory.

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

### 4a.3 PROTOCOL AMENDMENT C.3 (2026-10-26, written before any decision-set result was seen)

Provenance: the owner's budget decision for the home run ("a couple dozen dialogues is enough",
USD 20-30 in total) and the cost review of the first home-run tooling. No record of any decision set
exists at the time of writing (only offline fixtures and cents-level smoke runs). As with C, C.1 and
C.2, nothing below may be edited after the first decision-set record exists; a deviation is Amendment
D. C.3 changes how the model calls are BILLED and how many units are sampled; it does not touch any
metric, threshold, arm, budget formula (B, note cap, recall cap), parity gate, prompt text, judge, or
the section-5 WIN / FAIL rules.

**C.3-1. Provider prompt caching is a COST-ONLY change.** Anthropic prompt caching marks a stable
prompt prefix (`cache_control`); a request that hits the cache bills those tokens at a lower price and
returns the same completion distribution for the same input. It does not alter what the model is
given. Concretely:

- The text of every model input is byte-identical with and without caching. Direct calls
  (`providers.chat`, used by the long-session runner for the summariser and answer calls) send the
  same system / user strings; with caching ON the last user message is split at a character offset
  into two text blocks whose concatenation is the original string, and `cache_control` marks the
  first block (answers: the memory note + verbatim tail shared by all questions asked at one
  boundary) and the system prompt. The summariser's one-shot prompt is not cached (a cache write that
  is never read costs 1.25x for nothing). Sampling parameters are omitted exactly as before
  (profile policy) and `thinking` stays `disabled` for the direct summariser calls.
- Caching-enabled direct calls use Anthropic's native `/v1/messages` endpoint (the OpenAI-compatible
  endpoint cannot carry `cache_control`); calls with nothing to cache (judge, liveness pings) stay on
  the compatible endpoint, unchanged. The endpoint is a property of the decision set, set once by the
  profile (`cache.enabled`, `--prompt-cache auto|on|off`) and recorded in every run MANIFEST
  (`prompt_cache` block: cached calls, cache read / write tokens, cost, cost without cache); it is
  never switched inside a decision set (C.1-4 applies: a change of the setting is a new decision set).
- The pi agent loop (OpenHands chain track) already marks its requests with `cache_control` by default
  in pi-ai 1.4.0 (retention "short", 5 minutes, refreshed by every hit; `PI_CACHE_RETENTION=long` =
  1 hour). The runner now records this explicitly (`NC_PROMPT_CACHE`, `PI_CACHE_RETENTION`,
  `NC_SUM_CACHE_RETENTION` in the per-arm env printed by `--dry-run`); pi has no switch to disable it
  for the agent loop, so `--prompt-cache off` only affects the cost model and the direct calls. The
  extension's summarizer call keeps `cacheRetention: "none"` (default of `NC_SUM_CACHE_RETENTION`).
- Every arm is cached identically: the same profile setting, the same env for baseline / nc-chunks /
  nc-graph / nc-note-only (`NC_PROMPT_CACHE`, `NC_SUM_CACHE_RETENTION`, `PI_CACHE_RETENTION` are equal
  across arms; the unit test pins this), the same prefix construction for all arms and for the `orig`
  arm. The cost-per-question comparison of section 5 (FAIL rule "cost/Q > 3x baseline") is therefore
  evaluated on cache-priced costs for ALL arms; the recorded `cost_without_cache_usd` allows the
  uncached comparison as a sensitivity check.
- Prices (profile `anthropic`, fields `cache_read`, `cache_write`, `cache_write_1h`; status per entry;
  Anthropic documentation via WebSearch, 2026-10-26): 5-minute cache write = 1.25 x base input price,
  1-hour write = 2 x, cache read = 0.1 x (Haiku 5.5: write USD 0.125, read USD 0.01 per MTok; above
  100k prompt tokens the tier prices apply). Third-party pages report a 0.05 x read multiplier for
  the Sonnet 5.5 generation; the profile keeps the 0.05 x value (USD 0.10) but it is NOT confirmed,
  and it only matters for Sonnet calls, which carry no cached prefix in this design.

**C.3-2. Sample sizes of the home run (fixed here, before results).**

- *OpenHands chain track:* the 14 chains of artifact `20261026T144511Z-oh-multihop-qa`, unchanged.
  Budget cell **B = 1536 only**; the sweep 1536,1024,512 is opt-in (`[oh] budgets`). The S1-S3
  family is defined at the primary B = 1536 (C.2-1), so it is unaffected.
- *BEAM-1M:* **N = 10 chats** replace "all chats" of C.2-3. The ten are drawn deterministically, before
  any run: the chat numbers found under `chats/1M/` are sorted numerically and
  `random.Random("20261026:1M").sample(numbers, 10)` is taken, then sorted; `home_run.py` writes the
  population, seed and selection to `beam/selection.json` and passes the list to `import_beam.py
  --chats`. No chat is added or dropped after seeing results. `MIN_CHAINS = 8` is met (10 chats). B =
  **1536 only**: L2 and L3 need the sweep and are **n/a (p = 1) unless the sweep is enabled**
  (`[beam] budgets = [1536, 1024, 512]`); the Holm family stays m = 5 and L1, L4, L5 are tested at B =
  1536 as in C.2-3. With ten chats the intervals are wide (L4 / L5 rest on roughly twenty
  `multi_session_reasoning` items); an INCONCLUSIVE verdict is the expected outcome for small
  effects and is reported as such, never as a failure of the architecture.
- *Claude Code sessions:* descriptive only (C.2-3). The owner picks the session(s)
  (`home_run.py cc --find TITLE | --session-id ID | --list`); the pick is recorded in
  `cc/selected-session.json`. Generated questions: 12 single + 4 multihop per session
  (`gen_longsession_qa.py` verbatim-evidence gates unchanged).

**C.3-3. Cost ceiling values (fills the placeholders of C.1-5).** Suggested hard caps for the
home run: QA generation USD 3, runs (replay / compaction / answers + the flagged HARS estimate) USD 20,
judge USD 5, total USD 30 across all tracks (`[caps]` are campaign-wide per phase unless a per-track
override exists). `home_run.py dry-run` prints the estimate against the total before any spend.

### 4a.4 PROTOCOL AMENDMENT C.4 (2026-10-26, written before any decision-set result was seen)

Provenance: the owner's home smoke. With the baseline cap equal to B (1536) the OpenHands calibration
chain failed twice: the baseline summaries ended `stop=length` on claude-haiku-5-5 (CLIP FAIL) and the
run parity ratio nc / baseline was 1.31 against the window [0.80, 1.05]; the runner's own advice
(`cap_advice`: realized / cap = fill, cap = target / fill) was a baseline cap near 2010. Two tooling
defects kept that advice from being applied: in the per-B sweep path the runner overwrote any explicit
`--baseline-sum-max-tokens` with B, and `home_run.py` never passed a cap. No record of any decision set
exists at the time of writing (only offline fixtures and the cents-level smoke runs). As with C to C.3,
nothing below may be edited after the first decision-set record exists; a deviation is Amendment D. C.4
changes the BASELINE CAP value and how it is obtained; it does not touch any metric, threshold, arm,
the parity window, the realized-token parity gate, the nc note cap, the nc recall cap formula, the
prompt text, the judge, or the section-5 WIN / FAIL rules.

**C.4-1. The baseline cap is a calibrated parity parameter, not a constant.** Amendment A fixed the
baseline cap at B on the assumption that a summarizer fills what it is given. Measured on the
protocol model it does not (fill 0.43-0.63 in the first protocol run, 1183 vs 2047 realized tokens
across runs of one chain, section 4), so "cap = B" does not give realized-memory parity. What is pre-registered is
the PARITY, i.e. the unchanged gate on REALIZED ask-time tokens (`ratio = nc_memory_total /
baseline_memory_total` in [0.80, 1.05], mean over chains, per B and per treatment). The cap is the knob
that makes the baseline's natural fill land there and is therefore set per B by the calibration below,
before any decision-set record, and then frozen. Consequences stated openly:

- With cap(B) > B the baseline's hard ceiling is above B (e.g. 2010 at B = 1536). The baseline is NOT
  given more realized memory than the window allows: the gate is on realized tokens, and the window
  admits a baseline up to 1/0.80 = 1.25 x the nc memory (the unchanged, pre-registered tolerance).
- ONE value feeds everything the baseline summarizer sees: the prompt's "hard output limit" and its
  85 % soft target, the provider `max_tokens` of the generation, the deterministic section trim
  (`NC_SUM_SECTION_TRIM=1`) and the CLIP detection (`stop=length` at that `max_tokens`).
  Generation and trim therefore share the cap by construction; they cannot diverge.
- The nc side is unchanged: note cap round(640 B / 1536), recall cap = clamp(B - realized note,
  round(256 B / 1536), round(1024 B / 1536)). The cap never enters the nc arithmetic.

**C.4-2. Calibration procedure** (`home_run.py calibrate`, `run_chainreplay.py --calibrate`):

1. *Chain.* The calibration chain is the FIRST chain in a seeded order of the artifact's chains:
   the sorted chain ids shuffled by `random.Random("20261026:oh-calibration")` (seed `[oh]
   calibration_seed`), first element. Fixed by the artifact and the seed, so it cannot be picked after
   seeing any result; it is written to `calibration.json` (`chain_selection`).
2. *Run.* The chain goes through both live arms (baseline, nc-chunks) up to C3 at the primary B (1536;
   each further B listed in `[oh] budgets` is calibrated in turn) with NO QA asks and no judge, at a
   starting cap (default B, or an explicit `[oh] baseline_cap` / `--baseline-cap`).
3. *Acceptance* (both must hold): (a) CLIP-free by the same rule as `--require-no-clip` (no arm with more
   than 1/6 unrepaired `stop=length` compactions; with three compactions per arm that is none), and (b)
   the realized ratio of that run inside [0.80, 1.05].
4. *Iteration.* If not accepted, the next cap is the runner's own advice (`recommended_cap` = target
   realized / measured fill); a clipped baseline always grows (at least +10 %); a clipped nc note, a
   missing ratio, or advice that does not move the cap end the calibration as FAILED. At most `N`
   calibration runs per B (`--max-iterations`, `[oh] calibration_max_iterations`, default 2: the initial
   run plus one retry), so the calibration costs a bounded number of cents-level runs (ledger, run cap).
5. *Record.* `calibration.json` in the campaign: status, chain id and selection, per-B cap with every
   iteration (cap, ratio, clip verdicts, advice, cost, run dir), `caps`, `scale`, `ratio` achieved,
   total cost, `excluded_from_decision_set`. A FAILED calibration is recorded as such and is a
   blocking condition (C.4-4), not a result.

**C.4-3. Freeze and uncalibrated B.** After a passing calibration the caps are frozen for the campaign
(`oh/caps.json`; a later invocation with different caps is refused, a new campaign is a new decision
set). A B that was not calibrated uses `scale = cap(primary) / primary`: `cap(B) = round(B * scale)`;
the run records that cap as `basis = scale` (calibrated B: `basis = map`), together with its source
(`default` | `cli` | `calibrated:<run dir>`) per B in `chain_budget`, `records`, `compactions`,
`run_manifest.json` (`baseline_caps`) and `run_budget.json`. Precedence in the runner: explicit per-B
map > scale > legacy `--baseline-sum-max-tokens` (one B only) > default B; an explicit value is never
replaced by B. The per-B parity gate is evaluated on the decision set as before; a scale-derived B that
lands outside the window fails the gate and is reported as such (not re-tuned on decision data).

**C.4-4. The calibration chain is excluded from the decision set.** The chain used to choose the cap
has been seen at run time with the cap tuned against it, which is exactly the tuning-on-decision-chains
failure recorded as M7 in section 6. The calibration chain is therefore removed from the OpenHands
decision set automatically (13 of 14 chains): `home_run.py` writes the decision-set chain file without
it, passes `--exclude-chains` to every run, `merge_runs.py` refuses a merged set that contains it and
`metrics_v2.py` gets `--exclude-sid`; the exclusion is recorded in `calibration.json`, `oh/caps.json`,
`oh_inputs.json` and the run / merge manifests. The `oh` track refuses to run without a passing
calibration unless a cap is given explicitly (`--baseline-cap`, `--baseline-cap-scale` or `[oh]
baseline_cap`, recorded with source `cli`; an explicit cap is a statement of the owner, not a
calibration, and is labelled so). The `smoke` command calibrates on its own single chain WITHOUT
excluding it (a smoke is not a decision set; its `calibration.json` is not reusable).

### 4a.5 PROTOCOL AMENDMENT C.5 (2026-10-26, written before any decision-set result was seen)

Provenance: the owner's home bundle (`20261026T172729Z-homerun-home`, `calibration.json`, `oh/calibration/b1536-i*/`,
`logs/oh.calibrate.*.log`). With Anthropic `claude-haiku-5-5` the baseline summarizer hit `finish_reason=length` at
`max_tokens=1536` (NATIVE Claude tokens) while the very same summary measured **959 tokens in tiktoken cl100k and 1102 in
chars/3.6**. Parity and every budget were measured with cl100k (runner, packer) or chars/3.6 (pi extension `realized_tokens`,
the `summary_chars/3.6` fallback): three counters for one quantity, none of them the model's; the generation caps
(`max_tokens`) are native tokens. So the nc arm's recall + note, packed to ~1536 "our" tokens, was ~1.4-1.6 x more NATIVE tokens
than the baseline's 1536-token generation could ever hold, and the C.4 calibration ratio (1.32 at cap 1536, 1.20 at cap 2026)
was mostly this exchange rate (plus noise: the calibration looked at C3 of ONE chain, whose baseline C3 was 1051/1069 against
C1/C2 ~1400). Measured here with the free `count_tokens` endpoint on claude-haiku-5-5 (6000-character samples, request
framing overhead of 8 tokens subtracted): native / cl100k = **1.57 (Python source), 1.53 (protocol prose), 1.64 (TypeScript)**,
native / (chars/3.6) = 1.58-1.60 (the chars/3.6 label was never conservative on Claude). As with C to C.4, nothing below may be
edited after the first decision-set record exists (a deviation is Amendment D). C.5 changes the UNIT in which parity and budgets
are measured; it changes no metric, threshold, arm, window, formula, prompt, judge or section-5 rule.

**C.5-1. Parity and all budgets are measured in the answering model's native tokenizer.** One layer, `scripts/tokcount.py`:

- Anthropic: `POST /v1/messages/count_tokens` (`{"model", "messages":[{"role":"user","content":text}]}` -> `{"input_tokens"}`);
  Gemini: `models/{m}:countTokens` (`totalTokens`); both free (Anthropic: subject to per-tier requests-per-minute limits, hence
  the retry/backoff, the concurrency limit and the on-disk cache keyed by (provider, model, sha256(text))). The request's
  constant framing overhead (8 tokens, measured per model as `count("x") - 1`) is subtracted, so a count is the cost of the TEXT.
  Other providers (OpenRouter, custom): tiktoken cl100k, `native=false`, flagged. Offline/dry-run/tests: a calibrated-ratio
  estimator, `native=false, estimated=true`. A live Anthropic/Gemini run that cannot count natively is REFUSED unless
  `--token-counter tiktoken` accepts a flagged, non-native measurement.
- Applies to everything parity-related, at compaction time and ask time: the baseline's and the nc note's realized tokens,
  the recall payload AS INJECTED (wrapper and header included), `memory_tokens`, `arm_note_tokens`, the recall cap arithmetic,
  the calibration. The tokenizer id and the native flag are stamped on every record, `chain_budget`, `calibration.json`,
  `run_manifest.json` (`token_counter`), `run_budget.json`, `caps.json`.
- Hard trims are to the NATIVE budget: the extension trims the baseline summary and the nc note to `NC_SUM_MAX_TOKENS`
  verified against `count_tokens` (the section trimmer runs at an estimator cap that is lowered until the native count fits);
  the recall packer packs with the fast estimator and then verifies the injected block natively, re-scaling until it fits
  (and growing back toward the budget when it undershoots); long-session notes use the same fit. The extension's chars/3.6
  `realized_tokens` is no longer a budget source (logged as `est_tokens_chars36`; the runner re-counts anyway).
- Consequence for the numbers: B = 1536, the nc note cap (640 at B = 1536) and the recall cap clamp are now NATIVE tokens.
  The formulas are unchanged; in absolute content the nc note and recall shrink by the exchange rate (~1.5x fewer cl100k
  tokens), the baseline summary does not (its generation was always native).

**C.5-2. The baseline cap defaults to cap(B) = B again.** With native counting the cap and the realized memory are the same unit,
so cap(B) = B is the intended default. The C.4 machinery (`--baseline-cap`, `--baseline-cap-scale`, calibration, `caps.json`)
stays; the calibration (a) counts natively, (b) aggregates over all compactions C1..C3 of every calibration chain AND the
ask-time memory and reports both (`ratio_ask_time` = nc (C3 note + injected recall) / baseline C3, the quantity the gate measures,
and `ratio_pooled_all_compactions` over 3k baseline summaries; the cap advice uses the pooled fill), (c) may use k chains
(`[oh] calibration_chains`, default 1; all excluded from the decision set), and (d) accepts when CLIP-free and the run ratio is
in [0.80, 1.05]. The calibration chains are still excluded from the decision set.

**C.5-3. What this invalidates.** All earlier gate/batch parity numbers (cl100k or chars/3.6) overstate nc's share of the
memory by the tokenizer exchange rate on Claude models (~1.5x before the cap), so their ratios must not be compared with
C.5 numbers; the `calibration.json` of the 20261026T172729Z home campaign (and any calibration without `amendment: "C.5"`) is
VOID and `home_run.py oh` refuses to use it. Non-Anthropic/Gemini runs keep a flagged cl100k measurement (`native=false`),
which is a documented limitation of those runs, not a parity claim in the model's tokens.

### 4a.6 PROTOCOL AMENDMENT C.6 (2026-10-27, written before any decision-set result was seen)

Provenance: owner decision of 2026-10-27 (final). The OpenHands track must NOT replay sessions through a pi agent turn by turn.
For the memory question it is enough that, at each compaction boundary, the summariser (baseline) and HARS `memory_compact` + the short
note (nc arms) see the dropped span, and that the questions are asked at the end with each arm's memory plus the same tail: exactly what
`run_longsession.py` already does for BEAM and Claude Code. The pi replay (`run_chainreplay.py`) costs many agent turns and produced the
calibration chain of C.4 and C.5 (cap iteration, tokenizer exchange rate, a calibration chain removed from the decision set). As with C
to C.5, no record of any decision set exists at the time of writing (only offline fixtures and cents-level smoke runs), nothing below may
be edited after the first decision-set record exists, and a deviation is Amendment D. C.6 changes the RUNNER of the OpenHands track and the
meaning of its calibration; it changes no arm, no budget B, no formula, no prompt, no judge, no metric, no threshold, no window and no
section-5 WIN / FAIL rule.

**C.6-1. Runner.** The OpenHands track runs `chains.jsonl` through `run_longsession.py`:

1. `scripts/chains_to_longsession.py` converts each valid chain to a `longsession/1` session. Input: `chains.jsonl` ONLY. A chain record
   already carries the raw events of its three spans (`spans[k].segments = [{role, text}]`, the replay window the questions were verified
   against); the imported OpenHands session JSONs are not read. `sid` = `chain_id`; `messages` = the events of span 1, 2, 3 in order
   (empty texts skipped as in `split_segments`); `compactions` = the chain's synthetic boundaries, one per span end (`at_idx` = first
   message of the next span, `at_idx = len(messages)` for C3 so nothing survives and the tail is empty, as in the replay where the
   questions were asked right after C3); `original_summary = null` (no `orig` arm). Questions: every qa item with its `qid` unchanged,
   `qtype = canon_qtype(stored)` (stored `multihop` -> `xspan_locator`, Amendment C.2; the stored label stays in `source_qtype`),
   `answer` = `canonical_short_answer`, `answerable`, `gold_anchor` and `slice` (HEAD / TAIL / NONE) passed through; evidence = the
   char-exact position(s) `[{msg_idx, start, end}]` of the verbatim evidence snippet(s) (multihop: `evidence_spans`, link hop in its span and
   answer hop in its span); a snippet that cannot be located falls back to its anchor and, failing that, is a conversion ERROR (never a
   silent drop); unanswerable probes get empty evidence. The conversion is deterministic (sorted keys, no clock), and `.conversion.json`
   records sha256 of the input and of every output.
2. `run_longsession.py --use-real-compactions --keep-tail 0 --checkpoints end --arms baseline,nc-chunks,nc-graph,nc-note-only
   --mem-budgets 1536` (the sweep 1536,1024,512 stays opt-in). Boundaries are the converted ones; after C3 the lags are exactly the
   chain's (span 1 -> 3, span 2 -> 2, span 3 -> 1; xspan_locator items 3), which a unit test asserts. Every question is asked once, at the
   end checkpoint, to every arm. Summariser, HARS ingest (the whole dropped span in <= 8000-char pieces via `memory_compact`), note cap,
   recall caps, recall call shapes (`nc-graph` differs from `nc-chunks` ONLY in `session_graph: true`), the ask prompt, native-token
   measurement (C.5) and validity rules (CLIP, ingest, recall, status gate) are the long-session runner's, unchanged from C.1 to C.5.
3. `export_longsession_judge.py` -> `judge_v2.py` (blind, same judge, same cap rules) -> `metrics_v2.py --track longsession --end-only`.
   `visible_context` = memory + injected recall + the (empty) tail; `primed_hit` / `clean_anchor_hit` keep their definition.
   `funnel_chains.py` is run best effort on the run directory (it reads the per-boundary summaries from `memory/<sid>.json`; the stages
   that need pi session directories print n/a).

**C.6-2. What this changes for the protocol, stated openly.**

- The agent no longer "lives" the span. In the replay the model produced the span's turns, tool calls and replies, and the compaction was
  triggered inside the agent loop (extension path); here the summariser and HARS ingest see the dropped span directly, the same text for
  every arm, and the answer is a single call per (question, arm) with memory (+ empty tail) in the prompt. The comparison is therefore
  about the fidelity of each arm's memory of the dropped text, not about an agent working with its own memory. The summariser prompts are
  the hand-synced Python port of `neurocompact.ts` (`PROMPT_VERSION`), not the extension's runtime.
- Boundaries are fixed by the dataset (three, at the span ends), not by a token window; there is no tail.
- Cost and time drop by an order of magnitude (no agent turns); the 14 chains give 14 sessions x 3 compactions and 140 questions x 4 arms.
- All earlier chain-replay numbers, including every gate, ratio and calibration of the pi replay (C.4 and C.5 calibrations, the home
  campaigns, smoke runs), are NOT comparable to C.6 numbers and are not decision-set records. `--legacy-replay` (`home_run.py oh
  --legacy-replay`, `calibrate --legacy-replay`, `smoke --legacy-replay`, `dry-run --legacy-replay`, `doctor --legacy-replay`) keeps the
  pi path, with the C.4 cap iteration, for reproducing them; it is not the decision runner.

**C.6-3. Hypotheses.** Unchanged in content, computed by `metrics_v2.py --track longsession --end-only`, one record per (question, arm, B):

- Primary: the section-5 WIN / FAIL rules of C (`evaluate_gates`, B = 1536, treatment `nc-graph`) including the realized ask-time memory
  ratio window [0.80, 1.05] in native tokens (C.5), MIN_CHAINS = 8 distinct sessions, cluster bootstrap over sessions (`sid` = chain id; the
  14 chains come from 14 distinct OpenHands sessions, the converter warns if two chains ever share one). Unchanged.
- Secondary: the contents are the same (graph vs chunks at B = 1536; token-saving non-inferiority of `nc-graph` at 1024 / 512 against
  `baseline` at 1536, which needs the opt-in sweep and is `n/a` without it; the `xspan_locator` stratum is descriptive). The long-session
  track evaluates them as its family L1 to L3 (L4 and L5, genuine multi-hop, have no data on OpenHands: `multi_session_reasoning` does not
  occur) with the long-session test metric (judge score; clean-anchor is reported alongside) and Holm over the fixed family of m = 5 (C.2).

**C.6-3a (2026-10-27, owner decision, before any decision-set result).** On the OpenHands track the secondary family is restricted to
the hypotheses that have data there: **L1-L3, Holm step-down over the fixed family of m = 3** at one-sided alpha 0.025 (judge score
metric, NI score leg only, as in C.2). L4/L5 (genuine multi-hop, `multi_session_reasoning`) are not computed on OpenHands: they cannot
occur there and would only cost power as n/a = p 1 members. The `xspan_locator` stratum stays descriptive. BEAM and Claude Code keep
L1-L5 (m = 5). Implementation: `metrics_v2.py --ls-family oh` (default `full`); `home_run.py oh` passes it.
  The only consequence for OpenHands is that the correction is over 5, not over the 3 of the chain track; this is the price of one
  metrics path for all three tracks. `home_run oh` also writes the chain-track reading of the same records (`--track chains`: S1 to S3 on
  clean_anchor_hit, m = 3) as `REPORT-oh-chains-family.md`: it is DESCRIPTIVE and never replaces the L-family; when the two differ, the
  L-family is the reported inference.

**C.6-4. Calibration semantics (C.4 and C.5 superseded for the default path).** With native token counting (C.5) and the long-session
summariser with its hard trim to B, parity holds by construction, so there is no baseline-cap calibration and no cap iteration on the
default path: `cap(B) = B` (C.5-2). `home_run.py calibrate` (and the parity step of `smoke`) is a cheap parity CHECK on the
`run_longsession` path over k sessions (`[oh] calibration_chains`, `--check-sessions`, default 1; the first k of the seeded order of C.4-2):

- It compacts the session exactly like the real run (baseline summaries, nc notes, HARS ingest) and asks ONLY two fixed, question-independent
  probe questions per arm (sid `parity-<chain id>`, never a decision-set qid, no gold, never judged). It therefore spends no decision-set
  answer and tunes nothing; it is NOT excluded from the decision set (all 14 chains stay in; the C.4-4 exclusion applies to
  `--legacy-replay` only). Had it asked decision-set questions it would have to be excluded (C.4-4 reasoning); that is the reason for the
  probe design.
- Acceptance, per B: every row valid, CLIP-free (no arm with more than 1/6 unrepaired `stop=length` summaries: with three compactions,
  none), native counting (`tokenizer_native`) on native providers, and the realized ask-time memory ratio (mean over sessions of
  nc memory tokens / baseline memory tokens, memory = note + recall as injected) of BOTH `nc-chunks` and `nc-graph` inside [0.80, 1.05].
  The probes are generic queries, so the check is a proxy of the real-question recall fill; the decision run's own gate
  (`win_mem_token_ratio`, section 5) is authoritative and unchanged.
- Result `calibration.json` (`amendment: "C.6"`, `mode: "parity-check"`). `oh` runs the check first when absent, refuses to run after a
  FAILED check (no iteration is offered: investigate, or `--accept-parity-fail` which records the override and lets the run's own gate
  report the same fact), and treats any calibration without `amendment: "C.6"` as VOID. `--skip-parity-check` skips it (recorded).

**C.6-5. Cost and time** (offline `home_run.py dry-run --tracks oh`, 14 chains at B = 1536, prompt cache ON, 4 arms): run (summarisers,
HARS extraction ESTIMATE, answers, incl. the parity check) about USD 0.33 (the pi replay projected 0.75); judge about USD 1.90 (560 items);
wall-clock about 2.3 h by the runners' own model (the pi replay: 6.9 h). Two inputs of that model are guesses and the first live parity
check (one chain, claude-haiku-5-5, a local HARS with a hash-shim embedder, 2026-10-27) contradicts both: HARS ingest took 146 / 163 / 164 s per
boundary (3-4 pieces of 8000 chars each: about 50 s per piece, the model assumes 11 s), i.e. about 8 min per chain, about 2 h for 14 chains
and about 3 h in total; and HARS made 59 / 64 / 69 extraction chunks per boundary (192 per chain) where the estimate assumes 18, so the
HARS-extraction part of the run estimate (about USD 0.15 of the 0.33) is a lower bound, plausibly up to 10x too low (no extractor usage is
logged on the client side: unverified). The run cap (USD 20) is unaffected. Opt-in sweep: about 3x the asks.

### 4a.7 PROTOCOL AMENDMENT C.7 (2026-10-28, written AFTER the OpenHands question-time result was seen and BEFORE any `nc-enriched` record exists)

Provenance and honesty. The OpenHands home run under C.6 finished first: arms `nc-chunks` / `nc-graph` recall from HARS AT QUESTION TIME with the
question as the query. Result (14 sessions, 140 questions, B = 1536): no significant gain over `baseline`, more hallucination than `baseline`,
`nc-graph` indistinguishable from `nc-chunks`. C.7 is therefore a post-result amendment of the design, not a pre-registration of the old one:
its hypotheses are fixed here before the first `nc-enriched` record exists, and nothing below may be edited after that record exists (a
deviation is Amendment D), but the choice of the new arm was informed by the old result and by the owner's actual hypothesis, which the
question-time arms never tested. C.7 changes the TREATMENT ARM, the default arm set, the metrics regime and the version string. It changes no
budget B, no formula, no ask prompt, no summariser prompt, no judge, no thresholds of the section-5 WIN / FAIL rules, no window.

**C.7-1. Demotion of the question-time arms.** `nc-chunks` and `nc-graph` become EXPLORATORY (descriptive side arms in every report; they
never invalidate a chain of the primary and never enter an inference). Cause found after the result: they were run with a mis-specified
memory call. The frozen call `memory_recall(session_id, question, top_k=8)` sends NO `ll_keywords` / `hl_keywords`, so the server resolves the
query to LightRAG `naive` mode and (HARS_MEMORY_GRAPH_BRANCH=auto) skips its graph branch entirely; `top_k = 8` is below the shipped default
(DEFAULT_QUERY_TOP_K = 20); and our own packer (`run_chainreplay` packing v2) then discards about 80% of the roughly 16k-char result to fit
about 820 tokens. The repo's usage guide (CLAUDE.md, "Using the hars-memory MCP") prescribes the opposite on all three points. The old arms
therefore say nothing about the owner's memory used correctly; their result is kept as a record of that mis-specification, not as evidence
about memory. `nc-graph` additionally differs from `nc-chunks` only by `session_graph`, which does not change this.

**C.7-2. The treatment: `nc-enriched`, ONE memory configuration.** The owner's hypothesis: at COMPACTION time call the memory many times,
fold the recalled facts into the post-compaction context, then let the model work WITHOUT memory access, and compare with a plain summary of
the same token budget. At each boundary that precedes asks (end-only tracks: the last boundary), after the rolling note for budget B exists:

1. The summariser model writes K = 8 (`--enrich-queries`) short retrieval queries, each with `ll_keywords` (exact identifiers: paths,
   function names, error strings, numbers, verbatim) and `hl_keywords` (2-3 topics), from the NOTE and the kept recent context ONLY. A question
   can never reach that prompt: `enrich_query_prompt(note, recent, k)` has no other parameter, and a test asserts it end to end with a sentinel.
2. Each query is one `memory_recall(session_id, question = <query>, ll_keywords, hl_keywords, mode = "hybrid", max_chars)`: the server's
   DEFAULT `top_k` (not sent), no `session_graph`, the output budgeted by the SERVER'S OWN `max_chars` (token cap -> chars with the chars per
   token measured on the session's dropped text with the run tokenizer, verified in native tokens). This is the only memory configuration of
   the decision run (the default session recall); no other configuration is run in the decision.
3. The chunks of the K answers are deduplicated across queries (exact normalised content), interleaved by rank (rank 1 of every query first)
   and filled whole into the cap (a chunk that no longer fits is skipped, like the server's own assembly). `--packer legacy` keeps packing v2 for
   comparison only.
4. Cap = `recall_cap(B, realized note tokens)` = clamp(B - note, floor, ceil) of C.1 (native tokens, C.5), wrapper included. The ask context is
   note + the "facts recalled from long-term memory" block. There is NO recall at ask time. The block is computed ONCE per (sid, boundary, B)
   and persisted under `enrich/` (queries, keywords, per-query latency / errors / graph_branch.enabled / lightrag_mode, raw payload files,
   packed text, tokens); raw recall text also goes to the sidecar `recall_raw.jsonl` (all recall arms), never inline in records.
5. Validity of a block: at most `--enrich-max-failed-frac` (25%) of its recalls failed or came back empty, keywords present on at least half of
   the queries, `graph_branch.enabled` true on at least half of the keyworded recalls (else the server is not in the intended configuration),
   no degraded channel unless `--allow-degraded`, at least one chunk packed, status gate (entities, vectors_indexed). Rows carry
   `anchor_in_enriched` (anchor regex over the packed block) and `anchor_in_enriched_raw` (over the raw recall of all queries), so a loss is
   attributable to retrieval or to the budget; `anchor_in_recall` / `anchor_in_recall_raw` are aliases for the funnel and for the old arms.

**C.7-3. Hypotheses.** Computed by `metrics_v2.py --track longsession --end-only` (auto-detects the regime from the presence of
`nc-enriched` rows): required arm set `baseline` / `nc-enriched` / `nc-note-only`; clustering, MIN_CHAINS = 8, bootstrap as before.

- PRIMARY: `nc-enriched` vs `baseline` (plain summary of the same B): the section-5 WIN / FAIL rules with treatment `nc-enriched`, thresholds
  unchanged, including the realized ask-time memory ratio `nc-enriched / baseline` in [0.80, 1.05] (native tokens, memory = note + block).
- KEY SECONDARY: `nc-enriched` vs `nc-note-only` (the same note without the recalled facts: isolates the facts). Family of m = 2, Holm
  step-down at one-sided alpha 0.025, judge score: E1 `nc-enriched - nc-note-only`, E2 `nc-enriched - baseline` (superiority). Strata
  (knowledge_update, temporal_reasoning, contradiction_resolution, xspan_locator) are descriptive.
- HALLUCINATION BREAKDOWN (descriptive, always reported): per arm, hallucination rate on answerable vs unanswerable items (k/n), paired deltas
  `hallucination_answerable` and `false_answer` (unanswerable).
- `nc-chunks` / `nc-graph`, when present: descriptive pairs only.
- Cost and latency: the compaction-time query-writing call is part of the arm's cost per question (amortised over the questions of its
  boundary) so `fail_cost` cannot be dodged by moving spend out of the ask; ask latency is the ask. The wall-clock of the enrichment is
  recorded (`enrich_seconds`) and is not an ask latency.

**C.7-4. Reuse rule (no re-ingest, no re-summarising, no re-judging).** `run_longsession.py --reuse-run DIR` and the home commands
`home_run.py oh-probe --resume <campaign>` / `oh-enrich --resume <campaign>` run ONLY the new arm against a finished `oh` campaign:

- the HARS sessions `ls-<old run id>-<sid>` are used as they stand; `memory_compact` is never called; a HARS status gate per session
  (compactions > 0, vectors_indexed present and > 0, entities) must pass or nothing is spent;
- the note is the one the old `nc-note-only` arm used: from `memory/<sid>.json` when present (sha-checked against the config), else the
  `memory_text` of the old `nc-note-only` rows (identical across the sid's rows, hashed against the `summary_sha256` of the last note entry of
  `compactions.jsonl`; only the last boundary, hence end-only checkpoints are required: an earlier checkpoint would see HARS data of later
  boundaries); a missing or differing note is an error (`--reuse-regenerate-notes` is the explicit override);
- the finished, VALID rows of `baseline` and `nc-note-only` are copied with all original fields plus `reused_from`; their judge verdicts are
  reused (rid join), the judge sees only the NEW rows, the verdict sets are merged before metrics. The judge model of the old verdicts must
  equal the current profile's judge (else the paired contrast would cross two judges: refused);
- static checks before any spend: input files equal (sha256), the copied arms answered exactly the questions this run asks, the old cells valid,
  `hars_per_budget` equal; then the parity of the realized memory before any judge spend (`--accept-parity-fail` records an override).

Why this is valid, and where its limit is. EVAL-VERSION moves to `2026-10-28.c7`. The paid-step guard keeps refusing every campaign made under
another version, with ONE exception: the commands `oh-probe` / `oh-enrich` accept a campaign stamped `2026-10-27.c6`, nothing older and no other
command (`REUSE_COMPATIBLE_FROM`; the campaign's own stamp is not rewritten, the reuse is recorded under `c7_reuse` in `campaign.json` and the
c7 scripts are snapshotted under `scripts/c7/`). Reuse is valid because everything it reuses was produced by code that C.7 does not touch:
conversion and boundaries (`chains_to_longsession`), the summariser prompts / caps / trimmer, HARS ingest, the ask prompt, native token
counting (C.5), the judge and its blinding. C.7 adds an arm, changes defaults and the metrics regime, and widens the judge's masking list for the
new arm's strings. The enriched arm and the reused arms therefore see the same dropped text, the same note, the same tail (empty), the same
question set and the same ask prompt; only the memory block differs. Residual risks, stated: (a) retrieval is not deterministic across calls
(section 6, item 2), so the enrichment is one realisation, persisted so that it is auditable; (b) the HARS server may have changed since the
ingest (the status gate and the recall validity rules are the guard, not a proof); (c) the new rows are judged in a separate pass after the old
ones (same model, same blinding; a drift would bias against nothing in particular but cannot be excluded); (d) the question set was seen
after the old result: the enrichment never sees a question and the new arm was not tuned on any answer.

**C.7-5. The recall probe (diagnostic, not a gate).** `oh-probe` (`run_longsession.py --recall-probe --reuse-run DIR`) asks no answer model and
calls no judge. For every answerable question with a gold anchor it records side by side (A) the old frozen shape + packing v2 at the old cap,
(B) the correct shape raw (server default budget), (C) the correct shape via `max_chars` at the old cap and at a sweep (800 / 1536 / 3000 /
6000 tokens), (D) BM25 over the full dropped context at the same caps, plus the anchor in the note and in the dropped context (the ceiling), the
server's `graph_branch.enabled` / `lightrag_mode`, the share of recalled chunks that belong to the session itself, and the keyword rate. The
probe's only decision rights: it may stop the paid run (a retrieval ceiling of zero is a finding, not a reason to spend). It selects no parameter
of the decision run: the decision cap stays the B = 1536 parity cap. The sweep exists because that cap is self-imposed and the owner's own
earlier evaluations did not cap memory that hard; a run at a larger cap would break the parity window and is a separate, labelled exploratory
contrast (Amendment D), never a replacement of the primary.

**C.7-6. Defaults and migration.** `run_longsession.py` default arms: `baseline,nc-enriched,nc-note-only` (`orig` with real compactions);
`nc-chunks` / `nc-graph` remain available with `--arms`. `home_run.py oh` runs the new arm set (fresh campaign: ingest happens in the run;
parity check on the enriched arm). The legacy 4-arm regime still analyses (rows without `nc-enriched`), flagged exploratory by C. The
section-5 gate table is unchanged and now reads `treatment = nc-enriched`.

### 4a.8 PROTOCOL AMENDMENT C.8 (2026-10-28, written AFTER the C.6 result and the C.7 design, BEFORE any project-layout record exists)

**C.8-1. The finding (verified in the server code, not in a result).** In `HARS_MEMORY_SESSION_MODE=split` (the server default and the mode of
every run so far) `memory_recall(session_id=...)` runs the real pipeline (keywords, hybrid dense + BM25 fusion, LightRAG graph branch,
`max_chars` budgeting) only over the MAIN index of the project (`default`, the owner's global KB). The session's own data reaches the answer only
through `_compute_session_channel` (`hars_memory/mcp_server.py`): `max(cosine, lexical F1)` of the QUESTION TEXT alone against the session's
chunks, appended after the fused chunks. It is a side channel, not the memory; ll/hl keywords, the graph branch and the fusion never see the
session. With keywords (C.7) the same call additionally pulled unrelated `default` KB chunks into the result. Consequently the C.6 arms
`nc-chunks` / `nc-graph`, and every C.7 `nc-enriched` / probe run on the split layout (the layout of `oh-probe` / `oh-enrich` as shipped with C.7),
measured that side channel, NOT the owner's memory. They are DIAGNOSTIC ONLY (`--hars-layout split-session`, `--layout split-session`); their
records stay in the artifacts, flagged, and never enter an inference about memory.

**C.8-2. The one memory configuration: the project layout.** "Global versus session" is expressed with what the server already has: the dropped
context of each eval session is written into a NEIGHBOUR project of the same server and read back with the very call used for the global memory.
No new service, tool, server mode or env switch.
1. One isolated project per eval session, `nc-<hars run id>-<sid>` (sanitised, <= 63 chars, never `default`), provisioned by writing
   `<projects dir>/<pid>/project.json` (`home_run.py oh-provision`); the owner restarts the existing server ONCE (the registry reads its projects
   dir at construction only; there is no MCP/HTTP reload). Each project has its own `index_dir`, BM25 index, graph and Qdrant collection
   `<base>_<project id>`.
2. Write: `memory_upsert_document(project=<pid>, file_path=<virtual absolute path>, content=<document>)` - the global memory's document tool, full
   LightRAG extraction into the project's main index. The documents carry the same sections as the split layout's compaction documents (the rolling
   note as `## Summary` in the first call of each boundary, the dropped span as `## Dropped context facts`); piece splitting (`--piece-chars`,
   `--pieces-per-call`) and boundaries are unchanged. `memory_compact` is not used. The write is idempotent per (project, boundary, call).
3. Read: `memory_recall(project=<pid>, question=<query>, ll_keywords, hl_keywords, mode="hybrid", max_chars)`, the server's default `top_k`, NO
   `session_id`, no `session_graph` - identical to how global memory is read, and the C.7-2 shape otherwise unchanged (K queries from the note and the
   kept tail only, dedupe, rank interleave, cap = `recall_cap`, block computed once per boundary, validity rules). Per query `graph_branch.enabled`,
   `lightrag_mode` and the keywords are recorded.
4. Gates (before any spend, not skippable by `--no-preflight`): the project is registered on the server, is not `default`, and a fresh ingest finds
   it EMPTY (a resumed run is exempt); after ingest the project's index holds entities and vectors (`memory_status(project)`). The eval refuses to
   write into `default`.
5. Known limit, stated: the block is built from the `Document Chunks` section of each recall (C.7-2 step 3); entity / relation sections of the
   full pipeline's context are not packed. The graph branch still shapes which chunks come back.

**C.8-3. What is reused, and why it is valid; what is not.** Reused from the finished C.6 campaign exactly as in C.7-4: the `baseline` and
`nc-note-only` rows, their judge verdicts (same judge model, refused otherwise), and the exact rolling notes of every boundary (`memory/<sid>.json`
of the old run, sha-checked). These were produced by code that C.8 does not touch (conversion, boundaries, summariser prompts and trimmer, ask prompt,
native token counting, judge and blinding); `baseline` and `nc-note-only` never read HARS. NOT reusable: the HARS data (a different layout and
different projects). The dropped spans are therefore RE-INGESTED into the new projects (`oh-ingest`), with the same pieces, the same boundaries and
the old notes, so the only thing that differs between `nc-enriched` and `nc-note-only` is the block recalled from the session's own memory.
Residual risks as in C.7-4 (a)-(d), plus: the new ingest is a new realisation of the extractor (not deterministic), persisted in the projects.

**C.8-4. Version and guard.** EVAL-VERSION moves to `2026-10-28.c8`. The paid-step guard keeps refusing every campaign made under another version,
with ONE exception, for `oh-provision` / `oh-ingest` / `oh-probe` / `oh-enrich` only: a campaign stamped `2026-10-27.c6` or `2026-10-28.c7`
(`REUSE_COMPATIBLE_FROM`). No hypothesis, threshold, budget, formula or judge of C.7-3 changes; `oh-ingest` spends only the flagged estimate of the
server-side extraction, `oh-probe` cents, and the probe must precede `oh-enrich` as before.

### 4a.9 PROTOCOL AMENDMENT C.9 (2026-10-28, written BEFORE any agentic record exists; the PRIMARY question and design)

Provenance: the owner's own question, 2026-10-28: *does our memory give a gain when used locally for a session - does the agent spend
MORE tokens finding the needed information WITHOUT memory than when it fetches facts from memory?* Neither arm ever sees the previous
session's transcript. The C.7 / C.8 fixed-budget single-shot arms (`nc-enriched` vs `baseline` vs `nc-note-only`, one call with memory in
the prompt) measure the fidelity of a memory block at a fixed budget; they do not measure what the owner asks (effort to find a fact), so
they become SECONDARY. Nothing below changes a C.6 record, the judge model, the summariser, the ingest or the C.8 projects; it reuses them.

**C.9-1. Design.** For each question of the finished OH campaign (14 SWE-bench sessions, 140 questions: answerable and unanswerable,
end-of-session checkpoint) an agent works in a tool loop until it answers (`run_agentic.py`):

- `sum`: the full baseline summary (the C.6 `baseline` rows' `memory_text`, byte for byte, sha256 recorded) + workspace tools;
- `sum+mem`: the SAME summary + workspace tools + `memory_recall`. The agent writes query, `ll_keywords`, `hl_keywords` itself; the call goes to
  HARS with `project = nc-<run>-<sid>` (the C.8 project), mode hybrid, the server-default `top_k`, `max_chars` = 6000 (fixed), no `session_id`
  (the `recall_args_correct` shape minus `session_id`); the tool returns HARS's `context` as-is, graph sections included;
- `note+mem` (OPTIONAL, `--arms`, off by default): the short C.6 note (`nc-note-only` `memory_text`) + workspace tools + `memory_recall`. Run it only if
  `sum+mem` shows a gain over `sum`.
Workspace tools, restricted to the session's workspace dir (realpath-checked, symlinks never followed out): `ls`, `read_file(path, start, end)`
(<= 250 lines and <= 6000 chars per call), `grep(pattern, path)` (ripgrep, <= 15 matches per file, <= 120 lines, <= 6000 chars), and the final `answer(text)`
(the agent may say it does not know). The summary sits in the system prompt, so the static prefix (tools + system + summary) is prompt-cached; a
moving cache breakpoint at the end of the history caches the earlier steps. Cache reads are priced separately and reported separately. Settings:
agent model = the profile's agent (haiku-5-5), at most 12 steps (the LAST step forces the `answer` tool, `tool_choice`), per-question token cap
400 000 (a runaway guard: a question that hits it is `no_answer`), `temperature 0` where the profile allows it (Anthropic 5.x: omitted, as everywhere),
the tool-use call is vendor-neutral (`providers.chat_tools`: Anthropic `/v1/messages` and OpenAI-compatible `tools`).
Questions of a session run sequentially, arms alternate their order by question parity, so cache warmth is not an arm effect.

**C.9-2. The workspace, and what it cannot be (stated).** The OH import keeps `instance_id` and `repo` of each session (`sessions/<id>.json`) but
NEITHER `base_commit` NOR the model's final patch (verified in `import_openhands.py` / `select_openhands.py`). `home_run.py oh-workspaces` therefore builds the
repository at `base_commit` (from `--instances FILE`, or at home from the public SWE-bench Verified dataset, `base_commit` only) and applies the session's
FINAL PATCH only when `--instances` supplies `model_patch`; the dataset's gold patch is never used. Without a patch (the default with the OH data as
shipped) the workspace is `base_commit` only and `oh-workspaces` says so. Consequence, a real limit: the session's own edits are not in the workspace, so
a question about what the session changed is answerable from the summary or from memory only, which favours `sum+mem` on exactly those questions. `.git`
is removed (the upstream fix lives in later commits). The transcript is never in the workspace: a leak scan (80-char heads of up to 60 transcript
messages, searched in every workspace file) runs before every session and stops the run on a hit; the workspace may not overlap the campaign or the
session data; a test plants a leak to prove the scan fires.

**C.9-3. Recorded per question per arm** (`records.jsonl`; full traces in `traces.jsonl`): every step's uncached-input / cache-read / cache-write /
output tokens (the API's native counts), `tokens_total` = all input incl. cache reads + output (the PRIMARY token measure), `tokens_fresh` = everything
except cache reads, USD, number and type of tool calls, the memory calls (query, keywords, returned chars, `graph_branch`), the final answer, steps,
wall-clock, and whether the gold anchor appeared in a tool result, separately for workspace and memory results.

**C.9-4. Outcome taxonomy and hallucination (methodology review of the C.6 bundle).** In the C.6 judged rows `abstained`, `hallucination` and the pass
score overlap (baseline answerable: 17 rows abstained AND passed, 12 abstained AND hallucinated). The agentic metrics use ONE mutually exclusive outcome per
answer, decided by the judge with an explicit precedence (a new field `outcome` in the judge schema, `judge_v2.py --outcome`; the default prompt and every
older judge file are unchanged): 1. `correct` - the answer states the recorded fact (a hedge does not change this); 2. `partial` - part of the fact, no false
specific; 3. `wrong` - asserts a false, contradicted, fabricated or different specific (= hallucination), even with a hedge; 4. `abstained` - declines and asserts
nothing relevant. Items with no recorded fact: a clear refusal is `abstained` (the right behaviour), an asserted specific is `wrong`. A hedged answer that states
the gold fact is correct or partial, never abstained. A run that produced no answer (token cap) is `no_answer`. If a judged row lacks `outcome` the metrics derive it with the same precedence from
the legacy fields and mark it `derived` (the count is printed). Hallucination is reported BOTH over all questions and conditional on answered
(wrong / (correct + partial + wrong)); it is a GUARD, not an endpoint: the decision does not reward silence. The mechanical rule `unsupported_no_anchor`
(judge_v2 post-pass) is reported separately and never counted as hallucination in the guard.

**C.9-5. One anchor matcher; subsets.** `anchors.anchor_found` (token-bounded, case-insensitive, identical to `judge_v2.anchor_re`, pinned by a test) is the only
anchor matcher of the track: tool results (workspace vs memory), the summary, the dropped context, the answer. (The C.6/C.7 funnel used different matchers over
different corpora: BM25 found the anchor in 61% of rows where `anchor_in_dropped_context` is 50%; nothing here uses that code.) Everything is reported on
(a) all questions and (b) the MEMORY-RELEVANT subset = anchor present in the dropped context (messages before the last boundary) and NOT in the summary.
**The primary analysis is (b).** Unanswerable probes are trivial (100% refusal in every arm so far): kept as a guard only; harder near-miss unanswerables (a plausible
but never-recorded detail close to a recorded one) are future work.

**C.9-6. Endpoints and statistics.** PRIMARY contrasts, `sum+mem - sum`, paired by question, on subset (b): (1) total tokens per question; (2) accuracy
(outcome `correct` over answerable questions). Secondary: tokens on the subset where BOTH arms are correct, fresh tokens, USD per question, USD and tokens per correct
answer, steps, tool calls, accuracy counting partial, the same for subset (a), and `note+mem` contrasts if that arm ran. Inference: cluster bootstrap over SESSIONS
(`metrics_v2.resample_indices`, 10 000 resamples, seed 1337), Wilcoxon over per-session means as a cross-check. **No WIN / FAIL threshold is applied**: n = 14 sessions
cannot reach the C-family thresholds by design, so the verdict is an estimate with its 95% CI, read as 'not distinguishable from zero at this n' when the CI contains zero.
**Power.** The primary endpoint is continuous and paired. The minimal detectable effect (80% power, alpha .05) is MDE = 2.8 x SD / sqrt(n_sessions), with SD the SD of the
per-session mean paired differences. Before the run, `oh-agentic --dry-run` prints a guess assuming SD = 50% of the mean tokens per question (about 37% of the mean at
n = 14: a guess, labelled as such); after the run `metrics_agentic.py` prints the MDE from the OBSERVED paired SD (and the optimistic question-level bound).

**C.9-7. Cost and time** (offline model, `oh-agentic --dry-run`; stated assumptions: 8 steps per question, 150 output tokens per step, 1200 tokens per workspace
result, 600 tokens of tool schemas, 330 of system prompt, a 1500-token summary; step k re-sends the whole history, cache reads cover the history of step k-1):
14 sessions x 10 questions x 2 arms = 280 agent runs, about USD 0.65 with cache / 1.86 without, upper bound (every question runs 12 steps, no cache) about USD 2.8; judge
280 items about USD 1.2; about 61 000 tokens per question including cache reads. These are estimates; the first live run replaces them.

**C.9-8. Guard, validity, limits.** EVAL-VERSION `2026-10-28.c9`. `oh-workspaces` / `oh-agentic` accept a campaign stamped `2026-10-27.c6`, `2026-10-28.c7` or
`2026-10-28.c8` (`REUSE_COMPATIBLE_FROM`; the campaign's stamp is not rewritten, the reuse is recorded under `c7_reuse`), nothing older. `oh-agentic` requires `oh-ingest`
done (C.8 project layout), the workspaces, the same judge model as the C.6 campaign (else refused), the caps and the per-HARS lock. The judge is blind to the arm (the tool
name `memory_recall` in an answer is masked like the other arm strings). Limits, stated: one realisation of a stochastic agent per (question, arm) (haiku at provider default
sampling); the retrieval is not deterministic across calls; the agent chooses its own keywords, so the result measures memory + the agent's use of it; without the final patch
the workspace lacks the session's edits (C.9-2); wall-clock per step is not controlled; the memory is the C.8 ingest as realised once.

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
