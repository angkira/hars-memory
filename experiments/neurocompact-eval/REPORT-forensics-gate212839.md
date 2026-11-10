# Forensic report — runs/gate-212839 (diagnostic, not a gate verdict)

Judge: openai/gpt-6-luna, 42/42 parsed, **$0.0064 spend** (cap $0.20). Blind: mapping in
judge-mapping.json; 1 prompt flagged for the token "baseline" — benign, it is inside the
candidate answer's own prose ("The deterministic baseline already passed E2E").

## Funnel (answerable items only, n=4 per lag per arm; anchor_hit from records.jsonl)

| arm | lag | n | anchor_in_recall | anchor_in_C3_summary | anchor_hit | LEAK |
|---|---|---|---|---|---|---|
| baseline | 1 | 4 | – | 1 | 1 | 0 |
| baseline | 2 | 4 | – | 0 | 0 | 0 |
| baseline | 3 | 4 | – | 1 | 2 | 0 |
| nc | 1 | 4 | 1 | 0 | 1 | 0 |
| nc | 2 | 4 | 1 | 0 | 1 | 0 |
| nc | 3 | 4 | 1 | 0 | 1 | 0 |
| nc-note-only | 1 | 4 | 0 | 0 | 0 | 0 |
| nc-note-only | 2 | 4 | 0 | 0 | 0 | 0 |
| nc-note-only | 3 | 4 | 0 | 0 | 1 | 0 |

LEAK check (gold anchor token-bounded, case-insensitive, in the question text): **0 violations
across all 42**. Funnel anchor_in_recall matches records.jsonl's own field on all answerable nc items.

Conditionals (answerable): nc P(hit | in_recall)=3/3, P(hit | NOT in_recall)=0/9,
P(halluc | in_recall)=0/3; baseline P(hit | anchor_in_C3_summary)=1/2, P(hit | NOT)=2/10,
P(halluc | in_summary)=0/2, P(halluc | NOT)=3/10.

## Drift, C1→C2→C3 (baseline arm-level session summaries)

chain_qq8Vqbow_c0: anchor survives all three compactions for _1_2 (hars.execution.monitor.feedback)
and appears only at C3 for _3_2 (1.2.1); 4 of 6 anchors are absent from every compaction.
chain_y062eGZR_c0: 0 of 6 anchors survive any compaction. No clean "survives C1, dies by C3"
case — the dominant failure is that compaction never captures the anchor at all (10/12 absent
from C3), with one visible C3 length-truncation (stop=length, 6497 chars).

Anchor support behind the 7 raw anchor_hits: nc 3/3 recall-backed; baseline 1/3
(1.2.1 in C3); baseline _1_1 (importlib.metadata) and _y062eGZR_1_2 (deepseek-r1-distill model id)
hit with the anchor absent from every summary and every post-compaction message — pure
question-primed reconstruction; nc-note-only's single hit (_1_1) is the same phenomenon.

## Judge metrics (42 records)

| arm | n | score mean | abstained | hallucinated | anchor_hit |
|---|---|---|---|---|---|
| baseline | 14 | 0.57 | 7 (6/12 answerable) | 4 (3/12 ans, 1/2 unans) | 3 |
| nc | 14 | 0.71 | 11 (9/12 ans, 2/2 unans) | 2 (2/12 ans, 0/2 unans) | 3 |
| nc-note-only | 14 | 0.36 | 8 (7/12 ans, 1/2 unans) | 8 (7/12 ans, 1/2 unans) | 1 |

Paired deltas: nc−baseline +0.143 (3W/9T/2L); nc-note-only−nc −0.357 (2W/8T/4L). n=2 chains —
wildly underpowered; diagnostic only. Unanswerables: nc 2/2 clean abstains (judge score 2.0);
baseline and nc-note-only 1/2 each hallucinated on the never-mentioned probes.

Post-pass caveat: the exported records carried no transcript field, so the transcript-verification
half of the hallucination post-pass ran in its degenerate mode (everything unsupported becomes
"unverifiable"). Offline re-verification of the 12 flagged answers against each model's actually
visible context (final compaction summary + post-compaction messages) flips none of them: all 12
assert a wrong specific (wrong topic string, wrong helper, wrong timestamp) or a false
meta-claim on top of a hedge. The hallucination labels stand; the rule name is just imprecise.

## Six answer readings

- baseline HIT _y062eGZR_c0_1_2 (lag 3): exact anchor `openrouter/deepseek/deepseek-r1-distill-llama-70b`
  produced with the string absent from the visible context — reconstructed from the question's own
  description, plus unverifiable garnish ("currently set for Quick/Parallel tasks in ~/.ninja-mcp.env").
  "…— the OpenRouter catalog entry for the DeepSeek-R1 distilled variant served on Llama-70B."
- baseline MISS _qq8Vqbow_c0_2_1 (lag 2): hedges ("I can't name a plan `.md` without guessing") but
  grounds the hedge in a false claim — the plan path WAS cited in the session. The "baseline never
  abstains" headline is wrong: 6/12 answerable baseline answers are judge-flagged abstain-flavored,
  and several still smuggle in a wrong specific.
- nc HIT _y062eGZR_c0_2_2 (lag 2): clean correct answer straight out of the injected recall block
  ("was **1786274314** (Unix epoch seconds, from `tmux list-sessions…`)"). The reader is fine.
- nc MISS _qq8Vqbow_c0_1_1 (lag 3): pure, honest abstention despite recall being present-but-packed:
  "I do not know — neither my working session context nor the recalled memory records the
  environment probe's module choice". The recall packing (81% char loss) dropped the anchor.
- nc-note-only MISS _qq8Vqbow_c0_1_2 (lag 3): confident wrong specific — asserts MONITOR =
  `hars.monitor` when the session recorded `hars.execution.monitor.feedback`, decorated with a
  fabricated consistency argument. The note arm's signature failure.
- nc-note-only HIT _qq8Vqbow_c0_1_1 (lag 3): not a memory hit — derives
  "since it's the standard-library way to query installed distribution metadata"; the note never
  contained the anchor.

## Verdict

The nc abstention wave is overwhelmingly (a) honest recall misses, with a hard floor of 3/12:
P(anchor in injected recall)=3/12 answerable, P(hit | in recall)=3/3, P(hit | recall absent)=0/9,
and the abstentions that follow recall misses are clean (both unanswerable probes perfectly
declined, median latency 4.9s vs baseline 7.3s — nc answers faster because abstaining is cheap).
It is not (b) reader failure — every time the packed recall actually carried the anchor, nc read
and used it — and not (c) a prompt/artifact problem — all 14 injected blocks were non-empty
(2.0–3.1 KB chars), correctly wrapped, and the question text leaks nothing (LEAK=0 everywhere).
The real lever is the packer: 77–88% of raw recall chars are dropped and 9/12 anchors don't
survive; raising recall_cap (or rank-diversifying the pack) is the highest-value fix. Baseline's
confident-miss pattern is partly real, partly metric artifact: 7/14 judged answers assert wrong
specifics or false meta-claims (4 hallucinations incl. one on an unanswerable probe — genuine
risk the judge supports with specific contradictions), but 2 of its 3 anchor_hits are
question-primed lucky reconstructions (anchors provably absent from context), so raw anchor_hit
flatters baseline; judge-based scoring shows baseline and nc tied at 0.50 mean on answerables,
with nc buying its parity at 934 injected tokens and zero unanswerable hallucinations.
nc-note-only is the worst arm on every axis — and note it was also starved (584–600 note tokens
vs baseline's 1183–1805 summary tokens, so the parity window was violated): 0/12 anchors in the
note, 7/12 answerable hallucination rate built from garbled note paraphrases. Its collapse is
confounded by the budget mismatch and cannot be read as a format verdict.

## Files created

- runs/gate-212839/judge-input.jsonl (42 records, export 42/42)
- runs/gate-212839/judged.jsonl + runs/gate-212839/judge-mapping.json
- runs/gate-212839/funnel.json (per-record booleans, drift table, conditionals, leak/mismatch checks)
- funnel_gate212839.py (deterministic funnel script, re-runnable, $0)
