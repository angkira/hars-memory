# nc-eval harness fixes (from Opus review of 9091373)

Scope: experiments/neurocompact-eval/ only. No dataset rebuild, no experiment runs (services/data not available here).

## Work packages
- WP-A runner (scripts/run_chainreplay.py)
  - B1 recall gate: invalid if raw recall empty / units_kept==0 / payload not ok / session error; preflight recall call; recall_error invalidates the chain.
  - B2 writer: buffer a chain's rows, write once at chain end with final `valid`; no orphan rows.
  - B5 abstention sentence identical in all arms' prompts.
  - M1 recall cap = clamp(mem_budget - realized note), as documented; baseline realized only reported.
  - M4 persistence gate also asserts entities>0 and vectors_indexed>0.
  - emit fields metrics_v2 expects (memory_tokens, persisted_compaction, recall_seconds in latency or separate field).
  - minor: open outputs "w" (or refuse non-empty --out), AGENT_DIR path, missing run_pool import documented/guarded.
- WP-B metrics (scripts/metrics_v2.py, export_judge_input.py, judge_v2.py)
  - B2 drop rows with valid != True; require complete arm triplets; hard-fail on duplicate (qid, arm).
  - B3 field names, ratio pair nc/baseline, unanswerable WIN rule, delta-anchor>=0, clean_anchor_hit for gates, missing -> INCONCLUSIVE, wire --boot/--seed, cluster by session_id not chain.
  - M8 blindness regex covers `-nc-` and "recalled memory" style leaks.
- WP-C docs: sync RUNBOOK/PROTOCOL/HANDOFF to code; log B4, M2, M3, M7, M9 as open protocol issues (need owner decision / data).

## Not done here (needs owner)
B4 dataset overlap, M2 graph vs chunk recall, M3 12k+12k payload, M5 530 diagnosis, M7 tuning-chain exclusion.
