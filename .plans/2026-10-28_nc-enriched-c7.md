# Plan: Amendment C.7 - nc-enriched (compaction-time recall), reuse of a finished OH campaign, recall probe

Scope: experiments/neurocompact-eval only (no change to hars_memory/).

1. run_longsession.py: arm `nc-enriched` (queries + ll/hl keywords from note + kept recent context only; memory_recall hybrid, server top_k,
   budget via max_chars; dedupe; fill to B - note; no ask-time recall), `--reuse-run` (HARS sessions, notes, copied arms, gate),
   `--recall-probe`, raw-recall sidecar + anchor_in_recall_raw for all recall arms, default arms baseline,nc-enriched,nc-note-only.
2. metrics_v2 / export / judge / funnel: accept nc-enriched; primary nc-enriched vs baseline, key secondary vs nc-note-only; hallucination
   breakdown answerable vs unanswerable.
3. home_run.py: `oh-probe`, `oh-enrich` on an existing campaign; EVAL-VERSION 2026-10-28.c7 with a narrow reuse exemption for 2026-10-27.c6.
4. REPORT-PROTOCOL.md 4a.7 (C.7), HOME-RUNBOOK.md 6.0b.
5. Tests offline (FakeHars, mock LLM): tests/test_c7_enriched.py, tests/test_c7_home.py + adjustments of the arm-set expectations.
