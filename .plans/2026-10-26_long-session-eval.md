# Long-session eval (1M+ tokens, 5-100 compactions) for the nc-eval

Status: plan written before implementation, 2026-10-26. Code + OFFLINE tests only; the OWNER runs it at
home (own models, real embedder, own HARS). Nothing is run for real in this task (optional smoke < $0.05).

Inputs read: CLAUDE.md (run-artifact rules), `experiments/neurocompact-eval/HANDOFF.md`,
`.plans/2026-10-26_session-graph-recall.md`, `scripts/run_chainreplay.py` (another agent edits it: IMPORT its
pieces lazily, never edit/copy), `scripts/import_opencode.py`, `scripts/run_pool.py`, `docs/mcp-reference.md`
(`session_graph`, commit 1df090d).

## Why

The chained eval (3 compactions, ~30k-token spans) cannot say whether graph memory pays off when a session is
10-300x larger than any summary: baseline = rolling summary re-summarised N times, nc = HARS fed the whole
dropped span at every boundary. Public data with 1M-10M token conversations exists (BEAM), and the owner's own
Claude Code transcripts carry real compactions + real summaries (a free "orig" arm).

## Deliverables (all under `experiments/neurocompact-eval/`)

| file | what |
|---|---|
| `scripts/longsession.py` | common format (docstring = spec), IO, validation, tokens, boundary computation, provider-agnostic LLM client (openrouter / anthropic / mock), lazy shim to `run_chainreplay` |
| `scripts/ls_redact.py` | secret redaction (sk-, sk-ant-, sk-or-v1, ghp_, glpat-, AWS, PRIVATE KEY, password=/token=) with counts |
| `scripts/import_beam.py` | BEAM repo -> long sessions (+ questions with msg_idx evidence) |
| `scripts/import_claude_code.py` | Claude Code transcript JSONL -> long sessions (+ real compactions, redaction ON) |
| `scripts/gen_longsession_qa.py` | QA generator (single + multihop across windows, verbatim evidence checks) for sessions without questions |
| `scripts/run_longsession.py` | the runner: boundaries, arms baseline / nc-chunks / nc-graph / orig, budget sweep, checkpoints, preflight, dry-run, resume, artifacts |
| `scripts/tests/test_*.py` | offline tests, synthetic fixtures only |
| `README-longsession.md` | format + usage + cost formulas |

## Design decisions

1. **Format** `longsession/1`: `{sid, source, title, messages[{idx,role,text,ts?}], compactions[{at_idx,
   original_summary?}], questions[{qid,qtype,question,answer,evidence[{msg_idx,start?,end?}],rubric?}]}`;
   `.json` (one session or a list) or `.jsonl` (one per line). `at_idx` = index of the FIRST message that
   survives the compaction (messages[:at_idx] were summarised away).
2. **Boundaries** are arm-independent (otherwise arms see different histories): synthetic mode fires when
   `tokens(live messages) + reserve > W`, keeps the last <= T tokens (cut at message boundaries) and the
   dropped region of boundary j is `[at_{j-1}, at_j)`, so every message is summarised exactly once. A forced
   final boundary makes "ask at the end" = "ask just after a compaction". Real mode (`--use-real-compactions`)
   takes `at_idx` from the session; the post-boundary tail is the following messages up to T tokens.
3. **Arms**: `baseline` (rolling LLM summary trimmed to B, nothing in HARS), `nc-chunks` and `nc-graph`
   (rolling short note + `memory_compact` of the dropped span in ~8k-char pieces at message boundaries; ONE HARS
   session per (sid) shared by both arms, they differ only in `session_graph` on `memory_recall` - the Amendment C
   ablation), `orig` (the session's real summary, only at real boundaries). Note cap / recall cap are the frozen
   functions of B from `run_chainreplay` (`note_cap_formula`, `recall_cap_for`).
4. **Deviation (documented)**: across the `--mem-budgets` sweep the HARS session is shared (the stored note is the
   one of the largest B); `--hars-per-budget` gives the strict per-B sessions at B-times the extraction cost/time.
5. **Validity** is per (sid, arm, B): recall error / empty / degraded (lexical_only) recall / graph block missing /
   persistence failure -> `valid=false` (`--allow-degraded` records degraded only). Rows are buffered per arm and
   written once with the final flag; `done.jsonl` marks finished (sid, arm, B); memory artifacts are cached per
   boundary for resume.
6. **Concurrency**: `--session-concurrency` (default 1), compact semaphore (default 1), recall semaphore
   (default 2) - the host has an OOM history.
7. **Artifacts** per CLAUDE.md: `/mnt/datasets/ft/artifacts/<UTC>-longsession-<name>/` with samples.jsonl,
   summary.json, run.log, inputs/ (sha256 + canonical path), scripts/ (+ cmdline), MANIFEST.json.
8. **Cost** is estimated offline by formulas from token counts only (printed per arm by `--dry-run`).
9. **Redaction** runs in the Claude Code importer by default (private transcripts); counts are stored in the
   session meta and printed.

## Tests (offline)

format/boundaries, redaction, both importers (synthetic fixtures built in the test, no BEAM text, no transcript
content), QA validators, dry-run determinism, record schema (compat keys of `export_judge_input` /
`metrics_v2`), resume skip, end-to-end with a fake HARS + mock LLM.

## Out of scope / open

Judging (reuse `judge_v2.py` on exported records), BEAM-native rubric judge, tokenizer exactness (tiktoken only if
installed), 10M-token HARS wall time (hours; the runner resumes).
