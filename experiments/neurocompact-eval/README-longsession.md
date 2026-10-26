# Long-session eval (1M+ tokens, 5-100 compactions)

Code + offline tests only; the owner runs it on their machine (own models, own embedder, own HARS).
Plan: `.plans/2026-10-26_long-session-eval.md`. All scripts live in `scripts/`, tests in `scripts/tests/`
(`cd experiments/neurocompact-eval/scripts && python -m pytest tests`).

## Pipeline

```
BEAM repo  --import_beam.py-------------------\
Claude Code *.jsonl --import_claude_code.py ---+--> longsession/1 sessions --gen_longsession_qa.py (if no questions)
                                               \--> run_longsession.py --dry-run | live | --resume
```

| script | what it does |
|---|---|
| `longsession.py` | the format (docstring = spec), IO/validation, token counting, arm-independent boundary computation, provider-agnostic LLM client (openrouter / anthropic / mock), lazy shim to `run_chainreplay.py` |
| `import_beam.py` | BEAM `chats/<size>/<n>/chat.json` + `probing_questions.json` -> sessions. json only (never pickles), run with `python3 -I`. Keeps information_extraction (-> `single`), knowledge_update, temporal_reasoning, multi_session_reasoning (-> `multihop`), contradiction_resolution; other types dropped and counted (`--types all` keeps them). Evidence = `source_chat_ids` mapped to message indices |
| `import_claude_code.py` | Claude Code transcripts -> sessions: user/assistant text, tool_use+tool_result merged into one `tool` message (import_opencode format), `compact_boundary` + compact-summary user line -> `compactions[{at_idx, original_summary}]`, sidechains excluded (flag to include), parentUuid ordering with dead-branch and parallel-tool-result handling, resumed files merged, **secret redaction on by default** with a count report |
| `ls_redact.py` | the redaction pass (sk-, sk-ant-, sk-or-v1, ghp_/github_pat_, glpat-, AWS keys, PEM private keys, JWT, bearer; `NAME=value` assignments only when NAME has a secret-like word segment (token/secret/password/api_key/auth/bearer/credential...) AND the value looks like a secret: >= 12 chars, mixed classes, entropy; `tokenizer=tiktoken:cl100k_base`, model ids, paths, URLs stay). Per-pattern counts are reported; `home_run bundle` uses the same module |
| `gen_longsession_qa.py` | QA for sessions without questions. Default `--mode menu`: per compaction window the code pre-extracts window-unique anchors with verbatim >= 8-word evidence, the model only phrases questions (~2 single per window up to 40, ~10 multihop via a code-verified shared entity, a few unanswerable probes verified absent); worst-case budget pre-check before every call. `--mode legacy`: free-text excerpts + post-hoc gates. Full prompt/answer log in `qa-raw.jsonl` |
| `run_longsession.py` | the runner (below) |
| `export_longsession_judge.py` | run_longsession records + the session files -> `judge_v2` input: blind, one +-500-char evidence window per hop, BEAM `rubric` reaches the judge, `visible_context` (memory + recall + verbatim tail, question/answer excluded), memory channel fields dropped, `metrics_v2.validate_rows` filter |
| `home_run.py` | the owner's single entry point (HOME-RUNBOOK.md): doctor / dry-run / smoke / oh / beam / cc / bundle |

## Format `longsession/1`

```jsonc
{ "format": "longsession/1", "sid": "beam-1M-3", "source": "beam|claude_code|...", "title": "...",
  "meta": {},
  "messages":    [ {"idx": 0, "role": "user|assistant|tool|system", "text": "...", "ts": "optional"} ],
  "compactions": [ {"at_idx": 412, "original_summary": "text|null", "auto": true, "pre_tokens": 167000} ],
  "questions":   [ {"qid": "...", "qtype": "single|multihop|knowledge_update|...", "question": "...",
                    "answer": "...", "answerable": true,
                    "evidence": [ {"msg_idx": 50}, {"msg_idx": 52, "start": 10, "end": 90} ],
                    "gold_anchor": "optional", "rubric": ["..."], "source_qtype": "optional"} ] }
```

`.json` (one session or a list) or `.jsonl` (one per line); directories are read recursively. `at_idx` = first
message that survives the compaction. `compactions` holds only REAL compactions (BEAM has none).

## Runner semantics (`run_longsession.py`)

* **Boundaries are arm-independent.** Synthetic: after message *i*, if `tokens(live) + reserve > W` the context
  is compacted: the last <= `--keep-tail` T tokens (whole messages) stay as the tail, the rest since the previous
  boundary is dropped, so each message is summarised exactly once. `reserve` is a constant (2048), not tied to
  the budget sweep. A forced final boundary makes "ask at the end" = "ask right after a compaction".
  `--use-real-compactions`: boundaries from the session; no forced final boundary.
* **Arms** (per session, per budget B in `--mem-budgets`): `baseline` (rolling LLM summary trimmed to B, no HARS),
  `nc-chunks` and `nc-graph` (rolling short note of `note_cap(B)` tokens + the whole dropped span to HARS via
  `memory_compact` in ~8k-char pieces cut at message boundaries; ask = note + packed `memory_recall`, recall cap =
  `clamp(B - realized note, floor(B), ceil(B))`; `nc-graph` = same HARS session, `session_graph: true`),
  `nc-note-only` (same note, no recall), `orig` (the session's real summary; real boundaries only). Caps scale
  with B exactly like `run_chainreplay` (640/256/1024 at B = 1536). One HARS session per sid is shared across the
  budget sweep (stored note = largest B's); `--hars-per-budget` gives strict per-B sessions.
* **Questions** are asked at `--checkpoints` (`end`, `all`, `every:N`, or a list of boundary numbers). A question
  is eligible once its evidence has happened; `lag` = boundaries between the oldest evidence and the ask (0 = the
  evidence is still in the tail). Mid-checkpoint rows get `qid = <qid>@c<j>` (so `metrics_v2` sees unique
  `(qid, arm, B)`), `base_qid` keeps the original. `--lags`, `--qtypes`, `--max-questions` thin them out.
* **Validity** per (sid, arm, B): summariser/ingest failure, empty/errored/degraded (`lexical_only`) recall
  (`--allow-degraded` only records it), nc-graph without a `session_graph` block, HARS status gate, unrepaired
  `stop=length` on > 1/6 of summaries, note alone busting B -> `valid=false` with `chain_invalid_reasons`.
* **Preflight**: HARS `/healthz` + one `memory_recall`, API-key presence; nothing is spent when it fails.
* **Resume**: `--resume <artifact dir>` skips done `(sid, arm, B)` (`done.jsonl`), reuses per-boundary summaries
  (`memory/`), HARS ingest markers and already-answered rows (`partial/`); a changed config is refused
  (`--force-config`). Concurrency: `--session-concurrency` (1), `--compact-concurrency` (1),
  `--recall-concurrency` (2), `--ask-concurrency` (2) - the host has an OOM history. `--max-cost-usd` hard-stops
  LLM spend (HARS extraction is billed by your HARS process).
* **Records** (`records.jsonl` == `samples.jsonl` rows) carry the `run_chainreplay` / `metrics_v2` /
  `export_judge_input` keys: `rid`, `qid`, `chain_id` (= sid), `arm`, `mem_budget`, `b_tag`, `qtype`, `valid`,
  `memory_tokens`, `arm_note_tokens`, `memory_tokens_injected`, `anchor_hit`, `anchor_in_recall`, recall stats,
  `latency_s`, `prompt_tokens`, `completion_tokens`, `cost_usd`, plus `checkpoint`, `lag`, `tail_tokens`,
  `memory_text`, `injected`, `rubric`, `rubric_hit_frac` (a cheap proxy, NOT a judge). `metrics_v2.validate_rows`
  accepts them with `arms=(baseline, nc-chunks, nc-graph, nc-note-only)`. `export_judge_input.py` is chain-shaped; the long-session twin is `export_longsession_judge.py` (below).

### Artifacts (CLAUDE.md "never lose a run")

`<out>` defaults to `/mnt/datasets/ft/artifacts/<UTC>-longsession-<name>/`: `samples.jsonl`, `records.jsonl`,
`compactions.jsonl` (per boundary per arm: usage, cost, latency, clipping, HARS ingest/status), `summary.json`,
`run.log`, `inputs/inputs.json` (sha256 + canonical path; `--copy-inputs` copies), `scripts/` (+ `cmdline.txt`,
the exact `run_chainreplay.py` used), `recall/` (raw HARS payloads), `memory/` (raw + trimmed summaries, prompt
sha256), `MANIFEST.json` (machine, models, params, start/end, cost, git head, `run_chainreplay` sha).

## Cost estimate (`--dry-run`, offline, deterministic)

Tokens from the session (`tiktoken` if installed, else chars/3.6 - printed). Prices come from ONE place, the
`providers.py` profile tables (`profiles/<name>.json`, each entry with status/source; tier-aware: Haiku 5.5 =
$0.10/$0.50 per MTok, $0.50/$2.50 above a 100k prompt; Sonnet 5.5 = $2/$10); `--price-in/--price-out` are explicit
overrides. A live run needs a hard `--budget-usd` (alias `--max-cost-usd`); the HARS-side extraction is charged to the
same cap as a flagged ESTIMATE per ingested piece. With J boundaries, dropped tokens D_j, FILL = 0.85:

```
baseline_j  in = s_j*(P_b + S_(j-1)) + D_j     out = s_j * 0.85*B          s_j = ceil(D_j / (sum_max_input - P_b - B))
note_j      in = s_j*(P_n + N_(j-1)) + D_j     out = s_j * 0.85*note_cap(B)
summariser $ = sum_j (in*p_in + out*p_out)/1e6          (P_b, P_n = tokens of the empty summariser prompt)
HARS $       = chunks * (2700*e_in + 600*e_out)/1e6      chunks = sum_j ceil(D_j/1200)   [2700/600 per chunk: UNVERIFIED]
ask $        = sum_asks (P_ask + tail_j + q + memory)*p_in + 100*p_out        memory ~ B (orig: its size)
nc-graph +   = n_recalls * (450*e_in + 80*e_out)         (keyword LLM call, run_chainreplay.KEYWORD_*)
total        = baseline + (note + HARS once + asks of nc-chunks/graph/note-only) + orig asks     [x budgets for asks]
```

Examples (this repo's data, `--dry-run`, W = 150k, T = 20k, B = 1536, all at Haiku 5.5 0.10/0.50 for the agent AND
the HARS extractor, chars/3.6 tokens, end checkpoint, 10 questions):

| session | tokens | boundaries | baseline arm | nc arms (note + asks) | HARS extraction | total (4 arms) |
|---|---|---|---|---|---|---|
| BEAM 1M (`beam-1M-3`) | 1.01M | 8 | $0.13 | $0.13 + 0.02 + 0.02 | $0.47 (~1.4 h ingest at 11 s/piece) | $0.78 |
| BEAM 10M (`beam-10M-1`) | 13.8M | 107 | $1.57 | $1.48 + 0.02 + 0.02 | $6.58 (~19 h at 11 s/piece) | $9.67 |

Two things dominate wall-clock and are guesses: HARS ingest time per piece (`--extract-s-per-piece`, default 11 s: measured 10-12 s/piece in a Haiku smoke with 2.5k-char pieces; 8k-char pieces are slower) and the
extractor tokens per chunk (`--extract-in-per-chunk/--extract-out-per-chunk`). Calibrate them on the first real
session (`compactions.jsonl` has the per-boundary ingest seconds) and re-run `--dry-run` for the rest.

## Typical commands

```bash
# BEAM 1M (35 chats), end checkpoint; --profile selects the vendor (anthropic | openrouter | gemini | custom)
python3 -I scripts/import_beam.py --repo /path/to/beam-repo --size 1M --out data/beam1M
python3 scripts/run_longsession.py --sessions data/beam1M --dry-run --profile anthropic --mem-budgets 1536,1024,512
ANTHROPIC_API_KEY=... python3 scripts/run_longsession.py --sessions data/beam1M --name beam1M --profile anthropic \
    --mem-budgets 1536 --mcp http://127.0.0.1:8307/mcp --budget-usd 40
python3 scripts/export_longsession_judge.py --records <run>/records.jsonl --sessions data/beam1M --out <run>/judge-input.jsonl
python3 scripts/judge_v2.py --in <run>/judge-input.jsonl --profile anthropic --allow-same-family-judge --budget-usd 8 \
    --out <run>/judged.jsonl --mapping <run>/judge-mapping.json
python3 scripts/metrics_v2.py --records <run>/records.jsonl --judge <run>/judged.jsonl --mapping <run>/judge-mapping.json \
    --judge-input <run>/judge-input.jsonl --track longsession --include-sid '^beam-1M-' --end-only --json <run>/metrics.json

# your own Claude Code sessions (redacted), QA generated, real compactions, orig arm, lag analysis (descriptive)
python3 -I scripts/import_claude_code.py ~/.claude/projects/<proj> --out data/cc --min-tokens 300000
ANTHROPIC_API_KEY=... python3 scripts/gen_longsession_qa.py --sessions data/cc --out data/cc-qa --profile anthropic \
    --use-real-compactions --n-single 40 --n-multihop 10 --budget-usd 3   # menu mode; --dry-run first
python3 scripts/run_longsession.py --sessions data/cc-qa --use-real-compactions --checkpoints every:3 --dry-run
```

`home_run.py` chains all of this with caps, resume and a bundle (HOME-RUNBOOK.md).

Limits: BEAM ships 2 questions per type per chat (10 kept per session); BEAM answers are mostly sentences, so
`gold_anchor` exists for few of them - score with the rubric/judge. Notes: the summariser prompts are a
hand-synced Python port of `pi-agent/extensions/neurocompact.ts` (`PROMPT_VERSION` in the cache key).
