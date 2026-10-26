# HOME-RUNBOOK - running the neurocompact eval on your own machine

Written 2026-10-26. You run everything here with your HARS, your real embedder and your Anthropic key
(later maybe OpenRouter / Gemini through a profile), and send ONE file back: the bundle (step 9).
The protocol is frozen in `REPORT-PROTOCOL.md` section 4a (Amendments C, C.1, **C.2**). Nothing in this
runbook changes a metric or a rule; the tooling is `scripts/home_run.py` (one entry point) on top of the
existing scripts. All paths below are relative to the repo root.

```
home_run.py doctor | hars-env | dry-run | smoke | oh | beam | cc | bundle
```

## 0. What you are going to spend and wait (read this first)

Verified list prices (profile `anthropic`, `profiles/anthropic.json`, status `confirmed`): Haiku 5.5
**$0.10 in / $0.50 out per MTok**, **$0.50 / $2.50 for a prompt above 100k tokens**; Sonnet 5.5 **$2 / $10**
(the judge and the Claude-Code QA generator). `dry-run` computes everything below from your actual data; these
are the numbers of today's data (tokenizer = the chars/3.6 estimator because tiktoken was not installed where
they were computed; with tiktoken they move by roughly +-15%).

| Track | what | run (replay/compaction/answers + flagged HARS estimate) | judge (Sonnet) | gen (QA) | wall-clock |
|---|---|---|---|---|---|
| `oh` B=1536 only | 14 chains x 10 items x 4 arms | **$3.64** | $1.90 (560 items) | - | ~7 h |
| `oh` sweep 1536,1024,512 (protocol) | same x 3 B | **$10.91** | $5.71 (1680 items) | - | ~21 h |
| `beam` B=1536 only | BEAM-1M, 35 chats, 45M tokens, 360 boundaries, 20k HARS pieces | **$34.29** (of which HARS extraction $21.07, an ESTIMATE) | $4.76 (1400 items) | - | ~68 h (61 h HARS ingest at the guessed 11 s/piece) |
| `beam` sweep 1536,1024,512 | same, asks x3 | **$59.88** | $14.28 | - | ~82 h |
| `cc` (your sessions) | per session with ~7 real boundaries, 28 generated questions, every:3 checkpoints | ~$1-1.5 / session | ~$0.5-1 / session | ~$2 / session (Sonnet, upper-ish bound) | ~2-3 h / session |

Formulas (also in `README-longsession.md`): judge = items x (1300 in x $2 + 80 out x $10)/1e6 = $0.0034 per
item; BEAM/CC run = baseline summaries + note summaries + `chunks x (2700 in, 600 out)` HARS extraction +
asks; OH run = `run_chainreplay --dry-run` projection (agent turns + extractor + keyword calls).
**Two numbers are guesses and dominate the BEAM wall-clock: HARS ingest seconds per piece (11 s) and extractor
tokens per chunk.** Calibrate on the first session (`<campaign>/beam/run/compactions.jsonl`, rows with
`"arm": "hars"` have `seconds`) and re-run `dry-run`. (Measured in the build container with a Haiku extractor: 6.4-7.1 s per 4000-char piece;
BEAM pieces are 8000 chars, so 11 s/piece is a plausible middle, but it is a guess.) If 68 h is too much, the BEAM run is **resumable**
(Ctrl-C, re-run the same command); if you cannot afford all 35 chats, decide the number N BEFORE running
(`--beam-limit N`, first N in file order), state it in your reply, and do not pick chats after seeing results.

## 1. Clean checkout

```bash
git clone <your hars-memory remote> && cd hars-memory && git checkout claude/youthful-thompson-yv6xqd
uv sync --extra dev                       # NOT `--group dev`; python >= 3.13; tiktoken is a core dependency
uv run python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); print('tiktoken ok')"   # warms the BPE cache
npm i -g @earendil-works/pi-coding-agent@1.4.0     # only the `oh` track needs pi; 1.4.x is pinned (event parser)
#   no global npm?  npm i --prefix ~/.local/pi @earendil-works/pi-coding-agent@1.4.0 && export PATH=~/.local/pi/node_modules/.bin:$PATH
pi --version && node --version             # 1.4.x and Node >= 20
cp experiments/neurocompact-eval/home_run.toml.example home_run.toml     # you fill it in below
```

The tokenizer is part of the decision set (every record carries it; `merge_runs` refuses a mix): run everything
on this one machine with tiktoken available.

## 2. Keys

```bash
export ANTHROPIC_API_KEY=...        # never paste it anywhere, never put it in home_run.toml
```
(Another variable name: `[general] key_env = "EVAL_API_KEY"` or `--key-env`.) The same key must be visible to the
HARS server process, see step 3. No key value is ever printed or logged by `home_run.py`.

## 3. HARS server with the profile env and the REAL embedder

```bash
S=experiments/neurocompact-eval/scripts
uv run python $S/home_run.py hars-env > ~/hars-env.sh     # extractor + keyword LLM = the profile, session-mode/engine settings
```
Edit `~/hars-env.sh`: set `<HARS_DATA_DIR>` (a fresh dir on a big disk, it grows) and **uncomment/set your real
embedder** (the repo default is `HARS_MEMORY_EMBED_MODEL=intfloat/e5-large-v2` + `HARS_MEMORY_EMBED_LOCAL_FILES_ONLY=1`,
see `config/.env.example`; any real embedder is fine but it is FROZEN for the decision set, C.1-3). Then:

```bash
source ~/hars-env.sh && uv run hars-longterm-memory-mcp-https      # serves http://127.0.0.1:8307/mcp
uv run python $S/home_run.py doctor --deep                          # in another shell
```
`doctor` must end with **0 FAIL**. It checks: HARS `/healthz`; the embedder id from `memory_status`; **a
behavioural paraphrase probe** (a tiny `memory_compact` + `memory_recall`; a hash/shim or broken embedder FAILS it -
`memory_status` alone cannot be trusted: it reports `HARS_MEMORY_EMBED_MODEL` even when `HARS_MEMORY_EMBED_PROVIDER=http`
serves a different model, see the findings in `HANDOFF.md`); the cents-level liveness ping of every profile model;
the server's extractor/keyword model vs the profile; keys; pi 1.4.x; node; free disk; tiktoken; your inputs.
Do not run two clients against the HARS engine at once (1 thread / queue 32 by design; the host has an OOM history).

## 4. Inputs

* **OpenHands chains**: copy the artifact dir (14 chains, single + `xspan_locator` + unanswerable) to the machine, e.g.
  `~/hars-data/20261026T144511Z-oh-multihop-qa/` (it holds `chains.jsonl`, `chains-manifest.json`, `verify-report.md`).
  Set `[oh] artifact_dir` (or `--oh-dir`). It is not in git. Public SWE-bench sessions, no secrets.
* **BEAM**: `git clone https://github.com/mohammadtavakoli78/BEAM ~/data/BEAM` (4 GB). Set `[beam] repo`.
  Only the JSON files are read (never pickles); `chats/1M/` has 35 chats.
* **Claude Code sessions**: point `[cc] paths` at project dirs/files under `~/.claude/projects/<proj>/*.jsonl`
  (subagent files are excluded by default). Only sessions that contain real `/compact` boundaries can run. Secret
  **redaction is ON and cannot be turned off through home_run**; read the redaction counts the import prints.
  Your private text lives in `<campaign>/cc/data*` and in `cc/run/*`: none of it leaves the machine except the
  excerpts inside `records.jsonl` / `judged.jsonl` (see step 9).

## 5. Caps (hard, yours to choose)

Edit `home_run.toml` `[caps]` (0 = not filled in = every paid step refuses): `gen` (QA generation), `run`
(replays, compactions, answers + the flagged HARS estimate), `judge`. Run `dry-run` (step 6) and set each cap a bit
above its estimate; per-track overrides: `[caps.oh]`, `[caps.beam]`, `[caps.cc]`. Each underlying script gets
`--budget-usd = cap - already spent` (ledger in the campaign dir) and stops at it ("BUDGET STOP"); the runs are
B-major, so a stop still leaves the primary B=1536 cell complete. A run whose estimate is above its cap is refused
unless `--accept-partial`.

## 6. The command sequence

All commands: `uv run python experiments/neurocompact-eval/scripts/home_run.py [--config home_run.toml] ...`
(abbreviated `HR` below). Everything lands in ONE campaign dir
`/mnt/datasets/ft/artifacts/<UTC>-homerun-home/` (change `artifact_root`; never `/tmp`, it is swept), logs and exact
command lines included. Re-running a subcommand skips finished steps and resumes unfinished runs.

```bash
HR dry-run                       # zero spend: per-track gen/run/judge estimate vs your caps, wall-clock
HR smoke                         # 1 cheapest OH chain + 1 tiny synthetic long session, ~$0.5-0.8 total (own campaign); must PASS
HR oh                            # OpenHands chains: batches of 4, B-major (1536 first), retries invalid cells once,
                                 #   merge -> export -> judge -> metrics_v2 -> funnel
HR beam                          # BEAM-1M: import -> run_longsession (end checkpoint, B sweep per [beam] budgets)
                                 #   -> export_longsession_judge -> judge (rubric) -> metrics_v2 --end-only --include-sid ^beam-1M-
HR cc                            # your sessions: import (redacted) -> QA generation -> run (real boundaries, orig arm)
                                 #   -> judge -> metrics (all rows, and --end-only)
HR bundle                        # step 9
```
Useful flags: `--budgets 1536` (primary only), `--skip-judge` (stop after export, judge later), `--retries N`,
`--resume DIR|latest`, `--new`, `--force-step NAME`, `--accept-partial`. The judge is always the profile's judge
(`claude-sonnet-5-5`, same family as the Haiku agent, so `--allow-same-family-judge` is passed and the limitation is
printed in every report, C.1-2). The vendor switch (OpenRouter / Gemini) is `--profile NAME` and a NEW decision set
(C.1-4): never mix profiles or embedders inside one campaign.

Recommended order: `doctor --deep` -> `smoke` -> `dry-run` -> `oh` (cheapest, ~7 h at B=1536) -> `beam` -> `cc` -> `bundle`.
Run `bundle` after each track if you want an early look; it is idempotent.

## 7. What each track gives you (and what it cannot)

* `oh`: primary = nc-graph vs baseline at B=1536 on 14 sessions, S1-S3 Holm family (C.2-1). The old "multihop"
  items are analysed as `xspan_locator` (descriptive S4d/S5d). `REPORT-chains.md`, `metrics.json`, `funnel.md`.
* `beam`: the C.2-3 long-session track: primary nc-graph vs baseline at B=1536 on all BEAM-1M chats, family L1-L5
  on the rubric-graded score, genuine multi-hop = `multi_session_reasoning`. Only 34 of 350 items carry a
  deterministic anchor, so anchor-based rules are n/a or low-power by construction; the score legs carry the result.
* `cc`: descriptive only (C.2-3): `orig` (Claude Code's own summary) vs baseline vs nc arms, lag and checkpoint tables.

## 8. If something goes wrong

* `doctor` FAIL on `hars.semantic_probe` or `hars.embedder`: you are on a shim/broken embedder. Fix HARS first.
* A step fails: the message names the log (`<campaign>/logs/<step>.log`). Fix, re-run the same command. A crashed OH batch
  dir is moved to `*.failed-<UTC>` (kept, never merged) and redone; a crashed BEAM/CC run is resumed with `--resume`.
* `BUDGET STOP`: raise the cap in `home_run.toml`, re-run.
* Anything that leaves HARS in a weird state: stop the server, move `<HARS_DATA_DIR>` aside, restart, `doctor --deep`,
  re-run (finished steps stay finished; the HARS store of a finished run is only needed for the funnel's graph stage).
* Host OOM: keep `batch_chains = 4` and `session_concurrency = 1`.

## 9. What to send back

```bash
HR bundle                         # prints BUNDLE: <path>.tar.gz, size and sha256
```
The bundle holds: summaries (`summary.json`, `MANIFEST.json`, `run_manifest.json`, `run_budget.json`,
`MERGE_MANIFEST.json`), metrics (`REPORT-*.md`, `metrics*.json`), funnel, per-sample files (`records.jsonl`,
`samples.jsonl`, `compactions.jsonl`, `chain_budget.jsonl`, `judged.jsonl`, `judge-mapping.json`), the campaign
`logs/`, `ledger.json` (spend), `doctor.json`, `dry-run.json`, the verify reports, the exact command lines and a copy
of the scripts. It EXCLUDES raw transcripts (`chains.jsonl`, imported sessions, `data*/`), HARS/session graph stores
(`sessions/`), raw recall payloads (`recall/`), resume caches (`memory/`, `partial/`), per-chain pi logs and
`judge-input.jsonl` (they contain transcript text); `--include-raw` adds them (still redacted). The Claude Code
`records.jsonl` keep questions and answers (excerpts of your sessions) but drop `memory_text` / `injected`; use
`--tracks oh,beam` to leave the cc track out entirely.

Before packing, the staged copy is scanned with the redaction patterns (sk-/sk-ant-/sk-or-, GitHub/GitLab/AWS/Slack/
Google tokens, JWT, PEM, `password=`-style assignments) plus the exact values of every `*KEY*/*TOKEN*/*SECRET*`
environment variable; hits are replaced by `[REDACTED:kind]` in the packed copy only, and
`bundle-redaction-report.json` (inside the archive) lists the files and counts. **Read the REDACTION SCAN line it
prints**; if it says hits > 0 skim the report before sending. Send the `.tar.gz` and its sha256.

## 10. Decisions that are yours

1. The three caps. 2. How many BEAM chats (all 35 = ~$39 and ~3 days at the guessed ingest speed; decide N in advance if
fewer). 3. Whether to run the B sweep on BEAM (L2/L3 need it; B=1536 alone leaves them n/a, p = 1). 4. Your embedder
(frozen afterwards). 5. Which Claude Code sessions (descriptive only).
