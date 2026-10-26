# HOME-RUNBOOK - running the neurocompact eval on your own machine

## START HERE (read this block, then skip to step 1)

1. **Use ONLY this checkout.** `git clone <your hars-memory remote> && cd hars-memory && git checkout claude/youthful-thompson-yv6xqd`,
   then work from **`experiments/neurocompact-eval/`** inside that repo. Everything below
   (`scripts/home_run.py`, `home_run.toml.example`, `profiles/`, `pi-agent/`) lives there and nowhere else.
2. **`/tmp/opencode/nc-eval` is OBSOLETE - do not use it.** It is an old working copy: old runner, OpenRouter-only judge and
   extractor, Haiku 4.5 defaults, the 5-session dataset. Nothing in it matches the protocol (`REPORT-PROTOCOL.md` Amendments C-C.3).
   `home_run.py doctor` now checks this itself: the `stale_copy` row is **FAIL** when a script, a profile or the
   `EVAL-VERSION` marker is missing or different (and WARN when the path looks like a scratch copy or is not a git checkout).
3. **Models** (printed by `doctor` as a role -> model table): agent, answer, summarizer, HARS extractor and keyword model are all
   **`claude-haiku-5-5`**; judge and QA generator **`claude-sonnet-5-5`**. If you see anything else, you are not on this checkout.
4. **The OpenHands dataset** is one tarball, `oh-multihop-qa-20261026T144511Z.tar.gz` (4.5 MB, 14 chains, not in git). Put it anywhere and
   let home_run unpack it into the artifacts root (safe extraction, then it is auto-detected, no config edit):
   ```bash
   cd experiments/neurocompact-eval
   uv run python scripts/home_run.py --oh-tar ~/Downloads/oh-multihop-qa-20261026T144511Z.tar.gz dry-run --tracks oh
   #   same thing by hand:  mkdir -p /mnt/datasets/ft/artifacts && tar -xzf oh-multihop-qa-20261026T144511Z.tar.gz -C /mnt/datasets/ft/artifacts
   #   -> /mnt/datasets/ft/artifacts/20261026T144511Z-oh-multihop-qa/chains.jsonl  (found automatically; or set [oh] artifact_dir)
   ```
5. **Small and cheap by default** (Amendment C.3, budget USD 20-30 in total): OpenHands 14 chains at B=1536 only; BEAM-1M 10 seeded chats at
   B=1536 only; Claude Code: the session(s) you pick. Sweeps (1536,1024,512) are opt-in in `home_run.toml`. Prompt caching is ON for the
   anthropic profile (cost-only, same model inputs). `home_run.py dry-run` prints the all-tracks total against USD 30 before any spend.
6. **Minimal sequence** (after `cp home_run.toml.example ../../home_run.toml`, the only edits are paths):
   ```bash
   cd experiments/neurocompact-eval/scripts
   uv run python home_run.py doctor --deep          # 0 FAIL (stale_copy, keys, real embedder, pi, models table)
   uv run python home_run.py dry-run --find "HARS Experts Research"     # total vs $30, nothing spent
   uv run python home_run.py oh && uv run python home_run.py beam
   uv run python home_run.py cc --find "HARS Experts Research"          # prints candidates, you pick; then import -> QA -> run -> judge
   uv run python home_run.py bundle                 # send the ONE .tar.gz back
   ```

---

Written 2026-10-26. You run everything here with your HARS, your real embedder and your Anthropic key
(later maybe OpenRouter / Gemini through a profile), and send ONE file back: the bundle (step 9).
The protocol is frozen in `REPORT-PROTOCOL.md` section 4a (Amendments C, C.1, C.2, C.3, C.4, **C.5**). Nothing in this
runbook changes a metric or a rule; the tooling is `scripts/home_run.py` (one entry point) on top of the
existing scripts. All paths below are relative to the repo root.

```
home_run.py doctor | hars-env | dry-run | smoke | calibrate | oh | beam | cc | bundle
```

## 0. What you are going to spend and wait (read this first)

Verified list prices (profile `anthropic`, `profiles/anthropic.json`, status `confirmed`): Haiku 5.5
**$0.10 in / $0.50 out per MTok**, **$0.50 / $2.50 for a prompt above 100k tokens**; Sonnet 5.5 **$2 / $10**
(the judge and the Claude-Code QA generator). `dry-run` computes everything below from your actual data; these
are the numbers of today's data (tokenizer = the chars/3.6 estimator because tiktoken was not installed where
they were computed; with tiktoken they move by roughly +-15%).

**Default run (what `home_run.toml.example` does, Amendment C.3): about USD 15-20 of the USD 30 ceiling.**

| Track | what | run (replay/compaction/answers + flagged HARS estimate) | judge (Sonnet) | gen (QA) | wall-clock |
|---|---|---|---|---|---|
| `oh` DEFAULT: B=1536 only | 14 chains x 10 items x 4 arms | **$0.75** with prompt cache (uncached $3.64) | $1.90 (560 items) | - | ~7 h |
| `beam` DEFAULT: B=1536 only, **N=10 seeded chats** | BEAM-1M, ~13M tokens, ~100 boundaries, ~5.7k HARS pieces | **~$9.8** (of which HARS extraction ~$6.0, an ESTIMATE; scaled from the 35-chat figure, `dry-run` gives the real one; answer-call cache saving not included) | ~$1.4 (400 items) | - | ~19 h (~17 h HARS ingest at the guessed 11 s/piece) |
| `cc` DEFAULT: the session(s) you pick (`cc --find`) | per session with ~7 real boundaries, 54 generated questions (40 single + 10 multihop + 4 unanswerable; menu mode), every:3 checkpoints | ~$1-1.5 / session | ~$0.3-0.7 / session | ~$1-2 / session (Sonnet, upper-ish bound) | ~2-3 h / session |
| **DEFAULT TOTAL, one cc session** | | **~$12** | **~$3.7** | **~$1.5** | **~$17**; caps 3 / 20 / 5, total 30 |

Opt-in (not run by default; `dry-run` prices them from your `[oh]/[beam]` budgets, figures below are UNCACHED list-price estimates of the C.2 numbers):

| Opt-in | what | run | judge | wall-clock |
|---|---|---|---|---|
| `oh` sweep `budgets = [1536,1024,512]` | same x 3 B | $10.91 (about $2.3 cached) | $5.71 | ~21 h |
| `beam` all 35 chats (`chats = "all"`) | 45M tokens, 360 boundaries, 20k HARS pieces | $34.29 (HARS extraction $21.07, an ESTIMATE) | $4.76 | ~68 h |
| `beam` sweep (L2/L3) | asks x3 | + about 75% of the run, judge x3 | $14.28 (35 chats) | + |

**Prompt caching (Amendment C.3, cost-only):** Anthropic bills a cached prompt prefix at 0.1x the input price (cache read) after one
write at 1.25x (5 min) or 2x (1 h). The model inputs are byte-identical; only the bill changes, and every arm is cached identically.
Where it applies: the pi agent's growing replay context (the OH track, where it removes about 80% of the agent cost) and the memory + tail
prefix of the long-session answers. The one-shot summariser / compaction calls are not cached. `--prompt-cache off` (or
`cache.enabled=false` in `profiles/anthropic.json`) shows the uncached figures; the run MANIFEST records `prompt_cache` (cached calls,
read/write tokens, cost, cost without cache).

Formulas (also in `README-longsession.md`): judge = items x (1300 in x $2 + 80 out x $10)/1e6 = $0.0034 per
item; BEAM/CC run = baseline summaries + note summaries + `chunks x (2700 in, 600 out)` HARS extraction +
asks; OH run = `run_chainreplay --dry-run` projection (agent turns + extractor + keyword calls).
**Two numbers are guesses and dominate the BEAM wall-clock: HARS ingest seconds per piece (11 s) and extractor
tokens per chunk.** Calibrate on the first session (`<campaign>/beam/run/compactions.jsonl`, rows with
`"arm": "hars"` have `seconds`) and re-run `dry-run`. (Measured in the build container with a Haiku extractor: 6.4-7.1 s per 4000-char piece;
BEAM pieces are 8000 chars, so 11 s/piece is a plausible middle, but it is a guess.) If ~19 h is too much, the BEAM run is **resumable**
(Ctrl-C, re-run the same command). The default N=10 chats is fixed in `[beam] n_chats` (seeded sample, written to
`<campaign>/beam/selection.json` before the run, Amendment C.3-2); change N BEFORE running (`--beam-n N`), never after seeing results.

## 1. Clean checkout

```bash
git clone <your hars-memory remote> && cd hars-memory && git checkout claude/youthful-thompson-yv6xqd
uv sync --extra dev                       # NOT `--group dev`; python >= 3.13; tiktoken is a core dependency
uv run python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); print('tiktoken ok')"   # warms the BPE cache
npm i -g @earendil-works/pi-coding-agent@1.4.0     # only the `oh` track needs pi; 1.4.x is pinned (event parser)
#   no global npm?  npm i --prefix ~/.local/pi @earendil-works/pi-coding-agent@1.4.0 && export PATH=~/.local/pi/node_modules/.bin:$PATH
pi --version && node --version             # 1.4.x and Node >= 20
cp experiments/neurocompact-eval/home_run.toml.example home_run.toml     # defaults are the small run + suggested caps; edit paths only
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
the server's extractor/keyword model vs the profile; keys; **`tokcount.native` (Amendment C.5: one free `count_tokens` call on the answering model must succeed; FAIL = a run would be refused)**; pi 1.4.x; node; free disk; tiktoken; your inputs.
Do not run two clients against the HARS engine at once (1 thread / queue 32 by design; the host has an OOM history).

## 4. Inputs

* **OpenHands chains**: the tarball `oh-multihop-qa-20261026T144511Z.tar.gz` (14 chains, single + `xspan_locator` + unanswerable; not in git;
  public SWE-bench sessions, no secrets). `home_run.py --oh-tar PATH.tar.gz <subcommand>` unpacks it into the artifact root
  (`/mnt/datasets/ft/artifacts/20261026T144511Z-oh-multihop-qa/`: `chains.jsonl`, `chains-manifest.json`, `verify-report.md`);
  `[oh] artifact_dir` empty = auto-detect the newest `<artifact_root>/*-oh-multihop-qa`; or set it (`--oh-dir`) to a directory you unpacked yourself.
* **BEAM**: `git clone https://github.com/mohammadtavakoli78/BEAM ~/data/BEAM` (4 GB). Set `[beam] repo`.
  Only the JSON files are read (never pickles); `chats/1M/` has 35 chats. Default: **10 of them**, drawn once with
  `random.Random("20261026:1M").sample(...)` and written to `<campaign>/beam/selection.json` before the run (`chats = "all"` = all 35).
* **Claude Code sessions**: no path needed. `home_run.py cc --find "HARS Experts Research"` (title or first-message substring),
  `cc --session-id <cse_... or uuid>` or `cc --list` (longest sessions with their compaction counts) search
  `~/.claude/projects/*/*.jsonl` (`[cc] root` / `--projects-root` to change). It prints each candidate (session id, project, size, date
  range, message count, **#compact boundaries**, title) and you pick (`--pick N`, or the prompt on a terminal); only a session with real
  `/compact` boundaries can run. The pick is recorded in `<campaign>/cc/selected-session.json`. (The target session is probably
  `~/.claude/projects/-home-angkira-Project-software-cortex/03023d8a-4fa0-4ff8-8345-a67624798daf.jsonl`; `cc --session-id 03023d8a-4fa0-4ff8-8345-a67624798daf`
  confirms it.) Explicit files still work: `[cc] paths` / `--cc-paths`. Secret **redaction is ON and cannot be turned off through home_run**;
  read the redaction counts the import prints. Your private text lives in `<campaign>/cc/data*` and in `cc/run/*`: none of it leaves the machine
  except the excerpts inside `records.jsonl` / `judged.jsonl` (see step 9).

## 5. Caps (hard)

`home_run.toml.example` ships suggested caps: `gen` 3, `run` 20, `judge` 5 and `total` 30 (USD). `gen` = QA generation, `run` = replays,
compactions, answers + the flagged HARS estimate, `judge` = the blind judge. They are **campaign-wide per phase** (summed over the tracks)
unless a `[caps.<track>]` override exists; `total` is the hard ceiling over everything (0 in the file = every paid step refuses). Run
`dry-run` (step 6): it prints the all-tracks total against USD 30 and each phase against its cap. Each underlying script gets
`--budget-usd = min(cap - already spent, total - already spent)` (ledger in the campaign dir) and stops at it ("BUDGET STOP"); the runs are
B-major, so a stop still leaves the primary B=1536 cell complete. A run whose estimate is above its cap is refused unless `--accept-partial`.

## 6. The command sequence

All commands: `uv run python experiments/neurocompact-eval/scripts/home_run.py [--config home_run.toml] ...`
(abbreviated `HR` below). Everything lands in ONE campaign dir
`/mnt/datasets/ft/artifacts/<UTC>-homerun-home/` (change `artifact_root`; never `/tmp`, it is swept), logs and exact
command lines included. Re-running a subcommand skips finished steps and resumes unfinished runs.

```bash
HR dry-run [--find "TITLE"]      # zero spend: per-track gen/run/judge estimate, ALL-TRACKS TOTAL vs $30, cache saving, wall-clock
HR smoke                         # 1 cheapest OH chain (incl. its baseline-cap calibration) + 1 tiny synthetic long session, ~$0.5-0.8 plus the calibration runs, cents each (own campaign); must PASS
HR calibrate                     # Amendment C.4, ONCE per decision set, BEFORE `oh`: baseline cap on the calibration chain
                                 #   (first chain in seeded order; no QA asks, ~cents) -> <campaign>/calibration.json; must say PASS
HR oh                            # OpenHands chains (add --oh-tar TAR the first time): batches of 4, B-major, retries invalid cells once,
                                 #   merge -> export -> judge -> metrics_v2 -> funnel
HR beam                          # BEAM-1M: seeded 10-chat import -> run_longsession (end checkpoint, B per [beam] budgets)
                                 #   -> export_longsession_judge -> judge (rubric) -> metrics_v2 --end-only --include-sid ^beam-1M-
HR cc --find "TITLE"             # (or --session-id ID / --list): pick the session, then import (redacted) -> QA generation
                                 #   -> run (real boundaries, orig arm)
                                 #   -> judge -> metrics (all rows, and --end-only)
HR bundle                        # step 9
```
Useful flags: `--budgets 1536` (primary only), `--skip-judge` (stop after export, judge later), `--retries N`,
`--resume DIR|latest`, `--new`, `--force-step NAME`, `--accept-partial`. The judge is always the profile's judge
(`claude-sonnet-5-5`, same family as the Haiku agent, so `--allow-same-family-judge` is passed and the limitation is
printed in every report, C.1-2). The vendor switch (OpenRouter / Gemini) is `--profile NAME` and a NEW decision set
(C.1-4): never mix profiles or embedders inside one campaign.

Recommended order: `doctor --deep` -> `smoke` -> `dry-run` -> `calibrate` -> `oh` (cheapest, ~7 h at B=1536) -> `cc` (2-3 h) -> `beam` (~19 h) -> `bundle`.
Run `bundle` after each track if you want an early look; it is idempotent.

### 6.0 Amendment C.5 - tokens are native (read first)

Parity and every budget are now counted in the answering model's NATIVE tokens (`count_tokens`, free; needs the same key; cached
in `~/.cache/nc-tokcount/cache.jsonl`, override `NC_TOKCOUNT_CACHE`). Your bundle showed why: a baseline summary of 959 cl100k
tokens hit `max_tokens=1536` (~1.6x). The calibration of `20261026T172729Z-homerun-home` is VOID: use a NEW campaign (`--new`),
refresh the scripts (EVAL-VERSION `2026-10-26.c5`), run `doctor` (row `tokcount.native` must be OK), then `calibrate`. The cap
defaults to cap(B) = B and should now calibrate at ~1.0; `[oh] calibration_chains = 3` (default 1) averages the fill over more
chains (all excluded from the decision set; costs ~3x the cents-level calibration). `calibration.json` shows `ratio_ask_time`
(the gate's quantity) and `ratio_pooled_all_compactions`. Offline: `NC_TOKCOUNT=estimate`; `--token-counter tiktoken` accepts a
flagged non-native measurement (not for a decision set on Claude).

### 6a. Baseline-cap calibration (Amendment C.4) - why `oh` refuses without it

With the baseline cap equal to B the baseline summaries can be cut at the cap (`stop=length`, CLIP FAIL) and the
realized parity ratio nc/baseline can leave [0.80, 1.05] (your home smoke: 1.31). The cap is therefore a calibrated parity
parameter. `HR calibrate` runs `run_chainreplay.py --calibrate` on ONE chain (the first in the seeded order of
`[oh] calibration_seed`, default 20261026; no QA asks, no judge), reads the runner's own advice (`recommended_cap`) and
re-runs at that cap, at most `--max-iterations` runs per B (default 2: initial + one retry; `[oh] calibration_max_iterations`).
It accepts when the run is CLIP-free AND the ratio is inside the window, then writes `<campaign>/calibration.json`
(per-B caps, `scale`, ratio achieved, chain id, cost, every iteration). Then:

* `HR oh` takes the calibrated caps (`--baseline-cap 1536:<cap>` + `--baseline-cap-scale <cap/1536>` + `--baseline-cap-source
  calibrated:<run dir>` go to every `run_chainreplay.py`); a B you did not calibrate (`--budgets 1536,1024`) uses the scale;
  both are recorded per B in `oh/caps.json`, `chain_budget.jsonl`, `records.jsonl`, `run_manifest.json`, `run_budget.json`.
* The calibration chain is EXCLUDED from the decision set automatically (13 of 14 chains run; `oh/inputs/chains.jsonl` is
  the decision set, `chains.all.jsonl` the full artifact; `--exclude-chains` to the runner and merge, `--exclude-sid` to metrics).
* The caps are frozen for the campaign (`oh/caps.json`); other caps need `--new` (a new decision set).
* `HR oh` without a passing `calibration.json` REFUSES (before any spend) unless you give a cap yourself: `HR oh --baseline-cap
  1536:2010` (or `--baseline-cap-scale 1.31`, or `[oh] baseline_cap = "1536:2010"` / `baseline_cap_scale` in `home_run.toml`).
  An explicit cap is recorded with source `cli` and is NOT a calibration; `HR calibrate` also starts from such a value.
* `HR smoke` runs the same calibration on its own single chain (not excluded) and then the OH flow at the calibrated cap.

**If the calibration FAILS** (`[calibrate] FAIL: ...`, `calibration.json` has `status: "fail"`, `per_b.<B>.stopped_because` and
`next_cap_suggestion`): nothing was run on the decision set. In order: (1) the reason `max iterations ... reached` - the advice
still moved the cap; re-run with more room `HR calibrate --max-iterations 4` (finished iterations are kept and skipped; the
extra runs cost cents); (2) `NC note itself is clipped` - the nc note cap is frozen by the protocol, not tunable here: send the
bundle (`HR bundle`) and the log `logs/oh.calibrate.*.log`, do NOT change `--note-sum-max-tokens`; (3) `advice does not move the
cap` / `no measured ratio` - the realized baseline does not grow with the cap (the model saturates) or a run produced no
tokens: send the bundle, do not hand-tune the cap on decision chains; (4) you may always pass the cap you want explicitly
(`HR oh --baseline-cap 1536:<cap>`), but that is then an uncalibrated choice and the run says so.

## 7. What each track gives you (and what it cannot)

* `oh`: primary = nc-graph vs baseline at B=1536 on 13 sessions (the 14th is the calibration chain, C.4-4), S1-S3 Holm family (C.2-1). The old "multihop"
  items are analysed as `xspan_locator` (descriptive S4d/S5d). `REPORT-chains.md`, `metrics.json`, `funnel.md`.
* `beam`: the C.2-3 / C.3-2 long-session track: primary nc-graph vs baseline at B=1536 on the 10 seeded BEAM-1M chats, family L1-L5 (L2/L3 n/a without the sweep)
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

1. The caps (suggested 3 / 20 / 5, total 30). 2. Whether to enlarge a track (BEAM `chats = "all"` or a larger `n_chats`; the OH or BEAM B sweep:
L2/L3 need the sweep, B=1536 alone leaves them n/a, p = 1) - decide BEFORE running. 3. Your embedder (frozen afterwards). 4. Which Claude Code
sessions (descriptive only).
