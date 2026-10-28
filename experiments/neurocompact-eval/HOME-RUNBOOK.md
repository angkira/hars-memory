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
   Amendment C.6: `oh` reads **`chains.jsonl` only** (the chain records carry the raw span events; `chains_to_longsession.py` turns them
   into long sessions). The `sessions/` imports of the dataset are not needed and not read. **No pi agent, no Node**: the OpenHands
   track runs on the same streaming runner as BEAM and Claude Code (`REPORT-PROTOCOL.md` 4a.6).
5. **Small and cheap by default** (Amendment C.3, budget USD 20-30 in total): OpenHands 14 chains at B=1536 only; BEAM-1M 10 seeded chats at
   B=1536 only; Claude Code: the session(s) you pick. Sweeps (1536,1024,512) are opt-in in `home_run.toml`. Prompt caching is ON for the
   anthropic profile (cost-only, same model inputs). `home_run.py dry-run` prints the all-tracks total against USD 30 before any spend.
6. **Minimal sequence** (after `cp home_run.toml.example ../../home_run.toml`, the only edits are paths):
   ```bash
   cd experiments/neurocompact-eval/scripts
   uv run python home_run.py doctor --deep          # 0 FAIL (stale_copy, keys, real embedder, tokcount.native, models table)
   uv run python home_run.py dry-run --find "HARS Experts Research"     # total vs $30, nothing spent
   uv run python home_run.py calibrate              # C.6: cheap parity CHECK on 1 OH session (cents); must say PASS
   uv run python home_run.py oh && uv run python home_run.py beam      # oh: convert -> run_longsession -> judge -> metrics
   uv run python home_run.py cc --find "HARS Experts Research"          # prints candidates, you pick; then import -> QA -> run -> judge
   uv run python home_run.py bundle                 # send the ONE .tar.gz back
   ```
   **You already have a finished `oh` campaign (question-time arms)?** Do NOT re-run it. Read section 6.0c (Amendment C.8): `oh-provision`,
   restart your server once, `doctor`, `oh-ingest`, `oh-probe`, `oh-enrich` (nc-enriched against the session's own project). **Amendment C.9
   makes the agentic runs the PRIMARY question: after `oh-ingest`, read section 6.0d (`oh-workspaces`, `oh-agentic`).** The split-session
   HARS data of that campaign measured a server side channel, not the memory (6.0b is kept for the record).

---

Written 2026-10-26, updated 2026-10-27 (Amendment C.6: no pi). You run everything here with your HARS, your real embedder and your Anthropic key
(later maybe OpenRouter / Gemini through a profile), and send ONE file back: the bundle (step 9).
The protocol is frozen in `REPORT-PROTOCOL.md` section 4a (Amendments C, C.1, C.2, C.3, C.4, C.5, **C.6**). Nothing in this
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
| `oh` DEFAULT: B=1536 only (Amendment C.6: `run_longsession`, no pi) | 14 chains x 3 boundaries, 10 items x 4 arms | **$0.33** with prompt cache (main run $0.31 + the 1-session parity check $0.02; the pi replay was $0.75 / ~7 h); the HARS-extraction share is a LOWER bound (see below) | $1.90 (560 items) | - | model ~2.3 h; **measured ~3 h** (HARS ingest ~50 s per 8000-char piece, ~2 h of it) |
| `beam` DEFAULT: B=1536 only, **N=10 seeded chats** | BEAM-1M, ~13M tokens, ~100 boundaries, ~5.7k HARS pieces | **~$9.8** (of which HARS extraction ~$6.0, an ESTIMATE; scaled from the 35-chat figure, `dry-run` gives the real one; answer-call cache saving not included) | ~$1.4 (400 items) | - | ~19 h (~17 h HARS ingest at the guessed 11 s/piece) |
| `cc` DEFAULT: the session(s) you pick (`cc --find`) | per session with ~7 real boundaries, 54 generated questions (40 single + 10 multihop + 4 unanswerable; menu mode), every:3 checkpoints | ~$1-1.5 / session | ~$0.3-0.7 / session | ~$1-2 / session (Sonnet, upper-ish bound) | ~2-3 h / session |
| **DEFAULT TOTAL, one cc session** | | **~$12** | **~$3.7** | **~$1.5** | **~$17**; caps 3 / 20 / 5, total 30 |

Opt-in (not run by default; `dry-run` prices them from your `[oh]/[beam]` budgets, figures below are UNCACHED list-price estimates of the C.2 numbers):

| Opt-in | what | run | judge | wall-clock |
|---|---|---|---|---|
| `oh` sweep `budgets = [1536,1024,512]` | same x 3 B | ~$0.62 (main run $0.59 + parity check) | $5.71 | ~5.6 h |
| `beam` all 35 chats (`chats = "all"`) | 45M tokens, 360 boundaries, 20k HARS pieces | $34.29 (HARS extraction $21.07, an ESTIMATE) | $4.76 | ~68 h |
| `beam` sweep (L2/L3) | asks x3 | + about 75% of the run, judge x3 | $14.28 (35 chats) | + |

**Prompt caching (Amendment C.3, cost-only):** Anthropic bills a cached prompt prefix at 0.1x the input price (cache read) after one
write at 1.25x (5 min) or 2x (1 h). The model inputs are byte-identical; only the bill changes, and every arm is cached identically.
Where it applies: the memory + tail prefix of the long-session answers (OH since C.6, BEAM, CC; the pi agent's growing replay context
only under `--legacy-replay`). The one-shot summariser / compaction calls are not cached. `--prompt-cache off` (or
`cache.enabled=false` in `profiles/anthropic.json`) shows the uncached figures; the run MANIFEST records `prompt_cache` (cached calls,
read/write tokens, cost, cost without cache).

Formulas (also in `README-longsession.md`): judge = items x (1300 in x $2 + 80 out x $10)/1e6 = $0.0034 per
item; BEAM/CC run = baseline summaries + note summaries + `chunks x (2700 in, 600 out)` HARS extraction +
asks; OH run = the same `run_longsession --dry-run` projection on the converted chains (Amendment C.6; `dry-run --legacy-replay`
prices the old pi replay with `run_chainreplay --dry-run`).
**Two numbers are guesses and dominate the BEAM (and OH) wall-clock: HARS ingest seconds per piece (11 s) and extractor
tokens per chunk.** (First live OH check, 2026-10-27, one chain, hash-shim embedder: 146-164 s per boundary = ~50 s per piece, and HARS
made 59-69 extraction chunks per boundary where `run_longsession` assumes ~6: the dry-run HARS dollars are a lower bound, the hours too.) Calibrate on the first session (`<campaign>/beam/run/compactions.jsonl`, rows with
`"arm": "hars"` have `seconds`) and re-run `dry-run`. (Measured in the build container with a Haiku extractor: 6.4-7.1 s per 4000-char piece;
BEAM pieces are 8000 chars, so 11 s/piece is a plausible middle, but it is a guess.) If ~19 h is too much, the BEAM run is **resumable**
(Ctrl-C, re-run the same command). The default N=10 chats is fixed in `[beam] n_chats` (seeded sample, written to
`<campaign>/beam/selection.json` before the run, Amendment C.3-2); change N BEFORE running (`--beam-n N`), never after seeing results.

## 1. Clean checkout

```bash
git clone <your hars-memory remote> && cd hars-memory && git checkout claude/youthful-thompson-yv6xqd
uv sync --extra dev                       # NOT `--group dev`; python >= 3.13; tiktoken is a core dependency
uv run python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); print('tiktoken ok')"   # warms the BPE cache
#   Amendment C.6: NO pi / Node needed. Only for the legacy replay (`oh --legacy-replay`, reproducing old numbers):
#     npm i -g @earendil-works/pi-coding-agent@1.4.0 ; pi --version && node --version    # 1.4.x and Node >= 20
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
the server's extractor/keyword model vs the profile; keys; **`tokcount.native` (Amendment C.5: one free `count_tokens` call on the answering model must succeed; FAIL = a run would be refused)**; free disk (pi 1.4.x and node only with `doctor --legacy-replay`); tiktoken; your inputs.
Do not run two clients against the HARS engine at once (1 thread / queue 32 by design; the host has an OOM history). home_run now enforces this (lock file per HARS endpoint, section 6.2).

## 4. Inputs

* **OpenHands chains**: the tarball `oh-multihop-qa-20261026T144511Z.tar.gz` (14 chains, single + `xspan_locator` + unanswerable; not in git;
  public SWE-bench sessions, no secrets). `home_run.py --oh-tar PATH.tar.gz <subcommand>` unpacks it into the artifact root
  (`/mnt/datasets/ft/artifacts/20261026T144511Z-oh-multihop-qa/`: `chains.jsonl`, `chains-manifest.json`, `verify-report.md`; only `chains.jsonl` is read);
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
HR smoke                         # 1 smallest OH chain (parity check + the whole streaming flow) + 1 tiny synthetic long session, cents-level
                                 #   (own campaign); must PASS
HR calibrate                     # Amendment C.6: cheap parity CHECK on the run_longsession path (1 session of the seeded order, compaction + 2 fixed
                                 #   probe asks, ~cents; spends no decision-set answer, so nothing is excluded) -> <campaign>/calibration.json; must say PASS
HR oh                            # OpenHands chains (add --oh-tar TAR the first time): chains_to_longsession -> parity check (if not done) ->
                                 #   run_longsession (3 arms baseline,nc-enriched,nc-note-only [C.7], end checkpoint, B=1536) -> export_longsession_judge -> judge -> metrics_v2 --track longsession
                                 #   (+ the descriptive chain-family reading) -> funnel. `--legacy-replay` = the old pi path (not the decision runner)
HR beam                          # BEAM-1M: seeded 10-chat import -> run_longsession (end checkpoint, B per [beam] budgets)
                                 #   -> export_longsession_judge -> judge (rubric) -> metrics_v2 --end-only --include-sid ^beam-1M-
HR cc --find "TITLE"             # (or --session-id ID / --list): pick the session, then import (redacted) -> QA generation
                                 #   -> run (real boundaries, orig arm)
                                 #   -> judge -> metrics (all rows, and --end-only)
HR oh-probe  --resume CAMP       # Amendment C.7 (section 6.0b): recall probe on a FINISHED oh campaign - cents, no answer model, no judge
HR oh-enrich --resume CAMP       # Amendment C.7: ONLY nc-enriched against that campaign (reuses its HARS sessions, notes, judged baseline /
                                 #   nc-note-only), judges the new rows only, merges, metrics. Run oh-probe first; --dry-run on both
HR oh-provision / oh-ingest --resume CAMP   # Amendment C.8 (section 6.0c): the session's own project on your existing server, then ingest
                                 #   (oh-probe / oh-enrich default to this project layout; `--layout split-session` = DIAGNOSTIC ONLY)
HR oh-workspaces / oh-agentic --resume CAMP   # Amendment C.9 (section 6.0d, PRIMARY): per-session workspaces (network, no spend), then agents with / without
                                 #   memory_recall per question of the finished campaign -> judge (--outcome) -> metrics_agentic -> bundle
HR bundle                        # step 9
```
Useful flags: `--budgets 1536` (primary only), `--skip-judge` (stop after export, judge later), `--retries N` (legacy replay only),
`--skip-parity-check` / `--accept-parity-fail` (oh, recorded), `--check-sessions K` (calibrate), `--resume [ID|PATH]`, `--new`, `--force-step NAME`, `--accept-partial`, `--force-shared-hars`. The judge is always the profile's judge
(`claude-sonnet-5-5`, same family as the Haiku agent, so `--allow-same-family-judge` is passed and the limitation is
printed in every report, C.1-2). The vendor switch (OpenRouter / Gemini) is `--profile NAME` and a NEW decision set
(C.1-4): never mix profiles or embedders inside one campaign.

### 6.1 Which campaign does a command use? (NO implicit resume)

Since 2026-10-27 home_run never attaches to "the last campaign" on its own (a `calibrate` used to silently join an old,
hand-touched one). The rule:

| you type | what happens |
|---|---|
| `HR calibrate` / `oh` / `beam` / `cc` / `dry-run` (no flag) | a **NEW** campaign is created and its path printed (`[campaign] NEW campaign ...`) |
| `HR oh --resume` | the **newest campaign by creation time** (`campaign.json` `created_utc`, never mtime; smoke / dry-run campaigns are skipped), announced in a `!!!` banner |
| `HR oh --resume 20261027T150000Z-homerun-home` (or a path, or a unique fragment of the name) | exactly that campaign |
| `HR oh --resume --new` / `--new` | a new campaign (`--new` wins) |
| `HR bundle`, `HR status` (read-only) | refuse without `--resume` and list the newest campaigns; `status` alone just lists them |
| `HR doctor` | needs no campaign (report goes to `<root>/.doctor/`); `doctor --resume ID` stores `doctor.json` in that campaign |

Put `--resume` AFTER the subcommand (`HR oh --resume`); before it, `--resume oh` would read `oh` as the campaign id.

So the normal chain is: `HR calibrate` (prints the new campaign path) -> `HR oh --resume` (or `--resume <that id>`) -> `HR cc ... --resume`
-> `HR bundle --resume`. `oh` in a NEW campaign has no `calibration.json` and runs its own cheap parity check first (it tells you if an earlier
campaign already holds a passing one). `HR status --resume ID` prints created time, EVAL-VERSION, steps, spend, calibration state and whether the
campaign may still spend.

### 6.2 Safety guards on paid steps

* **Void or foreign campaigns never spend.** Every paid step (run, judge, QA generation, calibrate, oh, beam, cc) first checks: the campaign's
  `eval_version` (recorded in `campaign.json` at creation; older campaigns fall back to their own `scripts/home_run.py` copy) equals the code's
  `EVAL-VERSION`; `EVAL-VERSION` file and `home_run.py` agree; `calibration.json` is of Amendment C.6 (C.5 for `--legacy-replay`), not marked
  `void`, and written under this EVAL-VERSION; `oh/caps.json` is not void. Otherwise: `refusing to spend ... Start a new campaign`, nothing spent.
* **One client per HARS.** Every step that writes to HARS (smoke, calibrate, oh, beam, cc, `doctor --deep`) takes an advisory lock
  `~/.cache/nc-eval/locks/<sha1(mcp url)>.lock` (pid, campaign, start time; a dead pid = stale, replaced). A second `home_run` refuses and prints the holder.
  `--force-shared-hars` overrides (at your own risk). If the server exposes `memory_lease_acquire` / `memory_lease_release`, home_run also takes a lease
  tagged with the campaign id and sends it as `run_tag` on `memory_compact` writes (`run_longsession --run-tag`); an older server without the tools is tolerated.
* **`doctor --deep` probe**: one throwaway session `doctor-probe-<UTC>` receives THREE separate realistic notes (~330 tokens each, above the
  server's 200-token chunk minimum) via `memory_compact`; one zero-overlap paraphrase query per note; the row `hars.semantic_probe` lists per-note hit/MISS and
  scores (>= 2 of 3 passes). Row `hars.probe_isolation` is OK only in `HARS_MEMORY_SESSION_MODE=split` (the generated env): in `subgraph` mode a compaction is indexed into the MAIN graph
  and the row FAILS (remove `session:doctor-probe-*` docs and use a throwaway data dir).

Recommended order: `doctor --deep` -> `smoke` -> `dry-run` -> `calibrate` -> `oh` (cheapest, ~2.3 h at B=1536) -> `cc` (2-3 h) -> `beam` (~19 h) -> `bundle`.
Run `bundle` after each track if you want an early look; it is idempotent.

### 6.0d Amendment C.9 - the agentic runs (PRIMARY question; read after `oh-ingest`)

The owner's question: does the agent spend MORE tokens finding a fact WITHOUT memory than when it fetches it from memory? `oh-agentic` runs, for each of the 140
questions, an agent with workspace tools (ls / read_file / grep over the session's repo at `base_commit`) and the baseline summary (`sum`), and the same plus a
`memory_recall` tool against the session's C.8 project (`sum+mem`). Neither arm sees the transcript. Everything else (the C.6 summary, the C.8 ingest, the judge model)
is reused. The C.7/C.8 fixed-budget arms are now secondary (protocol 4a.9).

Exact sequence after `oh-ingest` (the server from 6.0c keeps running; `CAMP` = the finished oh campaign):

```bash
cd experiments/neurocompact-eval/scripts
CAMP=<the finished oh campaign>
# 1) workspaces (NETWORK; spends nothing): clones github.com/<repo> into a cache, checks out base_commit, deletes .git. The OH data has neither
#    base_commit nor the final patch: base_commit comes from the SWE-bench Verified dataset; to apply each session's FINAL patch pass --instances FILE
#    (jsonl/json rows {instance_id, base_commit[, model_patch]}). Without it the workspace is base_commit only and the command says so.
uv run python home_run.py oh-workspaces --resume $CAMP --dry-run                 # the plan: instance, repo, base_commit source, patch yes/no
uv run python home_run.py oh-workspaces --resume $CAMP [--instances FILE] [--cache-dir DIR]   # default cache ~/.cache/nc-eval/workspaces
# 2) the cost estimate (zero spend): the steps x tokens model with its assumptions, the cache / no-cache run cost, the judge, the pre-run power guess
uv run python home_run.py oh-agentic --resume $CAMP --dry-run
# 3) the run (PAID): sum + sum+mem per question -> judge (--outcome schema, same judge model as C.6) -> metrics_agentic -> report
uv run python home_run.py oh-agentic --resume $CAMP [--max-steps 12] [--max-question-tokens 400000] [--memory-max-chars 6000] [--concurrency 2]
uv run python home_run.py bundle --resume $CAMP                                   # send the .tar.gz back
# 4) ONLY if sum+mem showed a gain over sum (the report says): the optional note+mem arm
uv run python home_run.py oh-agentic --resume $CAMP --arms note+mem
```
- Requirements (enforced): `oh-ingest` done (the C.8 projects hold the data; your HARS server running, the same one), the workspaces step done, the same judge model as
  the C.6 campaign, the caps (`[caps] run / judge / total`), the per-HARS lock. A campaign stamped `2026-10-27.c6`, `2026-10-28.c7` or `2026-10-28.c8` is accepted (code `2026-10-28.c9`).
- It is resumable: re-running the same command keeps finished (question, arm) rows (`oh/agentic-run/records.jsonl`) and finished steps. Outputs in `oh/agentic-run/`:
  `records.jsonl` (per step tokens, totals, USD, tool and memory calls, answer, anchor-in-tool-result flags), `traces.jsonl` (full traces, raw: not in the default bundle),
  `samples.jsonl`, `judge-input.jsonl`, `judged.jsonl`, `REPORT-oh-agentic.md`, `metrics-agentic.json`, `summary.json`, `MANIFEST.json`, `scripts/`, `inputs/`.
- Reading the report: the PRIMARY block (subset b = anchor in the dropped context and not in the summary): `sum+mem - sum` total tokens per question and accuracy, an estimate with
  a 95% CI over sessions and the MDE from the observed SD. Hallucination is a guard (over all and given answered); `unsupported_no_anchor` is listed apart. No WIN / FAIL.
- Keys: the run uses `ANTHROPIC_API_KEY` (or the profile's key env), never printed. Nothing here edits your global KB (`default` is only read through the project, never written).

### 6.0c Amendment C.8 - the memory is the session's own PROJECT (read before `oh-ingest` / `oh-enrich`)

The split-session runs (C.6 arms, C.7 as first shipped) measured the server's session side channel, not the memory (protocol 4a.8). The
decision configuration writes each eval session's dropped context into ITS OWN project on your EXISTING server (`memory_upsert_document`,
`project=nc-<id>-<sid>`) and reads it back with `memory_recall(project=..., ll/hl keywords, hybrid, max_chars)`, no `session_id` - like the
global memory. No second server, no mode switch, no new tool. The old split path stays as `--layout split-session`, DIAGNOSTIC ONLY.

```bash
cd experiments/neurocompact-eval/scripts
CAMP=<the finished oh campaign>
uv run python home_run.py oh-provision --resume $CAMP [--projects-dir DIR]   # 1) writes <projects dir>/nc-<id>-<sid>/project.json for all 14 sessions
#    2) RESTART your existing HARS server ONCE (projects are read at start only) with HARS_MEMORY_PROJECTS_DIR=<that dir>
#       (the default dir ~/.local/share/hars-longterm-memory/projects needs no env; otherwise also set [hars] projects_dir in home_run.toml)
uv run python home_run.py doctor    --resume $CAMP        # 3) rows projects.registered (14/14), projects.default_untouched, projects.state (all empty)
uv run python home_run.py oh-ingest --resume $CAMP --dry-run   #  estimate: ingest wall-clock + server-side extraction cost
uv run python home_run.py oh-ingest --resume $CAMP        # 4) writes the dropped spans (old notes, same pieces / boundaries); no model call of ours
uv run python home_run.py oh-probe  --resume $CAMP        # 5) cents: anchor recall on the ingested projects (A old shape, B/C correct shape, D BM25)
uv run python home_run.py oh-enrich --resume $CAMP --dry-run
uv run python home_run.py oh-enrich --resume $CAMP        # 6) PAID: nc-enriched only (judge = new rows only), merge, metrics
uv run python home_run.py bundle    --resume $CAMP        # 7) send the .tar.gz back
```
- `oh-provision` only creates directories under the projects dir and records `c8_projects` in `campaign.json`. It starts and stops nothing, never
  touches `default` or your global KB, and is idempotent (same run id on a re-run).
- `oh-ingest` is resumable (the idempotent upsert + the run's ingest cache). A fresh ingest refuses a project that is not empty; after a crash wipe
  that project's dir under the projects dir and restart the server, or simply re-run the same command (a resumed run is exempt).
- `oh-probe` / `oh-enrich` in this layout need `oh-ingest` done; `oh-enrich` without it would ingest inside the run (the dry-run then shows the
  ingest). Outputs: `oh/ingest-run/`, `oh/probe-run/`, `oh/enrich-run/`; the split path writes `oh/probe-run-split/`, `oh/enrich-run-split/`.
- The estimate of `oh-ingest` / `oh-enrich --dry-run` includes the ingest wall-clock (pieces x `extract_s_per_piece`, a guess printed with its
  value) and the server-side extraction (extract chunks x the extractor model's price, charged to the run cap as a flagged estimate).
- Guard: a campaign stamped `2026-10-27.c6` or `2026-10-28.c7` is accepted by these four commands (and, with C.9, `2026-10-28.c8` plus `oh-workspaces` / `oh-agentic`); the code is `2026-10-28.c9`.

### 6.0b Amendment C.7 - the OH result, the mis-specified recall, and the ONE paid command that follows (read before `oh-enrich`)

> **C.8 supersedes the storage layout of this section:** the commands below default to the project layout (6.0c); what is described here
> against the campaign's stored `ls-<id>-<sid>` HARS sessions is `--layout split-session`, DIAGNOSTIC ONLY.

The question-time arms (`nc-chunks`, `nc-graph`) of the finished OH campaign recalled at ask time with a call that sends no keywords, top_k = 8
and goes through our own packer; they are EXPLORATORY now (`REPORT-PROTOCOL.md` 4a.7). The primary treatment is **`nc-enriched`**: facts
recalled at COMPACTION time (query + `ll_keywords` + `hl_keywords`, mode hybrid, the server's default top_k, budget via the server's
`max_chars`), folded into the post-compaction context, no recall at ask time. It runs ONLY against your finished campaign: no re-ingest, no
re-summarising, no re-judging of baseline / nc-note-only. Name the campaign explicitly (`--resume <ID|PATH>`, both commands refuse without it);
its HARS server must be the one it was ingested on (same data dir) and running.

```bash
cd experiments/neurocompact-eval/scripts
CAMP=<the finished oh campaign, e.g. 20261027T...-homerun-home>
uv run python home_run.py oh-probe  --resume $CAMP --dry-run     # estimate, zero spend
uv run python home_run.py oh-probe  --resume $CAMP               # 1) cents, NO answer model, NO judge: where does the anchor get lost?
uv run python home_run.py oh-enrich --resume $CAMP --dry-run     # estimate (no ingest, new rows only), zero spend
uv run python home_run.py oh-enrich --resume $CAMP               # 2) PAID: nc-enriched only, judge only the new rows, merge, metrics
uv run python home_run.py bundle    --resume $CAMP               # send the .tar.gz back
```

- **`oh-probe` must run before `oh-enrich`** (enforced; `--skip-probe` is recorded). It writes `oh/probe-run/REPORT-probe.md`: per answerable
  question, side by side, (A) the old frozen shape + packing v2 at the old cap, (B) the correct shape raw, (C) the correct shape budgeted by the
  server's `max_chars` at the old cap and at 800 / 1536 / 3000 / 6000 tokens, (D) BM25 over the full dropped context at the same caps, plus the
  anchor in the note and in the dropped context (the ceiling). It costs one keyword call per question (cents); the HARS recalls are free. If the
  recoverable ceiling is low or (B)/(C) do not beat (A)/(D), read the report before paying for `oh-enrich`.
- **`oh-enrich`** reuses `oh/run` of that campaign (`--reuse-dir DIR` to point elsewhere): its HARS sessions `ls-<old run id>-<sid>` (HARS status
  gate first: compactions > 0, vectors_indexed > 0, else it stops before spending), the exact note of the old `nc-note-only` arm, the valid
  `baseline` / `nc-note-only` rows and their judge verdicts. Output: `oh/enrich-run/` (`records.jsonl` with the copied rows marked
  `reused_from`, `enrich/<sid>/j3.b1536.json` = queries, keywords, per-query latency / errors / graph_branch, packed block, tokens;
  `recall_raw.jsonl` = raw recall text; `judged.jsonl` = old + new verdicts; `REPORT-oh-enrich.md` + `metrics.json`; `funnel.md`).
  Parity (realized `nc-enriched / baseline` ask-time memory in [0.80, 1.05]) is checked BEFORE the judge is paid; `--accept-parity-fail`
  overrides (recorded). Flags: `--enrich-queries K` (default 8), `--packer server|legacy` (legacy = packing v2, comparison only),
  `--skip-judge`, `--accept-partial`, `--budgets`.
- **Guard rules.** The campaign was made under EVAL-VERSION `2026-10-27.c6`; the code is `2026-10-28.c9`. Every paid command still refuses
  such a campaign EXCEPT `oh-probe` / `oh-enrich` (why that is valid: protocol 4a.7, "Reuse rule"). Caps (`[caps]` run / judge / total), the
  per-HARS lock / lease, the dry-run estimate check and the ledger apply as everywhere; the estimate contains NO ingest and NO baseline / note
  summaries. Typical size for 14 sessions / 140 questions: run well under USD 0.1, judge about 140 x USD 0.0034 = USD 0.48.
- A fresh campaign (`oh`, `beam`, `cc`) now runs `baseline,nc-enriched,nc-note-only` (ingest happens in the run); `--arms` on
  `run_longsession.py` brings `nc-chunks` / `nc-graph` back for exploration.


### 6.0 Amendment C.6 - the OpenHands track is a long-session run (read first)

Owner decision 2026-10-27: no more pi replay of the OH chains. `HR oh` converts `chains.jsonl` to `longsession/1` sessions
(`scripts/chains_to_longsession.py`, deterministic, into `<campaign>/oh/ls-data/`, manifest `.conversion.json`) and runs
`run_longsession.py` (`--use-real-compactions --keep-tail 0 --checkpoints end`, arms baseline / nc-chunks / nc-graph / nc-note-only, B=1536,
the sweep is opt-in). At each of the chain's three boundaries the summariser and HARS `memory_compact` see the dropped span; every question
is asked once at the end. Outputs: `<campaign>/oh/run/` (`records.jsonl`, `compactions.jsonl`, `REPORT-oh.md` + `metrics.json` from
`metrics_v2.py --track longsession --end-only`, `REPORT-oh-chains-family.md` = the descriptive chain-family reading, `funnel.md`).
Restart from scratch with `--new`: old chain-replay campaigns/calibrations are void for this runner (`oh` says so).
**Parity check, not calibration**: `HR calibrate` (also run by `oh` when `calibration.json` is absent) compacts `[oh] calibration_chains`
(default 1; `--check-sessions K`) session(s) like the real run and asks only two fixed probe questions (sid `parity-<chain>`), then requires
all rows valid, CLIP-free and the ask-time memory ratio of nc-chunks AND nc-graph vs baseline in [0.80, 1.05]. It tunes nothing, asks no
decision-set question and therefore does NOT exclude any chain (14 sessions stay in the decision set). If it FAILS, `oh` refuses (before the
main run); see `calibration.json` `stopped_because`. Do not hand-tune anything: send the bundle, or `HR oh --accept-parity-fail` to run
anyway (recorded; the run's own `win_mem_token_ratio` gate will say the same). The old C.4 cap loop exists only as `calibrate --legacy-replay`.

### 6.0a Amendment C.5 - tokens are native

Parity and every budget are now counted in the answering model's NATIVE tokens (`count_tokens`, free; needs the same key; cached
in `~/.cache/nc-tokcount/cache.jsonl`, override `NC_TOKCOUNT_CACHE`). Your bundle showed why: a baseline summary of 959 cl100k
tokens hit `max_tokens=1536` (~1.6x). The calibration of `20261026T172729Z-homerun-home` is VOID: use a NEW campaign (`--new`),
refresh the scripts (EVAL-VERSION `2026-10-27.c6`), run `doctor` (row `tokcount.native` must be OK), then `calibrate`. The cap
defaults to cap(B) = B (the C.6 parity check uses it as is); `[oh] calibration_chains = 3` (default 1) checks more sessions.
`calibration.json` (C.6) shows `per_b.<B>.ratio_nc_chunks / ratio_nc_graph` (the gate's quantity, ask-time memory in native tokens) and the
pooled summary sizes. Offline: `NC_TOKCOUNT=estimate`; `--token-counter tiktoken` accepts a
flagged non-native measurement (not for a decision set on Claude).

### 6a. LEGACY (`--legacy-replay` only): baseline-cap calibration (Amendment C.4) - why the pi `oh` refuses without it

Everything in this section applies to `HR oh --legacy-replay` / `HR calibrate --legacy-replay` / `HR smoke --legacy-replay`
(the pi chain replay of C to C.5), which exist to reproduce old numbers and need pi + Node (`doctor --legacy-replay`).

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

* `oh`: primary = nc-graph vs baseline at B=1536 on 14 sessions (Amendment C.6: the parity check asks only probe questions, nothing is excluded),
  computed by the long-session track: secondary L1-L3 (L4/L5 n/a: no `multi_session_reasoning` in OpenHands; L2/L3 need the sweep), Holm m=5 (C.6-3). The old
  "multihop" items are analysed as `xspan_locator` (descriptive). `oh/run/REPORT-oh.md`, `metrics.json`, `REPORT-oh-chains-family.md` (descriptive S1-S3 reading), `funnel.md`.
  What it cannot tell you: how an agent works WITH its memory while living the span (the pi replay did; C.6 dropped it): this is memory fidelity at
  the end of three compactions, the same for every arm.
* `beam`: the C.2-3 / C.3-2 long-session track: primary nc-graph vs baseline at B=1536 on the 10 seeded BEAM-1M chats, family L1-L5 (L2/L3 n/a without the sweep)
  on the rubric-graded score, genuine multi-hop = `multi_session_reasoning`. Only 34 of 350 items carry a
  deterministic anchor, so anchor-based rules are n/a or low-power by construction; the score legs carry the result.
* `cc`: descriptive only (C.2-3): `orig` (Claude Code's own summary) vs baseline vs nc arms, lag and checkpoint tables.

## 8. If something goes wrong

* `doctor` FAIL on `hars.semantic_probe` or `hars.embedder`: you are on a shim/broken embedder. Fix HARS first.
* A step fails: the message names the log (`<campaign>/logs/<step>.log`). Fix, re-run the same command. A crashed OH/BEAM/CC run is
  resumed (`run_longsession --resume`, finished (session, arm, B) cells are skipped); with `--legacy-replay` a crashed batch dir is
  moved to `*.failed-<UTC>` (kept, never merged) and redone.
* `BUDGET STOP`: raise the cap in `home_run.toml`, re-run.
* Anything that leaves HARS in a weird state: stop the server, move `<HARS_DATA_DIR>` aside, restart, `doctor --deep`,
  re-run (finished steps stay finished; the HARS store of a finished run is only needed for the funnel's graph stage).
* Host OOM: keep `session_concurrency = 1` (`batch_chains = 4` for `--legacy-replay`).

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
