# Chain-replay execution layer — build + DoD report

**Scope:** the execution layer of the session-memory A/B (3 chained compactions,
questions asked only after C3). Nothing outside `/tmp/opencode/nc-eval` was
modified; no container was created/stopped/restarted; HARS `:8305` was used
read/write only through MCP; no GPU work; at most 1 `pi` process at a time; no
`pkill`; the API key was read from `cortex/.env` into the environment and never
written to disk.

Deliverables:

| file | what it is |
|---|---|
| `run_chainreplay.py` | the chain runner (arms, integrity gates, recall packer, per-qid forks, dry-run). 1299 lines |
| `mk_fixture.py` → `chains.fixture.jsonl` | 2 chains × 3 spans × 3 segments, 2 QA each (structural smoke test) |
| `probe_prompt_schema.py` → `probe-schema/report.json` | the forced-compaction proof for both arms + the note-only fallback |
| `pi-agent/extensions/neurocompact.ts` | schema-matched prompts, `NC_NOTE_ONLY`, `[nc-metrics]`, default caps 1536/512 |
| `run_longreplay.py` | untouched (single-boundary mode still works — verified) |

CLI: `--chains --arms --limit --dry-run --out --agent-model --qa-model [--sum-model --mcp --run-id]`.
Defaults: agent/qa `qwen/qwen3.8-flash`, HARS `http://127.0.0.1:8305/mcp`,
`--out runs/chains`. `--dry-run` = zero API calls, zero HARS writes, zero
settings rewrites, full projection.

---

## 1. Exact env per arm (as printed by `--dry-run`)

Identical everywhere except the four marked fields: same model, same prompt
schema, same reasoning setting, same force machinery, same caps file.

```
[baseline]      ask: no recall, no memory tools (-nt --no-mcp)
  NC_PROVIDER=openrouter            NC_MODEL_ID=qwen/qwen3.8-flash
  NC_PERSIST=0                      NC_NOTE_ONLY=0
  NC_SESSION_ID=cr-<tag>-<chain>-baseline-<run_id>
  NC_MCP_URL=http://127.0.0.1:8305/mcp
  NC_SUM_PROVIDER=openrouter        NC_SUM_MODEL_ID=qwen/qwen3.8-flash
  NC_SUM_MAX_TOKENS=1536            NC_SUM_REASONING_EFFORT=off

[nc]            ask: inject deterministic memory_recall, whole injected block <=1024 tok
  NC_PERSIST=1                      NC_NOTE_ONLY=0
  NC_SESSION_ID=cr-<tag>-<chain>-nc-<run_id>
  NC_SUM_MAX_TOKENS=512             NC_SUM_REASONING_EFFORT=off
  (+ NC_MCP_URL, NC_SUM_PROVIDER/MODEL_ID as above)

[nc-note-only]  ask: forked nc post-C3 dir (cp -a), no recall, no memory tools
  NC_PERSIST=0                      NC_NOTE_ONLY=1
  NC_SESSION_ID=cr-<tag>-<chain>-nc-<run_id>   <-- the NC session id, on purpose
  NC_SUM_MAX_TOKENS=512             NC_SUM_REASONING_EFFORT=off

settings.json during a run: compaction.reserveTokens=2000, keepRecentTokens=200
(both arms equally; restored byte-for-byte afterwards — verified with diff).
NC_FORCE_COMPACT_THRESHOLD is parked at 1e9 except for the single boundary turn.
```

Why `NC_SUM_REASONING_EFFORT=off`: on the wire `reasoning:{effort:"off"}` is a
400 (valid: max|xhigh|high|medium|low|minimal|none); pi maps thinking level
`off` → `reasoning:{effort:"none"}`, verified with a $0.0000052 probe
(`finish=stop`, 0 reasoning tokens). So the name `off` is right, the raw value
would not be.

Why note-only keeps the **nc** session id: pi resolves `--session-id` against
the session *file*; a fresh id silently creates an empty session and the arm
would measure nothing. The runner refuses `--arms nc-note-only` without `nc`
(exit 2) and asserts the note is inside the fork before asking.

## 2. Static validation

- `node --check pi-agent/extensions/neurocompact.ts` → OK.
- `tsc -p tsconfig.check.json` (typescript 5.9.3): **1 error**, the same
  pre-existing `model | undefined` at the `streamSimple(model, …)` call. The
  pristine file (`neurocompact.orig.ts`) has **2**: that one plus
  `Property 'text' does not exist on type 'TextContent | ThinkingContent | ToolCall'`,
  which this pass fixed with an explicit `(c: any)` annotation. No new error.
- `python -m py_compile run_chainreplay.py probe_prompt_schema.py mk_fixture.py` → OK.

## 3. Forced-compaction proof (`probe_prompt_schema.py`, real stack, `:8305`)

One forced compaction per arm on 2 real fixture segments (~3.5k chars), compactor
`qwen/qwen3.8-flash`, `NC_SUM_REASONING_EFFORT=off`:

| arm | stop | summary | cap | 4 headers | persisted | HARS compactions | verdict |
|---|---|---|---|---|---|---|---|
| baseline | `stop` | 4556 ch / **801 tok** | 1536 | all 4 | false | 0 (correct) | PASS |
| nc | `stop` | 2655 ch / **487 tok** | 512 | all 4 | true | 1 | PASS |
| nc-note-only | `reused` | = previous note verbatim | 512 | n/a | false | unchanged (1) | PASS |

Note-only checks: `usage=None` (no model call), `persisted=false`, compaction
entry appended `1 → 2`, reused note byte-identical to the note in the fork.
Re-run after the gate hardening: `PROBE RESULT: PASS` (exit 0).

## 4. End-to-end run (`--limit 1`, real stack, 3 arms)

Command:

```
OPENROUTER_API_KEY=… .venv-tok/bin/python run_chainreplay.py \
  --chains chains.fixture.jsonl --arms baseline,nc,nc-note-only \
  --limit 1 --out runs/chain1-final
```

Two full runs and one extra confirmation run were made; the canonical artifacts
are `runs/chain1-final` (produced by the current code, `truncated` present) and
the earlier `runs/chain1` (same protocol, before the `truncated` field and the
recall-wrapper budget fix). All three arms completed in every run.

| arm | compactions | stop reasons | realized tokens | cap | asks | anchor_hit |
|---|---|---|---|---|---|---|
| baseline | 3 | stop, stop, stop | 659 / 877 / 961 | 1536 | 2 | 2/2 |
| nc | 3 | stop, **length**, **length** | 454 / 462 / 459 | 512 | 2 | 2/2 |
| nc-note-only | 0 (fork, by design) | – | – | – | 2 | 2/2 |

Gates that fired on the successful run:

```
[nc] gate: HARS compactions == 3 ✓                     (memory_session_status)
   gate: baseline HARS compactions == 0 ✓              (readable, not -1)
[nc-note-only] gate: fork carries the nc note ✓ (note 2074ch, fork last compaction 2072ch)
[nc] recall: raw=12436ch (2893tok) -> body=890tok, injected=963tok, chunks 2/5
[nc] recall: raw=16120ch (3712tok) -> body=896tok, injected=969tok, chunks 2/7
    (relations: the :8305 extractor wrote none for these spans — logged, not fatal)
BUDGET baseline span 1: realized summary 659 tok < 800 (warn only)
BUDGET nc span 2/3: note CUT AT THE CAP (stop=length) — see §6.1
```

Wall clock 3m22s for the whole chain (baseline compactions 56 s, nc 96 s,
note-only 0 s); the dry-run's 12.5 min projection for the same chain was
pessimistic because compactions were modelled at the historical 180–205 s.

Cost of the whole exercise: compaction+ask tokens of the canonical run
$0.0051 (18.0k in / 5.1k out at $0.15/$0.47 per 1M), replays ~$0.0015, HARS
glm extractor ~$0.0025, probes ~$0.007, two earlier full/aborted runs ~$0.012 →
**≈ $0.028 total, well under the $0.10 cap.** (Per-turn `usage.cost` from this
provider is all zeros, so cost is computed from tokens, not read.)

`--dry-run --limit 2` → exit 0, "2 chain(s) runnable, 0 skipped, 0 alignment
failures", 12 compactions / 12 asks / ~25 min / $0.0273 projected.
`run_longreplay.py --dry-run` → exit 0, unchanged behaviour.

## 5. Bugs the gates caught (both fixed)

1. **pi no-ops a compaction and still emits `compaction_end`.** When everything
   is already before `keepRecentTokens`, pi appends **no** entry, yet the event
   and `result.summary` may still arrive. A run would have looked like a valid
   lag-3/2/1 chain while nothing was dropped. Fix: `count_compaction_entries()`
   counts `"type":"compaction"` lines, every boundary requires `before+1` (one
   retry at half the threshold, then `ChainFailure`), and an empty summary is a
   hard failure. This is also what the note-only probe needed: a forced
   compaction immediately after a compaction is a no-op, so the probe now
   replays 2 more segments into the fork first.
2. **Relative `--out` split the session files from the records.** pi runs with
   `cwd=workspace/`, so `--out runs/chain1` put the sessions under
   `workspace/runs/chain1/` while the runner wrote records under
   `runs/chain1/`: the arm dir looked empty and the entry gate aborted the chain
   (correctly — after ~45 s and one paid compaction). Fix:
   `args.out = os.path.abspath(args.out)`, plus an immediate
   `ChainFailure` if the first replay turn did not create a session file.
3. **The recall wrapper was outside the budget.** Packing the body to ≤1024 gave
   a 1054-token injection (the ~70-token scaffold: `<memory>` fence + "use only
   if relevant" instruction). nc was therefore getting ~70 tokens more ask-time
   budget than the protocol allows, in an arm where the whole point is an
   equalized budget. Fix: `pack_injection()` budgets the **whole** block
   (body 811/917 tok → injected 883/990 tok), asserts ≤1024, and the record
   stores `memory_tokens_injected` from the real injected text.

## 6. What is wrong or ambiguous in the spec (please decide before the real run)

### 6.1 The 512-token nc note truncates at the cap — 3 of 6 preserved compactions

`stop=length` on 2/3 nc compactions in the canonical run (`runs/chain1-final`)
and 1/3 in the earlier preserved run (`runs/chain1`); one intermediate run hit
3/3 before its artifacts were overwritten by the canonical rerun. So 3 of the 6
compactions whose records survive, and 6 of 9 across all runs, were cut off.
These are not near-misses with a tidy ending — the notes are cut mid-sentence,
and because the fourth section is `ACTIVE STATE`, **the truncated part is the
freshest state**, i.e. exactly what a lag-1 question needs:

```
canonical run, span2 tail: "…avoiding polished retrospectives to save time on re-reading old context.\nIn"
canonical run, span3 tail: "…checklist links (pointing to post"
earlier run,  span1 tail: "…awaiting the opening of the operational window"
```

The model reliably fills ~90–97% of the cap (454–495 tok by `cl100k_base`, 512
by the API tokenizer), so at a 512 cap truncation is close to the expected
outcome, not an accident. Consequences: the nc/note-only arms lose their tail
non-deterministically (1/3, 2/3, 3/3 across otherwise identical runs), which adds variance
exactly at the boundary the experiment is supposed to measure, and it biases nc
against itself. Options, in order of my preference:

1. **Raise the nc/note-only cap to ~700** so the schema-matched note fits
   (`CAPS = {"baseline": 1536, "nc": 700, "nc-note-only": 700}`): one line in
   `run_chainreplay.py` plus the env knob, no extension change. It breaks the
   512+1024 = 1536 identity, so the note (700) + recall (836, hard-packed) would
   have to be rebalanced to keep the ask-time budget equal.
2. **Keep 512 and register truncation as part of the treatment.** The record
   now carries `truncated: true/false` per compaction, so metrics can slice it;
   but with 3/3 in one run the "nc" arm is then partly a "truncated note" arm.
3. **Keep 512 and make the prompt allocate the budget per section** (character
   ceilings per header) so the cut, when it happens, lands on the oldest section
   instead of `ACTIVE STATE`. This is the only option that keeps the numbers as
   specified, but it is an untested prompt change and I did not apply it.

I implemented only the measurement (`truncated` in the record, an explicit
warning that the tail was cut) — changing the treatment silently is not my call.

### 6.2 The "nc > 700 tok" warning is unreachable

With `NC_SUM_MAX_TOKENS=512` enforced by the provider, a note can never exceed
512 API tokens (~500 by `cl100k_base`), so `BUDGET_MAX_NC_TOKENS=700` can never
fire. The only observable cap effect is truncation (§6.1). Either lower the
threshold to ~500 or accept that the warning is dead code for this arm.

### 6.3 Baseline under-fills its cap, so equal caps ≠ equal memory

Baseline realized 659/877/961 tok against a 1536 cap (43–63%), and the first one
tripped the `< 800` warning in every run so far. nc realizes ~460 + ~900–970
injected recall tokens, i.e. ~1.4k. So the *realized* ask-time memory is
baseline ~0.7–1.0k vs nc ~1.4k: the arms are equal in cap, not in bytes on the
table. Worth stating in the paper, and worth deciding whether "equalized budget"
means equal caps (what I implemented) or equal realized tokens (which would
require a feedback loop on baseline).

### 6.4 The fixture is not discriminative — it proves plumbing, not effect

All 18 asks across the three runs hit their gold anchor in all 3 arms, including
baseline at lag 3. The reason is structural: the fixture facts are exactly the
kind of thing the schema's `KEY FIGURES & IDENTIFIERS` section is designed to
preserve, so the baseline summary keeps them and there is no headroom for nc to
win. That
is fine for what this fixture is for (proving that 3 chained compactions, forks,
recall packing and the gates all work end-to-end), but **no effect size can be
read off `chains.fixture.jsonl`**. The real chains from `make_chains.py` need
questions whose anchors are *not* key-identifier-shaped (buried in prose, causal
details, intermediate numbers), otherwise the experiment will produce a
ceiling of 1.0 for every arm. The runner emits
`!!! nc recall … contained 0 gold anchors` for the opposite degeneracy, but
there is no warning for "all arms score 1.0" — that belongs in the metrics layer.

### 6.5 Smaller notes

- `--qa-model` is the model that answers; `--agent-model` replays/compacts.
  No judge is implemented: `anchor_hit` is an exact-substring test on the gold
  anchor and is the only objective metric written. `judge_v2.py` is the other
  agent's file and I did not wire it in.
- The `:8305` extractor returned **0 relations** for every recall on this
  fixture (chunks only). The packer logs `rels=0/0` and continues; if the real
  chains expect graph relations to matter, that extractor config needs a look.
- `keepRecentTokens=200` means the last compaction drops almost everything —
  that is deliberate (otherwise lag 3/2/1 is cosmetic), but it also means each
  boundary's `tokensBefore` (~1.9k→3.1k) is far below a realistic
  production compaction. Both arms are affected equally.
- Each ask gets its own `cp -a` of the post-C3 dir, so question N's injected
  recall can never leak into N+1. Cost: one extra copy per ask (cheap).
- HARS unreachable → `-1` compactions is treated as a hard failure in the chain
  path (`!!!` on screen), not just a warning: a silent `-1` would otherwise pass
  the "baseline 0 compactions" gate.

## 7. How to run the real chains

```
cd /tmp/opencode/nc-eval
.venv-tok/bin/python run_chainreplay.py --chains chains.jsonl --limit 2 --dry-run
OPENROUTER_API_KEY=… .venv-tok/bin/python run_chainreplay.py \
  --chains chains.jsonl --arms baseline,nc,nc-note-only --limit 2 --out runs/real1
```

`records.jsonl` (per ask: answer, anchor_hit, memory_tokens_injected,
recall_packed_tokens, packed_loss_pct) and `compactions.jsonl` (per boundary:
stop, truncated, realized tokens vs cap, persisted HARS count, wall) are the
inputs the metrics layer should read; `logs/<chain>/<arm>-spanN.{stderr,summary.txt}`
keep the raw `[nc-metrics]` line and the note itself as the audit trail.
