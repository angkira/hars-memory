# HANDOFF — session-memory A/B (chained long-replay, nc-eval)

Written 2026-10-26 by the interim agent. Read with: RUNBOOK-gate.md, REPORT-PROTOCOL.md,
INFRA-NOTES.md, REPORT-forensics-gate212839.md.

## Code locations (where this lives)

- **Home repo:** `github.com/angkira/hars-memory` (private), branch
  `exp/neurocompact-eval-clean`. It is a snapshot import of `hars-longterm-memory` at
  `feat/compaction-session-graph` (ffd3bd6) WITHOUT upstream history. Upstream (public):
  `iuriimedvedev-dev/hars-longterm-memory`. In-repo path: `experiments/neurocompact-eval/`.
  The old branch `exp/neurocompact-eval` on the upstream GitHub repo is leaked (see "Data is
  NOT in git") and must not be used.
- **Standalone mirror:** `github.com/angkira/neurocompact-eval`, branch
  `exp/neurocompact-session-memory` (private; contains private session content —
  keep private).
- **Live working copy:** `/tmp/opencode/nc-eval` (untracked originals: `chains.jsonl`,
  `longreplay/`, `runs/`).
- **Do NOT put this into the cortex repo** — it caused a branch collision with a
  concurrent perception agent there on 2026-10-26 (resolved: their branch restored
  to their tip, eval content removed).

## Mission (3 lines)

Test whether HARS externalized graph memory (short note + packed recall) matches or beats an
in-context 1536-token summary on 10 chained-compaction sessions, 3 arms (baseline / nc /
nc-note-only), questions only after C3, blind judge gpt-6-luna, pre-registered WIN/FAIL gates.
Protocol frozen after external review; gates 1-6 done; decisive 10-chain run is IN PROGRESS in
batches — batch 1 (chains 1-3) valid under the OLD runner (see OPEN PROTOCOL ISSUES: not
protocol-identical to new runs), batch A (chains 4-7) completed but INVALID, batch B (chains
8-10) not started. The runner/metrics were fixed after a code review (commits touching
`run_chainreplay.py`, `metrics_v2.py`, `export_judge_input.py`, `judge_v2.py`; plan
`.plans/2026-10-26_nc-eval-harness-fixes.md`); several protocol questions are still open
(section "OPEN PROTOCOL ISSUES").

## CURRENT STATE

### DONE

- [x] Protocol per external review (argon_answer_r1.md), frozen in REPORT-PROTOCOL.md
      (3 arms, C=3 chained compactions, ask-only-after-C3, budget parity B=1536, blind judge,
      chain-cluster bootstrap, kill/blame funnel).
- [x] 10-chain x 70-item dataset built and independently re-validated, all gates 1.000
      (dataset_stats.md, verify_chains.report.md). QA generator z-ai/glm-5.3-flash (different
      family from agent), 60 answerable + 10 unanswerable.
- [x] Judge calibrated on gate6-013036: agreement 0.976 (>=0.90), kappa 0.956 (>=0.80),
      P(unsupported AND score=2)=0.024 (<=0.05) — PASS
      (runs/gate6-013036/judge-calibration.json). Caveat: kappa on the unsupported flag alone
      is 0.624 — note in final report.
- [x] Gate history (one line each):
      - gate1 `runs/gate-212839` (baseline cap 1575): baseline C3 clipped (CLIP FAIL) + chain2
        ratio 1.261 -> Amendment A; forensic pack found the rank-prefix packer and abstention
        wave (REPORT-forensics-gate212839.md).
      - gate2 `runs/gate2-220255` (cap 2240): run ratio 0.714 FAIL; same-chain baseline fill
        1183 vs 2047 across runs -> ceiling-only parity is a lottery.
      - gate3 `runs/gate3-230010` (cap 1536 + section trim): run ratio 1.163 FAIL (baseline
        under-fill 0.688; per-chain 1.438).
      - gate4 `runs/gate4-233211`: ratio 0.989 PASS; chain_y062eGZR_c0 CLIP FAIL (baseline
        stop=length).
      - gate5 `runs/gate5-001652`: ratio 0.991 PASS; chain_y062eGZR_c0 CLIP FAIL again (nc,
        unrepaired).
      - gate6 `runs/gate6-005402` (1 chain) + `runs/gate6-013036` (2 chains): ratio 0.991 PASS
        both; gate6-013036 is the calibrated-judge run with metrics.json clean.
      - (also in runs/: gate7-011604, 2 chains aborted by the mid-replay compaction guard
        "force hook must be parked" — runner guard, pre-full-run; not a gate verdict.)
- [x] Parity mechanism (final design, as enforced by the CURRENT runner): baseline summary
      hard-trimmed to mem_budget=1536 (`NC_SUM_SECTION_TRIM=1`, `--baseline-sum-max-tokens
      1536`); nc note cap 640; recall cap `clamp(mem_budget - realized nc C3 note, 256, 1024)`
      (`recall_cap_for`; the baseline's realized C3 is only REPORTED, never enters the cap);
      gate is the run-mean ratio in [0.80, 1.05] (runner: `RUN BUDGET` + run_budget.json;
      metrics_v2: `win_mem_token_ratio` = run-mean nc `memory_tokens` / run-mean baseline
      `memory_tokens` over the kept records). The per-chain window is a warning only
      (`[budget warn]`); a chain is invalidated only by structural defects: CLIP FAIL (>1/6
      unrepaired `stop=length` per arm, `--require-no-clip` on), nc note alone busting the
      budget (recall cap clamped low), RECALL FAIL (any empty/errored nc recall), baseline
      writing to HARS, or a chain abort. Rows are buffered per chain and written once at chain
      end with the final `valid` flag. All arms share one ask prompt incl. the abstention
      sentence (`ASK_PROMPT`). Persistence gate: nc `compactions == 3`, `entities > 0`,
      `vectors_indexed > 0` when reported. Runner preflight: one `memory_recall` before any
      spend, abort on a dead channel. Runner refuses a non-empty `--out`.

### IN FLIGHT (as found by this agent)

- [x] Batch 1 `runs/full-20261026T021613` — chains 1-3 (qq8Vqbow_c0, y062eGZR_c0, gcKDB2Q7_c0)
      all COMPLETE and valid=true, per-chain ratios 0.987 / 0.992 / 0.982, 0 recall errors,
      anchor_in_recall on 9/21 nc asks. Killed by HOST OOM at 04:17 (full-run.log mtime
      04:17:41; no run_budget.json — killed before run end, recomputed at merge).
- [x] Batch A `runs/fullB-075703` — chains 4-7 (GFkz8FeB_c0, OAITa4mS_c0, qq8Vqbow_c1,
      y062eGZR_c1). `pgrep -af run_chainreplay` matches only the pgrep wrapper itself — the
      process is NOT running; the batch COMPLETED (last chain_budget ts 10:00:05). It is
      INVALID for decision purposes:
      - recall_error=530 and recall_raw_chars=0 on ALL 28/28 nc asks (the HARS recall channel
        returned an error payload; nc got 108 injected tokens = note+empty wrapper only;
        anchor_in_recall 0/28). Batch 1 and gate6-013036 had 0 recall errors, raw recall
        12.3-16.6 kchars — the channel broke between 04:17 and 07:57.
      - RUN BUDGET FAIL: run ratio 0.579 (window [0.80, 1.05]); baseline fill dropped to
        0.71-0.80 (was 0.92-0.94 in batch 1).
      - Under the CURRENT runner this batch would be stopped by the preflight call or marked
        valid=false by RECALL FAIL; it ran under the old runner, which marked it valid.
      - 2/4 chains valid=false: GFkz8FeB_c0 and qq8Vqbow_c1 (CLIP FAIL, nc unrepaired
        stop=length). Per-chain cap advice is in chain_budget.jsonl.
      -> Diagnose the recall_error=530 first (cause is UNDIAGNOSED, see OPEN PROTOCOL ISSUES
      M5; one memory_recall call by hand against :8307, then a 1-chain --dry-run + smoke).
      Do NOT merge fullB-075703 nc rows into the decision set as-is; re-run batch A clean.
- [ ] Batch B — chains 8-10 (gcKDB2Q7_c1, GFkz8FeB_c1, OAITa4mS_c1): NOT STARTED.

### TODO (in order)

0. Owner decision on the OPEN PROTOCOL ISSUES below (B4 above all) BEFORE spending on more
   batches: B4 decides whether a decisive verdict is reachable at all.
1. Diagnose batch A recall_error=530 (M5 checklist); if the fix changes nc behavior, re-run
   batch A (4 chains) with the same command shape as fullB (subset chains file, small batch).
   Batch 1 ran under the old prompt/cap: re-run chains 1-3 too, or document the mix (M7).
2. Run batch B (chains 8-10), after `--dry-run`:

```bash
cd /tmp/opencode/nc-eval
export OPENROUTER_API_KEY=$(grep -m1 '^OPENROUTER_API_KEY=' /home/angkira/Project/software/cortex/.env | cut -d= -f2- | tr -d '"'"'"'[:space:]')
python3 -c "import json;ws=['chain_gcKDB2Q7_c1','chain_GFkz8FeB_c1','chain_OAITa4mS_c1'];open('chains-batchB.jsonl','w').write(''.join(l for l in open('chains.jsonl') if json.loads(l)['chain_id'] in ws))"
python3 run_chainreplay.py --dry-run --chains chains-batchB.jsonl --arms baseline,nc,nc-note-only \
  --agent-model qwen/qwen3.8-flash --mem-budget 1536 --require-no-clip --mcp http://127.0.0.1:8307/mcp
python3 run_chainreplay.py --chains chains-batchB.jsonl --arms baseline,nc,nc-note-only \
  --out runs/fullC-$(date -u +%Y%m%dT%H%M%S) \
  --agent-model qwen/qwen3.8-flash \
  --mem-budget 1536 --require-no-clip --mcp http://127.0.0.1:8307/mcp
```

3. Merge the three run dirs (see procedure below) — only with a CLEAN batch A. The merge
   MUST yield exactly one row per (qid, arm) and per rid: export_judge_input.py and
   metrics_v2.py exit 2 on duplicates (do not concatenate the invalid fullB-075703 rows with
   their re-run).
4. Export -> judge -> metrics (RUNBOOK-gate.md full-run block, with `--judge-input` added on
   metrics_v2 so clean_anchor_hit / primed_hit are computed; gate6-013036 also passed
   `--json`):

```bash
cd /tmp/opencode/nc-eval
export OPENROUTER_API_KEY=$(grep -m1 '^OPENROUTER_API_KEY=' /home/angkira/Project/software/cortex/.env | cut -d= -f2- | tr -d '"'"'"'[:space:]')
python3 export_judge_input.py --records "runs/<MERGED>/records.jsonl" --chains chains.jsonl \
  --out "runs/<MERGED>/judge-input.jsonl"
python3 judge_v2.py --in "runs/<MERGED>/judge-input.jsonl" \
  --model openai/gpt-6-luna --out "runs/<MERGED>/judged.jsonl" \
  --mapping "runs/<MERGED>/judge-mapping.json"
python3 metrics_v2.py --records runs/<MERGED>/records.jsonl \
  --judge runs/<MERGED>/judged.jsonl --judge-input runs/<MERGED>/judge-input.jsonl \
  --out REPORT-chains.md          # optional: --json, --boot N (10000), --seed S (1337), --cluster session|chain
```

   What metrics_v2 does with the merged records (code, not intent): drops whole chains with
   any row whose `valid` is not `true` or that has a `recall_error`; drops qids lacking the
   full baseline/nc/nc-note-only triplet; exits 2 on duplicate (qid, arm) or rid. Bootstrap
   clusters by SESSION (chain_id minus `_cN`), and the power check `MIN_CHAINS = 8` counts
   DISTINCT SESSIONS (fewer => verdict forced to INCONCLUSIVE, rule tables are diagnostics).
   Anchor gates use `clean_anchor_hit` when it is defined on every answerable record with an
   anchor verdict (needs `--judge-input`); otherwise they fall back to raw `anchor_hit` with
   a warning. A gate input that is missing is `n/a`: a WIN rule that is n/a cannot be met, an
   n/a FAIL rule blocks WIN => INCONCLUSIVE, never FAIL. Check the `[validity]` lines on stderr
   for which chains/qids were dropped. The judge masks arm-revealing strings (`-nc-` /
   `-baseline-` session-id parts, `[memory recall ...]`/`<memory>`/"Recalled session memory"
   headers, "recalled memory", HARS) in the judge-visible answer only and logs it in the
   mapping file (`answer_masked`, `masking`).

5. Kill/blame funnel + BM25 comparison (offline, $0): adapt `funnel_gate212839.py` /
   `funnel_clean_gate212839.py` to the merged dir; stages anchor_in_summary / anchor_in_graph
   / anchor_in_recall + BM25/FTS5 over the same dropped_context. Kill graph specifically if
   HARS anchor_in_recall <= BM25 + 5 pp (RUNBOOK-gate.md "Decision rules").
6. Archive to `/mnt/datasets/ft/artifacts/<ts>-neurocompact-chains/`:

```bash
TS=$(date -u +%Y%m%dT%H%M%S); D=/mnt/datasets/ft/artifacts/$TS-neurocompact-chains; mkdir -p "$D"
cp chains.jsonl chains-manifest.json dataset_stats.md verify_chains.report.md \
   REPORT-PROTOCOL.md REPORT-chains.md "$D"/
cp -a runs/full-20261026T021613 runs/<MERGED> "$D"/        # judged outputs + metrics inside
cp -a runs/<MERGED>/sessions "$D"/sessions                 # HARS graphs
```

7. Final REPORT numbers into REPORT-PROTOCOL.md section 7 + REPORT-chains.md: per-arm score
   mean, hallucination rate, clean_anchor_hit, unanswerable false-answer rate, median ask
   latency, realized memory tokens + parity ratio, persisted-compaction rate, per-lag (3/2/1)
   slices, cluster-bootstrap CIs, WIN/FAIL/INCONCLUSIVE verdict per the verbatim thresholds in
   RUNBOOK-gate.md, funnel table. Never report INCONCLUSIVE/UNDERPOWERED as a directional win.

## Merge procedure (three run dirs)

- Dirs: batch 1 = `runs/full-20261026T021613` (chains 1-3), batch A = the CLEAN re-run of
  `runs/fullB-075703` (chains 4-7), batch B = `runs/fullC-*` (chains 8-10).
- `records.jsonl`, `compactions.jsonl`, `chain_budget.jsonl`: plain concatenation in CHAIN
  order (chains.jsonl order = qq8Vqbow_c0, y062eGZR_c0, gcKDB2Q7_c0, GFkz8FeB_c0, OAITa4mS_c0,
  qq8Vqbow_c1, y062eGZR_c1, gcKDB2Q7_c1, GFkz8FeB_c1, OAITa4mS_c1). Chain validity flags come
  from `chain_budget.jsonl` (`valid`, `invalid_reasons`); records also carry `valid` +
  `chain_invalid_reasons`.
- `sessions/`: merge by relative path (`rsync -a D/sessions/ MERGED/sessions/` per dir).
- `run_budget.json`: NOT recomputed by `metrics_v2.py` (it only computes the run-mean
  `memory_tokens` ratio gate from the kept records and prints it; it never reads or writes
  run_budget.json). Do NOT hand-merge the per-batch run_budget.json files (batch 1 has none;
  it was killed pre-end, and the runner computes it over all chains incl. invalid ones).
  Use `scripts/merge_runs.py` (below); it recomputes run_budget.json and refuses duplicate (qid,arm),
  invalid chains, recall_error rows and mixed configs. The manual shell fallback follows it.

```bash
# preferred:
python3 scripts/merge_runs.py --out runs/merged-$(date -u +%Y%m%dT%H%M%S) \
  --chains chains.jsonl runs/<D1> runs/<D2> runs/<D3>
# funnel (step 5): python3 scripts/funnel_chains.py --run runs/<MERGED> --chains chains.jsonl \
#   --graph-store /tmp/opencode/nc-eval/hars-data-chain-glm/sessions
# manual fallback:
cd /tmp/opencode/nc-eval/runs
M=merged-$(date -u +%Y%m%dT%H%M%S); mkdir -p "$M"
D1=full-20261026T021613; D2=<CLEAN-BATCH-A-DIR>; D3=fullC-<TS>
for f in records.jsonl compactions.jsonl chain_budget.jsonl; do
  cat "$D1/$f" "$D2/$f" "$D3/$f" > "$M/$f"
done
mkdir -p "$M/sessions"
rsync -a "$D1/sessions/" "$M/sessions/"; rsync -a "$D2/sessions/" "$M/sessions/"; rsync -a "$D3/sessions/" "$M/sessions/"
echo "$M"   # use as <MERGED> below
```

## Environment facts

- HARS memory service: `nc-hars-chain-glm` on **:8307**, glm extractor, 1 worker / queue 32 /
  600 s timeout. DEDICATED to this experiment; must be up for any run. Check:

```bash
curl -sf -m 5 http://127.0.0.1:8307/healthz >/dev/null && echo ':8307 UP' || echo ':8307 DOWN'
```

  As found: UP, status ok, engine 428/428 submitted/completed, 0 failed, 0 overloads,
  run_ms max 53.7 s. Data dir `/tmp/opencode/nc-eval/hars-data-chain-glm` (only persistent
  state; recreate only via hars-api.sh, see INFRA-NOTES.md).
- Agent/compaction/question-answer model: `qwen/qwen3.8-flash` (runner header `qa=` is the
  ANSWER model, defaults to agent model; the dataset QA GENERATOR was `z-ai/glm-5.3-flash`,
  chains-manifest.json — different concepts, do not confuse).
- nc note cap **640** tokens; baseline hard trim **1536**; mem_budget **1536** realized tokens;
  parity window [0.80, 1.05] on the run mean; recall cap clamp(1536 - realized nc C3 note,
  256, 1024) (baseline realized C3 is only reported), packer v2, PACK_QUERY_BONUS=1.0 (off).
- Judge: `openai/gpt-6-luna`, blind (shuffled order + logged mapping), budget flag default
  $0.30/run.
- OpenRouter key: exported from `OPENROUTER_API_KEY` in `cortex/.env` (see the export
  one-liner above). Never read .env for anything else, never quote the key value.
- Balance tracking (run before/after spends):

```bash
curl -s -m 15 https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $OPENROUTER_API_KEY"
# balance = data.total_credits - data.usage
```

## Gotchas

- Host OOM killed the first full run (chains 1-3 done, died 04:17). Batch small (<=4 chains),
  serialize nc compactions; a full 10-chain run is ~2 h.
- :8305/:8080 trap: :8305 `nc-hars-ch-small` has NO extractor env -> silently falls back to
  repo-default `localhost:8080` (dead 405-nginx) and returns `ok=true` compactions with EMPTY
  graphs. Only ever point `--mcp` at `http://127.0.0.1:8307/mcp`.
- Datasets and run dirs contain private session content plus `sk-or-v1-REDACTED` markers: the
  repo must stay private; never re-import unredacted data into public places; never quote any
  `sk-or-v1` string — say "the OpenRouter key in cortex/.env".
- Free-model answers are sometimes empty or `<tool_call>`/`<invoke>` markup leaked through the
  adapter (seen gate3, all arms). The judge handles it, but note it in the report limitations.
- HARS retrieval is nondeterministic across repeated `memory_recall` calls (5/12 payloads
  differed on refetch); run-time records.jsonl is authoritative for anchor_in_recall — do not
  substitute offline refetches.
- Always `python3 run_chainreplay.py --dry-run ...` before any real run.
- Never `pkill -f` (it would kill this session too); use recorded pids (`kill.pids`).
- nc compactions ~60 s each on the glm extractor (healthz run_ms max 53.7 s, queue 600 s
  timeout); total run ~2 h — batch it and let it run.
- Batch A recall_error=530 on 28/28 nc asks (recall_raw_chars=0): the nc arm in fullB-075703
  ran effectively note-only. Resolve before trusting any fullB number (cause undiagnosed, M5).
- `scripts/run_pool.py` is now committed but is only a minimal re-implementation of `extract()`
  (pi 1.4.0 `--mode json` stdout parser) -- all `run_chainreplay.py` needs. `usage.input` =
  last assistant turn's input + cacheRead + cacheWrite (context size); `usage.output` = output
  summed over turns; `compactions` = count of `compaction_end`. The rest of the owner's
  original (`run_pi`, `judge`, `hars_recall`, `hars_compactions`) is still only in the live copy
  `/tmp/opencode/nc-eval`, so `run_longreplay.py` and `judge_backfill.py` still need that one.
- Judge calibration kappa on the `unsupported` flag alone is 0.624 (score kappa 0.956) —
  mention as a caveat in the final report.

## OPEN PROTOCOL ISSUES (owner decision needed)

Found by the code review; NOT fixed by the harness fixes (no dataset rebuild, no runs).
Until decided, treat any full-run number as exploratory.

- **B4 — only 5 independent sessions.** The 10 chains are 5 sessions x (c0, c1); c0 and c1 of
  a session share 2 of 3 spans, and 7/10 unanswerable probes use the same probe
  (`NC_PHANTOM_TTL_MS`). metrics_v2 clusters by session and requires `MIN_CHAINS = 8` distinct
  sessions, so with this dataset the power check can never pass: the verdict is forced to
  INCONCLUSIVE (per metrics_v2 code; rule tables printed as diagnostics, a FAIL rule that
  fires is not promoted to a FAIL verdict). Why it matters: the pre-registered WIN/FAIL gate
  is unreachable until the dataset is rebuilt (more independent sessions, distinct probes) or
  the protocol is amended (owner decision; amend BEFORE looking at results).
- **M2 — nc may be chunk-RAG, not graph.** The runner calls `memory_recall` with
  `session_id`, `question`, `top_k=8` and no `ll_keywords`/`hl_keywords`, so the retrieval mode
  resolves to naive and the graph branch (`HARS_MEMORY_GRAPH_BRANCH=auto`) is probably
  skipped (unverified on a live server). Why it matters: the claim under test is graph
  memory; if recall is chunk-only, nc vs baseline does not test the graph, and the funnel's
  `anchor_in_graph` / "kill graph" rule are meaningless. Check the recall payload sections
  (relations/entities present?) before interpreting.
- **M3 — mid-span anchors unreachable for HARS.** `neurocompact.ts` serializes the dropped
  context, cuts it to `RAW_CAP = 60000` chars, and sends only the first `DROP_HALF = 12000` and
  last 12000 chars to `memory_compact`; the baseline summarizer sees up to 60k. Anchors in
  the middle of a long dropped span are never ingested by HARS. Why it matters: it
  structurally disadvantages nc relative to baseline on the items the memory is for (share of
  anchors outside the head+tail window: unmeasured).
- **M5 — batch A recall_error=530 undiagnosed.** Suspects: embedding server :8791 down after
  the 04:17 host OOM, with no lexical fallback; the container restarted onto other code.
  Checks: container restart count / `StartedAt` of `nc-hars-chain-glm`; `:8791` health;
  `vdb_chunks.json` size in the batch-A session dirs; grep logs for `embedding failed`. Why it
  matters: batch A must be re-run, and a repeat would silently turn nc into note-only (the
  new preflight + RECALL FAIL gate catch it, but only the root cause prevents it).
- **M7 — tuning chains are in the decision set.** Packer v2 and cap tuning used chains
  `qq8Vqbow_c0` and `y062eGZR_c0`, which are decision chains. Report results with and without
  them. Also: batch 1 (`runs/full-20261026T021613`) ran under the old ask prompt / cap logic,
  so it is not protocol-identical to new runs — re-run chains 1-3 or document the mix.
  Judge calibration (agreement 0.976, kappa 0.956) was two temp-0 passes of ONE model: it
  measures repeatability, not accuracy against humans or another judge.

## Cost ledger so far

| Item | Amount | Source |
|---|---|---|
| QA dataset generation (10 chains, 70 items) | ~$0.07-0.30 | RUNBOOK-gate.md |
| gate-212839 judge run (42 records) | $0.0064 (cap $0.20) | REPORT-forensics-gate212839.md |
| Pre-freeze gate ceiling (gates 1-6, spent under) | $0.60 | RUNBOOK-gate.md |
| Key balance | $6.5162 (~$6.52) as of gate6 | `runs/full-spend.txt` (`balance_before=6.5162`, written 2026-10-26T03:16) |
| Ceiling for the remainder (full run + judging) | **$2.50 hard** | RUNBOOK-gate.md |

Note: RUNBOOK-gate.md still says "key balance ~$4.0" — stale; the recorded balance at full-run
launch was $6.5162. Judge spend per run is capped by `judge_v2.py --budget-usd` (default 0.30).

## Data is NOT in git (public-repo leak incident)

`export/calib_items.jsonl`, `export/chains*.jsonl`, `export/chains-manifest.json` and
`verify_chains.report.md` contain private session content/anchors and are git-ignored
(see `.gitignore` here; `runs/` and `sessions/` too). They live in `/tmp/opencode/nc-eval`
and must be archived to `/mnt/datasets/ft/artifacts/`. Scripts regenerate them
(`make_chains.py`, `verify_chains.py`).
Leak incident: earlier commits of the old branch `exp/neurocompact-eval` on the PUBLIC upstream
GitHub repo (`iuriimedvedev-dev/hars-longterm-memory`) contained them; that branch must be
deleted (status of the deletion: unverified). The current home repo
(`github.com/angkira/hars-memory`, private, branch `exp/neurocompact-eval-clean`) is a snapshot
import without upstream history and carries none of these files. Do not push this branch or any
data file to the upstream repo.
