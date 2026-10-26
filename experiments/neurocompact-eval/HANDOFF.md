# HANDOFF — session-memory A/B (chained long-replay, nc-eval)

Written 2026-10-26 by the interim agent. Read with: RUNBOOK-gate.md, REPORT-PROTOCOL.md,
INFRA-NOTES.md, REPORT-forensics-gate212839.md.

## Code locations (where this lives)

- **Home repo:** `hars-longterm-memory`, branch `exp/neurocompact-eval` (from
  `feat/compaction-session-graph`), pushed to GitHub
  (`iuriimedvedev-dev/hars-longterm-memory`) and knotgate gitlab. In-repo path:
  `experiments/neurocompact-eval/`.
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
batches — batch 1 (chains 1-3) valid, batch A (chains 4-7) completed but INVALID, batch B
(chains 8-10) not started.

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
- [x] Parity mechanism (final design): baseline summary hard-trimmed to mem_budget=1536
      (`NC_SUM_SECTION_TRIM=1`, `--baseline-sum-max-tokens 1536`); nc note cap 640, recall cap
      `clamp(1536 - realized_C3_note, 256, 1024)`; gate is the run-mean ratio in [0.80, 1.05]
      written to run_budget.json (per-chain window is a warning only); CLIP FAIL stays
      per-chain; `--require-no-clip` on.

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
      - 2/4 chains valid=false: GFkz8FeB_c0 and qq8Vqbow_c1 (CLIP FAIL, nc unrepaired
        stop=length). Per-chain cap advice is in chain_budget.jsonl.
      -> Diagnose the recall_error=530 first (one memory_recall call by hand against :8307,
      then a 1-chain --dry-run + smoke). Do NOT merge fullB-075703 nc rows into the decision
      set as-is; re-run batch A clean.
- [ ] Batch B — chains 8-10 (gcKDB2Q7_c1, GFkz8FeB_c1, OAITa4mS_c1): NOT STARTED.

### TODO (in order)

1. Diagnose batch A recall_error=530; if the fix changes nc behavior, re-run batch A (4
   chains) with the same command shape as fullB (subset chains file, small batch).
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

3. Merge the three run dirs (see procedure below) — only with a CLEAN batch A.
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
  --out REPORT-chains.md
```

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
- `run_budget.json`: recomputed by `metrics_v2.py` on the merged records — do NOT hand-merge
  the per-batch run_budget.json files (batch 1 has none anyway; it was killed pre-end).

```bash
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
- nc note cap **640** tokens; mem_budget **1536** realized tokens; parity window [0.80, 1.05];
  recall cap clamp(1536 - realized note, 256, 1024), packer v2, PACK_QUERY_BONUS=1.0 (off).
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
  ran effectively note-only. Resolve before trusting any fullB number.
- Judge calibration kappa on the `unsupported` flag alone is 0.624 (score kappa 0.956) —
  mention as a caveat in the final report.

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

## Data is NOT in git (repo is public)

`export/calib_items.jsonl`, `export/chains*.jsonl`, `export/chains-manifest.json` and
`verify_chains.report.md` contain private session content/anchors and are git-ignored
(see `.gitignore` here). They live in `/tmp/opencode/nc-eval` and must be archived to
`/mnt/datasets/ft/artifacts/`. Scripts regenerate them (`make_chains.py`, `verify_chains.py`).
Earlier commits of the old branch `exp/neurocompact-eval` on GitHub contained them; that branch
must be deleted.
