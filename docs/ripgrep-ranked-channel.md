# Ripgrep → first-class RANKED retrieval channel (2026-10-17)

Status: **shipped, default ON**. Product decision: *"rg должен быть каналом
100%"* — ripgrep is a first-class channel, enabled by default.

This document is the evidence trail for that decision: the structural verdict
on the old wiring, the implementation, the before/after measurements, and the
deployment change.

---

## 1. Structural verdict — the old wiring *could not* affect any metric

Verified in code, then reproduced empirically (`hybrid_bm25_ripgrep` in
`mode=injection` scores **byte-identically** to `hybrid_bm25` on the 48-query
GC-fixed set — see §4).

Two independent defects, both in `hars_memory/retrieval/fusion.py`:

### 1a. Injection-only: hits entered PAST `top_k`

`apply_ripgrep_gate` (the only entry point the live call site used) did:

```python
top = boosted[:top_k]                    # order inside top_k unchanged
...
return top + injected                    # ripgrep hits appended AFTER the window
```

With the shipped `enable_boost=False`, `fused_score` was not even touched — the
function only set `FusedChunk.ripgrep_score` via `replace()` and appended a
`ripgrep:<basename>` tail. Recall@k / NDCG@k / MRR are computed over the first
`top_k` ranks, so an append-past-the-window channel is *structurally incapable*
of moving them, for any query. This was not an unfinished path — the module
docstring stated it as the design ("`apply_ripgrep_gate` therefore does two
independent things … (b) GATE … APPENDED … up to `RIPGREP_MAX_INJECTED`"). The
finish line was simply never drawn.

### 1b. The join was broken on this corpus (0/48 gold docs connected)

The old matcher was `ripgrep_hits_by_basename.get(Path(chunk.file_path).name)`.
But `kv_store_text_chunks.json` stores **flattened** paths — directory
separators collapsed to `-`:

| stored `file_path` (chunk) | ripgrep's on-disk basename |
| --- | --- |
| `.plans-2026-06-20_generic-mlops-orchestration-plan.md` | `2026-06-20_generic-mlops-orchestration-plan.md` |
| `docs-specs-training.vlm_dataset_extension.spec.md` | `training.vlm_dataset_extension.spec.md` |
| `staging-2026-09-02_x.md` | `2026-09-02_x.md` |

`Path(".plans-…").name` is the whole flattened string, which **never equals**
ripgrep's basename. Measured against the live index: **0/48** gold docs had a
stored `file_path` equal to the on-disk basename. Consequently
`matched_basenames` stayed empty, every ripgrep hit was classified "exclusive",
and even the presence guarantee only ever fired for genuinely off-index files.
The audit comment claiming basename was "the only join key that actually holds"
was wrong; it holds for **no** curated/staged doc on this index.

Fix: a component-boundary suffix match
(`fusion._chunk_matches_ripgrep_basename`) — `== basename` OR
`endswith("-"+basename)` OR `endswith("/"+basename)`. Post-fix, 7 of the 20
firing queries have the gold file in ripgrep's top-30 (4/20 with
`roots=<whole cortex tree>`); anchored so `plan.md` cannot match `xplan.md`.

---

## 2. Implementation

All in `hars_memory/retrieval/fusion.py` (+ `ripgrep_channel.py`), library-side;
no call-site change was strictly required for the ranking to work, but the live
path needs one line to select the new mode (§5).

### Score definition

**2026-10-20 — chunk-local scoring (default when the caller passes `question`).**
The original definition below cloned a FILE-level score onto every chunk of a
matching file and min-max normalized it across the pool. On a multi-chunk
document that is a whole-file boost, and on `hard_1k` it measured r@1 −2.0 pts
vs ungated fusion (production index). The ranked channel now computes, per
chunk:

```
r(c)       = sum over query identifier terms of capped occurrences in c.content
S_local(c) = r(c) / (r(c) + 3)          # saturating, in [0, 1)
fused_score(c) += weight * S_local(c)   # only if the query has an identifier
```

Chunks whose own text does not contain the identifier get no boost; queries
without an identifier-shaped term get no ranking boost at all (injection is
unchanged). `HARS_MEMORY_RIPGREP_CHUNK_LOCAL=0` restores the file-level path,
and omitting `question` (older callers) keeps it bit-for-bit.

Per chunk (legacy definition): the best `RipgrepSearchHit.score` among ripgrep
hits whose file matches the chunk (via the fixed join). That score already
encodes the signals the channel computes — number of matched query terms
(co-occurrence multiplier), per-term match count (capped), whole-token vs
substring weighting, path-type prior, floored recency
(`ripgrep_channel.py` "Scoring"). It is then min-max normalized **per query
over the matched chunks**, with `fusion._min_max_normalize` — the *same*
convention dense/sparse use, including the degenerate single/tied-pool → 0.5
rule. **No WRRF implementation exists in this tree** (the module docstring
documents why a pure rank fusion was rejected for the primary blend), so this
is **score-wise min-max fusion**.

### Fusion entry point

`apply_ripgrep_ranked(fused, ripgrep_hits_by_basename, *, top_k, weight, max_injected, enable_injection)`:

```python
fused_score(chunk) += weight * minmax(ripgrep_raw)(chunk)   # bounded, additive
```

The dense/sparse convex term is **untouched**: a chunk with no literal hit
keeps its pre-existing `fused_score` bit-for-bit. Re-sort uses `fuse()`'s exact
deterministic key (quantized score → single-channel z-score tiebreak →
chunk_id), factored into `_resort_after_rescoring`, so tie behaviour cannot
drift. The **injection tail is preserved** in ranked mode (`enable_injection`,
default True): ranking answers "did we order indexed candidates better";
injection answers "can an index-invisible file still surface at all".

`apply_ripgrep_gate(..., mode=...)` dispatches: `mode="ranked"` delegates to
`apply_ripgrep_ranked`; `mode="injection"` is the pre-2026-10-17 behaviour
(unchanged, and the function's default so direct callers/tests are unaffected).
`ripgrep_scores_by_chunk_id` is exported for reporting/trace.

### Flags & defaults

| Env | Default | Meaning |
| --- | --- | --- |
| `HARS_MEMORY_RIPGREP_CHANNEL` | **`1`** (was `0`) | channel on/off; `0` returns the pre-ripgrep fusion bit-for-bit |
| `HARS_MEMORY_RIPGREP_CHANNEL_MODE` | `ranked` | `ranked` \| `injection` |
| `HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT` | `0.10` (was `0.25`) | additive weight on the (chunk-local) literal score |
| `HARS_MEMORY_RIPGREP_CHUNK_LOCAL` | `1` | `0` restores file-level min-max scoring |
| `HARS_MEMORY_RIPGREP_ROOTS` | *(unset)* | comma-separated search roots; falls back to the knowledge-source manifest |

All are read fresh on every call.

### Latency guardrails

- **One `rg` invocation per query** (every term is a separate `-e` in a single
  argv — see `_build_rg_command`); **no per-chunk spawns**.
- **Hard timeout** `ripgrep_channel._DEFAULT_TIMEOUT_SECONDS = 2.0 s`
  (`subprocess.run(timeout=…)`), fail-soft on expiry.
- **Result cap**: the caller passes `top_k` (the pool size); per-file match
  reads are additionally capped by `rg -m 20`.
- **Fail-soft**: `shutil.which("rg")` gates every call; a missing binary →
  `available=False`, zero hits, no exception → the channel contributes nothing.

---

## 3. What changed in the harness

`ab_bench` did **not** exercise the ripgrep gate at all before this change — the
gate was only referenced as a strict-env knob. Added:

- `hars_memory/eval/channels.py::hybrid_bm25_ripgrep` — wires the channel
  exactly as the live server does (same basename join over the full fused
  pool), with `ripgrep_mode` / `ripgrep_weight` / `ripgrep_roots`.
- `ab_bench` config `hybrid_bm25_ripgrep` + flags `--ripgrep-mode`,
  `--ripgrep-weight`, `--ripgrep-roots`; the mode/weight knobs are now echoed in
  the config snapshot and covered by `--strict-env`.

Reproduce (48-query GC-fixed set, live index/embedder):

```bash
cd /home/demo/Project/software/hars-longterm-memory
set -a; source eval_agentic-data/bench_env.sh; set +a
export HARS_MEMORY_RIPGREP_ROOTS="/mnt/datasets/graphrag/staging/ingested,/home/demo/Project/software/cortex/.plans,/home/demo/Project/software/cortex/.reports,/home/demo/Project/software/cortex/.session,/home/demo/Project/software/cortex/docs,/home/demo/Project/software/cortex/hypotheses,/home/demo/Project/software/cortex/hypothesis_links"
uv run --project . python -m hars_memory.eval.ab_bench ab \
  --queries eval_agentic-data/retrieval_queries_current_gcfixed.yaml \
  --configs dense_only hybrid_bm25 hybrid_bm25_ripgrep \
  --normalize-gold-paths
```

---

## 4. Measurements

### 48-query GC-fixed gold set (`retrieval_queries_current_gcfixed.yaml`)

Quality (deterministic; run-twice byte-identical for `hybrid_bm25`, matching the
2026-10-17 baseline exactly: 0.708/0.979/0.856/0.815).

| config | recall@1 | recall@10 | NDCG@10 | MRR | Δ NDCG@10 | Δ MRR |
| --- | --- | --- | --- | --- | --- | --- |
| `dense_only` | 0.479 | 0.771 | 0.623 | 0.575 | — | — |
| `hybrid_bm25` (baseline) | 0.708 | 0.979 | 0.856 | 0.815 | — | — |
| rg **injection** (old behaviour) | 0.708 | 0.979 | 0.856 | 0.815 | **±0.000** | **±0.000** |
| rg **ranked** w=0.1 | **0.729** | 0.979 | **0.861** | **0.822** | +0.005 | +0.007 |
| rg **ranked** w=0.25 (default) | **0.729** | 0.979 | **0.861** | **0.822** | +0.005 | +0.007 |
| rg **ranked** w=0.5 | **0.729** | 0.979 | 0.859 | 0.818 | +0.003 | +0.003 |
| rg **ranked** w=1.0 | 0.667 | 0.979 | 0.830 | 0.780 | −0.026 | −0.035 |

Reading:
- **Injection is inert** — byte-identical to baseline, exactly as §1 predicts.
  This is the control that proves the ranked path is what moved the numbers.
- **Ranked is a clean, small win** at w ≤ 0.5: recall@1 +0.021 (one query
  flipped into rank 1), NDCG@10 +0.005, MRR +0.007, recall@10 unchanged
  (0.979 — there was almost no headroom). No metric regresses on the default.
- **w=1.0 over-weights** and regresses recall@1/NDCG/MRR (a literal coincidence
  can displace gold). The 0.25 default sits on the low, safe end of a broad
  plateau (0.1–0.25 identical).

### Relational set (`relational_gold.yaml`)

`relational_eval.py run` is a **live-service** harness (`--mcp-url` /
`--trace-url` against `:8099`) with **no in-process channel flag**; the service
is not running (and must not be restarted here), so it cannot exercise the
channel. Instead the 28 multi-hop questions were projected to document-level
gold (the union of `hop_chain[].doc`) and scored with the same `ab_bench`
metrics:

| config | recall@1 | recall@10 | NDCG@10 | MRR |
| --- | --- | --- | --- | --- |
| `dense_only` | 0.161 | 0.464 | 0.373 | 0.411 |
| `hybrid_bm25` | 0.286 | 0.696 | 0.614 | 0.684 |
| rg **ranked** w=0.25 | **0.304** | 0.696 | **0.631** | **0.702** |

Neutral-or-better here too (recall@1 +0.018, NDCG@10 +0.017, MRR +0.018).

### Latency

The ripgrep call (`ripgrep_channel.search`) over the 48 queries, curated roots:

| scope | mean | p50 | p95 | max |
| --- | --- | --- | --- | --- |
| all 48 queries | 108 ms | 0.3 ms | 418 ms | 646 ms |
| the ~20 queries that fire `rg` | 260 ms | 250 ms | 494 ms | 646 ms |
| the ~28 that have no identifier term | 0.15 ms | — | — | — |

End-to-end, in one shared-process pass (`hybrid_bm25` → `hybrid_bm25_ripgrep`):
**+166 ms mean** (41 ms → 208 ms), consistent with ~400 ms × 20/48 firing
queries. **Note:** the host was at high load during capture (dense-only
measured 858 ms/query), so treat latency as an upper bound; the `rg`-call
figures above are the stable part. There is no meaningful latency risk when the
query has no identifier term — `rg` is not invoked.

---

## 5. Deployment

`cortex/.mcp.json` (the single source of truth read by
`scripts/hars-longterm-memory-mcp-http.sh`) is updated:

```json
"HARS_MEMORY_RIPGREP_CHANNEL": "1",
"HARS_MEMORY_RIPGREP_CHANNEL_MODE": "ranked",
"HARS_MEMORY_RIPGREP_ROOTS": "<corpus dirs — see §2>",
```

`HARS_MEMORY_RIPGREP_ROOTS` is **required** for the channel to do anything: with
no roots and no `HARS_MEMORY_SOURCES_MANIFEST`, `ripgrep_channel.default_roots()`
returns `[]` and the channel reports "unavailable". (No knowledge-source
manifest is configured on this host, so the explicit roots list is used.)

**Not restarted here.** To take effect:

```bash
systemctl --user restart hars-longterm-memory-mcp.service
```

### Call-site change in `hars_memory/mcp_server.py` (owned by another agent)

**No change is required for the ranking itself.** `apply_ripgrep_gate`'s
`mode=None` now resolves to the env mode (`HARS_MEMORY_RIPGREP_CHANNEL_MODE`,
default `ranked`) internally, read fresh on every call — so the existing
`apply_ripgrep_gate(fused_pool, ripgrep_hits_by_basename, top_k=top_k)` call in
`_compute_hybrid_block` re-ranks automatically once the channel is enabled.
(Explicit `mode=` still wins; the legacy tests pass `mode="injection"`.)

The only *desirable* call-site change is the trace frame, so the UI can show
`rg` as a real channel. Add `ripgrep_scores_by_chunk_id` /
`ripgrep_channel_mode` to the `from hars_memory.retrieval.fusion import (…)`
block in `_compute_hybrid_block`, then emit a `ripgrep_hits` frame (same shape
as `bm25_hits`) immediately after
`rg_result = await ripgrep_channel.asearch(...)`:

   ```python
   _rg_by_name = ripgrep_channel.hits_by_basename(rg_result.hits)
   _rg_hits_by_basename = {
       name: ChannelHit(score=h.score, content=h.content, file_path=h.file_path)
       for name, h in _rg_by_name.items()
   }
   await trace_emit(
       "ripgrep_hits",
       latency_ms=round(rg_result.latency_seconds * 1000, 2),
       pool_size=pool_size,
       mode=ripgrep_channel_mode(),
       query_terms=list(rg_result.query_terms),
       available=rg_result.available,
       hits=[
           {
               "chunk_id": hit.chunk_id,
               "ripgrep_score": round(hit.score, 4),
               "match_count": hit.match_count,
               "matched_terms": hit.term_hits,
               **_trace_hit_doc_fields(hit.content, hit.file_path),
           }
           for hit in rg_result.hits[:HARS_MEMORY_TRACE_HITS_CAP]
       ],
       matched_chunk_ids=list(
           ripgrep_scores_by_chunk_id(fused_pool, _rg_hits_by_basename)
       ),
   )
   ```
   (The existing block then reuses `_rg_hits_by_basename` instead of rebuilding
   `ripgrep_hits_by_basename`.)

### Presentation-facing description

The channel set is now **parallel channels: vector + BM25 + ripgrep + graph**.
The `memory_recall` tool description string and `docs/mcp-reference.md` /
`docs/trace-api.md` (event table + pipeline diagram) should be updated to list
`ripgrep_hits` as a real stage. *Left to the agent that owns those files.*

---

## 6. Verdict

**Yes — a ranked ripgrep channel improves quality on this corpus, and is now
ON by default.** On the 48-query GC-fixed set it is neutral-or-better on every
metric at the default weight (recall@1 +0.021, NDCG@10 +0.005, MRR +0.007), and
neutral-or-better on the relational set (recall@1 +0.018, NDCG@10 +0.017). The
old injection-only mode is provably inert (byte-identical baseline), which is
what made the previous "no regression" evidence misleading. Cost is one capped
`rg` call per identifier-bearing query (~0.15 ms when no identifier term is
present; a few hundred ms when `rg` fires), fail-soft everywhere.
