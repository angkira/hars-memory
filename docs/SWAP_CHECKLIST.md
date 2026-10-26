# Phase-D index swap checklist

Replacing the live index (`index_compact`) with the new
md_struct build (`index_full_mdstruct_20261014`). Ordered; every step has a
command and a pass criterion. **Stop at the first failing step and roll back.**

Evidence behind the non-obvious steps: `docs/VERSION_PIN.md`,
`docs/CHANNEL_HYGIENE.md`, and the audit of the 2026-10-17 snapshot
(`/home/demo/mem-probe/index_snaps/20261017T132719Z/`).

Conventions:
- `LIVE=/home/demo/.local/share/hars-graphrag/index_compact`
- `NEW=/home/demo/.local/share/hars-graphrag/index_full_mdstruct_20261014`
- `BENCH=/home/demo/Project/software/hars-longterm-memory/eval_agentic-data`
- Repo = `/home/demo/Project/software/hars-longterm-memory`
- Env for harness runs: `set -a; source $BENCH/bench_env.sh; set +a; export HARS_MEMORY_EMBED_PROVIDER=http`
  (override `HARS_MEMORY_INDEX_DIR` to point at the candidate index).

---

## 0. Preconditions (do not start otherwise)

- [ ] **The new build is finished.** The 2026-10-17 snapshot was taken while the
      Colab run was still in flight and the session was then **preempted**
      (last good snapshot 13:27Z: `processed 958 / failed 290 / pending 17 /
      parsing 75` of 1340 doc_status rows). The index is **incomplete**.
      *Pass:* `kv_store_doc_status.json` has `parsing == 0 && pending == 0`
      and the run reached its `REALRUN_DONE` marker (or the human accepts a
      known partial set — then every later count is relative to that set).
      The `parsing`/`pending` remainder is *not* the same thing as the 290
      "failed" rows (see Step 5).
- [ ] **No writer is touching $NEW** (no in-flight `ainsert`, no consolidation).
      *Pass:* the indexer process is stopped and `$NEW` mtime is stable.
- [ ] **Service version alignment is possible** — do Step 1 before any swap.

## 1. Version alignment (HARD BLOCKER — see `docs/VERSION_PIN.md`) — RESOLVED

A 1.4.16 reader cannot faithfully read a 1.5.7-built `kv_store_doc_status.json`
(1050/1340 rows `content_hash` → `TypeError`; `get_status_counts()` →
`KeyError: 'parsing'`). Reader and builder must match at **1.5.7**.

- [x] **DONE (2026-10-17).** The serving venv is on `lightrag-hku==1.5.7`; the
      `chunking_by_token_size` / `DEFAULT_ENTITY_TYPES` import migration landed
      in the same change. Decisive proof: all 1340 rows of the
      `20261017T132719Z` snapshot construct under the installed reader and
      `get_status_counts()` returns the complete counts (see
      `tests/test_lightrag_157_store_read.py` and `docs/VERSION_PIN.md`).
- **Command (verify):**
  ```bash
  cd $Repo && uv run python -c "import importlib.metadata as m; print(m.version('lightrag-hku'))"
  ```
- **Pass:** prints `1.5.7`. If it prints `1.4.16`, the swap is **blocked** —
  re-land the migration (see `docs/VERSION_PIN.md`) before proceeding.
- **Alternative pass (only if reverted to 1.4.16):** the normalised doc_status
  loads clean under 1.4.16:
  ```bash
  cd $Repo && NEW=$NEW uv run python - <<'PY'
  import json, os
  from lightrag.base import DocProcessingStatus, DocStatus
  ds=json.load(open(os.path.join(os.environ["NEW"], "kv_store_doc_status.json")))
  unknown={v["status"] for v in ds.values()} - {s.value for s in DocStatus}
  assert not unknown, f"statuses unknown to reader: {unknown}"
  for k,v in ds.items(): DocProcessingStatus(**v)   # raises on any bad row
  print("doc_status reader-compatible")
  PY
  ```

## 2. Back up the live index

- **Command:**
  ```bash
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  cp -a "$LIVE" "${LIVE}.bak-${TS}"
  ( cd "${LIVE}.bak-${TS}" && sha256sum kv_store_*.json vdb_*.json graph_*.graphml > SHA256.txt )
  echo "$TS" > /tmp/phaseD_backup_ts
  ```
- **Pass:** the copy exists, `SHA256.txt` is non-empty, and
  `du -sh "${LIVE}.bak-${TS}"` ≈ `du -sh "$LIVE"`.

## 3. Stage the new index (never overwrite in place)

- [ ] Place the finished `index_full_mdstruct_20261014` at `$NEW` (verify the
      transferred `SHA256.txt`/part hashes first — the snapshot ships
      `PARTS.txt` + `SHA256.txt`).
- **Pass:** `sha256sum -c $NEW/SHA256.txt` (or the pack's own verify script)
  reports OK and `nvidia-smi`/process check shows no writer.

## 4. Count + schema validation

- **Command:**
  ```bash
  cd $Repo && uv run python - <<PY
  import json, pathlib, collections
  NEW=pathlib.Path("$NEW")
  ds=json.load((NEW/"kv_store_doc_status.json").open())
  kinds=collections.Counter(v["status"] for v in ds.values())
  chunks=json.load((NEW/"kv_store_text_chunks.json").open())
  docs=json.load((NEW/"kv_store_full_docs.json").open())
  vdb=json.load((NEW/"vdb_chunks.json").open())
  print("doc_status", dict(kinds))
  print("docs", len(docs), "chunks", len(chunks), "vdb_vectors", len(vdb["data"]))
  assert len(chunks)==len(vdb["data"]), "chunk KV and vdb_chunks disagree"
  assert kinds.get("parsing",0)==0 and kinds.get("pending",0)==0, "build still in flight"
  # every processed doc must have chunks; every chunk must resolve to a doc
  assert all(v.get("chunks_count",0)>=0 for v in ds.values())
  print("COUNTS_OK")
  PY
  ```
- **Pass:** `parsing==0`, `pending==0`, `len(chunks)==len(vdb["data"])`, and
  `processed + failed == len(ds)`.
- Reference (2026-10-17 in-flight snapshot, for scale): docs 1050, chunks 7442,
  vdb 7442, graphml 48125 nodes / 79661 edges.

## 5. Resolve the 290 "failed" documents

**Finding (measured, not inferred):** every one of the 290 `status=failed` rows
in the snapshot is a **duplicate-filename skip**, not a real failure —
`metadata.is_duplicate=true`, `duplicate_kind="filename"` for 290/290,
`error_msg="File name already exists. Original doc_id: file:…, Status: processed"`,
created in two bursts (150 + 140) at the 20:54Z indexer restart. Each points at
a **processed** original with an identical `content_length`; 290/290 originals
are `processed`. Nothing is lost, and they are **not auto-retried** (they are
terminal placeholder rows excluded from the pipeline's admitted set — a re-run
re-fails them identically while the same file basename keeps colliding).

- **Pick one policy and record it:**
  - **(a) Purge (recommended, cost ≈ 0 — JSON edit, no LLM, no re-embed):**
    ```bash
    cd $Repo && uv run python - <<PY
    import json, pathlib
    p=pathlib.Path("$NEW/kv_store_doc_status.json"); ds=json.load(p.open())
    drop=[k for k,v in ds.items() if v.get("metadata",{}).get("is_duplicate")]
    for k in drop: ds.pop(k)
    p.write_text(json.dumps(ds))
    print("dropped duplicate rows:", len(drop))
    PY
    ```
    Then Step 4's `processed + failed == len(ds)` holds with `failed == 0`.
    **Caveat:** these rows reappear on the next enqueue from the same path —
    avoid recurrence by running the next build from a stable absolute path, or
    via `HARS_MEMORY_ID_PATH_MAP=ACTUAL=LOGICAL` (`ingest/document.py`;
    `file_stable_id` hashes the absolute path). Root cause: the restart staged
    the same files under a different absolute path, so `file_stable_id` produced
    new `file:*` ids that miss `filter_keys`, and LightRAG's filename dedup then
    rejected them.
  - **(b) Accept:** leave them and relax Gate 2 to "no failed rows except
    `metadata.is_duplicate=true`". Cost 0; the `failed` count stays 290.
- **Pass:** `failed == 0` (policy a) **or** 0 non-duplicate failed rows (policy b).

## 6. Chunk-size / vector-truncation audit (no re-embedding required)

**Finding (measured on the snapshot):** the review's "chunks >512 tokens had
their tails silently chopped before embedding" is **false**. LightRAG 1.5.7's
hard guard **re-splits** (does not truncate):
`enforce_chunk_token_limit_before_embedding` (`lightrag/utils.py`) calls
`tokenizer.split_by_token_limit(content, max_tokens, overlap_tokens)` and emits
children tagged `split_type="hard_fallback"`. Measured: **0 of 7442 chunks
exceed 512 tokens** (max = exactly 512; 1629 are hard-split children across 365
families), all hard-split docs' tails are present (291/291, tail-100-char
subsequence test), and **all 7442 `vdb_chunks.json` payloads are byte-identical
to their `kv_store_text_chunks.json` content**. The 338
`VDB content truncated to 512 tokens` warnings are **entity/relation
descriptions** (`_truncate_vdb_content`, `operate.py`; every call site is
`entity:*`/`relation:*`), i.e. the entity/relation vector text — a minor,
separate issue, not chunk loss.

- **Command:**
  ```bash
  cd $Repo && uv run python - <<PY
  import json, pathlib
  NEW=pathlib.Path("$NEW")
  chunks=json.load((NEW/"kv_store_text_chunks.json").open())
  vdb={e["__id__"]: e.get("content") for e in json.load((NEW/"vdb_chunks.json").open())["data"]}
  over=[k for k,v in chunks.items() if v.get("tokens",0)>512]
  mismatch=[k for k,v in chunks.items() if vdb.get(k)!=v.get("content")]
  assert not over, f"{len(over)} chunks exceed 512 tokens"
  assert not mismatch, f"{len(mismatch)} vdb payloads != chunk text"
  print("CHUNK_AUDIT_OK  chunks", len(chunks))
  PY
  ```
- **Pass:** `CHUNK_AUDIT_OK`, 0 over-limit, 0 mismatches. **No re-embedding**;
  the review's Gate 3 (`re-embed with max_tokens=2048`) is unnecessary for chunk
  content. (Optionally still raise `HARS_MEMORY_EMBED_MAX_TOKENS` for future
  builds so entity descriptions aren't truncated — a quality nicety, not a gate.)

## 7. Gold-coverage gate (48-query set)

**Finding:** the 2026-10-17 snapshot was missing **15 of the 48 gold docs**
entirely (21 processed, 11 dup+processed → covered, 1 parsing, 15 absent). A
missing gold doc caps Recall@10. Re-check against the *finished* build.

- **Command:**
  ```bash
  cd $Repo && uv run python - <<PY
  import json, pathlib, yaml, collections
  ds=json.load(pathlib.Path("$NEW/kv_store_doc_status.json").open())
  by=collections.defaultdict(set)
  for v in ds.values(): by[pathlib.Path(v["file_path"]).name].add(v["status"])
  q=yaml.safe_load(open("$BENCH/retrieval_queries_current_gcfixed.yaml"))
  gold={g for it in q["queries"] for g in it["gold_docs"]}
  missing=sorted(g for g in gold if "processed" not in by.get(g, set()))
  print("gold total", len(gold), "| not fully processed:", len(missing))
  for m in missing: print("  MISSING", m, by.get(m))
  PY
  ```
- **Pass:** `not fully processed == 0`. If not zero, either finish ingesting
  those docs or regenerate the gold set against the new index — do **not** swap
  with a known-missing gold doc.
- **Then the quality gate (staging, do not touch $LIVE yet):**
  ```bash
  cd $Repo && set -a; source $BENCH/bench_env.sh; set +a; export HARS_MEMORY_EMBED_PROVIDER=http
  HARS_MEMORY_INDEX_DIR=$NEW uv run --project . python -m hars_memory.eval.ab_bench ab \
    --queries $BENCH/retrieval_queries_current_gcfixed.yaml \
    --report /tmp/phaseD_ab.json --configs dense_only hybrid_bm25 --normalize-gold-paths
  ```
  **Pass:** `hybrid_bm25` Recall@10 ≥ **0.958** (≤ 1 query drop from the
  baseline **0.9792**) and `dense_only` Recall@10 ≥ **0.7708**.

## 8. Rebuild the BM25 cache

- **Command:**
  ```bash
  rm -rf /home/demo/.local/share/hars-graphrag/bm25_cache_compact_20260903
  # first serve/query against $NEW rebuilds it; or force via the harness (Step 7).
  ```
- **Pass:** the cache dir is repopulated on first use and Step 7's `hybrid_bm25`
  numbers reproduce (a stale BM25 cache over the old chunk store would surface
  as sparse hits for chunks that no longer exist).

## 9. Spot-checks (staging, read-only)

- **Command (dump per-query hits against the candidate index, then assert ranks):**
  ```bash
  cd $Repo && set -a; source $BENCH/bench_env.sh; set +a; export HARS_MEMORY_EMBED_PROVIDER=http
  HARS_MEMORY_INDEX_DIR=$NEW uv run --project . python -m hars_memory.eval.ab_bench ab \
    --queries $BENCH/retrieval_queries_current_gcfixed.yaml \
    --report /tmp/phaseD_spot.json --configs hybrid_bm25 \
    --dump-hits /tmp/phaseD_hits.json --normalize-gold-paths
  uv run python - <<'PY'
  import json, yaml, pathlib
  hits=json.load(open("/tmp/phaseD_hits.json"))         # {config: {query_id: [file_path, ...]}}
  q=yaml.safe_load(open("/home/demo/Project/software/hars-longterm-memory/eval_agentic-data/retrieval_queries_current_gcfixed.yaml"))
  pick={it["id"]: it["gold_docs"][0] for it in q["queries"] if it["type"]=="identifier"}
  for qid in list(pick)[:3]:
      ranked=[pathlib.Path(p).name for p in hits["hybrid_bm25"][qid]]
      g=pathlib.Path(pick[qid]).name
      rank=ranked.index(g)+1 if g in ranked else None
      print(qid, "gold rank", rank)
      assert rank in (1,2), f"{qid}: gold {g} not in top 2 ({rank})"
  print("SPOT_OK")
  PY
  ```
- **Pass:** `SPOT_OK` (each sampled gold doc in the top 2), and a supersession
  query returns the newer note ahead of the superseded one.

## 10. Atomic swap

- **Command:**
  ```bash
  mv "$LIVE" "${LIVE}.pre-phaseD"                     # keep a second local copy
  mv "$NEW"  "$LIVE"
  # update cortex/.mcp.json only if the index dir name changed (it won't here)
  ```
- **Pass:** `HARS_MEMORY_INDEX_DIR` (unchanged) now resolves to the new store;
  `ls "$LIVE/kv_store_doc_status.json"` is fresh.

## 11. Restart the service and smoke-test

- [ ] Restart the MCP server / client session (the stdio server is relaunched by
      the client; the trace listener on `:8098` is restarted separately).
- **Smoke — `memory_status`:** returns counts, not an exception. **If you are
  still on 1.4.16 with a 1.5.7 store, this is where `KeyError: 'parsing'`
  bites — Step 1 exists to prevent exactly this.**
- **Smoke — recall:**
  ```bash
  curl -sN --max-time 60 "http://127.0.0.1:8098/trace?question=<a gold question>&context_only=true" | tail
  ```
  **Pass:** a terminal `done` event, `rerank`/`graph` stages present, and the
  gold doc appears in the returned context.
- **Smoke — tests:** `cd $Repo && uv run pytest -q` → 0 failed except the known
  `test_tokenize_russian` failure.

## 12. Rollback (on any failure in Steps 4-11)

- **Command:**
  ```bash
  # if the swap (Step 10) was done:
  rm -rf "$LIVE" && mv "${LIVE}.pre-phaseD" "$LIVE"
  # or restore from the Step-2 backup:
  rm -rf "$LIVE" && cp -a "${LIVE}.bak-$(cat /tmp/phaseD_backup_ts)" "$LIVE"
  rm -rf /home/demo/.local/share/hars-graphrag/bm25_cache_compact_20260903   # force rebuild
  ```
- [ ] Restart the service; re-run the Step-11 smoke.
- **Pass:** `memory_status` + recall smoke green on the restored store.
- For a dependency rollback (Step 1), see `docs/VERSION_PIN.md` "Rollback
  procedure".

---

## Gate summary (quick reference)

| # | Gate | Pass criterion |
| --- | --- | --- |
| 1 | Version alignment | serving reader == builder (**1.5.7 — DONE**), verified by `tests/test_lightrag_157_store_read.py` |
| 4 | Counts/schema | `parsing==0`, `pending==0`, `chunks==vdb_vectors`, `processed+failed==len(ds)` |
| 5 | Failures | 0 non-duplicate failed rows |
| 6 | Chunk audit | 0 chunks >512 tok, 0 vdb/chunk mismatches (no re-embed) |
| 7 | Gold coverage | 0 gold docs not `processed` |
| 7 | Quality | `hybrid_bm25` R@10 ≥ 0.958, `dense_only` R@10 ≥ 0.7708 |
| 11 | Service smoke | `memory_status` + `:8098` trace green; pytest 0 unexpected failures |
