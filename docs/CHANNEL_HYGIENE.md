# Channel hygiene — audit 2026-10-17

Scope: the optional retrieval channels around the core `hybrid_bm25`
(dense `rag.chunks_vdb` + `bm25s`) engine. This records what was changed,
what was verified dead, and what is a deletion candidate for a human to run.

## Summary

| Channel / asset | Verdict | Action taken |
| --- | --- | --- |
| `ripgrep` (`retrieval/ripgrep_channel.py`) | **RANKED, default ON** | Rewired from injection-only to a first-class ranked fusion channel; the broken flattened-path join fixed; default OFF → **ON** and live env → `=1` (product decision, same day). See [`docs/ripgrep-ranked-channel.md`](ripgrep-ranked-channel.md). |
| `flat_dense` (`retrieval/flat_index.py`) | **KEEP, default OFF** | Redundant with the dense channel (proven); deletion deferred (live callers outside this file). |
| Parallel Qdrant (`:6335`) | **DELETION CANDIDATE** | Serving read path disproven; exact removal commands below — **not executed**. |
| `lang_route` (`retrieval/lang_route.py`) | **DORMANT (off)** | Behind `HARS_MEMORY_ROUTING=lang`; default `off`. No change. |
| `query_twin` (`retrieval/query_twin.py`) | **DORMANT (off)** | Only reachable from the `lang` routing branch; default `off`. No change. |
| `dense_sidecar` (`retrieval/dense_sidecar.py`) | **DORMANT (off)** | Only loaded when a routing sidecar dir is configured; not in the live env. |

---

## 1. `ripgrep` — RANKED, default ON (product decision 2026-10-17)

**From the same-day audit:** the code default was ON while the live env set
`=0`. That "fix" (flip the code default to OFF so the two agreed) is
superseded the same day by a product decision — **"rg должен быть каналом
100%"** — because the audit's own premise was wrong in an important way: the
channel could not affect any top-k metric *by construction*, so "no
regression" was evidence the channel was inert, not that it was harmless.

**Two structural defects fixed** (see
[`docs/ripgrep-ranked-channel.md`](ripgrep-ranked-channel.md) for the full
write-up, line references and measurements):

1. **Injection-only.** `apply_ripgrep_gate` only ANNOTATED `ripgrep_score`
   and appended a `ripgrep:<basename>` tail past `top_k`; with `enable_boost`
   defaulting False it could never reorder inside `top_k`.
2. **Broken join.** The pool stores FLATTENED paths
   (`.plans-2026-06-20_x.md`) while `rg` reports `.plans/2026-06-20_x.md`; the
   old `Path(file_path).name` equality matched **0/48** gold docs. A
   component-boundary suffix match (`_matching_ripgrep_basename`) fixes it.

**Now:** `_RIPGREP_CHANNEL_DEFAULT = "1"` and the live env
(`cortex/.mcp.json`) sets `HARS_MEMORY_RIPGREP_CHANNEL=1` +
`HARS_MEMORY_RIPGREP_CHANNEL_MODE=ranked`. `apply_ripgrep_ranked` blends a
min-max-normalized literal-match score into `fused_score` (same convention as
dense/sparse) and re-sorts; the injection tail is preserved as a separate
capability, and `HARS_MEMORY_RIPGREP_CHANNEL_MODE=injection` restores the old
behaviour.

**Latency guardrails:** one `rg` per query, hard 2 s subprocess timeout, fixed
result cap, fail-soft on a missing binary, no per-chunk spawns.

**Tests:** `tests/test_ripgrep_ranked_channel.py` (ranked path, join, cap,
fail-soft, timeout, defaults), plus updated
`tests/test_channel_defaults.py::test_ripgrep_channel_defaults_on` and
`tests/test_bm25_retrieval.py::TestRipgrepFusion::test_channel_enabled_env_default_and_override`.

**A/B:** measured on the 48-query GC-fixed set — see
[`docs/ripgrep-ranked-channel.md`](ripgrep-ranked-channel.md). Injection-only
is byte-identical to the no-ripgrep baseline (structural); ranked is
neutral-or-better on every quality metric.

## 2. `flat_dense` — KEEP, default already OFF

**Redundancy proven.** `retrieval/flat_index.py` builds a numpy matrix from the
**same** `kv_store_text_chunks.json` payload, embedded with the **same**
`rag.embedding_func` in the **same** asymmetric doc/query convention as
LightRAG's `rag.chunks_vdb`. Standalone it reproduces `dense_only`
(recall@10 0.6852 vs 0.685, ndcg@10 0.6096 vs 0.610) and rescues 0/6 simulated
gaps. Its only theoretical benefit is freshness when `kv_store_text_chunks.json`
diverges from `vdb_chunks.json` — never observed on this index.

**Decision:** KEEP with the default OFF (`_FLAT_CHANNEL_DEFAULT = "0"`, already
the case) and the live env (`HARS_MEMORY_FLAT_CHANNEL=0`) agrees. A decision
note was added above `apply_flat_dense_gate` in `fusion.py`. Deletion is
**deferred** because `flat_index.py` still has live callers outside the file —
`hars_memory/mcp_server.py`'s `_get_flat_dense_index` gate and
`hars_memory/corpus/query.py` (standalone corpus query) — plus its own test
module, so removal is a cross-module change.

**Deletion candidate (later):** `flat_index.py`, `apply_flat_dense_gate`,
`flat_channel_enabled`, the `HARS_MEMORY_FLAT_*` env/config, and
`flat_dense_cache_compact_20260903/`.

## 3. Parallel Qdrant (`:6335`) — DELETION CANDIDATE

**Read path disproven.** The servicing query path reads Qdrant only when the
vector backend is Qdrant:

- `server/lightrag_init.py:538` — `_vector_storage = os.environ.get("HARS_MEMORY_VECTOR_STORAGE", "NanoVectorDBStorage")`; the Qdrant branch is entered only `if "qdrant" in _vector_storage.lower()`.
- `mcp_server.py:844-848` — `memory_status` reads `vdb_*.json` for
  `NanoVectorDBStorage` and only imports `qdrant_client` in the
  `elif "qdrant" in HARS_MEMORY_VECTOR_STORAGE.lower()` branch.
- `server/embedder.py::resolve_index_embedding_dim` branches the same way and
  returns the `vdb_*.json` dimension for `NanoVectorDBStorage`.
- The live env (`cortex/.mcp.json`) sets `HARS_MEMORY_VECTOR_STORAGE=NanoVectorDBStorage`
  and does **not** set any `HARS_MEMORY_QDRANT_*` var. No `:6335` reference
  exists on the query path (`grep` over `hars_memory/`, config, and docs: only
  ingestion/ops scripts — `ingest/audit.py`, `ingest/migrate.py`,
  `scripts/qdrant_transplant.py`, `scripts/incremental_refresh.py`,
  `eval/strategy_bench.py` — reach Qdrant, and only when invoked explicitly).

**Staleness (measured 2026-10-17):**

| | Points/vectors |
| --- | ---: |
| Qdrant `lightrag_vdb_chunks` (this host) | **8565** |
| Qdrant `lightrag_vdb_entities` | 29832 |
| Qdrant `lightrag_vdb_relationships` | 39426 |
| Live NanoVectorDB `vdb_chunks.json` | **2813** |

So the Qdrant copy is **3.04× the live store** and read by nobody. Container:
`hars-memory-qdrant` (qdrant/qdrant:v1.18.0), up, `127.0.0.1:6335`,
volume `cortex_memory_qdrant_storage` → **422 MB** on disk.
Defined in `cortex/docker-compose.dev.yml` (`hars-memory-qdrant`,
`restart: unless-stopped`).

**Commands for a human to run — NOT run by this audit:**

```bash
# 1. Stop + remove the container and its volume (422 MB reclaimed).
docker stop hars-memory-qdrant
docker rm   hars-memory-qdrant
docker volume rm cortex_memory_qdrant_storage

# 2. Remove/disable the `hars-memory-qdrant` service block in
#    cortex/docker-compose.dev.yml — otherwise `restart: unless-stopped`
#    (or the next `docker compose up`) resurrects it.
```

Do **not** remove the `code-diver-qdrant` / `code-diver-qdrant-knotgate`
containers — those belong to a different service.

## 4. `lang_route` / `query_twin` — off and dead (confirmed)

- `mcp_server.py:642` — `HARS_MEMORY_ROUTING = os.environ.get("HARS_MEMORY_ROUTING", "off")`.
  The live env does not set it, so routing is `off`.
- `lang_route.detect_lang` is called only inside `if HARS_MEMORY_ROUTING == "lang"`.
- `query_twin.make_twin` is called only inside that same `lang` branch (it also
  synchronously calls `rag.llm_model_func`, the extractor LLM — the reason the
  review flagged it).
- `dense_sidecar` is loaded only when `HARS_MEMORY_ROUTING_SIDECAR_DIR` is set;
  it is not in the live env.

No code change required; both stay dormant behind `HARS_MEMORY_ROUTING=off`.

## 5. A/B delta (48-query, GC-fixed gold set)

The `ab_bench` harness does **not** exercise the ripgrep/flat gates — it only
*reports* `HARS_MEMORY_RIPGREP_CHANNEL` in its config snapshot. Re-run on
`retrieval_queries_current_gcfixed.yaml` (embedder up; query LLM `:8083` down,
so the LLM-free configs were run):

| config | recall@1 | recall@10 | nDCG@10 | MRR | Δ vs `RETRIEVAL_BASELINE_20261017.md` |
| --- | ---: | ---: | ---: | ---: | --- |
| `dense_only` | 0.4792 | 0.7708 | 0.6228 | 0.5752 | **0.0000 all metrics** |
| `hybrid_bm25` | 0.7083 | 0.9792 | 0.8563 | 0.8149 | **0.0000 all metrics** |

The only change in the report is metadata: the config snapshot now prints
`HARS_MEMORY_RIPGREP_CHANNEL = False [default]` (was `True [default]`). Quality
is unchanged, as expected — the ripgrep gate appends only past `top_k` and
`ab_bench` never calls it.
