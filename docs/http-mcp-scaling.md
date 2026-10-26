# Scaling the native HTTP MCP server

The native Streamable HTTP endpoint (`docs/http-mcp.md`) was a **single process
that served one recall at a time**. This document describes why, what was fixed,
how to deploy and size it for concurrent users, how to measure it, and what
still cannot scale.

Everything is opt-in at the environment level: with no new variables set, the
server behaves exactly as before (`HARS_MEMORY_MCP_MODE=combined`, Nano vector
store, JSON OAuth state).

**TL;DR for a shared service**

1. The concurrency ceiling is **not Python and not the thread pool** — it is the
   **embedding endpoint** (slots × latency) plus the **per-request CPU** of
   retrieval (ripgrep scan, BM25, the fused rerank).
2. Two real bugs were found and fixed: every recall embedded its question
   **twice**, and every embed opened a **new HTTP connection** (no keep-alive).
3. Size the embedder explicitly: `-np` ≈ expected concurrent recalls (2–8),
   `-t` ≤ half the cores, and **`-b/--ubatch-size 2048`** (3x lower latency per
   embed than the default `-b 512`). Give it dedicated cores; it competes with
   the retrieval threads otherwise.
4. For more than one engine, share the vectors through Qdrant and give each
   tenant its **own prefixed collections** (§5).

---

## 1. Concurrency model

```
            ┌──────────────────── one process (combined) ────────────────────┐
 MCP clients│  uvicorn event loop: TLS, OAuth (Ed25519 verify), MCP framing   │
   ────────►│  tools/list (static, no index)     tools/call ─┐                │
            │                                               ▼                │
            │            ┌────────────── bounded thread pool ──────────────┐  │
            │            │  numpy/Qdrant vector query   fuse + LTR ranker  │  │
            │            │  ripgrep ranked gate         diversity + build  │  │
            │            └──────────────────────────────────────────────────┘│
            │  serial on the loop: aquery graph branch, response rendering,  │
            │  LightRAG bookkeeping                                          │
            └────────────────────────────────────────────────────────────────┘
                    │                                        │
                    ▼                                        ▼
        embedder (HTTP, shared)                     Qdrant / Nano / BM25
```

* **One `Server`, one index per process.** `StreamableHTTPSessionManager` is
  stateless, so every request reuses the same `mcp_server.app` and therefore the
  same project registry, graph and BM25 caches.
* **Blocking work is offloaded** to `HARS_MEMORY_MCP_ENGINE_THREADS`
  (=`min(8, cpus)`, `0` = disabled) via `hars_memory/mcp_runtime.py`. The pool is
  bounded in both directions: threads *and* queue.
* **Backpressure instead of unbounded latency**: past
  `HARS_MEMORY_MCP_ENGINE_QUEUE` waiting calls (`4 × threads` default) or
  `HARS_MEMORY_MCP_ENGINE_QUEUE_TIMEOUT_MS` (10 s), a recall fails fast with a
  retryable `503 OverloadedError`; the `overloads` counter is on `/healthz`.
* **Observability**: `/healthz` reports the pool (`submitted`, `inflight`,
  `max_inflight`, `queue_wait_ms`, `run_ms`) and a **loop-lag monitor**
  (`loop.p50_ms/p95_ms/max_ms/over_50ms`) — the direct measure of "is the event
  loop free". `engine.max_ms ≈ 28 s` right after startup is the index warm-up.
* **Per-recall stage timings are in the response** (`hybrid.latency_ms`):
  `dense_embed`, `dense_backend`, `sparse_channel`, `ripgrep_channel`,
  `fusion_rank`, `select`, `assemble`. They are measured server-side, so a slow
  client cannot distort them; `scripts/mcp_stage_bench.py` aggregates them.

### What is parallel, what is still serial

| stage | before | now |
|---|---|---|
| HTTP/TLS/OAuth verify | loop | loop (cheap; N edge workers in split mode) |
| Nano/Qdrant vector query | **loop (blocked)** | thread pool |
| BM25 scoring | thread (default executor) | thread pool |
| fuse + LightGBM LTR + graph bridge | **loop (blocked)** | thread pool |
| ripgrep ranked gate | **loop (blocked)** | thread pool |
| diversity + response assembly | **loop (blocked)** | thread pool |
| `LightRAG.aquery` graph branch | **loop (blocked)** | loop, unchanged (only with keywords) |
| embedder call | HTTP | HTTP — shared external service, the ceiling |
| index load / warm-up | 1× per process, blocking | 1× per process, blocking (as before) |
| across processes | 1 | N engines × N edge workers (§4) |

---

## 2. The embedder is the ceiling

Every recall embeds the question once (`chunks_vdb.query` → `embedding_func`).
That call goes to `HARS_MEMORY_EMBED_BASE_URL`, normally a shared CPU
`llama-server`. Its **slot count** (`-np`) is a hard limit on concurrent
embeddings, and its **per-call latency** (`-b/--ubatch-size`) sets the floor of
every recall.

### Measured slot/thread sweep

`llama-server --embedding --pooling mean -m embeddinggemma-300M-Q8_0.gguf …`,
same question, persistent client per thread, 8 requests per level
(`bench/embedder-sizing.json`):

| flags | n=1 p50 | n=8 p50 | n=16 p50 | throughput @16 |
|---|---|---|---|---|
| `-t 4 -np 2 -c 2048 -b 512` (was production) | 76.5 ms | 150.8 ms | 251.1 ms | **33/s** |
| `-t 4 -np 2 -c 4096 -b 2048` (**recommended**) | 26.8 ms | 109.9 ms | 194.6 ms | **46/s** |
| `-t 8 -np 4 -c 4096 -b 1024` | 68.4 ms | 295.4 ms | 381.7 ms | 22/s |
| `-t 8 -np 8 -c 8192 -b 1024` | 35.4 ms | 155.4 ms | 216.3 ms | 48/s |
| `-t 16 -np 4 -c 8192 -b 2048` | **3055 ms** | — | — | 1/s |

Three things to take away:

1. **`-b 512 → 2048` is nearly free throughput**: 3x lower single-call latency
   and 1.4x throughput on the same cores. It is the single cheapest change.
2. **More slots are not automatically faster.** `-np 4..8` buys throughput only
   when the batch size and thread count are sane; `-np 8` with `-t 8` matched
   `-np 2` at 16 clients but had worse single-client latency.
3. **Never set `-t` near the core count** with multiple slots (the `-t 16` row
   collapses to 1 req/s). Rule of thumb: `-t ≤ cpus / 2`, `-np ≈ expected
   concurrent recalls` (cap 8), `-b = -ub = 2048`, and pin the embedder to
   dedicated cores (`taskset`/cgroup) so retrieval threads cannot starve it.

### Required sizing for N concurrent clients

The embedder must deliver ≈ `N / (recall_latency_target − non_embed_work)`
embeddings per second. Practical table (measured, non-embed work ≈ 60–120 ms
with ripgrep on):

| concurrent clients | embedder config | embed throughput needed | measured |
|---|---|---|---|
| 1–2 | `-np 2 -t 4 -b 2048` | ~8/s | 37–45/s |
| 4 | `-np 2 -t 4 -b 2048` | ~16/s | 41/s |
| 8 | `-np 2..4 -t 4..8 -b 2048` + dedicated 4 cores | ~30/s | 45–48/s |
| 16 | `-np 4..8` **on a separate host/VM** | ~60/s | 48/s (one box, shared) |

Batching: the service already sends whole input lists in one request
(`HARS_MEMORY_EMBED_BATCH_SIZE`, capped by
`HARS_MEMORY_EMBED_HTTP_MAX_INPUTS_PER_REQUEST`), which helps **ingestion**, not
recall — a recall embeds exactly one question. Recall concurrency is therefore a
function of **slots**, not batch size, unless you co-locate several questions
per request (not done: it would couple unrelated users' latency).

Client-side pool: `HARS_MEMORY_EMBED_HTTP_MAX_CONNECTIONS` (default 32) bounds
the keep-alive pool per process. Before this was fixed, **every embed opened a
new client/connection**; at 8 concurrent recalls that alone made the embed stage
~5x slower than the embedder's own capacity.

---

## 3. What was fixed (and the evidence)

| # | defect | effect | evidence |
|---|---|---|---|
| 1 | `_vector_query` detected the backend by `hasattr(vdb, "final_namespace")`; lightrag's `BaseVectorStorage` sets that attribute on **every** storage, so Nano took the Qdrant branch, raised on `effective_workspace`, and the fallback called `storage.query()` — **re-embedding the question** | 2x embedder load per recall, and the vector query stayed on the event loop (no offload at all) | `AttributeError: 'NanoVectorDBStorage' object has no attribute 'effective_workspace'` in the server log; fixed by class dispatch; pinned by `test_embeds_exactly_once_per_query` |
| 2 | the HTTP embedder created `httpx.AsyncClient` **per call** | TCP handshake per recall, no keep-alive, 5x worse embed latency at 8 concurrent | single-call 48 ms vs 8-concurrent 195 ms wall after pooling; `test_embedder_http_mode` still green |
| 3 | no per-stage visibility, so "slow" was unattributable | — | `hybrid.latency_ms` now splits embed / backend / bm25 / ripgrep / fusion / select / assemble |
| 4 | channel fan-out (BM25 ∥ dense ∥ ripgrep) | **reverted**: tripled the instantaneous runnable set; at 16 clients load average >40, p50 3x worse. Per-request CPU, not dependency chains, is the constraint | `bench/stages-fanout-*.json` vs `bench/stages-fixed-*` |
| 5 | Qdrant collection prefix ignored by lightrag 1.5.7 | all tenants landed in the same collections; counts could not be checked per index | §5 |

### Per-stage breakdown under load (Nano, 2-slot embedder, `-b 512`)

`scripts/mcp_stage_bench.py --mode server --concurrency 1,4,8,16`
(`bench/stages-nano-np2.json`):

| clients | total p50 | embed | numpy | BM25 | ripgrep | fusion+rank | transport/JSON |
|---|---|---|---|---|---|---|---|
| 1 | 220 ms | 52 ms | 17 ms | 1 ms | 77 ms | 26 ms | 16 ms |
| 4 | 448 ms | 191 ms | 49 ms | 4 ms | 132 ms | 22 ms | 29 ms |
| 8 | 565 ms | 241 ms | 56 ms | 14 ms | 142 ms | 29 ms | 66 ms |
| 16 | 1457 ms | 854 ms | 124 ms | 23 ms | 178 ms | 49 ms | 173 ms |

Reading: the embed term grows linearly with clients (2 slots), ripgrep and
numpy grow with CPU contention, and the Python fusion/rank stage stays flat and
small. **The embedder and per-request CPU — not the GIL-bound Python — set the
curve.** Turning the ripgrep channel off raised c=8 throughput from 6.1 to 14.1
recalls/s in the same window, which is the clearest evidence that per-request
CPU (the `rg` scan over `HARS_MEMORY_RIPGREP_ROOTS`) is the second limit:

| config | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|
| ripgrep on | 238 ms / 4.0/s | 714 ms / 5.5/s | 1236 ms / 6.1/s | 2615 ms / 5.7/s |
| ripgrep off | 133 ms / 7.8/s | 264 ms / 14.7/s | 520 ms / 14.1/s | 1206 ms / 12.4/s |

`HARS_MEMORY_RIPGREP_ROOTS` should therefore list only the trees that matter for
the corpus; the default in this deployment lists seven, including large plan and
report trees.

> **Host-load caveat.** All end-to-end numbers in this section were taken on a
> 16-core box that was simultaneously running a GPU-training job and other
> services (load average 15–31, ~8.5 cores already busy before any load test).
> Ratios measured *within one window* are meaningful; absolute latencies are
> not, and the 8-concurrent target (≥4x) was not reachable on this host — the
> embedder alone needs more slots than the loaded box can feed. On a quiet
> 16-core host, the arithmetic above (embed 45/s, ~60–120 ms non-embed work)
> gives ~25–35 recalls/s at 8 clients, i.e. 4–5x a 190 ms serial recall.

---

## 4. Process scaling (Qdrant shared, graph/BM25 per process)

`HARS_MEMORY_MCP_MODE=split`: N edge workers own TLS/OAuth/MCP, N engine
processes own the index. Vectors live in Qdrant, so engines are cheap-ish; the
graph (NetworkX, 91 089 nodes / 143 211 edges) and BM25 stay **per process**.

Measured (`bench/procscale-engines*.json`, same noisy window, c=8):

| engines | c=8 p50 | c=8 throughput | c=16 throughput | RSS per engine |
|---|---|---|---|---|
| 1 | 952 ms | 7.9/s | 6.5/s | 1.66 GB |
| 2 | 1137 ms | 3.4/s | 8.3/s | 1.66 GB |
| 4 | 680 ms | 11.2/s | 8.2/s | 1.66 GB |

Four engines is the best c=8 point; at c=16 every shape converges to ~8/s
because they queue on the same embedder. The edge itself is negligible
(**70 MB RSS**) and uses ~0 CPU: it only verifies JWTs and forwards.

Per-worker memory with Qdrant: **1.66 GB** (graph + KV stores + Python/torch +
BM25) versus **2.74 GB** with in-process NanoVectorDB vectors — i.e. Qdrant
saves ~1.1 GB per engine, and removes the 30–60 ms in-process matmul from every
recall (dense backend 100 ms → 34 ms in the same test).

**Sharding.** `HARS_MEMORY_MCP_ENGINE_SOCKETS` is either a round-robin *pool*
(`/run/mem-a.sock,/run/mem-b.sock`, identical index) or *project affinity*
(`projX=/run/mem-x.sock`). One engine per project is the natural shard, since
the registry already keeps one LightRAG per project — at ~1.66 GB per shard.

---

## 5. Qdrant tenants, prefixes and verification

lightrag 1.5.7 names collections `lightrag_vdb_{namespace}` and **ignores**
`HARS_MEMORY_QDRANT_COLLECTION_PREFIX`. With several indexes in one container
this makes plain collection counts meaningless (they sum every tenant) and puts
all tenants in one shared HNSW/dimension configuration.

`hars_memory/server/qdrant_prefix.py` fixes that with an opt-in storage subclass:
when `HARS_MEMORY_QDRANT_COLLECTION_PREFIX` is set, collections become
`{prefix}_lightrag_vdb_{chunks,entities,relationships}` — exactly what
`qdrant_transplant.py --collection-prefix` writes and what
`qdrant_collection_names()` reports. Unset ⇒ stock behaviour and stock names.

```bash
export HARS_MEMORY_QDRANT_COLLECTION_PREFIX=hars_memory_mdstruct
export HARS_MEMORY_QDRANT_COLLECTION=hars_memory_mdstruct_20261021T081415Z  # workspace id
```

### Migrating and verifying

```bash
# 1. what would move (no writes)
.venv/bin/python -m hars_memory.scripts.qdrant_transplant --dry-run \
    --collection-prefix "$HARS_MEMORY_QDRANT_COLLECTION_PREFIX" \
    --workspace "$HARS_MEMORY_QDRANT_COLLECTION" \
    --qdrant-url "$HARS_MEMORY_QDRANT_URL"

# 2. copy all three namespaces (creates missing collections) + verify 200 rows each
.venv/bin/python -m hars_memory.scripts.qdrant_transplant \
    --collection-prefix "$HARS_MEMORY_QDRANT_COLLECTION_PREFIX" \
    --workspace "$HARS_MEMORY_QDRANT_COLLECTION" \
    --qdrant-url "$HARS_MEMORY_QDRANT_URL" \
    --batch-size 500 --verify-samples 200

# 3. re-verify at any time (no copy)
.venv/bin/python -m hars_memory.scripts.qdrant_transplant --verify-only \
    --collection-prefix "$HARS_MEMORY_QDRANT_COLLECTION_PREFIX" \
    --workspace "$HARS_MEMORY_QDRANT_COLLECTION" \
    --qdrant-url "$HARS_MEMORY_QDRANT_URL" --verify-samples 200
```

Verification is three checks per namespace, and the process exits non-zero if
any fails: **count** (Qdrant vs `vdb_*.json`), **vector** (cosine of sampled rows
≥ 0.9999) and **retrieval** (a sampled record's own vector must return that
record as top-1; exact duplicate vectors in the source are accepted on numerical
identity).

Result for the production index (2026-10-21):

| namespace | index records | Qdrant (own collection) | cosine min | top-1 self-hit |
|---|---|---|---|---|
| chunks | 15 409 | 15 409 | 1.000000 | 100 % |
| entities | 90 892 | 90 892 | 1.000000 | 100 % |
| relationships | 143 211 | 143 211 | 1.000000 | 100 % |

### What "contaminated" counts looked like, and why

The shared `lightrag_vdb_*` collections reported 23 974 / 120 724 / 182 637
points, which does not match the index. Per-workspace counts explained it
exactly — a second, older tenant (`hars_longterm_memory`, points created in July)
shares those collections:

| collection | total | ours (`hars_memory_mdstruct_20261021T081415Z`) | other tenant |
|---|---|---|---|
| chunks | 23 974 | 15 409 | 8 565 |
| entities | 120 724 | 90 892 | 29 832 |
| relationships | 182 637 | 143 211 | 39 426 |

Cleanup performed: this index was re-copied into its own prefixed collections
(verified above) and **only** our workspace's points were deleted from the shared
collections (`bench/qdrant-cleanup.json`), leaving the other tenant intact:

| collection | before (total / ours) | after (total / ours) |
|---|---|---|
| lightrag_vdb_chunks | 23 974 / 15 409 | 8 565 / 0 |
| lightrag_vdb_entities | 120 724 / 90 892 | 29 832 / 0 |
| lightrag_vdb_relationships | 182 637 / 143 211 | 39 426 / 0 |

To check a deployment: compare `GET /collections/<name>` with
`count(filter=workspace_id == <yours>)` and with the source `vdb_*.json`
record counts. If the totals disagree but the workspace count matches, the
container is shared, not corrupt.

---

## 6. Deployment shapes

| shape | env | when |
|---|---|---|
| combined (default) | `HARS_MEMORY_MCP_MODE=combined` | one team, one box, ≤2 concurrent users or a fast embedder |
| combined + sized embedder | `+ HARS_MEMORY_EMBED_BASE_URL` at `-np 4 -t 4 -b 2048` | 4–8 concurrent users |
| split | `HARS_MEMORY_MCP_MODE=split` + engine socket(s) | ≥8 concurrent users, TLS/OAuth at the edge, restart-free scaling of the edge |
| split + Qdrant pool | `HARS_MEMORY_MCP_ENGINE_SOCKETS=a.sock,b.sock` | 8–16 users; 2–4 engines; one shared Qdrant |
| split + shards | `HARS_MEMORY_MCP_ENGINE_SOCKETS="projA=a.sock,default=b.sock"` | per-project isolation |

```bash
# engine (owns the index; the only writer)
HARS_MEMORY_MCP_MODE=engine \
HARS_MEMORY_MCP_ENGINE_SOCKET=/run/user/1000/hars/mem-engine.sock \
HARS_MEMORY_VECTOR_STORAGE=QdrantVectorDBStorage \
HARS_MEMORY_QDRANT_URL=http://127.0.0.1:6335 \
HARS_MEMORY_QDRANT_COLLECTION=hars_memory_mdstruct_20261021T081415Z \
HARS_MEMORY_QDRANT_COLLECTION_PREFIX=hars_memory_mdstruct \
HARS_MEMORY_EMBED_BASE_URL=http://127.0.0.1:8802/v1 \
HARS_MEMORY_MCP_HTTP_WARMUP=1 \
hars-longterm-memory-mcp-engine

# edge (no index; OAuth state shared via SQLite)
HARS_MEMORY_MCP_MODE=split \
HARS_MEMORY_MCP_EDGE_WORKERS=2 \
HARS_MEMORY_MCP_ENGINE_SOCKETS=/run/user/1000/hars/mem-engine.sock \
HARS_MEMORY_MCP_OAUTH_STORE=sqlite \
HARS_MEMORY_MCP_OAUTH_STORE_PATH=~/.local/share/hars-longterm-memory/oauth/state.sqlite3 \
HARS_MEMORY_MCP_TLS_CERTFILE=/etc/hars/server.pem \
HARS_MEMORY_MCP_TLS_KEYFILE=/etc/hars/server-key.pem \
hars-longterm-memory-mcp-https
```

Kubernetes mapping (one Deployment each): the engine as a **StatefulSet with
`replicas: 1` and a ReadWriteOnce volume** for the working dir (it is the only
writer), the edge as a normal Deployment (`replicas: N`) behind the Service; an
`emptyDir`-backed Unix socket cannot cross pods, so use **one pod with both
containers** sharing an `emptyDir`, or switch the engine transport to TCP
(`HARS_MEMORY_MCP_ENGINE_HOST/PORT` is the natural next step — the JSON-RPC
protocol is transport-agnostic). Scale the edge freely; scale engines only with
Qdrant and only as read replicas.

---

## 7. Environment reference

| Variable | Default | Meaning |
|---|---|---|
| `HARS_MEMORY_MCP_MODE` | `combined` | `combined` · `split` (edge) · `engine` |
| `HARS_MEMORY_MCP_ENGINE_THREADS` | `min(8, cpus)` | retrieval worker threads; `0` = inline (pre-scaling) |
| `HARS_MEMORY_MCP_ENGINE_QUEUE` | `4 × threads` | max admitted calls before `503 OverloadedError` |
| `HARS_MEMORY_MCP_ENGINE_QUEUE_TIMEOUT_MS` | `10000` | how long a call may wait for a slot |
| `HARS_MEMORY_MCP_ENGINE_STATS` | `1` | expose engine/loop stats on `/healthz` |
| `HARS_MEMORY_MCP_LOOP_LAG_INTERVAL_MS` | `50` | loop-lag sampling period |
| `HARS_MEMORY_EMBED_HTTP_MAX_CONNECTIONS` | `32` | keep-alive pool per process |
| `HARS_MEMORY_EMBED_HTTP_MAX_KEEPALIVE` | `= max_connections` | idle connections kept |
| `HARS_MEMORY_EMBED_BASE_URL` | — | embedder `/v1`; size it per §2 |
| `HARS_MEMORY_RIPGREP_ROOTS` | deployment-specific | **narrow this**; each root is scanned per recall |
| `HARS_MEMORY_MCP_EDGE_WORKERS` | `1` | uvicorn workers on the edge |
| `HARS_MEMORY_MCP_ENGINE_SOCKET` | `~/.local/share/…/mcp/engine.sock` | single engine socket |
| `HARS_MEMORY_MCP_ENGINE_SOCKETS` | — | pool (`a.sock,b.sock`) and/or affinity (`proj=a.sock`) |
| `HARS_MEMORY_MCP_ENGINE_MAX_INFLIGHT` | `64` | engine-side cap |
| `HARS_MEMORY_MCP_ENGINE_WARM_BM25` / `_WARM_RECALL` | `1` | prime BM25 / one full recall at startup (no cold shard) |
| `HARS_MEMORY_MCP_ENGINE_TRACE` | `0` | only **one** engine may own the trace SSE port |
| `HARS_MEMORY_MCP_EDGE_TIMEOUT_MS` | `600000` | edge → engine timeout |
| `HARS_MEMORY_MCP_OAUTH_STORE` | `sqlite` | `sqlite` · `redis` · `json` (legacy) · `memory` |
| `HARS_MEMORY_MCP_OAUTH_STORE_PATH` | `~/.local/share/…/oauth/state.sqlite3` | SQLite file (WAL) |
| `HARS_MEMORY_MCP_OAUTH_STORE_URL` / `_PREFIX` | — / `hars:oauth:` | Redis URL / key prefix |
| `HARS_MEMORY_MCP_OAUTH_REVOCATION_CACHE_TTL` | `5` | seconds before a worker re-reads the revocation set |

Access tokens stay **stateless Ed25519 JWTs**: no token-record lookup per
request, only a cached revocation check (5 s propagation across workers).

---

## 8. Benchmarks (how to reproduce)

```bash
# end-to-end, N closed-loop clients, JSON + healthz probe + CPU/RSS
for n in 1 4 8 16; do
  .venv/bin/python scripts/mcp_scale_bench.py --url http://127.0.0.1:8234 \
      --approval-token "$TOKEN" --concurrency $n --repeat 8 --warmup 2 \
      --server-pid "$EDGE_PID" --server-pid "$ENGINE_PID" \
      --out bench/c$n.json
done

# per-stage breakdown from the response's own timings
.venv/bin/python scripts/mcp_stage_bench.py --mode server \
    --url http://127.0.0.1:8232 --approval-token "$TOKEN" \
    --concurrency 1,4,8,16 --repeat 8 --out bench/stages.json

# which stages release the GIL (embedder, numpy, BM25, LTR, pure-Python fusion)
.venv/bin/python scripts/mcp_stage_bench.py --mode micro --out bench/micro.json
```

Always report the **host load** next to the numbers (`uptime`, or the
`load_avg` sampled by the driver): on a shared box, two runs of the identical
build can differ 2–3x. CPU is summed over the `--server-pid`s, RSS is reported
per process, and `healthz_p95_ms` is the event-loop responsiveness under load.

### Language/runtime decision

A Go (or Rust) rewrite of the server was considered and rejected for the
engine:

* profiling puts the time in the **embedder service** (HTTP, external) and in
  **numpy/LightGBM/bm25s** (native code that already releases the GIL), not in
  the Python glue; a rewrite would move the same bytes through the same two
  bottlenecks;
* the engine *is* LightRAG + sentence-transformers + bm25s + LightGBM + the
  project's retrieval logic — reimplementing that in Go is a multi-month project
  with a retrieval-quality regression risk, for no measured gain;
* the one place Go would genuinely help is a **thin edge tier** at very high
  connection counts (thousands of idle MCP streams, TLS termination): a
  stateless JWT-verifying proxy. That is a deployment option, not a rewrite of
  the engine — and the current edge already costs 70 MB and ~0.05 cores per
  worker, so it is not the limit today.

---

## 9. Rollback

| to undo | set | effect |
|---|---|---|
| split | `HARS_MEMORY_MCP_MODE=combined` | one process, in-process index |
| extra edge workers | `HARS_MEMORY_MCP_EDGE_WORKERS=1` | single uvicorn worker |
| thread pool | `HARS_MEMORY_MCP_ENGINE_THREADS=0` | inline on the loop (pre-scaling) |
| Qdrant | `HARS_MEMORY_VECTOR_STORAGE=NanoVectorDBStorage` | JSON vector stores (never deleted) |
| prefixed collections | unset `HARS_MEMORY_QDRANT_COLLECTION_PREFIX` | back to the shared `lightrag_vdb_*` names |
| SQLite OAuth state | `HARS_MEMORY_MCP_OAUTH_STORE=json` | the legacy `state.json` (migrated from, never deleted) |
| native server | `HARS_MEMORY_MCP_HTTP_NATIVE=0` | legacy `supergateway --stateful` bridge |

The Qdrant migration is additive and reversible (`--reverse` rebuilds
`vdb_*.json` from Qdrant).

---

## 10. Honest limits

1. **Embedder slots × latency** cap concurrent recalls. One shared CPU endpoint
   with `-np 2` cannot serve 8 concurrent users with low latency no matter what
   the server does; the server's job is to not *add* serialization (it did, via
   the double-embed and per-call connection bugs — both fixed).
2. **Per-request CPU** (ripgrep scan, BM25, numpy scan, LTR features) sets the
   floor. On this host a warm recall costs ~0.2–0.3 core-seconds; 8 concurrent
   recalls need ~2 cores of headroom *on top of* the embedder.
3. **The graph is per process.** Every engine loads its own NetworkX graph for
   `memory_entities`/`memory_related`/graph-bridge. N engines ⇒ N × ~1.66 GB.
4. **Pure-Python stages still hold the GIL** (fusion, diversity, response
   assembly). They are small (20–50 ms) but they serialize *within* an engine,
   which is why the answer is more processes rather than more threads.
5. **One writer.** Qdrant makes reads shareable; ingestion mutates the working
   directory. Do not point two engines at one working dir for writes.
6. **Qdrant is a single node** here and adds a dependency on the hot path.
7. **Warm-up is still blocking** per process (11–37 s depending on load); the
   edge is unaffected, and `_WARM_RECALL` keeps a round-robin pool from having a
   cold shard.

Next steps if more concurrency is needed, in order of value per effort:
embedder on dedicated cores/host (cheap, largest), narrow
`HARS_MEMORY_RIPGREP_ROOTS` (cheap), 2–4 Qdrant-backed engines (moderate), a
shared read-only graph snapshot or graph service (large), Qdrant cluster
(large).

---

## 11. Tests

* `tests/test_mcp_runtime.py` — pool bounds, inline mode, overload, loop lag.
* `tests/test_mcp_oauth_store.py` — SQLite concurrency, cross-worker refresh
  rotation/codes/revocations, idempotent JSON migration, Redis optional.
* `tests/test_mcp_engine.py` — Unix-socket protocol, router pool/affinity,
  edge-side scope gating, retryable engine errors.
* `tests/test_mcp_vector_query.py` — offloaded query == storage's own `query()`
  (Nano and Qdrant), and exactly one embed per query.
* `tests/test_qdrant_prefix.py` — prefix-aware collection naming matches the
  transplant tool; registration satisfies lightrag's validator.
* Existing `tests/test_mcp_oauth.py`, `tests/test_http_mcp_integration.py`,
  `tests/test_mcp_server.py`, `tests/test_embedder_http_mode.py` stay green.
