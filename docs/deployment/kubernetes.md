# Kubernetes deployment (HTTP MCP + OAuth)

The manifests live in [`deploy/k8s/`](../../deploy/k8s/README.md). That README
is the operational reference (apply order, secrets, restore jobs, rollback);
this page records the deployment topology and the **embedder sizing** rules.

## Topology

```
client ──https──> ingress-nginx ──http──> Service:8099 ──> Pod (1 replica)
                                                             ├─ /data (PVC): index, staging, OAuth state
                                                             ├─ Qdrant (external, lightrag_vdb_*)
                                                             └─ embedder (external URL, or optional in-cluster llama-server)
```

* One `hars-longterm-memory-mcp-https` process, `replicas: 1`,
  `strategy: Recreate`, ReadWriteOnce index PVC.
* TLS terminates at ingress-nginx (work profile) or is absent (local profile);
  app-terminated TLS is an optional component (W2).
* OAuth is **off** in the local profile and **on** in the work profile, with a
  per-cluster issuer host.
* Vectors live in Qdrant so they survive engine rollbacks; the index working
  directory holds the graph/KV and is not multi-writer.

## Embedder is the first bottleneck

Measured on the reference CPU-only host (16 cores) with llama-server at
`-t 4 -np 2`:

| Load | Throughput vs 1 client |
|---|---|
| 8 concurrent embedding requests | **1.46×** |
| 8 concurrent `memory_recall` calls | **1.51×** |

The engine processes used only ~0.3–0.6 cores while throughput saturated: the
requests were queued on two embedding slots. Adding app replicas against the
same endpoint lengthens the queue; it does not raise throughput.

Two server-side defects were making this worse and are fixed in the current
build: every recall embedded its question **twice** (a backend-dispatch bug in
the offloaded vector query), and every embed opened a **new HTTP connection**
(no keep-alive). A 16-way slot/thread/batch sweep is in
[`docs/http-mcp-scaling.md`](../http-mcp-scaling.md) §2. The headline: same
cores, `-b/-ub 512 → 2048` cut single-call latency 76.5 ms → 26.8 ms and raised
throughput 33 → 46 req/s; `-t 16 -np 4` collapses to ~1 req/s (thread
oversubscription), so `-t` must stay ≤ half the cores.

### Concurrency model

* One uvicorn worker and one shared index per pod.
* With the bounded engine pool (`HARS_MEMORY_MCP_ENGINE_THREADS`, default
  `min(8, cpus)`), up to that many retrieval calls run concurrently; the queue
  is `4 × threads` and past it calls fail fast with a retryable `503`
  (`/healthz` exposes `engine.overloads`, `queue_wait_ms` and `run_ms`). Set
  `HARS_MEMORY_MCP_ENGINE_THREADS=0` to get the pre-scaling inline behaviour.
* One recall issues ~one embedding request (`flat_dense` is off by default), so
  **slots ≈ simultaneous recalls**. Write tools also embed and compete.
* Parallel: HTTP handling, embedder slots, Qdrant, numpy, the pool's blocking
  stages. Serial: Python fusion/assembly per process (GIL), index writes, OAuth
  state writes. See `docs/http-mcp-scaling.md` §1 for the full table.
* Beyond one pod: `HARS_MEMORY_MCP_MODE=split` puts TLS/OAuth/MCP on N edge
  workers and the index in one engine per shard, with Qdrant shared.

### Sizing table

| Simultaneous recalls | `-np` slots | `-t` threads | Pod CPU req/limit | Memory req/limit |
|---|---|---|---|---|
| 1–2 (home) | 2 | 4 | 1 / 2 | 1Gi / 2Gi |
| 3–4 | 2–4 | 4 | 2 / 4 | 1Gi / 2Gi |
| 5–8 (work default) | 4 | 4–6 | 4 / 6 | 2Gi / 4Gi |
| 9–16 | 4–8 on a **dedicated node** | 4–8 | 6 / 8 | 2Gi / 4Gi |
| >16 | second embedder Service + split groups, or GPU | | | |

`EMBED_THREADS` must not exceed the CPU limit, and must stay **≤ half the node's
cores** (≥16 threads per slot group measured *worse* than 4–8: 1 req/s vs 46).
Keep `EMBED_UBATCH ≤ EMBED_CTX / EMBED_SLOTS`; the shipped default is
`EMBED_CTX=8192, EMBED_BATCH=UBATCH=2048, EMBED_SLOTS=4, EMBED_THREADS=4`.
Slots beyond ~4 only help when the pod gets dedicated cores — the embedder and
the retrieval threads otherwise starve each other on the same node.

### Enabling the in-cluster embedder

```bash
# uncomment in deploy/k8s/overlays/{work,local}/kustomization.yaml:
#   - ../../components/embedder
#   - path: embedder-sizing.patch.yaml
#   - path: embedder-resources.patch.yaml
#   - path: embedder-storageclass.patch.yaml
# model: create Secret embedder-model/url (init container downloads it) or
# pre-seed the embedder-models volume with the GGUF before the pod starts
kubectl apply -k deploy/k8s/overlays/work
```

The component creates a single-replica llama-server (`/app/llama-server`,
OpenAI-compatible `/v1/embeddings` on 8791) with a 5 GiB model PVC, an
init container that downloads the GGUF when a URL is configured, probes on
`/health`, and hardening (non-root, read-only rootfs, drop ALL). It rewrites
`HARS_MEMORY_EMBED_BASE_URL` to `http://embedder:8791/v1`.

### Measuring

```bash
# embedder: compare 1 vs N concurrent /v1/embeddings requests (README script)
kubectl -n hars-memory port-forward svc/embedder 8791:8791
# service: 8 clients, 5 recalls each
python scripts/mcp_http_smoke.py --url https://<issuer> \
  --approval-token "$HARS_MEMORY_MCP_OAUTH_APPROVAL_TOKEN" --clients 8 --repeat 5
# engine stats (when the pool is enabled): engine threads/queue, loop lag
curl -s https://<issuer>/healthz | jq '{index_warm, engine, loop}'
```

### Rule

1. Scale the **embedder** (slots, then threads/CPU) before adding app replicas.
2. `replicas > 1` needs shared Qdrant (default), a shared OAuth store, an
   identical signing key, and exactly one index writer. The committed server
   does not have the shared OAuth store yet, so keep one replica per engine.
3. For more projects, add namespaces/engines rather than replicas. A future
   edge/engine split is not implemented.

## Validation status

The bundle was validated by YAML parsing plus a reduced kustomize renderer
(resources/components/patches/prefixes, probes, security, PVC references,
profile invariants, embedder composition). `kubectl`/`kustomize` are not
installed on the build host, so no `--dry-run` was performed. Run
`kubectl apply -k deploy/k8s/overlays/work --dry-run=server` before the first
apply.
