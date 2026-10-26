# hars-longterm-memory documentation

Documentation index for the `hars-longterm-memory` package — a standalone,
model-agnostic long-term memory service built on LightRAG (graph+vector), Qdrant,
and CPU-only embeddings.

## Quick links

| Document | What it covers |
|---|---|
| [MCP tool reference](mcp-reference.md) | All 7 MCP tools: `memory_recall`, `memory_remember`, `memory_entities`, `memory_related`, `memory_status`, `memory_consolidate`, `memory_forget` — parameters, response shapes, examples |
| [Configuration reference](configuration.md) | All environment variables, their defaults, and semantics |
| [Architecture](architecture.md) | Module map, data flow, component relationships, retrieval pipeline |
| [Integration guide](integration-guide.md) | How to consume this package from another project — install from GitLab registry, set up MCP server, configure Qdrant, use the skill |
| [Incremental refresh](incremental-refresh.md) | `plan / apply / audit / repair-orphans` CLI for refreshing a large LightRAG index without a full rebuild |
| [Release process](release-process.md) | How to publish a new version — PyPI package, container image, CI/CD |
| [Index service](index-service.md) | HTTP index-job service — SDK, API, deployment, artifact stores |
| [Kubernetes deployment](deployment/kubernetes.md) | HTTP MCP + OAuth on k8s: topologies, embedder sizing, apply/restore |
| [gRPC reference](grpc-reference.md) | gRPC service — 7 RPCs, client usage, configuration, health check |
| [Ripgrep ranked channel](ripgrep-ranked-channel.md) | The literal-search channel as a first-class RANKED fusion channel (default ON): structural verdict, scoring, flags, measurements, deployment |
| [Channel hygiene](CHANNEL_HYGIENE.md) | Per-channel audit: what is live, dormant, redundant, or a deletion candidate |

### CLI Reference

The `memory` command provides a unified interface for all subsystems:

| Subcommand | Description |
|---|---|
| `memory recall <question>` | Query the LightRAG knowledge graph |
| `memory mcp` | Start MCP server (stdio) |
| `memory server` | Start HTTP index-job service |
| `memory grpc` | Start gRPC server |
| `memory build` | Build / incrementally re-index a corpus |
| `memory query` | Search a corpus-built index |
| `memory eval` | Run retrieval metrics against a corpus |
| `memory regress` | Compare two eval reports |
| `memory status` | Print corpus manifest summary |
| `memory consolidate` | Trigger incremental reindex |
| `memory export <output>` | Export index to a tar.gz archive |
| `memory import <archive>` | Import index from a tar.gz archive |
| `memory estimate-cost <paths>` | Estimate indexing cost before running (dry-run, no API calls) |
| `memory strategy-bench` | Run strategy matrix benchmark |
| `memory extractor-bench <run\|score\|report>` | Benchmark extractor LLMs on fixed docs |

Examples:

```bash
# Quick query
memory recall "What is Kubernetes?" --mode hybrid

# Start servers
memory mcp
memory server --port 8787
memory grpc --port 8788

# Corpus management
memory build --paths kb/ --index-dir /tmp/my-index
memory query --index-dir /tmp/my-index --question "how to deploy?"

# Export/import index
memory export /tmp/hars-index-backup.tar.gz
memory import /tmp/hars-index-backup.tar.gz --index-dir /tmp/new-index

# Estimate cost before indexing
memory estimate-cost kb/ --model gpt-5.6-luna --chunk-size 2048 --batch-size 256
memory estimate-cost kb/ --model deepseek-v3.2 --max-gleaning 1

# Extractor LLM benchmark (fixed docs + reference entities in
# hars_memory/eval/fixtures/extractor_bench_v1/; thinking is forced off via
# chat_template_kwargs.enable_thinking=false, leaks fail the run)
memory extractor-bench run --model qwen3.5-9b --out /tmp/bench-runs/qwen3.5-9b
memory extractor-bench score --run-dir /tmp/bench-runs/qwen3.5-9b
memory extractor-bench report --runs /tmp/bench-runs/*/ --out /tmp/bench-runs/report.txt
```

Legacy entrypoints (`memory-index`, `memory-mcp`, `memory-grpc`, `hars-longterm-memory-server`, etc.) remain available for backward compatibility.

## Quick start

```bash
uv sync --extra dev
cp config/.env.example config/.env
# edit config/.env with your paths
uv run pytest -q
```

## Repository structure

```
hars_memory/              # Python package
  cli.py                  # console-script entrypoints
  mcp_server.py           # MCP server: 7 tools, ~2900 lines
  ingest/                 # walker, chunker, change detection
  server/                 # index.py, lightrag_init.py, embedder, reranker
  retrieval/              # BM25, flat-dense, ripgrep, fusion, supersession
  corpus/                 # LLM-free build/query pipeline
  schema/                 # entity/relation type schema
  eval/                   # gold-question harness, battle test, metrics
  service/                # HTTP index-job API, artifact stores, SDK
  sdk.py                  # synchronous HTTP client
  grpc/                   # gRPC server, client, generated proto stubs
  strategies.py           # validated index/search strategies
  scripts/                # maintenance tools
cortex-scripts/           # Cortex consumer scripts (not part of installed package)
docs/                     # this documentation
config/.env.example       # full env-var reference
tests/                    # 742+ passing tests
```