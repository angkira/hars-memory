# hars-longterm-memory

Standalone, self-hosted, CPU-only long-term memory for AI coding agents:
a LightRAG graph+vector knowledge base exposed as an MCP server and a CLI.
Bring your own documents; query them from any MCP-capable session.

- **Graph + vector retrieval.** LightRAG entity/relation extraction over your
  corpus, layered with additive retrieval channels: dense vectors, BM25
  sparse, live-worktree ripgrep, and supersession-aware rescoring.
- **CPU-only by design.** Embeddings run on CPU
  (`unsloth/embeddinggemma-300m` default); `torch` resolves to the CPU wheel
  (see the override in [Install](#install)).
- **Two surfaces.** MCP server with 7 tools (`memory_recall`,
  `memory_remember`, `memory_forget`, `memory_consolidate`, `memory_status`,
  plus `memory_entities` / `memory_related`) for agents, and a `memory`
  CLI for scripting, CI, and maintenance.
- **Pluggable models.** Every model binding is an env var (OpenAI-compatible
  extractor / query endpoints, embedder, optional reranker) — swap models
  with a config edit, no code changes.

## What's new in 0.9.0

- **Russian sparse channel** (`HARS_MEMORY_SPARSE_STEMMER=ru|en|off`,
  default `off`). The BM25 tokenizer was ASCII-only, so Cyrillic documents
  produced zero tokens and the sparse channel was dead on Russian corpora.
  `ru` adds Snowball stemming plus a `stem[:9]` prefix token. Needs the
  optional extra: `uv add "hars-longterm-memory[sparse-stem] @ ..."`
  (PyPI distribution name `PyStemmer`). Measured: facts+ru-sparse 0.351
  vs 0.297/0.324 with the dead channel.
- **Content-dedup on upsert** (`HARS_MEMORY_UPSERT_CONTENT_DEDUP=1`,
  default off): identical content re-upserted under a new path answers
  `skipped_duplicate_of_existing` instead of storing a second copy.
- **Batch queue for MCP upserts** (`HARS_MEMORY_UPSERT_BATCH_MODE=queue`):
  upserts land in an inbox sidecar in milliseconds and
  `memory index-batch --source upsert-inbox` runs extraction through a
  provider Batch API (~50% cheaper).
- **Recall sizing guidance**: size `max_chars` to the corpus, not to
  politeness. Measured on a 91K-entity standing KB: 0.17 answer accuracy
  at `max_chars=6000` vs **0.42 at `max_chars=32000`** — pass **16–32K**
  for standing project KBs (server default `15000` via
  `HARS_MEMORY_CONTEXT_MAX_CHARS`; per-call `max_chars` overrides it).

Full history: [CHANGELOG.md](CHANGELOG.md).

## Install

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv add "hars-longterm-memory @ git+https://github.com/angkira/hars-memory@v0.9.0"
```

`torch` must stay CPU-only. A built wheel does not propagate the package's
own `[tool.uv.sources]` table, so repeat this override in **your**
`pyproject.toml` or `torch` resolves to the full CUDA build from PyPI:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

Optional extras: `sparse-stem` (Snowball stemmers for the BM25 channel),
`server` (HTTP index-job service), `tui` (terminal UI), `ltr`
(learned channel-fusion ranker training).

## Quickstart

Minimal configuration — everything is env vars, see
[config/.env.example](config/.env.example) for the full reference:

```bash
export HARS_MEMORY_INDEX_DIR="$HOME/.local/share/my-project-memory/index"
export HARS_MEMORY_STAGING_DIR="$HOME/.local/share/my-project-memory/staging"
export HARS_MEMORY_EMBED_MODEL="unsloth/embeddinggemma-300m"
export HARS_MEMORY_EXTRACTOR_BASE_URL="http://localhost:8080/v1"  # OpenAI-compatible
export HARS_MEMORY_EXTRACTOR_MODEL="Qwen3.6-27B-Q4_K_M"
export HARS_MEMORY_QUERY_BASE_URL="http://localhost:8081/v1"
export HARS_MEMORY_QUERY_MODEL="Qwen3.5-4B-Q4_K_M"
```

Register the MCP server in your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "hars-longterm-memory": {
      "command": "uv",
      "args": ["run", "--project", "<path-to-your-project>", "hars-longterm-memory-mcp"],
      "env": {
        "HARS_MEMORY_INDEX_DIR": "/home/you/.local/share/my-project-memory/index",
        "HARS_MEMORY_STAGING_DIR": "/home/you/.local/share/my-project-memory/staging",
        "HARS_MEMORY_EMBED_MODEL": "unsloth/embeddinggemma-300m",
        "HARS_MEMORY_EXTRACTOR_BASE_URL": "http://localhost:8080/v1",
        "HARS_MEMORY_QUERY_BASE_URL": "http://localhost:8081/v1"
      }
    }
  }
}
```

Then, from an AI session:

```
memory_status()

memory_recall(
  question="What broke the last storage migration?",
  ll_keywords=["migration", "storage"],
  hl_keywords=["vector storage cutover"],
  mode="hybrid",
  top_k=12,
  max_chars=32000,
)
```

`memory_status()` is GPU-free — call it first to confirm the index exists.
`memory_recall` returns raw retrieved context by default (`context_only=true`)
for the calling agent to synthesize; write `question` and keywords in the
language of the documents you expect to match.

## Documentation

| Document | What it covers |
|---|---|
| [docs/](docs/README.md) | Documentation index |
| [docs/mcp-reference.md](docs/mcp-reference.md) | All 7 MCP tools — parameters, response shapes, examples |
| [docs/configuration.md](docs/configuration.md) | Every env var, defaults, and semantics |
| [docs/architecture.md](docs/architecture.md) | Module map, data flow, retrieval pipeline |
| [docs/integration-guide.md](docs/integration-guide.md) | Consuming this package from another project |
| [config/.env.example](config/.env.example) | Full env-var reference with inline rationale |
| [.claude/skills/hars-longterm-memory/SKILL.md](.claude/skills/hars-longterm-memory/SKILL.md) | Portable skill: drop into any project's `.claude/skills/` to give an agent memory setup, query, and maintenance instructions |

## Build / test locally

```bash
uv sync --extra dev   # NOT `uv sync --group dev` — dev is an optional-dependency
uv run pytest -q
```

`.python-version` pins Python 3.13; `uv.lock` is the pinned lock file.

## License and attribution

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

If you use this software in a product, publication or service, you must keep
the NOTICE file and cite the project:

    https://github.com/angkira/hars-memory
