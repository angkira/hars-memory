# Integration guide

How to consume `hars-longterm-memory` from another project — install from the
GitLab registry, set up the MCP server, configure Qdrant, and use the skill.

## Installation

### 1. Add the GitLab registry

Add the private GitLab PyPI registry to your project's `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "hars-longterm-memory"
url = "https://code.knotgate.com/api/v4/projects/14/packages/pypi/simple"
explicit = true
```

Set up authentication (deploy token):

```bash
export UV_INDEX_HARS_LONGTERM_MEMORY_USERNAME="<deploy-token-username>"
export UV_INDEX_HARS_LONGTERM_MEMORY_PASSWORD="<deploy-token-password>"
```

Store these in a gitignored secrets file, never committed.

### 2. Add the dependency

```toml
[project]
dependencies = [
    "hars-longterm-memory>=0.5.0",
]

[tool.uv.sources]
hars-longterm-memory = { index = "hars-longterm-memory" }
```

### 3. CPU-only torch

The package depends on `torch` (CPU-only). Repeat the CPU-only torch index in
your own `pyproject.toml` — `[tool.uv.sources]` does NOT propagate through a
built wheel:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

### 4. Install

```bash
uv sync
```

## MCP server setup

### Option A: Per-project MCP (recommended for AI coding sessions)

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "hars-longterm-memory": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "tools/memory",
        "memory-mcp"
      ],
      "env": {
        "HARS_MEMORY_INDEX_DIR": "/path/to/index",
        "HARS_MEMORY_STAGING_DIR": "/path/to/staging",
        "HARS_MEMORY_EXTRACTOR_BASE_URL": "http://localhost:8080/v1",
        "HARS_MEMORY_EXTRACTOR_MODEL": "Qwen3.6-27B-Q4_K_M",
        "HARS_MEMORY_QUERY_BASE_URL": "http://localhost:8080/v1",
        "HARS_MEMORY_QUERY_MODEL": "Qwen3.6-27B-Q4_K_M",
        "HARS_MEMORY_EMBED_MODEL": "embeddinggemma-300m",
        "HARS_MEMORY_QDRANT_URL": "http://localhost:6335",
        "HARS_MEMORY_QDRANT_COLLECTION": "my-project-workspace"
      }
    }
  }
}
```

### Option B: Standalone MCP entrypoint

```bash
uv run --project <project> memory-mcp
```

Or the standalone entrypoint (installed via pip/uv):

```bash
hars-longterm-memory-mcp
```

### Required env vars

| Variable | Why |
|---|---|
| `HARS_MEMORY_INDEX_DIR` | Path to the LightRAG working directory. No default — must be set explicitly |
| `HARS_MEMORY_STAGING_DIR` | Directory for `memory_remember` notes. Must exist and be writable |

### Optional but recommended

| Variable | Why |
|---|---|
| `HARS_MEMORY_QDRANT_URL` | Qdrant endpoint. Default `http://localhost:6333` |
| `HARS_MEMORY_QDRANT_COLLECTION` | NOT a collection name — LightRAG tenant id. Use a unique value per project |

## Qdrant setup

### Local development (Docker)

```yaml
version: "3"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6335:6333"
      - "6336:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

Start:

```bash
docker compose up -d qdrant
curl http://localhost:6335/readyz
```

### Production

Deploy Qdrant as a StatefulSet in K8s or use the Qdrant Cloud service. Set
`HARS_MEMORY_QDRANT_URL` to the endpoint.

### Gotcha: collection naming

`HARS_MEMORY_QDRANT_COLLECTION` is NOT a Qdrant collection name. LightRAG uses
it as the `workspace_id` tenant id written into every payload. The actual
collections are always `lightrag_vdb_{chunks,entities,relationships}`. If you
share a Qdrant instance across projects, use unique `HARS_MEMORY_QDRANT_COLLECTION`
values per project to isolate them.

## Optional: install the skill for AI agents

Copy `.claude/skills/hars-longterm-memory/SKILL.md` from the repo into your
project's `.claude/skills/hars-longterm-memory/SKILL.md`. This skill teaches
an AI coding agent how to install, configure, and use the MCP tools.

For Gemini Code Assist, the skill also works under `.gemini/skills/`.

## First index

After installation, build the initial index:

```bash
# Index files (GPU must be free if guard is configured)
HARS_MEMORY_EXTRACTOR_BASE_URL=http://localhost:8080/v1 \
HARS_MEMORY_EXTRACTOR_MODEL=Qwen3.6-27B-Q4_K_M \
HARS_MEMORY_QUERY_BASE_URL=http://localhost:8080/v1 \
HARS_MEMORY_QUERY_MODEL=Qwen3.6-27B-Q4_K_M \
uv run --project <project> python -m hars_memory.server.index \
    --paths docs/ knowledge/
```

Dry-run (no LLM, just document counts):

```bash
uv run --project <project> python -m hars_memory.server.index \
    --paths docs/ knowledge/ --dry-run
```

## Usage

### From an AI agent (MCP)

```python
# Check index status
memory_status()

# Query the knowledge graph
memory_recall(
    question="How do I configure GKE ingress?",
    mode="hybrid",
    top_k=12,
    ll_keywords=["GKE", "Ingress", "networking"],
    hl_keywords=["Kubernetes", "networking", "configuration"],
)

# Save a note
memory_remember(
    title="gke-ingress-tls",
    content="GKE ingress requires a managed certificate...",
    importance="normal",
    tags=["gke", "ingress", "tls"],
)
```

### From Python (SDK)

```python
from pathlib import Path
from hars_memory.sdk import HarsMemoryClient

with HarsMemoryClient("http://127.0.0.1:8787", "your-api-key") as client:
    job = client.create_index(
        [Path("docs/")],
        engine="corpus",  # CPU-only, LLM-free
        idempotency_key="docs-v1",
    )
    completed = client.wait_job(job.id)
    version = client.get_latest_index(completed.index_id)
    client.download_artifact(completed.index_id, Path("docs-index.tar.gz"))
```

### From CLI

```bash
# Index
uv run --project <project> memory-index --paths docs/

# MCP server
uv run --project <project> memory-mcp

# Evaluation
uv run --project <project> memory-eval-battle --paths docs/ --cases 100
```

## Updating the index

```bash
# Incremental reindex (dry-run first)
memory_consolidate(paths=["docs/", "knowledge/"], dry_run=True)
# Then actual
memory_consolidate(paths=["docs/", "knowledge/"], dry_run=False)
```

## Removing stale documents

```bash
# Dry-run: list candidates older than 90 days
memory_forget(older_than_days=90, protect=[".*important.*"], apply=False)

# Apply: actually delete
memory_forget(older_than_days=90, protect=[".*important.*"], apply=True)
```

## Performance tuning

| Setting | Recommended | Notes |
|---|---|---|
| `top_k` | 20 | Default. Measured sweet spot on this corpus |
| `fetch_top_k` | 1.7-2x `top_k` | Helpful when `top_k` is narrow (e.g. `top_k=10, fetch_top_k=18`) |
| `HARS_MEMORY_HYBRID_ALPHA` | 0.5 | Default. Measured optimum |
| `HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER` | 1.0 | Default. No benefit from widening at `top_k=20` |
| Qdrant | Production | NanoVectorDB is file-based and not suitable for production |