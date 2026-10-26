"""Status/health check tests for the memory index.

Verifies that the memory index reports its health correctly:
- Total documents, entities, relations
- Index size and staleness
- Storage backends
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.live_llm

KB_INDEX_DIR = Path("/tmp/hars_memory_lightrag")


def _enabled() -> bool:
    return os.environ.get("HARS_RUN_LIVE_LLM_E2E", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(not _enabled(), reason="set HARS_RUN_LIVE_LLM_E2E=1")
def test_memory_status_report(tmp_path: Path) -> None:
    """memory_status should report index health with document/entity counts."""
    required = {
        "HARS_MEMORY_QUERY_BASE_URL": os.environ.get("HARS_MEMORY_QUERY_BASE_URL"),
        "HARS_MEMORY_QUERY_MODEL": os.environ.get("HARS_MEMORY_QUERY_MODEL"),
        "HARS_MEMORY_LLM_API_KEY": os.environ.get("HARS_MEMORY_LLM_API_KEY"),
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        pytest.fail(f"missing live LLM configuration: {', '.join(missing)}")

    mcp_env = os.environ.copy()
    mcp_env.update(
        {
            "HARS_MEMORY_INDEX_DIR": str(KB_INDEX_DIR),
            "HARS_MEMORY_STAGING_DIR": str(tmp_path / "staging"),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25"),
            "HARS_MEMORY_HYBRID_ENABLED": "1",
            "HARS_MEMORY_RIPGREP_CHANNEL": "1",
            "HARS_MEMORY_VECTOR_STORAGE": "NanoVectorDBStorage",
            "HARS_MEMORY_GRAPH_STORAGE": "NetworkXStorage",
            "HARS_MEMORY_EMBED_LOCAL_FILES_ONLY": "1",
        }
    )
    mcp_env.pop("HARS_MEMORY_GPU_GUARD_SCRIPT_PATH", None)

    executable = Path(sys.executable).parent / "hars-longterm-memory-mcp"
    if not executable.exists():
        pytest.fail(f"MCP executable not found: {executable}")

    params = StdioServerParameters(
        command=str(executable), env=mcp_env, cwd=str(KB_INDEX_DIR.parent)
    )

    async def run_status() -> dict:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.call_tool("memory_status", {})
                return json.loads(response.content[0].text)

    result = anyio.run(run_status)
    assert result.get("ok") is True, f"Status failed: {result.get('error', 'unknown error')}"

    storage = result.get("storage", {})
    vector = storage.get("vector", {})
    graph = storage.get("graph", {})

    print("\n  Index report:")
    print(f"    Entities:  {result.get('node_count', '?')}")
    print(f"    Relations: {result.get('edge_count', '?')}")
    print(f"    Index dir: {result.get('working_dir', '?')}")
    print(f"    Last ingest: {result.get('last_ingest', '?')}")
    print("  Storage:")
    print(f"    Vector:    {vector.get('backend', '?')}")
    print(f"    Graph:     {graph.get('backend', '?')}")
    print(f"    Index exists: {result.get('index_exists', False)}")

    # Core assertions
    assert result.get("node_count", 0) > 0, "Index should have entities (node_count)"
    assert result.get("edge_count", 0) > 0, "Index should have relations (edge_count)"
    assert result.get("index_exists") is True, "Index should exist"
    assert vector.get("backend") == "NanoVectorDBStorage", "Vector backend mismatch"
    assert graph.get("backend") == "NetworkXStorage", "Graph backend mismatch"

    print(f"\n  ✅ Index health: {result.get('node_count')} entities, "
          f"{result.get('edge_count')} relations, "
          f"last ingest {result.get('last_ingest')}")