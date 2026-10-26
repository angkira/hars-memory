"""Quick E2E test: query the indexed KB and check the quality of answers."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.live_llm

KB_INDEX_DIR = Path("/tmp/hars_memory_lightrag")

# Questions that test different KB domains
KB_QUESTIONS = [
    {
        "question": "What is the LiteLLM budget per user per month?",
        "expected_keywords": ["1000", "USD", "month"],
        "domain": "litellm",
    },
    {
        "question": "Which cloud provider handles the main SRE K8s clusters in Europe?",
        "expected_keywords": ["GCP", "Google", "europe-west"],
        "domain": "cloud",
    },
    {
        "question": "What is the Harbor runtime class used for evaluations?",
        "expected_keywords": ["gvisor", "runtime"],
        "domain": "harbor",
    },
    {
        "question": "What does the KB cover? List the top-level domains.",
        "expected_keywords": ["aws", "azure", "k8s", "monitoring", "services", "tools"],
        "domain": "pier",
    },
    {
        "question": "What is the main team's on-call responsibility?",
        "expected_keywords": ["24/7", "monitoring", "incident", "on-call"],
        "domain": "team",
    },
]


def _enabled() -> bool:
    return os.environ.get("HARS_RUN_LIVE_LLM_E2E", "").lower() in {"1", "true", "yes"}


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.mark.skipif(not _enabled(), reason="set HARS_RUN_LIVE_LLM_E2E=1")
@pytest.mark.parametrize("qdata", KB_QUESTIONS, ids=lambda q: q["domain"])
def test_kb_recall(tmp_path: Path, qdata: dict) -> None:
    """Query the pre-indexed KB via MCP and check answer quality."""
    required = {
        "HARS_MEMORY_QUERY_BASE_URL": os.environ.get("HARS_MEMORY_QUERY_BASE_URL"),
        "HARS_MEMORY_QUERY_MODEL": os.environ.get("HARS_MEMORY_QUERY_MODEL"),
        "HARS_MEMORY_LLM_API_KEY": os.environ.get("HARS_MEMORY_LLM_API_KEY"),
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        pytest.fail(f"missing live LLM configuration: {', '.join(missing)}")

    # Start the MCP server pointing at the KB index
    mcp_env = os.environ.copy()
    mcp_env.update(
        {
            "HARS_MEMORY_INDEX_DIR": str(KB_INDEX_DIR),
            "HARS_MEMORY_STAGING_DIR": str(tmp_path / "staging"),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25"),
            "HARS_MEMORY_HYBRID_ENABLED": "0",
            "HARS_MEMORY_RIPGREP_CHANNEL": "0",
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

    async def run_query() -> dict:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.call_tool(
                    "memory_recall",
                    {
                        "question": qdata["question"],
                        "mode": "naive",
                        "top_k": 5,
                        "fetch_top_k": 10,
                        "context_only": False,
                    },
                )
                return json.loads(response.content[0].text)

    result = anyio.run(run_query)
    assert result.get("ok") is True, f"Query failed: {result.get('error', 'unknown error')}"

    answer_text = str(result.get("answer", ""))
    context_text = str(result.get("context", ""))

    # Check that at least one expected keyword appears in answer or context
    found = [kw for kw in qdata["expected_keywords"] if kw.lower() in answer_text.lower() or kw.lower() in context_text.lower()]
    assert len(found) > 0, (
        f"Answer for '{qdata['domain']}' contained none of the expected keywords "
        f"{qdata['expected_keywords']}.\n"
        f"Answer: {answer_text[:500]}\n"
        f"Context: {context_text[:500]}"
    )
    print(f"\n  [{qdata['domain']}] ✅ matched: {found}")
    print(f"  Answer (first 300 chars): {answer_text[:300]}")