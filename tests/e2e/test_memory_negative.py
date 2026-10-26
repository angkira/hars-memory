"""Negative tests: out-of-domain queries should produce low-confidence answers.

Tests that the memory system correctly identifies queries it cannot answer
(rather than hallucinating plausible-sounding but unsupported answers).
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
KB_ROOT = Path("/Users/iurii.medvedev/Work/sre-support-pier")

# Out-of-domain queries — memory should NOT find relevant KB content
OUT_OF_DOMAIN_QUERIES = [
    ("What is the capital of France?", "geo"),
    ("How to make pizza dough from scratch?", "cooking"),
    ("What is the meaning of life according to philosophy?", "philosophy"),
    ("Who won the World Cup in 2022?", "sports"),
    ("Explain quantum field theory in simple terms", "physics"),
]


def _enabled() -> bool:
    return os.environ.get("HARS_RUN_LIVE_LLM_E2E", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(not _enabled(), reason="set HARS_RUN_LIVE_LLM_E2E=1")
@pytest.mark.parametrize(
    "question,domain",
    OUT_OF_DOMAIN_QUERIES,
    ids=lambda tc: tc[-1] if isinstance(tc, tuple) else tc,
)
def test_out_of_domain_low_confidence(question: str, domain: str, tmp_path: Path) -> None:
    """Out-of-domain queries should trigger low_confidence flag."""
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
            "HARS_MEMORY_RIPGREP_ROOTS": str(KB_ROOT),
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
                # Test hybrid mode (most informative for confidence)
                response = await session.call_tool(
                    "memory_recall",
                    {
                        "question": question,
                        "mode": "hybrid",
                        "top_k": 10,
                        "fetch_top_k": 20,
                        "context_only": False,
                    },
                )
                return json.loads(response.content[0].text)

    result = anyio.run(run_query)
    answer_text = str(result.get("answer", ""))

    # Two acceptable behaviors for out-of-domain queries:
    # 1. Graceful refusal: ok=False, answer contains "not able" / "no-context"
    # 2. Low confidence: ok=True, hybrid.confidence.low_confidence=True
    is_refusal = not result.get("ok") and ("not able" in answer_text.lower() or "no-context" in answer_text.lower())
    hybrid = result.get("hybrid", {})
    confidence = hybrid.get("confidence", {}) if isinstance(hybrid, dict) else {}
    is_low_conf = result.get("ok") and confidence.get("low_confidence") is True

    print(f"\n  [{domain}] ok={result.get('ok')} | low_confidence={confidence.get('low_confidence')} | "
          f"top_dense_score={confidence.get('top_dense_score')}")

    if is_refusal:
        print(f"  ✅ Graceful refusal: {answer_text[:150]}")
        assert True, "Memory correctly refused to answer out-of-domain query"
    elif is_low_conf:
        print(f"  ✅ Low confidence: citations={len(result.get('citations', []))}")
        print(f"  Answer: {answer_text[:200]}")
        assert True, "Memory correctly flagged out-of-domain query as low-confidence"
    else:
        # Hallucination — memory answered confidently but shouldn't have
        citations = result.get("citations", [])
        print("  ⚠️  Memory answered confidently — possible hallucination:")
        print(f"  Answer: {answer_text[:200]}")
        print(f"  Citations: {len(citations)}")
        # Soft fail — log but don't assert (the LLM may still give a reasonable answer
        # from its own knowledge, which is not necessarily a memory problem)
        if len(citations) == 0:
            print("  (No citations — answer is from LLM knowledge, not memory)")
        pytest.fail(
            f"Memory did not refuse or flag as low-confidence for out-of-domain query "
            f"'{domain}'. Answer: {answer_text[:200]}"
        )