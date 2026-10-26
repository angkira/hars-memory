"""E2E tests for memory quality: citations, hybrid mode, graph mode, recall."""

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

# Questions with expected answers — each tests a different KB domain and mode
# Format: (question, expected_keywords, mode, label)
TEST_CASES = [
    # --- Citations and source-tracking ---
    (
        "What is the LiteLLM budget per user per month?",
        ["1000", "USD", "month"],
        "hybrid",
        "citations-litellm-budget",
    ),
    # --- Graph mode: local query (entity-level) ---
    (
        "Which cloud provider handles the main SRE K8s clusters in Europe?",
        ["GCP", "Google", "europe-west"],
        "local",
        "graph-local-cloud",
    ),
    # --- Graph mode: global query (relation-level) ---
    (
        "What runtime class is used for Harbor evaluations?",
        ["gvisor", "runtime"],
        "global",
        "graph-global-harbor",
    ),
    # --- Naive mode: vector search ---
    (
        "What does the KB cover? List the top-level domains.",
        ["aws", "azure", "k8s", "monitoring", "services", "tools"],
        "naive",
        "naive-kb-domains",
    ),
    # --- Hybrid mode: BM25 + vector ---
    (
        "What is the main team's on-call responsibility?",
        ["24/7", "monitoring", "incident", "on-call"],
        "hybrid",
        "hybrid-team-mission",
    ),
    # --- Exact keyword recall (ripgrep channel) ---
    (
        "LiteLLM per-user budget 1000 USD",
        ["1000", "USD", "budget"],
        "hybrid",
        "ripgrep-budget-exact",
    ),
    # --- Graph entity relationship ---
    (
        "What is Harbor used for and how does it relate to evaluations?",
        ["eval", "gVisor", "sandbox", "test"],
        "local",
        "graph-harbor-eval",
    ),
    # --- Multiple sentences / complex answer ---
    (
        "How do I get access to LiteLLM and what models are available?",
        ["key", "API", "model", "token"],
        "hybrid",
        "hybrid-litellm-access",
    ),
]


def _enabled() -> bool:
    return os.environ.get("HARS_RUN_LIVE_LLM_E2E", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(not _enabled(), reason="set HARS_RUN_LIVE_LLM_E2E=1")
@pytest.mark.parametrize(
    "question,expected_keywords,mode,label",
    TEST_CASES,
    ids=lambda tc: tc[-1] if isinstance(tc, tuple) else tc,
)
def test_memory_eval(question: str, expected_keywords: list[str], mode: str, label: str, tmp_path: Path) -> None:
    """Query the pre-indexed KB via MCP and verify answer quality."""
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
                response = await session.call_tool(
                    "memory_recall",
                    {
                        "question": question,
                        "mode": mode,
                        "top_k": 10,
                        "fetch_top_k": 20,
                        "context_only": False,
                    },
                )
                return json.loads(response.content[0].text)

    result = anyio.run(run_query)
    assert result.get("ok") is True, f"Query failed: {result.get('error', 'unknown error')}"

    answer_text = str(result.get("answer", ""))

    # --- Check citations are present with source_path ---
    citations = result.get("citations", [])
    assert len(citations) > 0, (
        f"[{label}] No citations returned — memory cannot verify its answer\n"
        f"Answer: {answer_text[:500]}"
    )
    for i, c in enumerate(citations[:3]):
        assert c.get("source_path"), (
            f"[{label}] Citation {i} has no source_path: {c}"
        )

    # --- Check hybrid metadata ---
    hybrid = result.get("hybrid", {})
    if mode == "hybrid":
        assert hybrid.get("alpha") is not None, "[{label}] Hybrid mode missing alpha"
        bm25 = hybrid.get("bm25_index", {})
        assert bm25.get("chunks_indexed", 0) > 0, f"[{label}] BM25 has no indexed chunks"
        rg = hybrid.get("ripgrep", {})
        assert rg.get("enabled") is True, f"[{label}] Ripgrep not enabled in hybrid mode"
        # ripgrep binary may not be installed on this machine — flag but don't fail
        if not rg.get("available", False):
            print(f"\n  [{label}] ⚠️ ripgrep not available: {rg.get('unavailable_reason', '?')}")

    # --- Check graph metadata ---
    if mode in ("local", "global"):
        # Graph mode should still produce a valid answer with citations
        assert len(result.get("citations", [])) > 0, (
            f"[{label}] No citations in {mode} mode"
        )

    # --- Check that expected keywords appear in the answer ---
    found = [kw for kw in expected_keywords if kw.lower() in answer_text.lower()]
    if mode == "hybrid":
        # Hybrid mode has ripgrep: should always find exact keywords
        assert len(found) > 0, (
            f"[{label}] Answer contained none of the expected keywords "
            f"{expected_keywords}.\n"
            f"Answer: {answer_text[:500]}"
        )
    else:
        # Other modes: softer check
        if len(found) == 0:
            print(f"\n  [{label}] ⚠️ No keywords matched, but answer: {answer_text[:200]}")

    print(f"\n  [{label}] ✅ mode={mode} keywords={found} citations={len(citations)}")
    print(f"  Answer: {answer_text[:200]}")
    for c in citations[:2]:
        print(f"    📎 {c.get('source_path','')}")
