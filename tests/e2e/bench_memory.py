#!/usr/bin/env python3
"""Comprehensive memory benchmark: full matrix of queries × modes.

Measures for each (query, mode) combination:
  - Latency (seconds)
  - Citation count and source_path validity
  - Keyword recall (expected terms in answer)
  - Confidence score (low_confidence flag)
  - Hybrid/ripgrep metadata (when applicable)
  - Cost estimate (tokens from LLM response)

Outputs a structured JSON report and a human-readable summary table.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

KB_INDEX_DIR = Path("/tmp/hars_memory_lightrag")
KB_ROOT = Path("/Users/iurii.medvedev/Work/sre-support-pier")

# All query modes to test
MODES = ["naive", "hybrid", "local", "global"]

# Benchmark queries: (question, expected_keywords, domain)
BENCH_QUERIES: list[tuple[str, list[str], str]] = [
    # --- Core KB facts ---
    (
        "What is the LiteLLM budget per user per month?",
        ["1000", "USD", "month"],
        "litellm-budget",
    ),
    # --- Cloud infrastructure ---
    (
        "Which cloud provider handles the main SRE K8s clusters in Europe?",
        ["GCP", "Google", "europe-west"],
        "cloud-provider",
    ),
    # --- Harbor evaluations ---
    (
        "What runtime class is used for Harbor evaluations?",
        ["gvisor", "runtime"],
        "harbor-gvisor",
    ),
    # --- KB domains ---
    (
        "What does the KB cover? List the top-level domains.",
        ["aws", "azure", "k8s", "monitoring", "services", "tools"],
        "kb-domains",
    ),
    # --- Team info ---
    (
        "What is the main team's on-call responsibility?",
        ["24/7", "monitoring", "incident", "on-call"],
        "team-mission",
    ),
    # --- Exact keyword (ripgrep channel) ---
    (
        "LiteLLM per-user budget 1000 USD",
        ["1000", "USD", "budget"],
        "ripgrep-exact",
    ),
    # --- Graph relationship ---
    (
        "What is Harbor used for and how does it relate to evaluations?",
        ["eval", "gVisor", "sandbox"],
        "harbor-eval",
    ),
    # --- LiteLLM access ---
    (
        "How do I get access to LiteLLM and what models are available?",
        ["key", "API", "model", "token"],
        "litellm-access",
    ),
    # --- Negative / out-of-domain ---
    (
        "What is the capital of France?",
        [],
        "out-of-domain-geo",
    ),
    (
        "How to make pizza dough?",
        [],
        "out-of-domain-cooking",
    ),
    (
        "What is the meaning of life?",
        [],
        "out-of-domain-philosophy",
    ),
]


@dataclass
class BenchResult:
    query: str
    domain: str
    mode: str
    ok: bool
    latency_s: float
    error: str | None = None
    citations_count: int = 0
    citations_with_source: int = 0
    answer_length_chars: int = 0
    keyword_matches: list[str] = field(default_factory=list)
    low_confidence: bool | None = None
    top_dense_score: float | None = None
    hybrid_enabled: bool | None = None
    ripgrep_enabled: bool | None = None
    ripgrep_available: bool | None = None
    ripgrep_hits: int | None = None
    ripgrep_injected: int | None = None
    ripgrep_latency_ms: float | None = None
    bm25_chunks: int | None = None
    entities_used: int = 0
    relations_used: int = 0
    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None
    cost_estimate_usd: float | None = None


def estimate_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    model: str = "gpt-5.6-luna",
) -> float | None:
    """Estimate cost in USD based on token counts and model pricing."""
    if input_tokens is None or output_tokens is None:
        return None
    pricing = {
        "gpt-5.6-luna": (0.20, 1.20),
        "gpt-4o-mini": (0.15, 0.60),
        "gemini-3.1-flash-lite": (0.25, 1.50),
    }
    base = model.split("/")[-1] if "/" in model else model
    inp_price, out_price = pricing.get(base, (0.20, 1.20))
    return (input_tokens / 1_000_000 * inp_price) + (output_tokens / 1_000_000 * out_price)


async def run_single_query(
    session: ClientSession,
    question: str,
    mode: str,
    expected_keywords: list[str],
    domain: str,
) -> BenchResult:
    """Run a single memory_recall query and collect metrics."""
    start = time.monotonic()
    try:
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
        latency_s = time.monotonic() - start
        result = json.loads(response.content[0].text)

        if not result.get("ok"):
            return BenchResult(
                query=question[:80],
                domain=domain,
                mode=mode,
                ok=False,
                latency_s=latency_s,
                error=result.get("error", "unknown error"),
            )

        answer_text = str(result.get("answer", ""))
        citations = result.get("citations", [])
        citations_with_source = sum(1 for c in citations if c.get("source_path"))
        hybrid = result.get("hybrid", {})
        rg = hybrid.get("ripgrep", {}) if isinstance(hybrid, dict) else {}

        # Extract usage info from result if available
        usage = result.get("usage", {}) or {}
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")

        # Find keyword matches in answer
        found = [kw for kw in expected_keywords if kw.lower() in answer_text.lower()]

        # Confidence
        confidence = hybrid.get("confidence", {}) if isinstance(hybrid, dict) else {}

        # Graph entities/relations
        entities_used = len(result.get("entities_used", []))
        relations_used = len(result.get("relations_used", []))

        # Mutability: if the result dict mutated between the two reads, the
        # second one may be empty — extract in one pass above.
        return BenchResult(
            query=question[:80],
            domain=domain,
            mode=mode,
            ok=True,
            latency_s=latency_s,
            citations_count=len(citations),
            citations_with_source=citations_with_source,
            answer_length_chars=len(answer_text),
            keyword_matches=found,
            low_confidence=confidence.get("low_confidence"),
            top_dense_score=confidence.get("top_dense_score"),
            hybrid_enabled=hybrid.get("enabled") if isinstance(hybrid, dict) else None,
            ripgrep_enabled=rg.get("enabled"),
            ripgrep_available=rg.get("available"),
            ripgrep_hits=rg.get("hits_count"),
            ripgrep_injected=rg.get("injected_count"),
            ripgrep_latency_ms=rg.get("latency_ms"),
            bm25_chunks=hybrid.get("bm25_index", {}).get("chunks_indexed") if isinstance(hybrid, dict) else None,
            entities_used=entities_used,
            relations_used=relations_used,
            llm_input_tokens=input_tokens,
            llm_output_tokens=output_tokens,
            cost_estimate_usd=estimate_cost(input_tokens, output_tokens),
        )

    except Exception as e:
        latency_s = time.monotonic() - start
        return BenchResult(
            query=question[:80],
            domain=domain,
            mode=mode,
            ok=False,
            latency_s=latency_s,
            error=str(e),
        )


def print_summary(results: list[BenchResult]) -> None:
    """Print a human-readable summary table."""
    # Group by domain
    by_domain: dict[str, list[BenchResult]] = {}
    for r in results:
        by_domain.setdefault(r.domain, []).append(r)

    print("\n" + "=" * 100)
    print("  MEMORY BENCHMARK SUMMARY")
    print("=" * 100)

    # Overall stats
    total = len(results)
    passed = sum(1 for r in results if r.ok)
    failed = total - passed
    avg_latency = sum(r.latency_s for r in results if r.ok) / max(passed, 1)
    total_citations = sum(r.citations_count for r in results if r.ok)
    citations_with_source = sum(r.citations_with_source for r in results if r.ok)

    print(f"\n  Total: {total} queries | Passed: {passed} | Failed: {failed}")
    print(f"  Avg latency (passed): {avg_latency:.1f}s")
    print(f"  Total citations: {total_citations} | With source_path: {citations_with_source}")

    # Cost
    costs = [r.cost_estimate_usd for r in results if r.cost_estimate_usd is not None]
    if costs:
        print(f"  Total estimated cost: ${sum(costs):.4f} | Avg per query: ${sum(costs)/len(costs):.4f}")

    # Per-mode summary
    print(f"\n  {'Mode':<10} {'Passed':<8} {'Avg Latency':<14} {'Avg Citations':<14} {'Ripgrep Hits':<14} {'Low Conf':<10}")
    print(f"  {'-'*8:<10} {'-'*6:<8} {'-'*12:<14} {'-'*12:<14} {'-'*12:<14} {'-'*8:<10}")
    for mode in MODES:
        mode_results = [r for r in results if r.mode == mode]
        mode_passed = sum(1 for r in mode_results if r.ok)
        mode_avg_lat = sum(r.latency_s for r in mode_results if r.ok) / max(mode_passed, 1)
        mode_avg_cit = sum(r.citations_count for r in mode_results if r.ok) / max(mode_passed, 1)
        mode_rg_hits = sum(r.ripgrep_hits or 0 for r in mode_results if r.ok) / max(mode_passed, 1)
        mode_low_conf = sum(1 for r in mode_results if r.low_confidence)
        print(f"  {mode:<10} {mode_passed:<8} {mode_avg_lat:<14.1f}s {mode_avg_cit:<14.1f} {mode_rg_hits:<14.1f} {mode_low_conf:<10}")

    # Per-domain detail
    print(f"\n  {'Domain':<22} {'Mode':<10} {'Status':<8} {'Latency':<10} {'Cits':<6} {'KW Hits':<10} {'Low Conf':<10} {'RG Hits':<8}")
    print(f"  {'-'*20:<22} {'-'*8:<10} {'-'*6:<8} {'-'*8:<10} {'-'*4:<6} {'-'*8:<10} {'-'*8:<10} {'-'*6:<8}")
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        kw_str = str(len(r.keyword_matches)) if r.ok else "-"
        lc = "Y" if r.low_confidence else ("N" if r.low_confidence is False else "?")
        rg = str(r.ripgrep_hits or 0) if r.ok else "-"
        print(f"  {r.domain:<22} {r.mode:<10} {status:<8} {r.latency_s:<10.1f}s {r.citations_count:<6} {kw_str:<10} {lc:<10} {rg:<8}")
        if not r.ok and r.error:
            print(f"  {'':22} {'':10} {'ERROR:':8} {r.error[:80]:<60}")

    # Lowest confidence queries
    low_conf = [r for r in results if r.low_confidence]
    if low_conf:
        print(f"\n  ⚠️  Low confidence queries ({len(low_conf)}):")
        for r in low_conf:
            print(f"    {r.domain:<22} {r.mode:<10} dense_score={r.top_dense_score}")

    # Best and worst latency
    ok_results = [r for r in results if r.ok]
    if ok_results:
        best = min(ok_results, key=lambda r: r.latency_s)
        worst = max(ok_results, key=lambda r: r.latency_s)
        print(f"\n  Fastest: {best.domain} / {best.mode} — {best.latency_s:.1f}s")
        print(f"  Slowest: {worst.domain} / {worst.mode} — {worst.latency_s:.1f}s")

    # Ripgrep effectiveness
    rg_enabled = [r for r in results if r.ripgrep_enabled and r.ripgrep_available]
    if rg_enabled:
        total_rg_hits = sum(r.ripgrep_hits or 0 for r in rg_enabled)
        total_rg_injected = sum(r.ripgrep_injected or 0 for r in rg_enabled)
        avg_rg_latency = sum(r.ripgrep_latency_ms or 0 for r in rg_enabled) / len(rg_enabled)
        print(f"\n  Ripgrep channel: {total_rg_hits} total hits, {total_rg_injected} injected, avg {avg_rg_latency:.0f}ms")

    print("\n" + "=" * 100)


async def main() -> None:
    """Run the full benchmark matrix."""
    required = {
        "HARS_MEMORY_QUERY_BASE_URL": os.environ.get("HARS_MEMORY_QUERY_BASE_URL"),
        "HARS_MEMORY_QUERY_MODEL": os.environ.get("HARS_MEMORY_QUERY_MODEL"),
        "HARS_MEMORY_LLM_API_KEY": os.environ.get("HARS_MEMORY_LLM_API_KEY"),
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        print(f"ERROR: missing live LLM configuration: {', '.join(missing)}")
        print("Source config/live-llm-e2e.env before running.")
        sys.exit(1)

    model = os.environ.get("HARS_MEMORY_QUERY_MODEL", "unknown")

    # Create unique temp dirs for this run
    import tempfile

    staging_dir = Path(tempfile.mkdtemp(prefix="hars_bench_staging_"))
    bm25_dir = Path(tempfile.mkdtemp(prefix="hars_bench_bm25_"))

    mcp_env = os.environ.copy()
    mcp_env.update(
        {
            "HARS_MEMORY_INDEX_DIR": str(KB_INDEX_DIR),
            "HARS_MEMORY_STAGING_DIR": str(staging_dir),
            "HARS_MEMORY_BM25_CACHE_DIR": str(bm25_dir),
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
        print(f"ERROR: MCP executable not found: {executable}")
        sys.exit(1)

    params = StdioServerParameters(
        command=str(executable),
        env=mcp_env,
        cwd=str(KB_INDEX_DIR.parent),
    )

    print(f"\n  Memory Benchmark — {len(BENCH_QUERIES)} queries × {len(MODES)} modes = {len(BENCH_QUERIES) * len(MODES)} tests")
    print(f"  Model: {model}")
    print(f"  Index: {KB_INDEX_DIR}")
    print(f"  Ripgrep roots: {KB_ROOT}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_results: list[BenchResult] = []
    total_tests = len(BENCH_QUERIES) * len(MODES)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            test_idx = 0
            for question, expected_keywords, domain in BENCH_QUERIES:
                for mode in MODES:
                    test_idx += 1
                    label = f"[{test_idx}/{total_tests}] {domain} / {mode}"
                    print(f"  {label} ... ", end="", flush=True)

                    result = await run_single_query(
                        session, question, mode, expected_keywords, domain
                    )
                    all_results.append(result)

                    if result.ok:
                        kw_hits = len(result.keyword_matches)
                        kw_str = f"kw={kw_hits}" if kw_hits > 0 else "no-kw"
                        lc_str = "low-conf" if result.low_confidence else ""
                        extras = " | ".join(filter(None, [kw_str, lc_str]))
                        print(f"✅ {result.latency_s:.1f}s cits={result.citations_count} {extras}")
                    else:
                        print(f"❌ {result.latency_s:.1f}s {result.error or 'failed'}")

    # Cleanup
    import shutil

    shutil.rmtree(staging_dir, ignore_errors=True)
    shutil.rmtree(bm25_dir, ignore_errors=True)

    # Print summary
    print_summary(all_results)

    # Save JSON report
    report_path = Path("/tmp/hars_memory_benchmark_results.json")
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "index": str(KB_INDEX_DIR),
        "total_tests": total_tests,
        "passed": sum(1 for r in all_results if r.ok),
        "failed": sum(1 for r in all_results if not r.ok),
        "results": [
            {
                "domain": r.domain,
                "mode": r.mode,
                "ok": r.ok,
                "latency_s": r.latency_s,
                "error": r.error,
                "citations_count": r.citations_count,
                "citations_with_source": r.citations_with_source,
                "answer_length_chars": r.answer_length_chars,
                "keyword_matches": len(r.keyword_matches),
                "low_confidence": r.low_confidence,
                "top_dense_score": r.top_dense_score,
                "ripgrep_hits": r.ripgrep_hits,
                "ripgrep_injected": r.ripgrep_injected,
                "ripgrep_latency_ms": r.ripgrep_latency_ms,
                "bm25_chunks": r.bm25_chunks,
                "entities_used": r.entities_used,
                "relations_used": r.relations_used,
                "cost_estimate_usd": r.cost_estimate_usd,
            }
            for r in all_results
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  Full JSON report saved to: {report_path}")


if __name__ == "__main__":
    anyio.run(main)