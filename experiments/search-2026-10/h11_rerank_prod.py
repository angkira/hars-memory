"""Measure production recall with the HTTP reranker off and enabled."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from common import load_cases, load_env_file
from metrics import compare, load_results, save_results, score_case

INDEX_DIR = Path("/tmp/eval-mig/after")
STAGING_DIR = "/tmp/search-exp/rr_staging"
BM25_CACHE_DIR = "/tmp/search-exp/rr_bm25"
FLAT_CACHE_DIR = "/tmp/search-exp/rr_flat"
METRIC_KEYS = (
    "file_hit@1", "file_hit@3", "file_hit@5", "fact@1", "fact@3", "fact@5",
    "fact_rr", "chars@5",
)


def _configure_runtime(backend: str, url: str) -> None:
    """Mirror the isolated offline runtime in h78_graph without touching live data."""
    load_env_file()
    import os

    os.environ.update({
        "HARS_MEMORY_INDEX_DIR": str(INDEX_DIR),
        "HARS_MEMORY_STAGING_DIR": STAGING_DIR,
        "HARS_MEMORY_BM25_CACHE_DIR": BM25_CACHE_DIR,
        "HARS_MEMORY_FLAT_DENSE_CACHE_DIR": FLAT_CACHE_DIR,
        "HARS_MEMORY_RIPGREP_CHANNEL": "0",
        "HARS_MEMORY_FLAT_CHANNEL": "0",
        "HARS_MEMORY_EMBED_MODEL": "unsloth/embeddinggemma-300m",
        "HARS_MEMORY_EMBED_LOCAL_FILES_ONLY": "1",
        "HF_HUB_OFFLINE": "1",
        "HARS_MEMORY_RERANK_BACKEND": backend,
        "HARS_MEMORY_RERANK_HTTP_URL": url,
    })


def _load_chunk_records() -> dict[str, dict[str, Any]]:
    path = INDEX_DIR / "kv_store_text_chunks.json"
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected chunk metadata mapping at {path}")
    return data


def _latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": p95,
    }


async def _recall(args: dict[str, Any]) -> dict[str, Any]:
    from hars_memory import mcp_server

    result = await mcp_server.call_tool("memory_recall", args)
    if not result:
        raise RuntimeError("memory_recall returned no content")
    text = getattr(result[0], "text", None)
    if not isinstance(text, str):
        raise TypeError("memory_recall did not return TextContent.text")
    response = json.loads(text)
    if not isinstance(response, dict):
        raise ValueError("memory_recall response was not a JSON object")
    return response


async def _run_measurement(backend: str, url: str, limit: int | None) -> None:
    _configure_runtime(backend, url)
    cases = load_cases()
    if len(cases) != 120:
        raise RuntimeError(f"Expected 120 cases in cases.yaml, found {len(cases)}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        cases = cases[:limit]

    chunks = _load_chunk_records()
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    applied_counts: Counter[bool] = Counter()
    fallback_counts: Counter[str] = Counter()

    for index, case in enumerate(cases, start=1):
        args = {
            "question": str(case["question"]),
            "top_k": 5,
            "context_only": True,
            "mode": "naive",
            "view": "full",
        }
        started = time.perf_counter()
        response = await _recall(args)
        latency_s = time.perf_counter() - started
        latencies.append(latency_s)

        hybrid = response.get("hybrid")
        if not isinstance(hybrid, dict) or not isinstance(hybrid.get("fused_chunks"), list):
            raise ValueError(f"Case {case['id']} did not return hybrid.fused_chunks")
        fused = [chunk for chunk in hybrid["fused_chunks"][:5] if isinstance(chunk, dict)]
        ranked_ids = [str(chunk["chunk_id"]) for chunk in fused if chunk.get("chunk_id") is not None]
        rerank = hybrid.get("rerank")
        if not isinstance(rerank, dict):
            rerank = {}
        applied = bool(rerank.get("applied", False))
        fallback_reason = rerank.get("fallback_reason")
        applied_counts[applied] += 1
        fallback_counts[str(fallback_reason or "none")] += 1
        rows.append({
            "case_id": str(case["id"]),
            **score_case(case, ranked_ids, chunks),
            "latency_s": latency_s,
            "rerank_applied": applied,
            "rerank_fallback_reason": fallback_reason,
        })
        print(f"progress {index}/{len(cases)} case={case['id']} latency={latency_s:.2f}s", flush=True)

    result_name = f"h11_prod_{backend}"
    result_path = save_results(result_name, rows)
    print(f"saved {len(rows)} rows to {result_path}")
    print(f"rerank applied counts: {dict(applied_counts)}")
    print(f"rerank fallback counts: {dict(fallback_counts)}")
    latency = _latency_stats(latencies)
    print(
        f"{backend} latency_s: mean={latency['mean']:.3f} "
        f"median={latency['median']:.3f} p95={latency['p95']:.3f}"
    )


def _compare_results() -> None:
    off = load_results("h11_prod_off")
    http = load_results("h11_prod_http")
    deltas = compare(http, off, METRIC_KEYS)
    print("Metric deltas (http - off), paired bootstrap 95% CI:")
    for key in METRIC_KEYS:
        item = deltas[key]
        print(
            f"  {key}: {float(item['mean']):+.4f} "
            f"[{float(item['lo']):+.4f}, {float(item['hi']):+.4f}] (n={item['n']})"
        )

    for name, rows in (("off", off), ("http", http)):
        latency = [float(row["latency_s"]) for row in rows if row.get("latency_s") is not None]
        stats = _latency_stats(latency)
        print(
            f"{name} latency_s: mean={stats['mean']:.3f} "
            f"median={stats['median']:.3f} p95={stats['p95']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("off", "http"), default="http")
    parser.add_argument("--url", default="http://127.0.0.1:8081/v1/rerank")
    parser.add_argument("--limit", type=int, help="Run only the first N cases (smoke testing).")
    parser.add_argument("--compare", action="store_true", help="Compare saved off/http result files.")
    args = parser.parse_args()

    if args.compare:
        _compare_results()
    else:
        asyncio.run(_run_measurement(args.backend, args.url, args.limit))


if __name__ == "__main__":
    main()