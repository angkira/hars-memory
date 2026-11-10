#!/usr/bin/env python3
"""Per-stage concurrency benchmark for the memory MCP server.

Two modes, because "it is slow under load" is never one number:

``--mode server`` (default)
    N closed-loop MCP clients issue ``memory_recall`` and the script aggregates
    the **server-side stage timings** the response already carries
    (``hybrid.latency_ms``: ``dense_embed``, ``dense_backend``, ``sparse_channel``,
    ``ripgrep_channel``, ``fusion_rank``, ``select``, ``assemble``) against the
    client-observed end-to-end latency. The difference is transport + MCP
    framing + JSON, i.e. what the client pays outside retrieval.

``--mode micro``
    Measures each retrieval stage *in isolation* at 1, 2, 4, 8 and 16
    concurrent calls in threads, and reports the speedup. A stage that does not
    speed up is holding the GIL (or is a serialized external service). Stages:
    the HTTP embedder, the numpy dense scan (real matrix shape), BM25 (real
    index), the LightGBM/MLP LTR ranker (real artifact), and pure-Python fusion.

Examples::

    # server stages at 1/4/8/16 concurrent clients
    python scripts/mcp_stage_bench.py --mode server --url http://127.0.0.1:8232 \
        --approval-token "$TOKEN" --concurrency 1,4,8,16 --repeat 8 \
        --out bench/stages.json

    # which stages release the GIL
    python scripts/mcp_stage_bench.py --mode micro --out bench/micro.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_http_smoke import Client  # noqa: E402

QUESTION = (
    "What did we decide about the retrained md_struct LTR ranker and the "
    "graph bridge damping defaults?"
)

STAGE_KEYS = (
    "dense_embed",
    "dense_backend",
    "sparse_channel",
    "ripgrep_channel",
    "fusion_rank",
    "select",
    "assemble",
)


def _pct(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return round(ordered[index], 2)


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50": _pct(values, 0.50),
        "p95": _pct(values, 0.95),
        "max": round(max(values), 2) if values else None,
        "mean": round(statistics.fmean(values), 2) if values else None,
    }


# ---------------------------------------------------------------------------
# Server mode: aggregate the stage timings the server already reports
# ---------------------------------------------------------------------------


def run_server_mode(args: argparse.Namespace) -> dict[str, Any]:
    bootstrap = Client(args.url, ca=args.ca, timeout=args.timeout)
    if args.no_auth:
        bootstrap.access_token = None
    elif args.access_token:
        bootstrap.access_token = args.access_token
    else:
        if not args.approval_token:
            raise SystemExit("--approval-token (or --access-token / --no-auth) is required")
        bootstrap.login(scope="memory:read memory:write", approval_token=args.approval_token)
    token = bootstrap.access_token

    report: dict[str, Any] = {
        "mode": "server",
        "url": args.url,
        "question": args.question,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": [],
    }

    for concurrency in args.concurrency:
        clients = []
        for _ in range(concurrency):
            client = Client(args.url, ca=args.ca, timeout=args.timeout)
            client.access_token = token
            client.initialize()
            client.recall(args.question, top_k=args.top_k)  # warm this worker's path
            clients.append(client)

        results: list[dict[str, Any]] = []
        lock = threading.Lock()

        def worker(client: Client) -> None:
            for _ in range(args.repeat):
                started = time.perf_counter()
                try:
                    _latency, body = client.recall(args.question, top_k=args.top_k)
                    elapsed = (time.perf_counter() - started) * 1000.0
                except Exception as exc:
                    with lock:
                        results.append({"error": f"{type(exc).__name__}: {exc}"})
                    continue
                stages = (body.get("hybrid") or {}).get("latency_ms") or {}
                with lock:
                    results.append({"total_ms": elapsed, **{k: stages.get(k) for k in STAGE_KEYS}})

        wall_started = time.perf_counter()
        threads = [threading.Thread(target=worker, args=(c,)) for c in clients]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = time.perf_counter() - wall_started

        ok = [r for r in results if "error" not in r]
        errors = [r["error"] for r in results if "error" in r]
        totals = [r["total_ms"] for r in ok]
        stages = {key: [r[key] for r in ok if r.get(key) is not None] for key in STAGE_KEYS}
        stage_sum = sum(statistics.fmean(values) for values in stages.values() if values)
        run = {
            "concurrency": concurrency,
            "repeat": args.repeat,
            "requests": len(ok),
            "errors": errors[:5],
            "wall_s": round(wall, 3),
            "throughput_rps": round(len(ok) / wall, 3) if wall else None,
            "total_ms": _summary(totals),
            "stages_ms": {key: _summary(values) for key, values in stages.items()},
            "unaccounted_ms": round(statistics.fmean(totals) - stage_sum, 2) if totals else None,
        }
        report["runs"].append(run)
        print(
            f"c={concurrency:<3} total p50={run['total_ms']['p50']:>7.1f} "
            f"p95={run['total_ms']['p95']:>7.1f} | "
            + " ".join(
                f"{key}={run['stages_ms'][key]['p50'] or 0:.1f}"
                for key in ("dense_embed", "dense_backend", "sparse_channel", "ripgrep_channel",
                            "fusion_rank", "select", "assemble")
            )
            + f" | unaccounted={run['unaccounted_ms']:.1f} thr={run['throughput_rps']}"
        )
    return report


# ---------------------------------------------------------------------------
# Micro mode: does each stage scale across threads (i.e. release the GIL)?
# ---------------------------------------------------------------------------


def _thread_scaling(fn: Callable[[], Any], concurrency: int, iterations: int) -> dict[str, Any]:
    barrier = threading.Barrier(concurrency)
    latencies: list[float] = []
    lock = threading.Lock()

    def one() -> None:
        barrier.wait()
        for _ in range(iterations):
            started = time.perf_counter()
            fn()
            with lock:
                latencies.append((time.perf_counter() - started) * 1000.0)

    started = time.perf_counter()
    threads = [threading.Thread(target=one) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    return {
        "concurrency": concurrency,
        "calls": len(latencies),
        "wall_s": round(wall, 3),
        "throughput_per_s": round(len(latencies) / wall, 2) if wall else None,
        "p50_ms": _pct(latencies, 0.50),
        "p95_ms": _pct(latencies, 0.95),
    }


def _scaling_curve(
    name: str, fn: Callable[[], Any], levels: list[int], iterations: int
) -> dict[str, Any]:
    curve = [_thread_scaling(fn, level, iterations) for level in levels]
    base = curve[0]["throughput_per_s"] or 1.0
    for point in curve:
        point["speedup_vs_1"] = round((point["throughput_per_s"] or 0) / base, 2)
    return {"stage": name, "curve": curve}


def run_micro_mode(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    levels = args.concurrency
    iterations = args.micro_iterations
    report: dict[str, Any] = {
        "mode": "micro",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": [],
    }

    # 1. The HTTP embedder (the suspected ceiling). Use ONE persistent client
    # per thread: the interesting number is the server's slot behaviour, not
    # TCP setup.
    import httpx

    embed_url = args.embed_url
    if embed_url:
        local = threading.local()

        def embed_client() -> httpx.Client:
            client = getattr(local, "client", None)
            if client is None:
                client = local.client = httpx.Client(timeout=120.0)
            return client

        payload = {"model": args.embed_model, "input": [QUESTION]}

        def embed() -> None:
            response = embed_client().post(f"{embed_url}/embeddings", json=payload)
            response.raise_for_status()

        embed()
        report["stages"].append(_scaling_curve("embed_http", embed, levels, iterations))
        report["embed_url"] = embed_url

    # 2. numpy dense scan over the real matrix shape.
    matrix = np.random.rand(args.vectors, args.dim).astype(np.float32)
    query = np.random.rand(args.dim).astype(np.float32)

    def dense_scan() -> None:
        scores = matrix @ query
        np.argpartition(scores, -args.top_k)[-args.top_k :]

    report["stages"].append(_scaling_curve("numpy_dense_scan", dense_scan, levels, iterations))

    # 3. BM25 (real index if available).
    try:
        from hars_memory.retrieval.bm25_index import get_or_build_index

        index_dir = os.environ.get("HARS_MEMORY_INDEX_DIR", "")
        cache_dir = os.environ.get("HARS_MEMORY_BM25_CACHE_DIR", "")
        bm25, _stats = get_or_build_index(index_dir, cache_dir)

        def bm25_search() -> None:
            bm25.search(QUESTION, 40)

        report["stages"].append(_scaling_curve("bm25_search", bm25_search, levels, iterations))
        report["bm25_chunks"] = bm25.chunk_count
    except Exception as exc:  # pragma: no cover - optional
        report["bm25_error"] = f"{type(exc).__name__}: {exc}"

    # 4. The LTR ranker on a realistic pool.
    try:
        from hars_memory.ranking.rerank import load_default_ranker
        from hars_memory.ranking.features import extract_pool_features
        from hars_memory.retrieval.fusion import ChannelHit, FusedChunk

        ranker = load_default_ranker()
        pool = [
            FusedChunk(
                chunk_id=f"chunk-{i}",
                fused_score=1.0 - i / 100.0,
                dense_score=0.8 - i / 200.0,
                sparse_score=0.5 - i / 300.0,
                content=f"chunk {i} content about the ranker and the graph bridge",
                file_path=f"docs/file{i % 7}.md",
                dense_rank=i,
                sparse_rank=i + 1,
            )
            for i in range(80)
        ]
        context = None
        rows = extract_pool_features(pool, QUESTION, context=context)
        if ranker is not None:
            def ltr_score() -> None:
                ranker.score(rows)

            report["stages"].append(_scaling_curve("ltr_score", ltr_score, levels, iterations))
    except Exception as exc:  # pragma: no cover - optional
        report["ltr_error"] = f"{type(exc).__name__}: {exc}"

    # 5. Pure-Python fusion (the part threads cannot help).
    try:
        from hars_memory.retrieval.fusion import fuse

        dense = {f"c{i}": ChannelHit(score=0.9 - i / 200, content="x", file_path="f") for i in range(120)}
        sparse = {f"c{i}": ChannelHit(score=0.7 - i / 200, content="x", file_path="f") for i in range(120)}

        def fusion() -> None:
            fuse(dense, sparse, 0.5)

        report["stages"].append(_scaling_curve("python_fusion", fusion, levels, iterations))
    except Exception as exc:  # pragma: no cover - optional
        report["fusion_error"] = f"{type(exc).__name__}: {exc}"

    for stage in report["stages"]:
        curve = stage["curve"]
        line = " ".join(f"n={p['concurrency']}:{p['speedup_vs_1']}x" for p in curve)
        print(f"{stage['stage']:<18} {line}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["server", "micro"], default="server")
    parser.add_argument("--url", default=os.environ.get("HARS_MEMORY_MCP_URL", "http://127.0.0.1:8232"))
    parser.add_argument("--ca", default=None)
    parser.add_argument("--access-token", default=None)
    parser.add_argument(
        "--approval-token", default=os.environ.get("HARS_MEMORY_MCP_OAUTH_APPROVAL_TOKEN")
    )
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=8, help="recalls per client (server mode)")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--concurrency", default="1,4,8,16")
    parser.add_argument("--micro-iterations", type=int, default=10)
    parser.add_argument(
        "--embed-url",
        default=os.environ.get("HARS_MEMORY_EMBED_BASE_URL", ""),
        help="embedder /v1 base URL for the micro mode (default: env)",
    )
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("HARS_MEMORY_EMBED_HTTP_MODEL", "google/embeddinggemma-300m"),
    )
    parser.add_argument("--vectors", type=int, default=15409, help="dense scan rows")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    args.concurrency = [int(value) for value in str(args.concurrency).split(",") if value]

    report = run_server_mode(args) if args.mode == "server" else run_micro_mode(args)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": args.out, "mode": report["mode"]}, indent=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
