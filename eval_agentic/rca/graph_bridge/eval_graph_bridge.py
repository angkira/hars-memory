#!/usr/bin/env python3
"""Offline measurement of the `G_chunks` graph/citation bridge channel.

Answers one question only: **does the bridge channel beat the tuned fusion
baseline, on which sets, and at what latency?** Nothing here is a gate — the
numbers land in `metrics.json` next to this file, and `REPORT.md` quotes them.

Baseline (the current tuned config, `8e79bb5`):
    --top-k 12 --alpha 0.15 --pool-multiplier 2

Variants measured per set:
    fusion            hand-tuned dense+sparse fusion, first `top_k` CHUNKS
    fusion_rog        production default: ranked-ripgrep gate, then the window
    fusion_graph      fusion + graph bridge merged into the pre-truncation pool
    fusion_rog_graph  the same merge, then the ranked-ripgrep gate

The graph bridge is merged exactly where `_compute_hybrid_block` merges it: into
the FULL fused pool, before the gate and before the window is cut. Scoring, seed
extraction and traversal all come from `hars_memory.retrieval.graph_bridge`; this
script does not reimplement the channel.

Reported beyond the four headline metrics:
  * `answer_in_window` — literal substring containment of the `answer` field in
    the returned chunks (hard_1k only; the root-cause report's product metric).
  * `all@k` — every gold document inside the window (hard_1k's 2-gold queries).
  * bridge telemetry — how often the channel fires, how many of its candidates
    land in the final top-k, and how many queries it adds a gold document to.
  * per-stratum deltas on hard_1k, because an average over 8 strata that mean
    opposite things is not a measurement.

CPU only, read-only on the index (this script never writes to the index dirs).

Usage:
    set -a; source eval_agentic-data/bench_env.sh; set +a
    export HARS_MEMORY_BM25_CACHE_DIR=~/.local/share/hars-graphrag/bm25_cache_compact_20260903
    export HARS_MEMORY_RIPGREP_ROOTS=/home/angkira/mem-recovery/r7_stage/corpus_full
    .venv/bin/python eval_agentic/rca/graph_bridge/eval_graph_bridge.py \
        --top-k 12 --alpha 0.15 --pool-multiplier 2 \
        --out eval_agentic/rca/graph_bridge/metrics.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import yaml  # type: ignore[import-not-found]

from hars_memory.eval.channels import install_query_embedding_cache
from hars_memory.eval.ltr_dataset import build_pool, load_query_records, matches_gold
from hars_memory.eval.metrics import dedupe_preserve_order, ndcg_at_k, recall_at_k, reciprocal_rank
from hars_memory.retrieval import graph_bridge
from hars_memory.retrieval.fusion import apply_ripgrep_gate

logger = logging.getLogger("rca.graph_bridge.eval")

DEFAULT_SETS = (
    "eval_agentic-data/retrieval_queries_current_gcfixed.yaml",
    "eval_agentic-data/relational_gold.yaml",
    "eval_agentic-data/hard_1k/hard_1k_eval.yaml",
)
K_VALUES = (1, 5, 10)
NDCG_K = 10
BASE_VARIANTS = ("fusion", "fusion_rog", "fusion_graph", "fusion_rog_graph")


# ---------------------------------------------------------------------------
# Per-query bookkeeping
# ---------------------------------------------------------------------------


class VariantStats:
    def __init__(self) -> None:
        self.recall: dict[int, list[float]] = {k: [] for k in K_VALUES}
        self.ndcg: list[float] = []
        self.rr: list[float] = []
        self.any_at_k: list[float] = []
        self.all_at_k: list[float] = []
        self.answer_in_window: list[float] = []
        self.docs_in_window: list[int] = []
        self.latency_ms: list[float] = []
        # Per-query values for the metrics a paired bootstrap is run on —
        # paired CIs are the only honest interval on n=300/n=48 where the
        # unpaired Wilson interval (±5.4 / ±12.8 pts) swamps the effect.
        self.per_query: dict[str, list[float]] = {
            "recall@1": [],
            "recall@10": [],
            "ndcg@10": [],
            "mrr": [],
            "answer_in_window": [],
        }

    def add(
        self,
        ranked_docs: list[str],
        gold: set[str],
        *,
        window_texts: Sequence[str],
        answer: str | None,
    ) -> None:
        if not gold:
            return
        hits = {k: recall_at_k(ranked_docs, gold, k) for k in K_VALUES}
        ndcg = ndcg_at_k(ranked_docs, gold, NDCG_K)
        rr = reciprocal_rank(ranked_docs, gold)
        for k, value in hits.items():
            self.recall[k].append(value)
            if k in (1, 10):
                self.per_query[f"recall@{k}"].append(value)
        self.ndcg.append(ndcg)
        self.rr.append(rr)
        self.per_query["ndcg@10"].append(ndcg)
        self.per_query["mrr"].append(rr)
        in_window = {doc for doc in ranked_docs[:NDCG_K]}
        self.any_at_k.append(1.0 if in_window & gold else 0.0)
        self.all_at_k.append(1.0 if gold <= in_window else 0.0)
        self.docs_in_window.append(len(ranked_docs[:NDCG_K]))
        if answer:
            needle = " ".join(answer.split()).casefold()
            hit = any(needle in " ".join(text.split()).casefold() for text in window_texts)
            self.answer_in_window.append(1.0 if hit else 0.0)
            self.per_query["answer_in_window"].append(1.0 if hit else 0.0)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in K_VALUES:
            out[f"recall@{k}"] = _mean(self.recall[k])
        out[f"ndcg@{NDCG_K}"] = _mean(self.ndcg)
        out["mrr"] = _mean(self.rr)
        out["any@10"] = _mean(self.any_at_k)
        out["all@10"] = _mean(self.all_at_k)
        if self.answer_in_window:
            out["answer_in_window"] = _mean(self.answer_in_window)
        out["docs_in_window"] = _mean(self.docs_in_window)
        if self.latency_ms:
            ordered = sorted(self.latency_ms)
            out["latency_ms_mean"] = round(statistics.mean(ordered), 3)
            out["latency_ms_p95"] = round(
                ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3
            )
        return out


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], *, resamples: int = 10000, seed: int = 20261020
) -> dict[str, Any]:
    """Mean paired difference (a - b) with a 95 % bootstrap interval.

    Deterministic (fixed RNG seed) so the report's interval is reproducible.
    """
    import random

    diffs = [x - y for x, y in zip(a, b) if True]
    if not diffs:
        return {"mean_delta": None}
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return {
        "n": n,
        "mean_delta": round(statistics.mean(diffs), 4),
        "ci95_low": round(means[int(0.025 * resamples)], 4),
        "ci95_high": round(means[int(0.975 * resamples)], 4),
        "delta_pts": round(statistics.mean(diffs) * 100, 2),
    }


def _canonical(ranked_docs: list[str], gold_docs: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for doc in ranked_docs:
        match = next((g for g in gold_docs if matches_gold(doc, g)), None)
        out.append(match if match is not None else doc)
    return out


def _window(pool: Sequence[Any], top_k: int) -> list[Any]:
    return list(pool[:top_k])


def _doc_order(pool: Sequence[Any], top_k: int) -> list[str]:
    return dedupe_preserve_order(
        [chunk.file_path for chunk in _window(pool, top_k) if chunk.file_path]
    )


# ---------------------------------------------------------------------------
# Query metadata (stratum / answer) from the YAML itself, keyed by id
# ---------------------------------------------------------------------------


def load_query_meta(path: Path) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("queries") or data.get("questions") or []
    meta: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        qid = str(entry.get("id") or entry.get("query_id") or index)
        meta[qid] = {
            "stratum": str(entry.get("stratum") or entry.get("type") or "unknown"),
            "answer": entry.get("answer") or entry.get("answer_span"),
        }
    return meta


# ---------------------------------------------------------------------------
# Main measurement
# ---------------------------------------------------------------------------


async def measure(
    *,
    query_files: Sequence[Path],
    top_k: int,
    alpha: float,
    pool_multiplier: int,
    index_dir: str | None,
    weights: Sequence[float],
    limits: Sequence[int],
    shipped_weight: float,
    shipped_limit: int,
    max_hops: int = graph_bridge.MAX_HOPS,
) -> dict[str, Any]:
    from hars_memory.retrieval.bm25_index import get_or_build_index
    from hars_memory.server.lightrag_init import DEFAULT_WORKING_DIR, create_lightrag

    working_dir = str(
        index_dir or os.environ.get("HARS_MEMORY_INDEX_DIR") or DEFAULT_WORKING_DIR
    )
    logger.info("index: %s", working_dir)
    store = graph_bridge.load_store(working_dir)

    rag = create_lightrag()
    await rag.initialize_storages()
    install_query_embedding_cache(rag)
    cache_dir = os.environ.get("HARS_MEMORY_BM25_CACHE_DIR") or str(
        Path(working_dir).parent / "bm25_cache"
    )
    bm25_index, _ = await asyncio.to_thread(get_or_build_index, working_dir, cache_dir)

    report: dict[str, Any] = {
        "index_dir": working_dir,
        "config": {
            "top_k": top_k,
            "alpha": alpha,
            "pool_multiplier": pool_multiplier,
            "pool_size": max(top_k * pool_multiplier, top_k),
            "shipped_weight": shipped_weight,
            "shipped_limit": shipped_limit,
            "weight_sweep": list(weights),
            "limit_sweep": list(limits),
            "ripgrep_roots": os.environ.get("HARS_MEMORY_RIPGREP_ROOTS", ""),
            "graph_bridge_constants": {
                "hub_fanout_cap": graph_bridge.HUB_FANOUT_CAP,
                "hop_decay": graph_bridge.HOP_DECAY,
                "bridge_degree_cap": graph_bridge.BRIDGE_DEGREE_CAP,
                "citation_bonus": graph_bridge.CITATION_BONUS,
                "max_hops": max_hops,
                "limit": graph_bridge.DEFAULT_LIMIT,
            },
        },
        "sets": {},
    }

    for query_file in query_files:
        path = Path(query_file)
        records = [r for r in load_query_records(path) if r.gold_docs]
        meta = load_query_meta(path)
        logger.info("%s: %d scored queries", path.name, len(records))

        variants = {name: VariantStats() for name in BASE_VARIANTS}
        grid = [(weight, limit) for weight in weights for limit in limits]
        sweep = {
            f"w{weight}_n{limit}": {
                "fusion_graph": VariantStats(),
                "fusion_rog_graph": VariantStats(),
            }
            for weight, limit in grid
        }
        per_stratum: dict[str, dict[str, VariantStats]] = {}
        pool_latencies: list[float] = []
        bridge_stats = {
            "queries": 0,
            "fired": 0,
            "candidates_returned": 0,
            "candidates_in_top_k": 0,
            "candidates_that_are_gold": 0,
            "queries_gaining_gold_doc": 0,
            "queries_losing_gold_doc": 0,
            "candidate_doc_is_answer": 0,
            "answer_doc_added_by_bridge": 0,
            "latencies_ms": [],
            "scored_candidates": [],
            "seed_entities": [],
        }

        # Warm the process before timing anything: production is a long-running
        # server, so a cold-start first query (graph degree cache, regex caches,
        # JSON page faults) is not the marginal cost this channel adds.
        warm_pools = []
        for warm_record in records[:5]:
            warm_bundle = await build_pool(
                rag,
                bm25_index,
                warm_record.question,
                top_k=top_k,
                alpha=alpha,
                pool_multiplier=pool_multiplier,
                annotate_ripgrep=False,
            )
            warm_pools.append((warm_record.question, warm_bundle.pool[: graph_bridge.ANCHOR_WINDOW]))
        for _ in range(3):
            for warm_question, warm_anchors in warm_pools:
                graph_bridge.bridge_candidates(
                    store, warm_question, warm_anchors, max_hops=max_hops
                )

        for record in records:
            started = time.perf_counter()
            bundle = await build_pool(
                rag,
                bm25_index,
                record.question,
                top_k=top_k,
                alpha=alpha,
                pool_multiplier=pool_multiplier,
                annotate_ripgrep=True,
                ll_keywords=record.ll_keywords,
            )
            pool_latencies.append((time.perf_counter() - started) * 1000.0)
            pool = bundle.pool
            if not pool:
                continue

            gold = set(record.gold_docs)
            answer = meta.get(record.query_id, {}).get("answer")
            stratum = meta.get(record.query_id, {}).get("stratum", "unknown")

            # -- the channel, measured through the production merge path ------
            bridge_started = time.perf_counter()
            candidates, bridge_report = graph_bridge.bridge_candidates(
                store,
                record.question,
                pool[: graph_bridge.ANCHOR_WINDOW],
                weight=1.0,
                max_hops=max_hops,
            )
            bridge_ms = (time.perf_counter() - bridge_started) * 1000.0
            shipped_pool, shipped_report = graph_bridge.augment_fused_pool(
                pool,
                record.question,
                store=store,
                weight=shipped_weight,
                limit=shipped_limit,
                candidates=[
                    replace(candidate, fused_score=shipped_weight * candidate.fused_score)
                    for candidate in candidates[:shipped_limit]
                ],
            )

            gated = apply_ripgrep_gate(
                pool, bundle.ripgrep_hits_by_basename, top_k=top_k, mode="ranked"
            )
            gated_graph = apply_ripgrep_gate(
                shipped_pool, bundle.ripgrep_hits_by_basename, top_k=top_k, mode="ranked"
            )

            ranked = {
                "fusion": _doc_order(pool, top_k),
                "fusion_rog": _doc_order(gated, top_k),
                "fusion_graph": _doc_order(shipped_pool, top_k),
                "fusion_rog_graph": _doc_order(gated_graph, top_k),
            }
            texts = {
                "fusion": [chunk.content for chunk in _window(pool, top_k)],
                "fusion_rog": [chunk.content for chunk in _window(gated, top_k)],
                "fusion_graph": [chunk.content for chunk in _window(shipped_pool, top_k)],
                "fusion_rog_graph": [chunk.content for chunk in _window(gated_graph, top_k)],
            }
            for name in BASE_VARIANTS:
                canonical = _canonical(ranked[name], record.gold_docs)
                variants[name].add(
                    canonical, gold, window_texts=texts[name], answer=answer
                )
            variants["fusion_graph"].latency_ms.append(bridge_ms)

            for weight, limit in grid:
                scaled = [
                    replace(candidate, fused_score=weight * candidate.fused_score)
                    for candidate in candidates[:limit]
                ]
                merged, _ = graph_bridge.augment_fused_pool(
                    pool,
                    record.question,
                    store=store,
                    weight=weight,
                    limit=limit,
                    candidates=scaled,
                )
                merged_gated = apply_ripgrep_gate(
                    merged, bundle.ripgrep_hits_by_basename, top_k=top_k, mode="ranked"
                )
                for variant_name, variant_pool in (
                    ("fusion_graph", merged),
                    ("fusion_rog_graph", merged_gated),
                ):
                    stats = sweep[f"w{weight}_n{limit}"][variant_name]
                    stats.add(
                        _canonical(_doc_order(variant_pool, top_k), record.gold_docs),
                        gold,
                        window_texts=[chunk.content for chunk in _window(variant_pool, top_k)],
                        answer=answer,
                    )

            # -- per-stratum (hard_1k only has strata that differ) ------------
            bucket = per_stratum.setdefault(
                stratum, {name: VariantStats() for name in BASE_VARIANTS}
            )
            for name in BASE_VARIANTS:
                bucket[name].add(
                    _canonical(ranked[name], record.gold_docs),
                    gold,
                    window_texts=texts[name],
                    answer=answer,
                )

            # -- bridge telemetry --------------------------------------------
            bridge_stats["queries"] += 1
            bridge_stats["latencies_ms"].append(bridge_ms)
            bridge_stats["scored_candidates"].append(bridge_report.get("scored", 0))
            bridge_stats["seed_entities"].append(
                bridge_report.get("seed_question_entities", 0)
                + bridge_report.get("seed_anchor_entities", 0)
            )
            if candidates:
                bridge_stats["fired"] += 1
            window_ids = {chunk.chunk_id for chunk in _window(shipped_pool, top_k)}
            landed = [c for c in candidates if c.chunk_id in window_ids]
            bridge_stats["candidates_returned"] += len(candidates)
            bridge_stats["candidates_in_top_k"] += len(landed)
            bridge_stats["candidates_that_are_gold"] += sum(
                1 for c in landed if any(matches_gold(c.file_path, g) for g in record.gold_docs)
            )
            base_gold = {d for d in ranked["fusion"] if d in gold}
            graph_gold = {d for d in ranked["fusion_graph"] if d in gold}
            if graph_gold - base_gold:
                bridge_stats["queries_gaining_gold_doc"] += 1
            if base_gold - graph_gold:
                bridge_stats["queries_losing_gold_doc"] += 1
            answer_gold = record.gold_docs[-1] if record.gold_docs else None
            if answer_gold:
                if any(matches_gold(c.file_path, answer_gold) for c in landed):
                    bridge_stats["candidate_doc_is_answer"] += 1
                answer_in_base = any(matches_gold(doc, answer_gold) for doc in ranked["fusion"])
                answer_in_graph = any(
                    matches_gold(doc, answer_gold) for doc in ranked["fusion_graph"]
                )
                if answer_in_graph and not answer_in_base:
                    bridge_stats["answer_doc_added_by_bridge"] += 1

        latencies = sorted(bridge_stats.pop("latencies_ms"))
        scored = bridge_stats.pop("scored_candidates")
        seeds = bridge_stats.pop("seed_entities")
        n = max(bridge_stats["queries"], 1)
        bridge_stats.update(
            {
                "fire_rate": round(bridge_stats["fired"] / n, 4),
                "mean_candidates_returned": round(bridge_stats["candidates_returned"] / n, 2),
                "mean_candidates_in_top_k": round(
                    bridge_stats["candidates_in_top_k"] / n, 2
                ),
                "candidates_in_top_k_per_firing_query": round(
                    bridge_stats["candidates_in_top_k"] / max(bridge_stats["fired"], 1), 2
                ),
                "latency_ms_mean": round(statistics.mean(latencies), 3) if latencies else None,
                "latency_ms_p50": round(latencies[len(latencies) // 2], 3) if latencies else None,
                "latency_ms_p95": round(
                    latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))], 3
                )
                if latencies
                else None,
                "latency_ms_max": round(latencies[-1], 3) if latencies else None,
                "mean_scored_candidates": round(statistics.mean(scored), 1) if scored else None,
                "mean_seed_entities": round(statistics.mean(seeds), 1) if seeds else None,
            }
        )

        bootstrap = {}
        if variants.get("fusion") is not None and variants.get("fusion_graph") is not None:
            for metric in ("recall@1", "recall@10", "ndcg@10", "mrr", "answer_in_window"):
                baseline_values = variants["fusion"].per_query.get(metric) or []
                graph_values = variants["fusion_graph"].per_query.get(metric) or []
                if len(baseline_values) == len(graph_values) and baseline_values:
                    bootstrap[f"fusion_graph_vs_fusion:{metric}"] = paired_bootstrap(
                        graph_values, baseline_values
                    )
        if variants.get("fusion_rog") is not None and variants.get("fusion_rog_graph") is not None:
            for metric in ("recall@1", "recall@10", "answer_in_window"):
                baseline_values = variants["fusion_rog"].per_query.get(metric) or []
                graph_values = variants["fusion_rog_graph"].per_query.get(metric) or []
                if len(baseline_values) == len(graph_values) and baseline_values:
                    bootstrap[f"fusion_rog_graph_vs_fusion_rog:{metric}"] = paired_bootstrap(
                        graph_values, baseline_values
                    )

        report["sets"][path.name] = {
            "n_queries": len(records),
            "variants": {name: stats.summary() for name, stats in variants.items()},
            "paired_bootstrap": bootstrap,
            "pool_latency_ms": {
                "mean": round(statistics.mean(pool_latencies), 3) if pool_latencies else None,
                "p95": round(
                    sorted(pool_latencies)[
                        min(len(pool_latencies) - 1, int(0.95 * len(pool_latencies)))
                    ],
                    3,
                )
                if pool_latencies
                else None,
            },
            "sweep": {
                key: {name: stats.summary() for name, stats in per_variant.items()}
                for key, per_variant in sweep.items()
            },
            "bridge": bridge_stats,
            "per_stratum": {
                stratum: {name: stats.summary() for name, stats in bucket.items()}
                for stratum, bucket in sorted(per_stratum.items())
            },
        }
    return report


def _print_summary(report: dict[str, Any]) -> None:
    for set_name, payload in report["sets"].items():
        print(f"\n=== {set_name} ({payload['n_queries']} queries) ===")
        header = (
            f"{'variant':<18}{'r@1':>7}{'r@5':>7}{'r@10':>7}"
            f"{'ndcg10':>8}{'mrr':>7}{'ans@10':>8}{'ms':>8}"
        )
        print(header)
        for name, metrics in payload["variants"].items():
            print(
                f"{name:<18}{_fmt(metrics.get('recall@1')):>7}{_fmt(metrics.get('recall@5')):>7}"
                f"{_fmt(metrics.get('recall@10')):>7}{_fmt(metrics.get('ndcg@10')):>8}"
                f"{_fmt(metrics.get('mrr')):>7}{_fmt(metrics.get('answer_in_window')):>8}"
                f"{_fmt(metrics.get('latency_ms_mean')):>8}"
            )
        bridge = payload["bridge"]
        print(
            f"bridge: fired {bridge['fire_rate']:.1%} of queries, "
            f"{bridge['mean_candidates_in_top_k']:.2f} candidates/query land in top-k, "
            f"{bridge['latency_ms_mean']:.2f} ms mean / {bridge['latency_ms_p95']:.2f} ms p95"
        )
        print(
            f"        gold doc gained in {bridge['queries_gaining_gold_doc']} queries, "
            f"lost in {bridge['queries_losing_gold_doc']}, "
            f"answer doc added in {bridge['answer_doc_added_by_bridge']}"
        )


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sets", type=Path, nargs="+", default=[Path(p) for p in DEFAULT_SETS])
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--pool-multiplier", type=int, default=2)
    parser.add_argument("--index-dir", type=str, default=None)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--limits", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--shipped-weight", type=float, default=0.25)
    parser.add_argument("--shipped-limit", type=int, default=20)
    parser.add_argument("--max-hops", type=int, default=graph_bridge.MAX_HOPS)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("metrics.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    report = asyncio.run(
        measure(
            query_files=args.sets,
            top_k=args.top_k,
            alpha=args.alpha,
            pool_multiplier=args.pool_multiplier,
            index_dir=args.index_dir,
            weights=args.weights,
            limits=args.limits,
            shipped_weight=args.shipped_weight,
            shipped_limit=args.shipped_limit,
            max_hops=args.max_hops,
        )
    )
    _print_summary(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
