#!/usr/bin/env python3
"""Mandatory offline gate for the learned channel-fusion ranker.

Evaluates, on one or more HELD-OUT gold sets (never used for training or model
selection), the document-level metrics that matter:

* ``fusion``        -- the hand-tuned ``alpha*dense + (1-alpha)*sparse`` order
                       over the full pre-truncation pool (the baseline being
                       replaced).
* ``fusion_rog``    -- the ranked-ripgrep gate over that pool (chunk-local
                       scoring when the question is passed), truncated to
                       top-k ("rerank=off": no cross-encoder).
* ``fusion_div``    -- the FIXED serving window: the ranked gate runs on the
                       FULL pool, source-diversity selection
                       (`retrieval/diversity.py`) ranks and slices it, and
                       additive-channel exclusives are re-appended.
* ``ltr_full``      -- the learned ranker over the FULL pool (recommended wiring).
* ``ltr_gated``     -- the learned ranker over the already top-k-truncated
                       gated pool (what you get if the ranker is hooked LAST).
* ``fusion_graph``  -- hand-tuned fusion + the hub-damped graph/citation bridge
                       channel (`retrieval/graph_bridge.py`, `G_chunks`),
                       merged into the pool BEFORE the window is cut — exactly
                       where `_compute_hybrid_block` merges it.
* ``fusion_rog_graph`` -- the same merge, then the production ranked-ripgrep
                       gate on top.

The window convention is DOCUMENTS by default: `top_k` distinct `file_path`s
are collected from the ranked pool. `--window chunks` reproduces the old
"first top_k chunks, then dedupe" convention (kept only so published numbers
can be re-derived); on an 8.4-chunks/document index it measured far fewer
documents than its name implied.

The two graph variants are skipped (and omitted from the report) when the
index's graph stores cannot be loaded, so the gate still runs on a bare index.
The merge never *replaces* an existing candidate's provenance — see
`graph_bridge.augment_fused_pool`'s contract.

Also reports the ranker's scoring latency (per query and per candidate), which
must stay well under 1 ms/candidate.

Usage:
    python -m hars_memory.eval.ltr_eval \
        --queries eval_agentic-data/retrieval_queries_current_gcfixed.yaml \
                  eval_agentic-data/relational_gold.yaml \
        --model hars_memory/ranking/artifacts/ranker.json \
        --report eval_agentic-data/ltr_offline_gate.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from hars_memory.eval.channels import DEFAULT_POOL_MULTIPLIER, install_query_embedding_cache
from hars_memory.eval.ltr_dataset import (
    PoolBundle,
    QueryRecord,
    build_pool,
    load_query_records,
    matches_gold,
)
from hars_memory.eval.metrics import dedupe_preserve_order, ndcg_at_k, recall_at_k, reciprocal_rank
from hars_memory.ranking.features import LtrCorpusContext
from hars_memory.ranking.rerank import apply_ltr_ranker, load_default_ranker
from hars_memory.retrieval.fusion import apply_ripgrep_gate

logger = logging.getLogger("memory.eval.ltr_eval")

K_VALUES = (1, 10)
NDCG_K = 10
DEFAULT_TOP_K = 10
DEFAULT_ALPHA = 0.5
GRAPH_VARIANTS = ("fusion_graph", "fusion_rog_graph")


@dataclass
class VariantScores:
    recall: dict[int, list[float]] = field(default_factory=lambda: {k: [] for k in K_VALUES})
    ndcg: list[float] = field(default_factory=list)
    rr: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    candidates: int = 0


def _canonicalize(ranked_docs: list[str], gold_docs: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for doc in ranked_docs:
        match = next((gold for gold in gold_docs if matches_gold(doc, gold)), None)
        out.append(match if match is not None else doc)
    return out


def _doc_order(pool: list[Any], top_k: int, *, window: str = "documents") -> list[str]:
    """Document order from a CHUNK ordering.

    ``window="documents"`` (default, the fixed convention): walk the ranked
    pool and collect up to ``top_k`` DISTINCT ``file_path`` values. The metric
    then means what its name says — recall@10 over ten retrieved documents.

    ``window="chunks"`` (the historical convention, kept for reproducing
    published numbers): keep the first ``top_k`` CHUNKS, then dedupe to
    documents. On md_struct (8.4 chunks/document) that window held as few as
    ~3 distinct documents, so its "recall@10" was really recall@~3 documents —
    which is why the md_struct-vs-production comparison had to be re-run.
    """
    if window == "chunks":
        return dedupe_preserve_order(
            [chunk.file_path for chunk in pool[:top_k] if chunk.file_path]
        )
    out: list[str] = []
    seen: set[str] = set()
    for chunk in pool:
        path = getattr(chunk, "file_path", "")
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
        if len(out) >= top_k:
            break
    return out


def _production_selection(
    pool: list[Any],
    ripgrep_hits: dict[str, Any],
    *,
    question: str,
    top_k: int,
    ll_keywords: Sequence[str] | None = None,
    diversity: bool = True,
) -> list[Any]:
    """The serving path's window: gate the FULL pool, diversify, then slice.

    Mirrors `mcp_server` after the 2026-10-20 fixes: the ranked gate runs with
    ``top_k=len(pool)`` so it cannot pre-slice away candidates the diversity
    pass needs, additive-channel exclusives are re-appended after the
    selection, and the selection itself is source-diversified before slicing.
    """
    from hars_memory.retrieval.diversity import select_diverse_fused_chunks

    gated = apply_ripgrep_gate(
        pool,
        ripgrep_hits,
        top_k=len(pool),
        mode="ranked",
        question=question,
        ll_keywords=ll_keywords,
    )
    pool_ids = {chunk.chunk_id for chunk in pool}
    tail = [chunk for chunk in gated if chunk.chunk_id not in pool_ids]
    selected = select_diverse_fused_chunks(gated, top_k, enable=diversity)
    selected_ids = {chunk.chunk_id for chunk in selected}
    selected.extend(chunk for chunk in tail if chunk.chunk_id not in selected_ids)
    return selected


def _score(record: QueryRecord, ranked_docs: list[str], variant: VariantScores) -> None:
    gold = set(record.gold_docs)
    if not gold:
        return
    scored = _canonicalize(ranked_docs, record.gold_docs)
    for k in K_VALUES:
        variant.recall[k].append(recall_at_k(scored, gold, k))
    variant.ndcg.append(ndcg_at_k(scored, gold, NDCG_K))
    variant.rr.append(reciprocal_rank(scored, gold))


def _aggregate(variant: VariantScores, n_queries: int) -> dict[str, Any]:
    out: dict[str, Any] = {"n_queries": n_queries}
    for k in K_VALUES:
        values = variant.recall[k]
        out[f"recall@{k}"] = round(statistics.mean(values), 4) if values else None
    out[f"ndcg@{NDCG_K}"] = round(statistics.mean(variant.ndcg), 4) if variant.ndcg else None
    out["mrr"] = round(statistics.mean(variant.rr), 4) if variant.rr else None
    if variant.latency_ms:
        ordered = sorted(variant.latency_ms)
        out["latency_ms_mean"] = round(statistics.mean(ordered), 4)
        out["latency_ms_p95"] = round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4)
        out["latency_ms_per_candidate"] = round(statistics.mean(ordered) / max(variant.candidates / n_queries, 1), 5)
    return out


def _run_variants(
    record: QueryRecord,
    bundle: PoolBundle,
    *,
    top_k: int,
    context: LtrCorpusContext | None,
    ranker: Any,
    per_variant: dict[str, VariantScores],
    graph_store: Any = None,
    window: str = "documents",
    diversity: bool = True,
) -> None:
    pool = bundle.pool
    if not pool:
        return
    # 1. hand-tuned fusion over the full pool (the baseline)
    _score(record, _doc_order(pool, top_k, window=window), per_variant["fusion"])

    # 2. production default: ranked-ripgrep gate, truncated to top-k ("rerank=off")
    gated = apply_ripgrep_gate(
        pool,
        bundle.ripgrep_hits_by_basename,
        top_k=top_k,
        mode="ranked",
        question=record.question,
        ll_keywords=record.ll_keywords,
    )
    _score(record, _doc_order(gated, top_k, window=window), per_variant["fusion_rog"])

    # 2a. the FIXED serving window: gate the full pool, source-diversify, slice
    selected = _production_selection(
        pool,
        bundle.ripgrep_hits_by_basename,
        question=record.question,
        top_k=top_k,
        ll_keywords=record.ll_keywords,
        diversity=diversity,
    )
    _score(record, _doc_order(selected, top_k, window=window), per_variant["fusion_div"])

    # 2b. graph/citation bridge channel, merged where production merges it:
    # into the full pre-truncation pool, before the gate and the window cut.
    graph_pool = None
    if graph_store is not None:
        from hars_memory.retrieval.graph_bridge import augment_fused_pool

        start = time.perf_counter()
        graph_pool, _report = augment_fused_pool(pool, record.question, store=graph_store)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_variant["fusion_graph"].latency_ms.append(elapsed_ms)
        per_variant["fusion_graph"].candidates += len(graph_pool)
        _score(record, _doc_order(graph_pool, top_k, window=window), per_variant["fusion_graph"])

        start = time.perf_counter()
        gated_graph = apply_ripgrep_gate(
            graph_pool,
            bundle.ripgrep_hits_by_basename,
            top_k=top_k,
            mode="ranked",
            question=record.question,
            ll_keywords=record.ll_keywords,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_variant["fusion_rog_graph"].latency_ms.append(elapsed_ms)
        per_variant["fusion_rog_graph"].candidates += len(gated_graph)
        _score(record, _doc_order(gated_graph, top_k, window=window), per_variant["fusion_rog_graph"])

    # 3 + 4. learned ranker, over the full pool and over the gated pool
    if ranker is not None:
        start = time.perf_counter()
        ranked_full = apply_ltr_ranker(pool, record.question, model=ranker, context=context)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_variant["ltr_full"].latency_ms.append(elapsed_ms)
        per_variant["ltr_full"].candidates += len(pool)
        _score(record, _doc_order(ranked_full, top_k), per_variant["ltr_full"])

        start = time.perf_counter()
        ranked_gated = apply_ltr_ranker(gated, record.question, model=ranker, context=context)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_variant["ltr_gated"].latency_ms.append(elapsed_ms)
        per_variant["ltr_gated"].candidates += len(gated)
        _score(record, _doc_order(ranked_gated, top_k), per_variant["ltr_gated"])


async def run_gate(
    *,
    query_files: list[Path],
    model_path: Path | None,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
    pool_multiplier: int = DEFAULT_POOL_MULTIPLIER,
    index_dir: str | None = None,
    include_graph_bridge: bool = True,
    window: str = "documents",
    diversity: bool = True,
) -> dict[str, Any]:
    from hars_memory.server.lightrag_init import DEFAULT_WORKING_DIR, create_lightrag
    from hars_memory.retrieval.bm25_index import get_or_build_index

    working_dir = str(index_dir or os.environ.get("HARS_MEMORY_INDEX_DIR") or DEFAULT_WORKING_DIR)
    rag = create_lightrag()
    await rag.initialize_storages()
    install_query_embedding_cache(rag)
    cache_dir = os.environ.get("HARS_MEMORY_BM25_CACHE_DIR") or str(Path(working_dir).parent / "bm25_cache")
    bm25_index, _ = await asyncio.to_thread(get_or_build_index, working_dir, cache_dir)
    context = LtrCorpusContext.load(working_dir)
    ranker = load_default_ranker(model_path) if model_path is not None else load_default_ranker()

    graph_store = None
    if include_graph_bridge:
        try:
            from hars_memory.retrieval.graph_bridge import load_store

            graph_store = load_store(working_dir)
        except Exception as exc:  # noqa: BLE001 - optional offline variant
            logger.warning("fusion_graph variants skipped: %s", exc)

    variant_names = ["fusion", "fusion_rog", "fusion_div"]
    if graph_store is not None:
        variant_names.extend(GRAPH_VARIANTS)
    variant_names.extend(["ltr_full", "ltr_gated"])

    report: dict[str, Any] = {
        "index_dir": working_dir,
        "model_path": str(model_path) if model_path else None,
        "ranker_loaded": ranker is not None,
        "graph_bridge_loaded": graph_store is not None,
        "top_k": top_k,
        "alpha": alpha,
        "pool_multiplier": pool_multiplier,
        "window": window,
        "diversity": diversity,
        "sets": {},
    }

    for query_file in query_files:
        records = [r for r in load_query_records(query_file) if r.gold_docs]
        per_variant = {name: VariantScores() for name in variant_names}
        for record in records:
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
            _run_variants(
                record,
                bundle,
                top_k=top_k,
                context=context,
                ranker=ranker,
                per_variant=per_variant,
                graph_store=graph_store,
                window=window,
                diversity=diversity,
            )
        report["sets"][query_file.name] = {
            "n_queries": len(records),
            "variants": {name: _aggregate(v, len(records)) for name, v in per_variant.items()},
        }
    return report


def _print_report(report: dict[str, Any]) -> None:
    for set_name, payload in report["sets"].items():
        print(f"\n=== {set_name} ({payload['n_queries']} queries) ===")
        header = f"{'variant':<14}{'recall@1':>10}{'recall@10':>11}{'ndcg@10':>9}{'mrr':>8}{'ms/query':>10}"
        print(header)
        for name, metrics in payload["variants"].items():
            latency = metrics.get("latency_ms_mean")
            latency_txt = f"{latency:.3f}" if latency is not None else "-"
            print(
                f"{name:<14}{metrics.get('recall@1', 0):>10.4f}{metrics.get('recall@10', 0):>11.4f}"
                f"{metrics.get('ndcg@10', 0):>9.4f}{metrics.get('mrr', 0):>8.4f}{latency_txt:>10}"
            )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, nargs="+", required=True)
    parser.add_argument("--model", type=Path, default=None, help="Ranker manifest; default artifact when omitted.")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--pool-multiplier", type=int, default=DEFAULT_POOL_MULTIPLIER)
    parser.add_argument("--index-dir", type=str, default=None)
    parser.add_argument(
        "--window",
        choices=("documents", "chunks"),
        default="documents",
        help="documents: collect top_k DISTINCT files (fixed). chunks: legacy "
        "top_k chunks then dedupe (only for reproducing pre-2026-10-20 numbers).",
    )
    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Disable the source-diversity selection in the fusion_div variant.",
    )
    parser.add_argument(
        "--no-graph-bridge",
        action="store_true",
        help="Skip the fusion_graph / fusion_rog_graph variants.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    report = asyncio.run(
        run_gate(
            query_files=args.queries,
            model_path=args.model,
            top_k=args.top_k,
            alpha=args.alpha,
            pool_multiplier=args.pool_multiplier,
            index_dir=args.index_dir,
            include_graph_bridge=not args.no_graph_bridge,
            window=args.window,
            diversity=not args.no_diversity,
        )
    )
    _print_report(report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
