#!/usr/bin/env python3
"""Offline search for a bridge ranking/expansion that actually places the answer.

Two experiments, both with retrieval held fixed (one live pool per query):

1. **Aggregation** — same traversal, six ways of turning reached entities into a
   chunk score (`sum`, `max`, `mean`, `sqrt`, `bonus_first`, `max_bonus`).
2. **Expansion sharpness** — the `bonus_first` aggregation under a grid of
   hub fan-out caps (10/20/40/80) and hop depths (1/2). The hypothesis: the
   answer document is reached through a *distinctive* bridge entity, while the
   ~1 500 competing candidate chunks are reached through generic ones, so a
   tighter hub cap should shrink the candidate pool around the answer.

The reported quantity is the rank of the answering document (`gold_docs[-1]`)
inside the channel's own candidate list, measured on the queries where the
answer document is NOT already in the fusion window (the only queries where the
channel can add anything). `answer_in_top20_actionable` is therefore the
channel's realistic ceiling for a 20-candidate injection.

Read-only. Writes `rank_experiment.json` next to this file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import statistics
from pathlib import Path
from typing import Any, Sequence

from hars_memory.eval.channels import install_query_embedding_cache
from hars_memory.eval.ltr_dataset import build_pool, load_query_records, matches_gold
from hars_memory.retrieval import graph_bridge

logger = logging.getLogger("rca.graph_bridge.rank_experiment")

SCORERS = ("sum", "max", "mean", "sqrt", "bonus_first", "max_bonus")
HUB_CAPS = (10, 20, 40, 80)
HOPS = (1, 2)


def _score_variants(
    store: Any,
    question: str,
    anchors: Sequence[Any],
    *,
    max_hops: int = graph_bridge.MAX_HOPS,
    hub_fanout_cap: int = graph_bridge.HUB_FANOUT_CAP,
) -> dict[str, list[tuple[str, float, bool]]]:
    """For one query: chunk -> (score, has_bonus) per scorer, sorted descending."""
    seeds = store.anchor_seeds(question, anchors)
    reached = graph_bridge._traverse(
        store,
        seeds,
        max_hops=max_hops,
        hub_fanout_cap=hub_fanout_cap,
        hop_decay=graph_bridge.HOP_DECAY,
        bridge_degree_cap=graph_bridge.BRIDGE_DEGREE_CAP,
    )
    anchor_paths = {str(getattr(a, "file_path", "") or "") for a in anchors}

    sums: dict[str, float] = {}
    maximum: dict[str, float] = {}
    counts: dict[str, int] = {}
    for entity, (weight, _hop) in reached.items():
        chunk_ids = store.entity_chunks.get(entity)
        if not chunk_ids:
            continue
        contribution = weight / math.log(1.0 + len(chunk_ids))
        for chunk_id in chunk_ids:
            sums[chunk_id] = sums.get(chunk_id, 0.0) + contribution
            if contribution > maximum.get(chunk_id, 0.0):
                maximum[chunk_id] = contribution
            counts[chunk_id] = counts.get(chunk_id, 0) + 1

    bonus: set[str] = set()
    cited = {name.casefold() for name in seeds.cited_files}
    for file_path, chunk_ids in store.file_chunks.items():
        if file_path.replace("\\", "/").rsplit("/", 1)[-1].casefold() in cited:
            bonus.update(chunk_ids)
    for uuid_cf in seeds.rare_ids:
        bonus.update(store.chunks_for_uuid(uuid_cf))

    def filtered(items: dict[str, float]) -> list[tuple[str, float, bool]]:
        rows = [
            (chunk_id, value, chunk_id in bonus)
            for chunk_id, value in items.items()
            if value > 0 and store.chunk_path(chunk_id) not in anchor_paths
        ]
        rows.sort(key=lambda row: (-row[1], row[0]))
        return rows

    out: dict[str, list[tuple[str, float, bool]]] = {}
    out["sum"] = filtered(sums)
    out["max"] = filtered(maximum)
    out["mean"] = filtered({cid: sums[cid] / counts[cid] for cid in sums})
    out["sqrt"] = filtered({cid: sums[cid] / math.sqrt(counts[cid]) for cid in sums})
    out["bonus_first"] = sorted(
        [
            (cid, value, cid in bonus)
            for cid, value in sums.items()
            if store.chunk_path(cid) not in anchor_paths
        ],
        key=lambda row: (not row[2], -row[1], row[0]),
    )
    out["max_bonus"] = sorted(
        [
            (cid, maximum[cid] + (1.0 if cid in bonus else 0.0), cid in bonus)
            for cid in maximum
            if store.chunk_path(cid) not in anchor_paths
        ],
        key=lambda row: (-row[1], row[0]),
    )
    return out


async def run(
    *, query_file: Path, index_dir: str | None, top_k: int, alpha: float, pool_multiplier: int
) -> dict[str, Any]:
    from hars_memory.retrieval.bm25_index import get_or_build_index
    from hars_memory.server.lightrag_init import DEFAULT_WORKING_DIR, create_lightrag

    working_dir = str(index_dir or os.environ.get("HARS_MEMORY_INDEX_DIR") or DEFAULT_WORKING_DIR)
    store = graph_bridge.load_store(working_dir)
    rag = create_lightrag()
    await rag.initialize_storages()
    install_query_embedding_cache(rag)
    cache_dir = os.environ.get("HARS_MEMORY_BM25_CACHE_DIR") or str(
        Path(working_dir).parent / "bm25_cache"
    )
    bm25_index, _ = await asyncio.to_thread(get_or_build_index, working_dir, cache_dir)

    records = [r for r in load_query_records(query_file) if r.gold_docs]
    scorer_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SCORERS}
    grid_rows: dict[str, list[dict[str, Any]]] = {
        f"hub{cap}_hop{hops}": [] for cap in HUB_CAPS for hops in HOPS
    }
    grid_pool_sizes: dict[str, list[int]] = {key: [] for key in grid_rows}

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
        if not bundle.pool:
            continue
        anchors = bundle.pool[: graph_bridge.ANCHOR_WINDOW]
        base_window = [chunk.file_path for chunk in bundle.pool[:top_k]]
        answer_doc = record.gold_docs[-1]
        already = any(matches_gold(path, answer_doc) for path in base_window)

        scored = _score_variants(store, record.question, anchors)
        for name, rows in scored.items():
            rank = next(
                (
                    index
                    for index, (chunk_id, _value, _bonus) in enumerate(rows)
                    if matches_gold(store.chunk_path(chunk_id), answer_doc)
                ),
                None,
            )
            scorer_rows[name].append({"already": already, "rank": rank})

        for cap in HUB_CAPS:
            for hops in HOPS:
                key = f"hub{cap}_hop{hops}"
                rows = _score_variants(
                    store, record.question, anchors, max_hops=hops, hub_fanout_cap=cap
                )["bonus_first"]
                rank = next(
                    (
                        index
                        for index, (chunk_id, _value, _bonus) in enumerate(rows)
                        if matches_gold(store.chunk_path(chunk_id), answer_doc)
                    ),
                    None,
                )
                grid_rows[key].append({"already": already, "rank": rank})
                grid_pool_sizes[key].append(len(rows))

    def summarize(rows: list[dict[str, Any]], pool_sizes: list[int] | None = None) -> dict[str, Any]:
        actionable = [row for row in rows if not row["already"]]
        ranks = [row["rank"] for row in rows if row["rank"] is not None]
        out = {
            "answer_scored_rate": round(len(ranks) / max(len(rows), 1), 4),
            "answer_in_top20_all_queries": round(
                sum(1 for row in rows if row["rank"] is not None and row["rank"] < 20)
                / max(len(rows), 1),
                4,
            ),
            "actionable_queries": len(actionable),
            "answer_in_top5_actionable": round(
                sum(1 for row in actionable if row["rank"] is not None and row["rank"] < 5)
                / max(len(actionable), 1),
                4,
            ),
            "answer_in_top20_actionable": round(
                sum(1 for row in actionable if row["rank"] is not None and row["rank"] < 20)
                / max(len(actionable), 1),
                4,
            ),
            "median_rank_when_scored": statistics.median(ranks) if ranks else None,
        }
        if pool_sizes:
            out["median_scored_chunks"] = statistics.median(pool_sizes)
        return out

    return {
        "n_queries": len(records),
        "aggregation_at_default_expansion": {
            name: summarize(rows) for name, rows in scorer_rows.items()
        },
        "expansion_grid_bonus_first": {
            key: summarize(rows, grid_pool_sizes[key]) for key, rows in grid_rows.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries", type=Path, default=Path("eval_agentic-data/hard_1k/hard_1k_eval.yaml")
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--pool-multiplier", type=int, default=2)
    parser.add_argument("--index-dir", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("rank_experiment.json"))
    args = parser.parse_args(argv)
    result = asyncio.run(
        run(
            query_file=args.queries,
            index_dir=args.index_dir,
            top_k=args.top_k,
            alpha=args.alpha,
            pool_multiplier=args.pool_multiplier,
        )
    )
    print(json.dumps(result, indent=2))
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
