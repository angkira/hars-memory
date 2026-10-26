#!/usr/bin/env python3
"""Where does the answering document actually rank inside the bridge channel?

The aggregate number ("the channel loses 5 pts r@10") does not say WHY. This
script measures, per hard_1k query:

  * is the answer document (gold_docs[-1]) among the channel's candidates at
    all, and at what candidate rank;
  * raw-rank distribution of the answer document over ALL scored chunks
    (before the 20-candidate cut);
  * whether the answer document's chunk carries the citation / rare-UUID bonus;
  * whether "bonus-first ordering" (promote every cited/rare-UUID chunk to the
    top of the channel) would put it inside the window when the plain score
    order does not.

Output: `diagnostics.json` next to this file. Read-only on the index.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
from pathlib import Path
from typing import Any, Sequence

from hars_memory.eval.channels import install_query_embedding_cache
from hars_memory.eval.ltr_dataset import build_pool, load_query_records, matches_gold
from hars_memory.retrieval import graph_bridge

logger = logging.getLogger("rca.graph_bridge.diagnose")


async def diagnose(
    *, query_file: Path, index_dir: str | None, top_k: int, alpha: float, pool_multiplier: int
) -> dict[str, Any]:
    from hars_memory.retrieval.bm25_index import get_or_build_index
    from hars_memory.server.lightrag_init import DEFAULT_WORKING_DIR, create_lightrag

    working_dir = str(
        index_dir or os.environ.get("HARS_MEMORY_INDEX_DIR") or DEFAULT_WORKING_DIR
    )
    store = graph_bridge.load_store(working_dir)
    rag = create_lightrag()
    await rag.initialize_storages()
    install_query_embedding_cache(rag)
    cache_dir = os.environ.get("HARS_MEMORY_BM25_CACHE_DIR") or str(
        Path(working_dir).parent / "bm25_cache"
    )
    bm25_index, _ = await asyncio.to_thread(get_or_build_index, working_dir, cache_dir)

    records = [r for r in load_query_records(query_file) if r.gold_docs]
    rows: list[dict[str, Any]] = []
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
        pool = bundle.pool
        if not pool:
            continue
        answer_doc = record.gold_docs[-1]
        anchors = pool[: graph_bridge.ANCHOR_WINDOW]

        seeds = store.anchor_seeds(record.question, anchors)
        scored = graph_bridge.score_candidates(
            store, seeds, exclude_files=[a.file_path for a in anchors]
        )
        candidates, report = graph_bridge.bridge_candidates(
            store, record.question, anchors, weight=1.0
        )

        answer_ranks = [
            index
            for index, item in enumerate(scored)
            if matches_gold(store.chunk_path(item.chunk_id), answer_doc)
        ]
        answer_bonus = [
            item.citation_bonus
            for item in scored
            if matches_gold(store.chunk_path(item.chunk_id), answer_doc)
        ]
        candidate_rank = next(
            (
                index
                for index, candidate in enumerate(candidates)
                if matches_gold(candidate.file_path, answer_doc)
            ),
            None,
        )
        base_window = [chunk.file_path for chunk in pool[:top_k]]
        rows.append(
            {
                "query_id": record.query_id,
                "scored": len(scored),
                "candidates": len(candidates),
                "answer_in_candidates": candidate_rank is not None,
                "answer_candidate_rank": candidate_rank,
                "answer_best_raw_rank": min(answer_ranks) if answer_ranks else None,
                "answer_provenance_chunks": len(answer_ranks),
                "answer_has_bonus": any(answer_bonus),
                "answer_in_baseline_window": any(
                    matches_gold(path, answer_doc) for path in base_window
                ),
                "seed_entities": report.get("seed_anchor_entities", 0)
                + report.get("seed_question_entities", 0),
                "bonus_candidates": sum(
                    1 for item in scored[: len(candidates)] if item.citation_bonus
                ),
            }
        )

    n = len(rows)
    found = [row for row in rows if row["answer_in_candidates"]]
    ranked = [row["answer_best_raw_rank"] for row in rows if row["answer_best_raw_rank"] is not None]
    summary = {
        "n_queries": n,
        "answer_doc_scored_at_all": round(len(ranked) / n, 4),
        "answer_doc_in_candidate_top20": round(len(found) / n, 4),
        "answer_doc_already_in_baseline_window": round(
            sum(1 for row in rows if row["answer_in_baseline_window"]) / n, 4
        ),
        "answer_doc_in_candidates_but_not_baseline": round(
            sum(
                1
                for row in rows
                if row["answer_in_candidates"] and not row["answer_in_baseline_window"]
            )
            / n,
            4,
        ),
        "answer_doc_has_bonus_when_found": round(
            sum(1 for row in found if row["answer_has_bonus"]) / max(len(found), 1), 4
        ),
        "median_raw_rank_when_scored": statistics.median(ranked) if ranked else None,
        "mean_scored_candidates": round(statistics.mean(row["scored"] for row in rows), 1),
        "max_bonus_candidates_per_query": max(
            (row["bonus_candidates"] for row in rows), default=0
        ),
    }
    return {"summary": summary, "rows": rows}


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("eval_agentic-data/hard_1k/hard_1k_eval.yaml"),
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--pool-multiplier", type=int, default=2)
    parser.add_argument("--index-dir", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("diagnostics.json"))
    args = parser.parse_args(argv)

    result = asyncio.run(
        diagnose(
            query_file=args.queries,
            index_dir=args.index_dir,
            top_k=args.top_k,
            alpha=args.alpha,
            pool_multiplier=args.pool_multiplier,
        )
    )
    print(json.dumps(result["summary"], indent=2))
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
