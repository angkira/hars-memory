"""Evaluate the dense, BM25, and production hybrid retrieval baselines."""

from __future__ import annotations

import asyncio
from typing import Any

from common import load_cases, ws
from metrics import breakdown, load_chunks, save_results, score_case, table
from retr import build_candidates, fuse_hybrid, open_retriever, rank_bm25, rank_dense

TOP_K = 20
METRIC_KEYS = ("file_hit@1", "file_hit@3", "file_hit@5", "fact@1", "fact@3", "fact@5", "fact_rr", "chars@5")


def _fact_in_candidates(case: dict[str, Any], candidate_ids: list[str], chunks: dict[str, dict[str, Any]]) -> bool:
    gold_ids = {str(chunk_id) for chunk_id in case.get("gold_chunk_ids", []) or []}
    gold_fact = ws(str(case.get("gold_fact", "")))
    for chunk_id in candidate_ids:
        if chunk_id in gold_ids:
            return True
        content = ws(str(chunks.get(chunk_id, {}).get("content", "")))
        if gold_fact and gold_fact in content:
            return True
    return False


def _print_breakdown(rows: list[dict[str, Any]], cases: list[dict[str, Any]], by: str) -> None:
    groups = breakdown(rows, cases, by=by)
    keys = ("file_hit@1", "fact@5", "fact_rr")
    print(f"\nBase hybrid by {by}:")
    print("| Group | " + " | ".join(keys) + " |")
    print("|---|" + "|".join("---:" for _ in keys) + "|")
    for group, stats in groups.items():
        cells = [f"{float(stats[key]['mean']):.3f}" for key in keys]
        print("| " + group + " | " + " | ".join(cells) + " |")


async def _run() -> None:
    cases = load_cases()
    if not cases:
        raise RuntimeError("No cases found in experiments/search-2026-10/cases.yaml")

    chunks = load_chunks()
    await open_retriever()
    candidates = await build_candidates(cases, pool=100, tag="base")
    rows_by_method: dict[str, list[dict[str, Any]]] = {
        "base_dense": [],
        "base_bm25": [],
        "base_hybrid": [],
    }

    for case in cases:
        case_id = str(case["id"])
        record = candidates[case_id]
        ranked_by_method = {
            "base_dense": rank_dense(record["dense"], TOP_K),
            "base_bm25": rank_bm25(record["bm25"], TOP_K),
            "base_hybrid": fuse_hybrid(record["dense"], record["bm25"], alpha=0.5, pool=15, top_k=TOP_K),
        }
        for name, ranked_ids in ranked_by_method.items():
            rows_by_method[name].append({"case_id": case_id, **score_case(case, ranked_ids, chunks)})

    for name, rows in rows_by_method.items():
        save_results(name, rows)

    print("Baseline retrieval metrics (means with bootstrap 95% CIs; Δ rows compare against dense):")
    print(table(["base_dense", "base_bm25", "base_hybrid"], METRIC_KEYS))
    _print_breakdown(rows_by_method["base_hybrid"], cases, "qtype")
    _print_breakdown(rows_by_method["base_hybrid"], cases, "lang")

    gold_ids = {str(chunk_id) for case in cases for chunk_id in case.get("gold_chunk_ids", []) or []}
    indexed_gold = gold_ids.intersection(chunks)
    dense_misses = 0
    bm25_misses = 0
    for case in cases:
        record = candidates[str(case["id"])]
        dense_ids = [chunk_id for chunk_id, _ in record["dense"][:100]]
        sparse_ids = [chunk_id for chunk_id, _ in record["bm25"][:100]]
        dense_misses += not _fact_in_candidates(case, dense_ids, chunks)
        bm25_misses += not _fact_in_candidates(case, sparse_ids, chunks)
    print(
        f"\nGold chunk IDs present in index: {len(indexed_gold)}/{len(gold_ids)}; "
        f"cases with no fact in top-100: dense {dense_misses}/{len(cases)}, "
        f"BM25 {bm25_misses}/{len(cases)}."
    )


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()