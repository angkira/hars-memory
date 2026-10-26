#!/usr/bin/env python3
"""Per-query retrieval trace for miss analysis (READ-ONLY).

For one index + the two gold sets, records for every query:

  * the exact dense (vector) hit list and BM25 hit list, with per-doc ranks
    (so a miss can be attributed to a CHANNEL, not inferred);
  * the ripgrep channel's availability/hits (the channel is file-level);
  * the fused pool, the LTR re-rank, and the doc-level order at several
    chunk-window widths (the published metric uses 10 chunks);
  * gold-doc reachability in the index chunk store (does the doc have ANY
    chunk, and how many), independent of retrieval;
  * for relational gold, whether the gold EVIDENCE chunk id exists.

Outputs are plain JSON/JSONL so every number in the memo is re-derivable
without re-running retrieval.

Usage:
    HARS_MEMORY_INDEX_DIR=<index> HARS_MEMORY_BM25_CACHE_DIR=<cache> \
      python trace_misses.py --label base --out traces/trace_base.json \
                              --candidates traces/candidates_base.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path("/home/angkira/Project/software/hars-longterm-memory")
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from hars_memory.eval.ltr_dataset import (  # noqa: E402
    build_pool,
    load_query_records,
    matches_gold,
)
from hars_memory.eval.metrics import (  # noqa: E402
    dedupe_preserve_order,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from hars_memory.eval.channels import install_query_embedding_cache  # noqa: E402
from hars_memory.ranking.features import LtrCorpusContext, extract_pool_features  # noqa: E402
from hars_memory.ranking.rerank import apply_ltr_ranker, load_default_ranker  # noqa: E402
from hars_memory.retrieval import ripgrep_channel as ripgrep_mod  # noqa: E402
from hars_memory.retrieval.bm25_index import get_or_build_index  # noqa: E402
from hars_memory.retrieval.fusion import apply_ripgrep_gate  # noqa: E402
from hars_memory.server.lightrag_init import create_lightrag  # noqa: E402

QUERY_SETS = [
    REPO / "eval_agentic-data/retrieval_queries_current_gcfixed.yaml",
    REPO / "eval_agentic-data/relational_gold.yaml",
]
TOP_K = 10
ALPHA = 0.5
POOL_MULT = 3  # hars_memory.eval.channels.DEFAULT_POOL_MULTIPLIER (must match the gate)


def _chunk_store(index_dir: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads((index_dir / "kv_store_text_chunks.json").read_text())
    return raw


def _full_docs(index_dir: Path) -> dict[str, Any]:
    return json.loads((index_dir / "kv_store_full_docs.json").read_text())


def _doc_status(index_dir: Path) -> dict[str, Any]:
    return json.loads((index_dir / "kv_store_doc_status.json").read_text())


def _docs_for_gold(chunks: dict[str, dict[str, Any]], gold: str) -> list[str]:
    """Chunk ids whose file_path matches this gold entry."""
    return [cid for cid, ch in chunks.items() if matches_gold(str(ch.get("file_path", "")), gold)]


def _doc_order(pool: list[Any], window: int | None) -> list[str]:
    sliced = pool if window is None else pool[:window]
    return dedupe_preserve_order([c.file_path for c in sliced if c.file_path])


def _rank_of(doc_order: list[str], gold: str) -> int | None:
    for i, path in enumerate(doc_order, start=1):
        if matches_gold(path, gold):
            return i
    return None


def _gold_chunk_rank(hits: list[tuple[str, float]], chunks: dict[str, dict[str, Any]], gold: str) -> int | None:
    """Best (lowest) 1-based rank among channel hits belonging to `gold`."""
    for i, (cid, _score) in enumerate(hits, start=1):
        ch = chunks.get(cid)
        if ch is None:
            continue
        if matches_gold(str(ch.get("file_path", "")), gold):
            return i
    return None


def _channel_hits_dense(entries: list[dict[str, Any]]) -> list[tuple[str, float]]:
    return [(str(e["id"]), float(e.get("distance", 0.0))) for e in entries]


def _channel_hits_sparse(hits: list[Any]) -> list[tuple[str, float]]:
    return [(h.chunk_id, float(h.score)) for h in hits]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test only: stop after N queries per set.")
    args = ap.parse_args()

    index_dir = Path(os.environ["HARS_MEMORY_INDEX_DIR"])
    chunks = _chunk_store(index_dir)
    full_docs = _full_docs(index_dir)
    doc_status = _doc_status(index_dir)
    print(f"[{args.label}] chunks={len(chunks)} full_docs={len(full_docs)}", file=sys.stderr)

    rag = create_lightrag()
    await rag.initialize_storages()
    install_query_embedding_cache(rag)
    cache_dir = os.environ.get("HARS_MEMORY_BM25_CACHE_DIR") or str(index_dir.parent / "bm25_cache")
    bm25_index, _stats = await asyncio.to_thread(get_or_build_index, str(index_dir), cache_dir)
    context = LtrCorpusContext.load(str(index_dir))
    ranker = load_default_ranker(None)
    rg_roots = ripgrep_mod.default_roots()

    records_out: list[dict[str, Any]] = []
    cand_path = args.candidates
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    cand_handle = cand_path.open("w", encoding="utf-8")

    for qf in QUERY_SETS:
        records = [r for r in load_query_records(qf) if r.gold_docs]
        if args.limit:
            records = records[: args.limit]
        for record in records:
            pool_size = max(TOP_K * POOL_MULT, TOP_K)
            pool_bundle = await build_pool(
                rag, bm25_index, record.question,
                top_k=TOP_K, alpha=ALPHA, pool_multiplier=POOL_MULT,
                annotate_ripgrep=True, ll_keywords=record.ll_keywords,
            )
            pool = pool_bundle.pool

            # Independent channel hit lists (what each channel actually returned).
            sparse_hits = bm25_index.search(record.question, pool_size)
            dense_entries = await rag.chunks_vdb.query(record.question, top_k=pool_size)
            rg_result = await ripgrep_mod.asearch(
                record.question, ll_keywords=list(record.ll_keywords), roots=rg_roots, top_k=pool_size
            ) if rg_roots else None

            gated = apply_ripgrep_gate(pool, pool_bundle.ripgrep_hits_by_basename, top_k=TOP_K, mode="ranked")
            ranked_full = apply_ltr_ranker(pool, record.question, model=ranker, context=context)
            ranked_gated = apply_ltr_ranker(gated, record.question, model=ranker, context=context)

            # All candidate features + ranker score, for oracle re-fusion / error analysis.
            rows = extract_pool_features(pool, record.question, context=context)
            scores = ranker.score(rows) if ranker is not None else None
            ltr_rank = {}
            if scores is not None:
                order = sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i].chunk_id))
                ltr_rank = {pool[i].chunk_id: r + 1 for r, i in enumerate(order)}
            for i, (chunk, row) in enumerate(zip(pool, rows, strict=True)):
                cand_handle.write(json.dumps({
                    "set": qf.name,
                    "query_id": record.query_id,
                    "pool_rank": i + 1,
                    "chunk_id": chunk.chunk_id,
                    "file_path": chunk.file_path,
                    "dense_score": chunk.dense_score,
                    "sparse_score": chunk.sparse_score,
                    "dense_norm": chunk.dense_norm,
                    "sparse_norm": chunk.sparse_norm,
                    "fused_score": chunk.fused_score,
                    "ltr_score": None if scores is None else float(scores[i]),
                    "ltr_rank": ltr_rank.get(chunk.chunk_id),
                    "features": list(row.features),
                }) + "\n")

            gold_rows = []
            for gold in record.gold_docs:
                chunk_ids = _docs_for_gold(chunks, gold)
                entries = {
                    "gold": gold,
                    "chunks_in_index": len(chunk_ids),
                    "in_index": bool(chunk_ids),
                    "doc_status": None,
                }
                # doc_status key is the LightRAG doc id; map via full_docs file_path.
                for did, d in full_docs.items():
                    if matches_gold(str(d.get("file_path", "")), gold):
                        entries["doc_status"] = doc_status.get(did, {}).get("status")
                        break
                entries["rank_doc_window10"] = _rank_of(_doc_order(pool, TOP_K), gold)
                entries["rank_doc_window20"] = _rank_of(_doc_order(pool, TOP_K * 2), gold)
                entries["rank_doc_window50"] = _rank_of(_doc_order(pool, 50), gold)
                entries["rank_doc_fullpool"] = _rank_of(_doc_order(pool, None), gold)
                entries["rank_doc_ltr_full"] = _rank_of(_doc_order(ranked_full, None), gold)
                entries["rank_doc_ltr_gated"] = _rank_of(_doc_order(ranked_gated, None), gold)
                entries["sparse_channel_rank"] = _gold_chunk_rank(
                    _channel_hits_sparse(sparse_hits), chunks, gold)
                entries["dense_channel_rank"] = _gold_chunk_rank(
                    _channel_hits_dense(dense_entries), chunks, gold)
                entries["in_pool"] = any(matches_gold(c.file_path, gold) for c in pool)
                entries["pool_chunk_ranks"] = [
                    i + 1 for i, c in enumerate(pool) if matches_gold(c.file_path, gold)
                ]
                entries["in_ltr_top10_chunks"] = _rank_of(_doc_order(ranked_full, TOP_K), gold)
                gold_rows.append(entries)

            records_out.append({
                "set": qf.name,
                "query_id": record.query_id,
                "question": record.question,
                "qtype": record.qtype,
                "ll_keywords": list(record.ll_keywords),
                "gold": gold_rows,
                "pool_size": len(pool),
                "n_dense_hits": len(dense_entries),
                "n_sparse_hits": len(sparse_hits),
                "ripgrep": {
                    "roots": [str(p) for p in rg_roots],
                    "available": bool(rg_result.available) if rg_result is not None else False,
                    "n_hits": len(rg_result.hits) if rg_result is not None else 0,
                },
                "doc_order_window10": _doc_order(pool, TOP_K)[:50],
                "doc_order_fullpool": _doc_order(pool, None)[:50],
                "doc_order_ltr_full": _doc_order(ranked_full, TOP_K)[:50],
                "doc_order_ltr_full_wide": _doc_order(ranked_full, None)[:50],
            })
            print(f"  {record.query_id}: pool={len(pool)} gold={[g['gold'] for g in gold_rows]}",
                  file=sys.stderr)

    cand_handle.close()

    # aggregate over the same metric convention as the published gate
    agg: dict[str, dict[str, float]] = {}
    for set_name in {r["set"] for r in records_out}:
        rows = [r for r in records_out if r["set"] == set_name]
        for variant, key in (("fusion", "doc_order_window10"),
                             ("ltr_full", "doc_order_ltr_full")):
            rs, ns, ms, rr = [], [], [], []
            for r in rows:
                golds = [g["gold"] for g in r["gold"]]
                order = r[key]
                gold_set = set(golds)
                canon = [next((g for g in golds if matches_gold(d, g)), d) for d in order]
                rs.append(recall_at_k(canon, gold_set, 1))
                ns.append(recall_at_k(canon, gold_set, 10))
                ms.append(ndcg_at_k(canon, gold_set, 10))
                rr.append(reciprocal_rank(canon, gold_set))
            agg.setdefault(set_name, {})[variant] = {
                "n": len(rows),
                "recall@1": round(sum(rs) / len(rows), 4),
                "recall@10": round(sum(ns) / len(rows), 4),
                "ndcg@10": round(sum(ms) / len(rows), 4),
                "mrr": round(sum(rr) / len(rows), 4),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "label": args.label,
        "index_dir": str(index_dir),
        "top_k": TOP_K,
        "alpha": ALPHA,
        "pool_multiplier": POOL_MULT,
        "index_stats": {
            "n_chunks": len(chunks),
            "n_full_docs": len(full_docs),
            "n_doc_status": len(doc_status),
        },
        "aggregate": agg,
        "per_query": records_out,
    }, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
