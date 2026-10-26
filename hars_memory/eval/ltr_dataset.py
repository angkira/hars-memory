#!/usr/bin/env python3
"""Export training rows for the learned channel-fusion ranker from the live index.

This does NOT fork the retrieval path: it calls the SAME production modules the
eval harness and the MCP server call (`retrieval.fusion.fuse`,
`retrieval.fusion.ripgrep_scores_by_chunk_id`, `retrieval.bm25_index`,
`retrieval.ripgrep_channel`, the LightRAG chunk store) and only normalizes the
resulting pool into feature rows. Every candidate the pool returns becomes one
row ``{query_id, features, label}``; the writer emits an optional header line
carrying the ``feature_names`` manifest.

GRADED LABELS (documented, deterministic, no LLM)
-------------------------------------------------
* ``2`` -- the chunk belongs to a gold document AND contains the query's
  validating answer span (when the query carries one); for span-less queries,
  every gold-document chunk is ``2``.
* ``1`` -- "same source document" chunk: it belongs to a gold document but is
  not the evidence chunk that contains the span. Only produced for queries that
  carry an ``answer_span``.
* ``0`` -- everything else.

READ-ONLY on the index. Output goes to a gitignored scratch dir by default.

Usage:
    python -m hars_memory.eval.ltr_dataset \
        --queries eval_agentic-data/synthetic_queries_scaled.yaml \
        --out eval_agentic-data/ltr_train.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml  # type: ignore[import-not-found]

from hars_memory.eval.channels import DEFAULT_POOL_MULTIPLIER, install_query_embedding_cache
from hars_memory.ranking.features import LTR_FEATURE_NAMES, LtrCorpusContext, extract_pool_features
from hars_memory.retrieval.fusion import (
    ChannelHit,
    FusedChunk,
    fuse,
    ripgrep_scores_by_chunk_id,
)

logger = logging.getLogger("memory.eval.ltr_dataset")

DEFAULT_ALPHA = 0.5
DEFAULT_TOP_K = 10
GRADE_EVIDENCE = 2
GRADE_SAME_DOC = 1
GRADE_IRRELEVANT = 0


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    question: str
    qtype: str
    gold_docs: tuple[str, ...]
    answer_span: str | None = None
    ll_keywords: tuple[str, ...] = ()


def _extract_gold_docs(raw: Any) -> tuple[str, ...]:
    docs: list[str] = []
    for entry in raw or ():
        if isinstance(entry, dict):
            doc = entry.get("doc") or entry.get("path") or entry.get("file_path")
            if doc:
                docs.append(str(doc))
        elif entry:
            docs.append(str(entry))
    return tuple(docs)


def load_query_records(path: Path | str) -> list[QueryRecord]:
    """Load a query YAML. Accepts the ab_bench schema (``queries:`` with
    ``gold_docs``), the relational schema (``questions:`` with dict-valued
    ``gold_docs``), and the synthetic schema (``queries:`` with ``answer_span``).
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw_queries = data.get("queries") or data.get("questions") or []
    records: list[QueryRecord] = []
    for raw in raw_queries:
        gold = _extract_gold_docs(raw.get("gold_docs"))
        if not gold and raw.get("correct_docs"):
            gold = _extract_gold_docs(raw.get("correct_docs"))
        span = raw.get("answer_span") or raw.get("answer")
        records.append(
            QueryRecord(
                query_id=str(raw.get("id") or raw.get("query_id") or len(records)),
                question=str(raw["question"]),
                qtype=str(raw.get("type") or "fact"),
                gold_docs=gold,
                answer_span=str(span) if span else None,
                ll_keywords=tuple(str(k) for k in (raw.get("ll_keywords") or ())),
            )
        )
    if not records:
        raise ValueError(f"{path} contains no queries.")
    return records


def matches_gold(served_file_path: str, gold_entry: str) -> bool:
    """Portable gold matching, mirroring ``eval.ab_bench._matches_gold``.

    The index stores some paths FLATTENED (``docs/specs/x.md`` ->
    ``docs-specs-x.md``); gold entries produced from the index carry that same
    flattened form, so basename equality holds for those. A gold entry that
    still contains a separator is matched by suffix.
    """
    served = served_file_path.replace("\\", "/")
    gold = gold_entry.replace("\\", "/")
    if "/" in gold:
        return served.endswith(gold)
    return Path(served).name == gold or served == gold


def grade_candidate(
    file_path: str,
    content: str,
    gold_docs: Iterable[str],
    answer_span: str | None,
) -> int:
    """The graded label for one candidate (see the module docstring)."""
    matched = any(matches_gold(file_path, gold) for gold in gold_docs)
    if not matched:
        return GRADE_IRRELEVANT
    if answer_span:
        return GRADE_EVIDENCE if answer_span.casefold() in content.casefold() else GRADE_SAME_DOC
    return GRADE_EVIDENCE


def build_rows(record: QueryRecord, pool: Sequence[FusedChunk], context: LtrCorpusContext | None) -> list[dict[str, Any]]:
    """Feature rows + graded labels for one query's candidate pool."""
    rows = extract_pool_features(pool, record.question, context=context)
    out: list[dict[str, Any]] = []
    for row, chunk in zip(rows, pool, strict=True):
        label = grade_candidate(chunk.file_path, chunk.content or "", record.gold_docs, record.answer_span)
        out.append({
            "query_id": record.query_id,
            "question": record.question,
            "item_id": row.item_id,
            "path": row.path,
            "rank": row.rank,
            "label": label,
            "features": list(row.features),
        })
    return out


def write_jsonl(rows: Sequence[dict[str, Any]], out_path: Path, feature_names: Sequence[str] = LTR_FEATURE_NAMES) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"feature_names": list(feature_names)}) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Live pool construction (mirrors channels.hybrid_bm25_ripgrep, but keeps the
# raw FusedChunk objects the harness throws away)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolBundle:
    """One question's candidate pool plus the ripgrep annotations used to build it."""

    pool: list[FusedChunk]
    ripgrep_hits_by_basename: dict[str, ChannelHit]


async def build_pool(
    rag: Any,
    bm25_index: Any,
    question: str,
    *,
    top_k: int,
    alpha: float,
    pool_multiplier: int = DEFAULT_POOL_MULTIPLIER,
    annotate_ripgrep: bool = True,
    ll_keywords: Sequence[str] | None = None,
    ripgrep_roots: list[Any] | None = None,
) -> PoolBundle:
    """The full pre-truncation fused pool for one question.

    Dense + BM25 -> ``fuse()``. When ``annotate_ripgrep`` is set the pool's
    ``ripgrep_score`` fields are populated with the SAME per-chunk raw score
    ``apply_ripgrep_ranked`` would blend (``ripgrep_scores_by_chunk_id``), but
    the pool order is NOT changed by ripgrep — the learned ranker sees the
    ripgrep feature and decides its weight itself. This is deliberate: the
    point of the ranker is to LEARN the channel weights the hand-tuned path
    fixes.
    """
    pool_size = max(top_k * pool_multiplier, top_k)
    sparse_hits = await asyncio.to_thread(bm25_index.search, question, pool_size)
    dense_raw = await rag.chunks_vdb.query(question, top_k=pool_size)

    sparse = {
        hit.chunk_id: ChannelHit(score=hit.score, content=hit.content, file_path=hit.file_path)
        for hit in sparse_hits
    }
    dense = {
        str(entry["id"]): ChannelHit(
            score=float(entry.get("distance", 0.0)),
            content=str(entry.get("content", "")),
            file_path=str(entry.get("file_path", "")),
        )
        for entry in dense_raw
    }
    pool = fuse(dense, sparse, alpha)

    by_basename: dict[str, ChannelHit] = {}
    if annotate_ripgrep:
        from hars_memory.retrieval import ripgrep_channel as ripgrep_mod

        roots = ripgrep_roots if ripgrep_roots is not None else ripgrep_mod.default_roots()
        rg_result = await ripgrep_mod.asearch(
            question, ll_keywords=list(ll_keywords or ()), roots=roots, top_k=pool_size
        )
        if rg_result.available and rg_result.hits:
            by_basename = {
                Path(hit.file_path).name: ChannelHit(
                    score=hit.score, content=hit.content, file_path=hit.file_path
                )
                for hit in rg_result.hits
            }
            rg_scores = ripgrep_scores_by_chunk_id(pool, by_basename)
            from dataclasses import replace

            pool = [
                replace(chunk, ripgrep_score=rg_scores.get(chunk.chunk_id, chunk.ripgrep_score))
                for chunk in pool
            ]
    return PoolBundle(pool=pool, ripgrep_hits_by_basename=by_basename)


async def export_dataset(
    *,
    queries: list[QueryRecord],
    out_path: Path,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
    pool_multiplier: int = DEFAULT_POOL_MULTIPLIER,
    index_dir: str | None = None,
    context: LtrCorpusContext | None = None,
    annotate_ripgrep: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the live retrieval path for every query and write the JSONL."""
    from hars_memory.server.lightrag_init import DEFAULT_WORKING_DIR, create_lightrag
    from hars_memory.retrieval.bm25_index import get_or_build_index

    working_dir = str(index_dir or os.environ.get("HARS_MEMORY_INDEX_DIR") or DEFAULT_WORKING_DIR)
    rag = create_lightrag()
    await rag.initialize_storages()
    install_query_embedding_cache(rag)

    cache_dir = os.environ.get("HARS_MEMORY_BM25_CACHE_DIR") or str(Path(working_dir).parent / "bm25_cache")
    bm25_index, _stats = await asyncio.to_thread(get_or_build_index, working_dir, cache_dir)

    if context is None:
        context = LtrCorpusContext.load(working_dir)

    all_rows: list[dict[str, Any]] = []
    stats = {"queries": 0, "queries_with_positive": 0, "rows": 0, "positives": 0}
    for i, record in enumerate(queries, start=1):
        bundle = await build_pool(
            rag,
            bm25_index,
            record.question,
            top_k=top_k,
            alpha=alpha,
            pool_multiplier=pool_multiplier,
            annotate_ripgrep=annotate_ripgrep,
            ll_keywords=record.ll_keywords,
        )
        pool = bundle.pool
        rows = build_rows(record, pool, context)
        positives = sum(1 for row in rows if row["label"] > 0)
        stats["queries"] += 1
        stats["rows"] += len(rows)
        stats["positives"] += positives
        if positives:
            stats["queries_with_positive"] += 1
            all_rows.extend(rows)
        else:
            logger.warning("query %s produced no positive candidate (pool=%d)", record.query_id, len(pool))
        if progress and i % 25 == 0:
            logger.info("exported %d/%d queries (%d rows)", i, len(queries), stats["rows"])

    write_jsonl(all_rows, out_path)
    stats["out_path"] = str(out_path)
    return stats


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, required=True, help="Query YAML (ab_bench / relational / synthetic).")
    parser.add_argument("--out", type=Path, required=True, help="JSONL output path.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--pool-multiplier", type=int, default=DEFAULT_POOL_MULTIPLIER)
    parser.add_argument("--index-dir", type=str, default=None)
    parser.add_argument("--no-ripgrep", action="store_true", help="Do not annotate the ripgrep feature.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    records = load_query_records(args.queries)
    stats = asyncio.run(
        export_dataset(
            queries=records,
            out_path=args.out,
            top_k=args.top_k,
            alpha=args.alpha,
            pool_multiplier=args.pool_multiplier,
            index_dir=args.index_dir,
            annotate_ripgrep=not args.no_ripgrep,
        )
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
