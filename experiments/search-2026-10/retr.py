"""Production-backed retrieval candidates for the 2026-10 search experiment."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from common import CLONE, EXP
from hars_memory.retrieval.fusion import ChannelHit, fuse

_RAG: Any | None = None
_BM25: Any | None = None


def _load_live_env() -> None:
    """Load optional local runtime settings silently, preserving explicit env."""
    env_file = Path(__file__).resolve().parents[2] / "config" / "live-llm-e2e.env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text("utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


async def open_retriever() -> tuple[Any, Any]:
    """Initialize LightRAG against the experiment clone and build/load BM25."""
    global _RAG, _BM25
    if _RAG is not None and _BM25 is not None:
        return _RAG, _BM25

    os.environ.setdefault("HARS_MEMORY_INDEX_DIR", str(CLONE))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HARS_MEMORY_EMBED_MODEL", "unsloth/embeddinggemma-300m")
    _load_live_env()
    if Path(os.environ["HARS_MEMORY_INDEX_DIR"]).resolve() != CLONE.resolve():
        raise RuntimeError("HARS_MEMORY_INDEX_DIR does not point at the read-only experiment clone")

    from hars_memory.retrieval.bm25_index import get_or_build_index
    from hars_memory.server.lightrag_init import create_lightrag

    _RAG = create_lightrag(working_dir=str(CLONE))
    await _RAG.initialize_storages()
    _BM25 = (await asyncio.to_thread(get_or_build_index, str(CLONE), "/tmp/search-exp/bm25_base"))[0]
    return _RAG, _BM25


def _install_embedding_timer(rag: Any) -> dict[str, float]:
    """Accumulate time spent inside LightRAG's embedding function."""
    embedding_func = rag.embedding_func
    original = embedding_func.func
    timing = {"seconds": 0.0}

    async def timed_embedding(texts: list[str], context: str | None = None, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await original(texts, context=context, **kwargs)
        finally:
            timing["seconds"] += time.perf_counter() - started

    embedding_func.func = timed_embedding
    return timing


def _as_pairs(values: list[list[Any]] | list[tuple[str, float]]) -> list[tuple[str, float]]:
    return [(str(chunk_id), float(score)) for chunk_id, score in values]


async def build_candidates(
    cases: list[dict[str, Any]], pool: int = 100, tag: str = "base"
) -> dict[str, dict[str, Any]]:
    """Retrieve dense and sparse candidate lists and cache them with embed timings."""
    if pool < 1:
        raise ValueError("pool must be at least 1")
    case_ids = [str(case["id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")

    cache_path = EXP / "cache" / f"cands_{tag}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text("utf-8"))
        if cached.get("pool") == pool and set(cached.get("cases", {})) >= set(case_ids):
            return {
                case_id: {
                    **cached["cases"][case_id],
                    "dense": _as_pairs(cached["cases"][case_id].get("dense", [])),
                    "bm25": _as_pairs(cached["cases"][case_id].get("bm25", [])),
                }
                for case_id in case_ids
            }

    rag, bm25 = await open_retriever()
    timing = _install_embedding_timer(rag)
    candidates: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case["id"])
        before_embed = timing["seconds"]
        dense_raw = await rag.chunks_vdb.query(str(case["question"]), top_k=pool)
        query_embedding_seconds = timing["seconds"] - before_embed
        sparse_raw = await asyncio.to_thread(bm25.search, str(case["question"]), pool)
        candidates[case_id] = {
            "dense": [
                (str(hit["id"]), float(hit.get("distance", 0.0)))
                for hit in dense_raw
                if hit.get("id") is not None
            ],
            "bm25": [(hit.chunk_id, float(hit.score)) for hit in sparse_raw],
            "query_embedding_seconds": query_embedding_seconds,
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "pool": pool,
        "query_embedding_seconds_total": sum(
            float(candidates[case_id]["query_embedding_seconds"]) for case_id in case_ids
        ),
        "cases": candidates,
    }
    cache_path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    return candidates


def rank_dense(dense: list[tuple[str, float]], top_k: int) -> list[str]:
    return [chunk_id for chunk_id, _ in dense[: max(0, top_k)]]


def rank_bm25(sparse: list[tuple[str, float]], top_k: int) -> list[str]:
    return [chunk_id for chunk_id, _ in sparse[: max(0, top_k)]]


def fuse_hybrid(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    alpha: float = 0.5,
    pool: int = 15,
    top_k: int = 5,
) -> list[str]:
    """Fuse the first ``pool`` hits from each channel using production fusion."""
    dense_hits = {
        chunk_id: ChannelHit(score=float(score), content="", file_path="")
        for chunk_id, score in dense[: max(0, pool)]
    }
    sparse_hits = {
        chunk_id: ChannelHit(score=float(score), content="", file_path="")
        for chunk_id, score in sparse[: max(0, pool)]
    }
    return [hit.chunk_id for hit in fuse(dense_hits, sparse_hits, alpha)[: max(0, top_k)]]


def fuse_rrf(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    k: int = 60,
    pool: int = 15,
    top_k: int = 5,
) -> list[str]:
    """Reciprocal-rank fuse the first ``pool`` candidates of both channels."""
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = {}
    for hits in (dense[: max(0, pool)], sparse[: max(0, pool)]):
        for rank, (chunk_id, _) in enumerate(hits, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return ranked[: max(0, top_k)]


__all__ = [
    "open_retriever",
    "build_candidates",
    "fuse_hybrid",
    "fuse_rrf",
    "rank_dense",
    "rank_bm25",
]