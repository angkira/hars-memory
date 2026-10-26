"""LLM-free, CPU-only retrieval over an index built by ``corpus.build``.

Three modes:

``sparse``
    BM25 only (``retrieval/bm25_index.py``). Zero model load — the fastest
    possible path, and the only one that is truly "instant" (no embedder
    import, no torch/sentence-transformers touched at all).
``dense``
    Flat dense only (``retrieval/flat_index.py`` + ``server/embedder.py``).
    Opt-in: the CPU sentence-embedding model is loaded ONLY when this mode
    (or ``fusion``) is requested — ``sparse`` mode never imports
    ``server/embedder.py`` or ``asyncio``-drives anything.
``fusion``
    Both channels combined via ``retrieval/fusion.py::fuse`` — its exact
    real signature, ``fuse(dense_hits, sparse_hits, alpha) -> list[FusedChunk]``,
    is reused unmodified (it fits this module's dict-of-``ChannelHit``
    shape directly; no local reimplementation of dense/sparse blending was
    needed).

CPU-only enforcement: this module refuses to run ``dense``/``fusion`` mode
if ``HARS_MEMORY_EMBED_DEVICE`` is set to anything other than ``cpu``/unset
(see ``_ensure_cpu_only``) — ``server/embedder.py`` itself already defaults
to CPU, but this is a belt-and-braces guard specific to this subsystem's
hard "CPU only, never GPU" constraint, checked BEFORE the embedder module is
ever imported.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

from hars_memory.retrieval.bm25_index import (
    CHUNKS_FILENAME,
    get_or_build_index as _bm25_get_or_build_index,
)
from hars_memory.retrieval.flat_index import (
    get_or_build_index as _flat_get_or_build_index,
)
from hars_memory.retrieval.fusion import ChannelHit, DEFAULT_HYBRID_ALPHA, fuse

logger = logging.getLogger(__name__)

DEFAULT_TOP_K: Final[int] = 10
# Matches eval/channels.py's HYBRID_CANDIDATE_POOL_MULTIPLIER convention:
# each channel retrieves top_k * this many candidates before fusion, so the
# fused ranking has real headroom to reorder beyond a naive top_k union.
DEFAULT_POOL_MULTIPLIER: Final[int] = 3
DEFAULT_ALPHA: Final[float] = DEFAULT_HYBRID_ALPHA
# Matches server/embedder.py::make_embedding_func's own default model.
DEFAULT_EMBED_MODEL: Final[str] = "unsloth/embeddinggemma-300m"
DEFAULT_SNIPPET_CHARS: Final[int] = 240
# Cache subdirectory names used when the caller does not supply an explicit
# bm25_cache_dir/flat_cache_dir — always resolved to an ABSOLUTE path
# (index_dir is resolved first), satisfying bm25_index.py/flat_index.py's
# own "cache_dir must be absolute" contract.
_DEFAULT_BM25_CACHE_SUBDIR: Final[str] = ".bm25_cache"
_DEFAULT_FLAT_CACHE_SUBDIR: Final[str] = ".flat_cache"

_HARS_MEMORY_EMBED_DEVICE_ENV: Final[str] = "HARS_MEMORY_EMBED_DEVICE"


class SearchMode(str, Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    FUSION = "fusion"


class CorpusQueryError(Exception):
    """Base class for every corpus-query-specific failure in this module."""


class CorpusIndexNotFoundError(CorpusQueryError):
    """Raised when ``index_dir`` has no ``kv_store_text_chunks.json`` —
    i.e. ``corpus.build.build_corpus`` was never run against it (or the
    directory is simply wrong)."""

    def __init__(self, index_dir: Path) -> None:
        super().__init__(
            f"No {CHUNKS_FILENAME} at {index_dir} — build a corpus index "
            "first (corpus.build.build_corpus / `memory build`)."
        )
        self.index_dir = index_dir


class GpuNotAllowedError(CorpusQueryError):
    """Raised when ``HARS_MEMORY_EMBED_DEVICE`` requests a non-CPU device
    for a ``dense``/``fusion`` mode query. This subsystem is CPU-only by
    hard constraint — refused before the embedder module is even imported,
    rather than silently loading a GPU model."""

    def __init__(self, device: str) -> None:
        super().__init__(
            f"{_HARS_MEMORY_EMBED_DEVICE_ENV}={device!r} requests a non-CPU "
            "device, but tools/memory/corpus is CPU-only by hard constraint. "
            f"Unset {_HARS_MEMORY_EMBED_DEVICE_ENV} or set it to 'cpu'."
        )
        self.device = device


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked result. ``channel`` records provenance: which retrieval
    path produced this hit (``"sparse"``, ``"dense"``, or ``"fusion"`` — the
    requested ``mode``, not a per-hit sub-channel breakdown)."""

    chunk_id: str
    file_path: str
    score: float
    channel: str
    snippet: str


def _snippet(content: str, max_chars: int = DEFAULT_SNIPPET_CHARS) -> str:
    stripped = content.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars].rstrip() + "…"


def _ensure_cpu_only() -> None:
    device = os.environ.get(_HARS_MEMORY_EMBED_DEVICE_ENV, "cpu").strip().lower()
    if device not in {"cpu", ""}:
        raise GpuNotAllowedError(device)


def _resolve_bm25_cache_dir(index_dir: Path, bm25_cache_dir: str | None) -> str:
    return bm25_cache_dir or str(index_dir / _DEFAULT_BM25_CACHE_SUBDIR)


def _resolve_flat_cache_dir(index_dir: Path, flat_cache_dir: str | None) -> str:
    return flat_cache_dir or str(index_dir / _DEFAULT_FLAT_CACHE_SUBDIR)


def _search_sparse(
    index_dir: Path, question: str, *, top_k: int, bm25_cache_dir: str | None
) -> list[SearchHit]:
    working_dir = str(index_dir)
    cache_dir = _resolve_bm25_cache_dir(index_dir, bm25_cache_dir)
    index, stats = _bm25_get_or_build_index(working_dir, cache_dir)
    logger.debug(
        "BM25 index ready: %d chunks (cache_hit=%s)", index.chunk_count, stats.cache_hit
    )
    hits = index.search(question, top_k)
    return [
        SearchHit(
            chunk_id=h.chunk_id,
            file_path=h.file_path,
            score=h.score,
            channel=SearchMode.SPARSE.value,
            snippet=_snippet(h.content),
        )
        for h in hits
    ]


async def _search_dense_or_fusion(
    index_dir: Path,
    question: str,
    *,
    top_k: int,
    mode: SearchMode,
    bm25_cache_dir: str | None,
    flat_cache_dir: str | None,
    embed_model: str,
    alpha: float,
) -> list[SearchHit]:
    _ensure_cpu_only()
    from hars_memory.server.embedder import make_embedding_func  # lazy: see module docstring

    working_dir = str(index_dir)
    flat_cache = _resolve_flat_cache_dir(index_dir, flat_cache_dir)
    embed_func = make_embedding_func(model_name=embed_model)

    flat_index, flat_stats = await _flat_get_or_build_index(
        working_dir, flat_cache, embed_func, embed_model=embed_model
    )
    logger.debug(
        "Flat dense index ready: %d chunks (update_path=%s)",
        flat_index.chunk_count, flat_stats.update_path,
    )

    if mode is SearchMode.DENSE:
        dense_hits = await flat_index.search(question, embed_func, top_k)
        return [
            SearchHit(
                chunk_id=h.chunk_id,
                file_path=h.file_path,
                score=h.score,
                channel=SearchMode.DENSE.value,
                snippet=_snippet(h.content),
            )
            for h in dense_hits
        ]

    # fusion: pull a wider pool from each channel, then let fuse() re-rank.
    pool_size = max(top_k * DEFAULT_POOL_MULTIPLIER, top_k)
    dense_pool = await flat_index.search(question, embed_func, pool_size)
    bm25_cache = _resolve_bm25_cache_dir(index_dir, bm25_cache_dir)
    bm25_index, _bm25_stats = _bm25_get_or_build_index(working_dir, bm25_cache)
    sparse_pool = bm25_index.search(question, pool_size)

    dense = {
        h.chunk_id: ChannelHit(score=h.score, content=h.content, file_path=h.file_path)
        for h in dense_pool
    }
    sparse = {
        h.chunk_id: ChannelHit(score=h.score, content=h.content, file_path=h.file_path)
        for h in sparse_pool
    }
    fused = fuse(dense, sparse, alpha)[:top_k]
    return [
        SearchHit(
            chunk_id=c.chunk_id,
            file_path=c.file_path,
            score=c.fused_score,
            channel=SearchMode.FUSION.value,
            snippet=_snippet(c.content),
        )
        for c in fused
    ]


def search(
    index_dir: Path,
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    mode: SearchMode | str = SearchMode.FUSION,
    bm25_cache_dir: str | None = None,
    flat_cache_dir: str | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    alpha: float = DEFAULT_ALPHA,
) -> list[SearchHit]:
    """Search an index built by ``corpus.build.build_corpus``.

    Parameters
    ----------
    index_dir:
        Directory containing ``kv_store_text_chunks.json`` (as emitted by
        ``build_corpus``).
    question:
        Query text. Must be non-empty.
    top_k:
        Number of results to return. Must be >= 1.
    mode:
        ``"sparse"`` | ``"dense"`` | ``"fusion"`` — see module docstring.
    bm25_cache_dir, flat_cache_dir:
        Absolute cache directories for the two channel indexes. Default to
        ``index_dir/.bm25_cache`` and ``index_dir/.flat_cache`` respectively.
    embed_model:
        Sentence-transformers model id, forwarded to
        ``server/embedder.py::make_embedding_func``. Unused in ``sparse``
        mode.
    alpha:
        Convex-combination weight forwarded to ``retrieval/fusion.py::fuse``
        (dense weight; ``1 - alpha`` is the sparse weight). Unused outside
        ``fusion`` mode.

    Raises
    ------
    ValueError
        Empty ``question`` or ``top_k < 1``.
    CorpusIndexNotFoundError
        ``index_dir`` has no chunk store.
    GpuNotAllowedError
        ``dense``/``fusion`` mode requested with a non-CPU
        ``HARS_MEMORY_EMBED_DEVICE``.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    search_mode = mode if isinstance(mode, SearchMode) else SearchMode(mode)
    resolved_index_dir = Path(index_dir).resolve()
    if not (resolved_index_dir / CHUNKS_FILENAME).is_file():
        raise CorpusIndexNotFoundError(resolved_index_dir)

    if search_mode is SearchMode.SPARSE:
        return _search_sparse(
            resolved_index_dir, question, top_k=top_k, bm25_cache_dir=bm25_cache_dir
        )
    return asyncio.run(
        _search_dense_or_fusion(
            resolved_index_dir,
            question,
            top_k=top_k,
            mode=search_mode,
            bm25_cache_dir=bm25_cache_dir,
            flat_cache_dir=flat_cache_dir,
            embed_model=embed_model,
            alpha=alpha,
        )
    )


__all__ = [
    "DEFAULT_TOP_K",
    "DEFAULT_POOL_MULTIPLIER",
    "DEFAULT_ALPHA",
    "DEFAULT_EMBED_MODEL",
    "SearchMode",
    "CorpusQueryError",
    "CorpusIndexNotFoundError",
    "GpuNotAllowedError",
    "SearchHit",
    "search",
]
