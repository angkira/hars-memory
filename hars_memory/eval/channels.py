"""Retrieval-channel adapters consumed by tools/memory/eval/ab_bench.py.

Every function here calls the SAME production modules
(tools/memory/retrieval/*, tools/memory/server/reranker.py, and
LightRAG's own `rag.chunks_vdb` / `rag.aquery`) exactly as
plugins/hars-longterm-memory/scripts/hars_longterm_memory_mcp.py does — this module does
not reimplement or alter any retrieval algorithm, it only calls it and
normalizes the result shape so ab_bench.py can score every channel with the
same metrics.py functions. Per the task constraints, none of the constrained
files (hars_longterm_memory_mcp.py, server/{embedder,lightrag_init,reranker}.py,
retrieval/*) are edited here — only imported and called.

`dense_only` / `hybrid_bm25` / `hybrid_bm25_rerank` mirror
`_compute_hybrid_block()` in hars_longterm_memory_mcp.py exactly (same
`HYBRID_CANDIDATE_POOL_MULTIPLIER`, same `fuse()` call) so this harness
measures what production actually does, not a reimplementation of it.

`lightrag_mode` drives LightRAG's own naive/local/global/hybrid(mix) context
assembly via `only_need_context=True` (no LLM call) — see `derive_keywords`
for why: local/global/hybrid modes need ll_keywords/hl_keywords to run
without an LLM keyword-extraction call, which this offline harness cannot
make (no GPU, no LLM, per task constraints). `derive_keywords` is a
deterministic heuristic standing in for the role hars_longterm_memory_mcp.py's own
tool docstring assigns to the calling agent ("you are the LLM, extract them
from your question yourself") — it is harness plumbing, not a retrieval
technique.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from hars_memory.retrieval.bm25_index import BM25SparseIndex
from hars_memory.retrieval.fusion import ChannelHit, apply_ripgrep_gate, fuse
from hars_memory.retrieval.tokenizer import extract_identifier_terms

# Matches HYBRID_CANDIDATE_POOL_MULTIPLIER in hars_longterm_memory_mcp.py — each
# channel retrieves top_k * this many candidates before fusion/rerank.
DEFAULT_POOL_MULTIPLIER = 3
DEFAULT_RERANK_POOL_SIZE = 15


@dataclass(frozen=True)
class RankedHit:
    rank: int  # 1-indexed
    file_path: str
    chunk_id: str | None
    score: float | None


def ranked_file_paths(hits: list[RankedHit]) -> list[str]:
    """Extract just the file_path column, in rank order, dropping empties."""
    return [h.file_path for h in hits if h.file_path]


# ---------------------------------------------------------------------------
# Channel 1: dense-only (pure vector search over LightRAG's chunk store)
# ---------------------------------------------------------------------------


async def dense_only(rag: Any, question: str, top_k: int) -> list[RankedHit]:
    raw = await rag.chunks_vdb.query(question, top_k=top_k)
    return [
        RankedHit(
            rank=i,
            file_path=str(entry.get("file_path", "")),
            chunk_id=str(entry.get("id", "")) or None,
            score=float(entry.get("distance", 0.0)),
        )
        for i, entry in enumerate(raw, start=1)
    ]


# ---------------------------------------------------------------------------
# Channel 2: hybrid dense + BM25 sparse fusion (retrieval/fusion.py)
# ---------------------------------------------------------------------------


async def hybrid_bm25(
    rag: Any,
    bm25_index: BM25SparseIndex,
    question: str,
    top_k: int,
    alpha: float,
    pool_multiplier: int = DEFAULT_POOL_MULTIPLIER,
) -> list[RankedHit]:
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
    fused = fuse(dense, sparse, alpha)[:top_k]
    return [
        RankedHit(rank=i, file_path=c.file_path, chunk_id=c.chunk_id, score=c.fused_score)
        for i, c in enumerate(fused, start=1)
    ]


# ---------------------------------------------------------------------------
# Channel 2b: hybrid dense + BM25 + the RANKED ripgrep channel
# (retrieval/ripgrep_channel.py + retrieval/fusion.py's apply_ripgrep_gate
# with mode="ranked"). Added 2026-10-17: before this, `ab_bench` never
# exercised the ripgrep gate at all (it was only referenced as a strict-env
# knob), so a ranked-ripgrep change could not be measured — this is the
# harness entry point that makes it measurable through the same scoring path
# as every other channel.
# ---------------------------------------------------------------------------


async def hybrid_bm25_ripgrep(
    rag: Any,
    bm25_index: BM25SparseIndex,
    question: str,
    top_k: int,
    alpha: float,
    pool_multiplier: int = DEFAULT_POOL_MULTIPLIER,
    *,
    ripgrep_mode: str = "ranked",
    ripgrep_weight: float | None = None,
    ripgrep_roots: list[Any] | None = None,
    ll_keywords: list[str] | None = None,
) -> list[RankedHit]:
    """`hybrid_bm25` plus the ripgrep channel, wired EXACTLY as the live
    server wires it (`apply_ripgrep_gate` over the full pre-truncation fused
    pool, keyed by basename). `ripgrep_mode="ranked"` re-ranks inside top_k;
    `"injection"` reproduces the pre-2026-10-17 append-only behaviour. Fails
    soft: when `rg` is unavailable or finds nothing, the result is identical
    to `hybrid_bm25`.
    """
    from hars_memory.retrieval import ripgrep_channel as ripgrep_mod

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
    fused_pool = fuse(dense, sparse, alpha)

    roots = ripgrep_roots if ripgrep_roots is not None else ripgrep_mod.default_roots()
    keywords = ll_keywords if ll_keywords is not None else derive_keywords(question)[0]
    rg_result = await ripgrep_mod.asearch(
        question, ll_keywords=keywords, roots=roots, top_k=pool_size
    )
    by_basename = {
        name: ChannelHit(score=hit.score, content=hit.content, file_path=hit.file_path)
        for name, hit in ripgrep_mod.hits_by_basename(rg_result.hits).items()
    }
    gated = apply_ripgrep_gate(
        fused_pool,
        by_basename,
        top_k=top_k,
        mode=ripgrep_mode,
        weight=ripgrep_weight,
    )
    return [
        RankedHit(rank=i, file_path=c.file_path, chunk_id=c.chunk_id, score=c.fused_score)
        for i, c in enumerate(gated[:top_k], start=1)
    ]


# ---------------------------------------------------------------------------
# Channel 3: hybrid fusion + CPU cross-encoder reranker (server/reranker.py)
# ---------------------------------------------------------------------------


async def hybrid_bm25_rerank(
    rag: Any,
    bm25_index: BM25SparseIndex,
    rerank_func: Callable[..., Awaitable[list[dict[str, Any]]]],
    question: str,
    top_k: int,
    alpha: float,
    pool_multiplier: int = DEFAULT_POOL_MULTIPLIER,
    rerank_pool_size: int = DEFAULT_RERANK_POOL_SIZE,
) -> list[RankedHit]:
    pool_size = max(top_k * pool_multiplier, rerank_pool_size, top_k)
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
    fused = fuse(dense, sparse, alpha)[:rerank_pool_size]
    if not fused:
        return []

    documents = [c.content for c in fused]
    scored = await rerank_func(question, documents, top_n=top_k)

    hits: list[RankedHit] = []
    for i, item in enumerate(scored[:top_k], start=1):
        candidate = fused[int(item["index"])]
        hits.append(
            RankedHit(
                rank=i,
                file_path=candidate.file_path,
                chunk_id=candidate.chunk_id,
                score=float(item["relevance_score"]),
            )
        )
    return hits


# ---------------------------------------------------------------------------
# Channel 4: LightRAG's own mode-based context assembly (naive/local/global/hybrid)
# ---------------------------------------------------------------------------

_DOCUMENT_CHUNKS_HEADER = "Document Chunks ("
_REFERENCE_LIST_HEADER = "Reference Document List ("
_REFERENCE_LINE_RE = re.compile(r"^\[(?P<ref_id>[^\]]+)\]\s*(?P<path>.+)$")


def _extract_fenced_block(context: str, header: str) -> str | None:
    header_idx = context.find(header)
    if header_idx == -1:
        return None
    fence_start = context.find("```", header_idx)
    if fence_start == -1:
        return None
    content_start = context.find("\n", fence_start)
    if content_start == -1:
        return None
    content_start += 1
    fence_end = context.find("```", content_start)
    if fence_end == -1:
        return context[content_start:]
    return context[content_start:fence_end]


def parse_reference_document_list(context: str) -> dict[str, str]:
    """Map reference_id -> file_path from LightRAG's `only_need_context=True`
    output. Format confirmed against a live query against the real index:
    ``[1] some/file_path.md`` lines inside a fenced block after the
    "Reference Document List (" header.
    """
    block = _extract_fenced_block(context, _REFERENCE_LIST_HEADER)
    if block is None:
        return {}
    mapping: dict[str, str] = {}
    for line in block.splitlines():
        match = _REFERENCE_LINE_RE.match(line.strip())
        if match:
            mapping[match.group("ref_id")] = match.group("path").strip()
    return mapping


def parse_document_chunks_order(context: str) -> list[str]:
    """Return reference_ids in the order LightRAG placed them in the
    "Document Chunks (" fenced JSON-lines block — this order IS the final
    rank LightRAG hands to the answering LLM, which is what this channel
    measures.
    """
    block = _extract_fenced_block(context, _DOCUMENT_CHUNKS_HEADER)
    if block is None:
        return []
    ref_ids: list[str] = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ref_id = str(obj.get("reference_id", ""))
        if ref_id:
            ref_ids.append(ref_id)
    return ref_ids


_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "did", "does", "do",
        "what", "which", "who", "how", "why", "when", "where", "and", "or",
        "of", "to", "in", "on", "for", "with", "that", "this", "it", "be",
        "has", "have", "had", "not", "no", "yes", "as", "by", "from", "at",
        "we", "our", "its", "their", "about", "than", "into", "over",
        "still", "actually", "really", "also",
    }
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def derive_keywords(question: str) -> tuple[list[str], list[str]]:
    """Deterministic ll_keywords/hl_keywords split, no LLM required.

    ll_keywords: exact identifier-shaped terms (reuses
    retrieval.tokenizer.extract_identifier_terms — the same detector the
    production BM25 identifier-lookup path uses).
    hl_keywords: remaining content words (stopwords and identifiers
    excluded, deduped, capped) — a crude proxy for "themes", standing in for
    what the calling LLM agent is expected to supply per
    hars_longterm_memory_mcp.py's own memory_recall tool description.
    """
    ll_keywords = extract_identifier_terms(question)
    ll_lower = {k.casefold() for k in ll_keywords}
    hl_keywords: list[str] = []
    seen: set[str] = set()
    for word in _WORD_RE.findall(question):
        lower = word.casefold()
        if len(word) < 4 or lower in _STOPWORDS or lower in ll_lower or lower in seen:
            continue
        seen.add(lower)
        hl_keywords.append(word)
    return ll_keywords, hl_keywords[:8]


async def lightrag_mode(
    rag: Any,
    question: str,
    mode: str,
    top_k: int,
    chunk_top_k: int | None = None,
    *,
    max_entity_tokens: int | None = None,
    max_relation_tokens: int | None = None,
    max_total_tokens: int | None = None,
) -> list[RankedHit]:
    """Run one of LightRAG's own retrieval modes (naive/local/global/hybrid)
    via `only_need_context=True` (no LLM call) and return a ranked file_path
    list derived from the rendered context's Document Chunks order.

    No per-chunk score is available in this text format (LightRAG's context
    renderer only encodes rank order, not a numeric score) — `score=None` is
    correct here, not a missing value.

    `max_entity_tokens` / `max_relation_tokens` / `max_total_tokens` map
    directly onto `QueryParam`'s fields of the same name (LightRAG defaults:
    6000 / 8000 / 30000 — see `lightrag.constants.DEFAULT_MAX_ENTITY_TOKENS`
    etc). `None` (the default here) omits the kwarg entirely so LightRAG's own
    default applies unmodified — this parameter exists for
    `ab_bench.py --token-budget-sweep`, which needs to vary these budgets
    without touching `server/lightrag_init.py` (a constrained file).
    """
    from lightrag import QueryParam  # type: ignore[import-not-found]

    lightrag_mode_name = "mix" if mode == "hybrid" else mode
    ll_keywords, hl_keywords = derive_keywords(question)
    kw_args = (
        {"ll_keywords": ll_keywords, "hl_keywords": hl_keywords}
        if (ll_keywords or hl_keywords)
        else {}
    )
    if max_entity_tokens is not None:
        kw_args["max_entity_tokens"] = max_entity_tokens
    if max_relation_tokens is not None:
        kw_args["max_relation_tokens"] = max_relation_tokens
    if max_total_tokens is not None:
        kw_args["max_total_tokens"] = max_total_tokens
    context = await rag.aquery(
        question,
        param=QueryParam(
            mode=lightrag_mode_name,
            top_k=top_k,
            chunk_top_k=chunk_top_k if chunk_top_k is not None else top_k,
            only_need_context=True,
            **kw_args,
        ),
    )
    if not isinstance(context, str) or not context.strip():
        return []

    ref_map = parse_reference_document_list(context)
    ref_order = parse_document_chunks_order(context)
    hits: list[RankedHit] = []
    rank = 0
    for ref_id in ref_order:
        file_path = ref_map.get(ref_id, "")
        if not file_path:
            continue
        rank += 1
        hits.append(RankedHit(rank=rank, file_path=file_path, chunk_id=None, score=None))
        if rank >= top_k:
            break
    return hits


# ---------------------------------------------------------------------------
# Query-embedding cache — wraps the LightRAG instance's own embedding_func at
# runtime (mutates the constructed object's attribute, never the on-disk
# server/embedder.py module) so repeated identical query texts across
# multiple channels/configs in one ab_bench.py run pay the ~89ms/text CPU
# embed cost once, not once per channel.
# ---------------------------------------------------------------------------


def install_query_embedding_cache(rag: Any) -> Callable[[], dict[str, int]]:
    original = rag.embedding_func.func
    cache: dict[tuple[str, str | None], Any] = {}
    stats = {"hits": 0, "misses": 0}

    async def cached_embed(texts: list[str], context: str | None = None, **kwargs: Any) -> Any:
        import numpy as np

        results: list[Any] = [None] * len(texts)
        miss_positions: list[int] = []
        miss_texts: list[str] = []
        for i, text in enumerate(texts):
            key = (text, context)
            if key in cache:
                results[i] = cache[key]
                stats["hits"] += 1
            else:
                miss_positions.append(i)
                miss_texts.append(text)

        if miss_texts:
            stats["misses"] += len(miss_texts)
            fetched = await original(miss_texts, context=context, **kwargs)
            for position, vector in zip(miss_positions, fetched):
                cache[(texts[position], context)] = vector
                results[position] = vector

        return np.stack(results)

    rag.embedding_func.func = cached_embed
    return lambda: dict(stats)


__all__ = [
    "RankedHit",
    "ranked_file_paths",
    "dense_only",
    "hybrid_bm25",
    "hybrid_bm25_ripgrep",
    "hybrid_bm25_rerank",
    "lightrag_mode",
    "parse_reference_document_list",
    "parse_document_chunks_order",
    "derive_keywords",
    "install_query_embedding_cache",
    "DEFAULT_POOL_MULTIPLIER",
    "DEFAULT_RERANK_POOL_SIZE",
]
