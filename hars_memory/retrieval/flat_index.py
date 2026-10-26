"""Flat (chunk-level) dense retrieval index — no graph, no LLM extraction.

WHY this channel exists: the system's retrieval portfolio has a gap between
BM25 (exact tokens, index-bound freshness) and the LightRAG graph (multi-hop
relations, ~3h47m GPU build for 241 documents). Graph extraction is ~100% of
that build cost — it needs an LLM to produce entities/relationships. Plain
chunk embedding needs none of that: chunk text already exists on disk
(`kv_store_text_chunks.json`), and CPU sentence-embedding is the only cost.
That makes this channel rebuildable on a much shorter cycle (minutes, not
hours) than the graph, at the price of the graph's multi-hop edge (measured
elsewhere: `naive` matches `hybrid` on overall ndcg, but loses specifically
on multihop queries) — see the module's companion report for the measured
tradeoff on this corpus's labeled query set.

WHY reuse `kv_store_text_chunks.json` chunks, not re-chunk from source docs:
id-compatibility with every other channel. `retrieval/bm25_index.py` and
LightRAG's own dense channel (`rag.chunks_vdb`) both key their hits by this
same chunk `_id`; `retrieval/fusion.py::fuse()` merges dense_hits/sparse_hits
dicts on that key. Re-chunking from source documents with a retrieval-suited
strategy (sentence/paragraph-aware, not LightRAG's 2048-BYTE/256-overlap
byte-tokenizer split that has been observed to cut chunks mid-sentence)
would very likely improve standalone quality, but it breaks that shared id
space: a flat index built from different chunk boundaries cannot be fused
against BM25 at the chunk level without a separate id-remapping layer this
module does not build. This module ships the id-compatible (a) choice;
the re-chunked (b) alternative was measured separately (read-only, outside
this module — see the companion report) specifically to quantify what (a)
gives up, not implemented as a second on-disk index here.

WHY the same embedder.py doc/query asymmetric convention (no document
prefix; `HARS_MEMORY_EMBED_QUERY_PROMPT_NAME` on the query side, default
"query" -> "task: search result | query: "): this module never loads its
own embedding model — it takes an already-constructed embed function
(the same async `embed(texts, context=...)` callable
`server/embedder.py::make_embedding_func()` returns, and that
`server/lightrag_init.py` already wires into the live LightRAG instance) as
a parameter. Reusing that function, not a second model, is what keeps this
channel's vectors in the exact same space as `dense_only`'s (LightRAG's
`rag.chunks_vdb`) vectors automatically, including its asymmetry: calling
`embed_func(contents, context="document")` to build the index and
`embed_func([query], context="query")` to search reproduces exactly what
`server/lightrag_init.py` already does for the production dense channel, no
separate prefix constant to keep in sync by hand.

WHY plain numpy dot product, no ANN library (faiss/hnswlib/annoy): 7,184 x
768 float32 is ~22 MB, trivially resident in RAM; scoring one query is a
single (N, D) @ (D,) matmul, sub-millisecond on any CPU. ANN indexes exist
to avoid an O(N) scan over corpora many orders of magnitude larger than
this one, at the cost of approximate recall and real index-build/tuning
complexity. Revisit only if this corpus's chunk count grows into the
hundreds of thousands, or once a shared vector store (Qdrant, out of scope
for this module — a concurrent migration owns that) makes a matrix-in-numpy
index redundant rather than complementary.

WHY persisted (`.npy` vectors + JSON sidecar) + mtime-invalidated: mirrors
`retrieval/bm25_index.py`'s cache exactly (see that module's docstring for
the rationale) — the expensive part here is N CPU embedding calls, worth
paying once per source-file version, not once per process.

WHY incremental update (`update_index`), on top of the mtime-cache above: a
plain mtime-invalidated cache only answers "is anything stale", not "what
changed" — any source-file touch (even a single new document appended)
forces `get_or_build_index` to re-embed all 7,184+ chunks at ~570ms/chunk
(measured; the ~89ms/text figure in `server/embedder.py`'s docstring is for
short query-length text, not ~2,000-byte document chunks), i.e. the full
68-minute cost, every time. That is the wrong cost model for this channel's
actual job: making a handful of freshly-consolidated documents (a few
hundred chunks) searchable promptly after `memory_remember`/KB-update runs,
without waiting for the LightRAG graph's next ~3h47m GPU cycle. Incremental
update changes the per-update cost from O(corpus size) to O(chunks changed
since last cache write) by reusing every unchanged row's vector as-is and
only calling `embed_func` for chunk ids that are new or whose content
changed. `get_or_build_index` picks incremental over full rebuild whenever
a structurally-usable, same-embed_model cache exists; see its docstring for
the exact fallback-to-full-rebuild conditions.

WHY hash-based change detection, not id-based: chunk ids
(`file:<hash>-chunk-NNN`) are stable across LightRAG re-ingests of the SAME
underlying document, but the CONTENT under a given id is not — LightRAG
re-chunks whenever a document is re-ingested (e.g. `memory_remember`
updating an existing note), so `file:<hash>-chunk-003`'s text after a
re-ingest can differ from what it was when this module last embedded it,
while the id string stays identical. Detecting "already indexed" by id
membership alone would treat that chunk as unchanged and skip re-embedding
it — i.e. silently keep serving the OLD vector under the id that now points
at different text, forever, since nothing would ever invalidate it again.
That is the same failure shape as the path-based dedup bug already hit
elsewhere in this project (matching by a supposedly-stable key while the
content underneath it changes). `update_index` therefore hashes each
chunk's content (sha256) at build and at every update, and re-embeds
whenever the CURRENT content's hash disagrees with the hash recorded in the
cache for that same id — id match is necessary but never sufficient for
"unchanged".

WHY cache_dir is NOT defaulted to `/tmp` here (unlike
`bm25_index.py`'s caller-side `HARS_MEMORY_BM25_CACHE_DIR` default of
`/tmp/hars_memory_bm25`): that default is fine for BM25 because losing it
costs a few seconds. Losing this index's cache costs up to 68 minutes (a
full rebuild) — `/tmp` can be cleared on reboot or by system tmp-cleaning
policies with no relationship to this project's lifecycle, which would
convert a routine reboot into an unplanned hour-plus rebuild the next time
anything queries this channel. This module itself takes `cache_dir` as a
plain parameter (no default, no env var) and does not prescribe a location;
whatever wires it into the running server should point it at
persistent storage that survives reboots (e.g. a subdirectory next to
`/home/demo/.local/share/hars-graphrag/index_v4` itself, which is
exactly the kind of location the source `kv_store_text_chunks.json` already
lives in) — not `/tmp`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from hars_memory.retrieval.bm25_index import CHUNKS_FILENAME

logger = logging.getLogger(__name__)

_CACHE_META_FILENAME = "flat_dense_cache_meta.json"
_CACHE_VECTORS_FILENAME = "flat_dense_vectors.npy"

# `embed_func(texts, context=None, **kwargs) -> np.ndarray` — the exact shape
# of the async callable `server/embedder.py::make_embedding_func()` returns.
# Typed as `Any` return (not `np.ndarray`) to avoid a hard top-level `numpy`
# type-only import cycle; the real return is always coerced via
# `np.asarray(...)` before use below.
EmbedFunc = Callable[..., Awaitable[Any]]

# WHY every call to `embed_func` below is BATCHED (never one call for an
# entire content list), confirmed empirically, not guessed — 2026-08-01:
#
# This module's own docstring above ("WHY the same embedder.py doc/query
# asymmetric convention") documents that the intended `embed_func` for a
# live server is `rag.embedding_func` — the SAME callable
# `server/lightrag_init.py::create_lightrag()` wires into the running
# LightRAG instance, not a second model load. What that section does not
# say (discovered only when this channel was actually wired into
# `hars_longterm_memory_mcp.py` and measured against the real object): by
# the time `create_lightrag()` returns, LightRAG's OWN `__post_init__`
# (lightrag/lightrag.py) has already REPLACED `rag.embedding_func.func`
# with `lightrag.utils.priority_limit_async_func_call(...)`'s wrapped
# version — a bounded worker-pool decorator that kills any single call
# exceeding a hard wall-clock ceiling (`asyncio.wait_for(..., timeout=
# max_execution_timeout)`) with `WorkerTimeoutError`/`TimeoutError`, not a
# graceful slow-down. That ceiling is `default_embedding_timeout * 2`
# (LightRAG's own `EMBEDDING_TIMEOUT` env, default 30s per
# `lightrag.constants.DEFAULT_EMBEDDING_TIMEOUT`) = 60 SECONDS, regardless
# of how this module is configured — it is baked into `rag.embedding_func`
# itself before this module ever sees it, entirely outside this module's
# (or hars_longterm_memory_mcp.py's) control.
#
# CONFIRMED against the live production embed_func (unsloth/embeddinggemma-
# 300m, CPU, this corpus's real ~1643-char average chunk content, `env -i`
# controlled): n=64 chunks in ONE call = 39.6s (618ms/chunk, safely under
# 60s); n=128 chunks in ONE call exceeded the ceiling and raised
# `TimeoutError: Embedding func: Worker execution timeout after 60s` —
# every chunk n=128 was embedding was simply LOST (the underlying
# `asyncio.to_thread` computation keeps running to completion in an
# orphaned background thread — CPU wasted, not saved — but `wait_for`
# already cancelled the awaiting future, so the caller gets an exception,
# not a slow-but-correct result). Before this fix, `build_index`'s single
# `await embed_func(contents, context="document")` call for this corpus's
# real 7,184 chunks (measured full-corpus cost: 68.2 minutes) would ALWAYS
# hit this 60-second ceiling when driven by `rag.embedding_func` — not
# "slow", outright BROKEN, silently turning "build could take a while" into
# "build always raises". The measured "68.2 min / 570 ms-per-chunk" full-
# build number this module's own docstring cites elsewhere predates this
# discovery and was necessarily measured with embed_func run OUTSIDE
# LightRAG's wrapping (a standalone script calling
# `server/embedder.py::make_embedding_func()` directly) — i.e. under a
# calling convention this module's OWN documented "WHY reuse rag.
# embedding_func" contract does not actually protect against in production.
#
# FIX: never submit more than `DEFAULT_EMBED_BATCH_SIZE` texts to
# `embed_func` in one call, regardless of caller-supplied contents length —
# `_embed_in_batches` below sequentially awaits each batch and concatenates
# results, so `build_index`/`update_index` behave identically from the
# caller's point of view (same input, same output shape/order/dtype),
# just safe against ANY wrapping the passed-in `embed_func` might carry.
# Sequential (not concurrent-batches-via-gather): a full 7,184-chunk build
# is a one-time, off-the-request-path cost (see the module docstring's own
# "incremental update" rationale — this is exactly the expensive path that
# rationale exists to make rare) where correctness and a bounded, easy-to-
# reason-about memory/thread footprint matter more than shaving minutes off
# a per-index, once-per-corpus-lifetime operation; concurrent submission
# also risks oversubscribing this machine's CPU threads against
# sentence-transformers' OWN internal BLAS/OMP multi-threading per call
# (`server/embedder.py`'s `model.encode(..., batch_size=...)`), a real
# regression risk that was not measured here and should not be assumed
# free.
#
# SIZING: `DEFAULT_EMBED_BATCH_SIZE=48` at the measured worst-case ~711ms/
# chunk (this corpus's real content, n=32 sample) costs ~34s per batch —
# comfortable margin under the 60s ceiling (safety factor ~1.75x) without
# being so small that per-call Python/asyncio overhead starts to matter.
# Not swept/tuned beyond this safety-margin argument; revisit only if a
# future corpus's average chunk length changes enough to threaten the
# margin (`chunk_token_size` in `server/lightrag_init.py` bounds this, so a
# silent drift is unlikely) or if LightRAG's own `EMBEDDING_TIMEOUT`
# default changes.
DEFAULT_EMBED_BATCH_SIZE = 48


async def _embed_in_batches(
    embed_func: EmbedFunc, contents: list[str], context: str, batch_size: int
) -> np.ndarray:
    """Call `embed_func` in sequential chunks of at most `batch_size` texts,
    concatenating the results — see the module-level "WHY every call ...
    is BATCHED" note above for why a single unbatched call is unsafe
    against `rag.embedding_func` specifically. Behaves identically to one
    unbatched `await embed_func(contents, context=context)` call from the
    caller's perspective (same output shape/dtype/row-order), for any
    `batch_size >= 1`.

    Returns a `(0, 0)` float32 array for empty `contents` — callers already
    branch on `if contents:` before calling this (see `build_index`/
    `update_index`), so this is defensive, not a documented public contract
    for the empty case.
    """
    if not contents:
        return np.zeros((0, 0), dtype=np.float32)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    chunks_of_vectors: list[np.ndarray] = []
    for start in range(0, len(contents), batch_size):
        batch = contents[start : start + batch_size]
        raw = await embed_func(batch, context=context)
        vecs = np.asarray(raw, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] != len(batch):
            raise RuntimeError(
                f"embed_func returned shape {vecs.shape} for a batch of {len(batch)} "
                f"texts (offset {start}), expected ({len(batch)}, dim)"
            )
        chunks_of_vectors.append(vecs)
    return np.concatenate(chunks_of_vectors, axis=0)


class FlatIndexUnavailableError(RuntimeError):
    """Raised when the source `kv_store_text_chunks.json` does not exist yet.

    Same "nothing to do yet, not a real error" distinction as
    `retrieval.bm25_index.BM25IndexUnavailableError` — the LightRAG index
    itself hasn't been built.
    """


class FlatIndexDimensionMismatchError(RuntimeError):
    """Raised mid-incremental-update when freshly embedded vectors disagree
    in width with the cached matrix `update_index` is extending, despite an
    `embed_model` label match (e.g. the label was left unchanged by hand
    while the underlying model/config actually changed underneath it).

    Never silently concatenate mismatched-width rows: `FlatDenseIndex.search`
    does `self.vectors @ query_vec`, a single matmul over the whole matrix,
    so a width mismatch is not a localized bug — it is a hard shape error
    (or worse, a shape that happens to still multiply but scores garbage)
    for every future query, not just the newly-updated rows. Callers
    (`get_or_build_index`) catch this and fall back to a full rebuild.
    """


@dataclass(frozen=True)
class FlatSearchHit:
    chunk_id: str
    score: float
    content: str
    file_path: str


@dataclass(frozen=True)
class FlatBuildStats:
    chunk_count: int
    build_seconds: float
    source_mtime: float
    cache_hit: bool
    cache_dir: str
    embed_model: str
    # Populated for both the incremental and full-rebuild paths (zero/"" on
    # a pure cache hit, where nothing was compared or embedded); see
    # `get_or_build_index` — kept as trailing defaulted fields so existing
    # positional/keyword call sites that predate incremental update keep
    # working unchanged.
    update_path: str = "cache_hit"  # "cache_hit" | "incremental" | "full_rebuild"
    new_count: int = 0
    modified_count: int = 0
    deleted_count: int = 0
    reused_count: int = 0


@dataclass
class FlatDenseIndex:
    """In-memory flat dense index: a normalized (chunk_count, dim) float32
    matrix + chunk metadata. No ANN structure — see module docstring.
    """

    chunk_ids: list[str]
    vectors: np.ndarray  # (chunk_count, dim), float32, L2-normalized rows
    chunk_meta: dict[str, dict[str, str]]
    # sha256 hex digest of each chunk's content, keyed by chunk_id — the
    # change-detection signal `update_index` compares against on every
    # incremental update. See module docstring ("WHY hash-based change
    # detection, not id-based") for why id membership alone is not enough.
    chunk_hashes: dict[str, str]
    source_mtime: float
    embed_model: str

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_ids)

    async def search(self, query: str, embed_func: EmbedFunc, top_k: int) -> list[FlatSearchHit]:
        """Rank indexed chunks against `query`; up to `top_k` hits.

        `context="query"` is passed to `embed_func` deliberately — see module
        docstring for why this must match the query-side convention
        `server/embedder.py` applies for the production dense channel.

        `top_k` is clamped to the corpus size (mirrors
        `BM25SparseIndex.search`'s clamping — asking for more results than
        exist is a caller convenience, not a real request for padding).
        """
        if not query or self.chunk_count == 0:
            return []
        k = max(1, min(top_k, self.chunk_count))

        raw = await embed_func([query], context="query")
        query_vec = np.asarray(raw, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query_vec))
        if norm > 0.0:
            query_vec = query_vec / norm

        sims = self.vectors @ query_vec
        # argpartition for the top-k candidate set, then exact sort within
        # it — avoids a full O(N log N) sort for a corpus this size, though
        # at 7,184 rows either is sub-millisecond; kept for headroom as the
        # corpus grows toward the "revisit ANN" threshold noted above.
        if k >= len(sims):
            top_idx = np.argsort(-sims)
        else:
            candidate_idx = np.argpartition(-sims, k - 1)[:k]
            top_idx = candidate_idx[np.argsort(-sims[candidate_idx])]

        hits: list[FlatSearchHit] = []
        for idx in top_idx[:k]:
            chunk_id = self.chunk_ids[int(idx)]
            meta = self.chunk_meta.get(chunk_id, {})
            hits.append(
                FlatSearchHit(
                    chunk_id=chunk_id,
                    score=float(sims[int(idx)]),
                    content=str(meta.get("content", "")),
                    file_path=str(meta.get("file_path", "")),
                )
            )
        return hits


def _chunks_file_path(working_dir: str) -> Path:
    return Path(working_dir) / CHUNKS_FILENAME


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / _CACHE_META_FILENAME


def _vectors_path(cache_dir: Path) -> Path:
    return cache_dir / _CACHE_VECTORS_FILENAME


def _content_hash(content: str) -> str:
    """sha256 hex digest of a chunk's content — the change-detection unit
    for `update_index`. See module docstring ("WHY hash-based change
    detection, not id-based"): chunk ids are stable across re-ingests, chunk
    CONTENT under a given id is not, so id equality alone is never
    sufficient to call a chunk "unchanged".
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def build_index(
    working_dir: str,
    embed_func: EmbedFunc,
    *,
    embed_model: str = "",
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> tuple[FlatDenseIndex, float]:
    """Build a fresh `FlatDenseIndex` from `working_dir/kv_store_text_chunks.json`,
    embedding every chunk unconditionally.

    Returns `(index, build_seconds)`. Raises `FlatIndexUnavailableError` if
    the source file does not exist. `embed_model` is stored alongside the
    vectors purely as a label for cache-invalidation / diagnostics (see
    `get_or_build_index`) — this module never loads a model itself, so it
    cannot independently verify the vectors' dimensionality matches the
    label; that is the caller's responsibility (same division of
    responsibility as `server/embedder.py::validate_embedder_against_index`
    uses for the production LightRAG index).

    `embed_batch_size` is forwarded to `_embed_in_batches` (see its
    module-level "WHY every call ... is BATCHED" note above) — `embed_func`
    is NEVER called with more than this many texts at once, regardless of
    corpus size.

    This is the expensive, O(corpus size) path (measured: ~570-710ms/chunk
    against this corpus's real content, 4,094s / 68.2min for 7,184 chunks
    at the pre-batching unbatched cost) — `get_or_build_index` prefers
    `update_index` (O(chunks changed)) whenever a usable cache exists; this
    function is the fallback for "no cache", "cache unusable", and
    "different embed_model" (see `get_or_build_index`).
    """
    chunks_file = _chunks_file_path(working_dir)
    if not chunks_file.is_file():
        raise FlatIndexUnavailableError(
            f"No text-chunk store at {chunks_file} — build the LightRAG index first "
            "(python tools/memory/server/index.py)."
        )

    start = time.monotonic()
    with chunks_file.open("r", encoding="utf-8") as fh:
        raw_chunks: dict[str, dict[str, Any]] = json.load(fh)

    chunk_ids = sorted(raw_chunks.keys())  # deterministic positional order, matches bm25_index.py
    contents: list[str] = []
    chunk_meta: dict[str, dict[str, str]] = {}
    chunk_hashes: dict[str, str] = {}
    for chunk_id in chunk_ids:
        entry = raw_chunks[chunk_id]
        content = str(entry.get("content", ""))
        contents.append(content)
        chunk_meta[chunk_id] = {
            "content": content,
            "file_path": str(entry.get("file_path", "")),
        }
        chunk_hashes[chunk_id] = _content_hash(content)

    if contents:
        vectors = await _embed_in_batches(embed_func, contents, "document", embed_batch_size)
        # Defensive re-normalization: server/embedder.py's own encode() call
        # already sets normalize_embeddings=True, but this module takes
        # embed_func as an opaque parameter and must not silently trust an
        # unnormalized caller — cosine similarity via plain dot product (see
        # FlatDenseIndex.search) is only correct on unit-norm rows.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        vectors = (vectors / norms).astype(np.float32)
    else:
        vectors = np.zeros((0, 0), dtype=np.float32)
    build_seconds = time.monotonic() - start

    source_mtime = chunks_file.stat().st_mtime
    index = FlatDenseIndex(
        chunk_ids=chunk_ids,
        vectors=vectors,
        chunk_meta=chunk_meta,
        chunk_hashes=chunk_hashes,
        source_mtime=source_mtime,
        embed_model=embed_model,
    )
    logger.info(
        "Built flat dense index (full rebuild): %d chunks in %.3fs (source=%s, embed_model=%s)",
        len(chunk_ids), build_seconds, chunks_file, embed_model or "<unspecified>",
    )
    return index, build_seconds


@dataclass(frozen=True)
class FlatUpdateStats:
    """Stats for one `update_index` call — the observability this module's
    incremental path is built for: which chunks actually needed the
    ~570ms/chunk embedding cost, versus which rows were reused untouched.
    """

    chunk_count: int
    new_count: int
    modified_count: int
    deleted_count: int
    reused_count: int
    update_seconds: float
    source_mtime: float
    embed_model: str

    @property
    def embedded_count(self) -> int:
        return self.new_count + self.modified_count


async def update_index(
    working_dir: str,
    embed_func: EmbedFunc,
    cached: FlatDenseIndex,
    *,
    embed_model: str = "",
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> tuple[FlatDenseIndex, FlatUpdateStats]:
    """Bring `cached` up to date with the current
    `working_dir/kv_store_text_chunks.json`, embedding only chunks that are
    new or whose content changed, reusing every unchanged row's vector
    as-is, and dropping rows for chunks no longer present in the source.

    Change detection, precisely (see module docstring for the "why"):
    for each chunk id currently in the source, compare `_content_hash` of
    its CURRENT content against `cached.chunk_hashes.get(chunk_id)`:
      - id absent from `cached.chunk_ids`            -> NEW,  must embed.
      - id present, hash differs                     -> MODIFIED, must embed.
      - id present, hash matches                     -> UNCHANGED, reuse row.
    Any id in `cached.chunk_ids` no longer present in the current source
    (removed by `memory_forget` / `cleanup_kb.py`, or a re-chunk that
    dropped it) is a DELETION: its row is simply not carried into the
    output matrix — an index with orphan rows would let deleted content
    keep being returned by `search()`, which is exactly the case this
    function exists to prevent.

    Output row order is `sorted(current_chunk_ids)` — identical to what
    `build_index` would produce for the same source file — so an
    incrementally-updated index and a freshly full-rebuilt one are
    order-for-order identical, not just set-equal.

    `embed_batch_size` is forwarded to `_embed_in_batches` (see its
    module-level "WHY every call ... is BATCHED" note above) — `embed_func`
    is NEVER called with more than this many texts at once, regardless of
    how many chunks in `to_embed_ids` need embedding.

    Raises `FlatIndexUnavailableError` if the source chunk store does not
    exist. Raises `FlatIndexDimensionMismatchError` if newly embedded
    vectors disagree in width with `cached.vectors` (see that error's
    docstring) — callers should treat this as "cache unusable, fall back to
    `build_index`", not retry.
    """
    chunks_file = _chunks_file_path(working_dir)
    if not chunks_file.is_file():
        raise FlatIndexUnavailableError(
            f"No text-chunk store at {chunks_file} — build the LightRAG index first "
            "(python tools/memory/server/index.py)."
        )

    start = time.monotonic()
    with chunks_file.open("r", encoding="utf-8") as fh:
        raw_chunks: dict[str, dict[str, Any]] = json.load(fh)

    current_ids = sorted(raw_chunks.keys())
    current_id_set = set(current_ids)
    cached_id_set = set(cached.chunk_ids)
    cached_row_by_id = {chunk_id: i for i, chunk_id in enumerate(cached.chunk_ids)}

    new_ids: list[str] = []
    modified_ids: list[str] = []
    reused_ids: list[str] = []
    current_hashes: dict[str, str] = {}
    current_meta: dict[str, dict[str, str]] = {}

    for chunk_id in current_ids:
        entry = raw_chunks[chunk_id]
        content = str(entry.get("content", ""))
        content_hash = _content_hash(content)
        current_hashes[chunk_id] = content_hash
        current_meta[chunk_id] = {
            "content": content,
            "file_path": str(entry.get("file_path", "")),
        }
        if chunk_id not in cached_id_set:
            new_ids.append(chunk_id)
        elif cached.chunk_hashes.get(chunk_id) != content_hash:
            # Stable id, changed content — e.g. LightRAG re-chunked this
            # document on re-ingest. Comparing by id alone here would treat
            # this chunk as already-indexed and skip it, silently keeping
            # the OLD vector under an id that now points at different text
            # forever (nothing else would ever re-trigger embedding for it).
            # The hash comparison above is the only thing standing between
            # this and that stale-vector-under-stable-id trap.
            modified_ids.append(chunk_id)
        else:
            reused_ids.append(chunk_id)

    deleted_ids = cached_id_set - current_id_set
    to_embed_ids = new_ids + modified_ids

    embedded_by_id: dict[str, np.ndarray] = {}
    embedded_dim: int | None = None
    if to_embed_ids:
        contents_to_embed = [current_meta[cid]["content"] for cid in to_embed_ids]
        new_vectors = await _embed_in_batches(
            embed_func, contents_to_embed, "document", embed_batch_size
        )
        # Same defensive re-normalization as build_index — cosine similarity
        # via plain dot product is only correct on unit-norm rows.
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        new_vectors = (new_vectors / norms).astype(np.float32)
        embedded_dim = new_vectors.shape[1]
        if cached.vectors.ndim == 2 and cached.vectors.shape[0] > 0 and cached.vectors.shape[1] != embedded_dim:
            raise FlatIndexDimensionMismatchError(
                f"Freshly embedded vectors are {embedded_dim}-dim but the cached "
                f"matrix being updated is {cached.vectors.shape[1]}-dim "
                f"(embed_model label={cached.embed_model!r}). Cache is unusable "
                "for incremental update; caller should fall back to build_index."
            )
        embedded_by_id = {cid: new_vectors[i] for i, cid in enumerate(to_embed_ids)}

    dim = embedded_dim
    if dim is None:
        dim = cached.vectors.shape[1] if cached.vectors.ndim == 2 else 0
    out_vectors = np.zeros((len(current_ids), dim), dtype=np.float32)
    for row, chunk_id in enumerate(current_ids):
        if chunk_id in embedded_by_id:
            out_vectors[row] = embedded_by_id[chunk_id]
        else:
            out_vectors[row] = cached.vectors[cached_row_by_id[chunk_id]]

    update_seconds = time.monotonic() - start
    source_mtime = chunks_file.stat().st_mtime
    resolved_embed_model = embed_model or cached.embed_model

    index = FlatDenseIndex(
        chunk_ids=current_ids,
        vectors=out_vectors,
        chunk_meta=current_meta,
        chunk_hashes=current_hashes,
        source_mtime=source_mtime,
        embed_model=resolved_embed_model,
    )
    stats = FlatUpdateStats(
        chunk_count=len(current_ids),
        new_count=len(new_ids),
        modified_count=len(modified_ids),
        deleted_count=len(deleted_ids),
        reused_count=len(reused_ids),
        update_seconds=update_seconds,
        source_mtime=source_mtime,
        embed_model=resolved_embed_model,
    )
    logger.info(
        "Incrementally updated flat dense index: %d chunks total "
        "(%d new, %d modified, %d deleted, %d reused, %.3fs, source=%s, embed_model=%s)",
        stats.chunk_count, stats.new_count, stats.modified_count,
        stats.deleted_count, stats.reused_count, update_seconds, chunks_file,
        resolved_embed_model or "<unspecified>",
    )
    return index, stats


def save_index(index: FlatDenseIndex, cache_dir: str) -> None:
    """Persist `index` to `cache_dir`: vectors as `.npy`, everything else
    (chunk_ids/chunk_meta/chunk_hashes/source_mtime/embed_model) as a JSON
    sidecar — mirrors `retrieval/bm25_index.py::save_index`'s split between
    a library-native binary format and a hand-written metadata file.
    """
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(_vectors_path(out_dir), index.vectors)
    meta = {
        "chunk_ids": index.chunk_ids,
        "chunk_meta": index.chunk_meta,
        "chunk_hashes": index.chunk_hashes,
        "source_mtime": index.source_mtime,
        "embed_model": index.embed_model,
    }
    _meta_path(out_dir).write_text(json.dumps(meta), encoding="utf-8")


def load_index(cache_dir: str) -> FlatDenseIndex | None:
    """Load a previously persisted index from `cache_dir`, or `None` if
    absent OR unreadable.

    `None` (not an exception) on a missing cache mirrors
    `retrieval/bm25_index.py::load_index` — "no cache yet" is expected
    first-run state, not an error; `get_or_build_index` decides what to do.

    A cache that exists but fails to parse (truncated write, disk
    corruption, a JSON sidecar that isn't valid JSON, an `.npy` file that
    isn't a valid numpy array) is treated the same way — logged and
    returned as `None` — rather than propagating the raw parse exception:
    per this module's contract, a broken on-disk cache must fall back to a
    full rebuild (via `get_or_build_index`), never crash the caller or get
    silently trusted. Structural-but-parseable inconsistency (e.g. row
    counts that don't line up) is a separate, stricter check —
    `get_or_build_index` runs it on the returned index before deciding
    whether to trust this cache for an *incremental* update; a load that
    succeeds here can still be judged "too inconsistent to increment,
    rebuild instead" there.

    `chunk_hashes` defaults to `{}` for a sidecar written before this field
    existed (pre-incremental-update cache format) — deliberately, not
    treated as corruption: `get_or_build_index`'s usability check then finds
    every current chunk id missing from an empty `chunk_hashes`, so the
    first read of such a legacy cache is naturally treated as "not usable
    for incremental update" (one full rebuild to adopt the new format), not
    as license to silently skip hashing chunks it has no hash for.

    Vectors are loaded with `mmap_mode="r"` — cheap process startup (mirrors
    bm25s' own `mmap=True` used by `retrieval/bm25_index.py::load_index`);
    the returned array is read-only, which is correct here since nothing
    mutates `FlatDenseIndex.vectors` in place after a build/update (both
    always produce a fresh array).
    """
    out_dir = Path(cache_dir)
    meta_file = _meta_path(out_dir)
    vectors_file = _vectors_path(out_dir)
    if not meta_file.is_file() or not vectors_file.is_file():
        return None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        vectors = np.load(vectors_file, mmap_mode="r")
        return FlatDenseIndex(
            chunk_ids=list(meta["chunk_ids"]),
            vectors=vectors,
            chunk_meta=dict(meta["chunk_meta"]),
            chunk_hashes=dict(meta.get("chunk_hashes", {})),
            source_mtime=float(meta["source_mtime"]),
            embed_model=str(meta.get("embed_model", "")),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        logger.warning(
            "Flat dense cache at %s is unreadable/corrupt (%s: %s) — treating as absent; "
            "get_or_build_index will fall back to a full rebuild.",
            cache_dir, type(exc).__name__, exc,
        )
        return None


def _cache_is_usable_for_update(cached: FlatDenseIndex) -> bool:
    """Structural sanity check run on a *successfully loaded* cache before
    `get_or_build_index` trusts it as the base for an incremental
    `update_index` call.

    `load_index` already turns unparseable files into `None`; this catches
    the narrower case of a cache that parses fine but is internally
    inconsistent — row count / id list / metadata / hash dict out of
    lockstep (e.g. a partial write that landed a stale `.npy` next to a
    fresher sidecar, or a hand-edited/legacy file). Incrementing such a
    cache would silently propagate the inconsistency (an off-by-one here is
    exactly the "wrong document's content for a correct-looking id" bug
    this module's tests are required to guard against) — so any failure
    here routes to a full rebuild instead of an increment.
    """
    n = len(cached.chunk_ids)
    if len(set(cached.chunk_ids)) != n:
        return False  # duplicate ids — position-based lookups below would be ambiguous
    if cached.vectors.ndim != 2 or cached.vectors.shape[0] != n:
        return False
    if set(cached.chunk_meta.keys()) != set(cached.chunk_ids):
        return False
    if set(cached.chunk_hashes.keys()) != set(cached.chunk_ids):
        return False
    return True


async def get_or_build_index(
    working_dir: str,
    cache_dir: str,
    embed_func: EmbedFunc,
    *,
    embed_model: str = "",
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> tuple[FlatDenseIndex, FlatBuildStats]:
    """Return a ready-to-query `FlatDenseIndex`, choosing between three paths
    — cheapest safe option first:

    1. **Cache hit** (`update_path="cache_hit"`, cost O(1)): a usable cache
       exists, its `source_mtime` matches the current source file exactly,
       and its `embed_model` label matches. Nothing changed; return as-is.
    2. **Incremental update** (`update_path="incremental"`, cost O(chunks
       changed) — see `update_index`): a cache exists, passes
       `_cache_is_usable_for_update`, and its `embed_model` label matches
       (or none was requested). Only new/modified chunks (by content hash,
       not id — see `update_index`) are embedded; unchanged rows are reused
       and deleted chunks are dropped.
    3. **Full rebuild** (`update_path="full_rebuild"`, cost O(corpus size)
       — see `build_index`): no cache, a corrupt/structurally-inconsistent
       cache, a cache built under a different `embed_model` label (its
       vectors live in a different embedding space entirely — none of them
       are reusable, so there is nothing to increment), or an incremental
       update that hit `FlatIndexDimensionMismatchError` mid-way.

    Every path is logged (see `build_index` / `update_index` / this
    function) with which one ran and how many chunks were
    embedded/reused/dropped — this choice, and its cost, is the entire
    reason this module supports two update strategies instead of one; see
    the module docstring.

    Raises `FlatIndexUnavailableError` if the source chunk store itself does
    not exist (LightRAG index not built yet).
    """
    chunks_file = _chunks_file_path(working_dir)
    if not chunks_file.is_file():
        raise FlatIndexUnavailableError(
            f"No text-chunk store at {chunks_file} — build the LightRAG index first "
            "(python tools/memory/server/index.py)."
        )
    current_mtime = chunks_file.stat().st_mtime

    cached = load_index(cache_dir)  # None on missing OR corrupt (see load_index)

    embed_model_matches = not embed_model or cached is None or cached.embed_model in ("", embed_model)

    if cached is not None and cached.source_mtime == current_mtime and embed_model_matches:
        logger.info(
            "Flat dense index cache hit: %d chunks, source unchanged (cache=%s).",
            cached.chunk_count, cache_dir,
        )
        return cached, FlatBuildStats(
            chunk_count=cached.chunk_count,
            build_seconds=0.0,
            source_mtime=current_mtime,
            cache_hit=True,
            cache_dir=cache_dir,
            embed_model=cached.embed_model,
            update_path="cache_hit",
            reused_count=cached.chunk_count,
        )

    if cached is not None and not embed_model_matches:
        logger.info(
            "Flat dense cache embed_model mismatch (cached=%r, requested=%r) — vectors are "
            "in a different embedding space, nothing to reuse; full rebuild (cache=%s).",
            cached.embed_model, embed_model, cache_dir,
        )
        cached = None
    elif cached is not None and not _cache_is_usable_for_update(cached):
        logger.warning(
            "Flat dense cache at %s failed structural consistency checks — "
            "full rebuild rather than trusting it for an incremental update.",
            cache_dir,
        )
        cached = None

    if cached is not None:
        try:
            index, update_stats = await update_index(
                working_dir, embed_func, cached,
                embed_model=embed_model or cached.embed_model,
                embed_batch_size=embed_batch_size,
            )
        except FlatIndexDimensionMismatchError as exc:
            logger.warning(
                "Flat dense cache at %s unusable for incremental update (%s) — full rebuild.",
                cache_dir, exc,
            )
        else:
            save_index(index, cache_dir)
            return index, FlatBuildStats(
                chunk_count=index.chunk_count,
                build_seconds=update_stats.update_seconds,
                source_mtime=current_mtime,
                cache_hit=False,
                cache_dir=cache_dir,
                embed_model=index.embed_model,
                update_path="incremental",
                new_count=update_stats.new_count,
                modified_count=update_stats.modified_count,
                deleted_count=update_stats.deleted_count,
                reused_count=update_stats.reused_count,
            )

    index, build_seconds = await build_index(
        working_dir, embed_func, embed_model=embed_model, embed_batch_size=embed_batch_size
    )
    save_index(index, cache_dir)
    return index, FlatBuildStats(
        chunk_count=index.chunk_count,
        build_seconds=build_seconds,
        source_mtime=current_mtime,
        cache_hit=False,
        cache_dir=cache_dir,
        embed_model=embed_model,
        update_path="full_rebuild",
        new_count=index.chunk_count,
    )


__all__ = [
    "CHUNKS_FILENAME",
    "EmbedFunc",
    "DEFAULT_EMBED_BATCH_SIZE",
    "FlatIndexUnavailableError",
    "FlatIndexDimensionMismatchError",
    "FlatSearchHit",
    "FlatBuildStats",
    "FlatUpdateStats",
    "FlatDenseIndex",
    "build_index",
    "update_index",
    "save_index",
    "load_index",
    "get_or_build_index",
]
