"""BM25 sparse retrieval index built from LightRAG's text-chunk KV store.

WHY bm25s (not rank_bm25 or a hand-rolled index): its default backend needs
only `numpy` (already a hard dependency of this project via `torch` /
`sentence-transformers` — verified before adding: `scipy` also happens to
already be present, but bm25s' numpy backend doesn't require it). `bm25s`
stores the term-document matrix as a sparse CSC array and ships built-in
`save()`/`load()` (with `mmap=True` support) — exactly the persist/invalidate
behaviour this module needs, without hand-rolling pickling. `rank_bm25` is
pure-Python and re-scores the *entire* corpus per query with no persistence
story, which does not scale past ~6.8k chunks (measured: 4-5x slower per
query than `bm25s`, and provides no save/load at all — every process start
pays the full tokenization+build cost). bm25s adds exactly 1 new PyPI
dependency to this project's lockfile.

WHY built from `kv_store_text_chunks.json`, not the vector index: it is
LightRAG's own on-disk record of every chunk's full text, keyed by chunk_id,
written with **no vectors and no LLM output** — the same source LightRAG
itself chunked from. Reading it never touches `vdb_*.json`, the GraphML graph,
or any embedding/LLM call, satisfying the "no reindexation" constraint
exactly: this is a read-only, CPU-only, sub-second operation over a file that
already exists.

WHY persisted to disk + mtime-invalidated: mirrors the `_graph_cache` pattern
already in `hars_longterm_memory_mcp.py` (`_get_graph()`) — cheap to rebuild once
per process, wasteful to rebuild on every query within a process, and
pointless to force every new server process to pay the build cost when a
previous process already built an index against the same, unchanged source
file.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hars_memory.retrieval.tokenizer import tokenize_identifiers

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback keeps the package importable
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CHUNKS_FILENAME = "kv_store_text_chunks.json"
_CACHE_META_FILENAME = "bm25_cache_meta.json"
_LOAD_RETRY_ATTEMPTS = 3
_LOAD_RETRY_SLEEP_SECONDS = 0.05


@contextlib.contextmanager
def _cache_lock(out_dir: Path, *, exclusive: bool):
    """Hold a process-safe lock while a published cache is swapped or opened.

    Atomic directory renames alone are insufficient for mmap-backed readers:
    bm25s opens several files by path, so a swap between those opens can mix
    files from two generations. A stable sibling lock serializes publication
    against that short open window; readers may still search their already-open
    mmap after releasing it.
    """
    lock_path = out_dir.parent / f".{out_dir.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        if fcntl is None:
            yield
            return
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(lock_file.fileno(), operation)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class BM25IndexUnavailableError(RuntimeError):
    """Raised when the source `kv_store_text_chunks.json` does not exist yet.

    This means the LightRAG index itself hasn't been built — there is nothing
    for BM25 to index. Distinct from a build/load *failure*, which is a real
    error; this is "nothing to do yet".
    """


class BM25CacheDirNotAbsoluteError(ValueError):
    """Raised when `cache_dir` is not an absolute path.

    A relative (or empty-string) `cache_dir` silently resolves against
    whatever the *calling process's* cwd happens to be, not a deliberate,
    configured location — `Path("")` and `Path(".")` both resolve to cwd.
    This is exactly how a stray direct call (bypassing the sanctioned
    `hars_longterm_memory_mcp.py` / `eval/ab_bench.py` entry points, both of
    which already pass absolute `/tmp/...` defaults) dumped this index's six
    on-disk files straight into the repo root instead of a cache directory
    (observed 2026-08-07: an out-of-band invocation landed six bm25s cache
    files at repo root while both sanctioned callers' own caches, correctly
    absolute, sat untouched in `/tmp`). Every caller must resolve `cache_dir`
    to an absolute path — env var, config, or explicit constant — before
    calling `save_index` / `load_index` / `get_or_build_index`.
    """


@dataclass(frozen=True)
class BM25SearchHit:
    chunk_id: str
    score: float
    content: str
    file_path: str


@dataclass(frozen=True)
class BM25BuildStats:
    chunk_count: int
    build_seconds: float
    source_mtime: float
    cache_hit: bool
    cache_dir: str


@dataclass
class BM25SparseIndex:
    """In-memory BM25 index + chunk metadata.

    `retriever` is a `bm25s.BM25` instance; typed `Any` here to keep `bm25s`
    an optional, lazily-imported dependency of this module (see `build_index`
    / `load_index`) rather than a hard import-time cost for callers that never
    touch the sparse channel.
    """

    retriever: Any
    chunk_ids: list[str]
    chunk_meta: dict[str, dict[str, str]]
    source_mtime: float

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_ids)

    def search(self, query: str, top_k: int) -> list[BM25SearchHit]:
        """Rank indexed chunks against `query`; up to `top_k` hits, score > 0 only.

        `top_k` is clamped to the corpus size — asking bm25s for more results
        than exist raises internally.
        """
        query_tokens = tokenize_identifiers(query)
        if not query_tokens or not self.chunk_ids:
            return []
        k = max(1, min(top_k, len(self.chunk_ids)))
        results, scores = self.retriever.retrieve(
            [query_tokens], k=k, show_progress=False, sorted=True
        )
        hits: list[BM25SearchHit] = []
        for idx, score in zip(results[0], scores[0]):
            score_f = float(score)
            if score_f <= 0.0:
                # bm25s pads to `k` with zero-score entries when fewer than k
                # documents actually match any query token — not real hits.
                continue
            chunk_id = self.chunk_ids[int(idx)]
            meta = self.chunk_meta.get(chunk_id, {})
            hits.append(
                BM25SearchHit(
                    chunk_id=chunk_id,
                    score=score_f,
                    content=str(meta.get("content", "")),
                    file_path=str(meta.get("file_path", "")),
                )
            )
        return hits

    async def asearch(self, query: str, top_k: int) -> list[BM25SearchHit]:
        """Async wrapper over `search()`, offloaded to a worker thread.

        `retriever.retrieve()` is a synchronous numpy scan over the whole
        term-document matrix (tens of ms at this corpus size, more as it
        grows) — running it inline in a coroutine blocks the event loop and
        starves concurrent queries. Async callers must use this wrapper.
        """
        return await asyncio.to_thread(self.search, query, top_k)


def _chunks_file_path(working_dir: str) -> Path:
    return Path(working_dir) / CHUNKS_FILENAME


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / _CACHE_META_FILENAME


def _require_absolute_cache_dir(cache_dir: str) -> Path:
    """Resolve `cache_dir` to a `Path`, refusing anything not already absolute.

    Fail fast and loud here rather than silently writing to `Path(cache_dir)`
    — see `BM25CacheDirNotAbsoluteError` for why a relative/empty value is a
    real hazard, not just a style nit.
    """
    path = Path(cache_dir)
    if not path.is_absolute():
        raise BM25CacheDirNotAbsoluteError(
            f"cache_dir must be an absolute path, got {cache_dir!r} — a "
            f"relative value silently resolves to {path.resolve()} (the "
            "calling process's cwd). Pass an absolute, explicitly configured "
            "directory."
        )
    return path


def build_index(working_dir: str) -> tuple[BM25SparseIndex, float]:
    """Build a fresh `BM25SparseIndex` from `working_dir/kv_store_text_chunks.json`.

    Returns `(index, build_seconds)`. Raises `BM25IndexUnavailableError` if the
    source file does not exist.
    """
    import bm25s  # lazy: keep bm25s off the import path for callers that never
    # touch the sparse channel (mirrors the lazy `import networkx` already
    # used for `_get_graph()` in hars_longterm_memory_mcp.py).

    chunks_file = _chunks_file_path(working_dir)
    if not chunks_file.is_file():
        raise BM25IndexUnavailableError(
            f"No text-chunk store at {chunks_file} — build the LightRAG index first "
            "(python tools/memory/server/index.py)."
        )

    start = time.monotonic()
    with chunks_file.open("r", encoding="utf-8") as fh:
        raw_chunks: dict[str, dict[str, Any]] = json.load(fh)

    chunk_ids = sorted(raw_chunks.keys())  # deterministic positional order
    corpus_tokens: list[list[str]] = []
    chunk_meta: dict[str, dict[str, str]] = {}
    for chunk_id in chunk_ids:
        entry = raw_chunks[chunk_id]
        content = str(entry.get("content", ""))
        corpus_tokens.append(tokenize_identifiers(content))
        chunk_meta[chunk_id] = {
            "content": content,
            "file_path": str(entry.get("file_path", "")),
        }

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    build_seconds = time.monotonic() - start

    source_mtime = chunks_file.stat().st_mtime
    index = BM25SparseIndex(
        retriever=retriever,
        chunk_ids=chunk_ids,
        chunk_meta=chunk_meta,
        source_mtime=source_mtime,
    )
    logger.info(
        "Built BM25 index: %d chunks in %.3fs (source=%s)",
        len(chunk_ids),
        build_seconds,
        chunks_file,
    )
    return index, build_seconds


def save_index(index: BM25SparseIndex, cache_dir: str) -> None:
    """Persist `index` to `cache_dir` (bm25s' native sparse-matrix format + a
    sidecar JSON for chunk_ids/chunk_meta/source_mtime, which bm25s itself
    does not track).

    `corpus=None` on `retriever.save()` deliberately skips bm25s' own
    `corpus.jsonl` — we already keep `chunk_meta` (content + file_path) in the
    sidecar, so persisting the chunk text a second time would double the
    on-disk footprint for no benefit.

    Published atomically: bm25s writes several files (the CSC matrix parts,
    `params.index.json`, ...) plus this module's sidecar, one after another.
    Written straight into `cache_dir`, a concurrent reader (another MCP server
    process calling `load_index`) can observe a half-written directory — a
    fresh sidecar next to a stale/truncated matrix — and either crash or
    silently mis-rank. So the whole set is staged in a sibling temp directory
    and published with a single `os.replace` of the directory, which is atomic
    on POSIX. The sidecar is written last *inside* the staging directory, so a
    published `cache_dir` is never missing it.
    """
    out_dir = _require_absolute_cache_dir(cache_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=str(out_dir.parent))
    staging_dir = Path(staging_parent) / out_dir.name
    try:
        staging_dir.mkdir(parents=True, exist_ok=True)
        index.retriever.save(str(staging_dir), corpus=None, show_progress=False)
        meta = {
            "chunk_ids": index.chunk_ids,
            "chunk_meta": index.chunk_meta,
            "source_mtime": index.source_mtime,
        }
        _meta_path(staging_dir).write_text(json.dumps(meta), encoding="utf-8")

        with _cache_lock(out_dir, exclusive=True):
            if out_dir.exists():
                # os.replace() onto an existing directory fails (non-empty
                # destination), so swap the old one aside first, publish, then
                # drop it while readers are excluded from the multi-file open.
                retired_dir = Path(staging_parent) / f"{out_dir.name}.retired"
                os.replace(str(out_dir), str(retired_dir))
            os.replace(str(staging_dir), str(out_dir))
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def load_index(cache_dir: str) -> BM25SparseIndex | None:
    """Load a previously persisted index from `cache_dir`, or `None` if absent.

    Returning `None` (not raising) on a missing cache is deliberate: "no cache
    yet" is the expected first-run state, not an error — `get_or_build_index`
    is the caller that decides what to do about it (build).
    """
    out_dir = _require_absolute_cache_dir(cache_dir)
    # `save_index` publishes by renaming directories, so `cache_dir` is
    # briefly absent (two rename() syscalls) even though each published
    # state is complete. A reader that opened the sidecar just before the
    # swap would otherwise hit FileNotFoundError on the matrix files; retry
    # a couple of times so a concurrent publish degrades to a few ms of
    # latency rather than an error.
    last_error: OSError | None = None
    for attempt in range(_LOAD_RETRY_ATTEMPTS):
        try:
            with _cache_lock(out_dir, exclusive=False):
                return _load_index_once(out_dir)
        except FileNotFoundError as exc:
            last_error = exc
            if attempt + 1 < _LOAD_RETRY_ATTEMPTS:
                time.sleep(_LOAD_RETRY_SLEEP_SECONDS)
    logger.warning("BM25 cache at %s unreadable after retries: %s", out_dir, last_error)
    return None


def _load_index_once(out_dir: Path) -> BM25SparseIndex | None:
    import bm25s

    meta_file = _meta_path(out_dir)
    if not meta_file.is_file():
        return None
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    retriever = bm25s.BM25.load(str(out_dir), load_corpus=False, mmap=True, show_progress=False)
    return BM25SparseIndex(
        retriever=retriever,
        chunk_ids=list(meta["chunk_ids"]),
        chunk_meta=dict(meta["chunk_meta"]),
        source_mtime=float(meta["source_mtime"]),
    )


def invalidate_cache(cache_dir: str) -> bool:
    """Drop the persisted BM25 cache at `cache_dir`. True iff something was removed.

    `get_or_build_index` already rebuilds when the source chunk store's mtime
    moves, which covers the normal insert path. This exists for the ONE case
    that mtime alone does not make obvious: document garbage collection
    (`server/index.py` deleting doc_ids whose files are gone). There the chunk
    store shrinks, and a cache still holding the removed chunks would keep
    serving deleted content as sparse hits until the next mtime-visible write.
    Removing the directory outright is the fail-safe move: the worst case is
    one extra rebuild.

    Removal is itself atomic-ish: the directory is renamed aside first, so a
    concurrent reader never walks a partially deleted cache.
    """
    out_dir = _require_absolute_cache_dir(cache_dir)
    if not out_dir.exists():
        return False
    retired = out_dir.with_name(f".{out_dir.name}.invalidated-{os.getpid()}")
    try:
        os.replace(str(out_dir), str(retired))
    except OSError as exc:
        logger.warning("Could not invalidate BM25 cache %s: %s", out_dir, exc)
        return False
    shutil.rmtree(retired, ignore_errors=True)
    logger.info("Invalidated BM25 cache at %s", out_dir)
    return True


def get_or_build_index(working_dir: str, cache_dir: str) -> tuple[BM25SparseIndex, BM25BuildStats]:
    """Return a ready-to-query `BM25SparseIndex`, rebuilding only when the
    source `kv_store_text_chunks.json` mtime has changed since the on-disk
    cache (in `cache_dir`) was written, or no cache exists yet.

    Raises `BM25IndexUnavailableError` if the source chunk store itself does
    not exist (LightRAG index not built yet).
    """
    chunks_file = _chunks_file_path(working_dir)
    if not chunks_file.is_file():
        raise BM25IndexUnavailableError(
            f"No text-chunk store at {chunks_file} — build the LightRAG index first "
            "(python tools/memory/server/index.py)."
        )
    current_mtime = chunks_file.stat().st_mtime

    cached = load_index(cache_dir)
    if cached is not None and cached.source_mtime == current_mtime:
        return cached, BM25BuildStats(
            chunk_count=cached.chunk_count,
            build_seconds=0.0,
            source_mtime=current_mtime,
            cache_hit=True,
            cache_dir=cache_dir,
        )

    index, build_seconds = build_index(working_dir)
    save_index(index, cache_dir)
    return index, BM25BuildStats(
        chunk_count=index.chunk_count,
        build_seconds=build_seconds,
        source_mtime=current_mtime,
        cache_hit=False,
        cache_dir=cache_dir,
    )


__all__ = [
    "CHUNKS_FILENAME",
    "BM25IndexUnavailableError",
    "BM25CacheDirNotAbsoluteError",
    "BM25SearchHit",
    "BM25BuildStats",
    "BM25SparseIndex",
    "build_index",
    "invalidate_cache",
    "save_index",
    "load_index",
    "get_or_build_index",
]
