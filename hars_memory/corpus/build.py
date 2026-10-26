"""LLM-free corpus build: walk directories, chunk text, emit a queryable index.

No entity extraction, no LLM call, no GPU — this is the CPU-only sibling of
``server/index.py`` (the LightRAG graph builder). It reuses:

- ``ingest.walker.walk`` for directory traversal / glob filtering / binary
  and size guards / ``.memoryignore`` support.
- ``ingest.chunker.chunk_text`` for overlapping character-level chunking.
- ``ingest.change_detection.compute_fingerprint`` for the sha256 content hash
  used both per-document (incremental diffing) and to derive the corpus-wide
  fingerprint.

Output contract
----------------
``build_corpus`` emits exactly two files under ``index_dir``:

``kv_store_text_chunks.json``
    ``chunk_id -> {"content": str, "file_path": str, "chunk_index": int,
    "source_doc_id": str}``. This is the SAME shape (and filename) that
    ``retrieval/bm25_index.py`` and ``retrieval/flat_index.py`` already read
    via ``CHUNKS_FILENAME`` — both existing retrieval channels work against
    a corpus-built index with zero code change. Extra keys beyond
    ``content``/``file_path`` are additive; both readers only ever access
    those two via ``entry.get(...)``.
``corpus_manifest.json``
    Provenance record: one entry per source document (absolute path,
    ``file_stable_id``, content sha256, size, mtime, chunk ids) plus a
    top-level build record (timestamp, chunk_size/overlap, globs, tool
    version, ``corpus_fingerprint``, ``chunk_store_sha256``). This is the
    reproducibility record ``eval/corpus_eval.py`` and ``eval/regression.py``
    key off.

Chunk id scheme
----------------
``chunk-{sha256(f"{doc_id}:{chunk_index}")[:16]}`` — deterministic and stable
across rebuilds of identical content: the same document at the same chunk
position always yields the same id, so an unchanged document's chunk ids are
byte-identical build over build (verified by
``tests/test_corpus_build.py::test_deterministic_rebuild``).

Incremental re-index
---------------------
Every build is computed as a full walk + full re-chunk in memory (chunking
is pure-Python and CPU-cheap — there is no embedding step to amortize here,
unlike ``retrieval/flat_index.py``'s incremental update, which exists
specifically to avoid re-running the expensive embedding model). "Incremental"
in this module means *comparison and reporting*, not skipped work: if a
manifest already exists at ``index_dir``, each currently-walked document is
classified against it (added / changed / unchanged) by comparing content
sha256 (not just doc_id presence — see
``ingest.change_detection``'s module docstring for why content, not id,
is the correct change signal), and any document present in the OLD manifest
but absent from the current walk is a deletion. Because the chunk store is
always rebuilt from exactly what ``ingest.walker.walk`` returns for the
CURRENT filesystem state, a renamed file (which walker sees as "old path
gone, new path appeared", since ``file_stable_id`` hashes the path) can never
leave orphan chunks: the old path's chunks simply do not exist in the fresh
output.

Build isolation
-----------------
All computation (walk, chunk, manifest diff) happens in memory BEFORE any
file is written under ``index_dir`` or its temp sibling. Output is written
to a temp directory next to ``index_dir`` and only swapped into place via
``os.replace`` (atomic rename) once every file has been written successfully
— see ``_atomic_swap``. A build that raises at any point before the swap
step touches nothing under ``index_dir``; a pre-existing index is therefore
never left partially overwritten by a failed/interrupted build.

This builder refuses outright (``LiveGraphIndexGuardError``) to write into a
directory that already contains LightRAG graph artifacts
(``graph_chunk_entity_relation.graphml`` / ``kv_store_full_entities.json``)
— see that class's docstring. This check runs FIRST, before any walk/chunk
work, so a misdirected ``--index-dir`` fails immediately and loudly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from hars_memory.ingest.change_detection import compute_fingerprint
from hars_memory.ingest.chunker import chunk_text
from hars_memory.ingest.document import Document
from hars_memory.ingest.walker import WalkStats, walk
from hars_memory.retrieval.bm25_index import CHUNKS_FILENAME

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named defaults — no magic numbers inline in build_corpus's signature.
# ---------------------------------------------------------------------------

# Matches ingest.chunker.chunk_text's own defaults exactly (named here too so
# build_corpus's public signature is self-documenting and CLI --chunk-size /
# --chunk-overlap flags have an explicit constant to reference, not a bare
# literal reaching into chunker.py).
DEFAULT_CHUNK_SIZE: Final[int] = 1200
DEFAULT_CHUNK_OVERLAP: Final[int] = 200

MANIFEST_FILENAME: Final[str] = "corpus_manifest.json"
TOOL_VERSION: Final[str] = "1.0.0"

# LightRAG graph-index artifacts — presence of EITHER means index_dir is a
# live GraphRAG index, not a corpus-build target. Matches the exact filenames
# server/index.py's LightRAG instance writes (verified against the deployed
# index at /home/demo/.local/share/hars-graphrag/index_v4).
_LIGHTRAG_GRAPH_ARTIFACT_FILENAMES: Final[tuple[str, ...]] = (
    "graph_chunk_entity_relation.graphml",
    "kv_store_full_entities.json",
)

_TMP_DIR_PREFIX: Final[str] = ".build-tmp-"
_BACKUP_DIR_PREFIX: Final[str] = ".build-old-"


class CorpusBuildError(Exception):
    """Base class for every corpus-build-specific failure in this module."""


class NoPathsProvidedError(CorpusBuildError):
    """Raised when ``build_corpus`` is called with an empty ``paths`` list."""

    def __init__(self) -> None:
        super().__init__("build_corpus requires at least one path in `paths`")


class CorpusPathNotFoundError(CorpusBuildError):
    """Raised when one of ``paths`` does not exist on disk.

    ``ingest.walker.walk`` itself only *warns and skips* a missing path
    (appropriate for its own multi-root, best-effort contract) — this
    builder is stricter: a nonexistent path is almost always an operator
    typo, and silently indexing zero documents from it is exactly the kind
    of "ran, did nothing, looked fine" failure this module's fail-fast
    contract exists to prevent.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(f"Path does not exist: {path}")
        self.path = path


class EmptyCorpusError(CorpusBuildError):
    """Raised when a walk over valid, existing paths yields zero documents.

    E.g. every candidate file was excluded by globs/`.memoryignore`, was
    binary, or exceeded the walker's size guard. An index with zero chunks
    is never a legitimate build output.
    """

    def __init__(self, paths: list[Path]) -> None:
        super().__init__(
            f"No documents found under {[str(p) for p in paths]} — check "
            "include/exclude globs and .memoryignore."
        )
        self.paths = paths


class LiveGraphIndexGuardError(CorpusBuildError):
    """Raised when ``index_dir`` already contains LightRAG graph artifacts.

    This builder is LLM-free and never produces (or understands) a
    ``graph_chunk_entity_relation.graphml`` / entity/relationship store —
    overwriting a directory that has one would silently destroy a live
    GraphRAG index's graph layer while leaving behind a chunk-store-only
    directory that LOOKS like a valid index (both readers only check
    ``kv_store_text_chunks.json``) but has lost every entity/relationship a
    caller's graph-mode queries depend on. This guard refuses outright,
    before any work happens — there is no ``--force`` override for it.
    """

    def __init__(self, index_dir: Path, found: list[str]) -> None:
        super().__init__(
            f"Refusing to build into {index_dir}: it already contains LightRAG "
            f"graph artifact(s) {found} — this is a LIVE GRAPH INDEX, not a "
            "corpus-build target. This LLM-free builder must never overwrite "
            "it. Point --index-dir at a different, corpus-build-owned "
            "directory."
        )
        self.index_dir = index_dir
        self.found = found


class UnsafeOverwriteTargetError(CorpusBuildError):
    """Raised when ``index_dir`` exists, is non-empty, and was never itself
    produced by ``build_corpus`` (no ``corpus_manifest.json`` marker) —
    e.g. an operator's ``--index-dir`` typo pointing at an unrelated,
    populated directory. Distinct from ``LiveGraphIndexGuardError`` (which
    is never bypassable): this is a softer "are you sure" guard, and
    ``force=True`` (CLI ``--force``) opts out of it. A directory that
    already has a ``corpus_manifest.json`` (i.e. a prior corpus build) is
    always safe to rebuild into and never needs ``force``.
    """

    def __init__(self, index_dir: Path) -> None:
        super().__init__(
            f"Refusing to build into {index_dir}: it exists, is non-empty, and "
            "was not produced by a prior build_corpus run (no "
            f"{MANIFEST_FILENAME} found). Pass force=True (CLI --force) to "
            "overwrite it anyway."
        )
        self.index_dir = index_dir


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """One document's provenance entry in ``corpus_manifest.json``."""

    doc_id: str
    source_path: str
    content_sha256: str
    size_bytes: int
    mtime: float
    chunk_count: int
    chunk_ids: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "file_stable_id": self.doc_id,
            "content_sha256": self.content_sha256,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime,
            "chunk_count": self.chunk_count,
            "chunk_ids": list(self.chunk_ids),
        }


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Outcome of one ``build_corpus`` call."""

    index_dir: Path
    manifest_path: Path
    chunk_store_path: Path
    corpus_fingerprint: str
    chunk_store_sha256: str
    document_count: int
    chunk_count: int
    added: int
    changed: int
    unchanged: int
    deleted: int
    walk_stats: WalkStats
    build_seconds: float


def _chunk_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic chunk id — see module docstring "Chunk id scheme"."""
    digest = hashlib.sha256(f"{doc_id}:{chunk_index}".encode("utf-8")).hexdigest()
    return f"chunk-{digest[:16]}"


def _corpus_fingerprint(records: list[DocumentRecord]) -> str:
    """sha256 over the sorted ``(source_path, content_hash)`` pairs.

    Sorted so the fingerprint is independent of walk/dict iteration order —
    the same set of (path, content) pairs always yields the same fingerprint
    regardless of how the filesystem happened to enumerate them.
    """
    pairs = sorted((r.source_path, r.content_sha256) for r in records)
    canonical = "\n".join(f"{path}:{content_hash}" for path, content_hash in pairs)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _guard_against_live_graph_index(index_dir: Path) -> None:
    if not index_dir.is_dir():
        return
    found = [
        name
        for name in _LIGHTRAG_GRAPH_ARTIFACT_FILENAMES
        if (index_dir / name).exists()
    ]
    if found:
        raise LiveGraphIndexGuardError(index_dir, found)


def _guard_against_unsafe_overwrite(index_dir: Path, *, force: bool) -> None:
    if force or not index_dir.is_dir():
        return
    has_content = any(index_dir.iterdir())
    if not has_content:
        return
    if (index_dir / MANIFEST_FILENAME).is_file():
        return  # a prior corpus build — always safe to rebuild into
    raise UnsafeOverwriteTargetError(index_dir)


def _load_previous_manifest(index_dir: Path) -> dict[str, dict[str, object]]:
    """Return ``{doc_id: document_record_json}`` from a prior build's
    manifest at ``index_dir``, or ``{}`` if none exists (first build).
    """
    manifest_path = index_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Existing manifest at %s is unreadable (%s) — treating as no "
            "prior build for incremental-diff reporting purposes.",
            manifest_path, exc,
        )
        return {}
    documents = raw.get("documents", {})
    return documents if isinstance(documents, dict) else {}


@dataclass(slots=True)
class _DiffCounts:
    added: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0


def _classify(
    current_records: list[DocumentRecord], previous_documents: dict[str, dict[str, object]]
) -> _DiffCounts:
    counts = _DiffCounts()
    current_ids = {r.doc_id for r in current_records}
    for record in current_records:
        prev = previous_documents.get(record.doc_id)
        if prev is None:
            counts.added += 1
        elif prev.get("content_sha256") != record.content_sha256:
            counts.changed += 1
        else:
            counts.unchanged += 1
    counts.deleted = sum(1 for doc_id in previous_documents if doc_id not in current_ids)
    return counts


def _build_chunk_store_and_records(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> tuple[dict[str, dict[str, object]], list[DocumentRecord]]:
    chunk_store: dict[str, dict[str, object]] = {}
    records: list[DocumentRecord] = []
    for doc in docs:
        chunks = chunk_text(doc.content, doc.doc_id, chunk_size, chunk_overlap)
        chunk_ids: list[str] = []
        for chunk in chunks:
            cid = _chunk_id(doc.doc_id, chunk.chunk_index)
            chunk_ids.append(cid)
            chunk_store[cid] = {
                "content": chunk.text,
                "file_path": doc.source_path,
                "chunk_index": chunk.chunk_index,
                "source_doc_id": doc.doc_id,
            }
        size_bytes = int(doc.metadata.get("size_bytes", 0) or 0)
        mtime = float(doc.metadata.get("mtime", 0.0) or 0.0)
        records.append(
            DocumentRecord(
                doc_id=doc.doc_id,
                source_path=doc.source_path,
                content_sha256=compute_fingerprint(doc.content),
                size_bytes=size_bytes,
                mtime=mtime,
                chunk_count=len(chunk_ids),
                chunk_ids=tuple(chunk_ids),
            )
        )
    return chunk_store, records


def _atomic_swap(tmp_dir: Path, index_dir: Path) -> None:
    """Atomically publish ``tmp_dir`` as ``index_dir``.

    Two ``os.replace`` (atomic rename) calls, not one: POSIX ``rename``
    cannot replace a non-empty directory in a single call. If ``index_dir``
    already exists it is first moved aside to a backup sibling, then
    ``tmp_dir`` is renamed into place; the backup is only removed after that
    second rename succeeds. If the second rename fails, the backup is moved
    back so a pre-existing index is never left missing.
    """
    if not index_dir.exists():
        os.replace(tmp_dir, index_dir)
        return

    backup_dir = index_dir.parent / f"{_BACKUP_DIR_PREFIX}{index_dir.name}-{uuid.uuid4().hex[:8]}"
    os.replace(index_dir, backup_dir)
    try:
        os.replace(tmp_dir, index_dir)
    except OSError:
        os.replace(backup_dir, index_dir)
        raise
    else:
        shutil.rmtree(backup_dir, ignore_errors=True)


def build_corpus(
    paths: list[Path],
    index_dir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    include_globs: tuple[str, ...] | None = None,
    exclude_globs: tuple[str, ...] | None = None,
    force: bool = False,
) -> BuildResult:
    """Build (or incrementally re-index) an LLM-free, CPU-only corpus index.

    Parameters
    ----------
    paths:
        Directories (or individual files) to walk. Must be non-empty; every
        entry must exist.
    index_dir:
        Target directory for ``kv_store_text_chunks.json`` +
        ``corpus_manifest.json``. Never written to directly — see module
        docstring "Build isolation".
    chunk_size, chunk_overlap:
        Forwarded to ``ingest.chunker.chunk_text``.
    include_globs, exclude_globs:
        Forwarded to ``ingest.walker.walk``. ``None`` (the default) lets
        the walker apply its own defaults rather than this module
        duplicating those glob lists as a second set of magic values.
    force:
        Bypass ``UnsafeOverwriteTargetError`` (an existing, non-empty
        ``index_dir`` that was never itself produced by ``build_corpus``).
        Never bypasses ``LiveGraphIndexGuardError`` — that guard has no
        override.

    Raises
    ------
    NoPathsProvidedError, CorpusPathNotFoundError, EmptyCorpusError,
    LiveGraphIndexGuardError, UnsafeOverwriteTargetError
    """
    if not paths:
        raise NoPathsProvidedError()
    for path in paths:
        if not path.exists():
            raise CorpusPathNotFoundError(path)

    index_dir = index_dir.resolve()
    _guard_against_live_graph_index(index_dir)
    _guard_against_unsafe_overwrite(index_dir, force=force)

    start = time.monotonic()

    walk_kwargs: dict[str, object] = {}
    if include_globs is not None:
        walk_kwargs["include_globs"] = include_globs
    if exclude_globs is not None:
        walk_kwargs["exclude_globs"] = exclude_globs
    docs, walk_stats = walk(paths, **walk_kwargs)  # type: ignore[arg-type]
    if not docs:
        raise EmptyCorpusError(paths)

    chunk_store, records = _build_chunk_store_and_records(docs, chunk_size, chunk_overlap)
    previous_documents = _load_previous_manifest(index_dir)
    diff = _classify(records, previous_documents)

    corpus_fingerprint = _corpus_fingerprint(records)
    chunk_store_json = json.dumps(chunk_store, sort_keys=True, indent=2)
    chunk_store_sha256 = hashlib.sha256(chunk_store_json.encode("utf-8")).hexdigest()

    manifest = {
        "build": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "include_globs": list(include_globs) if include_globs is not None else None,
            "exclude_globs": list(exclude_globs) if exclude_globs is not None else None,
            "tool_version": TOOL_VERSION,
            "corpus_fingerprint": corpus_fingerprint,
            "chunk_store_sha256": chunk_store_sha256,
            "chunk_id_scheme": 'chunk-{sha256(f"{doc_id}:{chunk_index}")[:16]}',
            "paths": [str(p.resolve()) for p in paths],
        },
        "documents": {r.doc_id: r.to_json() for r in records},
    }
    manifest_json = json.dumps(manifest, sort_keys=True, indent=2)

    tmp_dir = index_dir.parent / f"{_TMP_DIR_PREFIX}{index_dir.name}-{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        (tmp_dir / CHUNKS_FILENAME).write_text(chunk_store_json, encoding="utf-8")
        (tmp_dir / MANIFEST_FILENAME).write_text(manifest_json, encoding="utf-8")
        _atomic_swap(tmp_dir, index_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    build_seconds = time.monotonic() - start
    logger.info(
        "Corpus build complete: %d documents, %d chunks (added=%d changed=%d "
        "unchanged=%d deleted=%d) in %.3fs -> %s",
        len(records), len(chunk_store), diff.added, diff.changed,
        diff.unchanged, diff.deleted, build_seconds, index_dir,
    )
    return BuildResult(
        index_dir=index_dir,
        manifest_path=index_dir / MANIFEST_FILENAME,
        chunk_store_path=index_dir / CHUNKS_FILENAME,
        corpus_fingerprint=corpus_fingerprint,
        chunk_store_sha256=chunk_store_sha256,
        document_count=len(records),
        chunk_count=len(chunk_store),
        added=diff.added,
        changed=diff.changed,
        unchanged=diff.unchanged,
        deleted=diff.deleted,
        walk_stats=walk_stats,
        build_seconds=build_seconds,
    )


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "MANIFEST_FILENAME",
    "TOOL_VERSION",
    "CorpusBuildError",
    "NoPathsProvidedError",
    "CorpusPathNotFoundError",
    "EmptyCorpusError",
    "LiveGraphIndexGuardError",
    "UnsafeOverwriteTargetError",
    "DocumentRecord",
    "BuildResult",
    "build_corpus",
]
