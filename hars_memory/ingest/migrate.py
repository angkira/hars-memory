"""In-place migration of an existing LightRAG index to the location-aware chunk layout.

Why this exists
---------------
An index built before chunk location metadata existed has chunks without
``heading_path`` / ``start_line`` / ``end_line`` / ``section`` and with only a
*flattened* file name (``services-litellm-readme.md``, see
``server/index.py::_insert_all_batches`` — LightRAG dedupes by basename, so the
indexer encodes the relative directory with ``/`` -> ``-``).  Re-indexing to get
those fields would repeat the multi-hour GPU entity extraction.

What it does (and deliberately does not)
----------------------------------------
Zero LLM calls, and no embedder either: nothing here ever re-embeds.  This
module imports neither ``lightrag`` nor ``server.lightrag_init``; it edits the
on-disk JSON stores directly:

* ``kv_store_text_chunks.json`` — gains ``source_path``, ``heading_path``,
  ``section``, ``start_line``, ``end_line`` on every chunk it can locate.
* ``vdb_chunks.json`` (NanoVectorDB) — same fields in the per-chunk payload;
  the vector matrix is never decoded unless chunks are removed (dedupe), so
  metadata back-fill leaves every vector byte-identical.

Chunk *boundaries* are never changed.  LightRAG chunk ids are
``md5(chunk text)`` and the graph (entities/relations) references them through
``source_id``; re-cutting chunks would orphan the graph and force the very
re-extraction this command exists to avoid.  Chunks are instead *located*
inside the stored full document (``kv_store_full_docs.json`` — the exact text
that was chunked), which yields line numbers and the heading path.

Source text for line numbers is the stored full document with the
``[Document: ... | Section: ... | Date: ...]`` attribution header (inserted by
the walker, not present in the file on disk) subtracted, so numbers refer to the
real file as it was when indexed.  The real path is recovered by re-walking
``--root`` with the normal walker: ``doc_id`` is a hash of the absolute path, so
the walk maps ``doc_id -> relative path``.  Documents the walk cannot find still
get line/heading metadata, just no ``source_path``.

Safety
------
* ``dry_run`` computes and reports everything, writes nothing.
* A real run copies every file it is about to modify into
  ``<index>/migrate-index-backup-<UTC timestamp>/`` first; each file is then
  replaced atomically (write temp, ``os.replace``).
* Idempotent: a second run finds every chunk already carrying identical
  metadata, reports it as "kept", and writes nothing (no backup either).
* Stop the MCP server / any indexer first — the JSON stores are rewritten whole.

Dedupe
------
Opt-in.  Documents whose body (header stripped) is byte-identical are
duplicates; the canonical one is the document the walk still finds, then the
shortest relative path.  The duplicate's chunks that the canonical document does
not also own are removed from the chunk KV and chunk vector store; chunks they
share (same text => same id) are kept and re-pointed at the canonical document.
The duplicate's ``full_docs`` / ``doc_status`` / fingerprint records are kept on
purpose so a later ``memory-index`` run still sees it as "already processed"
instead of re-extracting it; the decision is recorded in ``migration_dedup.json``.
The graph is not edited, so entity ``source_id`` values may still name a removed
chunk id; retrieval skips ids with no chunk record.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hars_memory.ingest.chunker import heading_path_at, line_number_at, scan_headings

logger = logging.getLogger(__name__)

CHUNKS_FILENAME = "kv_store_text_chunks.json"
FULL_DOCS_FILENAME = "kv_store_full_docs.json"
DOC_STATUS_FILENAME = "kv_store_doc_status.json"
CHUNKS_VDB_FILENAME = "vdb_chunks.json"
DEDUP_LOG_FILENAME = "migration_dedup.json"
BACKUP_DIR_PREFIX = "migrate-index-backup-"

# Fields this migration owns on a chunk record / vector payload.
LOCATION_FIELDS = ("source_path", "heading_path", "section", "start_line", "end_line")

# Matches ``build_source_header`` output exactly (header line + blank line).
_HEADER_RE = re.compile(r"^\[Document: [^\n]*\| Date: [^\n]*\]\n\n", re.MULTILINE)
_HEADER_SEARCH_WINDOW = 4000
_MARKDOWN_SUFFIXES = (".md", ".markdown")


class MigrationError(RuntimeError):
    """Raised for a precondition failure (missing store, unsupported backend)."""


@dataclass(slots=True)
class MigrationReport:
    """Counts printed by ``memory migrate-index`` (all zero-LLM)."""

    chunks_total: int = 0
    chunks_kept: int = 0  # already carried identical metadata
    chunks_metadata_only: int = 0  # metadata back-filled, vector untouched
    chunks_reembedded: int = 0  # always 0: boundaries/text are never changed
    chunks_deleted: int = 0  # removed as part of a duplicate document
    chunks_unlocated: int = 0  # text not found in its stored document
    duplicate_docs: int = 0
    docs_without_path: int = 0  # not found by the walk: no source_path
    vector_store: str = "skipped"
    dry_run: bool = False
    backup_dir: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- JSON helpers -----------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise MigrationError(f"{path} does not contain a JSON object")
    return data


def _write_json_atomic(path: Path, data: Any, **dump_kwargs: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".migrate.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, **dump_kwargs), encoding="utf-8")
    os.replace(tmp, path)


# --- location computation ---------------------------------------------------


def _strip_header(doc_text: str) -> tuple[str, int, int]:
    """Return ``(body, header_line, header_line_count)``.

    ``header_line`` is the 1-based line of the attribution header in
    *doc_text* (0 when there is none); the header occupies
    ``header_line_count`` lines that do not exist in the file on disk.
    """
    match = _HEADER_RE.search(doc_text, 0, _HEADER_SEARCH_WINDOW)
    if match is None:
        return doc_text, 0, 0
    header = match.group(0)
    body = doc_text[: match.start()] + doc_text[match.end() :]
    return body, line_number_at(doc_text, match.start()), header.count("\n")


def _source_line(doc_line: int, header_line: int, header_lines: int) -> int:
    """Map a line of the stored (headered) document to the file on disk."""
    if not header_lines or doc_line <= header_line:
        return doc_line
    if doc_line < header_line + header_lines:
        return header_line
    return doc_line - header_lines


@dataclass(slots=True)
class _DocView:
    text: str
    headings: list[tuple[int, int, str]]
    header_line: int
    header_lines: int
    body_hash: str


def _build_doc_view(content: str, file_name: str) -> _DocView:
    body, header_line, header_lines = _strip_header(content)
    headings = scan_headings(content) if file_name.lower().endswith(_MARKDOWN_SUFFIXES) else []
    return _DocView(
        text=content,
        headings=headings,
        header_line=header_line,
        header_lines=header_lines,
        body_hash=hashlib.sha256(body.strip().encode("utf-8")).hexdigest(),
    )


def _locate(view: _DocView, chunk_text: str, hint: int) -> tuple[int, int] | None:
    """Find *chunk_text* in the stored document; return ``(start, end)`` chars."""
    start = view.text.find(chunk_text, hint)
    if start < 0 and hint:
        start = view.text.find(chunk_text)
    if start < 0:
        return None
    return start, start + len(chunk_text)


def _location_fields(view: _DocView, start: int, end: int, source_path: str | None) -> dict[str, Any]:
    path = heading_path_at(view.headings, start) if view.headings else ()
    fields: dict[str, Any] = {
        "heading_path": list(path),
        "section": path[-1] if path else "",
        "start_line": _source_line(line_number_at(view.text, start), view.header_line, view.header_lines),
        "end_line": _source_line(
            line_number_at(view.text, max(start, end - 1)), view.header_line, view.header_lines
        ),
    }
    if source_path:
        fields["source_path"] = source_path
    return fields


def _has_fields(record: dict[str, Any], fields: dict[str, Any]) -> bool:
    return all(record.get(key) == value for key, value in fields.items())


# --- path discovery ---------------------------------------------------------


def discover_paths(root: Path) -> dict[str, str]:
    """Map ``doc_id -> repo-relative posix path`` by re-walking *root*.

    Uses the normal walker (so ``.memoryignore`` applies) in dry-run mode: it
    only needs identity, not content.
    """
    from hars_memory.ingest.walker import walk

    resolved = root.resolve()
    docs, _ = walk([resolved], dry_run=True)
    mapping: dict[str, str] = {}
    for doc in docs:
        try:
            rel = Path(doc.source_path).resolve().relative_to(resolved)
        except ValueError:
            rel = Path(str(doc.metadata.get("relative_path") or doc.source_path))
        mapping[doc.doc_id] = rel.as_posix()
    return mapping


# --- the migration ----------------------------------------------------------


def migrate_index(
    index_dir: Path,
    *,
    root: Path | None = None,
    path_by_doc_id: dict[str, str] | None = None,
    dry_run: bool = False,
    dedupe: bool = False,
) -> MigrationReport:
    """Back-fill chunk location metadata (and optionally dedupe) in *index_dir*.

    *path_by_doc_id* short-circuits the walk (used right after an index run,
    which already holds the documents); otherwise *root* is walked, and with
    neither given no ``source_path`` is recorded.
    """
    chunks_file = index_dir / CHUNKS_FILENAME
    docs_file = index_dir / FULL_DOCS_FILENAME
    if not chunks_file.is_file() or not docs_file.is_file():
        raise MigrationError(
            f"{index_dir} is not a LightRAG index: need {CHUNKS_FILENAME} and {FULL_DOCS_FILENAME}"
        )

    report = MigrationReport(dry_run=dry_run)
    vdb_file = index_dir / CHUNKS_VDB_FILENAME
    qdrant = "qdrant" in os.environ.get("HARS_MEMORY_VECTOR_STORAGE", "").lower()
    if qdrant:
        report.notes.append(
            "Qdrant vector storage configured: payloads are NOT updated and --dedupe is refused "
            "(KV store only)."
        )
        if dedupe:
            raise MigrationError("--dedupe needs the NanoVectorDB chunk store; refusing with Qdrant configured")
    elif not vdb_file.is_file():
        report.notes.append(f"{CHUNKS_VDB_FILENAME} not found: vector payloads not updated")

    paths = path_by_doc_id
    if paths is None:
        paths = discover_paths(root) if root is not None else {}

    chunks = _read_json(chunks_file)
    full_docs = _read_json(docs_file)
    status = _read_json(index_dir / DOC_STATUS_FILENAME) if (index_dir / DOC_STATUS_FILENAME).is_file() else {}

    report.chunks_total = len(chunks)

    # Group chunks by owning document, in reading order.
    by_doc: dict[str, list[str]] = {}
    for chunk_id, record in chunks.items():
        by_doc.setdefault(str(record.get("full_doc_id", "")), []).append(chunk_id)
    for ids in by_doc.values():
        ids.sort(key=lambda cid: int(chunks[cid].get("chunk_order_index", 0) or 0))

    views: dict[str, _DocView] = {}
    # Identical documents share chunk ids (id = md5(text)), so a duplicate may
    # own no chunk record at all; status.chunks_list still names it.
    candidates = set(by_doc) | {
        doc_id for doc_id, rec in status.items()
        if isinstance(rec, dict) and any(c in chunks for c in rec.get("chunks_list") or [])
    }
    for doc_id in sorted(candidates):
        ids = by_doc.get(doc_id, [])
        doc_rec = full_docs.get(doc_id)
        content = doc_rec.get("content") if isinstance(doc_rec, dict) else None
        if isinstance(content, str):
            file_name = str(doc_rec.get("file_path") or (chunks[ids[0]].get("file_path", "") if ids else ""))
            views[doc_id] = _build_doc_view(content, file_name)
        else:
            report.chunks_unlocated += len(ids)

    # Dedupe is planned first so deleted chunks are never back-filled and
    # shared chunks are located against the document they will belong to.
    deleted: set[str] = set()
    repoint: dict[str, str] = {}
    dedup_log: dict[str, str] = {}
    if dedupe:
        deleted, repoint, dedup_log = _plan_dedupe(
            by_doc, views, paths, status, chunks, already=_read_dedup_log(index_dir)
        )
        report.duplicate_docs = len(dedup_log)
        report.chunks_deleted = len(deleted)

    updates: dict[str, dict[str, Any]] = {}  # chunk_id -> fields to set
    docs_without_path: set[str] = set()
    for doc_id, ids in by_doc.items():
        if doc_id not in views:
            continue
        hints: dict[str, int] = {}
        for chunk_id in ids:
            if chunk_id in deleted:
                continue
            owner = repoint.get(chunk_id, doc_id)
            source_path = paths.get(owner)
            if not source_path:
                docs_without_path.add(owner)
            record = chunks[chunk_id]
            span = _locate(views[owner], str(record.get("content", "")), hints.get(owner, 0))
            if span is None:
                if "heading_path" in record:
                    # Written by the markdown chunker: text carries a breadcrumb
                    # so it is not a verbatim slice, but it already has its
                    # location fields; only the path can still be added.
                    if source_path and record.get("source_path") != source_path:
                        updates[chunk_id] = {"source_path": source_path}
                    else:
                        report.chunks_kept += 1
                    continue
                report.chunks_unlocated += 1
                continue
            hints[owner] = span[0] + 1
            fields = _location_fields(views[owner], span[0], span[1], source_path)
            if _has_fields(record, fields):
                report.chunks_kept += 1
            else:
                updates[chunk_id] = fields
    report.docs_without_path = len(docs_without_path)
    report.chunks_metadata_only = len(updates)

    if qdrant:
        report.vector_store = "skipped (qdrant)"
    elif vdb_file.is_file():
        report.vector_store = "nano"

    if dry_run or not (updates or deleted or repoint or dedup_log):
        return report

    # --- write phase: backup first, then atomic per-file replace ----------
    targets = [chunks_file] + ([vdb_file] if report.vector_store == "nano" else [])
    backup_dir = index_dir / f"{BACKUP_DIR_PREFIX}{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    backup_dir.mkdir(parents=True)
    for target in targets:
        shutil.copy2(target, backup_dir / target.name)
    report.backup_dir = str(backup_dir)

    for chunk_id, fields in updates.items():
        chunks[chunk_id].update(fields)
    for chunk_id, canonical in repoint.items():
        chunks[chunk_id]["full_doc_id"] = canonical
    for chunk_id in deleted:
        chunks.pop(chunk_id, None)

    if report.vector_store == "nano":
        _rewrite_nano_vdb(vdb_file, updates, repoint, deleted)
    _write_json_atomic(chunks_file, chunks, indent=2)
    if dedup_log:
        log_path = index_dir / DEDUP_LOG_FILENAME
        existing = _read_dedup_log(index_dir)
        existing.update(dedup_log)
        _write_json_atomic(log_path, existing, indent=2, sort_keys=True)
    return report


def _read_dedup_log(index_dir: Path) -> dict[str, str]:
    log_path = index_dir / DEDUP_LOG_FILENAME
    return {str(k): str(v) for k, v in _read_json(log_path).items()} if log_path.is_file() else {}


def _plan_dedupe(
    by_doc: dict[str, list[str]],
    views: dict[str, _DocView],
    paths: dict[str, str],
    status: dict[str, Any],
    chunks: dict[str, Any],
    already: dict[str, str],
) -> tuple[set[str], dict[str, str], dict[str, str]]:
    """Return ``(chunk_ids_to_delete, shared_chunk_id -> canonical_doc, {dup: canonical})``."""
    groups: dict[str, list[str]] = {}
    for doc_id, view in views.items():
        if doc_id not in already and view.text.strip():
            groups.setdefault(view.body_hash, []).append(doc_id)

    def owned(doc_id: str) -> set[str]:
        listed = (status.get(doc_id) or {}).get("chunks_list") or []
        return set(by_doc.get(doc_id, [])) | {c for c in listed if c in chunks}

    deleted: set[str] = set()
    repoint: dict[str, str] = {}
    log: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        # Canonical: still on disk (walk found it), then shortest path, then id.
        canonical = min(
            members,
            key=lambda d: (d not in paths, len(paths.get(d, "~")), paths.get(d, ""), d),
        )
        canonical_ids = owned(canonical)
        for dup in members:
            if dup == canonical:
                continue
            log[dup] = canonical
            for chunk_id in owned(dup):
                if chunk_id in canonical_ids:
                    repoint[chunk_id] = canonical
                else:
                    deleted.add(chunk_id)
    return deleted, repoint, log


def _rewrite_nano_vdb(
    vdb_file: Path,
    updates: dict[str, dict[str, Any]],
    repoint: dict[str, str],
    deleted: set[str],
) -> None:
    """Edit payloads in place; drop matrix rows only when chunks are deleted."""
    store = _read_json(vdb_file)
    data = store.get("data")
    if not isinstance(data, list):
        raise MigrationError(f"{vdb_file} has no 'data' list")
    keep_rows: list[int] = []
    for row, entry in enumerate(data):
        chunk_id = entry.get("__id__")
        if chunk_id in deleted:
            continue
        keep_rows.append(row)
        if chunk_id in updates:
            entry.update(updates[chunk_id])
        if chunk_id in repoint:
            entry["full_doc_id"] = repoint[chunk_id]
    if len(keep_rows) != len(data):
        import numpy as np

        dim = int(store["embedding_dim"])
        matrix = np.frombuffer(base64.b64decode(store["matrix"]), dtype=np.float32).reshape(-1, dim)
        if matrix.shape[0] != len(data):
            raise MigrationError(
                f"{vdb_file}: matrix rows ({matrix.shape[0]}) != payload rows ({len(data)}); refusing to edit"
            )
        store["matrix"] = base64.b64encode(matrix[keep_rows].tobytes()).decode()
        store["data"] = [data[row] for row in keep_rows]
    _write_json_atomic(vdb_file, store)


def backfill_after_ingest(working_dir: object, docs: list[Any]) -> None:
    """Best-effort metadata back-fill for chunks LightRAG just wrote.

    Called by the indexers after ``finalize_storages()`` so freshly ingested
    chunks carry the same location fields a migrated index does.  Never raises:
    the index is already durable and this is purely additive.
    """
    if not isinstance(working_dir, (str, os.PathLike)) or not docs:
        return
    index_dir = Path(working_dir)
    if not (index_dir / CHUNKS_FILENAME).is_file():
        return
    cwd = Path.cwd().resolve()
    paths: dict[str, str] = {}
    for doc in docs:
        try:
            paths[doc.doc_id] = Path(doc.source_path).resolve().relative_to(cwd).as_posix()
        except (ValueError, OSError):
            rel = (doc.metadata or {}).get("relative_path")
            if rel:
                paths[doc.doc_id] = str(rel)
    try:
        report = migrate_index(index_dir, path_by_doc_id=paths)
        logger.info("Chunk location metadata: %d chunk(s) back-filled", report.chunks_metadata_only)
    except Exception as exc:  # noqa: BLE001 - additive step, never fail an index run
        logger.warning("Chunk location back-fill skipped: %s", exc)


def format_report(report: MigrationReport) -> str:
    """Human-readable summary for the CLI."""
    lines = [
        f"migrate-index ({'DRY RUN - nothing written' if report.dry_run else 'applied'})",
        f"  chunks total:          {report.chunks_total}",
        f"  kept (up to date):     {report.chunks_kept}",
        f"  metadata-only:         {report.chunks_metadata_only}",
        f"  re-embedded:           {report.chunks_reembedded}",
        f"  deleted (dup docs):    {report.chunks_deleted}",
        f"  duplicate docs:        {report.duplicate_docs}",
        f"  unlocated chunks:      {report.chunks_unlocated}",
        f"  docs without path:     {report.docs_without_path}",
        f"  vector store:          {report.vector_store}",
    ]
    if report.backup_dir:
        lines.append(f"  backup:                {report.backup_dir}")
    lines.extend(f"  note: {note}" for note in report.notes)
    return "\n".join(lines)


__all__ = [
    "LOCATION_FIELDS",
    "MigrationError",
    "MigrationReport",
    "backfill_after_ingest",
    "discover_paths",
    "format_report",
    "migrate_index",
]
