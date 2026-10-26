"""Document model — the unit that flows from walker/postgres-exporter into LightRAG.

No LLM calls here.  This layer is fully GPU-free.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final

# Literal date value used when no real ingest/commit date could be determined
# for a document. Matches the convention already used by cleanup_kb.py's
# HEADER_RE (`[0-9]{4}-[0-9]{2}-[0-9]{2}|unknown`) — an "unknown"-dated doc is
# never purged by cleanup_kb.py's date-based retention sweep.
HEADER_DATE_UNKNOWN: Final[str] = "unknown"


class SourceKind(str, Enum):
    MARKDOWN = "markdown"
    PYTHON = "python"
    JSON = "json"
    TEXT = "text"
    # Generic tag for documents that did not come from a walked file on
    # disk -- e.g. rows exported from an external database (Cortex's
    # experiments/hypotheses tables, or any other caller's own source).
    # Deliberately not one member per external system/table: hars_memory
    # itself must stay free of any particular external system's schema or
    # vocabulary (see ingest/api.py). Per-source detail belongs in
    # Document.metadata, not in this enum.
    EXTERNAL = "external"


@dataclass(slots=True)
class Document:
    """A normalised unit of content ready for LightRAG ingestion.

    Attributes
    ----------
    doc_id:
        Stable, globally unique ID.  File-based documents use a path-hash
        prefix; Postgres rows use ``exp:<id>``, ``hyp:<id>``, etc.
    content:
        Full text content — either prose or a JSON-serialised record.
    source_kind:
        Enum tag identifying how this document was produced.
    source_path:
        Absolute path or a pseudo-path like ``postgres://experiments/42``.
    metadata:
        Arbitrary key-value pairs for provenance (mtime, row id, etc.).
    """

    doc_id: str
    content: str
    source_kind: SourceKind
    source_path: str
    metadata: dict[str, object] = field(default_factory=dict)


_INGEST_ARCHIVE_DIR_NAME: Final[str] = "ingested"


def _strip_archive_dir(resolved: Path) -> Path:
    """Drop any path component literally named ``ingested`` before hashing.

    ``scripts/update_kb.sh`` archives every successfully-indexed staging note
    by moving it from ``staging/<name>.md`` to ``staging/ingested/<name>.md``
    after a run (see that script's ``mkdir -p "$STAGING/ingested"`` /
    ``mv ... "$STAGING/ingested/"`` step). That move inserts exactly one path
    segment named ``ingested`` directly ahead of the filename, and nothing
    else about the path changes.

    Without this normalisation, ``file_stable_id`` (hashing the absolute
    path) would mint a brand-new ID for the identical file the moment it is
    archived, so the walker recursing into ``staging/ingested/`` on the next
    run (nothing currently excludes it — see ``scripts/update_kb.sh``'s
    generated ``.memoryignore``, which is the belt to this braces) would
    look like ~all-new documents to LightRAG and trigger a full re-extraction
    of content already in the index. See
    ``.session/2026-07-30_longterm-memory-overhaul.md`` for the incident this
    fixes: 248 of 249 staged files were about to be re-extracted after a
    single archival round-trip.

    Stripping the segment makes ``staging/x.md`` and ``staging/ingested/x.md``
    hash identically — which is correct, since the archived file IS the
    document that was ingested from the staging path — and reproduces every
    ID already recorded in the live ``kv_store_doc_status.json`` for files
    that predate this fix. Only an exact ``ingested`` component is dropped
    (case-sensitive, no prefix/suffix match), keeping collision risk with an
    unrelated directory that happens to be named ``ingested`` elsewhere in a
    walked root minimal and easy to reason about.
    """
    parts = tuple(part for part in resolved.parts if part != _INGEST_ARCHIVE_DIR_NAME)
    return Path(*parts) if parts else resolved


def file_stable_id(path: Path) -> str:
    """Derive a stable document ID from a file path.

    Uses a short SHA-256 prefix of the absolute path string so the ID
    survives directory renames while staying reproducible for the same file.
    A path component literally named ``ingested`` (the staging-archive
    directory created by ``scripts/update_kb.sh``) is normalised out first —
    see ``_strip_archive_dir`` — so archiving a file never changes its ID.

    >>> import re
    >>> doc_id = file_stable_id(Path("/some/path/report.md"))
    >>> bool(re.match(r"^file:[0-9a-f]{12}$", doc_id))
    True
    >>> staged = file_stable_id(Path("/staging/report.md"))
    >>> archived = file_stable_id(Path("/staging/ingested/report.md"))
    >>> staged == archived
    True
    """
    normalized = _remap_id_path(_strip_archive_dir(path.resolve()))
    digest = hashlib.sha256(str(normalized).encode()).hexdigest()[:12]
    return f"file:{digest}"


#: Optional ``ACTUAL=LOGICAL`` prefix mappings (``os.pathsep``-separated) applied
#: to a resolved path before it is hashed. Document ids are the hash of the
#: ABSOLUTE path of the machine that first indexed a file; this lets a refresh
#: run from a different checkout location (a CI clone, a git worktree, a temp
#: dir) reproduce the ids already stored in an existing index. Unset = no-op.
ID_PATH_MAP_ENV: Final[str] = "HARS_MEMORY_ID_PATH_MAP"


def _remap_id_path(path: Path) -> Path:
    raw = os.environ.get(ID_PATH_MAP_ENV, "").strip()
    if not raw:
        return path
    text = str(path)
    for entry in raw.split(os.pathsep):
        actual, sep, logical = entry.partition("=")
        actual, logical = actual.rstrip("/"), logical.rstrip("/")
        if not sep or not actual:
            raise ValueError(f"{ID_PATH_MAP_ENV} entry {entry!r} is not ACTUAL=LOGICAL")
        if text == actual or text.startswith(actual + "/"):
            return Path(logical + text[len(actual):])
    return path


def build_source_header(*, document_name: str, section: str, date: str) -> str:
    """Build a ``[Document: ... | Section: ... | Date: ...]`` attribution header.

    MUST byte-for-byte match the format already emitted by
    ``plugins/hars-longterm-memory/scripts/hars_longterm_memory_mcp.py``'s ``memory_remember``
    tool, and parsed by ``scripts/cleanup_kb.py``'s ``HEADER_RE``:
        ``[Document: <name> | Section: <section> | Date: <YYYY-MM-DD|unknown>]``

    This is what downstream retrieval uses for source attribution/recency, and
    what the KB-cleanup date sweep parses out of the first 400 chars of every
    stored document — changing the format silently breaks both.
    """
    return f"[Document: {document_name} | Section: {section} | Date: {date}]\n\n"


def infer_source_kind(path: Path) -> SourceKind:
    """Return the SourceKind for a given file extension."""
    suffix = path.suffix.lower()
    match suffix:
        case ".md":
            return SourceKind.MARKDOWN
        case ".py":
            return SourceKind.PYTHON
        case ".json":
            return SourceKind.JSON
        case ".txt":
            return SourceKind.TEXT
        case _:
            return SourceKind.TEXT
