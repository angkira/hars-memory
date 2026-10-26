"""Session compaction memory: persist agent compaction summaries as facts.

The problem
-----------
A long agent session compacts its context 10-50 times. Each compaction drops a
large block of the conversation and keeps only a short in-context summary, so
everything that was not explicitly re-stated is lost. This module writes every
compaction (summary + the facts from the dropped context) into the long-term
memory index as ordinary LightRAG documents, so the knowledge survives both the
compaction and the session itself.

Two modes, one switch
---------------------
``split`` (default; HYPOTHESIS B, Zep/Graphiti lineage, arXiv:2501.13956)
    Every session gets its own isolated LightRAG working directory
    ``<sessions_dir>/<session_id>/`` with its own GraphML graph, Nano vector
    store, document status and fingerprint sidecar. Compactions never touch the
    main index, so an in-progress session cannot pollute shared knowledge.
    ``merge_session()`` later consolidates the session into the MAIN index as
    one chronological document carrying entity-alias hints, and LightRAG's own
    extraction performs the entity resolution (GraphRAG-incremental-style
    consolidation). A ``merged.json`` marker makes the merge idempotent.

``subgraph`` (HYPOTHESIS A; Mem0/A-Mem lineage)
    Compaction documents are ingested straight into the MAIN index and LightRAG
    extraction links their entities to the existing graph nodes. Session
    identity lives in the document header and ``file_path``. There is nothing
    to merge, so ``merge_session()`` is a no-op.

``HARS_MEMORY_SESSION_MODE=split|subgraph`` selects the mode.

Provenance
----------
Every compaction document starts with a
``[Document: session-<sid>-c<seq> | Section: session | Date: YYYY-MM-DD]``
header built by :func:`hars_memory.ingest.document.build_source_header`. It
keeps exactly three pipe fields so it stays byte-compatible with
``scripts/cleanup_kb.py``'s ``HEADER_RE``; ``Section: session`` is already a
recognised section for ``memory_forget``/KB cleanup.

Sequence sidecar
----------------
``<sessions_dir>/<sid>/seq.json`` records the last allocated sequence number.
It is written atomically (tmp + rename) *before* the ingest, and restored on
ingest failure, so a failed compaction never burns a sequence number.

Configuration
-------------
- ``HARS_MEMORY_SESSION_MODE``: ``split`` (default) or ``subgraph``.
- ``HARS_MEMORY_SESSIONS_DIR``: default ``<HARS_MEMORY_INDEX_DIR>/sessions``.
- ``HARS_MEMORY_SESSION_INDEX_BS``: compaction indexing batch size, default 25,
  clamped to 1..256.
"""

from __future__ import annotations

import datetime as dt
import difflib
import json
import logging
import os
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hars_memory.ingest.api import ingest_documents
from hars_memory.ingest.document import (
    Document,
    SourceKind,
    build_source_header,
)

logger = logging.getLogger(__name__)

MODE_ENV = "HARS_MEMORY_SESSION_MODE"
SESSIONS_DIR_ENV = "HARS_MEMORY_SESSIONS_DIR"
INDEX_BS_ENV = "HARS_MEMORY_SESSION_INDEX_BS"
MIN_SCORE_ENV = "HARS_MEMORY_SESSION_MIN_SCORE"

MODE_SPLIT = "split"
MODE_SUBGRAPH = "subgraph"
VALID_MODES = (MODE_SPLIT, MODE_SUBGRAPH)
DEFAULT_MODE = MODE_SPLIT

DEFAULT_INDEX_BS = 25
MIN_INDEX_BS = 1
MAX_INDEX_BS = 256

DEFAULT_MIN_SCORE = 0.2

#: LightRAG's GraphML file name inside a working directory.
GRAPH_FILE_NAME = "graph_chunk_entity_relation.graphml"
#: LightRAG's full-document KV store (``{doc_id: {"content": str}}``).
FULL_DOCS_FILENAME = "kv_store_full_docs.json"
SEQ_FILENAME = "seq.json"
MERGED_FILENAME = "merged.json"

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_COMPACTION_DOC_RE = re.compile(r"^session:(?P<sid>[^:]+):c(?P<seq>\d+)$")
_COMPACTION_SEQ_RE = re.compile(r"^Seq:\s*(?P<seq>\d+)\s*$", re.MULTILINE)
_HEADER_LINE_RE = re.compile(r"^\[Document:[^\]]*\]\s*$")
_ENTITY_SEPARATOR_RE = re.compile(r"[_\-\s]+")

#: Storage backends a session directory MUST use regardless of the process
#: environment: a Qdrant-backed main index would otherwise put an isolated
#: session's vectors into the same collections as the main graph.
_SESSION_VECTOR_STORAGE = "NanoVectorDBStorage"
_SESSION_GRAPH_STORAGE = "NetworkXStorage"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def resolve_mode(mode: str | None = None) -> str:
    """Return the effective session mode (explicit arg > env > ``split``)."""
    raw = mode if mode is not None else os.environ.get(MODE_ENV, DEFAULT_MODE)
    resolved = str(raw or DEFAULT_MODE).strip().lower()
    if resolved not in VALID_MODES:
        raise ValueError(
            f"{MODE_ENV} must be one of {list(VALID_MODES)}, got {resolved!r}"
        )
    return resolved


def clamp_index_bs(value: int | str) -> int:
    """Clamp a batch size to the supported ``1..256`` range."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{INDEX_BS_ENV} must be an integer, got {value!r}") from exc
    return max(MIN_INDEX_BS, min(MAX_INDEX_BS, number))


def resolve_index_bs(index_bs: int | None = None) -> int:
    """Return the effective compaction indexing batch size.

    Precedence: explicit argument > ``HARS_MEMORY_SESSION_INDEX_BS`` > 25. A
    malformed env value falls back to the default (logged, never fatal) so a
    typo cannot stop the MCP server from starting.
    """
    if index_bs is not None:
        return clamp_index_bs(index_bs)
    raw = os.environ.get(INDEX_BS_ENV, "").strip()
    if not raw:
        return DEFAULT_INDEX_BS
    try:
        return clamp_index_bs(raw)
    except ValueError:
        logger.warning("Ignoring malformed %s=%r; using %d", INDEX_BS_ENV, raw, DEFAULT_INDEX_BS)
        return DEFAULT_INDEX_BS


def resolve_min_score(value: float | str | None = None) -> float:
    """Return the recall session-channel score cutoff (default 0.2)."""
    raw: Any = value if value is not None else os.environ.get(MIN_SCORE_ENV, "")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_MIN_SCORE
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring malformed %s=%r; using %.2f", MIN_SCORE_ENV, raw, DEFAULT_MIN_SCORE)
        return DEFAULT_MIN_SCORE


def validate_session_id(session_id: object) -> str:
    """Validate a session id against the traversal-safe alphabet.

    ``^[A-Za-z0-9._-]{1,128}$`` rejects path separators, NUL bytes, unicode and
    over-long ids. ``.`` and ``..`` also match that regex but would still let a
    caller escape ``sessions_dir``, so they are rejected explicitly.
    """
    if not isinstance(session_id, str):
        raise ValueError("session_id must be a string")
    sid = session_id.strip()
    if not _SESSION_ID_RE.match(sid) or sid in {".", ".."}:
        raise ValueError(
            "invalid session_id: expected 1-128 characters from [A-Za-z0-9._-] "
            f"(got {session_id!r})"
        )
    return sid


# ---------------------------------------------------------------------------
# Small file helpers (all fail-soft on read, atomic on write)
# ---------------------------------------------------------------------------


def _today_iso() -> str:
    return dt.date.today().isoformat()


def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object; missing/corrupt/non-object files read as ``{}``."""
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug("Session memory: cannot read %s: %s", path, exc)
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write *payload* as JSON atomically (tmp sibling + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _normalize_dropped_context(value: object) -> list[str]:
    """Flatten ``dropped_context`` into one clean bullet per fact.

    A single string is split on newlines (each non-empty line is one fact); a
    sequence contributes one fact per item. Every fact's internal newlines are
    collapsed into spaces, blank items are dropped.
    """
    if value is None:
        return []
    if isinstance(value, str):
        candidates: list[object] = list(value.splitlines())
    elif isinstance(value, (list, tuple)):
        candidates = list(value)
    else:
        candidates = [value]
    facts: list[str] = []
    for candidate in candidates:
        text = " ".join(str(candidate).split())
        if text:
            facts.append(text)
    return facts


def _normalize_entity_text(text: str) -> str:
    """Case-fold and collapse ``_``/``-``/whitespace, mirroring the MCP search."""
    return _ENTITY_SEPARATOR_RE.sub(" ", text).strip().casefold()


class SessionMemoryStore:
    """Store and consolidate per-session compaction documents.

    Parameters
    ----------
    index_dir:
        The MAIN LightRAG working directory. In ``subgraph`` mode compaction
        documents are ingested here directly; in ``split`` mode this is where
        the ``sessions/`` (or ``HARS_MEMORY_SESSIONS_DIR``) subtree lives and
        where :meth:`merge_session` writes the merged document.
    sessions_dir:
        Session working directories. Defaults to ``HARS_MEMORY_SESSIONS_DIR``
        or ``<index_dir>/sessions``.
    mode:
        ``split`` (default) or ``subgraph``. Defaults to
        ``HARS_MEMORY_SESSION_MODE``.
    index_bs:
        Documents per ingest batch. Defaults to
        ``HARS_MEMORY_SESSION_INDEX_BS`` (25), clamped to 1..256.
    """

    def __init__(
        self,
        index_dir: str | Path,
        sessions_dir: str | Path | None = None,
        mode: str | None = None,
        index_bs: int | None = None,
    ) -> None:
        self.index_dir = Path(index_dir).expanduser()
        configured_sessions_dir = sessions_dir
        if configured_sessions_dir is None:
            env_sessions_dir = os.environ.get(SESSIONS_DIR_ENV, "").strip()
            configured_sessions_dir = env_sessions_dir or (self.index_dir / "sessions")
        self.sessions_dir = Path(configured_sessions_dir).expanduser()
        self.mode = resolve_mode(mode)
        self.index_bs = resolve_index_bs(index_bs)

    # -- paths ------------------------------------------------------------

    def session_dir(self, session_id: str) -> Path:
        """Return the working directory for *session_id* (validated)."""
        return self.sessions_dir / validate_session_id(session_id)

    def _target_index_dir(self, session_id: str) -> Path:
        if self.mode == MODE_SUBGRAPH:
            return self.index_dir
        return self.session_dir(session_id)

    def _storage_kwargs(self) -> dict[str, str | None]:
        """Storage overrides for an ingest call.

        Split mode MUST pin file-backed storages: inheriting a Qdrant
        environment would write session vectors into the main collections.
        Subgraph mode passes ``None`` so the ambient env-driven behaviour is
        unchanged.
        """
        if self.mode == MODE_SUBGRAPH:
            return {"vector_storage": None, "graph_storage": None}
        return {
            "vector_storage": _SESSION_VECTOR_STORAGE,
            "graph_storage": _SESSION_GRAPH_STORAGE,
        }

    # -- compaction documents --------------------------------------------

    def _build_compaction_document(
        self,
        session_id: str,
        seq: int,
        summary: str,
        dropped_context: list[str] | str | None,
        metadata: dict[str, Any] | None,
    ) -> Document:
        """Build the LightRAG document for one compaction event.

        The first line is exactly three pipe fields
        (``[Document: ... | Section: session | Date: ...]``) so
        ``cleanup_kb.HEADER_RE`` keeps parsing it; anything else the caller
        wants to keep goes in the body (including the optional single-line
        ``Metadata: <compact json>``).
        """
        name = f"session-{session_id}-c{seq}"
        header = build_source_header(
            document_name=name, section="session", date=_today_iso()
        )
        lines = [
            f"Session: {session_id}",
            f"Seq: {seq}",
            "Tags: session, compaction",
        ]
        if metadata:
            lines.append(
                "Metadata: "
                + json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            )
        lines += ["", "## Summary", summary.strip()]
        facts = _normalize_dropped_context(dropped_context)
        if facts:
            lines += ["", "## Dropped context facts", *(f"- {fact}" for fact in facts)]
        content = header + "\n".join(lines).rstrip() + "\n"
        return Document(
            doc_id=f"session:{session_id}:c{seq}",
            content=content,
            source_kind=SourceKind.MARKDOWN,
            source_path=f"sessions/{session_id}/compaction-{seq}.md",
            metadata={
                "session_id": session_id,
                "seq": seq,
                "kind": "session_compaction",
            },
        )

    def _max_seen_seq(self, session_id: str) -> int:
        """Highest sequence number already recorded for *session_id*.

        Reads the ``seq.json`` sidecar AND the session's document store, so a
        lost sidecar cannot make a new compaction reuse an existing seq (which
        would silently overwrite a document instead of adding one).
        """
        session_dir = self.session_dir(session_id)
        seen = 0
        seq_payload = _read_json(session_dir / SEQ_FILENAME)
        try:
            seen = max(seen, int(seq_payload.get("seq", 0) or 0))
        except (TypeError, ValueError):
            pass
        full_docs = _read_json(session_dir / FULL_DOCS_FILENAME)
        for doc_id, record in full_docs.items():
            match = _COMPACTION_DOC_RE.match(str(doc_id))
            if match and match.group("sid") == session_id:
                seen = max(seen, int(match.group("seq")))
                continue
            if isinstance(record, dict):
                content = record.get("content")
                if isinstance(content, str):
                    seq_match = _COMPACTION_SEQ_RE.search(content)
                    if seq_match and f"Session: {session_id}" in content:
                        seen = max(seen, int(seq_match.group("seq")))
        return seen

    def store_compaction(
        self,
        session_id: str,
        summary: str,
        dropped_context: list[str] | str | None = None,
        metadata: dict[str, Any] | None = None,
        seq: int | None = None,
    ) -> dict[str, Any]:
        """Persist one compaction (summary + dropped-context facts).

        ``seq=None`` allocates ``1 + max seen``. The ``seq.json`` sidecar is
        advanced atomically before the ingest and restored verbatim if the
        ingest reports errors, so a failed compaction is retryable with the
        same sequence number.
        """
        sid = validate_session_id(session_id)
        summary_text = str(summary or "").strip()
        if not summary_text:
            raise ValueError("summary is required")
        if seq is not None:
            if isinstance(seq, bool) or not isinstance(seq, int):
                raise ValueError(f"seq must be an integer, got {seq!r}")
            if seq < 1:
                raise ValueError(f"seq must be >= 1, got {seq!r}")
            resolved_seq = seq
        else:
            resolved_seq = self._max_seen_seq(sid) + 1
        meta = dict(metadata) if isinstance(metadata, Mapping) else None

        session_dir = self.session_dir(sid)
        session_dir.mkdir(parents=True, exist_ok=True)
        seq_path = session_dir / SEQ_FILENAME
        previous_seq_json = seq_path.read_text(encoding="utf-8") if seq_path.is_file() else None

        document = self._build_compaction_document(
            sid, resolved_seq, summary_text, dropped_context, meta
        )
        target_dir = self._target_index_dir(sid)
        try:
            _write_json_atomic(
                seq_path,
                {"session_id": sid, "seq": resolved_seq, "updated_at": _now_iso()},
            )
            result = ingest_documents(
                [document],
                index_dir=target_dir,
                batch_size=self.index_bs,
                embed_batch_size=self.index_bs,
                **self._storage_kwargs(),
            )
            if result.errors:
                raise RuntimeError("; ".join(result.errors))
        except Exception:
            # Roll the sidecar back so the retry allocates the same seq again.
            if previous_seq_json is None:
                seq_path.unlink(missing_ok=True)
            else:
                seq_path.write_text(previous_seq_json, encoding="utf-8")
            raise

        logger.info(
            "Stored session compaction %s seq=%d (%d dropped facts) in %s mode at %s",
            sid,
            resolved_seq,
            len(_normalize_dropped_context(dropped_context)),
            self.mode,
            target_dir,
        )
        return {
            "ok": True,
            "session_id": sid,
            "seq": resolved_seq,
            "doc_id": document.doc_id,
            "mode": self.mode,
            "index_dir": str(target_dir),
        }

    # -- merge ------------------------------------------------------------

    def _load_session_compactions(self, session_id: str) -> list[tuple[int, str]]:
        """Return ``[(seq, body_without_header)]`` ordered chronologically."""
        full_docs = _read_json(self.session_dir(session_id) / FULL_DOCS_FILENAME)
        by_seq: dict[int, str] = {}
        for doc_id, record in full_docs.items():
            if not isinstance(record, dict):
                continue
            content = record.get("content")
            if not isinstance(content, str):
                continue
            seq: int | None = None
            match = _COMPACTION_DOC_RE.match(str(doc_id))
            if match and match.group("sid") == session_id:
                seq = int(match.group("seq"))
            else:
                seq_match = _COMPACTION_SEQ_RE.search(content)
                if seq_match and f"Session: {session_id}" in content:
                    seq = int(seq_match.group("seq"))
            if seq is None:
                continue
            by_seq.setdefault(seq, self._strip_document_header(content))
        return sorted(by_seq.items(), key=lambda item: item[0])

    @staticmethod
    def _strip_document_header(content: str) -> str:
        """Drop the ``[Document: ...]`` header line, keep the body."""
        lines = content.splitlines()
        if lines and _HEADER_LINE_RE.match(lines[0].strip()):
            lines = lines[1:]
        body = "\n".join(lines).strip()
        return body

    def _graph_entity_names(self, working_dir: Path) -> list[str]:
        """Node ids from a LightRAG GraphML file; ``[]`` on any failure."""
        graph_file = working_dir / GRAPH_FILE_NAME
        if not graph_file.is_file():
            return []
        try:
            import networkx as nx

            graph = nx.read_graphml(str(graph_file))
        except Exception as exc:  # noqa: BLE001 -- stats/alias hints are fail-soft
            logger.debug("Session memory: cannot read graph %s: %s", graph_file, exc)
            return []
        return [str(node) for node in graph.nodes]

    def _graph_counts(self, working_dir: Path) -> tuple[int, int]:
        """``(nodes, edges)`` for a GraphML file; ``(0, 0)`` on any failure."""
        graph_file = working_dir / GRAPH_FILE_NAME
        if not graph_file.is_file():
            return 0, 0
        try:
            import networkx as nx

            graph = nx.read_graphml(str(graph_file))
            return int(graph.number_of_nodes()), int(graph.number_of_edges())
        except Exception as exc:  # noqa: BLE001 -- status must never fail
            logger.debug("Session memory: cannot count graph %s: %s", graph_file, exc)
            return 0, 0

    def _alias_hints(self, session_id: str) -> list[str]:
        """Map session-graph entities onto main-graph entities.

        Exact normalized-name matches first, then ``difflib`` near-duplicates
        (cutoff 0.92). The hint list is fed to the merge document so LightRAG's
        own extraction resolves aliases instead of duplicating entities.
        """
        main_names = self._graph_entity_names(self.index_dir)
        session_names = self._graph_entity_names(self.session_dir(session_id))
        if not main_names or not session_names:
            return []
        main_by_norm: dict[str, str] = {}
        for name in main_names:
            main_by_norm.setdefault(_normalize_entity_text(name), name)
        hints: list[str] = []
        seen: set[str] = set()
        for name in session_names:
            target = main_by_norm.get(_normalize_entity_text(name))
            if target is None:
                close = difflib.get_close_matches(name, main_names, n=1, cutoff=0.92)
                target = close[0] if close else None
            if target is None or target == name:
                continue
            hint = f"{name} -> {target}"
            if hint not in seen:
                seen.add(hint)
                hints.append(hint)
        return hints

    def _build_merged_document(
        self, session_id: str, compactions: list[tuple[int, str]], alias_hints: list[str]
    ) -> Document:
        """Build one chronological merged document for the main index."""
        parts = [
            build_source_header(
                document_name=f"session-{session_id}-merged",
                section="session",
                date=_today_iso(),
            )
        ]
        for seq, body in compactions:
            parts.append(f"## Compaction {seq}\n\n{body}\n")
        if alias_hints:
            parts.append("## Entity aliases\n\n" + "\n".join(alias_hints) + "\n")
        content = "\n".join(parts)
        return Document(
            doc_id=f"session:{session_id}:merged",
            content=content,
            source_kind=SourceKind.MARKDOWN,
            source_path=f"sessions/{session_id}/merged.md",
            metadata={
                "session_id": session_id,
                "kind": "session_merged",
                "compactions": len(compactions),
                "alias_hints": len(alias_hints),
            },
        )

    def merge_session(self, session_id: str, delete_source: bool = False) -> dict[str, Any]:
        """Consolidate a split-mode session into the MAIN index.

        Idempotent via ``merged.json``: a second call (with the session
        directory still present) returns the recorded payload plus
        ``already_merged: true`` and does NOT re-ingest.
        """
        sid = validate_session_id(session_id)
        if self.mode == MODE_SUBGRAPH:
            return {
                "ok": True,
                "session_id": sid,
                "merged": False,
                "reason": "subgraph mode indexes compactions directly into the main graph",
            }
        session_dir = self.session_dir(sid)
        if not session_dir.is_dir():
            raise ValueError(f"unknown session: {sid}")
        marker = session_dir / MERGED_FILENAME
        if marker.is_file():
            payload = _read_json(marker)
            return {"ok": True, "session_id": sid, **payload, "already_merged": True}

        compactions = self._load_session_compactions(sid)
        if not compactions:
            raise ValueError(f"session {sid} has no compaction documents to merge")
        alias_hints = self._alias_hints(sid)
        document = self._build_merged_document(sid, compactions, alias_hints)

        result = ingest_documents(
            [document],
            index_dir=self.index_dir,
            batch_size=self.index_bs,
            embed_batch_size=self.index_bs,
            vector_storage=None,
            graph_storage=None,
        )
        if result.errors:
            raise RuntimeError("; ".join(result.errors))

        payload: dict[str, Any] = {
            "merged_at": _now_iso(),
            "compactions": len(compactions),
            "chars": len(document.content),
            "alias_hints": alias_hints,
            "merged_doc_id": document.doc_id,
            "delete_source": bool(delete_source),
        }
        _write_json_atomic(marker, payload)
        self._invalidate_caches()
        logger.info(
            "Merged session %s into %s (%d compactions, %d chars, %d alias hints)",
            sid,
            self.index_dir,
            len(compactions),
            len(document.content),
            len(alias_hints),
        )
        if delete_source:
            shutil.rmtree(session_dir, ignore_errors=True)
        return {"ok": True, "session_id": sid, "merged": True, **payload}

    def _invalidate_caches(self) -> None:
        """Drop the MCP server's in-memory graph/BM25 caches after a merge.

        Uses the live module from ``sys.modules`` first (the normal MCP path)
        and falls back to a lazy import, which keeps this module free of a
        hard import cycle with ``hars_memory.mcp_server``. Any failure (e.g. a
        standalone script with no MCP env) is logged and ignored: cache
        invalidation is an optimisation, not part of the merge contract.
        """
        module = sys.modules.get("hars_memory.mcp_server")
        try:
            if module is None:
                import hars_memory.mcp_server as module  # noqa: F811
            module._clear_in_memory_graph_cache(None)
            module._invalidate_bm25_cache(None)
        except Exception as exc:  # noqa: BLE001 -- best effort only
            logger.debug("Session memory: cache invalidation skipped: %s", exc)

    # -- status -----------------------------------------------------------

    def _session_detail(self, session_id: str) -> dict[str, Any]:
        sid = validate_session_id(session_id)
        session_dir = self.session_dir(sid)
        detail: dict[str, Any] = {
            "session_id": sid,
            "mode": self.mode,
            "exists": session_dir.is_dir(),
            "index_dir": str(self._target_index_dir(sid)),
            "compactions": 0,
            "merged": False,
            "merged_at": None,
            "entities": 0,
            "relations": 0,
        }
        if not session_dir.is_dir():
            return detail
        full_docs = _read_json(session_dir / FULL_DOCS_FILENAME)
        compactions = 0
        for doc_id, record in full_docs.items():
            match = _COMPACTION_DOC_RE.match(str(doc_id))
            if match and match.group("sid") == sid:
                compactions += 1
                continue
            if isinstance(record, dict):
                content = record.get("content")
                if (
                    isinstance(content, str)
                    and "Tags: session, compaction" in content
                    and f"Session: {sid}" in content
                ):
                    compactions += 1
        detail["compactions"] = compactions
        if self.mode == MODE_SUBGRAPH:
            # In subgraph mode the documents live in the MAIN index, so the
            # session directory only holds the sequence sidecar. Report its
            # high-water mark (the allocated sequence) instead of zero.
            seq_payload = _read_json(session_dir / SEQ_FILENAME)
            try:
                detail["compactions"] = max(
                    compactions, int(seq_payload.get("seq", 0) or 0)
                )
            except (TypeError, ValueError):
                pass
        merged = _read_json(session_dir / MERGED_FILENAME)
        if merged:
            detail["merged"] = True
            detail["merged_at"] = merged.get("merged_at")
        entities, relations = self._graph_counts(session_dir)
        detail["entities"] = entities
        detail["relations"] = relations
        return detail

    def session_status(self, session_id: str | None = None) -> dict[str, Any]:
        """Session inventory (no arg) or one session's detail (with arg).

        Every statistic is fail-soft: corrupt JSON or a missing GraphML file
        reads as zeros rather than raising.
        """
        if session_id is not None:
            return self._session_detail(session_id)
        sessions: list[dict[str, Any]] = []
        if self.sessions_dir.is_dir():
            for entry in sorted(self.sessions_dir.iterdir(), key=lambda item: item.name):
                if not entry.is_dir():
                    continue
                try:
                    sessions.append(self._session_detail(entry.name))
                except ValueError:
                    logger.debug("Session memory: skipping invalid session dir %s", entry)
        return {
            "mode": self.mode,
            "sessions_dir": str(self.sessions_dir),
            "sessions": sessions,
        }


__all__ = [
    "DEFAULT_INDEX_BS",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_MODE",
    "INDEX_BS_ENV",
    "MAX_INDEX_BS",
    "MIN_INDEX_BS",
    "MIN_SCORE_ENV",
    "MODE_ENV",
    "MODE_SPLIT",
    "MODE_SUBGRAPH",
    "SESSIONS_DIR_ENV",
    "SessionMemoryStore",
    "VALID_MODES",
    "clamp_index_bs",
    "resolve_index_bs",
    "resolve_min_score",
    "resolve_mode",
    "validate_session_id",
]
