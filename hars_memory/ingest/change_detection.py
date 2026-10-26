"""Content-fingerprint change detection for previously-ingested documents.

Why this exists
----------------
``doc_id`` is a hash of the file/row *path* only (see ``document.file_stable_id``
and ``schema.entity_types.make_stable_id``). LightRAG's own dedupe is a pure
set-diff on ``doc_id`` (``JsonDocStatusStorage.filter_keys``): a doc_id that is
already ``processed`` is never reprocessed, no matter how its content changed.
So a file edited in place — or a Postgres row whose status/config changed
after first ingest — is silently skipped forever.

We deliberately do NOT fold a content hash into ``doc_id`` itself: that would
change every existing doc_id, make all currently-indexed documents look brand
new, and trigger a full multi-day re-extraction. Instead we store a content
fingerprint *alongside* the stable doc_id and compare it on each run that
opts in via ``--refresh-changed``. When the fingerprint differs, the caller
issues a targeted ``adelete_by_doc_id(doc_id)`` followed by a normal reinsert
of that one document — not a rebuild.

Sidecar-JSON storage rationale
-------------------------------
LightRAG's ``JsonDocStatusStorage`` on-disk schema is a private implementation
detail of the library, not a documented public contract — depending on it
directly would silently break on a LightRAG upgrade. A dedicated sidecar file
keyed by doc_id needs no LightRAG API beyond the already-used
``adelete_by_doc_id``, is trivial to inspect/diff/back up independently of the
LightRAG working dir, and is exactly the failure surface we want (a missing or
corrupt sidecar degrades to "treat everything as unknown", never to
"everything looks new" or "delete everything").

Safety contract
----------------
A document with NO previously recorded fingerprint is treated as
"do not touch" — it is never reported as changed / never deleted. Its current
fingerprint is simply recorded for future comparisons. This is what makes the
feature safe to turn on for a working set (2k+ docs) that predates it: nothing
gets deleted on the first run, only new drift is caught. Whether ANY of this
runs at all is gated by the caller behind the opt-in ``--refresh-changed`` CLI
flag (default OFF) — see ``server/index.py::_apply_refresh_changed``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from hars_memory.ingest.document import Document

logger = logging.getLogger(__name__)

_FINGERPRINT_STORE_FILENAME = "doc_fingerprints.json"


class FingerprintStoreError(RuntimeError):
    """Raised when the fingerprint sidecar file exists but is unreadable/malformed."""


def compute_fingerprint(content: str) -> str:
    """Return a stable sha256 hex digest of *content*."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def default_fingerprint_store_path(working_dir: str | Path) -> Path:
    """Default sidecar location: alongside the LightRAG working dir."""
    return Path(working_dir) / _FINGERPRINT_STORE_FILENAME


@dataclass(slots=True)
class FingerprintStore:
    """Sidecar JSON mapping ``doc_id -> content fingerprint`` (sha256 hex)."""

    path: Path
    _data: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> FingerprintStore:
        """Load the store from *path*. A missing file is a valid empty store."""
        if not path.exists():
            return cls(path=path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FingerprintStoreError(
                f"Cannot read fingerprint store {path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise FingerprintStoreError(
                f"Fingerprint store {path} does not contain a JSON object"
            )
        return cls(path=path, _data={str(k): str(v) for k, v in raw.items()})

    def get(self, doc_id: str) -> str | None:
        return self._data.get(doc_id)

    def keys(self) -> frozenset[str]:
        """Every doc_id with a recorded fingerprint (i.e. previously ingested).

        This is the only record of "what we have indexed before" that does not
        depend on LightRAG's private on-disk schema, so it is what deleted-
        document detection diffs the current walk against.
        """
        return frozenset(self._data)

    def set(self, doc_id: str, fingerprint: str) -> None:
        self._data[doc_id] = fingerprint

    def discard(self, doc_id: str) -> None:
        """Remove any recorded fingerprint for *doc_id*, if present.

        Used to walk back an in-memory ``set()`` (e.g. from
        ``detect_changed_documents()``) once the caller learns the insert it
        was provisionally recorded for did not actually reach a durable
        success state -- so the doc_id reverts to "no fingerprint on
        record" (retry-eligible) rather than being persisted as current on
        the next ``save()``.
        """
        self._data.pop(doc_id, None)

    def save(self) -> None:
        """Write the store to disk atomically (write-tmp then replace)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8"
        )
        tmp_path.replace(self.path)


@dataclass(slots=True, frozen=True)
class ChangeReport:
    """Outcome of comparing a batch of documents against a ``FingerprintStore``."""

    changed_doc_ids: tuple[str, ...]
    unchanged_doc_ids: tuple[str, ...]  # byte-identical to the stored fingerprint
    unchanged_count: int
    no_fingerprint_count: int  # never deleted — fingerprint recorded going forward
    # Previously fingerprinted doc_ids absent from this walk: the source file
    # was deleted or moved out of every ingest root. Their entities, chunks and
    # vectors otherwise stay in the graph forever and keep being retrieved.
    deleted_doc_ids: tuple[str, ...] = ()


def detect_changed_documents(
    docs: list[Document],
    store: FingerprintStore,
) -> ChangeReport:
    """Compare each document's current fingerprint against *store*.

    Mutates *store* in memory (records every document's current fingerprint —
    call ``store.save()`` separately to persist). Does NOT delete or insert
    anything; that is the caller's responsibility, driven by
    ``ChangeReport.changed_doc_ids``.

    Classification per document:
      * no stored fingerprint  -> "no fingerprint on record": NEVER reported
        as changed (do not touch); fingerprint recorded for next time.
      * stored == current      -> unchanged, no-op.
      * stored != current      -> changed: caller must ``adelete_by_doc_id``
        then reinsert.

    Additionally, any doc_id that HAS a stored fingerprint but is absent from
    *docs* is reported in ``deleted_doc_ids`` and dropped from the store: its
    source file is gone, so the caller must ``adelete_by_doc_id`` it with no
    reinsert. Dropping the fingerprint (rather than keeping it) means a file
    that later comes back is treated as "no fingerprint on record" — the safe
    do-not-touch class — instead of spuriously "changed".

    Note this diff is only as sound as the walk it is handed: it must be the
    FULL document set for every ingest root, never a filtered subset, or
    every unwalked document looks deleted. That is why the caller gates it
    behind the same opt-in ``--refresh-changed`` flag.
    """
    changed: list[str] = []
    unchanged: list[str] = []
    no_fingerprint = 0
    for doc in docs:
        current = compute_fingerprint(doc.content)
        previous = store.get(doc.doc_id)
        if previous is None:
            no_fingerprint += 1
        elif previous != current:
            changed.append(doc.doc_id)
            logger.info(
                "Content change detected for doc_id=%s (%s): will delete+reinsert",
                doc.doc_id,
                doc.source_path,
            )
        else:
            unchanged.append(doc.doc_id)
        store.set(doc.doc_id, current)

    current_doc_ids = {doc.doc_id for doc in docs}
    deleted = sorted(store.keys() - current_doc_ids)
    for doc_id in deleted:
        store.discard(doc_id)
        logger.info(
            "Document gone from all ingest roots: doc_id=%s — will delete from index",
            doc_id,
        )

    return ChangeReport(
        changed_doc_ids=tuple(changed),
        unchanged_doc_ids=tuple(unchanged),
        unchanged_count=len(unchanged),
        no_fingerprint_count=no_fingerprint,
        deleted_doc_ids=tuple(deleted),
    )


__all__ = [
    "ChangeReport",
    "FingerprintStore",
    "FingerprintStoreError",
    "compute_fingerprint",
    "default_fingerprint_store_path",
    "detect_changed_documents",
]
