"""Sidecar inbox for batch-queueing MCP upserts (``HARS_MEMORY_UPSERT_BATCH_MODE=queue``).

Why this exists
----------------
``memory_upsert_document`` extracts synchronously: two LLM calls per chunk
(initial + gleaning) at full price, inside the MCP call.  Provider Batch APIs
are half price but asynchronous, and ``memory index-batch`` already knows how
to prime LightRAG's LLM cache from batch answers — but only for documents
coming from the filesystem walker.

With the opt-in queue mode the MCP tool does NOT insert: it stores the FINAL
document in this sidecar inbox (inside the index directory) and answers
``{"action": "queued_for_batch"}``.  ``memory index-batch --source
upsert-inbox`` later uses the inbox as its document source, and ``apply``
drains entries that really reached ``processed``.

Byte-for-byte prompt invariant
------------------------------
LightRAG's extraction cache key is ``md5(user_prompt + system_prompt +
history)``.  Answers primed by the batch only hit the synchronous path's cache
if the chunk TEXT is identical, which means the document stored here must be
exactly what ``memory_upsert_document`` would have inserted: the source header
(``build_source_header`` + ``apply_source_header``) is applied BEFORE the
document is queued, and the chunker configuration is pinned in
``batch_state.json`` at collect time (``ingest.batch.chunker_config``).

Safety contract
----------------
Queuing records NO fingerprint in ``FingerprintStore`` and purges nothing:
until ``apply`` really processed the document, a retry must re-queue (not
answer "unchanged") and the index must stay untouched.  ``apply`` records the
fingerprint and drains the inbox entry only for docs that ended up
``processed`` — the same "record after the fact" rule as the synchronous
upsert.  A failed extraction leaves the entry queued for the next run.

Storage follows the ``ContentDedupStore`` pattern (``ingest/change_detection.py``):
a plain JSON sidecar next to the LightRAG working dir, written atomically
(tmp + replace), trivially inspectable, and failing loud — never silently
dropping queued work — when the file is malformed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Final

from hars_memory.ingest.change_detection import FingerprintStoreError
from hars_memory.ingest.document import Document, SourceKind

_UPSERT_INBOX_FILENAME = "upsert_inbox.json"

#: Fields persisted per queued document.  ``content`` is the FINAL content
#: (source header already applied); ``fingerprint`` is the sha256 the queueing
#: upsert computed, reused by ``apply`` when it records the doc as indexed.
ENTRY_FIELDS: Final[tuple[str, ...]] = (
    "doc_id",
    "content",
    "fingerprint",
    "source_kind",
    "source_path",
    "metadata",
    "queued_at",
)


def default_upsert_inbox_path(working_dir: str | Path) -> Path:
    """Default sidecar location: alongside the LightRAG working dir (the index)."""
    return Path(working_dir) / _UPSERT_INBOX_FILENAME


class UpsertInboxStore:
    """Sidecar JSON mapping ``doc_id -> queued upsert entry``.

    Keyed by ``doc_id`` (``file_stable_id`` of the upserted path), so a repeat
    upsert of the same path overwrites its own entry — idempotent queueing.
    A missing sidecar is a valid empty inbox; a malformed one raises instead of
    silently dropping queued documents (their content exists nowhere else).
    """

    path: Path
    _data: dict[str, dict[str, Any]]

    def __init__(self, path: Path, _data: dict[str, dict[str, Any]] | None = None) -> None:
        self.path = path
        self._data = {str(k): dict(v) for k, v in dict(_data or {}).items()}

    @classmethod
    def load(cls, path: Path) -> UpsertInboxStore:
        """Load the store from *path*. A missing file is a valid empty store."""
        if not path.exists():
            return cls(path=path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FingerprintStoreError(f"Cannot read upsert inbox {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise FingerprintStoreError(f"Upsert inbox {path} does not contain a JSON object")
        return cls(path=path, _data={str(k): dict(v) for k, v in raw.items()})

    # -- inspection -------------------------------------------------------
    def get(self, doc_id: str) -> dict[str, Any] | None:
        """The queued entry for *doc_id*, if any."""
        return self._data.get(doc_id)

    def doc_ids(self) -> list[str]:
        """Every queued doc_id, sorted (deterministic document set for batch phases)."""
        return sorted(self._data)

    def entries(self) -> list[dict[str, Any]]:
        """Queued entries sorted by doc_id."""
        return [self._data[doc_id] for doc_id in self.doc_ids()]

    def __len__(self) -> int:
        return len(self._data)

    # -- mutation ---------------------------------------------------------
    def put_document(
        self,
        doc: Document,
        *,
        fingerprint: str,
        queued_at: float | None = None,
    ) -> str:
        """Queue *doc* under its ``doc_id`` (overwriting any older entry).

        *fingerprint* is the content fingerprint the caller computed for
        ``doc.content`` — stored so ``apply`` can record it in the
        ``FingerprintStore`` without recomputing (and so operators can diff
        the inbox against the index).  Returns the entry id (== ``doc_id``).
        """
        entry: dict[str, Any] = {
            "doc_id": doc.doc_id,
            "content": doc.content,
            "fingerprint": fingerprint,
            "source_kind": getattr(doc.source_kind, "value", doc.source_kind),
            "source_path": doc.source_path,
            "metadata": dict(doc.metadata or {}),
            "queued_at": queued_at if queued_at is not None else time.time(),
        }
        self._data[doc.doc_id] = entry
        return doc.doc_id

    def discard(self, doc_id: str) -> None:
        """Drop the queued entry for *doc_id* (no-op when absent)."""
        self._data.pop(doc_id, None)

    def documents(self) -> list[Document]:
        """Materialise the queued entries as :class:`Document` objects (sorted by doc_id).

        This is the document source for ``memory index-batch --source
        upsert-inbox``; the reconstructed Documents carry the same content /
        metadata the synchronous upsert would have inserted, so chunking (and
        therefore every extraction prompt) is identical.
        """
        docs: list[Document] = []
        for entry in self.entries():
            try:
                kind = SourceKind(str(entry.get("source_kind", SourceKind.EXTERNAL.value)))
            except ValueError as exc:
                raise FingerprintStoreError(
                    f"Upsert inbox {self.path} has an invalid source_kind for doc_id="
                    f"{entry.get('doc_id')!r}: {exc}"
                ) from exc
            docs.append(
                Document(
                    doc_id=str(entry["doc_id"]),
                    content=str(entry["content"]),
                    source_kind=kind,
                    source_path=str(entry.get("source_path", "")),
                    metadata=dict(entry.get("metadata") or {}),
                )
            )
        return docs

    # -- persistence ------------------------------------------------------
    def save(self) -> None:
        """Write the store to disk atomically (write-tmp then replace)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self._data, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)


__all__ = [
    "UpsertInboxStore",
    "default_upsert_inbox_path",
]
