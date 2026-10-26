"""Public ingest API for hars-longterm-memory.

External callers (in-repo scripts, or another project entirely) that have
their own document source -- a database export, an API pull, anything that
isn't a plain file on disk -- construct `Document` objects themselves and
hand them to `ingest_documents()`. This is the one supported integration
point for non-file-walk ingestion; it is what keeps hars_memory itself free
of any particular external system's schema or vocabulary.

Pipeline reuse
--------------
This module does NOT duplicate the chunk -> embed -> extract -> insert
pipeline. It builds a LightRAG instance the same way the CLI entrypoint
(``hars_memory.server.index``) does, via
``hars_memory.server.lightrag_init.create_lightrag()``, and hands the
documents to that same module's ``_insert_all_batches()`` helper -- the
exact function ``server/index.py``'s ``--paths``/``--db-export`` walk-driven
CLI already uses to call ``rag.ainsert()`` in batches. ``ingest_documents()``
is a thin wrapper parameterised by an explicit ``list[Document]`` instead of
a file-walk result.

"documents_written" semantics
------------------------------
A document counts as "written" once it has been hand off to LightRAG's
``ainsert()`` pipeline without that call raising -- i.e. its content and
chunks are durably enqueued in the index. It does NOT require that
downstream entity/relation extraction (the GPU-bound LLM call) succeeded:
LightRAG marks a document ``FAILED`` (with an error message, not a raised
exception) when extraction fails, e.g. because the extractor LLM endpoint is
unreachable, and automatically retries FAILED documents the next time the
enqueue/process pipeline runs (see
``LightRAG.apipeline_process_enqueue_documents``, which pulls
``PENDING``/``PROCESSING``/``FAILED`` docs on every call). Treating "written"
as "durably enqueued for indexing (with self-healing retry)" rather than
"fully graph-extracted this call" is what lets ``ingest_documents()`` be
called safely regardless of whether the extractor LLM happens to be up right
now -- exactly the GPU-optional posture the rest of this package's ingest
path (file walk, change detection) already has. ``errors`` therefore reports
failures of the *insertion call itself* -- ``rag.initialize_storages()``
raising (e.g. a broken vector store), ``rag.adelete_by_doc_id()`` raising, or
``rag.ainsert()`` raising a programming error -- not per-document extraction
failures, which are LightRAG's own concern to retry.

Content fingerprinting DOES gate ``ainsert()`` -- but only for documents that
were previously inserted successfully
--------------------------------------------------------------------------
Unlike ``server/index.py``'s walk-driven CLI (which re-hands its *entire*
document set to ``ainsert()`` on every run, gated only by the opt-in
``--refresh-changed`` delete-before-reinsert step), ``ingest_documents()``
uses the content fingerprint (see ``change_detection.py``) to decide, per
document, whether to call ``ainsert()`` at all this call:

* Unchanged content, previously recorded fingerprint matches -- the document
  is SKIPPED, not handed to ``ainsert()``. This is required, not just an
  optimisation: the installed ``lightrag-hku`` package's default
  ``JsonDocStatusStorage.filter_keys()`` is a blunt existence check (any
  doc_id already present in doc_status counts as "seen", regardless of
  whether its status is ``PROCESSED`` or ``FAILED``), and
  ``apipeline_enqueue_documents`` writes a brand-new, non-idempotent
  ``FAILED``/``is_duplicate`` doc_status record (plus a warning log) for
  every doc_id it is asked to enqueue that is already present -- there is no
  free "already processed, no-op" path inside LightRAG itself. Resubmitting
  unchanged, already-successful documents on every call would therefore
  permanently accumulate junk duplicate-marker records in doc_status storage
  on every repeat ingest of the same corpus. Skipping them here is what
  keeps repeat calls over an unchanged corpus (the normal staging ->
  periodic-reingest cycle) side-effect-free.
* No previously recorded fingerprint (never inserted, OR a prior call never
  confirmed a durable success for it -- see below) -- the document IS handed
  to ``ainsert()``. This is what gives a document LightRAG marked ``FAILED``,
  or a document this module never got to persist a fingerprint for, a real
  retry path.
* Changed content (fingerprint differs from what's on record) -- the
  document IS handed to ``ainsert()``, and its stale content is
  force-deleted first (``adelete_by_doc_id``) so LightRAG's doc_id-based
  dedupe cannot silently keep the old content forever.

A fingerprint is only ever persisted (``store.save()``) for a doc_id once
BOTH of the following hold: the delete+insert call for its batch completed
without raising, AND that specific doc_id's ``doc_status`` -- queried
afterwards via ``rag.doc_status.get_docs_by_status(DocStatus.PROCESSED)``
(mirroring the doc-selection pattern
``LightRAG.apipeline_process_enqueue_documents`` itself uses internally) --
came back ``PROCESSED``. The insert call not raising is necessary but NOT
sufficient: per the self-healing model above, a per-document extraction
failure is caught internally and leaves that one doc_id ``FAILED`` without
raising anything, so a batch can complete "successfully" while individual
documents inside it did not. Any doc_id in the batch that is not confirmed
``PROCESSED`` has its provisional fingerprint (recorded in memory by
``detect_changed_documents()`` before the insert ran) discarded before
``store.save()``, so it reverts to "no fingerprint on record" and stays
retry-eligible on the next call -- exactly like a doc_id whose insert call
itself raised. A failed/errored call (an exception anywhere in the
try-block) leaves the on-disk store completely untouched, for the same
reason. Restoring a document to "unchanged, skip it" status therefore only
happens once THAT document, specifically, has been confirmed ``PROCESSED``
-- never merely because the batch call around it didn't raise.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path

from hars_memory.ingest.change_detection import (
    FingerprintStore,
    default_fingerprint_store_path,
    detect_changed_documents,
)
from hars_memory.ingest.document import Document
from hars_memory.server.index import _insert_all_batches

# Docs-per-ainsert() batch size. Mirrors server/index.py's own
# HARS_MEMORY_INSERT_BATCH_SIZE env var/default so both entrypoints into the
# same insertion helper behave identically unless an operator overrides it.
_INSERT_BATCH_SIZE_ENV = "HARS_MEMORY_INSERT_BATCH_SIZE"
_DEFAULT_INSERT_BATCH_SIZE = 10


@dataclass(slots=True)
class IngestResult:
    """Outcome of an `ingest_documents()` call.

    `documents_skipped` counts documents whose content fingerprint matched a
    previously *successful* insert -- these were excluded from `ainsert()`
    entirely this call (see the module docstring for why: LightRAG's own
    doc_status dedupe is not a free no-op for already-seen doc_ids).
    """

    documents_written: int
    documents_skipped: int
    errors: list[str] = field(default_factory=list)


def ingest_documents(
    docs: list[Document],
    *,
    index_dir: Path,
    vector_storage: str | None = None,
    graph_storage: str | None = None,
    batch_size: int | None = None,
    embed_batch_size: int | None = None,
) -> IngestResult:
    """Ingest pre-built `Document` objects into the index at `index_dir`.

    This is the integration point for any document source that isn't a
    plain file on disk (e.g. rows exported from an external database).
    File-based ingestion should continue to use the `memory build`/
    `memory-index` walker path instead -- this function does not walk
    anything, it only accepts documents the caller has already built.

    Uses content-fingerprint change detection (see ``change_detection.py``)
    to decide, per document, whether to call LightRAG's ``ainsert()`` at all
    this call (see this module's docstring, "Content fingerprinting DOES
    gate ``ainsert()``"): documents byte-identical to a previously
    *successful* insert are skipped; documents with no recorded fingerprint
    (never inserted, or a prior insert of them failed) or with changed
    content are handed to ``ainsert()``, with changed-content documents
    additionally deleted (by ``doc_id``) first so LightRAG's doc_id-based
    dedupe cannot silently keep stale content. ``documents_skipped`` reports
    how many incoming documents were excluded from ``ainsert()`` this call
    for exactly that reason. See this module's docstring for what "written"
    does and does not guarantee about downstream LLM extraction.

    ``vector_storage`` / ``graph_storage``:
        Optional explicit LightRAG storage backends for THIS index directory.
        ``None`` (the default) keeps the existing env-driven behaviour
        (``HARS_MEMORY_VECTOR_STORAGE`` / ``HARS_MEMORY_GRAPH_STORAGE``), so
        every pre-existing caller is unaffected. Callers that write an
        isolated sub-index (e.g. ``session_memory``'s per-session working
        directory) pass file-backed names explicitly so a Qdrant-backed main
        index cannot mix that data into its own collections.

    ``batch_size``:
        Documents handed to ``ainsert()`` per call. ``None`` (the default)
        keeps reading ``HARS_MEMORY_INSERT_BATCH_SIZE`` exactly as before.

    ``embed_batch_size``:
        CPU embedding batch size for the LightRAG instance built for this
        call. ``None`` (the default) keeps ``create_lightrag``'s own default
        (32), so pre-existing callers are unchanged. Session compaction
        indexing passes its own batch-size knob here.
    """
    return asyncio.run(
        _ingest_documents_async(
            docs,
            index_dir,
            vector_storage=vector_storage,
            graph_storage=graph_storage,
            batch_size=batch_size,
            embed_batch_size=embed_batch_size,
        )
    )


async def _ingest_documents_async(
    docs: list[Document],
    index_dir: Path,
    *,
    vector_storage: str | None = None,
    graph_storage: str | None = None,
    batch_size: int | None = None,
    embed_batch_size: int | None = None,
) -> IngestResult:
    if not docs:
        return IngestResult(documents_written=0, documents_skipped=0, errors=[])

    fingerprint_path = default_fingerprint_store_path(index_dir)
    store = FingerprintStore.load(fingerprint_path)
    report = detect_changed_documents(docs, store)

    # Pre-filter: documents whose content fingerprint matched a previously
    # *successful* insert are never handed to ainsert() at all -- see the
    # module docstring's "Content fingerprinting DOES gate ainsert()"
    # section for why this is required, not optional. Everything else
    # (no recorded fingerprint yet, or a fingerprint mismatch) is inserted.
    unchanged_doc_ids = set(report.unchanged_doc_ids)
    to_insert = [doc for doc in docs if doc.doc_id not in unchanged_doc_ids]

    from hars_memory.server.lightrag_init import create_lightrag
    from lightrag.base import DocStatus  # type: ignore[import-not-found]

    rag = create_lightrag(
        working_dir=str(index_dir),
        vector_storage=vector_storage,
        graph_storage=graph_storage,
        **({"embed_batch_size": embed_batch_size} if embed_batch_size is not None else {}),
    )
    written = 0
    errors: list[str] = []
    initialized = False
    try:
        await rag.initialize_storages()
        initialized = True
        for doc_id in report.changed_doc_ids:
            # Content changed since the last recorded fingerprint for this
            # doc_id -- delete the stale version before reinsert so
            # LightRAG's doc_id-based dedupe cannot silently keep it.
            await rag.adelete_by_doc_id(doc_id)  # type: ignore[attr-defined]
        if to_insert:
            effective_batch_size = (
                batch_size
                if batch_size is not None
                else int(
                    os.environ.get(
                        _INSERT_BATCH_SIZE_ENV, str(_DEFAULT_INSERT_BATCH_SIZE)
                    )
                )
            )
            await _insert_all_batches(rag, to_insert, effective_batch_size)

            inserted_ids = [doc.doc_id for doc in to_insert]
            # _insert_all_batches()/ainsert() not raising only means the
            # insert *call* completed -- it says nothing about whether any
            # individual document actually reached a durable success state.
            # Per LightRAG's self-healing model (see module docstring),
            # ainsert() awaits apipeline_process_enqueue_documents() to
            # completion, and that method's own doc-selection logic reads
            # status via ``self.doc_status.get_docs_by_statuses(...)`` (see
            # lightrag.py: it pulls PENDING/PROCESSING/FAILED docs to
            # (re)process on every call) -- that per-doc-status storage,
            # ``rag.doc_status`` (a ``DocStatusStorage``), is the
            # authoritative source this mirrors here. Note:
            # ``LightRAG.aget_docs_by_ids()`` is NOT used for this -- despite
            # its type hint promising ``DocProcessingStatus`` values, the
            # installed lightrag-hku's JSON backend has it wrap
            # ``doc_status.get_by_id()``, which actually returns raw
            # ``dict[str, Any]`` records (confirmed against
            # ``JsonDocStatusStorage.get_by_ids()``), while
            # ``get_docs_by_status(es)`` reliably normalises into typed
            # ``DocProcessingStatus`` objects for every backend, per the
            # ``DocStatusStorage`` ABC's own contract.
            processed_docs = await rag.doc_status.get_docs_by_status(  # type: ignore[attr-defined]
                DocStatus.PROCESSED
            )
            processed_ids = set(processed_docs)
            for doc_id in inserted_ids:
                if doc_id not in processed_ids:
                    # FAILED (or PENDING/PROCESSING left mid-flight, or
                    # missing from doc_status entirely) -- walk back the
                    # provisional fingerprint detect_changed_documents()
                    # recorded in memory for this doc_id so it is NOT
                    # persisted below. It reverts to "no fingerprint on
                    # record", which keeps it eligible for the pre-filter to
                    # include (and hence resubmit / trigger LightRAG's own
                    # FAILED-doc retry sweep) on the next call.
                    store.discard(doc_id)
        written = len(to_insert)
        # Only persist fingerprints once the delete+insert above is fully
        # done -- an exception anywhere in this block leaves the on-disk
        # store untouched, so a retried call sees the same previous/current
        # comparison it would have seen if this call had never happened.
        # This is what lets a document whose insert call itself failed (or
        # whose individual extraction LightRAG marked FAILED, per the
        # per-doc status check above) be retried on the next call (its
        # fingerprint never made it to disk, so it is classified as "no
        # recorded fingerprint" -- not "unchanged, skip it" -- next time).
        # Documents that were pre-filtered out above are re-recorded here
        # too, but with the exact same value they already had on disk, so
        # this is a no-op for them.
        store.save()
    except Exception as exc:  # noqa: BLE001 -- reported to the caller, not swallowed
        errors.append(str(exc))
    finally:
        if initialized:
            await rag.finalize_storages()
            if to_insert:
                from hars_memory.ingest.migrate import backfill_after_ingest

                backfill_after_ingest(getattr(rag, "working_dir", None), to_insert)

    return IngestResult(
        documents_written=written,
        documents_skipped=report.unchanged_count,
        errors=errors,
    )


__all__ = ["IngestResult", "ingest_documents"]
