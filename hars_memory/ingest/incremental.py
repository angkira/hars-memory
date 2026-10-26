"""Incremental index refresh: plan (read-only), apply (delete + insert), batched deletion.

Why this exists
----------------
``memory-index --refresh-changed`` already detects changed and deleted documents,
but running it against a large index is impractical:

* every run hands ALL walked documents to ``ainsert``; LightRAG answers each
  already-indexed one with a ``dup-...`` doc_status record (one write per
  document per run, forever growing);
* each ``adelete_by_doc_id`` flushes every storage twice (the 50 MB graphml and
  the JSON KV files are rewritten, ~25 s) and re-parses the cached extractions of
  every entity it touches. Hub entities ("kubectl": 361 chunks) are rebuilt once
  PER deleted document, so N deletions cost N full rebuilds (~1-2.5 min each);
* the fingerprint sidecar is saved only at the very end, so an interrupted run
  repeats work.

This module keeps the fingerprint contract of ``change_detection`` but

* classifies documents against BOTH the fingerprint sidecar and LightRAG's
  ``kv_store_doc_status.json`` (read-only) and only inserts what is new, changed
  or previously failed - an unchanged index is a strict no-op;
* deletes many documents in one pass: LightRAG's own ``adelete_by_doc_id`` does
  the per-document graph analysis, but the entity/relation rebuild and the
  storage flushes are deferred and executed once for the union of everything
  that was touched;
* checkpoints fingerprints after every verified insert batch.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from hars_memory.ingest.change_detection import FingerprintStore, compute_fingerprint
from hars_memory.ingest.document import Document

logger = logging.getLogger("memory.incremental")

DOC_STATUS_FILENAME = "kv_store_doc_status.json"
PROCESSED = "processed"

#: Same guard as ``server.index._DELETION_GC_MIN_CURRENT_RATIO``: a walk that sees
#: less than half of the fingerprinted documents is a narrowed --paths, not a
#: mass deletion.
GC_MIN_CURRENT_RATIO = 0.5


class PlanError(RuntimeError):
    """The walk looks wrong (e.g. mass deletion); refuse to produce an apply plan."""


def load_doc_status(index_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Read LightRAG's doc_status JSON (read-only; missing file = empty index)."""
    path = Path(index_dir) / DOC_STATUS_FILENAME
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def load_indexed_content(index_dir: str | Path) -> dict[str, str]:
    """Content LightRAG actually stored per doc id (kv_store_full_docs.json), read-only."""
    path = Path(index_dir) / "kv_store_full_docs.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v["content"] for k, v in raw.items() if isinstance(v, dict) and isinstance(v.get("content"), str)}


def _is_real_doc(doc_id: str, rec: dict[str, Any]) -> bool:
    """False for LightRAG's ``dup-...`` markers (re-ingest of an existing file name)."""
    return not doc_id.startswith("dup-") and not (rec.get("metadata") or {}).get("is_duplicate")


@dataclass
class RefreshPlan:
    """What a refresh run will do. All lists hold doc ids except where noted."""

    new: list[Document] = field(default_factory=list)  # never indexed
    changed: list[Document] = field(default_factory=list)  # processed, fingerprint differs
    retry: list[Document] = field(default_factory=list)  # doc_status exists but not processed
    adopt: list[Document] = field(default_factory=list)  # processed, no fingerprint: record only
    unchanged: list[str] = field(default_factory=list)
    gone: list[str] = field(default_factory=list)  # fingerprinted, no longer walked, not kept
    kept: list[str] = field(default_factory=list)  # fingerprinted, not walked, protected by keep list
    unknown_processed: list[str] = field(default_factory=list)  # processed, no fingerprint, not walked: never touched
    gc_skipped_reason: str | None = None

    @property
    def to_insert(self) -> list[Document]:
        return [*self.new, *self.changed, *self.retry]

    @property
    def to_delete(self) -> list[str]:
        return [*(d.doc_id for d in self.changed), *(d.doc_id for d in self.retry), *self.gone]

    def is_noop(self) -> bool:
        return not (self.new or self.changed or self.retry or self.gone or self.adopt)

    def summary(self) -> dict[str, Any]:
        return {
            "new": len(self.new),
            "changed": len(self.changed),
            "retry": len(self.retry),
            "adopt": len(self.adopt),
            "unchanged": len(self.unchanged),
            "gone": len(self.gone),
            "kept": len(self.kept),
            "unknown_processed": len(self.unknown_processed),
            "chars_to_insert": sum(len(d.content) for d in self.to_insert),
            "gc_skipped_reason": self.gc_skipped_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        def rows(docs: Iterable[Document]) -> list[dict[str, Any]]:
            return [
                {"doc_id": d.doc_id, "path": d.metadata.get("relative_path") or d.source_path, "chars": len(d.content)}
                for d in docs
            ]

        return {
            "summary": self.summary(),
            "new": rows(self.new),
            "changed": rows(self.changed),
            "retry": rows(self.retry),
            "adopt": rows(self.adopt),
            "gone": list(self.gone),
            "kept": list(self.kept),
            "unknown_processed": list(self.unknown_processed),
        }


def build_plan(
    docs: list[Document],
    store: FingerprintStore,
    status: dict[str, dict[str, Any]],
    *,
    keep_ids: Iterable[str] = (),
    gc: bool = True,
    indexed: dict[str, str] | None = None,
) -> RefreshPlan:
    """Classify *docs* against the fingerprint sidecar and doc_status. Pure, read-only.

    Per walked document:
      processed + same fingerprint        -> unchanged (no work, no write)
      processed + different fingerprint   -> changed   (delete, then re-insert)
      processed + no fingerprint          -> adopt     (record fingerprint only; never touched)
      status present but not processed    -> retry     (delete leftovers, re-insert)
      no status                           -> new       (insert)
    When *indexed* (doc id -> content LightRAG stored, see ``load_indexed_content``) is given it
    is the ground truth for "what is in the index": a document whose sidecar fingerprint matches
    but whose stored content differs from the walked content (a sidecar that was recorded without
    re-indexing, e.g. by an earlier --refresh-changed run that adopted unfingerprinted documents)
    is ``changed``, and an adopted document is only adopted when the stored content really matches.
    A fingerprinted id that is not walked is ``gone`` (deleted from the index) unless it
    is in *keep_ids*. GC is refused for a walk covering less than half of the
    fingerprinted ids (``PlanError``) and is skipped entirely when ``gc`` is False.
    """
    keep = frozenset(keep_ids)
    plan = RefreshPlan()
    walked: set[str] = set()
    if indexed is not None:
        from lightrag.utils import sanitize_text_for_encoding  # what ainsert stores (strip, unescape, control chars)
    for doc in docs:
        if doc.doc_id in walked:
            raise PlanError(f"two walked documents share doc_id {doc.doc_id} ({doc.source_path})")
        walked.add(doc.doc_id)
        rec = status.get(doc.doc_id)
        prev = store.get(doc.doc_id)
        if rec is None:
            plan.new.append(doc)
        elif rec.get("status") != PROCESSED:
            plan.retry.append(doc)
        else:
            stored = indexed.get(doc.doc_id) if indexed is not None else None
            stale = stored is not None and stored != sanitize_text_for_encoding(doc.content)
            if prev is None:
                (plan.changed if stale else plan.adopt).append(doc)
            elif prev != compute_fingerprint(doc.content) or stale:
                plan.changed.append(doc)
            else:
                plan.unchanged.append(doc.doc_id)

    fingerprinted = store.keys()
    missing = sorted(fingerprinted - walked)
    plan.kept = [i for i in missing if i in keep]
    candidates = [i for i in missing if i not in keep]
    if candidates and fingerprinted and len(walked) < len(fingerprinted) * GC_MIN_CURRENT_RATIO:
        plan.gc_skipped_reason = (
            f"walk covers {len(walked)} documents, below {GC_MIN_CURRENT_RATIO:.0%} of "
            f"{len(fingerprinted)} fingerprinted: --paths looks narrowed, deletion GC refused"
        )
        candidates = []
    if not gc and candidates:
        plan.gc_skipped_reason = plan.gc_skipped_reason or "gc disabled (--no-gc)"
        candidates = []
    plan.gone = candidates
    plan.unknown_processed = sorted(
        i
        for i, rec in status.items()
        if _is_real_doc(i, rec) and rec.get("status") == PROCESSED and i not in walked and i not in fingerprinted
    )
    return plan


# ---------------------------------------------------------------------------
# Batched deletion
# ---------------------------------------------------------------------------


@dataclass
class DeleteReport:
    deleted: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    entities_rebuilt: int = 0
    relations_rebuilt: int = 0
    seconds_delete: float = 0.0
    seconds_rebuild: float = 0.0


async def delete_documents(
    rag: Any,
    doc_ids: list[str],
    *,
    progress: Callable[[str], None] | None = None,
) -> DeleteReport:
    """Delete *doc_ids* from the index in one pass.

    LightRAG's ``adelete_by_doc_id`` stays the source of truth for what a deletion
    means (chunks, vectors, graph contributions, bookkeeping). Two costs that scale
    with the NUMBER of deletions are removed by deferring them:

    * ``rebuild_knowledge_from_chunks`` (re-derives entity/relation descriptions of
      everything that still has other sources) is collected and run once at the end
      for the union, re-reading the CURRENT chunk list of each entity/relation. An
      entity that a later deletion removed entirely is skipped.
    * ``_insert_done`` (full storage flush, ~25 s on this index) runs once at the end.

    A crash in between leaves the stores unflushed on disk while Qdrant is already
    updated; the next run still sees the documents as processed + changed/gone and
    repeats the deletion (idempotent), see ``audit``/``repair`` for verification.
    """
    import lightrag.lightrag as lr_module
    from lightrag.utils import make_relation_chunk_key

    say = progress or (lambda m: logger.info("%s", m))
    report = DeleteReport()
    if not doc_ids:
        return report

    deferred_entities: set[str] = set()
    deferred_relations: set[tuple[str, str]] = set()
    captured: dict[str, Any] = {}
    original_rebuild = lr_module.rebuild_knowledge_from_chunks
    original_done = rag._insert_done

    async def collect(**kwargs: Any) -> None:
        deferred_entities.update(kwargs.get("entities_to_rebuild") or {})
        for pair in kwargs.get("relationships_to_rebuild") or {}:
            deferred_relations.add(tuple(pair))  # type: ignore[arg-type]
        captured.update(kwargs)

    async def skip_flush(*_a: Any, **_k: Any) -> None:
        return None

    import time

    t0 = time.monotonic()
    lr_module.rebuild_knowledge_from_chunks = collect
    rag._insert_done = skip_flush
    try:
        for n, doc_id in enumerate(doc_ids, 1):
            result = await rag.adelete_by_doc_id(doc_id)
            status = getattr(result, "status", None)
            if status == "success":
                report.deleted.append(doc_id)
            elif status == "not_found":
                report.not_found.append(doc_id)
            else:
                report.failed[doc_id] = str(getattr(result, "message", result))[:300]
            if n % 25 == 0 or n == len(doc_ids):
                say(f"deleted {n}/{len(doc_ids)} documents ({time.monotonic() - t0:.0f}s)")
    finally:
        lr_module.rebuild_knowledge_from_chunks = original_rebuild
        rag._insert_done = original_done
    report.seconds_delete = time.monotonic() - t0

    t1 = time.monotonic()
    await original_done()  # persist the deletions before the (long) rebuild

    entities_to_rebuild: dict[str, list[str]] = {}
    for name in sorted(deferred_entities):
        node = await rag.chunk_entity_relation_graph.get_node(name)
        if not node:
            continue  # removed entirely by a later deletion
        stored = await rag.entity_chunks.get_by_id(name) if rag.entity_chunks else None
        remaining = [c for c in (stored or {}).get("chunk_ids", []) if c]
        if remaining:
            entities_to_rebuild[name] = remaining
    relationships_to_rebuild: dict[tuple[str, str], list[str]] = {}
    for src, tgt in sorted(deferred_relations):
        if not await rag.chunk_entity_relation_graph.has_edge(src, tgt):
            continue
        stored = await rag.relation_chunks.get_by_id(make_relation_chunk_key(src, tgt)) if rag.relation_chunks else None
        remaining = [c for c in (stored or {}).get("chunk_ids", []) if c]
        if remaining:
            relationships_to_rebuild[(src, tgt)] = remaining

    if (entities_to_rebuild or relationships_to_rebuild) and captured:
        captured["entities_to_rebuild"] = entities_to_rebuild
        captured["relationships_to_rebuild"] = relationships_to_rebuild
        say(
            f"rebuilding {len(entities_to_rebuild)} entities and {len(relationships_to_rebuild)} relations "
            f"once for {len(report.deleted)} deleted documents"
        )
        await original_rebuild(**captured)
        await original_done()
    report.entities_rebuilt = len(entities_to_rebuild)
    report.relations_rebuilt = len(relationships_to_rebuild)
    report.seconds_rebuild = time.monotonic() - t1
    return report


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


@dataclass
class ApplyReport:
    deleted: DeleteReport = field(default_factory=DeleteReport)
    inserted_ok: list[str] = field(default_factory=list)
    insert_failed: list[str] = field(default_factory=list)
    skipped_delete_failed: list[str] = field(default_factory=list)
    adopted: int = 0
    gc_deleted: list[str] = field(default_factory=list)
    seconds_insert: float = 0.0

    @property
    def ok(self) -> bool:
        return not (self.insert_failed or self.deleted.failed or self.skipped_delete_failed)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def _unique_file_key(doc: Document) -> str:
    from hars_memory.server.index import _unique_file_key as key

    return key(doc)


async def _verify_processed(rag: Any, ids: list[str]) -> dict[str, bool]:
    rows = await rag.doc_status.get_by_ids(ids)
    out: dict[str, bool] = {}
    for doc_id, row in zip(ids, rows):
        status = (row or {}).get("status")
        out[doc_id] = (getattr(status, "value", status) == PROCESSED)
    return out


async def apply_plan(
    rag: Any,
    plan: RefreshPlan,
    store: FingerprintStore,
    *,
    batch_size: int = 25,
    progress: Callable[[str], None] | None = None,
) -> ApplyReport:
    """Execute *plan* against an initialised LightRAG instance.

    Order: (1) one batched deletion of changed + retry + gone documents, (2) insert of
    new + changed + retry documents in batches, each batch verified against doc_status
    before its fingerprints are persisted, (3) record adopted fingerprints.
    Fingerprints of documents whose deletion or insert did not complete are left
    untouched, so the next run classifies them as changed/retry again.
    """
    say = progress or (lambda m: logger.info("%s", m))
    report = ApplyReport()
    import time

    if plan.to_delete:
        say(f"deleting {len(plan.to_delete)} documents ({len(plan.changed)} changed, {len(plan.retry)} retry, {len(plan.gone)} gone)")
        report.deleted = await delete_documents(rag, plan.to_delete, progress=say)
        for doc_id in plan.gone:
            if doc_id in report.deleted.deleted or doc_id in report.deleted.not_found:
                store.discard(doc_id)
                report.gc_deleted.append(doc_id)
        store.save()

    blocked = set(report.deleted.failed)
    todo: list[Document] = []
    for doc in plan.to_insert:
        if doc.doc_id in blocked:
            report.skipped_delete_failed.append(doc.doc_id)
        else:
            todo.append(doc)

    t0 = time.monotonic()
    total = (len(todo) + batch_size - 1) // batch_size
    for n, start in enumerate(range(0, len(todo), batch_size), 1):
        batch = todo[start : start + batch_size]
        ids = [d.doc_id for d in batch]
        try:
            await rag.ainsert([d.content for d in batch], ids=ids, file_paths=[_unique_file_key(d) for d in batch])
        except Exception as exc:  # noqa: BLE001 - verified per document below
            logger.error("insert batch %d/%d raised: %s", n, total, exc)
        verified = await _verify_processed(rag, ids)
        for doc in batch:
            if verified.get(doc.doc_id):
                store.set(doc.doc_id, compute_fingerprint(doc.content))
                report.inserted_ok.append(doc.doc_id)
            else:
                report.insert_failed.append(doc.doc_id)
        store.save()  # checkpoint: a later interruption never repeats a verified batch
        say(f"inserted batch {n}/{total}: {sum(verified.values())}/{len(batch)} ok ({time.monotonic() - t0:.0f}s)")
    report.seconds_insert = time.monotonic() - t0

    if not report.insert_failed and not report.deleted.failed:
        for doc in plan.adopt:
            store.set(doc.doc_id, compute_fingerprint(doc.content))
            report.adopted += 1
        if plan.adopt:
            store.save()
    return report
