from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from lightrag.base import DocProcessingStatus, DocStatus

from hars_memory.ingest import api as api_module
from hars_memory.ingest.api import IngestResult, ingest_documents
from hars_memory.ingest.document import Document, SourceKind


def test_ingest_documents_returns_result_with_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = [
        Document(
            doc_id="ext:1",
            content="Experiment E1 confirmed hypothesis H3.",
            source_kind=SourceKind.EXTERNAL,
            source_path="postgres://experiments/1",
            metadata={"origin_table": "experiments", "row_id": "1"},
        ),
    ]
    fake_rag = _make_fake_rag()
    monkeypatch.setattr(
        "hars_memory.server.lightrag_init.create_lightrag", lambda **kwargs: fake_rag
    )

    async def fake_insert(rag, inserted_docs, batch_size):
        rag._known_doc_ids.update(doc.doc_id for doc in inserted_docs)

    monkeypatch.setattr(api_module, "_insert_all_batches", fake_insert)

    result = ingest_documents(docs, index_dir=tmp_path)
    assert isinstance(result, IngestResult)
    assert result.documents_written == 1
    assert result.documents_skipped == 0
    assert result.errors == []


def _make_fake_rag(doc_statuses: dict[str, DocStatus] | None = None) -> AsyncMock:
    """Build a fake LightRAG instance, mirroring the real
    `rag.doc_status.get_docs_by_status(DocStatus)` API `_ingest_documents_async`
    now queries after an insert (see `hars_memory.ingest.api`).

    Callers' `_insert_all_batches` replacements must record which doc_ids
    they "inserted" via `rag._known_doc_ids.update(doc_id, ...)` -- the fake
    `get_docs_by_status` only knows about doc_ids explicitly registered this
    way, exactly like a real doc_status store only knows about doc_ids it
    has actually seen.

    `doc_statuses` overrides the per-doc_id status a *registered* doc_id is
    reported as; any registered doc_id not present there defaults to
    `DocStatus.PROCESSED` (i.e. "insert succeeded for real"), which is what
    every pre-existing invariant test in this module expects.
    """
    overrides = doc_statuses or {}
    known_doc_ids: set[str] = set()

    async def fake_get_docs_by_status(
        status: DocStatus,
    ) -> dict[str, DocProcessingStatus]:
        result: dict[str, DocProcessingStatus] = {}
        for doc_id in known_doc_ids:
            effective_status = overrides.get(doc_id, DocStatus.PROCESSED)
            if effective_status != status:
                continue
            result[doc_id] = DocProcessingStatus(
                content_summary="stub",
                content_length=0,
                file_path="stub",
                status=effective_status,
                created_at="2026-01-01T00:00:00",
                updated_at="2026-01-01T00:00:00",
            )
        return result

    rag = AsyncMock()
    rag.initialize_storages = AsyncMock(return_value=None)
    rag.finalize_storages = AsyncMock(return_value=None)
    rag.adelete_by_doc_id = AsyncMock(return_value=None)
    rag.doc_status = AsyncMock()
    rag.doc_status.get_docs_by_status = AsyncMock(side_effect=fake_get_docs_by_status)
    rag._known_doc_ids = known_doc_ids
    return rag


def test_failed_insert_does_not_permanently_skip_unchanged_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A doc must be retried on the next call even if its content is
    byte-identical to a previous call that errored out.

    This is the specific failure mode of the Critical review finding on
    commit 53067f1d: persisting the fingerprint unconditionally (before/
    regardless of insert success) meant a document that failed to insert
    once would be silently skipped forever on every subsequent call with
    unchanged content -- defeating LightRAG's own FAILED-doc retry story
    that this module's docstring relies on.

    This covers the *call-level* failure mode: _insert_all_batches() itself
    raises. See
    test_per_doc_failed_status_inside_non_raising_insert_is_not_persisted
    below for the distinct, more subtle case fixed in round 3: the insert
    call returns normally but LightRAG marked one of its documents FAILED
    internally.
    """
    doc = Document(
        doc_id="ext:retry-1",
        content="Experiment E9 confirmed hypothesis H1.",
        source_kind=SourceKind.EXTERNAL,
        source_path="postgres://experiments/9",
        metadata={},
    )

    fake_rag = _make_fake_rag()
    monkeypatch.setattr(
        "hars_memory.server.lightrag_init.create_lightrag", lambda **kwargs: fake_rag
    )

    call_doc_id_batches: list[list[str]] = []

    async def failing_insert(rag, docs, batch_size):
        call_doc_id_batches.append([d.doc_id for d in docs])
        raise RuntimeError("simulated extractor-down insert failure")

    monkeypatch.setattr(api_module, "_insert_all_batches", failing_insert)

    first = ingest_documents([doc], index_dir=tmp_path)
    assert first.documents_written == 0
    assert first.documents_skipped == 0
    assert first.errors  # the failure is reported, not swallowed
    assert call_doc_id_batches == [["ext:retry-1"]]

    async def succeeding_insert(rag, docs, batch_size):
        call_doc_id_batches.append([d.doc_id for d in docs])
        rag._known_doc_ids.update(d.doc_id for d in docs)

    monkeypatch.setattr(api_module, "_insert_all_batches", succeeding_insert)

    second = ingest_documents([doc], index_dir=tmp_path)
    assert second.errors == []
    # The critical assertion: even though content is byte-identical to the
    # first (failed) call, the doc must be handed to _insert_all_batches
    # again -- NOT silently skipped because a fingerprint match was recorded
    # despite the earlier failure.
    assert call_doc_id_batches == [["ext:retry-1"], ["ext:retry-1"]]
    assert second.documents_written == 1


def test_per_doc_failed_status_inside_non_raising_insert_is_not_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The insert *call* not raising must not be conflated with "this
    document was actually processed successfully".

    Per LightRAG's documented self-healing model (see this module's
    docstring): when the extractor LLM is unreachable, ainsert() does NOT
    raise -- LightRAG catches the per-document extraction failure
    internally, marks that individual document's doc_status FAILED, and
    returns normally. This test mocks exactly that: _insert_all_batches()
    returns normally (no exception), but the doc's simulated doc_status
    comes back FAILED. The fingerprint for that doc_id must NOT be
    persisted -- it must remain eligible for resubmission (and hence
    LightRAG's own internal FAILED-doc retry sweep) on the next call with
    unchanged content, exactly like a call-level failure.
    """
    doc = Document(
        doc_id="ext:extraction-failed-1",
        content="Experiment E13 confirmed hypothesis H6.",
        source_kind=SourceKind.EXTERNAL,
        source_path="postgres://experiments/13",
        metadata={},
    )

    fake_rag = _make_fake_rag(
        doc_statuses={"ext:extraction-failed-1": DocStatus.FAILED}
    )
    monkeypatch.setattr(
        "hars_memory.server.lightrag_init.create_lightrag", lambda **kwargs: fake_rag
    )

    call_doc_id_batches: list[list[str]] = []

    async def non_raising_insert(rag, docs, batch_size):
        # Simulates LightRAG's self-healing behaviour: the call itself
        # completes normally even though the document's own extraction
        # failed internally. Registering the doc_id here mimics LightRAG
        # actually having written a (FAILED) doc_status record for it --
        # `rag.doc_status.get_docs_by_status` is queried separately below.
        call_doc_id_batches.append([d.doc_id for d in docs])
        rag._known_doc_ids.update(d.doc_id for d in docs)

    monkeypatch.setattr(api_module, "_insert_all_batches", non_raising_insert)

    first = ingest_documents([doc], index_dir=tmp_path)
    # The call-level contract is unchanged: "written" only means "handed to
    # ainsert() without that call raising", per the module docstring -- it
    # does NOT require extraction to have succeeded.
    assert first.documents_written == 1
    assert first.documents_skipped == 0
    assert first.errors == []
    assert call_doc_id_batches == [["ext:extraction-failed-1"]]
    fake_rag.doc_status.get_docs_by_status.assert_awaited_once_with(DocStatus.PROCESSED)

    second = ingest_documents([doc], index_dir=tmp_path)
    assert second.errors == []
    # The critical assertion: even though content is byte-identical to the
    # first call, and that first call's _insert_all_batches() did NOT raise,
    # the doc must be handed to _insert_all_batches() again -- because its
    # doc_status came back FAILED, not PROCESSED, so no fingerprint was ever
    # persisted for it.
    assert call_doc_id_batches == [
        ["ext:extraction-failed-1"],
        ["ext:extraction-failed-1"],
    ]
    assert second.documents_written == 1
    assert second.documents_skipped == 0


def test_initialize_storages_failure_is_reported_via_errors_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken vector store (rag.initialize_storages() raising) must surface
    via IngestResult.errors, per the module docstring's own example -- not
    propagate as an unhandled exception out of ingest_documents()."""
    doc = Document(
        doc_id="ext:broken-store",
        content="Experiment E10 confirmed hypothesis H2.",
        source_kind=SourceKind.EXTERNAL,
        source_path="postgres://experiments/10",
        metadata={},
    )

    fake_rag = _make_fake_rag()
    fake_rag.initialize_storages = AsyncMock(
        side_effect=RuntimeError("simulated corrupt vector store")
    )
    monkeypatch.setattr(
        "hars_memory.server.lightrag_init.create_lightrag", lambda **kwargs: fake_rag
    )

    insert_called = False

    async def unexpected_insert(rag, docs, batch_size):
        nonlocal insert_called
        insert_called = True

    monkeypatch.setattr(api_module, "_insert_all_batches", unexpected_insert)

    result = ingest_documents([doc], index_dir=tmp_path)

    assert result.documents_written == 0
    assert result.errors and "simulated corrupt vector store" in result.errors[0]
    assert insert_called is False
    # finalize_storages() must not be invoked on a rag whose storages never
    # finished initializing.
    fake_rag.finalize_storages.assert_not_called()


def test_unchanged_previously_successful_doc_is_not_reinserted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A doc that succeeded on a previous call must NOT be handed to
    _insert_all_batches() again on a later call with byte-identical content.

    This is the regression this fix-round addresses: round 1 removed the
    fingerprint pre-filter entirely, so every call unconditionally
    resubmitted every document -- including unchanged, already-successful
    ones -- to LightRAG's ainsert(), which (per the installed lightrag-hku
    JsonDocStatusStorage.filter_keys()/apipeline_enqueue_documents behavior)
    permanently writes a new duplicate FAILED doc_status record on every
    such resubmission. The fingerprint gate must skip these documents
    entirely, not just mark them "skipped" for reporting purposes.
    """
    doc = Document(
        doc_id="ext:stable-1",
        content="Experiment E11 confirmed hypothesis H4.",
        source_kind=SourceKind.EXTERNAL,
        source_path="postgres://experiments/11",
        metadata={},
    )

    fake_rag = _make_fake_rag()
    monkeypatch.setattr(
        "hars_memory.server.lightrag_init.create_lightrag", lambda **kwargs: fake_rag
    )

    call_doc_id_batches: list[list[str]] = []

    async def tracking_insert(rag, docs, batch_size):
        call_doc_id_batches.append([d.doc_id for d in docs])
        rag._known_doc_ids.update(d.doc_id for d in docs)

    monkeypatch.setattr(api_module, "_insert_all_batches", tracking_insert)

    first = ingest_documents([doc], index_dir=tmp_path)
    assert first.documents_written == 1
    assert first.documents_skipped == 0
    assert first.errors == []
    assert call_doc_id_batches == [["ext:stable-1"]]

    second = ingest_documents([doc], index_dir=tmp_path)
    assert second.errors == []
    # The critical assertion: _insert_all_batches must NOT be called again
    # for this doc -- the batch list must have exactly the one entry from
    # the first call, not two.
    assert call_doc_id_batches == [["ext:stable-1"]]
    assert second.documents_written == 0
    assert second.documents_skipped == 1


def test_changed_content_is_always_resubmitted_and_deletes_stale_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A doc whose content changed since a previous successful call must
    always be resubmitted to _insert_all_batches(), regardless of its prior
    (successful) status -- and its stale content must be deleted first."""
    doc_v1 = Document(
        doc_id="ext:changing-1",
        content="Experiment E12 confirmed hypothesis H5.",
        source_kind=SourceKind.EXTERNAL,
        source_path="postgres://experiments/12",
        metadata={},
    )
    doc_v2 = Document(
        doc_id="ext:changing-1",
        content="Experiment E12 confirmed hypothesis H5 -- REVISED.",
        source_kind=SourceKind.EXTERNAL,
        source_path="postgres://experiments/12",
        metadata={},
    )

    fake_rag = _make_fake_rag()
    monkeypatch.setattr(
        "hars_memory.server.lightrag_init.create_lightrag", lambda **kwargs: fake_rag
    )

    call_doc_id_batches: list[list[str]] = []

    async def tracking_insert(rag, docs, batch_size):
        call_doc_id_batches.append([d.doc_id for d in docs])
        rag._known_doc_ids.update(d.doc_id for d in docs)

    monkeypatch.setattr(api_module, "_insert_all_batches", tracking_insert)

    first = ingest_documents([doc_v1], index_dir=tmp_path)
    assert first.documents_written == 1
    assert call_doc_id_batches == [["ext:changing-1"]]
    fake_rag.adelete_by_doc_id.assert_not_awaited()  # nothing to delete yet

    second = ingest_documents([doc_v2], index_dir=tmp_path)
    assert second.errors == []
    assert second.documents_written == 1
    assert second.documents_skipped == 0
    # The critical assertion: the changed doc IS resubmitted a second time.
    assert call_doc_id_batches == [["ext:changing-1"], ["ext:changing-1"]]
    # And its stale content was deleted before the reinsert.
    fake_rag.adelete_by_doc_id.assert_awaited_once_with("ext:changing-1")


def test_source_kind_has_no_postgres_specific_members() -> None:
    from hars_memory.ingest.document import SourceKind

    names = {member.name for member in SourceKind}
    assert "POSTGRES_EXPERIMENT" not in names
    assert "POSTGRES_HYPOTHESIS" not in names
    assert "POSTGRES_HYPOTHESIS_LINK" not in names
    assert "EXTERNAL" in names
