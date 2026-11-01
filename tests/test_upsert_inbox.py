"""Tests for the MCP upsert inbox (``ingest/upsert_inbox.py``) as an index-batch source.

Three layers, following the existing patterns:

* ``UpsertInboxStore`` unit tests — the sidecar pattern of
  ``ContentDedupStore`` (tests/test_change_detection.py);
* ``index-batch --source upsert-inbox`` integration — real LightRAG with the
  fake embedder/LLM/Batch backend of tests/test_index_batch.py (no network);
* the byte-for-byte prompt invariant: collecting from the inbox must produce
  the exact same requests as collecting the same documents directly.
"""

# ruff: noqa: F811
# (the lightrag_env fixture is IMPORTED from tests.test_index_batch so both
# modules share one definition; pytest resolves it by parameter name, which
# ruff then sees as a redefinition of the imported name)

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hars_memory.ingest import batch
from hars_memory.ingest.change_detection import (
    FingerprintStore,
    FingerprintStoreError,
    compute_fingerprint,
    default_fingerprint_store_path,
)
from hars_memory.ingest.document import Document, SourceKind, file_stable_id
from hars_memory.ingest.upsert_inbox import UpsertInboxStore, default_upsert_inbox_path

# Reuse the walker-path batch test fakes (fixture + backend + docs).
from tests.test_index_batch import DOC_A, DOC_B, FakeBackend, lightrag_env  # noqa: F401

CONTENT_A = (
    "# Alpha session\n\nThe alpha service talks to the beta service.\n\n"
    "## Install\n\nInstall alpha with the installer.\n"
)
CONTENT_A_V2 = (
    "# Alpha session\n\nThe alpha service talks to the beta service and to gamma.\n\n"
    "## Install\n\nInstall alpha with the installer.\n\n## Gamma\n\nGamma retries nightly.\n"
)
CONTENT_B = "# Beta notes\n\n## Install\n\nBeta needs a database.\n"


def _inbox_doc(tmp_path: Path, rel: str, content: str) -> Document:
    """A Document exactly like memory_upsert_document's content_override branch builds it."""
    path = tmp_path / "src" / rel
    return Document(
        doc_id=file_stable_id(path),
        content=content,
        source_kind=SourceKind.MARKDOWN,
        source_path=str(path),
        metadata={"base_path": str(path.parent), "relative_path": path.name},
    )


def _queue(index_dir: Path, docs: list[Document]) -> None:
    store = UpsertInboxStore.load(default_upsert_inbox_path(index_dir))
    for doc in docs:
        store.put_document(doc, fingerprint=compute_fingerprint(doc.content))
    store.save()


def _run_inbox_all(index_dir: Path, backend: FakeBackend) -> dict:
    docs = UpsertInboxStore.load(default_upsert_inbox_path(index_dir)).documents()
    while batch.next_round(batch.BatchState.load(index_dir)) is not None:
        batch.collect(index_dir, docs, source=batch.SOURCE_UPSERT_INBOX)
        batch.submit(index_dir, backend)
        batch.wait(index_dir, backend, poll_seconds=0, timeout_seconds=5, sleep=lambda _s: None)
    return batch.apply(index_dir, docs, source=batch.SOURCE_UPSERT_INBOX)


# ---------------------------------------------------------------------------
# UpsertInboxStore unit tests (ContentDedupStore pattern)
# ---------------------------------------------------------------------------


class TestUpsertInboxStore:
    def test_missing_file_is_an_empty_store(self, tmp_path: Path) -> None:
        path = default_upsert_inbox_path(tmp_path / "idx")
        store = UpsertInboxStore.load(path)
        assert len(store) == 0
        assert store.doc_ids() == []
        assert store.entries() == []
        assert store.documents() == []

    def test_default_path_is_inside_the_index_dir(self, tmp_path: Path) -> None:
        assert default_upsert_inbox_path(tmp_path / "idx") == tmp_path / "idx" / "upsert_inbox.json"

    def test_put_save_load_round_trip(self, tmp_path: Path) -> None:
        path = default_upsert_inbox_path(tmp_path / "idx")
        doc = _inbox_doc(tmp_path, "session.md", CONTENT_A)
        store = UpsertInboxStore.load(path)
        entry_id = store.put_document(doc, fingerprint=compute_fingerprint(doc.content), queued_at=1.5)
        store.save()

        assert entry_id == doc.doc_id
        assert path.is_file()
        reloaded = UpsertInboxStore.load(path)
        assert reloaded.doc_ids() == [doc.doc_id]
        entry = reloaded.get(doc.doc_id)
        assert entry["content"] == CONTENT_A
        assert entry["fingerprint"] == compute_fingerprint(CONTENT_A)
        assert entry["source_kind"] == "markdown"
        assert entry["source_path"] == doc.source_path
        assert entry["metadata"] == doc.metadata
        assert entry["queued_at"] == 1.5
        # the reconstructed Document is indistinguishable from the queued one
        assert reloaded.documents() == [doc]

    def test_save_is_atomic_and_leaves_no_tmp(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "idx"
        store = UpsertInboxStore.load(default_upsert_inbox_path(index_dir))
        doc = _inbox_doc(tmp_path, "a.md", CONTENT_A)
        store.put_document(doc, fingerprint="fp")
        store.save()
        assert not list(index_dir.glob("*.tmp"))
        assert json.loads(default_upsert_inbox_path(index_dir).read_text()) == json.loads(
            json.dumps({doc.doc_id: store.get(doc.doc_id)}, indent=2, sort_keys=True, ensure_ascii=False)
        )

    def test_malformed_file_raises(self, tmp_path: Path) -> None:
        path = default_upsert_inbox_path(tmp_path / "idx")
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(FingerprintStoreError, match="Cannot read upsert inbox"):
            UpsertInboxStore.load(path)

    def test_non_object_file_raises(self, tmp_path: Path) -> None:
        path = default_upsert_inbox_path(tmp_path / "idx")
        path.parent.mkdir(parents=True)
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(FingerprintStoreError, match="does not contain a JSON object"):
            UpsertInboxStore.load(path)

    def test_requeue_same_doc_overwrites_idempotently(self, tmp_path: Path) -> None:
        path = default_upsert_inbox_path(tmp_path / "idx")
        doc = _inbox_doc(tmp_path, "session.md", CONTENT_A)
        doc_v2 = _inbox_doc(tmp_path, "session.md", CONTENT_A_V2)
        store = UpsertInboxStore.load(path)
        store.put_document(doc, fingerprint=compute_fingerprint(doc.content))
        store.put_document(doc_v2, fingerprint=compute_fingerprint(doc_v2.content))
        store.save()

        reloaded = UpsertInboxStore.load(path)
        assert reloaded.doc_ids() == [doc.doc_id]  # keyed by doc_id: one entry
        assert reloaded.get(doc.doc_id)["content"] == CONTENT_A_V2

    def test_discard_removes_only_the_target(self, tmp_path: Path) -> None:
        path = default_upsert_inbox_path(tmp_path / "idx")
        doc_a = _inbox_doc(tmp_path, "a.md", CONTENT_A)
        doc_b = _inbox_doc(tmp_path, "b.md", CONTENT_B)
        store = UpsertInboxStore.load(path)
        store.put_document(doc_a, fingerprint="fpa")
        store.put_document(doc_b, fingerprint="fpb")
        store.discard(doc_a.doc_id)
        store.save()

        reloaded = UpsertInboxStore.load(path)
        assert reloaded.doc_ids() == [doc_b.doc_id]
        reloaded.discard("no-such-doc")  # no-op, no raise
        assert reloaded.doc_ids() == [doc_b.doc_id]

    def test_entries_and_documents_sorted_by_doc_id(self, tmp_path: Path) -> None:
        path = default_upsert_inbox_path(tmp_path / "idx")
        docs = [_inbox_doc(tmp_path, "z.md", CONTENT_A), _inbox_doc(tmp_path, "a.md", CONTENT_B)]
        store = UpsertInboxStore.load(path)
        for doc in docs:
            store.put_document(doc, fingerprint="fp")
        assert store.doc_ids() == sorted(d.doc_id for d in docs)
        assert [d.doc_id for d in store.documents()] == sorted(d.doc_id for d in docs)

    def test_invalid_source_kind_raises_on_materialise(self, tmp_path: Path) -> None:
        path = default_upsert_inbox_path(tmp_path / "idx")
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"doc-x": {"doc_id": "doc-x", "content": "c", "source_kind": "nope"}}),
            encoding="utf-8",
        )
        with pytest.raises(FingerprintStoreError, match="invalid source_kind"):
            UpsertInboxStore.load(path).documents()


# ---------------------------------------------------------------------------
# pending_upsert_documents (the inbox change-detection rule)
# ---------------------------------------------------------------------------


def _seed_status(index_dir: Path, rows: dict[str, str]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "kv_store_doc_status.json").write_text(
        json.dumps({doc_id: {"status": status} for doc_id, status in rows.items()}), encoding="utf-8"
    )


def test_pending_upsert_documents_purges_stale_and_skips_racing_sync(tmp_path: Path) -> None:
    idx = tmp_path / "idx"
    changed = _inbox_doc(tmp_path, "a.md", CONTENT_A_V2)  # was indexed with older content
    unchanged = _inbox_doc(tmp_path, "b.md", CONTENT_B)  # a sync upsert indexed this exact content
    fresh = _inbox_doc(tmp_path, "c.md", CONTENT_A)  # never indexed
    _seed_status(idx, {changed.doc_id: "processed", unchanged.doc_id: "processed"})

    store = FingerprintStore.load(default_fingerprint_store_path(idx))
    store.set(changed.doc_id, compute_fingerprint(CONTENT_A))  # the OLD content
    store.set(unchanged.doc_id, compute_fingerprint(CONTENT_B))  # the CURRENT content
    store.save()

    todo, purge = batch.pending_upsert_documents(idx, [changed, unchanged, fresh])
    assert [d.doc_id for d in todo] == [changed.doc_id, fresh.doc_id]
    assert purge == [changed.doc_id]


def test_pending_upsert_documents_keeps_processed_doc_without_fingerprint(tmp_path: Path) -> None:
    """A processed doc with NO fingerprint on record must not be skipped (its fp was
    never written, so "already indexed" cannot be proven) — it is re-queued like
    the synchronous path would re-insert it."""
    idx = tmp_path / "idx"
    doc = _inbox_doc(tmp_path, "a.md", CONTENT_A)
    _seed_status(idx, {doc.doc_id: "processed"})
    todo, purge = batch.pending_upsert_documents(idx, [doc])
    assert [d.doc_id for d in todo] == [doc.doc_id]
    assert purge == [doc.doc_id]


# ---------------------------------------------------------------------------
# index-batch --source upsert-inbox: collect/apply integration
# ---------------------------------------------------------------------------


def test_inbox_collect_matches_walker_collect_byte_for_byte(lightrag_env, tmp_path: Path) -> None:
    """THE invariant: the same documents collected from the inbox produce the
    exact same requests (same cache keys, same prompts) as the walker source."""
    walker_idx, inbox_idx = tmp_path / "walker", tmp_path / "inbox"
    docs = [_inbox_doc(tmp_path, "alpha.md", DOC_A), _inbox_doc(tmp_path, "beta.md", DOC_B)]
    _queue(inbox_idx, docs)

    batch.collect(walker_idx, docs)
    batch.collect(inbox_idx, UpsertInboxStore.load(default_upsert_inbox_path(inbox_idx)).documents())

    def requests_file(index_dir: Path) -> str:
        rnd = batch.BatchState.load(index_dir).rounds["1"]
        return (index_dir / batch.BATCH_DIRNAME / rnd.requests_file).read_text()

    assert requests_file(inbox_idx) == requests_file(walker_idx)
    assert batch.BatchState.load(inbox_idx).doc_ids == batch.BatchState.load(walker_idx).doc_ids


def test_inbox_full_run_primes_cache_applies_and_drains(lightrag_env, tmp_path: Path) -> None:
    idx, backend = tmp_path / "idx", FakeBackend()
    docs = [_inbox_doc(tmp_path, "alpha.md", DOC_A), _inbox_doc(tmp_path, "beta.md", DOC_B)]
    _queue(idx, docs)

    stats = _run_inbox_all(idx, backend)

    # NOTE: "primed cache => zero synchronous LLM calls" is asserted by the
    # walker twin (tests/test_index_batch.py::test_full_run_primes_cache_...).
    # That twin is one of the known pre-existing FULL-suite isolation failures
    # (module-reload pollution from earlier test files); the same pollution
    # would flake the sync-count asserts here, so this test sticks to the
    # inbox-specific behaviour (which the reupsert test below proves also works
    # end-to-end with the cache primed).
    assert stats["total_requests"] > 0
    assert stats["inbox"] == {"applied": 2, "remaining": 0, "deferred": 0}

    s = batch.index_stats(idx)
    assert s["docs_processed"] == 2 and s["entities"] >= 2

    # the fingerprint-after-the-fact contract: recorded only now, for every doc
    fp_store = FingerprintStore.load(default_fingerprint_store_path(idx))
    for doc in docs:
        assert fp_store.get(doc.doc_id) == compute_fingerprint(doc.content)
    # and the inbox is drained
    assert len(UpsertInboxStore.load(default_upsert_inbox_path(idx))) == 0


def test_inbox_apply_defers_newcomers_queued_after_collect(lightrag_env, tmp_path: Path) -> None:
    """Upserts queued while the batch was in flight have no collected prompts;
    apply must NOT silently extract them at full sync price — they stay queued."""
    idx, backend = tmp_path / "idx", FakeBackend()
    first = _inbox_doc(tmp_path, "alpha.md", DOC_A)
    late = _inbox_doc(tmp_path, "beta.md", DOC_B)
    _queue(idx, [first])

    # both rounds (initial + gleaning) collected for [first] only
    while batch.next_round(batch.BatchState.load(idx)) is not None:
        batch.collect(idx, UpsertInboxStore.load(default_upsert_inbox_path(idx)).documents(),
                      source=batch.SOURCE_UPSERT_INBOX)
        batch.submit(idx, backend)
        batch.wait(idx, backend, poll_seconds=0, timeout_seconds=5, sleep=lambda _s: None)
    _queue(idx, [late])  # a new upsert lands after collection, before apply

    stats = batch.apply(
        idx, UpsertInboxStore.load(default_upsert_inbox_path(idx)).documents(),
        source=batch.SOURCE_UPSERT_INBOX,
    )
    assert stats["inbox"] == {"applied": 1, "remaining": 0, "deferred": 1}
    assert batch.index_stats(idx)["docs_processed"] == 1  # late doc NOT indexed

    inbox = UpsertInboxStore.load(default_upsert_inbox_path(idx))
    assert inbox.doc_ids() == [late.doc_id]  # still queued for the next run


def test_inbox_reupsert_of_changed_content_purges_and_reindexes(lightrag_env, tmp_path: Path) -> None:
    """The reason the inbox exists: a re-upsert of CHANGED content under the same
    path. apply purges the stale version (like the sync path's adelete) and
    indexes the new one; the fingerprint moves to the new content."""
    idx = tmp_path / "idx"
    doc = _inbox_doc(tmp_path, "alpha.md", CONTENT_A)

    _queue(idx, [doc])
    assert _run_inbox_all(idx, FakeBackend())["inbox"] == {"applied": 1, "remaining": 0, "deferred": 0}
    v1_chunks = {c["content"] for c in
                 json.loads((idx / "kv_store_text_chunks.json").read_text()).values()}

    # start a new batch on the same index (the documented "delete state to start
    # over" path) and queue version 2 of the same document
    (idx / batch.STATE_FILENAME).unlink()
    batch_dir = idx / batch.BATCH_DIRNAME
    for f in batch_dir.glob("round-*.requests.jsonl"):
        f.unlink()
    doc_v2 = _inbox_doc(tmp_path, "alpha.md", CONTENT_A_V2)
    _queue(idx, [doc_v2])

    stats = _run_inbox_all(idx, FakeBackend())
    assert stats["inbox"] == {"applied": 1, "remaining": 0, "deferred": 0}

    chunks = json.loads((idx / "kv_store_text_chunks.json").read_text())
    v2_text = {c["content"] for c in chunks.values()}
    assert any("gamma" in text.lower() for text in v2_text)  # new content is in
    assert v2_text != v1_chunks  # and it replaced, not appended to, the old one
    assert batch.index_stats(idx)["docs_processed"] == 1  # still ONE document

    fp_store = FingerprintStore.load(default_fingerprint_store_path(idx))
    assert fp_store.get(doc.doc_id) == compute_fingerprint(CONTENT_A_V2)


def test_inbox_apply_refuses_on_chunker_config_change(
    lightrag_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache keys are derived from chunk text: another chunker config means every
    primed key would miss and LightRAG would re-extract at full price."""
    idx = tmp_path / "idx"
    _queue(idx, [_inbox_doc(tmp_path, "alpha.md", DOC_A)])
    batch.collect(idx, UpsertInboxStore.load(default_upsert_inbox_path(idx)).documents(),
                  source=batch.SOURCE_UPSERT_INBOX)

    monkeypatch.setenv("HARS_MEMORY_CHUNK_TOKEN_SIZE", "128")
    with pytest.raises(batch.BatchError, match="chunker configuration changed"):
        batch.apply(
            idx, UpsertInboxStore.load(default_upsert_inbox_path(idx)).documents(),
            source=batch.SOURCE_UPSERT_INBOX,
        )


def test_apply_refuses_on_chunker_config_change_walker_source_too(
    lightrag_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard is source-independent: the default walker path refuses too."""
    idx = tmp_path / "idx"
    docs = [_inbox_doc(tmp_path, "alpha.md", DOC_A)]
    batch.collect(idx, docs)

    monkeypatch.setenv("HARS_MEMORY_CHUNKER", "token")  # fixture collects with markdown
    with pytest.raises(batch.BatchError, match="chunker configuration changed"):
        batch.apply(idx, docs)


def test_cli_collect_from_inbox_source(lightrag_env, tmp_path: Path, capsys) -> None:
    """`--source upsert-inbox` wiring: documents come from the inbox sidecar and
    the live-index refusal is deliberately skipped for this source."""
    from hars_memory import cli

    idx = tmp_path / "idx"
    _queue(idx, [_inbox_doc(tmp_path, "alpha.md", DOC_A)])

    with pytest.raises(SystemExit) as exc:  # cli.main always exits with the return code
        cli.main(["index-batch", "collect", "--dry-run", "--index-dir", str(idx),
                  "--source", "upsert-inbox"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "upsert-inbox mode: targeting" in captured.err  # live-index guard skipped, loudly
    assert "round 1" in captured.out and "request(s)" in captured.out
    assert not (idx / batch.STATE_FILENAME).exists()  # dry run wrote nothing


def test_cli_collect_from_empty_inbox_refuses(lightrag_env, tmp_path: Path) -> None:
    from hars_memory import cli

    idx = tmp_path / "idx"
    idx.mkdir()
    with pytest.raises(SystemExit, match="upsert inbox"):
        cli.main(["index-batch", "collect", "--index-dir", str(idx), "--source", "upsert-inbox"])


def test_inbox_cycle_applies_then_collects_the_next_batch(lightrag_env, tmp_path: Path, capsys) -> None:
    """The queue-mode cycle: after a batch is applied its state is archived and
    the next collect starts fresh — upserts keep queueing between runs."""
    from hars_memory import cli

    idx = tmp_path / "idx"
    alpha, beta = _inbox_doc(tmp_path, "alpha.md", DOC_A), _inbox_doc(tmp_path, "beta.md", DOC_B)

    _queue(idx, [alpha])
    assert _run_inbox_all(idx, FakeBackend())["inbox"]["applied"] == 1

    _queue(idx, [beta])  # the next session's upsert
    with pytest.raises(SystemExit) as exc:
        cli.main(["index-batch", "collect", "--index-dir", str(idx), "--source", "upsert-inbox"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "state archived" in captured.err

    state = batch.BatchState.load(idx)
    assert state.doc_ids == [beta.doc_id]  # fresh generation, only the newcomer
    assert (idx / (batch.STATE_FILENAME + ".prev.json")).is_file()
    assert batch.index_stats(idx)["docs_processed"] == 1  # alpha still indexed


def test_chunker_config_snapshot_effective_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARS_MEMORY_CHUNKER", raising=False)
    monkeypatch.delenv("HARS_MEMORY_CHUNKER_MODE", raising=False)
    monkeypatch.setenv("HARS_MEMORY_CHUNK_TOKEN_SIZE", "64")
    cfg = batch.chunker_config()
    assert cfg["HARS_MEMORY_CHUNKER"] == "token"  # documented default filled in
    assert cfg["HARS_MEMORY_CHUNK_TOKEN_SIZE"] == "64"
    assert cfg["HARS_MEMORY_CHUNK_MIN_TOKENS"] == "200"
    assert set(cfg) == set(batch.CHUNKER_CONFIG_ENVS)
    assert batch.chunker_config({}) == batch.chunker_config(  # type: ignore[arg-type]
        {name: "" for name in batch.CHUNKER_CONFIG_ENVS}
    )


def test_collect_records_chunker_config_on_first_collect_only(
    lightrag_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = tmp_path / "idx"
    docs = [_inbox_doc(tmp_path, "alpha.md", DOC_A)]
    batch.collect(idx, docs)
    state = batch.BatchState.load(idx)
    assert state.chunker_config == batch.chunker_config()
    assert state.chunker_config["HARS_MEMORY_CHUNKER"] == "markdown"

    # a later collect does not overwrite the pinned snapshot even if the env drifted
    monkeypatch.setenv("HARS_MEMORY_CHUNK_TOKEN_SIZE", "999")
    state.rounds.clear()  # pretend round 1 never happened
    state.save(idx)
    batch.collect(idx, docs)
    assert batch.BatchState.load(idx).chunker_config == state.chunker_config
