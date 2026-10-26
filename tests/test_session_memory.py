"""Hermetic tests for hars_memory.session_memory.SessionMemoryStore.

No network, no LLM, no real LightRAG: every test either replaces
``hars_memory.session_memory.ingest_documents`` with a recording fake or
monkeypatches the ingest pipeline's ``create_lightrag`` /
``_insert_all_batches`` (the same pattern tests/test_ingest_api.py uses).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

import hars_memory.session_memory as session_memory_module
from hars_memory.ingest.api import IngestResult, ingest_documents
from hars_memory.session_memory import (
    DEFAULT_INDEX_BS,
    SessionMemoryStore,
    clamp_index_bs,
    validate_session_id,
)
from hars_memory.scripts.cleanup_kb import HEADER_RE


class _RecordingIngest:
    """Stand-in for ``ingest_documents`` that records and never indexes."""

    def __init__(self, *, errors: list[str] | None = None, raises: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.errors = errors or []
        self.raises = raises

    def __call__(
        self,
        docs: list[Any],
        *,
        index_dir: Path,
        vector_storage: str | None = None,
        graph_storage: str | None = None,
        batch_size: int | None = None,
        embed_batch_size: int | None = None,
    ) -> IngestResult:
        self.calls.append({
            "docs": list(docs),
            "index_dir": Path(index_dir),
            "vector_storage": vector_storage,
            "graph_storage": graph_storage,
            "batch_size": batch_size,
            "embed_batch_size": embed_batch_size,
        })
        if self.raises:
            raise RuntimeError("simulated ingest failure")
        return IngestResult(
            documents_written=len(docs), documents_skipped=0, errors=list(self.errors)
        )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_graphml(path: Path, nodes: list[str]) -> None:
    import networkx as nx

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node, entity_type="Concept", description="stub")
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, str(path))


def _make_fake_rag() -> AsyncMock:
    """Minimal LightRAG double for exercising the REAL ingest_documents path."""
    from lightrag.base import DocStatus

    known: set[str] = set()

    async def fake_get_docs_by_status(status: DocStatus) -> dict[str, Any]:
        if status is not DocStatus.PROCESSED:
            return {}
        return {doc_id: object() for doc_id in known}

    rag = AsyncMock()
    rag.initialize_storages = AsyncMock(return_value=None)
    rag.finalize_storages = AsyncMock(return_value=None)
    rag.adelete_by_doc_id = AsyncMock(return_value=None)
    rag.doc_status = AsyncMock()
    rag.doc_status.get_docs_by_status = AsyncMock(side_effect=fake_get_docs_by_status)
    rag._known_doc_ids = known
    return rag


# ---------------------------------------------------------------------------
# (a) header + body shape
# ---------------------------------------------------------------------------


def test_compaction_document_header_is_three_fields_and_parses(tmp_path: Path) -> None:
    store = SessionMemoryStore(index_dir=tmp_path)
    document = store._build_compaction_document(
        "sess-1", 3, "Summary text.", ["fact one", "fact\ntwo"], {"attempt": 2}
    )

    first_line = document.content.splitlines()[0]
    assert first_line.count("|") == 2
    match = HEADER_RE.search(document.content[:400])
    assert match is not None
    assert match.group("name").strip() == "session-sess-1-c3"
    assert match.group("section").strip() == "session"

    assert document.doc_id == "session:sess-1:c3"
    assert document.source_path == "sessions/sess-1/compaction-3.md"
    assert "Session: sess-1" in document.content
    assert "Seq: 3" in document.content
    assert "Tags: session, compaction" in document.content
    assert 'Metadata: {"attempt":2}' in document.content
    assert "## Summary" in document.content
    assert "## Dropped context facts" in document.content
    assert "- fact one" in document.content
    assert "- fact two" in document.content  # internal newline collapsed


def test_compaction_document_omits_optional_sections(tmp_path: Path) -> None:
    store = SessionMemoryStore(index_dir=tmp_path)
    document = store._build_compaction_document("sess", 1, "only summary", None, None)
    assert "Metadata:" not in document.content
    assert "## Dropped context facts" not in document.content


# ---------------------------------------------------------------------------
# (b) stable ids + fingerprint no-op through the real ingest gate
# ---------------------------------------------------------------------------


def test_doc_ids_are_stable_across_builds(tmp_path: Path) -> None:
    store = SessionMemoryStore(index_dir=tmp_path)
    first = store._build_compaction_document("s", 7, "same", ["f"], None)
    second = store._build_compaction_document("s", 7, "same", ["f"], None)
    assert first.doc_id == second.doc_id == "session:s:c7"
    assert first.content == second.content


def test_reingest_same_session_seq_is_a_fingerprint_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hars_memory.ingest import api as api_module

    fake_rag = _make_fake_rag()
    monkeypatch.setattr(
        "hars_memory.server.lightrag_init.create_lightrag", lambda **kwargs: fake_rag
    )
    inserted: list[list[str]] = []

    async def fake_insert(rag: Any, docs: list[Any], batch_size: int) -> None:
        inserted.append([doc.doc_id for doc in docs])
        rag._known_doc_ids.update(doc.doc_id for doc in docs)

    monkeypatch.setattr(api_module, "_insert_all_batches", fake_insert)
    # Guarantee the real `ingest_documents` is the one under test.
    assert session_memory_module.ingest_documents is ingest_documents

    store = SessionMemoryStore(index_dir=tmp_path)
    first = store.store_compaction("sess-a", "the same summary", seq=1)
    second = store.store_compaction("sess-a", "the same summary", seq=1)

    assert first["seq"] == second["seq"] == 1
    assert inserted == [["session:sess-a:c1"]]
    assert (tmp_path / "sessions" / "sess-a" / "doc_fingerprints.json").is_file()


# ---------------------------------------------------------------------------
# (c) seq allocation + rollback
# ---------------------------------------------------------------------------


def test_seq_auto_increments_from_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _RecordingIngest()
    monkeypatch.setattr(session_memory_module, "ingest_documents", recorder)
    store = SessionMemoryStore(index_dir=tmp_path)

    seqs = [store.store_compaction("sess", f"summary {index}")["seq"] for index in range(3)]

    assert seqs == [1, 2, 3]
    seq_payload = json.loads(
        (tmp_path / "sessions" / "sess" / "seq.json").read_text(encoding="utf-8")
    )
    assert seq_payload["seq"] == 3


@pytest.mark.parametrize(
    "fake",
    [
        _RecordingIngest(raises=True),
        _RecordingIngest(errors=["extractor down"]),
    ],
    ids=["exception", "errors-list"],
)
def test_ingest_failure_rolls_back_seq_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: _RecordingIngest
) -> None:
    good = _RecordingIngest()
    monkeypatch.setattr(session_memory_module, "ingest_documents", good)
    store = SessionMemoryStore(index_dir=tmp_path)
    store.store_compaction("sess", "first")  # seq 1 persisted
    seq_path = tmp_path / "sessions" / "sess" / "seq.json"
    assert json.loads(seq_path.read_text(encoding="utf-8"))["seq"] == 1

    monkeypatch.setattr(session_memory_module, "ingest_documents", fake)
    with pytest.raises(RuntimeError):
        store.store_compaction("sess", "second")

    assert json.loads(seq_path.read_text(encoding="utf-8"))["seq"] == 1


def test_ingest_failure_removes_seq_sidecar_when_none_existed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failing = _RecordingIngest(raises=True)
    monkeypatch.setattr(session_memory_module, "ingest_documents", failing)
    store = SessionMemoryStore(index_dir=tmp_path)
    seq_path = tmp_path / "sessions" / "fresh" / "seq.json"

    with pytest.raises(RuntimeError):
        store.store_compaction("fresh", "summary")

    assert not seq_path.exists()


def test_explicit_seq_is_validated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_memory_module, "ingest_documents", _RecordingIngest())
    store = SessionMemoryStore(index_dir=tmp_path)
    with pytest.raises(ValueError):
        store.store_compaction("sess", "summary", seq=0)
    with pytest.raises(ValueError):
        store.store_compaction("sess", "summary", seq=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        store.store_compaction("sess", "")


# ---------------------------------------------------------------------------
# (d) target dir + forced storages per mode
# ---------------------------------------------------------------------------


def test_split_mode_ingests_into_session_dir_with_forced_file_storages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A Qdrant main index must NOT leak into the session's storages.
    monkeypatch.setenv("HARS_MEMORY_VECTOR_STORAGE", "QdrantVectorDBStorage")
    recorder = _RecordingIngest()
    monkeypatch.setattr(session_memory_module, "ingest_documents", recorder)
    store = SessionMemoryStore(index_dir=tmp_path, mode="split")

    payload = store.store_compaction("sess-q", "summary", dropped_context=["a"])

    assert payload["mode"] == "split"
    assert payload["index_dir"] == str(tmp_path / "sessions" / "sess-q")
    call = recorder.calls[0]
    assert call["index_dir"] == tmp_path / "sessions" / "sess-q"
    assert call["vector_storage"] == "NanoVectorDBStorage"
    assert call["graph_storage"] == "NetworkXStorage"


def test_subgraph_mode_ingests_into_main_index_with_env_driven_storages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _RecordingIngest()
    monkeypatch.setattr(session_memory_module, "ingest_documents", recorder)
    store = SessionMemoryStore(index_dir=tmp_path, mode="subgraph")

    payload = store.store_compaction("sess-s", "summary")

    assert payload["mode"] == "subgraph"
    assert payload["index_dir"] == str(tmp_path)
    call = recorder.calls[0]
    assert call["index_dir"] == tmp_path
    assert call["vector_storage"] is None
    assert call["graph_storage"] is None


# ---------------------------------------------------------------------------
# (e) batch size defaults / env override / clamping
# ---------------------------------------------------------------------------


def test_index_bs_default_env_override_and_clamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HARS_MEMORY_SESSION_INDEX_BS", raising=False)
    assert SessionMemoryStore(index_dir=tmp_path).index_bs == DEFAULT_INDEX_BS == 25

    monkeypatch.setenv("HARS_MEMORY_SESSION_INDEX_BS", "8")
    assert SessionMemoryStore(index_dir=tmp_path).index_bs == 8

    monkeypatch.setenv("HARS_MEMORY_SESSION_INDEX_BS", "999")
    assert SessionMemoryStore(index_dir=tmp_path).index_bs == 256
    monkeypatch.setenv("HARS_MEMORY_SESSION_INDEX_BS", "0")
    assert SessionMemoryStore(index_dir=tmp_path).index_bs == 1

    assert clamp_index_bs(-5) == 1
    assert clamp_index_bs(10_000) == 256
    with pytest.raises(ValueError):
        clamp_index_bs("nope")


def test_index_bs_is_passed_to_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _RecordingIngest()
    monkeypatch.setattr(session_memory_module, "ingest_documents", recorder)
    store = SessionMemoryStore(index_dir=tmp_path, index_bs=8)
    store.store_compaction("sess", "summary")
    assert recorder.calls[0]["batch_size"] == 8
    assert recorder.calls[0]["embed_batch_size"] == 8


# ---------------------------------------------------------------------------
# (f) session id traversal guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    ["../etc", "", "a/b", "sess id", "юникод", ".", "..", "a" * 129, "sess\nid"],
)
def test_invalid_session_ids_raise(tmp_path: Path, bad_id: str) -> None:
    store = SessionMemoryStore(index_dir=tmp_path)
    with pytest.raises(ValueError):
        validate_session_id(bad_id)
    with pytest.raises(ValueError):
        store.store_compaction(bad_id, "summary")


def test_valid_session_id_alphabet() -> None:
    assert validate_session_id(" Agent.Session_1-2 ") == "Agent.Session_1-2"


# ---------------------------------------------------------------------------
# (g) merge_session
# ---------------------------------------------------------------------------


def _prepare_split_session(
    tmp_path: Path,
    session_id: str = "sess-merge",
    main_nodes: list[str] | None = None,
    session_nodes: list[str] | None = None,
) -> SessionMemoryStore:
    store = SessionMemoryStore(index_dir=tmp_path / "index", mode="split")
    session_dir = tmp_path / "index" / "sessions" / session_id
    full_docs: dict[str, Any] = {}
    for seq in (1, 2, 3):
        document = store._build_compaction_document(
            session_id, seq, f"summary number {seq}", [f"fact {seq}"], None
        )
        full_docs[document.doc_id] = {"content": document.content}
    _write_json(session_dir / "kv_store_full_docs.json", full_docs)
    _write_graphml(
        tmp_path / "index" / session_memory_module.GRAPH_FILE_NAME,
        main_nodes or ["LightRAG", "NanoVectorDB", "Qdrant"],
    )
    _write_graphml(
        session_dir / session_memory_module.GRAPH_FILE_NAME,
        session_nodes or ["lightrag", "NanoVectorDBX", "UniqueThing"],
    )
    return store


def test_merge_builds_one_chronological_doc_with_alias_hints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _prepare_split_session(tmp_path)
    recorder = _RecordingIngest()
    monkeypatch.setattr(session_memory_module, "ingest_documents", recorder)

    payload = store.merge_session("sess-merge")

    assert payload["ok"] is True
    assert payload["merged"] is True
    assert payload["compactions"] == 3
    assert payload["merged_doc_id"] == "session:sess-merge:merged"
    assert payload["chars"] > 0
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["index_dir"] == tmp_path / "index"  # the MAIN index
    assert len(call["docs"]) == 1
    merged = call["docs"][0]
    assert merged.doc_id == "session:sess-merge:merged"
    assert merged.content.index("## Compaction 1") < merged.content.index("## Compaction 2")
    assert merged.content.index("## Compaction 2") < merged.content.index("## Compaction 3")
    assert merged.content.startswith(
        "[Document: session-sess-merge-merged | Section: session | Date: "
    )
    assert "## Entity aliases" in merged.content
    hints = payload["alias_hints"]
    assert "lightrag -> LightRAG" in hints
    assert "NanoVectorDBX -> NanoVectorDB" in hints
    assert all("UniqueThing" not in hint for hint in hints)

    marker = tmp_path / "index" / "sessions" / "sess-merge" / "merged.json"
    stats = json.loads(marker.read_text(encoding="utf-8"))
    assert stats["compactions"] == 3
    assert stats["merged_doc_id"] == "session:sess-merge:merged"
    assert stats["chars"] == payload["chars"]
    assert stats["alias_hints"] == hints


def test_merge_is_idempotent_via_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _prepare_split_session(tmp_path)
    recorder = _RecordingIngest()
    monkeypatch.setattr(session_memory_module, "ingest_documents", recorder)

    first = store.merge_session("sess-merge")
    second = store.merge_session("sess-merge")

    assert second["already_merged"] is True
    assert second["merged_at"] == first["merged_at"]
    assert len(recorder.calls) == 1  # no second ingest


def test_merge_delete_source_removes_session_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _prepare_split_session(tmp_path, session_id="sess-del")
    monkeypatch.setattr(session_memory_module, "ingest_documents", _RecordingIngest())

    payload = store.merge_session("sess-del", delete_source=True)

    assert payload["delete_source"] is True
    assert not (tmp_path / "index" / "sessions" / "sess-del").exists()


def test_merge_subgraph_mode_is_a_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingIngest()
    monkeypatch.setattr(session_memory_module, "ingest_documents", recorder)
    store = SessionMemoryStore(index_dir=tmp_path, mode="subgraph")

    payload = store.merge_session("sess")

    assert payload["ok"] is True
    assert payload["merged"] is False
    assert "subgraph" in payload["reason"]
    assert recorder.calls == []


def test_merge_unknown_session_raises(tmp_path: Path) -> None:
    store = SessionMemoryStore(index_dir=tmp_path, mode="split")
    with pytest.raises(ValueError, match="unknown session"):
        store.merge_session("missing")


def test_merge_ingest_failure_does_not_write_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _prepare_split_session(tmp_path)
    monkeypatch.setattr(
        session_memory_module, "ingest_documents", _RecordingIngest(raises=True)
    )
    with pytest.raises(RuntimeError):
        store.merge_session("sess-merge")
    assert not (tmp_path / "index" / "sessions" / "sess-merge" / "merged.json").exists()


# ---------------------------------------------------------------------------
# (h) session_status fail-soft
# ---------------------------------------------------------------------------


def test_session_status_lists_sessions_and_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _prepare_split_session(tmp_path, session_id="sess-ok")
    monkeypatch.setattr(session_memory_module, "ingest_documents", _RecordingIngest())
    store.merge_session("sess-ok")

    # A second session whose stores are corrupt: every stat must read as zero.
    corrupt_dir = tmp_path / "index" / "sessions" / "sess-corrupt"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "kv_store_full_docs.json").write_text("{not json", encoding="utf-8")
    (corrupt_dir / session_memory_module.GRAPH_FILE_NAME).write_text("<broken>", encoding="utf-8")

    listing = store.session_status()
    assert listing["mode"] == "split"
    assert listing["sessions_dir"] == str(tmp_path / "index" / "sessions")
    by_id = {entry["session_id"]: entry for entry in listing["sessions"]}
    assert set(by_id) == {"sess-ok", "sess-corrupt"}
    assert by_id["sess-ok"]["compactions"] == 3
    assert by_id["sess-ok"]["merged"] is True
    assert by_id["sess-ok"]["merged_at"]
    assert by_id["sess-corrupt"]["compactions"] == 0
    assert by_id["sess-corrupt"]["entities"] == 0
    assert by_id["sess-corrupt"]["relations"] == 0
    assert by_id["sess-corrupt"]["merged"] is False

    detail = store.session_status("sess-ok")
    assert detail["session_id"] == "sess-ok"
    assert detail["exists"] is True
    assert detail["index_dir"] == str(tmp_path / "index" / "sessions" / "sess-ok")
    assert detail["compactions"] == 3
    assert detail["entities"] == 3  # graphml stub nodes
    assert detail["relations"] == 0

    unknown = store.session_status("never-seen")
    assert unknown["exists"] is False
    assert unknown["compactions"] == 0
    assert unknown["merged"] is False


def test_session_status_rejects_invalid_id(tmp_path: Path) -> None:
    store = SessionMemoryStore(index_dir=tmp_path)
    with pytest.raises(ValueError):
        store.session_status("../escape")


def test_session_status_subgraph_counts_sequence_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subgraph compactions live in the main index; the session dir only has
    the seq sidecar, whose high-water mark is the best available count."""
    monkeypatch.setattr(session_memory_module, "ingest_documents", _RecordingIngest())
    store = SessionMemoryStore(index_dir=tmp_path / "index", mode="subgraph")
    for index in range(3):
        store.store_compaction("sess-sub", f"summary {index}")

    detail = store.session_status("sess-sub")

    assert detail["mode"] == "subgraph"
    assert detail["index_dir"] == str(tmp_path / "index")
    assert detail["compactions"] == 3
    assert detail["merged"] is False


def test_session_status_ignores_non_session_directories(tmp_path: Path) -> None:
    store = SessionMemoryStore(index_dir=tmp_path / "index")
    (tmp_path / "index" / "sessions" / "not a session").mkdir(parents=True)
    (tmp_path / "index" / "sessions" / "loose-file.txt").write_text("x", encoding="utf-8")
    listing = store.session_status()
    assert listing["sessions"] == []
