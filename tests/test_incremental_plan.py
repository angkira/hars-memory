"""Classification logic of the incremental refresh planner (pure, no LightRAG / Qdrant)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hars_memory.ingest.change_detection import FingerprintStore, compute_fingerprint
from hars_memory.ingest.document import ID_PATH_MAP_ENV, Document, SourceKind, file_stable_id
from hars_memory.ingest.incremental import PlanError, build_plan


def _doc(doc_id: str, content: str) -> Document:
    return Document(doc_id=doc_id, content=content, source_kind=SourceKind.MARKDOWN, source_path=f"/x/{doc_id}.md")


def _store(tmp_path: Path, **fps: str) -> FingerprintStore:
    store = FingerprintStore(path=tmp_path / "fp.json")
    for doc_id, content in fps.items():
        store.set(doc_id, compute_fingerprint(content))
    return store


def _ok(**kw: object) -> dict:
    return {"status": "processed", **kw}


def test_classification(tmp_path: Path) -> None:
    docs = [_doc("a", "same"), _doc("b", "edited"), _doc("c", "brand new"), _doc("d", "was failed"), _doc("e", "adopt me")]
    store = _store(tmp_path, a="same", b="old", d="was failed")
    status = {"a": _ok(), "b": _ok(), "d": {"status": "failed"}, "e": _ok(), "dup-1": {"status": "failed", "metadata": {"is_duplicate": True}}}
    plan = build_plan(docs, store, status)
    assert [d.doc_id for d in plan.new] == ["c"]
    assert [d.doc_id for d in plan.changed] == ["b"]
    assert [d.doc_id for d in plan.retry] == ["d"]
    assert [d.doc_id for d in plan.adopt] == ["e"]
    assert plan.unchanged == ["a"]
    assert plan.to_delete == ["b", "d"]
    assert not plan.is_noop()


def test_unchanged_index_is_a_noop(tmp_path: Path) -> None:
    docs = [_doc("a", "one"), _doc("b", "two")]
    plan = build_plan(docs, _store(tmp_path, a="one", b="two"), {"a": _ok(), "b": _ok(), "dup-x": {"status": "failed"}})
    assert plan.is_noop() and plan.unchanged == ["a", "b"] and plan.to_insert == [] and plan.to_delete == []


def test_gone_keep_and_unknown(tmp_path: Path) -> None:
    docs = [_doc(f"k{i}", "x") for i in range(4)]
    store = _store(tmp_path, k0="x", k1="x", k2="x", k3="x", gone1="g", kept1="k")
    status = {f"k{i}": _ok() for i in range(4)} | {"gone1": _ok(), "kept1": _ok(), "legacy": _ok()}
    plan = build_plan(docs, store, status, keep_ids={"kept1"})
    assert plan.gone == ["gone1"] and plan.kept == ["kept1"] and plan.unknown_processed == ["legacy"]
    assert build_plan(docs, store, status, keep_ids={"kept1"}, gc=False).gone == []


def test_partial_walk_refuses_gc(tmp_path: Path) -> None:
    ids = [f"d{i}" for i in range(10)]
    store = _store(tmp_path, **{i: "x" for i in ids})
    plan = build_plan([_doc("d0", "x")], store, {i: _ok() for i in ids})
    assert plan.gone == [] and "narrowed" in (plan.gc_skipped_reason or "")


def test_indexed_content_beats_sidecar(tmp_path: Path) -> None:
    pytest.importorskip("lightrag")
    # sidecar says "current", but the index stores older text (sidecar adopted without re-indexing)
    store = _store(tmp_path, a="new text\n", b="new text")
    status = {"a": _ok(), "b": _ok(), "c": _ok()}
    docs = [_doc("a", "new text\n"), _doc("b", "new text"), _doc("c", "adopted text")]
    plan = build_plan(docs, store, status, indexed={"a": "new text", "b": "OLD text", "c": "adopted text"})
    assert plan.unchanged == ["a"]  # trailing newline is stripped by LightRAG, still equal
    assert [d.doc_id for d in plan.changed] == ["b"]
    assert [d.doc_id for d in plan.adopt] == ["c"]


def test_duplicate_walked_ids_rejected(tmp_path: Path) -> None:
    with pytest.raises(PlanError):
        build_plan([_doc("a", "1"), _doc("a", "2")], _store(tmp_path), {})


def test_id_path_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "kb" / "x.md"
    f.parent.mkdir()
    f.write_text("x")
    plain = file_stable_id(f)
    monkeypatch.setenv(ID_PATH_MAP_ENV, f"{tmp_path.resolve()}=/logical/root")
    mapped = file_stable_id(f)
    assert mapped != plain
    # identical to a file that really lives at the logical path
    monkeypatch.delenv(ID_PATH_MAP_ENV)
    import hashlib

    assert mapped == "file:" + hashlib.sha256(b"/logical/root/kb/x.md").hexdigest()[:12]
    monkeypatch.setenv(ID_PATH_MAP_ENV, "bogus")
    with pytest.raises(ValueError):
        file_stable_id(f)


# ---------------------------------------------------------------------------
# Batched deletion: rebuild + flush are deferred and run once
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, status: str) -> None:
        self.status, self.message = status, ""


class _KV:
    def __init__(self, data: dict) -> None:
        self.data = data

    async def get_by_id(self, key: str):
        return self.data.get(key)


class _Graph:
    def __init__(self, nodes: set, edges: set) -> None:
        self.nodes, self.edges = nodes, edges

    async def get_node(self, name: str):
        return {"entity_id": name} if name in self.nodes else None

    async def has_edge(self, a: str, b: str) -> bool:
        return (a, b) in self.edges or (b, a) in self.edges


@pytest.mark.asyncio
async def test_delete_documents_defers_rebuild_and_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("lightrag")
    import lightrag.lightrag as lr

    from hars_memory.ingest.incremental import delete_documents

    rebuilds: list[dict] = []
    flushes: list[int] = []
    original_calls: list[int] = []

    async def real_rebuild(**kw):
        rebuilds.append(kw)

    async def real_flush(*_a, **_k):
        flushes.append(1)

    monkeypatch.setattr(lr, "rebuild_knowledge_from_chunks", real_rebuild)

    class FakeRag:
        _insert_done = staticmethod(real_flush)
        chunk_entity_relation_graph = _Graph({"kubectl", "Harbor"}, {("kubectl", "Harbor")})
        entity_chunks = _KV({"kubectl": {"chunk_ids": ["c1", "c2"]}, "Harbor": {"chunk_ids": ["c9"]}, "Gone": {"chunk_ids": ["c3"]}})
        relation_chunks = _KV({"Harbor<SEP>kubectl": {"chunk_ids": ["c1"]}})

        async def adelete_by_doc_id(self, doc_id: str):
            # what LightRAG does per document: ask for a rebuild and flush twice
            await lr.rebuild_knowledge_from_chunks(
                entities_to_rebuild={"kubectl": ["c1"], "Gone": ["c3"]},
                relationships_to_rebuild={("Harbor", "kubectl"): ["c1"]},
            )
            await self._insert_done()
            await self._insert_done()
            original_calls.append(1)
            return _Result("success" if doc_id != "missing" else "not_found")

    report = await delete_documents(FakeRag(), ["d1", "d2", "missing"])
    assert report.deleted == ["d1", "d2"] and report.not_found == ["missing"] and not report.failed
    assert len(original_calls) == 3
    assert len(rebuilds) == 1, "rebuild must run once for the union, not once per document"
    # entity "Gone" no longer has a graph node -> skipped; "kubectl" is rebuilt from its CURRENT chunk list
    assert rebuilds[0]["entities_to_rebuild"] == {"kubectl": ["c1", "c2"]}
    assert rebuilds[0]["relationships_to_rebuild"] == {("Harbor", "kubectl"): ["c1"]}
    assert len(flushes) == 2, "one flush after the deletions, one after the rebuild"
    assert lr.rebuild_knowledge_from_chunks is real_rebuild  # patch removed again
