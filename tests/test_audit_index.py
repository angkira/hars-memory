"""audit_index on a tiny on-disk index with a fake Qdrant client (no services needed)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import networkx as nx

from hars_memory.ingest.audit import audit_index

WS = "ws"


class FakeQdrant:
    """Just enough of QdrantClient for audit.qdrant_snapshot_view."""

    def __init__(self, collections: dict[str, list[dict[str, Any]]]) -> None:
        self._c = collections

    def collection_exists(self, name: str) -> bool:
        return name in self._c

    def scroll(self, collection_name: str, **_: Any):
        pts = [SimpleNamespace(id=i, payload=p) for i, p in enumerate(self._c[collection_name])]
        return pts, None


def _write(d: Path, name: str, data: Any) -> None:
    (d / name).write_text(json.dumps(data))


def _index(tmp_path: Path, edges: list[tuple[str, str]]) -> tuple[Path, FakeQdrant]:
    _write(tmp_path, "kv_store_full_docs.json", {"d1": {"content": "x"}})
    _write(tmp_path, "kv_store_doc_status.json", {"d1": {"status": "processed", "chunks_list": ["c1"]}})
    _write(tmp_path, "kv_store_text_chunks.json", {"c1": {"full_doc_id": "d1"}})
    _write(tmp_path, "doc_fingerprints.json", {"d1": "fp"})
    nodes = sorted({n for e in edges for n in e})
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    nx.write_graphml(g, tmp_path / "graph_chunk_entity_relation.graphml")
    _write(tmp_path, "kv_store_entity_chunks.json", {n: {"chunk_ids": ["c1"]} for n in nodes})
    _write(tmp_path, "kv_store_relation_chunks.json", {f"{a}<SEP>{b}": {"chunk_ids": ["c1"]} for a, b in edges})
    q = {
        "lightrag_vdb_chunks": [{"id": "c1", "full_doc_id": "d1"}],
        "lightrag_vdb_entities": [{"entity_name": n} for n in nodes],
        "lightrag_vdb_relationships": [{"src_id": a, "tgt_id": b} for a, b in edges],
    }
    return tmp_path, FakeQdrant(q)


def test_consistent_index_is_ok(tmp_path: Path) -> None:
    index_dir, client = _index(tmp_path, [("a", "b"), ("b", "c")])
    report = audit_index(index_dir, client, WS)
    assert report["ok"], report["failed"]


def test_missing_relationship_vector_fails(tmp_path: Path) -> None:
    index_dir, client = _index(tmp_path, [("a", "b"), ("b", "c")])
    client._c["lightrag_vdb_relationships"].pop()
    report = audit_index(index_dir, client, WS)
    assert not report["ok"]
    assert any("qdrant relationships" in name for name in report["failed"])


def test_md5_relationship_id_clash_is_tolerated(tmp_path: Path) -> None:
    # ("a/b/", "c.yaml") and ("a/", "b/c.yaml") concatenate to the same string, so LightRAG
    # gives both edges one point id and only one of them has a vector.
    edges = [("a/b/", "c.yaml"), ("a/", "b/c.yaml")]
    index_dir, client = _index(tmp_path, edges)
    client._c["lightrag_vdb_relationships"].pop()
    report = audit_index(index_dir, client, WS)
    assert report["ok"], report["failed"]
    assert len(report["relationship_id_collisions"]["unprefixed"]) == 1


def test_self_loop_edge_does_not_crash_the_audit(tmp_path: Path) -> None:
    index_dir, client = _index(tmp_path, [("a", "b"), ("c", "c")])
    client._c["lightrag_vdb_relationships"].pop(0)  # force the clash lookup over all edges
    report = audit_index(index_dir, client, WS)
    assert not report["ok"]  # a genuinely missing vector is still reported
    assert any("qdrant relationships" in name for name in report["failed"])
