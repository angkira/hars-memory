"""Cosmos export endpoints (/stats /graph /coords) — hermetic tests.

A tiny synthetic index (graphml + kv stores + minimal vdb matrices) is written
into tmp_path and served through the real trace HTTP listener on 127.0.0.1:0.
Nothing touches the live index or the network. The coords tests force the
"umap unavailable" branch via ``sys.modules`` so they are deterministic even on
a machine that has umap installed.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

import hars_memory.trace as trace
from hars_memory import graph_export

NS = "{http://graphml.graphdrawing.org/xmlns}"
GRAPHML_URI = "http://graphml.graphdrawing.org/xmlns"  # literal URI for the XML below


# ---------------------------------------------------------------------------
# synthetic index
# ---------------------------------------------------------------------------

GRAPHML = f"""<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="{GRAPHML_URI}" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <key id="d0" for="node" attr.name="entity_id" attr.type="string" />
  <key id="d1" for="node" attr.name="entity_type" attr.type="string" />
  <key id="d2" for="node" attr.name="description" attr.type="string" />
  <graph edgedefault="undirected">
    <node id="hub"> <data key="d0">hub</data><data key="d1">concept</data>
      <data key="d2">{'x' * 400}</data></node>
    <node id="a"> <data key="d0">a</data><data key="d1">tool</data><data key="d2">alpha</data></node>
    <node id="b"> <data key="d0">b</data><data key="d1">tool</data><data key="d2">beta</data></node>
    <node id="c"> <data key="d0">c</data><data key="d1">doc</data><data key="d2">gamma</data></node>
    <node id="lonely"><data key="d0">lonely</data><data key="d1">doc</data></node>
    <edge source="hub" target="a" />
    <edge source="hub" target="b" />
    <edge source="a" target="b" />
    <edge source="hub" target="c" />
    <edge source="a" target="ghost" />
  </graph>
</graphml>
"""


def _write_matrix(path: Path, n: int, dim: int, ids: list[str], name_key: str | None = None) -> None:
    import struct

    # Same shape as the real vdb files: one base64 float32 blob for the whole
    # matrix + per-row records in "data".
    blob = b"".join(struct.pack("<f", float((i + 1) % 7) / 7.0) for i in range(n * dim))
    data = []
    for i in range(n):
        rec: dict[str, Any] = {"__id__": ids[i]}
        if name_key:
            rec[name_key] = ids[i]
        data.append(rec)
    payload = {"data": data, "matrix": base64.b64encode(blob).decode(), "embedding_dim": dim}
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def index_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    idx = tmp_path / "idx_demo"
    idx.mkdir()
    (idx / "graph_chunk_entity_relation.graphml").write_text(GRAPHML, encoding="utf-8")
    (idx / "kv_store_full_docs.json").write_text(json.dumps({"d1": {}, "d2": {}}), encoding="utf-8")
    (idx / "kv_store_text_chunks.json").write_text(json.dumps({f"c{i}": {} for i in range(7)}), encoding="utf-8")
    (idx / "kv_store_doc_status.json").write_text(json.dumps({"d1": {"status": "processed"}}), encoding="utf-8")
    _write_matrix(idx / "vdb_entities.json", 5, 8, ["hub", "a", "b", "c", "lonely"], name_key="entity_name")
    _write_matrix(idx / "vdb_chunks.json", 3, 8, ["chunk-1", "chunk-2", "chunk-3"])
    monkeypatch.setenv("HARS_MEMORY_INDEX_DIR", str(idx))
    # isolation: the in-memory graph/stats caches must not leak between tests
    monkeypatch.setattr(graph_export, "_graph_cache", None)
    monkeypatch.setattr(graph_export, "_stats_cache", None)
    return idx


def _server():
    return trace.start_trace_server("127.0.0.1", 0, keepalive_s=1.0)


async def _get(url: str, **kw: Any) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30) as client:
        return await client.get(url, **kw)


# ---------------------------------------------------------------------------
# /stats
# ---------------------------------------------------------------------------

class TestStats:
    async def test_identity_and_counts(self, index_dir: Path) -> None:
        server = await _server()
        port = server.sockets[0].getsockname()[1]
        try:
            r = await _get(f"http://127.0.0.1:{port}/stats")
        finally:
            server.close()
            await server.wait_closed()
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        # identity = basename of HARS_MEMORY_INDEX_DIR (the contract the UI badge reads)
        assert body["index_id"] == index_dir.name
        assert body["docs"] == 2
        assert body["chunks"] == 7
        assert body["nodes"] == 5 and body["edges"] == 4  # dangling ghost edge dropped
        assert body["entities"] == 5 and body["relations"] == 4
        assert body["last_ingest"]

    async def test_no_index_dir_is_503_with_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HARS_MEMORY_INDEX_DIR", raising=False)
        server = await _server()
        port = server.sockets[0].getsockname()[1]
        try:
            r = await _get(f"http://127.0.0.1:{port}/stats")
        finally:
            server.close()
            await server.wait_closed()
        assert r.status_code == 503
        body = r.json()
        assert body["ok"] is False and body["error"] == "index_dir_unconfigured"
        assert "HARS_MEMORY_INDEX_DIR" in body["hint"]


# ---------------------------------------------------------------------------
# /graph
# ---------------------------------------------------------------------------

class TestGraph:
    async def test_shape_and_dangling_edge_dropped(self, index_dir: Path) -> None:
        server = await _server()
        port = server.sockets[0].getsockname()[1]
        try:
            r = await _get(f"http://127.0.0.1:{port}/graph")
        finally:
            server.close()
            await server.wait_closed()
        assert r.status_code == 200
        body = r.json()
        meta = body["meta"]
        assert meta["index_id"] == index_dir.name
        assert meta["full_nodes"] == 5 and meta["full_edges"] == 4
        assert meta["sampled"] is False
        assert meta["docs"] == 2
        assert len(body["nodes"]) == 5 and len(body["edges"]) == 4
        node = body["nodes"][0]
        for key in ("id", "label", "type", "degree", "desc_head"):
            assert key in node
        # desc_head is truncated to DESC_HEAD chars
        hub = next(n for n in body["nodes"] if n["id"] == "hub")
        assert len(hub["desc_head"]) == graph_export.DESC_HEAD
        # edges are index pairs into the returned array
        max_idx = len(body["nodes"]) - 1
        assert all(isinstance(s, int) and isinstance(t, int) and 0 <= s <= max_idx and 0 <= t <= max_idx
                   for s, t in body["edges"])

    async def test_degree_sampling_and_edge_remap(self, index_dir: Path) -> None:
        server = await _server()
        port = server.sockets[0].getsockname()[1]
        try:
            r = await _get(f"http://127.0.0.1:{port}/graph", params={"limit_nodes": "3", "limit_edges": "10"})
        finally:
            server.close()
            await server.wait_closed()
        body = r.json()
        assert body["meta"]["sampled"] is True
        assert body["meta"]["nodes"] == 3
        # top-3 by degree: hub(3), a(2), b(2); 'c' and 'lonely' dropped
        assert {n["id"] for n in body["nodes"]} == {"hub", "a", "b"}
        # edge hub-c did not survive; hub-a, hub-b, a-b did
        assert len(body["edges"]) == 3
        names = [n["id"] for n in body["nodes"]]
        for s, t in body["edges"]:
            assert names[s] in {"hub", "a", "b"} and names[t] in {"hub", "a", "b"}

    async def test_edge_cap_applies(self, index_dir: Path) -> None:
        server = await _server()
        port = server.sockets[0].getsockname()[1]
        try:
            r = await _get(f"http://127.0.0.1:{port}/graph", params={"limit_nodes": "5", "limit_edges": "2"})
        finally:
            server.close()
            await server.wait_closed()
        assert len(r.json()["edges"]) == 2

    async def test_garbage_limits_fall_back_to_defaults(self, index_dir: Path) -> None:
        server = await _server()
        port = server.sockets[0].getsockname()[1]
        try:
            r = await _get(f"http://127.0.0.1:{port}/graph", params={"limit_nodes": "abc", "limit_edges": "-5"})
        finally:
            server.close()
            await server.wait_closed()
        meta = r.json()["meta"]
        assert meta["node_limit"] == graph_export.DEFAULT_GRAPH_NODES
        assert meta["edge_limit"] == graph_export.DEFAULT_GRAPH_EDGES


# ---------------------------------------------------------------------------
# /coords — cached-file contract (compute happens offline, not in the service)
# ---------------------------------------------------------------------------

def _write_fresh_cache(idx: Path, dim: int = 3) -> None:
    cache = {
        "meta": {
            "index_id": idx.name,
            "dim": dim,
            "source_mtimes": graph_export._coords_fingerprint(idx),
            "generated_at": "2026-10-18T00:00:00Z",
        },
        "entities": {
            "ids": ["hub", "a", "b", "c", "lonely"],
            "xy": [[0.0, 0.0], [0.5, 0.1], [0.2, 0.9], [0.9, 0.4], [0.1, 0.2]],
            "xyz": [[0.0, 0.0, 0.1], [0.5, 0.1, 0.2], [0.2, 0.9, 0.3], [0.9, 0.4, 0.4], [0.1, 0.2, 0.5]],
        },
        "chunks": {"chunk-1": [0.1, 0.1, 0.1]},
    }
    (idx / graph_export.COORDS_CACHE_NAME).write_text(json.dumps(cache), encoding="utf-8")


class TestCoords:
    async def test_serves_fresh_cache(self, index_dir: Path) -> None:
        _write_fresh_cache(index_dir)
        server = await _server()
        port = server.sockets[0].getsockname()[1]
        try:
            r = await _get(f"http://127.0.0.1:{port}/coords")
        finally:
            server.close()
            await server.wait_closed()
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["meta"]["index_id"] == index_dir.name
        assert body["entities"]["ids"] == ["hub", "a", "b", "c", "lonely"]
        assert len(body["entities"]["xyz"]) == 5
        assert body["chunks"]["chunk-1"] == [0.1, 0.1, 0.1]

    async def test_dim2_omits_xyz(self, index_dir: Path) -> None:
        _write_fresh_cache(index_dir, dim=3)
        server = await _server()
        port = server.sockets[0].getsockname()[1]
        try:
            r = await _get(f"http://127.0.0.1:{port}/coords", params={"dim": "2"})
        finally:
            server.close()
            await server.wait_closed()
        body = r.json()
        assert "xyz" not in body["entities"]
        assert len(body["entities"]["xy"]) == 5

    async def test_missing_cache_without_umap_serves_fallback(
        self, index_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Deterministically force the "umap not importable" branch (this venv
        # may or may not have umap; the service venv does not).
        monkeypatch.setitem(sys.modules, "umap", None)
        server = await _server()
        port = server.sockets[0].getsockname()[1]
        try:
            r = await _get(f"http://127.0.0.1:{port}/coords")
        finally:
            server.close()
            await server.wait_closed()
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert len(body["entities"]["ids"]) == 5
        assert len(body["entities"]["xyz"]) == 5

    async def test_stale_cache_is_served(self, index_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_fresh_cache(index_dir)
        # bump a vdb mtime past the cached fingerprint -> must still serve cached projection
        monkeypatch.setitem(sys.modules, "umap", None)
        import os as _os

        st = (index_dir / "vdb_entities.json").stat()
        _os.utime(index_dir / "vdb_entities.json", (st.st_atime + 10, st.st_mtime + 10))
        server = await _server()
        port = server.sockets[0].getsockname()[1]
        try:
            r = await _get(f"http://127.0.0.1:{port}/coords")
        finally:
            server.close()
            await server.wait_closed()
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert r.json()["entities"]["ids"] == ["hub", "a", "b", "c", "lonely"]

    async def test_head_coords_returns_200(self, index_dir: Path) -> None:
        _write_fresh_cache(index_dir)
        server = await _server()
        port = server.sockets[0].getsockname()[1]
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.head(f"http://127.0.0.1:{port}/coords?dim=3")
        finally:
            server.close()
            await server.wait_closed()
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/json; charset=utf-8"
        assert int(r.headers["content-length"]) > 0
        assert r.content == b""

    async def test_graph_stats_prefix_handled(self, index_dir: Path) -> None:
        _write_fresh_cache(index_dir)
        server = await _server()
        port = server.sockets[0].getsockname()[1]
        try:
            r = await _get(f"http://127.0.0.1:{port}/graph-stats/coords?dim=3")
        finally:
            server.close()
            await server.wait_closed()
        assert r.status_code == 200
        assert r.json()["ok"] is True

    async def test_post_is_405(self, index_dir: Path) -> None:
        server = await _server()
        port = server.sockets[0].getsockname()[1]
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(f"http://127.0.0.1:{port}/graph")
        finally:
            server.close()
            await server.wait_closed()
        assert r.status_code == 405


# ---------------------------------------------------------------------------
# inline compute path (runs only when umap IS importable in this venv)
# ---------------------------------------------------------------------------

class TestInlineCompute:
    async def test_compute_and_cache_written(self, index_dir: Path) -> None:
        pytest.importorskip("umap", reason="inline UMAP path needs umap-learn installed")
        import numpy  # noqa: F401

        server = await _server()
        port = server.sockets[0].getsockname()[1]
        try:
            r = await _get(f"http://127.0.0.1:{port}/coords")
        finally:
            server.close()
            await server.wait_closed()
        assert r.status_code == 200
        body = r.json()
        assert body["meta"]["index_id"] == index_dir.name
        assert len(body["entities"]["ids"]) == 5
        assert (index_dir / graph_export.COORDS_CACHE_NAME).is_file()
