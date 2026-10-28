"""A neighbour (non-default) project on the SAME server must be written and read like the global memory.

Regression tests for the neurocompact eval's project layout (one isolated project per eval session, written with
memory_upsert_document, read with memory_recall(project=...)): the Qdrant workspace of a project must be its own,
the process-global default BM25 / graph caches must never answer for another project, and a write to a neighbour
must not evict the default project's caches.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from hars_memory.projects.models import ProjectMetadata
from hars_memory.projects.registry import ProjectRegistry, qdrant_workspace_for
from tests.test_mcp_server import _load_mcp_module_with_env


def _registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *pids: str) -> ProjectRegistry:
    projects = tmp_path / "projects"
    for pid in pids:
        (projects / pid).mkdir(parents=True)
    monkeypatch.setenv("HARS_MEMORY_PROJECTS_DIR", str(projects))
    monkeypatch.delenv("HARS_MEMORY_PROJECTS_CONFIG", raising=False)
    monkeypatch.setenv("HARS_MEMORY_QDRANT_COLLECTION", "base_coll")
    return ProjectRegistry()


def test_qdrant_workspace_for_default_discovered_and_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reg = _registry(tmp_path, monkeypatch, "nc-run1-s1")
    assert reg.get_project("nc-run1-s1").index_dir == (tmp_path / "projects" / "nc-run1-s1" / "index").resolve()
    assert qdrant_workspace_for(reg.get_project("default")) == "base_coll"
    assert qdrant_workspace_for(reg.get_project("nc-run1-s1")) == "base_coll_nc-run1-s1"
    explicit = ProjectMetadata(project_id="x", name="x", index_dir=tmp_path, qdrant_collection="mine")
    assert qdrant_workspace_for(explicit) == "mine"


def test_memory_status_reports_the_projects_own_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reg = _registry(tmp_path, monkeypatch, "nc-run1-s1")
    mod: Any = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path / "main")})
    monkeypatch.setattr(mod, "HARS_MEMORY_VECTOR_STORAGE", "QdrantVectorDBStorage")
    monkeypatch.setattr(mod, "HARS_MEMORY_QDRANT_COLLECTION_PREFIX", "")
    status = mod._index_status(reg.get_project("nc-run1-s1"))
    assert status["storage"]["vector"]["workspace"] == "base_coll_nc-run1-s1"


def test_default_bm25_and_graph_caches_never_answer_for_another_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod: Any = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path / "main")})
    default_bm25, default_graph = object(), object()
    project_bm25, project_graph = object(), object()

    class _Reg:
        def __init__(self) -> None:
            self.invalidated: list[str] = []

        async def get_bm25_index(self, pid: str) -> Any:
            assert pid == "p1"
            return project_bm25, "rebuild"

        async def get_graph(self, pid: str) -> Any:
            assert pid == "p1"
            return project_graph, "rebuild"

        def invalidate_caches(self, pid: str) -> None:
            self.invalidated.append(pid)

    reg = _Reg()
    monkeypatch.setattr(mod, "_bm25_cache", default_bm25)
    monkeypatch.setattr(mod, "_graph_cache", default_graph)
    monkeypatch.setattr(mod, "get_default_project_registry", lambda: reg)
    assert asyncio.run(mod._get_bm25_index("p1")) == (project_bm25, "rebuild")
    assert asyncio.run(mod._get_graph("p1")) == (project_graph, "rebuild")
    assert asyncio.run(mod._get_bm25_index("default")) == (default_bm25, "hit")
    assert asyncio.run(mod._get_graph("default")) == (default_graph, "hit")

    # a write to the neighbour invalidates the neighbour only
    mod._invalidate_bm25_cache("p1")
    mod._clear_in_memory_graph_cache("p1")
    assert mod._bm25_cache is default_bm25 and mod._graph_cache is default_graph
    assert reg.invalidated == ["p1", "p1"]
    mod._invalidate_bm25_cache("default")
    assert mod._bm25_cache is None


def test_explicit_collection_wins_over_a_workspace_left_by_an_earlier_instance() -> None:
    """create_lightrag: the QDRANT_WORKSPACE rule (source-level pin; the factory itself needs lightrag + qdrant)."""
    src = (Path(__file__).resolve().parents[1] / "hars_memory" / "server" / "lightrag_init.py").read_text()
    assert 'if qdrant_collection or not os.environ.get("QDRANT_WORKSPACE"):' in src


# --- memory_upsert_document records the fingerprint only after real processing ---------------
def test_doc_processing_state_reads_lightrag_doc_status():
    import asyncio

    from hars_memory import mcp_server as ms

    class Store:
        def __init__(self, row):
            self.row = row

        async def get_by_id(self, _id):
            return self.row

    class Rag:
        def __init__(self, row):
            self.doc_status = Store(row)

    run = asyncio.run
    assert run(ms._doc_processing_state(Rag({"status": "processed"}), "d")) == "processed"
    assert run(ms._doc_processing_state(Rag({"status": "pending"}), "d")) == "pending"
    assert run(ms._doc_processing_state(Rag(None), "d")) == "missing"
    assert run(ms._doc_processing_state(Rag("not-a-dict"), "d")) is None
    assert run(ms._doc_processing_state(object(), "d")) is None
