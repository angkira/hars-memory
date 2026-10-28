"""Two LightRAG indexes in one process must not share in-memory storage.

LightRAG keys its process-global shared dicts by "<workspace>:<namespace>" and
defaults the workspace to "" for every instance, so before the fix every
project / session index of a server process bound to the first one's
full_docs / doc_status / graph data. `isolated_workspace` gives each
non-default index its own workspace while keeping its files in place.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

import hars_memory.server.lightrag_init as li


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.delenv("WORKSPACE", raising=False)
    monkeypatch.setenv("HARS_MEMORY_INDEX_DIR", str(tmp_path / "default-index"))
    monkeypatch.setenv("HARS_MEMORY_WORKSPACE_LINK_ROOT", str(tmp_path / "links"))


def test_default_index_keeps_the_legacy_empty_workspace(tmp_path):
    d = str(tmp_path / "default-index")
    assert li.isolated_workspace(d) == (d, "")


def test_other_indexes_get_unique_workspaces_and_files_stay_in_place(tmp_path):
    dirs = [tmp_path / "p1" / "index", tmp_path / "p2" / "index", tmp_path / "sessions" / "s1"]
    got = [li.isolated_workspace(str(d)) for d in dirs]
    assert len({ws for _, ws in got}) == 3                       # same basename "index" does not collide
    for d, (wdir, ws) in zip(dirs, got):
        assert ws and "/" not in ws
        assert os.path.realpath(os.path.join(wdir, ws)) == os.path.realpath(d)
    assert li.isolated_workspace(str(dirs[0])) == got[0]        # stable for the same directory


def test_operator_workspace_env_is_respected(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE", "ops")
    d = str(tmp_path / "x")
    assert li.isolated_workspace(d) == (d, "ops")


def test_rag_index_dir_resolves_workspace_and_ignores_non_str():
    class R:
        working_dir, workspace = "/a", "b"

    class M:
        working_dir, workspace = "/a", object()

    assert li.rag_index_dir(R()) == os.path.realpath(os.path.join("/a", "b"))
    assert li.rag_index_dir(M()) == "/a"
    assert li.rag_index_dir(object()) is None


def test_two_kv_stores_in_one_process_are_isolated_on_disk_and_in_memory(tmp_path):
    """The real LightRAG storage: with the workspaces from isolated_workspace two indexes
    keep separate data; with the legacy "" workspace for both they share it."""
    pytest.importorskip("lightrag")
    from lightrag.kg.json_kv_impl import JsonKVStorage
    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

    async def run(isolate: bool) -> tuple[dict, dict]:
        initialize_share_data()
        # the same directory NAME in two places: the case every auto-discovered project (<id>/index) hits
        a = tmp_path / ("iso" if isolate else "leg") / "p1" / "index"
        b = tmp_path / ("iso" if isolate else "leg") / "p2" / "index"
        stores = []
        for d in (a, b):
            d.mkdir(parents=True)
            wdir, ws = li.isolated_workspace(str(d)) if isolate else (str(d), "")
            s = JsonKVStorage(namespace="full_docs", workspace=ws, global_config={"working_dir": wdir},
                              embedding_func=None)
            await s.initialize()
            stores.append(s)
        await stores[0].upsert({"doc-a": {"content": "A"}})
        await stores[1].upsert({"doc-b": {"content": "B"}})
        for s in stores:
            await s.index_done_callback()
        out = tuple(json.loads((d / "kv_store_full_docs.json").read_text()) if (d / "kv_store_full_docs.json").exists()
                    else None for d in (a, b))
        finalize_share_data()
        return out

    fa, fb = asyncio.run(run(isolate=True))
    assert set(fa) == {"doc-a"} and set(fb) == {"doc-b"}

    la, lb = asyncio.run(run(isolate=False))
    # legacy: both stores share one dict, so neither file holds exactly its own document
    # (observed: index a gets both documents, index b's file is never written)
    assert (la is None or set(la) != {"doc-a"}) or (lb is None or set(lb) != {"doc-b"}), (
        "legacy layout was expected to leak between indexes (the bug this guards against)"
    )
