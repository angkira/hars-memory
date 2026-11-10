"""Extractor LLM usage per insert (`extract_usage` of memory_upsert_document, `track_extract_usage`)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from hars_memory.server import lightrag_init as li
from tests.test_llm_reasoning_content_fallback import _client_factory_using_transport, _make_llm_func_for_test
from tests.test_mcp_server import _load_mcp_module_with_env


def _patch_client_factory(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    """Hand make_llm_func a client on a mock transport WITHOUT importing the real lightrag.llm.openai (an earlier test of
    the full suite may have blocked `tiktoken`, which that module imports)."""
    import sys
    import types

    stub = types.ModuleType("lightrag.llm.openai")
    stub.create_openai_async_client = _client_factory_using_transport(httpx.MockTransport(handler))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lightrag.llm.openai", stub)


def _reply(usage: dict | None, content: str = "ok answer") -> httpx.Response:
    body: dict[str, Any] = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def test_usage_comes_from_the_provider_and_is_scoped_to_the_block(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client_factory(monkeypatch, lambda req: _reply({"prompt_tokens": 100, "completion_tokens": 7}))
    llm = _make_llm_func_for_test()
    g0 = dict(li.LLM_USAGE)

    async def go() -> dict:
        await llm("outside before")
        with li.track_extract_usage() as acc:
            await llm("p1")
            await llm("p2", system_prompt="s")
        await llm("outside after")
        return acc.to_dict()

    u = asyncio.run(go())
    assert (u["calls"], u["prompt_tokens"], u["completion_tokens"], u["total_tokens"]) == (2, 200, 14, 214)
    assert u["estimated"] is False and u["calls_estimated"] == 0
    assert li.LLM_USAGE["calls"] - g0["calls"] == 4            # the process-global counter still sees everything
    assert li.LLM_USAGE["prompt_tokens"] - g0["prompt_tokens"] == 400


def test_missing_usage_is_estimated_and_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client_factory(monkeypatch, lambda req: _reply(None, content="x" * 360))
    llm = _make_llm_func_for_test()

    async def go() -> dict:
        with li.track_extract_usage() as acc:
            await llm("y" * 720)
        return acc.to_dict()

    u = asyncio.run(go())
    assert u["estimated"] is True and u["calls_estimated"] == 1 and u["calls"] == 1
    assert u["prompt_tokens"] == round(720 / li.ESTIMATE_CHARS_PER_TOKEN) == u["estimated_prompt_tokens"]
    assert u["completion_tokens"] == round(360 / li.ESTIMATE_CHARS_PER_TOKEN)


def test_concurrent_blocks_do_not_mix(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        n = len(json.loads(req.content)["messages"][-1]["content"])
        return _reply({"prompt_tokens": n, "completion_tokens": 1})

    _patch_client_factory(monkeypatch, handler)
    llm = _make_llm_func_for_test()

    async def one(tag: str, n_calls: int, size: int) -> dict:
        with li.track_extract_usage() as acc:
            async def call(i: int) -> None:
                await asyncio.sleep(0.001 * (i % 3))
                await llm(tag * size)
            await asyncio.gather(*(call(i) for i in range(n_calls)))
        return acc.to_dict()

    async def go() -> list[dict]:
        return await asyncio.gather(one("a", 7, 10), one("b", 3, 1000), one("c", 12, 1))

    a, b, c = asyncio.run(go())
    assert (a["calls"], a["prompt_tokens"]) == (7, 70)
    assert (b["calls"], b["prompt_tokens"]) == (3, 3000)
    assert (c["calls"], c["prompt_tokens"]) == (12, 12)


def test_upsert_response_carries_extract_usage(tmp_path: Path) -> None:
    index_dir, staging = tmp_path / "index", tmp_path / "staging"
    doc = tmp_path / "doc.md"
    doc.write_text("# D\nfresh knowledge", encoding="utf-8")
    mod: Any = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(index_dir), "HARS_MEMORY_STAGING_DIR": str(staging)})

    async def fake_insert(rag: object, docs: list, batch_size: int = 1) -> None:
        # what the extractor does inside the insert: three completions through make_llm_func's recorder
        class U:
            prompt_tokens, completion_tokens = 1000, 50
        for _ in range(3):
            li._record_extract_usage(U(), [{"role": "user", "content": "x"}], "out")
        li._record_extract_usage(None, [{"role": "user", "content": "z" * 36}], "o" * 18)   # one without usage

    rag = AsyncMock()
    with patch.object(mod, "_get_rag", new=AsyncMock(return_value=rag)), \
         patch("hars_memory.server.index._insert_all_batches", new=fake_insert), \
         patch.object(mod, "_invalidate_bm25_cache"), patch.object(mod, "_clear_in_memory_graph_cache"):
        data = json.loads(asyncio.run(mod.call_tool("memory_upsert_document", {"file_path": str(doc)}))[0].text)
    assert data["ok"] is True and data["action"] == "inserted"
    eu = data["extract_usage"]
    assert eu["calls"] == 4 and eu["calls_estimated"] == 1 and eu["estimated"] is True
    assert eu["prompt_tokens"] == 3000 + 10 and eu["completion_tokens"] == 150 + 5
    assert eu["scope"] == "this_call" and eu["process_delta"]["calls"] == 4
