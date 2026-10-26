"""LightRAG 1.5.7 role-LLM routing: the `query` role must actually resolve to
the query model after memory_recall's synthesis call.

Regression guard for a silent no-op found by the 1.5.7 migration: after init,
`rag.llm_model_func = ...` does nothing — LightRAG builds per-role wrappers once
and reads `global_config["role_llm_funcs"]` on every query. The synthesis path
therefore kept using the EXTRACTOR model with no error and no test noticing.
The supported runtime lever is
`await rag.aupdate_llm_role_config("query", model_func=...)`.

These tests use a LightRAG-1.5.7-shaped stub (a `role_llm_funcs` mapping plus
`_role_llm_states[role].raw_func`), so they assert on the RESOLVED function the
query would actually call, not on the assignment.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace


class _Marker:
    """A callable with a stable, identifiable repr (stands in for an LLM func)."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __call__(self) -> str:  # pragma: no cover - never invoked here
        return self.tag

    def __repr__(self) -> str:
        return self.tag


class _RoleAwareRag:
    """Minimal LightRAG-1.5.7-shaped rag exposing the role runtime API.

    `role_llm_funcs` is what `_build_global_config()` would hand to the query
    path; `_role_llm_states[role].raw_func` is what `aupdate_llm_role_config`
    rewrites. Recording what was resolved *inside* `aquery_llm` is the point:
    an after-init `llm_model_func` assignment would leave these untouched.
    """

    def __init__(self) -> None:
        self.orig_query_func = _Marker("ORIG_QUERY")
        self.extract_func = _Marker("EXTRACT")
        self._role_llm_states = {
            "query": SimpleNamespace(raw_func=self.orig_query_func),
            "extract": SimpleNamespace(raw_func=self.extract_func),
        }
        self.role_llm_funcs: dict[str, object] = {
            "query": self.orig_query_func,
            "extract": self.extract_func,
        }
        self.observed_query_funcs: list[object] = []
        self.observed_extract_funcs: list[object] = []

    async def aupdate_llm_role_config(self, role: str, *, model_func: object = None, **_kw: object) -> None:
        if model_func is not None:
            self._role_llm_states[role].raw_func = model_func
            self.role_llm_funcs[role] = model_func

    async def aquery_llm(self, question: str, param: object) -> dict:
        await asyncio.sleep(0)
        self.observed_query_funcs.append(self.role_llm_funcs["query"])
        self.observed_extract_funcs.append(self.role_llm_funcs["extract"])
        await asyncio.sleep(0)
        self.observed_query_funcs.append(self.role_llm_funcs["query"])
        return {
            "status": "success",
            "llm_response": {"content": f"answer for {question}"},
            "data": {"chunks": [], "references": []},
        }


def _load_module(tmp_path: Path) -> object:
    import importlib
    import os

    import hars_memory.mcp_server as mod

    os.environ["HARS_MEMORY_INDEX_DIR"] = str(tmp_path / "index")
    importlib.reload(mod)
    return mod


def _stub_recall_dependencies(mod: object, rag: _RoleAwareRag, monkeypatch) -> None:
    mod._rag_instance = rag  # type: ignore[attr-defined]

    counter = {"n": 0}

    def _fake_create_query_model_func() -> _Marker:
        counter["n"] += 1
        return _Marker(f"QUERY_FUNC_{counter['n']}")

    import hars_memory.server.lightrag_init as lightrag_init

    monkeypatch.setattr(lightrag_init, "create_query_model_func", _fake_create_query_model_func)

    async def _fake_compute_hybrid_block(
        rag_arg: object,
        question: str,
        top_k: int,
        ll_keywords: list[str] | None = None,
        *,
        fetch_top_k: int | None = None,
    ) -> dict:
        return {"enabled": True, "fused_chunks": [], "confidence": {"low_confidence": True}}

    mod._compute_hybrid_block = _fake_compute_hybrid_block  # type: ignore[attr-defined]


def _run_recall(mod: object) -> dict:
    async def _main() -> dict:
        results = await mod.call_tool(  # type: ignore[attr-defined]
            "memory_recall",
            {"question": "q", "ll_keywords": ["kw"], "context_only": False},
        )
        return json.loads(results[0].text)

    return asyncio.run(_main())


class TestRoleLlmResolution:
    def test_resolved_query_role_changes_and_is_restored(self, tmp_path: Path, monkeypatch) -> None:
        mod = _load_module(tmp_path)
        rag = _RoleAwareRag()
        _stub_recall_dependencies(mod, rag, monkeypatch)

        payload = _run_recall(mod)

        assert payload["ok"]
        # The function the query path would ACTUALLY call changed to the query
        # model — this is what the old attribute swap failed to do.
        assert rag.observed_query_funcs, "aquery_llm never ran"
        assert all(str(f).startswith("QUERY_FUNC_") for f in rag.observed_query_funcs), (
            f"query role still resolved to the extractor func: {rag.observed_query_funcs}"
        )
        # Extraction is a different role and must be unaffected.
        assert all(f is rag.extract_func for f in rag.observed_extract_funcs)
        # The original query func is put back after the call.
        assert rag.role_llm_funcs["query"] is rag.orig_query_func
        assert rag._role_llm_states["query"].raw_func is rag.orig_query_func


class TestAsetQueryRoleLlmFunc:
    def test_role_api_returns_true_and_changes_resolved_func(self, tmp_path: Path) -> None:
        mod = _load_module(tmp_path)
        rag = _RoleAwareRag()
        new_func = _Marker("NEW")

        used = asyncio.run(mod._aset_query_role_llm_func(rag, new_func))

        assert used is True
        assert rag.role_llm_funcs["query"] is new_func
        assert mod._query_role_raw_llm_func(rag) is new_func

    def test_missing_role_api_falls_back(self, tmp_path: Path) -> None:
        mod = _load_module(tmp_path)

        class _LegacyRag:
            llm_model_func = _Marker("EXTRACT")

        assert asyncio.run(mod._aset_query_role_llm_func(_LegacyRag(), _Marker("Q"))) is False
        assert mod._query_role_raw_llm_func(_LegacyRag()) is None
