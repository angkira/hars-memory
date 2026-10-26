"""`_QUERY_MODEL_LOCK`: concurrent memory_recall calls must not swap the
shared `rag.llm_model_func` out from under each other.

LightRAG 1.5.6 has no QueryParam.model_func, so memory_recall temporarily
installs the query model on the shared instance attribute. Without a lock,
two concurrent recalls interleave: the first one's `finally` restores the
ORIGINAL extractor func while the second is still mid-query (so the second
silently synthesises with the wrong model), and the loser of the race can
leave the query model permanently installed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

_CONCURRENT_RECALLS = 10


def _load_module(tmp_path: Path) -> object:
    import importlib
    import os

    import hars_memory.mcp_server as mod

    os.environ["HARS_MEMORY_INDEX_DIR"] = str(tmp_path / "index")
    importlib.reload(mod)
    return mod


class _FakeRag:
    """Records what `llm_model_func` was installed while each query ran, plus
    the peak number of queries running inside the swap window at once."""

    def __init__(self) -> None:
        self.llm_model_func = "ORIGINAL_EXTRACTOR_FUNC"
        self.observed_funcs: list[object] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def aquery_llm(self, question: str, param: object) -> dict:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            # Yield control: with no lock this is exactly where another recall
            # would enter and overwrite/restore llm_model_func.
            await asyncio.sleep(0)
            self.observed_funcs.append(self.llm_model_func)
            await asyncio.sleep(0)
            self.observed_funcs.append(self.llm_model_func)
            return {
                "status": "success",
                "llm_response": {"content": f"answer for {question}"},
                "data": {"chunks": [], "references": []},
            }
        finally:
            self.in_flight -= 1


def _stub_module(mod: object, rag: _FakeRag, monkeypatch: object) -> None:
    mod._rag_instance = rag  # type: ignore[attr-defined]

    counter = {"n": 0}

    def _fake_create_query_model_func() -> str:
        counter["n"] += 1
        return f"QUERY_FUNC_{counter['n']}"

    # memory_recall imports this lazily from server.lightrag_init inside the
    # call, so the patch has to land on the SOURCE module, not on mcp_server.
    import hars_memory.server.lightrag_init as lightrag_init

    monkeypatch.setattr(  # type: ignore[attr-defined]
        lightrag_init, "create_query_model_func", _fake_create_query_model_func
    )

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


def _run_concurrent_recalls(mod: object, count: int) -> list[dict]:
    async def _main() -> list[dict]:
        results = await asyncio.gather(*[
            mod.call_tool("memory_recall", {
                "question": f"q{i}",
                "ll_keywords": [f"kw{i}"],
                "context_only": False,
            })
            for i in range(count)
        ])
        return [json.loads(r[0].text) for r in results]

    return asyncio.run(_main())


class TestQueryModelLock:
    def test_concurrent_recalls_never_share_the_swap_window(self, tmp_path: Path, monkeypatch) -> None:
        mod = _load_module(tmp_path)
        rag = _FakeRag()
        _stub_module(mod, rag, monkeypatch)

        payloads = _run_concurrent_recalls(mod, _CONCURRENT_RECALLS)

        assert len(payloads) == _CONCURRENT_RECALLS
        assert all(p["ok"] for p in payloads)
        assert rag.max_in_flight == 1, (
            "the llm_model_func swap window must be serialised — "
            f"{rag.max_in_flight} queries were inside it at once"
        )

    def test_each_query_sees_a_query_model_not_the_extractor(self, tmp_path: Path, monkeypatch) -> None:
        mod = _load_module(tmp_path)
        rag = _FakeRag()
        _stub_module(mod, rag, monkeypatch)

        _run_concurrent_recalls(mod, _CONCURRENT_RECALLS)

        assert rag.observed_funcs, "no observations recorded"
        assert all(str(func).startswith("QUERY_FUNC_") for func in rag.observed_funcs), (
            f"a query ran with the extractor func still installed: {rag.observed_funcs}"
        )
        # Both observations of one query must be the SAME func object: nobody
        # replaced it half-way through.
        pairs = list(zip(rag.observed_funcs[0::2], rag.observed_funcs[1::2]))
        assert all(first == second for first, second in pairs), (
            f"llm_model_func changed mid-query: {pairs}"
        )

    def test_original_func_is_restored_after_all_recalls(self, tmp_path: Path, monkeypatch) -> None:
        mod = _load_module(tmp_path)
        rag = _FakeRag()
        _stub_module(mod, rag, monkeypatch)

        _run_concurrent_recalls(mod, _CONCURRENT_RECALLS)

        assert rag.llm_model_func == "ORIGINAL_EXTRACTOR_FUNC"

    def test_answers_are_not_mixed_up_between_concurrent_calls(self, tmp_path: Path, monkeypatch) -> None:
        mod = _load_module(tmp_path)
        rag = _FakeRag()
        _stub_module(mod, rag, monkeypatch)

        payloads = _run_concurrent_recalls(mod, _CONCURRENT_RECALLS)

        for i, payload in enumerate(payloads):
            assert payload["question"] == f"q{i}"
            assert f"q{i}" in json.dumps(payload["answer"])
