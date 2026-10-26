"""Tests for server/reranker.py's local CPU cross-encoder rerank function, and
for its wiring (gated by HARS_MEMORY_RERANK_MODEL) into
server/lightrag_init.py::create_lightrag().

No real model is loaded in the shape/ordering/laziness tests below —
``reranker._load_model`` is monkeypatched with a fake CrossEncoder stand-in so
these tests run fast and offline. The end-to-end "does the real model
actually rerank" demonstration lives outside the test suite (see the task
report) since it requires the real ~68M-param model and is not something a
unit test should depend on downloading/loading on every run.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock

import pytest

from hars_memory.server.reranker import make_rerank_func


class _FakeCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder.

    Assigns each (query, doc) pair a score derived from the doc's position in
    a fixed relevance ranking, so tests can assert on exact reordering.
    """

    def __init__(self, score_by_doc: dict[str, float]) -> None:
        self._score_by_doc = score_by_doc

    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[float]:
        return [self._score_by_doc[doc] for _query, doc in pairs]


class TestMakeRerankFuncShapeAndOrdering:
    def test_returns_index_relevance_score_dicts_sorted_descending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        docs = ["irrelevant doc", "somewhat relevant doc", "highly relevant doc"]
        # Deliberately NOT in score order — index 2 should end up first.
        scores_by_doc = {docs[0]: 0.1, docs[1]: 5.0, docs[2]: 11.5}
        fake_model = _FakeCrossEncoder(scores_by_doc)
        monkeypatch.setattr(
            "hars_memory.server.reranker._load_model",
            lambda *a, **kw: fake_model,
        )

        rerank = make_rerank_func(model_name="fake/reranker")
        results = asyncio.run(rerank(query="what is relevant?", documents=docs))

        assert [r["index"] for r in results] == [2, 1, 0]
        assert results[0]["relevance_score"] == pytest.approx(11.5)
        assert results[1]["relevance_score"] == pytest.approx(5.0)
        assert results[2]["relevance_score"] == pytest.approx(0.1)
        # Contract: every element is exactly {"index": int, "relevance_score": float}
        for r in results:
            assert set(r.keys()) == {"index", "relevance_score"}
            assert isinstance(r["index"], int)
            assert isinstance(r["relevance_score"], float)

    def test_top_n_truncates_after_sorting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        docs = ["a", "b", "c", "d"]
        scores_by_doc = {"a": 1.0, "b": 4.0, "c": 3.0, "d": 2.0}
        monkeypatch.setattr(
            "hars_memory.server.reranker._load_model",
            lambda *a, **kw: _FakeCrossEncoder(scores_by_doc),
        )

        rerank = make_rerank_func(model_name="fake/reranker")
        results = asyncio.run(rerank(query="q", documents=docs, top_n=2))

        assert len(results) == 2
        assert [r["index"] for r in results] == [1, 2]  # "b" (4.0) then "c" (3.0)

    def test_empty_documents_returns_empty_without_loading_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        load_mock = MagicMock(side_effect=AssertionError("must not load model for empty input"))
        monkeypatch.setattr("hars_memory.server.reranker._load_model", load_mock)

        rerank = make_rerank_func(model_name="fake/reranker")
        results = asyncio.run(rerank(query="q", documents=[]))

        assert results == []
        load_mock.assert_not_called()


class TestMakeRerankFuncLazyLoad:
    def test_constructing_rerank_func_does_not_load_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        load_mock = MagicMock(side_effect=AssertionError("model must not load at construction time"))
        monkeypatch.setattr("hars_memory.server.reranker._load_model", load_mock)

        # Constructing the closure must be free of side effects — this is
        # what lets create_lightrag() build rerank_model_func unconditionally
        # cheap when HARS_MEMORY_RERANK_MODEL happens to be set but no query has
        # run yet.
        make_rerank_func(model_name="fake/reranker")

        load_mock.assert_not_called()

    def test_model_loads_only_on_first_actual_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_model = _FakeCrossEncoder({"doc": 1.0})
        load_mock = MagicMock(return_value=fake_model)
        monkeypatch.setattr("hars_memory.server.reranker._load_model", load_mock)

        rerank = make_rerank_func(model_name="fake/reranker")
        load_mock.assert_not_called()

        asyncio.run(rerank(query="q", documents=["doc"]))
        load_mock.assert_called_once()


class TestCreateLightragRerankGating:
    """create_lightrag() wiring: HARS_MEMORY_RERANK_MODEL gates rerank_model_func."""

    def _base_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
        # Real local embedder (already cached at HF_HOME) — the same model
        # every other create_lightrag() caller in this repo uses, and no
        # network is required (HARS_MEMORY_EMBED_LOCAL_FILES_ONLY=1 below).
        monkeypatch.setenv("HARS_MEMORY_INDEX_DIR", str(tmp_path))
        monkeypatch.setenv("HARS_MEMORY_VECTOR_STORAGE", "NanoVectorDBStorage")
        monkeypatch.setenv("HARS_MEMORY_GRAPH_STORAGE", "NetworkXStorage")
        monkeypatch.setenv("HARS_MEMORY_EMBED_MODEL", "unsloth/embeddinggemma-300m")
        monkeypatch.setenv("HARS_MEMORY_EMBED_LOCAL_FILES_ONLY", "1")
        monkeypatch.setenv("HF_HOME", os.environ.get("HF_HOME", "/mnt/datasets/models/.hf_home"))
        monkeypatch.delenv("HARS_MEMORY_RERANK_MODEL", raising=False)

    @pytest.mark.skipif(
        not os.path.isdir(
            os.path.join(
                os.environ.get("HF_HOME", "/mnt/datasets/models/.hf_home"),
                "hub",
                "models--unsloth--embeddinggemma-300m",
            )
        ),
        reason="embeddinggemma-300m not present in local HF cache in this environment",
    )
    def test_rerank_model_func_is_none_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        self._base_env(monkeypatch, tmp_path)
        load_mock = MagicMock(side_effect=AssertionError("reranker must not load when disabled"))
        monkeypatch.setattr("hars_memory.server.reranker._load_model", load_mock)

        from hars_memory.server.lightrag_init import create_lightrag

        rag = create_lightrag()

        assert rag.rerank_model_func is None
        load_mock.assert_not_called()

    @pytest.mark.skipif(
        not os.path.isdir(
            os.path.join(
                os.environ.get("HF_HOME", "/mnt/datasets/models/.hf_home"),
                "hub",
                "models--unsloth--embeddinggemma-300m",
            )
        ),
        reason="embeddinggemma-300m not present in local HF cache in this environment",
    )
    def test_rerank_model_func_wired_when_env_set_without_loading_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        self._base_env(monkeypatch, tmp_path)
        monkeypatch.setenv("HARS_MEMORY_RERANK_MODEL", "cross-encoder/ettin-reranker-68m-v1")
        load_mock = MagicMock(side_effect=AssertionError("model must not load at create_lightrag() time"))
        monkeypatch.setattr("hars_memory.server.reranker._load_model", load_mock)

        from hars_memory.server.lightrag_init import create_lightrag

        rag = create_lightrag()

        assert rag.rerank_model_func is not None
        assert asyncio.iscoroutinefunction(rag.rerank_model_func)
        load_mock.assert_not_called()
        assert rag.min_rerank_score == pytest.approx(0.0)


class TestRerankDoesNotBlockEventLoop:
    """Cross-encoder inference is a blocking CPU call; `rerank()` must keep it
    in a worker thread so concurrent coroutines stay schedulable.
    """

    _BLOCKING_SECONDS = 0.3
    _TICK_SECONDS = 0.01

    def test_concurrent_coroutine_keeps_ticking_during_inference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time as _time

        class _SlowCrossEncoder:
            def predict(self, pairs, **kwargs):  # type: ignore[no-untyped-def]
                _time.sleep(TestRerankDoesNotBlockEventLoop._BLOCKING_SECONDS)
                return [float(len(pairs) - i) for i in range(len(pairs))]

        monkeypatch.setattr(
            "hars_memory.server.reranker._load_model",
            lambda *args, **kwargs: _SlowCrossEncoder(),
        )
        rerank = make_rerank_func(model_name="fake", device="cpu")

        async def scenario() -> tuple[int, list[dict]]:
            ticks = 0
            done = False

            async def ticker() -> None:
                nonlocal ticks
                while not done:
                    await asyncio.sleep(TestRerankDoesNotBlockEventLoop._TICK_SECONDS)
                    ticks += 1

            ticker_task = asyncio.create_task(ticker())
            results = await rerank("q", ["doc a", "doc b"])
            done = True
            await ticker_task
            return ticks, results

        ticks, results = asyncio.run(scenario())
        assert [r["index"] for r in results] == [0, 1]
        assert ticks >= 10, f"event loop appeared blocked: only {ticks} ticks"
