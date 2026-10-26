"""Tests for HARS_MEMORY_RERANK_PROVIDER=http (hars_memory/server/reranker.py).

All HTTP traffic is mocked via httpx.MockTransport injected by monkeypatching
reranker._make_async_client — no real network anywhere in this file.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from hars_memory.server import reranker

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _clear_rerank_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "HARS_MEMORY_RERANK_PROVIDER",
        "HARS_MEMORY_RERANK_BASE_URL",
        "HARS_MEMORY_RERANK_MODEL",
        "HARS_MEMORY_RERANK_HTTP_MODEL",
        "HARS_MEMORY_RERANK_HTTP_API_KEY",
        "HARS_MEMORY_RERANK_HTTP_TIMEOUT",
        "HARS_MEMORY_RERANK_HTTP_MAX_DOCS_PER_REQUEST",
        "HARS_MEMORY_RERANK_HTTP_MIN_INTERVAL_MS",
        "HARS_MEMORY_RERANK_DEVICE",
        "HARS_MEMORY_RERANK_BATCH_SIZE",
        "HARS_MEMORY_RERANK_LOCAL_FILES_ONLY",
    ):
        monkeypatch.delenv(var, raising=False)


def _install_transport(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[httpx.Request]:
    """Patch reranker._make_async_client with a MockTransport-backed client."""
    requests: list[httpx.Request] = []

    def _transport_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    def _fake_client(timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=timeout,
            transport=httpx.MockTransport(_transport_handler),
        )

    monkeypatch.setattr(reranker, "_make_async_client", _fake_client)
    return requests


def _ok_response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content.decode())
    docs = body["documents"]
    # Deterministic pseudo-rerank: score = index within the chunk.
    return httpx.Response(
        200,
        json={
            "results": [
                {"index": i, "relevance_score": float(len(docs) - i)}
                for i in range(len(docs))
            ]
        },
    )


def _make_func(monkeypatch: pytest.MonkeyPatch, **kwargs: str):
    return reranker.make_rerank_func(
        model_name=kwargs.get("model_name", "Qwen3-Reranker-0.6B")
    )


# --------------------------------------------------------------------------- #
# provider resolution
# --------------------------------------------------------------------------- #


def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    assert reranker.make_rerank_func_from_env() is None


def test_auto_model_only_selects_local(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_MODEL", "cross-encoder/ettin-reranker-68m-v1")
    # make_rerank_func (local) must be returned untouched — it is lazy and never
    # loads the model, so this is safe without any weights present.
    func = reranker.make_rerank_func_from_env()
    assert callable(func)


def test_auto_base_url_selects_http(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "http://127.0.0.1:8792/v1")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MODEL", "Qwen3-Reranker-0.6B")
    _install_transport(monkeypatch, _ok_response)
    func = reranker.make_rerank_func_from_env()
    assert callable(func)


def test_unknown_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "lokcal")
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "http://127.0.0.1:8792/v1")
    with pytest.raises(ValueError, match="invalid"):
        reranker.make_rerank_func_from_env()


def test_http_without_base_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MODEL", "Qwen3-Reranker-0.6B")
    with pytest.raises(ValueError, match="BASE_URL"):
        reranker.make_rerank_func_from_env()


def test_http_without_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "http://127.0.0.1:8792/v1")
    with pytest.raises(ValueError, match="model id"):
        reranker.make_rerank_func_from_env()


def test_http_model_falls_back_to_rerank_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "http://127.0.0.1:8792/v1")
    monkeypatch.setenv("HARS_MEMORY_RERANK_MODEL", "Qwen3-Reranker-0.6B")
    requests = _install_transport(monkeypatch, _ok_response)
    func = _make_func(monkeypatch)
    result = asyncio.run(func("q", ["a", "b"]))
    assert json.loads(requests[0].content.decode())["model"] == "Qwen3-Reranker-0.6B"
    assert result[0]["index"] == 0  # score = len(docs) - i → index 0 wins


# --------------------------------------------------------------------------- #
# http behaviour
# --------------------------------------------------------------------------- #


def test_request_body_and_result_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "http://127.0.0.1:8792/v1")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MODEL", "Qwen3-Reranker-0.6B")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["query"] == "what is memory"
        assert body["documents"] == ["doc0", "doc1", "doc2"]
        assert "top_n" not in body  # slicing is client-side by design
        return httpx.Response(
            200,
            json={"results": [{"index": 2, "relevance_score": 0.9},
                              {"index": 0, "relevance_score": 0.1}]},
        )

    _install_transport(monkeypatch, handler)
    func = _make_func(monkeypatch)
    result = asyncio.run(func("what is memory", ["doc0", "doc1", "doc2"], top_n=2))
    assert [r["index"] for r in result] == [2, 0]
    assert result[0]["relevance_score"] == 0.9


def test_chunked_documents_merge_and_top_n(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "http://127.0.0.1:8792/v1")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MODEL", "m")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MAX_DOCS_PER_REQUEST", "2")

    requests = _install_transport(monkeypatch, _ok_response)
    func = _make_func(monkeypatch)
    docs = [f"d{i}" for i in range(5)]  # → chunks [0,1], [2,3], [4]
    result = asyncio.run(func("q", docs, top_n=3))
    assert len(requests) == 3
    # Scores are len(chunk)-i per chunk: d0=2, d1=1, d2=2, d3=1, d4=1.
    # Global sort desc, stable on ties → top-3 indexes [0, 2, 1].
    assert [r["index"] for r in result] == [0, 2, 1]
    for req, expected_docs in zip(requests, (["d0", "d1"], ["d2", "d3"], ["d4"])):
        assert json.loads(req.content.decode())["documents"] == expected_docs


def test_empty_documents_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "http://127.0.0.1:8792/v1")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MODEL", "m")
    requests = _install_transport(monkeypatch, _ok_response)
    func = _make_func(monkeypatch)
    assert asyncio.run(func("q", [])) == []
    assert requests == []


def test_non_string_documents_coerced(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "http://127.0.0.1:8792/v1")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MODEL", "m")
    requests = _install_transport(monkeypatch, _ok_response)
    func = _make_func(monkeypatch)
    asyncio.run(func("q", ["text", 42]))
    assert json.loads(requests[0].content.decode())["documents"] == ["text", "42"]


def test_429_honours_retry_after_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "http://127.0.0.1:8792/v1")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MODEL", "m")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MIN_INTERVAL_MS", "0")

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, text="slow down")
        return _ok_response(request)

    _install_transport(monkeypatch, handler)
    func = _make_func(monkeypatch)
    result = asyncio.run(func("q", ["a", "b"]))
    assert calls["n"] == 2
    assert result[0]["index"] == 0


def test_429_exhausted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "http://127.0.0.1:8792/v1")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MODEL", "m")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MIN_INTERVAL_MS", "0")
    _install_transport(
        monkeypatch, lambda request: httpx.Response(429, text="rate limited")
    )
    func = _make_func(monkeypatch)
    with pytest.raises(RuntimeError, match="rate limited"):
        asyncio.run(func("q", ["a"]))


def test_5xx_retries_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "http://127.0.0.1:8792/v1")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MODEL", "m")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MIN_INTERVAL_MS", "0")
    requests = _install_transport(
        monkeypatch, lambda request: httpx.Response(500, text="boom")
    )
    func = _make_func(monkeypatch)
    with pytest.raises(RuntimeError, match="500"):
        asyncio.run(func("q", ["a"]))
    assert len(requests) == 3  # _HTTP_MAX_ATTEMPTS


def test_4xx_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "http://127.0.0.1:8792/v1")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MODEL", "m")
    requests = _install_transport(
        monkeypatch, lambda request: httpx.Response(422, text="bad json")
    )
    func = _make_func(monkeypatch)
    with pytest.raises(RuntimeError, match="422"):
        asyncio.run(func("q", ["a"]))
    assert len(requests) == 1


def test_api_key_sent_as_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "https://api.jina.ai/v1")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MODEL", "jina-reranker-v3")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_API_KEY", "secret-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret-key"
        assert str(request.url) == "https://api.jina.ai/v1/rerank"
        return _ok_response(request)

    _install_transport(monkeypatch, handler)
    func = _make_func(monkeypatch)
    asyncio.run(func("q", ["a"]))


def test_malformed_results_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "http://127.0.0.1:8792/v1")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MODEL", "m")
    _install_transport(
        monkeypatch,
        lambda request: httpx.Response(200, json={"unexpected": []}),
    )
    func = _make_func(monkeypatch)
    with pytest.raises(RuntimeError, match="results"):
        asyncio.run(func("q", ["a"]))


def test_out_of_range_index_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_RERANK_BASE_URL", "http://127.0.0.1:8792/v1")
    monkeypatch.setenv("HARS_MEMORY_RERANK_HTTP_MODEL", "m")
    _install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={"results": [{"index": 7, "relevance_score": 0.5}]},
        ),
    )
    func = _make_func(monkeypatch)
    with pytest.raises(RuntimeError, match="out-of-range"):
        asyncio.run(func("q", ["a", "b"]))


# --------------------------------------------------------------------------- #
# local path untouched
# --------------------------------------------------------------------------- #


def test_local_provider_still_builds_lazy_func(monkeypatch: pytest.MonkeyPatch) -> None:
    """provider=local must return the in-process closure (never load weights)."""
    _clear_rerank_env(monkeypatch)
    monkeypatch.setenv("HARS_MEMORY_RERANK_PROVIDER", "local")
    monkeypatch.setenv("HARS_MEMORY_RERANK_MODEL", "cross-encoder/ettin-reranker-68m-v1")
    func = reranker.make_rerank_func_from_env()
    # Empty documents short-circuit before any model load — proves laziness.
    assert asyncio.run(func("q", [])) == []
