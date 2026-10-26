"""Tests for the bounded retry/backoff wrapping ``make_llm_func``'s LLM call
in ``server/lightrag_init.py``.

Context: ``make_llm_func`` is a thin adapter over the official OpenAI SDK's
``AsyncOpenAI`` client (built via
``lightrag.llm.openai.create_openai_async_client``). It deliberately does
NOT call ``lightrag.llm.openai.openai_complete_if_cache`` — see
``make_llm_func``'s docstring for why (that function's ``reasoning_content``
handling and built-in tenacity retry both have a different contract than
this deployment needs). The SDK client's own built-in retry
(``AsyncOpenAI(max_retries=...)``) is explicitly disabled
(``client_configs={"max_retries": 0}``) so this module's own tenacity-based
retry/backoff — env-configurable, structured-logged — is the only retry
layer exercised here.

These tests substitute the ``AsyncOpenAI`` client's transport with one bound
to an ``httpx.MockTransport`` (via a fake ``create_openai_async_client``) so
no real network/process is involved.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import httpx
import openai
import pytest

from hars_memory.server.lightrag_init import make_llm_func


def _client_factory_using_transport(
    transport: httpx.MockTransport,
) -> Callable[..., openai.AsyncOpenAI]:
    """Build a ``create_openai_async_client``-compatible factory pinned to a
    mock transport.

    Mirrors ``make_llm_func``'s call shape
    (``create_openai_async_client(api_key=..., base_url=..., timeout=...,
    client_configs={"max_retries": 0})``) but routes all requests through
    ``transport`` instead of the network, and honours the caller's
    ``client_configs["max_retries"]`` so tests exercising the (disabled) SDK
    retry path stay meaningful if that default ever changes.
    """

    def _factory(
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        client_configs: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> openai.AsyncOpenAI:
        max_retries = (client_configs or {}).get("max_retries", 0)
        return openai.AsyncOpenAI(
            api_key=api_key or "test",
            base_url=base_url,
            timeout=timeout,
            http_client=httpx.AsyncClient(transport=transport),
            max_retries=max_retries,
        )

    return _factory


def _chat_response(content: str = "ok") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def _make_llm_func_for_test() -> object:
    return make_llm_func(
        base_url="http://localhost:8083/v1",
        model="gemma-4-12b",
        max_tokens=64,
        temperature=0.1,
    )


def _patch_client_factory(monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport) -> None:
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory_using_transport(transport),
    )


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink retry backoff so retry tests run in milliseconds, not seconds.

    Production defaults are 1s / 8s; these values just need to be > 0 so the
    retry/backoff code path (not merely "retry with zero delay") is exercised.
    """
    monkeypatch.setenv("HARS_MEMORY_LLM_RETRY_BACKOFF_INITIAL_SECONDS", "0.001")
    monkeypatch.setenv("HARS_MEMORY_LLM_RETRY_BACKOFF_MAX_SECONDS", "0.002")


class TestMakeLlmFuncRetriesTransientFailures:
    def test_retries_connection_error_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                # Simulates llama-server refusing/dropping a connection mid
                # prompt-cache reorganisation pass.
                raise httpx.ConnectError("connection refused", request=request)
            return _chat_response("recovered")

        _patch_client_factory(monkeypatch, httpx.MockTransport(handler))

        llm_func = _make_llm_func_for_test()
        result = asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert result == "recovered"
        assert calls["n"] == 3

    def test_retries_503_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(503, text="server busy")
            return _chat_response("ok-after-503")

        _patch_client_factory(monkeypatch, httpx.MockTransport(handler))

        llm_func = _make_llm_func_for_test()
        result = asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert result == "ok-after-503"
        assert calls["n"] == 2

    def test_retries_429_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(429, text="rate limited")
            return _chat_response("ok-after-429")

        _patch_client_factory(monkeypatch, httpx.MockTransport(handler))

        llm_func = _make_llm_func_for_test()
        result = asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert result == "ok-after-429"
        assert calls["n"] == 2

    def test_logs_a_warning_per_retry_attempt(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.ConnectError("connection refused", request=request)
            return _chat_response("recovered")

        _patch_client_factory(monkeypatch, httpx.MockTransport(handler))

        llm_func = _make_llm_func_for_test()
        with caplog.at_level(logging.WARNING, logger="hars_memory.server.lightrag_init"):
            asyncio.run(llm_func("hello"))  # type: ignore[operator]

        retry_records = [r for r in caplog.records if "transient failure" in r.message]
        assert len(retry_records) == 2  # 2 failures before the 3rd (successful) attempt
        assert "gemma-4-12b" in retry_records[0].message
        assert "retrying in" in retry_records[0].message


class TestMakeLlmFuncDoesNotRetryFatalErrors:
    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
    def test_does_not_retry_non_429_4xx(
        self, monkeypatch: pytest.MonkeyPatch, status_code: int
    ) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(status_code, text="fatal client error")

        _patch_client_factory(monkeypatch, httpx.MockTransport(handler))

        llm_func = _make_llm_func_for_test()

        with pytest.raises(RuntimeError, match="LLM endpoint unavailable"):
            asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert calls["n"] == 1  # no retry attempted — this can never succeed

    def test_runtime_error_chains_the_original_openai_status_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chained cause is now an ``openai.APIStatusError`` (subclass
        ``AuthenticationError`` for 401) carrying ``.status_code`` directly,
        rather than the pre-SDK ``httpx.HTTPStatusError`` with a nested
        ``.response.status_code`` — the OpenAI SDK's typed error taxonomy
        exposes the status code as a first-class attribute on the exception
        itself.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="unauthorized")

        _patch_client_factory(monkeypatch, httpx.MockTransport(handler))

        llm_func = _make_llm_func_for_test()

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert isinstance(exc_info.value.__cause__, openai.APIStatusError)
        assert exc_info.value.__cause__.status_code == 401


class TestMakeLlmFuncGivesUpAfterBoundedAttempts:
    def test_gives_up_after_max_attempts_and_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            # Permanently failing endpoint: must give up, not hang or retry
            # indefinitely.
            raise httpx.ConnectError("connection refused", request=request)

        _patch_client_factory(monkeypatch, httpx.MockTransport(handler))
        monkeypatch.setenv("HARS_MEMORY_LLM_RETRY_ATTEMPTS", "3")

        llm_func = _make_llm_func_for_test()

        with pytest.raises(RuntimeError, match="LLM endpoint unavailable"):
            asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert calls["n"] == 3  # exactly the configured cap — bounded, not infinite

    def test_retry_attempts_env_var_of_one_disables_retrying(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            raise httpx.ConnectError("connection refused", request=request)

        _patch_client_factory(monkeypatch, httpx.MockTransport(handler))
        monkeypatch.setenv("HARS_MEMORY_LLM_RETRY_ATTEMPTS", "1")

        llm_func = _make_llm_func_for_test()

        with pytest.raises(RuntimeError, match="LLM endpoint unavailable"):
            asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert calls["n"] == 1
