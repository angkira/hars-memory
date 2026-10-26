"""Tests proving ``make_llm_func``'s ``reasoning_content`` fallback survives
the httpx -> OpenAI SDK migration (see ``server/lightrag_init.py``'s module
and ``make_llm_func`` docstrings).

Context: Qwen3.6-27B's thinking-mode output previously leaked into
extraction results — the model would emit an empty regular ``content`` field
and put its entire (non-reasoning) answer in the OpenAI-compatible
``reasoning_content`` field instead, invalidating indexing runs that only
read ``content``. ``make_llm_func`` recovers the real answer in that case by
falling back to ``reasoning_content`` verbatim, and separately strips any
inline ``<think>...</think>`` block that shows up WITHIN ``content`` itself
(a different, also-observed failure mode). Neither behaviour is expressible
through ``lightrag.llm.openai.openai_complete_if_cache``'s own
Chain-of-Thought handling (see ``make_llm_func``'s docstring for why), so
this module implements it directly against the OpenAI SDK's response
object.

These tests substitute the ``AsyncOpenAI`` client's transport with one bound
to an ``httpx.MockTransport`` (via a fake ``create_openai_async_client``) so
no real network/process is involved.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import openai
import pytest

from hars_memory.server.lightrag_init import make_llm_func


def _client_factory_using_transport(transport: httpx.MockTransport):
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


def _make_llm_func_for_test() -> object:
    return make_llm_func(
        base_url="http://localhost:8080/v1",
        model="Qwen3.6-27B-Q4_K_M",
        max_tokens=8192,
        temperature=0.1,
    )


def _patch_client_factory(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory_using_transport(httpx.MockTransport(handler)),
    )


class TestReasoningContentFallback:
    def test_empty_content_falls_back_to_reasoning_content_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Qwen3.6's thinking-mode leak: ``content`` is empty/blank and the
        entire extraction payload is in ``reasoning_content`` instead. The
        clean fallback must be used AS THE ANSWER, not discarded and not
        wrapped in ``<think>`` markers.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning_content": (
                                    '("entity"<|>ORDER-123<|>identifier<|>'
                                    "The order id referenced in the ticket)"
                                ),
                            }
                        }
                    ]
                },
            )

        _patch_client_factory(monkeypatch, handler)

        llm_func = _make_llm_func_for_test()
        result = asyncio.run(llm_func("extract entities"))  # type: ignore[operator]

        assert result == (
            '("entity"<|>ORDER-123<|>identifier<|>The order id referenced in the ticket)'
        )
        # The reasoning trace is the extraction result, not a discardable
        # thought — it must NOT be wrapped in <think> tags.
        assert "<think>" not in result

    def test_whitespace_only_content_also_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Some servers emit whitespace (not a true empty string) in
        ``content`` when the answer is entirely in ``reasoning_content``.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "   \n", "reasoning_content": "the real answer"}}
                    ]
                },
            )

        _patch_client_factory(monkeypatch, handler)

        llm_func = _make_llm_func_for_test()
        result = asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert result == "the real answer"

    def test_nonempty_content_ignores_reasoning_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the server correctly separates reasoning from the answer
        (``content`` is non-empty), the reasoning trace must be ignored —
        only the well-formed case (empty ``content``) triggers the fallback.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "the clean answer",
                                "reasoning_content": "internal chain of thought, discard me",
                            }
                        }
                    ],
                },
            )

        _patch_client_factory(monkeypatch, handler)

        llm_func = _make_llm_func_for_test()
        result = asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert result == "the clean answer"

    def test_inline_think_block_in_content_is_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A separate failure mode: the server inlines a ``<think>...</think>``
        block directly INSIDE ``content`` (not the ``reasoning_content``
        field). This must be stripped out, leaving only the real answer.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "<think>let me reason about this</think>the real answer"
                                )
                            }
                        }
                    ]
                },
            )

        _patch_client_factory(monkeypatch, handler)

        llm_func = _make_llm_func_for_test()
        result = asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert result == "the real answer"

    def test_no_reasoning_content_field_at_all_is_unaffected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Servers that never send ``reasoning_content`` (the common case)
        must behave exactly as before: plain ``content`` passed through.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "plain answer"}}]})

        _patch_client_factory(monkeypatch, handler)

        llm_func = _make_llm_func_for_test()
        result = asyncio.run(llm_func("hello"))  # type: ignore[operator]

        assert result == "plain answer"
