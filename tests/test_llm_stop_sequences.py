"""Tests for ``make_llm_func``'s optional server-side stop sequences (env
``HARS_MEMORY_LLM_STOP``, sent as the Chat Completions ``stop`` parameter).

Context: the 2026-10-16 full re-index measured ~24% of all generated tokens
being produced AFTER the ``<|COMPLETE|>`` delimiter the pipeline already
discards (post-delimiter loops plus residual thinking-channel output — see
``eval_agentic/rca/index_perf_analysis.md`` §3b). Stopping generation at the
marker therefore loses zero retained content. The switch is env-only and
default-off so the payload is byte-identical to before unless the operator
opts in, and it is cache-safe because response-cache keys are prompt-based,
not payload-based.

Behaviour under test:
* env set    -> the payload's ``stop`` is the comma-split list (whitespace
  trimmed, empty segments dropped);
* env unset/blank -> no effective ``stop`` in the payload (key absent or
  ``null``), i.e. byte-identical behaviour to the pre-change deployment;
* a per-call ``stop`` primitive forwarded through LightRAG's ``**kwargs``
  overrides the env default (mirrors the pre-existing
  ``payload.update()``-style override semantics for ``max_tokens`` /
  ``temperature``).

These tests substitute the ``AsyncOpenAI`` client's transport with one bound
to an ``httpx.MockTransport`` (via a fake ``create_openai_async_client``) so
no real network/process is involved — same approach as
``test_llm_retry.py`` / ``test_llm_reasoning_content_fallback.py``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import openai
import pytest

from hars_memory.server.lightrag_init import make_llm_func

COMPLETE_MARKER = "<|COMPLETE|>"


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


def _patch_client_factory(monkeypatch: pytest.MonkeyPatch, handler) -> dict[str, Any]:
    """Patch the client factory and return a handle capturing the last
    request's JSON body (so tests can assert on the serialized payload)."""
    captured: dict[str, Any] = {}

    def handler_capturing(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": f'("entity"<|#|>THING<|#|>type<|#|>x){COMPLETE_MARKER}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8},
            },
        )

    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory_using_transport(httpx.MockTransport(handler_capturing)),
    )
    return captured


def _make_llm_func_for_test() -> object:
    return make_llm_func(
        base_url="http://localhost:8080/v1",
        model="Qwen3.8-27B-UD-Q4_K_M",
        max_tokens=8192,
        temperature=0.1,
    )


class TestStopSequences:
    def test_env_set_adds_stop_list_to_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HARS_MEMORY_LLM_STOP set -> the request payload carries the
        comma-split stop list (the re-index case: stop at <|COMPLETE|>)."""
        captured = _patch_client_factory(monkeypatch, None)
        monkeypatch.setenv("HARS_MEMORY_LLM_STOP", COMPLETE_MARKER)

        llm_func = _make_llm_func_for_test()
        result = asyncio.run(llm_func("extract entities"))  # type: ignore[operator]

        assert captured["stop"] == [COMPLETE_MARKER]
        assert result.endswith(COMPLETE_MARKER)

    def test_env_comma_split_trims_and_drops_empty_segments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multiple markers are comma-split; whitespace is trimmed and empty
        segments (e.g. a trailing comma) are dropped."""
        captured = _patch_client_factory(monkeypatch, None)
        monkeypatch.setenv("HARS_MEMORY_LLM_STOP", f"{COMPLETE_MARKER}, ###END ,,")

        llm_func = _make_llm_func_for_test()
        asyncio.run(llm_func("extract entities"))  # type: ignore[operator]

        assert captured["stop"] == [COMPLETE_MARKER, "###END"]

    def test_env_unset_keeps_payload_without_stop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default (env unset) is off: no effective ``stop`` key in the
        payload, so the request is unchanged from the pre-change deployment."""
        captured = _patch_client_factory(monkeypatch, None)
        monkeypatch.delenv("HARS_MEMORY_LLM_STOP", raising=False)

        llm_func = _make_llm_func_for_test()
        asyncio.run(llm_func("extract entities"))  # type: ignore[operator]

        assert captured.get("stop") is None

    def test_env_blank_or_commas_only_is_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A whitespace-only or comma-only value must not produce a garbage
        stop list (e.g. ['']) — it deactivates the feature."""
        captured = _patch_client_factory(monkeypatch, None)
        monkeypatch.setenv("HARS_MEMORY_LLM_STOP", "  ,,")

        llm_func = _make_llm_func_for_test()
        asyncio.run(llm_func("extract entities"))  # type: ignore[operator]

        assert captured.get("stop") is None

    def test_caller_stop_kwarg_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``stop`` primitive forwarded through LightRAG's ``**kwargs``
        overrides the env default — same override semantics as
        ``max_tokens``/``temperature`` — and must not be duplicated into
        ``extra_body``."""
        captured = _patch_client_factory(monkeypatch, None)
        monkeypatch.setenv("HARS_MEMORY_LLM_STOP", COMPLETE_MARKER)

        llm_func = _make_llm_func_for_test()
        asyncio.run(llm_func("extract entities", stop="###END"))  # type: ignore[operator]

        assert captured["stop"] == "###END"

    def test_env_stop_does_not_leak_into_extra_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The env-configured stop list goes out as the top-level ``stop``
        parameter; ``extra_body`` (e.g. reasoning effort) rides alongside it
        unchanged."""
        captured = _patch_client_factory(monkeypatch, None)
        monkeypatch.setenv("HARS_MEMORY_LLM_STOP", COMPLETE_MARKER)

        llm_func = _make_llm_func_for_test()
        asyncio.run(llm_func("extract entities", reasoning_effort="low"))  # type: ignore[operator]

        assert captured["stop"] == [COMPLETE_MARKER]
        assert captured.get("reasoning_effort") == "low"
