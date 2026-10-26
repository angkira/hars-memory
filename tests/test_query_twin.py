from __future__ import annotations

import asyncio

from hars_memory.retrieval.query_twin import clear_twin_cache, make_twin


def test_make_twin_strips_quotes_and_preserves_prompt_language() -> None:
    clear_twin_cache()
    prompts: list[str] = []

    async def llm(prompt: str) -> str:
        prompts.append(prompt)
        return '"check Qdrant status"'

    twin, reason = asyncio.run(make_twin(llm, "проверить Qdrant", 1.0))
    assert twin == "check Qdrant status"
    assert reason is None
    assert "into English" in prompts[0]
    assert "Qdrant" in prompts[0]


def test_mixed_script_query_twin_targets_english_below_language_threshold() -> None:
    clear_twin_cache()
    prompts: list[str] = []

    async def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "translated"

    asyncio.run(make_twin(llm, "a very long latin identifier plus я", 1.0))
    assert "into English" in prompts[0]


def test_make_twin_timeout_exception_and_empty_response_fail_softly() -> None:
    clear_twin_cache()

    async def slow(_prompt: str) -> str:
        await asyncio.sleep(0.05)
        return "late"

    async def broken(_prompt: str) -> str:
        raise RuntimeError("unavailable")

    async def empty(_prompt: str) -> str:
        return "  “ ”  "

    assert asyncio.run(make_twin(slow, "hello", 0.001)) == (None, "timeout")
    assert asyncio.run(make_twin(broken, "hello again", 1.0)) == (None, "llm_error:RuntimeError")
    assert asyncio.run(make_twin(empty, "hello empty", 1.0)) == (None, "empty_response")


def test_make_twin_caches_success_by_query() -> None:
    clear_twin_cache()
    calls = 0

    async def llm(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "translated"

    async def run() -> None:
        assert await make_twin(llm, "cache me", 1.0) == ("translated", None)
        assert await make_twin(llm, "cache me", 1.0) == ("translated", None)

    asyncio.run(run())
    assert calls == 1