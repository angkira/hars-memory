"""Fail-soft cross-lingual query-twin generation."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any, Awaitable, Callable

from hars_memory.retrieval.lang_route import detect_lang

_TWIN_PROMPT = """Translate/rephrase the following search query into {language} as a short, natural search query.
Keep every identifier, flag, command, and resource name verbatim, including exact spelling and
capitalization. Do not translate or alter those names. Output only the query text, with no quotes,
explanation, or label.

QUERY:
<<<
{query}
>>>"""
_CACHE_LIMIT = 256
_twin_cache: OrderedDict[str, str] = OrderedDict()


def _target_language(query: str) -> str:
    has_cyrillic = any(0x0400 <= ord(char) <= 0x052F or char in "Ёё" for char in query)
    has_latin = any("A" <= char <= "Z" or "a" <= char <= "z" for char in query)
    if has_cyrillic and has_latin:
        return "English"
    return "English" if detect_lang(query) == "ru" else "Russian"


def clear_twin_cache() -> None:
    """Clear the process-local successful-twin cache (useful for tests)."""
    _twin_cache.clear()


async def make_twin(
    llm_func: Callable[[str], Awaitable[Any]], query: str, timeout_s: float
) -> tuple[str | None, str | None]:
    """Ask ``llm_func`` for a short query in the other language, failing softly."""
    if query in _twin_cache:
        _twin_cache.move_to_end(query)
        return _twin_cache[query], None
    if timeout_s <= 0:
        return None, "invalid_timeout"
    language = _target_language(query)
    prompt = _TWIN_PROMPT.format(language=language, query=query)
    try:
        response = await asyncio.wait_for(llm_func(prompt), timeout=timeout_s)
    except (TimeoutError, asyncio.TimeoutError):
        return None, "timeout"
    except Exception as exc:
        return None, f"llm_error:{type(exc).__name__}"
    twin = " ".join(str(response or "").strip().split()).strip("\"'“”‘’")
    twin = " ".join(twin.split())
    if not twin:
        return None, "empty_response"
    _twin_cache[query] = twin
    _twin_cache.move_to_end(query)
    while len(_twin_cache) > _CACHE_LIMIT:
        _twin_cache.popitem(last=False)
    return twin, None


__all__ = ["clear_twin_cache", "make_twin"]