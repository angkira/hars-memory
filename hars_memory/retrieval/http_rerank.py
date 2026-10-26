"""Fail-soft async HTTP client for cross-encoder reranking."""

from __future__ import annotations

import asyncio
import math
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

DEFAULT_RERANK_URL = "http://127.0.0.1:8081/v1/rerank"
DEFAULT_RERANK_MODEL = "x"
DOCUMENT_CHAR_LIMIT = 2500


class _MalformedResponse(ValueError):
    """The server returned a response that cannot safely reorder candidates."""


def normalize_rerank_url(url: str) -> str:
    """Resolve a host or ``/v1`` base URL to a rerank endpoint."""
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("rerank URL must be an absolute HTTP(S) URL")

    path = parsed.path.rstrip("/")
    if path.endswith("/v1/rerank"):
        endpoint = path
    elif path.endswith("/v1"):
        endpoint = f"{path}/rerank"
    elif not path:
        endpoint = "/v1/rerank"
    else:
        endpoint = path
    return urlunsplit((parsed.scheme, parsed.netloc, endpoint, parsed.query, parsed.fragment))


def _ordered_indices(response: Any, document_count: int) -> list[int]:
    if not isinstance(response, dict):
        raise _MalformedResponse("response must be a JSON object")
    results = response.get("results")
    if not isinstance(results, list) or len(results) != document_count:
        raise _MalformedResponse("results must contain one item per document")

    scores: dict[int, float] = {}
    for item in results:
        if not isinstance(item, dict):
            raise _MalformedResponse("each result must be an object")
        index = item.get("index")
        score = item.get("relevance_score")
        if isinstance(index, bool) or not isinstance(index, int):
            raise _MalformedResponse("result index must be an integer")
        if index < 0 or index >= document_count or index in scores:
            raise _MalformedResponse("result indices must be unique and in range")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise _MalformedResponse("relevance_score must be numeric")
        try:
            numeric_score = float(score)
        except OverflowError as exc:
            raise _MalformedResponse("relevance_score must be finite") from exc
        if not math.isfinite(numeric_score):
            raise _MalformedResponse("relevance_score must be finite")
        scores[index] = numeric_score

    if set(scores) != set(range(document_count)):
        raise _MalformedResponse("results must include every document index")
    return sorted(scores, key=lambda index: (-scores[index], index))


async def rerank_http(
    query: str,
    documents: list[str],
    *,
    url: str = DEFAULT_RERANK_URL,
    model: str = DEFAULT_RERANK_MODEL,
    timeout_s: float = 8.0,
    doc_char_limit: int = DOCUMENT_CHAR_LIMIT,
) -> tuple[list[int] | None, str | None]:
    """Return ranked document indices, or ``(None, reason)`` on any failure.

    Each document is truncated to ``doc_char_limit`` chars in the request body:
    huge documents saturate cross-encoder scores (they add latency and destroy
    the score spread without adding signal — see
    HARS_MEMORY_RERANK_HTTP_DOC_CHARS in mcp_server.py for the measurement).

    The HTTP request is made once. The explicit ``wait_for`` complements the
    transport timeout so the caller has a hard upper bound even if a transport
    implementation does not enforce its own deadline as expected.
    """
    if not documents:
        return None, "empty_pool"

    try:
        endpoint = normalize_rerank_url(url)
        payload = {
            "model": model,
            "query": query,
            "documents": [document[: max(0, doc_char_limit)] for document in documents],
        }

        async def post() -> list[int]:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                try:
                    body = response.json()
                except (ValueError, UnicodeDecodeError) as exc:
                    raise _MalformedResponse("response body is not valid JSON") from exc
            return _ordered_indices(body, len(documents))

        return await asyncio.wait_for(post(), timeout=timeout_s), None
    except (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException):
        return None, "timeout"
    except _MalformedResponse:
        return None, "malformed_response"
    except Exception:
        return None, "http_error"


__all__ = ["DEFAULT_RERANK_MODEL", "DEFAULT_RERANK_URL", "normalize_rerank_url", "rerank_http"]