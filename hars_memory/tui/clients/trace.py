"""Recall-trace SSE client.

The server framing is ``hars_memory.trace.format_sse``: ``event: NAME`` + one
single-line JSON ``data:`` + blank line, with ``: ping`` comment keepalives.
Parsed by hand so no SSE dependency is needed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

from hars_memory.tui.clients.base import Services, describe_exc

# Query params the listener's ``_recall_args`` honours (trace.py). Anything else
# (session_id, session_graph, view, debug) is silently dropped by the server, so
# the TUI routes those requests through MCP instead (plan GAP-1).
TRACE_SUPPORTED = frozenset(
    {"question", "mode", "top_k", "fetch_top_k", "context_only", "project", "ll_keywords", "hl_keywords", "access_token"}
)
STAGE_ORDER = (
    "trace_start", "bm25_hits", "vector_hits", "ripgrep_hits", "fusion",
    "rerank", "graph_expand", "context", "result", "done", "error",
)


class SseParser:
    """Incremental SSE parser: feed lines, collect ``(event, data)`` frames."""

    def __init__(self) -> None:
        self._event = "message"
        self._data: list[str] = []

    def feed(self, line: str) -> tuple[str, Any] | None:
        line = line.rstrip("\r\n")
        if line == "":
            frame = self._flush()
            return frame
        if line.startswith(":"):
            return None  # keepalive comment
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        return None

    def _flush(self) -> tuple[str, Any] | None:
        if not self._data:
            self._event = "message"
            return None
        raw = "\n".join(self._data)
        event = self._event
        self._event, self._data = "message", []
        try:
            return event, json.loads(raw)
        except ValueError:
            return event, {"raw": raw}


def parse_sse(lines: Iterable[str]) -> list[tuple[str, Any]]:
    """Parse a complete SSE body (used by tests and the buffered path)."""
    parser, out = SseParser(), []
    for line in lines:
        frame = parser.feed(line)
        if frame:
            out.append(frame)
    tail = parser.feed("")
    if tail:
        out.append(tail)
    return out


def supports(args: dict[str, Any]) -> bool:
    """True when every non-empty arg can be expressed on the /trace endpoint."""
    for key, value in args.items():
        if value in (None, "", [], False) and key not in ("context_only",):
            continue
        if key not in TRACE_SUPPORTED:
            return False
    return True


def query_params(args: dict[str, Any]) -> dict[str, str]:
    params: dict[str, str] = {}
    for key in ("question", "mode", "project"):
        if args.get(key):
            params[key] = str(args[key])
    for key in ("top_k", "fetch_top_k"):
        if args.get(key) not in (None, ""):
            params[key] = str(int(args[key]))
    params["context_only"] = "true" if args.get("context_only", True) else "false"
    for key in ("ll_keywords", "hl_keywords"):
        values = [str(v).strip() for v in (args.get(key) or []) if str(v).strip()]
        if values:
            # The listener splits on commas AND spaces, so a multi-word keyword
            # arrives as several single-word keywords over /trace (documented limit;
            # the MCP path keeps them intact).
            params[key] = ",".join(values)
    if args.get("access_token"):
        params["access_token"] = str(args["access_token"])
    return params


async def stream_trace(services: Services, args: dict[str, Any]) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Yield ``(event, payload)`` from ``GET <trace>/trace``; a transport failure becomes
    a terminal synthetic ``transport_error`` event (so the screen can fall back to MCP)."""
    base = services.config.trace_url
    if not base:
        yield "transport_error", {"error": "trace listener not configured"}
        return
    parser = SseParser()
    params = query_params(args)
    if services.config.token:
        params["access_token"] = services.config.token  # the listener has no header auth
    try:
        async with services.http.stream(
            "GET",
            base + "/trace",
            params=params,
            headers={"accept": "text/event-stream", **services.config.auth_headers()},
            timeout=None,
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")[:200]
                yield "transport_error", {"error": f"HTTP {resp.status_code}: {body}"}
                return
            async for line in resp.aiter_lines():
                frame = parser.feed(line)
                if frame:
                    yield frame
            tail = parser.feed("")
            if tail:
                yield tail
    except Exception as exc:  # noqa: BLE001
        yield "transport_error", {"error": describe_exc(exc)}
