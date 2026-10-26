"""Shared adapter plumbing: the ``Reply`` envelope and the ``Services`` bundle."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from hars_memory.tui.config import ADMIN_TOOLS, WRITE_TOOLS, TuiConfig


@dataclass
class Reply:
    """Typed result of an adapter call. A dead endpoint is ``ok=False``, never an exception."""

    ok: bool
    data: Any = None
    error: str = ""
    latency_ms: float | None = None
    status: int | None = None

    @classmethod
    def unavailable(cls, reason: str, *, latency_ms: float | None = None, status: int | None = None) -> Reply:
        return cls(False, None, reason, latency_ms, status)


def describe_exc(exc: BaseException) -> str:
    """One-line, secret-free description of a transport error."""
    name = type(exc).__name__
    text = str(exc).strip()
    return f"{name}: {text}" if text else name


class Services:
    """Everything a screen needs to reach the world: config, one shared httpx client.

    Tests pass ``http=httpx.AsyncClient(transport=httpx.MockTransport(...))``;
    production builds a real client with the configured timeout.
    """

    def __init__(self, config: TuiConfig, http: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(config.timeout_s, connect=5.0))
        self._tools: set[str] | None = None

    async def aclose(self) -> None:
        await self.http.aclose()

    # -- raw HTTP ---------------------------------------------------------

    async def get_json(
        self, url: str, *, auth: bool = True, timeout: float | None = None
    ) -> Reply:
        """GET ``url`` and decode JSON. Auth header only goes to configured HARS hosts."""
        started = time.monotonic()
        headers = self.config.auth_headers() if auth else {}
        try:
            resp = await self.http.get(url, headers=headers, timeout=timeout or self.config.timeout_s)
        except Exception as exc:  # noqa: BLE001 - adapters never raise
            return Reply.unavailable(describe_exc(exc), latency_ms=_ms(started))
        latency = _ms(started)
        if resp.status_code >= 400:
            return Reply.unavailable(f"HTTP {resp.status_code}", latency_ms=latency, status=resp.status_code)
        try:
            return Reply(True, resp.json(), "", latency, resp.status_code)
        except ValueError:
            return Reply(True, resp.text, "", latency, resp.status_code)

    async def probe(self, url: str, *, timeout: float = 2.5) -> Reply:
        """Reachability probe for a third-party URL: any HTTP answer counts as reachable.

        Never sends our token (the target is an embedder / LLM endpoint, not HARS).
        """
        started = time.monotonic()
        try:
            resp = await self.http.get(url, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return Reply.unavailable(describe_exc(exc), latency_ms=_ms(started))
        return Reply(True, resp.status_code, "", _ms(started), resp.status_code)

    # -- MCP --------------------------------------------------------------

    async def mcp_rpc(self, method: str, params: dict[str, Any] | None = None) -> Reply:
        """One JSON-RPC call to the stateless streamable-HTTP endpoint (json_response=1)."""
        started = time.monotonic()
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        headers = {"accept": "application/json, text/event-stream", **self.config.auth_headers()}
        try:
            resp = await self.http.post(
                self.config.mcp_url, json=body, headers=headers, timeout=self.config.timeout_s * 4
            )
        except Exception as exc:  # noqa: BLE001
            return Reply.unavailable(describe_exc(exc), latency_ms=_ms(started))
        latency = _ms(started)
        if resp.status_code >= 400:
            return Reply.unavailable(f"HTTP {resp.status_code}", latency_ms=latency, status=resp.status_code)
        payload = _decode_rpc_body(resp)
        if payload is None:
            return Reply.unavailable("unparseable MCP response", latency_ms=latency, status=resp.status_code)
        if "error" in payload:
            err = payload["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            return Reply.unavailable(f"MCP error: {msg}", latency_ms=latency, status=resp.status_code)
        return Reply(True, payload.get("result"), "", latency, resp.status_code)

    async def list_tools(self) -> Reply:
        reply = await self.mcp_rpc("tools/list")
        if reply.ok:
            tools = (reply.data or {}).get("tools") or []
            self._tools = {str(t.get("name")) for t in tools if isinstance(t, dict)}
            return Reply(True, sorted(self._tools), "", reply.latency_ms, reply.status)
        return reply

    @property
    def known_tools(self) -> set[str] | None:
        """Tool names from the last ``tools/list`` (capability probe), or None if unknown."""
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Reply:
        """Call a ``memory_*`` tool. Admin tools are refused unless ``allow_admin``; writes
        are refused in read-only mode (defence in depth: the screens also gate these)."""
        cfg = self.config
        if name in ADMIN_TOOLS and not cfg.allow_admin:
            return Reply.unavailable(f"{name} is an admin tool; start the TUI with --allow-admin")
        if name in (WRITE_TOOLS | ADMIN_TOOLS) and cfg.read_only:
            return Reply.unavailable(f"{name} refused: TUI is in read-only mode")
        args = dict(arguments or {})
        if cfg.token and "access_token" not in args:
            args["access_token"] = cfg.token
        reply = await self.mcp_rpc("tools/call", {"name": name, "arguments": args})
        if not reply.ok:
            return reply
        result = reply.data or {}
        text = ""
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text") or "")
                break
        try:
            payload = json.loads(text) if text else {}
        except ValueError:
            payload = {"raw": text}
        if result.get("isError"):
            msg = payload.get("error") if isinstance(payload, dict) else None
            return Reply(False, payload, str(msg or text or "tool error")[:300], reply.latency_ms, reply.status)
        if isinstance(payload, dict) and payload.get("ok") is False:
            return Reply(False, payload, str(payload.get("error") or "ok=false")[:300], reply.latency_ms, reply.status)
        return Reply(True, payload, "", reply.latency_ms, reply.status)

    async def healthz(self) -> Reply:
        return await self.get_json(self.config.base_url + "/healthz")


def _ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000.0, 1)


def _decode_rpc_body(resp: httpx.Response) -> dict[str, Any] | None:
    """Accept both ``application/json`` and a single-event SSE body."""
    ctype = resp.headers.get("content-type", "")
    text = resp.text
    if "text/event-stream" in ctype:
        for line in text.splitlines():
            if line.startswith("data:"):
                text = line[5:].strip()
                break
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None

