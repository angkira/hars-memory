#!/usr/bin/env python3
"""Smoke-test the native HTTP(S) memory MCP endpoint end to end.

Exercises the real deployment surface with nothing but ``httpx``:

1. RFC 9728 / RFC 8414 discovery
2. RFC 7591 dynamic client registration
3. authorization-code + PKCE S256 (``--approval-token`` or ``--access-token``)
4. ``initialize`` + ``tools/list`` over ``/mcp`` with the bearer token
5. ``memory_recall`` latency for N independent clients

Examples::

    # plain HTTP, OAuth, one client
    python scripts/mcp_http_smoke.py --url http://127.0.0.1:8109 \\
        --approval-token "$HARS_MEMORY_MCP_OAUTH_APPROVAL_TOKEN"

    # HTTPS with the mkcert root CA, 4 clients x 3 recalls
    python scripts/mcp_http_smoke.py --url https://127.0.0.1:8109 \\
        --ca ~/.local/share/mkcert/rootCA.pem --clients 4 --repeat 3

    # no OAuth (legacy unauthenticated endpoint)
    python scripts/mcp_http_smoke.py --url http://127.0.0.1:8099 --no-auth
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import statistics
import sys
import time
from typing import Any
from urllib.parse import urlencode

import httpx

MCP_HEADERS = {"accept": "application/json, text/event-stream"}
PROTOCOL_VERSION = "2025-06-18"


def _json_or_sse(response: httpx.Response) -> dict[str, Any]:
    if "text/event-stream" in response.headers.get("content-type", ""):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                if "id" in payload:
                    return payload
        raise RuntimeError(f"no JSON-RPC response in SSE stream: {response.text[:400]}")
    return response.json()


class Client:
    def __init__(self, url: str, *, ca: str | None, timeout: float) -> None:
        self.url = url.rstrip("/")
        self.http = httpx.Client(verify=ca or True, timeout=timeout, follow_redirects=False)
        self.access_token: str | None = None
        self._rpc_id = 0

    def close(self) -> None:
        self.http.close()

    # -- OAuth ------------------------------------------------------------

    def discover(self) -> dict[str, Any]:
        response = self.http.get(f"{self.url}/.well-known/oauth-protected-resource")
        response.raise_for_status()
        return response.json()

    def register(self, scope: str) -> dict[str, Any]:
        response = self.http.post(
            f"{self.url}/register",
            json={
                "client_name": "hars-memory-smoke",
                "redirect_uris": ["http://127.0.0.1:33418/callback"],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
                "scope": scope,
            },
        )
        response.raise_for_status()
        return response.json()

    def authorize(self, client_id: str, scope: str, approval_token: str) -> str:
        verifier = secrets.token_urlsafe(64)[:96]
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": "http://127.0.0.1:33418/callback",
            "scope": scope,
            "state": secrets.token_urlsafe(8),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "approval_token": approval_token,
        }
        response = self.http.get(f"{self.url}/authorize?{urlencode(params)}")
        if response.status_code != 302:
            raise RuntimeError(f"/authorize returned {response.status_code}: {response.text[:400]}")
        location = response.headers["location"]
        if "error=" in location:
            raise RuntimeError(f"/authorize error: {location}")
        code = location.split("code=", 1)[1].split("&", 1)[0]
        token_response = self.http.post(
            f"{self.url}/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "http://127.0.0.1:33418/callback",
                "client_id": client_id,
                "code_verifier": verifier,
            },
        )
        token_response.raise_for_status()
        self.access_token = token_response.json()["access_token"]
        return self.access_token

    def login(self, *, scope: str, approval_token: str) -> dict[str, Any]:
        discovery = self.discover()
        registration = self.register(scope)
        self.authorize(registration["client_id"], scope, approval_token)
        return {"discovery": discovery, "client_id": registration["client_id"]}

    # -- MCP --------------------------------------------------------------

    def rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._rpc_id += 1
        headers = dict(MCP_HEADERS)
        if self.access_token:
            headers["authorization"] = f"Bearer {self.access_token}"
        started = time.perf_counter()
        response = self.http.post(
            f"{self.url}/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": self._rpc_id,
                "method": method,
                "params": params or {},
            },
        )
        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            raise RuntimeError(f"{method} -> HTTP {response.status_code}: {response.text[:400]}")
        payload = _json_or_sse(response)
        if "error" in payload:
            raise RuntimeError(f"{method} -> JSON-RPC error: {payload['error']}")
        payload["_elapsed_s"] = elapsed
        return payload

    def initialize(self) -> dict[str, Any]:
        return self.rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "hars-memory-smoke", "version": "1.0"},
            },
        )

    def list_tools(self) -> list[str]:
        return [tool["name"] for tool in self.rpc("tools/list")["result"]["tools"]]

    def recall(self, question: str, top_k: int = 8) -> tuple[float, dict[str, Any]]:
        payload = self.rpc(
            "tools/call",
            {
                "name": "memory_recall",
                "arguments": {"question": question, "mode": "hybrid", "top_k": top_k},
            },
        )
        text = payload["result"]["content"][0]["text"]
        return payload["_elapsed_s"], json.loads(text)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--url", default=os.environ.get("HARS_MEMORY_MCP_URL", "http://127.0.0.1:8109")
    )
    parser.add_argument("--ca", default=None, help="CA bundle for TLS verification")
    parser.add_argument("--access-token", default=None, help="Skip the OAuth flow")
    parser.add_argument(
        "--approval-token", default=os.environ.get("HARS_MEMORY_MCP_OAUTH_APPROVAL_TOKEN")
    )
    parser.add_argument("--no-auth", action="store_true", help="No bearer token (legacy endpoint)")
    parser.add_argument("--scope", default="memory:read memory:write")
    parser.add_argument("--question", default="What projects are tracked in HARS memory?")
    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {
        "url": args.url,
        "clients": args.clients,
        "repeat": args.repeat,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    started = time.perf_counter()
    clients: list[Client] = []
    try:
        setup = Client(args.url, ca=args.ca, timeout=args.timeout)
        if args.no_auth:
            setup.access_token = None
        elif args.access_token:
            setup.access_token = args.access_token
        else:
            if not args.approval_token:
                parser.error("--approval-token (or --access-token / --no-auth) is required")
            login = setup.login(scope=args.scope, approval_token=args.approval_token)
            report["discovery"] = login["discovery"]
            report["client_id"] = login["client_id"]
        clients.append(setup)

        # One extra independent client per requested concurrency.
        for _ in range(max(0, args.clients - 1)):
            extra = Client(args.url, ca=args.ca, timeout=args.timeout)
            if not args.no_auth and not args.access_token:
                extra.login(scope=args.scope, approval_token=args.approval_token)  # type: ignore[arg-type]
            else:
                extra.access_token = setup.access_token
            clients.append(extra)

        init_latency: list[float] = []
        for client in clients:
            init_latency.append(client.initialize()["_elapsed_s"])

        report["tools"] = clients[0].list_tools()
        report["initialize_s"] = {
            "max": round(max(init_latency), 3),
            "p50": round(statistics.median(init_latency), 3),
        }

        latencies: list[float] = []
        first_call_latency: list[float] = []
        sample: dict[str, Any] = {}
        per_client: list[dict[str, Any]] = []
        for index, client in enumerate(clients):
            client_latencies: list[float] = []
            for run in range(args.repeat):
                elapsed, result = client.recall(args.question)
                latencies.append(elapsed)
                client_latencies.append(elapsed)
                if run == 0:
                    first_call_latency.append(elapsed)
                if index == 0 and run == 0:
                    sample = {
                        "ok": result.get("ok"),
                        "error": result.get("error"),
                        "context_chars": len(result.get("context") or ""),
                        "mode": result.get("mode"),
                        "lightrag_mode": result.get("lightrag_mode"),
                        "graph_branch": result.get("graph_branch"),
                        "hybrid": bool(result.get("hybrid")),
                    }
            per_client.append(
                {
                    "client": index,
                    "first_s": round(client_latencies[0], 3),
                    "p50_s": round(statistics.median(client_latencies), 3),
                }
            )

        report["recall"] = {
            "calls": len(latencies),
            "first_call_max_s": round(max(first_call_latency), 3),
            "p50_s": round(statistics.median(latencies), 3),
            "p95_s": round(_percentile(latencies, 0.95), 3),
            "min_s": round(min(latencies), 3),
            "max_s": round(max(latencies), 3),
            "per_client": per_client,
            "sample": sample,
        }
        report["total_s"] = round(time.perf_counter() - started, 2)
        report["ok"] = bool(sample.get("ok", True))
    finally:
        for client in clients:
            client.close()

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("ok", False) else 1


if __name__ == "__main__":
    sys.exit(main())
