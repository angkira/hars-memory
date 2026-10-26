"""Unit tests for the edge/engine split: socket protocol, router, forwarding.

No index, no network, no GPU: the engine handler is a fake in-process callable,
which is exactly the seam the split introduces.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
from typing import Any

import pytest
from mcp.types import TextContent

from hars_memory import mcp_engine
from hars_memory.mcp_engine import (
    EngineClient,
    EngineError,
    EngineRouter,
    EngineServer,
    EngineUnavailable,
    router_from_env,
)


def _run(coro):
    return asyncio.run(coro)


_MCP_HEADERS = {"accept": "application/json, text/event-stream"}


class FakeHandler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def __call__(self, name: str, arguments: Any) -> list[TextContent]:
        self.calls.append((name, arguments))
        if name == "explode":
            raise RuntimeError("handler blew up")
        if name == "slow":
            await asyncio.sleep(0.05)
        return [TextContent(type="text", text=json.dumps({"ok": True, "tool": name}))]


class TestEngineServerClient:
    def test_roundtrip(self, tmp_path):
        handler = FakeHandler()
        socket_path = tmp_path / "engine.sock"

        async def go():
            server = EngineServer(socket_path, handler=handler)
            await server.start()
            serve = asyncio.create_task(server.serve_forever())
            try:
                client = EngineClient(socket_path)
                content = await client.call_tool("memory_status", {"project": "default"})
                assert len(content) == 1
                assert json.loads(content[0].text)["tool"] == "memory_status"
                assert handler.calls == [("memory_status", {"project": "default"})]
                info = await client.info()
                assert info["pid"] > 0 and info["protocol"] == mcp_engine.PROTOCOL_VERSION
                await client.aclose()
            finally:
                serve.cancel()
                await asyncio.gather(serve, return_exceptions=True)
                await server.stop()

        _run(go())

    def test_socket_permissions_are_owner_only(self, tmp_path):
        socket_path = tmp_path / "engine.sock"

        async def go():
            server = EngineServer(socket_path, handler=FakeHandler())
            await server.start()
            try:
                mode = socket_path.stat().st_mode & 0o777
                assert oct(mode) == "0o600"
            finally:
                await server.stop()

        _run(go())

    def test_many_inflight_calls_on_one_connection(self, tmp_path):
        socket_path = tmp_path / "engine.sock"

        async def go():
            server = EngineServer(socket_path, handler=FakeHandler())
            await server.start()
            serve = asyncio.create_task(server.serve_forever())
            try:
                client = EngineClient(socket_path)
                results = await asyncio.gather(
                    *(client.call_tool("slow", {"i": i}) for i in range(12))
                )
                assert len(results) == 12
                assert server.requests == 12
                assert server.max_seen_inflight > 1
                await client.aclose()
            finally:
                serve.cancel()
                await asyncio.gather(serve, return_exceptions=True)
                await server.stop()

        _run(go())

    def test_handler_error_becomes_engine_error(self, tmp_path):
        socket_path = tmp_path / "engine.sock"

        async def go():
            server = EngineServer(socket_path, handler=FakeHandler())
            await server.start()
            serve = asyncio.create_task(server.serve_forever())
            try:
                client = EngineClient(socket_path)
                with pytest.raises(EngineError) as excinfo:
                    await client.call_tool("explode", {})
                assert "handler blew up" in excinfo.value.message
                assert not excinfo.value.overloaded
                await client.aclose()
            finally:
                serve.cancel()
                await asyncio.gather(serve, return_exceptions=True)
                await server.stop()

        _run(go())

    def test_unknown_method_is_rejected(self, tmp_path):
        socket_path = tmp_path / "engine.sock"

        async def go():
            server = EngineServer(socket_path, handler=FakeHandler())
            await server.start()
            serve = asyncio.create_task(server.serve_forever())
            try:
                client = EngineClient(socket_path)
                with pytest.raises(EngineError) as excinfo:
                    await client.call("nonsense")
                assert excinfo.value.code == mcp_engine.ERR_TOOL_FAILED
                await client.aclose()
            finally:
                serve.cancel()
                await asyncio.gather(serve, return_exceptions=True)
                await server.stop()

        _run(go())

    def test_saturation_is_reported_as_overloaded(self, tmp_path):
        socket_path = tmp_path / "engine.sock"

        async def go():
            server = EngineServer(socket_path, handler=FakeHandler(), max_inflight=1)
            await server.start()
            serve = asyncio.create_task(server.serve_forever())
            try:
                client = EngineClient(socket_path)
                first = asyncio.create_task(client.call_tool("slow", {}))
                await asyncio.sleep(0.01)
                with pytest.raises(EngineError) as excinfo:
                    await client.call_tool("slow", {})
                assert excinfo.value.overloaded
                await first
                await client.aclose()
            finally:
                serve.cancel()
                await asyncio.gather(serve, return_exceptions=True)
                await server.stop()

        _run(go())

    def test_stale_socket_file_is_replaced(self, tmp_path):
        socket_path = tmp_path / "engine.sock"
        # Simulate a crash: a socket file with nobody listening.
        import socket as _socket

        stale = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        stale.bind(str(socket_path))
        stale.close()

        async def go():
            server = EngineServer(socket_path, handler=FakeHandler())
            await server.start()
            await server.stop()

        _run(go())
        assert not socket_path.exists()

    def test_non_socket_path_is_not_deleted(self, tmp_path):
        socket_path = tmp_path / "engine.sock"
        socket_path.write_text("precious", encoding="utf-8")

        async def go():
            server = EngineServer(socket_path, handler=FakeHandler())
            with pytest.raises(RuntimeError):
                await server.start()

        _run(go())
        assert socket_path.read_text(encoding="utf-8") == "precious"

    def test_missing_socket_raises_engine_unavailable(self, tmp_path):
        client = EngineClient(tmp_path / "absent.sock", timeout_s=1.0)

        async def go():
            with pytest.raises((EngineUnavailable, FileNotFoundError)):
                await client.call_tool("memory_status", {})
            await client.aclose()

        _run(go())

    def test_ping_and_info_include_provider_block(self, tmp_path):
        socket_path = tmp_path / "engine.sock"

        async def go():
            server = EngineServer(
                socket_path,
                handler=FakeHandler(),
                info_provider=lambda: {"index_warm": True, "vector_storage": "QdrantVectorDBStorage"},
            )
            await server.start()
            serve = asyncio.create_task(server.serve_forever())
            try:
                client = EngineClient(socket_path)
                info = await client.info()
                assert info["index_warm"] is True
                assert info["vector_storage"] == "QdrantVectorDBStorage"
                assert (await client.call("ping"))["ok"] is True
                await client.aclose()
            finally:
                serve.cancel()
                await asyncio.gather(serve, return_exceptions=True)
                await server.stop()

        _run(go())


class TestRouter:
    def test_pool_round_robin(self, tmp_path):
        router = EngineRouter(["/a.sock", "/b.sock", "/c.sock"])
        first = router.client_for("default")
        second = router.client_for("default")
        third = router.client_for("default")
        fourth = router.client_for("default")
        assert {first.socket_path, second.socket_path, third.socket_path} == {
            "/a.sock", "/b.sock", "/c.sock"
        }
        assert fourth.socket_path == first.socket_path  # wrapped around

    def test_affinity_wins_over_the_pool(self):
        router = EngineRouter(["/a.sock"], {"proj2": "/b.sock"})
        assert router.client_for("proj2").socket_path == "/b.sock"
        assert router.client_for("other").socket_path == "/a.sock"

    def test_affinity_only(self):
        router = EngineRouter([], {"default": "/x.sock", "p2": "/y.sock"})
        assert router.client_for("default").socket_path == "/x.sock"
        assert router.client_for("p2").socket_path == "/y.sock"
        assert router.client_for("unknown").socket_path in {"/x.sock", "/y.sock"}

    def test_empty_router_returns_none(self):
        assert EngineRouter([]).client_for("default") is None

    def test_from_env_pool_and_affinity(self, monkeypatch):
        monkeypatch.setenv(
            "HARS_MEMORY_MCP_ENGINE_SOCKETS", "/run/a.sock, p2=/run/b.sock"
        )
        monkeypatch.setenv("HARS_MEMORY_MCP_ENGINE_SOCKET", "/run/single.sock")
        router = router_from_env()
        assert router is not None
        assert router.mapping == {"p2": "/run/b.sock"}
        assert router.sockets == ["/run/a.sock", "/run/single.sock"]

    def test_from_env_absent_is_combined_mode(self, monkeypatch):
        monkeypatch.delenv("HARS_MEMORY_MCP_ENGINE_SOCKETS", raising=False)
        monkeypatch.delenv("HARS_MEMORY_MCP_ENGINE_SOCKET", raising=False)
        assert router_from_env() is None

    def test_stats_shape(self):
        router = EngineRouter(["/a.sock"], {"p": "/b.sock"})
        router.client_for("p")
        stats = router.stats()
        assert stats["shards"] == 2
        assert stats["affinity"] == {"p": "/b.sock"}
        assert stats["clients"][0]["socket"] == "/b.sock"


# ---------------------------------------------------------------------------
# Forwarding from the MCP layer
# ---------------------------------------------------------------------------


class FakeRemoteClient:
    def __init__(self, *, fail: Exception | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.fail = fail
        self.requests = 0
        self.errors = 0
        self._writer = object()

    async def call_tool(self, name: str, arguments: Any) -> list[TextContent]:
        if self.fail is not None:
            raise self.fail
        self.calls.append((name, arguments))
        return [TextContent(type="text", text=json.dumps({"ok": True, "remote": True}))]

    async def aclose(self) -> None:
        return None


class FakeRouter:
    def __init__(self, client: FakeRemoteClient) -> None:
        self.client = client

    def client_for(self, project_id: str) -> FakeRemoteClient:
        return self.client

    def stats(self) -> dict[str, Any]:
        return {"shards": 1, "pool": ["/fake.sock"], "affinity": {}, "clients": [], "engines": {}}


@pytest.fixture
def app_factory(monkeypatch):
    from hars_memory.http_server import create_app
    from hars_memory.mcp_oauth import OAuthProvider, OAuthSettings

    BASE = "https://memory.test"
    provider = OAuthProvider(
        OAuthSettings(
            enabled=True,
            issuer=BASE,
            resource=f"{BASE}/mcp",
            auto_approve=True,
            state_file=None,
            signing_key="integration-secret-0123456789abcdef0123456789abcdef",
            access_ttl=600,
        )
    )
    return create_app(provider, warmup_project=None), BASE


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    return verifier, challenge


def _token(client, base: str, scope: str) -> str:
    redirect = "http://127.0.0.1:7777/callback"
    registration = client.post(
        "/register",
        json={
            "client_name": "split-test",
            "redirect_uris": [redirect],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            "scope": scope,
        },
    ).json()
    verifier, challenge = _pkce()
    response = client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": registration["client_id"],
            "redirect_uri": redirect,
            "scope": scope,
            "state": "s",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )
    code = response.headers["location"].split("code=", 1)[1].split("&", 1)[0]
    tokens = client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect,
            "client_id": registration["client_id"],
            "code_verifier": verifier,
        },
    ).json()
    return tokens["access_token"]


class TestEdgeForwarding:
    def test_tool_call_is_forwarded(self, monkeypatch, app_factory):
        from starlette.testclient import TestClient

        app, base = app_factory
        remote = FakeRemoteClient()
        monkeypatch.setattr("hars_memory.mcp_server.get_router", lambda: FakeRouter(remote))

        with TestClient(app, base_url=base) as client:
            token = _token(client, base, "memory:read memory:write")
            headers = {"authorization": f"Bearer {token}", **_MCP_HEADERS}
            init = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                 "clientInfo": {"name": "t", "version": "1"}}},
                headers=headers,
            )
            assert init.status_code == 200

            listing = client.post(
                "/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                headers=headers,
            )
            # tools/list is answered on the edge — no index, no engine call.
            assert listing.status_code == 200
            assert any(t["name"] == "memory_recall" for t in listing.json()["result"]["tools"])
            assert remote.calls == []

            call = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                      "params": {"name": "memory_recall", "arguments": {"question": "q"}}},
                headers=headers,
            )
            assert call.status_code == 200
            body = json.loads(call.json()["result"]["content"][0]["text"])
            assert body["remote"] is True
            assert remote.calls == [("memory_recall", {"question": "q"})]

    def test_scope_gate_runs_before_forwarding(self, monkeypatch, app_factory):
        from starlette.testclient import TestClient

        app, base = app_factory
        remote = FakeRemoteClient()
        monkeypatch.setattr("hars_memory.mcp_server.get_router", lambda: FakeRouter(remote))

        with TestClient(app, base_url=base) as client:
            token = _token(client, base, "memory:read")
            response = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                      "params": {"name": "memory_forget", "arguments": {"apply": True}}},
                headers={"authorization": f"Bearer {token}", **_MCP_HEADERS},
            )
            text = response.json()["result"]["content"][0]["text"]
            assert '"code": 403' in text
            assert remote.calls == []  # the engine never saw it

    def test_engine_unavailable_is_a_retryable_tool_error(self, monkeypatch, app_factory):
        from starlette.testclient import TestClient

        app, base = app_factory
        remote = FakeRemoteClient(fail=EngineUnavailable("socket gone"))
        monkeypatch.setattr("hars_memory.mcp_server.get_router", lambda: FakeRouter(remote))

        with TestClient(app, base_url=base) as client:
            token = _token(client, base, "memory:read")
            response = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                      "params": {"name": "memory_recall", "arguments": {"question": "q"}}},
                headers={"authorization": f"Bearer {token}", **_MCP_HEADERS},
            )
            body = json.loads(response.json()["result"]["content"][0]["text"])
            assert body["code"] == 503 and body["retryable"] is True
            assert body["error_type"] == "EngineUnavailableError"

    def test_overloaded_engine_is_retryable(self, monkeypatch, app_factory):
        from starlette.testclient import TestClient

        app, base = app_factory
        remote = FakeRemoteClient(
            fail=EngineError(mcp_engine.ERR_OVERLOADED, "engine at max in-flight")
        )
        monkeypatch.setattr("hars_memory.mcp_server.get_router", lambda: FakeRouter(remote))

        with TestClient(app, base_url=base) as client:
            token = _token(client, base, "memory:read")
            response = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                      "params": {"name": "memory_recall", "arguments": {"question": "q"}}},
                headers={"authorization": f"Bearer {token}", **_MCP_HEADERS},
            )
            body = json.loads(response.json()["result"]["content"][0]["text"])
            assert body["error_type"] == "OverloadedError" and body["retryable"] is True

    def test_combined_mode_still_executes_locally(self, monkeypatch, app_factory):
        from starlette.testclient import TestClient

        app, base = app_factory
        monkeypatch.setattr("hars_memory.mcp_server.get_router", lambda: None)
        with TestClient(app, base_url=base) as client:
            token = _token(client, base, "memory:read")
            listing = client.post(
                "/mcp", json={"jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {}},
                headers={"authorization": f"Bearer {token}", **_MCP_HEADERS},
            )
            assert listing.status_code == 200
