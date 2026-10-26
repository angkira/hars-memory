"""End-to-end OAuth flow against the composed ASGI app, fully in-process.

register -> authorize (PKCE S256) -> token -> /mcp initialize with the bearer
token; plus 401 without it, scope gating of destructive tools, refresh-token
rotation and revocation. No network, no real IdP, no index load.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from typing import Any

import pytest
from starlette.testclient import TestClient

from hars_memory.mcp_oauth import (
    SCOPE_ADMIN,
    SCOPE_READ,
    SCOPE_WRITE,
    OAuthProvider,
    OAuthSettings,
)


def create_app(provider, **kwargs):
    """Import the server lazily: mcp_server validates HARS_MEMORY_* at import
    time, and tests/conftest.py sets those per-test (not at collection)."""
    from hars_memory.http_server import create_app as _create_app

    return _create_app(provider, **kwargs)


BASE = "https://memory.test"
REDIRECT = "http://127.0.0.1:7777/callback"
PROTOCOL_VERSION = "2025-06-18"

MCP_HEADERS = {"accept": "application/json, text/event-stream"}


def make_provider(**overrides) -> OAuthProvider:
    settings = OAuthSettings(
        enabled=True,
        issuer=BASE,
        resource=f"{BASE}/mcp",
        auto_approve=True,
        state_file=None,
        signing_key="integration-secret-0123456789abcdef0123456789abcdef",
        access_ttl=600,
        **overrides,
    )
    return OAuthProvider(settings)


@pytest.fixture
def client():
    app = create_app(make_provider(), warmup_project=None, trace=None)
    with TestClient(app, base_url=BASE, raise_server_exceptions=False) as test_client:
        yield test_client


def rpc(
    client: TestClient,
    token: str | None,
    method: str,
    params: dict | None = None,
    request_id: int = 1,
) -> Any:
    headers = dict(MCP_HEADERS)
    if token:
        headers["authorization"] = f"Bearer {token}"
    return client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        },
        headers=headers,
    )


def initialize(client: TestClient, token: str | None):
    return rpc(
        client,
        token,
        "initialize",
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "oauth-integration-test", "version": "0.0.1"},
        },
    )


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    return verifier, challenge


def register(
    client: TestClient, *, scope: str = f"{SCOPE_READ} {SCOPE_WRITE}", auth_method: str = "none"
) -> dict:
    response = client.post(
        "/register",
        json={
            "client_name": "integration-test",
            "redirect_uris": [REDIRECT],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": auth_method,
            "scope": scope,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def authorize(client: TestClient, *, client_id: str, scope: str = f"{SCOPE_READ} {SCOPE_WRITE}"):
    verifier, challenge = pkce_pair()
    response = client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": REDIRECT,
            "scope": scope,
            "state": "state-123",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )
    return response, verifier


def exchange_code(
    client: TestClient, *, code: str, client_id: str, verifier: str, secret: str | None = None
) -> dict:
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT,
        "client_id": client_id,
        "code_verifier": verifier,
    }
    if secret:
        data["client_secret"] = secret
    response = client.post("/token", data=data)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["token_type"] == "Bearer"
    assert body["expires_in"] == 600
    return body


def full_flow(client: TestClient, *, scope: str = f"{SCOPE_READ} {SCOPE_WRITE}") -> dict:
    registration = register(client, scope=scope)
    response, verifier = authorize(client, client_id=registration["client_id"], scope=scope)
    assert response.status_code == 302, response.text
    location = response.headers["location"]
    assert location.startswith(REDIRECT)
    assert "state=state-123" in location
    code = location.split("code=", 1)[1].split("&", 1)[0]
    tokens = exchange_code(
        client, code=code, client_id=registration["client_id"], verifier=verifier
    )
    tokens["client_id"] = registration["client_id"]
    return tokens


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscoveryEndpoints:
    def test_protected_resource(self, client):
        response = client.get("/.well-known/oauth-protected-resource")
        assert response.status_code == 200
        body = response.json()
        assert body["resource"] == f"{BASE}/mcp"
        assert body["authorization_servers"] == [BASE]

    def test_protected_resource_with_path_suffix(self, client):
        # RFC 9728 §3.1: the path-suffixed variant must also resolve.
        assert client.get("/.well-known/oauth-protected-resource/mcp").status_code == 200

    def test_authorization_server(self, client):
        response = client.get("/.well-known/oauth-authorization-server")
        assert response.status_code == 200
        body = response.json()
        assert body["issuer"] == BASE
        assert body["token_endpoint"] == f"{BASE}/token"
        assert body["code_challenge_methods_supported"] == ["S256"]

    def test_authorization_server_with_path_suffix(self, client):
        response = client.get("/.well-known/oauth-authorization-server/mcp")
        assert response.status_code == 200

    def test_healthz_is_public(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["oauth"] is True
        assert body["index_shared"] is True


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


class TestOAuthFlow:
    def test_register_authorize_token_initialize(self, client):
        tokens = full_flow(client)
        response = initialize(client, tokens["access_token"])
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["result"]["serverInfo"]["name"] == "hars-longterm-memory"
        assert "tools" in body["result"]["capabilities"]

    def test_initialize_returns_mcp_session_id(self, client):
        """Stateless mode assigns no session; the gateway still must return one
        because Streamable HTTP clients (the memory visualizer) require it."""
        tokens = full_flow(client)
        response = initialize(client, tokens["access_token"])
        assert response.status_code == 200, response.text
        session_id = response.headers.get("mcp-session-id")
        assert session_id
        assert len(session_id) == 32

        follow = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers={
                **MCP_HEADERS,
                "authorization": f"Bearer {tokens['access_token']}",
                "mcp-session-id": session_id,
            },
        )
        assert follow.status_code == 200, follow.text
        assert "tools" in follow.json()["result"]

    def test_missing_bearer_is_401_with_resource_metadata(self, client):
        response = initialize(client, None)
        assert response.status_code == 401
        challenge = response.headers["www-authenticate"]
        assert challenge.startswith("Bearer")
        assert f'resource_metadata="{BASE}/.well-known/oauth-protected-resource"' in challenge
        assert response.json()["error"] == "unauthorized"

    def test_invalid_bearer_is_401(self, client):
        response = initialize(client, "not.a.jwt")
        assert response.status_code == 401
        assert 'error="invalid_token"' in response.headers["www-authenticate"]

    def test_token_from_another_server_is_rejected(self, client):
        foreign = OAuthProvider(
            OAuthSettings(
                enabled=True,
                issuer="https://other.test",
                resource="https://other.test/mcp",
                auto_approve=True,
                state_file=None,
                signing_key="integration-secret-0123456789abcdef0123456789abcdef",
            )
        )
        token = foreign.issue_token_pair(
            "operator", "c1", [SCOPE_READ], "https://other.test", with_refresh=False
        )["access_token"]
        assert initialize(client, token).status_code == 401

    def test_tools_list_hides_admin_tools(self, client):
        tokens = full_flow(client)
        response = rpc(client, tokens["access_token"], "tools/list")
        assert response.status_code == 200
        names = {tool["name"] for tool in response.json()["result"]["tools"]}
        assert "memory_recall" in names
        assert "memory_remember" in names
        assert "memory_forget" not in names
        assert "memory_delete_document" not in names
        assert "memory_consolidate" not in names

    def test_destructive_tool_is_scope_gated(self, client):
        tokens = full_flow(client)
        response = rpc(
            client,
            tokens["access_token"],
            "tools/call",
            {"name": "memory_forget", "arguments": {"apply": True}},
        )
        assert response.status_code == 200  # MCP-level success, tool-level refusal
        content = response.json()["result"]["content"][0]["text"]
        assert '"code": 403' in content.replace("\n", " ")
        assert '"required_scope": "memory:admin"' in content.replace("\n", " ")

    def test_read_only_token_cannot_write(self, client):
        tokens = full_flow(client, scope=SCOPE_READ)
        listing = rpc(client, tokens["access_token"], "tools/list")
        names = {tool["name"] for tool in listing.json()["result"]["tools"]}
        assert "memory_recall" in names
        assert "memory_remember" not in names

        response = rpc(
            client,
            tokens["access_token"],
            "tools/call",
            {"name": "memory_remember", "arguments": {"content": "nope"}},
        )
        content = response.json()["result"]["content"][0]["text"].replace("\n", " ")
        assert '"required_scope": "memory:write"' in content

    def test_admin_scope_exposes_and_allows(self):
        # This app allows admin in max_scopes; destructive tools become visible.
        app = create_app(
            make_provider(max_scopes=(SCOPE_READ, SCOPE_WRITE, SCOPE_ADMIN)),
            warmup_project=None,
        )
        with TestClient(app, base_url=BASE, raise_server_exceptions=False) as admin_client:
            tokens = full_flow(admin_client, scope=f"{SCOPE_READ} {SCOPE_ADMIN}")
            listing = rpc(admin_client, tokens["access_token"], "tools/list")
            names = {tool["name"] for tool in listing.json()["result"]["tools"]}
            assert "memory_forget" in names
            assert "memory_remember" in names  # admin implies write

    def test_pkce_is_mandatory(self, client):
        registration = register(client)
        response = client.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": registration["client_id"],
                "redirect_uri": REDIRECT,
                "state": "s",
            },
            follow_redirects=False,
        )
        assert response.status_code == 302
        assert "error=invalid_request" in response.headers["location"]

    def test_authorize_rejects_unregistered_redirect(self, client):
        registration = register(client)
        response = client.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": registration["client_id"],
                "redirect_uri": "http://127.0.0.1:9999/evil",
                "code_challenge": "x" * 43,
                "code_challenge_method": "S256",
            },
            follow_redirects=False,
        )
        assert response.status_code == 400

    def test_token_rejects_wrong_verifier(self, client):
        registration = register(client)
        response, verifier = authorize(client, client_id=registration["client_id"])
        code = response.headers["location"].split("code=", 1)[1].split("&", 1)[0]
        bad = client.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT,
                "client_id": registration["client_id"],
                "code_verifier": "z" * 64,
            },
        )
        assert bad.status_code == 400
        assert bad.json()["error"] == "invalid_grant"

    def test_refresh_rotation_and_revocation(self, client):
        tokens = full_flow(client)
        refreshed = client.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "client_id": tokens["client_id"],
            },
        )
        assert refreshed.status_code == 200
        new_tokens = refreshed.json()
        assert new_tokens["refresh_token"] != tokens["refresh_token"]
        assert initialize(client, new_tokens["access_token"]).status_code == 200

        # Old refresh token was rotated out.
        reuse = client.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "client_id": tokens["client_id"],
            },
        )
        assert reuse.status_code == 400

        # Revoke the new refresh token, then it stops working.
        revoked = client.post(
            "/revoke",
            data={"token": new_tokens["refresh_token"], "client_id": tokens["client_id"]},
        )
        assert revoked.status_code == 200
        after = client.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": new_tokens["refresh_token"],
                "client_id": tokens["client_id"],
            },
        )
        assert after.status_code == 400

    def test_confidential_client_requires_secret(self, client):
        registration = register(client, auth_method="client_secret_post")
        response, verifier = authorize(client, client_id=registration["client_id"])
        code = response.headers["location"].split("code=", 1)[1].split("&", 1)[0]
        missing = client.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT,
                "client_id": registration["client_id"],
                "code_verifier": verifier,
            },
        )
        assert missing.status_code == 401

    def test_confidential_client_with_secret(self, client):
        registration = register(client, auth_method="client_secret_post")
        response, verifier = authorize(client, client_id=registration["client_id"])
        code = response.headers["location"].split("code=", 1)[1].split("&", 1)[0]
        tokens = exchange_code(
            client,
            code=code,
            client_id=registration["client_id"],
            verifier=verifier,
            secret=registration["client_secret"],
        )
        assert initialize(client, tokens["access_token"]).status_code == 200

    def test_cors_preflight(self, client):
        response = client.options(
            "/mcp",
            headers={
                "origin": "https://claude.ai",
                "access-control-request-method": "POST",
                "access-control-request-headers": "authorization,content-type",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] in {"*", "https://claude.ai"}


# ---------------------------------------------------------------------------
# OAuth disabled (legacy deployment) still serves /mcp
# ---------------------------------------------------------------------------


class TestOAuthDisabled:
    def test_mcp_works_without_token(self):
        app = create_app(OAuthProvider(OAuthSettings(enabled=False)), warmup_project=None)
        with TestClient(
            app, base_url="http://127.0.0.1:8099", raise_server_exceptions=False
        ) as legacy:
            response = initialize(legacy, None)
            assert response.status_code == 200
            assert legacy.get("/.well-known/oauth-authorization-server").status_code == 404
            assert legacy.get("/healthz").json()["oauth"] is False
