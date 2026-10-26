"""Unit tests for the OAuth 2.1 authorization server (hars_memory.mcp_oauth).

No network, no real IdP: everything here is pure functions, an in-memory
provider, or a Starlette ``TestClient`` over the composed ASGI app.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jwt
import pytest

from hars_memory.mcp_oauth import (
    ALL_SCOPES,
    SCOPE_ADMIN,
    SCOPE_READ,
    SCOPE_WRITE,
    Grant,
    OAuthProvider,
    OAuthSettings,
    pkce_s256,
    required_scope_for_tool,
    resolve_signing_material,
)

BASE = "https://memory.test"


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def make_settings(**overrides) -> OAuthSettings:
    defaults = dict(
        enabled=True,
        issuer=BASE,
        resource=f"{BASE}/mcp",
        auto_approve=True,
        state_file=None,
        signing_key="unit-test-secret-0123456789abcdef0123456789abcdef",
        access_ttl=600,
    )
    defaults.update(overrides)
    return OAuthSettings(**defaults)


@pytest.fixture
def provider() -> OAuthProvider:
    return OAuthProvider(make_settings())


def register(provider: OAuthProvider, **overrides) -> dict:
    payload = {
        "client_name": "unit-test",
        "redirect_uris": ["http://127.0.0.1:7777/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "scope": "memory:read memory:write",
    }
    payload.update(overrides)
    body, status = provider.register_client(payload, BASE)
    assert status == 201, body
    return body


def make_grant(scopes: set[str]) -> Grant:
    return Grant(
        subject="operator",
        client_id="c1",
        scopes=frozenset(scopes),
        jti="at_test",
        expires_at=int(time.time()) + 600,
    )


# ---------------------------------------------------------------------------
# PKCE and scopes
# ---------------------------------------------------------------------------


class TestPkce:
    def test_rfc7636_test_vector(self):
        verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        assert pkce_s256(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"

    def test_challenge_is_unpadded_base64url(self):
        challenge = pkce_s256("a" * 64)
        assert len(challenge) == 43
        assert "=" not in challenge
        assert all(ch.isalnum() or ch in "-_" for ch in challenge)


class TestScopes:
    def test_hierarchy(self):
        read = make_grant({SCOPE_READ})
        write = make_grant({SCOPE_WRITE})
        admin = make_grant({SCOPE_ADMIN})
        assert read.has_scope(SCOPE_READ) and not read.has_scope(SCOPE_WRITE)
        assert write.has_scope(SCOPE_WRITE) and write.has_scope(SCOPE_READ)
        assert not write.has_scope(SCOPE_ADMIN)
        assert admin.has_scope(SCOPE_ADMIN) and admin.has_scope(SCOPE_WRITE)

    def test_tool_policy(self):
        assert required_scope_for_tool("memory_recall") == SCOPE_READ
        assert required_scope_for_tool("memory_remember") == SCOPE_WRITE
        assert required_scope_for_tool("memory_forget") == SCOPE_ADMIN
        # Unknown tools fail closed at write level.
        assert required_scope_for_tool("memory_unknown") == SCOPE_WRITE

    def test_scope_order_constant(self):
        assert ALL_SCOPES == ("memory:read", "memory:write", "memory:admin")


# ---------------------------------------------------------------------------
# Settings and signing
# ---------------------------------------------------------------------------


class TestSettings:
    def test_defaults_are_disabled(self, monkeypatch):
        for name in list(__import__("os").environ):
            if name.startswith("HARS_MEMORY_MCP_OAUTH_"):
                monkeypatch.delenv(name, raising=False)
        settings = OAuthSettings.from_env()
        assert settings.enabled is False
        assert settings.max_scopes == (SCOPE_READ, SCOPE_WRITE)

    def test_enabled_without_credential_fails_closed(self):
        with pytest.raises(RuntimeError, match="no approval credential"):
            OAuthProvider(make_settings(auto_approve=False))

    def test_enabled_with_password_is_allowed(self):
        provider = OAuthProvider(make_settings(auto_approve=False, authorize_password="s3cret"))
        assert provider.enabled

    def test_max_scopes_must_be_supported(self):
        with pytest.raises(ValueError, match="not in"):
            OAuthProvider(make_settings(max_scopes=("memory:root",)))

    def test_env_parsing(self, monkeypatch):
        monkeypatch.setenv("HARS_MEMORY_MCP_OAUTH_ENABLED", "1")
        monkeypatch.setenv("HARS_MEMORY_MCP_OAUTH_ISSUER", "https://m.example")
        monkeypatch.setenv("HARS_MEMORY_MCP_OAUTH_APPROVAL_TOKEN", "tok")
        monkeypatch.setenv("HARS_MEMORY_MCP_OAUTH_MAX_SCOPES", "memory:read")
        monkeypatch.setenv("HARS_MEMORY_MCP_CORS_ORIGINS", "https://a.example,https://b.example")
        settings = OAuthSettings.from_env()
        assert settings.enabled
        assert settings.issuer == "https://m.example"
        assert settings.approval_token == "tok"
        assert settings.max_scopes == (SCOPE_READ,)
        assert settings.cors_origins == ("https://a.example", "https://b.example")


class TestSigning:
    def test_hs256_secret(self):
        material = resolve_signing_material(make_settings())
        assert material.algorithm == "HS256"
        assert material.kid == "hs1"

    def test_missing_key_file_is_fatal(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            resolve_signing_material(
                make_settings(signing_key=None, signing_key_file=tmp_path / "missing.key")
            )

    def test_generated_key_is_persisted_0600(self, tmp_path: Path, monkeypatch):
        import hars_memory.mcp_oauth as oauth_module

        monkeypatch.setattr(oauth_module, "DEFAULT_SIGNING_KEY_FILE", tmp_path / "signing.key")
        material = resolve_signing_material(make_settings(signing_key=None))
        assert material.algorithm == "EdDSA"
        key_file = tmp_path / "signing.key"
        assert key_file.is_file()
        assert (key_file.stat().st_mode & 0o777) == 0o600
        # Reloading yields the same key id (tokens survive a restart).
        again = resolve_signing_material(make_settings(signing_key=None))
        assert again.kid == material.kid


# ---------------------------------------------------------------------------
# Discovery payloads
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_protected_resource_metadata(self, provider):
        body = provider.protected_resource_metadata(BASE)
        assert body["resource"] == f"{BASE}/mcp"
        assert body["authorization_servers"] == [BASE]
        assert body["bearer_methods_supported"] == ["header"]
        assert set(body["scopes_supported"]) == set(ALL_SCOPES)

    def test_authorization_server_metadata(self, provider):
        body = provider.authorization_server_metadata(BASE)
        assert body["issuer"] == BASE
        assert body["authorization_endpoint"] == f"{BASE}/authorize"
        assert body["token_endpoint"] == f"{BASE}/token"
        assert body["registration_endpoint"] == f"{BASE}/register"
        assert body["revocation_endpoint"] == f"{BASE}/revoke"
        assert body["response_types_supported"] == ["code"]
        assert body["code_challenge_methods_supported"] == ["S256"]
        assert "authorization_code" in body["grant_types_supported"]
        assert "refresh_token" in body["grant_types_supported"]

    def test_registration_endpoint_hidden_when_disabled(self):
        provider = OAuthProvider(make_settings(registration_enabled=False))
        body = provider.authorization_server_metadata(BASE)
        assert "registration_endpoint" not in body

    def test_issuer_derived_from_baseline_when_unset(self):
        provider = OAuthProvider(make_settings(issuer=None, resource=None))
        assert provider.issuer_for("http://10.0.0.5:8099") == "http://10.0.0.5:8099"
        assert provider.resource_for("http://10.0.0.5:8099") == "http://10.0.0.5:8099/mcp"


# ---------------------------------------------------------------------------
# Dynamic client registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_public_client(self, provider):
        body = register(provider)
        assert body["client_id"].startswith("hmc_")
        assert "client_secret" not in body
        assert body["token_endpoint_auth_method"] == "none"
        assert provider.get_client(body["client_id"]) is not None

    def test_confidential_client_gets_secret(self, provider):
        body = register(provider, token_endpoint_auth_method="client_secret_post")
        assert body["client_secret"]
        stored = provider.get_client(body["client_id"])
        assert stored.client_secret_hash is not None
        assert body["client_secret"] not in json.dumps(stored.to_dict())

    def test_rejects_plain_http_non_loopback(self, provider):
        body, status = provider.register_client({"redirect_uris": ["http://evil.example/cb"]}, BASE)
        assert status == 400
        assert body["error"] == "invalid_redirect_uri"

    def test_rejects_fragment(self, provider):
        body, status = provider.register_client(
            {"redirect_uris": ["https://good.example/cb#frag"]}, BASE
        )
        assert status == 400

    def test_rejects_scope_above_max(self, provider):
        body, status = provider.register_client(
            {"redirect_uris": ["http://127.0.0.1:1/cb"], "scope": "memory:admin"}, BASE
        )
        assert status == 400
        assert body["error"] == "invalid_scope"

    def test_rejects_unknown_grant_type(self, provider):
        body, status = provider.register_client(
            {"redirect_uris": ["http://127.0.0.1:1/cb"], "grant_types": ["password"]}, BASE
        )
        assert status == 400

    def test_allowed_redirect_hosts(self):
        provider = OAuthProvider(make_settings(allowed_redirect_hosts=("claude.ai",)))
        body, status = provider.register_client(
            {"redirect_uris": ["https://evil.example/cb"]}, BASE
        )
        assert status == 400
        body, status = provider.register_client({"redirect_uris": ["https://claude.ai/cb"]}, BASE)
        assert status == 201

    def test_registration_disabled(self):
        provider = OAuthProvider(make_settings(registration_enabled=False))
        body, status = provider.register_client({"redirect_uris": ["http://127.0.0.1:1/cb"]}, BASE)
        assert status == 403

    def test_state_survives_restart(self, tmp_path: Path):
        state_file = tmp_path / "state.json"
        provider = OAuthProvider(make_settings(state_file=state_file))
        body = register(provider)
        assert state_file.is_file()
        assert (state_file.stat().st_mode & 0o777) == 0o600
        reopened = OAuthProvider(make_settings(state_file=state_file))
        assert reopened.get_client(body["client_id"]) is not None


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


class TestTokens:
    def test_token_is_bound_to_issuer_and_audience(self):
        # With a dynamic origin (issuer unset) the aud/iss claims pin the token
        # to the origin that issued it.
        provider = OAuthProvider(make_settings(issuer=None, resource=None))
        token = provider.issue_token_pair(
            "operator", "c1", [SCOPE_READ], "https://a.test", with_refresh=False
        )["access_token"]
        grant, error = provider.validate_access_token(token, "https://a.test")
        assert error is None and grant is not None

        grant, error = provider.validate_access_token(token, "https://b.test")
        assert grant is None and error == "invalid_token"

    def test_garbage_token(self, provider):
        assert provider.validate_access_token("not-a-jwt", BASE)[0] is None

    def test_expired_token(self, provider):
        body = register(provider)
        token = provider.issue_token_pair(
            "operator", body["client_id"], [SCOPE_READ], BASE, with_refresh=False
        )["access_token"]
        # Re-sign the same claims with an exp in the past.
        claims = jwt.decode(token, options={"verify_signature": False})
        claims["exp"] = int(time.time()) - 5
        expired = jwt.encode(
            claims,
            provider.signing.encode_key,
            algorithm="HS256",
            headers={"typ": "at+jwt", "kid": provider.signing.kid},
        )
        assert provider.validate_access_token(expired, BASE)[0] is None

    def test_wrong_signing_key(self, provider):
        body = register(provider)
        token = provider.issue_token_pair(
            "operator", body["client_id"], [SCOPE_READ], BASE, with_refresh=False
        )["access_token"]
        other = OAuthProvider(make_settings(signing_key="a-different-secret-0123456789abcdef"))
        assert other.validate_access_token(token, BASE)[0] is None

    def test_revoked_access_token(self, provider):
        body = register(provider)
        client = provider.get_client(body["client_id"])
        issued = provider.issue_token_pair(
            "operator", body["client_id"], [SCOPE_READ], BASE, with_refresh=False
        )
        provider.revoke_token(issued["access_token"], client)
        assert provider.validate_access_token(issued["access_token"], BASE)[0] is None

    def test_code_reuse_is_rejected(self, provider):
        body = register(provider)
        client = provider.get_client(body["client_id"])
        verifier = "v" * 64
        code = provider._issue_code(
            client,
            "http://127.0.0.1:7777/callback",
            [SCOPE_READ],
            "operator",
            pkce_s256(verifier),
        )
        first, status = provider.exchange_code(
            code=code,
            client=client,
            redirect_uri="http://127.0.0.1:7777/callback",
            verifier=verifier,
            base=BASE,
        )
        assert status == 200 and first["access_token"]
        second, status = provider.exchange_code(
            code=code,
            client=client,
            redirect_uri="http://127.0.0.1:7777/callback",
            verifier=verifier,
            base=BASE,
        )
        assert status == 400 and second["error"] == "invalid_grant"

    def test_wrong_pkce_verifier(self, provider):
        body = register(provider)
        client = provider.get_client(body["client_id"])
        code = provider._issue_code(
            client,
            "http://127.0.0.1:7777/callback",
            [SCOPE_READ],
            "operator",
            pkce_s256("v" * 64),
        )
        out, status = provider.exchange_code(
            code=code,
            client=client,
            redirect_uri="http://127.0.0.1:7777/callback",
            verifier="w" * 64,
            base=BASE,
        )
        assert status == 400 and out["error"] == "invalid_grant"

    def test_redirect_uri_mismatch(self, provider):
        body = register(provider)
        client = provider.get_client(body["client_id"])
        code = provider._issue_code(
            client,
            "http://127.0.0.1:7777/callback",
            [SCOPE_READ],
            "operator",
            pkce_s256("v" * 64),
        )
        out, status = provider.exchange_code(
            code=code,
            client=client,
            redirect_uri="http://127.0.0.1:9999/callback",
            verifier="v" * 64,
            base=BASE,
        )
        assert status == 400

    def test_refresh_rotation(self, provider):
        body = register(provider)
        client = provider.get_client(body["client_id"])
        issued = provider.issue_token_pair("operator", client.client_id, [SCOPE_READ], BASE)
        refresh = issued["refresh_token"]
        out, status = provider.exchange_refresh(
            refresh_token=refresh, client=client, scope=None, base=BASE
        )
        assert status == 200 and out["refresh_token"] != refresh
        # The old refresh token is single-use.
        out2, status2 = provider.exchange_refresh(
            refresh_token=refresh, client=client, scope=None, base=BASE
        )
        assert status2 == 400 and out2["error"] == "invalid_grant"

    def test_refresh_cannot_escalate_scope(self, provider):
        body = register(provider)
        client = provider.get_client(body["client_id"])
        issued = provider.issue_token_pair("operator", client.client_id, [SCOPE_READ], BASE)
        out, status = provider.exchange_refresh(
            refresh_token=issued["refresh_token"],
            client=client,
            scope=SCOPE_ADMIN,
            base=BASE,
        )
        assert status == 400 and out["error"] == "invalid_scope"

    def test_refresh_bound_to_client(self, provider):
        first = register(provider)
        second = register(provider, client_name="other")
        client = provider.get_client(first["client_id"])
        issued = provider.issue_token_pair("operator", client.client_id, [SCOPE_READ], BASE)
        other = provider.get_client(second["client_id"])
        out, status = provider.exchange_refresh(
            refresh_token=issued["refresh_token"], client=other, scope=None, base=BASE
        )
        assert status == 400

    def test_token_typ_and_claims(self, provider):
        body = register(provider)
        token = provider.issue_token_pair(
            "operator", body["client_id"], [SCOPE_READ, SCOPE_WRITE], BASE, with_refresh=False
        )["access_token"]
        header = jwt.get_unverified_header(token)
        claims = jwt.decode(token, options={"verify_signature": False})
        assert header["typ"] == "at+jwt"
        assert claims["aud"] == f"{BASE}/mcp"
        assert claims["iss"] == BASE
        assert claims["scope"] == "memory:read memory:write"
        assert claims["token_use"] == "access"
        assert claims["client_id"] == body["client_id"]
