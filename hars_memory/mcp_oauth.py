"""OAuth 2.1 authorization server for the native HTTP MCP endpoint.

Why this exists
---------------
The memory MCP server can be served natively over Streamable HTTP by
:mod:`hars_memory.http_server`. That endpoint may be reachable by agents that are
not on this host, so it needs an authorization story that the MCP spec
(2025-06-18, "Authorization") actually describes:

* the MCP server is an OAuth 2.1 **resource server** — every ``/mcp`` request
  needs ``Authorization: Bearer <access token>``;
* this module is also the **authorization server** (RFC 8414 metadata,
  RFC 9728 protected-resource metadata, RFC 7591 dynamic client registration,
  authorization-code + PKCE S256, refresh tokens, RFC 7009 revocation).

Design notes
------------
* **Self-contained.** :mod:`hars_memory.auth.jwt` exists, but its tokens carry no
  ``aud`` claim and its issuer is a fixed ``hars-memory``; mixing it into an
  RFC 9068 access token would silently drop the audience binding the MCP spec
  requires. This module therefore signs its own tokens and only reuses the
  house style (Ed25519 by default, encrypted-at-rest secrets elsewhere).
* **Fail closed.** Enabling OAuth without an approval credential (password,
  pre-shared approval token, or explicit ``AUTO_APPROVE``) raises at startup
  instead of exposing an unauthenticated authorization endpoint.
* **Scopes are hierarchical.** ``memory:admin`` ⊃ ``memory:write`` ⊃
  ``memory:read``. Read-only tools are visible/usable with ``memory:read``;
  mutating tools need ``memory:write``; the destructive ones need
  ``memory:admin``. ``tools/list`` hides what the caller cannot call.
* **Persistence.** Dynamic clients, hashed refresh tokens and the access-token
  revocation set live in a 0600 JSON file so a service restart does not force
  every client to re-register and re-consent. Authorization codes stay in
  memory (they live for ``CODE_TTL`` seconds by design).
* **No external IdP, no network calls.** Everything is local, which also makes
  the integration test hermetic.

All configuration is environment-driven; see ``docs/http-mcp.md`` for the table.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import html
import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final
from urllib.parse import urlencode, urlparse

import jwt
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

logger = logging.getLogger("memory.oauth")

# ---------------------------------------------------------------------------
# Scopes and tool policy
# ---------------------------------------------------------------------------

SCOPE_READ: Final[str] = "memory:read"
SCOPE_WRITE: Final[str] = "memory:write"
SCOPE_ADMIN: Final[str] = "memory:admin"
ALL_SCOPES: Final[tuple[str, ...]] = (SCOPE_READ, SCOPE_WRITE, SCOPE_ADMIN)

#: Tool -> minimum scope. Unknown tools default to ``memory:write`` (fail closed).
TOOL_SCOPES: Final[dict[str, str]] = {
    "memory_recall": SCOPE_READ,
    "memory_status": SCOPE_READ,
    "memory_entities": SCOPE_READ,
    "memory_related": SCOPE_READ,
    "memory_inspect_entity": SCOPE_READ,
    "memory_sync_status": SCOPE_READ,
    "memory_session_status": SCOPE_READ,
    "memory_list_projects": SCOPE_READ,
    "memory_remember": SCOPE_WRITE,
    "memory_compact": SCOPE_WRITE,
    "memory_merge_session": SCOPE_WRITE,
    "memory_upsert_document": SCOPE_WRITE,
    "memory_delete_document": SCOPE_ADMIN,
    "memory_forget": SCOPE_ADMIN,
    "memory_consolidate": SCOPE_ADMIN,
}

#: ASGI ``scope`` key carrying the validated :class:`Grant` into the MCP layer.
GRANT_SCOPE_KEY: Final[str] = "hars_memory.oauth.grant"

# --- env var names (public: documented in docs/http-mcp.md) -----------------
ENV_ENABLED = "HARS_MEMORY_MCP_OAUTH_ENABLED"
ENV_ISSUER = "HARS_MEMORY_MCP_OAUTH_ISSUER"
ENV_RESOURCE = "HARS_MEMORY_MCP_OAUTH_RESOURCE"
ENV_SCOPES = "HARS_MEMORY_MCP_OAUTH_SCOPES"
ENV_MAX_SCOPES = "HARS_MEMORY_MCP_OAUTH_MAX_SCOPES"
ENV_ACCESS_TTL = "HARS_MEMORY_MCP_OAUTH_ACCESS_TTL"
ENV_REFRESH_TTL = "HARS_MEMORY_MCP_OAUTH_REFRESH_TTL"
ENV_CODE_TTL = "HARS_MEMORY_MCP_OAUTH_CODE_TTL"
ENV_STATE_FILE = "HARS_MEMORY_MCP_OAUTH_STATE_FILE"
ENV_SIGNING_KEY = "HARS_MEMORY_MCP_OAUTH_SIGNING_KEY"
ENV_SIGNING_KEY_FILE = "HARS_MEMORY_MCP_OAUTH_SIGNING_KEY_FILE"
ENV_AUTHORIZE_USER = "HARS_MEMORY_MCP_OAUTH_AUTHORIZE_USER"
ENV_AUTHORIZE_PASSWORD = "HARS_MEMORY_MCP_OAUTH_AUTHORIZE_PASSWORD"
ENV_APPROVAL_TOKEN = "HARS_MEMORY_MCP_OAUTH_APPROVAL_TOKEN"
ENV_AUTO_APPROVE = "HARS_MEMORY_MCP_OAUTH_AUTO_APPROVE"
ENV_REGISTRATION_ENABLED = "HARS_MEMORY_MCP_OAUTH_REGISTRATION_ENABLED"
ENV_REGISTRATION_TOKEN = "HARS_MEMORY_MCP_OAUTH_REGISTRATION_TOKEN"
ENV_CLIENTS_FILE = "HARS_MEMORY_MCP_OAUTH_CLIENTS_FILE"
ENV_ALLOWED_REDIRECT_HOSTS = "HARS_MEMORY_MCP_OAUTH_ALLOWED_REDIRECT_HOSTS"
ENV_CORS_ORIGINS = "HARS_MEMORY_MCP_CORS_ORIGINS"
ENV_STORE = "HARS_MEMORY_MCP_OAUTH_STORE"
ENV_STORE_PATH = "HARS_MEMORY_MCP_OAUTH_STORE_PATH"
ENV_STORE_URL = "HARS_MEMORY_MCP_OAUTH_STORE_URL"
ENV_REVOCATION_CACHE_TTL = "HARS_MEMORY_MCP_OAUTH_REVOCATION_CACHE_TTL"

DEFAULT_STATE_DIR: Final[Path] = Path.home() / ".local" / "share" / "hars-longterm-memory" / "oauth"
DEFAULT_STATE_FILE: Final[Path] = DEFAULT_STATE_DIR / "state.json"
DEFAULT_STORE_PATH: Final[Path] = DEFAULT_STATE_DIR / "state.sqlite3"
DEFAULT_SIGNING_KEY_FILE: Final[Path] = DEFAULT_STATE_DIR / "signing.key"

DEFAULT_ACCESS_TTL: Final[int] = 3600
DEFAULT_REFRESH_TTL: Final[int] = 30 * 24 * 3600
DEFAULT_CODE_TTL: Final[int] = 300

_PKCE_VERIFIER_RE: Final[str] = r"^[A-Za-z0-9\-._~]{43,128}$"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r; using %s", name, raw, default)
        return default


def _list_env(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [part for part in raw.replace(",", " ").split() if part]


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r; using %s", name, raw, default)
        return default


def b64url(data: bytes) -> str:
    """Unpadded base64url (RFC 7636 / RFC 7515)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def pkce_s256(verifier: str) -> str:
    """The S256 code challenge for a verifier."""
    return b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> int:
    return int(time.time())


def _iso(ts: int) -> str:
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()


def required_scope_for_tool(name: str) -> str:
    """Minimum OAuth scope for an MCP tool name (unknown -> write)."""
    return TOOL_SCOPES.get(name, SCOPE_WRITE)


@dataclass(frozen=True)
class Grant:
    """A validated access token, attached to the ASGI scope for one request."""

    subject: str
    client_id: str
    scopes: frozenset[str]
    jti: str
    expires_at: int

    def has_scope(self, scope: str) -> bool:
        """Hierarchical check: admin ⊃ write ⊃ read."""
        if scope == SCOPE_READ:
            return bool(self.scopes & {SCOPE_READ, SCOPE_WRITE, SCOPE_ADMIN})
        if scope == SCOPE_WRITE:
            return bool(self.scopes & {SCOPE_WRITE, SCOPE_ADMIN})
        return scope in self.scopes


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class OAuthSettings:
    """Every knob of the authorization server (see docs/http-mcp.md)."""

    enabled: bool = False
    #: Fixed public origin, e.g. ``https://memory.example``. When unset the
    #: origin is derived from each request (correct behind a TLS terminator as
    #: long as uvicorn's ``--proxy-headers`` is trusted).
    issuer: str | None = None
    #: Canonical resource identifier; defaults to ``<issuer>/mcp``.
    resource: str | None = None
    scopes_supported: tuple[str, ...] = ALL_SCOPES
    max_scopes: tuple[str, ...] = (SCOPE_READ, SCOPE_WRITE)
    access_ttl: int = DEFAULT_ACCESS_TTL
    refresh_ttl: int = DEFAULT_REFRESH_TTL
    code_ttl: int = DEFAULT_CODE_TTL
    state_file: Path | None = None
    signing_key: str | None = None
    signing_key_file: Path | None = None
    authorize_user: str = "operator"
    authorize_password: str | None = None
    approval_token: str | None = None
    auto_approve: bool = False
    registration_enabled: bool = True
    registration_token: str | None = None
    clients_file: Path | None = None
    allowed_redirect_hosts: tuple[str, ...] = ()
    cors_origins: tuple[str, ...] = ("*",)
    #: ``sqlite`` (default, multi-process safe) | ``redis`` | ``json`` (the
    #: pre-scaling single-process file) | ``memory`` (tests).
    store_kind: str | None = None
    store_path: Path | None = None
    store_url: str | None = None
    store_prefix: str | None = None
    #: How long an in-process cache of the revocation set may serve validation
    #: before refreshing from the store. 0 = query the store on every request.
    revocation_cache_ttl: float = 5.0

    @classmethod
    def from_env(cls) -> "OAuthSettings":
        state_file_raw = os.environ.get(ENV_STATE_FILE, "").strip()
        key_file_raw = os.environ.get(ENV_SIGNING_KEY_FILE, "").strip()
        clients_raw = os.environ.get(ENV_CLIENTS_FILE, "").strip()
        store_path_raw = os.environ.get(ENV_STORE_PATH, "").strip()
        scopes = tuple(_list_env(ENV_SCOPES)) or ALL_SCOPES
        default_max = tuple(s for s in scopes if s != SCOPE_ADMIN) or (SCOPE_READ,)
        max_scopes = tuple(_list_env(ENV_MAX_SCOPES)) or default_max
        cors = tuple(_list_env(ENV_CORS_ORIGINS)) or ("*",)
        return cls(
            enabled=_flag(ENV_ENABLED),
            issuer=os.environ.get(ENV_ISSUER, "").strip() or None,
            resource=os.environ.get(ENV_RESOURCE, "").strip() or None,
            scopes_supported=scopes,
            max_scopes=max_scopes,
            access_ttl=_int_env(ENV_ACCESS_TTL, DEFAULT_ACCESS_TTL),
            refresh_ttl=_int_env(ENV_REFRESH_TTL, DEFAULT_REFRESH_TTL),
            code_ttl=_int_env(ENV_CODE_TTL, DEFAULT_CODE_TTL),
            state_file=Path(state_file_raw).expanduser() if state_file_raw else DEFAULT_STATE_FILE,
            signing_key=os.environ.get(ENV_SIGNING_KEY, "").strip() or None,
            signing_key_file=Path(key_file_raw).expanduser() if key_file_raw else None,
            authorize_user=os.environ.get(ENV_AUTHORIZE_USER, "").strip() or "operator",
            authorize_password=os.environ.get(ENV_AUTHORIZE_PASSWORD, "").strip() or None,
            approval_token=os.environ.get(ENV_APPROVAL_TOKEN, "").strip() or None,
            auto_approve=_flag(ENV_AUTO_APPROVE),
            registration_enabled=_flag(ENV_REGISTRATION_ENABLED, default=True),
            registration_token=os.environ.get(ENV_REGISTRATION_TOKEN, "").strip() or None,
            clients_file=Path(clients_raw).expanduser() if clients_raw else None,
            allowed_redirect_hosts=tuple(_list_env(ENV_ALLOWED_REDIRECT_HOSTS)),
            cors_origins=cors,
            store_kind=os.environ.get(ENV_STORE, "").strip() or "sqlite",
            store_path=Path(store_path_raw).expanduser() if store_path_raw else None,
            store_url=os.environ.get(ENV_STORE_URL, "").strip() or None,
            store_prefix=os.environ.get("HARS_MEMORY_MCP_OAUTH_STORE_PREFIX", "").strip() or None,
            revocation_cache_ttl=_float_env(ENV_REVOCATION_CACHE_TTL, 5.0),
        )

    def validate(self) -> None:
        unknown = [s for s in self.max_scopes if s not in self.scopes_supported]
        if unknown:
            raise ValueError(f"{ENV_MAX_SCOPES} contains scopes not in {ENV_SCOPES}: {unknown}")
        if not self.enabled:
            return
        if not self.auto_approve and not self.authorize_password and not self.approval_token:
            raise RuntimeError(
                "OAuth is enabled but no approval credential is configured. Set "
                f"{ENV_AUTHORIZE_PASSWORD} (browser login), {ENV_APPROVAL_TOKEN} "
                f"(headless pre-shared secret), or {ENV_AUTO_APPROVE}=1 for a "
                "loopback-only development server."
            )
        if self.access_ttl <= 0 or self.refresh_ttl <= 0 or self.code_ttl <= 0:
            raise ValueError("OAuth token/code TTLs must be positive integers")


# ---------------------------------------------------------------------------
# Signing material
# ---------------------------------------------------------------------------


@dataclass
class SigningMaterial:
    algorithm: str
    encode_key: Any
    decode_key: Any
    kid: str


def _public_pem(private_key: Any) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _load_pem(pem_text: str) -> SigningMaterial:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519, rsa

    key = serialization.load_pem_private_key(pem_text.encode("utf-8"), password=None)
    if isinstance(key, ed25519.Ed25519PrivateKey):
        algorithm = "EdDSA"
    elif isinstance(key, rsa.RSAPrivateKey):
        algorithm = "RS256"
    else:  # pragma: no cover - defensive
        raise ValueError("OAuth signing key must be Ed25519 or RSA")
    public_pem = _public_pem(key)
    kid = b64url(hashlib.sha256(public_pem).digest())[:16]
    return SigningMaterial(algorithm, pem_text, public_pem.decode("utf-8"), kid)


def _generate_ed25519(path: Path) -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    key = ed25519.Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(path), flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(pem)
    return pem


def resolve_signing_material(settings: OAuthSettings) -> SigningMaterial:
    """Resolve (or create) the access-token signing key.

    Precedence: inline ``..._SIGNING_KEY`` → ``..._SIGNING_KEY_FILE`` → a
    generated Ed25519 key persisted at ``~/.local/share/.../oauth/signing.key``.
    A PEM value selects EdDSA/RS256; any other string is treated as an HS256
    secret (intended for tests and single-host deployments).
    """
    if settings.signing_key:
        if "-----BEGIN" in settings.signing_key:
            return _load_pem(settings.signing_key)
        return SigningMaterial("HS256", settings.signing_key, settings.signing_key, "hs1")

    key_path = settings.signing_key_file
    if key_path is not None:
        if not key_path.is_file():
            raise FileNotFoundError(f"{ENV_SIGNING_KEY_FILE} not found: {key_path}")
        text = key_path.read_text(encoding="utf-8").strip()
        if "-----BEGIN" in text:
            return _load_pem(text)
        return SigningMaterial("HS256", text, text, "hs1")

    default_path = DEFAULT_SIGNING_KEY_FILE
    if default_path.is_file():
        return _load_pem(default_path.read_text(encoding="utf-8"))
    logger.info("Generating OAuth signing key at %s", default_path)
    return _load_pem(_generate_ed25519(default_path))


# ---------------------------------------------------------------------------
# Persisted state
# ---------------------------------------------------------------------------


@dataclass
class ClientRecord:
    client_id: str
    client_name: str
    redirect_uris: list[str]
    scope: str
    grant_types: list[str] = field(default_factory=lambda: ["authorization_code", "refresh_token"])
    response_types: list[str] = field(default_factory=lambda: ["code"])
    token_endpoint_auth_method: str = "none"
    client_secret_hash: str | None = None
    created_at: int = 0
    dynamic: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "redirect_uris": list(self.redirect_uris),
            "scope": self.scope,
            "grant_types": list(self.grant_types),
            "response_types": list(self.response_types),
            "token_endpoint_auth_method": self.token_endpoint_auth_method,
            "client_secret_hash": self.client_secret_hash,
            "created_at": self.created_at,
            "dynamic": self.dynamic,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClientRecord":
        return cls(
            client_id=str(data["client_id"]),
            client_name=str(data.get("client_name") or data["client_id"]),
            redirect_uris=[str(u) for u in data.get("redirect_uris") or []],
            scope=str(data.get("scope") or ""),
            grant_types=[str(g) for g in data.get("grant_types") or ["authorization_code"]],
            response_types=[str(r) for r in data.get("response_types") or ["code"]],
            token_endpoint_auth_method=str(data.get("token_endpoint_auth_method") or "none"),
            client_secret_hash=data.get("client_secret_hash"),
            created_at=int(data.get("created_at") or 0),
            dynamic=bool(data.get("dynamic", True)),
        )


@dataclass
class _AuthCode:
    code: str
    client_id: str
    redirect_uri: str
    scopes: tuple[str, ...]
    subject: str
    code_challenge: str
    expires_at: int


class OAuthProvider:
    """OAuth 2.1 authorization server + RFC 9068 access-token validation."""

    def __init__(self, settings: OAuthSettings, store: Any = None) -> None:
        settings.validate()
        self.settings = settings
        self.enabled = settings.enabled
        self.signing = resolve_signing_material(settings) if settings.enabled else None
        self._lock = threading.RLock()
        # Mutable cross-worker state lives in the store (SQLite WAL by default,
        # Redis optional, legacy JSON for rollback); access tokens stay
        # stateless JWTs and are never looked up here.
        self.store: Any = store if store is not None else self._build_store()
        self._rev_cache: set[str] | None = None
        self._rev_cache_at = 0.0
        if settings.enabled:
            migrated = self.store.migrate_json(settings.state_file)
            if any(migrated.values()):
                logger.info("OAuth store migrated from legacy JSON: %s", migrated)
            self._load_static_clients()

    def _build_store(self) -> Any:
        from hars_memory.mcp_oauth_store import (
            JsonOAuthStore,
            MemoryOAuthStore,
            build_store,
        )

        # Precedence, deliberately explicit:
        #   1. an explicit ``store_kind`` wins (``from_env`` always sets one, so
        #      the deployed server is SQLite unless told otherwise);
        #   2. otherwise ``state_file=None`` -> pure in-memory (tests);
        #   3. otherwise an explicit ``state_file`` -> the legacy JSON store.
        # (3) is the compatibility rule that keeps every pre-scaling caller —
        # and the documented one-flag rollback — behaving exactly as before.
        if self.settings.store_kind is None:
            if self.settings.state_file is None:
                return MemoryOAuthStore()
            return JsonOAuthStore(self.settings.state_file)
        if self.settings.store_kind == "json" and self.settings.store_path is None:
            return JsonOAuthStore(self.settings.state_file)
        return build_store(
            self.settings.store_kind,
            path=self.settings.store_path,
            url=self.settings.store_url,
            prefix=self.settings.store_prefix,
        )

    # -- state ------------------------------------------------------------

    def get_client(self, client_id: str) -> ClientRecord | None:
        data = self.store.get_client(client_id)
        if data is None:
            return None
        try:
            return ClientRecord.from_dict(data)
        except (KeyError, TypeError, ValueError):
            return None

    def _put_client(self, record: ClientRecord) -> None:
        self.store.put_client(record.client_id, record.to_dict())

    def _load_static_clients(self) -> None:
        path = self.settings.clients_file
        if path is None:
            return
        if not path.is_file():
            raise FileNotFoundError(f"{ENV_CLIENTS_FILE} not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("clients", payload) if isinstance(payload, dict) else payload
        for entry in entries:
            secret = str(entry.get("client_secret") or "")
            record = ClientRecord(
                client_id=str(entry["client_id"]),
                client_name=str(entry.get("client_name") or entry["client_id"]),
                redirect_uris=[str(u) for u in entry.get("redirect_uris") or []],
                scope=str(entry.get("scope") or " ".join(self.settings.max_scopes)),
                grant_types=list(entry.get("grant_types") or ["authorization_code"]),
                response_types=list(entry.get("response_types") or ["code"]),
                token_endpoint_auth_method=str(
                    entry.get("token_endpoint_auth_method")
                    or ("client_secret_post" if secret else "none")
                ),
                client_secret_hash=_token_hash(secret) if secret else None,
                created_at=_now(),
                dynamic=False,
            )
            self._put_client(record)
        logger.info("Loaded %d static OAuth clients from %s", len(entries), path)

    # -- metadata ---------------------------------------------------------

    @staticmethod
    def base_url(request: Request) -> str:
        host = request.headers.get("host") or request.url.netloc
        return f"{request.url.scheme}://{host}".rstrip("/")

    def issuer_for(self, base: str) -> str:
        return (self.settings.issuer or base).rstrip("/")

    def resource_for(self, base: str) -> str:
        if self.settings.resource:
            return self.settings.resource
        return f"{self.issuer_for(base)}/mcp"

    def protected_resource_metadata(self, base: str) -> dict[str, Any]:
        issuer = self.issuer_for(base)
        return {
            "resource": self.resource_for(base),
            "authorization_servers": [issuer],
            "bearer_methods_supported": ["header"],
            "scopes_supported": list(self.settings.scopes_supported),
            "resource_name": "HARS long-term memory",
            "resource_documentation": f"{issuer}/docs/http-mcp",
        }

    def authorization_server_metadata(self, base: str) -> dict[str, Any]:
        issuer = self.issuer_for(base)
        auth_methods = ["none", "client_secret_post", "client_secret_basic"]
        metadata: dict[str, Any] = {
            "issuer": issuer,
            "authorization_endpoint": f"{issuer}/authorize",
            "token_endpoint": f"{issuer}/token",
            "revocation_endpoint": f"{issuer}/revoke",
            "scopes_supported": list(self.settings.scopes_supported),
            "response_types_supported": ["code"],
            "response_modes_supported": ["query"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": auth_methods,
            "revocation_endpoint_auth_methods_supported": auth_methods,
            "code_challenge_methods_supported": ["S256"],
            "service_documentation": f"{issuer}/docs/http-mcp",
        }
        if self.settings.registration_enabled:
            metadata["registration_endpoint"] = f"{issuer}/register"
        return metadata

    # -- clients ----------------------------------------------------------

    def register_client(self, payload: dict[str, Any], base: str) -> tuple[dict[str, Any], int]:
        """RFC 7591 dynamic registration. Returns ``(body, status)``."""
        if not self.settings.registration_enabled:
            return {"error": "invalid_request", "error_description": "registration disabled"}, 403

        redirect_uris = payload.get("redirect_uris")
        if not isinstance(redirect_uris, list) or not redirect_uris:
            return {
                "error": "invalid_redirect_uri",
                "error_description": "redirect_uris required",
            }, 400
        for uri in redirect_uris:
            problem = self._redirect_uri_problem(str(uri))
            if problem:
                return {"error": "invalid_redirect_uri", "error_description": problem}, 400

        grant_types = list(payload.get("grant_types") or ["authorization_code", "refresh_token"])
        bad_grants = [g for g in grant_types if g not in {"authorization_code", "refresh_token"}]
        if bad_grants:
            return {
                "error": "invalid_client_metadata",
                "error_description": f"grant_types {bad_grants} not supported",
            }, 400

        response_types = list(payload.get("response_types") or ["code"])
        if any(r != "code" for r in response_types):
            return {
                "error": "invalid_client_metadata",
                "error_description": "only response_type=code is supported",
            }, 400

        auth_method = str(payload.get("token_endpoint_auth_method") or "none")
        if auth_method not in {"none", "client_secret_post", "client_secret_basic"}:
            return {
                "error": "invalid_client_metadata",
                "error_description": f"unsupported token_endpoint_auth_method {auth_method}",
            }, 400

        requested = self._parse_scope(payload.get("scope")) or list(self.settings.max_scopes)
        disallowed = [s for s in requested if s not in self.settings.max_scopes]
        if disallowed:
            return {
                "error": "invalid_scope",
                "error_description": f"scopes not available: {disallowed}",
            }, 400

        client_id = f"hmc_{secrets.token_urlsafe(18)}"
        client_secret = None
        secret_hash = None
        if auth_method != "none":
            client_secret = secrets.token_urlsafe(32)
            secret_hash = _token_hash(client_secret)

        record = ClientRecord(
            client_id=client_id,
            client_name=str(payload.get("client_name") or "MCP client")[:200],
            redirect_uris=[str(u) for u in redirect_uris],
            scope=" ".join(requested),
            grant_types=grant_types,
            response_types=response_types,
            token_endpoint_auth_method=auth_method,
            client_secret_hash=secret_hash,
            created_at=_now(),
            dynamic=True,
        )
        with self._lock:
            self._put_client(record)

        body: dict[str, Any] = {
            "client_id": client_id,
            "client_id_issued_at": record.created_at,
            "client_name": record.client_name,
            "redirect_uris": record.redirect_uris,
            "grant_types": record.grant_types,
            "response_types": record.response_types,
            "token_endpoint_auth_method": record.token_endpoint_auth_method,
            "scope": record.scope,
        }
        if client_secret is not None:
            body["client_secret"] = client_secret
            body["client_secret_expires_at"] = 0
        return body, 201

    def _redirect_uri_problem(self, uri: str) -> str | None:
        parsed = urlparse(uri)
        if parsed.fragment or parsed.username or parsed.password:
            return f"redirect_uri must not contain a fragment or credentials: {uri}"
        if parsed.scheme == "https":
            if (
                self.settings.allowed_redirect_hosts
                and parsed.hostname not in self.settings.allowed_redirect_hosts
            ):
                return f"redirect_uri host not allowed: {parsed.hostname}"
            return None
        if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
            return None
        return f"redirect_uri must be https:// (or http://loopback): {uri}"

    def _authenticate_client(
        self, request: Request, form: Any
    ) -> tuple[ClientRecord | None, str | None]:
        """Resolve and verify the client for /token and /revoke."""
        header = request.headers.get("authorization", "")
        client_id = str(form.get("client_id") or "")
        client_secret = str(form.get("client_secret") or "")
        if header.lower().startswith("basic "):
            try:
                decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
                basic_id, _, basic_secret = decoded.partition(":")
                client_id = client_id or basic_id
                client_secret = client_secret or basic_secret
            except Exception:
                return None, "invalid_client"
        if not client_id:
            return None, "invalid_client"
        record = self.get_client(client_id)
        if record is None:
            return None, "invalid_client"
        if record.client_secret_hash is not None:
            if not client_secret or not hmac.compare_digest(
                _token_hash(client_secret), record.client_secret_hash
            ):
                return None, "invalid_client"
        return record, None

    # -- codes and tokens -------------------------------------------------

    @staticmethod
    def _parse_scope(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            parts: list[str] = []
            for item in value:
                parts.extend(str(item).replace(",", " ").split())
            return [p for p in parts if p]
        return [p for p in str(value).replace(",", " ").split() if p]

    def _issue_code(
        self,
        client: ClientRecord,
        redirect_uri: str,
        scopes: list[str],
        subject: str,
        challenge: str,
    ) -> str:
        code = secrets.token_urlsafe(32)
        record = _AuthCode(
            code=code,
            client_id=client.client_id,
            redirect_uri=redirect_uri,
            scopes=tuple(scopes),
            subject=subject,
            code_challenge=challenge,
            expires_at=_now() + self.settings.code_ttl,
        )
        # Authorization codes live in the same store as refresh tokens: the
        # /authorize response and the /token request may hit different edge
        # workers, so a process-local dict would break the flow.
        self.store.put_code(_token_hash(code), {
            "client_id": record.client_id,
            "redirect_uri": record.redirect_uri,
            "scopes": list(record.scopes),
            "subject": record.subject,
            "code_challenge": record.code_challenge,
            "expires_at": record.expires_at,
        })
        return code

    def _consume_code(self, code: str) -> _AuthCode | None:
        data = self.store.pop_code(_token_hash(code))
        if data is None:
            return None
        try:
            return _AuthCode(
                code=code,
                client_id=str(data["client_id"]),
                redirect_uri=str(data["redirect_uri"]),
                scopes=tuple(data.get("scopes") or ()),
                subject=str(data["subject"]),
                code_challenge=str(data["code_challenge"]),
                expires_at=int(data["expires_at"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _sign_access_token(
        self, subject: str, client_id: str, scopes: list[str], issuer: str, resource: str
    ) -> tuple[str, int, str]:
        assert self.signing is not None
        now = _now()
        expires_at = now + self.settings.access_ttl
        jti = f"at_{secrets.token_urlsafe(16)}"
        payload = {
            "iss": issuer,
            "sub": subject,
            "aud": resource,
            "client_id": client_id,
            "scope": " ".join(scopes),
            "token_use": "access",
            "iat": now,
            "exp": expires_at,
            "jti": jti,
        }
        headers = {"typ": "at+jwt", "kid": self.signing.kid}
        token = jwt.encode(
            payload, self.signing.encode_key, algorithm=self.signing.algorithm, headers=headers
        )
        return token, expires_at, jti

    def issue_token_pair(
        self,
        subject: str,
        client_id: str,
        scopes: list[str],
        base: str,
        *,
        with_refresh: bool = True,
    ) -> dict[str, Any]:
        token, _exp, _jti = self._sign_access_token(
            subject, client_id, scopes, self.issuer_for(base), self.resource_for(base)
        )
        body: dict[str, Any] = {
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": self.settings.access_ttl,
            "scope": " ".join(scopes),
        }
        if with_refresh and "refresh_token" in (
            (self.get_client(client_id).grant_types if self.get_client(client_id) else [])
        ):
            refresh = secrets.token_urlsafe(32)
            self.store.put_refresh(_token_hash(refresh), {
                "client_id": client_id,
                "subject": subject,
                "scopes": list(scopes),
                "created_at": _now(),
                "expires_at": _now() + self.settings.refresh_ttl,
            })
            body["refresh_token"] = refresh
        return body

    def exchange_code(
        self, *, code: str, client: ClientRecord, redirect_uri: str, verifier: str, base: str
    ) -> tuple[dict[str, Any], int]:
        record = self._consume_code(code)
        if record is None:
            return {"error": "invalid_grant", "error_description": "unknown or expired code"}, 400
        if record.client_id != client.client_id:
            return {
                "error": "invalid_grant",
                "error_description": "code issued to another client",
            }, 400
        if record.redirect_uri != redirect_uri:
            return {"error": "invalid_grant", "error_description": "redirect_uri mismatch"}, 400
        if not verifier:
            return {"error": "invalid_request", "error_description": "code_verifier required"}, 400
        if not re.fullmatch(_PKCE_VERIFIER_RE, verifier):
            return {"error": "invalid_grant", "error_description": "malformed code_verifier"}, 400
        if not hmac.compare_digest(pkce_s256(verifier), record.code_challenge):
            return {"error": "invalid_grant", "error_description": "PKCE verification failed"}, 400
        return self.issue_token_pair(
            record.subject, client.client_id, list(record.scopes), base
        ), 200

    def exchange_refresh(
        self, *, refresh_token: str, client: ClientRecord, scope: str | None, base: str
    ) -> tuple[dict[str, Any], int]:
        key = _token_hash(refresh_token)
        # Atomic read-and-delete: with N edge workers, two concurrent refreshes
        # of the same token must not both succeed (rotation is one-time use).
        consume = getattr(self.store, "consume_refresh", None)
        if consume is not None:
            record = consume(key)
        else:  # pragma: no cover - every shipped store implements it
            record = self.store.get_refresh(key)
            if record is not None:
                self.store.delete_refresh(key)
        if record is None:
            return {"error": "invalid_grant", "error_description": "unknown refresh token"}, 400
        if int(record["expires_at"]) < _now():
            return {"error": "invalid_grant", "error_description": "refresh token expired"}, 400
        if record["client_id"] != client.client_id:
            return {
                "error": "invalid_grant",
                "error_description": "refresh token issued to another client",
            }, 400
        requested = self._parse_scope(scope) or list(record["scopes"])
        if any(s not in record["scopes"] for s in requested):
            return {
                "error": "invalid_scope",
                "error_description": "scope exceeds the original grant",
            }, 400
        return self.issue_token_pair(record["subject"], client.client_id, requested, base), 200

    # -- access-token validation -----------------------------------------

    def validate_access_token(self, token: str, base: str) -> tuple[Grant | None, str | None]:
        """Verify signature, issuer, audience, expiry and revocation."""
        if self.signing is None:
            return None, "invalid_token"
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            return None, "invalid_token"
        if header.get("typ") not in (None, "at+jwt"):
            return None, "invalid_token"
        if header.get("kid") not in (None, self.signing.kid):
            return None, "invalid_token"
        try:
            payload = jwt.decode(
                token,
                self.signing.decode_key,
                algorithms=[self.signing.algorithm],
                issuer=self.issuer_for(base),
                audience=self.resource_for(base),
                options={"require": ["exp", "iat", "sub", "aud", "iss"]},
            )
        except jwt.PyJWTError as exc:
            logger.debug("Access token rejected: %s", exc)
            return None, "invalid_token"
        if payload.get("token_use") not in (None, "access"):
            return None, "invalid_token"
        jti = str(payload.get("jti") or "")
        if jti and self._is_revoked(jti):
            return None, "invalid_token"
        scopes = frozenset(self._parse_scope(payload.get("scope")))
        if not scopes:
            return None, "insufficient_scope"
        return (
            Grant(
                subject=str(payload["sub"]),
                client_id=str(payload.get("client_id") or ""),
                scopes=scopes,
                jti=jti,
                expires_at=int(payload["exp"]),
            ),
            None,
        )

    def revoke_token(self, token: str, client: ClientRecord) -> None:
        key = _token_hash(token)
        record = self.store.get_refresh(key)
        if record is not None and record.get("client_id") == client.client_id:
            self.store.delete_refresh(key)
            return
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
        except jwt.PyJWTError:
            return
        if payload.get("client_id") not in (None, client.client_id):
            return
        jti = str(payload.get("jti") or "")
        exp = int(payload.get("exp") or 0)
        if jti:
            self.store.revoke_jti(jti, exp)
            if self._rev_cache is not None:
                self._rev_cache.add(jti)

    def _is_revoked(self, jti: str) -> bool:
        """Revocation check with a short in-process cache.

        Access tokens are stateless, so this is the only store read on the hot
        path. With a shared store the cache bounds cross-worker revocation
        propagation to ``revocation_cache_ttl`` seconds (default 5) while
        keeping validation at ~microseconds — the JWT signature check still
        happens on every request.
        """
        ttl = float(self.settings.revocation_cache_ttl or 0.0)
        if ttl <= 0 or not getattr(self.store, "is_shared", False):
            return self.store.is_revoked(jti)
        now = time.monotonic()
        if self._rev_cache is None or (now - self._rev_cache_at) > ttl:
            try:
                self._rev_cache = set(self.store.revoked_snapshot())
            except Exception as exc:  # pragma: no cover - store outage
                logger.warning("Revocation snapshot failed (%s); falling back to direct check", exc)
                return self.store.is_revoked(jti)
            self._rev_cache_at = now
        return jti in self._rev_cache

    # -- authorization endpoint -------------------------------------------

    def authorize(self, request: Request, params: dict[str, Any]) -> Response:
        base = self.base_url(request)
        client_id = str(params.get("client_id") or "")
        client = self.get_client(client_id)
        if client is None:
            return self._json_error(400, "invalid_request", f"unknown client_id {client_id!r}")

        redirect_uri = str(params.get("redirect_uri") or "")
        if not redirect_uri:
            if len(client.redirect_uris) == 1:
                redirect_uri = client.redirect_uris[0]
            else:
                return self._json_error(400, "invalid_request", "redirect_uri required")
        problem = self._redirect_uri_problem(redirect_uri)
        if problem or redirect_uri not in client.redirect_uris:
            return self._json_error(400, "invalid_request", "redirect_uri is not registered")

        state = str(params.get("state") or "")
        response_type = str(params.get("response_type") or "")
        if response_type != "code":
            return self._redirect_error(
                redirect_uri,
                state,
                "unsupported_response_type",
                "only response_type=code is supported",
            )

        challenge = str(params.get("code_challenge") or "")
        method = str(params.get("code_challenge_method") or "")
        if not challenge or method != "S256":
            return self._redirect_error(
                redirect_uri,
                state,
                "invalid_request",
                "PKCE with code_challenge_method=S256 is required",
            )
        if len(challenge) != 43 or any(
            ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
            for ch in challenge
        ):
            return self._redirect_error(
                redirect_uri, state, "invalid_request", "malformed code_challenge"
            )

        requested = self._parse_scope(params.get("scope")) or self._parse_scope(client.scope)
        allowed = set(self._parse_scope(client.scope)) & set(self.settings.max_scopes)
        if not requested:
            requested = sorted(allowed)
        if any(s not in allowed for s in requested):
            return self._redirect_error(
                redirect_uri,
                state,
                "invalid_scope",
                "one or more scopes are not available to this client",
            )

        resource = str(params.get("resource") or "")
        if resource and resource != self.resource_for(base):
            return self._redirect_error(
                redirect_uri, state, "invalid_target", "resource does not identify this MCP server"
            )

        if not self._is_approved(request, params):
            if "text/html" not in request.headers.get("accept", "text/html"):
                response = self._json_error(401, "access_denied", "authorization required")
                response.headers["WWW-Authenticate"] = 'Basic realm="HARS memory"'
                return response
            return HTMLResponse(self._login_page(params, client), status_code=200)

        code = self._issue_code(
            client, redirect_uri, requested, self.settings.authorize_user, challenge
        )
        query = {"code": code}
        if state:
            query["state"] = state
        query["iss"] = self.issuer_for(base)
        separator = "&" if "?" in redirect_uri else "?"
        return RedirectResponse(f"{redirect_uri}{separator}{urlencode(query)}", status_code=302)

    def _is_approved(self, request: Request, params: dict[str, Any]) -> bool:
        if self.settings.auto_approve:
            return True
        if self.settings.approval_token:
            supplied = str(params.get("approval_token") or "")
            if supplied and hmac.compare_digest(supplied, self.settings.approval_token):
                return True
        password = self.settings.authorize_password
        if not password:
            return False
        header = request.headers.get("authorization", "")
        if header.lower().startswith("basic "):
            try:
                decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
                user, _, supplied = decoded.partition(":")
            except Exception:
                return False
            if user == self.settings.authorize_user and hmac.compare_digest(supplied, password):
                return True
        form_password = str(params.get("password") or "")
        if form_password and hmac.compare_digest(form_password, password):
            user = str(params.get("username") or self.settings.authorize_user)
            return user == self.settings.authorize_user
        return False

    def _login_page(self, params: dict[str, Any], client: ClientRecord) -> str:
        hidden = []
        for key in (
            "response_type",
            "client_id",
            "redirect_uri",
            "scope",
            "state",
            "code_challenge",
            "code_challenge_method",
            "resource",
        ):
            if key in params and params[key] not in (None, ""):
                hidden.append(
                    f'<input type="hidden" name="{html.escape(key)}" '
                    f'value="{html.escape(str(params[key]))}">'
                )
        scopes = html.escape(" ".join(self._parse_scope(params.get("scope"))))
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Authorize {html.escape(client.client_name)}</title>
<style>
 body{{font:16px/1.5 system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaf0}}
 main{{max-width:26rem;margin:12vh auto;padding:2rem;background:#171a21;border-radius:12px}}
 h1{{font-size:1.15rem;margin:0 0 .25rem}} p{{color:#9aa3b2;margin:.25rem 0 1.25rem}}
 label{{display:block;font-size:.85rem;color:#9aa3b2;margin:.75rem 0 .25rem}}
 input[type=text],input[type=password]{{width:100%;padding:.6rem;border-radius:8px;border:1px solid #2c313c;background:#0f1115;color:#e8eaf0}}
 button{{margin-top:1.25rem;width:100%;padding:.7rem;border:0;border-radius:8px;background:#3b82f6;color:#fff;font-weight:600;cursor:pointer}}
 code{{font-size:.8rem;color:#c7cdd9}}
</style></head><body><main>
<h1>Authorize {html.escape(client.client_name)}</h1>
<p>Requested scopes: <code>{scopes or "default"}</code></p>
<form method="post" action="/authorize">
{"".join(hidden)}
<label for="username">User</label>
<input id="username" type="text" name="username" value="{html.escape(self.settings.authorize_user)}" autocomplete="username">
<label for="password">Password</label>
<input id="password" type="password" name="password" autocomplete="current-password" autofocus>
<button type="submit">Approve access</button>
</form></main></body></html>"""

    # -- responses --------------------------------------------------------

    @staticmethod
    def _json_error(status: int, error: str, description: str) -> JSONResponse:
        return JSONResponse(
            {"error": error, "error_description": description},
            status_code=status,
            headers={"Cache-Control": "no-store"},
        )

    def _redirect_error(
        self, redirect_uri: str, state: str, error: str, description: str
    ) -> RedirectResponse:
        query = {"error": error, "error_description": description}
        if state:
            query["state"] = state
        separator = "&" if "?" in redirect_uri else "?"
        return RedirectResponse(f"{redirect_uri}{separator}{urlencode(query)}", status_code=302)


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


def build_oauth_routes(provider: OAuthProvider) -> list[Route]:
    """Routes served next to ``/mcp`` (discovery + AS + health)."""

    async def protected_resource(request: Request) -> Response:
        return JSONResponse(
            provider.protected_resource_metadata(provider.base_url(request)),
            headers={"Cache-Control": "public, max-age=300"},
        )

    async def authorization_server(request: Request) -> Response:
        return JSONResponse(
            provider.authorization_server_metadata(provider.base_url(request)),
            headers={"Cache-Control": "public, max-age=300"},
        )

    async def register(request: Request) -> Response:
        if provider.settings.registration_token:
            supplied = request.headers.get("authorization", "")
            expected = f"Bearer {provider.settings.registration_token}"
            if not hmac.compare_digest(supplied, expected):
                return provider._json_error(401, "invalid_token", "registration token required")
        try:
            payload = await request.json()
        except Exception:
            return provider._json_error(400, "invalid_client_metadata", "JSON body required")
        if not isinstance(payload, dict):
            return provider._json_error(400, "invalid_client_metadata", "JSON object required")
        body, status = provider.register_client(payload, provider.base_url(request))
        return JSONResponse(body, status_code=status, headers={"Cache-Control": "no-store"})

    async def authorize_get(request: Request) -> Response:
        return provider.authorize(request, dict(request.query_params))

    async def authorize_post(request: Request) -> Response:
        form = await request.form()
        params = dict(request.query_params)
        params.update({key: form[key] for key in form})
        return provider.authorize(request, params)

    async def token(request: Request) -> Response:
        form = await request.form()
        client, error = provider._authenticate_client(request, form)
        if client is None:
            response = provider._json_error(401, "invalid_client", error or "invalid_client")
            response.headers["WWW-Authenticate"] = 'Basic realm="HARS memory"'
            return response
        grant_type = str(form.get("grant_type") or "")
        base = provider.base_url(request)
        if grant_type == "authorization_code":
            body, status = provider.exchange_code(
                code=str(form.get("code") or ""),
                client=client,
                redirect_uri=str(form.get("redirect_uri") or ""),
                verifier=str(form.get("code_verifier") or ""),
                base=base,
            )
        elif grant_type == "refresh_token":
            body, status = provider.exchange_refresh(
                refresh_token=str(form.get("refresh_token") or ""),
                client=client,
                scope=form.get("scope"),
                base=base,
            )
        else:
            body, status = (
                {"error": "unsupported_grant_type", "error_description": grant_type},
                400,
            )
        return JSONResponse(
            body,
            status_code=status,
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    async def revoke(request: Request) -> Response:
        form = await request.form()
        client, error = provider._authenticate_client(request, form)
        if client is None:
            return provider._json_error(401, "invalid_client", error or "invalid_client")
        token_value = str(form.get("token") or "")
        if token_value:
            provider.revoke_token(token_value, client)
        return JSONResponse({}, headers={"Cache-Control": "no-store"})

    return [
        Route("/.well-known/oauth-protected-resource", protected_resource, methods=["GET"]),
        Route("/.well-known/oauth-protected-resource/mcp", protected_resource, methods=["GET"]),
        Route("/.well-known/oauth-authorization-server", authorization_server, methods=["GET"]),
        Route("/.well-known/oauth-authorization-server/mcp", authorization_server, methods=["GET"]),
        Route("/register", register, methods=["POST"]),
        Route("/authorize", authorize_get, methods=["GET"]),
        Route("/authorize", authorize_post, methods=["POST"]),
        Route("/token", token, methods=["POST"]),
        Route("/revoke", revoke, methods=["POST"]),
    ]


def build_oauth_app(
    provider: OAuthProvider, *, health: Callable[[], dict[str, Any]] | None = None
) -> Starlette:
    """Starlette sub-app: OAuth routes (when enabled) + ``/healthz``."""

    async def healthz(request: Request) -> Response:
        body = dict(health() if health is not None else {"status": "ok"})
        body.setdefault("oauth", provider.enabled)
        body.setdefault("resource", provider.resource_for(provider.base_url(request)))
        return JSONResponse(body, headers={"Cache-Control": "no-store"})

    routes: list[Route] = []
    if provider.enabled:
        routes.extend(build_oauth_routes(provider))
    routes.append(Route("/healthz", healthz, methods=["GET"]))
    routes.append(Route("/health", healthz, methods=["GET"]))
    return Starlette(routes=routes)


__all__ = [
    "ALL_SCOPES",
    "GRANT_SCOPE_KEY",
    "Grant",
    "OAuthProvider",
    "OAuthSettings",
    "SCOPE_ADMIN",
    "SCOPE_READ",
    "SCOPE_WRITE",
    "TOOL_SCOPES",
    "build_oauth_app",
    "build_oauth_routes",
    "pkce_s256",
    "required_scope_for_tool",
    "resolve_signing_material",
]
