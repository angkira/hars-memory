"""Fail-closed API-key authentication for the HTTP service."""

from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass
from typing import Final, Mapping

from fastapi import Header, HTTPException, status

API_KEYS_ENV: Final[str] = "HARS_MEMORY_API_KEYS_JSON"


class AuthConfigurationError(RuntimeError):
    """Raised when the server's API-key mapping is absent or invalid."""


@dataclass(frozen=True, slots=True)
class Principal:
    tenant_id: str


class APIKeyAuthenticator:
    """Resolve opaque API keys to tenant IDs without exposing either value."""

    def __init__(self, key_to_tenant: Mapping[str, str]) -> None:
        normalized = {
            key.strip(): tenant.strip()
            for key, tenant in key_to_tenant.items()
            if key.strip() and tenant.strip()
        }
        if not normalized:
            raise AuthConfigurationError("at least one API-key mapping is required")
        self._key_to_tenant = normalized

    @classmethod
    def from_env(cls) -> APIKeyAuthenticator:
        raw = os.environ.get(API_KEYS_ENV, "").strip()
        if not raw:
            raise AuthConfigurationError(f"{API_KEYS_ENV} is required")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuthConfigurationError(f"{API_KEYS_ENV} must be valid JSON") from exc
        if not isinstance(parsed, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()
        ):
            raise AuthConfigurationError(
                f"{API_KEYS_ENV} must be a JSON object mapping API keys to tenant IDs"
            )
        return cls(parsed)

    def authenticate(self, api_key: str | None) -> Principal:
        candidate = api_key or ""
        tenant_id = next(
            (
                tenant
                for known_key, tenant in self._key_to_tenant.items()
                if hmac.compare_digest(candidate, known_key)
            ),
            None,
        )
        if tenant_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "invalid_api_key", "message": "Invalid API key"},
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return Principal(tenant_id=tenant_id)

    def dependency(self, x_api_key: str | None = Header(default=None)) -> Principal:
        return self.authenticate(x_api_key)


__all__ = [
    "API_KEYS_ENV",
    "APIKeyAuthenticator",
    "AuthConfigurationError",
    "Principal",
]
