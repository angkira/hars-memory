"""Token store and token verification."""

from __future__ import annotations

import hmac
import logging
import os
from pathlib import Path
from typing import Any, Final

import yaml

from hars_memory.auth.models import AccessToken, TokenContext

logger = logging.getLogger(__name__)

DEFAULT_TOKENS_FILE: Final[Path] = (
    Path.home() / ".local" / "share" / "hars-longterm-memory" / "auth" / "tokens.json"
)
TOKENS_CONFIG_ENV: Final[str] = "HARS_MEMORY_TOKENS_CONFIG"


class TokenStore:
    """In-memory token repository with disk and environment variable backends."""

    def __init__(
        self,
        config_path: Path | str | None = None,
        auto_load: bool = True,
        tokens: list[AccessToken] | None = None,
    ) -> None:
        self._tokens: dict[str, AccessToken] = {}
        self._config_path = Path(config_path) if config_path else None
        if tokens:
            for t in tokens:
                self.add_token(t)
        elif auto_load:
            self.reload()

    def clear(self) -> None:
        """Clear all registered tokens."""
        self._tokens.clear()

    def add_token(self, token: AccessToken) -> None:
        """Register or update an AccessToken."""
        clean_token = token.token.strip()
        if not clean_token:
            raise ValueError("Token string cannot be empty")
        self._tokens[clean_token] = token

    def remove_token(self, token_str: str) -> bool:
        """Remove a token by its token string. Returns True if removed."""
        clean_token = self._normalize_token(token_str)
        return self._tokens.pop(clean_token, None) is not None

    def list_tokens(self) -> list[AccessToken]:
        """Return list of all registered AccessTokens."""
        return list(self._tokens.values())

    @staticmethod
    def _normalize_token(token_str: str | None) -> str:
        if not token_str:
            return ""
        candidate = token_str.strip()
        if candidate.lower().startswith("bearer "):
            candidate = candidate[7:].strip()
        return candidate

    def verify_token(self, token_str: str | None) -> TokenContext | None:
        """Verify an authentication token string.

        Supports signed JWT tokens and static opaque tokens.
        Tokens may optionally be prefixed with 'Bearer '.
        Performs constant-time comparison for static keys to prevent timing attacks.
        Returns TokenContext on success, or None on failure.
        """
        candidate = self._normalize_token(token_str)
        if not candidate:
            return None

        # 1. Check signed JWT first if candidate looks like a JWT
        if candidate.count(".") == 2:
            try:
                from hars_memory.auth.jwt import get_default_jwt_manager  # noqa: PLC0415

                jwt_ctx = get_default_jwt_manager().verify_token(candidate)
                if jwt_ctx is not None:
                    return jwt_ctx
            except Exception as exc:
                logger.debug("JWT verification error: %s", exc)

        # 2. Check static registered tokens with constant-time comparison
        for known_key, access_token in self._tokens.items():
            if hmac.compare_digest(candidate, known_key):
                return TokenContext.from_access_token(access_token)
        return None

    def reload(self) -> None:
        """Reload tokens from env HARS_MEMORY_TOKENS_CONFIG or default storage file."""
        self._tokens.clear()

        # 1. Check explicit config path if set
        if self._config_path and self._config_path.is_file():
            self._load_from_file(self._config_path)
            return

        # 2. Check HARS_MEMORY_TOKENS_CONFIG env
        env_config = os.environ.get(TOKENS_CONFIG_ENV, "").strip()
        if env_config:
            self._load_from_env_string(env_config)
            return

        # 3. Check default path ~/.local/share/hars-longterm-memory/auth/tokens.json
        if DEFAULT_TOKENS_FILE.is_file():
            self._load_from_file(DEFAULT_TOKENS_FILE)

    def _load_from_env_string(self, content: str) -> None:
        candidate_path = Path(content).expanduser()
        if candidate_path.is_file():
            self._load_from_file(candidate_path)
            return

        # Parse inline JSON or YAML
        try:
            parsed = yaml.safe_load(content)
            self._parse_and_ingest(parsed)
        except Exception as exc:
            logger.warning("Failed to parse %s content: %s", TOKENS_CONFIG_ENV, exc)

    def _load_from_file(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
            parsed = yaml.safe_load(text)
            self._parse_and_ingest(parsed)
        except Exception as exc:
            logger.warning("Failed to load tokens from %s: %s", path, exc)

    def _parse_and_ingest(self, data: Any) -> None:
        if not data:
            return

        # Dict format: {"tokens": [...]} or {"token_string": {user_id, ...}}
        if isinstance(data, dict):
            if "tokens" in data and isinstance(data["tokens"], list):
                for item in data["tokens"]:
                    if isinstance(item, dict):
                        tok = AccessToken.from_dict(item)
                        if tok.token:
                            self._tokens[tok.token] = tok
            else:
                for key, val in data.items():
                    if isinstance(val, dict):
                        tok = AccessToken.from_dict(val, default_token=key)
                        if tok.token:
                            self._tokens[tok.token] = tok

        # List format: [{"token": "...", "user_id": "..."}, ...]
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    tok = AccessToken.from_dict(item)
                    if tok.token:
                        self._tokens[tok.token] = tok


    def authenticate(self, token_str: str | None) -> TokenContext | None:
        """Alias for verify_token."""
        return self.verify_token(token_str)


_default_token_store: TokenStore | None = None


def get_default_token_store() -> TokenStore:
    """Return the global default TokenStore singleton."""
    global _default_token_store
    if _default_token_store is None:
        _default_token_store = TokenStore()
    return _default_token_store


def reset_default_token_store() -> None:
    """Reset the global default TokenStore singleton."""
    global _default_token_store
    _default_token_store = None
