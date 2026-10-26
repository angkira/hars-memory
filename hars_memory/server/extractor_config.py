"""Config-driven selection of the extraction-LLM backend (local llama.cpp vs OpenRouter).

Everything is read from environment variables; there is NO model id in code.

Provider ``local`` (default, unchanged behaviour)
    ``HARS_MEMORY_EXTRACTOR_BASE_URL`` / ``HARS_MEMORY_EXTRACTOR_MODEL`` select the
    OpenAI-compatible llama-server. Auth: ``HARS_MEMORY_LLM_API_KEY`` or a dummy.

Provider ``openrouter``
    ``HARS_MEMORY_OPENROUTER_MODEL``     REQUIRED model id (e.g. ``vendor/model``).
    ``HARS_MEMORY_OPENROUTER_BASE_URL``  optional, default :data:`OPENROUTER_BASE_URL`.
    ``HARS_MEMORY_EXTRACTOR_API_KEY_ENV`` NAME of the env var holding the key
        (default ``OPENROUTER_API_KEY``). The key value is never logged/printed.
    ``HARS_MEMORY_EXTRACTOR_API_KEY_FILE``  optional dotenv-style file consulted ONLY when that
        env var is unset/empty (lets an MCP launched without the secrets exported still
        authenticate; only the one named variable is read, never logged).
    The generic ``HARS_MEMORY_EXTRACTOR_BASE_URL``/``_MODEL`` are deliberately
    ignored here: they keep pointing at the local server so flipping the provider
    never inherits a local model name.

Both providers
    ``HARS_MEMORY_EXTRACTOR_REASONING_EFFORT``  sent as ``reasoning.effort`` when set
        (openrouter default ``low`` -- "none" measured worse, see below; local default: not sent, the
        server is started with reasoning disabled). ``unset`` disables sending, ``none`` = off.
    ``HARS_MEMORY_EXTRACTOR_EXTRA_BODY_JSON``  JSON object merged into every request.
    Timeout, retry/backoff and concurrency are the existing
    ``HARS_MEMORY_LLM_TIMEOUT_SECONDS`` / ``HARS_MEMORY_LLM_RETRY_*`` /
    ``HARS_MEMORY_LLM_MAX_ASYNC``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

PROVIDER_LOCAL: Final[str] = "local"
PROVIDER_OPENROUTER: Final[str] = "openrouter"
PROVIDERS: Final[frozenset[str]] = frozenset({PROVIDER_LOCAL, PROVIDER_OPENROUTER})

#: Service endpoint of the hosted provider (not a model id).
OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_KEY_ENV: Final[str] = "OPENROUTER_API_KEY"
# Measured 2026-10-07 (5 docs, 7 chunks): effort "none" made the hosted model emit junk lines
# (2.0 non-record lines/chunk at T=0.1; gemma baseline 0.14); "low" gave 0.0 and better recall.
DEFAULT_OPENROUTER_REASONING_EFFORT: Final[str] = "low"

_LOCAL_DEFAULT_BASE_URL: Final[str] = "http://localhost:8080/v1"
_LOCAL_DEFAULT_MODEL: Final[str] = "Qwen3.6-27B-Q4_K_M"
_LOCAL_DUMMY_API_KEY: Final[str] = "not-needed"
_REASONING_UNSET: Final[str] = "unset"


class ExtractorConfigError(ValueError):
    """Invalid or incomplete extractor-backend configuration (explicit, fail-fast)."""


@dataclass(frozen=True, slots=True)
class ExtractorBackend:
    """Resolved extraction-LLM backend. ``api_key`` is excluded from ``repr``."""

    provider: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    extra_body: Mapping[str, Any] = field(default_factory=dict)


def _clean(env: Mapping[str, str], name: str) -> str:
    return (env.get(name) or "").strip()


def _key_from_file(path: str, name: str) -> str:
    """Return ``NAME`` from a dotenv-style file ('' if no file/var). Never logs values."""
    if not path:
        return ""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExtractorConfigError(f"cannot read HARS_MEMORY_EXTRACTOR_API_KEY_FILE {path!r}: {exc.strerror}") from exc
    prefix = f"{name}="
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if line.startswith(prefix):
            return line[len(prefix):].strip().strip("\"'")
    return ""


def resolve_openrouter_credentials(env: Mapping[str, str]) -> tuple[str, str]:
    """``(base_url, api_key)`` for OpenRouter; raises when the key is missing.

    Shared by the sync extractor and the Batch API backend so both read the same
    variables (and neither ever logs the key).
    """
    base_url = _clean(env, "HARS_MEMORY_OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL
    key_env = _clean(env, "HARS_MEMORY_EXTRACTOR_API_KEY_ENV") or DEFAULT_OPENROUTER_KEY_ENV
    api_key = _clean(env, key_env) or _key_from_file(_clean(env, "HARS_MEMORY_EXTRACTOR_API_KEY_FILE"), key_env)
    if not api_key:
        raise ExtractorConfigError(
            "OpenRouter needs an API key in the environment "
            f"variable {key_env!r} (name configurable via HARS_MEMORY_EXTRACTOR_API_KEY_ENV); it is unset or empty"
        )
    return base_url, api_key


def resolve_extractor_backend(
    env: Mapping[str, str],
    *,
    base_url: str | None = None,
    model: str | None = None,
) -> ExtractorBackend:
    """Resolve the extractor backend from ``env`` (explicit args win for ``local``)."""
    provider = _clean(env, "HARS_MEMORY_EXTRACTOR_PROVIDER").lower() or PROVIDER_LOCAL
    if provider not in PROVIDERS:
        raise ExtractorConfigError(
            f"HARS_MEMORY_EXTRACTOR_PROVIDER={provider!r} is invalid; expected one of {sorted(PROVIDERS)}"
        )

    if provider == PROVIDER_LOCAL:
        resolved_url = base_url or _clean(env, "HARS_MEMORY_EXTRACTOR_BASE_URL") or _LOCAL_DEFAULT_BASE_URL
        resolved_model = model or _clean(env, "HARS_MEMORY_EXTRACTOR_MODEL") or _LOCAL_DEFAULT_MODEL
        api_key = _clean(env, "HARS_MEMORY_LLM_API_KEY") or _LOCAL_DUMMY_API_KEY
        default_effort = ""
    else:
        resolved_model = model or _clean(env, "HARS_MEMORY_OPENROUTER_MODEL")
        if not resolved_model:
            raise ExtractorConfigError(
                "HARS_MEMORY_EXTRACTOR_PROVIDER=openrouter requires HARS_MEMORY_OPENROUTER_MODEL "
                "(the OpenRouter model id); no default is baked in"
            )
        resolved_url, api_key = resolve_openrouter_credentials(env)
        if base_url:
            resolved_url = base_url
        default_effort = DEFAULT_OPENROUTER_REASONING_EFFORT

    extra_body: dict[str, Any] = {}
    effort = _clean(env, "HARS_MEMORY_EXTRACTOR_REASONING_EFFORT") or default_effort
    if effort and effort.lower() != _REASONING_UNSET:
        extra_body["reasoning"] = {"effort": effort}

    raw_extra = _clean(env, "HARS_MEMORY_EXTRACTOR_EXTRA_BODY_JSON")
    if raw_extra:
        try:
            parsed = json.loads(raw_extra)
        except json.JSONDecodeError as exc:
            raise ExtractorConfigError(
                f"HARS_MEMORY_EXTRACTOR_EXTRA_BODY_JSON is not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ExtractorConfigError("HARS_MEMORY_EXTRACTOR_EXTRA_BODY_JSON must be a JSON object")
        extra_body.update(parsed)

    return ExtractorBackend(
        provider=provider,
        base_url=resolved_url,
        model=resolved_model,
        api_key=api_key,
        extra_body=extra_body,
    )
