"""TUI configuration: environment variables first, CLI flags override.

Environment (all optional):

    HARS_MEMORY_TUI_URL          base URL of the HARS HTTP server (health + MCP)
    HARS_MEMORY_TUI_MCP_URL      full MCP endpoint (default: <base><HARS_MEMORY_MCP_HTTP_PATH or /mcp>)
    HARS_MEMORY_TUI_TRACE_URL    base URL of the trace SSE listener (a trailing /trace is accepted)
    HARS_MEMORY_TUI_JOBS_URL     base URL of the index-job service (health probe only)
    HARS_MEMORY_TUI_TOKEN        bearer / access token (falls back to HARS_MEMORY_ACCESS_TOKEN)
    HARS_MEMORY_TUI_SESSIONS_DIR session root for the interim local-file session browser
                                 (falls back to HARS_MEMORY_SESSIONS_DIR)
    HARS_MEMORY_TUI_EVAL_ROOT    directory scanned by the eval results viewer
    HARS_MEMORY_TUI_PROJECT      default project id (default: ``default``)
    HARS_MEMORY_TUI_STATE_DIR    where the recall history is kept
    HARS_MEMORY_TUI_ALLOW_ADMIN  1 = unlock admin-class tools (off by default)
    HARS_MEMORY_TUI_READ_ONLY    1 = refuse every write, even behind the confirm modal
    HARS_MEMORY_TUI_REFRESH_S    health/session polling interval (default 5)

The server's own variables are honoured as defaults so a TUI started next to
the service needs no configuration: ``HARS_MEMORY_MCP_HTTP_HOST/PORT/PATH``,
``HARS_MEMORY_TRACE_URL`` / ``HARS_MEMORY_TRACE_HTTP_HOST/PORT`` and
``HARS_MEMORY_ACCESS_TOKEN`` / ``HARS_MEMORY_SESSIONS_DIR``.

The token is read from the environment (or from the variable *named* by
``--token-env``); there is deliberately no ``--token VALUE`` flag, so secrets
never land in shell history or ``ps``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_MCP_PORT = 8099
DEFAULT_TRACE_PORT = 8098
DEFAULT_PROJECT = "default"
DEFAULT_REFRESH_S = 5.0

# Tools that mutate the shared index in a way the TUI treats as admin-class
# (plan section 2). The MCP client refuses them unless allow_admin is set.
ADMIN_TOOLS = frozenset({"memory_forget", "memory_delete_document", "memory_consolidate"})
# Write tools the MVP exposes (always behind the confirm modal).
WRITE_TOOLS = frozenset({"memory_remember", "memory_compact", "memory_merge_session"})


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _pick(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _loopback(host: str) -> str:
    host = host.strip() or "127.0.0.1"
    return "127.0.0.1" if host in ("0.0.0.0", "::", "[::]") else host


@dataclass(frozen=True)
class TuiConfig:
    base_url: str = f"http://127.0.0.1:{DEFAULT_MCP_PORT}"
    mcp_url: str = f"http://127.0.0.1:{DEFAULT_MCP_PORT}/mcp"
    trace_url: str | None = f"http://127.0.0.1:{DEFAULT_TRACE_PORT}"
    jobs_url: str | None = None
    token: str | None = field(default=None, repr=False)
    sessions_dir: Path | None = None
    eval_root: Path | None = None
    project: str = DEFAULT_PROJECT
    state_dir: Path | None = None
    allow_admin: bool = False
    read_only: bool = False
    refresh_s: float = DEFAULT_REFRESH_S
    timeout_s: float = 15.0
    probe_models: bool = True

    def with_overrides(self, **overrides: object) -> TuiConfig:
        """Return a copy with every non-None override applied."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @property
    def host(self) -> str:
        return urlsplit(self.base_url).hostname or "127.0.0.1"


def from_env(env: Mapping[str, str] | None = None) -> TuiConfig:
    """Build a config from environment variables (see module docstring)."""
    env = os.environ if env is None else env
    scheme = "https" if _pick(env, "HARS_MEMORY_MCP_TLS_CERTFILE") else "http"
    host = _loopback(_pick(env, "HARS_MEMORY_MCP_HTTP_HOST", "HARS_MEMORY_MCP_BIND_HOST") or "127.0.0.1")
    port = _pick(env, "HARS_MEMORY_MCP_HTTP_PORT") or str(DEFAULT_MCP_PORT)
    base = (_pick(env, "HARS_MEMORY_TUI_URL") or f"{scheme}://{host}:{port}").rstrip("/")
    path = _pick(env, "HARS_MEMORY_MCP_HTTP_PATH") or "/mcp"
    if not path.startswith("/"):
        path = "/" + path
    mcp_url = (_pick(env, "HARS_MEMORY_TUI_MCP_URL") or f"{base}{path}").rstrip("/")

    trace_url = _trace_url(env, base)
    jobs = _pick(env, "HARS_MEMORY_TUI_JOBS_URL")

    sessions = _pick(env, "HARS_MEMORY_TUI_SESSIONS_DIR", "HARS_MEMORY_SESSIONS_DIR")
    eval_root = _pick(env, "HARS_MEMORY_TUI_EVAL_ROOT")
    state = _pick(env, "HARS_MEMORY_TUI_STATE_DIR")
    try:
        refresh = float(_pick(env, "HARS_MEMORY_TUI_REFRESH_S") or DEFAULT_REFRESH_S)
    except ValueError:
        refresh = DEFAULT_REFRESH_S
    return TuiConfig(
        base_url=base,
        mcp_url=mcp_url,
        trace_url=trace_url,
        jobs_url=jobs.rstrip("/") if jobs else None,
        token=_pick(env, "HARS_MEMORY_TUI_TOKEN", "HARS_MEMORY_ACCESS_TOKEN"),
        sessions_dir=Path(sessions).expanduser() if sessions else None,
        eval_root=Path(eval_root).expanduser() if eval_root else None,
        project=_pick(env, "HARS_MEMORY_TUI_PROJECT") or DEFAULT_PROJECT,
        state_dir=Path(state).expanduser() if state else None,
        allow_admin=_truthy(env.get("HARS_MEMORY_TUI_ALLOW_ADMIN")),
        read_only=_truthy(env.get("HARS_MEMORY_TUI_READ_ONLY")),
        refresh_s=max(1.0, refresh),
    )


def _trace_url(env: Mapping[str, str], base: str) -> str | None:
    explicit = _pick(env, "HARS_MEMORY_TUI_TRACE_URL", "HARS_MEMORY_TRACE_URL")
    if explicit:
        explicit = explicit.rstrip("/")
        return explicit[: -len("/trace")] if explicit.endswith("/trace") else explicit
    raw_port = env.get("HARS_MEMORY_TRACE_HTTP_PORT")
    if raw_port is not None and raw_port.strip() in ("", "0"):
        return None  # listener explicitly disabled
    port = (raw_port or "").strip() or str(DEFAULT_TRACE_PORT)
    host = _pick(env, "HARS_MEMORY_TRACE_HTTP_HOST")
    host = _loopback(host) if host else (urlsplit(base).hostname or "127.0.0.1")
    scheme = urlsplit(base).scheme or "http"
    return f"{scheme}://{host}:{port}"
