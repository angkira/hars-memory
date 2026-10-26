"""``memory-tui`` / ``memory tui`` entry point. Imports Textual lazily with a clean error."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from hars_memory.tui.config import TuiConfig, from_env

INSTALL_HINT = (
    "The hars-memory TUI needs the optional 'tui' extra (Textual).\n"
    "Install it with:  pip install 'hars-longterm-memory[tui]'   (or: uv sync --extra tui)"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memory-tui",
        description="Terminal UI for hars-longterm-memory (health, recall trace, sessions, projects, eval results).",
        epilog="Environment: HARS_MEMORY_TUI_URL, HARS_MEMORY_TUI_TRACE_URL, HARS_MEMORY_TUI_TOKEN, "
               "HARS_MEMORY_TUI_SESSIONS_DIR, HARS_MEMORY_TUI_EVAL_ROOT (see docs/tui.md). "
               "Flags override the environment.",
    )
    p.add_argument("--url", help="base URL of the HARS HTTP server (health + MCP)")
    p.add_argument("--mcp-url", help="full MCP endpoint (default: <url>/mcp)")
    p.add_argument("--trace-url", help="trace SSE listener base URL; 'off' disables stage tracing")
    p.add_argument("--jobs-url", help="index-job service base URL (health probe only)")
    p.add_argument("--token-env", metavar="NAME", help="name of the environment variable holding the access token "
                   "(default: HARS_MEMORY_TUI_TOKEN, then HARS_MEMORY_ACCESS_TOKEN). The token itself is never a flag.")
    p.add_argument("--project", help="default project id (default: 'default')")
    p.add_argument("--sessions-dir", type=Path, help="session working-dir root for the session browser")
    p.add_argument("--eval-root", type=Path, help="directory scanned by the eval results viewer")
    p.add_argument("--state-dir", type=Path, help="where the recall history is stored")
    p.add_argument("--refresh", type=float, metavar="SECONDS", help="polling interval (default 5)")
    p.add_argument("--allow-admin", action="store_true", help="unlock admin-class tools in the client gate (off by default)")
    p.add_argument("--read-only", action="store_true", help="refuse every write, even behind the confirm modal")
    p.add_argument("--no-probe", action="store_true", help="do not probe embedder/extractor URLs from this host")
    p.add_argument("--screen", choices=("health", "recall", "sessions", "projects", "eval", "help"), default="health",
                   help="screen to open first")
    return p


def config_from_args(args: argparse.Namespace, env: Mapping[str, str] | None = None) -> TuiConfig:
    """Environment first, then CLI flags on top."""
    env_map = dict(os.environ if env is None else env)
    cfg = from_env(env_map)
    token = cfg.token
    if args.token_env:
        token = env_map.get(args.token_env, "").strip() or None
    base = args.url.rstrip("/") if args.url else None
    mcp = args.mcp_url.rstrip("/") if args.mcp_url else (f"{base}/mcp" if base else None)
    cfg = cfg.with_overrides(
        base_url=base, mcp_url=mcp, jobs_url=args.jobs_url.rstrip("/") if args.jobs_url else None,
        token=token, project=args.project, sessions_dir=args.sessions_dir, eval_root=args.eval_root,
        state_dir=args.state_dir, refresh_s=max(1.0, args.refresh) if args.refresh else None,
        allow_admin=True if args.allow_admin else None, read_only=True if args.read_only else None,
        probe_models=False if args.no_probe else None,
    )
    if args.trace_url:
        off = args.trace_url.lower() in ("off", "none", "0")
        cfg = replace(cfg, trace_url=None if off else args.trace_url.rstrip("/").removesuffix("/trace"))
    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        from hars_memory.tui.app import HarsTuiApp
    except ImportError as exc:
        if (getattr(exc, "name", None) or "").split(".")[0] == "textual":
            print(INSTALL_HINT, file=sys.stderr)
            return 2
        raise
    HarsTuiApp(config_from_args(args), start=args.screen).run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
