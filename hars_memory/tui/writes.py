"""The single path every mutating action takes: gate -> confirm modal -> MCP call -> notify."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from textual.screen import Screen

from hars_memory.tui.clients.base import Reply, Services
from hars_memory.tui.config import ADMIN_TOOLS
from hars_memory.tui.screens.modals import ConfirmWriteModal


def request_write(
    screen: Screen,
    services: Services,
    tool: str,
    project: str,
    args: dict[str, Any],
    *,
    notes: list[str] | None = None,
    on_done: Callable[[Reply], None] | None = None,
) -> bool:
    """Open the confirm modal for a write. Returns False when the gate refused it outright."""
    cfg = services.config
    app = screen.app
    if cfg.read_only:
        app.notify("read-only mode: writes are disabled", severity="warning")
        return False
    if tool in ADMIN_TOOLS and not cfg.allow_admin:
        app.notify(f"{tool} is an admin tool (start with --allow-admin)", severity="warning")
        return False
    full_args = {**args, "project": project}

    def after(confirmed: bool | None) -> None:
        if not confirmed:
            app.notify("write cancelled")
            return

        async def run() -> None:
            reply = await services.call_tool(tool, full_args)
            if reply.ok:
                app.notify(f"{tool} ok", severity="information")
            else:
                app.notify(f"{tool} failed: {reply.error}", severity="error", timeout=10)
            if on_done:
                on_done(reply)

        screen.run_worker(run(), exclusive=False, name=f"write:{tool}")

    app.push_screen(ConfirmWriteModal(tool, project, full_args, notes), after)
    return True
