"""The Textual application: five mode screens + help, wired to one ``Services`` bundle."""

from __future__ import annotations

from typing import Any

import httpx
from textual.app import App
from textual.binding import Binding

from hars_memory.tui.clients.base import Services
from hars_memory.tui.config import TuiConfig
from hars_memory.tui.screens.evals import EvalScreen
from hars_memory.tui.screens.health import HealthScreen
from hars_memory.tui.screens.help import HelpScreen
from hars_memory.tui.screens.modals import RememberModal
from hars_memory.tui.screens.projects import ProjectsScreen
from hars_memory.tui.screens.recall import RecallScreen
from hars_memory.tui.screens.sessions import SessionsScreen
from hars_memory.tui.styles import CSS
from hars_memory.tui.writes import request_write

SCREEN_MODES = ("health", "recall", "sessions", "projects", "eval", "help")


class HarsTuiApp(App):
    TITLE = "hars-memory"
    CSS = CSS
    ENABLE_COMMAND_PALETTE = False
    MODES = {
        "health": HealthScreen,
        "recall": RecallScreen,
        "sessions": SessionsScreen,
        "projects": ProjectsScreen,
        "eval": EvalScreen,
        "help": HelpScreen,
    }
    DEFAULT_MODE = "health"
    BINDINGS = [
        Binding("1,f1", "go('health')", "Health"),
        Binding("2,f2", "go('recall')", "Recall"),
        Binding("3,f3", "go('sessions')", "Sessions"),
        Binding("4,f4", "go('projects')", "Projects"),
        Binding("5,f5", "go('eval')", "Eval"),
        Binding("question_mark,f10", "go('help')", "Help"),
        Binding("ctrl+t", "remember", "Remember"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, config: TuiConfig, *, http: httpx.AsyncClient | None = None, start: str = "health") -> None:
        super().__init__()
        self.services = Services(config, http)
        self.DEFAULT_MODE = start if start in SCREEN_MODES else "health"

    def on_mount(self) -> None:
        cfg = self.services.config
        self.sub_title = cfg.base_url + ("  [read-only]" if cfg.read_only else "") + ("  [ADMIN]" if cfg.allow_admin else "")

    async def action_go(self, mode: str) -> None:
        await self.switch_mode(mode)

    def action_remember(self) -> None:
        cfg = self.services.config

        def got(result: dict[str, Any] | None) -> None:
            if result:
                request_write(self.screen, self.services, "memory_remember", result["project"], result["args"],
                              notes=["Writes a staged note file; it is merged into the graph by the next KB update."])
        self.push_screen(RememberModal(cfg.project), got)

    async def on_unmount(self) -> None:
        await self.services.aclose()
