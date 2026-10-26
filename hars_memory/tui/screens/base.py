"""Shared screen helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.screen import Screen
from textual.widgets import DataTable

from hars_memory.tui.clients.base import Services

if TYPE_CHECKING:  # pragma: no cover
    from hars_memory.tui.app import HarsTuiApp

STATE_STYLE = {"OK": "bold green", "WARN": "bold yellow", "DOWN": "bold red", "N/A": "dim"}


def state_text(state: str) -> Text:
    return Text(state, style=STATE_STYLE.get(state, ""))


class HarsScreen(Screen):
    """Base for the five mode screens; gives typed access to the shared services."""

    @property
    def services(self) -> Services:
        return self.app.services  # type: ignore[attr-defined]

    @property
    def tui(self) -> HarsTuiApp:
        return self.app  # type: ignore[return-value]


def safe(value: object) -> Text:
    """Render untrusted text literally: DataTable/Tree would otherwise parse ``[x]`` as markup."""
    return Text(str(value))


def add_safe_row(table: DataTable, *values: object, key: str | None = None) -> None:
    """``DataTable.add_row`` with every plain value wrapped in a literal ``Text`` cell."""
    table.add_row(*[v if isinstance(v, Text) else safe(v) for v in values], key=key)
