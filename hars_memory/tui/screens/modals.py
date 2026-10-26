"""Modal dialogs: the write-confirmation gate and the small input forms."""

from __future__ import annotations

import json
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static, TextArea

from hars_memory.tui.config import ADMIN_TOOLS

GUARDED_PROJECT = "default"  # the shared SRE KB: writes need a second, typed confirmation


class ConfirmWriteModal(ModalScreen[bool]):
    """Shows project + tool + args. Writes to the shared ``default`` project need a typed second confirmation."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, tool: str, project: str, args: dict[str, Any], notes: list[str] | None = None) -> None:
        super().__init__()
        self.tool, self.project, self.args_, self.notes = tool, project, args, notes or []
        self.guarded = project == GUARDED_PROJECT
        self._stage = 1

    def compose(self) -> ComposeResult:
        shown = {k: v for k, v in self.args_.items() if k != "access_token"}
        body = "\n".join([
            f"tool:    {self.tool}" + ("   [ADMIN]" if self.tool in ADMIN_TOOLS else ""),
            f"project: {self.project}",
            "args:",
            json.dumps(shown, indent=2, ensure_ascii=False)[:1800],
            *([""] + self.notes if self.notes else []),
        ])
        with Vertical(id="confirm-dialog"):
            yield Label("Confirm write", id="confirm-title")
            with VerticalScroll(id="confirm-body"):
                yield Static(body, id="confirm-text", markup=False)
            yield Static("", id="confirm-guard", markup=False)
            yield Input(placeholder=f"type '{self.project}' to confirm", id="confirm-typed", classes="hidden")
            with Horizontal(id="confirm-buttons"):
                yield Button("Confirm", id="confirm", variant="warning")
                yield Button("Cancel", id="cancel", variant="default")

    def on_mount(self) -> None:
        self.query_one("#cancel", Button).focus()

    def _arm_second_stage(self) -> None:
        self._stage = 2
        self.query_one("#confirm-guard", Static).update(
            f"SECOND CONFIRMATION: '{self.project}' is the shared knowledge base. "
            f"Type '{self.project}' below, then press Confirm."
        )
        typed = self.query_one("#confirm-typed", Input)
        typed.remove_class("hidden")
        typed.focus()
        self.query_one("#confirm", Button).disabled = True

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "confirm-typed":
            self.query_one("#confirm", Button).disabled = event.value.strip() != self.project

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "confirm-typed" and event.value.strip() == self.project:
            self.dismiss(True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "cancel":
            self.dismiss(False)
        elif self.guarded and self._stage == 1:
            self._arm_second_stage()
        else:
            self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class RememberModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, project: str) -> None:
        super().__init__()
        self.project = project

    def compose(self) -> ComposeResult:
        with Vertical(id="form-dialog"):
            yield Label("Remember a note (staged; merged by the next KB update)")
            yield Input(self.project, id="f-project", placeholder="project")
            yield Input(placeholder="title (kebab-case slug)", id="f-title")
            yield Input(placeholder="tags, comma separated", id="f-tags")
            yield Select([("normal", "normal"), ("critical", "critical"), ("low", "low")], value="normal",
                         allow_blank=False, id="f-importance", compact=True)
            yield TextArea(id="f-content")
            with Horizontal(id="form-buttons"):
                yield Button("Review", id="ok", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        title = self.query_one("#f-title", Input).value.strip()
        content = self.query_one("#f-content", TextArea).text.strip()
        if not title or not content:
            self.app.notify("title and content are required", severity="warning")
            return
        tags = [t.strip() for t in self.query_one("#f-tags", Input).value.split(",") if t.strip()]
        select_value = self.query_one("#f-importance", Select).value
        self.dismiss({
            "project": self.query_one("#f-project", Input).value.strip() or self.project,
            "args": {"title": title, "content": content, "tags": tags,
                     "importance": str(select_value) if isinstance(select_value, str) else "normal"},
        })

    def action_cancel(self) -> None:
        self.dismiss(None)


class CompactModal(ModalScreen[dict[str, Any] | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, project: str, session_id: str = "") -> None:
        super().__init__()
        self.project, self.session_id = project, session_id

    def compose(self) -> ComposeResult:
        with Vertical(id="form-dialog"):
            yield Label("Store a session compaction (manual testing; agents normally do this)")
            yield Input(self.session_id, id="f-sid", placeholder="session id [A-Za-z0-9._-]")
            yield Label("Summary")
            yield TextArea(id="f-summary")
            yield Label("Dropped-context facts (one per line)")
            yield TextArea(id="f-dropped")
            with Horizontal(id="form-buttons"):
                yield Button("Review", id="ok", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        sid = self.query_one("#f-sid", Input).value.strip()
        summary = self.query_one("#f-summary", TextArea).text.strip()
        if not sid or not summary:
            self.app.notify("session id and summary are required", severity="warning")
            return
        dropped = [ln.strip() for ln in self.query_one("#f-dropped", TextArea).text.splitlines() if ln.strip()]
        self.dismiss({"project": self.project, "args": {"session_id": sid, "summary": summary, "dropped_context": dropped}})

    def action_cancel(self) -> None:
        self.dismiss(None)
