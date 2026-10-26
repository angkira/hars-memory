"""Screen 4: projects and index status (read-only; project CRUD is out of scope, plan GAP-5)."""

from __future__ import annotations

import json
from typing import Any

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static

from hars_memory.tui.screens.base import HarsScreen, add_safe_row


def status_text(project: str, status: dict[str, Any]) -> str:
    vec = (status.get("storage") or {}).get("vector") or {}
    graph = (status.get("storage") or {}).get("graph") or {}
    models = status.get("configured_models") or {}
    backlog = status.get("staging_backlog") or {}
    lines = [
        f"project: {project} ({status.get('project_name')})",
        f"index_exists: {status.get('index_exists')}   nodes: {status.get('node_count')}   edges: {status.get('edge_count')}",
        f"last_ingest: {status.get('last_ingest')}",
        f"working_dir: {status.get('working_dir')}",
        f"vector: {json.dumps(vec, ensure_ascii=False)[:600]}",
        f"graph:  {graph.get('backend')}",
        f"embedder:  {models.get('embedder')}",
        f"extractor: {models.get('extractor')}",
        f"query:     {models.get('query')}",
        f"staging: {backlog.get('pending_notes')} pending notes (oldest {backlog.get('oldest_note_age_days')} days) in {backlog.get('staging_dir')}",
    ]
    if status.get("message"):
        lines.append(f"message: {status['message']}")
    return "\n".join(lines)


class ProjectsScreen(HarsScreen):
    BINDINGS = [Binding("r", "refresh", "Refresh")]

    def __init__(self) -> None:
        super().__init__()
        self.projects: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield DataTable(id="projects", cursor_type="row", zebra_stripes=True)
            with VerticalScroll(id="project-detail-scroll"):
                yield Static("", id="project-detail", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#projects", DataTable).add_columns("project", "name", "visibility", "actions", "owner")
        self.refresh_projects()

    def action_refresh(self) -> None:
        self.refresh_projects()

    @work(exclusive=True, group="projects")
    async def refresh_projects(self) -> None:
        detail = self.query_one("#project-detail", Static)
        reply = await self.services.call_tool("memory_list_projects", {})
        table = self.query_one("#projects", DataTable)
        table.clear()
        if not reply.ok or not isinstance(reply.data, dict):
            self.projects = []
            detail.update(f"memory_list_projects unavailable: {reply.error}")
            return
        self.projects = [p for p in reply.data.get("projects") or [] if isinstance(p, dict)]
        for p in self.projects:
            add_safe_row(table, p.get("project_id"), p.get("name"), p.get("visibility"),
                         ",".join(p.get("actions") or []), p.get("owner_user_id") or "-", key=str(p.get("project_id")))
        if not self.projects:
            detail.update("No accessible projects.")

    @on(DataTable.RowHighlighted, "#projects")
    def _highlight(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value:
            self.load_status(str(event.row_key.value))

    @work(exclusive=True, group="project-status")
    async def load_status(self, project: str) -> None:
        detail = self.query_one("#project-detail", Static)
        meta = next((p for p in self.projects if p.get("project_id") == project), {})
        reply = await self.services.call_tool("memory_status", {"project": project})
        head = f"{meta.get('description') or ''}\ndepartments: {', '.join(meta.get('shared_departments') or []) or '-'}\n\n"
        if reply.ok and isinstance(reply.data, dict):
            detail.update(head + status_text(project, reply.data))
        else:
            detail.update(head + f"memory_status({project}) failed: {reply.error}")
