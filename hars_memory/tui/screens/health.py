"""Screen 1: service health (HARS API, MCP, trace, index, embedder, extractor, Qdrant, jobs)."""

from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Static

from hars_memory.tui.health import HealthRow, run_health_checks
from hars_memory.tui.screens.base import HarsScreen, safe, state_text


class HealthScreen(HarsScreen):
    BINDINGS = [Binding("r", "refresh", "Refresh now")]

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[HealthRow] = []
        self._keys: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static("", id="health-banner", markup=False)
            yield DataTable(id="health-table", cursor_type="row", zebra_stripes=True)
            yield Static("Select a row for details.", id="health-detail", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        cfg = self.services.config
        table = self.query_one("#health-table", DataTable)
        table.add_columns("Service", "State", "Latency", "Detail")
        self.query_one("#health-banner", Static).update(
            f"{cfg.base_url}  project={cfg.project}  token={'set' if cfg.token else 'none'}  "
            f"admin={'ON' if cfg.allow_admin else 'off'}  writes={'DISABLED' if cfg.read_only else 'confirm-gated'}  "
            f"refresh={cfg.refresh_s:g}s"
        )
        self.set_interval(cfg.refresh_s, self.refresh_health)
        self.refresh_health()

    def action_refresh(self) -> None:
        self.refresh_health()

    @work(exclusive=True, group="health")
    async def refresh_health(self) -> None:
        rows = await run_health_checks(self.services)
        self.rows = rows
        self._render_rows(rows)

    def _render_rows(self, rows: list[HealthRow]) -> None:
        table = self.query_one("#health-table", DataTable)
        keys = [r.service for r in rows]

        def cells(r: HealthRow) -> tuple:
            lat = f"{r.latency_ms:.0f} ms" if r.latency_ms is not None else "-"
            return safe(r.service), state_text(r.state), lat, safe(r.detail)
        if keys == self._keys:
            for i, r in enumerate(rows):
                for col, value in enumerate(cells(r)):
                    table.update_cell_at((i, col), value)
        else:
            table.clear()
            for r in rows:
                table.add_row(*cells(r), key=r.service)
            self._keys = keys
        self._show_detail(table.cursor_row)

    def _show_detail(self, index: int) -> None:
        if not (0 <= index < len(self.rows)):
            return
        r = self.rows[index]
        text = f"{r.service}: {r.state}\n{r.detail}"
        if r.error:
            text += f"\nlast error: {r.error}"
        self.query_one("#health-detail", Static).update(text)

    @on(DataTable.RowHighlighted, "#health-table")
    def _highlight(self, event: DataTable.RowHighlighted) -> None:
        self._show_detail(event.cursor_row)
