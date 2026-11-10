"""Screen 5: eval results viewer. Shows what the tools wrote; never recomputes a metric."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Input, Static, TabbedContent, TabPane

from hars_memory.tui import evalruns as ev
from hars_memory.tui.screens.base import HarsScreen, add_safe_row


def _fill(table: DataTable, data: ev.Table) -> None:
    table.clear(columns=True)
    table.add_columns(*data.columns)
    for row in data.rows:
        add_safe_row(table, *row)


def _kv(lines: list[tuple[str, str]]) -> str:
    return "\n".join(f"{k:>18}: {v}" for k, v in lines)


class EvalScreen(HarsScreen):
    AUTO_FOCUS = "#runs"
    BINDINGS = [Binding("r", "reload", "Rescan"), Binding("slash", "focus_path", "Path"),
                Binding("escape", "focus_runs", "Back to list", show=False)]

    def __init__(self) -> None:
        super().__init__()
        self.runs: list[ev.RunInfo] = []
        self.detail: ev.RunDetail | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="runs / artifacts directory (Enter to scan)", id="eval-path", compact=True)
        with Horizontal():
            with Vertical(id="eval-left"):
                yield DataTable(id="runs", cursor_type="row", zebra_stripes=True)
                yield Static("", id="eval-status", markup=False)
            with TabbedContent(id="eval-tabs"):
                with TabPane("Summary", id="tab-summary"):
                    with VerticalScroll():
                        yield Static("", id="eval-summary", markup=False)
                with TabPane("Arms", id="tab-arms"):
                    yield DataTable(id="arms", cursor_type="row")
                    yield DataTable(id="deltas", cursor_type="row")
                with TabPane("Strata", id="tab-strata"):
                    yield DataTable(id="strata", cursor_type="row")
                with TabPane("Gates", id="tab-gates"):
                    yield DataTable(id="gates", cursor_type="row")
                with TabPane("Invalid chains", id="tab-invalid"):
                    yield DataTable(id="invalid", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        root = self.services.config.eval_root
        self.query_one("#runs", DataTable).add_columns("run", "verdict", "files")
        if root is not None:
            self.query_one("#eval-path", Input).value = str(root)
            self.scan(root)
        else:
            self.query_one("#eval-status", Static).update(
                "No eval root configured. Type a directory above (or use --eval-root / HARS_MEMORY_TUI_EVAL_ROOT).")
            self.query_one("#eval-path", Input).focus()

    def action_focus_runs(self) -> None:
        self.query_one("#runs", DataTable).focus()

    def action_focus_path(self) -> None:
        self.query_one("#eval-path", Input).focus()

    def action_reload(self) -> None:
        text = self.query_one("#eval-path", Input).value.strip()
        if text:
            self.scan(Path(text).expanduser())

    @on(Input.Submitted, "#eval-path")
    def _path_submitted(self, event: Input.Submitted) -> None:
        if event.value.strip():
            self.scan(Path(event.value.strip()).expanduser())

    @work(exclusive=True, group="eval-scan")
    async def scan(self, root: Path) -> None:
        runs = await asyncio.to_thread(ev.scan_runs, root)
        self.runs = runs
        table = self.query_one("#runs", DataTable)
        table.clear()
        for r in runs:
            verdict = "-"
            if r.metrics_files:
                d = await asyncio.to_thread(ev.load_run, r.path)
                verdict = ev.verdict(d)
            rel = r.name if r.path != root else f"{r.name} (root)"
            add_safe_row(table, rel, verdict, ",".join(m.removesuffix(".json").removesuffix(".jsonl") for m in r.markers) or "metrics", key=str(r.path))
        status = self.query_one("#eval-status", Static)
        status.update(f"{len(runs)} run dir(s) under {root}" if runs else
                      f"no run directories under {root} (looked for run_manifest.json, MANIFEST.json, summary.json, metrics_v2 JSON, *.jsonl budgets)")
        if not runs:
            self.query_one("#eval-summary", Static).update("")

    @on(DataTable.RowHighlighted, "#runs")
    def _highlight(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value:
            self.load(Path(str(event.row_key.value)))

    @work(exclusive=True, group="eval-load")
    async def load(self, path: Path) -> None:
        detail = await asyncio.to_thread(ev.load_run, path)
        self.detail = detail
        self._show(detail)

    def _show(self, d: ev.RunDetail) -> None:
        parts = [f"{d.path}", f"verdict: {ev.verdict(d)}"]
        gates = (d.metrics or {}).get("gates") or {}
        if gates.get("why"):
            parts.append(f"why: {gates['why']}")
        if d.metrics:
            parts += ["", f"-- metrics ({d.metrics_name}) --", _kv(ev.meta_lines(d))]
        parts += ["", "-- manifest --", _kv(ev.manifest_lines(d)), "", "-- spend --", _kv(ev.spend_lines(d))]
        if d.run_budgets:
            parts += ["", "-- run budgets (token parity) --", _kv(ev.budget_lines(d))]
        parts += [f"note: {n}" for n in d.notes]
        self.query_one("#eval-summary", Static).update("\n".join(parts))
        _fill(self.query_one("#arms", DataTable), ev.arms_table(d))
        _fill(self.query_one("#deltas", DataTable), ev.deltas_table(d))
        _fill(self.query_one("#strata", DataTable), ev.strata_table(d))
        _fill(self.query_one("#gates", DataTable), ev.gates_table(d))
        _fill(self.query_one("#invalid", DataTable), ev.invalid_table(d))
