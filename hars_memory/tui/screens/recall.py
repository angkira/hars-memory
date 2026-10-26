"""Screen 2: recall with trace. Streams stage events over SSE when the listener supports the query."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (Button, Checkbox, DataTable, Footer, Header, Input, Label, Select, Static,
                             TabbedContent, TabPane)

from hars_memory.tui.clients import trace as trace_client
from hars_memory.tui.recall_view import build_view, summarize_stage
from hars_memory.tui.screens.base import HarsScreen, add_safe_row
from hars_memory.tui.state import HistoryStore

MODES = [("hybrid", "hybrid"), ("local", "local"), ("global", "global"), ("naive", "naive")]


def _split(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


class RecallScreen(HarsScreen):
    BINDINGS = [
        Binding("ctrl+r", "run", "Run recall"),
        Binding("ctrl+p", "history_prev", "Prev query", show=False),
        Binding("ctrl+n", "history_next", "Next query", show=False),
        Binding("o", "open_hit", "Open hit"),
        Binding("y", "copy_hit", "Copy chunk id"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.history: HistoryStore | None = None
        self._hist_pos = 0
        self.last_result: dict[str, Any] | None = None
        self.last_view: dict[str, Any] | None = None
        self.stage_payloads: list[tuple[str, dict[str, Any]]] = []

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        cfg = self.services.config
        yield Header()
        with Vertical(id="recall-form"):
            with Horizontal(classes="frow"):
                yield Label("question")
                yield Input(placeholder="natural-language question (write it in the language of the documents)", id="q", compact=True)
                yield Button("Run", id="run", variant="primary", compact=True)
            with Horizontal(classes="frow"):
                yield Label("project")
                yield Input(cfg.project, id="project", compact=True)
                yield Label("mode")
                yield Select(MODES, value="hybrid", allow_blank=False, id="mode", compact=True)
                yield Label("top_k")
                yield Input(placeholder="server default", id="top_k", compact=True, type="integer")
            with Horizontal(classes="frow"):
                yield Label("ll_keywords")
                yield Input(placeholder="specific entities / codes, comma separated", id="ll", compact=True)
                yield Label("hl_keywords")
                yield Input(placeholder="themes, comma separated", id="hl", compact=True)
            with Horizontal(classes="frow"):
                yield Label("session_id")
                yield Input(placeholder="optional", id="sid", compact=True)
                yield Checkbox("session_graph", id="sg", compact=True)
                yield Checkbox("stage trace (SSE)", cfg.trace_url is not None, id="trace", compact=True)
        yield Static("Enter a question and press Ctrl+R (or Run).", id="recall-status", markup=False)
        with Horizontal(id="recall-main"):
            with Vertical(id="stages-pane"):
                yield DataTable(id="stages", cursor_type="row")
                with VerticalScroll(id="stage-detail-scroll"):
                    yield Static("", id="stage-detail", markup=False)
            with TabbedContent(id="recall-tabs"):
                with TabPane("Packed context", id="tab-context"):
                    with VerticalScroll():
                        yield Label("Knowledge-graph entities", classes="sect")
                        yield DataTable(id="kg-ent", cursor_type="row")
                        yield Label("Knowledge-graph relations", classes="sect")
                        yield DataTable(id="kg-rel", cursor_type="row")
                        yield Label("Document chunks", classes="sect")
                        yield DataTable(id="chunks", cursor_type="row")
                with TabPane("Scored hits", id="tab-hits"):
                    yield DataTable(id="hits", cursor_type="row")
                    with VerticalScroll(id="hit-detail-scroll"):
                        yield Static("", id="hit-detail", markup=False)
                with TabPane("Meta", id="tab-meta"):
                    with VerticalScroll():
                        yield Static("", id="meta", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        cfg = self.services.config
        self.history = HistoryStore(cfg.state_dir)
        self._hist_pos = len(self.history.items)
        self.query_one("#stages", DataTable).add_columns("#", "stage", "t+ms", "stage ms", "summary")
        self.query_one("#kg-ent", DataTable).add_columns("entity", "type", "description")
        self.query_one("#kg-rel", DataTable).add_columns("from", "to", "description")
        self.query_one("#chunks", DataTable).add_columns("ref", "score", "source", "content")
        self.query_one("#hits", DataTable).add_columns("fused", "dense", "sparse", "location", "section", "chunk_id")
        self.query_one("#q", Input).focus()

    # -- form <-> args --------------------------------------------------------

    def form_args(self) -> dict[str, Any]:
        q = lambda i: self.query_one(i, Input).value.strip()  # noqa: E731
        args: dict[str, Any] = {
            "question": q("#q"),
            "project": q("#project") or self.services.config.project,
            "mode": str(self.query_one("#mode", Select).value),
            "context_only": True,
        }
        if q("#top_k"):
            args["top_k"] = int(q("#top_k"))
        if q("#ll"):
            args["ll_keywords"] = _split(q("#ll"))
        if q("#hl"):
            args["hl_keywords"] = _split(q("#hl"))
        if q("#sid"):
            args["session_id"] = q("#sid")
        if self.query_one("#sg", Checkbox).value:
            args["session_graph"] = True
        return args

    def load_entry(self, entry: dict[str, Any]) -> None:
        self.query_one("#q", Input).value = str(entry.get("question", ""))
        self.query_one("#project", Input).value = str(entry.get("project", self.services.config.project))
        self.query_one("#top_k", Input).value = str(entry.get("top_k", ""))
        self.query_one("#ll", Input).value = ", ".join(entry.get("ll_keywords") or [])
        self.query_one("#hl", Input).value = ", ".join(entry.get("hl_keywords") or [])
        self.query_one("#sid", Input).value = str(entry.get("session_id", ""))
        self.query_one("#sg", Checkbox).value = bool(entry.get("session_graph"))
        if entry.get("mode") in {m for m, _ in MODES}:
            self.query_one("#mode", Select).value = entry["mode"]

    # -- actions ------------------------------------------------------------

    def action_run(self) -> None:
        self.run_recall()

    @on(Button.Pressed, "#run")
    def _run_pressed(self) -> None:
        self.run_recall()

    @on(Input.Submitted, "#q")
    def _q_submitted(self) -> None:
        self.run_recall()

    def action_history_prev(self) -> None:
        if self.history and self.history.items and self._hist_pos > 0:
            self._hist_pos -= 1
            self.load_entry(self.history.items[self._hist_pos])

    def action_history_next(self) -> None:
        if self.history and self._hist_pos < len(self.history.items) - 1:
            self._hist_pos += 1
            self.load_entry(self.history.items[self._hist_pos])

    def _selected_fused(self) -> dict[str, Any] | None:
        table = self.query_one("#hits", DataTable)
        fused = (self.last_view or {}).get("fused") or []
        return fused[table.cursor_row] if 0 <= table.cursor_row < len(fused) else None

    def action_copy_hit(self) -> None:
        hit = self._selected_fused()
        if hit:
            self.app.copy_to_clipboard(str(hit.get("chunk_id")))
            self.app.notify(f"copied {hit.get('chunk_id')}")

    def action_open_hit(self) -> None:
        hit = self._selected_fused()
        if not hit or not hit.get("file_path"):
            self.app.notify("select a row in 'Scored hits' first", severity="warning")
            return
        path = Path(str(hit["file_path"])).expanduser()
        if not path.is_file():
            self.app.notify(f"not found on this host: {path}", severity="warning")
            return
        line = int(hit.get("start_line") or 1)
        pager = shlex.split(os.environ.get("EDITOR") or os.environ.get("PAGER") or "less")
        with self.app.suspend():
            subprocess.run([*pager, f"+{line}", str(path)], check=False)

    # -- running --------------------------------------------------------------

    def _reset_output(self) -> None:
        for sel in ("#stages", "#kg-ent", "#kg-rel", "#chunks", "#hits"):
            self.query_one(sel, DataTable).clear()
        for sel in ("#stage-detail", "#hit-detail", "#meta"):
            self.query_one(sel, Static).update("")
        self.stage_payloads = []
        self.last_result = self.last_view = None

    def _add_stage(self, event: str, payload: dict[str, Any]) -> None:
        table = self.query_one("#stages", DataTable)
        lat, summary = summarize_stage(event, payload)
        t_ms = payload.get("t_ms")
        add_safe_row(table, payload.get("seq", len(self.stage_payloads)), event,
                     f"{t_ms:.0f}" if isinstance(t_ms, (int, float)) else "-", lat, summary)
        self.stage_payloads.append((event, payload))

    @work(exclusive=True, group="recall")
    async def run_recall(self) -> None:
        status = self.query_one("#recall-status", Static)
        args = self.form_args()
        if not args["question"]:
            status.update("A question is required.")
            return
        self._reset_output()
        svc = self.services
        want_trace = self.query_one("#trace", Checkbox).value
        notes: list[str] = []
        use_trace = False
        if want_trace:
            if not svc.config.trace_url:
                notes.append("trace listener not configured -> MCP (no stage events)")
            elif not trace_client.supports(args):
                notes.append("session_id/session_graph are not accepted by /trace (GAP-1) -> MCP (no stage events)")
            else:
                use_trace = True
        status.update("running ...")
        started = time.monotonic()
        result: dict[str, Any] | None = None
        error = ""
        via = "MCP memory_recall"
        if use_trace:
            via = "SSE trace"
            got_any = False
            async for event, payload in trace_client.stream_trace(svc, args):
                if event == "transport_error":
                    if got_any:
                        error = payload.get("error", "stream interrupted")
                    else:
                        notes.append(f"trace listener unavailable ({payload.get('error')}) -> MCP (no stage events)")
                        use_trace = False
                    break
                got_any = True
                self._add_stage(event, payload)
                if event == "result":
                    result = payload
                elif event == "error":
                    error = f"{payload.get('error_type')}: {payload.get('error')}"
        if not use_trace and result is None and not error:
            via = "MCP memory_recall"
            reply = await svc.call_tool("memory_recall", args)
            if reply.ok and isinstance(reply.data, dict):
                result = reply.data
            else:
                error = reply.error or "recall failed"
                if isinstance(reply.data, dict):
                    result = reply.data
            self._add_stage("mcp_result", {"seq": 0, "t_ms": (time.monotonic() - started) * 1000, "ok": not error,
                                           "error": error, "context_chars": len((result or {}).get("context") or "")})
        total_ms = (time.monotonic() - started) * 1000
        if self.history is not None:
            self.history.add({k: v for k, v in args.items() if k != "context_only"})
            self._hist_pos = len(self.history.items)
        if result is not None:
            self._show_result(result, args, via, total_ms)
        suffix = "  " + " | ".join(notes) if notes else ""
        if error:
            status.update(f"FAILED via {via}: {error}{suffix}")
        else:
            status.update((self.last_view or {}).get("summary", "") + suffix)

    def _show_result(self, result: dict[str, Any], args: dict[str, Any], via: str, total_ms: float) -> None:
        view = build_view(result, args, via, total_ms)
        self.last_result, self.last_view = result, view
        for sel, key in (("#kg-ent", "entities"), ("#kg-rel", "relations"), ("#chunks", "chunks"), ("#hits", "fused_rows")):
            table = self.query_one(sel, DataTable)
            for row in view[key]:
                add_safe_row(table, *row)
        self.query_one("#meta", Static).update(view["meta"])

    # -- detail panes ---------------------------------------------------------

    @on(DataTable.RowHighlighted, "#stages")
    def _stage_highlight(self, event: DataTable.RowHighlighted) -> None:
        i = event.cursor_row
        if 0 <= i < len(self.stage_payloads):
            name, payload = self.stage_payloads[i]
            shown = payload if name != "result" else {k: v for k, v in payload.items() if k not in ("context", "hybrid")}
            self.query_one("#stage-detail", Static).update(f"{name}\n{json.dumps(shown, indent=1, ensure_ascii=False, default=str)[:4000]}")

    @on(DataTable.RowHighlighted, "#hits")
    def _hit_highlight(self, event: DataTable.RowHighlighted) -> None:
        hit = self._selected_fused()
        if hit:
            self.query_one("#hit-detail", Static).update(
                f"{hit.get('chunk_id')}  {hit.get('file_path')}:{hit.get('start_line', '?')}\n\n{str(hit.get('content') or hit.get('snippet') or '')[:3000]}")
