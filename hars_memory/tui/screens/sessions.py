"""Screen 3: session browser (read-only; merge / compact behind the confirm modal).

Inventory comes from the MCP ``memory_session_status`` tool; compactions and the
session graph are read from the session working dirs on disk (plan GAP-2 interim:
there is no API for them yet), so those panes only work on the machine that runs
the service, or with a mounted sessions dir.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Input, Static, TabbedContent, TabPane, Tree

from hars_memory.tui import sessions_fs as fs
from hars_memory.tui.screens.base import HarsScreen, add_safe_row, safe
from hars_memory.tui.screens.modals import CompactModal
from hars_memory.tui.writes import request_write

GAP2 = "needs local access to the sessions dir (API gap GAP-2: no memory_session_compactions / memory_session_graph tool yet)"


def filter_rows(rows: list[dict[str, Any]], needle: str) -> list[dict[str, Any]]:
    needle = needle.strip().lower()
    return [r for r in rows if not needle or needle in str(r.get("session_id", "")).lower()]


def graph_matches(graph: fs.SessionGraph, needle: str) -> list[fs.GraphEntity]:
    needle = needle.strip().lower()
    if not needle:
        return list(graph.entities)
    touching: dict[str, list[str]] = {}
    for r in graph.relations:
        text = f"{r.source} {r.target} {r.description} {r.keywords}".lower()
        touching.setdefault(r.source, []).append(text)
        touching.setdefault(r.target, []).append(text)
    return [e for e in graph.entities
            if needle in f"{e.name} {e.type} {e.description}".lower()
            or any(needle in t for t in touching.get(e.name, ()))]


class SessionsScreen(HarsScreen):
    AUTO_FOCUS = "#sessions"
    BINDINGS = [
        Binding("escape", "focus_table", "Back to list", show=False),
        Binding("slash", "focus_filter", "Filter"),
        Binding("r", "refresh", "Refresh"),
        Binding("M", "merge", "Merge into main"),
        Binding("C", "compact", "Compact (manual)"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.all_rows: list[dict[str, Any]] = []
        self.sid: str | None = None
        self.graph = fs.SessionGraph()
        self.comps: list[fs.Compaction] = []
        self._signature: tuple | None = None
        self.source = ""
        self._server_dir: Path | None = None

    def sessions_root(self) -> Path | None:
        """Configured sessions dir, else the server-reported one when it exists on this host."""
        cfg_dir = self.services.config.sessions_dir
        if cfg_dir is not None:
            return cfg_dir
        return self._server_dir if self._server_dir and self._server_dir.is_dir() else None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sess-left"):
                yield Input(placeholder="filter sessions  (press /)", id="sess-filter", compact=True)
                yield DataTable(id="sessions", cursor_type="row", zebra_stripes=True)
                yield Static("", id="sess-source", markup=False)
            with Vertical(id="sess-right"):
                yield Static("Select a session.", id="sess-title", markup=False)
                with TabbedContent(id="sess-tabs"):
                    with TabPane("Compactions", id="tab-comps"):
                        yield DataTable(id="comps", cursor_type="row")
                        with VerticalScroll():
                            yield Static("", id="comp-detail", markup=False)
                    with TabPane("Graph", id="tab-graph"):
                        yield Input(placeholder="filter entities / relations", id="graph-filter", compact=True)
                        yield Tree("session graph", id="graph")
                    with TabPane("Info", id="tab-info"):
                        with VerticalScroll():
                            yield Static("", id="sess-info", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#sessions", DataTable).add_columns("session", "cmp", "ent", "rel", "chk", "vec", "mrg")
        self.query_one("#comps", DataTable).add_columns("seq", "summary", "facts")
        self.query_one("#graph", Tree).show_root = True
        self.set_interval(self.services.config.refresh_s, self.poll)
        self.refresh_sessions()

    # -- inventory ------------------------------------------------------------

    def action_focus_table(self) -> None:
        self.query_one("#sessions", DataTable).focus()

    def action_focus_filter(self) -> None:
        self.query_one("#sess-filter", Input).focus()

    def action_refresh(self) -> None:
        self.refresh_sessions()

    def poll(self) -> None:
        self.refresh_sessions()

    @work(exclusive=True, group="sessions")
    async def refresh_sessions(self) -> None:
        cfg = self.services.config
        reply = await self.services.call_tool("memory_session_status", {"project": cfg.project})
        rows: list[dict[str, Any]]
        if reply.ok and isinstance(reply.data, dict):
            rows = [r for r in reply.data.get("sessions") or [] if isinstance(r, dict)]
            self.source = f"source: MCP memory_session_status (mode {reply.data.get('mode')})"
            if reply.data.get("sessions_dir"):
                # usable for the file panes only if it exists on this host
                self._server_dir = Path(str(reply.data["sessions_dir"]))
        else:
            root = self.sessions_root()
            why = fs.sessions_dir_status(root)
            if why:
                rows = []
                self.source = f"MCP unavailable ({reply.error}); {why}"
            else:
                rows = await asyncio.to_thread(
                    lambda: [fs.local_session_row(root, s) for s in fs.list_session_ids(root)])
                self.source = f"source: local files {root} (MCP unavailable: {reply.error})"
        self.all_rows = rows
        self._render_rows()
        await self._maybe_reload_detail()

    def _render_rows(self) -> None:
        table = self.query_one("#sessions", DataTable)
        rows = filter_rows(self.all_rows, self.query_one("#sess-filter", Input).value)
        table.clear()
        for r in rows:
            add_safe_row(table, r.get("session_id"), r.get("compactions", 0), r.get("entities", 0),
                         r.get("relations", 0), r.get("chunks_indexed", 0), r.get("vectors_indexed", 0),
                         "yes" if r.get("merged") else "-", key=str(r.get("session_id")))
        if self.sid:
            for i, r in enumerate(rows):
                if r.get("session_id") == self.sid:
                    table.move_cursor(row=i)
                    break
        self.query_one("#sess-source", Static).update(f"{self.source}\n{len(rows)}/{len(self.all_rows)} sessions shown")

    @on(Input.Changed, "#sess-filter")
    def _filter_changed(self) -> None:
        self._render_rows()

    @on(DataTable.RowHighlighted, "#sessions")
    def _row_highlight(self, event: DataTable.RowHighlighted) -> None:
        sid = event.row_key.value if event.row_key is not None else None
        if sid and sid != self.sid:
            self.sid = str(sid)
            self._signature = None
            self.load_detail()

    # -- detail ---------------------------------------------------------------

    def _sig(self) -> tuple | None:
        root = self.sessions_root()
        if root is None or not self.sid:
            return None
        d = root / self.sid
        try:
            return tuple((d / n).stat().st_mtime_ns if (d / n).exists() else 0 for n in (fs.GRAPH_FILE, fs.FULL_DOCS, fs.MERGED_FILE))
        except OSError:
            return None

    async def _maybe_reload_detail(self) -> None:
        if self.sid and self._sig() != self._signature:
            self.load_detail()

    @work(exclusive=True, group="session-detail")
    async def load_detail(self) -> None:
        sid = self.sid
        if not sid:
            return
        cfg = self.services.config
        root = self.sessions_root()
        why = fs.sessions_dir_status(root)
        self._signature = self._sig()
        status = await self.services.call_tool("memory_session_status", {"session_id": sid, "project": cfg.project})
        if why:
            self.comps, self.graph = [], fs.SessionGraph(error=why + " - " + GAP2)
        else:
            self.comps = await asyncio.to_thread(fs.read_compactions, root, sid)
            self.graph = await asyncio.to_thread(fs.read_graph, root, sid)
        if sid != self.sid:
            return  # selection moved on while we were reading
        self._render_detail(sid, status.data if status.ok and isinstance(status.data, dict) else None, why)

    def _render_detail(self, sid: str, status: dict[str, Any] | None, why: str) -> None:
        row = status or next((r for r in self.all_rows if r.get("session_id") == sid), {})
        self.query_one("#sess-title", Static).update(
            f"{sid}  mode={row.get('mode')}  compactions={row.get('compactions')}  "
            f"entities={row.get('entities')}  relations={row.get('relations')}  merged={row.get('merged')}")
        comps = self.query_one("#comps", DataTable)
        comps.clear()
        for c in self.comps:
            add_safe_row(comps, c.seq, " ".join(c.summary.split())[:90], len(c.facts), key=str(c.seq))
        detail = self.query_one("#comp-detail", Static)
        if why:
            detail.update(f"{why}\n{GAP2}")
        elif not self.comps:
            detail.update("No compaction documents found in this session dir.")
        else:
            detail.update("Select a compaction.")
        info = {k: v for k, v in row.items()}
        self.query_one("#sess-info", Static).update("\n".join(f"{k}: {v}" for k, v in info.items()) or "no status")
        self._render_graph()

    @on(DataTable.RowHighlighted, "#comps")
    def _comp_highlight(self, event: DataTable.RowHighlighted) -> None:
        i = event.cursor_row
        if 0 <= i < len(self.comps):
            c = self.comps[i]
            facts = "\n".join(f"  - {f}" for f in c.facts) or "  (none)"
            self.query_one("#comp-detail", Static).update(f"{c.doc_id}  seq {c.seq}\n\nSUMMARY\n{c.summary}\n\nDROPPED-CONTEXT FACTS ({len(c.facts)})\n{facts}")

    @on(Input.Changed, "#graph-filter")
    def _graph_filter(self) -> None:
        self._render_graph()

    def _render_graph(self) -> None:
        tree = self.query_one("#graph", Tree)
        tree.clear()
        g = self.graph
        if g.error:
            tree.root.set_label(safe(f"graph: {g.error}"))
            return
        needle = self.query_one("#graph-filter", Input).value
        shown = graph_matches(g, needle)
        tree.root.set_label(safe(f"session graph: {len(g.entities)} entities, {len(g.relations)} relations"
                                 + (f"  (showing {len(shown)} matching '{needle}')" if needle.strip() else "")))
        for e in sorted(shown, key=lambda x: x.name.lower()):
            node = tree.root.add(safe(f"{e.name}  [{e.type or '?'}]"), expand=False)
            if e.description:
                node.add_leaf(safe(f"= {' '.join(e.description.split())[:160]}"))
            for r in g.relations:
                if r.source == e.name:
                    node.add_leaf(safe(f"-> {r.target}: {' '.join(r.description.split())[:140]}"))
                elif r.target == e.name:
                    node.add_leaf(safe(f"<- {r.source}: {' '.join(r.description.split())[:140]}"))
        tree.root.expand()

    # -- writes ---------------------------------------------------------------

    def action_merge(self) -> None:
        if not self.sid:
            self.app.notify("select a session first", severity="warning")
            return
        row = next((r for r in self.all_rows if r.get("session_id") == self.sid), {})
        notes = [
            f"Merges session '{self.sid}' ({row.get('entities', '?')} session-graph entities, {row.get('compactions', '?')} compactions) into the MAIN index.",
            "This re-ingests a chronological document through the extractor LLM (costs LLM calls) and is a MAIN-INDEX write.",
            "delete_source=false: the session working dir is kept.",
        ]
        request_write(self, self.services, "memory_merge_session", self.services.config.project,
                      {"session_id": self.sid, "delete_source": False}, notes=notes,
                      on_done=lambda _r: self.refresh_sessions())

    def action_compact(self) -> None:
        project = self.services.config.project

        def got(result: dict[str, Any] | None) -> None:
            if result:
                request_write(self, self.services, "memory_compact", result["project"], result["args"],
                              notes=["Stores one compaction document and indexes it (may call the extractor LLM)."],
                              on_done=lambda _r: self.refresh_sessions())
        self.app.push_screen(CompactModal(project, self.sid or ""), got)
