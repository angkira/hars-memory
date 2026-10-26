"""Help screen: keys, and the later phases that are intentionally stubs."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

HELP = """\
hars-memory TUI  (Phase 1 MVP)

SCREENS               1 Health   2 Recall   3 Sessions   4 Projects   5 Eval results   ? Help   q Quit
GLOBAL                Ctrl+T remember a note (confirm modal)      r refresh the current screen
RECALL                Enter / Ctrl+R run    Ctrl+P / Ctrl+N history    o open hit at start_line    y copy chunk id
SESSIONS              / filter    M merge session into MAIN index (confirm)    C manual compaction (confirm)
EVAL                  / change directory    r rescan

WRITES                every write (remember, compact, merge) opens a confirm modal showing project + tool + args.
                      Writes into project 'default' (the shared SRE KB) need a SECOND, typed confirmation.
                      --read-only refuses all writes.  Admin tools (forget, delete_document, consolidate) are
                      OFF; --allow-admin unlocks the client gate but no admin screen exists in the MVP.

NOT IN THE MVP (stubs, by plan .plans/2026-10-26_tui.md)
  Phase 2   Remember/Upsert document forms, Entities/Related/Inspect browser, Sync-status, artifact browser
  Phase 3   Indexing jobs (index-job service, index-batch state reader, migrate/consolidate runner)
  Phase 4   Eval launcher (dry-run gate, live progress, cost tracking)  -- never to be built: graph canvas,
            project CRUD, token management, metric recomputation
  Phase 5   themes, snapshot tests, publish

OPEN API GAPS
  GAP-1     /trace ignores session_id / session_graph  -> session recall runs over MCP without stage events
  GAP-2     no read API for session compactions / graph -> Sessions reads the session dir on disk (same host only)
  GAP-3     no deep health endpoint -> embedder / extractor are probed from this host via GET <base>/models
  GAP-4/6   no job list / progress, no machine-readable progress for index-batch / migrate (Phase 3)
"""


class HelpScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static(HELP, id="help-text", markup=False)
        yield Footer()
