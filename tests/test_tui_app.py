"""Pilot tests: every screen driven headless against the fake HTTP backend (no sockets, no live server)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.widgets import Button, Checkbox, DataTable, Input, Static, Tree  # noqa: E402

from hars_memory.tui.app import HarsTuiApp  # noqa: E402
from hars_memory.tui.screens.modals import ConfirmWriteModal  # noqa: E402
from tests.test_tui_core import make_run  # noqa: E402
from tests.tui_support import FakeHars, make_config, write_session  # noqa: E402

SIZE = (150, 48)


async def settle(app, pilot, rounds: int = 3) -> None:
    for _ in range(rounds):
        await pilot.pause()
        await app.workers.wait_for_complete()


def cells(table: DataTable) -> list[list[str]]:
    return [[str(c) for c in table.get_row_at(i)] for i in range(table.row_count)]


def make_app(fake: FakeHars, tmp_path: Path, **cfg) -> HarsTuiApp:
    cfg.setdefault("state_dir", tmp_path / "state")
    return HarsTuiApp(make_config(**cfg), http=fake.client())


# ------------------------------------------------------------------ health

async def test_health_screen_lists_every_service(tmp_path: Path) -> None:
    fake = FakeHars()
    app = make_app(fake, tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)
        rows = {r[0]: r for r in cells(app.screen.query_one("#health-table", DataTable))}
        banner = str(app.screen.query_one("#health-banner", Static).render())
    assert rows["HARS API"][1] == "OK" and rows["MCP endpoint"][1] == "OK"
    assert rows["Embedder"][1] == "OK" and "embeddinggemma" in rows["Embedder"][3]
    assert rows["Extractor LLM"][1] == "DOWN" and rows["Qdrant"][1] == "N/A"
    assert "token=none" in banner and "writes=confirm-gated" in banner


async def test_health_with_dead_backend_shows_down_rows_not_a_traceback(tmp_path: Path) -> None:
    fake = FakeHars()
    fake.health, fake.mcp_up, fake.trace_up = None, False, False
    app = make_app(fake, tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot)
        rows = {r[0]: r for r in cells(app.screen.query_one("#health-table", DataTable))}
        app.screen.query_one("#health-table", DataTable).move_cursor(row=0)
        await settle(app, pilot, 1)
        detail = str(app.screen.query_one("#health-detail", Static).render())
    assert rows["HARS API"][1] == "DOWN" and rows["MCP endpoint"][1] == "DOWN" and rows["Index"][1] == "DOWN"
    assert "ConnectError" in detail


# ------------------------------------------------------------------ recall

async def run_recall(app, pilot, **fields) -> None:
    await pilot.press("f2")
    await settle(app, pilot, 2)
    scr = app.screen
    scr.query_one("#q", Input).value = fields.pop("q", "retry limit")
    for sel, value in fields.items():
        widget = scr.query_one(f"#{sel}")
        widget.value = value
    await pilot.press("ctrl+r")
    await settle(app, pilot, 4)


async def test_recall_streams_stage_events_in_order_and_renders_packed_context(tmp_path: Path) -> None:
    fake = FakeHars()
    app = make_app(fake, tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        await run_recall(app, pilot, ll="payments queue, Maria", hl="retry policy")
        scr = app.screen
        stages = [r[1] for r in cells(scr.query_one("#stages", DataTable))]
        ents = cells(scr.query_one("#kg-ent", DataTable))
        rels = cells(scr.query_one("#kg-rel", DataTable))
        chunks = cells(scr.query_one("#chunks", DataTable))
        hits = cells(scr.query_one("#hits", DataTable))
        status = str(scr.query_one("#recall-status", Static).render())
        meta = str(scr.query_one("#meta", Static).render())
    assert stages == ["trace_start", "bm25_hits", "vector_hits", "fusion", "rerank", "graph_expand", "context", "result", "done"]
    assert ents[0][:2] == ["Payments Queue", "artifact"] and rels[0][:2] == ["Maria", "Payments Queue"]
    assert chunks[0][:3] == ["1", "0.910", "docs/queues.md"]  # score joined from hybrid.fused_chunks by chunk_id
    assert hits[0][0] == "0.910" and hits[0][3] == "docs/queues.md:12" and hits[0][5] == "chunk-a"
    assert "SSE trace" in status and "keyword_source=caller" in status and "1 KG entities" in status
    assert "rerank: backend=off" in meta and "disabled_by_default" in meta
    # the SSE request carried the form values
    import httpx

    params = httpx.URL(next(u for m, u, _ in fake.calls if "/trace?" in u)).params
    assert params["question"] == "retry limit" and params["ll_keywords"] == "payments queue,Maria"
    assert params["hl_keywords"] == "retry policy" and params["project"] == "default"
    assert not fake.tool_calls("memory_recall"), "SSE path must not also call MCP"


async def test_recall_with_session_graph_goes_over_mcp_and_says_why(tmp_path: Path) -> None:
    fake = FakeHars()
    fake.tools["memory_recall"] = dict(fake.tools["memory_recall"], session_graph={
        "status": "ok", "keyword_source": "llm", "entities": 2, "relations": 1, "latency_ms": 300.0})
    app = make_app(fake, tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("f2")
        await settle(app, pilot, 1)
        app.screen.query_one("#sg", Checkbox).value = True
        await run_recall(app, pilot, sid="sg1")
        scr = app.screen
        stages = [r[1] for r in cells(scr.query_one("#stages", DataTable))]
        status = str(scr.query_one("#recall-status", Static).render())
    assert stages == ["mcp_result"]
    assert "GAP-1" in status and "MCP memory_recall" in status and "keyword_source=llm" in status
    (tool, args), = fake.tool_calls("memory_recall")
    assert args["session_id"] == "sg1" and args["session_graph"] is True and args["context_only"] is True
    assert not any("/trace?" in u for _, u, _ in fake.calls)


async def test_recall_falls_back_to_mcp_when_the_trace_listener_is_down(tmp_path: Path) -> None:
    fake = FakeHars()
    fake.trace_up = False
    app = make_app(fake, tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        await run_recall(app, pilot)
        status = str(app.screen.query_one("#recall-status", Static).render())
        chunk_rows = cells(app.screen.query_one("#chunks", DataTable))
    assert "trace listener unavailable" in status and "MCP" in status
    assert chunk_rows and len(fake.tool_calls("memory_recall")) == 1


async def test_recall_reports_server_errors_and_requires_a_question(tmp_path: Path) -> None:
    fake = FakeHars()
    fake.tools["memory_recall"] = {"ok": False, "error": "index is warming"}
    app = make_app(fake, tmp_path, trace_url=None)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("f2")
        await settle(app, pilot, 1)
        await pilot.press("ctrl+r")
        await settle(app, pilot, 2)
        assert "question is required" in str(app.screen.query_one("#recall-status", Static).render())
        await run_recall(app, pilot)
        status = str(app.screen.query_one("#recall-status", Static).render())
    assert status.startswith("FAILED") and "index is warming" in status


async def test_recall_trace_stream_cut_midway_is_reported_not_fatal(tmp_path: Path) -> None:
    fake = FakeHars()
    fake.trace_truncate_after = 3  # no result/done frames
    app = make_app(fake, tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        await run_recall(app, pilot)
        stages = [r[1] for r in cells(app.screen.query_one("#stages", DataTable))]
        assert stages == ["trace_start", "bm25_hits", "vector_hits"]
        assert app.screen.last_result is None  # nothing invented for the missing result


async def test_recall_history_is_kept_and_recallable(tmp_path: Path) -> None:
    fake = FakeHars()
    app = make_app(fake, tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        await run_recall(app, pilot, q="first question", ll="alpha")
        await run_recall(app, pilot, q="second question")
        scr = app.screen
        scr.query_one("#q", Input).value = ""
        await pilot.press("ctrl+p")
        assert scr.query_one("#q", Input).value == "second question"
        await pilot.press("ctrl+p")
        assert scr.query_one("#q", Input).value == "first question"
        assert scr.query_one("#ll", Input).value == "alpha"
    assert (tmp_path / "state" / "recall-history.json").is_file()


# ------------------------------------------------------------------ sessions

async def open_sessions(app, pilot, sessions_dir: Path | None) -> None:
    await pilot.press("f3")
    await settle(app, pilot, 4)


async def test_sessions_browser_lists_compactions_and_graph_from_the_session_dir(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    write_session(root)
    fake = FakeHars()
    app = make_app(fake, tmp_path, sessions_dir=root)
    async with app.run_test(size=SIZE) as pilot:
        await open_sessions(app, pilot, root)
        scr = app.screen
        rows = cells(scr.query_one("#sessions", DataTable))
        assert [r[0] for r in rows] == ["sg1", "other-9"] and rows[1][6] == "yes"
        comps = cells(scr.query_one("#comps", DataTable))
        assert [c[0] for c in comps] == ["1", "2"] and "Billing migration notes." in comps[0][1] and comps[0][2] == "2"
        scr.query_one("#comps", DataTable).move_cursor(row=0)
        await settle(app, pilot, 1)
        detail = str(scr.query_one("#comp-detail", Static).render())
        assert "The retry limit is 7." in detail and "DROPPED-CONTEXT FACTS (2)" in detail
        tree = scr.query_one("#graph", Tree)
        assert "3 entities, 2 relations" in str(tree.root.label)
        labels = [str(n.label) for n in tree.root.children]
        assert labels == ["Exponential Backoff  [concept]", "Maria  [person]", "Payments Queue  [artifact]"]
        maria = next(n for n in tree.root.children if str(n.label).startswith("Maria"))
        assert any("-> Payments Queue" in str(leaf.label) for leaf in maria.children)
        # search / filter: sessions list and graph
        scr.query_one("#sess-filter", Input).value = "other"
        await settle(app, pilot, 2)
        assert [r[0] for r in cells(scr.query_one("#sessions", DataTable))] == ["other-9"]
        scr.query_one("#sess-filter", Input).value = ""
        await settle(app, pilot, 2)
        scr.query_one("#sessions", DataTable).move_cursor(row=0)  # the selection followed the filter; go back to sg1
        await settle(app, pilot, 3)
        scr.query_one("#graph-filter", Input).value = "backoff"
        await settle(app, pilot, 1)
        assert [str(n.label) for n in scr.query_one("#graph", Tree).root.children] == [
            "Exponential Backoff  [concept]", "Payments Queue  [artifact]"]


async def test_sessions_without_local_access_say_so_instead_of_crashing(tmp_path: Path) -> None:
    fake = FakeHars()
    app = make_app(fake, tmp_path)  # no sessions_dir; server-reported dir does not exist here
    async with app.run_test(size=SIZE) as pilot:
        await open_sessions(app, pilot, None)
        scr = app.screen
        assert len(cells(scr.query_one("#sessions", DataTable))) == 2  # inventory still comes from MCP
        assert "GAP-2" in str(scr.query_one("#comp-detail", Static).render())
        assert "GAP-2" in str(scr.query_one("#graph", Tree).root.label)


async def test_sessions_fall_back_to_files_when_mcp_is_down(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    write_session(root)
    fake = FakeHars()
    fake.mcp_up = False
    app = make_app(fake, tmp_path, sessions_dir=root)
    async with app.run_test(size=SIZE) as pilot:
        await open_sessions(app, pilot, root)
        scr = app.screen
        assert [r[0] for r in cells(scr.query_one("#sessions", DataTable))] == ["sg1"]
        assert "local files" in str(scr.query_one("#sess-source", Static).render())


async def press_button(app, pilot, button_id: str) -> None:
    app.screen.query_one(f"#{button_id}", Button).press()
    await settle(app, pilot, 3)


async def test_merge_is_gated_by_a_confirm_modal_and_default_needs_a_second_typed_confirmation(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    write_session(root)
    fake = FakeHars()
    app = make_app(fake, tmp_path, sessions_dir=root)
    async with app.run_test(size=SIZE) as pilot:
        await open_sessions(app, pilot, root)
        await pilot.press("M")
        await settle(app, pilot, 2)
        assert isinstance(app.screen, ConfirmWriteModal)
        text = str(app.screen.query_one("#confirm-text", Static).render())
        assert "memory_merge_session" in text and "project: default" in text and '"session_id": "sg1"' in text
        assert "MAIN-INDEX write" in text
        # cancel: nothing sent
        await press_button(app, pilot, "cancel")
        assert not fake.tool_calls("memory_merge_session")
        # confirm once: still not sent (project 'default' is guarded)
        await pilot.press("M")
        await settle(app, pilot, 2)
        await press_button(app, pilot, "confirm")
        assert isinstance(app.screen, ConfirmWriteModal) and not fake.tool_calls("memory_merge_session")
        assert "SECOND CONFIRMATION" in str(app.screen.query_one("#confirm-guard", Static).render())
        assert app.screen.query_one("#confirm", Button).disabled
        app.screen.query_one("#confirm-typed", Input).value = "defaul"
        await settle(app, pilot, 1)
        assert app.screen.query_one("#confirm", Button).disabled
        app.screen.query_one("#confirm-typed", Input).value = "default"
        await settle(app, pilot, 1)
        await press_button(app, pilot, "confirm")
        await settle(app, pilot, 3)
        assert not isinstance(app.screen, ConfirmWriteModal)
    (tool, args), = fake.tool_calls("memory_merge_session")
    assert args["session_id"] == "sg1" and args["project"] == "default" and args["delete_source"] is False


async def test_non_default_project_needs_a_single_confirmation(tmp_path: Path) -> None:
    fake = FakeHars()
    app = make_app(fake, tmp_path, sessions_dir=tmp_path / "x", project="lab")
    async with app.run_test(size=SIZE) as pilot:
        await open_sessions(app, pilot, None)
        await pilot.press("M")
        await settle(app, pilot, 2)
        await press_button(app, pilot, "confirm")
        await settle(app, pilot, 2)
    assert len(fake.tool_calls("memory_merge_session")) == 1


async def test_read_only_mode_refuses_writes_without_opening_a_modal(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    write_session(root)
    fake = FakeHars()
    app = make_app(fake, tmp_path, sessions_dir=root, read_only=True)
    async with app.run_test(size=SIZE) as pilot:
        await open_sessions(app, pilot, root)
        await pilot.press("M")
        await settle(app, pilot, 2)
        assert not isinstance(app.screen, ConfirmWriteModal)
    assert not fake.tool_calls("memory_merge_session")


async def test_remember_goes_through_the_modal_and_the_default_guard(tmp_path: Path) -> None:
    fake = FakeHars()
    app = make_app(fake, tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        await settle(app, pilot, 2)
        await pilot.press("ctrl+t")
        await settle(app, pilot, 2)
        app.screen.query_one("#f-title", Input).value = "retry-limit"
        app.screen.query_one("#f-content").text = "The retry limit is 7."
        await press_button(app, pilot, "ok")
        assert isinstance(app.screen, ConfirmWriteModal)
        await press_button(app, pilot, "confirm")      # stage 1
        app.screen.query_one("#confirm-typed", Input).value = "default"
        await settle(app, pilot, 1)
        await press_button(app, pilot, "confirm")      # stage 2
        await settle(app, pilot, 2)
    (tool, args), = fake.tool_calls("memory_remember")
    assert args["title"] == "retry-limit" and args["project"] == "default" and args["content"] == "The retry limit is 7."


# ------------------------------------------------------------------ projects

async def test_projects_screen_lists_projects_and_shows_status(tmp_path: Path) -> None:
    fake = FakeHars()
    app = make_app(fake, tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("f4")
        await settle(app, pilot, 4)
        scr = app.screen
        rows = cells(scr.query_one("#projects", DataTable))
        detail = str(scr.query_one("#project-detail", Static).render())
        scr.query_one("#projects", DataTable).move_cursor(row=1)
        await settle(app, pilot, 3)
        detail_lab = str(scr.query_one("#project-detail", Static).render())
    assert [r[0] for r in rows] == ["default", "lab"] and rows[1][3] == "read"
    assert "nodes: 120" in detail and "embeddinggemma" in detail and "NanoVectorDBStorage" in detail
    assert "memory_status" in [c[0] for c in fake.tool_calls()] or fake.tool_calls("memory_status")
    assert "project: lab" in detail_lab or "lab" in detail_lab


# ------------------------------------------------------------------ eval

async def test_eval_viewer_shows_manifest_arms_strata_gates_and_invalid_chains(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    make_run(root)
    fake = FakeHars()
    app = make_app(fake, tmp_path, eval_root=root)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("f5")
        await settle(app, pilot, 5)
        scr = app.screen
        runs = cells(scr.query_one("#runs", DataTable))
        summary = str(scr.query_one("#eval-summary", Static).render())
        arms = cells(scr.query_one("#arms", DataTable))
        deltas = cells(scr.query_one("#deltas", DataTable))
        strata = cells(scr.query_one("#strata", DataTable))
        gates = cells(scr.query_one("#gates", DataTable))
        invalid = cells(scr.query_one("#invalid", DataTable))
    assert runs[0][:2] == ["run-a", "INCONCLUSIVE"]
    assert "INCONCLUSIVE" in summary and "underpowered" in summary and "p1" in summary and "gemma" in summary
    assert "spent_usd: 1.2345" in summary.replace("  ", " ") or "1.2345" in summary
    assert "SECRET" not in summary and "MY_KEY" in summary
    assert arms[0][0] == "baseline" and deltas[0][0] == "nc-baseline"
    assert any("multi_session_reasoning" in r[1] for r in strata)
    assert {r[0] for r in gates} == {"WIN", "FAIL"}
    assert any(r[0] == "c1" and "ratio 0.7" in r[3] for r in invalid)


async def test_eval_viewer_handles_an_empty_or_missing_directory(tmp_path: Path) -> None:
    fake = FakeHars()
    app = make_app(fake, tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("f5")
        await settle(app, pilot, 2)
        assert "No eval root configured" in str(app.screen.query_one("#eval-status", Static).render())
        app.screen.query_one("#eval-path", Input).value = str(tmp_path / "missing")
        await app.screen.query_one("#eval-path", Input).action_submit()
        await settle(app, pilot, 3)
        assert "no run directories" in str(app.screen.query_one("#eval-status", Static).render())


# ------------------------------------------------------------------ shell

async def test_help_screen_lists_stubbed_phases_and_gaps(tmp_path: Path) -> None:
    fake = FakeHars()
    app = make_app(fake, tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("question_mark")
        await settle(app, pilot, 2)
        text = str(app.screen.query_one("#help-text", Static).render())
    for needle in ("Phase 2", "Phase 3", "Phase 4", "GAP-1", "GAP-2", "SECOND, typed confirmation"):
        assert needle in text


async def test_screen_modes_switch_and_keep_their_state(tmp_path: Path) -> None:
    fake = FakeHars()
    app = make_app(fake, tmp_path)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("f2")
        await settle(app, pilot, 1)
        app.screen.query_one("#q", Input).value = "kept"
        await pilot.press("f1")
        await settle(app, pilot, 1)
        assert app.current_mode == "health"
        await pilot.press("f2")
        await settle(app, pilot, 1)
        assert app.screen.query_one("#q", Input).value == "kept"


async def test_untrusted_text_with_markup_characters_renders_literally(tmp_path: Path) -> None:
    fake = FakeHars()
    hostile = dict(fake.tools["memory_recall"])
    hostile["context"] = hostile["context"].replace("retry limit 7", "[/oops] [bold red]x[/] [concept]")
    fake.tools["memory_recall"] = hostile
    app = make_app(fake, tmp_path, trace_url=None)
    async with app.run_test(size=SIZE) as pilot:
        await run_recall(app, pilot)
        ents = cells(app.screen.query_one("#kg-ent", DataTable))
    assert "[/oops] [bold red]x[/] [concept]" in ents[0][2]


async def test_sessions_screen_picks_up_a_changed_graph_on_the_next_poll(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    write_session(root)
    fake = FakeHars()
    app = make_app(fake, tmp_path, sessions_dir=root)
    async with app.run_test(size=SIZE) as pilot:
        await open_sessions(app, pilot, root)
        scr = app.screen
        assert "3 entities" in str(scr.query_one("#graph", Tree).root.label)
        graph = root / "sg1" / "graph_chunk_entity_relation.graphml"
        graph.write_text(graph.read_text().replace(
            "</graph>", '<node id="New Entity"><data key="d1">concept</data></node></graph>'))
        scr.poll()  # what the interval timer calls
        await settle(app, pilot, 4)
        assert "4 entities" in str(scr.query_one("#graph", Tree).root.label)
