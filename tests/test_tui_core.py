"""TUI core (no Textual): config, adapters, SSE parser, health probes, local-file readers, eval reader."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hars_memory.trace import format_sse
from hars_memory.tui import evalruns as ev
from hars_memory.tui import sessions_fs as fs
from hars_memory.tui.cli import INSTALL_HINT, build_parser, config_from_args
from hars_memory.tui.clients import trace as tc
from hars_memory.tui.clients.base import Services
from hars_memory.tui.config import from_env
from hars_memory.tui.context_parse import parse_context
from hars_memory.tui.health import parse_model_ref, run_health_checks
from hars_memory.tui.state import HistoryStore
from tests.tui_support import BASE, RECALL_RESULT, FakeHars, make_config, write_session


# ---------------------------------------------------------------- config

def test_config_defaults_and_server_env_fallbacks() -> None:
    cfg = from_env({})
    assert cfg.base_url == "http://127.0.0.1:8099"
    assert cfg.mcp_url == "http://127.0.0.1:8099/mcp"
    assert cfg.trace_url == "http://127.0.0.1:8098"
    assert not cfg.allow_admin and not cfg.read_only

    cfg = from_env({"HARS_MEMORY_MCP_HTTP_HOST": "0.0.0.0", "HARS_MEMORY_MCP_HTTP_PORT": "8307",
                    "HARS_MEMORY_TRACE_HTTP_PORT": "8398", "HARS_MEMORY_ACCESS_TOKEN": "s3cret",
                    "HARS_MEMORY_SESSIONS_DIR": "/data/sessions"})
    assert cfg.base_url == "http://127.0.0.1:8307"
    assert cfg.trace_url == "http://127.0.0.1:8398"
    assert cfg.token == "s3cret" and cfg.sessions_dir == Path("/data/sessions")
    assert "s3cret" not in repr(cfg)


def test_config_tui_vars_win_and_trace_can_be_disabled() -> None:
    cfg = from_env({"HARS_MEMORY_TUI_URL": "https://mem.example:9000/", "HARS_MEMORY_TUI_TOKEN": "t",
                    "HARS_MEMORY_ACCESS_TOKEN": "other", "HARS_MEMORY_TRACE_URL": "http://h:1/trace",
                    "HARS_MEMORY_TUI_ALLOW_ADMIN": "1", "HARS_MEMORY_TUI_READ_ONLY": "true"})
    assert cfg.base_url == "https://mem.example:9000" and cfg.token == "t"
    assert cfg.trace_url == "http://h:1"
    assert cfg.allow_admin and cfg.read_only
    assert from_env({"HARS_MEMORY_TRACE_HTTP_PORT": "0"}).trace_url is None


def test_cli_flags_override_env_and_token_comes_from_named_env_var() -> None:
    args = build_parser().parse_args(["--url", "http://x:1", "--trace-url", "off", "--token-env", "MY_TOK",
                                      "--read-only", "--sessions-dir", "/s", "--no-probe"])
    cfg = config_from_args(args, {"MY_TOK": "abc", "HARS_MEMORY_TUI_TOKEN": "ignored"})
    assert cfg.base_url == "http://x:1" and cfg.mcp_url == "http://x:1/mcp"
    assert cfg.trace_url is None and cfg.token == "abc" and cfg.read_only and not cfg.probe_models
    assert cfg.sessions_dir == Path("/s")
    assert not any(a.dest == "token" for a in build_parser()._actions), "no --token VALUE flag may exist"


def test_main_prints_install_hint_without_textual(monkeypatch, capsys) -> None:
    from hars_memory.tui import cli

    monkeypatch.setitem(sys.modules, "textual", None)  # makes `import textual` raise ImportError
    for name in [m for m in sys.modules if m.startswith("hars_memory.tui.app") or m.startswith("hars_memory.tui.screens")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert cli.main([]) == 2
    assert INSTALL_HINT.splitlines()[0] in capsys.readouterr().err


def test_memory_tui_subcommand_delegates(monkeypatch) -> None:
    from hars_memory import cli as memory_cli

    seen = {}
    import hars_memory.tui.cli as tui_cli

    def fake_main(argv=None) -> int:
        seen["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(tui_cli, "main", fake_main)
    with pytest.raises(SystemExit) as exc:
        memory_cli.main(["tui", "--read-only", "--screen", "recall"])
    assert exc.value.code == 0 and seen["argv"] == ["--read-only", "--screen", "recall"]


def test_tui_import_is_light() -> None:
    code = ("import sys, hars_memory.tui.app; "
            "bad=[m for m in ('lightrag','torch','hars_memory.mcp_server') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    pytest.importorskip("textual")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------- SSE + adapters

def test_sse_parser_handles_comments_multiline_and_chunked_frames() -> None:
    body = (b": ping\n\n" + format_sse("bm25_hits", {"hits": [1, 2]}) + b"event: x\ndata: {\"a\":\ndata: 1}\n\n"
            + format_sse("done", {"ok": True}))
    frames = tc.parse_sse(body.decode().split("\n"))
    assert [f[0] for f in frames] == ["bm25_hits", "x", "done"]
    assert frames[0][1] == {"hits": [1, 2]} and frames[1][1] == {"a": 1}


def test_trace_supports_only_what_the_listener_honours() -> None:
    assert tc.supports({"question": "q", "project": "default", "ll_keywords": ["a"], "context_only": True})
    assert not tc.supports({"question": "q", "session_id": "s1"})
    assert not tc.supports({"question": "q", "session_graph": True})
    assert tc.supports({"question": "q", "session_id": "", "session_graph": False})
    params = tc.query_params({"question": "q", "ll_keywords": ["a b", "c"], "top_k": 5})
    assert params["ll_keywords"] == "a b,c" and params["top_k"] == "5" and params["context_only"] == "true"


async def test_stream_trace_yields_events_in_wire_order_and_reports_transport_errors() -> None:
    fake = FakeHars()
    svc = Services(make_config(token="tok"), fake.client())
    events = [e async for e, _ in tc.stream_trace(svc, {"question": "q"})]
    assert events == ["trace_start", "bm25_hits", "vector_hits", "fusion", "rerank", "graph_expand", "context", "result", "done"]
    # the listener has no header auth: the token goes in the query, and only there
    url = next(u for u, _ in fake.headers_seen if "/trace?" in u)
    assert "access_token=tok" in url
    fake.trace_up = False
    out = [x async for x in tc.stream_trace(svc, {"question": "q"})]
    assert out[0][0] == "transport_error" and "ConnectError" in out[0][1]["error"]
    await svc.aclose()


async def test_call_tool_envelope_gates_and_auth_scoping() -> None:
    fake = FakeHars()
    svc = Services(make_config(token="tok"), fake.client())
    ok = await svc.call_tool("memory_status", {"project": "default"})
    assert ok.ok and ok.data["node_count"] == 120
    sent = fake.tool_calls("memory_status")[0][1]
    assert sent["access_token"] == "tok"

    fake.tool_errors["memory_recall"] = "boom"
    bad = await svc.call_tool("memory_recall", {"question": "q"})
    assert not bad.ok and bad.error == "boom"

    refused = await svc.call_tool("memory_forget", {"x": 1})
    assert not refused.ok and "admin" in refused.error
    assert not fake.tool_calls("memory_forget")

    ro = Services(make_config(read_only=True), fake.client())
    refused = await ro.call_tool("memory_remember", {"title": "t", "content": "c"})
    assert not refused.ok and "read-only" in refused.error and not fake.tool_calls("memory_remember")

    admin = Services(make_config(allow_admin=True), fake.client())
    fake.tools["memory_forget"] = {"ok": True}
    assert (await admin.call_tool("memory_forget", {})).ok

    fake.mcp_up = False
    down = await svc.call_tool("memory_status", {})
    assert not down.ok and "ConnectError" in down.error  # typed Unavailable, never an exception
    await svc.aclose()


async def test_probe_never_sends_the_hars_token() -> None:
    fake = FakeHars()
    svc = Services(make_config(token="tok"), fake.client())
    await svc.probe("http://emb.test/v1/models")
    await svc.healthz()
    seen = dict(fake.headers_seen)
    assert "authorization" not in seen["http://emb.test/v1/models"]
    assert seen[BASE + "/healthz"]["authorization"] == "Bearer tok"
    await svc.aclose()


# ---------------------------------------------------------------- health

def test_parse_model_ref() -> None:
    assert parse_model_ref("gemma @ http://h/v1 (http)") == ("gemma", "http://h/v1", "http")
    assert parse_model_ref("local-model") == ("local-model", None, None)


async def test_health_rows_cover_every_service_and_survive_dead_ones() -> None:
    fake = FakeHars()
    svc = Services(make_config(jobs_url="http://jobs.test"), fake.client())
    rows = {r.service: r for r in await run_health_checks(svc)}
    assert rows["HARS API"].state == "OK" and rows["MCP endpoint"].state == "OK" and rows["Trace listener"].state == "OK"
    assert rows["Index"].state == "OK" and "nodes=120" in rows["Index"].detail
    assert rows["Embedder"].state == "OK" and "embeddinggemma" in rows["Embedder"].detail
    assert rows["Extractor LLM"].state == "DOWN" and rows["Extractor LLM"].error  # fake extractor is down
    assert rows["Qdrant"].state == "N/A"
    assert rows["Index-job service"].state == "DOWN"  # /healthz on jobs.test has no fake route -> 404

    fake.health = None
    fake.mcp_up = False
    fake.trace_up = False
    rows = {r.service: r for r in await run_health_checks(svc)}
    assert rows["HARS API"].state == "DOWN" and rows["MCP endpoint"].state == "DOWN"
    assert rows["Trace listener"].state == "WARN" and rows["Index"].state == "DOWN"
    await svc.aclose()


async def test_health_qdrant_row_from_memory_status() -> None:
    fake = FakeHars()
    fake.tools["memory_status"]["storage"]["vector"] = {
        "backend": "QdrantVectorDBStorage", "reachable": True, "points_count": 99, "dim": 768, "workspace": "w",
        "collections": {"chunks": {"exists": True}, "entities": {"exists": False}}}
    svc = Services(make_config(), fake.client())
    row = next(r for r in await run_health_checks(svc) if r.service == "Qdrant")
    assert row.state == "WARN" and "entities" in row.detail
    fake.tools["memory_status"]["storage"]["vector"] = {"backend": "QdrantVectorDBStorage", "reachable": False, "error": "refused"}
    row = next(r for r in await run_health_checks(svc) if r.service == "Qdrant")
    assert row.state == "DOWN" and row.error == "refused"
    await svc.aclose()


# ---------------------------------------------------------------- parsers / readers

def test_parse_context_sections() -> None:
    s = parse_context(RECALL_RESULT["context"])
    assert [e["entity"] for e in s.entities] == ["Payments Queue"]
    assert s.relations[0]["entity1"] == "Maria"
    assert s.chunks[0]["chunk_id"] == "chunk-a" and s.references == {"1": "docs/queues.md"}
    assert not parse_context("Sorry, no context.[no-context]").has_content


def test_session_files_compactions_and_graph(tmp_path: Path) -> None:
    write_session(tmp_path)
    comps = fs.read_compactions(tmp_path, "sg1")
    assert [c.seq for c in comps] == [1, 2]
    assert comps[0].summary == "Billing migration notes." and comps[0].facts == ["The retry limit is 7.", "Backoff starts at 250 ms."]
    g = fs.read_graph(tmp_path, "sg1")
    assert {e.name for e in g.entities} == {"Payments Queue", "Maria", "Exponential Backoff"}
    rel = next(r for r in g.relations if r.source == "Maria")
    assert rel.target == "Payments Queue" and rel.keywords == "owner" and rel.weight == 1.0
    row = fs.local_session_row(tmp_path, "sg1")
    assert (row["compactions"], row["entities"], row["relations"]) == (2, 3, 2)
    assert fs.list_session_ids(tmp_path) == ["sg1"]


def test_session_files_fail_soft(tmp_path: Path) -> None:
    assert "not readable" in fs.sessions_dir_status(tmp_path / "nope")
    assert "not configured" in fs.sessions_dir_status(None)
    assert fs.read_graph(tmp_path / "nope", "x").error
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / fs.GRAPH_FILE).write_text("<graphml><graph>")  # torn write
    assert "unreadable" in fs.read_graph(tmp_path, "s").error
    (tmp_path / "s" / fs.FULL_DOCS).write_text("{not json")
    assert fs.read_compactions(tmp_path, "s") == []


def test_history_store_roundtrip_and_never_persists_tokens(tmp_path: Path) -> None:
    h = HistoryStore(tmp_path, max_items=3)
    for i in range(5):
        h.add({"question": f"q{i}", "access_token": "nope"})
    assert [e["question"] for e in HistoryStore(tmp_path, max_items=3).items] == ["q2", "q3", "q4"]
    assert "nope" not in (tmp_path / "recall-history.json").read_text()


METRICS = {
    "meta": {"records": 42, "chains": 2, "sessions": 2, "questions": 14, "primary_b": 1536, "treatment": "nc",
             "validity": {"dropped_chains": {"chain_bad@B1536": ["chunk clipped"]}, "incomplete_qids": {"q9": ["nc"]}}},
    "summary": {"baseline": {"n": 14, "n_answerable": 12, "score": 1.04, "anchor_hit": 0.4, "hallucination": 0.25},
                "nc": {"n": 14, "n_answerable": 12, "score": 0.916, "anchor_hit": 0.5, "hallucination": 0.18}},
    "deltas": {"nc-baseline": {"metrics": {"score": {"point": -0.125, "ci_lo": -0.5, "ci_hi": 0.25, "n_chains": 2, "wilcoxon_p": 1.0}}}},
    "gates": {"verdict": "INCONCLUSIVE", "why": "underpowered: 2 chain(s) (< 8)",
              "win_rules": [{"name": "win_score_ci", "ok": False, "rule": "CI lower >= -0.05", "observed": "ci_lo=-0.500"}],
              "fail_rules": [{"name": "fail_score_bad", "ok": True, "rule": "d_score < -0.10", "observed": "point=-0.125"}]},
    "slices": {"baseline": {"HEAD": {"n": 6, "score": 0.75, "anchor_hit": 0.33, "clean_anchor_hit": 0.16}}},
    "longsession": {"strata": {"multi_session_reasoning": {"declared": "genuine multi-hop (L4/L5)", "per_arm": {
        "nc": {"n": 4, "score": 1.5, "anchor_hit": 0.5, "clean_anchor_hit": 0.25, "rubric_hit_frac": 0.6}}}}},
}


def make_run(root: Path) -> Path:
    run = root / "run-a"
    run.mkdir(parents=True)
    (run / "gate6_metrics.json").write_text(json.dumps(METRICS))
    (run / "run_manifest.json").write_text(json.dumps({
        "profile": "p1", "models": {"agent": "m-a", "judge": "m-j"}, "embedder": {"id": "gemma", "source": "server", "flags": []},
        "judge_same_family": False, "started_utc": "2026-10-26T00:00:00Z", "ended_utc": "2026-10-26T01:00:00Z",
        "argv": ["run.py", "--api-key", "SECRET", "--token=SECRET2", "--key-env", "MY_KEY", "--limit", "2"],
        "spend": {"budget_usd": 5.0, "spent_usd": 1.2345, "calls": 80, "pi_usd": 1.0}}))
    (run / "chain_budget.jsonl").write_text(
        json.dumps({"chain_id": "c0", "mem_budget": 1536, "valid": True, "invalid_reasons": []}) + "\n"
        + json.dumps({"chain_id": "c1", "mem_budget": 1536, "valid": False, "invalid_reasons": ["ratio 0.7 outside window"]}) + "\n")
    (run / "run_budget.json").write_text(json.dumps({"run_ratio": 0.991, "in_window": True}))
    return run


def test_eval_reader_shows_what_tools_wrote_and_masks_secrets(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run = make_run(root)
    (root / "empty").mkdir()
    found = ev.scan_runs(root)
    assert [r.name for r in found] == ["run-a"] and found[0].metrics_files == ["gate6_metrics.json"]
    d = ev.load_run(run)
    assert ev.verdict(d) == "INCONCLUSIVE"
    arms = ev.arms_table(d)
    assert arms.rows[0][0] == "baseline" and "1.040" in arms.rows[0] and "0.916" in arms.rows[1]
    assert ev.deltas_table(d).rows[0][:3] == ["nc-baseline", "score", "-0.125"]
    assert [r[0] for r in ev.gates_table(d).rows] == ["WIN", "FAIL"]
    strata = ev.strata_table(d)
    assert any("multi_session_reasoning" in r[1] and r[2] == "nc" for r in strata.rows)
    assert any(r[0] == "metrics_v2 slices" and r[1] == "HEAD" for r in strata.rows)
    invalid = {r[0]: r for r in ev.invalid_table(d).rows}
    assert "ratio 0.7 outside window" in invalid["c1"][3] and "c0" not in invalid
    assert "chunk clipped" in invalid["chain_bad@B1536"][3] and "missing nc" in invalid["q9"][3]
    manifest = dict(ev.manifest_lines(d))
    assert manifest["profile"] == "p1" and "gemma" in manifest["embedder"] and "m-a" in manifest["models"]
    cmd = manifest["command"]
    assert "SECRET" not in cmd and "MY_KEY" in cmd and "--limit 2" in cmd
    assert dict(ev.spend_lines(d))["spent_usd"] == "1.2345"


def test_eval_reader_summary_json_layout_and_failed_sessions(tmp_path: Path) -> None:
    run = tmp_path / "r"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({
        "run_id": "x", "failed_sessions": ["s9"], "llm_cost_usd": {"answer": 0.5},
        "by_arm_budget": {"baseline@B512": {"n": 6, "n_valid": 6, "anchor_hit_rate": 1.0, "by_qtype": {"single": {"n": 6, "anchor_hit_rate": 1.0, "rubric_hit_mean": 0.9, "latency_s_mean": 1.1}}}}}))
    (run / "done.jsonl").write_text(json.dumps({"sid": "s1", "arm": "nc-chunks", "B": 512, "valid": False, "reasons": ["raw recall empty"]}) + "\n")
    d = ev.load_run(run)
    assert ev.arms_table(d).rows[0][0] == "baseline@B512"
    assert ev.strata_table(d).rows[0][:2] == ["summary.json", "single"]
    inv = ev.invalid_table(d).rows
    assert ["s1", "nc-chunks B=512", "done.jsonl", "raw recall empty"] in inv and any(r[0] == "s9" for r in inv)
    assert dict(ev.spend_lines(d)) == {"answer": "0.5000"}
    assert ev.scan_runs(tmp_path / "missing") == []
