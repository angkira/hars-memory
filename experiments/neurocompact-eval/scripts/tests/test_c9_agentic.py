"""Amendment C.9: the agentic track (run_agentic, providers.chat_tools, oh_workspaces, judge --outcome, metrics_agentic, home_run).
Offline: scripted fake LLM, FakeHars in the C.8 project layout, a tiny fixture repo, local git remotes."""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

import anchors as A
import chains_to_longsession as C2L
import home_run as H
import judge_v2 as J
import longsession as L
import metrics_agentic as MA
import oh_workspaces as OW
import providers as P
import run_agentic as RA
import run_longsession as RL
import test_c7_home as T7
import test_home_run as thr
from oh_chain_fixture import make_chain

NEED = "NEEDLE_s1_1"


# --------------------------------------------------------------------------- fixtures
def make_ws(root: str, with_anchor: str | None = None) -> str:
    os.makedirs(os.path.join(root, "pkg"), exist_ok=True)
    open(os.path.join(root, "README.md"), "w").write("# widget\nA tiny repo.\n")
    open(os.path.join(root, "pkg", "util.py"), "w").write(
        "def helper(x):\n    return x + 1\n\n" + "\n".join(f"# filler line {i} about helpers" for i in range(400)) + "\n")
    open(os.path.join(root, "pkg", "core.py"), "w").write(
        "class Widget:\n    value = 41\n" + (f"    token = '{with_anchor}'\n" if with_anchor else ""))
    open(os.path.join(root, "blob.bin"), "wb").write(b"\x00\x01\x02" * 100)
    return root


def make_sessions(d: str, n: int = 2) -> dict:
    out = {}
    for i in range(n):
        s, _ = C2L.convert_chain(make_chain(f"chain_s{i}_c0", f"sess{i}", n_events=(6 + i, 5, 4)))
        C2L.write_session(s, d)
        out[s["sid"]] = s
    return out


def make_old_run(d: str, sessions: dict, note=True) -> None:
    os.makedirs(d, exist_ok=True)
    rows = []
    for sid, s in sessions.items():
        summ = f"SUMMARY<{sid}> The session fixed a bug.  Two spaces and trailing newline\n" + (f"Mentions {'NEEDLE_s3_0'} already.\n" if sid.endswith("s0_c0") else "")
        for q in s["questions"]:
            for arm, mem in (("baseline", summ), ("nc-note-only", f"NOTE<{sid}> short note")):
                rows.append({"rid": f"{q['qid']}::{arm}::b1536", "qid": q["qid"], "base_qid": q["qid"], "sid": sid, "chain_id": sid,
                             "arm": arm, "mem_budget": 1536, "valid": True, "is_end": True, "memory_text": mem})
    with open(os.path.join(d, "records.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def res(content, usage=None, stop=None, cost=0.001):
    """A scripted ToolChatResult. content: list of ('text', s) | ('tool', name, input)."""
    blocks, calls, text = [], [], ""
    n = 0
    for c in content:
        if c[0] == "text":
            blocks.append({"type": "text", "text": c[1]})
            text += c[1]
        else:
            n += 1
            cid = f"tu_{abs(hash((c[1], json.dumps(c[2], sort_keys=True)))) % 10**6}_{n}"
            blocks.append({"type": "tool_use", "id": cid, "name": c[1], "input": c[2]})
            calls.append({"id": cid, "name": c[1], "input": c[2]})
    u = {"prompt_tokens": 1000, "completion_tokens": 50, "cached_tokens": 0, "cache_write_tokens": 0, "reported_cost": None}
    u.update(usage or {})
    return P.ToolChatResult(text=text, tool_calls=calls, content=blocks, stop_reason=stop or ("tool_use" if calls else "end_turn"),
                            usage=u, cost_usd=cost, cost_source="table", model="fake", profile="fake", latency_s=0.01)


class ScriptLLM:
    """Plays a list of results (or callables(system, messages, tools, tool_choice) -> result)."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.model = "fake"

    def step(self, system, messages, tools, tool_choice=None):
        self.calls.append({"system": system, "n_messages": len(messages), "tools": [t["name"] for t in tools], "tool_choice": tool_choice})
        i = len(self.calls) - 1
        item = self.script[min(i, len(self.script) - 1)]
        return item(system, messages, tools, tool_choice) if callable(item) else item


def exec_for(ws_root, hars=None, project="nc-e1-x"):
    ws = RA.Workspace(ws_root)
    mem = RA.MemoryTool(hars, project, 6000) if hars is not None else None
    return RA.exec_tool_factory(ws, mem), ws


# --------------------------------------------------------------------------- anchor matcher
class AnchorMatcher(unittest.TestCase):
    CASES = [("see NEEDLE_s1_1 here", "NEEDLE_s1_1"), ("xNEEDLE_s1_1", "NEEDLE_s1_1"), ("needle_s1_1.", "NEEDLE_S1_1"),
             ("a.b.c is set", "a.b.c"), ("path/to/file.py:12", "file.py"), ("nothing", "NEEDLE_s1_1"), ("anything", ""), ("", "x"),
             ("  pad  ", "  pad ")]

    def test_one_function_equals_the_judge_and_longsession_matchers(self):
        for text, anc in self.CASES:
            want = bool(anc.strip()) and bool(text) and J.anchor_re(anc).search(text) is not None
            self.assertEqual(A.anchor_found(text, anc), want, (text, anc))
            if anc.strip() and text:
                self.assertEqual(A.anchor_found(text, anc), RL.anchor_pattern(anc.strip()).search(text) is not None)

    def test_same_text_same_verdict_for_tool_results_summary_flags_dropped_and_answer(self):
        anchor, text = "NEEDLE_s1_1", "log: value NEEDLE_s1_1 = 107"
        with tempfile.TemporaryDirectory() as d:
            ws = make_ws(os.path.join(d, "ws"), with_anchor=None)
            open(os.path.join(ws, "pkg", "core.py"), "a").write(f"    note = '{text}'\n")
            ex, _ = exec_for(ws)
            out, err, kind, _ = ex("grep", {"pattern": "NEEDLE"})
        q = {"gold_anchor": anchor}
        verdicts = {
            "tool_result": A.anchor_found(out, anchor),
            "summary_flag": RA.subset_flags(q, text, "")["anchor_in_summary"],
            "dropped_flag": RA.subset_flags(q, "", text)["anchor_in_dropped"],
            "answer": A.anchor_found(text, anchor),
            "trace_scan": RA.run_agent(ScriptLLM([res([("tool", "grep", {"pattern": "NEEDLE"})]),
                                                  res([("tool", "answer", {"text": "x"})])]), "s", "q", RA.tool_specs(False),
                                       lambda n, a: (text, False, "workspace", {}), max_steps=4, token_cap=10**6, anchor=anchor
                                       )["calls"][0]["anchor_in_result"],
        }
        self.assertEqual(set(verdicts.values()), {True}, verdicts)
        neg = "log: value NEEDLE_s1_10 = 1"
        self.assertFalse(A.anchor_found(neg, anchor))
        self.assertFalse(RA.subset_flags(q, neg, neg)["anchor_in_dropped"])


# --------------------------------------------------------------------------- sandbox
class Sandbox(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = make_ws(os.path.join(self.d.name, "ws"))
        self.outside = os.path.join(self.d.name, "secret.txt")
        open(self.outside, "w").write("TOP-SECRET-OUTSIDE\n")
        os.symlink(self.outside, os.path.join(self.root, "evil_link"))
        os.symlink(self.d.name, os.path.join(self.root, "evil_dir"))

    def tearDown(self):
        self.d.cleanup()

    def backends(self):
        return [False] + ([True] if RA.shutil.which("rg") else [])

    def test_no_path_escape(self):
        for rg in self.backends():
            ws = RA.Workspace(self.root, use_rg=rg)
            for bad in ("../secret.txt", "../../etc/passwd", "evil_link", "evil_dir/secret.txt", "pkg/../../secret.txt", "..", "\x00"):
                for call in (lambda p: ws.read_file(p), lambda p: ws.ls(p), lambda p: ws.grep("SECRET", p)):
                    with self.assertRaises(RA.ToolError, msg=(rg, bad)):
                        call(bad)
            # an absolute host path never reads the host: it is re-rooted inside the workspace (and does not exist there)
            with self.assertRaises(RA.ToolError):
                ws.read_file("/etc/passwd")
            self.assertNotIn("TOP-SECRET", ws.grep("TOP-SECRET", "."))        # the symlinks are not followed by grep
            self.assertIn("Widget", ws.read_file("/pkg/core.py"))              # "/" = the workspace root
            self.assertIn("Widget", ws.read_file(os.path.join(self.root, "pkg", "core.py")))   # an absolute path inside is fine

    def test_output_caps(self):
        for rg in self.backends():
            ws = RA.Workspace(self.root, out_chars=900, use_rg=rg)
            r = ws.read_file("pkg/util.py")
            self.assertLessEqual(len(r), 900 + 200)
            self.assertIn("continue with start=", r)
            g = RA.Workspace(self.root, out_chars=300, use_rg=rg).grep("filler", ".")
            self.assertLessEqual(len(g), 300 + 200)
            self.assertIn("truncated", g)
        ws = RA.Workspace(self.root)
        r = ws.read_file("pkg/util.py", 1, 10 ** 6)
        self.assertLessEqual(r.count("\n") + 1, RA.READ_MAX_LINES + 2)
        self.assertEqual(ws.read_file("pkg/util.py", 2, 3).splitlines()[0].split("\t")[0], "2")
        os.makedirs(os.path.join(self.root, "many"))
        for i in range(RA.LS_MAX_ENTRIES + 30):
            open(os.path.join(self.root, "many", f"f{i:04d}.txt"), "w").write("x")
        self.assertIn("more entries not shown", RA.Workspace(self.root, out_chars=10 ** 6).ls("many"))

    def test_grep_backends_agree_and_errors_are_clean(self):
        outs = {rg: RA.Workspace(self.root, use_rg=rg).grep(r"class \w+", ".") for rg in self.backends()}
        self.assertTrue(all("pkg/core.py:1:class Widget:" in o for o in outs.values()), outs)
        for rg in self.backends():
            with self.assertRaises(RA.ToolError):
                RA.Workspace(self.root, use_rg=rg).grep("(unclosed", ".")
            self.assertEqual(RA.Workspace(self.root, use_rg=rg).grep("zzz_not_there", "."), "(no matches)")
        with self.assertRaises(RA.ToolError):
            RA.Workspace(self.root).read_file("blob.bin")

    def test_dispatcher_turns_tool_errors_into_error_results_and_unknown_tools_into_errors(self):
        ex, _ = exec_for(self.root)
        t, err, kind, _ = ex("read_file", {"path": "../secret.txt"})
        self.assertTrue(err)
        self.assertIn("outside the workspace", t)
        self.assertTrue(ex("rm", {})[1])


# --------------------------------------------------------------------------- leaks
class Leaks(unittest.TestCase):
    def test_transcript_strings_are_not_in_the_workspace_and_the_scan_catches_a_planted_leak(self):
        with tempfile.TemporaryDirectory() as d:
            s, _ = C2L.convert_chain(make_chain("chain_s0_c0", "sess0"))
            ws = make_ws(os.path.join(d, "ws"))
            probes = RA.transcript_probes(s)
            self.assertGreaterEqual(len(probes), 5)
            # distinctive transcript-only strings (the NEEDLE markers and the message heads) are absent from the repo
            self.assertEqual(RA.leak_scan(ws, probes, extra_strings=[q["gold_anchor"] for q in s["questions"] if q.get("gold_anchor")]), [])
            RA.check_workspace_clean(ws, s, [d + "/campaign"])
            # negative control: plant the transcript -> detected, and the run refuses
            leak = os.path.join(ws, "docs_session.txt")
            open(leak, "w").write(L.serialize(s["messages"][:3]))
            self.assertTrue(RA.leak_scan(ws, probes))
            with self.assertRaises(SystemExit):
                RA.check_workspace_clean(ws, s, [])
            os.remove(leak)
            # the workspace may not overlap campaign / transcript dirs
            with self.assertRaises(SystemExit):
                RA.check_workspace_clean(ws, s, [d])

    def test_agent_cannot_find_a_transcript_string_through_the_tools(self):
        with tempfile.TemporaryDirectory() as d:
            ws = make_ws(os.path.join(d, "ws"))
            ex, _ = exec_for(ws)
            for pat in ("NEEDLE", "value NEEDLE_s1_1"):
                self.assertEqual(ex("grep", {"pattern": pat})[0], "(no matches)")


# --------------------------------------------------------------------------- providers.chat_tools
class Transport:
    def __init__(self, raw):
        self.raw, self.reqs = raw, []

    def __call__(self, req, timeout):
        self.reqs.append((req.full_url, dict(req.header_items()), json.loads(req.data)))
        return 200, self.raw


ANTH_RAW = {"content": [{"type": "text", "text": "thinking"}, {"type": "tool_use", "id": "tu1", "name": "grep", "input": {"pattern": "x"}}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 100, "output_tokens": 30, "cache_read_input_tokens": 4000,
                                                 "cache_creation_input_tokens": 500}}
OAI_RAW = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "c1", "type": "function",
                                                                   "function": {"name": "grep", "arguments": "{\"pattern\": \"x\"}"}}]},
                        "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 900, "completion_tokens": 20, "cost": 0.0042}}
HIST = [{"role": "user", "content": "Q"},
        {"role": "assistant", "content": [{"type": "text", "text": "hm"}, {"type": "tool_use", "id": "t1", "name": "ls", "input": {"path": "."}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "README.md"}]}]


class ChatTools(unittest.TestCase):
    def test_anthropic_wire_caching_pricing_and_no_temperature(self):
        prof = P.load_profile("anthropic")
        tr = Transport(ANTH_RAW)
        r = P.chat_tools("agent", "SYS", HIST, RA.tool_specs(True), profile=prof, max_tokens=300, temperature=0, environ={"ANTHROPIC_API_KEY": "k"},
                         transport=tr, budget=P.Budget(1.0))
        url, headers, body = tr.reqs[0]
        self.assertTrue(url.endswith("/messages"))
        self.assertNotIn("temperature", body)                                  # Anthropic 5.x rejects it: policy "omit"
        self.assertFalse(r.sent_temperature)
        self.assertEqual(body["system"][0]["cache_control"], {"type": "ephemeral"})     # static prefix (system incl. the summary)
        self.assertEqual(body["messages"][-1]["content"][-1]["cache_control"], {"type": "ephemeral"})   # moving breakpoint
        self.assertEqual([t["name"] for t in body["tools"]], ["ls", "read_file", "grep", "memory_recall", "answer"])
        self.assertEqual(r.tool_calls, [{"id": "tu1", "name": "grep", "input": {"pattern": "x"}}])
        self.assertEqual(r.stop_reason, "tool_use")
        self.assertEqual((r.usage["prompt_tokens"], r.usage["cached_tokens"], r.usage["cache_write_tokens"]), (4600, 4000, 500))
        want = prof.cost_usd("claude-haiku-5-5", 100, 30, cache_read=4000, cache_write=500, cache_ttl="5m")
        self.assertAlmostEqual(r.cost_usd, want)
        self.assertLess(want, prof.cost_usd("claude-haiku-5-5", 4600, 30))        # reads are cheaper than plain input
        self.assertIsNone(HIST[-1]["content"][-1].get("cache_control"))          # the caller's history is not mutated

    def test_forced_tool_choice_and_cache_off(self):
        prof = P.load_profile("anthropic", cache=False)
        tr = Transport(ANTH_RAW)
        P.chat_tools("agent", "SYS", HIST, RA.tool_specs(False), profile=prof, tool_choice={"type": "tool", "name": "answer"},
                     environ={"ANTHROPIC_API_KEY": "k"}, transport=tr)
        body = tr.reqs[0][2]
        self.assertEqual(body["tool_choice"], {"type": "tool", "name": "answer"})
        self.assertNotIn("cache_control", json.dumps(body))

    def test_openai_compatible_wire(self):
        prof = P.load_profile("openrouter")
        tr = Transport(OAI_RAW)
        r = P.chat_tools("agent", "SYS", HIST, RA.tool_specs(True), profile=prof, model="anthropic/claude-haiku-5.5", temperature=0,
                         tool_choice={"type": "tool", "name": "answer"}, environ={"OPENROUTER_API_KEY": "k"}, transport=tr)
        url, headers, body = tr.reqs[0]
        self.assertTrue(url.endswith("/chat/completions"))
        self.assertEqual(body["messages"][0], {"role": "system", "content": "SYS"})
        self.assertEqual(body["messages"][2]["tool_calls"][0]["function"], {"name": "ls", "arguments": "{\"path\": \".\"}"})
        self.assertEqual(body["messages"][3], {"role": "tool", "tool_call_id": "t1", "content": "README.md"})
        self.assertEqual(body["tools"][0]["function"]["name"], "ls")
        self.assertEqual(body["tool_choice"], {"type": "function", "function": {"name": "answer"}})
        self.assertEqual(r.tool_calls, [{"id": "c1", "name": "grep", "input": {"pattern": "x"}}])
        self.assertEqual(r.wire, "openai")
        self.assertEqual((r.usage["prompt_tokens"], r.usage["completion_tokens"]), (900, 20))
        self.assertAlmostEqual(r.cost_usd, 0.0042)                            # provider-reported
        self.assertEqual(r.content[0]["type"], "tool_use")                    # canonical block, ready for the history

    def test_budget_is_charged_with_the_actual_cost_and_refuses_when_exhausted(self):
        prof = P.load_profile("anthropic")
        b = P.Budget(1.0)
        P.chat_tools("agent", "S", HIST, RA.tool_specs(False), profile=prof, environ={"ANTHROPIC_API_KEY": "k"}, transport=Transport(ANTH_RAW), budget=b)
        self.assertGreater(b.spent, 0)
        with self.assertRaises(P.BudgetExceeded):
            P.chat_tools("agent", "S", HIST, RA.tool_specs(False), profile=prof, environ={"ANTHROPIC_API_KEY": "k"},
                         transport=Transport(ANTH_RAW), budget=P.Budget(1e-9))


# --------------------------------------------------------------------------- the loop
class Loop(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.ws = make_ws(os.path.join(self.d.name, "ws"), with_anchor=NEED)
        self.hars = RL.FakeHars()
        self.pid = "nc-e1-sess0"
        self.hars.register_project(self.pid)
        self.hars.upsert(self.pid, "/nc-eval/x/j1.md", f"## Dropped context facts\n- line 101: value {NEED} = 107 in tests/test_a.py\n- other fact")

    def tearDown(self):
        self.d.cleanup()

    def run_loop(self, script, mem=True, max_steps=6, cap=10 ** 7, anchor=NEED):
        ex, _ = exec_for(self.ws, self.hars if mem else None, self.pid)
        llm = ScriptLLM(script)
        return RA.run_agent(llm, "SYS", "Q?", RA.tool_specs(mem), ex, max_steps=max_steps, token_cap=cap, anchor=anchor), llm

    def test_tokens_are_summed_across_steps_including_cache(self):
        tr, _ = self.run_loop([
            res([("tool", "ls", {})], usage={"prompt_tokens": 1000, "completion_tokens": 40, "cache_write_tokens": 900}),
            res([("tool", "grep", {"pattern": "Widget"})], usage={"prompt_tokens": 1500, "completion_tokens": 25, "cached_tokens": 900}),
            res([("tool", "answer", {"text": "41"})], usage={"prompt_tokens": 2100, "completion_tokens": 10, "cached_tokens": 1500}),
        ])
        s = tr["steps"]
        self.assertEqual([x["input_total"] for x in s], [1000, 1500, 2100])
        self.assertEqual([x["cache_read"] for x in s], [0, 900, 1500])
        self.assertEqual([x["cache_write"] for x in s], [900, 0, 0])
        self.assertEqual([x["input_uncached"] for x in s], [100, 600, 600])
        self.assertEqual(tr["tokens_total"], 4600 + 75)
        q = {"qid": "q1", "question": "Q?", "gold_anchor": NEED, "answerable": True}
        row = RA.build_row("r", "sess0", "sum", q, tr, summary_text="SUM", flags={}, wall_s=1.0, args_meta={})
        self.assertEqual((row["tokens_input_total"], row["tokens_cache_read"], row["tokens_cache_write"], row["tokens_input_uncached"],
                          row["tokens_output"], row["tokens_total"]), (4600, 2400, 900, 1300, 75, 4675))
        self.assertEqual(row["tokens_fresh"], 1300 + 900 + 75)
        self.assertAlmostEqual(row["cost_usd"], 0.003)
        self.assertEqual(row["steps"], 3)
        self.assertEqual(row["tool_calls"], 2)
        self.assertEqual(row["tool_calls_by_type"], {"ls": 1, "grep": 1, "answer": 1})
        self.assertEqual([c["step"] for c in row["step_tokens"]], [1, 2, 3])

    def test_step_cap_forces_the_answer_tool_on_the_last_step(self):
        loop_forever = res([("tool", "ls", {})])
        tr, llm = self.run_loop([loop_forever] * 2 + [lambda s, m, t, tc: res([("tool", "answer", {"text": "forced"})]) if tc else loop_forever], max_steps=3)
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual([c["tool_choice"] for c in llm.calls], [None, None, {"type": "tool", "name": "answer"}])
        self.assertEqual((tr["final"], tr["status"]), ("forced", "answered"))
        self.assertTrue(tr["steps"][-1]["forced_answer_step"])
        # the "last step" notice reaches the model in the last tool-result message only
        self.assertIn(RA.LAST_STEP_NOTICE, json.dumps(tr["messages"][-2]))
        # a vendor that ignores the forcing: the loop still stops at max_steps with no answer
        tr2, llm2 = self.run_loop([loop_forever], max_steps=3)
        self.assertEqual((len(llm2.calls), tr2["status"], tr2["final"]), (3, "step_cap_no_answer", ""))

    def test_token_cap_stops_without_an_answer(self):
        tr, llm = self.run_loop([res([("tool", "ls", {})], usage={"prompt_tokens": 600, "completion_tokens": 10})], max_steps=9, cap=1000)
        self.assertEqual(tr["status"], "token_cap")
        self.assertEqual(len(llm.calls), 2)

    def test_arm_without_the_memory_tool_cannot_call_it(self):
        tr, llm = self.run_loop([res([("tool", "memory_recall", {"query": "q", "ll_keywords": [NEED]})]),
                                 res([("tool", "answer", {"text": "x"})])], mem=False)
        self.assertNotIn("memory_recall", llm.calls[0]["tools"])
        self.assertEqual(self.hars.recall_calls, 0)
        self.assertEqual(self.hars.correct_calls, [])
        c = tr["calls"][0]
        self.assertEqual((c["kind"], c["is_error"]), ("unknown", True))
        self.assertIn("unknown tool", c["result"])

    def test_memory_call_shape_is_the_c8_project_shape_and_the_result_is_returned_as_is(self):
        tr, _ = self.run_loop([
            res([("tool", "memory_recall", {"query": f"value of {NEED}", "ll_keywords": [NEED, "tests/test_a.py"], "hl_keywords": ["test values"]})]),
            res([("tool", "answer", {"text": "107"})])])
        self.assertEqual(len(self.hars.correct_calls), 1)
        call = self.hars.correct_calls[0]
        self.assertEqual((call["project"], call["sid"], call["max_chars"]), (self.pid, None, 6000))   # project, NO session_id
        self.assertEqual((call["ll_keywords"], call["hl_keywords"]), ([NEED, "tests/test_a.py"], ["test values"]))
        self.assertTrue(RL.FakeHars.recall_correct)                                   # same entry point the C.8 runner uses
        mc = tr["calls"][0]
        self.assertEqual(mc["kind"], "memory")
        self.assertIn("Document Chunks", mc["result"])                                # HARS context as-is
        self.assertEqual(mc["chars"], mc["meta"]["returned_chars"])
        self.assertTrue(mc["anchor_in_result"])
        self.assertTrue(mc["meta"]["graph_branch"])
        q = {"qid": "q1", "question": "Q", "gold_anchor": NEED, "answerable": True}
        row = RA.build_row("r", "sess0", "sum+mem", q, tr, summary_text="S", flags={}, wall_s=0.1, args_meta={"memory_max_chars": 6000})
        self.assertEqual(row["memory_calls_n"], 1)
        self.assertEqual(row["memory_calls"][0]["ll_keywords"], [NEED, "tests/test_a.py"])
        self.assertEqual(row["memory_calls"][0]["returned_chars"], mc["chars"])
        self.assertTrue(row["anchor_in_memory_results"])
        self.assertFalse(row["anchor_in_workspace_results"])
        self.assertEqual(row["workspace_calls"], 0)

    def test_workspace_and_memory_anchor_hits_are_recorded_separately(self):
        tr, _ = self.run_loop([res([("tool", "grep", {"pattern": "token"})]), res([("tool", "answer", {"text": "x"})])])
        q = {"qid": "q1", "question": "Q", "gold_anchor": NEED, "answerable": True}
        row = RA.build_row("r", "s", "sum", q, tr, summary_text="S", flags={}, wall_s=0, args_meta={})
        self.assertTrue(row["anchor_in_workspace_results"])
        self.assertFalse(row["anchor_in_memory_results"])
        self.assertTrue(row["anchor_in_any_result"])

    def test_plain_text_reply_is_the_answer_and_parallel_calls_all_get_results(self):
        tr, _ = self.run_loop([res([("text", "I do not know.")])])
        self.assertEqual((tr["final"], tr["ended_by"]), ("I do not know.", "text"))
        tr, _ = self.run_loop([res([("tool", "ls", {}), ("tool", "grep", {"pattern": "Widget"})]), res([("tool", "answer", {"text": "ok"})])])
        results = tr["messages"][2]["content"]
        self.assertEqual(len([b for b in results if b["type"] == "tool_result"]), 2)

    def test_memory_error_is_an_error_result_not_a_crash(self):
        self.hars.fail_recall = True
        tr, _ = self.run_loop([res([("tool", "memory_recall", {"query": "q"})]), res([("tool", "answer", {"text": "x"})])])
        c = tr["calls"][0]
        self.assertTrue(c["is_error"])
        self.assertIn("memory_recall error", c["result"])


# --------------------------------------------------------------------------- sources, byte identity, end to end
class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        D = self.d.name
        self.sdir = os.path.join(D, "ls-data")
        self.sessions = make_sessions(self.sdir)
        self.old = os.path.join(D, "old")
        make_old_run(self.old, self.sessions)
        self.wsroot = os.path.join(D, "ws")
        for sid in self.sessions:
            make_ws(os.path.join(self.wsroot, sid, "repo"))
        self.pmap = os.path.join(D, "pmap.json")
        json.dump({sid: f"nc-e1-{sid}" for sid in self.sessions}, open(self.pmap, "w"))

    def tearDown(self):
        self.d.cleanup()

    def test_summary_and_note_are_reused_byte_for_byte(self):
        src = RA.load_sources(self.old, self.sessions, ["sum", "sum+mem", "note+mem"])
        old_rows = RA.read_jsonl(os.path.join(self.old, "records.jsonl"))
        for sid in self.sessions:
            b = next(r for r in old_rows if r["sid"] == sid and r["arm"] == "baseline")["memory_text"]
            n = next(r for r in old_rows if r["sid"] == sid and r["arm"] == "nc-note-only")["memory_text"]
            self.assertEqual(src["summary"][sid], b)
            self.assertEqual(src["note"][sid], n)
            self.assertEqual(RA.sha256_text(src["summary"][sid]), RA.sha256_text(b))
            for arm in ("sum", "sum+mem"):
                self.assertIn(b, RA.system_for(arm, src["summary"][sid]))         # exact substring, no strip / re-wrap
            self.assertIn(n, RA.system_for("note+mem", src["note"][sid]))
        self.assertEqual(len(src["questions"][next(iter(self.sessions))]), 5)       # answerable AND unanswerable
        # disagreeing rows are an error, a missing note for note+mem too
        rows = old_rows + [dict(old_rows[0], memory_text="CHANGED", rid="x")]
        bad = os.path.join(self.d.name, "bad")
        os.makedirs(bad)
        open(os.path.join(bad, "records.jsonl"), "w").write("\n".join(json.dumps(r) for r in rows))
        with self.assertRaises(RA.SourceError):
            RA.load_sources(bad, self.sessions, ["sum"])

    def test_memory_relevant_flag_uses_the_one_matcher(self):
        src = RA.load_sources(self.old, self.sessions, ["sum"])
        s0 = self.sessions["chain_s0_c0"]
        dropped = RA.dropped_text(s0)
        q = next(x for x in s0["questions"] if x.get("gold_anchor") == "NEEDLE_s3_0")
        f = RA.subset_flags(q, src["summary"]["chain_s0_c0"], dropped)
        self.assertEqual((f["anchor_in_dropped"], f["anchor_in_summary"], f["memory_relevant"]), (True, True, False))   # the summary has it
        q1 = next(x for x in s0["questions"] if x.get("gold_anchor") == "NEEDLE_s1_1")
        self.assertTrue(RA.subset_flags(q1, src["summary"]["chain_s0_c0"], dropped)["memory_relevant"])
        unans = next(x for x in s0["questions"] if not x["answerable"])
        self.assertFalse(RA.subset_flags(unans, "", dropped)["memory_relevant"])

    def run_main(self, *extra, out=None):
        out = out or os.path.join(self.d.name, "run")
        argv = ["--sessions", self.sdir, "--reuse-run", self.old, "--workspaces", self.wsroot, "--projects-map", self.pmap, "--out", out,
                "--mock-llm", "--fake-hars", "--profile", "anthropic", "--concurrency", "2", *extra]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = RA.main(argv)
        return rc, out, buf.getvalue()

    def test_full_run_artifacts_resume_and_judge_export(self):
        rc, out, log = self.run_main("--arms", "sum,sum+mem")
        self.assertEqual(rc, 0, log)
        rows = RA.read_jsonl(os.path.join(out, "records.jsonl"))
        self.assertEqual(len(rows), 2 * 5 * 2)                                       # 2 sessions x 5 questions x 2 arms
        self.assertEqual({r["arm"] for r in rows}, {"sum", "sum+mem"})
        sm = [r for r in rows if r["arm"] == "sum+mem"]
        self.assertTrue(all(r["memory_calls_n"] == 1 for r in sm))
        self.assertTrue(all(r["memory_calls_n"] == 0 for r in rows if r["arm"] == "sum"))
        self.assertTrue(all(r["summary_sha256"] == RA.sha256_text(r["memory_text"]) for r in rows))
        for fn in ("samples.jsonl", "traces.jsonl", "summary.json", "run.log", "MANIFEST.json", "config.json"):
            self.assertTrue(os.path.isfile(os.path.join(out, fn)), fn)
        self.assertTrue(os.path.isfile(os.path.join(out, "scripts", "run_agentic.py")))
        self.assertTrue(os.path.isfile(os.path.join(out, "inputs", "inputs.json")))
        tr = RA.read_jsonl(os.path.join(out, "traces.jsonl"))
        self.assertEqual(len(tr), len(rows))
        self.assertTrue(all("messages" in t and t["system"] for t in tr))
        # the transcript is nowhere in the traces' system prompts beyond the summary; tools of the sum arm lack memory_recall
        self.assertTrue(all("memory_recall" not in t["tools"] for t in tr if t["arm"] == "sum"))
        # resume: nothing re-run
        rc2, _, log2 = self.run_main("--arms", "sum,sum+mem", out=out)
        self.assertEqual(rc2, 0, log2)
        self.assertEqual(len(RA.read_jsonl(os.path.join(out, "records.jsonl"))), len(rows))
        # judge export joins the evidence and builds the visible context from the summary + tool results
        self.assertEqual(RA.export_judge(out, self.sdir, os.path.join(out, "judge-input.jsonl")), 0)
        ji = RA.read_jsonl(os.path.join(out, "judge-input.jsonl"))
        self.assertEqual(len(ji), len(rows))
        r0 = ji[0]
        self.assertTrue(r0["span_text"] and r0["visible_context"].startswith("SUMMARY<"))
        self.assertNotIn("memory_text", r0)
        self.assertIn("agent_model", r0)

    def test_note_arm_is_off_by_default_and_enabled_by_arms(self):
        rc, out, log = self.run_main()
        self.assertEqual({r["arm"] for r in RA.read_jsonl(os.path.join(out, "records.jsonl"))}, {"sum", "sum+mem"})
        rc, out2, log = self.run_main("--arms", "note+mem", "--limit-questions", "2", out=os.path.join(self.d.name, "run2"))
        self.assertEqual(rc, 0, log)
        rows = RA.read_jsonl(os.path.join(out2, "records.jsonl"))
        self.assertEqual({r["arm"] for r in rows}, {"note+mem"})
        self.assertTrue(all(r["memory_text"].startswith("NOTE<") for r in rows))

    def test_leaky_workspace_stops_the_run(self):
        sid = next(iter(self.sessions))
        open(os.path.join(self.wsroot, sid, "repo", "oops.txt"), "w").write(L.serialize(self.sessions[sid]["messages"][:4]))
        rc, out, log = self.run_main("--arms", "sum")
        self.assertNotEqual(rc, 0)
        self.assertIn("LEAK", log)
        rows = RA.read_jsonl(os.path.join(out, "records.jsonl"))
        self.assertTrue(all(r["sid"] != sid for r in rows))

    def test_dry_run_estimate_is_deterministic_and_states_its_model(self):
        argv = ["--sessions", self.sdir, "--reuse-run", self.old, "--dry-run", "--json", "--profile", "anthropic", "--arms", "sum,sum+mem"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(RA.main(argv), 0)
        est = json.loads(buf.getvalue())
        self.assertEqual((est["questions"], est["judge_items"], est["sessions"]), (10, 20, 2))
        self.assertEqual(est["assumptions"]["steps_mean"], 8)
        self.assertGreater(est["run_usd_expected_with_cache"], 0)
        self.assertGreater(est["run_usd_expected_without_cache"], est["run_usd_expected_with_cache"])
        self.assertGreaterEqual(est["run_usd_upper_bound"], est["run_usd_expected_without_cache"])
        # hand check of the model for one (session, arm): step k input = P0 + (k-1)(o + r)
        prof = P.load_profile("anthropic")
        sm = L.count_tokens(self.sessions_summary("chain_s0_c0"))
        E = RA.EST
        p0 = E["system_tokens"] + E["tool_schema_tokens"] + sm + E["question_tokens"]
        inputs = [p0 + (k - 1) * (E["out_tokens_per_step"] + E["ws_result_tokens"]) for k in range(1, 9)]
        fresh = inputs[0] + sum(inputs[k] - inputs[k - 1] for k in range(1, 8))
        want = prof.cost_usd("claude-haiku-5-5", int(fresh), 8 * 150, cache_read=int(sum(inputs[:-1])))
        e1 = RA.estimate(["sum"], {("chain_s0_c0", "sum"): sm}, {"chain_s0_c0": 1}, prof, "claude-haiku-5-5", 12, 6000, 0.003)
        self.assertAlmostEqual(e1["per_arm"]["sum"]["usd_with_cache"], round(want, 4), places=4)
        self.assertEqual(e1["mde_prior"]["mde_relative_to_mean_tokens"], round(2.8 * 0.5 / 1.0, 3))
        # nothing was written by a dry run
        self.assertFalse(os.path.exists(os.path.join(self.d.name, "run")))

    def sessions_summary(self, sid):
        return next(r for r in RA.read_jsonl(os.path.join(self.old, "records.jsonl")) if r["sid"] == sid and r["arm"] == "baseline")["memory_text"]


# --------------------------------------------------------------------------- judge outcome schema
class JudgeOutcome(unittest.TestCase):
    REC = {"question": "What?", "answer": "I think it was 107.", "canonical_short_answer": "107", "span_text": "x"}

    def test_default_prompt_is_unchanged_and_outcome_mode_adds_the_precedence_and_the_field(self):
        J.OUTCOME_MODE = False
        base = J.build_prompt(self.REC, "ex")
        self.assertNotIn("outcome", base)
        try:
            J.OUTCOME_MODE = True
            p = J.build_prompt(self.REC, "ex")
        finally:
            J.OUTCOME_MODE = False
        self.assertIn("precedence", p)
        self.assertIn("hedged answer that states the recorded fact is correct", p)
        self.assertIn('"outcome":"correct|partial|wrong|abstained"', p)
        self.assertTrue(p.startswith(base.split("Reply ONLY JSON")[0].split("reason: <= 25 words.")[0]))

    def test_the_memory_tool_name_in_an_answer_is_masked_for_the_blind_judge(self):
        masked, hits = J.mask_arm_leaks("I used memory_recall and found it in tests/a.py")
        self.assertTrue(hits)
        self.assertNotIn("memory_recall", masked)

    def test_parse_verdict_reads_a_valid_outcome_only(self):
        v, ok = J.parse_verdict('{"score":2,"unsupported_claim":false,"abstained":true,"outcome":"Correct","reason":"r"}')
        self.assertTrue(ok)
        self.assertEqual(v["outcome"], "correct")
        v, _ = J.parse_verdict('{"score":2,"unsupported_claim":false,"abstained":true,"outcome":"maybe","reason":"r"}')
        self.assertNotIn("outcome", v)
        v, _ = J.parse_verdict('{"score":1,"unsupported_claim":false,"abstained":false,"reason":"r"}')
        self.assertNotIn("outcome", v)


# --------------------------------------------------------------------------- metrics
def toy_rows():
    """2 arms x 4 sessions x 3 questions; sum+mem is cheaper on memory-relevant questions and (slightly) more accurate."""
    recs, judged = [], []
    for si in range(4):
        sid = f"s{si}"
        for qi in range(3):
            qid = f"{sid}_q{qi}"
            relevant = qi < 2
            unans = qi == 2
            for arm, tokens, outcome in (("sum", 40000 + 1000 * si, "wrong" if (qi == 0 and si == 0) else ("correct" if qi == 1 else "abstained")),
                                         ("sum+mem", 25000 + 500 * si, "correct" if not unans else "abstained")):
                rid = f"{qid}::{arm}"
                recs.append({"rid": rid, "sid": sid, "cluster_session": sid, "qid": qid, "arm": arm, "valid": True,
                             "answerable": not unans, "gold_anchor": "" if unans else "A", "memory_relevant": relevant and not unans,
                             "tokens_total": tokens, "tokens_fresh": tokens // 4, "cost_usd": tokens * 1e-7, "steps": 6 if arm == "sum" else 4,
                             "tool_calls": 5 if arm == "sum" else 3, "memory_calls_n": 0 if arm == "sum" else 1, "workspace_calls": 4,
                             "anchor_in_workspace_results": False, "anchor_in_memory_results": arm == "sum+mem", "latency_s": 3.0})
                judged.append({"rid": rid, "score": {"correct": 2, "wrong": 0, "abstained": 2 if unans else 0, "partial": 1}[outcome],
                               "outcome": outcome, "hallucination": outcome == "wrong", "abstained": outcome == "abstained",
                               "hallucination_rule": "unsupported_no_anchor" if outcome == "wrong" else "judge_clean"})
    return recs, judged


class Metrics(unittest.TestCase):
    def test_toy_numbers_subsets_taxonomy_and_contrasts(self):
        recs, judged = toy_rows()
        res_ = MA.analyse(recs, judged, n_boot=500, seed=7)
        self.assertEqual((res_["n_questions_complete"], res_["n_sessions"], res_["outcome_source"]["judge"]), (12, 4, 24))
        a = res_["arms_summary"]["all"]["sum"]
        # taxonomy: mutually exclusive, sums to n
        self.assertEqual(sum(a["outcome_counts"].values()), 12)
        self.assertEqual(a["outcome_counts"], {"correct": 4, "partial": 0, "wrong": 1, "abstained": 7, "no_answer": 0})
        self.assertAlmostEqual(a["hallucination_rate_all"], 1 / 12)
        self.assertAlmostEqual(a["hallucination_given_answered"], 1 / 5)             # wrong / (correct + partial + wrong)
        self.assertEqual(a["mechanical_unsupported_no_anchor"], 1)                    # reported separately
        self.assertAlmostEqual(a["unanswerable_abstain_rate"], 1.0)
        # accuracy over answerables: 4 correct of 8
        self.assertAlmostEqual(a["accuracy_correct"], 4 / 8)
        mr = res_["arms_summary"]["memory_relevant"]
        self.assertEqual(mr["sum"]["n"], 8)
        self.assertAlmostEqual(mr["sum"]["tokens_total_mean"], sum(40000 + 1000 * s for s in range(4)) / 4)
        # tokens per correct = sum of tokens / number of correct answers
        want_tpc = sum(r["tokens_total"] for r in recs if r["arm"] == "sum+mem")
        self.assertAlmostEqual(res_["arms_summary"]["all"]["sum+mem"]["tokens_per_correct"]["point"],
                               want_tpc / res_["arms_summary"]["all"]["sum+mem"]["outcome_counts"]["correct"])
        # PRIMARY: sum+mem - sum on the memory-relevant subset, tokens
        c = res_["primary"]["tokens_total"]
        self.assertEqual((c["n_questions"], c["n_sessions"]), (8, 4))
        want = sum((25000 + 500 * s) - (40000 + 1000 * s) for s in range(4)) / 4
        self.assertAlmostEqual(c["point"], want)
        self.assertLess(c["ci_hi"], 0)
        self.assertAlmostEqual(c["relative_to_b"], want / (sum(40000 + 1000 * s for s in range(4)) / 4))
        # observed-SD power statement
        sds = [(25000 + 500 * s) - (40000 + 1000 * s) for s in range(4)]
        import statistics
        self.assertAlmostEqual(c["mde_80pct_session"], MA.Z_MDE * statistics.stdev(sds) / 2)
        # accuracy contrast: sum+mem correct on all 8 relevant, sum on 4 of 8
        acc = res_["primary"]["correct"]
        self.assertAlmostEqual(acc["point"], 0.5)
        # both-correct subset: only the qi == 1 questions are correct in both arms
        bc = res_["contrasts"]["sum+mem - sum | memory_relevant | tokens_total | both correct"]
        self.assertEqual(bc["n_questions"], 4)
        # no verdict anywhere
        md = MA.render(res_, {})
        self.assertNotIn("WIN", md)
        self.assertNotIn("FAIL", md)
        self.assertIn("MDE", md)

    def test_outcome_falls_back_to_a_documented_derivation_and_precedence(self):
        d = MA.derive_outcome
        base = {"answerable": True}
        self.assertEqual(d(base, {"score": 2, "hallucination": False, "abstained": True}), "correct")       # hedged but states the fact
        self.assertEqual(d(base, {"score": 2, "hallucination": True, "abstained": False}), "wrong")
        self.assertEqual(d(base, {"score": 1, "hallucination": False, "abstained": True}), "partial")
        self.assertEqual(d(base, {"score": 0, "hallucination": True, "abstained": True}), "wrong")           # wrong beats abstained
        self.assertEqual(d(base, {"score": 0, "hallucination": False, "abstained": True}), "abstained")
        self.assertEqual(d({"answerable": False}, {"score": 2, "hallucination": False, "abstained": True}), "abstained")
        self.assertEqual(d({"answerable": False}, {"score": 0, "hallucination": True, "abstained": False}), "wrong")
        self.assertEqual(d({"no_answer": True, "answerable": True}, {}), "no_answer")
        recs, judged = toy_rows()
        for j in judged:
            j.pop("outcome")
        res_ = MA.analyse(recs, judged, n_boot=50)
        self.assertEqual(res_["outcome_source"]["derived"], 24)

    def test_invalid_and_unpaired_rows_are_dropped(self):
        recs, judged = toy_rows()
        recs[0]["valid"] = False
        res_ = MA.analyse(recs, judged, n_boot=50)
        self.assertEqual(res_["n_questions_complete"], 11)
        self.assertIn(recs[0]["qid"], res_["dropped_incomplete_qids"])


# --------------------------------------------------------------------------- workspaces
def git(*a, cwd=None):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


class Workspaces(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        D = self.d.name
        self.remote = os.path.join(D, "remote", "acme", "widget")
        os.makedirs(self.remote)
        git("init", "-q", cwd=self.remote)
        open(os.path.join(self.remote, "a.py"), "w").write("x = 1\n")
        git("add", ".", cwd=self.remote)
        git("commit", "-qm", "base", cwd=self.remote)
        self.base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.remote, capture_output=True, text=True).stdout.strip()
        open(os.path.join(self.remote, "a.py"), "w").write("x = 2  # UPSTREAM_FIX_MARKER\n")        # a LATER commit: must not be visible
        git("commit", "-qam", "fix", cwd=self.remote)
        self.patch = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 99  # SESSION_PATCH\n"
        self.oh = os.path.join(D, "oh")
        os.makedirs(os.path.join(self.oh, "sessions"))
        json.dump({"instance_id": "acme__widget-1", "repo": "acme/widget"}, open(os.path.join(self.oh, "sessions", "sess0.json"), "w"))
        self.chains = [{"chain_id": "chain_s0_c0", "session_id": "sess0"}]
        self.tmpl = os.path.join(D, "remote", "{repo}")

    def tearDown(self):
        self.d.cleanup()

    def build(self, inst):
        return OW.provision(self.chains, self.oh, os.path.join(self.d.name, "cache"), os.path.join(self.d.name, "out"), inst, self.tmpl)

    def test_base_commit_only_without_a_patch_and_no_history_or_upstream_fix(self):
        s = self.build({"acme__widget-1": {"instance_id": "acme__widget-1", "base_commit": self.base}})
        self.assertEqual(s["errors"], {})
        m = s["workspaces"]["chain_s0_c0"]
        self.assertEqual((m["workspace_kind"], m["patch_applied"]), ("base_commit", False))
        repo = m["repo_dir"]
        self.assertEqual(open(os.path.join(repo, "a.py")).read(), "x = 1\n")
        self.assertFalse(os.path.exists(os.path.join(repo, ".git")))
        self.assertEqual(RA.Workspace(repo).grep("UPSTREAM_FIX_MARKER", "."), "(no matches)")
        self.assertTrue(os.path.isfile(os.path.join(self.d.name, "out", "chain_s0_c0.json")))     # meta lives OUTSIDE the workspace
        self.assertFalse(os.path.exists(os.path.join(repo, "chain_s0_c0.json")))

    def test_final_patch_applied_only_when_supplied(self):
        s = self.build({"acme__widget-1": {"instance_id": "acme__widget-1", "base_commit": self.base, "model_patch": self.patch}})
        m = s["workspaces"]["chain_s0_c0"]
        self.assertEqual((m["workspace_kind"], m["patch_applied"]), ("base_commit+final_patch", True))
        self.assertIn("SESSION_PATCH", open(os.path.join(m["repo_dir"], "a.py")).read())
        # idempotent
        s2 = self.build({"acme__widget-1": {"instance_id": "acme__widget-1", "base_commit": self.base, "model_patch": self.patch}})
        self.assertTrue(s2["workspaces"]["chain_s0_c0"].get("reused"))

    def test_a_patch_that_does_not_apply_falls_back_loudly(self):
        s = self.build({"acme__widget-1": {"instance_id": "acme__widget-1", "base_commit": self.base, "model_patch": "garbage"}})
        m = s["workspaces"]["chain_s0_c0"]
        self.assertEqual((m["workspace_kind"], m["patch_applied"]), ("base_commit", False))
        self.assertTrue(m["patch_error"])

    def test_missing_instance_or_commit_is_an_error_not_a_guess(self):
        s = self.build({})
        self.assertIn("chain_s0_c0", s["errors"])                                   # no base_commit and no network (fetch fails)
        os.remove(os.path.join(self.oh, "sessions", "sess0.json"))
        s = OW.provision(self.chains, self.oh, os.path.join(self.d.name, "c2"), os.path.join(self.d.name, "o2"), {}, self.tmpl)
        self.assertIn("instance_id", s["errors"]["chain_s0_c0"])

    def test_dataset_lookup_is_injectable_and_reads_only_base_commit(self):
        seen = []

        def fetch(url):
            seen.append(url)
            return {"rows": [{"row": {"base_commit": self.base, "patch": "THE GOLD PATCH MUST NEVER BE USED"}}]}
        s = OW.provision(self.chains, self.oh, os.path.join(self.d.name, "c3"), os.path.join(self.d.name, "o3"), {}, self.tmpl, fetch=fetch)
        self.assertEqual(s["errors"], {})
        self.assertEqual(s["workspaces"]["chain_s0_c0"]["workspace_kind"], "base_commit")
        self.assertIn("SWE-bench_Verified", seen[0])


# --------------------------------------------------------------------------- home_run
class AgenticRunner(T7.EnrichRunner):
    """EnrichRunner + stand-ins for run_agentic.py / metrics_agentic.py writing the real file shapes."""

    def __call__(self, argv, env, cwd, log_path, echo=True):
        script = next((os.path.basename(a) for a in argv if a.endswith(".py")), "")
        if script in ("run_agentic.py", "metrics_agentic.py"):
            self.calls.append((script, argv))
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            rc, text = (self.agentic if script == "run_agentic.py" else self.agmetrics)(argv)
            open(log_path, "a").write(text)
            return rc, text
        return super().__call__(argv, env, cwd, log_path, echo)

    def agentic(self, argv):
        if "--export-judge" in argv:
            out = self.arg(argv, "--out")
            open(os.path.join(out, "judge-input.jsonl"), "w").write(json.dumps({"rid": "q::sum"}) + "\n")
            return 0, "exported\n"
        out = self.arg(argv, "--resume") if "--resume" in argv else self.arg(argv, "--out")
        os.makedirs(out, exist_ok=True)
        json.dump({"run_id": "ag"}, open(os.path.join(out, "config.json"), "w"))
        json.dump({"rows": 4, "expected": 4, "n_missing": 0}, open(os.path.join(out, "summary.json"), "w"))
        json.dump({"budget_spent_all_invocations": 0.33, "failed_sessions": []}, open(os.path.join(out, "MANIFEST.json"), "w"))
        open(os.path.join(out, "records.jsonl"), "w").write(json.dumps({"rid": "q::sum", "valid": True}) + "\n")
        return 0, "done\n"

    def agmetrics(self, argv):
        open(self.arg(argv, "--out"), "w").write("# agentic\n")
        json.dump({}, open(self.arg(argv, "--json"), "w"))
        return 0, "ok\n"


class Home(T7.Base):
    def setUp(self):
        super().setUp()
        D = self.d.name
        self.stamp("2026-10-28.c8")
        old = os.path.join(self.camp, "oh", "run")
        rows = [json.loads(l) for l in open(os.path.join(old, "records.jsonl")) if l.strip()]
        with open(os.path.join(old, "records.jsonl"), "w") as f:
            for r in rows:
                r["memory_text"] = f"SUMMARY<{r['sid']}> text" if r["arm"] == "baseline" else f"NOTE<{r['sid']}>"
                f.write(json.dumps(r) + "\n")
        open(os.path.join(old, "judged.jsonl"), "w").write("{}\n")
        json.dump({"judge_model": H.P.resolve_models(H.P.load_profile("anthropic"), {})["judge"], "map": {}},
                  open(os.path.join(old, "judge-mapping.json"), "w"))
        # sessions/<session_id>.json in the OH artifact dir + a local git remote per repo
        self.remote = os.path.join(D, "remote")
        self.inst = {}
        os.makedirs(os.path.join(self.art, "sessions"), exist_ok=True)
        for i, c in enumerate(self.chains):
            repo = f"acme/w{i}"
            rd = os.path.join(self.remote, repo)
            os.makedirs(rd)
            git("init", "-q", cwd=rd)
            open(os.path.join(rd, "m.py"), "w").write(f"v = {i}\n")
            git("add", ".", cwd=rd)
            git("commit", "-qm", "b", cwd=rd)
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=rd, capture_output=True, text=True).stdout.strip()
            iid = f"acme__w{i}-{i + 1}"
            json.dump({"instance_id": iid, "repo": repo}, open(os.path.join(self.art, "sessions", f"{c['session_id']}.json"), "w"))
            self.inst[iid] = {"instance_id": iid, "base_commit": head}
        self.inst_file = os.path.join(D, "instances.json")
        json.dump(list(self.inst.values()), open(self.inst_file, "w"))
        self.cache = os.path.join(D, "cache")
        self.clone = os.path.join(self.remote, "{repo}")

    def mark(self, *steps):
        for s in steps:
            H.write_json(os.path.join(self.camp, "steps", f"{s}.json"), {"step": s})

    def provision_ws(self):
        rc, so, se = self.go(AgenticRunner(), "oh-workspaces", "--cache-dir", self.cache, "--instances", self.inst_file, "--clone-url", self.clone)
        self.assertEqual(rc, 0, so + se)
        return so

    def test_workspaces_build_from_base_commit_and_say_there_is_no_patch(self):
        rc, so, se = self.go(AgenticRunner(), "oh-workspaces", "--dry-run", "--cache-dir", self.cache, "--instances", self.inst_file)
        self.assertEqual(rc, 0, so + se)
        self.assertIn("final patch NOT available", so)
        self.assertNotIn("oh.workspaces", self.steps())
        so = self.provision_ws()
        self.assertIn("WITHOUT the session's final patch", so)
        cj = H.read_json(os.path.join(self.camp, "campaign.json"))
        self.assertEqual(cj["c9_workspaces"]["kinds"], {"base_commit": len(self.chains)})
        dest = cj["c9_workspaces"]["dest"]
        self.assertTrue(os.path.isfile(os.path.join(dest, self.chains[0]["chain_id"], "repo", "m.py")))
        self.assertIn("oh.workspaces", self.steps())

    def test_guard_accepts_c6_c7_c8_and_refuses_older(self):
        for v in ("2026-10-27.c6", "2026-10-28.c7", "2026-10-28.c8"):
            self.stamp(v)
            rc, so, se = self.go(AgenticRunner(), "oh-agentic", "--dry-run")
            self.assertEqual(rc, 0, (v, so, se))
        self.stamp("2026-10-26.c5")
        rc, so, se = self.go(AgenticRunner(), "oh-agentic", "--dry-run")
        self.assertEqual(rc, 2)
        self.assertIn("EVAL-VERSION", se)
        rc, so, se = self.go(AgenticRunner(), "oh-workspaces", "--dry-run", "--cache-dir", self.cache)
        self.assertEqual(rc, 0, so + se)                                            # no spend: no version guard needed
        self.assertEqual(H.REUSE_COMPATIBLE_FROM, ("2026-10-27.c6", "2026-10-28.c7", "2026-10-28.c8"))
        self.assertEqual(H.HOME_RUN_VERSION, "2026-10-28.c9")
        self.assertEqual(open(os.path.join(os.path.dirname(H.BASE), "EVAL-VERSION")).read().strip(), "2026-10-28.c9")

    def test_needs_an_explicit_campaign_ingest_and_workspaces(self):
        rc, so, se = self.go(AgenticRunner(), "oh-agentic", resume=False)
        self.assertEqual(rc, 2)
        self.assertIn("EXISTING campaign", se)
        rc, so, se = self.go(AgenticRunner(), "oh-agentic")
        self.assertEqual(rc, 2)
        self.assertIn("oh-ingest", se)
        self.mark("oh.ingest")
        H.write_json(os.path.join(self.camp, "campaign.json"), {**H.read_json(os.path.join(self.camp, "campaign.json")),
                     "c8_projects": {"hars_run_id": "e1", "pids": {c["chain_id"]: f"nc-e1-{c['chain_id']}" for c in self.chains}}})
        rc, so, se = self.go(AgenticRunner(), "oh-agentic")
        self.assertEqual(rc, 2)
        self.assertIn("oh-workspaces", se)

    def test_dry_run_prints_the_stated_estimate_and_power_guess_and_spends_nothing(self):
        r = AgenticRunner()
        rc, so, se = self.go(r, "oh-agentic", "--dry-run")
        self.assertEqual(rc, 0, so + se)
        self.assertIn("steps x tokens", so)
        self.assertIn("MDE", so)
        self.assertIn("upper bound", so)
        est = H.read_json(os.path.join(self.camp, "oh-agentic-estimate.json"))
        self.assertEqual(est["arms"], ["sum", "sum+mem"])
        self.assertEqual(est["questions"], sum(len(c["qa"]) for c in self.chains))
        self.assertFalse([1 for s, a in r.calls if s in ("run_agentic.py", "judge_v2.py")])
        self.assertFalse([e for e in H.read_json(os.path.join(self.camp, "ledger.json"), {"entries": []})["entries"] if "agentic" in e["step"]])

    def test_full_run_wires_run_export_judge_outcome_metrics_and_ledger(self):
        self.provision_ws()
        self.mark("oh.ingest")
        H.write_json(os.path.join(self.camp, "campaign.json"), {**H.read_json(os.path.join(self.camp, "campaign.json")),
                     "c8_projects": {"hars_run_id": "e1", "pids": {c["chain_id"]: f"nc-e1-{c['chain_id']}" for c in self.chains}}})
        r = AgenticRunner()
        rc, so, se = self.go(r, "oh-agentic", "--arms", "sum,sum+mem", "--max-steps", "7")
        self.assertEqual(rc, 0, so + se)
        run = [a for s, a in r.calls if s == "run_agentic.py" and "--export-judge" not in a][0]
        self.assertEqual(r.arg(run, "--arms"), "sum,sum+mem")
        self.assertEqual(r.arg(run, "--max-steps"), "7")
        self.assertEqual(r.arg(run, "--reuse-run"), os.path.join(self.camp, "oh", "run"))
        self.assertTrue(r.arg(run, "--workspaces").startswith(self.cache))
        self.assertEqual(json.load(open(r.arg(run, "--projects-map")))[self.chains[0]["chain_id"]], f"nc-e1-{self.chains[0]['chain_id']}")
        judge = [a for s, a in r.calls if s == "judge_v2.py"][0]
        self.assertIn("--outcome", judge)
        self.assertIn("--allow-same-family-judge", judge)
        self.assertTrue([a for s, a in r.calls if s == "metrics_agentic.py"])
        steps = self.steps()
        for s in ("oh.agentic.run", "oh.agentic.export", "oh.agentic.judge", "oh.agentic.metrics"):
            self.assertIn(s, steps)
        led = [e for e in H.read_json(os.path.join(self.camp, "ledger.json"))["entries"] if e["step"] == "oh.agentic.run"]
        self.assertEqual(len(led), 1)
        self.assertAlmostEqual(led[0]["usd"], 0.33)
        cj = H.read_json(os.path.join(self.camp, "campaign.json"))
        self.assertEqual(cj["eval_version"], "2026-10-28.c8")                       # the campaign's stamp is not rewritten
        self.assertIn("oh-agentic", cj["c7_reuse"])
        # re-run: finished steps are skipped
        r2 = AgenticRunner()
        rc, so, se = self.go(r2, "oh-agentic", "--arms", "sum,sum+mem")
        self.assertEqual(rc, 0, so + se)
        self.assertFalse([1 for s, a in r2.calls if s in ("run_agentic.py", "judge_v2.py")])

    def test_a_different_judge_model_than_the_c6_campaign_is_refused(self):
        self.provision_ws()
        self.mark("oh.ingest")
        H.write_json(os.path.join(self.camp, "campaign.json"), {**H.read_json(os.path.join(self.camp, "campaign.json")),
                     "c8_projects": {"hars_run_id": "e1", "pids": {c["chain_id"]: "p" for c in self.chains}}})
        mp = os.path.join(self.camp, "oh", "run", "judge-mapping.json")
        json.dump({"judge_model": "some-other-judge", "map": {}}, open(mp, "w"))
        r = AgenticRunner()
        rc, so, se = self.go(r, "oh-agentic")
        self.assertEqual(rc, 2)
        self.assertIn("same judge", se)
        self.assertFalse([1 for s, a in r.calls if s == "run_agentic.py"])


if __name__ == "__main__":
    unittest.main()
