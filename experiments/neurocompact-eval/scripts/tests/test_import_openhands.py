"""Synthetic tests for import_openhands.py / select_openhands.py / oh_pipeline.py guard.

No real trajectories: a generated OpenHands-shaped message list (no network, $0).
Run: python3 -m pytest -q experiments/neurocompact-eval/scripts/tests/test_import_openhands.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import import_openhands as io_oh  # noqa: E402
import make_chains as mc  # noqa: E402
import oh_pipeline as ohp  # noqa: E402
import select_openhands as sel  # noqa: E402

ISSUE = "Widget.frobnicate() crashes when the frob count is zero and the mode is 'strict'."
TEMPLATE_TAIL = "Follow these phases to resolve the issue:\nPhase 1. READING ... Phase 8. FINAL REVIEW"


def T(text):
    return [{"type": "text", "text": text}]


def bash_out(body, code=0, cwd="/workspace/acme__widget__1.0"):
    return (f"{body}\n[The command completed with exit code {code}.]\n[Current working directory: {cwd}]\n"
            f"[Python interpreter: /opt/miniconda3/envs/testbed/bin/python]\n"
            f"[Command finished with exit code {code}]")


def make_traj(n_steps=120, big=900, issue=ISSUE):
    msgs = [
        {"role": "system", "content": T("You are OpenHands agent, SYSTEM-BOILERPLATE-XYZ")},
        {"role": "user", "content": T(
            "<uploaded_files>\n/workspace/acme__widget__1.0\n</uploaded_files>\n\nI've uploaded a repo. "
            f"<issue_description>\n{issue}\n</issue_description>\n\nCan you help...\n{TEMPLATE_TAIL}")},
    ]
    for i in range(n_steps):
        cid = f"toolu_{i:04d}"
        kind = i % 4
        if kind == 0:
            msgs.append({"role": "assistant", "content": T(f"Step {i}: looking around."), "tool_calls": [
                {"id": cid, "type": "function", "function": {
                    "name": "execute_bash", "arguments": json.dumps({"command": f"grep -rn needle{i} src/"})}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "name": "execute_bash",
                         "content": T(bash_out(f"src/mod{i}.py:{i}: needle{i} = {i * 7}\n" + "x" * big, code=(i // 4) % 2))})
        elif kind == 1:
            msgs.append({"role": "assistant", "content": [], "tool_calls": [
                {"id": cid, "type": "function", "function": {
                    "name": "str_replace_editor", "arguments": json.dumps(
                        {"command": "view", "path": f"/workspace/acme__widget__1.0/src/mod{i}.py"})}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "name": "str_replace_editor",
                         "content": T(f"Here's the result of running `cat -n` on mod{i}.py:\n" + "y" * big)})
        elif kind == 2:
            msgs.append({"role": "assistant", "content": T("Thinking aloud."), "tool_calls": [
                {"id": cid, "type": "function", "function": {
                    "name": "think", "arguments": json.dumps({"thought": "PRIVATE-REASONING " * 20})}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "name": "think",
                         "content": T("Your thought has been logged.")})
        else:
            msgs.append({"role": "assistant", "content": [], "tool_calls": [
                {"id": cid, "type": "function", "function": {
                    "name": "str_replace_editor", "arguments": json.dumps(
                        {"command": "str_replace", "path": f"/workspace/acme__widget__1.0/src/mod{i}.py",
                         "old_str": "a = 1", "new_str": f"a = {i}  # FIXED-{i}"})}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "name": "str_replace_editor",
                         "content": T("The file mod.py has been edited. Here's the result of running `cat -n` "
                                      "on a snippet:\n 1\ta = 2\nReview the changes and make sure they are as "
                                      "expected. Edit the file again if necessary.")})
    return msgs


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.ev, self.st = io_oh.extract_events(make_traj(8), 400, 1200)

    def test_system_and_template_dropped(self):
        blob = "\n".join(e["text"] for e in self.ev)
        self.assertNotIn("SYSTEM-BOILERPLATE-XYZ", blob)
        self.assertNotIn("Phase 8", blob)
        self.assertEqual(self.ev[0], {"role": "user", "text": ISSUE, "ts": 0})
        self.assertTrue(self.st["issue_found"])

    def test_think_dropped_and_counted(self):
        self.assertNotIn("PRIVATE-REASONING", "\n".join(e["text"] for e in self.ev))
        self.assertEqual(self.st["think_dropped"], 2)

    def test_tool_format_matches_opencode_part_text(self):
        t = [e["text"] for e in self.ev if e["role"] == "tool"][0]
        self.assertTrue(t.startswith("[tool bash] in: {"), t)
        self.assertIn("\n[out] ", t)
        self.assertIn('"command": "grep -rn needle0 src/"', t)

    def test_bash_boilerplate_stripped_exit_kept(self):
        outs = [e["text"] for e in self.ev if e["text"].startswith("[tool bash]")]
        self.assertTrue(all("Python interpreter" not in o and "The command completed" not in o
                            and "Current working directory" not in o for o in outs))
        self.assertTrue(any("[exit 1]" in o for o in outs))

    def test_editor_mapping_and_tail_stripped(self):
        texts = [e["text"] for e in self.ev if e["role"] == "tool"]
        self.assertTrue(any(t.startswith("[tool read] in:") for t in texts))
        edit = next(t for t in texts if t.startswith("[tool edit] in:"))
        self.assertNotIn("Review the changes", edit)
        self.assertIn("newString", edit)

    def test_empty_assistant_content_not_emitted(self):
        self.assertTrue(all(e["text"].strip() for e in self.ev))

    def test_clip_head_tail(self):
        c = io_oh.clip("A" * 100 + "Z" * 100, 50)
        self.assertTrue(c.startswith("A" * 30) and c.endswith("Z" * 20))
        self.assertIn("chars elided", c)
        self.assertEqual(io_oh.clip("short", 50), "short")

    def test_unmatched_call_tolerated(self):
        m = make_traj(4)
        m = [x for x in m if not (x["role"] == "tool" and x["tool_call_id"] == "toolu_0000")]
        ev, st = io_oh.extract_events(m)
        self.assertEqual(st["unmatched_calls"], 1)

    def test_ids(self):
        self.assertEqual(io_oh.repo_of("sphinx-doc__sphinx-8595"), "sphinx-doc/sphinx")
        self.assertEqual(io_oh.sid_of("django__django-11099"), "django__django_11099")


class ConvertTests(unittest.TestCase):
    def setUp(self):
        self.s = io_oh.convert(make_traj(160), "acme__widget-123")

    def test_opencode_schema(self):
        s = self.s
        self.assertTrue({"sid", "title", "events", "boundaries", "span_chars"} <= set(s))
        for e in s["events"]:
            self.assertIn(e["role"], ("user", "assistant", "tool", "compaction"))
            self.assertTrue({"role", "text", "ts"} <= set(e))
        keys = {"idx", "auto", "original_summary", "original_summary_chars", "orig_summary_valid",
                "orig_summary_source", "orig_summary_reject", "orig_summary_note"}
        self.assertEqual(len(s["boundaries"]), 3)
        for b in s["boundaries"]:
            self.assertTrue(keys <= set(b))
            self.assertEqual(s["events"][b["idx"]]["role"], "compaction")
            self.assertFalse(b["orig_summary_valid"])
            self.assertEqual(b["original_summary"], "")
        self.assertEqual(len(s["span_chars"]), 3)

    def test_title_has_no_issue_text(self):
        self.assertNotIn("frobnicate", self.s["title"])

    def test_spans_are_exact_replay_windows(self):
        for k, sp in enumerate(mc.session_spans(self.s)[:3]):
            segs = mc.segments_struct(sp)
            win, dropped = mc.window_struct(segs, 8)
            self.assertEqual(len(win), 8)
            if k:
                self.assertEqual(dropped, 0)
                self.assertEqual(len(segs), 8)
            self.assertTrue(mc.selfcheck_span(sp, 8))
        self.assertEqual(io_oh.check_session(self.s), [])

    def test_make_chains_build_chain_consumes_it(self):
        ch = mc.build_chain(self.s, 0, 8, True)
        self.assertEqual(ch["reasons"], [])
        self.assertEqual([sp["lag"] for sp in ch["spans"]], [3, 2, 1])
        for sp in ch["spans"]:
            self.assertGreaterEqual(sp["chars"], mc.MIN_SPAN_CHARS)
            self.assertLess(sp["est_tokens"], 20000)

    def test_deterministic(self):
        self.assertEqual(json.dumps(io_oh.convert(make_traj(160), "acme__widget-123")),
                         json.dumps(self.s))

    def test_short_session_raises(self):
        with self.assertRaises(ValueError):
            io_oh.convert(make_traj(12), "acme__widget-1")

    def test_six_boundaries_two_disjoint_chains(self):
        s = io_oh.convert(make_traj(400), "acme__widget-9", n_boundaries=6)
        self.assertEqual(len(s["boundaries"]), 6)
        self.assertEqual(list(mc.chain_starts(6, disjoint=True)), [0, 3])


class CliTests(unittest.TestCase):
    def test_cli_writes_loadable_session_and_select_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src")
            os.makedirs(src)
            for n in ("acme__widget-1", "acme__widget-2", "beta__gadget-3"):
                with open(os.path.join(src, n + ".json"), "w") as fh:
                    json.dump(make_traj(200, issue=ISSUE + " variant " + n), fh)
            man = os.path.join(d, "man.json")
            argv = sys.argv
            try:
                sys.argv = ["select", "--src-dir", src, "--out", man, "--n", "3", "--min-raw-chars", "1000",
                            "--min-extracted-chars", "1000", "--min-tool-calls", "10",
                            "--min-error-outputs", "1", "--seed", "7"]
                self.assertEqual(sel.main(), 0)
                m = json.load(open(man))
                self.assertEqual(m["seed"], 7)
                self.assertEqual(len(m["selected"]), 3)
                self.assertEqual(len(m["selected"][0]["sha256"]), 64)
                out = os.path.join(d, "sessions")
                sys.argv = ["imp", "--out-dir", out, "--manifest", man, "--src-dir", src]
                self.assertEqual(io_oh.main(), 0)
            finally:
                sys.argv = argv
            files = sorted(os.listdir(out))
            self.assertEqual(len(files), 3)
            old = mc.SESS_DIR
            mc.SESS_DIR = out
            try:
                self.assertEqual(len(mc.load_sessions()), 3)
            finally:
                mc.SESS_DIR = old

    def test_pick_diversity_cap_and_seed(self):
        rows = [{"instance_id": f"{r}__{r}-{i}", "repo": f"{r}/{r}", "sid": f"{r}__{r}_{i}",
                 "issue_sha1": f"{r}{i}"} for r in "abcd" for i in range(10, 16)]
        a = sel.pick(rows, 7, 3, 1)
        self.assertEqual(a, sel.pick(rows, 7, 3, 1))
        self.assertEqual(len({r["repo"] for r in a[:4]}), 4)      # round 1: one per repo
        self.assertTrue(all(sum(1 for x in a if x["repo"] == r) <= 3 for r in {x["repo"] for x in a}))
        self.assertNotEqual([r["instance_id"] for r in a], [r["instance_id"] for r in sel.pick(rows, 7, 3, 2)])

    def test_refuses_tracked_output_path(self):
        here = os.path.join(SCRIPTS, "definitely_not_ignored_dir")
        with self.assertRaises(SystemExit):
            io_oh.ensure_not_tracked_path(here)
        io_oh.ensure_not_tracked_path(os.path.join(SCRIPTS, "longreplay-oh", "sessions"), is_dir=True)  # gitignored


class GuardTests(unittest.TestCase):
    def test_guard_reasons(self):
        s = io_oh.convert(make_traj(160), "acme__widget-123")
        issue, fix = ohp.contaminated_text(s)
        self.assertIn("frobnicate", issue)
        self.assertIn("FIXED-", fix)
        base = {"kind": "answerable", "question": "Which file did the agent grep for needle8?", "gold_anchor": "needle8"}
        self.assertEqual(ohp.guard_reasons(base, issue, fix), [])
        self.assertIn("oh_anchor_in_issue", ohp.guard_reasons(dict(base, gold_anchor="frobnicate"), issue, fix))
        self.assertIn("oh_anchor_in_fix", ohp.guard_reasons(dict(base, gold_anchor="FIXED-7"), issue, fix))
        q = {"kind": "answerable", "question": "When the frob count is zero and the mode is strict what happens?",
             "gold_anchor": "needle8"}
        self.assertIn("oh_question_restates_issue", ohp.guard_reasons(q, issue, fix))
        self.assertEqual(ohp.guard_reasons({"kind": "unanswerable", "question": q["question"]}, issue, fix), [])

    def test_install_guard_wraps_validate(self):
        class FakeMc:
            RULES = "R"
            def __init__(self, s): self.s = s
            def load_sessions(self): return [self.s]
            transcript_of = staticmethod(mc.transcript_of)
            @staticmethod
            def validate(item, spans, transcript): return True, [], {}
        s = io_oh.convert(make_traj(160), "acme__widget-123")
        f = FakeMc(s)
        f.load_sessions = f.load_sessions
        ohp.install_guard(f)
        self.assertIn("CONTAMINATION GUARD", f.RULES)
        f.load_sessions()
        ok, why, checks = f.validate({"kind": "answerable", "question": "q q q q q q", "gold_anchor": "frobnicate"},
                                     [], mc.transcript_of(s))
        self.assertFalse(ok)
        self.assertIn("oh_anchor_in_issue", why)
        self.assertFalse(checks["oh_guard"])


if __name__ == "__main__":
    unittest.main()
