"""Offline tests: Claude Code transcript importer on synthetic transcripts (built here)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import ls_fixtures as F  # noqa: F401
import import_claude_code as C
import longsession as L

SID = "sess1111-2222-3333-4444-555555555555"
_n = [0]


def ts():
    _n[0] += 1
    return f"2026-01-01T00:{_n[0] // 60:02d}:{_n[0] % 60:02d}.000Z"


def base(uuid, parent, typ, **kw):
    d = {"parentUuid": parent, "isSidechain": False, "type": typ, "uuid": uuid, "timestamp": ts(),
         "sessionId": SID}
    d.update(kw)
    return d


def U(uuid, parent, content, **kw):
    return base(uuid, parent, "user", message={"role": "user", "content": content}, **kw)


def A(uuid, parent, blocks, mid="m1", **kw):
    return base(uuid, parent, "assistant", message={"id": mid, "role": "assistant", "content": blocks}, **kw)


def T(text):
    return {"type": "text", "text": text}


def TU(i, name="Bash", inp=None):
    return {"type": "tool_use", "id": i, "name": name, "input": inp or {"command": "ls"}}


def TR(i, content, err=False):
    d = {"type": "tool_result", "tool_use_id": i, "content": content}
    if err:
        d["is_error"] = True
    return d


def write(path, entries, extra_lines=()):
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
        for l in extra_lines:
            f.write(l + "\n")


def run(tmp, entries, name="a.jsonl", **kw):
    p = os.path.join(tmp, name)
    write(p, entries)
    return C.import_paths([p], **kw)


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.tmp = self.d.name

    def tearDown(self):
        self.d.cleanup()

    def test_basic_roles_blocks_and_noise(self):
        es = [
            U("u1", None, "hello <system-reminder>noise</system-reminder> world"),
            {"type": "attachment", "uuid": "at1", "parentUuid": "u1", "isSidechain": False, "attachment": {"type": "x"}},
            A("a1", "at1", [{"type": "thinking", "thinking": "hmm", "signature": "s"}], mid="m1"),
            A("a2", "a1", [T("part one")], mid="m1"),
            A("a3", "a2", [T("part two"), TU("t1", "Bash", {"command": "echo " + "x" * 500})], mid="m1"),
            U("r1", "a3", [TR("t1", [{"type": "text", "text": "y" * 900}])], sourceToolAssistantUUID="a3"),
            A("a4", "r1", [T("done")], mid="m2"),
            U("meta1", "a4", "injected harness text", isMeta=True),
            base("sys1", "meta1", "system", subtype="stop_hook_summary"),
        ]
        (s,) = run(self.tmp, es)
        self.assertEqual([m["role"] for m in s["messages"]], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(s["messages"][0]["text"], "hello world")
        self.assertNotIn("noise", s["messages"][0]["text"])
        self.assertEqual(s["messages"][1]["text"], "part one\npart two")        # same message.id merged
        tool = s["messages"][2]["text"]
        self.assertTrue(tool.startswith("[tool Bash] in: {"))
        self.assertIn("\n[out] ", tool)
        self.assertLessEqual(len(tool.split("\n[out] ")[0]), len("[tool Bash] in: ") + C.ARGS_CAP)
        self.assertEqual(len(tool.split("\n[out] ")[1]), C.OUT_CAP)
        self.assertEqual(s["meta"]["meta_skipped"], 1)
        self.assertEqual(s["meta"]["thinking_dropped"], 1)
        self.assertEqual(L.validate_session(s), [])
        self.assertEqual(s["source"], "claude_code")
        self.assertTrue(SID_OK(s["sid"]))

    def test_parallel_tool_results_are_not_lost(self):
        es = [U("u1", None, "go"),
              A("a1", "u1", [TU("t1", "Read", {"p": 1})], mid="m1"),
              A("a2", "a1", [TU("t2", "Read", {"p": 2})], mid="m1"),
              U("r1", "a1", [TR("t1", "out-one")], sourceToolAssistantUUID="a1"),
              U("r2", "a2", [TR("t2", "out-two")], sourceToolAssistantUUID="a2"),
              A("a3", "r2", [T("both done")], mid="m2")]
        (s,) = run(self.tmp, es)
        tools = [m["text"] for m in s["messages"] if m["role"] == "tool"]
        self.assertEqual(len(tools), 2)
        self.assertTrue(any("out-one" in t for t in tools) and any("out-two" in t for t in tools))
        self.assertEqual(s["meta"]["dead_branch_entries"], 0)

    def test_unanswered_tool_use_and_error_result(self):
        es = [U("u1", None, "go"), A("a1", "u1", [TU("t1"), TU("t2", "Edit", {"f": "x"})], mid="m1"),
              U("r1", "a1", [TR("t1", "boom", err=True)], sourceToolAssistantUUID="a1")]
        (s,) = run(self.tmp, es)
        texts = [m["text"] for m in s["messages"] if m["role"] == "tool"]
        self.assertTrue(any("[out] [error] boom" in t for t in texts))
        self.assertTrue(any(t.endswith("[out] (no result)") for t in texts))
        self.assertEqual(s["meta"]["unanswered_tool_use"], 1)

    def test_compaction_boundary_and_summary(self):
        es = [U("u1", None, "first"), A("a1", "u1", [T("one")]), U("u2", "a1", "second"), A("a2", "u2", [T("two")], mid="m2"),
              base("cb", None, "system", subtype="compact_boundary", logicalParentUuid="a2",
                   compactMetadata={"trigger": "auto", "preTokens": 167000}),
              U("sum", "cb", C.SUMMARY_LEAD + " ... Summary: the work so far.", isCompactSummary=True, isVisibleInTranscriptOnly=True),
              U("u3", "sum", "third"), A("a3", "u3", [T("three")], mid="m3")]
        (s,) = run(self.tmp, es)
        self.assertEqual(len(s["messages"]), 6)            # the summary is NOT a message
        (c,) = s["compactions"]
        self.assertEqual(c["at_idx"], 4)
        self.assertTrue(c["original_summary"].endswith("the work so far."))
        self.assertEqual((c["auto"], c["pre_tokens"]), (True, 167000))
        self.assertEqual(s["messages"][4]["text"], "third")
        self.assertEqual(s["meta"]["n_summaries"], 1)
        self.assertEqual(L.validate_session(s), [])

    def test_summary_without_boundary_line_still_makes_a_compaction(self):
        es = [U("u1", None, "first"), A("a1", "u1", [T("one")]),
              U("sum", "a1", C.SUMMARY_LEAD + " ... Summary: S.", isCompactSummary=True), U("u2", "sum", "next")]
        (s,) = run(self.tmp, es)
        self.assertEqual([c["at_idx"] for c in s["compactions"]], [2])
        self.assertEqual(len(s["messages"]), 3)

    def test_rewound_branch_is_dropped(self):
        es = [U("u1", None, "q"), A("old", "u1", [T("OLD reply")], mid="m1"),
              A("new", "u1", [T("NEW reply")], mid="m2"), U("u2", "new", "next")]
        (s,) = run(self.tmp, es)
        texts = [m["text"] for m in s["messages"]]
        self.assertIn("NEW reply", texts)
        self.assertNotIn("OLD reply", texts)
        self.assertEqual(s["meta"]["dead_branch_entries"], 1)
        (f,) = run(self.tmp, es, name="b.jsonl", order="file")
        self.assertEqual(len(f["messages"]), 4)             # file order keeps everything

    def test_broken_parent_link_falls_back_to_file_order(self):
        es = [U("u1", None, "a"), A("a1", "u1", [T("b")]), U("u2", "MISSING-PARENT", "c"), A("a2", "u2", [T("d")], mid="m2")]
        (s,) = run(self.tmp, es)
        self.assertEqual([m["text"] for m in s["messages"]], ["a", "b", "c", "d"])

    def test_sidechains_excluded_by_default_and_prefixed_when_included(self):
        es = [U("u1", None, "main"), A("a1", "u1", [T("main reply")]),
              U("s1", None, "sub task", isSidechain=True, agentId="agent12345678"),
              A("s2", "s1", [T("sub reply")], mid="ms", isSidechain=True, agentId="agent12345678")]
        (s,) = run(self.tmp, es)
        self.assertEqual(len(s["messages"]), 2)
        self.assertEqual(s["meta"]["sidechain_entries"], 2)
        (s2,) = run(self.tmp, es, name="c.jsonl", include_sidechains=True)
        self.assertEqual(len(s2["messages"]), 4)
        self.assertTrue(any(m["text"].startswith("[subagent agent123] ") for m in s2["messages"]))

    def test_subagents_dir_not_scanned_by_default(self):
        os.makedirs(os.path.join(self.tmp, "proj", "x", "subagents"))
        write(os.path.join(self.tmp, "proj", "main.jsonl"), [U("u1", None, "m"), A("a1", "u1", [T("r")])])
        write(os.path.join(self.tmp, "proj", "x", "subagents", "agent-1.jsonl"),
              [U("s1", None, "s", isSidechain=True), A("s2", "s1", [T("t")], isSidechain=True)])
        self.assertEqual(len(C.discover([os.path.join(self.tmp, "proj")])), 1)
        self.assertEqual(len(C.discover([os.path.join(self.tmp, "proj")], include_subagents=True)), 2)

    def test_resumed_files_merge_into_one_session(self):
        e = [U("u1", None, "one"), A("a1", "u1", [T("two")]), U("u2", "a1", "three"), A("a2", "u2", [T("four")], mid="m2")]
        p1, p2 = os.path.join(self.tmp, "r1.jsonl"), os.path.join(self.tmp, "r2.jsonl")
        write(p1, e)
        later = [dict(x) for x in e] + [U("u3", "a2", "five"), A("a3", "u3", [T("six")], mid="m3")]
        for x in later:
            x["timestamp"] = "2027" + x["timestamp"][4:]
        write(p2, later)
        merged = C.import_paths([p1, p2])
        self.assertEqual(len(merged), 1)
        self.assertEqual([m["text"] for m in merged[0]["messages"]], ["one", "two", "three", "four", "five", "six"])
        self.assertEqual(merged[0]["meta"]["files"], ["r1.jsonl", "r2.jsonl"])
        split = C.import_paths([p1, p2], session_per_file=True)
        self.assertEqual(sorted(len(s["messages"]) for s in split), [4, 6])

    def test_unrelated_files_stay_separate_sessions(self):
        p1, p2 = os.path.join(self.tmp, "x1.jsonl"), os.path.join(self.tmp, "x2.jsonl")
        write(p1, [U("u1", None, "a"), A("a1", "u1", [T("b")])])
        write(p2, [U("v1", None, "c"), A("b1", "v1", [T("d")])])
        self.assertEqual(len(C.import_paths([p1, p2])), 2)

    def test_redaction_default_on_with_counts_and_opt_out(self):
        key = "s" + "k-ant-" + "api03-" + "Z9y8X7w6" * 5
        es = [U("u1", None, f"my key is {key} ok"), A("a1", "u1", [T("password=Zx9fQ2mK7pLw4RtY8vNb stored")])]
        (s,) = run(self.tmp, es)
        self.assertNotIn(key, s["messages"][0]["text"])
        self.assertIn("[REDACTED:sk_ant]", s["messages"][0]["text"])
        self.assertEqual(s["meta"]["redaction"]["counts"], {"sk_ant": 1, "assign": 1})
        (raw,) = run(self.tmp, es, name="n.jsonl", redact=False)
        self.assertIn(key, raw["messages"][0]["text"])
        self.assertFalse(raw["meta"]["redaction"]["enabled"])

    def test_bad_lines_are_counted_not_fatal(self):
        p = os.path.join(self.tmp, "bad.jsonl")
        write(p, [U("u1", None, "a"), A("a1", "u1", [T("b")])], extra_lines=["{not json", ""])
        (s,) = C.import_paths([p])
        self.assertEqual(s["meta"]["bad_lines"], 1)
        self.assertEqual(len(s["messages"]), 2)

    def test_cli_writes_session_and_reports_redaction(self):
        p = os.path.join(self.tmp, "cli.jsonl")
        write(p, [U("u1", None, "hi"), A("a1", "u1", [T("yo")])])
        out = os.path.join(self.tmp, "out")
        self.assertEqual(C.main([p, "--out", out]), 0)
        (s,) = L.load_sessions(out)
        self.assertEqual(s["messages"][1]["text"], "yo")
        self.assertEqual(C.main([p, "--out", out, "--min-tokens", "10000000"]), 1)


def SID_OK(sid):
    return bool(L.SID_RE.match(sid))


if __name__ == "__main__":
    unittest.main()
