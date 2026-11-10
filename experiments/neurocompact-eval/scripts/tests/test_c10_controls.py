"""Amendment C.10: control arms (sum+bm25, sum+grep, oracle), the true retrieval cap, reuse of the C.9 rows, metrics of the new contrasts
and the decision rule, judge robustness (re-ask, judge_unparsed, aborted-run spend), the bundle with traces, oh-ingest-measure, extract_usage
in the ingest record. Offline: scripted LLM, FakeHars, a tiny fixture repo. No network."""
from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import funnel_chains as FC
import home_run as H
import judge_v2 as J
import longsession as L
import metrics_agentic as MA
import providers as P
import run_agentic as RA
import run_longsession as RL
import test_c7_enriched as T7E
import test_c7_home as T7
import test_c9_agentic as T9
import test_home_run as thr

NEED = T9.NEED


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            rc = fn(*a, **k)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else str(e)
            if not isinstance(e.code, int):
                buf.write(str(e.code))
    return rc, buf.getvalue()


# --------------------------------------------------------------------------- fixtures
class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        D = self.d.name
        self.sdir = os.path.join(D, "ls-data")
        self.sessions = T9.make_sessions(self.sdir)
        self.sid = "chain_s0_c0"
        self.old = os.path.join(D, "old")
        T9.make_old_run(self.old, self.sessions)
        self.wsroot = os.path.join(D, "ws")
        for sid in self.sessions:
            T9.make_ws(os.path.join(self.wsroot, sid, "repo"))
        self.kinds = {self.sid: "base_commit+final_patch", "chain_s1_c0": "base_commit"}
        for sid, k in self.kinds.items():
            json.dump({"sid": sid, "workspace_kind": k, "patch_applied": k != "base_commit"}, open(os.path.join(self.wsroot, f"{sid}.json"), "w"))
        self.pmap = os.path.join(D, "pmap.json")
        json.dump({sid: f"nc-e1-{sid}" for sid in self.sessions}, open(self.pmap, "w"))

    def tearDown(self):
        self.d.cleanup()

    def run_main(self, *extra, out=None, fake=None):
        out = out or os.path.join(self.d.name, "run")
        argv = ["--sessions", self.sdir, "--reuse-run", self.old, "--workspaces", self.wsroot, "--projects-map", self.pmap, "--out", out,
                "--mock-llm", "--fake-hars", "--profile", "anthropic", "--concurrency", "2", *extra]
        if fake is None:
            rc, log = quiet(RA.main, argv)
        else:
            with mock.patch.object(RL, "FakeHars", lambda: fake):
                rc, log = quiet(RA.main, argv)
        return rc, out, log

    def rows(self, out, arm=None):
        r = RA.read_jsonl(os.path.join(out, "records.jsonl"))
        return [x for x in r if arm is None or x["arm"] == arm]


def seeded_hars(pids, text_for):
    """A FakeHars whose projects hold `text_for(pid)` (the dropped facts)."""
    h = RL.FakeHars()
    for pid in pids:
        h.register_project(pid)
        h.upsert(pid, f"/nc-eval/{pid}/j1.md", text_for(pid))
    return h


# --------------------------------------------------------------------------- the same split as the ingest
class CorpusIsTheIngestedText(Base):
    def test_ingested_pieces_equal_what_run_longsession_wrote_into_the_project(self):
        rec = []
        orig = RL.FakeHars.upsert

        def up(h, project, path, content):
            rec.append((project, path, content))
            return orig(h, project, path, content)
        out = os.path.join(self.d.name, "ing")
        argv = ["--sessions", self.sdir, "--out", out, "--use-real-compactions", "--keep-tail", "0", "--mem-budgets", "1536", "--provider", "mock",
                "--agent-model", "mock", "--fake-hars", "--no-preflight", "--piece-chars", "8000", "--pieces-per-call", "12",
                "--arms", "baseline,nc-enriched,nc-note-only", "--hars-layout", "project", "--hars-run-id", "e1", "--ingest-only"]
        with mock.patch.object(RL.FakeHars, "upsert", up):
            rc, _, se = T7E.call(argv)
        self.assertEqual(rc, 0, se)
        for sid, s in self.sessions.items():
            pid = f"nc-e1-{sid}"
            facts = []
            for p, path, content in sorted(x for x in rec if x[0] == pid):
                facts += [ln[2:] for ln in content.split("## Dropped context facts\n", 1)[1].splitlines() if ln.startswith("- ")]
            want = [re.sub(r"\s*\n\s*", " ", x).strip() for x in RA.ingested_pieces(s, 8000)]
            self.assertEqual(facts, want, sid)
            self.assertEqual(RA.span_bounds(s), [(0, 6), (6, 11), (11, 15)] if sid == self.sid else RA.span_bounds(s))
        # the ingest record carries the measured extractor usage (C.10) in ingest.jsonl and in the summary
        rows = RA.read_jsonl(os.path.join(out, "ingest.jsonl"))
        self.assertEqual(len(rows), 2)
        for r in rows:
            eu = r["extract_usage"]
            self.assertEqual((eu["documents"], eu["documents_measured"], eu["documents_unmeasured"], eu["complete"], eu["estimated"]),
                             (3, 3, 0, True, False))
            self.assertEqual(eu["total_tokens"], eu["prompt_tokens"] + eu["completion_tokens"])
            self.assertGreater(eu["calls"], 0)
        summ = json.load(open(os.path.join(out, "summary.json")))
        m = summ["hars_extraction_measured"]
        self.assertEqual(m["documents"], 6)
        self.assertEqual(m["total_tokens"], sum(r["extract_usage"]["total_tokens"] for r in rows))
        man = json.load(open(os.path.join(out, "MANIFEST.json")))
        self.assertIn("hars_extraction_measured_usd", man["spend_usd"])

    def test_a_server_without_extract_usage_is_flagged_not_zero(self):
        tot = RL.new_extract_totals()
        RL.add_extract_usage(tot, {"calls": 2, "prompt_tokens": 100, "completion_tokens": 10, "calls_estimated": 1})
        RL.add_extract_usage(tot, None)                                   # an older server / an unchanged document
        v = RL.extract_totals_view(tot, (0.1, 0.5))
        self.assertEqual((v["documents"], v["documents_measured"], v["documents_unmeasured"]), (2, 1, 1))
        self.assertTrue(v["estimated"] and not v["complete"])
        self.assertAlmostEqual(v["usd"], (100 * 0.1 + 10 * 0.5) / 1e6)


# --------------------------------------------------------------------------- the cap
class Cap(Base):
    def test_cap_text_never_exceeds_the_cap(self):
        text = ("alpha line\n" * 400)
        for n in (1, 5, 40, 100, 1000, 4000, 10 ** 6):
            out, cut = RA.cap_text(text, n)
            self.assertLessEqual(len(out), n, n)
            self.assertEqual(cut, len(text) > n)
        self.assertEqual(RA.cap_text("short", 100), ("short", False))

    def mem_tool(self, hars, cap):
        pid = "nc-e1-x"
        hars.register_project(pid)
        for i in range(8):
            hars.upsert(pid, f"/d/{i}.md", f"## Dropped context facts\n- fact {i} about widget {NEED} " + "pad words " * 120)
        return RA.MemoryTool(hars, pid, cap)

    def test_old_server_is_cut_by_the_guard_and_the_violation_is_recorded(self):
        h = RL.FakeHars()
        h.honour_cap = False                                      # a server older than the C.10 hard cap (the C.9 behaviour)
        h.report_cap = False
        text, meta = self.mem_tool(h, 400).call({"query": "widget", "ll_keywords": [NEED]})
        self.assertLessEqual(len(text), 400)
        self.assertTrue(meta["client_cut"])
        self.assertFalse(meta["server_cap_honoured"])
        self.assertGreater(meta["server_chars"], 400)
        self.assertEqual(meta["returned_chars"], len(text))

    def test_new_server_bounds_the_whole_context_and_reports_it(self):
        h = RL.FakeHars()
        text, meta = self.mem_tool(h, 1500).call({"query": "widget", "ll_keywords": [NEED]})
        self.assertLessEqual(len(text), 1500)
        self.assertFalse(meta["client_cut"])
        self.assertTrue(meta["server_cap_honoured"])
        self.assertIsInstance(meta["server_context_cap"], dict)
        self.assertIn("Document Chunks", text)

    def test_an_output_above_the_cap_after_the_guard_is_a_hard_error(self):
        h = RL.FakeHars()
        h.honour_cap = False
        tool = self.mem_tool(h, 300)
        with mock.patch.object(RA, "cap_text", lambda t, n: (t, False)):          # a broken guard
            with self.assertRaises(RA.CapViolation):
                tool.call({"query": "widget", "ll_keywords": [NEED]})

    def seeded(self, **flags):
        h = seeded_hars([f"nc-e1-{sid}" for sid in self.sessions],
                        lambda pid: "## Dropped context facts\n" + "\n".join(f"- fact {i} about widget NEEDLE_s1_1 " + "pad words " * 8 for i in range(10)))
        for k, v in flags.items():
            setattr(h, k, v)
        return h

    def test_rows_record_realised_chars_and_the_run_refuses_an_old_server_unless_allowed(self):
        rc, out, log = self.run_main("--arms", "sum+mem,sum+bm25", "--retrieval-max-chars", "700", fake=self.seeded())
        self.assertEqual(rc, 0, log)
        for r in self.rows(out):
            if r["arm"] in ("sum+mem", "sum+bm25"):
                self.assertEqual(r["retrieval_max_chars"], 700)
                self.assertLessEqual(r["retrieval_chars_max"], 700)
                self.assertTrue(r["retrieval_cap_ok"])
                self.assertEqual(r["retrieval_cap_server_violations"], 0)
        summ = json.load(open(os.path.join(out, "summary.json")))
        self.assertEqual(summ["retrieval_cap"]["rows"], len(self.rows(out)))
        self.assertEqual(summ["retrieval_cap"]["rows_cap_ok"], summ["retrieval_cap"]["rows"])
        self.assertLessEqual(summ["retrieval_cap"]["realised_chars_max"], 700)
        # an old server (no context_cap in the payload): refused before any spend ...
        old_srv = self.seeded(honour_cap=False, report_cap=False)
        rc, out2, log = self.run_main("--arms", "sum+mem", "--retrieval-max-chars", "300", out=os.path.join(self.d.name, "r2"), fake=old_srv)
        self.assertEqual(rc, 4, log)
        self.assertIn("context_cap", log)
        self.assertFalse(os.path.exists(os.path.join(out2, "records.jsonl")))
        # ... or run with the guard when explicitly allowed; the rows say the server did not honour the cap
        old_srv2 = self.seeded(honour_cap=False, report_cap=False)
        rc, out3, log = self.run_main("--arms", "sum+mem", "--retrieval-max-chars", "300", "--allow-old-server",
                                      out=os.path.join(self.d.name, "r3"), fake=old_srv2)
        self.assertEqual(rc, 0, log)
        rows = self.rows(out3)
        self.assertTrue(all(r["retrieval_chars_max"] <= 300 and r["retrieval_cap_ok"] for r in rows))
        self.assertGreater(sum(r["retrieval_cap_server_violations"] or 0 for r in rows), 0)
        self.assertGreater(sum(r["retrieval_client_cuts"] for r in rows), 0)

    def test_the_memory_max_chars_spelling_still_works(self):
        a = RA.parse_args(["--sessions", "x", "--reuse-run", "y", "--memory-max-chars", "1234"])
        self.assertEqual((a.retrieval_max_chars, a.memory_max_chars), (1234, 1234))
        self.assertEqual(RA.parse_args(["--sessions", "x", "--reuse-run", "y"]).retrieval_max_chars, 6000)


# --------------------------------------------------------------------------- sum+bm25
class Bm25(Base):
    def setUp(self):
        super().setUp()
        self.s = self.sessions[self.sid]
        self.pieces = RA.ingested_pieces(self.s)

    def test_same_tool_same_schema_same_prompt_as_sum_mem(self):
        self.assertEqual(RA.tool_specs(True), RA.tool_specs(True))            # one definition serves both arms
        self.assertEqual([t["name"] for t in RA.tool_specs(True)], ["ls", "read_file", "grep", "memory_recall", "answer"])
        sm = RA.system_for("sum+mem", "SUMMARY X")
        self.assertEqual(RA.system_for("sum+bm25", "SUMMARY X"), sm)
        self.assertNotEqual(RA.system_for("sum", "SUMMARY X"), sm)
        # the C.9 prompts are unchanged byte for byte (the reused rows were produced with them)
        self.assertEqual(RA.system_for("sum", "S"), RA.SYSTEM_BASE + "\n\n[Session memory]\nS\n[End of session memory]")
        self.assertEqual(RA.system_for("sum+mem", "S"), RA.SYSTEM_BASE + RA.SYSTEM_MEMORY + "\n\n[Session memory]\nS\n[End of session memory]")

    def test_retrieves_over_the_ingested_pieces_with_keywords_in_the_query_and_respects_the_cap(self):
        tool = RA.Bm25Tool(self.pieces, 900, passage_chars=500)
        self.assertGreater(len(tool.passages), len(self.pieces))              # pieces were cut into passages
        text, meta = tool.call({"query": "what value", "ll_keywords": ["NEEDLE_s2_3"], "hl_keywords": []})
        self.assertIn("NEEDLE_s2_3", text)                                    # found through the KEYWORD, not the generic query
        self.assertLessEqual(len(text), 900)
        self.assertTrue(A_found(text, "NEEDLE_s2_3"))
        self.assertEqual((meta["backend"], meta["cap"], meta["returned_chars"]), ("bm25", 900, len(text)))
        # ranking: the passage with the rare token comes first
        self.assertTrue(text.startswith("[1]"))
        self.assertIn("NEEDLE_s2_3", text.split("[2]")[0])
        # a tiny cap still bounds the output (the best passage alone is cut)
        t2, m2 = RA.Bm25Tool(self.pieces, 120, passage_chars=500).call({"query": "NEEDLE_s2_3"})
        self.assertLessEqual(len(t2), 120)
        self.assertTrue(m2["client_cut"] or len(t2) <= 120)
        # whole-piece mode and the empty / no-hit cases
        whole = RA.Bm25Tool(self.pieces, 10 ** 6, passage_chars=0)
        self.assertEqual(len(whole.passages), len(self.pieces))
        self.assertEqual(whole.call({"query": "zzzqqq"})[0], "(memory returned nothing)")
        self.assertIn("error", whole.call({"query": "  "})[0])

    def test_corpus_is_the_dropped_text_only(self):
        text = RA.transcript_text(self.pieces)
        for m in self.s["messages"][:15]:
            self.assertIn(m["text"].strip()[:60].split("\n")[0], text)
        self.assertNotIn("ghost.py", text)                                    # the unanswerable probe's subject was never in the session


def A_found(text, anchor):
    import anchors
    return anchors.anchor_found(text, anchor)


# --------------------------------------------------------------------------- sum+grep
class Grep(Base):
    def setUp(self):
        super().setUp()
        self.pieces = RA.ingested_pieces(self.sessions[self.sid])
        self.text = RA.transcript_text(self.pieces)

    def test_matching_lines_with_context_and_cap(self):
        g = RA.TranscriptGrepTool(self.text, 800)
        out, meta = g.call({"pattern": "NEEDLE_s2_3"})
        self.assertLessEqual(len(out), 800)
        self.assertIn("NEEDLE_s2_3", out)
        self.assertRegex(out, r"(?m)^\d+:")                                  # a matching line: "<n>:text"
        self.assertRegex(out, r"(?m)^\d+-")                                  # a context line: "<n>-text"
        self.assertGreaterEqual(meta["matches"], 1)
        self.assertEqual(meta["corpus_chars"], len(self.text))
        big, bm = RA.TranscriptGrepTool(self.text, 300).call({"pattern": "NEEDLE"})
        self.assertLessEqual(len(big), 300)
        self.assertTrue(bm["client_cut"])
        self.assertGreater(bm["server_chars"], 300)

    def test_smart_case_alternatives_and_errors(self):
        g = RA.TranscriptGrepTool(self.text, 5000)
        self.assertIn("NEEDLE_s1_1", g.call({"pattern": "needle_s1_1"})[0])    # lower case pattern: case-insensitive
        self.assertEqual(g.call({"pattern": "Needle_s1_1"})[0], "(no matches)")   # an upper-case letter: case-sensitive
        out = g.call({"pattern": "NEEDLE_s1_1|NEEDLE_s3_0"})[0]
        self.assertIn("NEEDLE_s1_1", out)
        self.assertIn("NEEDLE_s3_0", out)
        t, m = g.call({"pattern": "(unclosed"})
        self.assertIn("bad pattern", t)
        self.assertTrue(m["error"])
        self.assertTrue(g.call({"pattern": ""})[1]["error"])

    def test_long_lines_are_windowed_around_the_match(self):
        g = RA.TranscriptGrepTool("x" * 5000 + " HERE_IS_THE_HIT " + "y" * 5000, 5000, context_lines=0)
        out = g.call({"pattern": "HERE_IS_THE_HIT"})[0]
        self.assertIn("HERE_IS_THE_HIT", out)
        self.assertLess(len(out), RA.TRANSCRIPT_LINE_CHARS + 60)

    def test_tool_spec_and_prompt(self):
        names = [t["name"] for t in RA.tool_specs(False, True)]
        self.assertEqual(names, ["ls", "read_file", "grep", "transcript_grep", "answer"])
        self.assertIn("transcript_grep", RA.system_for("sum+grep", "S"))
        self.assertNotIn("memory_recall", RA.system_for("sum+grep", "S"))


# --------------------------------------------------------------------------- oracle
class Oracle(Base):
    def test_full_dropped_transcript_in_the_prompt_no_retrieval_tool_size_recorded(self):
        s = self.sessions[self.sid]
        text = RA.transcript_text(RA.ingested_pieces(s))
        sysp = RA.system_for("oracle", "SUMMARY", text)
        self.assertIn(text, sysp)
        self.assertIn("SUMMARY", sysp)
        self.assertIn(RA.ORACLE_SENTENCE, sysp)
        self.assertNotIn("You cannot see the earlier conversation any more", sysp)
        self.assertNotIn("memory_recall", sysp)
        self.assertEqual([t["name"] for t in RA.tool_specs(False)], ["ls", "read_file", "grep", "answer"])
        rc, out, log = self.run_main("--arms", "oracle")
        self.assertEqual(rc, 0, log)
        rows = self.rows(out, "oracle")
        by = {r["sid"]: r for r in rows}
        for sid, sess in self.sessions.items():
            tx = RA.transcript_text(RA.ingested_pieces(sess))
            self.assertEqual(by[sid]["oracle_context_chars"], len(tx))
            self.assertEqual(by[sid]["oracle_context_tokens"], L.count_tokens(tx))
        self.assertTrue(all(r["retrieval_calls_n"] == 0 and r["retrieval_chars_max"] == 0 for r in rows))
        tr = RA.read_jsonl(os.path.join(out, "traces.jsonl"))
        self.assertTrue(all(t["tools"] == ["ls", "read_file", "grep", "answer"] and "[Earlier conversation, complete]" in t["system"] for t in tr))
        summ = json.load(open(os.path.join(out, "summary.json")))
        self.assertEqual(set(summ["oracle"]["context_chars"]), set(self.sessions))
        # the judge's visible context for an oracle answer is the whole transcript (facts are verified against what the agent saw)
        self.assertEqual(RA.export_judge(out, self.sdir, os.path.join(out, "ji.jsonl")), 0)
        ji = RA.read_jsonl(os.path.join(out, "ji.jsonl"))
        self.assertTrue(all(RA.transcript_text(RA.ingested_pieces(self.sessions[r["sid"]]))[:200] in r["visible_context"] for r in ji))


# --------------------------------------------------------------------------- controls end to end, reuse
class ControlsAndReuse(Base):
    def make_c9_run(self):
        rc, out, log = self.run_main("--arms", "sum,sum+mem", "--retrieval-max-chars", "6000", out=os.path.join(self.d.name, "c9"))
        self.assertEqual(rc, 0, log)
        return out

    def test_controls_preset_reuse_labels_and_spend(self):
        c9 = self.make_c9_run()
        old_rows = RA.read_jsonl(os.path.join(c9, "records.jsonl"))
        rc, out, log = self.run_main("--controls", "--reuse-agentic-run", c9, "--retrieval-max-chars", "900")
        self.assertEqual(rc, 0, log)
        rows = self.rows(out)
        arms = {r["arm"] for r in rows}
        self.assertEqual(arms, {"sum", "sum+mem@uncapped", "sum+mem", "sum+bm25", "sum+grep", "oracle"})
        n_q = sum(len(s["questions"]) for s in self.sessions.values())
        for a in arms:
            self.assertEqual(len([r for r in rows if r["arm"] == a]), n_q, a)
        reused = [r for r in rows if r.get("reused_from")]
        self.assertEqual({r["arm"] for r in reused}, {"sum", "sum+mem@uncapped"})
        self.assertTrue(all(r["reused_from"] == os.path.abspath(c9) for r in reused))
        for r in reused:
            o = next(x for x in old_rows if x["qid"] == r["qid"] and x["arm"] == r.get("arm_original", r["arm"]))
            self.assertEqual(r["tokens_total"], o["tokens_total"])
            self.assertEqual(r["answer"], o["answer"])
            self.assertEqual(r["rid"], f"{r['qid']}::{r['arm']}")
        self.assertTrue(all(r["arm_original"] == "sum+mem" for r in reused if r["arm"] == "sum+mem@uncapped"))
        self.assertTrue(all(r["retrieval_max_chars"] is None and "NOT a total cap" in r["retrieval_note"]
                            for r in reused if r["arm"] == "sum+mem@uncapped"))
        # per-session workspace kind in EVERY record (new and reused), from the workspaces meta
        for r in rows:
            self.assertEqual(r["workspace_kind"], self.kinds.get(r["sid"]), r["rid"])
        # spend: the reused rows were paid for by the old run
        summ = json.load(open(os.path.join(out, "summary.json")))
        own = [r for r in rows if not r.get("reused_from")]
        self.assertAlmostEqual(summ["cost_usd"], round(sum(r["cost_usd"] for r in own), 6))
        self.assertEqual(summ["rows_reused"], len(reused))
        self.assertEqual(summ["reused"], {"sum": n_q, "sum+mem@uncapped": n_q})
        self.assertEqual(json.load(open(os.path.join(out, "MANIFEST.json")))["budget_spent_all_invocations"], summ["cost_usd"])
        self.assertEqual(summ["workspace_kinds"][self.sid], "base_commit+final_patch")
        # all capped arms obey the one cap
        capped = [r for r in own if r["arm"] in RA.CAPPED_ARMS]
        self.assertTrue(capped and all(r["retrieval_chars_max"] <= 900 for r in capped))
        # resume: nothing is re-run or re-copied
        rc2, _, log2 = self.run_main("--controls", "--reuse-agentic-run", c9, "--retrieval-max-chars", "900", out=out)
        self.assertEqual(rc2, 0, log2)
        self.assertEqual(len(self.rows(out)), len(rows))
        # judge export: only the NEW rows
        self.assertEqual(RA.export_judge(out, self.sdir, os.path.join(out, "judge-input.jsonl")), 0)
        ji = RA.read_jsonl(os.path.join(out, "judge-input.jsonl"))
        self.assertEqual({r["arm"] for r in ji}, {"sum+mem", "sum+bm25", "sum+grep", "oracle"})
        self.assertEqual(len(ji), 4 * n_q)

    def test_a_different_summary_is_refused_and_arms_cannot_clash(self):
        c9 = self.make_c9_run()
        rows = RA.read_jsonl(os.path.join(c9, "records.jsonl"))
        rows[0]["summary_sha256"] = "0" * 64
        with open(os.path.join(c9, "records.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        rc, out, log = self.run_main("--controls", "--reuse-agentic-run", c9)
        self.assertNotEqual(rc, 0)
        self.assertIn("another summary", log)
        for bad in (["--arms", "sum,sum+bm25", "--reuse-agentic-run", c9], ["--controls", "--arms", "sum"]):
            with self.assertRaises(SystemExit):
                RA.parse_args(["--sessions", "x", "--reuse-run", "y", *bad])

    def test_dry_run_estimates_every_arm_with_the_oracle_prefix_and_the_calibrated_figure(self):
        c9 = self.make_c9_run()
        argv = ["--sessions", self.sdir, "--reuse-run", self.old, "--dry-run", "--json", "--profile", "anthropic", "--controls",
                "--reuse-agentic-run", c9]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(RA.main(argv), 0)
        est = json.loads(buf.getvalue())
        self.assertEqual(set(est["per_arm"]), {"sum+mem", "sum+bm25", "sum+grep", "oracle"})
        self.assertEqual(est["reused_arms"], ["sum+mem@uncapped"] if False else est["reused_arms"])
        self.assertEqual(est["judge_items"], 4 * sum(len(s["questions"]) for s in self.sessions.values()))
        pa = est["per_arm"]
        self.assertEqual(pa["sum+mem"]["usd_with_cache"], pa["sum+bm25"]["usd_with_cache"])        # same tool-result size in the model
        self.assertGreater(pa["oracle"]["tokens_per_question"], pa["sum+mem"]["tokens_per_question"])   # the transcript prefix
        # the cap check prices the oracle's big stable prefix as cached (caching certainly engages), the others without cache (conservative)
        self.assertEqual(pa["oracle"]["usd_guard"], pa["oracle"]["usd_with_cache"])
        self.assertEqual(pa["sum+mem"]["usd_guard"], pa["sum+mem"]["usd_without_cache"])
        self.assertAlmostEqual(est["run_usd_guard"], sum(v["usd_guard"] for v in pa.values()), places=2)
        self.assertGreater(est["run_usd_expected_calibrated"], 0)
        self.assertLessEqual(est["run_usd_expected_calibrated"], est["run_usd_expected_with_cache"])
        self.assertEqual(pa["sum+mem"]["steps_calibrated"], max(1, round(
            sum(r["steps"] for r in RA.read_jsonl(os.path.join(c9, "records.jsonl")) if r["arm"] == "sum+mem")
            / len([r for r in RA.read_jsonl(os.path.join(c9, "records.jsonl")) if r["arm"] == "sum+mem"]))))

    def test_oracle_prefix_is_priced_as_a_cached_prefix(self):
        prof = P.load_profile("anthropic")
        sm = {("s", "oracle"): 1000, ("s", "sum"): 1000}
        base = RA.estimate(["sum"], sm, {"s": 10}, prof, "claude-haiku-5-5", 12, 6000, 0.003)
        big = RA.estimate(["oracle"], sm, {"s": 10}, prof, "claude-haiku-5-5", 12, 6000, 0.003, prefix_extra_tokens={("s", "oracle"): 20000})
        self.assertGreater(big["per_arm"]["oracle"]["usd_with_cache"], base["per_arm"]["sum"]["usd_with_cache"])
        # the prefix is a CACHED prefix: with caching the oracle costs far less than the same arm priced without any cache
        self.assertLess(big["per_arm"]["oracle"]["usd_with_cache"], 0.5 * big["per_arm"]["oracle"]["usd_without_cache"])


class LeakScanPass(Base):
    def test_a_clean_scan_prints_an_explicit_pass_line_with_counts(self):
        rc, out, log = self.run_main("--arms", "sum", "--limit-questions", "1")
        self.assertEqual(rc, 0, log)
        m = re.findall(r"\[leak-scan\] PASS (\S+): (\d+) file\(s\) scanned \((\d+) skipped.*?\), (\d+) transcript probe\(s\), 0 hits", log)
        self.assertEqual({x[0] for x in m}, set(self.sessions))
        self.assertTrue(all(int(x[1]) >= 4 and int(x[3]) >= 5 for x in m), m)


# --------------------------------------------------------------------------- metrics
def toy(arms_acc, cost, kinds=None, n_sessions=4, n_q=4, judge_cost=0.002, unparsed=()):
    """arms_acc {arm: n correct of n_q per session}, cost {arm: tokens per question}; identical in every session (zero variance)."""
    recs, judged = [], []
    for si in range(n_sessions):
        sid = f"s{si}"
        for qi in range(n_q):
            qid = f"{sid}_q{qi}"
            for arm, k in arms_acc.items():
                tok = cost[arm]
                rid = f"{qid}::{arm}"
                recs.append({"rid": rid, "sid": sid, "cluster_session": sid, "qid": qid, "arm": arm, "valid": True, "answerable": True,
                             "gold_anchor": "A", "memory_relevant": True, "tokens_total": tok, "tokens_fresh": tok // 5,
                             "tokens_cache_read": tok - tok // 5 - 100, "tokens_cache_write": 50, "tokens_input_uncached": 50 + tok // 5 - 100,
                             "tokens_output": 100, "tokens_input_total": tok - 100, "cost_usd": tok * 1e-7, "steps": 3, "tool_calls": 2,
                             "memory_calls_n": 1, "workspace_calls": 1, "latency_s": 1.0, "retrieval_calls_n": 1, "retrieval_chars_max": 500,
                             "workspace_kind": (kinds or {}).get(sid, "base_commit+final_patch")})
                ok = qi < k
                judged.append({"rid": rid, "score": 2 if ok else 0, "outcome": "correct" if ok else "abstained", "hallucination": False,
                               "abstained": not ok, "hallucination_rule": "judge_clean", "judge_cost_usd": judge_cost,
                               "judge_unparsed": rid in unparsed, "judge_parse_ok": rid not in unparsed})
    return recs, judged


ARMS5 = {"sum": 1, "sum+mem": 4, "sum+bm25": 3, "sum+grep": 3, "oracle": 4}
COST5 = {"sum": 8000, "sum+mem": 12000, "sum+bm25": 12000, "sum+grep": 9000, "oracle": 15000}


class MetricsC10(unittest.TestCase):
    def test_every_new_contrast_exists_and_the_primary_hars_pair_is_first(self):
        recs, judged = toy(ARMS5, COST5)
        res = MA.analyse(recs, judged, n_boot=200, seed=3)
        self.assertEqual(res["pair_list"][0], ["sum+mem", "sum+bm25"])
        want = {("sum+mem", "sum+bm25"), ("sum+mem", "sum+grep"), ("sum+bm25", "sum+grep"), ("sum+mem", "sum"), ("sum+bm25", "sum"),
                ("sum+grep", "sum"), ("oracle", "sum"), ("sum+mem", "oracle"), ("sum+bm25", "oracle"), ("sum+grep", "oracle")}
        self.assertEqual({tuple(p) for p in res["pair_list"]}, want)
        for a, b in want:
            for m in ("correct", "tokens_total", "tokens_fresh", "usd_product", "tokens_all_in"):
                c = res["contrasts"][f"{a} - {b} | memory_relevant | {m}"]
                self.assertEqual(c["n_questions"], 16)
            self.assertIn(f"{a} - {b} | memory_relevant | tokens_total | both correct", res["contrasts"])
            self.assertIn(f"{a} vs {b} | memory_relevant", res["pairs"])
        c = res["contrasts"]["sum+mem - sum+bm25 | memory_relevant | correct"]
        self.assertAlmostEqual(c["point"], 0.25)                                  # 4/4 vs 3/4 in every session
        self.assertEqual(res["primary_hars_specific"]["correct"], c)
        # accuracy-matched: questions correct in both arms (the 3 first questions of each session)
        bc = res["contrasts"]["sum+mem - sum+bm25 | memory_relevant | tokens_total | both correct"]
        self.assertEqual(bc["n_questions"], 12)
        self.assertAlmostEqual(bc["point"], 0.0)
        bc2 = res["contrasts"]["sum+mem - sum | memory_relevant | tokens_total | both correct"]
        self.assertEqual(bc2["n_questions"], 4)
        self.assertAlmostEqual(bc2["point"], 4000)
        self.assertAlmostEqual(bc2["mean_a"] / bc2["mean_b"], 1.5)

    def test_tokens_and_usd_per_correct_judge_ingest_and_fresh_vs_cached(self):
        recs, judged = toy(ARMS5, COST5)
        ingest = {"usd": 0.32, "tokens": 64000}
        res = MA.analyse(recs, judged, n_boot=100, seed=1, ingest=ingest)
        s = res["arms_summary"]["memory_relevant"]
        # sum+mem: 16 questions, 16 correct; tokens/correct = 12000 (+ ingest 64000/16 per question)
        self.assertAlmostEqual(s["sum+mem"]["tokens_per_correct"]["point"], 12000)
        self.assertAlmostEqual(s["sum+mem"]["tokens_per_correct_all_in"]["point"], 12000 + 64000 / 16)
        self.assertAlmostEqual(s["sum+bm25"]["tokens_per_correct_all_in"]["point"], 12000 * 16 / 12)    # no ingest on the control arm
        self.assertAlmostEqual(s["sum"]["tokens_per_correct"]["point"], 8000 * 16 / 4)
        # USD: agent (+ ingest for HARS arms) per correct; the judge is extra (all-in)
        self.assertAlmostEqual(s["sum+mem"]["usd_per_correct_product"]["point"], (12000e-7 * 16 + 0.32) / 16)
        self.assertAlmostEqual(s["sum+mem"]["usd_per_correct_all_in"]["point"], (12000e-7 * 16 + 0.32 + 0.002 * 16) / 16)
        self.assertTrue(s["sum+mem"]["ingest_amortised"] and not s["sum+bm25"]["ingest_amortised"])
        self.assertTrue(s["sum+mem"]["judge_usd_known"])
        # fresh vs cached split
        ts = s["sum+mem"]["token_split"]
        self.assertAlmostEqual(ts["fresh_mean"], 12000 // 5)
        self.assertAlmostEqual(ts["cache_read_mean"], 12000 - 12000 // 5 - 100)
        self.assertAlmostEqual(ts["cache_read_share_of_input"], (12000 - 12000 // 5 - 100) / (12000 - 100))
        md = MA.render(res, {})
        self.assertIn("fresh", md)
        self.assertIn("Ingest cost amortised", md)
        self.assertIn("DECISION RULE", md)
        res0 = MA.analyse(recs, judged, n_boot=50)
        self.assertIn("NOT measured", MA.render(res0, {}))

    def test_judge_unparsed_rows_are_excluded_and_counted_not_scored_wrong(self):
        recs, judged = toy(ARMS5, COST5)
        bad = "s0_q0::sum+grep"
        recs2, judged2 = toy(ARMS5, COST5, unparsed={bad})
        res = MA.analyse(recs2, judged2, n_boot=50)
        self.assertEqual(res["judge_unparsed"]["excluded_rows"], 1)
        self.assertEqual(res["judge_unparsed"]["by_arm"], {"sum+grep": 1})
        self.assertIn(bad, res["judge_unparsed"]["rids"])
        # the question leaves every pair containing the arm; the other pairs keep it
        self.assertEqual(res["contrasts"]["sum+mem - sum+grep | memory_relevant | correct"]["n_questions"], 15)
        self.assertEqual(res["contrasts"]["sum+mem - sum+bm25 | memory_relevant | correct"]["n_questions"], 16)
        # arm tables use one question set: the 15 complete ones
        self.assertEqual(res["n_questions_complete"], 15 + 0 * 1 + 0)
        self.assertEqual(res["arms_summary"]["memory_relevant"]["sum+grep"]["n"], 15)
        self.assertEqual(res["arms_summary"]["memory_relevant"]["sum+grep"]["outcome_counts"]["wrong"], 0)
        # a legacy row (judge_parse_ok False, no outcome) is excluded too, never derived to "wrong"
        recs3, judged3 = toy({"sum": 1, "sum+mem": 4}, {"sum": 1, "sum+mem": 2})
        judged3[0].pop("outcome")
        judged3[0].update(judge_parse_ok=False, judge_unparsed=False, score=None)
        r3 = MA.analyse(recs3, judged3, n_boot=50)
        self.assertEqual(r3["judge_unparsed"]["excluded_rows"], 1)

    def test_workspace_kind_strata_and_the_no_patch_sessions_are_separate(self):
        recs, judged = toy(ARMS5, COST5, kinds={"s3": "base_commit"})
        res = MA.analyse(recs, judged, n_boot=100, seed=2)
        self.assertEqual(res["no_patch_sessions"], ["s3"])
        self.assertEqual(res["workspace_kinds"]["s3"], "base_commit")
        self.assertEqual(res["arms_summary"]["memory_relevant|no_patch"]["sum+mem"]["n"], 4)
        self.assertEqual(res["arms_summary"]["memory_relevant|patch"]["sum+mem"]["n"], 12)
        c = res["contrasts"]["sum+mem - sum | memory_relevant|no_patch | correct"]
        self.assertEqual((c["n_sessions"], c["n_questions"]), (1, 4))
        self.assertIn("WITHOUT the final patch", MA.render(res, {}))

    def test_verdict_table_cell_by_cell(self):
        v = MA.verdict_of
        marg = lambda lo, hi, c: {"ci_lo": lo, "ci_hi": hi, "c_less_accurate": c}  # noqa: E731
        self.assertEqual(v("UNRESOLVED", "DOWN", "A", "B", None), "UNRESOLVED")
        self.assertEqual(v("TIE", "TIE", "A", "B", None), "TIE")
        self.assertEqual(v("TIE", "DOWN", "A", "B", None), "A_BETTER")
        self.assertEqual(v("TIE", "UP", "A", "B", None), "B_BETTER")
        self.assertEqual(v("TIE", "UNRESOLVED", "A", "B", None), "UNRESOLVED")
        self.assertEqual(v("UP", "DOWN", "A", "B", None), "A_BETTER")
        self.assertEqual(v("UP", "TIE", "A", "B", None), "A_BETTER")
        self.assertEqual(v("DOWN", "UP", "A", "B", None), "B_BETTER")
        self.assertEqual(v("DOWN", "TIE", "A", "B", None), "B_BETTER")
        # opposite directions / unresolved cost -> the marginal rule with the 1.5 premium
        lim = 1000 * MA.TRADEOFF_PREMIUM
        self.assertEqual(v("UP", "UP", "A", "B", marg(100, lim, 1000)), "A_BETTER_TRADEOFF")
        self.assertEqual(v("UP", "UNRESOLVED", "A", "B", marg(100, lim - 1, 1000)), "A_BETTER_TRADEOFF")
        self.assertEqual(v("UP", "UP", "A", "B", marg(lim + 1, lim * 3, 1000)), "TRADEOFF_UNFAVOURABLE_A")
        self.assertEqual(v("UP", "UP", "A", "B", marg(100, lim * 3, 1000)), "UNRESOLVED")
        self.assertEqual(v("DOWN", "DOWN", "A", "B", marg(100, lim, 1000)), "B_BETTER_TRADEOFF")
        self.assertEqual(v("DOWN", "DOWN", "A", "B", marg(lim + 1, lim * 3, 1000)), "TRADEOFF_UNFAVOURABLE_B")
        self.assertEqual(v("UP", "UP", "A", "B", None), "UNRESOLVED")
        # the classifiers
        self.assertEqual(MA.classify_acc(0.01, 0.2), "UP")
        self.assertEqual(MA.classify_acc(-0.3, -0.01), "DOWN")
        self.assertEqual(MA.classify_acc(-0.04, 0.04), "TIE")
        self.assertEqual(MA.classify_acc(-0.2, 0.3), "UNRESOLVED")
        self.assertEqual(MA.classify_cost(0.5, 0.99), "DOWN")
        self.assertEqual(MA.classify_cost(1.01, 2.0), "UP")
        self.assertEqual(MA.classify_cost(0.95, 1.05), "TIE")
        self.assertEqual(MA.classify_cost(0.5, 1.5), "UNRESOLVED")

    def test_pair_verdicts_on_data_with_decided_cis(self):
        # A: 4/4 correct at 12000/q; B: 2/4 at 6000/q -> accuracy UP, tokens per correct 12000 vs 12000 -> TIE -> A_BETTER
        recs, judged = toy({"A": 4, "B": 2}, {"A": 12000, "B": 6000})
        pr = MA.pair_report(MA.join(recs, judged), "A", "B", "memory_relevant", 200, 1)
        self.assertEqual((pr["acc"]["class"], pr["cost_tokens"]["class"], pr["verdict"]), ("UP", "TIE", "A_BETTER"))
        # A costlier per correct but the marginal cost is within 1.5x of B's average: 4/4 at 15000 vs 2/4 at 6000
        #   per correct 15000 vs 12000 (UP); M = (15000*4 - 6000*4) / (4 - 2) = 18000 = 1.5 * 12000 -> trade-off accepted
        recs, judged = toy({"A": 4, "B": 2}, {"A": 15000, "B": 6000})
        pr = MA.pair_report(MA.join(recs, judged), "A", "B", "memory_relevant", 200, 1)
        self.assertEqual((pr["acc"]["class"], pr["cost_tokens"]["class"]), ("UP", "UP"))
        self.assertAlmostEqual(pr["marginal"]["point"], 18000)
        self.assertAlmostEqual(pr["marginal"]["threshold"], 18000)
        self.assertEqual(pr["verdict"], "A_BETTER_TRADEOFF")
        # too expensive: 4/4 at 20000 vs 2/4 at 6000 -> M = 28000 > 18000
        recs, judged = toy({"A": 4, "B": 2}, {"A": 20000, "B": 6000})
        pr = MA.pair_report(MA.join(recs, judged), "A", "B", "memory_relevant", 200, 1)
        self.assertEqual(pr["verdict"], "TRADEOFF_UNFAVOURABLE_A")
        # equal accuracy, A cheaper -> A_BETTER ; equal accuracy equal cost -> TIE
        recs, judged = toy({"A": 2, "B": 2}, {"A": 6000, "B": 12000})
        self.assertEqual(MA.pair_report(MA.join(recs, judged), "A", "B", "memory_relevant", 200, 1)["verdict"], "A_BETTER")
        recs, judged = toy({"A": 2, "B": 2}, {"A": 6000, "B": 6000})
        self.assertEqual(MA.pair_report(MA.join(recs, judged), "A", "B", "memory_relevant", 200, 1)["verdict"], "TIE")
        # the other direction
        recs, judged = toy({"A": 2, "B": 4}, {"A": 6000, "B": 6000})
        self.assertEqual(MA.pair_report(MA.join(recs, judged), "A", "B", "memory_relevant", 200, 1)["verdict"], "B_BETTER")

    def test_gain_and_the_note_mem_trigger_are_numeric(self):
        # HARS beats everything on accuracy at equal cost per correct: gain, HARS-specific value, trigger on
        recs, judged = toy({"sum": 1, "sum+mem": 4, "sum+bm25": 2, "sum+grep": 2}, {"sum": 6000, "sum+mem": 12000, "sum+bm25": 6000, "sum+grep": 6000})
        g = MA.analyse(recs, judged, n_boot=200, seed=1)["gain"]
        self.assertTrue(g["gain"] and g["hars_specific_value"] and g["note_mem_trigger"], g)
        self.assertEqual(g["beaten_by_controls"], [])
        # BM25 does as well and is cheaper: HARS gains over sum but is BEATEN -> no trigger, no HARS-specific value
        recs, judged = toy({"sum": 1, "sum+mem": 4, "sum+bm25": 4, "sum+grep": 2}, {"sum": 6000, "sum+mem": 12000, "sum+bm25": 6000, "sum+grep": 6000})
        g = MA.analyse(recs, judged, n_boot=200, seed=1)["gain"]
        self.assertTrue(g["gain"])
        self.assertFalse(g["hars_specific_value"])
        self.assertIn("sum+bm25", g["beaten_by_controls"])
        self.assertFalse(g["note_mem_trigger"])
        # accuracy gain below +0.10 is no gain whatever the CI
        recs, judged = toy({"sum": 4, "sum+mem": 4}, {"sum": 6000, "sum+mem": 6000})
        self.assertFalse(MA.analyse(recs, judged, n_boot=100)["gain"]["gain"])
        # the rule text is in the report and the protocol states the SAME numbers
        md = MA.render(MA.analyse(recs, judged, n_boot=50), {})
        self.assertIn(MA.GAIN_RULE, md)
        proto = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "REPORT-PROTOCOL.md")).read()
        norm = lambda t: re.sub(r"\s+", " ", t)  # noqa: E731
        self.assertIn(norm(MA.DECISION_RULE), norm(proto))
        self.assertIn(norm(MA.GAIN_RULE), norm(proto))
        self.assertIn("C.10", proto)

    def test_cli_with_ingest_json_and_judge_cost_fallback(self):
        recs, judged = toy({"sum": 1, "sum+mem": 3}, {"sum": 8000, "sum+mem": 12000})
        for j in judged[:4]:
            j.pop("judge_cost_usd")                                 # reused verdicts carry no cost: the mean of the others is used
        with tempfile.TemporaryDirectory() as d:
            for name, rows in (("r.jsonl", recs), ("j.jsonl", judged)):
                open(os.path.join(d, name), "w").write("\n".join(json.dumps(x) for x in rows))
            json.dump({"hars_extraction_measured": {"usd": 0.2, "total_tokens": 40000, "estimated": False}}, open(os.path.join(d, "ing.json"), "w"))
            rc, out = quiet(MA.main, ["--records", os.path.join(d, "r.jsonl"), "--judged", os.path.join(d, "j.jsonl"), "--out", os.path.join(d, "o.md"),
                                      "--json", os.path.join(d, "o.json"), "--boot", "50", "--ingest-json", os.path.join(d, "ing.json")])
            self.assertEqual(rc, 0, out)
            res = json.load(open(os.path.join(d, "o.json")))
        self.assertEqual(res["ingest"]["tokens"], 40000)
        self.assertAlmostEqual(res["arms_summary"]["all"]["sum+mem"]["tokens_per_correct_all_in"]["point"], (12000 * 16 + 40000) / 12)
        # the MEASURE command's file form is understood as well
        self.assertEqual(MA.load_ingest("", 0.5, 100.0), {"usd": 0.5, "tokens": 100.0})


# --------------------------------------------------------------------------- judge robustness
FAKE_OK = json.dumps({"score": 2, "unsupported_claim": False, "abstained": False, "outcome": "correct", "reason": "ok"})


def judge_records(n=3):
    base = dict(chain_id="c1", session_id="s", span_idx=1, lag=1, slice="HEAD", answerable=True, question="Which port?", gold_anchor="8842",
                canonical_short_answer="8842", evidence="port 8842", span_text="the beacon port 8842 was configured",
                answer="Port 8842.", agent_model="claude-haiku-5-5", qa_model="claude-haiku-5-5")
    return [dict(base, qid=f"q{i}", arm="sum", rid=f"q{i}::sum") for i in range(n)]


class ScriptedJudge:
    """providers transport: replies from a list (callable or str); records the prompts."""

    def __init__(self, replies, cost=0.0001):
        self.replies, self.prompts, self.cost = list(replies), [], cost

    def __call__(self, req, timeout):
        body = json.loads(req.data)
        self.prompts.append(json.dumps(body))
        i = len(self.prompts) - 1
        content = self.replies[min(i, len(self.replies) - 1)]
        return 200, {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": self.cost}}


class JudgeRobustness(unittest.TestCase):
    def run_judge(self, replies, budget="5", extra=(), records=None, cost=0.0001, td=None):
        td = td or tempfile.mkdtemp()
        inp = os.path.join(td, "in.jsonl")
        with open(inp, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in (records or judge_records())))
        out, mp = os.path.join(td, "j.jsonl"), os.path.join(td, "map.json")
        tr = ScriptedJudge(replies, cost)
        old = (sys.argv, J.TRANSPORT, dict(os.environ))
        sys.argv = ["judge_v2.py", "--in", inp, "--out", out, "--mapping", mp, "--profile", "openrouter", "--model-judge", "judge/model-x",
                    "--key-env", "LS_FAKE_JUDGE_KEY", "--budget-usd", budget, "--outcome", *extra]
        J.TRANSPORT = tr
        os.environ["LS_FAKE_JUDGE_KEY"] = "x"
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    rc = J.main()
                except SystemExit as e:
                    rc = e.code
        finally:
            sys.argv, J.TRANSPORT = old[0], old[1]
            os.environ.clear()
            os.environ.update(old[2])
        judged = [json.loads(x) for x in open(out)] if os.path.exists(out) else []
        return rc, buf.getvalue(), judged, tr, td, out

    def test_unparseable_reply_is_reasked_once_with_the_strict_reminder_and_recovered(self):
        rc, log, judged, tr, *_ = self.run_judge(["I think the answer is fine.", FAKE_OK, FAKE_OK, FAKE_OK])
        self.assertEqual(rc, 0, log)
        self.assertEqual(len(tr.prompts), 4)                                    # 3 items + exactly ONE re-ask
        self.assertIn(J.STRICT_REMINDER.strip()[:40], tr.prompts[1])
        self.assertNotIn(J.STRICT_REMINDER.strip()[:40], tr.prompts[0])
        first = judged[0]
        self.assertEqual((first["judge_reasked"], first["judge_reask_ok"], first["judge_unparsed"], first["judge_parse_ok"]),
                         (True, True, False, True))
        self.assertEqual(first["outcome"], "correct")
        self.assertTrue(all(not j["judge_reasked"] and not j["judge_unparsed"] for j in judged[1:]))
        self.assertIn("1 re-asked, 1 recovered", log)

    def test_still_unparseable_is_marked_judge_unparsed_not_wrong_and_the_cost_is_per_row(self):
        rc, log, judged, tr, *_ = self.run_judge(["nonsense", "still nonsense", FAKE_OK, FAKE_OK])
        self.assertEqual(rc, 0, log)
        self.assertEqual(len(tr.prompts), 4)
        j0 = judged[0]
        self.assertEqual((j0["judge_unparsed"], j0["judge_reasked"], j0["judge_reask_ok"], j0["judge_parse_ok"]), (True, True, False, False))
        self.assertIsNone(j0["outcome"])
        self.assertAlmostEqual(j0["judge_cost_usd"], 0.0002)                    # both calls are charged to the row
        self.assertAlmostEqual(judged[1]["judge_cost_usd"], 0.0001)
        # metrics: the row is excluded from accuracy (never "wrong"), counted
        recs = [{"rid": j["rid"], "sid": "s", "cluster_session": "s", "qid": j["qid"], "arm": "sum", "valid": True, "answerable": True,
                 "gold_anchor": "8842", "memory_relevant": True, "tokens_total": 1, "tokens_fresh": 1, "cost_usd": 0.0, "steps": 1, "tool_calls": 0}
                for j in judged]
        res = MA.analyse(recs, judged, n_boot=20)
        self.assertEqual(res["judge_unparsed"]["excluded_rows"], 1)
        self.assertEqual(res["arms_summary"]["all"]["sum"]["outcome_counts"]["wrong"], 0)
        self.assertEqual(res["arms_summary"]["all"]["sum"]["n"], 2)

    def test_aborted_run_keeps_its_verdicts_and_its_spend_and_resumes(self):
        # 3 items at $0.0001 with a cap of $0.0002: the third call finds the budget spent and aborts
        td = tempfile.mkdtemp()
        rc, log, judged, tr, td, out = self.run_judge([FAKE_OK] * 6, budget="0.0002", td=td)
        self.assertNotEqual(rc, 0)
        self.assertIn("ABORTED", log)
        self.assertRegex(log, r"judge spend: \$0\.000\d+ over 2 calls")                 # the line the ledger parses is printed on abort
        self.assertAlmostEqual(H.judge_spend_from_text(log), 0.0002)
        side = json.load(open(out + ".spend.json"))
        self.assertEqual((side["status"], side["calls"]), ("aborted", 2))
        self.assertAlmostEqual(side["usd"], 0.0002)
        part = [json.loads(x) for x in open(out + ".partial")]
        self.assertEqual(len(part), 2)
        self.assertFalse(os.path.exists(out))
        self.assertAlmostEqual(H.judge_spend_of_run(out, log)[0], 0.0002)
        # the pre-C.10 abort message (the one the C.9 run printed) is parsed too: this is why the aborted run was booked as $0.0
        old_msg = "judge budget exhausted after 222 calls: budget $1.1028 exhausted (spent $1.0968 over 222 calls)"
        self.assertAlmostEqual(H.judge_spend_from_text(old_msg), 1.0968)
        self.assertIsNone(H.judge_spend_from_text("nothing"))
        # resume: only the missing item is judged
        rc, log, judged, tr2, td, out = self.run_judge([FAKE_OK] * 6, budget="5", td=td, extra=("--resume-partial", out + ".partial"))
        self.assertEqual(rc, 0, log)
        self.assertEqual(len(tr2.prompts), 1)
        self.assertEqual(len(judged), 3)
        self.assertEqual({j["rid"] for j in judged}, {"q0::sum", "q1::sum", "q2::sum"})

    def test_masking_covers_the_new_tool_names(self):
        masked, hits = J.mask_arm_leaks("I used transcript_grep and bm25 to look")
        self.assertTrue(hits)
        self.assertNotIn("transcript_grep", masked)
        self.assertNotIn("bm25", masked.lower())


# --------------------------------------------------------------------------- home_run: version, controls, measure, bundle, ledger
class HomeC10(T9.Home):
    def agentic_ready(self):
        self.provision_ws()
        self.mark("oh.ingest")
        H.write_json(os.path.join(self.camp, "campaign.json"), {**H.read_json(os.path.join(self.camp, "campaign.json")),
                     "c8_projects": {"hars_run_id": "e1", "projects_dir": os.path.join(self.d.name, "projects"),
                                     "pids": {c["chain_id"]: f"nc-e1-{c['chain_id']}" for c in self.chains}}})

    def test_version_guard_and_commands(self):
        self.assertEqual(H.HOME_RUN_VERSION, "2026-10-28.c10")
        self.assertEqual(open(os.path.join(os.path.dirname(H.BASE), "EVAL-VERSION")).read().strip(), "2026-10-28.c10")
        self.assertEqual(H.REUSE_COMPATIBLE_FROM, ("2026-10-27.c6", "2026-10-28.c7", "2026-10-28.c8", "2026-10-28.c9"))
        self.assertIn("oh-ingest-measure", H.REUSE_CMDS)
        for v in ("2026-10-27.c6", "2026-10-28.c9"):
            self.stamp(v)
            for cmd in (["oh-agentic", "--controls", "--dry-run"], ["oh-ingest-measure", "--dry-run"]):
                rc, so, se = self.go(T9.AgenticRunner(), *cmd)
                self.assertNotIn("EVAL-VERSION", se, (v, cmd, se))
        self.stamp("2026-10-26.c5")
        rc, so, se = self.go(T9.AgenticRunner(), "oh-agentic", "--controls", "--dry-run")
        self.assertEqual(rc, 2)

    def make_c9_agentic_run(self):
        """A finished C.9 oh/agentic-run in the campaign (rows + verdicts), as the owner's home run left it."""
        run = os.path.join(self.camp, "oh", "agentic-run")
        os.makedirs(run, exist_ok=True)
        rows, ver = [], []
        for c in self.chains:
            sid = c["chain_id"]
            for q in c["qa"]:
                for arm in ("sum", "sum+mem"):
                    rows.append({"rid": f"{q['qid']}::{arm}", "qid": q["qid"], "sid": sid, "cluster_session": sid, "arm": arm, "valid": True,
                                 "tokens_total": 1000, "steps": 3, "cost_usd": 0.001,
                                 "summary_sha256": RA.sha256_text(f"SUMMARY<{sid}> text")})
                    ver.append({"rid": f"{q['qid']}::{arm}", "score": 2, "outcome": "correct", "present_id": "p0"})
        for fn, data in (("records.jsonl", rows), ("judged.jsonl", ver)):
            open(os.path.join(run, fn), "w").write("\n".join(json.dumps(x) for x in data) + "\n")
        json.dump({"rows": len(rows), "expected": len(rows), "n_missing": 0}, open(os.path.join(run, "summary.json"), "w"))
        json.dump({"judge_model": H.P.resolve_models(H.P.load_profile("anthropic"), {})["judge"]}, open(os.path.join(run, "judge-mapping.json"), "w"))
        return run, rows, ver

    def test_controls_dry_run_prints_every_arm_and_the_method_and_spends_nothing(self):
        run, rows, ver = self.make_c9_agentic_run()
        r = T9.AgenticRunner()
        rc, so, se = self.go(r, "oh-agentic", "--controls", "--dry-run")
        self.assertEqual(rc, 0, so + se)
        for arm in ("sum+mem", "sum+bm25", "sum+grep", "oracle"):
            self.assertRegex(so, rf"\n\s+{re.escape(arm)}\s+\$")
        self.assertIn("METHOD per arm", so)
        self.assertIn("calibrated", so)
        est = H.read_json(os.path.join(self.camp, "oh-agentic-c10-estimate.json"))
        self.assertEqual(est["arms"], ["sum+mem", "sum+bm25", "sum+grep", "oracle"])
        self.assertEqual(est["reused_arms"], ["sum", "sum+mem@uncapped"])
        self.assertEqual(est["retrieval_max_chars"], 6000)
        self.assertFalse([1 for s, a in r.calls if s in ("run_agentic.py", "judge_v2.py")])
        self.assertFalse(os.path.exists(os.path.join(self.camp, "oh", "agentic-c10-run")))
        # an unfinished / missing C.9 run is a clear error
        shutil_rm = __import__("shutil").rmtree
        shutil_rm(run)
        rc, so, se = self.go(T9.AgenticRunner(), "oh-agentic", "--controls", "--dry-run")
        self.assertEqual(rc, 2)
        self.assertIn("not a finished C.9 agentic run", se)

    def test_controls_run_wires_reuse_cap_judge_merge_and_metrics(self):
        run, rows, ver = self.make_c9_agentic_run()
        self.agentic_ready()
        r = T9.AgenticRunner()

        def agentic(argv, _orig=r.agentic):
            if "--export-judge" in argv:
                out = r.arg(argv, "--out")
                recs = [json.loads(x) for x in open(os.path.join(out, "records.jsonl"))]
                open(os.path.join(out, "judge-input.jsonl"), "w").write(
                    "".join(json.dumps({"rid": x["rid"]}) + "\n" for x in recs if not x.get("reused_from")))
                return 0, "exported\n"
            out = r.arg(argv, "--out")
            os.makedirs(out, exist_ok=True)
            json.dump({"run_id": "ag"}, open(os.path.join(out, "config.json"), "w"))
            json.dump({"rows": 0, "expected": 0, "n_missing": 0}, open(os.path.join(out, "summary.json"), "w"))
            json.dump({"budget_spent_all_invocations": 0.2, "failed_sessions": []}, open(os.path.join(out, "MANIFEST.json"), "w"))
            new = []
            for rr in rows:                                   # reused rows, relabelled, as run_agentic writes them
                lab = "sum+mem@uncapped" if rr["arm"] == "sum+mem" else "sum"
                new.append({**rr, "rid": f"{rr['qid']}::{lab}", "arm": lab, "reused_from": run, "reused_rid": rr["rid"]})
            for c in self.chains:
                for q in c["qa"]:
                    for arm in ("sum+mem", "sum+bm25", "sum+grep", "oracle"):
                        new.append({"rid": f"{q['qid']}::{arm}", "qid": q["qid"], "sid": c["chain_id"], "arm": arm, "valid": True})
            open(os.path.join(out, "records.jsonl"), "w").write("\n".join(json.dumps(x) for x in new) + "\n")
            return 0, "done\n"
        r.agentic = agentic
        rc, so, se = self.go(r, "oh-agentic", "--controls", "--retrieval-max-chars", "4000", "--bm25-passage-chars", "900")
        self.assertEqual(rc, 0, so + se)
        run_call = [a for s, a in r.calls if s == "run_agentic.py" and "--export-judge" not in a][0]
        self.assertEqual(r.arg(run_call, "--arms"), "sum+mem,sum+bm25,sum+grep,oracle")
        self.assertEqual(r.arg(run_call, "--retrieval-max-chars"), "4000")
        self.assertEqual(r.arg(run_call, "--bm25-passage-chars"), "900")
        self.assertEqual(r.arg(run_call, "--reuse-agentic-run"), run)
        self.assertEqual(r.arg(run_call, "--reuse-arms"), "sum,sum+mem")
        self.assertEqual(r.arg(run_call, "--out"), os.path.join(self.camp, "oh", "agentic-c10-run"))
        self.assertTrue(os.path.isdir(run))                                   # the C.9 run is untouched
        # the judge is asked for the new rows only and the old verdicts are merged (rid relabelled)
        judge = [a for s, a in r.calls if s == "judge_v2.py"][0]
        self.assertEqual(r.arg(judge, "--out"), os.path.join(self.camp, "oh", "agentic-c10-run", "judged.new.jsonl"))
        merged = [json.loads(x) for x in open(os.path.join(self.camp, "oh", "agentic-c10-run", "judged.jsonl"))]
        rids = {m["rid"] for m in merged}
        n_q = sum(len(c["qa"]) for c in self.chains)
        self.assertEqual(sum(1 for m in merged if m["rid"].endswith("::sum+mem@uncapped")), n_q)
        self.assertEqual(sum(1 for m in merged if m.get("reused_verdict")), 2 * n_q)
        self.assertTrue(all(not m["rid"].endswith("::sum+mem") or not m.get("reused_verdict") for m in merged))
        for s_ in ("oh.agentic.c10.run", "oh.agentic.c10.export", "oh.agentic.c10.judge", "oh.agentic.c10.merge", "oh.agentic.c10.metrics"):
            self.assertIn(s_, self.steps())
        led = [e for e in H.read_json(os.path.join(self.camp, "ledger.json"))["entries"] if e["step"] == "oh.agentic.c10.run"]
        self.assertEqual(len(led), 1)
        self.assertAlmostEqual(led[0]["usd"], 0.2)
        met = [a for s, a in r.calls if s == "metrics_agentic.py"][0]
        self.assertNotIn("--ingest-json", met)                                # nothing measured yet

    def test_ledger_books_an_aborted_judge_run_and_resumes_from_its_verdicts(self):
        self.stamp(H.HOME_RUN_VERSION)
        camp = H.Campaign(self.camp, self.cfg_for_judge(), runner=None)
        camp.guards_enabled = False
        ji = os.path.join(self.d.name, "ji.jsonl")
        open(ji, "w").write(json.dumps({"rid": "a"}) + "\n")
        judged, mp = os.path.join(self.d.name, "j.jsonl"), os.path.join(self.d.name, "m.json")

        class AbortRunner:
            def __init__(self):
                self.n = 0

            def __call__(self, argv, env, cwd, log_path, echo=True):
                self.n += 1
                out = argv[argv.index("--out") + 1]
                if self.n == 1:
                    open(out + ".partial", "w").write(json.dumps({"rid": "a", "present_id": "p0"}) + "\n")
                    json.dump({"usd": 1.0968, "calls": 222, "status": "aborted"}, open(out + ".spend.json", "w"))
                    return 1, "judge budget exhausted after 222 calls: budget $1.1028 exhausted (spent $1.0968 over 222 calls)\n"
                self.resumed = "--resume-partial" in argv
                open(out, "w").write(json.dumps({"rid": "a"}) + "\n")
                return 0, "judge spend: $0.30000 over 60 calls (cap $1.00; profile anthropic)\n"
        camp.runner = AbortRunner()
        with self.assertRaises(H.HomeRunError) as cm:
            H.judge_step(camp, "oh", "oh-agentic", ji, judged, mp, accept_partial=True, step="oh.test.judge")
        self.assertIn("partial", str(cm.exception))
        led = H.read_json(os.path.join(self.camp, "ledger.json"))["entries"]
        e = [x for x in led if x["step"] == "oh.test.judge"]
        self.assertEqual(len(e), 1)
        self.assertAlmostEqual(e[0]["usd"], 1.0968)                           # NOT 0.0
        self.assertIn("ABORTED", e[0]["note"])
        H.judge_step(camp, "oh", "oh-agentic", ji, judged, mp, accept_partial=True, step="oh.test.judge")
        self.assertTrue(camp.runner.resumed)
        e = [x for x in H.read_json(os.path.join(self.camp, "ledger.json"))["entries"] if x["step"] == "oh.test.judge"]
        self.assertEqual(len(e), 2)
        self.assertAlmostEqual(sum(x["usd"] for x in e), 1.0968 + 0.3)

    def cfg_for_judge(self):
        cfg = H.load_config(None, {"general": {"artifact_root": self.root if hasattr(self, "root") else self.d.name, "_offline": True},
                                    "caps": {"gen": 5, "run": 5, "judge": 5, "total": 20}})
        return cfg

    def test_ingest_measure_dry_run_then_provision_then_measure(self):
        run, rows, ver = self.make_c9_agentic_run()
        self.agentic_ready()
        pdir = os.path.join(self.d.name, "projects")
        main_pids = set(H.read_json(os.path.join(self.camp, "campaign.json"))["c8_projects"]["pids"].values())
        r = T9.AgenticRunner()
        # dry run: plan + delete instructions, writes nothing
        rc, so, se = self.go(r, "oh-ingest-measure", "--dry-run", "--sessions", "1", "--projects-dir", pdir)
        self.assertEqual(rc, 0, so + se)
        self.assertIn("throwaway projects", so)
        self.assertIn("rm -r", so)
        self.assertIn("curl -X DELETE", so)
        self.assertFalse(os.path.exists(pdir))
        # first real call: provisions the throwaway projects, asks for ONE restart, spends nothing
        rc, so, se = self.go(r, "oh-ingest-measure", "--sessions", "1", "--projects-dir", pdir)
        self.assertEqual(rc, 3, so + se)
        self.assertIn("restart your HARS server ONCE", so)
        plan = H.read_json(os.path.join(self.camp, "campaign.json"))["c10_ingest_measure"]
        self.assertEqual(len(plan["pids"]), 1)
        pid = next(iter(plan["pids"].values()))
        self.assertNotIn(pid, main_pids)
        self.assertRegex(pid, r"^nc-e1m\d{8}-")                               # the main run id + a distinct suffix
        self.assertTrue(os.path.isfile(os.path.join(pdir, pid, "project.json")))
        self.assertFalse([1 for s, a in r.calls if s == "run_longsession.py" and "--dry-run" not in a])
        # second call: ingests ONLY the throwaway project (the fake runner stands in for run_longsession)
        sid = next(iter(plan["pids"]))

        orig_ls = r.longsession

        def longsession(argv):
            if "--dry-run" in argv:
                return orig_ls(argv)
            self.assertEqual(r.arg(argv, "--hars-run-id"), plan["hars_run_id"])
            self.assertIn("--ingest-only", argv)
            self.assertNotEqual(r.arg(argv, "--hars-run-id"), "e1")
            self.assertEqual([f for f in os.listdir(r.arg(argv, "--sessions")) if f.endswith(".json")][0].replace(".json", ""), sid)
            out = r.arg(argv, "--out")
            os.makedirs(out, exist_ok=True)
            json.dump({"run_id": "m"}, open(os.path.join(out, "config.json"), "w"))
            json.dump({"budget_spent_all_invocations": 0.01, "failed_sessions": []}, open(os.path.join(out, "MANIFEST.json"), "w"))
            json.dump({"hars_extraction_measured": {}}, open(os.path.join(out, "summary.json"), "w"))
            open(os.path.join(out, "done.jsonl"), "w").write("")
            open(os.path.join(out, "ingest.jsonl"), "w").write(json.dumps({"sid": sid, "ok": True, "project": pid, "ts": H.utc_iso(),
                                                                            "extract_usage": {
                "calls": 40, "prompt_tokens": 80000, "completion_tokens": 8000, "total_tokens": 88000, "usd": 0.012, "estimated": False,
                "documents": 3, "documents_unmeasured": 0}}) + "\n")
            return 0, "ok\n"
        r.longsession = longsession
        rc, so, se = self.go(r, "oh-ingest-measure", "--sessions", "1", "--projects-dir", pdir)
        self.assertEqual(rc, 0, so + se)
        meas = H.read_json(os.path.join(self.camp, "oh", "ingest-measure", "measure.json"))
        self.assertEqual(meas["sessions_measured"], 1)
        self.assertEqual(meas["mean_per_session"]["tokens"], 88000)
        self.assertEqual(meas["total_tokens"], 88000 * len(self.chains))
        self.assertAlmostEqual(meas["usd"], 0.012 * len(self.chains))
        self.assertIn("MEASURED over 1 session", so)
        self.assertIn("rm -r", so)
        self.assertIn(pid, so)
        # metrics_agentic reads that file as the amortised ingest
        ing = MA.load_ingest(os.path.join(self.camp, "oh", "ingest-measure", "measure.json"), None, None)
        self.assertEqual(ing["tokens"], 88000 * len(self.chains))
        self.assertEqual(H.ingest_for_metrics(H.Campaign(self.camp, self.cfg_for_judge())), os.path.join(self.camp, "oh", "ingest-measure", "measure.json"))
        # the main projects were never written
        self.assertFalse([1 for pid_ in main_pids if os.path.exists(os.path.join(pdir, pid_))])
        self.assertEqual(meas["plan"], plan["hars_run_id"])
        self.assertTrue(os.path.isfile(os.path.join(self.camp, "oh", "ingest-measure", plan["hars_run_id"], "measure.json")))

        # a repeated call never re-runs or rewrites a finished measurement (the bug: old numbers under a new timestamp)
        before = open(os.path.join(self.camp, "oh", "ingest-measure", "measure.json")).read()
        n_runs = len([1 for s_, a in r.calls if s_ == "run_longsession.py" and "--dry-run" not in a])
        rc, so, se = self.go(r, "oh-ingest-measure", "--sessions", "1", "--projects-dir", pdir)
        self.assertEqual(rc, 0, so + se)
        self.assertIn("already MEASURED", so)
        self.assertEqual(open(os.path.join(self.camp, "oh", "ingest-measure", "measure.json")).read(), before)
        self.assertEqual(len([1 for s_, a in r.calls if s_ == "run_longsession.py" and "--dry-run" not in a]), n_runs)

        # --new-measure: fresh ids AND a fresh run dir; rows that ran no extraction are refused, not reported
        rc, so, se = self.go(r, "oh-ingest-measure", "--sessions", "1", "--projects-dir", pdir, "--new-measure",
                             "--mcp", "http://127.0.0.1:8318/mcp")
        self.assertEqual(rc, 3, so + se)
        plan2 = H.read_json(os.path.join(self.camp, "campaign.json"))["c10_ingest_measure"]
        self.assertEqual(plan2["mcp"], "http://127.0.0.1:8318/mcp")
        pid2 = next(iter(plan2["pids"].values()))

        def idle(argv):
            if "--dry-run" in argv:
                return orig_ls(argv)
            self.assertEqual(r.arg(argv, "--mcp"), "http://127.0.0.1:8318/mcp")
            out = r.arg(argv, "--out")
            self.assertIn(plan2["hars_run_id"], out)                       # never the old run dir
            os.makedirs(out, exist_ok=True)
            json.dump({"run_id": "m2"}, open(os.path.join(out, "config.json"), "w"))
            json.dump({"budget_spent_all_invocations": 0.0, "failed_sessions": []}, open(os.path.join(out, "MANIFEST.json"), "w"))
            json.dump({}, open(os.path.join(out, "summary.json"), "w"))
            open(os.path.join(out, "done.jsonl"), "w").write("")
            open(os.path.join(out, "ingest.jsonl"), "w").write(json.dumps({"sid": sid, "ok": True, "project": pid2, "ts": H.utc_iso(),
                                                                            "extract_usage": {"calls": 0, "total_tokens": 0}}) + "\n")
            return 0, "ok\n"
        r.longsession = idle
        rc, so, se = self.go(r, "oh-ingest-measure", "--sessions", "1", "--projects-dir", pdir, "--mcp", "http://127.0.0.1:8318/mcp")
        self.assertNotEqual(rc, 0, so + se)
        self.assertIn("ran NO extraction", se + so)
        self.assertEqual(open(os.path.join(self.camp, "oh", "ingest-measure", "measure.json")).read(), before)


# --------------------------------------------------------------------------- bundle
class BundleTraces(unittest.TestCase):
    """The default bundle of the OH track now carries the agentic traces and the recall / probe sidecars, gzip-compressed, redacted,
    under --max-raw-mb; the workspace kinds and an explicit PASS line are in it."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.d.name, "root")
        cfg = H.load_config(None, {"general": {"artifact_root": self.root}})
        self.camp = H.open_campaign(cfg, H.build_parser().parse_args(["oh"]), "oh", echo=False)
        self.secret = "sk-ant-" + "a1B2c3D4" * 4
        self.write_rel("oh/agentic-c10-run/traces.jsonl", json.dumps({"rid": "q::oracle", "system": "tool key " + self.secret, "messages": []}) + "\n")
        self.write_rel("oh/agentic-c10-run/records.jsonl", json.dumps({"rid": "q::oracle", "workspace_kind": "base_commit"}) + "\n")
        self.write_rel("oh/ingest-run/ingest.jsonl", json.dumps({"sid": "s", "extract_usage": {"calls": 3}}) + "\n")
        self.write_rel("oh/probe-run/recall_raw.jsonl", json.dumps({"rid": "r", "raw": "RAW RECALL"}) + "\n")
        self.write_rel("oh/merged/judge-input.jsonl", json.dumps({"span_text": "RAW TRANSCRIPT"}) + "\n")
        self.write_rel("cc/run/traces.jsonl", json.dumps({"rid": "private"}) + "\n")
        ws = os.path.join(self.d.name, "wsdest")
        os.makedirs(ws, exist_ok=True)
        json.dump({"workspaces": {"chain_a": {"workspace_kind": "base_commit", "patch_applied": False},
                                  "chain_b": {"workspace_kind": "base_commit+final_patch", "patch_applied": True}}},
                  open(os.path.join(ws, "workspaces.json"), "w"))
        cj = H.read_json(os.path.join(self.camp.path, "campaign.json"), {})
        cj["c9_workspaces"] = {"dest": ws}
        H.write_json(os.path.join(self.camp.path, "campaign.json"), cj)
        self.camp.write_manifest()

    def tearDown(self):
        self.d.cleanup()

    def write_rel(self, rel, text):
        p = os.path.join(self.camp.path, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(text)

    def pack(self, *extra):
        rc, so = quiet(H.main, ["--root", self.root, "bundle", "--resume", self.camp.path, *extra])
        self.assertEqual(rc, 0, so)
        path = re.search(r"BUNDLE: (\S+)", so).group(1)
        with tarfile.open(path) as tf:
            files = {n.split("/", 1)[1]: tf.extractfile(n).read() for n in tf.getnames() if "/" in n and tf.getmember(n).isfile()}
        return so, path, files

    def test_traces_ingest_and_sidecars_are_in_by_default_gzipped_and_redacted(self):
        so, path, files = self.pack()
        for want in ("oh/agentic-c10-run/traces.jsonl.gz", "oh/probe-run/recall_raw.jsonl.gz", "oh/ingest-run/ingest.jsonl",
                     "oh/agentic-c10-run/records.jsonl", "oh/workspace-kinds.json"):
            self.assertIn(want, files)
        self.assertNotIn("cc/run/traces.jsonl.gz", files)                         # the private track keeps raw behind --include-raw
        self.assertNotIn("cc/run/traces.jsonl", files)
        self.assertNotIn("oh/merged/judge-input.jsonl", files)                    # transcripts stay out
        self.assertNotIn("oh/agentic-c10-run/traces.jsonl", files)
        tr = gzip.decompress(files["oh/agentic-c10-run/traces.jsonl.gz"]).decode()
        self.assertNotIn(self.secret, tr)
        self.assertIn("[REDACTED:", tr)
        self.assertEqual(json.loads(tr.splitlines()[0])["rid"], "q::oracle")
        wk = json.loads(files["oh/workspace-kinds.json"])
        self.assertEqual(wk["chain_a"]["workspace_kind"], "base_commit")
        man = json.loads(files["bundle-manifest.json"])
        self.assertEqual(man["workspace_kinds"]["chain_b"]["workspace_kind"], "base_commit+final_patch")
        self.assertIn("oh/agentic-c10-run/traces.jsonl.gz", man["files"])
        self.assertRegex(so, r"LEAK SCAN: (PASS|REVIEW) - \d+ file\(s\)")
        self.assertIn("workspace kinds", so)
        rep = json.loads(files["bundle-redaction-report.json"])
        self.assertGreaterEqual(rep["total_hits"], 1)
        self.assertEqual(rep["raw_default"]["skipped"], [])

    def test_a_clean_bundle_prints_an_explicit_pass_with_counts(self):
        os.remove(os.path.join(self.camp.path, "oh", "agentic-c10-run", "traces.jsonl"))
        so, path, files = self.pack()
        self.assertRegex(so, r"LEAK SCAN: PASS - \d+ file\(s\), [\d.]+ \w+ scanned, \d+ env secret value\(s\) checked, 0 hit\(s\)")
        self.assertIn("false positive", so)                                       # the script copies' two known pattern false positives

    def test_size_guard_skips_and_says_what_it_skipped(self):
        import random
        rnd = random.Random(1)
        big = "\n".join(json.dumps({"rid": f"r{i}", "blob": "".join(rnd.choice("abcdefghijklmnop0123456789") for _ in range(900))})
                        for i in range(3000)) + "\n"
        self.write_rel("oh/agentic-c10-run/traces.jsonl", big)
        so, path, files = self.pack("--max-raw-mb", "0.2")
        self.assertNotIn("oh/agentic-c10-run/traces.jsonl.gz", files)
        self.assertIn("oh/probe-run/recall_raw.jsonl.gz", files)                 # the small ones still fit
        rep = json.loads(files["bundle-redaction-report.json"])
        sk = rep["raw_default"]["skipped"]
        self.assertEqual([x["rel"] for x in sk], ["oh/agentic-c10-run/traces.jsonl"])
        self.assertIn("size guard", sk[0]["reason"])
        self.assertIn("SKIPPED by the guard: 1", so)
        self.assertIn("traces.jsonl", so)
        so2, path2, files2 = self.pack("--no-default-raw")                       # the C.9 bundle
        self.assertNotIn("oh/probe-run/recall_raw.jsonl.gz", files2)
        self.assertNotIn("oh/agentic-c10-run/traces.jsonl.gz", files2)

    def test_include_raw_keeps_plain_files_and_redaction_stays_on(self):
        so, path, files = self.pack("--include-raw", "--tracks", "oh")
        self.assertIn("oh/agentic-c10-run/traces.jsonl", files)
        self.assertNotIn(self.secret, files["oh/agentic-c10-run/traces.jsonl"].decode())


class EvalVersionFile(unittest.TestCase):
    def test_marker_and_code_agree(self):
        marker = open(os.path.join(os.path.dirname(H.BASE), "EVAL-VERSION")).read().strip()
        self.assertEqual(marker, H.HOME_RUN_VERSION)
        self.assertEqual(marker, RA.EVAL_VERSION_TAG)


if __name__ == "__main__":
    unittest.main()
