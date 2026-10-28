"""Offline tests: long-session judging path (exporter -> judge_v2 -> metrics_v2) and Amendment C.2."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import ls_fixtures as F
import export_longsession_judge as X
import judge_v2
import longsession as L
import metrics_v2 as M
import qtypes as QT
import run_longsession as R

COMMON = ["--window", "14000", "--keep-tail", "3000", "--provider", "mock", "--agent-model", "mock",
          "--fake-hars", "--no-preflight", "--piece-chars", "3000", "--pieces-per-call", "4",
          "--arms", "baseline,nc-chunks,nc-graph,nc-note-only"]
FAKE_VERDICT = json.dumps({"score": 2, "unsupported_claim": False, "abstained": False, "reason": "ok"})


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def beam_like(sid: str, real: bool = False) -> dict:
    s = F.make_session(sid, real=real)
    s["source"] = "beam"
    for q in s["questions"]:
        if q["qtype"] == "multihop":
            q["source_qtype"] = "multi_session_reasoning"
    s["questions"][0]["source_qtype"] = "information_extraction"
    s["questions"][1].update(qtype="knowledge_update", source_qtype="knowledge_update")
    s["questions"][2].update(qtype="temporal_reasoning", source_qtype="temporal_reasoning")
    return s


def judge_run(records_in: str, out: str, mapping: str, extra=()):
    judge_v2.TRANSPORT = lambda req, timeout: (200, {
        "choices": [{"message": {"content": FAKE_VERDICT}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0001}})
    old = sys.argv
    os.environ["LS_FAKE_JUDGE_KEY"] = "x"
    sys.argv = ["judge_v2.py", "--in", records_in, "--out", out, "--mapping", mapping, "--profile",
                "openrouter", "--model-judge", "judge/model-x", "--key-env", "LS_FAKE_JUDGE_KEY",
                "--budget-usd", "5", *extra]
    try:
        return quiet(judge_v2.main)
    finally:
        sys.argv = old
        judge_v2.TRANSPORT = None


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.TemporaryDirectory()
        cls.sd = os.path.join(cls.d.name, "in")
        for k in range(3):
            F.write_session(cls.sd, beam_like(f"beam-1M-{k}"))
        cls.rundir = os.path.join(cls.d.name, "run")
        rc = quiet(R.main, ["--sessions", cls.sd, "--out", cls.rundir, "--checkpoints", "2,end",
                            "--mem-budgets", "1536,512"] + COMMON)
        assert rc == 0
        cls.rec = os.path.join(cls.rundir, "records.jsonl")
        cls.ji = os.path.join(cls.rundir, "judge-input.jsonl")
        assert quiet(X.main, ["--records", cls.rec, "--sessions", cls.sd, "--out", cls.ji]) == 0
        cls.rows = [json.loads(l) for l in open(cls.ji)]

    @classmethod
    def tearDownClass(cls):
        cls.d.cleanup()

    def test_export_has_every_field_judge_v2_needs(self):
        need = {"rid", "qid", "chain_id", "arm", "question", "canonical_short_answer", "answer",
                "gold_anchor", "answerable", "span_text", "evidence", "rubric", "visible_context",
                "qtype", "source_qtype", "base_qid", "checkpoint", "lag", "mem_budget"}
        for r in self.rows:
            self.assertFalse(need - set(r), need - set(r))
            self.assertNotIn("memory_text", r)
            self.assertNotIn("injected", r)
        self.assertEqual(len({r["rid"] for r in self.rows}), len(self.rows))
        mh = next(r for r in self.rows if r["base_qid"].endswith("_m1"))
        self.assertIsInstance(mh["evidence"], list)
        self.assertEqual(len(mh["evidence"]), 2)                       # both hops get a window
        self.assertIn("JOB0010", mh["span_text"])
        self.assertIn("JOB0030", mh["span_text"])
        # the visible context holds the memory + the verbatim tail but not the question
        self.assertTrue(all(r["visible_context"] for r in self.rows))
        self.assertTrue(all(r["question"] not in r["visible_context"] for r in self.rows))

    def test_rubric_reaches_the_judge_prompt_only_when_present(self):
        with_rub = next(r for r in self.rows if r["rubric"])
        ex, _ = judge_v2.window_for(with_rub["span_text"], with_rub["evidence"])
        self.assertIn("<rubric>", judge_v2.build_prompt(with_rub, ex))
        self.assertIn(with_rub["rubric"][0][:20], judge_v2.build_prompt(with_rub, ex))
        no_rub = dict(with_rub, rubric=[])
        self.assertNotIn("rubric", judge_v2.build_prompt(no_rub, ex).lower())
        # chain items (no rubric key at all) keep the historical prompt byte-for-byte
        legacy = {k: v for k, v in no_rub.items() if k != "rubric"}
        self.assertEqual(judge_v2.build_prompt(legacy, ex), judge_v2._build_prompt(legacy, ex))

    def test_prompts_are_blind(self):
        for r in self.rows:
            ex, _ = judge_v2.window_for(r["span_text"], r["evidence"])
            p = judge_v2.build_prompt(r, ex)
            for forbidden in ("nc-graph", "nc-chunks", "nc-note-only", "baseline", "HARS", "[memory",
                              "Knowledge Graph", r["rid"], r["session_id"]):
                self.assertNotIn(forbidden, p, (r["rid"], forbidden))

    def test_multi_evidence_windows(self):
        span = "A" * 3000 + "FIRSTFACT goes here" + "B" * 3000 + "SECONDFACT goes here" + "C" * 3000
        w, how = judge_v2.window_for(span, ["FIRSTFACT goes here", "SECONDFACT goes here"])
        self.assertIn("FIRSTFACT", w)
        self.assertIn("SECONDFACT", w)
        self.assertEqual(how, "evidence_exact")
        w1, how1 = judge_v2.window_for(span, "FIRSTFACT goes here")          # str path unchanged
        self.assertEqual((w1, how1), judge_v2.window_around(span, "FIRSTFACT goes here"))
        _, weak = judge_v2.window_for(span, ["FIRSTFACT goes here", "nothing like this at all anywhere"])
        self.assertEqual(weak, "evidence_not_found")

    def test_oh_family_is_l1_l3_holm_m3(self):
        """Amendment C.6-3a: --ls-family oh -> fixed family L1-L3 (m = 3), no L4/L5,
        xspan_locator kept as a descriptive stratum."""
        judged, mapping = os.path.join(self.rundir, "judged.jsonl"), os.path.join(self.rundir, "map.json")
        self.assertEqual(judge_run(self.ji, judged, mapping), 0)
        md_path, js_path = os.path.join(self.rundir, "oh.md"), os.path.join(self.rundir, "oh.json")
        old = sys.argv
        sys.argv = ["metrics_v2.py", "--records", self.rec, "--judge", judged, "--mapping", mapping,
                    "--judge-input", self.ji, "--out", md_path, "--json", js_path, "--boot", "300",
                    "--ls-family", "oh"]
        try:
            self.assertEqual(quiet(M.main), 0)
        finally:
            sys.argv = old
        sec = json.load(open(js_path))["secondary"]
        self.assertEqual(sec["family"], list(M.LS_SECONDARY_IDS[:3]))
        self.assertIn("over the fixed family of 3", sec["correction"])
        self.assertNotIn("L4_msr_vs_baseline", sec["tests"])
        self.assertNotIn("L5_msr_graph_vs_chunks", sec["tests"])
        for d in M.DESCRIPTIVE_IDS:
            self.assertIn(d, sec["descriptive"])

    def test_full_chain_export_judge_metrics(self):
        judged, mapping = os.path.join(self.rundir, "judged.jsonl"), os.path.join(self.rundir, "map.json")
        self.assertEqual(judge_run(self.ji, judged, mapping), 0)
        mp = json.load(open(mapping))
        self.assertEqual(mp["judge_model"], "judge/model-x")
        self.assertTrue(all("arm" not in v or v["arm"] for v in mp["map"].values()))
        self.assertEqual(len([l for l in open(judged)]), len(self.rows))
        md_path, js_path = os.path.join(self.rundir, "m.md"), os.path.join(self.rundir, "m.json")
        old = sys.argv
        sys.argv = ["metrics_v2.py", "--records", self.rec, "--judge", judged, "--mapping", mapping,
                    "--judge-input", self.ji, "--out", md_path, "--json", js_path, "--boot", "300"]
        try:
            self.assertEqual(quiet(M.main), 0)
        finally:
            sys.argv = old
        d = json.load(open(js_path))
        md = open(md_path).read()
        self.assertEqual(d["meta"]["track"], "longsession")
        self.assertEqual(d["meta"]["sessions"], 3)                     # sid, not _cN-stripped chain
        self.assertEqual(d["secondary"]["family"], list(M.LS_SECONDARY_IDS))
        self.assertIn("L4_msr_vs_baseline", d["secondary"]["tests"])
        self.assertEqual(d["secondary"]["metric"], "score")                 # C.2: rubric-graded score
        self.assertEqual(list(d["secondary"]["tests"]["L2_ni_1024"]["legs"]), ["score"])
        self.assertEqual(d["secondary"]["tests"]["L1_graph_vs_chunks"]["metric"], "score")
        # genuine multihop = multi_session_reasoning; the stored 'multihop' label is NOT the stratum
        ls = d["longsession"]
        self.assertIn("multi_session_reasoning", ls["strata"])
        self.assertNotIn("multihop", ls["strata"])
        self.assertIn("knowledge_update", ls["strata"])
        self.assertGreater(ls["checkpoints"]["questions_asked_at_multiple_checkpoints"], 0)
        self.assertIn("end", ls["position"])
        self.assertIn("mid", ls["position"])
        self.assertTrue(ls["lag"])
        self.assertIn("Long-session track", md)
        self.assertIn("genuine multi-hop (L4/L5)", md)
        # the judge score reached the records
        s = d["by_budget"]["1536"]["summary"]["nc-graph"]
        self.assertAlmostEqual(s["score"], 2.0)

    def test_include_sid_selects_the_analysis_set(self):
        out = os.path.join(self.rundir, "sel.json")
        old = sys.argv
        sys.argv = ["metrics_v2.py", "--records", self.rec, "--out", os.path.join(self.rundir, "sel.md"),
                    "--json", out, "--boot", "100", "--include-sid", r"^beam-1M-[01]$"]
        try:
            self.assertEqual(quiet(M.main), 0)
        finally:
            sys.argv = old
        d = json.load(open(out))
        self.assertEqual(d["meta"]["sessions"], 2)
        self.assertIn("include=", d["meta"]["sid_filter"])


    def test_end_only_drops_mid_checkpoint_rows(self):
        out = os.path.join(self.rundir, "end.json")
        old = sys.argv
        sys.argv = ["metrics_v2.py", "--records", self.rec, "--out", os.path.join(self.rundir, "end.md"),
                    "--json", out, "--boot", "100", "--end-only"]
        try:
            self.assertEqual(quiet(M.main), 0)
        finally:
            sys.argv = old
        d = json.load(open(out))
        self.assertTrue(d["meta"]["end_only"])
        self.assertGreater(d["meta"]["mid_rows_dropped"], 0)
        self.assertEqual(d["longsession"]["checkpoints"]["questions_asked_at_multiple_checkpoints"], 0)
        self.assertNotIn("mid", d["longsession"]["position"])


class RealCompactionOrig(unittest.TestCase):
    def test_orig_arm_is_kept_descriptive_and_not_required(self):
        with tempfile.TemporaryDirectory() as d:
            sd = os.path.join(d, "in")
            for k in range(2):
                s = F.make_session(f"cc-{k}", real=True)
                s["source"] = "claude_code"
                F.write_session(sd, s)
            run = os.path.join(d, "run")
            rc = quiet(R.main, ["--sessions", sd, "--out", run, "--use-real-compactions", "--checkpoints", "all",
                                "--keep-tail", "1500", "--window", "14000", "--mem-budgets", "512",
                                "--provider", "mock", "--agent-model", "mock", "--fake-hars", "--no-preflight",
                                "--piece-chars", "3000", "--arms", "baseline,nc-chunks,nc-graph,nc-note-only,orig"])
            self.assertEqual(rc, 0)
            rows = [json.loads(l) for l in open(os.path.join(run, "records.jsonl"))]
            self.assertTrue(any(r["arm"] == "orig" for r in rows))
            kept, rep = M.validate_rows([dict(r) for r in rows])
            self.assertEqual(rep["rows_kept"], len(rows))
            self.assertEqual(rep["side_arms_kept"], ["orig"])
            self.assertEqual(rep["required_arms"], list(M.REQUIRED_ARMS))
            # an arm set WITHOUT orig at some qids is still complete
            js = os.path.join(d, "m.json")
            old = sys.argv
            sys.argv = ["metrics_v2.py", "--records", os.path.join(run, "records.jsonl"), "--out",
                        os.path.join(d, "m.md"), "--json", js, "--boot", "100"]
            try:
                self.assertEqual(quiet(M.main), 0)
            finally:
                sys.argv = old
            data = json.load(open(js))
            self.assertIn("orig", data["summary"])
            self.assertIn("nc-graph-orig", data["by_budget"]["512"]["deltas"])
            self.assertEqual(data["meta"]["sessions"], 2)


class QtypeRelabel(unittest.TestCase):
    def test_canon_qtype(self):
        self.assertEqual(QT.canon_qtype("multihop"), "xspan_locator")
        self.assertEqual(QT.canon_qtype("Multi-Hop"), "xspan_locator")
        self.assertEqual(QT.canon_qtype("multihop", "multi_session_reasoning"), "multi_session_reasoning")
        self.assertEqual(QT.canon_qtype("single", "information_extraction"), "information_extraction")
        self.assertEqual(QT.canon_qtype("single"), "single")
        self.assertEqual(QT.canon_qtype("multihop", legacy=True), "multihop")
        self.assertIsNone(QT.canon_qtype(None))

    def test_normalize_relabels_at_load_without_touching_data(self):
        raw = {"qid": "q", "arm": "baseline", "chain_id": "c_c0", "qtype": "multihop"}
        n = M.normalize(dict(raw))
        self.assertEqual((n["qtype"], n["qtype_stored"]), ("xspan_locator", "multihop"))
        self.assertEqual(raw["qtype"], "multihop")
        M.QTYPE_LEGACY = True
        try:
            self.assertEqual(M.normalize(dict(raw))["qtype"], "multihop")
        finally:
            M.QTYPE_LEGACY = False

    def test_split_qid_and_cluster(self):
        self.assertEqual(M.split_qid("beam-1M-3_q1@c12"), ("beam-1M-3_q1", 12))
        self.assertEqual(M.split_qid("beam-1M-3_q1", {"checkpoint": 9}), ("beam-1M-3_q1", 9))
        self.assertEqual(M.split_qid("x@c3", {"base_qid": "root"}), ("root", 3))
        n = M.normalize({"qid": "a@c2", "arm": "baseline", "chain_id": "sess_c1", "base_qid": "a",
                         "cluster_session": "sess_c1"})
        self.assertEqual(n["cluster_session"], "sess_c1")                 # no _cN stripping
        self.assertEqual((n["base_qid"], n["checkpoint"], n["is_end"]), ("a", 2, False))
        self.assertEqual(M.normalize({"qid": "a", "arm": "baseline", "chain_id": "sess_c1"})["cluster_session"], "sess")
        self.assertEqual([M.lag_bucket(x) for x in (None, 0, 1, 3, 5, 9, 40)],
                         ["n/a", "0", "1", "2-3", "4-7", "8-15", "16+"])

    def test_chain_family_is_s1_s3_with_descriptive_xspan(self):
        import test_amendment_c as TA
        rc_, d, md = TA.run_metrics(TA.synth_rows(10))
        sec = d["secondary"]
        self.assertEqual(sec["family"], list(M.SECONDARY_IDS))
        self.assertEqual(len(sec["family"]), 3)
        self.assertEqual(sec["descriptive"], list(M.DESCRIPTIVE_IDS))
        t = sec["tests"]["S4d_xspan_locator_vs_baseline"]
        self.assertTrue(t["decision"].startswith("descriptive"))
        self.assertEqual(t["n_pairs"], 10 * 2)             # the stored 'multihop' items = xspan_locator
        self.assertIsNone(t["p_holm"])
        self.assertIn("xspan_locator", md)
        # --legacy-qtypes: the stratum under its stored name is empty for the descriptive tests
        rc_, d2, _ = TA.run_metrics(TA.synth_rows(10), extra=("--legacy-qtypes",))
        self.assertEqual(d2["secondary"]["tests"]["S4d_xspan_locator_vs_baseline"]["n_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
