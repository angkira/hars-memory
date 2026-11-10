"""Offline tests: run_longsession plan / dry-run / records / resume / validity (mock LLM, fake HARS)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
import urllib.request
from unittest import mock

import ls_fixtures as F
import longsession as L
import run_longsession as R

HAVE_RC = L.rc() is not None and all(hasattr(L.rc(), n) for n in (
    "pack_injection", "recall_failure_reasons", "recall_degraded", "graph_failure_reasons", "graph_summary",
    "split_packed_sections", "status_gate_problems", "preflight_verdict"))
needs_rc = unittest.skipUnless(HAVE_RC, "run_chainreplay.py not importable (mid-edit?)")


def call(argv):
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        rc = R.main(argv)
    return rc, buf.getvalue(), err.getvalue()


def rows_of(out):
    return [json.loads(l) for l in open(os.path.join(out, "records.jsonl"))]


class TrimTests(unittest.TestCase):
    NOTE = ("GOAL\nShip the thing.\nKEY FIGURES & IDENTIFIERS\n" + "\n".join(f"- id{i} = {i * 111}" for i in range(30))
            + "\nDECISIONS & RATIONALE\n" + "\n".join(f"- decided {i} because reason number {i}." for i in range(20))
            + "\nACTIVE STATE\n- running job JOB1 now.")

    def test_trim_cuts_decisions_first_and_keeps_active_state(self):
        cap = L.count_tokens(self.NOTE) - 60
        out, info = R.trim_note(self.NOTE, cap)
        self.assertLessEqual(info["after"], cap)
        self.assertIn("ACTIVE STATE\n- running job JOB1 now.", out)
        self.assertEqual(info["trimmed_sections"][0], "DECISIONS & RATIONALE")
        self.assertTrue(info["trim_applied"])
        self.assertIn("KEY FIGURES & IDENTIFIERS", out)

    def test_under_cap_is_untouched(self):
        out, info = R.trim_note(self.NOTE, 10 ** 6)
        self.assertEqual(out, self.NOTE.strip())
        self.assertFalse(info["trim_applied"] or info["boundary_repaired"])

    def test_active_state_alone_over_cap_is_reported_not_cut(self):
        out, info = R.trim_note("GOAL\ng.\nACTIVE STATE\n" + "- word. " * 200, 20)
        self.assertTrue(info["over_cap"])
        self.assertIn("ACTIVE STATE", out)

    def test_boundary_repair_cuts_dangling_tail(self):
        note = "GOAL\ng.\nACTIVE STATE\n- first fact is done.\n- second fact is cut mi"
        out, info = R.trim_note(note, 10 ** 6, force_repair=True)
        self.assertTrue(info["boundary_repaired"])
        self.assertTrue(out.endswith("done."))
        out2, info2 = R.trim_note("GOAL\ng.\nACTIVE STATE\n- ok.\n3.", 10 ** 6, force_repair=True)
        self.assertTrue(out2.endswith("ok."))

    def test_unparsed_text_is_hard_cut_by_lines(self):
        out, info = R.trim_note("\n".join(f"line {i} of free text" for i in range(100)), 30)
        self.assertTrue(info["unparsed"])
        self.assertLessEqual(L.count_tokens(out), 30)


class PlanTests(unittest.TestCase):
    def test_checkpoint_specs(self):
        rc = R.resolve_checkpoints
        self.assertEqual(rc("end", 7), [7])
        self.assertEqual(rc("all", 3), [1, 2, 3])
        self.assertEqual(rc("every:3", 10), [3, 6, 9, 10])
        self.assertEqual(rc("2,end,99", 5), [2, 5])
        self.assertEqual(rc("99", 5), [5])
        self.assertEqual(rc("end", 0), [])

    def test_select_questions_filters_and_is_deterministic(self):
        s = F.make_session()
        bs = L.compute_boundaries(L.message_tokens(s["messages"]), window=14000, keep_tail=3000, reserve=500)
        kw = dict(qtypes=set(), lags=set(), max_q=0, seed=1)
        end = R.select_questions(s["questions"], bs, len(bs), **kw)
        self.assertEqual(len(end), len(s["questions"]))
        early = R.select_questions(s["questions"], bs, 1, **kw)
        self.assertLess(len(early), len(end))                        # later evidence not yet eligible
        self.assertTrue(all(L.question_eligible(q, bs[0]) for q, _ in early))
        capped = R.select_questions(s["questions"], bs, len(bs), **{**kw, "max_q": 3})
        again = R.select_questions(s["questions"], bs, len(bs), **{**kw, "max_q": 3})
        self.assertEqual([q["qid"] for q, _ in capped], [q["qid"] for q, _ in again])
        self.assertEqual(len(capped), 3)
        self.assertEqual({q["qtype"] for q, _ in R.select_questions(
            s["questions"], bs, len(bs), **{**kw, "qtypes": {"multihop"}})}, {"multihop"})
        lagged = R.select_questions(s["questions"], bs, len(bs), **{**kw, "lags": {len(bs)}})
        self.assertTrue(all(lag in (len(bs), None) for _, lag in lagged))

    def test_recall_cap_formula(self):
        self.assertEqual(R.recall_cap(1536, 417)["recall_cap"], 1024 if 1536 - 417 > 1024 else 1536 - 417)
        self.assertEqual(R.recall_cap(1536, 1400)["recall_cap"], 256)
        self.assertTrue(R.recall_cap(1536, 1400)["recall_cap_clamped_low"])
        self.assertEqual(R.recall_cap(512, 100)["recall_cap"], 341 if 512 - 100 > 341 else 412)
        self.assertEqual(L.note_cap_formula(1536), 640)

    @needs_rc
    def test_recall_cap_matches_run_chainreplay(self):
        rcm = L.rc()
        if not hasattr(rcm, "recall_cap_for"):
            self.skipTest("recall_cap_for gone")
        import types
        for B in (512, 1024, 1536, 4096):
            for note in (0, 100, 417, 640, 1500, 5000):
                ns = types.SimpleNamespace(mem_budget=B, recall_cap_min=L.recall_floor_formula(B),
                                           recall_cap_max=L.recall_ceil_formula(B))
                self.assertEqual(rcm.recall_cap_for(note, ns)["recall_cap"], R.recall_cap(B, note)["recall_cap"], (B, note))

    def test_rubric_proxy(self):
        r = ["LLM response should state: 97% success rate", "LLM response should mention: pytest coverage"]
        self.assertEqual(R.rubric_hit_frac("It is a 97% success rate.", r), 0.5)
        self.assertEqual(R.rubric_hit_frac("97% success rate and pytest coverage", r), 1.0)
        self.assertIsNone(R.rubric_hit_frac("x", []))


COMMON = ["--window", "14000", "--keep-tail", "3000", "--mem-budgets", "512", "--checkpoints", "end"]


class DryRunTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.sd = os.path.join(self.d.name, "in")
        F.write_session(self.sd, F.make_session("syn-1", n=160))   # enough text for >=5 boundaries under any tokenizer
        F.write_session(self.sd, F.make_session("syn-2", n=90))

    def tearDown(self):
        self.d.cleanup()

    def test_dry_run_is_deterministic_offline_and_creates_nothing(self):
        out = os.path.join(self.d.name, "never")
        argv = ["--sessions", self.sd, "--dry-run", "--out", out] + COMMON
        with mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("network used")):
            a = call(argv)
            b = call(argv)
            aj = call(argv + ["--json"])
            bj = call(argv + ["--json"])
        self.assertEqual(a, b)
        self.assertEqual(aj, bj)
        self.assertEqual(a[0], 0)
        self.assertFalse(os.path.exists(out))
        plan = json.loads(aj[1])
        self.assertEqual([s["sid"] for s in plan["sessions"]], ["syn-1", "syn-2"])
        s1 = plan["sessions"][0]
        self.assertGreaterEqual(s1["boundaries"], 5)
        self.assertGreater(s1["total_usd"], 0)
        self.assertEqual(set(s1["per_budget"]["512"]["usd"]), set(R.ALL_ARMS))
        self.assertIn("GRAND TOTAL", a[1])

    def test_dry_run_cost_scales_with_price_and_arms(self):
        # cache off: the price scaling is exact (cache-priced answers are covered in test_prompt_cache)
        base = ["--sessions", self.sd, "--dry-run", "--json", "--limit", "1", "--prompt-cache", "off"] + COMMON
        c1 = json.loads(call(base)[1])["grand_total_usd"]
        c2 = json.loads(call(base + ["--price-in", "0.2", "--price-out", "1.0", "--extractor-price-in", "0.2",
                                     "--extractor-price-out", "1.0"])[1])["grand_total_usd"]
        self.assertAlmostEqual(c2, 2 * c1, places=4)
        only_base = json.loads(call(base + ["--arms", "baseline"])[1])["grand_total_usd"]
        self.assertLess(only_base, c1)
        per_b = json.loads(call(base + ["--hars-per-budget", "--mem-budgets", "512,1024"])[1])
        shared = json.loads(call(base + ["--mem-budgets", "512,1024"])[1])
        self.assertGreater(per_b["grand_total_usd"], shared["grand_total_usd"])

    def test_dry_run_real_compactions_adds_orig_arm(self):
        sd = os.path.join(self.d.name, "real")
        F.write_session(sd, F.make_session("syn-r", real=True))
        r = call(["--sessions", sd, "--dry-run", "--json", "--use-real-compactions", "--keep-tail", "2000",
                  "--window", "14000", "--mem-budgets", "512", "--checkpoints", "all"])
        self.assertEqual(r[0], 0, r[2])
        p = json.loads(r[1])
        s = p["sessions"][0]
        self.assertIn("orig", p["config"]["arms"])
        self.assertEqual(s["boundaries"], 3)
        self.assertEqual(s["real_summaries"], 3)
        self.assertGreater(s["per_budget"]["512"]["asks"]["orig"], 0)

    def test_bad_configs_exit(self):
        with self.assertRaises(SystemExit):
            call(["--sessions", self.sd, "--dry-run", "--arms", "orig"] + COMMON)       # orig needs real
        with self.assertRaises(SystemExit):
            call(["--sessions", self.sd, "--dry-run", "--window", "2000", "--keep-tail", "3000"])
        with self.assertRaises(SystemExit):
            call(["--sessions", self.sd, "--dry-run", "--arms", "nope"])


class NoRunChainreplayTests(unittest.TestCase):
    """run_chainreplay.py is edited by another agent: dry-run must not depend on it."""

    def test_dry_run_and_live_path_without_it(self):
        with tempfile.TemporaryDirectory() as d:
            sd = os.path.join(d, "in")
            F.write_session(sd, F.make_session("syn-1"))
            broken = {"mod": None, "err": "SyntaxError: simulated mid-edit", "tried": True}
            with mock.patch.dict(L._RC, broken):
                rc, so, se = call(["--sessions", sd, "--dry-run"] + COMMON)
                self.assertEqual(rc, 0, se)
                self.assertIn("GRAND TOTAL", so)
                with self.assertRaises(SystemExit) as cm:
                    call(["--sessions", sd, "--out", os.path.join(d, "o"), "--provider", "mock", "--agent-model", "mock",
                          "--no-preflight"] + COMMON)
                self.assertIn("not importable", str(cm.exception))
                # baseline-only needs none of its recall pieces ... except the ask prompt fallback
                rc, so, se = call(["--sessions", sd, "--out", os.path.join(d, "o2"), "--provider", "mock",
                                   "--agent-model", "mock", "--fake-hars", "--no-preflight", "--arms", "baseline",
                                   "--piece-chars", "3000"] + COMMON)
                self.assertEqual(rc, 0, so + se)
                rows = rows_of(os.path.join(d, "o2"))
                self.assertTrue(rows and {r["arm"] for r in rows} == {"baseline"})


@needs_rc
class RunTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.sd = os.path.join(self.d.name, "in")
        F.write_session(self.sd, F.make_session("syn-1"))
        self.out = os.path.join(self.d.name, "out")
        a = R.parse_args(["--sessions", self.sd] + F.RUN_ARGS)
        self.J = len(R.build_plan(F.make_session("syn-1"), a)["boundaries"])
        self.assertGreaterEqual(self.J, 4)

    def tearDown(self):
        self.d.cleanup()

    def run_main(self, extra=(), out=None, base=None):
        argv = ["--sessions", self.sd, "--out", out or self.out] + (base or F.RUN_ARGS) + list(extra)
        return call(argv)

    def test_full_run_artifacts_and_record_schema(self):
        rc, so, se = self.run_main(["--checkpoints", "2,end"])
        self.assertEqual(rc, 0, so + se)
        for name in ("samples.jsonl", "records.jsonl", "compactions.jsonl", "summary.json", "run.log",
                     "MANIFEST.json", "done.jsonl", "config.json", os.path.join("inputs", "inputs.json"),
                     os.path.join("scripts", "cmdline.txt"), os.path.join("scripts", "run_longsession.py")):
            self.assertTrue(os.path.exists(os.path.join(self.out, name)), name)
        rows = rows_of(self.out)
        self.assertEqual(len(rows), len(open(os.path.join(self.out, "samples.jsonl")).readlines()))
        need = {"run_id", "chain_id", "qid", "base_qid", "rid", "arm", "mem_budget", "b_tag", "qtype", "valid",
                "chain_invalid_reasons", "memory_tokens", "arm_note_tokens", "memory_tokens_injected", "answer",
                "latency_s", "prompt_tokens", "completion_tokens", "cost_usd", "lag", "checkpoint", "is_end",
                "anchor_hit", "gold_anchor", "canonical_short_answer", "recall_raw_chars", "recall_packed_tokens",
                "recall_seconds", "recall_error", "recall_failure_reasons", "recall_degraded", "persisted_compaction",
                "tail_tokens", "memory_text", "agent_model", "tokenizer", "span_idx"}
        for r in rows:
            self.assertFalse(need - set(r), need - set(r))
        # arms and checkpoint rows
        arms = {r["arm"] for r in rows}
        self.assertEqual(arms, {"baseline", "nc-chunks", "nc-graph", "nc-note-only"})
        self.assertEqual({r["checkpoint"] for r in rows}, {2, self.J})
        self.assertTrue(all(r["valid"] is True for r in rows))
        # unique (qid, arm, B) and rid: metrics_v2.validate_rows refuses duplicates
        keys = [(r["qid"], r["arm"], r["mem_budget"]) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len({r["rid"] for r in rows}), len(rows))
        # rid shape of export_judge_input / metrics_v2: <qid>::<arm>::b<B>
        r0 = next(r for r in rows if r["is_end"])
        self.assertEqual(r0["rid"], f"{r0['qid']}::{r0['arm']}::b{r0['mem_budget']}")
        # nc arms: injected recall > 0, total within the budget; baseline: no injection, no HARS
        for r in rows:
            self.assertLessEqual(r["memory_tokens"], r["mem_budget"], r["rid"])
            if r["arm"] in R.NC_ARMS:
                self.assertGreater(r["memory_tokens_injected"], 0)
                self.assertTrue(r["persisted_compaction"])
                self.assertGreater(r["recall_raw_chars"], 0)
            else:
                self.assertEqual(r["memory_tokens_injected"], 0)
                self.assertEqual(r["recall_raw_chars"], 0)
        graph = [r for r in rows if r["arm"] == "nc-graph"]
        self.assertTrue(all(r.get("session_graph_status") for r in graph))
        # same tail for every arm at a checkpoint
        tails = {(r["qid"].split("@")[0], r["checkpoint"]): set() for r in rows}
        for r in rows:
            tails[(r["qid"].split("@")[0], r["checkpoint"])].add(r["tail_tokens"])
        self.assertTrue(all(len(v) == 1 for v in tails.values()))
        # summary + manifest
        summ = json.load(open(os.path.join(self.out, "summary.json")))
        self.assertIn("nc-graph@B512", summ["by_arm_budget"])
        man = json.load(open(os.path.join(self.out, "MANIFEST.json")))
        self.assertEqual(man["models"]["provider"], "mock")
        self.assertIn("run_chainreplay", man)
        self.assertTrue(open(os.path.join(self.out, "scripts", "cmdline.txt")).read().strip())
        inp = json.load(open(os.path.join(self.out, "inputs", "inputs.json")))
        self.assertEqual(len(next(iter(inp["files"].values()))["sha256"]), 64)
        # compatible with metrics_v2.validate_rows (explicit arm set; nc-note-only is a 4th arm there)
        try:
            import metrics_v2 as M
        except Exception:                      # noqa: BLE001
            return
        kept, rep = M.validate_rows([dict(r) for r in rows], arms=("baseline", "nc-chunks", "nc-graph", "nc-note-only"))
        self.assertEqual(rep["rows_kept"], len(rows))

    def test_resume_skips_done_and_does_not_recompute(self):
        rc, so, se = self.run_main()
        self.assertEqual(rc, 0, so + se)
        n1 = len(rows_of(self.out))
        rc, so, se = call(["--sessions", self.sd, "--resume", self.out] + F.RUN_ARGS)
        self.assertEqual(rc, 0, so + se)
        self.assertIn("all (arm, B) done - skipped", so)
        self.assertEqual(len(rows_of(self.out)), n1)
        # drop one done marker: only that arm is redone, from the summary cache (0 summariser calls)
        lines = open(os.path.join(self.out, "done.jsonl")).read().splitlines()
        keep = [l for l in lines if json.loads(l)["arm"] != "nc-graph"]
        open(os.path.join(self.out, "done.jsonl"), "w").write("\n".join(keep) + "\n")
        recs = [l for l in open(os.path.join(self.out, "records.jsonl")) if json.loads(l)["arm"] != "nc-graph"]
        open(os.path.join(self.out, "records.jsonl"), "w").write("".join(recs))
        rc, so, se = call(["--sessions", self.sd, "--resume", self.out] + F.RUN_ARGS)
        self.assertEqual(rc, 0, so + se)
        rows = rows_of(self.out)
        self.assertEqual(len(rows), n1)
        self.assertEqual(len([r for r in rows if r["arm"] == "nc-graph"]), n1 // 4)
        cached = [json.loads(l) for l in open(os.path.join(self.out, "compactions.jsonl"))]
        self.assertTrue(any(c.get("cached") for c in cached))

    def test_resume_refuses_changed_config_and_reused_dir(self):
        self.assertEqual(self.run_main()[0], 0)
        with self.assertRaises(SystemExit):
            call(["--sessions", self.sd, "--resume", self.out] + F.RUN_ARGS + ["--keep-tail", "2500"])
        with self.assertRaises(SystemExit):
            self.run_main()                                                    # same --out, finished run
        with self.assertRaises(SystemExit):
            call(["--sessions", self.sd, "--resume", os.path.join(self.d.name, "nope")] + F.RUN_ARGS)

    def test_run_is_deterministic_for_mock(self):
        self.assertEqual(self.run_main(["--checkpoints", "end"])[0], 0)
        o2 = os.path.join(self.d.name, "out2")
        self.assertEqual(self.run_main(["--checkpoints", "end"], out=o2)[0], 0)
        strip = lambda rs: [{k: v for k, v in r.items() if k not in ("run_id", "latency_s", "recall_seconds", "session_id", "injected", "prompt_sha256")}  # noqa: E731
                            for r in rs]
        self.assertEqual(strip(rows_of(self.out)), strip(rows_of(o2)))

    def test_budget_sweep_shares_hars_and_scales_caps(self):
        rc, so, se = self.run_main(["--mem-budgets", "512,1024"], base=[a for a in F.RUN_ARGS if a not in ("512",)][:0] + [
            "--window", "14000", "--keep-tail", "3000", "--provider", "mock", "--agent-model", "mock",
            "--fake-hars", "--no-preflight", "--piece-chars", "3000", "--pieces-per-call", "4",
            "--arms", "baseline,nc-chunks,nc-graph,nc-note-only"])
        self.assertEqual(rc, 0, so + se)
        rows = rows_of(self.out)
        self.assertEqual({r["mem_budget"] for r in rows}, {512, 1024})
        comp = [json.loads(l) for l in open(os.path.join(self.out, "compactions.jsonl"))]
        hars_calls = [c for c in comp if c.get("arm") == "hars" and c.get("ok")]
        self.assertEqual(len(hars_calls), self.J)             # one ingest per boundary, not per budget
        by_b = {B: [r for r in rows if r["mem_budget"] == B and r["arm"] == "nc-chunks"] for B in (512, 1024)}
        self.assertTrue(all(r["memory_tokens"] <= 512 for r in by_b[512]))
        self.assertTrue(all(r["memory_tokens"] <= 1024 for r in by_b[1024]))
        self.assertGreater(max(r["recall_cap"] for r in by_b[1024]), max(r["recall_cap"] for r in by_b[512]))
        self.assertEqual({r["recall_cap_max"] for r in by_b[1024]}, {683})

    def test_hars_per_budget_uses_separate_sessions(self):
        rc, so, se = self.run_main(["--hars-per-budget", "--mem-budgets", "512,1024"], base=[
            "--window", "14000", "--keep-tail", "3000", "--provider", "mock", "--agent-model", "mock",
            "--fake-hars", "--no-preflight", "--piece-chars", "3000", "--pieces-per-call", "4",
            "--arms", "baseline,nc-chunks,nc-graph,nc-note-only"])
        self.assertEqual(rc, 0, so + se)
        sids = {(r["mem_budget"], r["session_id"]) for r in rows_of(self.out) if r["arm"] == "nc-chunks"}
        self.assertEqual(len({s for _, s in sids}), 2)

    def test_recall_failure_invalidates_nc_arms_only(self):
        with mock.patch.object(R.FakeHars, "recall", return_value={"context": "", "error": "boom 530"}):
            rc, so, se = self.run_main()
        self.assertEqual(rc, 0)
        rows = rows_of(self.out)
        for r in rows:
            if r["arm"] in R.NC_ARMS:
                self.assertFalse(r["valid"], r["arm"])
                self.assertTrue(r["chain_invalid_reasons"])
                self.assertEqual(r["recall_error"], "boom 530")
            else:
                self.assertTrue(r["valid"], r["arm"])
        # metrics_v2 drops the invalid cell
        import metrics_v2 as M
        kept, rep = M.validate_rows([dict(r) for r in rows], arms=("baseline", "nc-chunks", "nc-graph", "nc-note-only"))
        self.assertEqual(rep["rows_kept"], 0)

    def test_degraded_recall_invalid_unless_allowed(self):
        orig = R.FakeHars.__init__

        def degraded_init(self_):
            orig(self_)
            self_.degraded = True
        with mock.patch.object(R.FakeHars, "__init__", degraded_init):
            self.run_main()
            bad = [r for r in rows_of(self.out) if r["arm"] in R.NC_ARMS]
            self.assertTrue(bad and all(not r["valid"] and r["recall_degraded"] == "lexical_only" for r in bad))
            self.assertTrue(any("degraded" in w for r in bad for w in r["chain_invalid_reasons"]))
            out2 = os.path.join(self.d.name, "out-allow")
            self.run_main(["--allow-degraded"], out=out2)
            ok = [r for r in rows_of(out2) if r["arm"] in R.NC_ARMS]
            self.assertTrue(ok and all(r["valid"] and r["allow_degraded"] and r["recall_degraded"] for r in ok))

    def test_ingest_failure_invalidates_and_blocks_further_ingest(self):
        calls = []

        def broken(self_, sid, summary, dropped, metadata):
            calls.append(1)
            return {"ok": False, "error": "extractor dead"}
        with mock.patch.object(R.FakeHars, "compact", broken):
            self.run_main()
        rows = rows_of(self.out)
        self.assertTrue(all(not r["valid"] for r in rows if r["arm"] in R.NC_ARMS))
        self.assertTrue(all(r["valid"] for r in rows if r["arm"] in ("baseline", "nc-note-only")))
        self.assertLessEqual(len(calls), 2)                    # one retry, then ingest stops

    def test_real_compactions_and_orig_arm(self):
        sd = os.path.join(self.d.name, "real")
        F.write_session(sd, F.make_session("syn-r", real=True))
        out = os.path.join(self.d.name, "out-real")
        rc, so, se = call(["--sessions", sd, "--out", out, "--use-real-compactions", "--keep-tail", "1500",
                           "--window", "14000", "--mem-budgets", "512", "--checkpoints", "all",
                           "--provider", "mock", "--agent-model", "mock", "--fake-hars", "--no-preflight",
                           "--piece-chars", "3000"])
        self.assertEqual(rc, 0, so + se)
        rows = rows_of(out)
        orig = [r for r in rows if r["arm"] == "orig"]
        self.assertTrue(orig)
        self.assertTrue(all(r["memory_text"].startswith("GOAL\nreal summary") for r in orig))
        self.assertTrue(all(r["memory_tokens"] == L.count_tokens(r["memory_text"]) for r in orig))
        self.assertEqual({r["checkpoint"] for r in rows}, {1, 2, 3})
        comp = [json.loads(l) for l in open(os.path.join(out, "compactions.jsonl")) if '"arm": "hars"' in l and '"ok": true' in l]
        self.assertEqual(len(comp), 3)

    def test_preflight_passes_and_fails(self):
        rc, so, se = self.run_main(base=[a for a in F.RUN_ARGS if a != "--no-preflight"])
        self.assertEqual(rc, 0, so + se)
        self.assertIn("[preflight] HARS healthz ok", so)
        out2 = os.path.join(self.d.name, "out-pf")
        with mock.patch.object(R.FakeHars, "health", side_effect=OSError("connection refused")):
            rc, so, se = self.run_main(out=out2, base=[a for a in F.RUN_ARGS if a != "--no-preflight"])
        self.assertEqual(rc, 3)
        self.assertIn("HARS healthz failed", se)
        self.assertFalse(os.path.exists(os.path.join(out2, "records.jsonl")))        # nothing spent
        out3 = os.path.join(self.d.name, "out-key")
        rc, so, se = call(["--sessions", self.sd, "--out", out3, "--provider", "anthropic", "--key-env",
                           "LS_TEST_NO_SUCH_KEY", "--fake-hars", "--window", "14000", "--keep-tail", "3000",
                           "--mem-budgets", "512"])
        self.assertEqual(rc, 3)
        self.assertIn("LS_TEST_NO_SUCH_KEY", se)

    def test_two_sessions_in_parallel(self):
        F.write_session(self.sd, F.make_session("syn-2", n=100))
        rc, so, se = self.run_main(["--session-concurrency", "2"])
        self.assertEqual(rc, 0, so + se)
        rows = rows_of(self.out)
        self.assertEqual({r["chain_id"] for r in rows}, {"syn-1", "syn-2"})
        self.assertTrue(all(r["valid"] for r in rows))
        done = [json.loads(l) for l in open(os.path.join(self.out, "done.jsonl"))]
        self.assertEqual(len(done), 8)                       # 2 sessions x 4 arms x 1 budget

    def test_stdout_is_restored_after_main(self):
        before = (os.sys.stdout, os.sys.stderr)
        self.run_main()
        self.assertEqual((os.sys.stdout, os.sys.stderr), before)

    def test_session_without_boundary_is_skipped_cleanly(self):
        sd = os.path.join(self.d.name, "short")
        F.write_session(sd, F.make_session("tiny", n=6, questions=False))
        rc, so, se = call(["--sessions", sd, "--out", os.path.join(self.d.name, "o-short")] + F.RUN_ARGS)
        self.assertEqual(rc, 0, so + se)


if __name__ == "__main__":
    unittest.main()
