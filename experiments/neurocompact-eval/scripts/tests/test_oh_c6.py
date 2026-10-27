"""Amendment C.6: the OpenHands track on the streaming long-session runner. home_run.py with a FAKE
run_longsession / judge / metrics runner (chains_to_longsession and the dry runs execute for real):
oh pipeline order and arguments, the parity CHECK (calibrate) semantics, refusal / overrides, the dry-run
projection, the legacy flag. No network, no pi."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest

import home_run as H
import longsession as L
import test_home_run as thr
from oh_chain_fixture import make_artifact

ARMS = ("baseline", "nc-chunks", "nc-graph", "nc-note-only")


class LSFakeRunner:
    """Stands in for run_longsession.py (writes records / compactions / done / MANIFEST in the real shapes), judge_v2,
    export_longsession_judge, metrics_v2, funnel_chains. Everything else (chains_to_longsession, dry runs) runs for real."""

    def __init__(self, ratio=0.95, clip=False, native=True, invalid_arm=None, stop_cost=0.05):
        self.calls = []
        self.ratio, self.clip, self.native, self.invalid_arm, self.cost = ratio, clip, native, invalid_arm, stop_cost

    def __call__(self, argv, env, cwd, log_path, echo=True):
        script = next((os.path.basename(a) for a in argv if a.endswith(".py")), "")
        self.calls.append((script, argv))
        h = {"run_longsession.py": self.longsession, "judge_v2.py": self.judge, "export_longsession_judge.py": self.export,
             "metrics_v2.py": self.metrics, "funnel_chains.py": self.funnel}.get(script)
        if h is None:
            return H.subprocess_runner(argv, env, cwd, log_path, False)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        rc, text = h(argv)
        open(log_path, "a").write(text)
        return rc, text

    @staticmethod
    def arg(argv, flag):
        return argv[argv.index(flag) + 1]

    def runs(self, parity=None):
        out = [a for s, a in self.calls if s == "run_longsession.py" and "--dry-run" not in a]
        if parity is None:
            return out
        return [a for a in out if (self.arg(a, "--name") == "oh-parity") == parity]

    def longsession(self, argv):
        if "--dry-run" in argv:
            ss = L.load_sessions([self.arg(argv, "--sessions")])
            return 0, "log line\n" + json.dumps({"sessions": [{"sid": x["sid"], "boundaries": 3, "per_budget": {
                "1536": {"asks": {"baseline": len(x["questions"])}}}} for x in ss], "grand_total_usd": 0.1}, indent=1)
        if "--resume" in argv:
            out = self.arg(argv, "--resume")
        else:
            out = self.arg(argv, "--out")
        os.makedirs(out, exist_ok=True)
        sessions = L.load_sessions([self.arg(argv, "--sessions")])
        budgets = [int(b) for b in self.arg(argv, "--mem-budgets").split(",")]
        recs, comps, done = [], [], []
        for s in sessions:
            sid = s["sid"]
            for B in budgets:
                for arm in ARMS:
                    ok = arm != self.invalid_arm
                    base, note, rec = 1500, 600, 1500 * self.ratio - 600
                    mem = {"baseline": base, "nc-chunks": note + rec, "nc-graph": note + rec, "nc-note-only": note}[arm]
                    for q in s["questions"]:
                        recs.append({"rid": f"{q['qid']}::{arm}::b{B}", "qid": q["qid"], "base_qid": q["qid"], "chain_id": sid,
                                     "sid": sid, "cluster_session": sid, "arm": arm, "mem_budget": B, "b_tag": True,
                                     "valid": ok, "chain_invalid_reasons": [] if ok else ["RECALL FAIL"],
                                     "qtype": q["qtype"], "answerable": q["answerable"], "is_end": True, "checkpoint": 3,
                                     "memory_tokens": mem, "arm_note_tokens": 0 if arm == "baseline" else note,
                                     "memory_tokens_injected": rec if arm in ("nc-chunks", "nc-graph") else 0,
                                     "clipped_summaries": {"n": 3, "unrepaired": 3 if (self.clip and arm == "baseline") else 0},
                                     "tokenizer": "anthropic:count_tokens:claude-haiku-5-5", "tokenizer_native": self.native,
                                     "cost_usd": 0.001})
                    done.append({"sid": sid, "arm": arm, "B": B, "n_rows": len(s["questions"]), "valid": ok})
                for j in (1, 2, 3):
                    comps.append({"sid": sid, "j": j, "arm": "baseline", "B": B, "realized_tokens": 1500, "cost_usd": 0.001})
                    comps.append({"sid": sid, "j": j, "arm": "note", "B": B, "realized_tokens": 600, "cost_usd": 0.001})
        for fn, rows in (("records.jsonl", recs), ("samples.jsonl", recs), ("compactions.jsonl", comps), ("done.jsonl", done)):
            with open(os.path.join(out, fn), "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        json.dump({"hash": "x", "run_id": "ls-fake"}, open(os.path.join(out, "config.json"), "w"))
        json.dump({"budget_spent_all_invocations": self.cost * len(sessions), "failed_sessions": [], "tokenizer": "x"},
                  open(os.path.join(out, "MANIFEST.json"), "w"))
        return 0, f"done: {len(recs)} rows\n"

    def export(self, argv):
        open(self.arg(argv, "--out"), "w").write("{}\n")
        return 0, "exported\n"

    def judge(self, argv):
        open(self.arg(argv, "--out"), "w").write("{}\n")
        open(self.arg(argv, "--mapping"), "w").write("{}\n")
        return 0, "judge spend: $0.50000 over 12 calls (cap $9.00; profile anthropic)\n"

    def metrics(self, argv):
        open(self.arg(argv, "--out"), "w").write("# metrics\n")
        json.dump({"gates": {"verdict": "INCONCLUSIVE"}}, open(self.arg(argv, "--json"), "w"))
        return 0, "**verdict: INCONCLUSIVE** (x)\n"

    def funnel(self, argv):
        open(self.arg(argv, "--out-md"), "w").write("funnel\n")
        open(self.arg(argv, "--out-json"), "w").write("{}\n")
        return 0, "funnel\n"


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.art = os.path.join(self.d.name, "oh-art")
        self.chains = make_artifact(self.art, 4)
        self.root = os.path.join(self.d.name, "root")
        self.base = ["--root", self.root, "--oh-dir", self.art, "--run-cap", "100", "--judge-cap", "100"]

    def tearDown(self):
        self.d.cleanup()

    def go(self, runner, *cmd, pre=()):
        argv = self.base + list(pre) + list(cmd)
        ptr = os.path.join(self.root, ".homerun-current")
        if os.path.exists(ptr) and "--new" not in argv and "--resume" not in argv:
            argv += ["--resume", open(ptr).read()]           # no implicit resume any more: continue the campaign explicitly
        return thr.quiet(H.main, argv, runner=runner)

    def camp(self):
        return open(os.path.join(self.root, ".homerun-current")).read()

    def ids(self):
        return [c["chain_id"] for c in self.chains]


class OhPipeline(Base):
    def test_the_oh_track_runs_through_run_longsession_without_pi_or_replay(self):
        r = LSFakeRunner()
        rc, so, se = self.go(r, "oh")
        self.assertEqual(rc, 0, so + se)
        scripts = [s for s, _ in r.calls]
        for banned in ("run_chainreplay.py", "merge_runs.py", "export_judge_input.py"):
            self.assertNotIn(banned, scripts)
        order = [s for s in scripts if s in ("chains_to_longsession.py", "run_longsession.py", "export_longsession_judge.py",
                                             "judge_v2.py", "metrics_v2.py", "funnel_chains.py")]
        dry = [s for s, a in r.calls if s == "run_longsession.py" and "--dry-run" in a]
        self.assertEqual(len(dry), 2)                                     # main + parity projection
        self.assertEqual([s for s in order if not (s == "run_longsession.py")][:1], ["chains_to_longsession.py"])
        self.assertEqual([s for s in order if s != "run_longsession.py"],
                         ["chains_to_longsession.py", "export_longsession_judge.py", "judge_v2.py", "metrics_v2.py",
                          "metrics_v2.py", "funnel_chains.py"])
        parity, main = r.runs(parity=True), r.runs(parity=False)
        self.assertEqual((len(parity), len(main)), (1, 1))
        # run_longsession arguments of the real run
        a = main[0]
        for flag in ("--use-real-compactions", "--keep-tail", "--arms", "--checkpoints", "--mem-budgets", "--budget-usd",
                     "--mcp", "--profile"):
            self.assertIn(flag, a)
        self.assertEqual(r.arg(a, "--keep-tail"), "0")
        self.assertEqual(r.arg(a, "--arms"), "baseline,nc-chunks,nc-graph,nc-note-only")
        self.assertEqual(r.arg(a, "--checkpoints"), "end")
        self.assertEqual(r.arg(a, "--mem-budgets"), "1536")
        self.assertNotIn("--baseline-cap", a)
        sess_dir = r.arg(a, "--sessions")
        self.assertEqual(sorted(s["sid"] for s in L.load_sessions([sess_dir])), sorted(self.ids()))
        # judge / metrics wiring: long-session track, end checkpoint, sessions dir feeds the exporter
        exp = [a for s, a in r.calls if s == "export_longsession_judge.py"][0]
        self.assertEqual(r.arg(exp, "--sessions"), sess_dir)
        met = [a for s, a in r.calls if s == "metrics_v2.py" and r.arg(a, "--track") == "longsession"][0]
        self.assertEqual(r.arg(met, "--track"), "longsession")
        self.assertIn("--end-only", met)
        self.assertNotIn("--exclude-sid", met)                           # the parity check excludes nothing
        chains_view = [a for s, a in r.calls if s == "metrics_v2.py" and r.arg(a, "--track") == "chains"]
        self.assertEqual(len(chains_view), 1)                            # the descriptive S-family companion (C.6-3)
        self.assertTrue(r.arg(chains_view[0], "--out").endswith("REPORT-oh-chains-family.md"))
        c = self.camp()
        for st in ("oh.convert", "oh.parity.run", "oh.estimate", "oh.run", "oh.export", "oh.judge", "oh.metrics",
                   "oh.metrics_chains", "oh.funnel"):
            self.assertTrue(os.path.isfile(os.path.join(c, "steps", st + ".json")), st)
        recs = H.read_jsonl(os.path.join(c, "oh", "run", "records.jsonl"))
        self.assertEqual({x["chain_id"] for x in recs}, set(self.ids()))
        self.assertEqual(len(recs), 4 * 5 * 4)                            # chains x questions x arms
        qids = {x["qid"] for x in recs}
        self.assertEqual(qids, {q["qid"] for ch in self.chains for q in ch["qa"]})      # qids preserved
        led = H.read_json(os.path.join(c, "ledger.json"))["entries"]
        self.assertTrue(any(e["step"] == "oh.parity.run" for e in led))
        self.assertTrue(any(e["step"] == "oh.run" and e["phase"] == "run" for e in led))
        self.assertAlmostEqual(sum(e["usd"] for e in led if e["phase"] == "judge"), 0.5)
        # second invocation: nothing is rerun
        n = len(r.calls)
        rc, so, se = self.go(r, "oh")
        self.assertEqual(rc, 0, so + se)
        self.assertEqual([s for s, _ in r.calls[n:]], [])

    def test_oh_needs_no_pi_even_without_a_pi_binary_on_the_path(self):
        r = LSFakeRunner()
        rc, so, se = self.go(r, "oh", "--skip-judge")
        self.assertEqual(rc, 0, so + se)
        self.assertNotIn("pi", [s for s, _ in r.calls])
        self.assertFalse(os.path.exists(os.path.join(self.camp(), "oh", "merged")))

    def test_conversion_uses_only_chains_jsonl(self):
        r = LSFakeRunner()
        self.go(r, "oh", "--skip-judge")
        conv = [a for s, a in r.calls if s == "chains_to_longsession.py"][0]
        self.assertEqual(conv[conv.index("--chains") + 1], os.path.join(self.camp(), "oh", "inputs", "chains.jsonl"))
        m = H.read_json(os.path.join(self.camp(), "oh", "ls-data", ".conversion.json"))
        self.assertEqual(len(m["sessions"]), 4)
        self.assertEqual(m["errors"], [])

    def test_sweep_is_opt_in_and_budgets_reach_the_runner(self):
        r = LSFakeRunner()
        rc, so, se = self.go(r, "oh", "--skip-judge")                    # default: the primary B only
        self.assertEqual(rc, 0, so + se)
        self.assertEqual(r.arg(r.runs(parity=False)[0], "--mem-budgets"), "1536")
        r = LSFakeRunner()
        with tempfile.TemporaryDirectory() as d2:
            rc, so, se = thr.quiet(H.main, ["--root", d2, "--oh-dir", self.art, "--run-cap", "100", "--judge-cap", "100",
                                            "oh", "--skip-judge", "--budgets", "1536,1024,512"], runner=r)
        self.assertEqual(rc, 0, so + se)
        self.assertEqual(r.arg(r.runs(parity=False)[0], "--mem-budgets"), "1536,1024,512")
        self.assertEqual(r.arg(r.runs(parity=True)[0], "--mem-budgets"), "1536,1024,512")      # the check covers every B

    def test_offline_selftest_runs_the_whole_oh_track_with_the_mock_llm(self):
        os.environ["NC_TOKCOUNT"] = "estimate"
        rc, so, se = thr.quiet(H.main, ["--root", self.root, "--oh-dir", self.art, "--offline-selftest", "--run-cap", "5",
                                        "--judge-cap", "5", "oh", "--skip-judge", "--accept-parity-fail"])
        self.assertEqual(rc, 0, so + se)
        recs = H.read_jsonl(os.path.join(self.camp(), "oh", "run", "records.jsonl"))
        self.assertEqual({x["arm"] for x in recs}, set(ARMS))
        self.assertEqual({x["chain_id"] for x in recs}, set(self.ids()))
        self.assertTrue(all(x["valid"] for x in recs))
        lags = {x["lag"] for x in recs if x["answerable"]}
        self.assertEqual(lags, {1, 2, 3})                                 # the chain's design lags survive the conversion
        self.assertEqual({x["qtype"] for x in recs}, {"single", "xspan_locator"})
        self.assertTrue(os.path.isfile(os.path.join(self.camp(), "oh", "run", "judge-input.jsonl")))
        # HEAD / TAIL slices of the chain items reach the records (metrics_v2 slice tables)
        self.assertEqual({x["slice"] for x in recs if x["answerable"]}, {"HEAD", "TAIL"})
        # the funnel reads the per-boundary summaries of a run_longsession run dir (memory/<sid>.json)
        import funnel_chains as F
        c = self.camp()
        out_json = os.path.join(c, "oh", "run", "funnel.json")
        rc, so2, se2 = thr.quiet(F.main, ["--run", os.path.join(c, "oh", "run"), "--chains",
                                          os.path.join(c, "oh", "inputs", "chains.jsonl"),
                                          "--out-md", os.path.join(c, "oh", "run", "funnel.md"), "--out-json", out_json])
        self.assertEqual(rc, 0, so2 + se2)
        pa = H.read_json(out_json)["per_arm"]
        self.assertIsInstance(pa["baseline"]["anchor_in_summary"], dict)
        self.assertIsInstance(pa["nc-chunks"]["anchor_in_note"], dict)


class ParityCheck(Base):
    def cal(self):
        return H.read_json(os.path.join(self.camp(), "calibration.json"))

    def test_calibrate_is_one_cheap_run_on_the_probe_session_without_iteration(self):
        r = LSFakeRunner(ratio=1.4)                                       # out of window: a FAIL is reported, never iterated
        rc, so, se = self.go(r, "calibrate")
        self.assertEqual(rc, 1, so + se)
        runs = r.runs()
        self.assertEqual(len(runs), 1)                                    # no cap iteration loop
        self.assertNotIn("run_chainreplay.py", [s for s, _ in r.calls])
        a = runs[0]
        self.assertEqual(r.arg(a, "--name"), "oh-parity")
        cid, order = H.pick_calibration_chain(self.ids(), 20261026)
        sess = L.load_sessions([r.arg(a, "--sessions")])
        self.assertEqual([s["sid"] for s in sess], [f"parity-{cid}"])
        s = sess[0]
        # probes only: no decision-set qid, no gold; same messages and boundaries as the converted chain
        self.assertTrue(all(q["qid"].startswith(f"parity-{cid}_p") for q in s["questions"]))
        self.assertTrue(all(q["evidence"] == [] and "gold_anchor" not in q for q in s["questions"]))
        self.assertEqual(len(s["questions"]), len(H.PARITY_PROBES))
        self.assertEqual([c["at_idx"] for c in s["compactions"]][-1], len(s["messages"]))
        cal = self.cal()
        self.assertEqual((cal["amendment"], cal["mode"], cal["status"]), ("C.6", "parity-check", "fail"))
        self.assertEqual(cal["excluded_from_decision_set"], [])
        self.assertIn("outside", " ".join(cal["stopped_because"]))
        self.assertEqual(cal["chain_ids"], [cid])
        self.assertAlmostEqual(cal["per_b"]["1536"]["ratio_nc_chunks"], 1.4)

    def test_pass_records_ratios_and_tokens(self):
        r = LSFakeRunner(ratio=0.95)
        rc, so, se = self.go(r, "calibrate")
        self.assertEqual(rc, 0, so + se)
        cal = self.cal()
        pb = cal["per_b"]["1536"]
        self.assertEqual(cal["status"], "pass")
        self.assertAlmostEqual(pb["ratio_nc_chunks"], 0.95)
        self.assertAlmostEqual(pb["ratio_nc_graph"], 0.95)
        self.assertEqual((pb["baseline_memory_tokens_mean"], pb["note_tokens_mean"]), (1500, 600))
        self.assertTrue(pb["clip_free"] and pb["all_valid"] and pb["tokenizer_native"])
        self.assertEqual(cal["window"], [0.8, 1.05])
        self.assertGreater(cal["spent_usd"], 0)
        led = H.read_json(os.path.join(self.camp(), "ledger.json"))["entries"]
        self.assertEqual([e["step"] for e in led], ["oh.parity.run"])
        # idempotent
        n = len(r.calls)
        self.assertEqual(self.go(r, "calibrate")[0], 0)
        self.assertEqual([s for s, _ in r.calls[n:]], [])

    def test_k_sessions(self):
        r = LSFakeRunner()
        rc, so, se = self.go(r, "calibrate", "--check-sessions", "2")
        self.assertEqual(rc, 0, so + se)
        cid, order = H.pick_calibration_chain(self.ids(), 20261026)
        self.assertEqual(self.cal()["chain_ids"], order[:2])
        self.assertEqual(len(L.load_sessions([r.arg(r.runs()[0], "--sessions")])), 2)

    def test_clip_invalid_cell_and_non_native_tokens_fail_the_check(self):
        for kw, needle in (({"clip": True}, "CLIP"), ({"invalid_arm": "nc-chunks"}, "invalid cells"),
                           ({"native": False}, "native=False")):
            with tempfile.TemporaryDirectory() as d2:
                base = ["--root", os.path.join(d2, "r"), "--oh-dir", self.art, "--run-cap", "100", "--judge-cap", "100"]
                rc, so, se = thr.quiet(H.main, base + ["calibrate"], runner=LSFakeRunner(**kw))
                self.assertEqual(rc, 1, (kw, so, se))
                cal = H.read_json(os.path.join(open(os.path.join(d2, "r", ".homerun-current")).read(), "calibration.json"))
                self.assertIn(needle, " ".join(cal["stopped_because"]), kw)

    def test_parity_from_records_is_pure_and_per_session_averaged(self):
        rows = []
        for sid, nc in (("a", 1200), ("b", 1800)):
            for arm, mem in (("baseline", 1500), ("nc-chunks", nc), ("nc-graph", nc), ("nc-note-only", 500)):
                for i in range(2):
                    rows.append({"chain_id": sid, "arm": arm, "mem_budget": 1536, "memory_tokens": mem, "valid": True,
                                 "arm_note_tokens": 600, "memory_tokens_injected": mem - 600,
                                 "clipped_summaries": {"n": 3, "unrepaired": 0}, "tokenizer_native": True})
        out = H.parity_from_records(rows, [], [1536])["1536"]
        self.assertAlmostEqual(out["ratio_nc_chunks"], (0.8 + 1.2) / 2)
        self.assertTrue(out["in_window"] and out["clip_free"] and out["all_valid"])
        self.assertEqual(out["sessions"], ["a", "b"])
        rows[0]["clipped_summaries"] = {"n": 3, "unrepaired": 1}
        self.assertFalse(H.parity_from_records(rows, [], [1536])["1536"]["clip_free"])
        self.assertEqual(H.parity_from_records([], [], [1536])["1536"]["rows"], 0)

    def test_oh_runs_the_check_first_and_refuses_when_it_fails(self):
        r = LSFakeRunner(ratio=1.4)
        rc, so, se = self.go(r, "oh")
        self.assertEqual(rc, 2)
        self.assertIn("parity check FAILED", se)
        self.assertEqual(len(r.runs(parity=False)), 0)                    # the real run never started
        self.assertEqual(len(r.runs(parity=True)), 1)
        # --accept-parity-fail runs anyway (recorded), without re-running the check
        rc, so, se = self.go(r, "oh", "--skip-judge", "--accept-parity-fail")
        self.assertEqual(rc, 0, so + se)
        self.assertEqual((len(r.runs(parity=False)), len(r.runs(parity=True))), (1, 1))
        self.assertEqual(H.read_json(os.path.join(self.camp(), "calibration.json"))["status"], "fail")

    def test_skip_parity_check_runs_nothing_for_it(self):
        r = LSFakeRunner(ratio=1.4)
        rc, so, se = self.go(r, "oh", "--skip-judge", "--skip-parity-check")
        self.assertEqual(rc, 0, so + se)
        self.assertEqual(len(r.runs(parity=True)), 0)
        self.assertFalse(os.path.exists(os.path.join(self.camp(), "calibration.json")))

    def test_a_pre_c6_calibration_is_void(self):
        r = LSFakeRunner()
        self.go(r, "dry-run", "--tracks", "oh") if False else None
        thr.quiet(H.main, self.base + ["dry-run", "--tracks", "oh"], runner=r)               # opens the campaign
        H.write_json(os.path.join(self.camp(), "calibration.json"),
                     {"status": "pass", "amendment": "C.5", "caps": {"1536": 2010}, "excluded_from_decision_set": ["x"]})
        rc, so, se = self.go(r, "oh")
        self.assertEqual(rc, 2)
        self.assertIn("pre-C.6", se)
        self.assertEqual(r.runs(), [])

    def test_a_budget_the_check_did_not_cover_is_refused(self):
        r = LSFakeRunner()
        self.assertEqual(self.go(r, "calibrate")[0], 0)
        rc, so, se = self.go(r, "oh", "--budgets", "1536,512")
        self.assertEqual(rc, 2)
        self.assertIn("B=[512]", se)

    def test_check_failure_message_is_actionable_for_the_shim_embedder_case(self):
        r = LSFakeRunner(invalid_arm="nc-graph")
        rc, so, se = self.go(r, "calibrate")
        self.assertEqual(rc, 1)
        self.assertIn("invalid cells", so)


class DryRunAndLegacy(Base):
    def test_dry_run_prices_the_oh_track_with_the_longsession_projection(self):
        os.environ["NC_TOKCOUNT"] = "estimate"
        rc, so, se = thr.quiet(H.main, self.base + ["--offline-selftest", "dry-run", "--tracks", "oh"])
        self.assertEqual(rc, 0, so + se)
        est = H.read_json(os.path.join(self.camp(), "dry-run.json"))["estimates"]["oh"]
        self.assertEqual(est["sessions"], 4)
        self.assertEqual(est["judged_items"], 4 * 5 * 4)           # chains x questions x arms
        self.assertIn("parity_check_usd", est)
        self.assertEqual(est["budgets"], [1536])
        cmds = open(os.path.join(self.camp(), "scripts", "cmdlines.txt")).read()
        self.assertIn("run_longsession.py", cmds)
        self.assertNotIn("run_chainreplay.py", cmds)

    def test_legacy_flag_routes_to_run_chainreplay_and_default_never_does(self):
        r = thr.FakeRunner()
        rc, so, se = thr.quiet(H.main, self.base + ["oh", "--legacy-replay", "--baseline-cap-scale", "1.0"], runner=r)
        self.assertEqual(rc, 0, so + se)
        scripts = [s for s, _ in r.calls]
        self.assertIn("run_chainreplay.py", scripts)
        self.assertNotIn("run_longsession.py", scripts)
        # calibrate --legacy-replay is the C.4 cap loop through pi
        r2 = thr.FakeRunner()
        with tempfile.TemporaryDirectory() as d2:
            rc, so, se = thr.quiet(H.main, ["--root", d2, "--oh-dir", self.art, "--run-cap", "100", "--judge-cap", "100",
                                            "calibrate", "--legacy-replay"], runner=r2)
        self.assertEqual(rc, 0, so + se)
        self.assertIn("--calibrate", [a for s, argv in r2.calls if s == "run_chainreplay.py" for a in argv])
        # default `oh` / `calibrate` with a run_chainreplay-only fake runner: the chain replay is never invoked
        r3 = LSFakeRunner()
        self.go(r3, "calibrate")
        self.assertNotIn("run_chainreplay.py", [s for s, _ in r3.calls])

    def test_legacy_dry_run_uses_the_replay_projection(self):
        r = thr.FakeRunner()
        rc, so, se = thr.quiet(H.main, self.base + ["dry-run", "--tracks", "oh", "--legacy-replay"], runner=r)
        self.assertEqual(rc, 0, so + se)
        self.assertIn("run_chainreplay.py", [s for s, _ in r.calls])


class Smoke(Base):
    def test_oh_smoke_converts_the_smallest_chain_and_runs_the_whole_streaming_flow(self):
        os.environ["HR_TEST_KEY_C6"] = "k-" + "x" * 20
        r = LSFakeRunner()
        with tempfile.TemporaryDirectory() as d2:
            rc, so, se = thr.quiet(H.main, ["--root", d2, "--key-env", "HR_TEST_KEY_C6", "--oh-dir", self.art, "smoke", "--skip-ls"],
                                   runner=r, probes=thr.FakeProbes(pi=None))        # no pi on this machine: fine
            self.assertEqual(rc, 0, so + se)
            camp = [x for x in os.listdir(d2) if x.endswith("-smoke")][0]
            chains = H.read_jsonl(os.path.join(d2, camp, "oh", "inputs", "chains.jsonl"))
            sizes = {c["chain_id"]: sum(len(e["text"]) for sp in c["spans"] for e in sp["segments"]) for c in self.chains}
            self.assertEqual([c["chain_id"] for c in chains], [min(sizes, key=lambda k: (sizes[k], k))])
            scripts = [s for s, _ in r.calls]
            for s in ("chains_to_longsession.py", "run_longsession.py", "export_longsession_judge.py", "judge_v2.py",
                      "metrics_v2.py"):
                self.assertIn(s, scripts)
            self.assertNotIn("run_chainreplay.py", scripts)
            rep = H.read_json(os.path.join(d2, camp, "smoke-report.json"))
            self.assertTrue(all(x["ok"] for x in rep), rep)
            oh = [x for x in rep if x["stage"].startswith("oh")][0]
            self.assertEqual(oh["parity"], "pass")
            self.assertAlmostEqual(oh["parity_ratio_nc_chunks"], 0.95)
            self.assertEqual(oh["memory_tokens"]["baseline"], 1500)


if __name__ == "__main__":
    unittest.main()
