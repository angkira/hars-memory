"""Offline tests for Protocol Amendment C.4 (baseline cap = calibrated parity parameter).

Runner side: precedence map > scale > legacy > default, an explicit value is never overwritten by the
B sweep, per-B cap + source in records / chain_budget / run_manifest / run_budget, the CLIP verdict of
`--calibrate`. Orchestrator side (home_run.py, fake runner): the calibrate loop (fail -> advice -> pass),
`oh` refusing without a calibration, the calibrated caps reaching the runner, the calibration chain
excluded from the decision set. No network, no key, $0.

Run:  cd experiments/neurocompact-eval/scripts && python -m pytest tests -q
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
sys.dont_write_bytecode = True
os.environ.setdefault("NC_PACK_QUIET", "1")
import home_run as H  # noqa: E402
import run_chainreplay as rc  # noqa: E402
import test_amendment_c as tac  # noqa: E402
import test_home_run as thr  # noqa: E402


def _ns(**kw):
    base = dict(baseline_cap=None, baseline_cap_scale=None, baseline_cap_legacy=None, baseline_cap_source=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


# --------------------------------------------------------------------------- resolver

class Precedence(unittest.TestCase):
    def test_default_is_b(self):
        r = rc.resolve_baseline_cap(_ns(), 1024)
        self.assertEqual((r["baseline_cap"], r["baseline_cap_basis"], r["baseline_cap_source"]),
                         (1024, "default", "default"))

    def test_legacy_beats_default_and_scale_beats_legacy_and_map_beats_scale(self):
        self.assertEqual(rc.resolve_baseline_cap(_ns(baseline_cap_legacy=1700), 1536)["baseline_cap"], 1700)
        r = rc.resolve_baseline_cap(_ns(baseline_cap_legacy=1700, baseline_cap_scale=1.31), 1536)
        self.assertEqual((r["baseline_cap"], r["baseline_cap_basis"], r["baseline_cap_scale"]), (2012, "scale", 1.31))
        r = rc.resolve_baseline_cap(_ns(baseline_cap_legacy=1700, baseline_cap_scale=1.31,
                                        baseline_cap="1536:2010,1024:1340"), 1536)
        self.assertEqual((r["baseline_cap"], r["baseline_cap_basis"]), (2010, "map"))
        self.assertEqual(rc.resolve_baseline_cap(_ns(baseline_cap="1536:2010,1024:1340"), 1024)["baseline_cap"], 1340)

    def test_map_without_this_b_falls_to_scale_then_legacy_then_default(self):
        a = _ns(baseline_cap="1536:2010", baseline_cap_scale=1.31)
        self.assertEqual(rc.resolve_baseline_cap(a, 512)["baseline_cap"], round(512 * 1.31))
        self.assertEqual(rc.resolve_baseline_cap(_ns(baseline_cap="1536:2010"), 512)["baseline_cap_basis"], "default")

    def test_source_label_follows_the_explicit_value(self):
        a = _ns(baseline_cap="1536:2010", baseline_cap_source="calibrated:/x/y")
        self.assertEqual(rc.resolve_baseline_cap(a, 1536)["baseline_cap_source"], "calibrated:/x/y")
        self.assertEqual(rc.resolve_baseline_cap(_ns(baseline_cap="1536:2010"), 1536)["baseline_cap_source"], "cli")
        self.assertEqual(rc.resolve_baseline_cap(_ns(baseline_cap_legacy=1600), 1536)["baseline_cap_source"], "cli")

    def test_map_syntax_errors(self):
        for bad in ("1536", "1536:", "a:b", "1536:0", "1536:2000,1536:2100"):
            with self.assertRaises(ValueError, msg=bad):
                rc.parse_baseline_cap_map(bad)

    def test_sweep_never_overwrites_an_explicit_cap(self):
        base = tac._ns(baseline_cap="1536:2010,1024:1340", baseline_cap_scale=None, baseline_cap_legacy=None,
                       baseline_cap_source="calibrated:/r")
        a1536, a1024 = (rc.args_for_budget(base, b, sweep=True) for b in (1536, 1024))
        self.assertEqual((a1536.baseline_sum_max_tokens, a1024.baseline_sum_max_tokens), (2010, 1340))
        self.assertEqual(a1024.baseline_cap_info["baseline_cap_source"], "calibrated:/r")
        # the frozen note / recall formulas still follow B, not the cap
        self.assertEqual((a1024.note_sum_max_tokens, a1024.recall_cap_max), (427, 683))
        # legacy value in a single-B sweep is kept too
        one = rc.args_for_budget(tac._ns(baseline_cap_legacy=1500), 1024, sweep=True)
        self.assertEqual(one.baseline_sum_max_tokens, 1500)


# --------------------------------------------------------------------------- runner end to end

class CapWorld(tac.FakeWorld):
    """FakeWorld that records NC_SUM_MAX_TOKENS per arm and can clip the baseline (stop=length)."""

    def __init__(self, clip_baseline=False, **kw):
        super().__init__(**kw)
        self.envs = []
        self.clip_baseline = clip_baseline

    def run_pi_turn(self, argv, env, timeout):
        r = super().run_pi_turn(argv, env, timeout)
        if r["metrics"]:
            self.envs.append((env["NC_PERSIST"], int(env["NC_SUM_MAX_TOKENS"]), env["NC_SUM_SECTION_TRIM"]))
            if self.clip_baseline and env["NC_PERSIST"] == "0":
                r["metrics"][0]["stop"] = "length"
        return r


def run_main(world, argv_extra, chains):
    td = tempfile.mkdtemp()
    cpath = os.path.join(td, "chains.jsonl")
    with open(cpath, "w") as f:
        for c in chains:
            f.write(json.dumps(c) + "\n")
    out = os.path.join(td, "run")
    names = ("run_pi_turn", "hars_status", "hars_recall", "write_settings", "write_mcp_url",
             "write_models_base_url", "restore", "restore_models_json", "WORKDIR")
    saved = {k: getattr(rc, k) for k in names}
    rc.run_pi_turn, rc.hars_status, rc.hars_recall = world.run_pi_turn, world.hars_status, world.hars_recall
    rc.write_settings = lambda *a, **k: "bk"
    rc.write_mcp_url = lambda *a, **k: "bk"
    rc.write_models_base_url = lambda *a, **k: None
    rc.restore = lambda *a, **k: None
    rc.restore_models_json = lambda *a, **k: None
    rc.WORKDIR = os.path.join(td, "wd")
    gstate = (rc.PROVIDER, rc.CTX_WINDOW_OVERRIDE, rc.AGENT_THINKING, dict(rc.CAPS))
    argv = ["run_chainreplay.py", "--chains", cpath, "--out", out, "--run-id", "t1",
            "--agent-model", "claude-haiku-5-5", "--provider", "anthropic", "--allow-same-family-judge",
            "--mcp", "http://127.0.0.1:1/mcp", "--budget-usd", "5", *argv_extra]
    old_argv, old_key = sys.argv, os.environ.get("ANTHROPIC_API_KEY")
    sys.argv = argv
    os.environ["ANTHROPIC_API_KEY"] = "x"
    buf, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = rc.main()
    finally:
        sys.argv = old_argv
        if old_key is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = old_key
        for k, v in saved.items():
            setattr(rc, k, v)
        rc.PROVIDER, rc.CTX_WINDOW_OVERRIDE, rc.AGENT_THINKING = gstate[:3]
        rc.CAPS.clear()
        rc.CAPS.update(gstate[3])
    return code, out, buf.getvalue() + err.getvalue(), td


def jl(path):
    return [json.loads(x) for x in open(path) if x.strip()]


class RunnerCaps(unittest.TestCase):
    def run_it(self, extra, world=None, chains=None):
        world = world or CapWorld()
        code, out, log, td = run_main(world, extra, chains or [tac._chain("chain_S0_c0", "S0")])
        self.addCleanup(shutil.rmtree, td, True)
        return code, out, log, world

    def test_explicit_map_reaches_generation_trim_records_manifest_and_run_budget(self):
        code, out, log, world = self.run_it(["--mem-budgets", "1536,1024", "--baseline-cap", "1536:2010,1024:1340",
                                             "--baseline-cap-source", "calibrated:/camp/oh/calibration/b1536-i2",
                                             "--exclude-chains", "chain_CAL_c0"])
        self.assertEqual(code, 0, log[-2500:])
        # ONE value is the prompt/generation max_tokens AND the hard trim: env of the baseline compactions
        base_envs = {c for p, c, t in world.envs if p == "0"}
        self.assertEqual(base_envs, {2010, 1340})
        self.assertTrue(all(t == "1" for _, _, t in world.envs))                      # NC_SUM_SECTION_TRIM=1
        comps = jl(os.path.join(out, "compactions.jsonl"))
        bcaps = {(r["mem_budget"], r["cap"]) for r in comps if r["arm"] == "baseline"}
        self.assertEqual(bcaps, {(1536, 2010), (1024, 1340)})
        buds = {b["mem_budget"]: b for b in jl(os.path.join(out, "chain_budget.jsonl"))}
        for B, cap in ((1536, 2010), (1024, 1340)):
            b = buds[B]
            self.assertEqual((b["baseline_sum_max_tokens"], b["baseline_cap_basis"], b["baseline_cap_source"]),
                             (cap, "map", "calibrated:/camp/oh/calibration/b1536-i2"))
        recs = jl(os.path.join(out, "records.jsonl"))
        self.assertEqual({(r["mem_budget"], r["baseline_sum_max_tokens"], r["baseline_cap_source"]) for r in recs},
                         {(1536, 2010, "calibrated:/camp/oh/calibration/b1536-i2"),
                          (1024, 1340, "calibrated:/camp/oh/calibration/b1536-i2")})
        man = json.load(open(os.path.join(out, "run_manifest.json")))
        self.assertEqual(man["baseline_caps"]["1536"]["baseline_cap"], 2010)
        self.assertEqual(man["baseline_caps"]["1024"]["baseline_cap_basis"], "map")
        self.assertEqual(man["excluded_chains"], ["chain_CAL_c0"])
        rb = json.load(open(os.path.join(out, "run_budget.json")))
        self.assertEqual(rb["baseline_cap"], 2010)
        self.assertEqual(rb["by_budget"]["1024"]["baseline_cap"], 1340)
        self.assertEqual(rb["by_budget"]["1536"]["baseline_cap_source"], "calibrated:/camp/oh/calibration/b1536-i2")
        self.assertEqual(rb["baseline_caps"]["1024"]["baseline_cap"], 1340)

    def test_scale_gives_rounded_caps_and_default_stays_b(self):
        code, out, log, world = self.run_it(["--mem-budgets", "1536,1024", "--baseline-cap-scale", "1.31"])
        self.assertEqual(code, 0, log[-2500:])
        self.assertEqual({c for p, c, _ in world.envs if p == "0"}, {2012, 1341})
        buds = {b["mem_budget"]: b for b in jl(os.path.join(out, "chain_budget.jsonl"))}
        self.assertEqual((buds[1024]["baseline_cap_basis"], buds[1024]["baseline_cap_scale"],
                          buds[1024]["baseline_cap_source"]), ("scale", 1.31, "cli"))
        code, out, log, world = self.run_it(["--mem-budgets", "1536,512"])
        self.assertEqual(code, 0, log[-2500:])
        self.assertEqual({c for p, c, _ in world.envs if p == "0"}, {1536, 512})
        self.assertEqual({b["baseline_cap_source"] for b in jl(os.path.join(out, "chain_budget.jsonl"))}, {"default"})

    def test_legacy_flag_still_works_for_one_b_and_is_refused_for_a_sweep(self):
        code, out, log, world = self.run_it(["--baseline-sum-max-tokens", "1700"])
        self.assertEqual(code, 0, log[-2500:])
        self.assertEqual({c for p, c, _ in world.envs if p == "0"}, {1700})
        recs = jl(os.path.join(out, "records.jsonl"))
        self.assertEqual({(r["baseline_sum_max_tokens"], r["baseline_cap_basis"]) for r in recs}, {(1700, "legacy")})
        # one-entry sweep: the explicit value is kept, not replaced by B
        code, out, log, world = self.run_it(["--mem-budgets", "1024", "--baseline-sum-max-tokens", "1500"])
        self.assertEqual(code, 0, log[-2500:])
        self.assertEqual({c for p, c, _ in world.envs if p == "0"}, {1500})
        # multi-B sweep: would be overwritten per B -> refuse instead of silently ignoring
        code, out, log, world = self.run_it(["--mem-budgets", "1536,1024", "--baseline-sum-max-tokens", "1700"])
        self.assertEqual(code, 2)
        self.assertIn("single-B knob", log)
        self.assertEqual(world.envs, [])

    def test_bad_cli_values_are_refused(self):
        for bad in (["--baseline-cap", "1536"], ["--baseline-cap-scale", "0"],
                    ["--baseline-cap-source", "whatever"], ["--baseline-sum-max-tokens", "0"]):
            code, out, log, world = self.run_it(bad)
            self.assertEqual(code, 2, bad)

    def test_exclude_chains_drops_the_chain_before_anything_runs(self):
        chains = [tac._chain("chain_S0_c0", "S0"), tac._chain("chain_S1_c0", "S1")]
        code, out, log, world = self.run_it(["--exclude-chains", "chain_S0_c0"], chains=chains)
        self.assertEqual(code, 0, log[-2500:])
        self.assertEqual({r["chain_id"] for r in jl(os.path.join(out, "records.jsonl"))}, {"chain_S1_c0"})
        self.assertEqual(json.load(open(os.path.join(out, "run_manifest.json")))["chains"], ["chain_S1_c0"])

    def test_calibrate_reports_clip_cap_source_and_recommendation(self):
        code, out, log, world = self.run_it(["--calibrate", "--arms", "baseline,nc-chunks", "--limit", "1",
                                             "--mem-budgets", "1536", "--baseline-cap", "1536:1536"],
                                            world=CapWorld(clip_baseline=True))
        self.assertEqual(code, 0, log[-2500:])
        cal = json.load(open(os.path.join(out, "calibration.json")))
        self.assertEqual((cal["baseline_cap"], cal["baseline_cap_source"], cal["mem_budget"]), (1536, "cli", 1536))
        self.assertTrue(cal["clip"]["baseline"]["clip_fail"])
        self.assertFalse(cal["clip"]["nc-chunks"]["clip_fail"])
        self.assertFalse(cal["clip_free"])
        self.assertIn("recommended_cap", cal)
        self.assertIsNotNone(cal["recommended_cap"])
        self.assertIn("--baseline-cap 1536:", log)                      # the advice names the new flag
        # a sweep of several B is not a calibration
        code, *_ = self.run_it(["--calibrate", "--arms", "baseline,nc-chunks", "--mem-budgets", "1536,1024"])
        self.assertEqual(code, 2)

    def test_per_chain_warning_names_the_baseline_cap_flag(self):
        comps = {"baseline": [{"span_idx": 3, "realized_tokens": 900, "cap": 1536}],
                 "nc-chunks": [{"span_idx": 3, "realized_tokens": 600, "cap": 640}]}
        b = rc.chain_budget("c1", tac._ns(), comps, [900], True, {"q1": []}, {"q1": None})
        self.assertTrue(any("--baseline-cap 1536:" in w for w in b["warnings"]), b["warnings"])
        self.assertIsNone(b["baseline_cap_source"])                     # no cap info on a bare namespace


# --------------------------------------------------------------------------- home_run: calibrate / oh

class CalibrationFlow(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.art = os.path.join(self.d.name, "oh-art")
        thr.make_oh_artifact(self.art, 4)
        self.root = os.path.join(self.d.name, "root")
        self.base = ["--root", self.root, "--oh-dir", self.art, "--run-cap", "100", "--judge-cap", "100"]
        self.toml = os.path.join(self.d.name, "t.toml")
        open(self.toml, "w").write("[oh]\nbatch_chains = 2\nbudgets = [1536]\nretries = 0\n")

    def tearDown(self):
        self.d.cleanup()

    def go(self, runner, *cmd, toml=None):
        argv = self.base + ["--config", toml or self.toml] + [cmd[0], "--legacy-replay"] + list(cmd[1:])
        ptr = os.path.join(self.root, ".homerun-current")
        if os.path.exists(ptr) and "--new" not in argv and "--resume" not in argv:
            argv += ["--resume", open(ptr).read()]           # no implicit resume any more: continue the campaign explicitly
        return thr.quiet(H.main, argv, runner=runner)

    def camp(self):
        return open(os.path.join(self.root, ".homerun-current")).read()

    @staticmethod
    def runs(r):
        return [a for s, a in r.calls if s == "run_chainreplay.py" and "--dry-run" not in a and "--calibrate" not in a]

    def chain_ids(self):
        return [json.loads(x)["chain_id"] for x in open(os.path.join(self.art, "chains.jsonl"))]

    # ---- the loop
    def test_fail_then_advice_then_pass_writes_calibration_json(self):
        r = thr.FakeRunner()
        r.calibration_script = [{"ratio": 1.31, "clip_baseline": True, "recommended_cap": 2010},
                                {"ratio": 0.97}]
        rc_, so, se = self.go(r, "calibrate")
        self.assertEqual(rc_, 0, so + se)
        caps = [int(a[a.index("--baseline-cap") + 1].split(":")[1]) for a in r.cal_argvs]
        self.assertEqual(caps, [1536, 2010])                          # iteration 2 runs at the runner's advice
        cal = H.read_json(os.path.join(self.camp(), "calibration.json"))
        self.assertEqual((cal["status"], cal["caps"], cal["primary_budget"]), ("pass", {"1536": 2010}, 1536))
        self.assertAlmostEqual(cal["scale"], 2010 / 1536, places=5)
        self.assertEqual(cal["ratio"], 0.97)
        self.assertEqual(len(cal["per_b"]["1536"]["iterations"]), 2)
        self.assertFalse(cal["per_b"]["1536"]["iterations"][0]["clip_free"])
        self.assertAlmostEqual(cal["spent_usd"], 0.5)
        cid, order = H.pick_calibration_chain(self.chain_ids(), 20261026)
        self.assertEqual(cal["chain_id"], cid)                        # first in seeded order
        self.assertEqual(cal["excluded_from_decision_set"], [cid])
        self.assertTrue(cal["source_run_dir"].endswith("b1536-i2"))
        led = H.read_json(os.path.join(self.camp(), "ledger.json"))["entries"]
        self.assertAlmostEqual(sum(e["usd"] for e in led if e["phase"] == "run"), 0.5)
        # resumable: a second invocation re-runs nothing
        n = len(r.calls)
        rc_, so, se = self.go(r, "calibrate")
        self.assertEqual(rc_, 0, so + se)
        self.assertEqual([s for s, _ in r.calls[n:]], [])

    def test_pass_at_once_needs_one_run_and_max_iterations_bounds_the_loop(self):
        r = thr.FakeRunner()
        rc_, so, se = self.go(r, "calibrate")
        self.assertEqual((rc_, len(r.cal_argvs)), (0, 1), so + se)
        # never converging: stops after max-iterations and says why
        r2 = thr.FakeRunner()
        r2.calibration_script = [{"ratio": 1.31, "clip_baseline": True, "recommended_cap": 2010},
                                 {"ratio": 1.20, "recommended_cap": 1900}, {"ratio": 0.95}]
        rc_, so, se = self.go(r2, "calibrate", "--max-iterations", "2", "--new")
        self.assertEqual(rc_, 1, so + se)
        self.assertEqual(len(r2.cal_argvs), 2)
        cal = H.read_json(os.path.join(self.camp(), "calibration.json"))
        self.assertEqual(cal["status"], "fail")
        self.assertIn("max iterations", cal["per_b"]["1536"]["stopped_because"])
        self.assertEqual(cal["per_b"]["1536"]["next_cap_suggestion"], 1900)
        self.assertEqual(cal["caps"], {})

    def test_a_clipped_nc_note_is_not_fixable_by_the_baseline_cap(self):
        r = thr.FakeRunner()
        r.calibration_script = [{"ratio": 0.95, "clip_nc": True}]
        rc_, so, se = self.go(r, "calibrate")
        self.assertEqual((rc_, len(r.cal_argvs)), (1, 1), so + se)
        cal = H.read_json(os.path.join(self.camp(), "calibration.json"))
        self.assertIn("NC note itself is clipped", cal["per_b"]["1536"]["stopped_because"])

    def test_next_cap_rules(self):
        f = H.next_calibration_cap
        self.assertEqual(f(1536, {"ratio": 1.31, "in_window": False, "recommended_cap": 2010, "clip": {}}), (2010, ""))
        # clipped but in window -> +10 %
        self.assertEqual(f(1536, {"ratio": 0.9, "in_window": True, "clip": {"baseline": {"clip_fail": True}}})[0], 1690)
        # clipped and the ratio says "shrink" -> still grow (clip first)
        self.assertEqual(f(1536, {"ratio": 0.7, "in_window": False, "recommended_cap": 1200,
                                  "clip": {"baseline": {"clip_fail": True}}})[0], 1690)
        self.assertIsNone(f(1536, {"ratio": None, "clip": {}})[0])
        self.assertIsNone(f(1536, {"ratio": 1.2, "in_window": False, "recommended_cap": 1536, "clip": {}})[0])

    def test_uncalibrated_b_use_the_calibrated_scale(self):
        r = thr.FakeRunner()
        r.calibration_script = [{"ratio": 1.31, "clip_baseline": True, "recommended_cap": 2010}, {"ratio": 0.97}]
        t = os.path.join(self.d.name, "t2.toml")
        open(t, "w").write("[oh]\nbatch_chains = 4\nbudgets = [1536, 1024]\nretries = 0\n")
        # calibrate only the primary
        rc_, so, se = self.go(r, "calibrate", "--budgets", "1536", toml=t)
        self.assertEqual(rc_, 0, so + se)
        rc_, so, se = self.go(r, "oh", toml=t)
        self.assertEqual(rc_, 0, so + se)
        plan = H.read_json(os.path.join(self.camp(), "oh", "caps.json"))
        self.assertEqual(plan["mode"], "calibrated")
        self.assertEqual(plan["caps"]["1536"], {"cap": 2010, "basis": "map", "source": plan["source"]})
        self.assertEqual((plan["caps"]["1024"]["cap"], plan["caps"]["1024"]["basis"]),
                         (round(1024 * 2010 / 1536), "scale"))
        self.assertTrue(plan["source"].startswith("calibrated:") and plan["source"].endswith("b1536-i2"))
        for a in self.runs(r):
            self.assertEqual(a[a.index("--baseline-cap") + 1], "1536:2010")
            self.assertAlmostEqual(float(a[a.index("--baseline-cap-scale") + 1]), 2010 / 1536, places=5)
            self.assertEqual(a[a.index("--baseline-cap-source") + 1], plan["source"])

    # ---- oh refuses / explicit
    def test_oh_refuses_without_a_calibration_and_spends_nothing(self):
        r = thr.FakeRunner()
        rc_, so, se = self.go(r, "oh")
        self.assertEqual(rc_, 2)
        self.assertIn("calibrat", se)
        self.assertIn("--baseline-cap", se)
        self.assertEqual(self.runs(r), [])
        self.assertFalse(os.path.exists(os.path.join(self.camp(), "ledger.json")))

    def test_oh_refuses_after_a_failed_calibration(self):
        r = thr.FakeRunner()
        r.calibration_script = [{"ratio": 1.31, "clip_baseline": True, "recommended_cap": 2010}]
        self.go(r, "calibrate", "--max-iterations", "1")
        rc_, so, se = self.go(r, "oh")
        self.assertEqual(rc_, 2)
        self.assertIn("FAILED", se)
        self.assertEqual(self.runs(r), [])

    def test_explicit_baseline_cap_runs_without_calibration_and_is_passed_through(self):
        r = thr.FakeRunner()
        rc_, so, se = self.go(r, "oh", "--baseline-cap", "1536:2010")
        self.assertEqual(rc_, 0, so + se)
        runs = self.runs(r)
        self.assertTrue(runs)
        for a in runs:
            self.assertEqual(a[a.index("--baseline-cap") + 1], "1536:2010")
            self.assertEqual(a[a.index("--baseline-cap-source") + 1], "cli")
            self.assertNotIn("--exclude-chains", a)                    # no calibration -> nothing to exclude
        plan = H.read_json(os.path.join(self.camp(), "oh", "caps.json"))
        self.assertEqual((plan["mode"], plan["origin"], plan["caps"]["1536"]["cap"]), ("explicit", "cli", 2010))
        # frozen per campaign
        rc_, so, se = self.go(r, "oh", "--baseline-cap", "1536:2200")
        self.assertEqual(rc_, 2)
        self.assertIn("FROZEN", se)

    def test_config_baseline_cap_and_scale_are_explicit(self):
        t = os.path.join(self.d.name, "t3.toml")
        open(t, "w").write('[oh]\nbatch_chains = 4\nbudgets = [1536, 1024]\nretries = 0\n'
                           'baseline_cap = "1536:2010"\nbaseline_cap_scale = 1.3\n')
        r = thr.FakeRunner()
        rc_, so, se = self.go(r, "oh", toml=t)
        self.assertEqual(rc_, 0, so + se)
        plan = H.read_json(os.path.join(self.camp(), "oh", "caps.json"))
        self.assertEqual((plan["origin"], plan["caps"]["1536"]["cap"], plan["caps"]["1024"]["cap"],
                          plan["caps"]["1024"]["basis"]), ("config", 2010, round(1024 * 1.3), "scale"))
        # TOML table form
        self.assertEqual(H.caps_to_str({1536: 2010, 1024: 1340}), "1536:2010,1024:1340")

    def test_explicit_map_missing_a_budget_without_scale_is_refused(self):
        r = thr.FakeRunner()
        t = os.path.join(self.d.name, "t4.toml")
        open(t, "w").write("[oh]\nbudgets = [1536, 1024]\nretries = 0\n")
        rc_, so, se = self.go(r, "oh", "--baseline-cap", "1536:2010", toml=t)
        self.assertEqual(rc_, 2)
        self.assertIn("no baseline cap for B=1024", se)

    # ---- exclusion
    def test_calibration_chain_is_excluded_from_run_merge_and_metrics(self):
        r = thr.FakeRunner()
        self.go(r, "calibrate")
        cal = H.read_json(os.path.join(self.camp(), "calibration.json"))
        cid = cal["chain_id"]
        rc_, so, se = self.go(r, "oh")
        self.assertEqual(rc_, 0, so + se)
        c = self.camp()
        ran = {json.loads(x)["chain_id"] for a in self.runs(r) for x in open(a[a.index("--chains") + 1])}
        self.assertEqual(ran, set(self.chain_ids()) - {cid})
        for a in self.runs(r):
            self.assertEqual(a[a.index("--exclude-chains") + 1], cid)
        merge = [a for s, a in r.calls if s == "merge_runs.py"][0]
        self.assertEqual(merge[merge.index("--exclude-chains") + 1], cid)
        metrics = [a for s, a in r.calls if s == "metrics_v2.py"][0]
        self.assertIn("--exclude-sid", metrics)
        self.assertEqual(metrics[metrics.index("--exclude-sid") + 1], f"^(?:{cid})$")
        recs = H.read_jsonl(os.path.join(c, "oh", "merged", "records.jsonl"))
        self.assertTrue(recs)
        self.assertNotIn(cid, {x["chain_id"] for x in recs})
        # the decision set file and the recorded exclusion
        self.assertEqual([x["chain_id"] for x in H.read_jsonl(os.path.join(c, "oh", "inputs", "chains.jsonl"))],
                         [x for x in self.chain_ids() if x != cid])
        self.assertEqual(len(H.read_jsonl(os.path.join(c, "oh", "inputs", "chains.all.jsonl"))), 4)
        self.assertEqual(H.read_json(os.path.join(c, "oh", "caps.json"))["excluded_from_decision_set"], [cid])
        self.assertEqual(H.read_json(os.path.join(c, "inputs", "oh_inputs.json"))["excluded_from_decision_set"], [cid])

    def test_merge_runs_refuses_a_leaked_calibration_chain(self):
        import subprocess
        d = self.d.name
        for name, cid in (("a", "chain_A"), ("b", "chain_CAL")):
            os.makedirs(os.path.join(d, name))
            thr_dir = os.path.join(d, name)
            rows = {"records.jsonl": [{"rid": f"{cid}1", "qid": f"{cid}_1", "chain_id": cid, "arm": "baseline",
                                       "valid": True, "mem_budget": 1536}],
                    "compactions.jsonl": [{"chain_id": cid, "arm": "baseline", "span_idx": 1, "mem_budget": 1536}],
                    "chain_budget.jsonl": [{"chain_id": cid, "mem_budget": 1536, "valid": True,
                                            "baseline_sum_max_tokens": 1536, "note_sum_max_tokens": 640}]}
            for fn, rs in rows.items():
                with open(os.path.join(thr_dir, fn), "w") as f:
                    f.write("".join(json.dumps(x) + "\n" for x in rs))
        p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "merge_runs.py"), "--out", os.path.join(d, "m"),
                            "--allow-mixed-config", "--exclude-chains", "chain_CAL",
                            os.path.join(d, "a"), os.path.join(d, "b")], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("calibration chain", p.stdout + p.stderr)

    # ---- smoke
    def test_smoke_calibrates_on_the_smoke_chain_without_excluding_it(self):
        os.environ["HR_TEST_KEY9"] = "k-" + "x" * 20
        r = thr.FakeRunner()
        r.calibration_script = [{"ratio": 1.31, "clip_baseline": True, "recommended_cap": 2010}, {"ratio": 0.97}]
        orig = r.chainreplay

        def chainreplay(argv):
            if "--dry-run" in argv:
                return 0, ("== chain_s1_c0\n  projected cost: a = $0.20\n== chain_s2_c0\n  projected cost: a = $0.30\n"
                           "TOTAL projected: 12 compactions, 80 asks, ~120.0 min, $0.5000\n")
            return orig(argv)
        r.chainreplay = chainreplay
        rc_, so, se = thr.quiet(H.main, ["--root", self.root, "--key-env", "HR_TEST_KEY9", "--oh-dir", self.art,
                                         "smoke", "--skip-ls", "--legacy-replay"], runner=r, probes=thr.FakeProbes())
        self.assertEqual(rc_, 0, so + se)
        camp = os.path.join(self.root, [x for x in os.listdir(self.root) if x.endswith("-smoke")][0])
        cal = H.read_json(os.path.join(camp, "calibration.json"))
        self.assertEqual((cal["status"], cal["caps"], cal["excluded_from_decision_set"]), ("pass", {"1536": 2010}, []))
        run = [a for s, a in r.calls if s == "run_chainreplay.py" and "--dry-run" not in a and "--calibrate" not in a][0]
        self.assertEqual(run[run.index("--baseline-cap") + 1], "1536:2010")      # the smoke run uses the calibrated cap
        rep = H.read_json(os.path.join(camp, "smoke-report.json"))
        self.assertTrue(all(x["ok"] for x in rep), rep)


if __name__ == "__main__":
    unittest.main()
