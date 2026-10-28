"""Amendment C.7 in home_run.py: `oh-probe` / `oh-enrich` against a FINISHED oh campaign (fake runner: run_longsession,
export, judge and metrics are stand-ins writing the real file shapes; chains_to_longsession and the guards run for real).
Covers: explicit campaign only, the EVAL-VERSION reuse rule, probe-before-enrich, no ingest in the estimate, new rows
only to the judge, merged verdicts, the parity stop, the judge-model check, ledger and artifacts."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import home_run as H
import longsession as L
import test_home_run as thr
import test_oh_c6 as t6
from oh_chain_fixture import make_artifact

OLD_ARMS = "baseline,nc-chunks,nc-graph,nc-note-only"


class EnrichRunner(t6.LSFakeRunner):
    """LSFakeRunner + the C.7 shapes: reuse-run / probe branches of run_longsession, rid-bearing export and judge."""

    def __init__(self, parity_ratio=0.95, judge_model=None, **kw):
        super().__init__(**kw)
        self.parity_ratio = parity_ratio
        self.judge_model = judge_model

    def longsession(self, argv):
        if "--dry-run" in argv:
            ss = L.load_sessions([self.arg(argv, "--sessions")])
            arms = self.arg(argv, "--arms").split(",") if "--arms" in argv else ["baseline"]
            res = {"sessions": [{"sid": x["sid"], "boundaries": 3, "hars": {"est_ingest_hours": 0.0 if "--reuse-run" in argv else 2.0},
                                 "per_budget": {"1536": {"asks": {a: len(x["questions"]) for a in arms}}}} for x in ss],
                   "grand_total_usd": 0.1}
            return 0, "log line\n" + json.dumps(res, indent=1)
        out = self.arg(argv, "--resume") if "--resume" in argv else self.arg(argv, "--out")
        os.makedirs(out, exist_ok=True)
        if "--recall-probe" in argv:
            json.dump({"hash": "p", "run_id": "ls-probe"}, open(os.path.join(out, "config.json"), "w"))
            json.dump({"n_questions": 7, "n_recoverable": 6, "all": {"A_raw": {"rate": 0.4}, "A_packed": {"rate": 0.2},
                                                                     "B_raw": {"rate": 0.8}, "C_old_cap": {"rate": 0.6},
                                                                     "D_old_cap": {"rate": 0.5}}},
                      open(os.path.join(out, "probe-summary.json"), "w"))
            open(os.path.join(out, "REPORT-probe.md"), "w").write("# probe report\n")
            json.dump({"budget_spent_all_invocations": 0.07, "failed_sessions": []}, open(os.path.join(out, "MANIFEST.json"), "w"))
            return 0, "probe done\n"
        if "--reuse-run" in argv:
            old = self.arg(argv, "--reuse-run")
            sessions = L.load_sessions([self.arg(argv, "--sessions")])
            old_rows = [json.loads(l) for l in open(os.path.join(old, "records.jsonl")) if l.strip()]
            recs = [dict(r, reused_from=old) for r in old_rows if r["arm"] in ("baseline", "nc-note-only")]
            for s in sessions:
                for q in s["questions"]:
                    recs.append({"rid": f"{q['qid']}::nc-enriched::b1536", "qid": q["qid"], "base_qid": q["qid"], "chain_id": s["sid"],
                                 "sid": s["sid"], "cluster_session": s["sid"], "arm": "nc-enriched", "mem_budget": 1536, "valid": True,
                                 "chain_invalid_reasons": [], "qtype": q["qtype"], "answerable": q["answerable"], "is_end": True,
                                 "checkpoint": 3, "memory_tokens": 1500 * self.parity_ratio, "cost_usd": 0.001})
            with open(os.path.join(out, "records.jsonl"), "a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            json.dump({"hash": "e", "run_id": "ls-enrich"}, open(os.path.join(out, "config.json"), "w"))
            json.dump({"budget_spent_all_invocations": 0.4, "failed_sessions": []}, open(os.path.join(out, "MANIFEST.json"), "w"))
            ok = 0.8 <= self.parity_ratio <= 1.05
            json.dump({"parity_enriched": {"1536": {"ratio_enriched_vs_baseline": self.parity_ratio, "in_window": ok,
                                                    "window": [0.8, 1.05]}}}, open(os.path.join(out, "summary.json"), "w"))
            with open(os.path.join(out, "done.jsonl"), "a") as f:
                for s in sessions:
                    for arm in ("baseline", "nc-note-only", "nc-enriched"):
                        f.write(json.dumps({"sid": s["sid"], "arm": arm, "B": 1536, "valid": True}) + "\n")
            return 0, f"done: {len(recs)} rows\n"
        return super().longsession(argv)

    def export(self, argv):
        recs = [json.loads(l) for l in open(self.arg(argv, "--records")) if l.strip()]
        skip = set()
        if "--skip-judged" in argv:
            skip = {json.loads(l)["rid"] for l in open(self.arg(argv, "--skip-judged")) if l.strip()}
        with open(self.arg(argv, "--out"), "w") as f:
            for r in recs:
                if r["rid"] not in skip:
                    f.write(json.dumps({"rid": r["rid"], "arm": r["arm"], "qid": r["qid"]}) + "\n")
        return 0, "exported\n"

    def judge(self, argv):
        rows = [json.loads(l) for l in open(self.arg(argv, "--in")) if l.strip()]
        self.judged_inputs = getattr(self, "judged_inputs", []) + [[r["rid"] for r in rows]]
        with open(self.arg(argv, "--out"), "w") as f:
            for r in rows:
                f.write(json.dumps({"rid": r["rid"], "score": 0.5, "pass": 1}) + "\n")
        jm = self.judge_model or H.P.resolve_models(H.P.load_profile("anthropic"), {})["judge"]
        json.dump({"judge_model": jm, "map": {}}, open(self.arg(argv, "--mapping"), "w"))
        return 0, "judge spend: $0.20000 over 12 calls (cap $9.00; profile anthropic)\n"


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.art = os.path.join(self.d.name, "oh-art")
        self.chains = make_artifact(self.art, 4)
        self.root = os.path.join(self.d.name, "root")
        self.base = ["--root", self.root, "--oh-dir", self.art, "--run-cap", "100", "--judge-cap", "100"]
        os.environ["NC_TOKCOUNT"] = "estimate"
        self.old_runner = EnrichRunner()
        with mock.patch.object(H, "OH_ARMS", OLD_ARMS):                    # the finished campaign: the question-time arms
            rc, so, se = thr.quiet(H.main, self.base + ["oh"], runner=self.old_runner)
        self.assertEqual(rc, 0, so + se)
        self.camp = open(os.path.join(self.root, ".homerun-current")).read()
        self.stamp("2026-10-27.c6")

    def tearDown(self):
        self.d.cleanup()

    def stamp(self, version):
        cj = H.read_json(os.path.join(self.camp, "campaign.json"))
        cj["eval_version"] = version
        H.write_json(os.path.join(self.camp, "campaign.json"), cj)
        cal = H.read_json(os.path.join(self.camp, "calibration.json"))
        if cal:
            cal["eval_version"] = version
            H.write_json(os.path.join(self.camp, "calibration.json"), cal)

    def go(self, runner, *cmd, resume=True, pre=()):
        cmd = list(cmd)
        if cmd and cmd[0] in ("oh-probe", "oh-enrich") and "--layout" not in cmd:
            cmd += ["--layout", "split-session"]       # C.8: these tests pin the (diagnostic) split-session path
        argv = self.base + list(pre) + cmd + (["--resume", self.camp] if resume else [])
        return thr.quiet(H.main, argv, runner=runner)

    def steps(self):
        return set(f[:-5] for f in os.listdir(os.path.join(self.camp, "steps")))


class Guards(Base):
    def test_both_commands_need_an_explicit_existing_campaign(self):
        for cmd in ("oh-probe", "oh-enrich"):
            rc, so, se = self.go(EnrichRunner(), cmd, resume=False)
            self.assertEqual(rc, 2, cmd)
            self.assertIn("EXISTING campaign", se)

    def test_a_c6_campaign_is_refused_by_every_other_paid_command_but_accepted_by_the_reuse_commands(self):
        r = EnrichRunner()
        rc, so, se = self.go(r, "oh")
        self.assertEqual(rc, 2)
        self.assertIn("EVAL-VERSION", se)
        self.assertIn("may reuse a 2026-10-27.c6", se)
        rc, so, se = self.go(r, "oh-probe")
        self.assertEqual(rc, 0, so + se)

    def test_other_versions_and_void_calibrations_are_still_refused(self):
        self.stamp("2026-10-26.c5")
        for cmd in ("oh-probe", "oh-enrich"):
            rc, so, se = self.go(EnrichRunner(), cmd, *(["--skip-probe"] if cmd == "oh-enrich" else []))
            self.assertEqual(rc, 2, cmd)
            self.assertIn("EVAL-VERSION", se)
            self.assertIn("Nothing was spent", se)
        self.stamp("2026-10-27.c6")
        cal = H.read_json(os.path.join(self.camp, "calibration.json"))
        cal["void"] = True
        H.write_json(os.path.join(self.camp, "calibration.json"), cal)
        rc, so, se = self.go(EnrichRunner(), "oh-probe")
        self.assertEqual(rc, 2)
        self.assertIn("VOID", se)

    def test_an_unfinished_old_run_is_not_reusable(self):
        import shutil
        shutil.rmtree(os.path.join(self.camp, "oh", "run"))
        rc, so, se = self.go(EnrichRunner(), "oh-probe")
        self.assertEqual(rc, 2)
        self.assertIn("not a finished `oh` run dir", se)


class Probe(Base):
    def test_probe_args_artifacts_and_ledger(self):
        r = EnrichRunner()
        rc, so, se = self.go(r, "oh-probe", "--probe-caps", "500,1000")
        self.assertEqual(rc, 0, so + se)
        runs = [a for s, a in r.calls if s == "run_longsession.py" and "--dry-run" not in a]
        self.assertEqual(len(runs), 1)
        a = runs[0]
        self.assertIn("--recall-probe", a)
        self.assertEqual(r.arg(a, "--reuse-run"), os.path.join(self.camp, "oh", "run"))
        self.assertEqual(r.arg(a, "--reuse-arms"), "baseline,nc-note-only")
        self.assertEqual(r.arg(a, "--probe-caps"), "500,1000")
        self.assertEqual(r.arg(a, "--arms"), "baseline,nc-enriched,nc-note-only")
        self.assertEqual(r.arg(a, "--out"), os.path.join(self.camp, "oh", "probe-run-split"))
        self.assertIn("oh.probe.split", self.steps())
        m = H.read_json(os.path.join(self.camp, "steps", "oh.probe.split.json"))
        self.assertEqual(m["n_questions"], 7)
        self.assertEqual(m["headline"]["B_raw"], 0.8)
        led = [e for e in H.read_json(os.path.join(self.camp, "ledger.json"))["entries"] if e["step"] == "oh.probe.split.run"]
        self.assertEqual(len(led), 1)
        self.assertIn("probe report", so)
        cj = H.read_json(os.path.join(self.camp, "campaign.json"))
        self.assertEqual(cj["eval_version"], "2026-10-27.c6")                # the campaign's own stamp is not rewritten
        self.assertEqual(cj["c7_reuse"]["oh-probe-split"]["eval_version"], H.HOME_RUN_VERSION)
        self.assertTrue(os.path.isfile(os.path.join(self.camp, "scripts", "c7", "run_longsession.py")))

    def test_probe_dry_run_spends_nothing_and_writes_no_step(self):
        r = EnrichRunner()
        rc, so, se = self.go(r, "oh-probe", "--dry-run")
        self.assertEqual(rc, 0, so + se)
        self.assertIn("estimate (zero spend)", so)
        self.assertNotIn("oh.probe.split", self.steps())
        self.assertFalse([1 for s, a in r.calls if s == "run_longsession.py" and "--dry-run" not in a])


class Enrich(Base):
    def test_the_probe_must_run_first(self):
        rc, so, se = self.go(EnrichRunner(), "oh-enrich")
        self.assertEqual(rc, 2)
        self.assertIn("oh-probe", se)
        self.assertIn("--skip-probe", se)

    def test_full_flow_judges_only_new_rows_and_merges_verdicts(self):
        self.assertEqual(self.go(EnrichRunner(), "oh-probe")[0], 0)
        r = EnrichRunner()
        old_judged = [json.loads(l) for l in open(os.path.join(self.camp, "oh", "run", "judged.jsonl")) if l.strip()]
        self.assertTrue(old_judged)
        rc, so, se = self.go(r, "oh-enrich")
        self.assertEqual(rc, 0, so + se)
        run = os.path.join(self.camp, "oh", "enrich-run-split")
        dry = [a for s, a in r.calls if s == "run_longsession.py" and "--dry-run" in a]
        real = [a for s, a in r.calls if s == "run_longsession.py" and "--dry-run" not in a]
        self.assertEqual((len(dry), len(real)), (1, 1))
        for a in dry + real:
            self.assertEqual(r.arg(a, "--reuse-run"), os.path.join(self.camp, "oh", "run"))
            self.assertEqual(r.arg(a, "--reuse-arms"), "baseline,nc-note-only")
            self.assertEqual(r.arg(a, "--arms"), "baseline,nc-enriched,nc-note-only")
            self.assertNotIn("--recall-probe", a)
        self.assertEqual(r.arg(real[0], "--out"), run)
        # the estimate is the reuse projection: only the new arm's asks are judge items, no ingest
        est = H.read_json(os.path.join(self.camp, "oh-enrich-estimate.json")) if os.path.exists(
            os.path.join(self.camp, "oh-enrich-split-estimate.json")) else H.read_json(os.path.join(self.camp, "steps", "oh.enrich.split.estimate.json"))["estimate"]
        n_q = sum(len(c["qa"]) for c in self.chains)
        self.assertEqual(est["judged_items"], n_q)
        self.assertEqual(est["ingest_hours"], 0.0)
        # judge: only the nc-enriched rows
        judged_in = r.judged_inputs[0]
        self.assertEqual(len(judged_in), n_q)
        self.assertTrue(all("::nc-enriched::" in rid for rid in judged_in))
        merged = [json.loads(l) for l in open(os.path.join(run, "judged.jsonl")) if l.strip()]
        arms = {x["rid"].split("::")[1] for x in merged}
        self.assertEqual(arms, {"baseline", "nc-note-only", "nc-enriched"})
        self.assertEqual(len(merged), 3 * n_q)
        scripts = [s for s, _ in r.calls if s in ("export_longsession_judge.py", "judge_v2.py", "metrics_v2.py")]
        self.assertEqual(scripts, ["export_longsession_judge.py", "export_longsession_judge.py", "judge_v2.py", "metrics_v2.py"])
        exp_new = [a for s, a in r.calls if s == "export_longsession_judge.py" and "--skip-judged" in a][0]
        self.assertEqual(r.arg(exp_new, "--skip-judged"), os.path.join(self.camp, "oh", "run", "judged.jsonl"))
        for st in ("oh.enrich.split.run", "oh.enrich.split.export", "oh.enrich.split.export_new", "oh.enrich.split.judge", "oh.enrich.split.merge", "oh.enrich.split.metrics"):
            self.assertIn(st, self.steps())
        led = {e["step"] for e in H.read_json(os.path.join(self.camp, "ledger.json"))["entries"]}
        self.assertTrue({"oh.enrich.split.run", "oh.enrich.split.judge"} <= led)
        self.assertTrue(os.path.isfile(os.path.join(run, "home_run-summary.json")))
        mt = [a for s, a in r.calls if s == "metrics_v2.py"][0]
        self.assertEqual(r.arg(mt, "--judge"), os.path.join(run, "judged.jsonl"))
        self.assertEqual(r.arg(mt, "--track"), "longsession")
        # idempotent: a second call skips every finished step
        n = len(r.calls)
        self.assertEqual(self.go(r, "oh-enrich")[0], 0)
        self.assertEqual([s for s, _ in r.calls[n:]], [])

    def test_dry_run_prints_the_reuse_estimate_and_changes_nothing(self):
        r = EnrichRunner()
        rc, so, se = self.go(r, "oh-enrich", "--dry-run")
        self.assertEqual(rc, 0, so + se)
        self.assertIn("NO ingest / extraction", so)
        self.assertFalse(any(s in ("export_longsession_judge.py", "judge_v2.py") for s, _ in r.calls))
        self.assertNotIn("oh.enrich.split.estimate", self.steps())

    def test_a_parity_miss_stops_before_any_judge_spend(self):
        self.assertEqual(self.go(EnrichRunner(), "oh-probe")[0], 0)
        r = EnrichRunner(parity_ratio=0.5)
        rc, so, se = self.go(r, "oh-enrich")
        self.assertEqual(rc, 2, so + se)
        self.assertIn("parity FAILED", se)
        self.assertNotIn("judge_v2.py", [s for s, _ in r.calls])
        rc, so, se = self.go(r, "oh-enrich", "--accept-parity-fail")
        self.assertEqual(rc, 0, so + se)
        self.assertIn("judge_v2.py", [s for s, _ in r.calls])

    def test_a_different_judge_model_is_refused(self):
        self.assertEqual(self.go(EnrichRunner(), "oh-probe")[0], 0)
        mp = os.path.join(self.camp, "oh", "run", "judge-mapping.json")
        json.dump({"judge_model": "some-other-judge", "map": {}}, open(mp, "w"))
        r = EnrichRunner()
        rc, so, se = self.go(r, "oh-enrich")
        self.assertEqual(rc, 2)
        self.assertIn("two judges", se)
        self.assertNotIn("run_longsession.py", [s for s, a in r.calls if "--dry-run" not in a])

    def test_skip_probe_is_allowed_explicitly(self):
        r = EnrichRunner()
        rc, so, se = self.go(r, "oh-enrich", "--skip-probe", "--skip-judge")
        self.assertEqual(rc, 0, so + se)
        self.assertIn("--skip-probe", so)
        self.assertNotIn("judge_v2.py", [s for s, _ in r.calls])

    def test_enrich_queries_and_packer_are_passed_through(self):
        r = EnrichRunner()
        rc, so, se = self.go(r, "oh-enrich", "--skip-probe", "--skip-judge", "--enrich-queries", "5", "--packer", "legacy")
        self.assertEqual(rc, 0, so + se)
        a = [a for s, a in r.calls if s == "run_longsession.py" and "--dry-run" not in a][0]
        self.assertEqual((r.arg(a, "--enrich-queries"), r.arg(a, "--packer")), ("5", "legacy"))


class FreshOhDefaults(unittest.TestCase):
    def test_default_oh_arms_are_the_c7_set_and_the_version_is_bumped(self):
        self.assertEqual(H.OH_ARMS, "baseline,nc-enriched,nc-note-only")
        self.assertEqual(H.HOME_RUN_VERSION, "2026-10-28.c9")
        self.assertEqual(open(os.path.join(os.path.dirname(H.BASE), "EVAL-VERSION")).read().strip(), H.HOME_RUN_VERSION)
        self.assertEqual(H.REUSE_COMPATIBLE_FROM, ("2026-10-27.c6", "2026-10-28.c7", "2026-10-28.c8"))


if __name__ == "__main__":
    unittest.main()
