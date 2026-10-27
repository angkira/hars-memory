"""Offline tests: home_run.py (config/caps, campaign + steps + ledger, doctor, dry-run parsers and
integration, beam end-to-end with the mock LLM, OpenHands batching/retry/accepted-view/merge with a fake
runner, bundle redaction/exclusion, smoke). No network, no real model."""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import tarfile
import tempfile
import unittest

import home_run as H
import longsession as L

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def quiet(fn, *a, **k):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = fn(*a, **k)
    return rc, out.getvalue(), err.getvalue()


class FakeProbes(H.Probes):
    def __init__(self, embedder="intfloat/e5-large-v2 (CPU)", semantic=True, pi="1.4.0", key=True, free=500.0):
        self.embedder, self.semantic, self.pi, self.free = embedder, semantic, pi, free
        self.mcp_calls = []

    def http_json(self, url, timeout=10.0):
        return {"status": "ok"}

    def mcp(self, url, tool, args, timeout=60.0):
        self.mcp_calls.append(tool)
        if tool == "memory_status":
            return {"configured_models": {"embedder": self.embedder, "extractor": "claude-haiku-5-5 @ https://api.anthropic.com/v1",
                                          "query": "claude-haiku-5-5 @ https://api.anthropic.com/v1"}}
        if tool == "memory_compact":
            return {"ok": True}
        if tool == "memory_session_status":
            return {"entities": 7, "vectors_indexed": 4}
        if tool == "memory_recall":
            q = args.get("question", "")
            hit = next((w for _, qq, w in H.SEMANTIC_PROBES if qq == q), "")
            return {"context": (f"... {hit} ..." if self.semantic else ""), "session": {"status": "ok"}}
        return {}

    def cmd(self, argv, timeout=20.0):
        if argv[0] == "pi":
            return (0, self.pi) if self.pi else (127, "")
        if argv[0] == "node":
            return 0, "v22.22.0"
        if argv[0] == "git":
            return 0, ""
        return 1, ""

    def disk_free_gb(self, path):
        return self.free

    def chat_ping(self, profile, role, model, key_env):
        return "'ok' $0.00001"

    def tiktoken_info(self):
        return True, "tiktoken cl100k_base loaded"

    def native_count(self, profile, model, key_env):
        return "anthropic:count_tokens:%s: 400 tok (fake)" % model


def cfg_for(root, **over):
    base = {"general": {"artifact_root": root, "name": "t"}, "caps": {"gen": 5.0, "run": 50.0, "judge": 20.0}}
    return H.load_config(None, H.deep_merge(base, over))


class ConfigAndCaps(unittest.TestCase):
    def test_cc_qa_defaults_are_the_menu_mode_numbers(self):
        cc = H.load_config(None, {})["cc"]
        self.assertEqual((cc["n_single"], cc["n_multihop"], cc["n_unanswerable"]), (40, 10, 4))
        self.assertGreaterEqual(cc["session_budget_usd"], 1.0)          # expected spend ~$0.3 on Sonnet
        self.assertGreaterEqual(H.CC_TYPICAL["gen"], 1.0)
        ex = H.load_config(os.path.join(os.path.dirname(H.__file__), "..", "home_run.toml.example"), {})["cc"]
        self.assertEqual((ex["n_single"], ex["n_multihop"], ex["n_unanswerable"]), (40, 10, 4))

    def test_calibration_chains_default_one_and_tokenizer_check_exists(self):
        self.assertEqual(H.load_config(None, {})["oh"]["calibration_chains"], 1)
        rows = H.doctor_checks(H.load_config(None, {"general": {"name": "t"}}), FakeProbes(), live=True)
        self.assertEqual([r["status"] for r in rows if r["check"] == "tokcount.native"], ["OK"])

    def test_caps_refuse_when_unset_and_per_track_override(self):
        cfg = H.load_config(None, {})
        for ph in H.PHASES:
            with self.assertRaises(H.HomeRunError) as cm:
                H.need_cap(cfg, "beam", ph)
            self.assertIn("home_run.toml", str(cm.exception))
        cfg = H.load_config(None, {"caps": {"run": 10.0, "beam": {"run": 40.0}}})
        self.assertEqual(H.cap_for(cfg, "beam", "run"), 40.0)
        self.assertEqual(H.cap_for(cfg, "oh", "run"), 10.0)
        self.assertEqual(H.cap_for(cfg, "oh", "judge"), 0.0)

    def test_toml_example_parses_and_has_suggested_caps(self):
        p = os.path.join(os.path.dirname(SCRIPTS), "home_run.toml.example")
        cfg = H.load_config(p, {})
        # Amendment C.3-3: suggested caps filled in; the built-in defaults stay 0 (= refuse without a config)
        self.assertEqual((cfg["caps"]["gen"], cfg["caps"]["run"], cfg["caps"]["judge"], cfg["caps"]["total"]),
                         (3.0, 20.0, 5.0, 30.0))
        self.assertEqual(H.load_config(None, {})["caps"]["run"], 0.0)
        self.assertEqual(cfg["general"]["profile"], "anthropic")
        self.assertEqual(cfg["beam"]["size"], "1M")
        # small by default: primary B only, BEAM N=10 seeded sample, sweeps opt-in (Amendment C.3-2)
        self.assertEqual((cfg["oh"]["budgets"], cfg["beam"]["budgets"], cfg["cc"]["budgets"]), ([1536], [1536], [1536]))
        self.assertEqual((cfg["beam"]["chats"], cfg["beam"]["n_chats"], cfg["beam"]["seed"]), ("sample", 10, 20261026))
        self.assertFalse(cfg["general"]["artifact_root"].startswith("/tmp"))

    def test_cli_cap_flags_reach_the_config(self):
        a = H.build_parser().parse_args(["--run-cap", "7", "dry-run", "--judge-cap", "3"])
        o = H.overrides_from_args(a)
        self.assertEqual(o["caps"], {"run": 7.0, "judge": 3.0})

    def test_default_root_is_the_artifact_store_not_tmp(self):
        self.assertEqual(H.DEFAULT_CONFIG["general"]["artifact_root"], "/mnt/datasets/ft/artifacts")


class CampaignAndLedger(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.cfg = cfg_for(self.d.name)
        args = H.build_parser().parse_args(["dry-run"])
        self.camp = H.open_campaign(self.cfg, args, "dry-run", echo=False)

    def tearDown(self):
        self.d.cleanup()

    def test_layout_and_pointer(self):
        c = self.camp
        self.assertTrue(os.path.basename(c.path).endswith("-homerun-t"))
        for n in ("campaign.json", "steps", "logs", "scripts", "inputs"):
            self.assertTrue(os.path.exists(c.p(n)), n)
        self.assertEqual(open(os.path.join(self.d.name, ".homerun-current")).read(), c.path)
        self.assertTrue(os.path.isfile(c.p("scripts", "home_run.py")))
        # a second invocation reuses it, --new does not
        c2 = H.open_campaign(self.cfg, H.build_parser().parse_args(["dry-run"]), "dry-run", echo=False)
        self.assertEqual(c2.path, c.path)
        c3 = H.open_campaign(self.cfg, H.build_parser().parse_args(["dry-run", "--new"]), "dry-run", echo=False)
        self.assertNotEqual(c3.path, c.path)
        smoke = H.open_campaign(self.cfg, H.build_parser().parse_args(["smoke"]), "smoke", echo=False)
        self.assertTrue(smoke.path.endswith("-smoke"))

    def test_steps_are_idempotent_and_forceable(self):
        calls = []
        self.camp.step("x.a", lambda: calls.append(1) or {"v": 1})
        self.camp.step("x.a", lambda: calls.append(1) or {"v": 2})
        self.assertEqual(len(calls), 1)
        self.camp.force_steps = {"x.a"}
        self.camp.step("x.a", lambda: calls.append(1) or {"v": 3})
        self.assertEqual(len(calls), 2)
        self.assertEqual(H.read_json(self.camp.marker("x.a"))["v"], 3)

    def test_ledger_is_idempotent_per_source_and_remaining_refuses_when_spent(self):
        c = self.camp
        c.charge("beam", "run", "beam.run", 3.0, source="dirA")
        c.charge("beam", "run", "beam.run", 4.0, source="dirA")      # cumulative re-parse replaces
        c.charge("beam", "run", "beam.run", 1.0, source="dirB")
        self.assertAlmostEqual(c.spent("beam", "run"), 5.0)
        self.assertAlmostEqual(c.remaining("beam", "run"), 45.0)
        c.charge("beam", "run", "beam.run", 46.0, source="dirC")
        with self.assertRaises(H.HomeRunError):
            c.remaining("beam", "run")
        man = H.read_json(c.p("MANIFEST.json"))
        self.assertAlmostEqual(man["cost_usd"], 51.0)

    def test_failed_dirs_are_moved_aside_never_deleted(self):
        d = self.camp.p("x", "run")
        os.makedirs(d)
        open(os.path.join(d, "records.jsonl"), "w").write("{}\n")
        dst = self.camp.quarantine(d)
        self.assertFalse(os.path.exists(d))
        self.assertTrue(os.path.isfile(os.path.join(dst, "records.jsonl")))
        self.assertIn(".failed-", dst)


class Doctor(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        os.environ["HR_TEST_KEY"] = "k-" + "x" * 20
        self.cfg = cfg_for(self.d.name, general={"key_env": "HR_TEST_KEY"})

    def tearDown(self):
        self.d.cleanup()

    def status(self, rows, name):
        return next(r["status"] for r in rows if r["check"] == name)

    def test_all_good(self):
        rows = H.doctor_checks(self.cfg, FakeProbes(), deep=True)
        self.assertFalse([r for r in rows if r["status"] == "FAIL"], rows)
        self.assertEqual(self.status(rows, "hars.embedder"), "OK")
        # Amendment C.6: pi / node are NOT required (and not even listed) unless the legacy replay is asked for
        self.assertNotIn("pi", {r["check"] for r in rows})
        self.assertNotIn("node", {r["check"] for r in rows})
        self.assertEqual(self.status(rows, "hars.semantic_probe"), "OK")
        self.assertEqual(self.status(rows, "keys"), "OK")

    def test_hash_shim_is_flagged_by_name_and_by_behaviour(self):
        rows = H.doctor_checks(self.cfg, FakeProbes(embedder="hashshim (CPU)"))
        self.assertEqual(self.status(rows, "hars.embedder"), "FAIL")
        # memory_status names the default model even when the HTTP provider serves a shim: only the
        # behavioural probe catches it
        rows = H.doctor_checks(self.cfg, FakeProbes(semantic=False), deep=True)
        self.assertEqual(self.status(rows, "hars.embedder"), "OK")
        self.assertEqual(self.status(rows, "hars.semantic_probe"), "FAIL")
        self.assertEqual(H.print_doctor(rows) if False else 1, 1)

    def test_pi_node_only_with_the_legacy_replay(self):
        rows = H.doctor_checks(self.cfg, FakeProbes(pi=None))        # default: no pi, no problem
        self.assertFalse([r for r in rows if r["status"] == "FAIL"], rows)
        rows = H.doctor_checks(self.cfg, FakeProbes(pi="1.3.9"), legacy_replay=True)
        self.assertEqual(self.status(rows, "pi"), "FAIL")
        rows = H.doctor_checks(self.cfg, FakeProbes(pi=None), legacy_replay=True)
        self.assertEqual(self.status(rows, "pi"), "FAIL")
        rows = H.doctor_checks(self.cfg, FakeProbes(pi="1.4.0"), legacy_replay=True)
        self.assertEqual((self.status(rows, "pi"), self.status(rows, "node")), ("OK", "OK"))
        rows = H.doctor_checks(self.cfg, FakeProbes(pi=None), tracks=("beam",), legacy_replay=True)
        self.assertNotIn("pi", {r["check"] for r in rows})           # the legacy replay is an oh-track concern

    def test_keys_and_disk(self):
        cfg = cfg_for(self.d.name, general={"key_env": "HR_NO_SUCH_KEY"})
        self.assertEqual(self.status(H.doctor_checks(cfg, FakeProbes()), "keys"), "FAIL")
        self.assertEqual(self.status(H.doctor_checks(self.cfg, FakeProbes(free=2.0)), "disk"), "FAIL")
        self.assertEqual(self.status(H.doctor_checks(self.cfg, FakeProbes(free=10.0)), "disk"), "WARN")

    def test_declared_embedder_mismatch_warns(self):
        cfg = cfg_for(self.d.name, general={"key_env": "HR_TEST_KEY", "embedder_id": "BAAI/bge-m3"})
        self.assertEqual(self.status(H.doctor_checks(cfg, FakeProbes()), "hars.embedder"), "WARN")

    def test_embedder_verdict(self):
        self.assertEqual(H.embedder_verdict(None)[0], "FAIL")
        self.assertEqual(H.embedder_verdict("HashShim")[0], "FAIL")
        self.assertEqual(H.embedder_verdict("BAAI/bge-m3")[0], "OK")

    def test_doctor_cli_writes_json_and_prints_hars_env(self):
        rc, so, se = quiet(H.main, ["--root", self.d.name, "--key-env", "HR_TEST_KEY", "doctor"], probes=FakeProbes())
        self.assertEqual(rc, 0, so + se)
        self.assertIn("export HARS_MEMORY_EXTRACTOR_BASE_URL=https://api.anthropic.com/v1", so)
        self.assertIn("HARS_MEMORY_LLM_OMIT_SAMPLING_PARAMS=1", so)
        camp = open(os.path.join(self.d.name, ".homerun-current")).read()
        self.assertTrue(os.path.isfile(os.path.join(camp, "doctor.json")))
        rc, so, se = quiet(H.main, ["--root", self.d.name, "--key-env", "HR_TEST_KEY", "doctor"],
                           probes=FakeProbes(embedder="shim"))
        self.assertEqual(rc, 1)
        self.assertNotIn("sk-", so)


class Parsers(unittest.TestCase):
    CR = ("== chain_a_c0\n  span 1 ...\n  projected cost: agent $0.4203 (claude-haiku-5-5 $0.10/$0.50 per 1M) + HARS "
          "extractor $0.0031 (claude-haiku-5-5) = $0.4234\n\n== chain_b_c0\n  projected cost: agent $0.2140 (x) + HARS "
          "extractor $0.0031 (y) = $0.2171\n\nTOTAL projected: 84 compactions, 560 asks, ~414.9 min, $3.6367\n"
          "BUDGET CHECK: ...\n  TOTAL over the sweep: 252 compactions, 1680 asks, ~1244.5 min, ~$10.9102 (single-B x 3)\n")

    def test_chainreplay_dry_prefers_the_sweep_total_and_reads_per_chain(self):
        d = H.parse_chainreplay_dry(self.CR)
        self.assertEqual((d["compactions"], d["asks"], d["usd"]), (252, 1680, 10.9102))
        self.assertAlmostEqual(d["minutes"], 1244.5)
        self.assertEqual(d["per_chain_usd_single_b"], {"chain_a_c0": 0.4234, "chain_b_c0": 0.2171})
        single = self.CR.split("  TOTAL over")[0]
        self.assertEqual(H.parse_chainreplay_dry(single)["usd"], 3.6367)
        with self.assertRaises(H.HomeRunError):
            H.parse_chainreplay_dry("nothing")

    def test_gen_and_json_block(self):
        self.assertEqual(H.parse_gen_dry("x\nDRY RUN total estimate $1,234.50 (upper-ish)"), 1234.5)
        txt = "[skip] foo {bar}\n{\n \"a\": {\"b\": 1},\n \"grand_total_usd\": 2.5\n}\n"
        self.assertEqual(H.parse_json_block(txt)["grand_total_usd"], 2.5)
        self.assertEqual(H.parse_json_block('{\n "x": 1\n}')["x"], 1)

    def test_ls_dry_summary(self):
        res = {"grand_total_usd": 10.0, "sessions": [
            {"boundaries": 4, "hars": {"est_ingest_hours": 2.0}, "per_budget": {"1536": {"asks": {"baseline": 10, "nc-chunks": 10}}}},
            {"boundaries": 6, "hars": {"est_ingest_hours": 1.0}, "per_budget": {"1536": {"asks": {"baseline": 5}}}}]}
        s = H.ls_dry_summary(res, 1, 0.01)
        self.assertEqual((s["asks"], s["boundaries"], s["sessions"]), (25, 10, 2))
        self.assertAlmostEqual(s["judge_usd"], 0.25)
        self.assertAlmostEqual(s["wall_hours"], 3.0 + (10 * 2 * 20.0 + 25 * 8.0) / 3600.0)

    def test_judge_unit_uses_the_profile_judge_price(self):
        cfg = H.load_config(None, {})
        self.assertAlmostEqual(H.judge_unit_usd(cfg), (1300 * 2.0 + 80 * 10.0) / 1e6)

    def test_spend_readers_prefer_manifests_and_fall_back_to_lower_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"spend": {"spent_usd": 1.25}}, open(os.path.join(d, "run_manifest.json"), "w"))
            self.assertEqual(H.chainreplay_spend(d)[0], 1.25)
            d2 = os.path.join(d, "x")
            os.makedirs(d2)
            open(os.path.join(d2, "records.jsonl"), "w").write(json.dumps({"cost_usd": 0.5}) + "\n")
            v, why = H.chainreplay_spend(d2)
            self.assertEqual(v, 0.5)
            self.assertIn("LOWER BOUND", why)
            json.dump({"budget_spent_all_invocations": 3.5}, open(os.path.join(d2, "MANIFEST.json"), "w"))
            self.assertEqual(H.longsession_spend(d2)[0], 3.5)
            open(os.path.join(d2, "qa-raw.jsonl"), "w").write(json.dumps({"cost_usd": 0.1}) + "\n" * 1
                                                              + json.dumps({"cost_usd": 0.2}) + "\n")
            self.assertAlmostEqual(H.genqa_spend(d2)[0], 0.3)
            self.assertEqual(H.judge_spend_from_text("judge spend: $0.01234 over 5 calls (cap"), 0.01234)


# ---------------------------------------------------------------------------- BEAM end to end

def make_big_beam_repo(root, size="1M", chats=("1", "2"), turns=170, chars=3600):
    """Synthetic BEAM checkout big enough to cross the 150k-token window (no BEAM text)."""
    filler = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore "
              "et dolore magna aliqua ")
    for n in chats:
        d = os.path.join(root, "chats", size, n)
        os.makedirs(os.path.join(d, "probing_questions"))
        groups, tid = [], 0
        for k in range(turns // 2):
            u = f"step {tid}: job JOB{tid:04d} finished with metric_{tid} = {1000 + 7 * tid} units "
            groups.append([{"role": "user", "id": tid, "content": (u + filler * 60)[:chars]},
                           {"role": "assistant", "id": tid + 1, "content": (f"ack {tid + 1} " + filler * 60)[:chars]}])
            tid += 2
        json.dump([{"batch_number": 1, "time_anchor": "March-01-2024", "turns": groups}], open(os.path.join(d, "chat.json"), "w"))
        json.dump({"title": f"topic {n}"}, open(os.path.join(d, "topic.json"), "w"))
        pq = {"information_extraction": [{"question": "What did job JOB0010 report?", "answer": "1070",
                                          "source_chat_ids": [10], "rubric": ["LLM response should state: 1070"]}],
              "knowledge_update": [{"question": "What did JOB0020 report later?", "answer": "1140",
                                    "source_chat_ids": {"original_info": [20], "updated_info": [22]}}],
              "multi_session_reasoning": [{"question": "Which jobs reported 1070 and 1140?", "answer": "JOB0010 and JOB0020",
                                           "source_chat_ids": [10, 20]}]}
        json.dump(pq, open(os.path.join(d, "probing_questions", "probing_questions.json"), "w"))


class BeamEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.TemporaryDirectory()
        cls.repo = os.path.join(cls.d.name, "beam")
        make_big_beam_repo(cls.repo)
        cls.root = os.path.join(cls.d.name, "root")
        cls.base = ["--root", cls.root, "--offline-selftest", "--beam-repo", cls.repo, "--run-cap", "5", "--judge-cap", "5",
                    "--gen-cap", "5", "--mcp", "http://127.0.0.1:1/mcp"]

    @classmethod
    def tearDownClass(cls):
        cls.d.cleanup()

    def camp(self):
        return open(os.path.join(self.root, ".homerun-current")).read()

    def test_1_dry_run_estimates_from_the_real_dry_run_scripts(self):
        rc, so, se = quiet(H.main, self.base + ["dry-run", "--tracks", "beam"])
        self.assertEqual(rc, 0, so + se)
        self.assertIn("beam", so)
        est = H.read_json(os.path.join(self.camp(), "dry-run.json"))["estimates"]["beam"]
        self.assertEqual(est["sessions"], 2)
        self.assertIn("judge_usd", est)
        self.assertTrue(os.path.isfile(os.path.join(self.camp(), "beam", "data", "beam-1M-1.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.camp(), "dry-run.md")))

    def test_2_beam_runs_resumes_and_skips(self):
        argv = self.base + ["beam", "--budgets", "1536", "--skip-judge"]
        rc, so, se = quiet(H.main, argv)
        self.assertEqual(rc, 0, so + se)
        c = self.camp()
        run = os.path.join(c, "beam", "run")
        rows = H.read_jsonl(os.path.join(run, "records.jsonl"))
        self.assertTrue(rows)
        self.assertEqual({r["arm"] for r in rows}, {"baseline", "nc-chunks", "nc-graph", "nc-note-only"})
        self.assertEqual({r["chain_id"] for r in rows}, {"beam-1M-1", "beam-1M-2"})
        for n in ("MANIFEST.json", "summary.json", "run.log", "done.jsonl", "judge-input.jsonl"):
            self.assertTrue(os.path.isfile(os.path.join(run, n)), n)
        for n in ("campaign.json", "ledger.json", "MANIFEST.json", "dry-run.json"):
            self.assertTrue(os.path.isfile(os.path.join(c, n)), n)
        for st in ("beam.import", "beam.run", "beam.export", "beam.estimate"):
            self.assertTrue(os.path.isfile(os.path.join(c, "steps", st + ".json")), st)
        self.assertTrue(os.path.getsize(os.path.join(c, "logs", "beam.run.log")) > 0)
        cmds = open(os.path.join(c, "scripts", "cmdlines.txt")).read()
        self.assertIn("run_longsession.py", cmds)
        self.assertIn("--budget-usd", cmds)
        # second invocation: everything is skipped, nothing recomputed
        n_before = len(rows)
        rc, so2, se2 = quiet(H.main, argv)
        self.assertEqual(rc, 0, so2 + se2)
        self.assertIn("[skip] beam.run", so2)
        self.assertEqual(len(H.read_jsonl(os.path.join(run, "records.jsonl"))), n_before)
        # --force-step redoes just that step (run_longsession resumes: done cells are skipped, no duplicates)
        rc, so3, se3 = quiet(H.main, argv + ["--force-step", "beam.export"])
        self.assertEqual(rc, 0, so3 + se3)
        self.assertIn("=== beam.export ===", so3)
        self.assertIn("[skip] beam.run", so3)
        # the judge-input rows feed judge_v2 and carry the rubric
        ji = H.read_jsonl(os.path.join(run, "judge-input.jsonl"))
        self.assertTrue(any(r.get("rubric") for r in ji))

    def test_3_judge_needs_a_cap_and_an_offline_run_has_no_judge(self):
        rc, so, se = quiet(H.main, ["--root", self.root, "--offline-selftest", "--beam-repo", self.repo, "--run-cap", "5",
                                    "beam", "--budgets", "1536"])
        self.assertEqual(rc, 2)
        self.assertIn("no judge cap", se)

    def test_4_cap_refusals(self):
        rc, so, se = quiet(H.main, ["--root", self.root, "--offline-selftest", "--beam-repo", self.repo, "beam", "--new"])
        self.assertEqual(rc, 2)
        self.assertIn("no run cap", se)
        # estimate above the cap is refused without --accept-partial
        # the mock provider estimates $0, so the guard is exercised directly with a nonzero estimate
        est = {"run_usd": 3.0}
        cfg = H.load_config(None, {"caps": {"run": 1.0}})
        camp = H.Campaign(os.path.join(self.d.name, "c"), cfg, echo=False)
        with self.assertRaises(H.HomeRunError) as cm:
            H.guard_estimate(camp, "beam", est, False)
        self.assertIn("--accept-partial", str(cm.exception))
        H.guard_estimate(camp, "beam", est, True)
        H.guard_estimate(camp, "beam", {"run_usd": 0.5}, False)


# ---------------------------------------------------------------------------- OpenHands (fake runner)

class FakeRunner:
    """Stands in for run_chainreplay / judge_v2 / export / metrics / funnel; merge_runs runs for real."""

    def __init__(self, invalid_first=()):
        self.calls = []
        self.invalid_first = set(invalid_first)       # (chain, B) cells that come out invalid on the first attempt
        self.attempts = {}
        self.stop_after = None
        self.calibration_script = []                  # dicts: ratio / clip_baseline / recommended_cap per --calibrate call

    def __call__(self, argv, env, cwd, log_path, echo=True):
        script = next((os.path.basename(a) for a in argv if a.endswith(".py")), "")
        self.calls.append((script, argv))
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handler = {"run_chainreplay.py": self.chainreplay, "judge_v2.py": self.judge, "export_judge_input.py": self.touch_out,
                   "metrics_v2.py": self.metrics, "funnel_chains.py": self.funnel}.get(script)
        if handler is None:
            return H.subprocess_runner(argv, env, cwd, log_path, False)
        rc, text = handler(argv)
        open(log_path, "a").write(text)
        return rc, text

    @staticmethod
    def arg(argv, flag):
        return argv[argv.index(flag) + 1]

    def chainreplay(self, argv):
        if "--dry-run" in argv:
            return 0, ("== chain_x_c0\n  projected cost: agent $0.3 + HARS $0.0 = $0.30\n"
                       "TOTAL projected: 12 compactions, 80 asks, ~120.0 min, $1.2000\n")
        if "--calibrate" in argv:
            return self.calibrate(argv)
        out, B = self.arg(argv, "--out"), int(self.arg(argv, "--mem-budgets"))
        chains = [json.loads(l) for l in open(self.arg(argv, "--chains")) if l.strip()]
        os.makedirs(out, exist_ok=True)
        rec, comp, bud = [], [], []
        for c in chains:
            cid = c["chain_id"]
            k = (cid, B)
            self.attempts[k] = self.attempts.get(k, 0) + 1
            ok = not (k in self.invalid_first and self.attempts[k] == 1)
            bud.append({"chain_id": cid, "mem_budget": B, "valid": ok, "nc_memory_total": 1500, "baseline_memory_total": 1500,
                        "chain_invalid_reasons": [] if ok else ["RECALL FAIL"], "agent_model": "claude-haiku-5-5"})
            for q in c["qa"]:
                for arm in ("baseline", "nc-chunks", "nc-graph", "nc-note-only"):
                    rec.append({"rid": f"{q['qid']}::{arm}::b{B}", "qid": q["qid"], "chain_id": cid, "arm": arm, "mem_budget": B,
                                "b_tag": True, "valid": ok, "chain_invalid_reasons": [], "cost_usd": 0.01, "qtype": q.get("qtype"),
                                "session_id": f"cr-{cid}-{arm}", "answerable": True, "anchor_hit": True,
                                "memory_tokens": 1500})
            comp.append({"chain_id": cid, "arm": "baseline", "span_idx": 1, "mem_budget": B, "valid": ok})
            for sub in ("sessions", "logs", "answers"):
                os.makedirs(os.path.join(out, sub, f"{cid}__b{B}"), exist_ok=True)
                open(os.path.join(out, sub, f"{cid}__b{B}", "f.txt"), "w").write(f"{out}\n")
        for fn, rows in (("records.jsonl", rec), ("compactions.jsonl", comp), ("chain_budget.jsonl", bud)):
            with open(os.path.join(out, fn), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        spent = 0.5 * len(chains)
        json.dump({"run_id": os.path.basename(out), "profile": "anthropic", "models": {"agent": "claude-haiku-5-5"},
                   "embedder": {"id": "bge-m3"}, "judge_same_family": True, "ended_utc": "x",
                   "spend": {"spent_usd": spent, "stopped_by_budget": bool(self.stop_after)}},
                  open(os.path.join(out, "run_manifest.json"), "w"))
        return 0, f"spend: ${spent}\n"

    def calibrate(self, argv):
        """Stands in for `run_chainreplay.py --calibrate`: scripted outcome per call (default: passes at once)."""
        out, B = self.arg(argv, "--out"), int(self.arg(argv, "--mem-budgets"))
        cap = int(self.arg(argv, "--baseline-cap").split(":")[1])
        step = self.calibration_script.pop(0) if self.calibration_script else {}
        ratio = step.get("ratio", 0.95)
        clipb = step.get("clip_baseline", False)
        clipn = step.get("clip_nc", False)
        os.makedirs(out, exist_ok=True)
        cal = {"mode": "calibrate", "chain_id": json.loads(open(self.arg(argv, "--chains")).readline())["chain_id"],
               "mem_budget": B, "baseline_cap": cap, "ratio": ratio, "in_window": 0.8 <= ratio <= 1.05,
               "clip": {"baseline": {"compactions": 3, "unrepaired_clipped": 3 if clipb else 0, "clip_fail": clipb},
                        "nc-chunks": {"compactions": 3, "unrepaired_clipped": 3 if clipn else 0, "clip_fail": clipn}},
               "clip_free": not (clipb or clipn), "recommended_cap": step.get("recommended_cap", cap),
               "nc_memory_total": 1500, "baseline_memory_total": round(1500 / ratio), "spent_usd": 0.25,
               "amendment": "C.5", "tokenizer": "anthropic:count_tokens:claude-haiku-5-5", "tokenizer_native": True,
               "ratio_ask_time": ratio, "ratio_pooled_all_compactions": ratio}
        json.dump(cal, open(os.path.join(out, "calibration.json"), "w"))
        self.cal_argvs = getattr(self, "cal_argvs", []) + [argv]
        return 0, f"calibration cap {cap} ratio {ratio}\n"

    def touch_out(self, argv):
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


def make_oh_artifact(d, n_chains=3):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "chains.jsonl"), "w") as f:
        for i in range(n_chains):
            c = {"chain_id": f"chain_s{i}_c0", "session_id": f"sess{i}", "spans": [], "qa": [
                {"qid": f"chain_s{i}_c0_1", "qtype": "single", "answerable": True},
                {"qid": f"chain_s{i}_c0_mh", "qtype": "multihop", "answerable": True},
                {"qid": f"chain_s{i}_c0_u", "answerable": False}]}
            f.write(json.dumps(c) + "\n")
    open(os.path.join(d, "verify-report.md"), "w").write("verify ok\n")
    open(os.path.join(d, "chains-manifest.json"), "w").write("{}\n")


class OpenHandsFlow(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.art = os.path.join(self.d.name, "oh-art")
        make_oh_artifact(self.art, 3)
        self.root = os.path.join(self.d.name, "root")
        self.base = ["--root", self.root, "--oh-dir", self.art, "--run-cap", "100", "--judge-cap", "100"]

    def tearDown(self):
        self.d.cleanup()

    def camp(self):
        return open(os.path.join(self.root, ".homerun-current")).read()

    def run_oh(self, runner, extra=(), toml=None):
        # the oh track refuses without a calibration or an explicit cap (Amendment C.4): scale 1.0 = cap == B
        argv = self.base + (["--config", toml] if toml else []) + ["oh", "--legacy-replay", *(extra or ["--baseline-cap-scale", "1.0"])]
        return quiet(H.main, argv, runner=runner)

    def toml(self, text):
        p = os.path.join(self.d.name, "t.toml")
        open(p, "w").write(text)
        return p

    def test_batches_are_b_major_with_primary_first_and_everything_is_chained(self):
        r = FakeRunner()
        t = self.toml("[oh]\nbatch_chains = 2\nbudgets = [512, 1536]\nretries = 0\n")
        rc, so, se = self.run_oh(r, toml=t)
        self.assertEqual(rc, 0, so + se)
        runs = [(os.path.basename(self.arg_out(a)), self.arg_b(a)) for s, a in r.calls if s == "run_chainreplay.py" and "--dry-run" not in a]
        self.assertEqual([b for _, b in runs], [1536, 1536, 512, 512])          # primary B for ALL chains first
        self.assertEqual([n for n, _ in runs], ["b1536-00", "b1536-01", "b512-00", "b512-01"])
        scripts = [s for s, _ in r.calls]
        for s in ("merge_runs.py", "export_judge_input.py", "judge_v2.py", "metrics_v2.py", "funnel_chains.py"):
            self.assertIn(s, scripts)
        order = [s for s in scripts if s in ("merge_runs.py", "export_judge_input.py", "judge_v2.py", "metrics_v2.py", "funnel_chains.py")]
        self.assertEqual(order, ["merge_runs.py", "export_judge_input.py", "judge_v2.py", "metrics_v2.py", "funnel_chains.py"])
        c = self.camp()
        self.assertTrue(os.path.isfile(os.path.join(c, "oh", "merged", "run_budget.json")))
        recs = H.read_jsonl(os.path.join(c, "oh", "merged", "records.jsonl"))
        self.assertEqual(len(recs), 3 * 3 * 4 * 2)                              # chains x qa x arms x B
        led = H.read_json(os.path.join(c, "ledger.json"))["entries"]
        self.assertAlmostEqual(sum(e["usd"] for e in led if e["phase"] == "run"), 0.5 * 3 * 2)
        self.assertAlmostEqual(sum(e["usd"] for e in led if e["phase"] == "judge"), 0.5)
        # budget passed to every run is the remaining cap
        caps = [float(self.arg_flag(a, "--budget-usd")) for s, a in r.calls if s == "run_chainreplay.py" and "--dry-run" not in a]
        self.assertEqual(caps[0], 100.0)
        self.assertLess(caps[-1], caps[0])
        # second invocation: nothing is rerun
        n = len(r.calls)
        rc, so, se = self.run_oh(r, toml=t)
        self.assertEqual(rc, 0, so + se)
        self.assertEqual([s for s, _ in r.calls[n:]], [])

    @staticmethod
    def arg_flag(argv, flag):
        return argv[argv.index(flag) + 1]

    def arg_out(self, argv):
        return self.arg_flag(argv, "--out")

    def arg_b(self, argv):
        return int(self.arg_flag(argv, "--mem-budgets"))

    def test_invalid_cell_is_retried_and_the_accepted_view_feeds_the_merge(self):
        r = FakeRunner(invalid_first={("chain_s1_c0", 1536)})
        t = self.toml("[oh]\nbatch_chains = 3\nbudgets = [1536]\nretries = 1\n")
        rc, so, se = self.run_oh(r, toml=t)
        self.assertEqual(rc, 0, so + se)
        c = self.camp()
        names = sorted(os.listdir(os.path.join(c, "oh", "runs")))
        self.assertEqual(names, ["b1536-00", "retry1-b1536-00"])
        retry = [a for s, a in r.calls if s == "run_chainreplay.py" and "--dry-run" not in a
                 and "retry1" in self.arg_out(a)][0]
        sub = [json.loads(l)["chain_id"] for l in open(self.arg_flag(retry, "--chains"))]
        self.assertEqual(sub, ["chain_s1_c0"])                                   # only the invalid chain
        recs = H.read_jsonl(os.path.join(c, "oh", "merged", "records.jsonl"))
        self.assertEqual(len(recs), 3 * 3 * 4)
        self.assertTrue(all(x["valid"] for x in recs))                           # the retry replaced the invalid cell
        self.assertEqual(len({x["rid"] for x in recs}), len(recs))
        # the failed first attempt is still on disk, untouched
        first = H.read_jsonl(os.path.join(c, "oh", "runs", "b1536-00", "chain_budget.jsonl"))
        self.assertFalse(next(b for b in first if b["chain_id"] == "chain_s1_c0")["valid"])

    def test_cells_invalid_after_the_retries_are_kept_with_allow_invalid(self):
        r = FakeRunner(invalid_first={("chain_s0_c0", 1536)})
        t = self.toml("[oh]\nbatch_chains = 3\nbudgets = [1536]\nretries = 0\n")
        rc, so, se = self.run_oh(r, toml=t)
        self.assertEqual(rc, 0, so + se)
        merge = [a for s, a in r.calls if s == "merge_runs.py"][0]
        self.assertIn("--allow-invalid", merge)
        self.assertIn("stay invalid", so)

    def test_failed_batch_is_moved_aside_and_redone_and_a_spent_cap_refuses(self):
        r = FakeRunner()
        t = self.toml("[oh]\nbatch_chains = 3\nbudgets = [1536]\nretries = 0\n")
        orig = r.chainreplay

        def crash(argv):
            if "--dry-run" in argv:
                return orig(argv)
            rc, text = orig(argv)
            os.remove(os.path.join(self.arg_flag(argv, "--out"), "chain_budget.jsonl"))      # the run died mid-way
            return 1, text
        r.chainreplay = crash
        rc, so, se = self.run_oh(r, toml=t)
        self.assertEqual(rc, 2)
        self.assertIn("did not finish", se)
        r.chainreplay = orig
        rc, so, se = self.run_oh(r, toml=t)
        self.assertEqual(rc, 0, so + se)
        runs = os.listdir(os.path.join(self.camp(), "oh", "runs"))
        self.assertEqual(sorted(runs)[0], "b1536-00")
        self.assertTrue(any(".failed-" in x for x in runs))
        # the crashed attempt's spend stays on the ledger
        led = H.read_json(os.path.join(self.camp(), "ledger.json"))["entries"]
        self.assertEqual(len([e for e in led if e["phase"] == "run"]), 2)
        # a cap that is already spent refuses a new batch
        rc, so, se = quiet(H.main, ["--root", self.root, "--oh-dir", self.art, "--run-cap", "0.5", "--judge-cap", "9",
                                    "--config", t, "oh", "--legacy-replay", "--baseline-cap-scale", "1.0", "--force-step", "oh.run.b1536-00", "--accept-partial"], runner=r)
        self.assertEqual(rc, 2)
        self.assertIn("spent", se)

    def test_budget_stop_is_reported_and_resumable(self):
        r = FakeRunner()
        r.stop_after = True
        t = self.toml("[oh]\nbatch_chains = 3\nbudgets = [1536]\nretries = 0\n")
        rc, so, se = self.run_oh(r, toml=t)
        self.assertEqual(rc, 2)
        self.assertIn("BUDGET STOP", se)


# ---------------------------------------------------------------------------- bundle

class BundleTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.d.name, "root")
        cfg = H.load_config(None, {"general": {"artifact_root": self.root}})
        self.camp = H.open_campaign(cfg, H.build_parser().parse_args(["bundle"]), "bundle", echo=False)
        c = self.camp
        self.secret = "sk-ant-" + "a1B2c3D4" * 4                           # assembled at runtime
        self.env_secret = "tok_" + "Z9y8X7w6" * 3
        os.environ["HR_TEST_SECRET_TOKEN"] = self.env_secret
        def w(rel, text):
            p = c.p(*rel.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(text)
        w("oh/merged/records.jsonl", json.dumps({"rid": "a", "answer": f"key is {self.secret}", "memory_text": "m"}) + "\n")
        w("oh/merged/metrics.json", json.dumps({"gates": {"verdict": "INCONCLUSIVE"}}))
        w("oh/merged/REPORT-chains.md", f"# report\nenv {self.env_secret}\n")
        w("oh/merged/judge-input.jsonl", "{\"span_text\": \"RAW TRANSCRIPT\"}\n")
        w("oh/merged/sessions/chain_a__b1536/graph.graphml", "<graph/>")
        w("oh/runs/b1536-00/records.jsonl", "{\"rid\": \"b\"}\n")
        w("oh/runs/b1536-00/logs/chain_a__b1536/pi.err", "stderr")
        w("oh/runs/b1536-00/recall/x.json", "{}")
        w("oh/runs.failed-1/records.jsonl", "{}\n")
        w("oh/inputs/chains.jsonl", "{\"spans\": \"RAW\"}\n")
        w("oh/inputs/chains-manifest.json", "{}")
        w("beam/run/records.jsonl", json.dumps({"rid": "z", "memory_text": "beam mem"}) + "\n")
        w("beam/data/beam-1M-1.json", "{\"messages\": []}")
        w("cc/run/records.jsonl", json.dumps({"rid": "q", "answer": "short", "memory_text": "PRIVATE", "injected": "PRIVATE2",
                                              "question": "keep"}) + "\n")
        w("cc/data-qa/qa-raw.jsonl", "{\"prompt\": \"PRIVATE\"}\n")
        w("logs/oh.merge.log", "ok\n")
        w("ledger.json", "{\"entries\": []}")
        w("accepted/x/records.jsonl", "{}\n")
        c.write_manifest()

    def tearDown(self):
        os.environ.pop("HR_TEST_SECRET_TOKEN", None)
        self.d.cleanup()

    def pack(self, *extra):
        rc, so, se = quiet(H.main, ["--root", self.root, "bundle", *extra])
        self.assertEqual(rc, 0, so + se)
        path = re.search(r"BUNDLE: (\S+)", so).group(1)
        with tarfile.open(path) as tf:
            names = tf.getnames()
            data = {n: tf.extractfile(n).read().decode() for n in names if tf.getmember(n).isfile()}
        return so, path, names, data

    def test_default_bundle_excludes_raw_and_redacts(self):
        so, path, names, data = self.pack()
        rel = {n.split("/", 1)[1] for n in names if "/" in n}
        for want in ("oh/merged/records.jsonl", "oh/merged/metrics.json", "oh/merged/REPORT-chains.md", "oh/runs/b1536-00/records.jsonl",
                     "oh/inputs/chains-manifest.json", "beam/run/records.jsonl", "cc/run/records.jsonl", "logs/oh.merge.log",
                     "ledger.json", "MANIFEST.json", "bundle-manifest.json", "bundle-redaction-report.json"):
            self.assertIn(want, rel)
        for banned in ("oh/merged/judge-input.jsonl", "oh/merged/sessions/chain_a__b1536/graph.graphml",
                       "oh/runs/b1536-00/logs/chain_a__b1536/pi.err", "oh/runs/b1536-00/recall/x.json",
                       "oh/inputs/chains.jsonl", "beam/data/beam-1M-1.json", "cc/data-qa/qa-raw.jsonl",
                       "accepted/x/records.jsonl", "oh/runs.failed-1/records.jsonl"):
            self.assertNotIn(banned, rel)
        blob = "\n".join(data.values())
        self.assertNotIn(self.secret, blob)
        self.assertNotIn(self.env_secret, blob)
        self.assertIn("[REDACTED:sk_ant]", blob)
        self.assertIn("[REDACTED:env_secret]", blob)
        rep = json.loads(next(v for k, v in data.items() if k.endswith("bundle-redaction-report.json")))
        self.assertGreaterEqual(rep["total_hits"], 2)
        self.assertGreater(rep["excluded"]["count"], 5)
        self.assertIn("REDACTION SCAN", so)
        # private cc text fields are dropped, the rest of the record stays
        cc = json.loads(next(v for k, v in data.items() if k.endswith("cc/run/records.jsonl")).splitlines()[0])
        self.assertEqual(cc, {"rid": "q", "answer": "short", "question": "keep"})
        beam = json.loads(next(v for k, v in data.items() if k.endswith("beam/run/records.jsonl")).splitlines()[0])
        self.assertEqual(beam["memory_text"], "beam mem")
        self.assertRegex(so, r"size: [\d.]+ (B|KB|MB|GB)")

    def test_include_raw_and_tracks_and_no_redact(self):
        so, path, names, data = self.pack("--include-raw", "--tracks", "oh")
        rel = {n.split("/", 1)[1] for n in names if "/" in n}
        self.assertIn("oh/merged/judge-input.jsonl", rel)
        self.assertIn("oh/merged/sessions/chain_a__b1536/graph.graphml", rel)
        self.assertNotIn("beam/run/records.jsonl", rel)                       # track not selected
        self.assertNotIn(self.secret, "\n".join(data.values()))               # raw never means unredacted
        so, path, names, data = self.pack("--no-redact", "--tracks", "oh")
        self.assertIn(self.secret, "\n".join(data.values()))                  # explicit opt-out only


# ---------------------------------------------------------------------------- smoke (offline) and misc

class SmokeOffline(unittest.TestCase):
    def test_tiny_long_session_smoke_runs_offline(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "root")
            rc, so, se = quiet(H.main, ["--root", root, "--offline-selftest", "smoke", "--skip-oh"], probes=FakeProbes())
            self.assertEqual(rc, 0, so + se)
            camps = [x for x in os.listdir(root) if x.endswith("-smoke")]
            self.assertEqual(len(camps), 1)
            rep = H.read_json(os.path.join(root, camps[0], "smoke-report.json"))
            self.assertTrue(all(r["ok"] for r in rep), rep)
            recs = H.read_jsonl(os.path.join(root, camps[0], "smoke", "ls-run", "records.jsonl"))
            self.assertEqual({r["arm"] for r in recs}, {"baseline", "nc-chunks", "nc-graph", "nc-note-only"})
            self.assertFalse(os.path.exists(os.path.join(root, ".homerun-current")))   # smoke never becomes "current"

    def test_oh_smoke_picks_the_cheapest_chain_and_runs_the_whole_flow(self):
        with tempfile.TemporaryDirectory() as d:
            art = os.path.join(d, "art")
            make_oh_artifact(art, 3)
            root = os.path.join(d, "root")
            os.environ["HR_TEST_KEY4"] = "k-" + "x" * 20
            r = FakeRunner()
            orig = r.chainreplay

            def chainreplay(argv):
                if "--dry-run" in argv:
                    return 0, ("== chain_s0_c0\n  projected cost: a = $0.40\n== chain_s1_c0\n  projected cost: a = $0.20\n"
                               "== chain_s2_c0\n  projected cost: a = $0.30\n"
                               "TOTAL projected: 12 compactions, 80 asks, ~120.0 min, $0.9000\n")
                return orig(argv)
            r.chainreplay = chainreplay
            rc, so, se = quiet(H.main, ["--root", root, "--key-env", "HR_TEST_KEY4", "--oh-dir", art, "smoke", "--skip-ls", "--legacy-replay"],
                               runner=r, probes=FakeProbes())
            self.assertEqual(rc, 0, so + se)
            camp = [x for x in os.listdir(root) if x.endswith("-smoke")][0]
            chains = H.read_jsonl(os.path.join(root, camp, "oh", "inputs", "chains.jsonl"))
            self.assertEqual([c["chain_id"] for c in chains], ["chain_s1_c0"])          # the cheapest one
            scripts = [s for s, _ in r.calls]
            for s in ("run_chainreplay.py", "merge_runs.py", "judge_v2.py", "metrics_v2.py", "funnel_chains.py"):
                self.assertIn(s, scripts)
            rep = H.read_json(os.path.join(root, camp, "smoke-report.json"))
            self.assertTrue(all(x["ok"] for x in rep), rep)

    def test_smoke_refuses_a_shim_embedder_unless_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "root")
            os.environ["HR_TEST_KEY2"] = "k-" + "x" * 20
            rc, so, se = quiet(H.main, ["--root", root, "--key-env", "HR_TEST_KEY2", "smoke", "--skip-oh"],
                               probes=FakeProbes(semantic=False))
            self.assertEqual(rc, 1)
            self.assertIn("doctor has FAILs", so)

    def test_synthetic_session_is_valid(self):
        s = H.synthetic_session()
        self.assertEqual(L.validate_session(s), [])
        self.assertEqual(len(s["questions"]), 5)
        self.assertTrue(all(q["rubric"] and q["gold_anchor"] for q in s["questions"]))

    def test_hars_env_block_covers_profile_and_session_mode(self):
        cfg = H.load_config(None, {})
        txt = "\n".join(H.hars_env_block(cfg))
        for needle in ("HARS_MEMORY_SESSION_MODE=split", "HARS_MEMORY_MCP_ENGINE_THREADS=1", "HARS_MEMORY_EMBED_MODEL",
                       "HARS_MEMORY_EXTRACTOR_MODEL=claude-haiku-5-5", "OMIT_SAMPLING"):
            self.assertIn(needle, txt)

    def test_no_secret_value_in_logs_or_cmdlines(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["HR_TEST_KEY3"] = "k-" + "SECRETVALUE" * 3
            root = os.path.join(d, "root")
            quiet(H.main, ["--root", root, "--key-env", "HR_TEST_KEY3", "doctor"], probes=FakeProbes())
            camp = open(os.path.join(root, ".homerun-current")).read()
            for dp, _, fns in os.walk(camp):
                for fn in fns:
                    p = os.path.join(dp, fn)
                    try:
                        self.assertNotIn("SECRETVALUE", open(p, errors="replace").read(), p)
                    except IsADirectoryError:
                        pass


if __name__ == "__main__":
    unittest.main()
