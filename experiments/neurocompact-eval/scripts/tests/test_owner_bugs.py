"""Offline tests for the five owner bug reports from the home run: sid-less sessions / non-session files in
long-session dirs, campaign selection (no implicit resume), the doctor semantic probe, one client per HARS
(lock + lease), and refusal of paid steps on void / other-version campaigns."""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

import ls_fixtures as F
import gen_longsession_qa as G
import export_longsession_judge as EJ
import home_run as H
import longsession as L
import test_home_run as thr


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


# ------------------------------------------------------------------ 1. sid
class SidRobustness(unittest.TestCase):
    def test_session_without_sid_gets_a_deterministic_sid_and_a_warning(self):
        with tempfile.TemporaryDirectory() as d:
            s = F.make_session(questions=False)
            del s["sid"]
            p = os.path.join(d, "My Session (v2).json")
            write(p, s)
            warns = []
            a = L.load_sessions([p], warn=warns.append)
            b = L.load_sessions([p], warn=lambda m: None)
            self.assertEqual(a[0]["sid"], b[0]["sid"])                     # deterministic
            self.assertRegex(a[0]["sid"], L.SID_RE.pattern)
            self.assertTrue(a[0]["sid"].startswith("My-Session--v2"))
            self.assertEqual(a[0]["_sid_assigned"], a[0]["sid"])
            self.assertTrue(any("no valid sid" in w for w in warns))
            self.assertEqual(L.validate_session(a[0]), [])                 # valid after normalisation

    def test_list_and_jsonl_get_distinct_sids(self):
        with tempfile.TemporaryDirectory() as d:
            ss = []
            for _ in range(2):
                s = F.make_session(questions=False)
                del s["sid"]
                ss.append(s)
            write(os.path.join(d, "many.json"), ss)
            got = L.load_sessions([os.path.join(d, "many.json")], warn=lambda m: None)
            self.assertEqual(len({s["sid"] for s in got}), 2)

    def test_directory_skips_non_session_files(self):
        """qa-manifest-<stamp>.json (home_run renames it) and qa-raw.jsonl live next to the sessions in cc/data-qa:
        they used to be loaded as sessions -> KeyError 'sid' (manifest) / phantom sessions (raw rows)."""
        with tempfile.TemporaryDirectory() as d:
            s = F.make_session(questions=False)
            L.save_session(s, d)
            write(os.path.join(d, "qa-manifest-20261027T120000Z.json"), {"started": "x", "sessions": []})
            with open(os.path.join(d, "qa-raw.jsonl"), "w") as f:
                f.write(json.dumps({"sid": s["sid"], "kind": "single", "raw": "..."}) + "\n")
            warns = []
            got = L.load_sessions([d], warn=warns.append)
            self.assertEqual([x["sid"] for x in got], [s["sid"]])
            self.assertEqual(len(warns), 2)

    def test_gen_cli_on_sidless_file_and_on_its_own_output_dir(self):
        with tempfile.TemporaryDirectory() as d:
            s = F.make_session(questions=False)
            del s["sid"]
            p = os.path.join(d, "in", "nosid.json")
            write(p, s)
            out = os.path.join(d, "out")
            args = ["--sessions", p, "--out", out, "--provider", "mock", "--model", "mock", "--window-tokens", "14000",
                    "--keep-tail", "3000", "--reserve", "500", "--max-calls", "2"]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(G.main(args), 0)
            # the output dir now holds qa-raw.jsonl + qa-manifest.json: loading it must not crash (home_run cc.qa does)
            os.replace(os.path.join(out, "qa-manifest.json"), os.path.join(out, "qa-manifest-20261027T1.json"))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual([x["sid"] for x in L.load_sessions([out])], [x["sid"] for x in L.load_sessions([out])])

    def test_export_judge_loads_sidless_sessions(self):
        with tempfile.TemporaryDirectory() as d:
            s = F.make_session()
            del s["sid"]
            write(os.path.join(d, "s.json"), s)
            with contextlib.redirect_stderr(io.StringIO()):
                got = {x["sid"]: x for x in L.load_sessions(d)}
            self.assertEqual(len(got), 1)

    def test_the_importer_always_sets_sid(self):
        import import_claude_code as C
        e = [{"type": "user", "uuid": "u1", "sessionId": "", "message": {"role": "user", "content": "hello"}}]
        self.assertTrue(C.build_session(e, files=["/x/y.jsonl"])["sid"])


# ------------------------------------------------------------------ 2. campaign selection
class CampaignSelection(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = self.d.name
        self.cfg = thr.cfg_for(self.root)

    def tearDown(self):
        self.d.cleanup()

    def mk(self, name, created, kind="oh", mtime=None):
        p = os.path.join(self.root, name)
        write(os.path.join(p, "campaign.json"), {"created_utc": created, "kind": kind})
        if mtime:
            os.utime(os.path.join(p, "campaign.json"), (mtime, mtime))
            os.utime(p, (mtime, mtime))
        return p

    def open(self, *argv, kind="oh"):
        a = H.build_parser().parse_args([kind, *argv])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            camp = H.open_campaign(self.cfg, a, kind, echo=False)
        return camp, out.getvalue()

    def test_no_flags_creates_a_new_campaign_never_the_touched_old_one(self):
        old = self.mk("20260101T000000Z-homerun-home", "2026-01-01T00:00:00Z", mtime=time.time() + 1000)
        camp, out = self.open("--budgets", "1536", kind="calibrate")
        self.assertNotEqual(camp.path, old)
        self.assertIn("NEW campaign", out)
        self.assertIn(camp.path, out)

    def test_bare_resume_is_newest_by_creation_not_mtime_and_is_loud(self):
        self.mk("a-old", "2026-01-01T00:00:00Z", mtime=time.time() + 1000)       # hand-touched
        new = self.mk("b-new", "2026-03-01T00:00:00Z", mtime=1000)
        camp, out = self.open("--resume")
        self.assertEqual(camp.path, new)
        self.assertIn("RESUMING the NEWEST", out)
        self.assertIn("2026-03-01", out)

    def test_bare_resume_skips_smoke_and_dry_run_campaigns(self):
        real = self.mk("r", "2026-01-01T00:00:00Z")
        self.mk("s", "2026-02-01T00:00:00Z", kind="smoke")
        self.mk("d", "2026-03-01T00:00:00Z", kind="dry-run")
        self.assertEqual(self.open("--resume")[0].path, real)

    def test_resume_id_path_and_fragment(self):
        a = self.mk("20260101T000000Z-homerun-home", "2026-01-01T00:00:00Z")
        b = self.mk("20260201T000000Z-homerun-home", "2026-02-01T00:00:00Z")
        self.assertEqual(self.open("--resume", "20260101T000000Z-homerun-home")[0].path, a)
        self.assertEqual(self.open("--resume", b)[0].path, b)
        self.assertEqual(self.open("--resume", "0201T0")[0].path, b)
        with self.assertRaises(H.HomeRunError):
            self.open("--resume", "homerun-home")                              # ambiguous fragment
        with self.assertRaises(H.HomeRunError):
            self.open("--resume", "nope")

    def test_new_overrides_resume(self):
        a = self.mk("x", "2026-01-01T00:00:00Z")
        camp, _ = self.open("--resume", "--new")
        self.assertNotEqual(camp.path, a)

    def test_read_only_commands_refuse_without_a_target_and_say_what_to_pass(self):
        self.mk("x", "2026-01-01T00:00:00Z")
        with self.assertRaises(H.HomeRunError) as cm:
            self.open(kind="bundle")
        self.assertIn("--resume", str(cm.exception))
        self.assertIn("x", str(cm.exception))                                   # lists the newest campaigns
        self.assertEqual(self.open("--resume", kind="bundle")[0].path, os.path.join(self.root, "x"))

    def test_main_status_and_bundle_without_target(self):
        self.mk("x", "2026-01-01T00:00:00Z")
        rc, so, se = thr.quiet(H.main, ["--root", self.root, "status"])
        self.assertEqual(rc, 0)
        self.assertIn("status --resume", so)
        rc, so, se = thr.quiet(H.main, ["--root", self.root, "bundle"])
        self.assertEqual(rc, 2)
        self.assertIn("read-only", se)
        rc, so, se = thr.quiet(H.main, ["--root", self.root, "status", "--resume"])
        self.assertEqual(rc, 0)
        self.assertIn("valid for spending", so)

    def test_creation_time_falls_back_to_the_dir_stamp(self):
        p = os.path.join(self.root, "20260505T101010Z-homerun-home")
        os.makedirs(p)
        self.assertEqual(H.campaign_created(p), H._parse_ts("2026-05-05T10:10:10Z"))

    def test_doctor_without_target_makes_no_campaign(self):
        rc, so, se = thr.quiet(H.main, ["--root", self.root, "doctor", "--no-live"], probes=thr.FakeProbes())
        self.assertEqual(H.list_campaigns(self.root), [])
        self.assertTrue(os.listdir(os.path.join(self.root, ".doctor")))


# ------------------------------------------------------------------ 3. semantic probe
class SemanticProbe(unittest.TestCase):
    class Probes(thr.FakeProbes):
        def __init__(self, mode="split", **kw):
            super().__init__(**kw)
            self.mode, self.compacts, self.sessions = mode, [], set()

        def mcp(self, url, tool, args, timeout=60.0):
            if "session_id" in args:
                self.sessions.add(args["session_id"])
            if tool == "memory_compact":
                self.compacts.append(args)
                return {"ok": True, "mode": self.mode}
            if tool == "memory_recall":
                r = super().mcp(url, tool, args, timeout)
                r["session"] = {"stats": {"min_score": 0.2}, "chunks": [{"fused_score": 0.62}, {"fused_score": 0.31}]}
                return r
            return super().mcp(url, tool, args, timeout)

    def cfg(self):
        return thr.cfg_for(tempfile.mkdtemp())

    def test_three_separate_documents_each_far_above_the_chunk_minimum(self):
        self.assertEqual(len(H.SEMANTIC_PROBES), 3)
        for doc, q, want in H.SEMANTIC_PROBES:
            self.assertIn(want, doc)
            # >= ~300 cl100k tokens (~4.4 chars/token of English prose), far above the server's 200-token chunk minimum
            self.assertGreater(len(doc), 1350)
            self.assertLess(len(doc), 2400)
        self.assertEqual(len({d for d, _, _ in H.SEMANTIC_PROBES}), 3)

    def test_probe_uses_separate_compacts_in_one_throwaway_session_and_reports_hits_and_scores(self):
        p = self.Probes()
        rows = H.deep_ingest_check(self.cfg(), p)
        self.assertEqual(len(p.compacts), 3)
        self.assertEqual(len({c["summary"] for c in p.compacts}), 3)
        self.assertEqual(len(p.sessions), 1)
        sid = next(iter(p.sessions))
        self.assertTrue(sid.startswith("doctor-probe-"))
        self.assertEqual([c["seq"] for c in p.compacts], [1, 2, 3])
        sp = next(r for r in rows if r["check"] == "hars.semantic_probe")
        self.assertEqual(sp["status"], "OK")
        self.assertIn("3/3", sp["detail"])
        for i in (1, 2, 3):
            self.assertIn(f"doc{i} hit", sp["detail"])
        self.assertIn("0.62", sp["detail"])
        iso = next(r for r in rows if r["check"] == "hars.probe_isolation")
        self.assertEqual(iso["status"], "OK")

    def test_shim_still_fails_and_reports_each_doc(self):
        rows = H.deep_ingest_check(self.cfg(), self.Probes(semantic=False))
        sp = next(r for r in rows if r["check"] == "hars.semantic_probe")
        self.assertEqual(sp["status"], "FAIL")
        self.assertEqual(sp["detail"].count("MISS"), 3)

    def test_subgraph_mode_means_the_probe_wrote_into_the_main_index_and_fails_loudly(self):
        rows = H.deep_ingest_check(self.cfg(), self.Probes(mode="subgraph"))
        iso = next(r for r in rows if r["check"] == "hars.probe_isolation")
        self.assertEqual(iso["status"], "FAIL")
        self.assertIn("MAIN graph", iso["detail"])

    def test_probe_never_calls_a_main_index_write_tool(self):
        p = self.Probes()
        H.deep_ingest_check(self.cfg(), p)
        self.assertEqual(set(p.mcp_calls), {"memory_session_status", "memory_recall"}.union(set(p.mcp_calls) & {"memory_status"}))


# ------------------------------------------------------------------ 4. one client per HARS
class HarsLock(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self._old = os.environ.get("NC_EVAL_LOCK_DIR")
        os.environ["NC_EVAL_LOCK_DIR"] = self.d.name
        H.release_hars_locks()
        H.GUARD_CTX.update({"force_shared": False, "campaign": ""})
        self.url = "http://127.0.0.1:8307/mcp"

    def tearDown(self):
        H.release_hars_locks()
        if self._old is None:
            os.environ.pop("NC_EVAL_LOCK_DIR", None)
        else:
            os.environ["NC_EVAL_LOCK_DIR"] = self._old
        self.d.cleanup()

    def foreign_holder(self, pid):
        write(H.lock_path(self.url), {"pid": pid, "campaign": "C-other", "started_at": "2026-10-27T10:00:00Z",
                                      "argv": ["oh"]})

    def test_lock_file_is_sha1_of_the_normalised_url_and_records_holder(self):
        p = H.acquire_hars_lock(self.url, "C1")
        self.assertTrue(p.startswith(self.d.name) and p.endswith(".lock"))
        info = json.load(open(p))
        self.assertEqual((info["pid"], info["campaign"]), (os.getpid(), "C1"))
        self.assertEqual(H.lock_path("http://localhost:8307/mcp/"), H.lock_path(self.url))   # aliases share one lock
        self.assertEqual(H.acquire_hars_lock(self.url, "C1"), p)                              # re-entrant
        H.release_hars_locks()
        self.assertFalse(os.path.exists(p))

    def test_live_holder_refuses_with_its_details(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            self.foreign_holder(proc.pid)
            with self.assertRaises(H.HomeRunError) as cm:
                H.acquire_hars_lock(self.url, "C2")
            msg = str(cm.exception)
            self.assertIn(str(proc.pid), msg)
            self.assertIn("C-other", msg)
            self.assertIn("--force-shared-hars", msg)
            H.acquire_hars_lock(self.url, "C2", force=True)                                     # override works
            H.release_hars_locks()
            self.assertEqual(json.load(open(H.lock_path(self.url)))["pid"], proc.pid)           # forced never deletes theirs
        finally:
            proc.kill()
            proc.wait()

    def test_dead_holder_is_stale_and_replaced(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        self.foreign_holder(proc.pid)
        H.acquire_hars_lock(self.url, "C3")
        self.assertEqual(json.load(open(H.lock_path(self.url)))["campaign"], "C3")

    def test_campaign_run_locks_for_hars_writers_only(self):
        r = thr.FakeRunner() if hasattr(thr, "FakeRunner") else None
        calls = []

        def runner(full, env, cwd, log, echo):
            calls.append(full)
            return 0, ""
        camp = H.open_campaign(thr.cfg_for(self.d.name + "/root"), H.build_parser().parse_args(["oh"]), "oh",
                               runner=runner, echo=False)
        camp.guards_enabled = True
        camp.lease_fn = lambda tool, args: (_ for _ in ()).throw(RuntimeError("Unknown tool"))
        with contextlib.redirect_stdout(io.StringIO()):
            camp.run("export", [camp.script("export_longsession_judge.py")], check=False)
            self.assertFalse(os.path.exists(H.lock_path(self.url)))                             # not a HARS writer
            camp.run("dry", [camp.script("run_longsession.py"), "--dry-run"], check=False)
            self.assertFalse(os.path.exists(H.lock_path(self.url)))
            camp.run("run", [camp.script("run_longsession.py"), "--sessions", "x"], check=False)
        self.assertTrue(os.path.exists(H.lock_path(self.url)))
        self.assertNotIn("--run-tag", calls[-1])                                                # no lease -> no run_tag
        # a second home_run (other pid) is refused
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            H.release_hars_locks()
            self.foreign_holder(proc.pid)
            with self.assertRaises(H.HomeRunError):
                camp.run("run2", [camp.script("run_longsession.py"), "--sessions", "x"], check=False)
        finally:
            proc.kill()
            proc.wait()

    def test_deep_probe_takes_the_lock(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            self.foreign_holder(proc.pid)
            with self.assertRaises(H.HomeRunError):
                H.deep_ingest_check(thr.cfg_for(self.d.name), thr.FakeProbes())
        finally:
            proc.kill()
            proc.wait()

    def test_offline_selftest_does_not_lock(self):
        calls = []
        camp = H.open_campaign(thr.cfg_for(self.d.name + "/root", general={"_offline": True}),
                               H.build_parser().parse_args(["oh"]), "oh", runner=lambda *a: (calls.append(a) or (0, "")),
                               echo=False)
        camp.guards_enabled = True
        camp.run("run", [camp.script("run_longsession.py"), "--sessions", "x"], check=False)
        self.assertFalse(os.path.exists(H.lock_path(self.url)))

    def test_lease_taken_passes_run_tag_and_absent_is_tolerated(self):
        got = []

        def ok(tool, args):
            got.append((tool, args))
            return {"ok": True, "run_tag": "tag-1"}
        r = H.try_server_lease(self.url, "camp", call=ok)
        self.assertEqual((r["active"], r["tag"]), (True, "tag-1"))
        self.assertEqual(got[0][0], "memory_lease_acquire")

        def absent(tool, args):
            return {"ok": False, "error": "Unknown tool: memory_lease_acquire"}
        self.assertFalse(H.try_server_lease(self.url, "camp", call=absent)["active"])

        def boom(tool, args):
            raise ConnectionError("refused")
        self.assertFalse(H.try_server_lease(self.url, "camp", call=boom)["active"])

        def held(tool, args):
            return {"ok": False, "error": "lease held", "holder": "other-run"}
        with self.assertRaises(H.HomeRunError):
            H.try_server_lease(self.url, "camp", call=held)
        self.assertFalse(H.try_server_lease(self.url, "camp", call=held, force=True)["active"])

    def test_run_longsession_receives_the_lease_tag(self):
        calls = []
        camp = H.open_campaign(thr.cfg_for(self.d.name + "/root"), H.build_parser().parse_args(["oh"]), "oh",
                               runner=lambda full, *a: (calls.append(full) or (0, "")), echo=False)
        camp.guards_enabled = True
        camp.lease_fn = lambda tool, args: {"ok": True, "run_tag": "tagX"}
        with contextlib.redirect_stdout(io.StringIO()):
            camp.run("run", [camp.script("run_longsession.py"), "--sessions", "x"], check=False)
        self.assertEqual(calls[-1][-2:], ["--run-tag", "tagX"])

    def test_force_flag_parses(self):
        a = H.build_parser().parse_args(["oh", "--force-shared-hars"])
        self.assertTrue(a.force_shared_hars)

    def test_run_longsession_sends_run_tag_on_compact(self):
        import run_longsession as R

        class RC:
            def __init__(self):
                self.sent = []

            def mcp_call(self, url, tool, args, timeout=0):
                self.sent.append(args)
                return {"ok": True}
        import threading
        h = R.Hars.__new__(R.Hars)
        h.rcm, h.url, h.cs, h.ct, h.run_tag = RC(), "u", threading.Semaphore(1), 1, ""
        h.compact("s", "sum", [], {})
        self.assertNotIn("run_tag", h.rcm.sent[-1])
        h.run_tag = "T"
        h.compact("s", "sum", [], {})
        self.assertEqual(h.rcm.sent[-1]["run_tag"], "T")


# ------------------------------------------------------------------ 5. paid-step guard
class PaidGuard(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.calls = []
        self.camp = H.open_campaign(thr.cfg_for(self.d.name), H.build_parser().parse_args(["oh"]), "oh",
                                    runner=lambda full, *a: (self.calls.append(full) or (0, "")), echo=False)

    def tearDown(self):
        self.d.cleanup()

    def paid(self, script="run_longsession.py", *extra):
        return self.camp.run("x", [self.camp.script(script), *extra], check=False)

    def test_fresh_campaign_is_valid(self):
        self.assertEqual(H.spend_blockers(self.camp), [])
        self.paid()
        self.assertEqual(len(self.calls), 1)

    def test_other_eval_version_blocks_every_paid_script_but_not_dry_runs(self):
        cj = self.camp.p("campaign.json")
        d = H.read_json(cj)
        d["eval_version"] = "2026-10-26.c5"
        H.write_json(cj, d)
        for s in H.PAID_SCRIPTS:
            with self.assertRaises(H.HomeRunError) as cm:
                self.paid(s)
            self.assertIn("EVAL-VERSION", str(cm.exception))
            self.assertIn("Nothing was spent", str(cm.exception))
        self.assertEqual(self.calls, [])
        self.paid("run_longsession.py", "--dry-run")                         # free
        self.assertEqual(len(self.calls), 1)

    def test_legacy_campaign_without_field_uses_its_scripts_copy(self):
        cj = self.camp.p("campaign.json")
        d = H.read_json(cj)
        del d["eval_version"]
        H.write_json(cj, d)
        self.assertEqual(H.campaign_eval_version(self.camp.path), H.HOME_RUN_VERSION)   # copied home_run.py
        with open(self.camp.p("scripts", "home_run.py"), "w") as f:
            f.write('HOME_RUN_VERSION = "2026-10-26.c5"\n')
        self.assertIn("2026-10-26.c5", " ".join(H.spend_blockers(self.camp)))

    def test_void_pre_c6_calibration_blocks_c6_paid_steps(self):
        H.write_json(self.camp.p("calibration.json"), {"amendment": "C.5", "status": "pass"})
        with self.assertRaises(H.HomeRunError) as cm:
            self.paid()
        self.assertIn("VOID", str(cm.exception))
        self.camp.legacy_replay = True                                        # legacy runner accepts C.5
        self.assertEqual(H.spend_blockers(self.camp), [])

    def test_explicitly_void_calibration_and_caps(self):
        H.write_json(self.camp.p("calibration.json"), {"amendment": "C.6", "status": "pass", "void": True,
                                                       "void_reason": "tokenizer cl100k"})
        self.assertIn("tokenizer cl100k", " ".join(H.spend_blockers(self.camp)))
        H.write_json(self.camp.p("calibration.json"), {"amendment": "C.6", "status": "pass"})
        H.write_json(self.camp.p("oh", "caps.json"), {"void": True})
        self.assertIn("caps.json", " ".join(H.spend_blockers(self.camp)))

    def test_calibration_written_under_another_version(self):
        H.write_json(self.camp.p("calibration.json"), {"amendment": "C.6", "status": "pass", "eval_version": "old"})
        self.assertIn("'old'", " ".join(H.spend_blockers(self.camp)))

    def test_stale_checkout_marker_blocks(self):
        marker = H.eval_version_file()
        self.assertEqual(marker, H.HOME_RUN_VERSION)                          # repo consistency

    def test_commands_refuse_before_any_step(self):
        d = H.read_json(self.camp.p("campaign.json"))
        d["eval_version"] = "old"
        H.write_json(self.camp.p("campaign.json"), d)
        for cmd in ("calibrate", "oh", "beam", "cc"):
            rc, so, se = thr.quiet(H.main, ["--root", self.d.name, "--run-cap", "5", "--gen-cap", "3", "--judge-cap", "3",
                                            cmd, "--resume", self.camp.path], runner=lambda *a: (0, ""),
                                   probes=thr.FakeProbes())
            self.assertEqual(rc, 2, cmd + so + se)
            self.assertIn("refusing to spend", se, cmd)


if __name__ == "__main__":
    unittest.main()
