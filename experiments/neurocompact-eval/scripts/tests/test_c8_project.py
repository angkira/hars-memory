"""Amendment C.8: the project layout. The session's dropped context is written with memory_upsert_document into its OWN
project on the same server and read back with memory_recall(project=..., no session_id) - the way the global memory is
used. Offline: mock LLM + FakeHars (project-aware). No network."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import home_run as H
import ls_fixtures as F
import longsession as L
import run_longsession as R
import test_c7_enriched as T7
import test_c7_home as T7H

BASE = [x for x in T7.BASE if x not in ("--hars-layout", "split-session")]
ARMS = ["--arms", "baseline,nc-enriched,nc-note-only"]
RID = ["--hars-run-id", "e1"]
PID = L.make_project_id("e1", "syn-1")


class Base(T7.Base):
    def run_ls(self, out, extra=(), base=None):
        return T7.call(["--sessions", self.sd, "--out", out] + (base or BASE) + list(extra))


class MakeProjectId(unittest.TestCase):
    def test_sanitised_limited_unique_never_default(self):
        self.assertEqual(L.make_project_id("e1", "syn-1"), "nc-e1-syn-1")
        self.assertEqual(L.make_project_id("e1", "a b/c:d"), "nc-e1-a-b-c-d")
        long1, long2 = L.make_project_id("e1", "x" * 200), L.make_project_id("e1", "x" * 199 + "y")
        self.assertLessEqual(len(long1), L.PROJECT_ID_MAX)
        self.assertNotEqual(long1, long2)
        self.assertEqual(L.make_project_id("e1", "s", 1536, True), "nc-e1-s-b1536")
        for sid in ("default", "", "-"):
            self.assertNotEqual(L.make_project_id("", sid), "default")


class ArgsAndRefusals(Base):
    def test_default_layout_is_project_and_question_time_arms_need_split(self):
        a = R.parse_args(["--sessions", self.sd] + BASE)
        self.assertEqual(a.hars_layout, "project")
        b = R.parse_args(["--sessions", self.sd] + BASE + ["--arms", "baseline,nc-chunks,nc-note-only"])
        self.assertEqual(b.hars_layout, "split-session")
        with self.assertRaises(SystemExit):
            R.parse_args(["--sessions", self.sd] + BASE + ["--arms", "baseline,nc-chunks", "--hars-layout", "project"])
        with self.assertRaises(SystemExit):
            R.parse_args(["--sessions", self.sd] + BASE + ["--hars-layout", "split-session", "--hars-ingested"])
        with self.assertRaises(SystemExit):
            R.parse_args(["--sessions", self.sd] + BASE + ["--recall-probe", "--reuse-run", "x"])    # probe needs --hars-ingested

    def test_never_writes_into_default(self):
        for bad in ("default", "", None):
            with self.assertRaises(ValueError):
                R.check_project_writable(bad)
            with self.assertRaises(ValueError):
                R.FakeHars().upsert(bad, "/x.md", "c")
        h = R.Hars("http://127.0.0.1:1/mcp", mock.MagicMock(), mock.MagicMock())
        with self.assertRaises(ValueError):
            h.upsert("default", "/x.md", "c")

    def test_unregistered_project_refuses_before_any_spend(self):
        with T7.SharedHars() as h:
            h.auto_register = False
            rc, so, se = self.run_ls(self.out, ARMS + RID)
        self.assertEqual(rc, 3, so + se)
        self.assertIn("not registered", se)
        self.assertFalse(os.path.exists(os.path.join(self.out, "records.jsonl")))
        self.assertEqual(h.upsert_calls, [])

    def test_non_empty_project_refuses_a_fresh_ingest(self):
        with T7.SharedHars() as h:
            h.register_project(PID)
            self.assertTrue(h.upsert(PID, "/old.md", "JOB0001 something")["ok"])
            rc, so, se = self.run_ls(self.out, ARMS + RID)
        self.assertEqual(rc, 3, so + se)
        self.assertIn("not empty", se)
        self.assertEqual(len(h.upsert_calls), 1)               # only the test's own write


class ProjectRun(Base):
    def test_ingest_into_the_project_recall_from_the_project_without_session_id(self):
        with T7.SharedHars() as h:
            h.default_kb = ["UNRELATED global KB chunk about kubernetes ingress 9999"] * 30
            h.recall = mock.Mock(side_effect=AssertionError("question-time recall"))
            rc, so, se = self.run_ls(self.out, ARMS + RID)
            self.assertEqual(rc, 0, so + se)
        self.assertEqual(h.compact_calls, 0)                                     # memory_compact is not used in this layout
        self.assertTrue(h.upsert_calls and all(c["project"] == PID for c in h.upsert_calls))
        self.assertTrue(all(c["path"].startswith(f"/nc-eval/{PID}/") for c in h.upsert_calls))
        self.assertEqual(set(h.projects), {PID})                                 # nothing else was created, `default` untouched
        self.assertEqual(len(h.correct_calls), 8)
        for c in h.correct_calls:
            self.assertEqual((c["project"], c["sid"]), (PID, None))              # NO session_id
            self.assertTrue(c["ll_keywords"] and c["hl_keywords"])
            self.assertIsInstance(c["max_chars"], int)
        rows = T7.rows_of(self.out)
        self.assertTrue(all(r["valid"] is True for r in rows), [r["chain_invalid_reasons"] for r in rows][:2])
        e = [r for r in rows if r["arm"] == "nc-enriched"]
        self.assertTrue(e and all(r["hars_layout"] == "project" and r["hars_project"] == PID for r in e))
        en = json.load(open(os.path.join(self.out, "enrich", "syn-1", os.listdir(os.path.join(self.out, "enrich", "syn-1"))[0])))
        self.assertEqual((en["hars_project"], en["hars_layout"]), (PID, "project"))
        self.assertNotIn("global KB", en["raw_union"])                            # no pollution from the `default` KB
        for q in en["queries"]:
            self.assertIn("graph_branch_enabled", q)
            self.assertIn("lightrag_mode", q)
            self.assertTrue(q["ll_keywords"])
        self.assertIn("JOB0", en["raw_union"])                                    # the project's own content came back

    def test_split_layout_is_polluted_by_the_global_kb_the_finding_of_c8(self):
        with T7.SharedHars() as h:
            h.default_kb = ["UNRELATED global KB chunk about kubernetes ingress 9999"] * 30
            rc, so, se = self.run_ls(self.out, ARMS + ["--hars-layout", "split-session"])
            self.assertEqual(rc, 0, so + se)
        self.assertGreater(h.compact_calls, 0)
        self.assertTrue(all(c["project"] is None and c["sid"] for c in h.correct_calls))

    def test_two_sessions_are_isolated_projects(self):
        F.write_session(self.sd, F.make_session("syn-2"))
        with T7.SharedHars() as h:
            rc, so, se = self.run_ls(self.out, ARMS + RID)
            self.assertEqual(rc, 0, so + se)
        p1, p2 = L.make_project_id("e1", "syn-1"), L.make_project_id("e1", "syn-2")
        self.assertEqual(set(h.projects), {p1, p2})
        self.assertTrue(all(c["path"].startswith(f"/nc-eval/{c['project']}/") for c in h.upsert_calls))
        self.assertFalse(set(h.projects[p1]["docs"]) & set(h.projects[p2]["docs"]))

    def test_resume_is_idempotent_and_does_not_trip_the_empty_gate(self):
        with T7.SharedHars() as h:
            self.assertEqual(self.run_ls(self.out, ARMS + RID)[0], 0)
            n = len(h.upsert_calls)
            os.remove(os.path.join(self.out, "done.jsonl"))
            rc, so, se = T7.call(["--sessions", self.sd, "--resume", self.out] + BASE + ARMS + RID)
        self.assertEqual(rc, 0, so + se)
        self.assertEqual(len(h.upsert_calls), n)                                  # ingest cache: no second write


class ReuseIngestProbeEnrich(Base):
    def first_run(self):
        rc, so, se = self.run_ls(self.out_n(1), ["--arms", "baseline,nc-chunks,nc-note-only", "--hars-layout", "split-session"])
        self.assertEqual(rc, 0, so + se)

    def test_ingest_only_then_probe_then_enrich_reuse_rows_and_notes(self):
        with T7.SharedHars() as h:
            self.first_run()
            c0 = h.compact_calls
            qa = []
            orig = R.LLM.complete

            def spy(self_, user, **kw):
                qa.append(user)
                return orig(self_, user, **kw)
            with mock.patch.object(R.LLM, "complete", spy):
                rc, so, se = self.run_ls(self.out_n(2), ["--reuse-run", self.out_n(1), "--ingest-only"] + RID)
            self.assertEqual(rc, 0, so + se)
            self.assertEqual(qa, [])                                              # ingest only: no model call at all
            self.assertEqual(h.compact_calls, c0)
            n_up = len(h.upsert_calls)
            self.assertGreater(n_up, 0)
            ing = T7.rows_of(self.out_n(2), "ingest.jsonl")
            self.assertEqual([(r["sid"], r["project"], r["ok"]) for r in ing], [("syn-1", PID, True)])
            self.assertFalse(os.path.exists(os.path.join(self.out_n(2), "records.jsonl")))
            # the rolling note of EVERY boundary was ingested (summary section of the first call of each boundary)
            docs = list(h.projects[PID]["docs"])
            self.assertTrue(any("j001-c01" in d for d in docs) and any("j007-c01" in d for d in docs))
            # probe on the ingested projects (no ingest, project-scoped recalls)
            rc, so, se = self.run_ls(self.out_n(3), ARMS + ["--reuse-run", self.out_n(1), "--recall-probe", "--probe-caps", "400,900",
                                                           "--hars-ingested"] + RID)
            self.assertEqual(rc, 0, so + se)
            self.assertEqual(len(h.upsert_calls), n_up)
            pr = T7.rows_of(self.out_n(3), "probe.jsonl")
            self.assertTrue(pr and all(r["hars_project"] == PID for r in pr))
            self.assertTrue(all(c["project"] == PID and c["sid"] is None for c in h.correct_calls))
            h.correct_calls.clear()
            # enrich on the ingested projects: copies baseline / nc-note-only, runs only nc-enriched
            rc, so, se = self.run_ls(self.out_n(4), ARMS + ["--reuse-run", self.out_n(1), "--hars-ingested"] + RID)
            self.assertEqual(rc, 0, so + se)
            self.assertEqual(len(h.upsert_calls), n_up)
            self.assertEqual(len(h.correct_calls), 8)
            self.assertTrue(all(c["project"] == PID and c["sid"] is None for c in h.correct_calls))
        rows = T7.rows_of(self.out_n(4))
        self.assertEqual({r["arm"] for r in rows}, {"baseline", "nc-enriched", "nc-note-only"})
        self.assertTrue(all(r["valid"] for r in rows))
        old = {(r["arm"], r["qid"]): r for r in T7.rows_of(self.out_n(1))}
        for r in rows:
            if r["arm"] != "nc-enriched":
                self.assertEqual({k: v for k, v in r.items() if k not in ("reused_from", "reused_run_id")}, old[(r["arm"], r["qid"])])
        note_only = {r["qid"]: r["memory_text"] for r in rows if r["arm"] == "nc-note-only"}
        self.assertTrue(all(r["memory_text"] == note_only[r["qid"]] for r in rows if r["arm"] == "nc-enriched"))

    def test_enrich_without_ingested_flag_ingests_in_the_run(self):
        with T7.SharedHars() as h:
            self.first_run()
            rc, so, se = self.run_ls(self.out_n(2), ARMS + ["--reuse-run", self.out_n(1)] + RID)
            self.assertEqual(rc, 0, so + se)
            self.assertGreater(len(h.upsert_calls), 0)
        self.assertTrue(all(r["valid"] for r in T7.rows_of(self.out_n(2))))

    def test_project_reuse_needs_the_old_rolling_notes(self):
        import shutil
        with T7.SharedHars():
            self.first_run()
            shutil.rmtree(os.path.join(self.out_n(1), "memory"))
            rc, so, se = self.run_ls(self.out_n(2), ARMS + ["--reuse-run", self.out_n(1)] + RID)
        self.assertEqual(rc, 3)
        self.assertIn("memory/syn-1.json", se)

    def test_dry_run_estimates_include_ingest_unless_already_ingested(self):
        with T7.SharedHars():
            self.first_run()

        def dry(extra):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                rc = R.main(["--sessions", self.sd, "--dry-run", "--json", "--budget-usd", "5"] + BASE + ARMS
                            + ["--reuse-run", self.out_n(1)] + RID + extra)
            self.assertEqual(rc, 0)
            return json.loads(buf.getvalue())
        full, done, only = dry([]), dry(["--hars-ingested"]), dry(["--ingest-only"])
        h_full, h_done = full["sessions"][0]["hars"], done["sessions"][0]["hars"]
        self.assertGreater(h_full["memory_compact_calls"], 0)
        self.assertGreater(h_full["est_ingest_hours"], 0)
        self.assertEqual((h_done["memory_compact_calls"], h_done["extract_usd"], h_done["est_ingest_hours"]), (0, 0.0, 0.0))
        self.assertGreaterEqual(full["grand_total_usd"], done["grand_total_usd"])
        self.assertAlmostEqual(only["grand_total_usd"], h_full["extract_usd"], places=6)       # (mock prices are 0)


class RealClientShape(unittest.TestCase):
    def test_hars_sends_project_and_no_session_id_and_upserts_with_a_virtual_path(self):
        calls = []
        rcm = L.rc()
        with mock.patch.object(rcm, "mcp_call", lambda url, tool, args, timeout=0: calls.append((tool, args)) or {"ok": True}):
            h = R.Hars("http://127.0.0.1:1/mcp", mock.MagicMock(), mock.MagicMock())
            h.recall_correct("ls-x-syn", "q", ["k"], ["t"], 3000, project=PID)
            h.recall("ls-x-syn", "q", False, project=PID)
            h.upsert(PID, R.project_doc_path(PID, 3, 1), "content")
        (t1, a1), (t2, a2), (t3, a3) = calls
        self.assertEqual(t1, "memory_recall")
        self.assertEqual(a1, {"project": PID, "question": "q", "mode": "hybrid", "ll_keywords": ["k"], "hl_keywords": ["t"],
                              "max_chars": 3000})
        self.assertNotIn("session_id", a1)
        self.assertNotIn("top_k", a1)
        self.assertNotIn("session_id", a2)
        self.assertEqual((t3, a3["project"], a3["file_path"]), ("memory_upsert_document", PID, f"/nc-eval/{PID}/j003-c01.md"))
        self.assertTrue(os.path.isabs(a3["file_path"]))

    def test_project_state_normalisation(self):
        ok = R.project_state_of({"ok": True, "project": PID, "node_count": 5, "edge_count": 3,
                                 "storage": {"vector": {"backend": "Nano", "file_count": 3}}}, PID)
        self.assertEqual((ok["registered"], ok["nodes"], ok["vectors"]), (True, 5, 3))
        self.assertFalse(R.project_state_of({"ok": False, "error": "Project 'x' not found", "code": 404}, PID)["registered"])
        self.assertFalse(R.project_state_of({"ok": True, "project": "default"}, PID)["registered"])      # answered for another project


class HomeProvision(T7H.Base):
    def go8(self, *cmd, resume=True, runner=None):
        argv = self.base + list(cmd) + (["--resume", self.camp] if resume else [])
        return T7H.thr.quiet(H.main, argv, runner=runner or T7H.EnrichRunner())

    def test_provision_writes_project_dirs_only_and_prints_the_restart_instruction(self):
        pdir = os.path.join(self.d.name, "projects")
        self.assertEqual(self.go8("oh-provision", "--dry-run", "--projects-dir", pdir)[0], 0)
        self.assertFalse(os.path.exists(pdir))
        rc, so, se = self.go8("oh-provision", "--projects-dir", pdir)
        self.assertEqual(rc, 0, so + se)
        cj = H.read_json(os.path.join(self.camp, "campaign.json"))
        pj = cj["c8_projects"]
        self.assertEqual(len(pj["pids"]), 4)
        names = sorted(os.listdir(pdir))
        self.assertEqual(names, sorted(pj["pids"].values()))
        self.assertNotIn("default", names)
        for pid in names:
            meta = json.load(open(os.path.join(pdir, pid, "project.json")))
            self.assertEqual(meta["project_id"], pid)
            self.assertEqual(meta["index_dir"], os.path.join(pdir, pid, "index"))
            self.assertTrue(pid.startswith(f"nc-{pj['hars_run_id']}-"))
        self.assertIn("RESTART your existing HARS server", so)
        self.assertIn(f"HARS_MEMORY_PROJECTS_DIR={pdir}", so)
        # idempotent: a second provision keeps the same run id
        self.assertEqual(self.go8("oh-provision", "--projects-dir", pdir)[0], 0)
        self.assertEqual(H.read_json(os.path.join(self.camp, "campaign.json"))["c8_projects"]["hars_run_id"], pj["hars_run_id"])

    def test_ingest_and_project_probe_need_provision_and_ingest(self):
        rc, so, se = self.go8("oh-ingest")
        self.assertEqual(rc, 2)
        self.assertIn("oh-provision", se)
        pdir = os.path.join(self.d.name, "projects")
        self.assertEqual(self.go8("oh-provision", "--projects-dir", pdir)[0], 0)
        rc, so, se = self.go8("oh-probe")
        self.assertEqual(rc, 2)
        self.assertIn("oh-ingest", se)
        self.assertEqual(self.go8("oh-ingest", "--dry-run")[0], 0)

    def test_doctor_rows_registered_missing_and_default_untouched(self):
        pdir = os.path.join(self.d.name, "projects")
        self.go8("oh-provision", "--projects-dir", pdir)
        camp = H.Campaign(self.camp, H.load_config(None, {"general": {"artifact_root": self.root}}), runner=mock.Mock(), echo=False)
        pids = sorted(H.read_json(os.path.join(self.camp, "campaign.json"))["c8_projects"]["pids"].values())

        class P:
            def __init__(self, have):
                self.have = have

            def mcp(self, url, tool, args, timeout=60):
                if tool == "memory_list_projects":
                    return {"ok": True, "projects": [{"project_id": p} for p in self.have]}
                return {"ok": True, "node_count": 0}
        rows = {r["check"]: r for r in H.project_doctor_rows(camp.cfg, P(pids + ["default"]), camp)}
        self.assertEqual(rows["projects.registered"]["status"], "OK")
        self.assertEqual(rows["projects.default_untouched"]["status"], "OK")
        self.assertIn("ready for oh-ingest", rows["projects.state"]["detail"])
        rows = {r["check"]: r for r in H.project_doctor_rows(camp.cfg, P(pids[:2]), camp)}
        self.assertEqual(rows["projects.registered"]["status"], "FAIL")
        self.assertIn("restart it once", rows["projects.registered"]["detail"])


if __name__ == "__main__":
    unittest.main()
