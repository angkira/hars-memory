"""Offline tests (synthetic JSONL fixtures, structure only): Claude Code session discovery
(cc_discover.py + `home_run.py cc --find | --session-id | --list`), the stale-copy doctor check, role->model
table, seeded BEAM sample, campaign-wide caps + total ceiling, dry-run total vs USD 30, OH tarball extraction."""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import tempfile
import unittest

import cc_discover as D
import home_run as H
from test_home_run import FakeProbes, cfg_for, quiet

SID_A = "03023d8a-4fa0-4ff8-8345-a67624798daf"
SID_B = "aaaa1111-2222-3333-4444-555566667777"
SID_C = "cccc9999-8888-7777-6666-555544443333"


def rec(typ, ts, **kw):
    return json.dumps({"type": typ, "timestamp": ts, "isSidechain": False, **kw})


def user(text, ts, uid, sid, **kw):
    return rec("user", ts, uuid=uid, sessionId=sid, message={"role": "user", "content": text}, **kw)


def asst(text, ts, uid, sid):
    return rec("assistant", ts, uuid=uid, sessionId=sid,
               message={"id": "m" + uid, "role": "assistant", "content": [{"type": "text", "text": text}]})


def write_session(path, sid, first, *, n=6, compactions=0, title=None, title_type="ai-title", sidechain=0):
    """Records are chained by parentUuid like real transcripts (the importer walks that chain)."""
    lines = []
    prev = [None]

    def chain(line_fn, uid, **kw):
        lines.append(line_fn(uid, prev[0], **kw))
        prev[0] = uid

    def u(text, ts, uid, **kw):
        chain(lambda i, p, **k: user(text, ts, i, sid, parentUuid=p, **k), uid, **kw)

    def a(text, ts, uid):
        chain(lambda i, p: json.dumps({**json.loads(asst(text, ts, i, sid)), "parentUuid": p}), uid)

    if title:
        key = {"ai-title": "aiTitle", "custom-title": "customTitle", "summary": "summary"}[title_type]
        lines.append(json.dumps({"type": title_type, key: title, "sessionId": sid}))
    lines.append(rec("attachment", "2026-10-01T09:59:00Z", uuid="att" + sid[:4], sessionId=sid))
    u("<command-name>/clear</command-name>", "2026-10-01T10:00:00Z", "u0" + sid[:4], isMeta=True)
    u(first, "2026-10-01T10:00:01Z", "u1" + sid[:4])
    for i in range(n):
        a(f"answer {i}", f"2026-10-0{2 + i % 5}T10:0{i}:00Z", f"a{i}" + sid[:4])
        u(f"follow up {i}", f"2026-10-0{2 + i % 5}T10:0{i}:30Z", f"f{i}" + sid[:4])
    for k in range(compactions):
        bid = f"cb{k}" + sid[:4]
        lines.append(rec("system", "2026-10-09T10:00:00Z", subtype="compact_boundary", uuid=bid, sessionId=sid,
                         parentUuid=None, logicalParentUuid=prev[0],
                         compactMetadata={"trigger": "auto", "preTokens": 150000}))
        prev[0] = bid
        u("This session is being continued from a previous conversation ... Summary: x",
          "2026-10-09T10:00:01Z", f"cs{k}" + sid[:4], isCompactSummary=True)
        a(f"after compaction {k}", "2026-10-09T10:01:00Z", f"ac{k}" + sid[:4])
        u(f"more {k}", "2026-10-09T10:02:00Z", f"mo{k}" + sid[:4])
    for k in range(sidechain):
        lines.append(json.dumps({"type": "user", "isSidechain": True, "timestamp": "2026-10-09T11:00:00Z",
                                 "uuid": f"sc{k}", "sessionId": sid, "message": {"role": "user", "content": "sub"}}))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


class Fixture(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.d.name, "projects")
        p1 = os.path.join(self.root, "-home-me-Project-software-cortex")
        p2 = os.path.join(self.root, "-home-me-other")
        os.makedirs(p1)
        os.makedirs(p2)
        write_session(os.path.join(p1, SID_A + ".jsonl"), SID_A, "Continue the work on HARS experts please",
                      n=30, compactions=3, title="HARS Experts Research", sidechain=2)
        write_session(os.path.join(p1, SID_B + ".jsonl"), SID_B, "fix the flaky test in hars experts research module",
                      n=8, compactions=0)
        write_session(os.path.join(p2, SID_C + ".jsonl"), SID_C, "unrelated chat about cooking", n=4, compactions=1,
                      title="Cooking", title_type="custom-title")
        os.makedirs(os.path.join(p1, SID_A, "subagents"))
        write_session(os.path.join(p1, SID_A, "subagents", "agent-1.jsonl"), "sub1", "subagent HARS Experts Research")

    def tearDown(self):
        self.d.cleanup()


class Discovery(Fixture):
    def test_list_orders_by_compactions_and_counts(self):
        c = D.discover(self.root, list_top=10)
        self.assertEqual([i["file_id"] for i in c], [SID_A, SID_C, SID_B])      # subagents/ never entered
        a = c[0]
        self.assertEqual(a["compactions"], 3)
        self.assertEqual(a["compact_summaries"], 3)
        self.assertEqual(a["messages"], 2 + 60 + 3 * 3)    # raw user/assistant records (meta line + first + 30 pairs + 3x(summary, assistant, user)); sidechains excluded
        self.assertEqual(a["first_ts"][:10], "2026-10-01")
        self.assertEqual(a["last_ts"][:10], "2026-10-09")
        self.assertEqual(a["titles"], ["HARS Experts Research"])
        self.assertEqual(a["first_user"][0], "Continue the work on HARS experts please")  # meta/command line skipped
        self.assertGreater(a["size"], 1000)

    def test_find_by_title_first_message_case_and_space_insensitive(self):
        c = D.discover(self.root, find="hars   experts research")
        self.assertEqual([i["file_id"] for i in c], [SID_A, SID_B])
        self.assertEqual(c[0]["matched"], "title")
        self.assertEqual(c[1]["matched"], "first message")
        self.assertEqual(D.discover(self.root, find="cooking")[0]["file_id"], SID_C)
        self.assertEqual(D.discover(self.root, find="no such words anywhere"), [])

    def test_find_by_session_id_full_prefix_cse_and_uuid_in_records(self):
        for q in (SID_A, "cse_" + SID_A, SID_A[:8].upper(), "u1" + SID_A[:4]):
            c = D.discover(self.root, session_id=q)
            self.assertTrue(c and c[0]["file_id"] == SID_A, q)
        self.assertEqual(D.discover(self.root, session_id="deadbeef-0000-0000-0000-000000000000"), [])

    def test_table_has_the_columns_the_owner_picks_from(self):
        t = D.format_candidates(D.discover(self.root, find="hars experts research"))
        for w in ("size", "dates", "msgs", "compact", SID_A[:11], "HARS Experts Research", "2026-10-01..2026-10-09"):
            self.assertIn(w, t)

    def test_cli_exit_codes(self):
        self.assertEqual(D.main(["--root", self.root, "--find", "cooking"]), 0)
        self.assertEqual(D.main(["--root", self.root, "--find", "nothing-matches-this"]), 1)


class HomeRunCc(Fixture):
    def ns(self, **kw):
        base = dict(projects_root=self.root, find=None, session_id=None, list=False, pick=None, top=15)
        base.update(kw)
        return argparse.Namespace(**base)

    def cfg(self):
        return H.load_config(None, {})

    def test_list(self):
        buf = io.StringIO()
        rc, ch = H.cc_discover_cmd(self.cfg(), self.ns(list=True), out=lambda s: buf.write(s + "\n"))
        self.assertEqual((rc, ch), (0, None))
        self.assertIn(SID_A[:11], buf.getvalue())
        self.assertIn("compact", buf.getvalue())

    def test_find_single_match_is_picked_and_ambiguous_needs_a_pick(self):
        out = []
        rc, ch = H.cc_discover_cmd(self.cfg(), self.ns(find="cooking"), out=out.append)
        self.assertEqual(rc, 0)
        self.assertEqual(ch["file_id"], SID_C)
        # two candidates, no terminal, no --pick: nothing chosen, exit 3, nothing spent
        rc, ch = H.cc_discover_cmd(self.cfg(), self.ns(find="hars experts research"), isatty=False, out=out.append)
        self.assertEqual((rc, ch), (3, None))
        self.assertIn("--pick N", "\n".join(out))
        # interactive pick
        rc, ch = H.cc_discover_cmd(self.cfg(), self.ns(find="hars experts research"), isatty=True,
                                   input_fn=lambda p: "1", out=out.append)
        self.assertEqual(ch["file_id"], SID_A)
        rc, ch = H.cc_discover_cmd(self.cfg(), self.ns(find="hars experts research"), isatty=True,
                                   input_fn=lambda p: "q", out=out.append)
        self.assertEqual((rc, ch), (3, None))
        # --pick
        rc, ch = H.cc_discover_cmd(self.cfg(), self.ns(find="hars experts research", pick=1), out=out.append)
        self.assertEqual(ch["file_id"], SID_A)
        with self.assertRaises(H.HomeRunError):
            H.cc_discover_cmd(self.cfg(), self.ns(find="hars experts research", pick=9), out=out.append)

    def test_session_without_compaction_and_no_match_are_refused_with_advice(self):
        with self.assertRaises(H.HomeRunError) as cm:
            H.cc_discover_cmd(self.cfg(), self.ns(session_id=SID_B), out=lambda s: None)
        self.assertIn("NO /compact", str(cm.exception))
        with self.assertRaises(H.HomeRunError) as cm:
            H.cc_discover_cmd(self.cfg(), self.ns(find="zzz"), out=lambda s: None)
        self.assertIn("--list", str(cm.exception))
        with self.assertRaises(H.HomeRunError):
            H.cc_discover_cmd(self.cfg(), self.ns(find="x", projects_root=os.path.join(self.d.name, "empty")),
                              out=lambda s: None)

    def test_main_find_imports_the_picked_session_with_redaction_and_records_the_pick(self):
        root = os.path.join(self.d.name, "art")
        argv = ["--root", root, "--mcp", "http://127.0.0.1:1/mcp", "cc", "--session-id", "cse_" + SID_A,
                "--projects-root", self.root, "--skip-judge", "--gen-cap", "1", "--run-cap", "1", "--judge-cap", "1"]
        rc, so, se = quiet(H.main, argv + ["--list"])               # --list never imports
        self.assertEqual(rc, 0, so + se)
        self.assertFalse(os.path.exists(os.path.join(root, ".homerun-current")))
        # a real run needs QA generation (LLM): stop at the missing key, AFTER the import + the recorded pick
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            rc, so, se = quiet(H.main, argv)
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old
        camp = open(os.path.join(root, ".homerun-current")).read()
        sel = H.read_json(os.path.join(camp, "cc", "selected-session.json"))
        self.assertTrue(sel["path"].endswith(SID_A + ".jsonl"))
        self.assertEqual(sel["compactions"], 3)
        self.assertTrue(os.path.isdir(os.path.join(camp, "cc", "data")), so + se)
        imported = os.listdir(os.path.join(camp, "cc", "data"))
        self.assertEqual(len(imported), 1)
        self.assertIn("redaction", so)
        # the same campaign refuses a different session
        rc, so, se = quiet(H.main, ["--root", root, "cc", "--find", "cooking", "--projects-root", self.root,
                                       "--resume", camp])
        self.assertEqual(rc, 2)
        self.assertIn("--new", se)


class StaleCopyAndModels(unittest.TestCase):
    def test_current_checkout_is_not_flagged_as_stale_by_files(self):
        rows = H.stale_copy_rows(FakeProbes())
        self.assertNotEqual(rows[0]["status"], "FAIL", rows)

    def test_incomplete_copy_fails(self):
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "nc-eval", "scripts")
            os.makedirs(base)
            rows = H.stale_copy_rows(FakeProbes(), base=base)
            self.assertEqual(rows[0]["status"], "FAIL")
            self.assertIn("STALE", rows[0]["detail"])
            self.assertIn("obsolete", rows[0]["detail"])

    def test_wrong_version_marker_fails_and_scratch_path_warns(self):
        with tempfile.TemporaryDirectory() as td:
            ev = os.path.join(td, "eval")
            sc = os.path.join(ev, "scripts")
            os.makedirs(sc)
            os.makedirs(os.path.join(ev, "profiles"))
            os.makedirs(os.path.join(ev, "pi-agent", "extensions"))
            for p in ("scripts/providers.py", "scripts/cc_discover.py", "scripts/run_longsession.py", "profiles/anthropic.json",
                      "pi-agent/extensions/neurocompact.ts", "REPORT-PROTOCOL.md"):
                open(os.path.join(ev, p), "w").write("Amendment C.3\n")
            open(os.path.join(ev, "EVAL-VERSION"), "w").write("1999\n")
            rows = H.stale_copy_rows(FakeProbes(), base=sc)
            self.assertEqual(rows[0]["status"], "FAIL")
            self.assertIn("version marker", rows[0]["detail"])
            open(os.path.join(ev, "EVAL-VERSION"), "w").write(H.HOME_RUN_VERSION + "\n")
            rows = H.stale_copy_rows(FakeProbes(), base=sc)
            self.assertEqual(rows[0]["status"], "WARN")          # fake git: not a checkout

    def test_version_file_matches_code(self):
        self.assertEqual(open(os.path.join(os.path.dirname(H.BASE), "EVAL-VERSION")).read().strip(), H.HOME_RUN_VERSION)

    def test_role_table_and_every_default_is_haiku_5_5(self):
        lines, warns = H.role_model_table(H.load_config(None, {}))
        txt = "\n".join(lines)
        for r in ("agent", "answer", "summarizer", "extractor", "keyword"):
            self.assertRegex(txt, rf"{r}\s+claude-haiku-5-5")
        self.assertRegex(txt, r"judge\s+claude-sonnet-5-5")
        self.assertEqual(warns, [])

    def test_no_stale_model_defaults_anywhere_in_scripts_and_profiles(self):
        import re
        ev = os.path.dirname(H.BASE)
        bad = re.compile(r"haiku-4|claude-3|haiku-4-5|3-5-haiku")
        hits = []
        for sub in ("scripts", "profiles", "pi-agent"):
            for root, _d, fns in os.walk(os.path.join(ev, sub)):
                if "tests" in root or "__pycache__" in root or "node_modules" in root:
                    continue
                for fn in fns:
                    if fn.endswith((".py", ".json", ".ts")):
                        for i, line in enumerate(open(os.path.join(root, fn), encoding="utf-8", errors="replace"), 1):
                            if bad.search(line):
                                hits.append(f"{fn}:{i}: {line.strip()[:100]}")
        self.assertEqual(hits, [])

    def test_doctor_includes_stale_row_and_table(self):
        rows = H.doctor_checks(H.load_config(None, {}), FakeProbes(), live=False, offline=True)
        self.assertIn("stale_copy", [r["check"] for r in rows])


class BeamSampleAndCaps(unittest.TestCase):
    def mk_repo(self, td, n=35):
        for i in range(1, n + 1):
            os.makedirs(os.path.join(td, "chats", "1M", str(i)))
        return td

    def test_seeded_sample_is_deterministic_sorted_and_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            repo = self.mk_repo(td)
            cfg = H.load_config(None, {"beam": {"repo": repo}})
            a, b = H.beam_selection(cfg), H.beam_selection(cfg)
            self.assertEqual(a, b)
            self.assertEqual(len(a["selected"]), 10)
            self.assertEqual(a["selected"], sorted(a["selected"]))
            self.assertEqual(a["population"], list(range(1, 36)))
            self.assertEqual(a["seed"], 20261026)
            other = H.beam_selection(H.load_config(None, {"beam": {"repo": repo, "seed": 7}}))
            self.assertNotEqual(a["selected"], other["selected"])
            # pinned: a change of the sampling procedure would silently change the decision set
            self.assertEqual(a["selected"], H.beam_selection(cfg)["selected"])
            self.assertIsNone(H.beam_selection(H.load_config(None, {"beam": {"repo": repo, "chats": "all"}})))
            small = H.beam_selection(H.load_config(None, {"beam": {"repo": self.mk_repo(os.path.join(td, "s"), 6)}}))
            self.assertEqual(small["selected"], [1, 2, 3, 4, 5, 6])

    def test_caps_are_campaign_wide_unless_overridden_and_total_is_hard(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = cfg_for(td, caps={"gen": 3.0, "run": 20.0, "judge": 5.0, "total": 24.0})
            camp = H.open_campaign(cfg, H.build_parser().parse_args(["dry-run"]), "dry-run", echo=False)
            camp.charge("oh", "run", "oh.run", 8.0, "x")
            camp.charge("beam", "run", "beam.run", 9.0, "y")
            self.assertAlmostEqual(camp.remaining("cc", "run"), 3.0)                 # 20 - 8 - 9
            camp.charge("cc", "run", "cc.run", 2.9, "z")
            camp.charge("oh", "judge", "oh.judge", 4.0, "j")                        # 19.9 + 4.0 = 23.9 of 24
            self.assertAlmostEqual(camp.remaining("beam", "judge"), 0.1)             # total binds before the judge cap
            with self.assertRaises(H.HomeRunError):
                camp.charge("beam", "judge", "beam.judge", 0.2, "j2")
                camp.remaining("beam", "judge")
            cfg2 = cfg_for(td + "/b", caps={"run": 20.0, "beam": {"run": 12.0}})
            c2 = H.open_campaign(cfg2, H.build_parser().parse_args(["dry-run"]), "dry-run", echo=False)
            c2.charge("oh", "run", "oh.run", 9.0, "x")
            self.assertAlmostEqual(c2.remaining("beam", "run"), 12.0)                # own cap, own spend
            self.assertAlmostEqual(c2.remaining("cc", "run"), 11.0)


class OhTar(unittest.TestCase):
    def test_extract_into_artifact_root_and_autodetect(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src", "20261026T144511Z-oh-multihop-qa")
            os.makedirs(src)
            open(os.path.join(src, "chains.jsonl"), "w").write('{"chain_id": "a"}\n')
            tp = os.path.join(td, H.OH_TAR_DEFAULT)
            with tarfile.open(tp, "w:gz") as tf:
                tf.add(src, arcname="20261026T144511Z-oh-multihop-qa")
            root = os.path.join(td, "root")
            d = H.extract_oh_tar(tp, root)
            self.assertTrue(os.path.isfile(os.path.join(d, "chains.jsonl")))
            cfg = H.load_config(None, {"general": {"artifact_root": root}})
            self.assertEqual(cfg["oh"]["artifact_dir"], d)                           # auto-detected, no config edit
            bad = os.path.join(td, "bad.tar.gz")
            with tarfile.open(bad, "w:gz") as tf:
                ti = tarfile.TarInfo("../evil.txt")
                ti.size = 1
                tf.addfile(ti, io.BytesIO(b"x"))
            with self.assertRaises(H.HomeRunError):
                H.extract_oh_tar(bad, os.path.join(td, "r2"))
            with self.assertRaises(H.HomeRunError):
                H.extract_oh_tar(os.path.join(td, "missing.tar.gz"), root)


if __name__ == "__main__":
    unittest.main()
