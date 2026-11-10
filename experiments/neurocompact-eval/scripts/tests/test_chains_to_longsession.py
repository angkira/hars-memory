"""Amendment C.6: chains_to_longsession.py - OpenHands chains.jsonl -> longsession/1. Offline, deterministic."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

import chains_to_longsession as C
import longsession as L
import qtypes as QT
import run_longsession as R
from oh_chain_fixture import make_chain, write_chains

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Convert(unittest.TestCase):
    def setUp(self):
        self.chain = make_chain()
        self.sess, self.notes = C.convert_chain(self.chain)

    def test_messages_are_the_span_events_in_order_and_ids_are_kept(self):
        s = self.sess
        self.assertEqual((s["format"], s["sid"], s["source"]), ("longsession/1", "chain_syn_c0", "openhands-chain"))
        self.assertEqual(L.validate_session(s), [])
        self.assertEqual(len(s["messages"]), 6 + 5 + 4)
        self.assertEqual([m["idx"] for m in s["messages"]], list(range(15)))
        self.assertEqual([m["role"] for m in s["messages"][:3]], ["tool", "assistant", "tool"])
        self.assertIn("NEEDLE_s2_0", s["messages"][6]["text"])             # span 2 starts at message 6
        self.assertIn("NEEDLE_s3_0", s["messages"][11]["text"])
        self.assertEqual([x["span_idx"] for x in s["meta"]["spans"]], [1, 2, 3])
        self.assertEqual([(x["start"], x["end"]) for x in s["meta"]["spans"]], [(0, 6), (6, 11), (11, 15)])

    def test_compactions_are_the_chains_synthetic_boundaries_with_no_tail(self):
        cs = self.sess["compactions"]
        self.assertEqual([c["at_idx"] for c in cs], [6, 11, 15])           # C3 at len(messages): nothing survives
        self.assertTrue(all(c["original_summary"] is None and c["synthetic"] for c in cs))
        tok = L.message_tokens(self.sess["messages"])
        bs = L.compute_boundaries(tok, window=150000, keep_tail=0, real=cs, final=False)
        self.assertEqual([(b["drop_start"], b["at_idx"]) for b in bs], [(0, 6), (6, 11), (11, 15)])
        self.assertEqual(bs[-1]["tail_tokens"], 0)
        self.assertEqual(bs[-1]["trigger_idx"], 14)

    def test_questions_keep_qids_map_qtype_and_pass_answers_anchors_slices(self):
        by = {q["qid"]: q for q in self.sess["questions"]}
        self.assertEqual(sorted(by), sorted(q["qid"] for q in self.chain["qa"]))
        mh = by["chain_syn_c0_mh_1"]
        self.assertEqual((mh["qtype"], mh["source_qtype"]), ("xspan_locator", "multihop"))
        self.assertEqual(mh["qtype"], QT.canon_qtype("multihop"))
        self.assertEqual(by["chain_syn_c0_1_1"]["qtype"], "single")
        self.assertEqual((mh["answer"], mh["gold_anchor"], mh["slice"]), ("314", "NEEDLE_s3_2", "TAIL"))
        self.assertEqual(mh["meta"]["hop_lags"], [3, 1])
        self.assertEqual(mh["meta"]["link_anchor"], "NEEDLE_s1_2")
        self.assertEqual(by["chain_syn_c0_1_1"]["answer"], "107")
        un = by["chain_syn_c0_0_1"]
        self.assertFalse(un["answerable"])
        self.assertEqual(un["evidence"], [])
        self.assertEqual(un["answer"], C.NOT_MENTIONED)
        self.assertNotIn("gold_anchor", un)

    def test_evidence_points_at_the_right_messages_and_chars(self):
        by = {q["qid"]: q for q in self.sess["questions"]}
        msgs = self.sess["messages"]
        e = by["chain_syn_c0_1_1"]["evidence"]
        self.assertEqual(len(e), 1)
        self.assertEqual(e[0]["msg_idx"], 1)                                # span 1, event 1
        self.assertEqual(msgs[e[0]["msg_idx"]]["text"][e[0]["start"]:e[0]["end"]], "line 101: value NEEDLE_s1_1 = 107")
        # the snippet that crosses two events yields one entry per overlapped message
        e2 = by["chain_syn_c0_2_2"]["evidence"]
        self.assertEqual([x["msg_idx"] for x in e2], [8, 9])                # span 2 starts at 6: events 2 and 3
        self.assertTrue(msgs[8]["text"][e2[0]["start"]:e2[0]["end"]].startswith("NEEDLE_s2_2"))
        self.assertTrue(msgs[9]["text"][e2[1]["start"]:e2[1]["end"]].endswith("NEEDLE_s2_3"))
        # a multihop item carries one hop per span: link in span 1, answer in span 3
        e3 = by["chain_syn_c0_mh_1"]["evidence"]
        spans = {1: range(0, 6), 3: range(11, 15)}
        self.assertIn(e3[0]["msg_idx"], spans[1])
        self.assertIn(e3[1]["msg_idx"], spans[3])
        self.assertEqual(msgs[e3[1]["msg_idx"]]["text"][e3[1]["start"]:e3[1]["end"]], "line 302: value NEEDLE_s3_2 = 314")

    def test_runner_lags_equal_the_chains_design_lags(self):
        tok = L.message_tokens(self.sess["messages"])
        bs = L.compute_boundaries(tok, window=150000, keep_tail=0, real=self.sess["compactions"], final=False)
        want = {"chain_syn_c0_1_1": 3, "chain_syn_c0_2_2": 2, "chain_syn_c0_3_1": 1, "chain_syn_c0_mh_1": 3}
        for q in self.sess["questions"]:
            if q["qid"] in want:
                self.assertEqual(L.question_lag(q, bs, 3), want[q["qid"]], q["qid"])
                self.assertTrue(L.question_eligible(q, bs[2]))
        un = [q for q in self.sess["questions"] if not q["answerable"]][0]
        self.assertIsNone(L.question_lag(un, bs, 3))

    def test_run_longsession_plans_the_converted_session_with_one_end_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            C.write_session(self.sess, d)
            a = R.parse_args(["--sessions", d, "--use-real-compactions", "--keep-tail", "0", "--arms",
                              "baseline,nc-chunks,nc-graph,nc-note-only", "--dry-run", "--provider", "mock"])
            sessions, plans = R.load_plans(a)
            p = plans[0]
            self.assertEqual((len(p["boundaries"]), p["checkpoints"]), (3, [3]))
            self.assertEqual(len(p["ask"][3]), 5)                          # every question is asked, once, at the end

    def test_fallback_to_the_anchor_and_error_when_nothing_matches(self):
        chain = make_chain(unlocatable=True)
        chain["qa"][0]["gold_anchor"] = "NEEDLE_s1_3"                      # present in span 1
        sess, notes = C.convert_chain(chain)
        q = [q for q in sess["questions"] if q["qid"].endswith("_1_1")][0]
        self.assertTrue(any("evidence_fallback" in n for n in notes))
        self.assertEqual(sess["messages"][q["evidence"][0]["msg_idx"]]["text"][q["evidence"][0]["start"]:q["evidence"][0]["end"]],
                         "NEEDLE_s1_3")
        with self.assertRaises(C.ConversionError):
            C.convert_chain(make_chain(unlocatable=True))

    def test_bad_chains_are_rejected(self):
        c = make_chain()
        c["spans"] = []
        with self.assertRaises(C.ConversionError):
            C.convert_chain(c)
        c = make_chain(cid="has space")
        with self.assertRaises(C.ConversionError):
            C.convert_chain(c)
        c = make_chain()
        c["qa"][0]["canonical_short_answer"] = ""
        with self.assertRaises(C.ConversionError):
            C.convert_chain(c)


class Files(unittest.TestCase):
    def test_deterministic_bytes_manifest_and_not_loaded_as_a_session(self):
        with tempfile.TemporaryDirectory() as d:
            chains = [make_chain("chain_a_c0", "sa"), make_chain("chain_b_c0", "sb"), make_chain("chain_bad_c0", "sc", valid=False)]
            src = os.path.join(d, "in", "chains.jsonl")
            write_chains(src, chains)
            outs = []
            for name in ("o1", "o2"):
                o = os.path.join(d, name)
                m = C.convert_file(src, o)
                outs.append(o)
            for fn in ("chain_a_c0.json", "chain_b_c0.json"):
                self.assertEqual(open(os.path.join(outs[0], fn), "rb").read(), open(os.path.join(outs[1], fn), "rb").read())
            self.assertFalse(os.path.exists(os.path.join(outs[0], "chain_bad_c0.json")))
            self.assertEqual([s["sid"] for s in m["sessions"]], ["chain_a_c0", "chain_b_c0"])
            self.assertEqual(m["skipped"], [{"chain_id": "chain_bad_c0", "why": "chain valid=false"}])
            self.assertEqual(m["chains_sha256"], C.sha256_file(src))
            self.assertEqual(m["errors"], [])
            self.assertTrue(os.path.isfile(os.path.join(outs[0], C.MANIFEST_NAME)))
            # the manifest must not be read as a session by the runner / exporter
            self.assertEqual(sorted(s["sid"] for s in L.load_sessions([outs[0]])), ["chain_a_c0", "chain_b_c0"])
            # include / exclude
            m2 = C.convert_file(src, os.path.join(d, "o3"), include=["chain_a_c0"])
            self.assertEqual([s["sid"] for s in m2["sessions"]], ["chain_a_c0"])
            m3 = C.convert_file(src, os.path.join(d, "o4"), exclude=["chain_a_c0"])
            self.assertEqual([s["sid"] for s in m3["sessions"]], ["chain_b_c0"])

    def test_cli_exit_codes_and_shared_session_warning(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "chains.jsonl")
            write_chains(src, [make_chain("chain_a_c0", "same"), make_chain("chain_b_c0", "same")])
            p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "chains_to_longsession.py"), "--chains", src,
                                "--out", os.path.join(d, "o")], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("share OpenHands session same", p.stderr)
            self.assertIn("converted 2/2 chain(s)", p.stdout)
            write_chains(src, [make_chain("chain_a_c0", "x", unlocatable=True)])
            p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "chains_to_longsession.py"), "--chains", src,
                                "--out", os.path.join(d, "o2")], capture_output=True, text=True)
            self.assertEqual(p.returncode, 1)
            self.assertIn("neither located verbatim nor by its anchor", p.stderr)
            p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "chains_to_longsession.py"), "--chains",
                                os.path.join(d, "missing.jsonl"), "--out", d], capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)


if __name__ == "__main__":
    unittest.main()
