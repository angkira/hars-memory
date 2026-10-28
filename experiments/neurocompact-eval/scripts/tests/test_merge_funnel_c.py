"""Amendment C shape for merge_runs.py / funnel_chains.py: 4 arms (baseline / nc-chunks /
nc-graph / nc-note-only), legacy `nc` alias, per-mem_budget keys, funnel stages of item 10.
Offline; fixtures are synthetic run dirs built by the scripts' own selftest helpers."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
sys.dont_write_bytecode = True
import funnel_chains as fc  # noqa: E402
import merge_runs as mr  # noqa: E402


def _quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = fn(*a)
    return rc, buf.getvalue()


class MergeRuns(unittest.TestCase):
    def test_builtin_selftest(self):
        rc, out = _quiet(mr.selftest)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertGreaterEqual(out.count("PASS"), 24)

    def test_four_arm_sweep_merge_across_dirs(self):
        with tempfile.TemporaryDirectory() as T:
            a, b, out = (os.path.join(T, x) for x in ("a", "b", "m"))
            mr._fake_run(a, ["s1", "s2"], budgets=(1536, 1024), nc_tok=1500, base_tok=1500)
            mr._fake_run(b, ["s3"], budgets=(1536, 1024), nc_tok=1200, base_tok=1600)
            rc, o = _quiet(mr.main, ["--out", out, a, b])
            self.assertEqual(rc, 0, o)
            recs = [json.loads(x) for x in open(os.path.join(out, "records.jsonl"))]
            # 3 chains x 2 B x 4 arms x 2 qids
            self.assertEqual(len(recs), 48)
            self.assertEqual({r["arm"] for r in recs}, set(mr.AMENDED_ARMS))
            self.assertEqual(len({r["rid"] for r in recs}), 48)
            rb = json.load(open(os.path.join(out, "run_budget.json")))
            self.assertEqual(set(rb["by_budget"]), {"1536", "1024"})
            self.assertEqual(rb["by_budget"]["1536"]["nc_chunks"]["chains"], 3)
            self.assertIn("nc_graph", rb["by_budget"]["1536"])
            self.assertEqual(rb["mem_budgets"], [1536, 1024])
            self.assertTrue(os.path.isfile(os.path.join(out, "sessions", "s3__b1024", "nc-chunks", "s.jsonl")))

    def test_per_cell_collision_and_invalid_cell(self):
        with tempfile.TemporaryDirectory() as T:
            a, b = os.path.join(T, "a"), os.path.join(T, "b")
            mr._fake_run(a, ["s1"], budgets=(1536, 512))
            mr._fake_run(b, ["s1"], budgets=(512,), invalid={"s1"})
            rc, o = _quiet(mr.main, ["--out", os.path.join(T, "m"), a, b])
            self.assertEqual(rc, 2)
            self.assertIn("s1__b512", o)                      # the colliding cell, not s1__b1536
            self.assertIn("(qid, arm, mem_budget)", o)

    def test_legacy_alias_and_no_mixing(self):
        with tempfile.TemporaryDirectory() as T:
            a, l = os.path.join(T, "a"), os.path.join(T, "l")
            mr._fake_run(a, ["s1"])
            mr._fake_run(l, ["l1"], legacy=True)
            self.assertEqual(mr.canon_arm("nc"), "nc-chunks")
            rc, o = _quiet(mr.main, ["--out", os.path.join(T, "m"), a, l])
            self.assertEqual(rc, 2)
            self.assertIn("Amendment C1", o)

    def test_vendor_switch_is_refused_unless_forced(self):
        man = {"profile": "anthropic", "models": {"agent": "claude-haiku-5-5"},
               "embedder": {"id": "bge-m3"}}
        with tempfile.TemporaryDirectory() as T:
            a, b = os.path.join(T, "a"), os.path.join(T, "b")
            mr._fake_run(a, ["s1"], manifest=man)
            mr._fake_run(b, ["s2"], manifest={**man, "profile": "gemini"})
            rc, o = _quiet(mr.main, ["--out", os.path.join(T, "m"), a, b])
            self.assertEqual(rc, 2)
            self.assertIn("disagree on profile", o)
            rc, o = _quiet(mr.main, ["--out", os.path.join(T, "m2"), a, b, "--allow-mixed-config"])
            self.assertEqual(rc, 0, o)

    def test_agent_model_mix_refused(self):
        with tempfile.TemporaryDirectory() as T:
            a, b = os.path.join(T, "a"), os.path.join(T, "b")
            mr._fake_run(a, ["s1"])
            mr._fake_run(b, ["s2"])
            rows = [json.loads(x) for x in open(os.path.join(b, "chain_budget.jsonl"))]
            rows[0]["agent_model"] = "gemini-3.7-flash"
            mr._w(os.path.join(b, "chain_budget.jsonl"), rows)
            rc, o = _quiet(mr.main, ["--out", os.path.join(T, "m"), a, b])
            self.assertEqual(rc, 2)
            self.assertIn("agent_model", o)


class FunnelChains(unittest.TestCase):
    def test_builtin_selftest(self):
        rc, out = _quiet(fc.selftest)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertGreaterEqual(out.count("PASS"), 19)

    def test_amendment_c_stages_exist_and_apply_by_arm(self):
        for s in ("anchor_in_graph_entities", "anchor_in_graph_relations", "anchor_in_graph",
                  "anchor_in_recall", "anchor_in_kg_recall", "anchor_in_chunk_recall"):
            self.assertIn(s, fc.STAGES)
        self.assertEqual(fc.APPLIES["anchor_in_kg_recall"], {"nc-graph"})
        self.assertEqual(fc.APPLIES["anchor_in_recall"], {"nc-chunks", "nc-graph", "nc-enriched"})
        self.assertIn("nc-note-only", fc.APPLIES["anchor_in_graph"])
        self.assertNotIn("baseline", fc.APPLIES["anchor_in_graph"])

    def test_four_arm_funnel_json_shape_and_merged_input(self):
        """merge_runs output feeds the funnel: per-B cells, nc-graph session store shared with
        nc-chunks (sessions/<chain>__b<B>/nc-chunks)."""
        with tempfile.TemporaryDirectory() as T:
            run, chains, gs = fc._selftest_fixture(T)
            merged = os.path.join(T, "merged")
            rc, o = _quiet(mr.main, ["--out", merged, run])
            self.assertEqual(rc, 0, o)
            oj = os.path.join(T, "f.json")
            rc, o = _quiet(fc.main, ["--run", merged, "--chains", chains, "--graph-store", gs,
                                     "--out-json", oj, "--out-md", os.path.join(T, "f.md")])
            self.assertEqual(rc, 0, o)
            res = json.load(open(oj))
            self.assertEqual(res["meta"]["cells"], ["b1536"])
            self.assertEqual(sorted(res["cells"]["b1536"]["per_arm"]),
                             ["baseline", "nc-chunks", "nc-graph", "nc-note-only"])
            self.assertEqual(res["cells"]["b1536"]["meta"]["treatment"], "nc-graph")
            self.assertIn("## Cell b1536 (primary)", o)


if __name__ == "__main__":
    unittest.main()
