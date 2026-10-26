"""Offline tests for Protocol Amendment C (nc-graph arm, budget sweep, secondaries, M3).

No network, no API key, no spend ($0): the runner end-to-end test replaces pi and HARS with
in-process fakes; the metrics tests use synthetic records; the TypeScript selftest runs under
node with stub packages.

Run:  cd experiments/neurocompact-eval/scripts && python -m pytest tests -q
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
sys.dont_write_bytecode = True
os.environ.setdefault("NC_PACK_QUIET", "1")
import judge_v2  # noqa: E402
import metrics_v2 as mv  # noqa: E402
import run_chainreplay as rc  # noqa: E402
import visible_context as vc  # noqa: E402

EXT = os.path.join(os.path.dirname(SCRIPTS), "pi-agent", "extensions", "neurocompact.ts")


def _ns(**kw):
    base = dict(mem_budget=1536, recall_cap_min=256, recall_cap_max=1024,
                baseline_sum_max_tokens=1536, note_sum_max_tokens=640,
                agent_model="claude-haiku-5-5", provider="anthropic",
                judge_model="openai/gpt-6-luna", allow_degraded=False,
                extractor_model="claude-haiku-5-5")
    base.update(kw)
    return types.SimpleNamespace(**base)


# --------------------------------------------------------------------------- runner

class FrozenFormulas(unittest.TestCase):
    def test_budget_table_is_the_frozen_one(self):
        got = {b: (rc.note_cap_formula(b), rc.recall_floor_formula(b), rc.recall_ceil_formula(b))
               for b in (1536, 1024, 512)}
        self.assertEqual(got, {1536: (640, 256, 1024), 1024: (427, 171, 683),
                               512: (213, 85, 341)})

    def test_args_for_budget_sets_every_knob_and_leaves_single_b_alone(self):
        base = _ns()
        self.assertIs(rc.args_for_budget(base, 1536, sweep=False), base)
        a = rc.args_for_budget(base, 1024, sweep=True)
        self.assertEqual((a.mem_budget, a.baseline_sum_max_tokens, a.note_sum_max_tokens,
                          a.recall_cap_min, a.recall_cap_max, a.b_tag),
                         (1024, 1024, 427, 171, 683, True))
        self.assertEqual(base.mem_budget, 1536)            # the original is untouched

    def test_recall_cap_clamps_per_budget(self):
        for b, note, want in ((1536, 600, 936), (1536, 300, 1024), (1024, 400, 624),
                              (1024, 200, 683), (512, 200, 312), (512, 100, 341),
                              (512, 450, 85)):
            a = rc.args_for_budget(_ns(), b, True)
            self.assertEqual(rc.recall_cap_for(note, a)["recall_cap"], want, (b, note))

    def test_frozen_recall_call_shapes(self):
        self.assertEqual(rc.recall_args("s", "q"), {"session_id": "s", "question": "q", "top_k": 8})
        self.assertEqual(rc.recall_args("s", "q", graph=True),
                         {"session_id": "s", "question": "q", "top_k": 8,
                          "session_graph": True, "session_graph_mode": "hybrid"})

    def test_arm_alias_and_order(self):
        self.assertEqual(rc.ARM_ALIASES["nc"], "nc-chunks")
        self.assertEqual(rc.ARM_ORDER, ("baseline", "nc-chunks", "nc-graph", "nc-note-only"))
        env_g = rc.base_env("nc-graph", "m", "sid", "http://x", None)
        env_c = rc.base_env("nc-chunks", "m", "sid", "http://x", None)
        self.assertEqual({k: v for k, v in env_g.items() if k.startswith("NC_")},
                         {k: v for k, v in env_c.items() if k.startswith("NC_")})
        self.assertEqual(rc.base_env("nc-note-only", "m", "s", "u", None)["NC_INJECT_RECALL"], "0")


class GraphGates(unittest.TestCase):
    def test_failure_reasons(self):
        ok = {"session_graph": {"status": "ok", "entities": 3, "relations": 2}}
        self.assertEqual(rc.graph_failure_reasons(ok), [])
        empty = {"session_graph": {"status": "empty", "entities": 0, "relations": 0}}
        self.assertEqual(rc.graph_failure_reasons(empty), [])        # empty KG is NOT a failure
        for st in ("error", "skipped", "unknown_session"):
            self.assertTrue(rc.graph_failure_reasons({"session_graph": {"status": st}}), st)
        self.assertTrue(rc.graph_failure_reasons({"context": "x"}))   # block absent

    def test_summary_is_tolerant(self):
        g = rc.graph_summary({"session_graph": {
            "status": "ok", "keyword_source": "llm", "ll_keywords": ["a"], "entities": 2,
            "relations": 0, "latency_ms": 1234.5, "keyword_usage": {"input": 400, "output": 50}}})
        self.assertEqual((g["keyword_source"], g["kg_empty"], g["keyword_tokens_in"],
                          g["keyword_tokens_out"]), ("llm", False, 400.0, 50.0))
        g0 = rc.graph_summary({"session_graph": {"status": "empty", "entities": 0, "relations": 0}})
        self.assertTrue(g0["kg_empty"])
        self.assertIsNone(g0["keyword_source"])
        self.assertEqual(rc.graph_summary({})["session_graph_status"], None)    # never raises

    def test_split_packed_sections(self):
        kg, ch = rc.split_packed_sections(
            "### Direct relations / entities\n- A -> B: x\n\n[rank 1] chunk text")
        self.assertIn("A -> B", kg)
        self.assertNotIn("A -> B", ch)
        self.assertIn("chunk text", ch)
        self.assertEqual(rc.split_packed_sections("[rank 1] only"), ("", "[rank 1] only"))

    def _bctx(self, graph_failures=None, kg_empty=None, chunk_failures=None, graph_tokens=None):
        comps = {"baseline": [{"span_idx": 3, "realized_tokens": 1400, "cap": 1536}],
                 "nc-chunks": [{"span_idx": 3, "realized_tokens": 600, "cap": 640}]}
        return rc.chain_budget("c1", _ns(), comps, [900, 910], True,
                               chunk_failures or {"q1": [], "q2": []}, {"q1": None, "q2": None},
                               graph_tokens, graph_failures, kg_empty)

    def test_graph_error_invalidates_the_chain(self):
        b = self._bctx({"q1": ["session_graph status=error: boom"], "q2": []},
                       {"q1": False, "q2": False}, graph_tokens=[900, 900])
        self.assertFalse(b["valid"])
        self.assertTrue(any("RECALL FAIL (nc-graph)" in r for r in b["invalid_reasons"]))
        self.assertEqual(b["graph_recall_failed_qids"], ["q1"])

    def test_empty_kg_is_reported_not_invalidating(self):
        b = self._bctx({"q1": [], "q2": []}, {"q1": True, "q2": True}, graph_tokens=[880, 890])
        self.assertTrue(b["valid"])
        self.assertEqual(b["kg_empty_rate"], 1.0)
        self.assertTrue(b["graph_not_exercised"])
        self.assertEqual(b["nc_graph_memory_total"], 600 + 885)
        self.assertEqual(b["ratio_graph"], round((600 + 885) / 1400, 3))

    def test_legacy_call_without_graph_args_unchanged(self):
        b = rc.chain_budget("c1", _ns(), {"baseline": [{"span_idx": 3, "realized_tokens": 1400,
                            "cap": 1536}], "nc-chunks": [{"span_idx": 3, "realized_tokens": 600,
                            "cap": 640}]}, [900], True, {"q": []}, {"q": None})
        self.assertTrue(b["valid"])
        self.assertIsNone(b["ratio_graph"])
        self.assertIsNone(b["kg_empty_rate"])


class AlignmentAndDryRun(unittest.TestCase):
    def test_multihop_legs_are_all_verified(self):
        chain = {"spans": [{"span_idx": i, "segments": [{"role": "a", "text": t}]}
                           for i, t in ((1, "foo set to 3"), (2, "mid"), (3, "foo is now 7 later"))],
                 "qa": [{"qid": "ok", "span_idx": 3, "evidence": "x", "answerable": True,
                         "evidence_spans": [{"span_idx": 1, "evidence": "foo set to 3"},
                                            {"span_idx": 3, "evidence": "foo is now 7"}]},
                        {"qid": "bad", "span_idx": 3, "evidence": "x", "answerable": True,
                         "evidence_spans": [{"span_idx": 1, "evidence": "foo set to 3"},
                                            {"span_idx": 2, "evidence": "NOT THERE"}]}]}
        res = {r["qid"]: r["ok"] for r in rc.check_alignment(chain)}
        self.assertEqual(res, {"ok": True, "bad": False})

    def test_dry_run_sweep_and_alias(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "chains.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps(_chain("chain_S0_c0", "S0")) + "\n")
            run = lambda *a: subprocess.run(   # noqa: E731
                [sys.executable, os.path.join(SCRIPTS, "run_chainreplay.py"), "--dry-run",
                 "--chains", path, "--agent-model", "claude-haiku-5-5", "--provider", "anthropic",
                 *a], capture_output=True, text=True, check=True).stdout
            out = run("--arms", "baseline,nc,nc-note-only")
            self.assertIn("arms: baseline,nc-chunks,nc-note-only", out)
            self.assertNotIn("BUDGET SWEEP", out)
            out = run("--mem-budgets", "512,1536,1024")
            self.assertIn("arms: baseline,nc-chunks,nc-graph,nc-note-only", out)
            self.assertIn("BUDGET SWEEP", out)
            self.assertIn("B in [1536, 1024, 512]", out)               # primary first
            self.assertIn("note cap 427 | recall cap = clamp(B - realized note, 171, 683)", out)
            self.assertIn('"session_graph": true', out)
            bad = subprocess.run([sys.executable, os.path.join(SCRIPTS, "run_chainreplay.py"),
                                  "--dry-run", "--chains", path, "--arms", "baseline,nc-graph"],
                                 capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)

    def test_judge_family_refusal_not_weakened(self):
        # the judge must differ in family from the agent: a claude judge is refused
        self.assertIsNotNone(rc.family_conflict("claude-haiku-5-5", "claude-sonnet-5-5"))
        self.assertIsNotNone(rc.family_conflict("claude-haiku-5-5", "anthropic/claude-opus-4"))
        self.assertIsNone(rc.family_conflict("claude-haiku-5-5", "openai/gpt-6-luna"))


def _chain(cid, sess, multihop=False):
    spans = []
    for i in (1, 2, 3):
        segs = [{"role": "assistant",
                 "text": f"span{i} seg{j} ran pytest tests/test_mod{i}.py and saw FooError_{i}{j} "
                         f"at line {10 * i + j}. " * 5} for j in range(2)]
        spans.append({"span_idx": i, "lag": 4 - i, "segments": segs})
    qa = []
    for i in (1, 2, 3):
        qa.append({"qid": f"{cid}_{i}_1", "chain_id": cid, "session_id": sess, "span_idx": i,
                   "lag": 4 - i, "slice": "HEAD", "question": f"Which error in span {i}?",
                   "canonical_short_answer": f"FooError_{i}0", "gold_anchor": f"FooError_{i}0",
                   "evidence": f"ran pytest tests/test_mod{i}.py and saw FooError_{i}0 at line "
                               f"{10 * i}.", "answerable": True})
    if multihop:
        qa.append({"qid": f"{cid}_mh_1", "chain_id": cid, "session_id": sess, "span_idx": 3,
                   "lag": 3, "slice": "TAIL", "question": "What came after FooError_10?",
                   "canonical_short_answer": "FooError_30", "gold_anchor": "FooError_30",
                   "qtype": "multihop", "answerable": True,
                   "evidence": "x", "evidence_spans": [
                       {"span_idx": 1, "evidence": "ran pytest tests/test_mod1.py and saw "
                                                   "FooError_10 at line 10."},
                       {"span_idx": 3, "evidence": "ran pytest tests/test_mod3.py and saw "
                                                   "FooError_30 at line 30."}]})
    qa.append({"qid": f"{cid}_0_1", "chain_id": cid, "session_id": sess, "span_idx": 0, "lag": 0,
               "slice": "NONE", "question": "What was NC_PHANTOM?", "canonical_short_answer": "",
               "gold_anchor": "", "evidence": "", "answerable": False})
    return {"chain_id": cid, "session_id": sess, "chain_idx": 0, "spans": spans, "qa": qa}


# ------------------------------------------------------- runner end to end (fake pi/HARS)

class FakeWorld:
    """In-process stand-in for pi + HARS so run_chainreplay.main() runs end to end offline."""

    def __init__(self, graph_status="ok", graph_block=True):
        self.compactions: dict[str, int] = {}
        self.recalls: list[dict] = []
        self.graph_status, self.graph_block = graph_status, graph_block

    # ---- pi
    def run_pi_turn(self, argv, env, timeout):
        d = argv[argv.index("--session-dir") + 1]
        sid = argv[argv.index("--session-id") + 1]
        prompt = argv[-1]
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "s.jsonl")
        events, metrics, ncomp, text = [], [], 0, "ok"
        thr = int(env["NC_FORCE_COMPACT_THRESHOLD"])
        with open(path, "a") as f:
            f.write(json.dumps({"type": "message", "message": {"role": "user", "content": [
                {"type": "text", "text": prompt}]}}) + "\n")
            if thr < rc.FORCE_OFF:                     # boundary turn: one compaction
                cap = int(env["NC_SUM_MAX_TOKENS"])
                n = int(cap * 0.85)
                summary = "## GOAL\n" + " ".join(["word"] * int(n * 0.9)) + "\nACTIVE STATE: x"
                f.write(json.dumps({"type": "compaction", "summary": summary}) + "\n")
                persisted = env["NC_PERSIST"] == "1"
                if persisted:
                    self.compactions[sid] = self.compactions.get(sid, 0) + 1
                events = [{"type": "compaction_end", "result": {
                    "summary": summary, "tokensBefore": 9000, "estimatedTokensAfter": 700}}]
                metrics = [{"stop": "stop", "realized_tokens": n, "max_tokens": cap,
                            "persisted": persisted, "model": "fake", "reasoning": "off"}]
                ncomp = 1
            elif prompt.startswith(rc.ASK_PROMPT):
                text = "FooError_10" if "<memory>" in prompt else "I do not know"
            f.write(json.dumps({"type": "message", "message": {"role": "assistant", "content": [
                {"type": "text", "text": text}]}}) + "\n")
        return {"out": "", "err": "", "wall": 0.01, "text": text,
                "usage": {"input": 5000, "output": 20}, "compactions": 0 if events == [] else 0,
                "ok": True, "metrics": metrics, "compaction_events": events}

    # ---- HARS
    def hars_status(self, mcp, sid):
        return {"compactions": self.compactions.get(sid, 0), "entities": 4, "relations": 3,
                "vectors_indexed": 9}

    def hars_recall(self, mcp, sid, question, timeout=600, graph=False):
        self.recalls.append({"sid": sid, "graph": graph})
        chunk = json.dumps({"reference_id": "1", "content": "ran pytest and saw FooError_10 at "
                            "line 10 in tests/test_mod1.py during span one replay."})
        raw = ["Document Chunks (each has a reference_id):", "```json", chunk, "```", ""]
        payload = {"ok": True, "session": {"status": "ok", "vectors_indexed": 9}}
        if graph:
            raw += ["Knowledge Graph Data (Entity):", "```json",
                    json.dumps({"entity": "FooError_10", "type": "ERROR",
                                "description": "raised by tests/test_mod1.py"}), "```", ""]
            if self.graph_block:
                payload["session_graph"] = {
                    "status": self.graph_status, "keyword_source": "llm",
                    "ll_keywords": ["FooError_10"], "hl_keywords": ["test failure"],
                    "entities": 1, "relations": 0, "latency_ms": 900.0, "keyword_latency_ms": 700.0}
        payload["context"] = "\n".join(raw)
        return payload


class RunnerEndToEnd(unittest.TestCase):
    def _run(self, world, argv_extra, chains):
        td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, td, True)
        cpath = os.path.join(td, "chains.jsonl")
        with open(cpath, "w") as f:
            for c in chains:
                f.write(json.dumps(c) + "\n")
        out = os.path.join(td, "run")
        saved = {k: getattr(rc, k) for k in (
            "run_pi_turn", "hars_status", "hars_recall", "write_settings", "write_mcp_url",
            "write_models_base_url", "restore", "restore_models_json", "WORKDIR")}
        rc.run_pi_turn, rc.hars_status, rc.hars_recall = (
            world.run_pi_turn, world.hars_status, world.hars_recall)
        rc.write_settings = lambda *a, **k: "bk"
        rc.write_mcp_url = lambda *a, **k: "bk"
        rc.write_models_base_url = lambda *a, **k: None
        rc.restore = lambda *a, **k: None
        rc.restore_models_json = lambda *a, **k: None
        rc.WORKDIR = os.path.join(td, "wd")
        glob_state = (rc.PROVIDER, rc.CTX_WINDOW_OVERRIDE, rc.AGENT_THINKING, dict(rc.CAPS))
        argv = ["run_chainreplay.py", "--chains", cpath, "--out", out, "--run-id", "t1",
                "--agent-model", "claude-haiku-5-5", "--provider", "anthropic",
                "--mcp", "http://127.0.0.1:1/mcp", *argv_extra]
        old_argv, old_key = sys.argv, os.environ.get("ANTHROPIC_API_KEY")
        sys.argv = argv
        os.environ["ANTHROPIC_API_KEY"] = "x"
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rcode = rc.main()
        finally:
            sys.argv = old_argv
            if old_key is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                os.environ["ANTHROPIC_API_KEY"] = old_key
            for k, v in saved.items():
                setattr(rc, k, v)
            rc.PROVIDER, rc.CTX_WINDOW_OVERRIDE, rc.AGENT_THINKING = glob_state[:3]
            rc.CAPS.clear()
            rc.CAPS.update(glob_state[3])
        recs = [json.loads(x) for x in open(os.path.join(out, "records.jsonl")) if x.strip()]
        return rcode, recs, out, buf.getvalue()

    def test_sweep_writes_per_budget_records_for_four_arms(self):
        world = FakeWorld()
        rcode, recs, out, log = self._run(
            world, ["--mem-budgets", "512,1536"], [_chain("chain_S0_c0", "S0", multihop=True)])
        self.assertEqual(rcode, 0, log[-2000:])
        by = {}
        for r in recs:
            by.setdefault((r["arm"], r["mem_budget"]), []).append(r)
        self.assertEqual(set(by), {(a, b) for a in ("baseline", "nc-chunks", "nc-graph",
                                                    "nc-note-only") for b in (1536, 512)})
        n_q = 5                                        # 3 single + 1 multihop + 1 unanswerable
        self.assertTrue(all(len(v) == n_q for v in by.values()))
        rids = [r["rid"] for r in recs]
        self.assertEqual(len(rids), len(set(rids)))                      # unique across B
        self.assertTrue(all(r["valid"] is True for r in recs), [r["chain_invalid_reasons"] for r in recs][:2])
        # primary budget ran first
        self.assertEqual(recs[0]["mem_budget"], 1536)
        # per-B caps in the records (512: ceiling 341)
        self.assertLessEqual(by[("nc-chunks", 512)][0]["recall_cap"], 341)
        self.assertLessEqual(by[("nc-chunks", 1536)][0]["recall_cap"], 1024)
        r512 = by[("nc-graph", 512)][0]
        self.assertLessEqual(r512["recall_packed_tokens"], 341)
        # graph rows carry the readout, chunk rows do not
        g = by[("nc-graph", 1536)][0]
        self.assertEqual((g["keyword_source"], g["session_graph_status"], g["kg_empty"],
                          g["keyword_model"], g["session_graph_mode"]),
                         ("llm", "ok", False, "claude-haiku-5-5", "hybrid"))
        self.assertNotIn("keyword_source", by[("nc-chunks", 1536)][0])
        mh = [r for r in by[("nc-graph", 1536)] if r["qtype"] == "multihop"]
        self.assertEqual(len(mh), 1)
        # nc-graph is a fork: same HARS session id as nc-chunks (same session dir), own ask dir
        self.assertEqual(by[("nc-graph", 1536)][0]["session_id"],
                         by[("nc-chunks", 1536)][0]["session_id"])
        self.assertTrue(by[("nc-graph", 1536)][0]["session_id"].endswith("-b1536-t1"))
        self.assertTrue(os.path.isdir(os.path.join(out, "sessions", "chain_S0_c0__b512", "asks",
                                                   "nc-graph")))
        # recall call shapes: chunks never asked for the graph, graph always
        self.assertTrue(any(c["graph"] for c in world.recalls))
        self.assertEqual(sum(1 for c in world.recalls if c["graph"]),
                         sum(1 for c in world.recalls if not c["graph"]) - 1)  # -1 preflight
        # budget json has per-B/per-treatment ratios
        rb = json.load(open(os.path.join(out, "run_budget.json")))
        self.assertEqual(set(rb["by_budget"]), {"1536", "512"})
        self.assertIn("nc_graph", rb["by_budget"]["1536"])
        # feeds metrics' validity filter as complete quadruplets (4 arms x 2 B)
        kept, rep = mv.validate_rows([dict(r) for r in recs])
        self.assertEqual(len(kept), len(recs))
        self.assertFalse(rep["legacy_arm_set"])

    def test_graph_error_marks_only_that_budget_chain_invalid(self):
        world = FakeWorld(graph_status="error")
        rcode, recs, out, log = self._run(world, ["--mem-budgets", "1536"],
                                          [_chain("chain_S0_c0", "S0")])
        self.assertTrue(recs)
        self.assertTrue(all(r["valid"] is False for r in recs))
        self.assertTrue(any("nc-graph" in x for x in recs[0]["chain_invalid_reasons"]))
        kept, rep = mv.validate_rows([dict(r) for r in recs])
        self.assertEqual(kept, [])                                      # metrics drops the cell

    def test_missing_session_graph_block_invalidates(self):
        world = FakeWorld(graph_block=False)
        _, recs, _, _ = self._run(world, [], [_chain("chain_S0_c0", "S0")])
        self.assertTrue(all(r["valid"] is False for r in recs))

    def test_old_arm_set_still_runs_and_is_legacy_to_metrics(self):
        world = FakeWorld()
        rcode, recs, out, log = self._run(world, ["--arms", "baseline,nc,nc-note-only"],
                                          [_chain("chain_S0_c0", "S0")])
        self.assertEqual(rcode, 0, log[-1500:])
        self.assertEqual({r["arm"] for r in recs}, {"baseline", "nc-chunks", "nc-note-only"})
        self.assertFalse(any(c["graph"] for c in world.recalls))
        _, rep = mv.validate_rows([dict(r) for r in recs])
        self.assertTrue(rep["legacy_arm_set"])


# ---------------------------------------------------------------------------- metrics

def synth_rows(n_sessions=10, budgets=(1536, 1024, 512), p=None, multihop_every=3, seed=7,
               invalid=()):
    """Synthetic 4-arm sweep records. p[(arm, B)] = P(anchor hit); outcomes are drawn from
    one per-item uniform so arms are positively paired (same item hardness)."""
    p = p or {}
    rng = random.Random(seed)
    rows = []
    for s in range(n_sessions):
        cid = f"chain_S{s:02d}_c0"
        for k in range(6):
            qid = f"{cid}_q{k}"
            qtype = "multihop" if k % multihop_every == 0 else "single"
            u = rng.random()
            for B in budgets:
                for arm in ("baseline", "nc-chunks", "nc-graph", "nc-note-only"):
                    pr = p.get((arm, B), p.get((arm, None), 0.5))
                    hit = u < pr
                    rows.append({
                        "rid": f"{qid}::{arm}::b{B}", "qid": qid, "chain_id": cid,
                        "session_id": f"cr-{arm}", "span_idx": 1 + k % 3, "lag": 3 - k % 3,
                        "slice": "HEAD" if k % 2 else "TAIL", "answerable": True, "arm": arm,
                        "mem_budget": B, "qtype": qtype, "score": 1.5 if hit else 0.5,
                        "anchor_hit": hit, "gold_anchor": "A", "visible_context": "A in ctx",
                        "hallucination": False, "abstained": not hit, "latency_s": 5.0,
                        "recall_seconds": 3.0 if arm == "nc-graph" else 0.0,
                        "memory_tokens": B if arm != "nc-note-only" else B // 3,
                        "persisted_compaction": arm in ("nc-chunks", "nc-graph"),
                        "tokens_in": 1000, "tokens_out": 50,
                        "keyword_source": "llm" if arm == "nc-graph" else None,
                        "kg_empty": False if arm == "nc-graph" else None,
                        "session_graph_status": "ok" if arm == "nc-graph" else None,
                        "valid": (cid, B, arm) not in invalid, "chain_invalid_reasons": [],
                    })
        qid = f"{cid}_u"
        for B in budgets:
            for arm in ("baseline", "nc-chunks", "nc-graph", "nc-note-only"):
                rows.append({"rid": f"{qid}::{arm}::b{B}", "qid": qid, "chain_id": cid,
                             "session_id": "x", "span_idx": 0, "lag": 0, "slice": "NONE",
                             "answerable": False, "arm": arm, "mem_budget": B, "score": None,
                             "anchor_hit": None, "hallucination": False, "abstained": True,
                             "latency_s": 5.0, "recall_seconds": 0.0,
                             "memory_tokens": B, "valid": True, "chain_invalid_reasons": [],
                             "keyword_source": "llm" if arm == "nc-graph" else None,
                             "kg_empty": False if arm == "nc-graph" else None,
                             "session_graph_status": "ok" if arm == "nc-graph" else None})
    return rows


def run_metrics(rows, extra=()):
    td = tempfile.mkdtemp()
    rp, jp = os.path.join(td, "r.jsonl"), os.path.join(td, "m.json")
    with open(rp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    old = sys.argv
    sys.argv = ["metrics_v2.py", "--records", rp, "--json", jp, "--out", os.path.join(td, "m.md"),
                "--boot", "600", *extra]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc_ = mv.main()
    finally:
        sys.argv = old
    data = json.load(open(jp))
    md = open(os.path.join(td, "m.md")).read()
    shutil.rmtree(td, True)
    return rc_, data, md


class HolmAndBootstrap(unittest.TestCase):
    def test_holm_step_down_fixed_family(self):
        adj = mv.holm_adjust({"a": 0.001, "b": 0.01, "c": 0.04, "d": None, "e": 0.5})
        self.assertAlmostEqual(adj["a"], 0.005)
        self.assertAlmostEqual(adj["b"], 0.04)
        self.assertAlmostEqual(adj["c"], 0.12)
        self.assertAlmostEqual(adj["e"], 1.0)
        self.assertIsNone(adj["d"])
        # a missing cell does not shrink the family: with d absent, m stays 5
        self.assertAlmostEqual(mv.holm_adjust({"a": 0.001, "b": None, "c": None,
                                               "d": None, "e": None})["a"], 0.005)

    def test_one_sided_bootstrap_p(self):
        n = 400
        self.assertAlmostEqual(mv.boot_p_le({"a": 0.2, "b": 0.3, "c": 0.25}, 0.0, n, 1),
                               1 / (n + 1))
        self.assertGreater(mv.boot_p_le({"a": -0.2, "b": -0.3}, 0.0, n, 1), 0.99)
        self.assertIsNone(mv.boot_p_le({}, 0.0, n, 1))
        # NI shape: a true 0 delta is far above -0.10
        self.assertLess(mv.boot_p_le({"a": 0.0, "b": 0.0, "c": 0.01}, -0.10, n, 1), 0.01)
        # ... and a -0.2 delta is not non-inferior
        self.assertGreater(mv.boot_p_le({"a": -0.2, "b": -0.2, "c": -0.19}, -0.10, n, 1), 0.99)


class MetricsAmendmentC(unittest.TestCase):
    def test_legacy_alias_and_three_arm_mode(self):
        rows = [{"qid": "q1", "chain_id": "c_c0", "arm": a, "answerable": True, "valid": True,
                 "score": 1.0, "anchor_hit": True, "mem_budget": 1536}
                for a in ("baseline", "nc", "nc-note-only")]
        kept, rep = mv.validate_rows(rows)
        self.assertEqual(len(kept), 3)
        self.assertTrue(rep["legacy_arm_set"])
        self.assertEqual(mv.normalize(rows[1])["arm"], "nc-chunks")
        self.assertEqual(mv.normalize(rows[1])["rid"], "q1::nc")        # old rid format kept

    def test_four_arms_required_once_graph_exists(self):
        rows = synth_rows(2, budgets=(1536,))
        rows = [r for r in rows if not (r["arm"] == "nc-note-only" and r["qid"].endswith("q0"))]
        kept, rep = mv.validate_rows(rows)
        self.assertEqual(len(kept), len(rows) - 3 * 2)                  # qid q0 of 2 chains: 3 arms each
        self.assertEqual(len(rep["incomplete_qids"]), 2)

    def test_duplicate_key_includes_budget(self):
        rows = synth_rows(1, budgets=(1536, 1024))
        mv.validate_rows([dict(r) for r in rows])                       # same qid/arm, different B: fine
        dup = [dict(r) for r in rows] + [dict(rows[0], rid="other")]
        with self.assertRaises(SystemExit) as cm:
            mv.validate_rows(dup)
        self.assertEqual(cm.exception.code, 2)
        dup_rid = [dict(r) for r in rows] + [dict(rows[0], qid="zz")]
        with self.assertRaises(SystemExit):
            mv.validate_rows(dup_rid)

    def test_validity_is_per_budget_cell(self):
        rows = synth_rows(2, budgets=(1536, 512), invalid={("chain_S00_c0", 512, "nc-graph")})
        kept, rep = mv.validate_rows(rows)
        keys = {(r["chain_id"], r["mem_budget"]) for r in kept}
        self.assertIn(("chain_S00_c0", 1536), keys)
        self.assertNotIn(("chain_S00_c0", 512), keys)
        self.assertIn(("chain_S01_c0", 512), keys)

    def test_keyword_cost_is_counted_and_flagged(self):
        base = {"qid": "q", "chain_id": "c_c0", "arm": "nc-graph", "answerable": True,
                "tokens_in": 1000, "tokens_out": 50, "keyword_source": "llm"}
        n = mv.normalize(dict(base))
        self.assertEqual((n["tokens_in"], n["tokens_out"], n["keyword_cost_estimated"]),
                         (1450.0, 130.0, True))                         # flagged estimate
        n = mv.normalize(dict(base, keyword_tokens_in=400, keyword_tokens_out=60))
        self.assertEqual((n["tokens_in"], n["tokens_out"], n["keyword_cost_estimated"]),
                         (1400.0, 110.0, False))
        n = mv.normalize(dict(base, keyword_source="caller"))
        self.assertEqual(n["tokens_in"], 1000.0)                        # no LLM call
        n = mv.normalize(dict(base, arm="nc-chunks"))
        self.assertEqual(n["tokens_in"], 1000.0)
        n = mv.normalize(dict(base, recall_seconds=3.0, latency_s=5.0))
        self.assertEqual(n["latency_s"], 8.0)                           # keyword call inside latency

    def test_end_to_end_primary_and_secondaries(self):
        p = {("baseline", None): 0.5, ("nc-note-only", None): 0.25, ("nc-chunks", None): 0.5,
             ("nc-graph", 1536): 0.75, ("nc-graph", 1024): 0.5, ("nc-graph", 512): 0.2}
        rc_, d, md = run_metrics(synth_rows(10, p=p))
        self.assertEqual(rc_, 0)
        self.assertEqual(d["meta"]["treatment"], "nc-graph")
        self.assertEqual(d["meta"]["primary_b"], 1536)
        self.assertEqual(sorted(d["by_budget"]), ["1024", "1536", "512"])
        self.assertIn("nc-graph-baseline", d["deltas"])                 # primary cell pairs
        self.assertIn("nc-graph-nc-chunks", d["deltas"])
        sec = d["secondary"]["tests"]
        self.assertEqual(list(sec), list(mv.SECONDARY_IDS))
        # S1: graph (0.75) vs chunks (0.5), paired on identical items -> clearly positive
        self.assertGreater(sec["S1_graph_vs_chunks"]["point"], 0.15)
        self.assertEqual(sec["S1_graph_vs_chunks"]["label"], "graph helps")
        # S2: nc-graph@1024 (0.5) vs baseline@1536 (0.5) -> non-inferior, supported
        self.assertEqual(sec["S2_ni_1024"]["decision"], "SUPPORTED")
        self.assertEqual(set(sec["S2_ni_1024"]["legs"]), {"clean_anchor_hit", "score"})
        # S3: nc-graph@512 (0.2) loses 0.3 -> NOT non-inferior
        self.assertEqual(sec["S3_ni_512"]["decision"], "not supported")
        self.assertGreater(sec["S3_ni_512"]["p"], 0.5)
        # multi-hop stratum tests exist and use only multihop items (k % 3 == 0 -> 2 of 6)
        self.assertGreater(sec["S4_multihop_vs_baseline"]["n_pairs"], 0)
        self.assertLess(sec["S4_multihop_vs_baseline"]["n_pairs"],
                        sec["S1_graph_vs_chunks"]["n_pairs"])
        self.assertEqual(sec["S4_multihop_vs_baseline"]["n_pairs"], 10 * 2)
        for t in sec.values():
            self.assertIn("p_holm", t)
        self.assertIn("Secondary hypotheses", md)
        self.assertIn("Budget sweep", md)
        self.assertIn("nc-graph diagnostics", md)
        # per-B grouping: every budget has all four arms
        for B in ("1536", "1024", "512"):
            self.assertEqual(set(d["by_budget"][B]["summary"]),
                             {"baseline", "nc-chunks", "nc-graph", "nc-note-only"})
        # primary verdict uses the nc-graph treatment at B=1536 (10 sessions >= 8)
        self.assertEqual(d["gates"]["n_sessions"], 10)
        names = [g["rule"] for g in d["gates"]["win_rules"]]
        self.assertTrue(any("nc-graph-baseline" in r for r in names))
        self.assertEqual(d["graph_diag"]["1536"]["keyword_source"], {"llm": d["graph_diag"]["1536"]["n"]})

    def test_underpowered_labels_secondaries_inconclusive(self):
        rc_, d, md = run_metrics(synth_rows(4, p={("nc-graph", 1536): 0.9, ("baseline", None): 0.4}))
        self.assertEqual(d["gates"]["verdict"], "INCONCLUSIVE")
        self.assertTrue(all(t["decision"].startswith(("INCONCLUSIVE", "n/a"))
                            for t in d["secondary"]["tests"].values()))

    def test_missing_cell_counts_as_p_one_not_a_shrunk_family(self):
        rc_, d, _ = run_metrics(synth_rows(10, budgets=(1536,)))
        sec = d["secondary"]["tests"]
        self.assertIsNone(sec["S3_ni_512"]["p"])
        self.assertTrue(sec["S3_ni_512"]["decision"].startswith("n/a"))
        self.assertEqual(len(d["secondary"]["family"]), 5)

    def test_legacy_run_gets_exploratory_banner_and_no_secondaries(self):
        rows = [r for r in synth_rows(9, budgets=(1536,)) if r["arm"] != "nc-graph"]
        for r in rows:
            r["arm"] = "nc" if r["arm"] == "nc-chunks" else r["arm"]
            r["rid"] = r["rid"].replace("::nc-chunks", "::nc")
        rc_, d, md = run_metrics(rows)
        self.assertTrue(d["meta"]["legacy"])
        self.assertIsNone(d["secondary"])
        self.assertIn("LEGACY / EXPLORATORY", md)
        self.assertEqual(d["meta"]["treatment"], "nc-chunks")


# ------------------------------------------------------------ judge / export / context

class BlindnessAndContext(unittest.TestCase):
    def test_masking_covers_new_arm_names_and_graph_headers(self):
        for leak in ("cr-claudehaiku55-chain_x_c0-nc-graph-b512-t1",
                     "see x-nc-chunks-b1536-run", "the nc-graph arm", "nc-chunks: answer",
                     "Knowledge Graph Data (Entity): foo", "Knowledge Graph Data (Relationship):",
                     "### Direct relations / entities", "Document Chunks (each has a reference_id):",
                     "session_graph on"):
            masked, hits = judge_v2.mask_arm_leaks(judge_v2.sanitize_answer(leak))
            if not leak.startswith("nc-chunks:"):      # a leading arm label is stripped, not masked
                self.assertTrue(hits, leak)
            for forbidden in ("nc-graph", "nc-chunks", "Knowledge Graph", "Direct relations",
                              "Document Chunks", "session_graph"):
                self.assertNotIn(forbidden, masked, (leak, masked))
        # a normal answer is untouched
        self.assertEqual(judge_v2.mask_arm_leaks("It raised FooError_10 in tests/test_mod1.py.")[1], [])

    def test_visible_context_budget_dirs_and_graph_recall(self):
        with tempfile.TemporaryDirectory() as td:
            for arm, extra in (("nc-graph", "memory"), ("nc-chunks", "memory")):
                d = os.path.join(td, "chain_x_c0__b1024", "asks", arm, "q1")
                os.makedirs(d)
                with open(os.path.join(d, "s.jsonl"), "w") as f:
                    f.write(json.dumps({"type": "compaction", "summary": "NOTE alpha"}) + "\n")
                    f.write(json.dumps({"type": "message", "message": {"role": "user", "content": [
                        {"type": "text", "text": "Q?\n<memory>\nRECALL_BETA\n</memory>"}]}}) + "\n")
            ctx, meta = vc.build_visible_context(td, "chain_x_c0", "nc-graph", "q1", budget=1024)
            self.assertIn("RECALL_BETA", ctx)
            self.assertTrue(meta["recall_in_context"])
            self.assertIsNone(vc.build_visible_context(td, "chain_x_c0", "nc-graph", "q1")[0])
            # legacy arm name resolves to the nc-chunks dir
            ctx2, _ = vc.build_visible_context(td, "chain_x_c0", "nc", "q1", budget=1024)
            self.assertIn("RECALL_BETA", ctx2)
            self.assertEqual(vc.chain_dirkey("c", 1024), rc.chain_dirkey("c", _ns(mem_budget=1024,
                                                                                  b_tag=True)))


# ------------------------------------------------------------------- M3 (TypeScript)

class M3Splitter(unittest.TestCase):
    def test_ts_selftest_with_stub_packages(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        with tempfile.TemporaryDirectory() as td:
            for pkg, body in (
                    ("@earendil-works/pi-ai", "export const uuidv7 = () => 'x';\n"),
                    ("@earendil-works/pi-coding-agent",
                     "export const convertToLlm = (x) => x;\nexport const serializeConversation = (x) => '';\n")):
                d = os.path.join(td, "node_modules", *pkg.split("/"))
                os.makedirs(d)
                open(os.path.join(d, "package.json"), "w").write(
                    json.dumps({"name": pkg, "type": "module", "main": "index.js"}))
                open(os.path.join(d, "index.js"), "w").write(body)
            shutil.copy(EXT, os.path.join(td, "nc.ts"))
            env = dict(os.environ, NC_SELFTEST="1")
            p = subprocess.run([node, "--experimental-strip-types", "nc.ts"], cwd=td, env=env,
                               capture_output=True, text=True, timeout=120)
            if "strip-types" in (p.stderr or "") and p.returncode not in (0, 1):
                self.skipTest("node without --experimental-strip-types")
            out = p.stdout + p.stderr
            self.assertEqual(p.returncode, 0, out[-3000:])
            self.assertIn("[selftest] RESULT: PASS", out)
            for name in ("split-60k-anchor-in-the-middle", "split-one-oversized-message",
                         "split-mixed-oversized", "split-empty"):
                self.assertIn(f"PASS {name}", out)

    def test_m3_report_selftest(self):
        p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "m3_anchor_report.py"),
                            "--selftest"], capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)


if __name__ == "__main__":
    unittest.main()
