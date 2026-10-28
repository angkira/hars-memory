"""Amendment C.7: the nc-enriched arm (memory recalled at COMPACTION time, none at ask time), --reuse-run, the recall
probe, the hallucination breakdown, the metrics regime. Offline: mock LLM + FakeHars. No network."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import ls_fixtures as F
import longsession as L
import metrics_v2 as M
import run_longsession as R

HAVE_RC = L.rc() is not None and all(hasattr(L.rc(), n) for n in (
    "pack_injection", "recall_failure_reasons", "recall_degraded", "status_gate_problems", "_parse_docs",
    "hars_recall_correct", "recall_args_correct"))
needs_rc = unittest.skipUnless(HAVE_RC, "run_chainreplay.py not importable (mid-edit?)")

BASE = ["--window", "14000", "--keep-tail", "3000", "--mem-budgets", "1536", "--provider", "mock", "--agent-model", "mock",
        "--fake-hars", "--no-preflight", "--piece-chars", "3000", "--pieces-per-call", "4"]
NEW_ARMS = ["--arms", "baseline,nc-enriched,nc-note-only"]


def call(argv):
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        rc = R.main(argv)
    return rc, buf.getvalue(), err.getvalue()


def rows_of(out, name="records.jsonl"):
    return [json.loads(l) for l in open(os.path.join(out, name)) if l.strip()]


class SharedHars:
    """One FakeHars for several run_longsession invocations in a test (a real server outlives a run)."""

    def __init__(self):
        self.h = R.FakeHars()
        self.p = mock.patch.object(R, "FakeHars", lambda: self.h)

    def __enter__(self):
        self.p.start()
        return self.h

    def __exit__(self, *a):
        self.p.stop()


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.sd = os.path.join(self.d.name, "in")
        F.write_session(self.sd, F.make_session("syn-1"))
        self.out = os.path.join(self.d.name, "out")

    def tearDown(self):
        self.d.cleanup()

    def out_n(self, n):
        return os.path.join(self.d.name, f"out{n}")

    def run_ls(self, out, extra=(), base=None):
        return call(["--sessions", self.sd, "--out", out] + (base or BASE) + list(extra))


class PureHelpers(unittest.TestCase):
    def test_query_prompt_has_no_question_input(self):
        import inspect
        params = list(inspect.signature(R.enrich_query_prompt).parameters)
        self.assertEqual(params, ["note", "recent", "k"])             # structurally: nothing else can be passed in
        p = R.enrich_query_prompt("GOAL\nship", "tail text", 5)
        self.assertIn("Write 5 short retrieval queries", p)
        self.assertIn("<note>\nGOAL\nship\n</note>", p)
        self.assertIn("<recent_context>\ntail text\n</recent_context>", p)

    def test_parse_queries_json_lines_bullets_dedupe_and_cap(self):
        txt = ('1. {"query": "where is job JOB0010 output", "ll_keywords": ["JOB0010", "/data/run010/out.bin"], "hl_keywords": ["jobs"]}\n'
               '{"query": "where is job JOB0010 output", "ll_keywords": [], "hl_keywords": []}\n'
               '- plain text query without keywords\n'
               '{"query": "third", "ll_keywords": "single", "hl_keywords": ["a"]}\n'
               'x\n')
        qs = R.parse_queries(txt, 3)
        self.assertEqual([q["query"] for q in qs], ["where is job JOB0010 output", "plain text query without keywords", "third"])
        self.assertEqual(qs[0]["ll_keywords"], ["JOB0010", "/data/run010/out.bin"])
        self.assertEqual(qs[1]["keywords_source"], "none")
        self.assertEqual(qs[2]["ll_keywords"], ["single"])
        self.assertEqual(len(R.parse_queries(txt, 2)), 2)
        arr = R.parse_queries('[{"query": "q one", "ll_keywords": ["k"]}, "q two"]', 8)
        self.assertEqual([q["query"] for q in arr], ["q one", "q two"])

    def test_parse_keywords(self):
        kw = R.parse_keywords('noise {"ll_keywords": ["a.py", "E42"], "hl_keywords": ["auth"]} tail')
        self.assertEqual((kw["ll_keywords"], kw["hl_keywords"], kw["source"]), (["a.py", "E42"], ["auth"], "llm"))
        self.assertEqual(R.parse_keywords("garbage")["source"], "none")

    @needs_rc
    def test_merge_dedupes_across_queries_and_interleaves_by_rank(self):
        def pay(*contents):
            lines = [json.dumps({"reference_id": str(i + 1), "content": c, "section": ""}) for i, c in enumerate(contents)]
            return {"ok": True, "context": "Document Chunks (each has a reference_id):\n```json\n" + "\n".join(lines) + "\n```",
                    "session": {"status": "ok", "vectors_indexed": 3}}
        p1, p2 = pay("alpha one", "beta two", "gamma three"), pay("beta  TWO", "delta four", "alpha one")
        merged, st = R.merge_recall_payloads([p1, p2, {"context": "", "error": "x"}])
        docs = merged.pop("_docs")
        self.assertEqual([d["content"] for d in docs], ["alpha one", "beta  TWO", "delta four", "gamma three"])
        self.assertEqual(st["chunks_total"], 6)
        self.assertEqual(st["chunks_unique"], 4)
        self.assertEqual(st["duplicates_removed"], 2)
        self.assertEqual(st["kept_from_query"], [0, 1, 1, 0])      # rank 1 of every query comes first
        self.assertTrue(merged["ok"])
        rcm = L.rc()
        again = rcm._parse_docs(merged["context"], rcm._HEADER_CHUNKS)
        self.assertEqual([d["content"] for d in again], [d["content"] for d in docs])      # the rebuilt context parses back
        empty, st0 = R.merge_recall_payloads([{"context": "", "error": "boom"}])
        self.assertFalse(empty["ok"])
        self.assertIn("error", empty)

    @needs_rc
    def test_fill_union_respects_the_cap_in_native_tokens_and_skips_what_does_not_fit(self):
        docs = [{"content": "w " * 80, "section": ""}, {"content": "x " * 400, "section": ""}, {"content": "y " * 20, "section": ""}]
        for cap in (120, 300, 2000):
            sel, inj, tot = R.fill_union(docs, "ls-x-syn", cap)
            self.assertLessEqual(tot, cap)
            self.assertEqual(tot, R.ntok(inj))
            self.assertIn("Facts recalled from long-term memory", inj)
        sel, inj, tot = R.fill_union(docs, "ls-x-syn", 300)
        self.assertEqual([d["content"].split()[0] for d in sel], ["w", "y"])    # the big middle chunk is skipped, the next fits

    def test_budget_recall_shrinks_max_chars_until_the_block_fits(self):
        calls = []

        def fn(mc):
            calls.append(mc)
            return {"context": "z" * mc}
        pay, inj, tot, att = R.budget_recall(fn, "sid", 300, 3.6)
        self.assertLessEqual(tot, 300)
        self.assertEqual(att[0]["max_chars"], int(300 * 3.6))
        self.assertTrue(tot >= 0.85 * 300)
        with self.assertRaises(RuntimeError):
            R.budget_recall(lambda mc: {"context": "z" * 5000}, "sid", 50, 3.6)    # ignores max_chars: cannot be budgeted

    def test_recall_args_correct_shape(self):
        rcm = L.rc()
        a = rcm.recall_args_correct("s", "q", ["k1"], ["t"], 3000)
        self.assertEqual(a, {"session_id": "s", "question": "q", "mode": "hybrid", "ll_keywords": ["k1"],
                             "hl_keywords": ["t"], "max_chars": 3000})
        self.assertNotIn("top_k", a)
        self.assertNotIn("session_graph", a)
        self.assertNotIn("max_chars", rcm.recall_args_correct("s", "q", [], []))
        old = rcm.recall_args("s", "q", False)
        self.assertEqual((old["top_k"], "ll_keywords" in old), (8, False))              # the frozen question-time shape


@needs_rc
class EnrichedRun(Base):
    def test_arms_default_to_baseline_enriched_note_only(self):
        a = R.parse_args(["--sessions", self.sd] + BASE)
        self.assertEqual(a.arms_list, ["baseline", "nc-enriched", "nc-note-only"])
        b = R.parse_args(["--sessions", self.sd, "--arms", "nc-chunks,nc-graph,baseline,nc-note-only"] + BASE)
        self.assertEqual(b.arms_list, ["baseline", "nc-chunks", "nc-graph", "nc-note-only"])

    def test_no_recall_at_ask_time_and_the_c7_call_shape(self):
        with SharedHars() as h:
            h.recall = mock.Mock(side_effect=AssertionError("ask-time recall"))       # the question-time call must never happen
            rc, so, se = self.run_ls(self.out, NEW_ARMS + ["--checkpoints", "end"])
            self.assertEqual(rc, 0, so + se)
            rows = rows_of(self.out)
            self.assertEqual({r["arm"] for r in rows}, {"baseline", "nc-enriched", "nc-note-only"})
            self.assertTrue(all(r["valid"] is True for r in rows), [r["chain_invalid_reasons"] for r in rows][:2])
            self.assertEqual(len(h.correct_calls), 8)                          # K = 8 queries, ONE boundary (end), ONE B
            for c in h.correct_calls:
                self.assertTrue(c["ll_keywords"] and c["hl_keywords"])
                self.assertIsInstance(c["max_chars"], int)
            e = [r for r in rows if r["arm"] == "nc-enriched"]
            self.assertTrue(all(r["recall_seconds"] == 0.0 for r in e))
            self.assertTrue(all(r["memory_tokens_injected"] > 0 for r in e))

    def test_enrich_artifacts_rows_and_anchor_fields(self):
        with SharedHars():
            rc, so, se = self.run_ls(self.out, NEW_ARMS)
        self.assertEqual(rc, 0, so + se)
        j = len(L.load_sessions([self.sd])[0]["messages"])
        arts = [os.path.join(self.out, "enrich", "syn-1", f) for f in os.listdir(os.path.join(self.out, "enrich", "syn-1"))]
        self.assertEqual(len(arts), 1)                                          # once per (sid, boundary, B), not per question
        en = json.load(open(arts[0]))
        for k in ("queries", "query_gen", "merge", "packed_text", "injected", "block_tokens", "cap_info", "reasons", "raw_union",
                  "recall_shape", "packer", "chars_per_token", "seconds"):
            self.assertIn(k, en)
        self.assertEqual(len(en["queries"]), 8)
        for q in en["queries"]:
            for k in ("query", "ll_keywords", "hl_keywords", "seconds", "error", "max_chars", "graph_branch_enabled",
                      "lightrag_mode", "payload_file", "raw_chars"):
                self.assertIn(k, q)
            self.assertTrue(os.path.isfile(os.path.join(self.out, q["payload_file"])))
        self.assertLessEqual(en["block_tokens"], en["cap_info"]["recall_cap"])
        self.assertGreater(en["merge"]["duplicates_removed"], 0)                # the fake store returns overlapping chunks
        self.assertIsNone(en["top_k_sent"])
        side = rows_of(self.out, "recall_raw.jsonl")
        self.assertEqual(len([x for x in side if x.get("kind") == "enrich_query"]), 8)
        self.assertTrue(all(x["raw"] for x in side))
        e = [r for r in rows_of(self.out) if r["arm"] == "nc-enriched"]
        for r in e:
            self.assertIn("anchor_in_enriched", r)
            self.assertIn("anchor_in_enriched_raw", r)
            self.assertEqual(r["anchor_in_recall"], r["anchor_in_enriched"])
            self.assertEqual(r["enrich_artifact"], f"enrich/syn-1/j{r['checkpoint']}.b1536.json")
            self.assertEqual(r["enrich_queries"], 8)
        anchored = [r for r in e if r["gold_anchor"]]
        self.assertTrue(any(r["anchor_in_enriched_raw"] for r in anchored))
        self.assertTrue(all(r["anchor_in_enriched"] <= r["anchor_in_enriched_raw"] for r in anchored))   # packed implies raw

    def test_query_generation_never_sees_the_question(self):
        seen = []
        orig = L.mock_complete

        def spy(user, system, max_tokens, model="mock"):
            seen.append(user)
            return orig(user, system, max_tokens, model)
        s = F.make_session("syn-1")
        marker = "ZEBRAQUESTIONMARKER"
        for q in s["questions"]:
            q["question"] += f" {marker}{q['qid']}"
        sd = os.path.join(self.d.name, "in2")
        F.write_session(sd, s)
        with SharedHars(), mock.patch.object(L, "mock_complete", spy):
            rc, so, se = call(["--sessions", sd, "--out", self.out] + BASE + NEW_ARMS + ["--checkpoints", "2,end"])
        self.assertEqual(rc, 0, so + se)
        qgen = [u for u in seen if "retrieval queries" in u]
        self.assertEqual(len(qgen), 2)                                          # one per checkpoint boundary (2 and end)
        self.assertTrue(any(marker in u for u in seen if "Question:" in u))      # the questions DID reach the answer prompts
        for u in qgen:
            self.assertNotIn(marker, u)
            self.assertNotIn("Question:", u)
        en = [json.load(open(os.path.join(self.out, "enrich", "syn-1", f))) for f in os.listdir(os.path.join(self.out, "enrich", "syn-1"))]
        for e in en:
            self.assertNotIn(marker, json.dumps([q["query"] for q in e["queries"]] + [q["ll_keywords"] for q in e["queries"]]))

    def test_parity_of_realized_memory_vs_baseline(self):
        with SharedHars():
            rc, so, se = self.run_ls(self.out, NEW_ARMS)
        self.assertEqual(rc, 0)
        rows = rows_of(self.out)
        par = R.parity_enriched(rows, [1536])["1536"]
        self.assertEqual(par["window"], [0.8, 1.05])
        e = [r for r in rows if r["arm"] == "nc-enriched"]
        for r in e:                                                              # note + block never exceeds B
            self.assertLessEqual(r["memory_tokens"], 1536)
            self.assertEqual(r["memory_tokens"], r["arm_note_tokens"] + r["memory_tokens_injected"])
        summ = json.load(open(os.path.join(self.out, "summary.json")))
        self.assertIn("1536", summ["parity_enriched"])
        # pure: in / out of the window
        mk = lambda arm, mem: {"chain_id": "s", "arm": arm, "mem_budget": 1536, "memory_tokens": mem}  # noqa: E731
        ok = R.parity_enriched([mk("baseline", 1500), mk("nc-enriched", 1400), mk("nc-note-only", 600)], [1536])["1536"]
        bad = R.parity_enriched([mk("baseline", 1500), mk("nc-enriched", 700)], [1536])["1536"]
        self.assertTrue(ok["in_window"])
        self.assertFalse(bad["in_window"])

    def test_every_recall_failing_invalidates_only_the_enriched_arm(self):
        with SharedHars() as h:
            h.fail_recall = True
            rc, so, se = self.run_ls(self.out, NEW_ARMS)
        rows = rows_of(self.out)
        by = {a: {r["valid"] for r in rows if r["arm"] == a} for a in ("baseline", "nc-enriched", "nc-note-only")}
        self.assertEqual(by["baseline"], {True})
        self.assertEqual(by["nc-note-only"], {True})
        self.assertEqual(by["nc-enriched"], {False})
        why = " ".join(next(r for r in rows if r["arm"] == "nc-enriched")["chain_invalid_reasons"])
        self.assertIn("enrichment recalls failed", why)

    def test_missing_keywords_invalidate_the_block(self):
        orig = L.mock_complete

        def no_kw(user, system, max_tokens, model="mock"):
            r = orig(user, system, max_tokens, model)
            if "retrieval queries" in user:
                r["text"] = "\n".join(json.loads(l)["query"] for l in r["text"].splitlines())      # plain lines, no keywords
            return r
        with SharedHars(), mock.patch.object(L, "mock_complete", no_kw):
            rc, so, se = self.run_ls(self.out, NEW_ARMS)
        e = [r for r in rows_of(self.out) if r["arm"] == "nc-enriched"]
        self.assertTrue(e and all(r["valid"] is False for r in e))
        self.assertIn("keywords missing", " ".join(e[0]["chain_invalid_reasons"]))

    def test_legacy_packer_switch_uses_packing_v2_without_max_chars(self):
        with SharedHars() as h:
            rc, so, se = self.run_ls(self.out, NEW_ARMS + ["--packer", "legacy"])
            self.assertEqual(rc, 0, so + se)
            self.assertTrue(all(c["max_chars"] is None for c in h.correct_calls))
        en = json.load(open(os.path.join(self.out, "enrich", "syn-1", os.listdir(os.path.join(self.out, "enrich", "syn-1"))[0])))
        self.assertEqual(en["packer"], "legacy")
        self.assertLessEqual(en["block_tokens"], en["cap_info"]["recall_cap"])

    def test_question_time_arms_still_work_and_now_persist_raw_recall(self):
        with SharedHars():
            rc, so, se = self.run_ls(self.out, ["--arms", "baseline,nc-chunks,nc-graph,nc-note-only"])
        self.assertEqual(rc, 0, so + se)
        rows = rows_of(self.out)
        nc = [r for r in rows if r["arm"] in ("nc-chunks", "nc-graph")]
        self.assertTrue(nc and all("anchor_in_recall_raw" in r for r in nc))
        self.assertTrue(all(r["anchor_in_recall"] <= r["anchor_in_recall_raw"] for r in nc if r["gold_anchor"]))
        side = rows_of(self.out, "recall_raw.jsonl")
        self.assertEqual({x["rid"] for x in side}, {r["rid"] for r in nc})
        self.assertTrue(all(x["raw_chars"] == r["recall_raw_chars"] for x, r in zip(sorted(side, key=lambda x: x["rid"]),
                                                                                    sorted(nc, key=lambda r: r["rid"]))))
        self.assertTrue(all("raw" not in r for r in rows))                         # never inline in records.jsonl


@needs_rc
class ReuseRun(Base):
    def first_run(self, h, arms="baseline,nc-chunks,nc-note-only"):
        rc, so, se = self.run_ls(self.out_n(1), ["--arms", arms])
        self.assertEqual(rc, 0, so + se)
        return json.load(open(os.path.join(self.out_n(1), "config.json")))["run_id"]

    def test_reuse_calls_no_memory_compact_copies_arms_and_uses_the_exact_old_note(self):
        with SharedHars() as h:
            old_id = self.first_run(h)
            n0 = h.compact_calls
            self.assertGreater(n0, 0)
            rc, so, se = self.run_ls(self.out_n(2), NEW_ARMS + ["--reuse-run", self.out_n(1)])
            self.assertEqual(rc, 0, so + se)
            self.assertEqual(h.compact_calls, n0)                                # NO memory_compact at all
            self.assertTrue(all(c["sid"].startswith(f"ls-{old_id}-") for c in h.correct_calls))
        rows = rows_of(self.out_n(2))
        self.assertEqual({r["arm"] for r in rows}, {"baseline", "nc-enriched", "nc-note-only"})
        old = {(r["arm"], r["qid"]): r for r in rows_of(self.out_n(1))}
        for r in rows:
            if r["arm"] in ("baseline", "nc-note-only"):
                self.assertEqual(r["reused_from"], os.path.abspath(self.out_n(1)))
                o = old[(r["arm"], r["qid"])]
                self.assertEqual({k: v for k, v in r.items() if k not in ("reused_from", "reused_run_id")}, o)   # original fields kept
            else:
                self.assertNotIn("reused_from", r)
        note_only = {r["qid"]: r["memory_text"] for r in rows if r["arm"] == "nc-note-only"}
        for r in rows:
            if r["arm"] == "nc-enriched":
                self.assertEqual(r["memory_text"], note_only[r["qid"]])           # enriched vs note-only isolates the facts
                self.assertEqual(r["session_id"], f"ls-{old_id}-syn-1")
        done = rows_of(self.out_n(2), "done.jsonl")
        self.assertEqual({d["arm"] for d in done if d.get("reused_from")}, {"baseline", "nc-note-only"})
        comps = rows_of(self.out_n(2), "compactions.jsonl")
        self.assertFalse([c for c in comps if c.get("arm") == "hars" and c.get("ok")])        # no ingest rows
        self.assertTrue([c for c in comps if c.get("arm") == "note" and c.get("reused")])
        summ = json.load(open(os.path.join(self.out_n(2), "summary.json")))
        self.assertEqual(summ["reuse"]["run_id"], old_id)
        self.assertEqual(summ["llm_cost_usd"]["summariser"], 0.0)

    def test_notes_come_from_the_records_when_memory_dir_is_missing(self):
        import shutil
        with SharedHars() as h:
            self.first_run(h)
            with_cache = self.run_ls(self.out_n(2), NEW_ARMS + ["--reuse-run", self.out_n(1)])
            shutil.rmtree(os.path.join(self.out_n(1), "memory"))
            rc, so, se = self.run_ls(self.out_n(3), NEW_ARMS + ["--reuse-run", self.out_n(1)])
            self.assertEqual(rc, 0, so + se)
        a = {r["qid"]: r["memory_text"] for r in rows_of(self.out_n(2)) if r["arm"] == "nc-enriched"}
        b = {r["qid"]: r["memory_text"] for r in rows_of(self.out_n(3)) if r["arm"] == "nc-enriched"}
        self.assertEqual(a, b)
        summ = json.load(open(os.path.join(self.out_n(3), "summary.json")))
        self.assertEqual(set(summ["reuse"]["note_sources"].values()), {"records.nc-note-only.memory_text"})
        self.assertEqual(json.load(open(os.path.join(self.out_n(3), "summary.json")))["llm_cost_usd"]["summariser"], 0.0)

    def test_hars_gate_fails_loudly_when_the_sessions_are_not_on_the_server(self):
        with SharedHars() as h:
            self.first_run(h)
        rc, so, se = self.run_ls(self.out_n(2), NEW_ARMS + ["--reuse-run", self.out_n(1)])        # a fresh, empty FakeHars
        self.assertEqual(rc, 3, so + se)
        self.assertIn("HARS gate FAIL", se)
        self.assertIn("nothing spent", se)
        self.assertFalse(os.path.exists(os.path.join(self.out_n(2), "records.jsonl")))

    def test_gate_checks_vectors_indexed_and_compactions(self):
        with SharedHars() as h:
            self.first_run(h)
            sid = next(iter(h.s))
            h.s[sid]["docs"] = []                                                  # vectors_indexed == 0 but compactions > 0
            rc, so, se = self.run_ls(self.out_n(2), NEW_ARMS + ["--reuse-run", self.out_n(1)])
        self.assertEqual(rc, 3)
        self.assertIn("vectors_indexed", se)

    def test_reuse_requires_end_only_checkpoints_and_a_matching_question_set(self):
        with SharedHars() as h:
            self.first_run(h)
            rc, so, se = self.run_ls(self.out_n(2), NEW_ARMS + ["--reuse-run", self.out_n(1), "--checkpoints", "2,end"])
            self.assertEqual(rc, 3)
            self.assertIn("end-only", se)
            rc, so, se = self.run_ls(self.out_n(3), NEW_ARMS + ["--reuse-run", self.out_n(1), "--max-questions", "2"])
            self.assertEqual(rc, 3)
            self.assertIn("must match the old run", se)

    def test_reuse_refuses_other_inputs_and_an_invalid_old_cell(self):
        with SharedHars() as h:
            self.first_run(h)
            s = F.make_session("syn-1")
            s["messages"][3]["text"] += " changed"
            sd2 = os.path.join(self.d.name, "in-changed")
            F.write_session(sd2, s)
            rc, so, se = call(["--sessions", sd2, "--out", self.out_n(2)] + BASE + NEW_ARMS + ["--reuse-run", self.out_n(1)])
            self.assertEqual(rc, 3)
            self.assertIn("another sha256", se)
            with open(os.path.join(self.out_n(1), "done.jsonl")) as f:
                done = [json.loads(l) for l in f if l.strip()]
            for d in done:
                if d["arm"] == "baseline":
                    d["valid"] = False
            open(os.path.join(self.out_n(1), "done.jsonl"), "w").write("\n".join(json.dumps(d) for d in done) + "\n")
            rc, so, se = self.run_ls(self.out_n(3), NEW_ARMS + ["--reuse-run", self.out_n(1)])
            self.assertEqual(rc, 3)
            self.assertIn("INVALID", se)

    def test_reuse_arms_may_not_include_the_paid_arm(self):
        with self.assertRaises(SystemExit):
            R.parse_args(["--sessions", self.sd, "--reuse-run", "x", "--reuse-arms", "nc-enriched"] + BASE + NEW_ARMS)

    def test_dry_run_with_reuse_has_no_ingest_and_no_reused_arm_cost(self):
        with SharedHars() as h:
            self.first_run(h)
        cost = lambda extra: json.loads(self._dry(extra))  # noqa: E731
        fresh = cost([])
        reuse = cost(["--reuse-run", self.out_n(1)])
        fs, rs = fresh["sessions"][0], reuse["sessions"][0]
        self.assertGreater(fs["hars"]["memory_compact_calls"], 0)
        self.assertEqual((rs["hars"]["memory_compact_calls"], rs["hars"]["extract_usd"], rs["hars"]["est_ingest_hours"]), (0, 0.0, 0.0))
        self.assertEqual(rs["per_budget"]["1536"]["usd"]["baseline"], 0.0)
        self.assertEqual(rs["per_budget"]["1536"]["usd"]["nc-note-only"], 0.0)
        self.assertLessEqual(reuse["grand_total_usd"], fresh["grand_total_usd"])
        self.assertEqual(set(rs["per_budget"]["1536"]["asks"]), {"baseline", "nc-enriched", "nc-note-only"})

    def _dry(self, extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = R.main(["--sessions", self.sd, "--dry-run", "--json", "--budget-usd", "5"] + BASE + NEW_ARMS + extra)
        self.assertEqual(rc, 0)
        return buf.getvalue()


@needs_rc
class RecallProbe(Base):
    def test_probe_reports_old_shape_correct_shape_and_bm25_side_by_side_without_an_answer_model(self):
        with SharedHars() as h:
            rc, so, se = self.run_ls(self.out_n(1), ["--arms", "baseline,nc-chunks,nc-note-only"])
            self.assertEqual(rc, 0)
            qa_calls = []
            orig = R.LLM.complete

            def spy(self_, user, **kw):
                qa_calls.append(user)
                return orig(self_, user, **kw)
            with mock.patch.object(R.LLM, "complete", spy):
                rc, so, se = self.run_ls(self.out_n(2), NEW_ARMS + ["--reuse-run", self.out_n(1), "--recall-probe",
                                                                      "--probe-caps", "400,900"])
            self.assertEqual(rc, 0, so + se)
        self.assertTrue(qa_calls and all("Produce keywords for a memory search" in u for u in qa_calls))   # only the keyword call
        self.assertFalse(os.path.exists(os.path.join(self.out_n(2), "records.jsonl")))                    # no answers, no rows
        summ = json.load(open(os.path.join(self.out_n(2), "probe-summary.json")))
        for col in ("A_raw", "A_packed", "B_raw", "C_old_cap", "D_old_cap", "C_400", "C_900", "D_400", "D_900", "anchor_in_note"):
            self.assertIn(col, summ["all"])
            self.assertIn(col, summ["recoverable"])
        self.assertGreater(summ["n_questions"], 0)
        self.assertEqual(summ["caps"], [400, 900])
        rows = rows_of(self.out_n(2), "probe.jsonl")
        self.assertEqual(len(rows), summ["n_questions"])
        for r in rows:
            self.assertTrue(r["keywords"])
            self.assertIn("B_graph_branch_enabled", r)
            self.assertIn("attempts", r["C"]["400"])
            self.assertLessEqual(r["C"]["400"]["tokens"], 400)
            self.assertLessEqual(r["C"]["900"]["tokens"], 900)
        md = open(os.path.join(self.out_n(2), "REPORT-probe.md")).read()
        for needle in ("(A) old shape, raw", "(B) correct shape, raw", "(C) correct shape, max_chars @ 400 tok", "(D) BM25 @ 900 tok"):
            self.assertIn(needle, md)
        side = rows_of(self.out_n(2), "recall_raw.jsonl")
        self.assertEqual({x["kind"] for x in side}, {"probe_A", "probe_B"})

    def test_probe_needs_reuse_run_and_its_dry_run_is_free_of_ingest(self):
        with self.assertRaises(SystemExit):
            R.parse_args(["--sessions", self.sd, "--recall-probe"] + BASE)
        with SharedHars():
            self.assertEqual(self.run_ls(self.out_n(1), ["--arms", "baseline,nc-chunks,nc-note-only"])[0], 0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = R.main(["--sessions", self.sd, "--dry-run", "--json", "--budget-usd", "5", "--recall-probe", "--reuse-run",
                         self.out_n(1)] + BASE + NEW_ARMS)
        self.assertEqual(rc, 0)
        res = json.loads(buf.getvalue())
        self.assertEqual(res["mode"], "recall-probe")
        self.assertGreater(res["probe"]["questions"], 0)


def synth(arms, n_sessions=9, n_q=4, hal=None):
    """Judged-looking records for metrics_v2 (long-session shape, one budget)."""
    rows = []
    for s in range(n_sessions):
        sid = f"s{s}"
        for k in range(n_q):
            answerable = k != n_q - 1
            for arm in arms:
                r = {"rid": f"{sid}_q{k}::{arm}::b1536", "qid": f"{sid}_q{k}", "base_qid": f"{sid}_q{k}", "chain_id": sid,
                     "cluster_session": sid, "arm": arm, "mem_budget": 1536, "valid": True, "answerable": answerable,
                     "qtype": "single", "is_end": True, "checkpoint": 3, "lag": 1 + k % 3, "memory_tokens": 1500,
                     "score": 0.5 + (0.2 if arm == "nc-enriched" else 0.0), "anchor_hit": arm == "nc-enriched" and answerable,
                     "gold_anchor": "X" if answerable else "", "cost_usd": 0.001, "latency_s": 2.0,
                     "hallucination": (hal or {}).get(arm, 0.0), "abstained": False, "persisted_compaction": arm != "baseline",
                     "recall_error": None}
                if arm == "nc-enriched":
                    r.update({"anchor_in_enriched": answerable, "anchor_in_enriched_raw": answerable, "enrich_cost_usd": 0.0005,
                              "enrich_queries": 8, "enrich_keywords_rate": 1.0, "enrich_graph_branch_enabled_rate": 1.0,
                              "enrich_chunks_unique": 20, "enrich_duplicates_removed": 12, "recall_packed_tokens": 900})
                rows.append(r)
    return rows


class MetricsRegime(unittest.TestCase):
    def test_enriched_arm_set_is_primary_and_exploratory_arms_never_drop_a_chain(self):
        rows = synth(["baseline", "nc-enriched", "nc-note-only", "nc-chunks", "nc-graph"], n_sessions=2)
        for r in rows:
            if r["arm"] == "nc-graph" and r["chain_id"] == "s0":
                r["valid"] = False                                       # an invalid exploratory cell
        kept, rep = M.validate_rows(rows)
        self.assertEqual(rep["required_arms"], ["baseline", "nc-enriched", "nc-note-only"])
        self.assertTrue(rep["enriched_arm_set"] and not rep["legacy_arm_set"])
        self.assertEqual({r["chain_id"] for r in kept if r["arm"] == "nc-enriched"}, {"s0", "s1"})   # s0 survives
        self.assertFalse([r for r in kept if r["arm"] == "nc-graph" and r["chain_id"] == "s0"])        # the bad cell itself is dropped
        self.assertIn("nc-chunks", rep["side_arms_kept"])
        # an invalid REQUIRED arm still drops the chain
        rows2 = synth(["baseline", "nc-enriched", "nc-note-only"], n_sessions=2)
        for r in rows2:
            if r["arm"] == "nc-enriched" and r["chain_id"] == "s0":
                r["valid"] = False
        kept2, rep2 = M.validate_rows(rows2)
        self.assertEqual({r["chain_id"] for r in kept2}, {"s1"})

    def test_cli_report_has_enriched_treatment_secondary_family_and_hallucination_breakdown(self):
        rows = synth(["baseline", "nc-enriched", "nc-note-only", "nc-chunks"], hal={"baseline": 1.0, "nc-enriched": 0.0, "nc-note-only": 1.0})
        with tempfile.TemporaryDirectory() as d:
            rp, op, jp = (os.path.join(d, x) for x in ("r.jsonl", "o.md", "o.json"))
            open(rp, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                import sys
                argv = sys.argv
                sys.argv = ["metrics_v2", "--records", rp, "--out", op, "--json", jp, "--track", "longsession", "--end-only",
                            "--boot", "300"]
                try:
                    rc = M.main()
                finally:
                    sys.argv = argv
            self.assertEqual(rc, 0)
            md, js = open(op).read(), json.load(open(jp))
        self.assertEqual(js["meta"]["treatment"], "nc-enriched")
        self.assertTrue(js["meta"]["enriched"])
        self.assertEqual(js["secondary"]["family"], list(M.LS_ENRICHED_IDS))
        self.assertEqual(js["secondary"]["correction"], "Holm step-down over the fixed family of 2")
        self.assertIn("nc-enriched-baseline", js["deltas"])
        self.assertIn("nc-enriched-nc-note-only", js["deltas"])
        self.assertIn("nc-enriched-nc-chunks", js["deltas"])                # exploratory arm: descriptive pair
        self.assertIn("Amendment C.7", md)
        self.assertIn("## Hallucination by answerability", md)
        self.assertIn("halluc@ANSWERABLE", md)
        self.assertIn("nc-enriched diagnostics", md)
        self.assertEqual(js["gates"]["fail_rules"][0]["name"], "fail_score_bad")
        s = js["summary"]
        self.assertEqual(s["nc-enriched"]["halluc_counts"]["answerable"], [0, 27])
        self.assertEqual(s["baseline"]["halluc_counts"]["answerable"], [27, 27])
        self.assertEqual(s["baseline"]["hallucination_unanswerable"], 1.0)
        self.assertEqual(s["nc-enriched"]["hallucination_unanswerable"], 0.0)
        self.assertIn("hallucination_answerable", js["deltas"]["nc-enriched-baseline"]["metrics"])
        self.assertAlmostEqual(js["deltas"]["nc-enriched-baseline"]["metrics"]["hallucination_answerable"]["point"], -1.0)
        self.assertAlmostEqual(js["deltas"]["nc-enriched-baseline"]["metrics"]["false_answer"]["point"], -1.0)
        d = js["enrich_diag"]["1536"]
        self.assertEqual(d["anchor_in_enriched_budgeted"], 1.0)
        # the amortised compaction-time query cost is part of the arm's cost per question
        self.assertAlmostEqual(s["nc-enriched"]["cost_per_question"], 0.0015)
        self.assertAlmostEqual(s["baseline"]["cost_per_question"], 0.001)

    def test_secondary_family_enriched_is_inconclusive_when_underpowered(self):
        recs = [M.normalize(r) for r in synth(["baseline", "nc-enriched", "nc-note-only"], n_sessions=3)]
        sec = M.secondary_family_enriched(recs, "clean_anchor_hit", "session", 200, 1337, 3)
        self.assertTrue(sec["underpowered"])
        self.assertTrue(all("INCONCLUSIVE" in sec["tests"][k]["decision"] for k in sec["family"]))
        self.assertTrue(all(sec["tests"][k]["p_holm"] is None for k in sec["descriptive"]))


class ExportSkipJudged(Base):
    def test_only_unjudged_rows_are_exported_but_validity_sees_all(self):
        import export_longsession_judge as E
        with SharedHars():
            self.assertEqual(self.run_ls(self.out, NEW_ARMS)[0], 0)
        recs = rows_of(self.out)
        judged = os.path.join(self.d.name, "j.jsonl")
        with open(judged, "w") as f:
            for r in recs:
                if r["arm"] != "nc-enriched":
                    f.write(json.dumps({"rid": r["rid"], "score": 0.5}) + "\n")
        full, new = os.path.join(self.d.name, "full.jsonl"), os.path.join(self.d.name, "new.jsonl")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(E.main(["--records", os.path.join(self.out, "records.jsonl"), "--sessions", self.sd, "--out", full]), 0)
            self.assertEqual(E.main(["--records", os.path.join(self.out, "records.jsonl"), "--sessions", self.sd, "--out", new,
                                     "--skip-judged", judged]), 0)
        n_full = [json.loads(l) for l in open(full)]
        n_new = [json.loads(l) for l in open(new)]
        self.assertEqual({r["arm"] for r in n_new}, {"nc-enriched"})
        self.assertEqual(len(n_new), len([r for r in n_full if r["arm"] == "nc-enriched"]))
        e = next(r for r in n_new if r["gold_anchor"])
        self.assertIn("Facts recalled from long-term memory", e["visible_context"])      # the judge verifies against the block
        self.assertNotIn("memory_text", e)
        self.assertNotIn("injected", e)


class JudgeBlindness(unittest.TestCase):
    def test_arm_name_and_block_wrapper_are_masked(self):
        import judge_v2 as J
        txt = "nc-enriched says: Facts recalled from long-term memory at compaction time (HARS session ls-1): <recalled-facts> x </recalled-facts>"
        masked, hits = J.mask_arm_leaks(txt)
        for needle in ("nc-enriched", "Facts recalled", "recalled-facts"):
            self.assertNotIn(needle, masked)
        self.assertTrue(hits)
        self.assertEqual(J.ARM_LABEL_RE.sub("", "nc-enriched: the answer").strip(), "the answer")


if __name__ == "__main__":
    unittest.main()
