"""Offline tests: QA generator with a scripted LLM (no network)."""
from __future__ import annotations

import json
import os
import re
import tempfile
import unittest

import ls_fixtures as F
import gen_longsession_qa as G
import longsession as L


class ScriptedLLM:
    """Looks at the prompt, copies a verbatim quote from the first message of each excerpt."""
    model, price = "scripted", (0.0, 0.0)

    def __init__(self, mode="good"):
        self.mode, self.cost, self.calls, self.prompts = mode, 0.0, 0, []

    def complete(self, prompt, *, system=None, max_tokens=0):
        self.calls += 1
        self.prompts.append(prompt)
        blocks = re.split(r"=== window (\d+) ===", prompt)
        first = {}                                   # window -> (idx, text)
        for k in range(1, len(blocks), 2):
            m = re.search(r"\[#(\d+) \w+\] (step \d+: job JOB\d+ finished with metric_\d+ = \d+ units in [\d.]+s;)", blocks[k + 1])
            if m:
                first[int(blocks[k])] = (int(m.group(1)), m.group(2))
        if "MULTI-HOP" in prompt:
            (wa, (ia, ta)), (wb, (ib, tb)) = sorted(first.items())[:2]
            va = 1000 + 7 * ia
            vb = 1000 + 7 * ib
            item = {"question": f"What did the run after the metric_{ia} reading report later on?",
                    "canonical_short_answer": str(vb), "link_anchor": f"metric_{ia}", "gold_anchor": f"JOB{ib:04d}",
                    "hops": [{"span_idx": wa, "msg_idx": ia, "evidence": ta}, {"span_idx": wb, "msg_idx": ib, "evidence": tb}]}
            if self.mode == "bad":
                item["hops"][1]["evidence"] = "this is not a verbatim quote of anything in the session at all"
            items = [item]
        else:
            w, (i, t) = sorted(first.items())[0]
            v = 1000 + 7 * i
            item = {"question": f"Which reading did the metric_{i} counter report during that step?", "answer": str(v),
                    "gold_anchor": f"JOB{i:04d}", "msg_idx": i, "evidence": t}
            if self.mode == "bad":
                item["evidence"] = "paraphrased text that never appears in the session itself ok"
            if self.mode == "leak":
                item["question"] = f"Tell me about JOB{i:04d} and its metric_{i} value please"
            items = [item]
        return {"text": json.dumps({"items": items}), "input_tokens": 10, "output_tokens": 10, "cost_usd": 0.0,
                "stop_reason": "stop", "latency_s": 0.0, "model": "scripted"}


KW = dict(window=14000, keep_tail=3000, reserve=500, use_real=False, n_single=3, n_multihop=1,
          excerpt_tokens=1500, seed=7, per_call=1, max_calls=40, budget_usd=1.0)


class QATests(unittest.TestCase):
    def setUp(self):
        self.s = F.make_session(questions=False)

    def test_accepts_verbatim_items_and_stores_exact_spans(self):
        qs, st = G.generate(self.s, ScriptedLLM(), **KW)
        self.assertEqual(st["single_ok"], 3)
        self.assertEqual(st["multi_ok"], 1)
        taken = set()
        for q in qs:
            self.assertIn(q["qtype"], ("single", "multihop"))
            for e in q["evidence"]:
                quote = self.s["messages"][e["msg_idx"]]["text"][e["start"]:e["end"]]
                self.assertGreaterEqual(len(quote.split()), 8)
            self.assertNotIn(q["gold_anchor"].lower(), q["question"].lower())
            self.assertNotIn(q["gold_anchor"], taken)
            taken.add(q["gold_anchor"])
        mh = next(q for q in qs if q["qtype"] == "multihop")
        self.assertEqual([e["role"] for e in mh["evidence"]], ["link", "answer"])
        self.assertLess(mh["evidence"][0]["msg_idx"], mh["evidence"][1]["msg_idx"])
        # the evidence windows differ (different compaction windows)
        self.assertEqual(len(set(mh["meta"]["windows"])), 2)
        self.s["questions"] = qs
        self.assertEqual(L.validate_session(self.s), [])

    def test_questions_only_about_dropped_regions(self):
        qs, _ = G.generate(self.s, ScriptedLLM(), **KW)
        tok = L.message_tokens(self.s["messages"])
        bs = L.compute_boundaries(tok, window=14000, keep_tail=3000, reserve=500, final=True)
        last_at = bs[-1]["at_idx"]
        for q in qs:
            self.assertTrue(all(e["msg_idx"] < last_at for e in q["evidence"]))

    def test_deterministic_for_a_seed(self):
        a, _ = G.generate(self.s, ScriptedLLM(), **KW)
        b, _ = G.generate(self.s, ScriptedLLM(), **KW)
        self.assertEqual([q["qid"] + q["question"] for q in a], [q["qid"] + q["question"] for q in b])

    def test_rejects_non_verbatim_and_leaky_items(self):
        qs, st = G.generate(self.s, ScriptedLLM("bad"), **{**KW, "max_calls": 10})
        self.assertEqual(qs, [])
        self.assertIn("evidence_not_verbatim_in_one_message", st["single_rejected"])
        self.assertTrue(st["multi_rejected"])
        qs, st = G.generate(self.s, ScriptedLLM("leak"), **{**KW, "max_calls": 4, "n_multihop": 0})
        self.assertEqual(qs, [])
        self.assertIn("anchor_in_question", st["single_rejected"])

    def test_budget_and_call_caps_stop_generation(self):
        llm = ScriptedLLM()
        qs, st = G.generate(self.s, llm, **{**KW, "max_calls": 2})
        self.assertLessEqual(llm.calls, 2)

    def test_validate_single_rules(self):
        wins = G.windows_for(self.s, 14000, 3000, 500, False)
        lo, hi = wins[0]
        m = self.s["messages"][lo]
        quote = " ".join(m["text"].split()[:12])
        anchor = re.search(r"JOB\d+", quote).group(0)
        good = {"question": "Which counter reading was logged for that step?", "answer": "1234",
                "gold_anchor": anchor, "evidence": quote, "msg_idx": lo}
        ok, why, span = G.validate_single(good, self.s, wins, 0, set())
        self.assertTrue(ok, why)
        self.assertEqual(self.s["messages"][span["msg_idx"]]["text"][span["start"]:span["end"]], quote)
        # whitespace differences in the quote are tolerated, wording differences are not
        spaced = dict(good, evidence=quote.replace(" ", "  ", 2))
        self.assertTrue(G.validate_single(spaced, self.s, wins, 0, set())[0])
        for patch, reason in (({"evidence": "too short"}, "evidence_lt_8_words"),
                              ({"answer": "w " * 40}, "answer_too_long"),
                              ({"question": f"What about {anchor} here exactly?"}, "anchor_in_question"),
                              ({"gold_anchor": "x"}, "anchor_bad_form")):
            ok, why, _ = G.validate_single(dict(good, **patch), self.s, wins, 0, set())
            self.assertFalse(ok)
            self.assertTrue(any(w.startswith(reason) for w in why), (patch, why))
        self.assertIn("anchor_taken", G.validate_single(good, self.s, wins, 0, {anchor.lower()})[1])

    def test_anchor_must_not_leak_into_other_windows(self):
        s = F.make_session(questions=False)
        s["messages"][100]["text"] += " also JOB0003 shows up here again"
        wins = G.windows_for(s, 14000, 3000, 500, False)
        lo, _ = wins[0]
        text = s["messages"][3]["text"]
        quote = " ".join(text.split()[:12])
        item = {"question": "Which counter reading was logged for that step?", "answer": "1021",
                "gold_anchor": "JOB0003", "evidence": quote, "msg_idx": 3}
        ok, why, _ = G.validate_single(item, s, wins, 0, set())
        self.assertFalse(ok)
        self.assertIn("anchor_outside_window", why)

    def test_cli_dry_run_and_mock_provider(self):
        with tempfile.TemporaryDirectory() as d:
            sd = os.path.join(d, "in")
            F.write_session(sd, F.make_session(questions=False))
            out = os.path.join(d, "out")
            self.assertEqual(G.main(["--sessions", sd, "--out", out, "--dry-run", "--window-tokens", "14000",
                                     "--keep-tail", "3000", "--reserve", "500"]), 0)
            self.assertFalse(os.path.exists(os.path.join(out, "qa-manifest.json")))
            # mock provider produces unparsable answers: nothing accepted, but artifacts are written
            self.assertEqual(G.main(["--sessions", sd, "--out", out, "--provider", "mock", "--model", "mock",
                                     "--window-tokens", "14000", "--keep-tail", "3000", "--reserve", "500",
                                     "--max-calls", "3"]), 0)
            self.assertTrue(os.path.exists(os.path.join(out, "qa-manifest.json")))
            self.assertTrue(os.path.exists(os.path.join(out, "qa-raw.jsonl")))
            raw = [json.loads(l) for l in open(os.path.join(out, "qa-raw.jsonl"))]
            self.assertTrue(all("prompt" in r and "raw" in r for r in raw))


if __name__ == "__main__":
    unittest.main()
