"""Offline tests: longsession format, IO, pieces, boundaries, lag, mock LLM."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import ls_fixtures as F  # noqa: F401  (puts scripts/ on sys.path)
import longsession as L


class FormatTests(unittest.TestCase):
    def test_valid_fixture_has_no_problems(self):
        self.assertEqual(L.validate_session(F.make_session()), [])

    def test_validation_catches_defects(self):
        s = F.make_session()
        s["messages"][3]["idx"] = 9
        self.assertTrue(any("idx" in w for w in L.validate_session(s)))
        s = F.make_session()
        s["compactions"] = [{"at_idx": 10}, {"at_idx": 10}]
        self.assertTrue(any("not increasing" in w for w in L.validate_session(s)))
        s = F.make_session()
        s["questions"][0]["evidence"] = [{"msg_idx": 9999}]
        self.assertTrue(any("out of range" in w for w in L.validate_session(s)))
        s = F.make_session()
        s["sid"] = "bad sid!"
        self.assertTrue(any("bad sid" in w for w in L.validate_session(s)))
        s = F.make_session()
        s["messages"][1]["role"] = "robot"
        self.assertTrue(any("bad role" in w for w in L.validate_session(s)))

    def test_json_and_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = F.make_session("a-1", n=8), F.make_session("b-1", n=8)
            L.save_session(a, d)
            with open(os.path.join(d, "many.jsonl"), "w") as f:
                f.write(json.dumps(b) + "\n")
            got = {s["sid"]: s for s in L.load_sessions(d)}
            self.assertEqual(set(got), {"a-1", "b-1"})
            self.assertEqual(got["a-1"]["messages"], a["messages"])
            self.assertEqual(got["a-1"]["format"], L.FORMAT)


class PieceTests(unittest.TestCase):
    def test_pieces_are_lossless_ordered_and_bounded(self):
        s = F.make_session(n=30, chars=1500)
        pieces = L.split_pieces(s["messages"], 4000)
        self.assertTrue(all(len(p) <= 4000 for p in pieces))
        joined = "\n\n".join(pieces)
        for m in s["messages"]:
            self.assertIn(m["text"], joined)
        order = [joined.index(f"job JOB{i:04d}") for i in range(30)]
        self.assertEqual(order, sorted(order))
        # pieces start at message boundaries (a role tag), never mid-message
        self.assertTrue(all(p.startswith("[") for p in pieces))

    def test_oversized_message_is_split_at_whitespace(self):
        big = {"idx": 0, "role": "tool", "text": ("word " * 3000).strip()}
        pieces = L.split_pieces([big], 2000)
        self.assertGreater(len(pieces), 7)
        self.assertTrue(all(len(p) <= 2000 for p in pieces))
        self.assertEqual("".join(pieces).replace(" ", "").replace("[Toolresult]:", ""),
                         ("word" * 3000))

    def test_empty(self):
        self.assertEqual(L.split_pieces([], 100), [])


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tok = [100] * 200            # 20k tokens

    def test_partition_tail_and_progress(self):
        bs = L.compute_boundaries(self.tok, window=5000, keep_tail=1000, reserve=500, final=True)
        self.assertGreaterEqual(len(bs), 4)
        prev = 0
        for i, b in enumerate(bs, 1):
            self.assertEqual(b["j"], i)
            self.assertEqual(b["drop_start"], prev)           # dropped regions tile [0, last at_idx)
            self.assertGreater(b["at_idx"], b["drop_start"])  # progress
            self.assertLessEqual(b["tail_tokens"], 1000)
            self.assertLessEqual(b["at_idx"], b["trigger_idx"] + 1)
            prev = b["at_idx"]
        self.assertTrue(bs[-1]["final"])
        self.assertEqual(bs[-1]["trigger_idx"], len(self.tok) - 1)

    def test_live_context_never_exceeds_window_before_a_boundary(self):
        bs = L.compute_boundaries(self.tok, window=5000, keep_tail=1000, reserve=500)
        for b in (x for x in bs if not x["final"]):         # the forced final one is not window-driven
            live = sum(self.tok[b["drop_start"]:b["trigger_idx"] + 1]) + 500
            self.assertGreater(live, 5000)                     # it fired because it exceeded
            self.assertLessEqual(live - self.tok[b["trigger_idx"]], 5000)   # not one message late

    def test_no_final_and_determinism(self):
        a = L.compute_boundaries(self.tok, window=5000, keep_tail=1000, reserve=500, final=False)
        b = L.compute_boundaries(self.tok, window=5000, keep_tail=1000, reserve=500, final=False)
        self.assertEqual(a, b)
        self.assertFalse(any(x["final"] for x in a))

    def test_bad_params_raise(self):
        with self.assertRaises(ValueError):
            L.compute_boundaries(self.tok, window=1000, keep_tail=900, reserve=200)

    def test_short_session_has_no_boundary(self):
        self.assertEqual(L.compute_boundaries([10] * 5, window=5000, keep_tail=1000, reserve=100,
                                              final=False), [])

    def test_real_mode_uses_session_at_idx_and_limits_tail(self):
        real = [{"at_idx": 50, "original_summary": "S1"}, {"at_idx": 120, "original_summary": None}]
        bs = L.compute_boundaries(self.tok, window=10 ** 6, keep_tail=2500, reserve=0, real=real)
        self.assertEqual([b["at_idx"] for b in bs], [50, 120])
        self.assertEqual([b["drop_start"] for b in bs], [0, 50])
        self.assertTrue(all(b["real"] for b in bs))
        self.assertEqual(bs[0]["original_summary"], "S1")
        self.assertEqual(bs[0]["tail_tokens"], 2500)          # 25 messages of 100
        self.assertEqual(bs[0]["trigger_idx"], 74)
        self.assertFalse(any(b["final"] for b in bs))
        # tail never crosses the next real boundary
        bs2 = L.compute_boundaries(self.tok, window=10 ** 6, keep_tail=10 ** 6, reserve=0,
                                   real=[{"at_idx": 50}, {"at_idx": 60}])
        self.assertEqual(bs2[0]["trigger_idx"], 59)

    def test_real_mode_zero_tail(self):
        bs = L.compute_boundaries(self.tok, window=10 ** 6, keep_tail=0, reserve=0,
                                  real=[{"at_idx": 50}])
        self.assertEqual((bs[0]["trigger_idx"], bs[0]["tail_tokens"]), (49, 0))

    def test_lag_and_eligibility(self):
        bs = L.compute_boundaries(self.tok, window=5000, keep_tail=1000, reserve=500, final=True)
        q = {"qid": "x", "evidence": [{"msg_idx": 3}]}
        r = L.region_of(bs, 3)
        self.assertEqual(r, 1)
        self.assertEqual(L.question_lag(q, bs, 1), 1)
        self.assertEqual(L.question_lag(q, bs, 4), 4)
        # evidence still inside the tail -> lag 0
        q2 = {"qid": "y", "evidence": [{"msg_idx": bs[0]["at_idx"]}]}
        self.assertEqual(L.question_lag(q2, bs, 1), 0)
        self.assertIsNone(L.question_lag({"qid": "z", "evidence": []}, bs, 1))
        # not yet happened -> not eligible; unanswerable always eligible
        late = {"qid": "l", "evidence": [{"msg_idx": 190}]}
        self.assertFalse(L.question_eligible(late, bs[0]))
        self.assertTrue(L.question_eligible(late, bs[-1]))
        self.assertTrue(L.question_eligible({"qid": "u", "answerable": False, "evidence": []}, bs[0]))


class MockLLMTests(unittest.TestCase):
    def test_mock_summary_respects_cap_and_has_sections(self):
        llm = L.LLM("mock", "mock")
        user = ("... exactly these four section headers ...\nthe hard output limit is 120 tokens.\n"
                "<conversation>\n" + " ".join(f"JOB{i:04d} metric_{i}" for i in range(200)) + "\n</conversation>")
        out = llm.complete(user)["text"]
        for h in ("GOAL", "KEY FIGURES & IDENTIFIERS", "DECISIONS & RATIONALE", "ACTIVE STATE"):
            self.assertIn(h, out)
        self.assertLessEqual(L.count_tokens(out), 130)

    def test_unknown_provider_and_missing_key(self):
        with self.assertRaises(ValueError):
            L.LLM("nope", "m")
        llm = L.LLM("anthropic", "claude-haiku-5-5", key_env="LS_TEST_NO_SUCH_KEY")
        self.assertIn("LS_TEST_NO_SUCH_KEY", llm.key_problem())

    def test_prices_known_and_override(self):
        self.assertEqual(L.price_for("claude-haiku-5-5"), (0.10, 0.50))
        self.assertEqual(L.price_for("claude-haiku-5-5", (1.0, 2.0)), (1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
