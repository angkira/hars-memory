"""Offline tests: BEAM importer on a synthetic repo layout (no BEAM text)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import ls_fixtures as F  # noqa: F401
import import_beam as B
import longsession as L


def turn(i, role, text, **kw):
    d = {"role": role, "id": i, "content": text}
    d.update(kw)
    return d


def make_repo(root, size="1M", n="1", ten=False):
    d = os.path.join(root, "chats", size, n)
    os.makedirs(os.path.join(d, "probing_questions"))
    b1 = {"batch_number": 1, "time_anchor": "March-01-2024", "turns": [
        [turn(0, "user", "My server port is 8461 and the db is orders_db.", index="1,3", question_type="main_question",
              time_anchor="March-01-2024"), turn(1, "assistant", "Noted: port 8461.")],
        [turn(2, "user", "Actually the port is now 9090."), turn(3, "assistant", "Updated.")]]}
    b2 = {"batch_number": 2, "time_anchor": None, "turns": [
        [turn(4, "user", "I never wrote tests.", time_anchor="March-16-2024"), turn(5, "assistant", "ok")],
        [turn(6, "user", "I have pytest tests at 88% coverage."), turn(7, "assistant", "great")]]}
    if ten:
        chat = [{"plan-1": [b1]}, {"plan-2": [dict(b2, turns=[[turn(8, "user", "x"), turn(9, "assistant", "y")]])]}]
    else:
        chat = [b1, b2]
    json.dump(chat, open(os.path.join(d, "chat.json"), "w"))
    json.dump({"title": "Synthetic topic"}, open(os.path.join(d, "topic.json"), "w"))
    pq = {
        "information_extraction": [{"question": "What port did I set?", "answer": "8461",
                                    "source_chat_ids": [0], "rubric": ["LLM response should state: 8461"]}],
        "knowledge_update": [{"question": "What is the port now?", "answer": "9090",
                              "source_chat_ids": {"original_info": [0], "updated_info": [2]}}],
        "temporal_reasoning": [{"question": "How many days between my notes?", "answer": "15 days",
                                "source_chat_ids": {"first_event": [0], "second_event": [4]}}],
        "multi_session_reasoning": [{"question": "Which numbers did I mention?", "answer": "Two numbers",
                                     "source_chat_ids": [0, 2, 9999]}],
        "contradiction_resolution": [{"question": "Did I write tests?", "ideal_answer": "Contradiction: you said never and 88%.",
                                      "source_chat_ids": {"first_statement": [4], "second_statement": [6]},
                                      "rubric": ["LLM response should mention: contradictory"]}],
        "abstention": [{"question": "What is my cat's name?", "ideal_response": "No information.", "rubric": []}],
        "summarization": [{"question": "Summarize.", "ideal_summary": "A summary.", "source_chat_ids": [0, 2]}],
        "event_ordering": [{"question": "Order?", "answer": "1, 2", "source_chat_ids": [0, 2]}],
    }
    json.dump(pq, open(os.path.join(d, "probing_questions", "probing_questions.json"), "w"))
    # a pickle must never be read: plant one that would execute code if unpickled
    with open(os.path.join(d, "chat.pickle"), "wb") as f:
        f.write(b"cos\nsystem\n(S'echo pwned > /tmp/ls_pickle_probe'\ntR.")


class BeamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        make_repo(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_flatten_and_ids(self):
        s = B.import_chat(self.tmp.name, "1M", "1", B.DEFAULT_TYPES)
        self.assertEqual(L.validate_session(s), [])
        self.assertEqual([m["idx"] for m in s["messages"]], list(range(8)))
        self.assertEqual(s["sid"], "beam-1M-1")
        self.assertEqual(s["title"], "Synthetic topic")
        self.assertEqual(s["messages"][0]["ts"], "March-01-2024")
        self.assertEqual(s["messages"][5]["ts"], "March-16-2024")       # carried forward
        self.assertEqual(s["compactions"], [])
        self.assertEqual([b["start_idx"] for b in s["meta"]["batches"]], [0, 4])

    def test_question_mapping_and_dropped_types(self):
        s = B.import_chat(self.tmp.name, "1M", "1", B.DEFAULT_TYPES)
        by = {q["source_qtype"]: q for q in s["questions"]}
        self.assertEqual(set(by), set(B.DEFAULT_TYPES))
        self.assertEqual(by["information_extraction"]["qtype"], "single")
        self.assertEqual(by["multi_session_reasoning"]["qtype"], "multihop")
        self.assertEqual(by["knowledge_update"]["qtype"], "knowledge_update")
        self.assertEqual(by["information_extraction"]["evidence"], [{"msg_idx": 0}])
        self.assertEqual(by["knowledge_update"]["evidence"],
                         [{"msg_idx": 0, "role": "original_info"}, {"msg_idx": 2, "role": "updated_info"}])
        self.assertEqual(by["contradiction_resolution"]["answer"], "Contradiction: you said never and 88%.")
        self.assertEqual(by["multi_session_reasoning"]["meta"]["unmapped_evidence"], 1)   # id 9999
        self.assertEqual(s["meta"]["dropped_types"],
                         {"abstention": 1, "event_ordering": 1, "summarization": 1})
        self.assertEqual(by["information_extraction"]["rubric"], ["LLM response should state: 8461"])
        self.assertEqual(L.validate_session(s), [])

    def test_gold_anchor_only_for_verbatim_short_answers(self):
        s = B.import_chat(self.tmp.name, "1M", "1", B.DEFAULT_TYPES)
        by = {q["source_qtype"]: q for q in s["questions"]}
        self.assertEqual(by["information_extraction"]["gold_anchor"], "8461")
        self.assertEqual(by["knowledge_update"]["gold_anchor"], "9090")
        self.assertNotIn("gold_anchor", by["temporal_reasoning"])
        self.assertNotIn("gold_anchor", by["contradiction_resolution"])

    def test_types_all_keeps_abstention_as_unanswerable(self):
        s = B.import_chat(self.tmp.name, "1M", "1", B.ALL_TYPES)
        ab = next(q for q in s["questions"] if q["source_qtype"] == "abstention")
        self.assertFalse(ab["answerable"])
        self.assertEqual(ab["evidence"], [])
        self.assertEqual(s["meta"]["dropped_types"], {})
        su = next(q for q in s["questions"] if q["source_qtype"] == "summarization")
        self.assertEqual(su["answer"], "A summary.")

    def test_ten_m_plan_layout_and_cli(self):
        with tempfile.TemporaryDirectory() as r2, tempfile.TemporaryDirectory() as out:
            make_repo(r2, "10M", "3", ten=True)
            rc = B.main(["--repo", r2, "--size", "10M", "--out", out])
            self.assertEqual(rc, 0)
            s = L.load_sessions(out)[0]
            self.assertEqual(s["sid"], "beam-10M-3")
            self.assertEqual([m["idx"] for m in s["messages"]], list(range(len(s["messages"]))))
            self.assertEqual([b["plan"] for b in s["meta"]["batches"]], ["plan-1", "plan-2"])
            self.assertEqual(len(s["messages"]), 6)

    def test_pickles_are_never_loaded(self):
        probe = "/tmp/ls_pickle_probe"
        if os.path.exists(probe):
            os.remove(probe)
        B.import_chat(self.tmp.name, "1M", "1", B.DEFAULT_TYPES)
        self.assertFalse(os.path.exists(probe))

    def test_bad_type_is_rejected(self):
        with self.assertRaises(SystemExit):
            B.main(["--repo", self.tmp.name, "--size", "1M", "--out", self.tmp.name + "/o", "--types", "nonsense"])


if __name__ == "__main__":
    unittest.main()
