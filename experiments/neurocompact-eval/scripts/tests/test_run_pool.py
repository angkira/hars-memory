"""Offline tests for run_pool.extract (pi 1.4.0 ``--mode json`` stdout parser)."""
from __future__ import annotations

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from run_pool import extract  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "pi_events.jsonl")


def _read() -> str:
    with open(FIXTURE) as f:
        return f.read()


class ExtractTests(unittest.TestCase):
    def test_fixture_two_turns_with_tool_call_and_compaction(self):
        ex = extract(_read())
        self.assertEqual(ex["text"], "The config is in a.txt.")   # final assistant text only
        # last turn context = input + cacheRead + cacheWrite = 7 + 2657 + 431
        self.assertEqual(ex["usage"]["input"], 3095)
        self.assertEqual(ex["usage"]["output"], 30 + 12)           # summed over turns
        self.assertEqual(ex["compactions"], 1)

    def test_cache_tokens_are_summed_into_input(self):
        line = json.dumps({"type": "message_end", "message": {
            "role": "assistant", "content": [{"type": "text", "text": "pong"}],
            "usage": {"input": 4, "output": 4, "cacheRead": 0, "cacheWrite": 2657}}})
        self.assertEqual(extract(line)["usage"], {"input": 2661, "output": 4})

    def test_streaming_duplicates_not_double_counted(self):
        # message_update / turn_end repeat the assistant message; only message_end counts.
        ex = extract(_read())
        self.assertEqual(ex["usage"]["output"], 42)

    def test_compactions_counts_each_compaction_end_not_start(self):
        lines = "\n".join(json.dumps(e) for e in [
            {"type": "compaction_start", "reason": "manual"},
            {"type": "compaction_end", "reason": "manual", "aborted": False},
            {"type": "compaction_start", "reason": "threshold"},
            {"type": "compaction_end", "reason": "threshold", "aborted": False},
        ])
        self.assertEqual(extract(lines)["compactions"], 2)

    def test_empty_garbage_and_non_json_lines(self):
        zero = {"text": "", "usage": {"input": 0, "output": 0}, "compactions": 0}
        self.assertEqual(extract(""), zero)
        self.assertEqual(extract(None), zero)
        self.assertEqual(extract("not json\n{broken\n[1,2]\n"), zero)

    def test_ignores_system_user_and_tool_result_messages(self):
        out = extract(_read())
        self.assertNotIn("Where is the config", out["text"])

    def test_string_content_and_missing_usage(self):
        line = json.dumps({"type": "message_end",
                           "message": {"role": "assistant", "content": "plain"}})
        self.assertEqual(extract(line),
                         {"text": "plain", "usage": {"input": 0, "output": 0}, "compactions": 0})

    def test_works_with_run_chainreplay_dead_run_check(self):
        # run_chainreplay treats empty text + 0/0 usage as a dead run.
        ex = extract('{"type":"agent_start"}\n{"type":"agent_settled","aborted":false}\n')
        self.assertTrue(not ex["text"].strip() and ex["usage"]["input"] == 0)


if __name__ == "__main__":
    unittest.main()
