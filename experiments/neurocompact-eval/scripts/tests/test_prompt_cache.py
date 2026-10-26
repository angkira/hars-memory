"""Amendment C.3: provider prompt caching is COST-ONLY. Offline tests (fake transport, no keys):
request shape (cache_control on system + stable prefix), the model sees the same text, cost with
cache write/read multipliers, default ON for the anthropic profile, OFF switch, pi env, dry-run."""
from __future__ import annotations

import json
import os
import sys
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
sys.dont_write_bytecode = True
import providers as P  # noqa: E402
import longsession as L  # noqa: E402
import run_chainreplay as rc  # noqa: E402

ENV = {"ANTHROPIC_API_KEY": "sk-test"}


def native_reply(inp, out, read=0, write=0, text="ans"):
    return 200, {"type": "message", "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                 "usage": {"input_tokens": inp, "output_tokens": out, "cache_read_input_tokens": read,
                           "cache_creation_input_tokens": write}}


class Tr:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, req, timeout):
        self.calls.append({"url": req.full_url, "headers": {k.lower(): v for k, v in req.header_items()},
                           "body": json.loads(req.data.decode())})
        return self.replies.pop(0)


class Multipliers(unittest.TestCase):
    def test_profile_prices_follow_anthropic_multipliers(self):
        p = P.load_profile("anthropic")
        for m in ("claude-haiku-5-5", "claude-sonnet-5-5"):
            e = p.price(m)
            self.assertAlmostEqual(e["cache_write"], e["in"] * 1.25)       # 5 min write
            self.assertAlmostEqual(e["cache_write_1h"], e["in"] * 2.0)     # 1 h write
        self.assertAlmostEqual(p.price("claude-haiku-5-5")["cache_read"], 0.1 * 0.1)
        self.assertNotIn("claude-haiku-4-5", p.data["prices"])

    def test_cost_write_read_and_ttl(self):
        p = P.load_profile("anthropic")
        c = p.cost_usd("claude-haiku-5-5", 1000, 100, cache_write=40_000)
        self.assertAlmostEqual(c, (1000 * 0.1 + 100 * 0.5 + 40_000 * 0.125) / 1e6)
        c = p.cost_usd("claude-haiku-5-5", 1000, 100, cache_write=40_000, cache_ttl="1h")
        self.assertAlmostEqual(c, (1000 * 0.1 + 100 * 0.5 + 40_000 * 0.2) / 1e6)
        c = p.cost_usd("claude-haiku-5-5", 1000, 100, cache_read=40_000)
        self.assertAlmostEqual(c, (1000 * 0.1 + 100 * 0.5 + 40_000 * 0.01) / 1e6)


class WireAndCost(unittest.TestCase):
    MSGS = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "PREFIX-LONG|question?"}]

    def test_default_on_for_anthropic_off_for_others(self):
        self.assertTrue(P.load_profile("anthropic").cache_enabled)
        for n in ("openrouter", "gemini", "custom"):
            self.assertFalse(P.load_profile(n).cache_enabled, n)
        self.assertFalse(P.load_profile("anthropic", cache=False).cache_enabled)

    def test_native_request_marks_prefix_and_keeps_text(self):
        p = P.load_profile("anthropic")
        tr = Tr([native_reply(30, 5, write=70)])
        r = P.chat("answer", self.MSGS, profile=p, environ=ENV, transport=tr, cache_prefix_chars=len("PREFIX-LONG|"))
        c = tr.calls[0]
        self.assertEqual(c["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(c["headers"]["x-api-key"], "sk-test")
        b = c["body"]
        self.assertEqual(b["system"], [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}])
        blocks = b["messages"][0]["content"]
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", blocks[1])
        # SAME text reaches the model as without caching
        self.assertEqual("".join(x["text"] for x in blocks), "PREFIX-LONG|question?")
        self.assertEqual(r.text, "ans")
        self.assertTrue(r.cached)
        self.assertEqual(r.usage["prompt_tokens"], 100)          # whole prompt
        self.assertEqual(r.usage["cache_write_tokens"], 70)
        exp = (30 * 0.1 + 5 * 0.5 + 70 * 0.125) / 1e6
        self.assertAlmostEqual(r.cost_usd, exp)
        self.assertAlmostEqual(r.cost_nocache_usd, (100 * 0.1 + 5 * 0.5) / 1e6)

    def test_second_call_reads_and_is_cheaper(self):
        p = P.load_profile("anthropic")
        tr = Tr([native_reply(30, 5, write=70), native_reply(30, 5, read=70)])
        a = P.chat("answer", self.MSGS, profile=p, environ=ENV, transport=tr, cache_prefix_chars=12)
        b = P.chat("answer", self.MSGS, profile=p, environ=ENV, transport=tr, cache_prefix_chars=12)
        self.assertLess(b.cost_usd, a.cost_usd)
        self.assertLess(b.cost_usd, b.cost_nocache_usd)

    def test_off_and_nothing_to_cache_use_plain_endpoint(self):
        plain = (200, {"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 1}})
        tr = Tr([plain, plain])
        P.chat("answer", self.MSGS, profile=P.load_profile("anthropic", cache=False), environ=ENV, transport=tr,
               cache_prefix_chars=12)
        self.assertTrue(tr.calls[0]["url"].endswith("/chat/completions"))
        self.assertEqual(tr.calls[0]["body"]["messages"], self.MSGS)
        P.chat("judge", [{"role": "user", "content": "hi"}], profile=P.load_profile("anthropic"), environ=ENV,
               transport=tr)
        self.assertTrue(tr.calls[1]["url"].endswith("/chat/completions"))      # no system, no prefix

    def test_cli_flag(self):
        import argparse
        ap = argparse.ArgumentParser()
        P.add_profile_args(ap)
        self.assertIsNone(P.cache_override(ap.parse_args([])))
        self.assertIs(P.cache_override(ap.parse_args(["--prompt-cache", "off"])), False)
        self.assertIs(P.cache_override(ap.parse_args(["--prompt-cache", "on"])), True)


class LLMWrapper(unittest.TestCase):
    def test_llm_tracks_cache_and_savings(self):
        tr = Tr([native_reply(30, 5, write=70), native_reply(30, 5, read=70)])
        llm = L.LLM("anthropic", "claude-haiku-5-5", role="answer", transport=tr)
        os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test")
        llm.complete("PREFIX-LONG|q1", cache_prefix_chars=12)
        llm.complete("PREFIX-LONG|q2", cache_prefix_chars=12)
        s = llm.cache_stats()
        self.assertEqual((s["cached_calls"], s["read_tokens"], s["write_tokens"]), (2, 70, 70))
        self.assertGreater(s["saved_usd"], 0)
        # the prefix is byte-identical in both requests
        first = lambda i: tr.calls[i]["body"]["messages"][0]["content"][0]["text"]  # noqa: E731
        self.assertEqual(first(0), first(1))


class PiSide(unittest.TestCase):
    def setUp(self):
        self._p, self._a = rc.PROMPT_CACHE, rc.ACTIVE_PROFILE

    def tearDown(self):
        rc.PROMPT_CACHE, rc.ACTIVE_PROFILE = self._p, self._a

    def test_env_identical_across_arms_and_does_not_change_model_inputs(self):
        rc.ACTIVE_PROFILE = P.load_profile("anthropic")
        rc.PROMPT_CACHE = True
        envs = {a: rc.base_env(a, "claude-haiku-5-5", "s", "http://x/mcp", None, provider="anthropic")
                for a in ("baseline", rc.ARM_CHUNKS, rc.ARM_GRAPH, rc.ARM_NOTE)}
        for k in ("NC_PROMPT_CACHE", "NC_SUM_CACHE_RETENTION", "PI_CACHE_RETENTION"):
            self.assertEqual(len({e.get(k) for e in envs.values()}), 1, k)
        self.assertEqual(envs["baseline"]["NC_PROMPT_CACHE"], "on")
        self.assertEqual(envs["baseline"]["NC_SUM_CACHE_RETENTION"], "none")
        rc.PROMPT_CACHE = False
        off = rc.base_env("baseline", "claude-haiku-5-5", "s", "http://x/mcp", None, provider="anthropic")
        self.assertEqual(off["NC_PROMPT_CACHE"], "off")
        # every model-input variable is unchanged by the cache switch
        for k in ("NC_MODEL_ID", "NC_SUM_MODEL_ID", "NC_SUM_MAX_TOKENS", "NC_SUM_REASONING_EFFORT",
                  "NC_SUM_SECTION_TRIM", "NC_INJECT_RECALL"):
            self.assertEqual(off[k], envs["baseline"][k], k)

    def test_long_retention_sets_pi_env(self):
        prof = P.load_profile("anthropic")
        prof.data["cache"]["pi_retention"] = "long"
        rc.ACTIVE_PROFILE, rc.PROMPT_CACHE = prof, True
        e = rc.base_env("baseline", "claude-haiku-5-5", "s", "http://x/mcp", None, provider="anthropic")
        self.assertEqual(e["PI_CACHE_RETENTION"], "long")

    def test_projection_cheaper_with_cache_same_tokens(self):
        chain = {"chain_id": "c", "session_id": "s", "chain_idx": 0,
                 "spans": [{"span_idx": i, "lag": 3 - i, "segments": [{"text": "x" * 4000}] * 20} for i in range(3)],
                 "qa": [{"qid": f"q{i}", "question": "q?"} for i in range(10)]}
        rc.ACTIVE_PROFILE = P.load_profile("anthropic")
        rc.PROMPT_CACHE = True
        on = rc.project(chain, ["baseline", rc.ARM_CHUNKS], "claude-haiku-5-5", "claude-haiku-5-5", "claude-haiku-5-5",
                        "http://x/mcp")
        self.assertLess(on["usd_agent"], on["usd_agent_nocache"])
        rc.PROMPT_CACHE = False
        off = rc.project(chain, ["baseline", rc.ARM_CHUNKS], "claude-haiku-5-5", "claude-haiku-5-5", "claude-haiku-5-5",
                         "http://x/mcp")
        self.assertAlmostEqual(off["usd_agent"], on["usd_agent_nocache"])
        self.assertEqual(off["est"]["replay_in"], on["est"]["replay_in"])


if __name__ == "__main__":
    unittest.main()
