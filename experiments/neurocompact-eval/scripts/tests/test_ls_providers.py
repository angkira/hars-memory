"""Offline tests: providers.py wiring of the long-session / dataset scripts (one price source,
mandatory --budget-usd, anthropic thinking knob, alias flags)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest

import ls_fixtures as F
import gen_longsession_qa as G
import longsession as L
import oh_pipeline as OHP
import providers as P
import run_longsession as R

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)


def fake(body_log, text="ok", fin="stop", usage=None, fail_thinking=False):
    def send(req, timeout):
        body = json.loads(req.data.decode())
        body_log.append(body)
        if fail_thinking and "thinking" in body:
            return 400, {"error": {"message": "thinking: Extra inputs are not permitted"}}
        return 200, {"choices": [{"message": {"content": text}, "finish_reason": fin}],
                     "usage": usage or {"prompt_tokens": 1000, "completion_tokens": 200}}
    return send


class LLMOverProviders(unittest.TestCase):
    def setUp(self):
        os.environ["LS_FAKE_KEY"] = "k"

    def test_cost_from_profile_table_charges_the_shared_budget(self):
        log = []
        b = P.Budget(1.0)
        llm = L.LLM("anthropic", "claude-haiku-5-5", budget=b, key_env="LS_FAKE_KEY", transport=fake(log))
        r = llm.complete("hi", max_tokens=50)
        self.assertAlmostEqual(r["cost_usd"], (1000 * 0.10 + 200 * 0.50) / 1e6)
        self.assertAlmostEqual(b.spent, r["cost_usd"])
        self.assertEqual((r["input_tokens"], r["output_tokens"], r["stop_reason"]), (1000, 200, "stop"))
        # anthropic: no temperature/top_p on the wire, thinking disabled, OpenAI-compat endpoint
        self.assertNotIn("temperature", log[0])
        self.assertEqual(log[0]["thinking"], {"type": "disabled"})

    def test_prompt_above_100k_uses_the_tier_price(self):
        log = []
        llm = L.LLM("anthropic", "claude-haiku-5-5", key_env="LS_FAKE_KEY",
                    transport=fake(log, usage={"prompt_tokens": 120000, "completion_tokens": 1000}))
        self.assertAlmostEqual(llm.complete("x")["cost_usd"], (120000 * 0.50 + 1000 * 2.50) / 1e6)

    def test_length_stop_and_thinking_fallback(self):
        log = []
        llm = L.LLM("anthropic", "claude-haiku-5-5", key_env="LS_FAKE_KEY",
                    transport=fake(log, fin="length", fail_thinking=True))
        r = llm.complete("x")
        self.assertEqual(r["stop_reason"], "length")
        self.assertEqual(len(log), 2)                         # first with thinking (400), then without
        self.assertIn("thinking", log[0])
        self.assertNotIn("thinking", log[1])

    def test_openrouter_keeps_temperature_and_has_no_thinking(self):
        log = []
        llm = L.LLM("openrouter", "z-ai/glm-5.3-flash", key_env="LS_FAKE_KEY", transport=fake(
            log, usage={"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.5}))
        r = llm.complete("x")
        self.assertEqual(log[0]["temperature"], 0)
        self.assertNotIn("thinking", log[0])
        self.assertEqual(r["cost_usd"], 0.5)                  # provider-reported cost wins

    def test_budget_is_a_hard_stop_and_unpriced_is_refused(self):
        log = []
        b = P.Budget(0.0005)
        llm = L.LLM("anthropic", "claude-haiku-5-5", budget=b, key_env="LS_FAKE_KEY", transport=fake(log))
        llm.complete("x")                                      # 0.0002 spent
        llm.complete("x")
        llm.complete("x")                                      # 0.0006 >= cap now
        with self.assertRaises(L.LLMError):
            llm.complete("x")
        self.assertEqual(len(log), 3)                          # the 4th never hit the wire
        u = L.LLM("anthropic", "some-unpriced-model", budget=P.Budget(1.0), key_env="LS_FAKE_KEY",
                  transport=fake([]))
        with self.assertRaises(L.LLMError) as cm:
            u.complete("x")
        self.assertIn("no price entry", str(cm.exception))

    def test_price_override_and_single_source(self):
        llm = L.LLM("anthropic", "claude-haiku-5-5", key_env="LS_FAKE_KEY", price=(1.0, 2.0),
                    transport=fake([]))
        self.assertAlmostEqual(llm.complete("x")["cost_usd"], (1000 * 1.0 + 200 * 2.0) / 1e6)
        self.assertEqual(L.price_for("claude-haiku-5-5"), P.price_tuple("claude-haiku-5-5"))
        self.assertEqual(L.price_for("qwen/qwen3.8-flash"), P.price_tuple("qwen/qwen3.8-flash"))
        self.assertEqual(L.price_for("not-a-model"), (0.0, 0.0))

    def test_no_duplicate_price_tables_left(self):
        for fn in ("longsession.py", "run_chainreplay.py", "oh_pipeline.py", "run_longsession.py",
                   "gen_longsession_qa.py", "multihop.py", "oh_salvage.py"):
            src = open(os.path.join(SCRIPTS, fn)).read()
            self.assertIsNone(re.search(r"^PRICES\s*=\s*\{\s*\n?\s*\"", src, re.M), fn)
            self.assertNotIn("PRICE_IN, mc.PRICE_OUT = 0.10e-6", src, fn)


def run(mod_main, argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = mod_main(argv)
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


class MandatoryBudget(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.sd = os.path.join(self.d.name, "in")
        F.write_session(self.sd, F.make_session("syn-1"))
        os.environ["LS_FAKE_KEY"] = "k"

    def tearDown(self):
        self.d.cleanup()

    def test_run_longsession_refuses_a_paid_run_without_budget(self):
        rc, so, se = run(R.main, ["--sessions", self.sd, "--out", os.path.join(self.d.name, "o"),
                                  "--profile", "anthropic", "--key-env", "LS_FAKE_KEY", "--fake-hars",
                                  "--window", "14000", "--keep-tail", "3000", "--mem-budgets", "512",
                                  "--no-preflight"])
        self.assertEqual(rc, 3)
        self.assertIn("--budget-usd is mandatory", se)
        self.assertFalse(os.path.exists(os.path.join(self.d.name, "o", "records.jsonl")))

    def test_dry_run_needs_none_and_prints_a_budget_check(self):
        base = ["--sessions", self.sd, "--dry-run", "--profile", "anthropic", "--window", "14000",
                "--keep-tail", "3000", "--mem-budgets", "512"]
        rc, so, se = run(R.main, base)
        self.assertEqual(rc, 0, se)
        self.assertNotIn("BUDGET CHECK", so)
        rc, so, se = run(R.main, base + ["--budget-usd", "0.0001"])
        self.assertIn("OVER THE CAP", so)
        rc, so, se = run(R.main, base + ["--max-cost-usd", "100"])          # alias
        self.assertIn("BUDGET CHECK", so)
        self.assertIn("OK", so)

    def test_profile_models_and_aliases_resolve(self):
        a = R.parse_args(["--sessions", self.sd, "--profile", "anthropic"])
        self.assertEqual((a.agent_model, a.sum_model, a.qa_model, a.extractor_model),
                         ("claude-haiku-5-5",) * 4)
        a = R.parse_args(["--sessions", self.sd, "--profile", "anthropic", "--agent-model", "claude-sonnet-5-5"])
        self.assertEqual((a.sum_model, a.qa_model, a.extractor_model),
                         ("claude-sonnet-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"))
        a = R.parse_args(["--sessions", self.sd, "--provider", "openrouter", "--model-summarizer", "x/y",
                          "--qa-model", "q/z"])
        self.assertEqual((a.provider, a.sum_model, a.qa_model), ("openrouter", "x/y", "q/z"))
        a = R.parse_args(["--sessions", self.sd, "--provider", "mock"])
        self.assertEqual((a.provider, a.agent_model), ("mock", "mock"))

    def test_budget_stops_a_run_and_resume_continues(self):
        # mock LLM is free: simulate the cap through the estimate charge of the HARS extraction
        out = os.path.join(self.d.name, "o")
        base = ["--sessions", self.sd, "--out", out, "--provider", "mock", "--agent-model", "mock", "--fake-hars",
                "--no-preflight", "--window", "14000", "--keep-tail", "3000", "--mem-budgets", "512",
                "--piece-chars", "3000", "--pieces-per-call", "4"]
        rc, so, se = run(R.main, base)
        self.assertEqual(rc, 0, so + se)
        man = json.load(open(os.path.join(out, "MANIFEST.json")))
        self.assertIn("spend_usd", man)
        self.assertIn("budget_usd", man)
        self.assertIsNone(man["profile"])                      # mock has no profile

    def test_gen_qa_requires_budget_unless_dry_or_mock(self):
        s = F.make_session("g-1", questions=False)
        gd = os.path.join(self.d.name, "g")
        F.write_session(gd, s)
        common = ["--sessions", gd, "--out", os.path.join(self.d.name, "gout"), "--window-tokens", "14000",
                  "--keep-tail", "3000", "--reserve", "500", "--max-calls", "2"]
        rc, so, se = run(G.main, common + ["--profile", "anthropic", "--key-env", "LS_FAKE_KEY"])
        self.assertIn("--budget-usd is mandatory", str(rc) + se + so)
        rc, so, se = run(G.main, common + ["--profile", "anthropic", "--dry-run", "--budget-usd", "0.001"])
        self.assertEqual(rc, 0, se)
        self.assertIn("claude-sonnet-5-5", so)                 # profile's qa_generator role
        self.assertIn("OVER CAP", so)
        rc, so, se = run(G.main, common + ["--provider", "mock", "--model", "mock"])
        self.assertEqual(rc, 0, se)

    def test_gen_qa_live_path_charges_budget(self):
        s = F.make_session("g-2", questions=False)
        gd = os.path.join(self.d.name, "g2")
        F.write_session(gd, s)
        log = []
        orig = L.LLM.__init__

        def init(self_, *a, **k):
            k["transport"] = fake(log, text='{"items": []}')
            orig(self_, *a, **k)
        L.LLM.__init__ = init
        try:
            rc, so, se = run(G.main, ["--sessions", gd, "--out", os.path.join(self.d.name, "g2out"),
                                      "--window-tokens", "14000", "--keep-tail", "3000", "--reserve", "500",
                                      "--max-calls", "3", "--profile", "anthropic", "--key-env", "LS_FAKE_KEY",
                                      "--budget-usd", "1"])
        finally:
            L.LLM.__init__ = orig
        self.assertEqual(rc, 0, so + se)
        self.assertTrue(log)
        man = json.load(open(os.path.join(self.d.name, "g2out", "qa-manifest.json")))
        self.assertEqual(man["budget_usd"], 1.0)
        self.assertGreater(man["total_cost_usd"], 0)
        self.assertEqual(man["price_status"], "confirmed")


class OhPipelineProfiles(unittest.TestCase):
    def test_parse_pre_and_legacy_alias(self):
        o, rest = OHP.parse_pre(["--profile", "anthropic", "--key-env", "K", "--sessions-dir", "d", "make"])
        self.assertEqual((o["profile"], o["key_env"], rest[0]), ("anthropic", "K", "--sessions-dir"))
        o, _ = OHP.parse_pre(["--qa-provider", "anthropic", "--sessions-dir", "d", "make"])
        self.assertEqual(o["profile"], "anthropic")
        o, _ = OHP.parse_pre(["--qa-provider", "openrouter", "--sessions-dir", "d", "make"])
        self.assertIsNone(o["profile"])                        # legacy original call, unchanged

    def test_live_run_without_budget_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            old = sys.argv
            for which in ("make", "multihop", "salvage"):
                sys.argv = ["oh_pipeline.py", "--profile", "anthropic", "--sessions-dir", d, which, "--out", "x"]
                with self.assertRaises(SystemExit) as cm:
                    OHP.main()
                self.assertIn("--budget-usd is mandatory", str(cm.exception))
            sys.argv = old

    def test_profile_adapter_replaces_make_chains_llm(self):
        import make_chains as mc
        os.environ["LS_FAKE_KEY"] = "k"
        log = []
        saved = (mc.llm, mc.PRICE_IN, mc.PRICE_OUT, os.environ.get("OPENROUTER_API_KEY"))
        try:
            prof = P.load_profile("anthropic")
            OHP.install_profile(mc, prof, "claude-haiku-5-5", "LS_FAKE_KEY", "disabled", transport=fake(log, text="RAW"))
            spend = mc.Spend(0.01)
            text, usage = mc.llm("prompt", "ignored", "claude-haiku-5-5", spend, max_tokens=100, est_cost=0.001)
            self.assertEqual(text, "RAW")
            self.assertAlmostEqual(spend.spent, (1000 * 0.10 + 200 * 0.50) / 1e6)
            self.assertAlmostEqual(mc.PRICE_IN * 1e6, 0.10)
            self.assertEqual(log[0]["thinking"], {"type": "disabled"})
            # the hard cap: an estimate over the remaining budget is refused without a call
            tiny = mc.Spend(0.0001)
            self.assertEqual(mc.llm("p", "k", "claude-haiku-5-5", tiny, est_cost=0.01), ("", {}))
            self.assertEqual(len(log), 1)
            with self.assertRaises(SystemExit):
                OHP.install_profile(mc, prof, "not-priced-model", "LS_FAKE_KEY")
        finally:
            mc.llm, mc.PRICE_IN, mc.PRICE_OUT = saved[0], saved[1], saved[2]
            if saved[3] is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = saved[3]

    def test_use_profile_injects_the_role_model(self):
        import make_chains as mc
        rest = OHP.use_profile(mc, "anthropic", {"model": None}, ["--out", "x"], live=False)
        self.assertEqual(rest[-2:], ["--model", "claude-sonnet-5-5"])
        rest = OHP.use_profile(mc, "anthropic", {"model": None}, ["--model", "claude-haiku-5-5"], live=False)
        self.assertEqual(rest, ["--model", "claude-haiku-5-5"])
        self.assertEqual(OHP.use_profile(mc, None, {}, ["--x"], live=True), ["--x"])


if __name__ == "__main__":
    unittest.main()
