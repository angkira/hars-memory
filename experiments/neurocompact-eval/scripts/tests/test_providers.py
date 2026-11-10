"""Offline tests for providers.py: profile loading, role->model mapping, sampling policy,
cost calc + tiers, budget, hars-env output, request building / chat() with a fake transport,
judge-family verdict + the --allow-same-family-judge flag in judge_v2 / run_chainreplay /
metrics_v2. No network, no keys."""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
sys.dont_write_bytecode = True
import providers as P  # noqa: E402
import judge_v2  # noqa: E402
import metrics_v2  # noqa: E402
import run_chainreplay as rc  # noqa: E402


class FakeTransport:
    def __init__(self, replies=None):
        self.calls = []
        self.replies = list(replies or [])

    def __call__(self, req, timeout):
        body = json.loads(req.data.decode())
        self.calls.append({"url": req.full_url, "headers": dict(req.header_items()), "body": body})
        if self.replies:
            return self.replies.pop(0)
        return 200, {"choices": [{"message": {"content": json.dumps(
            {"score": 2, "unsupported_claim": False, "abstained": False, "reason": "ok"})}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}


class Profiles(unittest.TestCase):
    def test_all_shipped_profiles_load(self):
        for n in P.PROFILE_NAMES:
            p = P.load_profile(n)
            self.assertEqual(p.name, n)
            self.assertIn(p.data["auth"]["style"], ("bearer", "x-api-key"))

    def test_unknown_profile(self):
        with self.assertRaises(P.ProviderError):
            P.load_profile("nope")

    def test_role_model_table(self):
        a = P.resolve_models(P.load_profile("anthropic"))
        self.assertEqual({r: a[r] for r in ("agent", "answer", "summarizer", "extractor", "keyword")},
                         {r: "claude-haiku-5-5" for r in ("agent", "answer", "summarizer",
                                                         "extractor", "keyword")})
        self.assertEqual(a["judge"], "claude-sonnet-5-5")
        o = P.resolve_models(P.load_profile("openrouter"))
        self.assertEqual((o["agent"], o["judge"]), ("anthropic/claude-haiku-5.5", "openai/gpt-6-luna"))
        g = P.resolve_models(P.load_profile("gemini"))
        self.assertEqual((g["agent"], g["extractor"]), ("gemini-3.7-flash", "gemini-3.7-flash"))
        self.assertEqual(set(a), set(P.ROLES))

    def test_per_role_override_and_unknown_role(self):
        m = P.resolve_models(P.load_profile("anthropic"), {"judge": "claude-opus-5", "agent": None})
        self.assertEqual((m["judge"], m["agent"]), ("claude-opus-5", "claude-haiku-5-5"))
        with self.assertRaises(P.ProviderError):
            P.resolve_models(P.load_profile("anthropic"), {"bogus": "x"})

    def test_custom_profile_needs_models_and_url(self):
        c = P.load_profile("custom", base_url="http://localhost:9/v1", key_env="EVAL_API_KEY",
                           pi_provider="mine")
        self.assertEqual(c.base_url, "http://localhost:9/v1")
        self.assertEqual(c.pi["provider"], "mine")
        m = P.resolve_models(c, {"judge": "j"})
        with self.assertRaises(P.ProviderError):
            P.require_models(m, ("judge", "agent"))
        P.require_models(m, ("judge",))

    def test_cli_args_model_flags(self):
        import argparse
        ap = argparse.ArgumentParser()
        P.add_profile_args(ap)
        a = ap.parse_args(["--profile", "anthropic", "--model-judge", "x", "--model-qa-generator", "q"])
        self.assertEqual(P.role_overrides(a), {"judge": "x", "qa_generator": "q"})


class Sampling(unittest.TestCase):
    def test_anthropic_omits_temperature_and_top_p(self):
        p = P.load_profile("anthropic")
        self.assertEqual(p.sampling_kwargs("claude-haiku-5-5", temperature=0, top_p=0.9), {})
        _, _, body = P.build_request(p, "claude-sonnet-5-5", [{"role": "user", "content": "x"}],
                                     max_tokens=10, temperature=0, key="k")
        self.assertNotIn("temperature", body)

    def test_gemini_allows_openrouter_omits_only_anthropic_family(self):
        g = P.load_profile("gemini")
        self.assertEqual(g.sampling_kwargs("gemini-3.7-flash", temperature=0), {"temperature": 0})
        o = P.load_profile("openrouter")
        self.assertEqual(o.sampling_kwargs("openai/gpt-6-luna", temperature=0), {"temperature": 0})
        self.assertEqual(o.sampling_kwargs("anthropic/claude-haiku-5.5", temperature=0), {})


class Costs(unittest.TestCase):
    def test_haiku_sonnet_prices_and_tier(self):
        a = P.load_profile("anthropic")
        self.assertAlmostEqual(a.cost_usd("claude-haiku-5-5", 50_000, 50_000), (0.10 + 0.50) * 0.05)
        self.assertAlmostEqual(a.cost_usd("claude-sonnet-5-5", 1_000_000, 1_000_000), 2.0 + 10.0)
        # prompt > 100k tokens -> Haiku 5.5 long-context tier 0.50 / 2.50
        self.assertAlmostEqual(a.cost_usd("claude-haiku-5-5", 200_000, 0), 0.2 * 0.50)
        # cache read counts toward the tier and is priced at cache_read
        c = a.cost_usd("claude-haiku-5-5", 10_000, 0, cache_read=100_000)
        self.assertAlmostEqual(c, (10_000 * 0.50 + 100_000 * 0.05) / 1e6)
        self.assertIsNone(a.cost_usd("not-a-model", 1, 1))

    def test_price_status_is_explicit_everywhere(self):
        for n in ("anthropic", "openrouter", "gemini"):
            for m, e in P.load_profile(n).data["prices"].items():
                self.assertIn(e["status"], ("confirmed", "catalog_only", "unverified"), (n, m))
                self.assertTrue(e["source"] and e["as_of"], (n, m))
        self.assertEqual(P.load_profile("gemini").price("gemini-3.1-pro-preview")["status"],
                         "catalog_only")

    def test_lookup_price_scans_profiles(self):
        hit = P.lookup_price("openai/gpt-6-luna")
        self.assertEqual((hit[0].name, hit[1]["in"], hit[1]["out"]), ("openrouter", 0.10, 0.50))
        self.assertIsNone(P.lookup_price("zzz"))


class BudgetAndChat(unittest.TestCase):
    def setUp(self):
        self.env = {"ANTHROPIC_API_KEY": "sk-test", "EVAL_API_KEY": "sk-eval"}

    def test_budget_requires_positive_cap(self):
        for bad in (None, 0, -1):
            with self.assertRaises(P.ProviderError):
                P.Budget(bad)

    def test_chat_anthropic_request_shape_and_cost(self):
        p = P.load_profile("anthropic")
        tr = FakeTransport()
        b = P.Budget(1.0)
        r = P.chat("judge", [{"role": "user", "content": "hi"}], profile=p, budget=b,
                   environ=self.env, transport=tr, temperature=0)
        c = tr.calls[0]
        self.assertEqual(c["url"], "https://api.anthropic.com/v1/chat/completions")
        self.assertEqual(c["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(c["body"]["model"], "claude-sonnet-5-5")
        self.assertNotIn("temperature", c["body"])
        self.assertAlmostEqual(r.cost_usd, (1000 * 2.0 + 100 * 10.0) / 1e6)
        self.assertEqual(r.cost_source, "table")
        self.assertAlmostEqual(b.spent, r.cost_usd)

    def test_chat_openrouter_reported_cost_and_usage_include(self):
        p = P.load_profile("openrouter")
        tr = FakeTransport([(200, {"choices": [{"message": {"content": "x"}}],
                                   "usage": {"prompt_tokens": 5, "completion_tokens": 5, "cost": 0.123}})])
        r = P.chat("judge", [{"role": "user", "content": "hi"}], profile=p, budget=P.Budget(1),
                   environ={"OPENROUTER_API_KEY": "k"}, transport=tr, temperature=0)
        self.assertEqual(tr.calls[0]["body"]["usage"], {"include": True})
        self.assertEqual(tr.calls[0]["body"]["temperature"], 0)      # gpt-6-luna: allowed
        self.assertEqual((r.cost_usd, r.cost_source), (0.123, "reported"))

    def test_chat_gemini_url_and_key_env_override(self):
        p = P.load_profile("gemini")
        tr = FakeTransport()
        P.chat("agent", [{"role": "user", "content": "hi"}], profile=p, key_env="EVAL_API_KEY",
               environ=self.env, transport=tr)
        self.assertEqual(tr.calls[0]["url"],
                         "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
        self.assertEqual(tr.calls[0]["headers"]["Authorization"], "Bearer sk-eval")

    def test_x_api_key_style(self):
        p = P.load_profile("anthropic")
        p.data["auth"]["style"] = "x-api-key"
        _, h, _ = P.build_request(p, "m", [], max_tokens=1, key="K")
        self.assertEqual(h["x-api-key"], "K")
        self.assertNotIn("Authorization", h)

    def test_budget_exhaustion_stops_before_the_call(self):
        p = P.load_profile("anthropic")
        tr = FakeTransport()
        b = P.Budget(0.0011)          # worst case of a 100-token call ~0.00104 passes; the fake reply costs 0.003
        P.chat("judge", [], profile=p, budget=b, max_tokens=100, environ=self.env, transport=tr)
        with self.assertRaises(P.BudgetExceeded):
            P.chat("judge", [], profile=p, budget=b, max_tokens=100, environ=self.env, transport=tr)
        self.assertEqual(len(tr.calls), 1)

    def test_unpriced_model_cannot_be_budgeted(self):
        p = P.load_profile("anthropic")
        with self.assertRaises(P.UnpricedModel):
            P.chat("judge", [], profile=p, model="mystery-1", budget=P.Budget(1),
                   environ=self.env, transport=FakeTransport())

    def test_missing_key_names_the_variable_not_a_value(self):
        p = P.load_profile("anthropic")
        with self.assertRaises(P.ProviderError) as cm:
            P.chat("judge", [], profile=p, environ={}, transport=FakeTransport())
        self.assertIn("ANTHROPIC_API_KEY", str(cm.exception))

    def test_retry_on_429_then_ok(self):
        p = P.load_profile("anthropic")
        tr = FakeTransport([(429, {"error": {"message": "slow"}}), (200, {
            "choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1,
                                                                    "completion_tokens": 1}})])
        r = P.chat("judge", [], profile=p, environ=self.env, transport=tr, sleep=lambda s: None)
        self.assertEqual((r.text, len(tr.calls)), ("ok", 2))

    def test_pi_env_maps_override_key_to_native_var(self):
        p = P.load_profile("gemini")
        self.assertEqual(P.pi_env(p, "EVAL_API_KEY", {"EVAL_API_KEY": "v"}), {"GEMINI_API_KEY": "v"})
        self.assertEqual(P.pi_env(p, None, {"GEMINI_API_KEY": "v"}), {})


class PiSupport(unittest.TestCase):
    def test_pi_flags(self):
        self.assertEqual(P.pi_flags(P.load_profile("anthropic"), "claude-haiku-5-5"),
                         ["--provider", "anthropic", "--model", "claude-haiku-5-5"])
        self.assertEqual(P.pi_flags(P.load_profile("gemini"), "gemini-3.7-flash"),
                         ["--provider", "google", "--model", "gemini-3.7-flash"])
        self.assertEqual(P.pi_flags(P.load_profile("openrouter"), "anthropic/claude-haiku-5.5"),
                         ["--model", "openrouter/anthropic/claude-haiku-5.5", "--provider", "openrouter"])

    def test_runner_table_from_profiles(self):
        t = rc.PROVIDERS
        self.assertEqual(t["anthropic"]["key_env"], "ANTHROPIC_API_KEY")
        self.assertEqual((t["gemini"]["pi_flags"], t["gemini"]["key_env"]), ("google", "GEMINI_API_KEY"))
        self.assertEqual(t["openrouter"]["ctx_window"], 32768)
        a = rc.pi_args("s", "/d", "gemini-3.7-flash", [], "P", provider="gemini")
        self.assertEqual(a[a.index("--provider"):a.index("--provider") + 4],
                         ["--provider", "google", "--model", "gemini-3.7-flash"])


class HarsEnv(unittest.TestCase):
    def test_anthropic_exports(self):
        txt = "\n".join(P.hars_env(P.load_profile("anthropic")))
        for line in ("export HARS_MEMORY_EXTRACTOR_PROVIDER=local",
                     "export HARS_MEMORY_EXTRACTOR_BASE_URL=https://api.anthropic.com/v1",
                     "export HARS_MEMORY_EXTRACTOR_MODEL=claude-haiku-5-5",
                     "export HARS_MEMORY_QUERY_BASE_URL=https://api.anthropic.com/v1",
                     "export HARS_MEMORY_QUERY_MODEL=claude-haiku-5-5",
                     'export HARS_MEMORY_LLM_API_KEY="$ANTHROPIC_API_KEY"',
                     "export HARS_MEMORY_LLM_OMIT_SAMPLING_PARAMS=1"):
            self.assertIn(line, txt)
        self.assertNotIn("sk-", txt)

    def test_openrouter_uses_server_openrouter_provider(self):
        txt = "\n".join(P.hars_env(P.load_profile("openrouter"), {"extractor": "openai/gpt-6-luna",
                                                                    "keyword": "openai/gpt-6-luna"}))
        self.assertIn("export HARS_MEMORY_EXTRACTOR_PROVIDER=openrouter", txt)
        self.assertIn("export HARS_MEMORY_OPENROUTER_MODEL=openai/gpt-6-luna", txt)
        self.assertIn("export HARS_MEMORY_EXTRACTOR_API_KEY_ENV=OPENROUTER_API_KEY", txt)
        self.assertNotIn("OMIT_SAMPLING_PARAMS=1", txt)     # non-Anthropic models

    def test_gemini_and_key_env_override(self):
        txt = "\n".join(P.hars_env(P.load_profile("gemini"), key_env="EVAL_API_KEY"))
        self.assertIn("https://generativelanguage.googleapis.com/v1beta/openai", txt)
        self.assertIn('HARS_MEMORY_LLM_API_KEY="$EVAL_API_KEY"', txt)
        self.assertNotIn("OMIT_SAMPLING_PARAMS=1", txt)

    def test_cli_prints_and_runs_offline(self):
        out = subprocess.run([sys.executable, os.path.join(SCRIPTS, "providers.py"), "hars-env",
                              "--profile", "anthropic"], capture_output=True, text=True, check=True).stdout
        self.assertIn("HARS_MEMORY_LLM_OMIT_SAMPLING_PARAMS=1", out)


class FamilyCheck(unittest.TestCase):
    def test_default_refuses_same_family(self):
        v = P.judge_family_verdict(["claude-haiku-5-5"], "claude-sonnet-5-5", False)
        self.assertFalse(v["ok"])
        self.assertIn("--allow-same-family-judge", v["error"])

    def test_cross_family_ok(self):
        v = P.judge_family_verdict(["claude-haiku-5-5"], "openai/gpt-6-luna", False)
        self.assertTrue(v["ok"])
        self.assertFalse(v["judge_same_family"])

    def test_flag_allows_different_model_with_warning_and_reason(self):
        v = P.judge_family_verdict(["claude-haiku-5-5"], "claude-sonnet-5-5", True)
        self.assertTrue(v["ok"] and v["judge_same_family"])
        self.assertIn("LOUD WARNING", v["warning"])
        self.assertIn("claude-sonnet-5-5", v["reason"])

    def test_flag_still_refuses_same_model_even_across_routers(self):
        for judge in ("claude-haiku-5-5", "anthropic/claude-haiku-5.5"):
            v = P.judge_family_verdict(["claude-haiku-5-5"], judge, True)
            self.assertFalse(v["ok"], judge)
            self.assertIn("judge model != agent model", v["error"])

    def test_every_agent_side_model_is_checked(self):
        v = P.judge_family_verdict(["claude-haiku-5-5", "claude-sonnet-5-5"], "claude-sonnet-5-5", True)
        self.assertFalse(v["ok"])

    def test_runner_wrappers_still_exported(self):
        self.assertIsNotNone(rc.family_conflict("claude-haiku-5-5", "claude-sonnet-5-5"))


def _judge_records():
    base = dict(chain_id="c1", session_id="s", span_idx=1, lag=1, slice="HEAD", answerable=True,
                question="Which port?", gold_anchor="8842", canonical_short_answer="8842",
                evidence="port 8842", span_text="the beacon port 8842 was configured",
                answer="Port 8842.", agent_model="claude-haiku-5-5", qa_model="claude-haiku-5-5")
    return [dict(base, qid="q1", arm="baseline"), dict(base, qid="q1", arm="nc-graph", rid="q1::nc-graph::b1536")]


class JudgeCli(unittest.TestCase):
    def _run(self, argv, transport=None, env=None):
        td = tempfile.mkdtemp()
        inp = os.path.join(td, "in.jsonl")
        with open(inp, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in _judge_records()))
        out, mp = os.path.join(td, "j.jsonl"), os.path.join(td, "map.json")
        old = (sys.argv, judge_v2.TRANSPORT, dict(os.environ))
        sys.argv = ["judge_v2.py", "--in", inp, "--out", out, "--mapping", mp, *argv]
        judge_v2.TRANSPORT = transport
        os.environ.update(env or {"ANTHROPIC_API_KEY": "k"})
        buf, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                try:
                    rcode = judge_v2.main()
                except SystemExit as e:
                    rcode = str(e)
        finally:
            sys.argv, judge_v2.TRANSPORT = old[0], old[1]
            os.environ.clear()
            os.environ.update(old[2])
        mapping = json.load(open(mp)) if os.path.exists(mp) else None
        judged = [json.loads(x) for x in open(out)] if os.path.exists(out) else []
        return rcode, buf.getvalue() + err.getvalue(), mapping, judged

    def test_same_family_judge_refused_by_default(self):
        rcode, log, mp, _ = self._run(["--profile", "anthropic", "--budget-usd", "1"], FakeTransport())
        self.assertIn("REFUSING", str(rcode))
        self.assertIsNone(mp)

    def test_flag_runs_and_records_limitation(self):
        tr = FakeTransport()
        rcode, log, mp, judged = self._run(["--profile", "anthropic", "--allow-same-family-judge",
                                            "--budget-usd", "1"], tr)
        self.assertEqual(rcode, 0, log)
        self.assertTrue(mp["judge_same_family"])
        self.assertIn("same model family", mp["judge_same_family_reason"])
        self.assertEqual((mp["profile"], mp["judge_model"]), ("anthropic", "claude-sonnet-5-5"))
        self.assertTrue(all(j["judge_same_family"] for j in judged))
        self.assertIn("LOUD WARNING", log)
        self.assertEqual({c["body"]["model"] for c in tr.calls}, {"claude-sonnet-5-5"})
        self.assertTrue(all("temperature" not in c["body"] for c in tr.calls))

    def test_flag_with_same_model_still_refused(self):
        rcode, log, mp, _ = self._run(["--profile", "anthropic", "--model-judge", "claude-haiku-5-5",
                                       "--allow-same-family-judge", "--budget-usd", "1"], FakeTransport())
        self.assertIn("judge model != agent model", str(rcode))

    def test_budget_is_mandatory_and_hard(self):
        rcode, _, _, _ = self._run(["--profile", "anthropic", "--allow-same-family-judge"], FakeTransport())
        self.assertIn("budget-usd", str(rcode))
        # a cap below the worst case of the very first call refuses it BEFORE the wire (pre-flight)
        tr = FakeTransport()
        rcode, log, _, _ = self._run(["--profile", "anthropic", "--allow-same-family-judge",
                                      "--budget-usd", "0.0001"], tr)
        self.assertIn("refusing", str(rcode))
        self.assertEqual(len(tr.calls), 0)

    def test_openrouter_cross_family_needs_no_flag(self):
        tr = FakeTransport()
        rcode, log, mp, _ = self._run(["--profile", "openrouter", "--budget-usd", "1"], tr,
                                      env={"OPENROUTER_API_KEY": "k"})
        self.assertEqual(rcode, 0, log)
        self.assertFalse(mp["judge_same_family"])
        self.assertEqual(mp["judge_model"], "openai/gpt-6-luna")
        self.assertEqual(tr.calls[0]["url"], "https://openrouter.ai/api/v1/chat/completions")


class RunnerWiring(unittest.TestCase):
    def _dry(self, *a):
        chains = os.path.join(tempfile.mkdtemp(), "c.jsonl")
        sys.path.insert(0, os.path.join(SCRIPTS, "tests"))
        import test_amendment_c as tac
        with open(chains, "w") as f:
            f.write(json.dumps(tac._chain("chain_S0_c0", "S0")) + "\n")
        return subprocess.run([sys.executable, os.path.join(SCRIPTS, "run_chainreplay.py"), "--dry-run",
                               "--chains", chains, *a], capture_output=True, text=True)

    def test_profile_anthropic_refuses_then_allows(self):
        bad = self._dry("--profile", "anthropic")
        self.assertEqual(bad.returncode, 2)
        self.assertIn("same family", bad.stderr)
        ok = self._dry("--profile", "anthropic", "--allow-same-family-judge", "--budget-usd", "5")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("LOUD WARNING", ok.stderr)
        self.assertIn("judge claude-sonnet-5-5", ok.stdout)
        self.assertIn("same-family judge: True", ok.stdout)
        self.assertIn("BUDGET CHECK", ok.stdout)

    def test_profile_openrouter_and_gemini_defaults(self):
        o = self._dry("--profile", "openrouter")
        self.assertEqual(o.returncode, 0, o.stderr)
        self.assertIn("anthropic/claude-haiku-5.5", o.stdout)
        self.assertIn("judge openai/gpt-6-luna", o.stdout)
        g = self._dry("--profile", "gemini", "--allow-same-family-judge")
        self.assertEqual(g.returncode, 0, g.stderr)
        self.assertIn("pi --provider gemini --model gemini-3.7-flash", g.stdout)
        self.assertIn("key env GEMINI_API_KEY", g.stdout)

    def test_defaults_are_the_anthropic_profile_haiku_5_5(self):
        """No flag at all = the anthropic profile: agent/extractor/keyword claude-haiku-5-5 (2026-10-26)."""
        d = self._dry("--allow-same-family-judge")
        self.assertEqual(d.returncode, 0, d.stderr)
        self.assertIn("agent model: claude-haiku-5-5 | sum model: claude-haiku-5-5 | qa model: claude-haiku-5-5", d.stdout)
        self.assertIn("judge claude-sonnet-5-5 | extractor claude-haiku-5-5 | keyword claude-haiku-5-5", d.stdout)
        self.assertNotIn("qwen", d.stdout)

    def test_explicit_agent_model_still_wins(self):
        d = self._dry("--profile", "openrouter", "--agent-model", "qwen/qwen3.8-flash")
        self.assertEqual(d.returncode, 0, d.stderr)
        self.assertIn("agent model: qwen/qwen3.8-flash", d.stdout)

    def test_per_role_override(self):
        d = self._dry("--profile", "openrouter", "--model-judge", "google/gemini-3.7-flash",
                      "--model-agent", "qwen/qwen3.8-flash")
        self.assertEqual(d.returncode, 0, d.stderr)
        self.assertIn("agent model: qwen/qwen3.8-flash", d.stdout)
        self.assertIn("judge google/gemini-3.7-flash", d.stdout)

    def test_live_run_requires_budget(self):
        chains = os.path.join(tempfile.mkdtemp(), "c.jsonl")
        sys.path.insert(0, os.path.join(SCRIPTS, "tests"))
        import test_amendment_c as tac
        with open(chains, "w") as f:
            f.write(json.dumps(tac._chain("chain_S0_c0", "S0")) + "\n")
        p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "run_chainreplay.py"),
                            "--chains", chains, "--profile", "openrouter",
                            "--out", os.path.join(tempfile.mkdtemp(), "o")],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("--budget-usd", p.stderr)

    def test_pi_spend_charged_per_turn_and_stops(self):
        saved = (rc.ACTIVE_PROFILE, rc.BUDGET, rc.STOP_RUN)
        try:
            rc.ACTIVE_PROFILE = P.load_profile("anthropic")
            rc.BUDGET = P.Budget(0.01)
            rc.STOP_RUN = False
            ev = json.dumps({"type": "message_end", "message": {"role": "assistant", "content": "x",
                             "usage": {"input": 5000, "output": 1000, "cacheRead": 0, "cacheWrite": 0}}})
            rc.charge_pi_spend(["--provider", "anthropic", "--model", "claude-haiku-5-5"], ev)
            self.assertAlmostEqual(rc.BUDGET.spent, (5000 * 0.10 + 1000 * 0.50) / 1e6)
            self.assertFalse(rc.STOP_RUN)
            for _ in range(2):
                rc.charge_pi_spend(["--provider", "anthropic", "--model", "claude-sonnet-5-5"], ev)
            self.assertTrue(rc.STOP_RUN)
            with self.assertRaises(rc.ChainFailure):
                rc.run_pi_turn(["-p"], {}, 1)
        finally:
            rc.ACTIVE_PROFILE, rc.BUDGET, rc.STOP_RUN = saved

    def test_embedder_info_precedence_and_flags(self):
        ok = rc.embedder_info("m", None, {}, mcp_fn=lambda *a, **k: {
            "configured_models": {"embedder": "BAAI/bge-m3 (CPU)"}})
        self.assertEqual((ok["id"], ok["flags"]), ("BAAI/bge-m3 (CPU)", []))
        self.assertIn("server", ok["source"])
        env = rc.embedder_info("m", "declared-1", {"HARS_MEMORY_EMBED_MODEL": "e"},
                               mcp_fn=lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
        self.assertEqual((env["id"], env["source"]), ("declared-1", "--embedder-id"))
        envonly = rc.embedder_info("m", None, {"HARS_MEMORY_EMBED_MODEL": "e"},
                                   mcp_fn=lambda *a, **k: {})
        self.assertEqual(envonly["id"], "e")
        self.assertIn("NOT verified", envonly["source"])
        shim = rc.embedder_info("m", None, {}, mcp_fn=lambda *a, **k: {
            "configured_models": {"embedder": "hash-bow-shim (CPU)"}})
        self.assertTrue(shim["flags"])
        unk = rc.embedder_info("m", None, {}, mcp_fn=lambda *a, **k: {})
        self.assertIsNone(unk["id"])
        self.assertIn("EMBEDDER UNKNOWN", unk["flags"][0])


class MetricsHeader(unittest.TestCase):
    def _meta(self, **kw):
        m = {"records": 1, "chains": 1, "sessions": 1, "questions": 1, "seed": 1, "boot_n": 1,
             "cluster": "session", "mem_ratio_pair": ("nc-graph", "baseline"), "primary_b": 1536,
             "budgets": [1536], "treatment": "nc-graph", "legacy": False}
        m.update(kw)
        return m

    def _render(self, meta):
        return "\n".join(metrics_v2.judge_limitation_lines(meta)) + "\nanalysis cell"

    def test_same_family_limitation_in_header(self):
        md = self._render(self._meta(judge_same_family=True, judge_model="claude-sonnet-5-5",
                                     judge_same_family_reason="same family anthropic"))
        self.assertIn("LIMITATION - SAME-FAMILY JUDGE", md)
        self.assertIn("claude-sonnet-5-5", md)
        self.assertLess(md.index("LIMITATION"), md.index("analysis cell"))

    def test_cross_family_has_no_limitation(self):
        md = self._render(self._meta(judge_same_family=False, judge_model="openai/gpt-6-luna"))
        self.assertNotIn("SAME-FAMILY", md)
        self.assertNotIn("judge independence not recorded", md)

    def test_unrecorded_is_flagged(self):
        md = self._render(self._meta(judge_same_family=None, judge_model=None))
        self.assertIn("judge independence not recorded", md)


if __name__ == "__main__":
    unittest.main()
