"""Offline tests for run_chainreplay.py: --provider {openrouter,anthropic}, degraded-recall
gating, preflight verdict and the loopback proxy-bypass opener. No network (the HTTP test
talks to a throwaway 127.0.0.1 server), no API keys, $0.

Run:  uvx --from pytest pytest -q experiments/neurocompact-eval/scripts/tests/test_chainreplay_provider.py
  or: python3 experiments/neurocompact-eval/scripts/tests/test_chainreplay_provider.py
"""
from __future__ import annotations

import http.server
import json
import os
import sys
import tempfile
import threading
import types
import unittest
import urllib.error
import urllib.request
from email.message import Message

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
sys.dont_write_bytecode = True
import run_chainreplay as rc  # noqa: E402


def _read(path):
    with open(path) as f:
        return f.read()


def _args(**kw):
    base = dict(mem_budget=1536, recall_cap_min=256, recall_cap_max=1024,
                baseline_sum_max_tokens=1536, note_sum_max_tokens=640,
                agent_model="claude-haiku-5-5", provider="anthropic",
                judge_model="openai/gpt-6-luna", allow_degraded=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


_COMPS = {"baseline": [{"span_idx": 3, "realized_tokens": 1500, "cap": 1536}],
          "nc": [{"span_idx": 3, "realized_tokens": 600, "cap": 640}]}


class ProviderConstruction(unittest.TestCase):
    def setUp(self):
        self._prov = (rc.PROVIDER, rc.CTX_WINDOW_OVERRIDE, rc.AGENT_THINKING)

    def tearDown(self):
        rc.PROVIDER, rc.CTX_WINDOW_OVERRIDE, rc.AGENT_THINKING = self._prov

    def test_pi_args_openrouter_is_the_legacy_shape(self):
        a = rc.pi_args("sid", "/d", "qwen/qwen3.8-flash", ["--no-mcp", "-nt"], "PROMPT", provider="openrouter")
        self.assertEqual(a, ["-p", "--mode", "json", "--session-id", "sid", "--session-dir", "/d",
                             "--model", "openrouter/qwen/qwen3.8-flash", "--provider", "openrouter",
                             "--no-mcp", "-nt", "PROMPT"])

    def test_pi_args_anthropic(self):
        a = rc.pi_args("sid", "/d", "claude-haiku-5-5", ["--no-mcp", "-nt"], "P", provider="anthropic")
        i = a.index("--provider")
        self.assertEqual(a[i:i + 4], ["--provider", "anthropic", "--model", "claude-haiku-5-5"])
        self.assertNotIn("openrouter", " ".join(a))
        self.assertEqual(a[-1], "P")

    def test_pi_args_follows_active_provider_and_thinking(self):
        rc.PROVIDER = "anthropic"
        rc.AGENT_THINKING = "off"
        a = rc.pi_args("s", "/d", "claude-haiku-5-5", [], "P")
        self.assertIn("anthropic", a)
        self.assertEqual(a[a.index("--thinking") + 1], "off")
        rc.PROVIDER, rc.AGENT_THINKING = "openrouter", None
        self.assertNotIn("--thinking", rc.pi_args("s", "/d", "m", [], "P"))

    def test_base_env_providers(self):
        rc.CAPS.update({"baseline": 1536, "nc": 640, "nc-note-only": 640})
        e_or = rc.base_env("nc", "qwen/qwen3.8-flash", "s", "http://x/mcp", None, provider="openrouter")
        self.assertEqual((e_or["NC_PROVIDER"], e_or["NC_SUM_PROVIDER"]), ("openrouter", "openrouter"))
        e_an = rc.base_env("nc", "claude-haiku-5-5", "s", "http://x/mcp", None, provider="anthropic")
        self.assertEqual((e_an["NC_PROVIDER"], e_an["NC_SUM_PROVIDER"]), ("anthropic", "anthropic"))
        self.assertEqual(e_an["NC_MODEL_ID"], "claude-haiku-5-5")
        self.assertEqual(e_an["NC_SUM_MODEL_ID"], "claude-haiku-5-5")
        self.assertEqual(e_an["NC_SUM_REASONING_EFFORT"], "off")
        # everything protocol-relevant is identical across providers
        for k in ("NC_SUM_MAX_TOKENS", "NC_SUM_SECTION_TRIM", "NC_PERSIST", "NC_INJECT_RECALL",
                  "NC_FORCE_COMPACT_THRESHOLD"):
            self.assertEqual(e_or[k], e_an[k], k)

    def test_glm_minimal_reasoning_only_on_openrouter(self):
        self.assertEqual(rc.sum_reasoning_default("z-ai/glm-5.3-flash", "openrouter"), "minimal")
        self.assertEqual(rc.sum_reasoning_default("qwen/qwen3.8-flash", "openrouter"), "off")
        self.assertEqual(rc.sum_reasoning_default("claude-haiku-5-5", "anthropic"), "off")

    def test_ctx_window_and_reserve_keep_the_native_line(self):
        self.assertEqual(rc.reserve_tokens_for("openrouter"), rc.RESERVE_TOKENS)   # 2000, unchanged
        self.assertEqual(rc.ctx_window_for("anthropic"), 200000)
        for prov in ("openrouter", "anthropic"):
            self.assertEqual(rc.ctx_window_for(prov) - rc.reserve_tokens_for(prov), rc.NATIVE_TRIGGER)
        self.assertEqual(rc.ctx_window_for("anthropic", 144000), 144000)
        self.assertEqual(rc.ctx_window_for("anthropic", 144000) - rc.reserve_tokens_for("anthropic", 144000),
                         rc.NATIVE_TRIGGER)

    def test_budget_logic_is_provider_independent(self):
        # B=1536 / recall-cap arithmetic never looks at the provider
        r1 = rc.recall_cap_for(600, _args(provider="anthropic"))
        r2 = rc.recall_cap_for(600, _args(provider="openrouter"))
        self.assertEqual(r1["recall_cap"], r2["recall_cap"])
        self.assertEqual(r1["recall_cap"], 936)

    def test_key_env_problem_never_returns_the_value(self):
        self.assertIn("ANTHROPIC_API_KEY", rc.key_env_problem("anthropic", {}))
        self.assertIn("ANTHROPIC_API_KEY", rc.key_env_problem("anthropic", {"ANTHROPIC_API_KEY": "  "}))
        self.assertIsNone(rc.key_env_problem("anthropic", {"ANTHROPIC_API_KEY": "sk-ant-SECRET"}))
        self.assertNotIn("SECRET", rc.key_env_problem("anthropic", {"ANTHROPIC_API_KEY": ""}))
        self.assertIn("OPENROUTER_API_KEY", rc.key_env_problem("openrouter", {}))

    def test_models_json_override_for_base_url(self):
        self.assertIsNone(rc.models_json_override({}, None))
        self.assertIsNone(rc.models_json_override({}, "  "))
        existing = {"providers": {"openrouter": {"modelOverrides": {"x": {}}}}}
        out = rc.models_json_override(existing, "https://proxy.example/anthropic")
        self.assertEqual(out["providers"]["anthropic"]["baseUrl"], "https://proxy.example/anthropic")
        self.assertIn("openrouter", out["providers"])
        self.assertNotIn("anthropic", existing["providers"])          # input not mutated

    def test_write_and_restore_models_json(self):
        with tempfile.TemporaryDirectory() as d:
            old = rc.MODELS_JSON
            rc.MODELS_JSON = os.path.join(d, "models.json")
            try:
                self.assertIs(rc.write_models_base_url("openrouter", {"ANTHROPIC_BASE_URL": "http://p"}), False)
                self.assertIs(rc.write_models_base_url("anthropic", {}), False)
                tok = rc.write_models_base_url("anthropic", {"ANTHROPIC_BASE_URL": "http://p"})
                self.assertEqual(_read(rc.MODELS_JSON) and json.loads(_read(rc.MODELS_JSON))["providers"]["anthropic"]["baseUrl"], "http://p")
                rc.restore_models_json(tok)
                self.assertFalse(os.path.exists(rc.MODELS_JSON))       # did not exist before
                with open(rc.MODELS_JSON, "w") as f:
                    f.write('{"providers": {"a": {}}}')
                tok = rc.write_models_base_url("anthropic", {"ANTHROPIC_BASE_URL": "http://p"})
                rc.restore_models_json(tok)
                self.assertEqual(_read(rc.MODELS_JSON), '{"providers": {"a": {}}}')
            finally:
                rc.MODELS_JSON = old


class FamilyCheck(unittest.TestCase):
    def test_families(self):
        f = rc.model_family
        self.assertEqual(f("claude-haiku-5-5"), "anthropic")
        self.assertEqual(f("anthropic/claude-sonnet-4"), "anthropic")
        self.assertEqual(f("openai/gpt-6-luna"), "openai")
        self.assertEqual(f("qwen/qwen3.8-flash"), "qwen")
        self.assertEqual(f("z-ai/glm-5.3-flash"), "zai")
        self.assertIsNone(f("some/unknown-model"))
        self.assertIsNone(f(""))

    def test_conflict(self):
        self.assertIsNone(rc.family_conflict("claude-haiku-5-5", "openai/gpt-6-luna"))
        self.assertIsNone(rc.family_conflict("qwen/qwen3.8-flash", "openai/gpt-6-luna"))
        self.assertIsNotNone(rc.family_conflict("gpt-5", "openai/gpt-6-luna"))
        self.assertIsNotNone(rc.family_conflict("anthropic/claude-haiku-5-5", "claude-opus-4"))
        self.assertIsNone(rc.family_conflict("mystery-model", "openai/gpt-6-luna"))   # unknown -> no refusal
        self.assertIsNone(rc.family_conflict("claude-haiku-5-5", None))


def _sess(**kw):
    return {"session_id": "s", "hits": 3, **kw}


_PACKED = {"units_kept": 3, "relations_kept": 0, "entities_kept": 0}


class DegradedGating(unittest.TestCase):
    def test_ok_payload_is_clean(self):
        p = {"ok": True, "context": "text", "session": _sess(status="ok")}
        self.assertEqual(rc.recall_failure_reasons(p, _PACKED), [])
        self.assertIsNone(rc.recall_degraded(p))

    def test_hard_failures_unchanged(self):
        for sess in (_sess(status="error", error="boom", error_kind="channel"),
                     _sess(status="unknown_session", error="unknown session")):
            why = rc.recall_failure_reasons({"ok": True, "context": "text", "session": sess}, _PACKED)
            self.assertTrue(any("status=" in w for w in why), why)
        for kind in ("session_channel", "recall_exception"):
            why = rc.recall_failure_reasons({"ok": True, "context": "text", "error_kind": kind}, _PACKED)
            self.assertIn(f"error_kind={kind}", why)
        self.assertTrue(rc.recall_failure_reasons({"context": "", "error": "HTTPError: 530"}, _PACKED))
        # no_hits kind by itself is not a failure reason (empty raw is, separately)
        self.assertEqual(rc.recall_failure_reasons({"ok": True, "context": "x", "error_kind": "no_hits"}, _PACKED), [])

    def test_degraded_is_not_a_failure_but_is_recorded(self):
        p = {"ok": True, "context": "lexical text", "session": _sess(
            status="degraded", degraded="lexical_only", error_kind="embedding",
            error="query embedding failed (TimeoutError): timed out")}
        self.assertEqual(rc.recall_failure_reasons(p, _PACKED), [])     # the embedding error text is NOT a dead channel
        self.assertEqual(rc.recall_degraded(p), "lexical_only")
        self.assertEqual(rc.recall_degraded({"session": {"status": "degraded"}}), "degraded")

    def test_vectors_indexed_zero_still_fails(self):
        p = {"ok": True, "context": "t", "session": _sess(status="ok", vectors_indexed=0)}
        self.assertTrue(rc.recall_failure_reasons(p, _PACKED))

    def test_chain_invalid_on_degraded_unless_allowed(self):
        deg = {"q1": "lexical_only", "q2": None, "q3": None}
        v = rc.chain_budget("c", _args(), _COMPS, [900], True, {"q1": [], "q2": [], "q3": []}, deg)
        self.assertFalse(v["valid"])
        self.assertTrue(any(r.startswith("RECALL DEGRADED 1/3") for r in v["invalid_reasons"]), v["invalid_reasons"])
        self.assertEqual(v["recall_degraded_qids"], ["q1"])
        v2 = rc.chain_budget("c", _args(allow_degraded=True), _COMPS, [900], True,
                             {"q1": [], "q2": [], "q3": []}, deg)
        self.assertTrue(v2["valid"])
        self.assertEqual(v2["recall_degraded_n"], 1)
        self.assertTrue(v2["allow_degraded"])
        self.assertTrue(any("RECALL DEGRADED" in w for w in v2["warnings"]))
        v3 = rc.chain_budget("c", _args(), _COMPS, [900], True, {"q1": []}, {"q1": None})
        self.assertTrue(v3["valid"])
        self.assertEqual(v3["recall_degraded_n"], 0)

    def test_chain_budget_row_records_model_and_provider(self):
        v = rc.chain_budget("c", _args(), _COMPS, [900], True)
        self.assertEqual((v["agent_model"], v["agent_provider"]), ("claude-haiku-5-5", "anthropic"))
        self.assertEqual(v["judge_model"], "openai/gpt-6-luna")

    def test_recall_fail_still_invalidates_alongside(self):
        v = rc.chain_budget("c", _args(), _COMPS, [900], True, {"q1": ["raw recall empty (0 chars)"]}, {"q1": "lexical_only"})
        self.assertTrue(any(r.startswith("RECALL FAIL") for r in v["invalid_reasons"]))
        self.assertTrue(any(r.startswith("RECALL DEGRADED") for r in v["invalid_reasons"]))


class Preflight(unittest.TestCase):
    OK = {"ok": True, "context": "ctx", "session": _sess(status="ok")}

    def test_ok(self):
        self.assertEqual(rc.preflight_verdict(self.OK, "ses_x"), ([], []))

    def test_throwaway_probe_session_is_tolerated(self):
        pf = {"ok": True, "context": "", "error": "no context retrieved for this question; session channel: unknown session",
              "error_kind": "no_hits", "session": {"status": "unknown_session", "error": "unknown session"}}
        self.assertEqual(rc.preflight_verdict(pf, None)[0], [])
        self.assertTrue(rc.preflight_verdict(pf, "ses_real")[0])        # real session: unknown is fatal

    def test_dead_channel_fails(self):
        self.assertTrue(rc.preflight_verdict({"context": "", "error": "HTTPError: 530"}, None)[0])
        self.assertTrue(rc.preflight_verdict({"ok": False, "context": "x", "error_kind": "recall_exception"}, None)[0])
        self.assertTrue(rc.preflight_verdict({"ok": True, "context": "x", "session": _sess(status="error", error="e")}, None)[0])

    def test_degraded(self):
        pf = {"ok": True, "context": "x", "session": _sess(status="degraded", degraded="lexical_only", error="emb")}
        fails, warns = rc.preflight_verdict(pf, "ses_x")
        self.assertTrue(any("degraded" in f for f in fails))
        fails, warns = rc.preflight_verdict(pf, "ses_x", allow_degraded=True)
        self.assertEqual(fails, [])
        self.assertTrue(any("degraded" in w for w in warns))

    def test_status_gate_m4(self):
        st = {"compactions": 3, "entities": 12, "vectors_indexed": 9}
        self.assertEqual(rc.status_gate_problems(st), ([], []))
        f, w = rc.status_gate_problems({"compactions": 3, "entities": 12})          # absent -> warning only
        self.assertEqual(f, [])
        self.assertTrue(any("vectors_indexed" in x for x in w))
        f, _ = rc.status_gate_problems({"compactions": 3, "entities": 12, "vectors_indexed": 0})   # present and 0 -> fail
        self.assertTrue(any("vectors_indexed" in x for x in f))
        f, _ = rc.status_gate_problems({"compactions": 3, "entities": 0, "vectors_indexed": 5})
        self.assertTrue(any("entities" in x for x in f))
        f, _ = rc.status_gate_problems({"compactions": -1, "unreachable": True, "error": "x"})
        self.assertTrue(f)
        f, w = rc.status_gate_problems({"compactions": 0})
        self.assertEqual(f, [])
        fails, _ = rc.preflight_verdict(self.OK, "ses_x", status={"compactions": 2, "entities": 0, "vectors_indexed": 4})
        self.assertTrue(fails)


class LoopbackOpener(unittest.TestCase):
    def test_loopback_hosts_get_an_empty_proxy_opener(self):
        saved = {k: os.environ.get(k) for k in ("http_proxy", "HTTP_PROXY")}
        saved_op = rc._NO_PROXY_OPENER
        try:
            os.environ["http_proxy"] = os.environ["HTTP_PROXY"] = "http://127.0.0.1:1"
            rc._NO_PROXY_OPENER = None
            # control: a default opener under this env DOES carry a proxy handler
            ctrl = urllib.request.build_opener()
            self.assertTrue(any(isinstance(h, urllib.request.ProxyHandler) for h in ctrl.handlers))
            for url in ("http://127.0.0.1:8307/mcp", "http://localhost:8307/mcp", "http://[::1]:8307/mcp"):
                op = rc.opener_for(url)
                self.assertIsNotNone(op, url)
                # ProxyHandler({}) registers no proxy_open methods, so no proxy handler at all
                self.assertFalse(any(isinstance(h, urllib.request.ProxyHandler) for h in op.handlers), url)
            self.assertIs(rc.opener_for("http://127.0.0.1/x"), rc.opener_for("http://localhost/y"))   # cached
        finally:
            rc._NO_PROXY_OPENER = saved_op
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_non_loopback_is_untouched(self):
        for url in ("http://example.com/mcp", "https://hars.internal:8307/mcp", "http://127.0.0.2/mcp",
                    "http://10.0.0.5:8307/mcp", "not a url", ""):
            self.assertIsNone(rc.opener_for(url), url)

    def test_http_error_text_has_status_server_via(self):
        h = Message()
        h["Server"] = "cloudflare"
        h["Via"] = "1.1 proxy"
        e = urllib.error.HTTPError("http://127.0.0.1:8307/mcp", 530, "<none>", h, None)
        txt = rc.describe_exception(e)
        self.assertTrue(txt.startswith("HTTPError: HTTP Error 530"))
        self.assertIn("status=530", txt)
        self.assertIn("server=cloudflare", txt)
        self.assertIn("via=1.1 proxy", txt)
        e2 = urllib.error.HTTPError("u", 502, "Bad Gateway", Message(), None)
        self.assertIn("server=- via=-", rc.describe_exception(e2))
        self.assertEqual(rc.describe_exception(ValueError("x")), "ValueError: x")

    def test_loopback_request_bypasses_a_dead_proxy_and_records_http_error(self):
        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response_only(530)      # no auto Server/Date headers
                self.send_header("Server", "stub-edge")
                self.send_header("Via", "1.1 stub")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        saved = {k: os.environ.get(k) for k in ("http_proxy", "HTTP_PROXY", "no_proxy", "NO_PROXY")}
        try:
            os.environ["http_proxy"] = os.environ["HTTP_PROXY"] = "http://127.0.0.1:1"   # dead proxy
            os.environ.pop("no_proxy", None)
            os.environ.pop("NO_PROXY", None)
            out = rc.hars_recall(f"http://127.0.0.1:{srv.server_address[1]}/mcp", "s", "q", timeout=5)
        finally:
            srv.shutdown()
            srv.server_close()
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        # reached the stub (HTTP 530 from it) instead of failing to connect to the proxy
        self.assertIn("HTTP Error 530", out["error"], out)
        self.assertIn("server=stub-edge", out["error"])
        self.assertIn("via=1.1 stub", out["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
