"""Offline tests for Protocol Amendment C.5 (parity and all budgets in the answering model's NATIVE tokens).

A mock native counter whose ratio differs from the fast estimator (~1.5x, with a digit-heavy surcharge, plus an
8-token request framing overhead like the real count_tokens endpoint) stands in for the API. No network, no key.

Run:  cd experiments/neurocompact-eval/scripts && python -m pytest tests -q
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from unittest import mock

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
sys.dont_write_bytecode = True
os.environ.setdefault("NC_PACK_QUIET", "1")
import run_chainreplay as rc  # noqa: E402
import tokcount  # noqa: E402

OVERHEAD = 8


def native_fn(text: str) -> int:
    """The 'API': ~1.5x the fast estimator, heavier on digits/ids, plus the constant framing overhead."""
    return max(1, int(tokcount.fast_count(text) * 1.4 + 0.1 * sum(c.isdigit() for c in text))) + OVERHEAD


class Calls:
    def __init__(self):
        self.n = 0
        self.inflight = 0
        self.max_inflight = 0
        self.lock = threading.Lock()
        self.delay = 0.0

    def __call__(self, text):
        with self.lock:
            self.n += 1
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        if self.delay:
            time.sleep(self.delay)
        with self.lock:
            self.inflight -= 1
        return native_fn(text)


def mk(tmp, calls=None, **kw):
    calls = calls or Calls()
    return tokcount.Counter("anthropic", "claude-haiku-5-5", mode="native", transport=calls,
                            cache_path=os.path.join(tmp, "tc.jsonl"), **kw), calls


def forge(chunks):
    out = ["Document Chunks (each has a reference_id):", "```json"]
    out += ['{"reference_id": "%d", "content": %s, "section": ""}' % (i + 1, json.dumps(c))
            for i, c in enumerate(chunks)]
    out += ["```", "", "Reference Document List (for reference):", "```", "[1] sessions-x-compaction-1.md", "```"]
    return "\n".join(out)


def raw_recall(n=40):
    chunks = [f"[User]: run {k} ran pytest tests/test_mod{k}.py and saw FooError_{k}{k} at line {k * 7}. "
              f"Config PORT_{k} = {8000 + k}. Retry backoff {k}.5 seconds; job id JOB-{k * 31:05d}. "
              f"Plain narration about the work, nothing factual here, filler words filler words." * 2
              for k in range(1, n + 1)]
    return forge(chunks)


class CounterBasics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_native_count_subtracts_framing_overhead_and_is_flagged(self):
        c, calls = mk(self.tmp)
        t = "def f(x):\n    return {'k': [x, x+1], 'id': 'a1b2c3'}\n" * 20
        self.assertEqual(c.count(t), native_fn(t) - OVERHEAD)
        self.assertTrue(c.native and not c.estimated)
        self.assertEqual(c.tokenizer_id, "anthropic:count_tokens:claude-haiku-5-5")
        m = c.manifest()
        self.assertTrue(m["native"])
        self.assertEqual(m["overhead_subtracted"], OVERHEAD)
        self.assertGreater(c.count(t), 1.3 * tokcount.fast_count(t))     # differs from the fast estimator

    def test_cache_in_memory_and_on_disk(self):
        c, calls = mk(self.tmp)
        t = "hello world " * 50
        a = c.count(t)
        n1 = calls.n                                  # overhead probe + the text
        self.assertEqual(c.count(t), a)
        self.assertEqual(calls.n, n1)                 # memory hit
        c.close()
        c2, calls2 = mk(self.tmp)                     # a new process: the disk cache answers
        self.assertEqual(c2.count(t), a)
        self.assertEqual(calls2.n, 0)
        self.assertEqual(c2.stats["cache_hits"], 1)
        # keyed by model: another model does not see it
        c3 = tokcount.Counter("anthropic", "claude-sonnet-5-5", mode="native", transport=Calls(),
                              cache_path=os.path.join(self.tmp, "tc.jsonl"))
        c3.count(t)
        self.assertEqual(c3.stats["cache_hits"], 0)

    def test_concurrency_limit_and_batching(self):
        calls = Calls()
        calls.delay = 0.02
        c, _ = mk(self.tmp, calls, concurrency=3)
        texts = [f"text number {i} " * 30 for i in range(24)] + ["text number 0 " * 30]   # one duplicate
        got = c.count_many(texts)
        self.assertEqual(len(got), 25)
        self.assertEqual(got[0], got[-1])
        self.assertLessEqual(calls.max_inflight, 3)
        self.assertGreaterEqual(calls.max_inflight, 2)
        self.assertEqual(calls.n, 24 + 1)             # unique texts + the overhead probe

    def test_retry_on_429_then_success(self):
        c = tokcount.Counter("anthropic", "claude-haiku-5-5", mode="native", key="k", cache_path="",
                             sleep=lambda s: None)
        seq = [urllib.error.HTTPError("u", 429, "rl", {"retry-after": "1"}, None)]

        class R:
            def __init__(self, n): self.n = n
            def read(self): return json.dumps({"input_tokens": self.n}).encode()
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_urlopen(req, timeout=0):
            if seq:
                raise seq.pop()
            body = json.loads(req.data.decode())
            return R(10 if body["messages"][0]["content"] == "x" else 50)

        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            self.assertEqual(c.count("some text to count"), 50 - 9)
        self.assertEqual(c.stats["retries"], 1)

    def test_non_native_provider_is_flagged(self):
        c = tokcount.Counter("openrouter", "qwen/qwen3.8-flash", mode="auto", cache_path="")
        self.assertFalse(c.native)
        self.assertEqual(c.tokenizer_id, tokcount.FAST_ID)
        with self.assertRaises(tokcount.TokCountError):
            tokcount.Counter("openrouter", "m", mode="native")

    def test_estimate_mode_is_calibrated_ratio_and_flagged(self):
        c = tokcount.Counter("anthropic", "m", mode="estimate", ratio=1.5)
        t = "word " * 400
        self.assertEqual(c.count(t), round(tokcount.fast_count(t) * 1.5))
        self.assertTrue(c.estimated and not c.native)
        self.assertIn("1.5", c.tokenizer_id)


class FitToNativeBudget(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.c, self.calls = mk(self.tmp)
        self.text = "".join(f"line {i}: PORT_{i} = {8000 + i}, retry {i}.5s, id JOB-{i * 37:05d}\n" for i in range(400))

    def test_max_prefix_is_the_largest_fitting_cut(self):
        k = self.c.max_prefix(self.text, 1000)
        self.assertLessEqual(self.c.count(self.text[:k]), 1000)
        self.assertGreater(self.c.count(self.text[:k + 1]), 1000)
        # binary search over cached counts: far fewer calls than characters
        self.assertLess(self.calls.n, 80)

    def test_trim_to_budget_fits_natively(self):
        out = self.c.trim_to_budget(self.text, 700, marker="…[truncated]")
        self.assertLessEqual(self.c.count(out), 700)
        self.assertGreater(self.c.count(out), 600)
        self.assertTrue(out.endswith("…[truncated]"))
        self.assertGreater(tokcount.fast_count(self.text[:len(out)]), 0)
        self.assertEqual(self.c.trim_to_budget("short", 700), "short")

    def test_fit_native_runs_the_estimator_trimmer_to_a_native_cap(self):
        def trim_fn(t, est_cap):                       # a trimmer that only knows the FAST estimator
            lines = t.splitlines()
            while lines and tokcount.fast_count("\n".join(lines)) > est_cap:
                lines.pop()
            out = "\n".join(lines)
            return out, {"after": tokcount.fast_count(out)}

        out, info = tokcount.fit_native(self.c, self.text, 900, trim_fn)
        n = self.c.count(out)
        self.assertLessEqual(n, 900)
        self.assertEqual(info["native_tokens"], n)
        self.assertFalse(info["native_over_cap"])
        self.assertGreater(n, 800)                     # not trimmed to 1/1.5 of the budget
        # the legacy (fast-estimator-only) trim at the same number overshoots the native budget
        legacy, _ = trim_fn(self.text, 900)
        self.assertGreater(self.c.count(legacy), 900 * 1.2)


class PackerAndParity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.c, self.calls = mk(self.tmp)
        self.raw = raw_recall()
        rc.set_counter(self.c)

    def tearDown(self):
        rc.set_counter(None)
        tokcount.set_default(None)

    def test_packer_fits_the_native_budget_wrapper_included(self):
        for budget in (1024, 683, 341):
            packed = rc.pack_injection(self.raw, "sid-1", budget=budget, query="PORT_3")
            injected = packed["injected"]
            self.assertLessEqual(self.c.count(injected), budget)
            self.assertEqual(packed["total_tokens"], self.c.count(injected))
            self.assertGreaterEqual(packed["total_tokens"], 0.80 * budget, (budget, packed["total_tokens"]))
            self.assertEqual(packed["tokens"], self.c.count(packed["text"]))
            self.assertTrue(packed["native"])
            self.assertEqual(packed["tokenizer"], self.c.tokenizer_id)

    def test_legacy_estimator_packing_would_overshoot_natively(self):
        rc.set_counter(None)
        legacy = rc.pack_injection(self.raw, "sid-1", budget=1024, query="PORT_3", counter=None)
        self.assertLessEqual(legacy["total_tokens"], 1024)                    # fits in "our" tokens ...
        self.assertGreater(self.c.count(legacy["injected"]), 1024 * 1.25)    # ... but not in native ones

    def test_note_tokens_use_the_counter_and_ext_value_is_diagnostic(self):
        note = "GOAL\nship.\nACTIVE STATE\n- JOB-00031 running on PORT_3 = 8003.\n" * 5
        n, src = rc.note_tokens_from({"realized_tokens": 5}, note)
        self.assertEqual(n, self.c.count(note))
        self.assertIn("tokcount:anthropic:count_tokens", src)
        rc.set_counter(None)
        n2, src2 = rc.note_tokens_from({"realized_tokens": 5}, note)
        self.assertEqual((n2, src2), (5, "nc-metrics.realized_tokens"))        # legacy rule only without a counter

    def test_parity_is_computed_natively_and_converges_to_one_at_cap_equals_B(self):
        """The Amendment C.5 finding in miniature: at cap(B)=B the baseline summary (cut by the provider at B
        NATIVE tokens) and the nc memory (note + recall) are only equal if the nc side is packed natively."""
        B = 1536
        # baseline: the model would write 1.3*B native tokens; max_tokens=B cuts it at B native
        long_summary = "".join(f"- fact {i}: PORT_{i} = {8000 + i}; retry {i}.5s; id JOB-{i * 37:05d}\n" for i in range(2000))
        base = self.c.trim_to_budget(long_summary, B)
        b_tok = self.c.count(base)
        self.assertLessEqual(b_tok, B)
        # nc: note of 640 native tokens, recall packed to clamp(B - note) natively
        note = self.c.trim_to_budget(long_summary[::-1][:20000][::-1], 640)
        n_tok = self.c.count(note)
        cap = max(256, min(1024, B - n_tok))
        packed = rc.pack_injection(self.raw, "sid-1", budget=cap, query="PORT_3")
        nc_total = n_tok + self.c.count(packed["injected"])
        ratio_native = nc_total / b_tok
        self.assertGreaterEqual(ratio_native, 0.80)
        self.assertLessEqual(ratio_native, 1.05)
        # the pre-C.5 pipeline (nc packed in estimator tokens, baseline cut by the provider in native tokens)
        rc.set_counter(None)
        legacy = rc.pack_injection(self.raw, "sid-1", budget=cap, query="PORT_3", counter=None)
        note_fast = rc.note_tokens_from({}, note, counter=None)[0]
        legacy_total_native = self.c.count(note) + self.c.count(legacy["injected"])
        self.assertGreater(legacy_total_native / b_tok, ratio_native + 0.08)
        self.assertGreater(note_fast, 0)

    def test_calibration_aggregate_converges_to_cap_equals_B_under_the_mock(self):
        B = 1536
        per_chain = []
        for ci in range(3):
            text = "".join(f"- c{ci} fact {i}: PORT_{i} = {8000 + i}; id JOB-{i * 37 + ci:05d}\n" for i in range(2000))
            base_by_span = {s: self.c.count(self.c.trim_to_budget(text, B - 40 * (s % 2))) for s in (1, 2, 3)}
            note = self.c.count(self.c.trim_to_budget(text[::-1][:30000][::-1], 640))
            packed = rc.pack_injection(self.raw, f"s{ci}", budget=max(256, min(1024, B - note)), query="PORT_3")
            recall = self.c.count(packed["injected"])
            per_chain.append({"chain_id": f"c{ci}", "baseline_by_span": base_by_span,
                              "nc_note_by_span": {1: note, 2: note, 3: note}, "baseline_c3": base_by_span[3],
                              "nc_recall_tokens": recall, "nc_total": note + recall})
        agg = rc.calibrate_aggregate(per_chain, B)
        self.assertEqual(agg["chains"], 3)
        self.assertEqual(agg["baseline_all_compactions_n"], 9)
        for k in ("ratio_ask_time", "ratio_pooled_all_compactions"):
            self.assertGreaterEqual(agg[k], 0.85, (k, agg))
            self.assertLessEqual(agg[k], 1.05, (k, agg))
        adv = agg["recommended_cap_pooled"]["recommended_cap"]
        self.assertLessEqual(abs(adv - B) / B, 0.12, agg)                     # the advised cap stays ~B


class Wiring(unittest.TestCase):
    def tearDown(self):
        rc.set_counter(None)
        tokcount.set_default(None)

    def test_setup_counter_refuses_live_anthropic_without_native(self):
        import types
        a = types.SimpleNamespace(provider="anthropic", qa_model="claude-haiku-5-5", agent_model="claude-haiku-5-5",
                                  token_counter="auto", tokcount_concurrency=2)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("NC_TOKCOUNT", None)
            with self.assertRaises(tokcount.TokCountError):
                rc.setup_counter(a, live=True)
            a.token_counter = "tiktoken"                       # explicit opt-out: allowed, flagged non-native
            c = rc.setup_counter(a, live=True)
            self.assertFalse(c.native)
            self.assertEqual(rc.tokenizer_info()["native"], False)

    def test_dry_run_uses_a_flagged_ratio_estimator(self):
        import types
        a = types.SimpleNamespace(provider="anthropic", qa_model="m", agent_model="m", token_counter=None)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NC_TOKCOUNT", None)
            c = rc.setup_counter(a, live=False)
        self.assertTrue(c.estimated and not c.native)
        self.assertGreater(c.ratio, 1.2)

    def test_base_env_tells_the_extension_to_count_natively_only_with_a_native_counter(self):
        c, _ = mk(tempfile.mkdtemp())
        rc.set_counter(c)
        env = rc.base_env("baseline", "claude-haiku-5-5", "sid", "http://x/mcp", None, cap=1536, provider="anthropic")
        self.assertEqual(env["NC_TOKCOUNT_NATIVE"], "anthropic")
        rc.set_counter(None)
        env = rc.base_env("baseline", "claude-haiku-5-5", "sid", "http://x/mcp", None, cap=1536, provider="anthropic")
        self.assertEqual(env["NC_TOKCOUNT_NATIVE"], "off")


class LongsessionCounting(unittest.TestCase):
    def setUp(self):
        import run_longsession as R
        self.R = R
        self.c, _ = mk(tempfile.mkdtemp())
        tokcount.set_default(self.c)

    def tearDown(self):
        tokcount.set_default(None)

    def test_trim_note_native_fits_the_native_cap(self):
        note = ("GOAL\nShip the thing.\nKEY FIGURES & IDENTIFIERS\n"
                + "\n".join(f"- id{i} = {i * 111} PORT_{i}" for i in range(60))
                + "\nDECISIONS & RATIONALE\n" + "\n".join(f"- decided {i} because reason number {i}." for i in range(40))
                + "\nACTIVE STATE\n- running job JOB1 now.")
        cap = 400
        out, info = self.R.trim_note_native(note, cap)
        self.assertLessEqual(self.c.count(out), cap)
        self.assertEqual(info["native_tokens"], self.c.count(out))
        self.assertIn("ACTIVE STATE\n- running job JOB1 now.", out)
        self.assertFalse(info["over_cap"])
        # the unfitted trimmer at the same number overshoots in native tokens
        legacy, _ = self.R.trim_note(note, cap)
        self.assertGreater(self.c.count(legacy), cap)

    def test_ntok_and_label_follow_the_default_counter(self):
        self.assertEqual(self.R.ntok("hello world " * 30), self.c.count("hello world " * 30))
        self.assertTrue(self.R.tok_native())
        self.assertIn("count_tokens", self.R.tok_label())
        tokcount.set_default(None)
        self.assertFalse(self.R.tok_native())


if __name__ == "__main__":
    unittest.main()
