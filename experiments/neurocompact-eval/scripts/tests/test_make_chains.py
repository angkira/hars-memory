"""Synthetic-data tests for make_chains.py independence options + verify_chains.py.

Everything here runs on SYNTHETIC sessions (generated below) and a deterministic fake
generator (no network, $0). The real OpenCode sessions / chains.jsonl are private and
not in the repo, so these tests prove mechanics, not the properties of the real dataset.

Run:  python3 -m pytest -q experiments/neurocompact-eval/scripts/tests/test_make_chains.py
  or: python3 experiments/neurocompact-eval/scripts/tests/test_make_chains.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import make_chains as mc  # noqa: E402
import verify_chains as vc  # noqa: E402

# make_chains.py as it was BEFORE the independence options (git blob of commit
# 79caeecd "evals: NeuroCompact session-memory A/B harness"). The default-behaviour
# regression runs this pristine copy and the current file on the same input and
# requires byte-identical outputs.
BASELINE_BLOB = "9082686d5af1420a3ca20f8e6b2d668e3897d54f"

WORDS = ("configured", "service", "on", "port", "with", "retry", "policy", "enabled",
         "after", "the", "rollout", "review")


# ------------------------------------------------------------ synthetic sessions

def synth_session(sid: str, n_bounds: int, events_per_span: int = 14) -> dict:
    """Session with n_bounds compaction boundaries. Every line carries a token
    (svc_<sid>_<b>_<e>_<l>) that occurs exactly once in the session -> clean anchors."""
    events, bounds = [], []
    tag = sid[-4:]
    for b in range(n_bounds):
        for e in range(events_per_span):
            lines = []
            for ln in range(6):
                port = 1000 + (b * 97 + e * 13 + ln) % 8000
                lines.append(f"step {e} {WORDS[ln % 12]} svc_{tag}_{b}_{e}_{ln} "
                             f"bound to port {port} with retry policy enabled after the "
                             f"rollout review number {b}{e}{ln}")
            events.append({"role": "user" if e % 2 == 0 else "assistant",
                           "text": "\n".join(lines)})
        bounds.append({"idx": len(events)})
        events.append({"role": "system", "text": "[compaction boundary]"})
    return {"sid": sid, "title": f"synthetic {sid}", "events": events, "boundaries": bounds}


def write_sessions(d: str, counts: list[int]) -> list[str]:
    os.makedirs(d, exist_ok=True)
    sids = []
    for i, n in enumerate(counts):
        sid = f"ses_synthetic{i:02d}AB{i:02d}xy"
        with open(os.path.join(d, f"{sid}.json"), "w") as w:
            json.dump(synth_session(sid, n), w)
        sids.append(sid)
    return sids


# ------------------------------------------------------------- fake generator

class FakeGen:
    """Deterministic stand-in for the QA generator, parsing the real prompt.

    Bias modelled on the observed generator: on the FIRST prompt of a chain it is
    stubborn -- it reuses the anchor it already chose for an identical evidence
    window (overlapping chains show the same span twice) and always asks about
    NC_PHANTOM_TTL_MS. In a repair prompt it follows the listed suggestions/stems.
    """

    def __init__(self):
        self.window_anchor: dict[str, str] = {}
        self.calls = 0

    @staticmethod
    def windows(prompt: str) -> dict[str, str]:
        out = {}
        for m in re.finditer(r"^(E\d[HT]\d+) \(\d+ words\): (.*?)(?=\nE\d[HT]\d+ \(|\n\[|\n\n|\Z)",
                             prompt, re.M | re.S):
            out[m.group(1)] = m.group(2)
        return out

    @staticmethod
    def suggestions(prompt: str, span: int, sl: str) -> list[tuple[str, str]]:
        m = re.search(rf"^\[SUGGESTED ANCHORS {sl} of SPAN {span} .*?\]\n(.*)$", prompt, re.M)
        if not m:
            return []
        return [tuple(x.split("->", 1)) for x in m.group(1).split(", ") if "->" in x]

    def answerable(self, prompt: str, span: int, sl: str, stubborn: bool) -> dict:
        wins = self.windows(prompt)
        sugg = self.suggestions(prompt, span, sl)
        wid, anchor = (sugg[0] if sugg else (None, ""))
        if stubborn:
            for w, txt in wins.items():
                if w.startswith(f"E{span}{sl[0]}") and txt in self.window_anchor:
                    wid, anchor = w, self.window_anchor[txt]
                    break
        ev = wins.get(wid, "")
        if ev and anchor:
            self.window_anchor.setdefault(ev, anchor)
        return {"kind": "answerable", "span_idx": span, "slice": sl,
                "question": "Which service identifier was bound in this configuration step?",
                "canonical_short_answer": f"It was {anchor}", "gold_anchor": anchor,
                "evidence": ev}

    @staticmethod
    def unanswerable(prompt: str, stubborn: bool) -> dict:
        m = re.search(r"\[UNANSWERABLE STEMS[^\n]*\]\n([^\n]+)", prompt)
        q = m.group(1) if m else "What does the --phantom-flag default to?"
        if stubborn:
            q = "What does NC_PHANTOM_TTL_MS default to when it is unset?"
        return {"kind": "unanswerable", "span_idx": 0, "slice": "NONE", "question": q,
                "canonical_short_answer": "", "gold_anchor": "", "evidence": ""}

    def __call__(self, body: dict, key: str, timeout: int = 300):
        self.calls += 1
        prompt = body["messages"][0]["content"]
        items = []
        if prompt.startswith("You previously wrote exam items"):
            for m in re.finditer(r"^\[SLOT \d+\] kind=(\w+) span_idx=(\d+) slice=(\w+)", prompt, re.M):
                if m.group(1) == "unanswerable":
                    items.append(self.unanswerable(prompt, stubborn=False))
                else:
                    items.append(self.answerable(prompt, int(m.group(2)), m.group(3), False))
        else:
            for span in (1, 2, 3):
                for sl in ("HEAD", "TAIL"):
                    items.append(self.answerable(prompt, span, sl, stubborn=True))
            items.append(self.unanswerable(prompt, stubborn=True))
        content = json.dumps({"items": items})
        return 200, json.dumps({"choices": [{"message": {"content": content},
                                             "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": len(prompt) // 3,
                                          "completion_tokens": 400, "cost": 0.001}})


# ------------------------------------------------------------------ runners

def load_baseline(tmp: str):
    """Pristine pre-change make_chains.py from git, or None when unavailable."""
    try:
        src = subprocess.run(["git", "-C", SCRIPTS, "cat-file", "-p", BASELINE_BLOB],
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    d = os.path.join(tmp, "baseline_mod")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "make_chains_baseline.py"), "wb").write(src)
    shutil.copy(os.path.join(SCRIPTS, "make_longreplay_dataset.py"), d)
    spec = importlib.util.spec_from_file_location(
        "make_chains_baseline", os.path.join(d, "make_chains_baseline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_main(mod, sess_dir: str, argv: list[str], fake: FakeGen | None = None) -> tuple[int, str]:
    """Run mod.main() in-process with a synthetic SESS_DIR and a fake generator."""
    old = (mod.SESS_DIR, mod.http_post, sys.argv, os.environ.get("OPENROUTER_API_KEY"))
    mod.SESS_DIR = sess_dir
    mod.http_post = fake or FakeGen()
    mod.RAW_LOG.update(path="", n=0)
    mod.BLOCKED["n"] = 0
    os.environ["OPENROUTER_API_KEY"] = "sk-or-synthetic-test"
    sys.argv = ["make_chains.py"] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = mod.main()
    finally:
        mod.SESS_DIR, mod.http_post, sys.argv = old[:3]
        if old[3] is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = old[3]
    return rc, buf.getvalue()


def strip_ts(obj):
    if isinstance(obj, dict):
        return {k: strip_ts(v) for k, v in obj.items() if k != "generated_at"}
    if isinstance(obj, list):
        return [strip_ts(x) for x in obj]
    return obj


def run_verify(sess_dir: str, chains: str, out: str, extra: list[str] = ()) -> tuple[int, str]:
    old = (mc.SESS_DIR, sys.argv)
    mc.SESS_DIR = sess_dir
    sys.argv = ["verify_chains.py", "--chains", chains, "--out", out, *extra]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = vc.main()
    finally:
        mc.SESS_DIR, sys.argv = old
    return rc, open(out).read()


def gen(tmp: str, name: str, sess_dir: str, flags: list[str], mod=mc, max_chains: int = 40):
    out = os.path.join(tmp, name)
    os.makedirs(out, exist_ok=True)
    chains, man, raw = (os.path.join(out, f) for f in
                        ("chains.jsonl", "chains-manifest.json", "raw.jsonl"))
    rc, log = run_main(mod, sess_dir, ["--max-chains", str(max_chains), "--budget-usd", "5",
                                       "--out", chains, "--manifest", man,
                                       "--save-raw", raw, *flags])
    return rc, log, chains, man, raw


# -------------------------------------------------------------------- tests

COUNTS = [3, 4, 6, 7, 3, 5, 6, 3, 4, 9]     # 10 synthetic sessions


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nc_chains_test_")
        self.sess = os.path.join(self.tmp, "sessions")
        self.sids = write_sessions(self.sess, COUNTS)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestDefaultUnchanged(Base):
    """Regression: without the new flags, outputs are byte-identical to the
    pre-change script (pristine git blob) on the same synthetic input."""

    def setUp(self):
        super().setUp()
        self.base = load_baseline(self.tmp)
        if self.base is None:
            self.skipTest("baseline git blob not available")

    def test_dry_run_identical(self):
        outs = []
        for mod, tag in ((self.base, "old"), (mc, "new")):
            sp = os.path.join(self.tmp, f"spans-{tag}.jsonl")
            rc, log = run_main(mod, self.sess, ["--dry-run", "--max-chains", "40",
                                                "--spans-out", sp])
            self.assertEqual(rc, 0)
            outs.append((log.replace(sp, "<SP>"), open(sp, "rb").read()))
        self.assertEqual(outs[0][0], outs[1][0])
        self.assertEqual(outs[0][1], outs[1][1])

    def test_generation_identical(self):
        res = []
        for mod, tag in ((self.base, "old"), (mc, "new")):
            rc, log, chains, man, raw = gen(self.tmp, tag, self.sess, [], mod=mod)
            self.assertEqual(rc, 0)
            res.append({
                "log": log.replace(os.path.join(self.tmp, tag), "<OUT>"),
                "chains": [strip_ts(json.loads(l)) for l in open(chains)],
                "manifest": strip_ts(json.load(open(man))),
                "raw": open(raw, "rb").read(),
            })
        old, new = res
        self.assertGreater(len(old["chains"]), 10)
        self.assertEqual(old["raw"], new["raw"])           # every prompt + response
        self.assertEqual(old["chains"], new["chains"])
        self.assertEqual(old["manifest"], new["manifest"])
        self.assertEqual(old["log"], new["log"])
        self.assertNotIn("build_flags", new["chains"][0])
        self.assertNotIn("independence", new["manifest"])


class TestEnumeration(unittest.TestCase):
    def test_overlap_default_and_disjoint(self):
        sess = [{"boundaries": [{}] * n} for n in (2, 3, 4, 5, 6, 7, 9)]
        self.assertEqual(mc.enumerate_chains(sess),
                         mc.enumerate_chains(sess, disjoint=False))
        dis = mc.enumerate_chains(sess, disjoint=True)
        by = {}
        for si, ci in dis:
            by.setdefault(si, []).append(ci)
        # session index i has n = (2,3,4,5,6,7,9)[i] boundaries
        self.assertEqual(by, {1: [0], 2: [0], 3: [0], 4: [0, 3], 5: [0, 3], 6: [0, 3, 6]})
        for cis in by.values():
            ranges = [set(range(c, c + mc.CHAIN_LEN)) for c in cis]
            for i in range(len(ranges)):
                for j in range(i + 1, len(ranges)):
                    self.assertFalse(ranges[i] & ranges[j])


class TestFlags(Base):
    def test_default_has_the_reviewed_defects(self):
        """Synthetic reproduction of the review finding (overlap, repeated anchors,
        one probe for everything) and that verify_chains reports them."""
        rc, _, chains, man, _ = gen(self.tmp, "default", self.sess, [], max_chains=40)
        rc, rep = run_verify(self.sess, chains, os.path.join(self.tmp, "v.md"))
        self.assertIn("## Independence", rep)
        self.assertRegex(rep, r"chain pairs sharing >= 1 span: [1-9]")
        self.assertRegex(rep, r"gold anchors repeated across chains: [1-9]")
        self.assertIn("nc_phantom_ttl_ms:", rep)
        self.assertIn("independence: PASS", rep)      # 10 sessions, nothing declared

    def test_disjoint(self):
        rc, log, chains, man, _ = gen(self.tmp, "dis", self.sess, ["--disjoint"])
        recs = [json.loads(l) for l in open(chains)]
        expect = sum(len(mc.chain_starts(n, True)) for n in COUNTS)
        self.assertEqual(len(recs), expect)           # 1+1+2+2+1+1+2+1+1+3 = 15
        used = set()
        for c in recs:
            for k in range(mc.CHAIN_LEN):
                key = (c["session_id"], c["chain_idx"] + k)
                self.assertNotIn(key, used)
                used.add(key)
            self.assertEqual(c["build_flags"]["disjoint"], True)
        self.assertTrue(json.load(open(man))["independence"]["disjoint"])
        rc, rep = run_verify(self.sess, chains, os.path.join(self.tmp, "v.md"))
        self.assertEqual(rc, 0, rep[-2000:])
        self.assertIn("chain pairs sharing >= 1 span: 0", rep)
        self.assertIn("declared build flags: disjoint", rep)

    def test_unique_probes(self):
        rc, log, chains, man, _ = gen(self.tmp, "up", self.sess, ["--unique-probes"])
        recs = [json.loads(l) for l in open(chains)]
        probes = [mc.probe_tokens(q["question"])[0] for c in recs for q in c["qa"]
                  if not q["answerable"]]
        self.assertEqual(len(probes), len(recs))
        self.assertEqual(len(set(p.lower() for p in probes)), len(probes))
        m = json.load(open(man))
        rej = [r["reason"] for c in m["chains"] for r in c["independence_rejections"]]
        # stubborn fake re-asks NC_PHANTOM_TTL_MS every chain; all but the first refused
        self.assertEqual(sum(r.startswith("probe_reused:NC_PHANTOM_TTL_MS") for r in rej),
                         len(recs) - 1)
        self.assertEqual(sum(c["rejections"].get("probe_reused", 0) for c in m["chains"]),
                         len(recs) - 1)
        self.assertEqual(m["independence"]["probe_pool_size"], 120)
        self.assertIn("NC_PHANTOM_TTL_MS", mc.probe_pool(1337))
        rc, rep = run_verify(self.sess, chains, os.path.join(self.tmp, "v.md"))
        self.assertIn("probes used by > 1 item: 0", rep)

    def test_unique_probes_pool_too_small(self):
        with self.assertRaises(SystemExit) as cm:
            gen(self.tmp, "up2", self.sess, ["--unique-probes", "--limit", "121"],
                max_chains=121)
        self.assertIn("pool has 120 unused", str(cm.exception))

    def test_dedup_anchors(self):
        rc, log, chains, man, _ = gen(self.tmp, "dd", self.sess, ["--dedup-anchors"])
        recs = [json.loads(l) for l in open(chains)]
        seen = {}
        for c in recs:
            for q in c["qa"]:
                if q["answerable"]:
                    k = (c["session_id"], q["gold_anchor"].lower())
                    self.assertNotIn(k, seen, f"{k} in {seen.get(k)} and {c['chain_id']}")
                    seen[k] = c["chain_id"]
        m = json.load(open(man))
        rej = [r for c in m["chains"] + m["dropped"]
               for r in c.get("independence_rejections", [])]
        self.assertTrue(rej and all(r["reason"].startswith("anchor_dup_across_chains:")
                                    for r in rej))
        self.assertTrue(any(c["rejections"].get("anchor_dup_across_chains")
                            for c in m["chains"]))
        rc, rep = run_verify(self.sess, chains, os.path.join(self.tmp, "v.md"))
        self.assertNotIn("INDEP-ANCHOR", rep)

    def test_all_flags_verify_clean(self):
        rc, log, chains, man, _ = gen(self.tmp, "all", self.sess,
                                      ["--disjoint", "--unique-probes", "--dedup-anchors"])
        rc, rep = run_verify(self.sess, chains, os.path.join(self.tmp, "v.md"))
        self.assertEqual(rc, 0, rep[-2000:])
        self.assertIn("independence: PASS", rep)


class TestVerifyIndependence(unittest.TestCase):
    @staticmethod
    def chain(sid, ci, anchors=("a1",), probe="NC_X_Y"):
        spans = [{"span_idx": k + 1, "segments": [{"role": "user", "text": f"{sid}{ci + k}"}]}
                 for k in range(3)]
        qa = [{"answerable": True, "gold_anchor": a, "question": "q"} for a in anchors]
        qa.append({"answerable": False, "gold_anchor": "",
                   "question": f"What does {probe} default to?"})
        return {"chain_id": f"chain_{sid}_c{ci}", "session_id": sid, "chain_idx": ci,
                "spans": spans, "qa": qa}

    def test_few_sessions_fail(self):
        cs = [self.chain(f"s{i}", 0, (f"a{i}",), f"NC_P_{i}") for i in range(5)]
        cs += [self.chain(f"s{i}", 1, (f"b{i}",), f"NC_Q_{i}") for i in range(5)]
        lines, fails = vc.independence(cs, {"disjoint": False})
        self.assertIn("INDEP-SESSIONS", [f[1] for f in fails])
        self.assertIn("chain pairs sharing >= 1 span: 5", "\n".join(lines))

    def test_disjoint_shared_span_fail(self):
        cs = [self.chain(f"s{i}", 0, (f"a{i}",), f"NC_P_{i}") for i in range(8)]
        cs.append(self.chain("s0", 2, ("zz",), "NC_Z_Z"))      # shares boundary 2 with s0 c0
        _, fails = vc.independence(cs, {"disjoint": True})
        self.assertEqual([f[1] for f in fails], ["INDEP-DISJOINT"])
        _, fails = vc.independence(cs, {"disjoint": False})
        self.assertEqual(fails, [])
        cs[-1] = self.chain("s0", 3, ("zz",), "NC_Z_Z")
        _, fails = vc.independence(cs, {"disjoint": True})
        self.assertEqual(fails, [])

    def test_duplicates_reported(self):
        cs = [self.chain(f"s{i}", 0, ("same",), "NC_PHANTOM_TTL_MS") for i in range(8)]
        lines, fails = vc.independence(cs, {})
        txt = "\n".join(lines)
        self.assertIn("gold anchors repeated across chains: 1", txt)
        self.assertIn("nc_phantom_ttl_ms: 8x", txt)
        self.assertEqual(fails, [])          # nothing declared -> report only
        _, fails = vc.independence(cs, {"unique_probes": True})
        self.assertEqual([f[1] for f in fails], ["INDEP-PROBE"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
