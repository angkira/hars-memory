"""Offline tests (synthetic fixtures, $0) for multihop.py, py_probes.py and the qtype-aware
verify_chains.py.

Run: cd experiments/neurocompact-eval/scripts && python -m pytest tests
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_chains as mc  # noqa: E402
import multihop as mh  # noqa: E402
import oh_pipeline as ohp  # noqa: E402
import py_probes as pp  # noqa: E402
import test_make_chains as tmc  # noqa: E402

_SAVED = {}
_PATCHED = ("evidence_windows", "suggested_anchors", "MAX_RETRIES_PER_SLOT", "llm", "SESS_DIR",
            "_orig_evidence_windows", "_orig_suggested_anchors")


def setUpModule():          # multihop.main() installs anchor-window patches on make_chains
    for k in _PATCHED:
        _SAVED[k] = (hasattr(mc, k), getattr(mc, k, None))


def tearDownModule():       # do not leak them into other test modules
    for k, (had, v) in _SAVED.items():
        if had:
            setattr(mc, k, v)
        elif hasattr(mc, k):
            delattr(mc, k)


# ------------------------------------------------------------------ fixtures

EV1 = "edited the file src/widgets/frobnicate_helper.py to guard the zero frob count branch now"
EV2 = "FAILED tests/test_widget_strict.py::test_zero_strict - AssertionError: expected 3 frobs after retry"
SPANS = {
    1: f"[assistant] intro line\n{EV1}\nnothing else here\n\n",
    2: "[tool] ran the whole suite and everything looked fine in this middle window of work\n\n",
    3: f"[tool] pytest output follows\n{EV2}\nsummary line 1 failed 12 passed in 3.2s\n\n",
}
ISSUE = "Widget.frobnicate() crashes when the frob count is zero and the mode is strict."


def good_item():
    return {
        "question": "Which test failed after the agent edited src/widgets/frobnicate_helper.py?",
        "canonical_short_answer": "tests/test_widget_strict.py::test_zero_strict",
        "link_anchor": "src/widgets/frobnicate_helper.py",
        "gold_anchor": "test_zero_strict",
        "hops": [{"span_idx": 1, "evidence": EV1}, {"span_idx": 3, "evidence": EV2}],
    }


def check(item, **kw):
    return mh.validate_multihop(item, kw.pop("spans", SPANS), kw.pop("issue", ISSUE),
                                kw.pop("fix", ""), kw.pop("taken", None))


class TestValidateMultihop(unittest.TestCase):
    def test_good_item_passes(self):
        ok, why, ck = check(good_item())
        self.assertTrue(ok, why)
        self.assertEqual(ck["slice_of_ev2"], "HEAD")
        self.assertTrue(ck["ns_link_not_in_ev2"] and ck["oh_guard"])

    def reasons(self, **changes):
        it = good_item()
        for k, v in changes.items():
            it[k] = v
        return check(it)[1]

    def test_evidence_must_be_verbatim_in_its_span(self):
        it = good_item()
        it["hops"][1]["evidence"] = EV2.replace("expected", "wanted")
        self.assertIn("ev2_not_verbatim", check(it)[1])
        it = good_item()
        it["hops"][0]["span_idx"] = 2                      # ev1 quoted from the wrong span
        self.assertIn("ev1_not_verbatim", check(it)[1])

    def test_evidence_min_words(self):
        it = good_item()
        it["hops"][0]["evidence"] = "edited the file src/widgets/frobnicate_helper.py"
        self.assertIn("ev1_lt_8_words", check(it)[1])

    def test_spans_must_differ_and_be_ordered(self):
        it = good_item()
        it["hops"][1] = {"span_idx": 1, "evidence": EV1}
        self.assertIn("same_span", check(it)[1])
        it = good_item()
        it["hops"] = it["hops"][::-1]
        self.assertIn("hop_order", check(it)[1])
        it = good_item()
        it["hops"] = it["hops"][:1]
        self.assertIn("hops_count", check(it)[1])
        it = good_item()
        it["hops"][1]["span_idx"] = 7
        self.assertIn("bad_span_idx", check(it)[1])

    def test_anchor_leak_and_link_in_question(self):
        self.assertIn("anchor_leaked_in_question", self.reasons(
            question="Did test_zero_strict fail after src/widgets/frobnicate_helper.py changed?"))
        self.assertIn("link_not_in_question", self.reasons(
            question="Which test failed after the agent edited the helper module again?"))
        self.assertIn("answer_in_question", self.reasons(
            question="Is it true that tests/test_widget_strict.py::test_zero_strict fails after "
                     "src/widgets/frobnicate_helper.py changed?"))

    def test_anchor_discrimination(self):
        sp = dict(SPANS)
        sp[2] += " later again src/widgets/frobnicate_helper.py was mentioned"
        self.assertIn("link_in_other_span", check(good_item(), spans=sp)[1])
        sp = dict(SPANS)
        sp[1] += " and test_zero_strict is also here"
        w = check(good_item(), spans=sp)[1]
        self.assertIn("gold_in_other_span", w)
        self.assertIn("single_ev1_has_answer", self.reasons(
            hops=[{"span_idx": 1, "evidence": EV1 + " test_zero_strict"}, good_item()["hops"][1]]))

    def test_single_evidence_heuristic(self):
        # (a) ev2 alone already carries the question's entity
        sp = dict(SPANS)
        ev2 = EV2 + " src/widgets/frobnicate_helper.py"
        sp[3] = sp[3].replace(EV2, ev2)
        it = good_item()
        it["hops"][1]["evidence"] = ev2
        w = check(it, spans=sp)[1]
        self.assertIn("single_ev2_has_link", w)
        # (c) the question names a token that only hop 2 holds
        w = self.reasons(question="Which test failed after src/widgets/frobnicate_helper.py "
                                  "when tests/test_widget_strict.py ran?")
        self.assertTrue(any(x.startswith("single_question_names_ev2_token") for x in w), w)

    def test_distinct_anchors_and_taken(self):
        self.assertIn("anchors_equal", self.reasons(gold_anchor="src/widgets/frobnicate_helper.py",
                                                    link_anchor="src/widgets/frobnicate_helper.py"))
        self.assertIn("anchor_duplicated_within_chain",
                      check(good_item(), taken={"test_zero_strict"})[1])

    def test_contamination_guard(self):
        w = check(good_item(), issue=ISSUE + " see test_zero_strict")[1]
        self.assertIn("oh_anchor_in_issue", w)
        w = check(good_item(), fix='"newString": "def frobnicate_helper(): src/widgets/'
                                   'frobnicate_helper.py"')[1]
        self.assertIn("oh_anchor_in_fix_link", w)
        w = self.reasons(question="Crashes when the frob count is zero and the mode is strict, "
                                  "what failed after src/widgets/frobnicate_helper.py?")
        self.assertIn("oh_question_restates_issue", w)

    def test_answer_length(self):
        self.assertTrue(any(x.startswith("answer_too_long")
                            for x in self.reasons(canonical_short_answer="word " * 40)))
        self.assertIn("empty_answer", self.reasons(canonical_short_answer=""))


class TestRecordRoundtrip(unittest.TestCase):
    def test_record_fields_and_back(self):
        ok, why, ck = check(good_item())
        chain = {"chain_id": "chain_x_c0", "session_id": "x"}
        rec = mh.to_record(good_item(), ck, chain, 2, SPANS)
        self.assertEqual((rec["qtype"], rec["qid"]), ("multihop", "chain_x_c0_mh_2"))
        self.assertEqual((rec["span_idxs"], rec["span_idx"], rec["lag"], rec["hop_lags"]),
                         ([1, 3], 3, 3, [3, 1]))
        self.assertTrue(rec["answerable"] and not rec["must_abstain"])
        self.assertIn("\n[...]\n", rec["evidence"])
        back = mh.item_from_record(rec)
        self.assertTrue(mh.validate_multihop(back, SPANS, ISSUE, "")[0])

    def test_parse(self):
        raw = "```json\n" + json.dumps({"items": [good_item()]}) + "\n```"
        got = mh.parse_mh(raw)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["hops"][1]["span_idx"], 3)
        self.assertEqual(mh.parse_mh("no json"), [])


# --------------------------------------------------------------- py_probes

PY_TRANSCRIPT = (
    "[tool read] in: {\"filePath\": \"/workspace/sphinx-doc__sphinx__4.0/sphinx/domains/python.py\"}\n"
    "[tool read] in: {\"filePath\": \"/workspace/sphinx-doc__sphinx__4.0/sphinx/domains/std.py\"}\n"
    "[tool read] in: {\"filePath\": \"/workspace/sphinx-doc__sphinx__4.0/sphinx/util/typing.py\"}\n"
    "[tool read] in: {\"filePath\": \"/workspace/sphinx-doc__sphinx__4.0/tests/test_domain_py.py\"}\n"
    "def ClassMaker(): pass  # LeaseBackoffError\n")


class TestPyProbes(unittest.TestCase):
    def test_repo_dirs_exclude_tests(self):
        d = pp.repo_dirs(PY_TRANSCRIPT)
        self.assertEqual(d[0], "sphinx/domains")
        self.assertIn("sphinx/util", d)
        self.assertFalse(any(x.startswith("tests") for x in d))

    def test_probes_fit_repo_and_are_absent(self):
        got = pp.probe_candidates(PY_TRANSCRIPT, [], limit=6)
        self.assertEqual(len(got), 6)
        for t in got:
            self.assertEqual(mc.probe_tokens(pp.stem_for(t)), [t], t)   # exactly one probe token
            self.assertNotRegex(t, r"^NC_")
            self.assertIsNone(mc.anchor_re(t).search(PY_TRANSCRIPT))
        self.assertTrue(any(t.startswith("sphinx/") and t.endswith(".py") for t in got))
        self.assertTrue(any(t.split("_")[0] in pp.VERBS for t in got))
        # a noun that really exists in the transcript (as a substring) is never offered
        self.assertFalse(any("lease_backoff" in t.lower() for t in got))

    def test_django_style_setting_is_upper(self):
        tr = PY_TRANSCRIPT.replace("sphinx", "django").replace("django-doc__django",
                                                              "django__django")
        toks = pp.probe_candidates(tr, [], limit=9)
        self.assertTrue(any(t.isupper() for t in toks), toks)

    def test_distinct_across_chains(self):
        used, seen = set(), []
        for k in range(8):
            tr = PY_TRANSCRIPT + f"\nsession marker {k}\n"
            first = pp.probe_candidates(tr, [], limit=1, used=used)[0]
            self.assertNotIn(first.lower(), used)
            used.add(first.lower())
            seen.append(first)
        self.assertEqual(len(set(seen)), 8)
        nouns = {re.sub(r"^(?:\w+/)+|\.py$|^[a-z]+_(?=\w+_\w+$)", "", t.lower()) for t in seen}
        self.assertGreaterEqual(len(nouns), 8)

    def test_deterministic_and_install(self):
        a = pp.probe_candidates(PY_TRANSCRIPT, [], 5, salt="s1")
        self.assertEqual(a, pp.probe_candidates(PY_TRANSCRIPT, [], 5, salt="s1"))

        class M:
            pass
        m = M()
        pp.install(m)
        out = m.absent_probe_candidates(["x"], PY_TRANSCRIPT, salt="s1")
        self.assertEqual(out, a)
        blk = m.absent_block({"absent_probes": out})
        self.assertIn("UNANSWERABLE STEMS", blk)
        used = {out[0].lower()}
        nxt = m.draw_unique_probes([], used, [], PY_TRANSCRIPT)
        self.assertNotIn(out[0], nxt)


# ------------------------------------------------- end to end (fake generator)

class ScriptedMH:
    """Stands in for mc.llm: parses the real multihop prompt, quotes real windows."""

    def __init__(self, bad_first=False, id_mode=False):
        self.calls, self.bad_first, self.id_mode = 0, bad_first, id_mode

    def __call__(self, prompt, key, model, spend, max_tokens=0, est_cost=0, temperature=0):
        self.calls += 1
        n = int(re.search(r"EXACTLY (\d+) MULTI-HOP", prompt).group(1))
        if "MENU RULES" in prompt:
            return self.menu(prompt, n)
        wins = tmc.FakeGen.windows(prompt)
        used = set(re.search(r"\[ANCHORS ALREADY USED\]\n([^\n]*)", prompt).group(1).split(", "))

        def sugg(span, sl):
            return [(w, t) for w, t in tmc.FakeGen.suggestions(prompt, span, sl) if t not in used]
        items = []
        l1, g3 = sugg(1, "TAIL") + sugg(1, "HEAD")[2:], sugg(3, "TAIL") + sugg(3, "HEAD")[2:]
        for k in range(n):
            (lw, la), (gw, ga) = l1[k], g3[k]
            hops = [{"span_idx": 1, "evidence": wins[lw]}, {"span_idx": 3, "evidence": wins[gw]}]
            if self.bad_first and self.calls == 1 and k == 0:
                hops[1]["span_idx"] = 1                       # same span -> rejected
                gw = lw
            if self.id_mode:
                items.append({"question": f"What was bound in the configuration step with {la}?",
                              "canonical_short_answer": f"It was {ga}",
                              "link": {"window": lw, "anchor": la},
                              "answer": {"window": gw, "anchor": ga}})
                continue
            items.append({"question": f"What was bound in the configuration step with {la}?",
                          "canonical_short_answer": f"It was {ga}", "link_anchor": la,
                          "gold_anchor": ga, "hops": hops})
        mc.log_raw({"kind": "llm", "prompt": prompt, "raw": "x"})
        return json.dumps({"items": items}), {"prompt_tokens": 1, "completion_tokens": 1}


    def menu(self, prompt, n):
        ents = re.findall(r"^\[(E(\d)[HT]\d+)\] ANCHOR: (\S+)", prompt, re.M)
        low = [e[0] for e in ents if e[1] == "1"]
        high = [e[0] for e in ents if e[1] == "3"]
        items = [{"link_window": low[k + 1], "answer_window": high[k + 1],
                  "question": "What was bound in the configuration step with {LINK} afterwards?",
                  "canonical_short_answer": "something"} for k in range(n)]
        if self.bad_first and self.calls == 1:
            items[0]["question"] = "What was bound in the configuration step?"   # no placeholder
        mc.log_raw({"kind": "llm", "prompt": prompt, "raw": "x"})
        return json.dumps({"items": items}), {"prompt_tokens": 1, "completion_tokens": 1}


class TestEndToEnd(tmc.Base):
    def run_phase(self, chains, tmp, fake, extra=()):
        out = os.path.join(tmp, "mh.jsonl")
        man = os.path.join(tmp, "mh-manifest.json")
        argv = ["--chains", chains, "--out", out, "--manifest-out", man, "--budget-usd", "5",
                "--verdicts", os.path.join(tmp, "verdicts.jsonl"),
                "--save-raw", os.path.join(tmp, "raw-mh.jsonl"), *extra]
        old = (mc.SESS_DIR, mc.llm)
        mc.SESS_DIR, mc.llm = self.sess, fake
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = mh.main(argv)
        finally:
            mc.SESS_DIR, mc.llm = old
        return rc, out, man

    def test_phase_then_verify_menu_mode(self):
        self.phase_then_verify(False, "menu")

    def test_phase_then_verify_verbatim_mode(self):
        self.phase_then_verify(False, "verbatim")

    def test_phase_then_verify_id_mode(self):
        self.phase_then_verify(True, "ids")

    def phase_then_verify(self, id_mode, mode):
        rc, _, chains, man, _ = tmc.gen(self.tmp, "base", self.sess, ["--disjoint"],
                                        max_chains=40)
        base = [json.loads(l) for l in open(chains)]
        fake = ScriptedMH(bad_first=True, id_mode=id_mode)
        rc, out, man2 = self.run_phase(chains, self.tmp, fake,
                                       ["--n", "3", "--mode", mode])
        self.assertEqual(rc, 0)
        recs = [json.loads(l) for l in open(out)]
        self.assertEqual(len(recs), len(base))
        for c, b in zip(recs, base):
            qt = [q.get("qtype") for q in c["qa"]]
            self.assertEqual(qt.count("single"), len(b["qa"]))        # old items tagged single
            self.assertEqual(qt.count("multihop"), 3)
            for q in c["qa"]:
                if q["qtype"] == "multihop":
                    self.assertEqual(len(set(q["span_idxs"])), 2)
        verdicts = [json.loads(l) for l in open(os.path.join(self.tmp, "verdicts.jsonl"))]
        self.assertTrue(any(not v["accepted"] and ("same_span" in v["why"] or
                                                    "link_placeholder" in v["why"])
                            for v in verdicts))
        m = json.load(open(man2))["multihop"]
        self.assertEqual(m["items_accepted"], 3 * len(base))
        self.assertGreater(m["calls"], len(base))                     # the repair round ran
        mix = next(iter(m["final_mix_per_chain"].values()))
        self.assertEqual(mix, {"single": 6, "multihop": 3, "unanswerable": 1})

        # verify: per-qtype counts, multihop gates pass
        rc, rep = tmc.run_verify(self.sess, out, os.path.join(self.tmp, "v.md"))
        self.assertIn("## Per-qtype counts", rep)
        self.assertRegex(rep, rf"multihop {3 * len(base)}")
        self.assertIn("MH-NS", rep)
        self.assertEqual(rc, 0, rep[-1500:])
        self.assertIn("hard-gate failures: 0", rep)
        mh_fail = [l for l in rep.splitlines() if l.startswith("- ") and "MH-" in l]
        self.assertEqual(mh_fail, [])

        # verify catches a tampered multihop item
        bad = copy.deepcopy(recs)
        q = next(x for x in bad[0]["qa"] if x["qtype"] == "multihop")
        q["evidence_spans"][1]["evidence"] += " tampered words that are not in the span"
        q["question"] += " " + q["gold_anchor"]
        p = os.path.join(self.tmp, "bad.jsonl")
        open(p, "w").write("\n".join(json.dumps(x) for x in bad) + "\n")
        rc, rep = tmc.run_verify(self.sess, p, os.path.join(self.tmp, "vb.md"))
        self.assertEqual(rc, 1)
        self.assertRegex(rep, r"MH-EVID")
        self.assertRegex(rep, r"MH-ANS")

    def test_old_dataset_without_qtype_is_all_single(self):
        rc, _, chains, man, _ = tmc.gen(self.tmp, "old", self.sess, ["--disjoint"])
        rc, rep = tmc.run_verify(self.sess, chains, os.path.join(self.tmp, "v.md"))
        self.assertIn("multihop 0", rep)
        self.assertRegex(rep, r"\| \*\*all\*\* \| \d+ \(\d+ pass\) \| 0 \(0 pass\) \|")


if __name__ == "__main__":
    unittest.main()


class TestAnchorHelp(unittest.TestCase):
    def test_suggestions_exclude_contaminated_and_widen(self):
        class M:      # minimal stand-in for the make_chains globals install_anchor_help patches
            MAX_RETRIES_PER_SLOT = 2

            @staticmethod
            def evidence_windows(text, want=8):
                return [f"w{n}" for n in range(want)]

            @staticmethod
            def suggested_anchors(wins, span, others, limit=6, exclude=None):
                return [(1, "tok_clean_1"), (2, "tok_in_fix"), (3, "tok_clean_2")][:limit]

            @staticmethod
            def load_sessions():
                return [{"events": [{"role": "user", "text": "issue about x"},
                                    {"role": "tool", "text": "diff --git a/f b/f\n+tok_in_fix = 1"}]}]
        ohp.install_anchor_help(M)
        self.assertEqual(len(M.evidence_windows("t")), 16)
        self.assertEqual(M.MAX_RETRIES_PER_SLOT, 3)
        self.assertEqual([t for _, t in M.suggested_anchors([], "", [])],
                         ["tok_clean_1", "tok_clean_2"])


class TestReprobe(tmc.Base):
    def test_fits_rejects_foreign_repo_probe(self):
        sph = PY_TRANSCRIPT
        mine = pp.probe_candidates(sph, [], 3)[0]
        self.assertTrue(pp.fits(sph, mine))
        other = "[tool read] in: {\"filePath\": \"/workspace/psf__requests__2.27/requests/api.py\"}\n"
        self.assertFalse(pp.fits(other, "src/_pytest/kestrel_merger.py"))
        self.assertFalse(pp.fits(sph, "NC_PHANTOM_TTL_MS"))

    def test_reprobe_replaces_with_distinct_repo_style_probes(self):
        import oh_salvage as ohs
        rc, _, chains, man, _ = tmc.gen(self.tmp, "rp", self.sess, ["--disjoint"])
        out = os.path.join(self.tmp, "rp.jsonl")
        old = mc.SESS_DIR
        mc.SESS_DIR = self.sess
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ohs.reprobe(["--chains", chains, "--out", out])
        finally:
            mc.SESS_DIR = old
        recs = [json.loads(l) for l in open(out)]
        probes = []
        for c in recs:
            u = [q for q in c["qa"] if not q["answerable"]]
            self.assertEqual(len(u), 1)
            self.assertEqual(u[0]["provenance"]["kind"], "template")
            toks = mc.probe_tokens(u[0]["question"])
            self.assertEqual(len(toks), 1)
            self.assertNotRegex(toks[0], r"^NC_")
            probes.append(toks[0].lower())
        self.assertEqual(len(set(probes)), len(probes))
        rc, rep = tmc.run_verify(self.sess, out, os.path.join(self.tmp, "v.md"))
        self.assertIn("probes used by > 1 item: 0", rep)
