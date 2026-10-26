"""Offline tests: window-targeted `menu` mode of gen_longsession_qa (synthetic sessions only, scripted LLM)."""
from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from collections import Counter

import ls_fixtures as F
import gen_longsession_qa as G
import longsession as L

KW = dict(window=14000, keep_tail=3000, reserve=500, use_real=False, n_single=8, n_multihop=2,
          n_unanswerable=2, seed=3, max_calls=60, budget_usd=5.0)


def make_menu_session(n: int = 120, bridge: bool = True) -> dict:
    """Each message carries anchors unique to it (SHARDnnnn, a part file, a row count) and one identifier
    that repeats in every message (cache_layer_v2). SHARD0010 is repeated in message 100 (so it is NOT
    window-unique); `bridge_core_cfg` links messages 5 and 95 (two different windows)."""
    s = F.make_session(n=n, questions=False)
    for i, m in enumerate(s["messages"]):
        head = (f"Step {i}: the worker finished shard SHARD{i:04d} and wrote {1000 + 7 * i} rows into "
                f"/data/out/part{i:04d}.parquet after {(i % 9) + 1}.{i % 7} seconds while the shared module "
                f"cache_layer_v2 stayed warm today. ")
        if bridge and i in (5, 95):
            head += "The run reused bridge_core_cfg for its settings. "
        if i == 100:
            head += "Earlier shard SHARD0010 was mentioned again here. "
        m["text"] = (head + F.FILLER * 40)[:1800]
    s["title"] = "synthetic menu session"
    return s


class MenuLLM:
    """Scripted model for menu prompts. mode: good | leak | badid | present_probe."""
    model, price = "scripted", (1.0, 5.0)

    def __init__(self, mode="good"):
        self.mode, self.cost, self.calls, self.prompts = mode, 0.0, 0, []

    def complete(self, prompt, *, system=None, max_tokens=0):
        self.calls += 1
        self.prompts.append(prompt)
        items = []
        if "multi-hop exam items" in prompt:
            for m in re.finditer(r"\[(P\d+)\] shared ENTITY: (\S+)\nHOP 1 \(window (\d+)\) LINK: (\S+)\n.*?"
                                 r"HOP 2 \(window (\d+), later\) GOLD: (\S+)\n", prompt, re.S):
                pid, ent, wa, link, wb, gold = m.groups()
                items.append({"pair": pid, "question": f"After the agent handled {{LINK}}, which result did the later "
                              f"run that reused {ent} report in its summary?", "answer": f"It reported {gold}."})
        elif "unanswerable" in prompt:
            toks = ["qzx_unseen_flag_77", "NoSuchModule_Xr9"] if self.mode != "present_probe" \
                else ["cache_layer_v2", "x"]
            items = [{"question": f"What did the agent set for {t} when it tuned things?", "absent_token": t}
                     for t in toks]
        else:
            for m in re.finditer(r"\[(W\d+C\d+)\] \((\w+)\) ANCHOR: (\S+)\nEVIDENCE: (.*)\n", prompt):
                cid, role, anchor, ev = m.groups()
                q = "What identifier did the worker use for the unit of work it completed in that stretch?"
                if self.mode == "leak":
                    q = f"What happened with {anchor} during that stretch of the session?"
                if self.mode == "badid":
                    cid = "W99C9"
                items.append({"id": cid, "question": q, "answer": anchor})
            n = int(re.search(r"Choose (\d+) DIFFERENT", prompt).group(1))
            items = items[:n]
        return {"text": json.dumps({"items": items}), "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.0,
                "stop_reason": "stop", "latency_s": 0.0, "model": "scripted"}


class TokenTests(unittest.TestCase):
    def test_extract_kinds(self):
        t = ("see /srv/app/conf/main.yaml and src/pkg/mod.py then commit 9fceb02d41a took 41.7ms with --max-retries "
             "on v2.31.4 raising ConnectionResetError and ENOENT at worker_pool_size 123456 plus 77MB")
        got = {tok: kind for (_, _, kind, tok) in G.extract_tokens(t)}
        self.assertEqual(got["/srv/app/conf/main.yaml"], "path")
        self.assertEqual(got["src/pkg/mod.py"], "path")
        self.assertEqual(got["9fceb02d41a"], "hash")
        self.assertEqual(got["41.7ms"], "numunit")
        self.assertEqual(got["--max-retries"], "flag")
        self.assertEqual(got["v2.31.4"], "version")
        self.assertEqual(got["ConnectionResetError"], "err")
        self.assertEqual(got["ENOENT"], "err")
        self.assertEqual(got["worker_pool_size"], "ident")
        self.assertEqual(got["123456"], "num")
        self.assertEqual(got["77MB"], "numunit")
        spans = sorted((s, e) for (s, e, _, _) in G.extract_tokens(t))
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])))        # no overlaps

    def test_plain_words_and_years_are_not_anchors(self):
        toks = [x[3] for x in G.extract_tokens("In 2024 we simply wrote the code and the test passed, true or false, 12 items")]
        self.assertEqual(toks, [])

    def test_evidence_bounds_word_aligned_and_min_words(self):
        text = "Header line\nThe job wrote results into the archive named QX_4471_final before the nightly sweep ran. Tail."
        s = text.index("QX_4471_final")
        a, b = G.evidence_bounds(text, s, s + len("QX_4471_final"))
        ev = text[a:b]
        self.assertIn("QX_4471_final", ev)
        self.assertGreaterEqual(len(ev.split()), 8)
        self.assertEqual(ev, ev.strip())
        self.assertIsNone(G.evidence_bounds("short QX_4471_final text", 6, 19))


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.s = make_menu_session()
        self.tok = L.message_tokens(self.s["messages"])
        self.wins = G.windows_for(self.s, 14000, 3000, 500, False, self.tok)
        self.toks, self.idx = G.build_token_index(self.s)
        self.cands = G.window_candidates(self.s, self.wins, self.toks, self.idx, seed=1)

    def test_every_window_gets_verified_unique_candidates(self):
        self.assertGreaterEqual(len(self.wins), 4)
        for w, cs in enumerate(self.cands, 1):
            self.assertGreaterEqual(len(cs), min(5, self.wins[w - 1][1] - self.wins[w - 1][0]), f"window {w}")
            lo, hi = self.wins[w - 1]
            for c in cs:
                self.assertEqual(G.verify_candidate(self.s, self.wins, c), [])
                hits = G.anchor_hits(self.s, c["anchor"])          # recomputed independently of the index
                self.assertLessEqual(len(hits), 3)
                self.assertTrue(all(lo <= h < hi for h in hits), (c["anchor"], hits, lo, hi))
                m = self.s["messages"][c["msg_idx"]]["text"]
                self.assertEqual(m[c["start"]:c["end"]], c["ev_text"])
                self.assertGreaterEqual(len(c["ev_text"].split()), 8)

    def test_repeated_identifiers_are_never_anchors(self):
        anchors = {c["anchor"].lower() for cs in self.cands for c in cs}
        self.assertNotIn("cache_layer_v2", anchors)             # in every message
        self.assertNotIn("shard0010", anchors)                  # repeats in message 100 (another window)
        self.assertTrue(any(a.startswith("shard") or a.startswith("/data/out/part") for a in anchors))
        # one candidate per anchor across the whole session
        flat = [c["anchor"].lower() for cs in self.cands for c in cs]
        self.assertEqual(len(flat), len(set(flat)))

    def test_anchor_repeating_within_its_window_up_to_3_messages_is_allowed_but_4_is_not(self):
        s = make_menu_session(bridge=False)
        for i in (2, 3, 4):
            s["messages"][i]["text"] = "the agent ran the nightly batch again and hit TokenBudgetError once more today"
        toks, idx = G.build_token_index(s)
        wins = G.windows_for(s, 14000, 3000, 500, False)
        flat = lambda cs: {c["anchor"].lower() for ws in cs for c in ws}
        self.assertIn("tokenbudgeterror", flat(G.window_candidates(s, wins, toks, idx, k=3000)))
        s["messages"][5]["text"] = "the agent ran the nightly batch again and hit TokenBudgetError once more today"
        toks, idx = G.build_token_index(s)
        self.assertNotIn("tokenbudgeterror", flat(G.window_candidates(s, wins, toks, idx, k=3000)))

    def test_candidate_diversity_per_message(self):
        for cs in self.cands:
            per = Counter(c["msg_idx"] for c in cs)
            self.assertLessEqual(max(per.values()), 2)

    def test_verify_candidate_flags_tampering(self):
        c = dict(self.cands[0][0])
        self.assertEqual(G.verify_candidate(self.s, self.wins, c), [])
        self.assertIn("evidence_not_verbatim_or_short", G.verify_candidate(self.s, self.wins, dict(c, ev_text="x y")))
        other = dict(self.cands[1][0], w=1)                      # claims window 1, lives in window 2
        self.assertIn("evidence_outside_window", G.verify_candidate(self.s, self.wins, other))
        self.assertIn("anchor_taken", G.verify_candidate(self.s, self.wins, c, {c["anchor"].lower()}))


class AllocationTests(unittest.TestCase):
    def test_two_per_window_up_to_forty(self):
        a = G.allocate_singles(22, 40)
        self.assertEqual(sum(a), 40)
        self.assertTrue(all(1 <= x <= 2 for x in a))
        a = G.allocate_singles(30, 40)
        self.assertEqual(sum(a), 40)
        self.assertTrue(all(x in (1, 2) for x in a))

    def test_small_cap_still_covers_all_strata(self):
        a = G.allocate_singles(21, 6)
        chosen = [i + 1 for i, n in enumerate(a) if n]
        self.assertEqual(sum(a), 6)
        self.assertEqual({G.stratum(w, 21) for w in chosen}, {"early", "mid", "late"})

    def test_few_windows_get_more_each_but_capped(self):
        self.assertEqual(G.allocate_singles(3, 12), [4, 4, 4])
        self.assertEqual(G.allocate_singles(2, 40, per_window_max=4), [4, 4])
        self.assertEqual(G.allocate_singles(0, 5), [])
        self.assertEqual(sum(G.allocate_singles(5, 7)), 7)

    def test_strata(self):
        self.assertEqual([G.stratum(w, 9) for w in range(1, 10)], ["early"] * 3 + ["mid"] * 3 + ["late"] * 3)


class PairTests(unittest.TestCase):
    def setUp(self):
        self.s = make_menu_session()
        self.wins = G.windows_for(self.s, 14000, 3000, 500, False)
        toks, idx = G.build_token_index(self.s)
        self.cands = G.window_candidates(self.s, self.wins, toks, idx, seed=1)

    def test_pairs_link_two_windows_through_a_shared_entity_in_both_contexts(self):
        pairs = G.find_pairs(self.cands, max_pairs=6)
        self.assertTrue(pairs)
        for p in pairs:
            c1, c2 = p["c1"], p["c2"]
            self.assertLess(c1["w"], c2["w"])
            self.assertIn(p["entity"], ("bridge_core_cfg", "shard0010"))
            erx = G._mc().anchor_re(p["entity"])
            self.assertTrue(erx.search(c1["ctx"]) and erx.search(c2["ctx"]))
            self.assertNotEqual(c1["anchor"].lower(), c2["anchor"].lower())
            # link anchor only in the earlier window, gold only in the later one
            lo1, hi1 = self.wins[c1["w"] - 1]
            lo2, hi2 = self.wins[c2["w"] - 1]
            self.assertTrue(all(lo1 <= h < hi1 for h in G.anchor_hits(self.s, c1["anchor"])))
            self.assertTrue(all(lo2 <= h < hi2 for h in G.anchor_hits(self.s, c2["anchor"])))
        # anchors are not reused across pairs
        flat = [x for p in pairs for x in (p["c1"]["anchor"].lower(), p["c2"]["anchor"].lower())]
        self.assertEqual(len(flat), len(set(flat)))

    def test_no_shared_entity_no_pairs(self):
        s = make_menu_session(bridge=False)
        toks, idx = G.build_token_index(s)
        cands = G.window_candidates(s, G.windows_for(s, 14000, 3000, 500, False), toks, idx, seed=1)
        self.assertEqual(G.find_pairs(cands, max_pairs=5), [])

    def test_entity_present_in_most_windows_is_too_generic(self):
        s = make_menu_session(bridge=False)
        for i in range(0, 120, 4):
            s["messages"][i]["text"] += " shared generic_helper_obj used"
        toks, idx = G.build_token_index(s)
        cands = G.window_candidates(s, G.windows_for(s, 14000, 3000, 500, False), toks, idx, seed=1)
        self.assertEqual(G.find_pairs(cands, max_pairs=5), [])


class GenerateMenuTests(unittest.TestCase):
    def setUp(self):
        self.s = make_menu_session()

    def test_accepts_menu_items_with_exact_spans(self):
        llm = MenuLLM()
        qs, st = G.generate_menu(self.s, llm, **KW)
        self.assertEqual(st["single_ok"], 8)
        self.assertGreaterEqual(st["multi_ok"], 1)
        self.assertEqual(st["probe_ok"], 2)
        self.assertEqual(st["accepted"], len(qs))
        self.assertLessEqual(st["calls"], llm.calls)
        self.assertGreater(st["accept_per_call"], 1.0)               # several accepted items per call
        wins = G.windows_for(self.s, 14000, 3000, 500, False)
        taken = set()
        for q in qs:
            if not q["answerable"]:
                self.assertEqual(q["evidence"], [])
                self.assertEqual(q["qtype"], "single")
                tok = q["meta"]["probe_token"]
                self.assertNotIn(tok.lower(), "\n".join(m["text"] for m in self.s["messages"]).lower())
                continue
            for e in q["evidence"]:
                quote = self.s["messages"][e["msg_idx"]]["text"][e["start"]:e["end"]]
                self.assertGreaterEqual(len(quote.split()), 8)
            self.assertNotIn(q["gold_anchor"].lower(), q["question"].lower())
            self.assertNotIn(q["gold_anchor"].lower(), taken)
            taken.add(q["gold_anchor"].lower())
            self.assertEqual(q["meta"]["mode"], "menu")
        singles = [q for q in qs if q["qtype"] == "single" and q["answerable"]]
        strata = {q["meta"]["stratum"] for q in singles}
        self.assertEqual(strata, {"early", "mid", "late"} & strata)
        self.assertGreaterEqual(len({q["meta"]["window"] for q in singles}), len(wins) - 1)   # spread over windows
        mh = next(q for q in qs if q["qtype"] == "multihop")
        self.assertEqual([e["role"] for e in mh["evidence"]], ["link", "answer"])
        self.assertLess(mh["evidence"][0]["msg_idx"], mh["evidence"][1]["msg_idx"])
        self.assertIn(mh["meta"]["link_anchor"], mh["question"])
        self.assertNotIn("{LINK}", mh["question"])
        self.assertIn(mh["meta"]["shared_entity"], ("bridge_core_cfg", "shard0010"))
        self.assertEqual(len(set(mh["meta"]["windows"])), 2)
        self.s["questions"] = qs
        self.assertEqual(L.validate_session(self.s), [])

    def test_deterministic(self):
        a, _ = G.generate_menu(self.s, MenuLLM(), **KW)
        b, _ = G.generate_menu(self.s, MenuLLM(), **KW)
        self.assertEqual([q["qid"] + q["question"] for q in a], [q["qid"] + q["question"] for q in b])

    def test_leaking_unknown_and_present_probe_items_are_rejected(self):
        qs, st = G.generate_menu(self.s, MenuLLM("leak"), **{**KW, "n_multihop": 0, "max_calls": 12})
        self.assertEqual(st["single_ok"], 0)
        self.assertIn("anchor_in_question", st["single_rejected"])
        qs, st = G.generate_menu(self.s, MenuLLM("badid"), **{**KW, "n_multihop": 0, "max_calls": 12})
        self.assertEqual(st["single_ok"], 0)
        self.assertIn("unknown_id", st["single_rejected"])
        qs, st = G.generate_menu(self.s, MenuLLM("present_probe"), **{**KW, "n_multihop": 0, "n_single": 0})
        self.assertEqual(st["probe_ok"], 0)
        self.assertIn("probe_present_in_session", st["probe_rejected"])

    def test_menu_prompt_shows_code_picked_anchors_not_free_text_excerpts(self):
        llm = MenuLLM()
        G.generate_menu(self.s, llm, **{**KW, "n_multihop": 0, "n_unanswerable": 0})
        p = llm.prompts[0]
        self.assertRegex(p, r"\[W\d+C1\] \(\w+\) ANCHOR: \S+\nEVIDENCE: ")
        self.assertRegex(p, r"window \d+ of 6")
        self.assertLessEqual(len(re.findall(r"ANCHOR: ", p)), G.MENU_SHOW)

    def test_retry_round_refills_windows_that_fell_short(self):
        class Flaky(MenuLLM):
            def complete(self, prompt, **kw):
                r = super().complete(prompt, **kw)
                if self.calls <= 2:                                  # the first calls return junk
                    r["text"] = "no json here"
                return r
        qs, st = G.generate_menu(self.s, Flaky(), **{**KW, "n_multihop": 0, "n_unanswerable": 0})
        self.assertEqual(st["single_ok"], 8)

    def test_multihop_answer_must_be_grounded_and_question_must_have_one_placeholder(self):
        pair = G.find_pairs(G.window_candidates(self.s, G.windows_for(self.s, 14000, 3000, 500, False),
                                                *G.build_token_index(self.s), seed=1), max_pairs=1)[0]
        wins = G.windows_for(self.s, 14000, 3000, 500, False)
        gold = pair["c2"]["anchor"]
        good = {"question": "After the agent handled {LINK}, which result did the later run report?",
                "answer": f"It reported {gold}."}
        why, q = G.validate_menu_multi(good, pair, self.s, wins, set())
        self.assertEqual(why, [], why)
        self.assertIn(pair["c1"]["anchor"], q)
        for patch, reason in (({"question": "No placeholder at all in this question text?"}, "link_placeholder"),
                              ({"question": "Handled {LINK} and {LINK} together, what was reported later on?"}, "link_placeholder"),
                              ({"question": f"After {{LINK}}, did {gold} show up in the later run?"}, "anchor_leaked_in_question"),
                              ({"answer": "Something entirely invented zorblax quuxification"}, "answer_unsupported"),
                              ({"answer": ""}, "empty_answer")):
            why, _ = G.validate_menu_multi(dict(good, **patch), pair, self.s, wins, set())
            self.assertTrue(any(w.startswith(reason) for w in why), (patch, why))
        # a pair whose shared entity is NOT in both contexts is refused
        why, _ = G.validate_menu_multi(good, dict(pair, entity="nowhere_to_be_found"), self.s, wins, set())
        self.assertIn("entity_not_in_both_windows", why)
        # answer anchor must live only in the later window
        bad2 = dict(pair, c2=dict(pair["c2"], hits=pair["c2"]["hits"] + [0]))
        why, _ = G.validate_menu_multi(good, bad2, self.s, wins, set())
        self.assertIn("hop2_anchor_outside_window", why)

    def test_validate_menu_single_gates(self):
        wins = G.windows_for(self.s, 14000, 3000, 500, False)
        toks, idx = G.build_token_index(self.s)
        c = G.window_candidates(self.s, wins, toks, idx, seed=1)[0][0]
        good = {"question": "Which identifier labelled the unit of work completed during that stretch?", "answer": c["anchor"]}
        self.assertEqual(G.validate_menu_single(good, c, set()), [])
        copy = " ".join(c["ev_text"].split()[:6])
        for patch, reason in (({"question": f"What was {c['anchor']} doing there at the time?"}, "anchor_in_question"),
                              ({"question": f"Why did it say: {copy} exactly?"}, "question_copies_evidence"),
                              ({"answer": "w " * 40}, "answer_too_long"),
                              ({"answer": "an unrelated fabricated statement about zorblax"}, "answer_unsupported"),
                              ({"question": "Too short?"}, "question_too_short")):
            self.assertTrue(any(w.startswith(reason) for w in G.validate_menu_single(dict(good, **patch), c, set())),
                            patch)
        self.assertIn("anchor_taken", G.validate_menu_single(good, c, {c["anchor"].lower()}))

    def test_validate_probe(self):
        full = "alpha beta cache_layer_v2 gamma"
        self.assertEqual(G.validate_probe({"question": "What did the agent set for qzx_unseen_flag_77 here?",
                                           "absent_token": "qzx_unseen_flag_77"}, full), [])
        self.assertIn("probe_present_in_session", G.validate_probe(
            {"question": "What did the agent set for cache_layer_v2 here today?", "absent_token": "cache_layer_v2"}, full))
        self.assertIn("probe_not_in_question", G.validate_probe(
            {"question": "What did the agent set for something here today?", "absent_token": "qzx_unseen_flag_77"}, full))
        self.assertIn("probe_bad_form", G.validate_probe({"question": "What did the agent set here today?",
                                                          "absent_token": "ab"}, full))


class BudgetPrecheckTests(unittest.TestCase):
    def setUp(self):
        self.s = make_menu_session()

    def test_local_estimate_never_lets_a_call_overshoot_the_cap(self):
        llm = MenuLLM()
        llm.price = (50.0, 250.0)                     # $/MTok: ~$0.35 worst case for a 1.4k-token answer + prompt

        class Costly(MenuLLM):
            pass
        orig = llm.complete

        def complete(prompt, **kw):                   # the real cost == the worst case it was gated on
            r = orig(prompt, **kw)
            llm.cost += (len(prompt) / 3.0 + 64) * llm.price[0] / 1e6 + kw["max_tokens"] * llm.price[1] / 1e6
            return r
        llm.complete = complete
        cap = 1.0
        qs, st = G.generate_menu(self.s, llm, **{**KW, "budget_usd": cap, "max_calls": 60})
        self.assertGreaterEqual(st["cap_refused"], 1)
        self.assertLessEqual(llm.cost, cap + 1e-9)
        self.assertGreaterEqual(st["calls"], 1)
        self.assertEqual(st["precheck"], "local-estimate")

    def test_llm_worst_case_hook_is_used_and_exact_arithmetic_holds(self):
        class Priced(MenuLLM):
            def worst_case(self, prompt, max_tokens):
                return 0.40

            def complete(self, prompt, **kw):
                r = super().complete(prompt, **kw)
                self.cost += 0.40
                return r
        llm = Priced()
        qs, st = G.generate_menu(self.s, llm, **{**KW, "budget_usd": 1.0})
        self.assertEqual(llm.calls, 2)                # 0.40 + 0.40 fit; a third could reach 1.20 > 1.00
        self.assertLessEqual(llm.cost, 1.0)
        self.assertGreaterEqual(st["cap_refused"], 1)
        self.assertEqual(st["precheck"], "LLM.worst_case")

    def test_shared_budget_remaining_is_respected(self):
        class FakeBudget:
            remaining = 0.05

        class W(MenuLLM):
            budget = FakeBudget()

            def worst_case(self, prompt, max_tokens):
                return 0.10
        llm = W()
        qs, st = G.generate_menu(self.s, llm, **KW)
        self.assertEqual(llm.calls, 0)
        self.assertEqual(qs, [])

    def test_phase_shares_leave_money_for_multihop_and_probes(self):
        class Priced(MenuLLM):
            def worst_case(self, prompt, max_tokens):
                return 0.10

            def complete(self, prompt, **kw):
                r = super().complete(prompt, **kw)
                self.cost += 0.10
                return r
        llm = Priced()
        qs, st = G.generate_menu(self.s, llm, **{**KW, "budget_usd": 1.0, "n_single": 40, "per_window_max": 4})
        self.assertLessEqual(llm.cost, 1.0 + 1e-9)
        self.assertGreaterEqual(st["multi_ok"] + st["probe_ok"], 1)       # singles did not eat the whole cap

    def test_max_calls_is_honoured(self):
        llm = MenuLLM()
        qs, st = G.generate_menu(self.s, llm, **{**KW, "max_calls": 3})
        self.assertEqual(llm.calls, 3)
        self.assertEqual(st["stopped"], "max_calls")


class RealWindowTests(unittest.TestCase):
    def test_real_compactions_give_exactly_the_dropped_spans_and_no_final_window(self):
        s = F.make_session(questions=False, real=True)            # real boundaries at 30, 70, 100
        self.assertEqual(G.windows_for(s, 14000, 3000, 500, True), [(0, 30), (30, 70), (70, 100)])
        self.assertGreater(len(G.windows_for(s, 14000, 3000, 500, False)), 0)


class HomeRunRedactionTests(unittest.TestCase):
    def test_bundle_log_staging_uses_ls_redact_and_keeps_log_lines_readable(self):
        import home_run as H
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "cc.qa.log"), os.path.join(d, "out", "cc.qa.log")
            key = "Zx9fQ2mK7pLw4RtY8vNb"
            with open(src, "w") as f:
                f.write("tokenizer=tiktoken:cl100k_base max_tokens=4096\nAPI_KEY=" + key + "\n")
            r = H.stage_file(src, dst, "logs/cc.qa.log", [], (), True)
            out = open(dst).read()
            self.assertIn("tokenizer=tiktoken:cl100k_base max_tokens=4096", out)
            self.assertNotIn(key, out)
            self.assertEqual(r["counts"], {"assign": 1})


class CLITests(unittest.TestCase):
    def test_dry_run_menu_prints_plan_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            sd = os.path.join(d, "in")
            F.write_session(sd, make_menu_session())
            out = os.path.join(d, "out")
            self.assertEqual(G.main(["--sessions", sd, "--out", out, "--dry-run", "--window-tokens", "14000",
                                     "--keep-tail", "3000", "--reserve", "500", "--n-single", "8"]), 0)
            self.assertFalse(os.path.exists(os.path.join(out, "qa-manifest.json")))

    def test_menu_is_the_default_and_legacy_is_a_flag(self):
        with tempfile.TemporaryDirectory() as d:
            sd = os.path.join(d, "in")
            F.write_session(sd, make_menu_session())
            base = ["--sessions", sd, "--provider", "mock", "--model", "mock", "--window-tokens", "14000",
                    "--keep-tail", "3000", "--reserve", "500", "--max-calls", "3"]
            o1, o2 = os.path.join(d, "o1"), os.path.join(d, "o2")
            self.assertEqual(G.main(base + ["--out", o1]), 0)
            self.assertEqual(G.main(base + ["--out", o2, "--mode", "legacy"]), 0)
            m1 = json.load(open(os.path.join(o1, "qa-manifest.json")))
            m2 = json.load(open(os.path.join(o2, "qa-manifest.json")))
            self.assertEqual(m1["sessions"][0]["mode"], "menu")
            self.assertNotIn("mode", m2["sessions"][0])
            raw = [json.loads(l) for l in open(os.path.join(o1, "qa-raw.jsonl"))]
            self.assertTrue(raw and all(r["mode"] == "menu" and "worst_case_usd" in r for r in raw))


if __name__ == "__main__":
    unittest.main()
