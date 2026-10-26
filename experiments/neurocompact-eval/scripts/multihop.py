#!/usr/bin/env python3
"""Cross-compaction MULTI-HOP QA items (qtype="multihop") for chained-compaction chains.

A chain has three spans (C1, C2, C3 compactions) and questions are asked after C3, so a
fact in span 1 has been summarized three times and a fact in span 3 once.  The `single`
items (make_chains.py) test whether ONE fact survived.  A multi-hop item tests whether the
memory keeps facts from DIFFERENT compaction windows in a form that can be COMBINED.

FROZEN DEFINITION (changing it changes the dataset; bump MH_VERSION)
--------------------------------------------------------------------
A multi-hop item is a question whose answer requires two evidence quotes (hops):

  hop 1 (link)    a verbatim quote of >= 8 words from span i, containing `link_anchor`;
  hop 2 (answer)  a verbatim quote of >= 8 words from span j > i, containing `gold_anchor`;

with i != j (j > i: the question reads "earlier event -> later consequence"), and:

  * the question names the entity of hop 1 (it contains `link_anchor`, token-bounded) and
    asks for the outcome recorded in hop 2; it does NOT contain `gold_anchor` or the answer;
  * `link_anchor` occurs in span i only (1..3 times) and in no other span; `gold_anchor`
    occurs in span j only (1..3 times) and in no other span (same discrimination rule as
    single items); the two anchors differ and differ from every other anchor of the chain;
  * `canonical_short_answer` is a short string (<= 30 words, checkable by the judge);
  * typical shapes: "file X edited in span 1 -> which test failed after that edit in span 3";
    "value of a setting after it was changed"; "a command that failed earlier -> what output
    did it produce when it later succeeded".

Gates (all LLM-free; mirrored in verify_chains.py; names are the rejection reasons):
  shape         hops_count!=2, bad_span_idx, same_span, hop_order, question/answer length
  evidence      mh_ev1/mh_ev2 verbatim in THEIR span (whitespace-normalised, case-sensitive),
                >= 8 words each
  anchors       link: form, in ev1, in question, hits in span i, absent from other spans;
                gold: form, in ev2, hits in span j, absent from other spans, absent from
                question; anchors distinct; answer text not in question
  not_single    LEXICAL one-evidence-alone heuristic (see below)
  contamination oh_pipeline guard on BOTH anchors (anchor in issue / in final diff) and the
                5-word-shingle question-restates-issue rule

"Not answerable from one evidence alone" - what is implemented and its weakness:
  (a) link_anchor must NOT occur in evidence 2: the only text that grounds the question's
      entity is hop 1, so evidence 2 alone cannot even be matched to the question;
  (b) gold_anchor must NOT occur in evidence 1 (nor, by the discrimination rule, anywhere
      in span i), and the canonical answer string must not be a substring of evidence 1:
      hop 1 alone does not contain the answer;
  (c) no concrete token of the question (other than link_anchor) may occur in evidence 2
      but not in evidence 1: the question must not name something that only hop 2 holds.
  GENERATION: the default `menu` mode shows the model only anchor-bearing windows with a
  code-chosen anchor and has it return window IDs + a question containing {LINK}; evidence
  and anchors are copied by code, so the gates above test the question/pairing, not copying.
  (Free-text evidence/anchors from Haiku 5.5 failed 85% of items in a pilot.)  The gates are
  identical in all modes.
  Weakness: these are lexical checks.  They prove each evidence is lexically necessary for
  grounding (A) and for the answer (B), not that the *reasoning* needs both: a question can
  still be answerable from evidence 2 by paraphrase or world knowledge, or the "link" can be
  decorative.  They also over-reject real update-style questions where the same entity
  legitimately appears in both quotes.  Item quality is therefore additionally spot-checked
  by hand (see the run's report), never claimed from the gates alone.

Record fields written for a multi-hop item (superset of the single schema so downstream
consumers keep working): qid (`<chain_id>_mh_<n>`), qtype="multihop", answerable=True,
must_abstain=False, question, canonical_short_answer, gold_anchor (= answer anchor),
link_anchor, evidence (= ev1 + "\\n[...]\\n" + ev2, for display/judging only),
evidence_spans=[{role:"link"|"answer", span_idx, anchor, evidence, slice}], span_idxs=[i, j],
span_idx (= j, the answer span), lag (= lag of the OLDEST evidence span i, i.e. the hardest
hop; hop_lags lists both), slice (half of span j holding ev2), checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import make_chains as mc  # noqa: E402
import oh_pipeline as ohp  # noqa: E402

MH_VERSION = 1
N_DEFAULT = 3
MAX_HITS = mc.MAX_ANCHOR_HITS_OWN_SPAN
MIN_WORDS = mc.MIN_EVIDENCE_WORDS


# ------------------------------------------------------------------ validation

def _anchor_form_ok(a: str) -> bool:
    return len(a) >= 2 and not re.fullmatch(r"\d", a) and a.lower() not in mc.ANCHOR_BAD


def validate_multihop(item: dict, span_texts: dict[int, str], issue: str = "", fix: str = "",
                      taken: set[str] | None = None) -> tuple[bool, list[str], dict]:
    """Pure gate function. item = {question, canonical_short_answer, link_anchor,
    gold_anchor, hops:[{span_idx, evidence}, {span_idx, evidence}]}; span_texts maps
    span_idx -> span text. `taken` = lowercase anchors of the chain's other items."""
    why: list[str] = []
    c: dict[str, bool] = {}
    q = mc.norm_ws(item.get("question") or "")
    ans = mc.norm_ws(item.get("canonical_short_answer") or "")
    link = (item.get("link_anchor") or "").strip()
    gold = (item.get("gold_anchor") or "").strip()
    hops = item.get("hops") or []

    c["question"] = len(q.split()) >= 5
    if not c["question"]:
        why.append("question_too_short")
    c["answer_len"] = bool(ans) and len(ans.split()) <= mc.MAX_Q_WORDS
    if not ans:
        why.append("empty_answer")
    elif len(ans.split()) > mc.MAX_Q_WORDS:
        why.append(f"answer_too_long:{len(ans.split())}w")

    c["hops_count"] = isinstance(hops, list) and len(hops) == 2 \
        and all(isinstance(h, dict) for h in hops)
    if not c["hops_count"]:
        return False, why + ["hops_count"], c
    try:
        i, j = int(hops[0].get("span_idx")), int(hops[1].get("span_idx"))
    except (TypeError, ValueError):
        return False, why + ["bad_span_idx"], c
    ev1, ev2 = hops[0].get("evidence") or "", hops[1].get("evidence") or ""
    if i not in span_texts or j not in span_texts:
        return False, why + ["bad_span_idx"], c
    c["different_spans"] = i != j
    c["hop_order"] = i < j
    if i == j:
        why.append("same_span")
    elif i > j:
        why.append("hop_order")
    A, B = span_texts[i], span_texts[j]

    # evidence: verbatim in ITS span, >= 8 words
    c["ev1_verbatim"], c["ev2_verbatim"] = mc.has_verbatim(A, ev1), mc.has_verbatim(B, ev2)
    c["ev1_words"] = len(mc.norm_ws(ev1).split()) >= MIN_WORDS
    c["ev2_words"] = len(mc.norm_ws(ev2).split()) >= MIN_WORDS
    for n, k in ((1, "ev1_verbatim"), (2, "ev2_verbatim")):
        if not c[k]:
            why.append(f"ev{n}_not_verbatim")
    for n, k in ((1, "ev1_words"), (2, "ev2_words")):
        if not c[k]:
            why.append(f"ev{n}_lt_8_words")

    # link anchor: in ev1, in the question, unique to span i
    c["link_form"] = _anchor_form_ok(link)
    if not c["link_form"]:
        why.append(f"link_bad_form:{link!r}")
    lrx = mc.anchor_re(link) if link else None
    c["link_in_ev1"] = bool(lrx and lrx.search(ev1))
    c["link_in_question"] = bool(lrx and lrx.search(q))
    l_hits = len(lrx.findall(A)) if lrx else 0
    c["link_hits_ok"] = 0 < l_hits <= MAX_HITS
    c["link_only_in_span"] = bool(lrx) and not any(
        lrx.search(t) for k, t in span_texts.items() if k != i)
    if not c["link_in_ev1"]:
        why.append("link_not_in_ev1")
    if not c["link_in_question"]:
        why.append("link_not_in_question")
    if l_hits == 0:
        why.append("link_not_in_span")
    elif l_hits > MAX_HITS:
        why.append(f"link_{l_hits}x_in_span")
    if not c["link_only_in_span"]:
        why.append("link_in_other_span")

    # gold anchor: in ev2, unique to span j, not leaked
    c["gold_form"] = _anchor_form_ok(gold)
    if not c["gold_form"]:
        why.append(f"gold_bad_form:{gold!r}")
    grx = mc.anchor_re(gold) if gold else None
    c["gold_in_ev2"] = bool(grx and grx.search(ev2))
    g_hits = len(grx.findall(B)) if grx else 0
    c["gold_hits_ok"] = 0 < g_hits <= MAX_HITS
    c["gold_only_in_span"] = bool(grx) and not any(
        grx.search(t) for k, t in span_texts.items() if k != j)
    c["gold_not_in_question"] = bool(gold) and gold.lower() not in q.lower()
    c["answer_not_in_question"] = not (ans and len(ans) >= 3 and ans.lower() in q.lower())
    if not c["gold_in_ev2"]:
        why.append("gold_not_in_ev2")
    if g_hits == 0:
        why.append("gold_not_in_span")
    elif g_hits > MAX_HITS:
        why.append(f"gold_{g_hits}x_in_span")
    if not c["gold_only_in_span"]:
        why.append("gold_in_other_span")
    if not c["gold_not_in_question"]:
        why.append("anchor_leaked_in_question")
    if not c["answer_not_in_question"]:
        why.append("answer_in_question")
    c["anchors_distinct"] = bool(link and gold) and link.lower() != gold.lower()
    if not c["anchors_distinct"]:
        why.append("anchors_equal")
    c["anchors_untaken"] = not (taken and ({link.lower(), gold.lower()} & taken))
    if not c["anchors_untaken"]:
        why.append("anchor_duplicated_within_chain")

    # not answerable from one evidence alone (lexical heuristic; see module docstring)
    c["ns_link_not_in_ev2"] = not (lrx and lrx.search(ev2))
    c["ns_gold_not_in_ev1"] = not (grx and grx.search(ev1))
    c["ns_answer_not_in_ev1"] = not (len(ans) >= 3 and ans.lower() in mc.norm_ws(ev1).lower())
    odd = [t for t in mc.probe_tokens(item.get("question") or "")
           if t.lower() != link.lower() and mc.anchor_re(t).search(ev2)
           and not mc.anchor_re(t).search(ev1) and t.lower() not in link.lower()]
    c["ns_question_token_not_only_ev2"] = not odd
    if not c["ns_link_not_in_ev2"]:
        why.append("single_ev2_has_link")
    if not c["ns_gold_not_in_ev1"] or not c["ns_answer_not_in_ev1"]:
        why.append("single_ev1_has_answer")
    if odd:
        why.append("single_question_names_ev2_token:" + odd[0])

    # contamination guard (both anchors + question shingle vs the issue)
    g1 = ohp.guard_reasons({"kind": "answerable", "gold_anchor": gold,
                            "question": item.get("question")}, issue, fix)
    g2 = ohp.guard_reasons({"kind": "answerable", "gold_anchor": link, "question": ""},
                           issue, fix)
    guard = g1 + [r + "_link" for r in g2]
    c["oh_guard"] = not guard
    why += guard

    # soft (reported, never a rejection)
    c["answer_has_gold_anchor"] = bool(gold) and gold.lower() in ans.lower()
    c["slice_of_ev2"] = ("TAIL" if mc.has_verbatim(mc.halves(B)[1], ev2) else "HEAD")
    return (not why), why, c


def item_from_record(qa: dict) -> dict:
    """Stored qa record -> the dict validate_multihop expects."""
    return {"question": qa.get("question"), "canonical_short_answer": qa.get("canonical_short_answer"),
            "link_anchor": qa.get("link_anchor"), "gold_anchor": qa.get("gold_anchor"),
            "hops": [{"span_idx": e.get("span_idx"), "evidence": e.get("evidence")}
                     for e in qa.get("evidence_spans") or []]}


def to_record(item: dict, checks: dict, chain: dict, n: int, spans_by_idx: dict) -> dict:
    (i, ev1), (j, ev2) = [(int(h["span_idx"]), h["evidence"]) for h in item["hops"]]
    lag = lambda k: mc.CHAIN_LEN - k + 1  # noqa: E731
    return {
        "qid": f"{chain['chain_id']}_mh_{n}", "chain_id": chain["chain_id"],
        "session_id": chain["session_id"], "qtype": "multihop",
        "span_idx": j, "lag": lag(i), "hop_lags": [lag(i), lag(j)], "span_idxs": [i, j],
        "slice": checks.get("slice_of_ev2", "HEAD"),
        "answerable": True, "must_abstain": False,
        "question": item["question"], "canonical_short_answer": item["canonical_short_answer"],
        "gold_anchor": item["gold_anchor"], "link_anchor": item["link_anchor"],
        "evidence": ev1 + "\n[...]\n" + ev2,
        "evidence_spans": [
            {"role": "link", "span_idx": i, "anchor": item["link_anchor"], "evidence": ev1,
             "slice": "TAIL" if mc.has_verbatim(mc.halves(spans_by_idx[i])[1], ev1) else "HEAD"},
            {"role": "answer", "span_idx": j, "anchor": item["gold_anchor"], "evidence": ev2,
             "slice": checks.get("slice_of_ev2", "HEAD")}],
        "checks": checks, "mh_version": MH_VERSION,
    }


# ----------------------------------------------------------------- generation

def window_table(chain: dict, half_chars: int, exclude: set[str] | None = None) -> dict[str, dict]:
    """Window id (as shown in the prompt: E<span><H|T><n>) -> {span_idx, slice, text,
    anchors}.  Mirrors make_chains.span_block exactly (same halves, same truncation, same
    evidence_windows / suggested_anchors), so an id the model copies resolves to the exact
    verbatim text it was shown.  Selecting windows by id removes the dominant failure of
    free-text evidence (reformatted / truncated quotes: 51 of 140 rejections)."""
    table: dict[str, dict] = {}
    for sp in chain["spans"]:
        head, tail = mc.halves(sp["text"])
        if len(head) > half_chars:
            head = head[:half_chars] + "\n[...HEAD truncated...]"
        if len(tail) > half_chars:
            tail = "[...TAIL truncated...]\n" + tail[-half_chars:]
        others = [o["text"] for o in chain["spans"] if o["span_idx"] != sp["span_idx"]]
        for sl, txt in (("HEAD", head), ("TAIL", tail)):
            wins = mc.evidence_windows(txt)
            sugg = mc.suggested_anchors(wins, sp["text"], others, exclude=exclude)
            for n, w in enumerate(wins, 1):
                table[f"E{sp['span_idx']}{sl[0]}{n}"] = {
                    "span_idx": sp["span_idx"], "slice": sl, "text": w,
                    "anchors": [t for k, t in sugg if k == n]}
    return table


def resolve_ids(item: dict, table: dict[str, dict]) -> tuple[dict | None, str]:
    """id-mode item {question, canonical_short_answer, link:{window,anchor},
    answer:{window,anchor}} -> validate_multihop's dict (evidence copied from the table)."""
    hops, anchors = [], []
    for k in ("link", "answer"):
        h = item.get(k) or {}
        w = table.get(str(h.get("window") or "").strip())
        if w is None:
            return None, f"unknown_window:{k}"
        hops.append({"span_idx": w["span_idx"], "evidence": w["text"]})
        anchors.append(mc.strip_wrappers(h.get("anchor") or ""))
    return {"question": item["question"],
            "canonical_short_answer": item["canonical_short_answer"],
            "link_anchor": anchors[0], "gold_anchor": anchors[1], "hops": hops}, ""


MENU_MAX_CHARS = 420


def menu_entries(table: dict[str, dict], taken: set[str]) -> list[dict]:
    """Windows that carry a usable code-chosen anchor: >= 5 chars, has a letter, not taken.
    The anchor is the first suggested token of the window, so it is unique to its span by
    construction (suggested_anchors) and the model never has to type one."""
    out = []
    for wid, w in table.items():
        for t in w["anchors"]:
            if len(t) >= 5 and re.search(r"[A-Za-z]", t) and t.lower() not in taken:
                out.append({"id": wid, "span_idx": w["span_idx"], "slice": w["slice"],
                            "anchor": t, "text": w["text"]})
                break
    return out


def menu_prompt(chain: dict, n: int, menu: list[dict], used: list[str], note: str = "") -> str:
    lines = []
    for sp in chain["spans"]:
        lines.append(f"=== SPAN {sp['span_idx']} (older = smaller number) ===")
        for m in menu:
            if m["span_idx"] == sp["span_idx"]:
                txt = m["text"] if len(m["text"]) <= MENU_MAX_CHARS \
                    else m["text"][:MENU_MAX_CHARS] + " [...]"
                lines.append(f"[{m['id']}] ANCHOR: {m['anchor']}\n{txt}")
    return (
        "You write exam items about ONE long coding session that was compacted 3 times. Below "
        "are excerpts (windows) of its three spans, oldest first. Each window has an ID and a "
        "fixed ANCHOR token that occurs in that window (and nowhere in the other spans).\n\n"
        + "\n\n".join(lines) + f"\n\nProduce EXACTLY {n} MULTI-HOP items in one JSON object. "
        "Each combines a fact from an EARLIER window (link_window) with a fact from a LATER "
        "span (answer_window), e.g. \"file X was edited early; which test failed after that "
        "edit\", \"what value did a setting have after it was changed\", \"a command failed "
        "earlier; what did it print when it later succeeded\". Answering must need BOTH.\n\n"
        + note + MENU_RULES + ohp.GUARD_RULES +
        f"\n\n[ANCHORS ALREADY USED] {', '.join(sorted(used)) or '(none)'}\n\n"
        'Schema: {"items":[{"link_window":"E1H3","answer_window":"E3T2","question":"... '
        '{LINK} ...","canonical_short_answer":"..."}]}\n\n'
        f"Session title: {chain['title'][:200] or chain['session_id']}")


MENU_RULES = """MENU RULES
- You do NOT copy evidence or anchors; you give only window IDs. link_window's span number must
  be SMALLER than answer_window's span number.
- The question MUST contain the literal placeholder {LINK} exactly once, standing for the
  ANCHOR token of link_window (the code substitutes it), e.g. "After the agent touched {LINK},
  which test run printed the failure that mentions an unexpected keyword?" Read the link
  window so the question is about what happens in it (the earlier event) and ask for the
  OUTCOME stated in answer_window. The question must not contain any other token that only the
  answer window contains, nor the answer_window's ANCHOR, nor the answer itself.
- "canonical_short_answer": <= 20 words, stated in answer_window (include its ANCHOR if natural).
- Do not use the same window twice across items; use varied span pairs where possible.
- Never ask about the issue statement, the root cause, the upstream fix or the final patch.
- Output JSON only.
"""


def resolve_menu(item: dict, menu: list[dict]) -> tuple[dict | None, str]:
    by_id = {m["id"]: m for m in menu}
    lw, aw = by_id.get(str(item.get("link_window") or "").strip()), \
        by_id.get(str(item.get("answer_window") or "").strip())
    if lw is None or aw is None:
        return None, "unknown_window"
    q = item.get("question") or ""
    if q.count("{LINK}") != 1:
        return None, "link_placeholder"
    return {"question": mc.strip_wrappers(q.replace("{LINK}", lw["anchor"])),
            "canonical_short_answer": item.get("canonical_short_answer") or "",
            "link_anchor": lw["anchor"], "gold_anchor": aw["anchor"],
            "hops": [{"span_idx": lw["span_idx"], "evidence": lw["text"]},
                     {"span_idx": aw["span_idx"], "evidence": aw["text"]}]}, ""


def parse_mh(raw: str) -> list[dict]:
    if not raw:
        return []
    lo, hi = raw.find("{"), raw.rfind("}")
    if lo < 0 or hi <= lo:
        return []
    for kw in ({}, {"strict": False}):
        try:
            j = json.loads(raw[lo:hi + 1], **kw)
        except Exception:  # noqa: BLE001
            continue
        out = []
        for it in j.get("items") or []:
            if not isinstance(it, dict):
                continue
            hops = it.get("hops") or []
            if "link_window" in it or "answer_window" in it:                # menu mode
                out.append({"link_window": it.get("link_window"),
                            "answer_window": it.get("answer_window"),
                            "question": it.get("question") or "",
                            "canonical_short_answer": mc.strip_wrappers(
                                it.get("canonical_short_answer") or "")})
                continue
            if not hops and (it.get("link") or it.get("answer")):       # id mode
                out.append({
                    "question": mc.strip_wrappers(it.get("question") or ""),
                    "canonical_short_answer": mc.strip_wrappers(
                        it.get("canonical_short_answer") or it.get("answer_text") or ""),
                    "link": it.get("link") if isinstance(it.get("link"), dict) else {},
                    "answer": it.get("answer") if isinstance(it.get("answer"), dict) else {}})
                continue
            out.append({
                "question": mc.strip_wrappers(it.get("question") or ""),
                "canonical_short_answer": mc.strip_wrappers(it.get("canonical_short_answer")
                                                            or it.get("answer") or ""),
                "link_anchor": mc.strip_wrappers(it.get("link_anchor") or ""),
                "gold_anchor": mc.strip_wrappers(it.get("gold_anchor") or ""),
                "hops": [{"span_idx": h.get("span_idx"),
                          "evidence": mc.strip_wrappers(h.get("evidence") or "")}
                         for h in hops if isinstance(h, dict)],
            })
        return out
    return []


MH_RULES = """MULTI-HOP ITEM RULES
- Each item combines TWO facts from TWO DIFFERENT spans (hop 1 "link" in an EARLIER span, hop 2
  "answer" in a LATER span): e.g. "file X was edited in span 1; which test failed after that
  edit in span 3", "what value did a setting have after it was changed", "a command failed
  earlier; what output did it produce when it later succeeded". The answer must need BOTH facts.
- You do NOT type any evidence. You pick windows by ID from the [CANDIDATE EVIDENCE] lists
  (e.g. "E1T4") and the code copies their text. Pick ONLY windows that have a token listed in
  [SUGGESTED ANCHORS] (the "E1T4->token" lines) and take the anchor from exactly that line.
- "link": {"window": ID in the EARLIER span, "anchor": a token suggested for THAT window}.
  The QUESTION MUST contain this anchor verbatim (it points at the earlier event). The anchor
  must not be needed in the answer window.
- "answer": {"window": ID in a LATER span (span number greater than the link's), "anchor": a
  token suggested for THAT window that proves the answer}. The question MUST NOT contain this
  anchor nor the answer text.
- The question must not mention "span", "log", "above", must be standalone, and must not
  contain any other concrete identifier that appears only in the answer window.
- "canonical_short_answer": <= 20 words, the exact checkable fact.
- All anchors of the items must be pairwise different and different from [ANCHORS ALREADY USED].
- Use DIFFERENT pairs of spans for the items where possible, and a variety of shapes.
- Output JSON only. No prose, no markdown fence.
"""


def mh_prompt(chain: dict, n: int, half_chars: int, used: list[str], context_note: str = "") -> str:
    blocks = "\n\n".join(
        mc.span_block(s, half_chars,
                      [o["text"] for o in chain["spans"] if o["span_idx"] != s["span_idx"]],
                      exclude={a.lower() for a in used} or None)
        for s in chain["spans"])
    return (
        "You write exam items about ONE long coding session that was compacted 3 times in a "
        "row. Below are the three spans, oldest first (span 1 is the oldest).\n\n"
        f"{blocks}\n\n"
        f"Produce EXACTLY {n} MULTI-HOP items in one JSON object, each combining facts from two "
        "different spans.\n\n" + context_note +
        f"[ANCHORS ALREADY USED]\n{', '.join(sorted(used)) or '(none)'}\n\n"
        + MH_RULES + ohp.GUARD_RULES + "\n\n"
        'Schema: {"items":[{"question":"...","canonical_short_answer":"...",'
        '"link":{"window":"E1H3","anchor":"..."},"answer":{"window":"E3T2","anchor":"..."}}]}\n\n'
        f"Session title: {chain['title'][:200] or chain['session_id']}")


def mh_repair_prompt(chain, rejected: list[tuple[int, str]], half_chars, used) -> str:
    asks = "\n".join(f"[REJECTED {k}] {w}" for k, w in rejected)
    note = (f"Your previous {len(rejected)} item(s) failed validation:\n{asks}\n"
            "Write replacement items (different facts if the cause was a shared anchor) and "
            "obey every rule literally: windows by ID only, anchors only from the suggested "
            "lines of that window, link anchor in the question, answer anchor not in it.\n\n")
    return mh_prompt(chain, len(rejected), half_chars, used, note)


# ------------------------------------------------------------------- driver

def run_phase(a) -> int:
    chains = [json.loads(l) for l in open(a.chains) if l.strip()]
    sessions = {s["sid"]: s for s in mc.load_sessions()}
    spend = mc.Spend(a.budget_usd)
    mc.RAW_LOG["path"] = a.save_raw
    if a.save_raw:
        open(a.save_raw, "w").close()
    verdicts = open(a.verdicts, "w") if a.verdicts else None
    key = os.environ.get("OPENROUTER_API_KEY", "")
    stats = {"calls": 0, "items_proposed": 0, "items_accepted": 0, "rejections": {},
             "chains_complete": 0}
    out_chains = []
    for c in chains:
        s = sessions[c["session_id"]]
        fresh = mc.build_chain(s, c["chain_idx"], a.segs, check=False)
        stored = [[(g["role"], g["text"]) for g in sp["segments"]] for sp in c["spans"]]
        rebuilt = [[(e.get("role", ""), e.get("text") or "") for e in sp["segments"]]
                   for sp in fresh["spans"]]
        if stored != rebuilt:
            raise SystemExit(f"{c['chain_id']}: stored spans differ from rebuild")
        span_texts = {sp["span_idx"]: sp["text"] for sp in fresh["spans"]}
        issue, fix = ohp.contaminated_text(s)
        for q in c["qa"]:
            q.setdefault("qtype", "single")
        have = [q for q in c["qa"] if q["qtype"] == "multihop"]
        c["qa"] = [q for q in c["qa"] if q["qtype"] != "multihop"] if a.redo else c["qa"]
        have = [] if a.redo else have
        taken = {(q.get("gold_anchor") or "").strip().lower()
                 for q in c["qa"] if q.get("gold_anchor")}
        taken |= {(q.get("link_anchor") or "").strip().lower()
                  for q in c["qa"] if q.get("link_anchor")}
        taken.discard("")
        got = list(have)
        need, rejected_last = a.n - len(got), []
        print(f"{c['chain_id']}: multihop have {len(got)} need {need}", flush=True)
        for rnd in range(a.rounds + 1):
            if need <= 0:
                break
            table = window_table(fresh, a.half_chars, {x for x in taken})
            menu = menu_entries(table, taken)
            if a.mode == "menu":
                note = "" if rnd == 0 else (
                    "Your previous item(s) failed validation:\n" + "\n".join(
                        f"- {w}" for _, w in rejected_last) + "\nWrite replacements.\n\n")
                prompt = menu_prompt(fresh, need, menu, sorted(taken), note)
            else:
                prompt = (mh_prompt(fresh, need, a.half_chars, sorted(taken)) if rnd == 0
                          else mh_repair_prompt(fresh, rejected_last, a.half_chars,
                                                sorted(taken)))
            est = sum(sp["chars"] for sp in fresh["spans"]) / mc.CHARS_PER_TOKEN * mc.PRICE_IN \
                + 6000 * mc.PRICE_OUT + 0.002
            before = mc.RAW_LOG["n"]
            raw, _ = mc.llm(prompt, key, a.model, spend, max_tokens=a.max_tokens,
                            est_cost=est, temperature=0.2)
            call_no = mc.RAW_LOG["n"] if mc.RAW_LOG["n"] > before else None
            stats["calls"] += 1
            items = parse_mh(raw)
            rejected_last = []
            if not items:
                stats["rejections"]["no_parseable_items"] = \
                    stats["rejections"].get("no_parseable_items", 0) + 1
                rejected_last = [(k + 1, "no parseable items returned (JSON only!)")
                                 for k in range(need)]
            for it in items:
                stats["items_proposed"] += 1
                if "hops" not in it:
                    res, err = (resolve_menu(it, menu) if "link_window" in it
                                else resolve_ids(it, table))
                    if res is None:
                        it = dict(it, _resolve_error=err)
                    else:
                        it = res
                if need <= 0:
                    verdict = {"accepted": False, "why": ["surplus_item"]}
                elif it.get("_resolve_error"):
                    ok, why, checks = False, [it["_resolve_error"]], {}
                    verdict = {"accepted": False, "why": why}
                    for w in why:
                        k = w.split(":")[0]
                        stats["rejections"][k] = stats["rejections"].get(k, 0) + 1
                    rejected_last.append((len(rejected_last) + 1, "; ".join(why)))
                else:
                    ok, why, checks = validate_multihop(it, span_texts, issue, fix, taken)
                    verdict = {"accepted": ok, "why": why}
                    if ok:
                        n = len(got) + 1
                        rec = to_record(it, checks, c, n, span_texts)
                        got.append(rec)
                        taken |= {it["gold_anchor"].strip().lower(),
                                  it["link_anchor"].strip().lower()}
                        need -= 1
                        stats["items_accepted"] += 1
                    else:
                        for w in why:
                            k = w.split(":")[0]
                            stats["rejections"][k] = stats["rejections"].get(k, 0) + 1
                        rejected_last.append((len(rejected_last) + 1, "; ".join(why)[:300]))
                if verdicts:
                    verdicts.write(json.dumps({"chain_id": c["chain_id"], "round": rnd,
                                               "call": call_no, "item": it, **verdict},
                                              ensure_ascii=False) + "\n")
                    verdicts.flush()
            if need > 0 and len(rejected_last) < need:     # model returned too few items
                rejected_last += [(len(rejected_last) + k + 1, "missing item") for k in
                                  range(need - len(rejected_last))]
            rejected_last = rejected_last[:need]
        c["qa"] = c["qa"] + [g for g in got if g not in c["qa"]]
        c["multihop"] = {"version": MH_VERSION, "target": a.n, "accepted": len(got),
                         "model": a.model,
                         "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        stats["chains_complete"] += int(len(got) >= a.n)
        out_chains.append(c)
        with open(a.out, "w") as w:
            for x in out_chains:
                w.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"  -> {len(got)}/{a.n} multihop; spend ${spend.spent:.4f}", flush=True)
    if verdicts:
        verdicts.close()
    man = {}
    if a.manifest_in and os.path.exists(a.manifest_in):
        man = json.load(open(a.manifest_in))
    man["multihop"] = dict(stats, version=MH_VERSION, target_per_chain=a.n, model=a.model,
                           spend_usd=round(spend.spent, 5), budget_usd=a.budget_usd,
                           final_mix_per_chain={
                               x["chain_id"]: {t: sum(1 for q in x["qa"] if q_type(q) == t)
                                               for t in ("single", "multihop", "unanswerable")}
                               for x in out_chains})
    if a.manifest_out:
        json.dump(man, open(a.manifest_out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in man["multihop"].items() if k != "final_mix_per_chain"}))
    return 0


def q_type(q: dict) -> str:
    """Report bucket: single / multihop / unanswerable (qtype stays single|multihop)."""
    if not q.get("answerable", True):
        return "unanswerable"
    return q.get("qtype") or "single"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="oh_pipeline.py ... multihop")
    ap.add_argument("--chains", required=True, help="chains.jsonl from `make` (not modified)")
    ap.add_argument("--out", required=True, help="chains.jsonl WITH qtype + multihop items")
    ap.add_argument("--manifest-in", default="")
    ap.add_argument("--manifest-out", default="")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--n", type=int, default=N_DEFAULT, help="multihop items per chain")
    ap.add_argument("--rounds", type=int, default=3, help="repair rounds after the first call")
    ap.add_argument("--segs", type=int, default=8)
    ap.add_argument("--half-chars", type=int, default=mc.PROMPT_HALF_CHARS)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--budget-usd", type=float, default=0.5)
    ap.add_argument("--save-raw", default="")
    ap.add_argument("--verdicts", default="", help="JSONL: every proposed item + accept/reject why")
    ap.add_argument("--mode", choices=("menu", "ids", "verbatim"), default="menu",
                    help="menu: model picks window IDs + writes a {LINK} question, code supplies "
                         "evidence and anchors (default); ids: model picks windows+anchors; "
                         "verbatim: model types evidence+anchors (all verified identically)")
    ap.add_argument("--windows", type=int, default=40, help="candidate windows per half-span")
    ap.add_argument("--suggestions", type=int, default=20, help="suggested anchors per half-span")
    ap.add_argument("--redo", action="store_true", help="drop existing multihop items first")
    a = ap.parse_args(argv)
    ohp.install_anchor_help(mc, windows=a.windows, suggestions=a.suggestions)
    return run_phase(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
