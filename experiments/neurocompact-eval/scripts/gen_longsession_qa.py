#!/usr/bin/env python3
"""QA generator for longsession/1 sessions that carry no questions (e.g. Claude Code imports).

TWO MODES. ``--mode menu`` (default): per compaction window (the dropped span between two boundaries) the
CODE extracts anchors that occur nowhere else in the session (paths, identifiers, numbers with units,
error names, commit hashes, flags, versions; <= 3 messages, all inside the window) each with a verbatim
>= 8-word evidence sentence; the model gets a small menu per window and only phrases the question.
Multihop = two such anchors in different windows joined by a shared entity the code verifies in both
evidence contexts. Calls are allocated over all windows (~2 single per window up to --n-single 40, golden-
ratio order so early/mid/late are all hit), ~10 multihop in pairs of 2 per call, a few unanswerable probes
(plausible token, verified absent from the whole session), retries last. Every call is pre-checked against
the worst-case cost (LLM.worst_case / providers.Budget.reserve, else a local estimate) so the cap is never
exceeded. ``--mode legacy``: the original free-text excerpt mode described below (post-hoc gates; wasteful
on long research sessions where identifiers repeat).

Provider-agnostic: any providers.py profile (anthropic | openrouter | gemini | custom) or the offline mock,
via longsession.LLM (role qa_generator; prices/keys/hard budget from providers.py). Builds on the machinery of
make_chains.py instead of re-inventing the gates:
  * text helpers  make_chains.norm_ws / has_verbatim / anchor_re / strip_wrappers / probe_tokens
  * multi-hop gate (validate_multihop below = the gates of multihop.validate_multihop, the FROZEN
    definition, minus its OpenHands contamination guard, re-implemented here so this script has
    no dependency on multihop.py/oh_pipeline.py): two verbatim >= 8-word quotes from two
    DIFFERENT spans, link_anchor in the question, gold_anchor only in the later span, lexical
    "not answerable from one evidence alone"; here a "span" = the dropped region
    of one compaction boundary (the session's real compactions with --use-real-compactions, else
    synthetic boundaries by --window-tokens/--keep-tail like run_longsession).
  * single gate (new, long-session sized): the verbatim quote (>= 8 words) must lie in ONE
    message, gold_anchor must be token-bounded in it, absent from the question, occur in <= 3
    messages of the whole session and ONLY inside the evidence's window; the answer must be
    short (<= 30 words) and not contained in the question.

Evidence is stored as ``{"msg_idx", "start", "end"}`` (exact char span of the quote inside the
message text, found with a whitespace-flexible match), so any consumer can re-check it. Questions
are asked only about the DROPPED regions (never the final tail), spread round-robin over the
windows so lags vary. Every LLM prompt + raw answer is appended to ``<out>/qa-raw.jsonl``
(full artifacts), accepted/rejected counts go to ``<out>/qa-manifest.json``.

Usage
  ANTHROPIC_API_KEY=... python3 gen_longsession_qa.py --sessions DIR --out OUT --profile anthropic \\
      --model-qa-generator claude-sonnet-5-5 --n-single 12 --n-multihop 4 --budget-usd 5.0
  (--budget-usd = HARD cap for the whole run, mandatory for a live run; --session-budget-usd
   optionally caps one session; --provider / --model are kept as aliases)
  python3 gen_longsession_qa.py --sessions DIR --out OUT --dry-run        # no network (menu: real menus, planned counts, worst-case $)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import providers as P  # noqa: E402
from longsession import (LLM, LLMError, compute_boundaries, fmt_usd,  # noqa: E402
                         load_sessions, message_tokens, save_session, serialize, validate_session)

MIN_WORDS = 8
MAX_ANSWER_WORDS = 30
MAX_ANCHOR_MSGS = 3
SHOW_CAP = 2000          # chars of one message shown to the generator
_IDENT = re.compile(r"[A-Za-z_][\w./:-]*\d[\w./:-]*|\b\d+(?:[.,]\d+)+%?|[A-Za-z_]+(?:[._/-][A-Za-z0-9_]+)+")


def _mc():
    import make_chains as mc  # noqa: PLC0415
    return mc


def _anchor_form_ok(a: str) -> bool:
    mc = _mc()
    return len(a) >= 2 and not re.fullmatch(r"\d", a) and a.lower() not in mc.ANCHOR_BAD


def validate_multihop(item: dict, span_texts: dict[int, str], taken: set[str] | None = None
                      ) -> tuple[bool, list[str]]:
    """Gates of multihop.validate_multihop (frozen definition) without the contamination guard.
    item = {question, canonical_short_answer, link_anchor, gold_anchor, hops:[{span_idx, evidence}]*2}."""
    mc = _mc()
    why: list[str] = []
    q = mc.norm_ws(item.get("question") or "")
    ans = mc.norm_ws(item.get("canonical_short_answer") or "")
    link = (item.get("link_anchor") or "").strip()
    gold = (item.get("gold_anchor") or "").strip()
    hops = item.get("hops") or []
    if len(q.split()) < 5:
        why.append("question_too_short")
    if not ans:
        why.append("empty_answer")
    elif len(ans.split()) > mc.MAX_Q_WORDS:
        why.append(f"answer_too_long:{len(ans.split())}w")
    if not (isinstance(hops, list) and len(hops) == 2 and all(isinstance(h, dict) for h in hops)):
        return False, why + ["hops_count"]
    try:
        i, j = int(hops[0].get("span_idx")), int(hops[1].get("span_idx"))
    except (TypeError, ValueError):
        return False, why + ["bad_span_idx"]
    ev1, ev2 = hops[0].get("evidence") or "", hops[1].get("evidence") or ""
    if i not in span_texts or j not in span_texts:
        return False, why + ["bad_span_idx"]
    if i == j:
        why.append("same_span")
    elif i > j:
        why.append("hop_order")
    A, B = span_texts[i], span_texts[j]
    for n, (sp, ev) in enumerate(((A, ev1), (B, ev2)), 1):
        if not mc.has_verbatim(sp, ev):
            why.append(f"ev{n}_not_verbatim")
        if len(mc.norm_ws(ev).split()) < MIN_WORDS:
            why.append(f"ev{n}_lt_8_words")
    if not _anchor_form_ok(link):
        why.append(f"link_bad_form:{link!r}")
    if not _anchor_form_ok(gold):
        why.append(f"gold_bad_form:{gold!r}")
    lrx = mc.anchor_re(link) if link else None
    grx = mc.anchor_re(gold) if gold else None
    if not (lrx and lrx.search(ev1)):
        why.append("link_not_in_ev1")
    if not (lrx and lrx.search(q)):
        why.append("link_not_in_question")
    l_hits = len(lrx.findall(A)) if lrx else 0
    if l_hits == 0:
        why.append("link_not_in_span")
    elif l_hits > MAX_ANCHOR_MSGS:
        why.append(f"link_{l_hits}x_in_span")
    if lrx and any(lrx.search(t) for k, t in span_texts.items() if k != i):
        why.append("link_in_other_span")
    if not (grx and grx.search(ev2)):
        why.append("gold_not_in_ev2")
    g_hits = len(grx.findall(B)) if grx else 0
    if g_hits == 0:
        why.append("gold_not_in_span")
    elif g_hits > MAX_ANCHOR_MSGS:
        why.append(f"gold_{g_hits}x_in_span")
    if grx and any(grx.search(t) for k, t in span_texts.items() if k != j):
        why.append("gold_in_other_span")
    if gold and gold.lower() in q.lower():
        why.append("anchor_leaked_in_question")
    if len(ans) >= 3 and ans.lower() in q.lower():
        why.append("answer_in_question")
    if link and gold and link.lower() == gold.lower():
        why.append("anchors_equal")
    if taken and ({link.lower(), gold.lower()} & taken):
        why.append("anchor_duplicated")
    # not answerable from one evidence alone (lexical heuristic, see multihop.py)
    if lrx and lrx.search(ev2):
        why.append("single_ev2_has_link")
    if (grx and grx.search(ev1)) or (len(ans) >= 3 and ans.lower() in mc.norm_ws(ev1).lower()):
        why.append("single_ev1_has_answer")
    odd = [t for t in mc.probe_tokens(item.get("question") or "")
           if t.lower() != link.lower() and mc.anchor_re(t).search(ev2)
           and not mc.anchor_re(t).search(ev1) and t.lower() not in link.lower()]
    if odd:
        why.append("single_question_names_ev2_token:" + odd[0])
    return (not why), why


# ------------------------------------------------------------------ windows / excerpts
def windows_for(session: dict, window: int, keep_tail: int, reserve: int, use_real: bool,
                tok: list[int] | None = None) -> list[tuple[int, int]]:
    tok = tok or message_tokens(session["messages"])
    real = session.get("compactions") if use_real else None
    bs = compute_boundaries(tok, window=window, keep_tail=keep_tail, reserve=reserve,
                            real=real, final=not use_real)   # real: the last real boundary is the end checkpoint (run_longsession adds none)
    return [(b["drop_start"], b["at_idx"]) for b in bs]


def pick_excerpt(session: dict, tok: list[int], lo: int, hi: int, target: int,
                 rng: random.Random, tries: int = 8) -> tuple[int, int]:
    """Contiguous message run [s, e) inside [lo, hi) of ~target tokens, preferring runs dense
    in identifiers/numbers (tool output), chosen among `tries` random starts."""
    best, best_score = (lo, min(hi, lo + 1)), -1
    for _ in range(tries):
        s = rng.randint(lo, max(lo, hi - 1))
        e, t = s, 0
        while e < hi and t < target:
            t += tok[e]
            e += 1
        txt = " ".join(m["text"] for m in session["messages"][s:e])
        score = len(_IDENT.findall(txt)) / max(1, len(txt)) * 1000
        if score > best_score:
            best, best_score = (s, e), score
    return best


def render_excerpt(session: dict, s: int, e: int, label: str) -> str:
    out = [f"=== {label} ==="]
    for m in session["messages"][s:e]:
        t = m["text"]
        out.append(f"[#{m['idx']} {m['role']}] {t[:SHOW_CAP]}" + (" ...[cut]" if len(t) > SHOW_CAP else ""))
    return "\n".join(out)


# ------------------------------------------------------------------ evidence location
def flex_regex(quote: str) -> re.Pattern | None:
    parts = quote.split()
    if not parts:
        return None
    return re.compile(r"\s+".join(re.escape(p) for p in parts))


def locate(session: dict, quote: str, lo: int, hi: int, hint: int | None = None) -> dict | None:
    """{"msg_idx","start","end"} of `quote` inside ONE message of [lo, hi) (hint tried first)."""
    rx = flex_regex(quote)
    if rx is None:
        return None
    order = ([hint] if isinstance(hint, int) and lo <= hint < hi else []) \
        + [i for i in range(lo, hi) if i != hint]
    for i in order:
        m = rx.search(session["messages"][i]["text"])
        if m:
            return {"msg_idx": i, "start": m.start(), "end": m.end()}
    return None


# ------------------------------------------------------------------ prompts
RULES = """ITEM RULES
- Ask about transcript-local specifics only: exact file paths, command lines, concrete numbers or
  values in tool output, error messages, names of things the agent created, decisions the user
  stated. A reader who has NOT seen this session must not be able to answer from general knowledge.
- "evidence": a CONTIGUOUS run of >= 8 words copied CHARACTER-FOR-CHARACTER from ONE message (the
  text after "[#N role]"), and "msg_idx" = N of that message. Never join lines, never reformat.
- "gold_anchor": ONE token (identifier, path, number with unit, error name) copied verbatim from the
  evidence that proves the answer. The QUESTION MUST NOT contain the gold_anchor nor the answer.
- "answer": <= 30 words, the exact checkable fact.
- The question must be standalone: no "above", "the log", "message N", "excerpt".
- Output JSON only, no prose, no fence."""

SINGLE_SCHEMA = ('Schema: {"items":[{"question":"...","answer":"...","gold_anchor":"...",'
                 '"msg_idx":N,"evidence":"..."}]}')
MH_RULES = """MULTI-HOP ITEM RULES
- Each item combines TWO facts from TWO DIFFERENT excerpts (hop 1 in the EARLIER excerpt, hop 2 in the
  LATER one); the answer needs BOTH (e.g. "file X was edited in excerpt A; which test failed after that
  edit in excerpt B", "what value did a setting have after it was changed").
- "hops": exactly 2 objects [hop 1, hop 2] = {"span_idx": WINDOW, "msg_idx": N, "evidence": "..."} with
  hop 1's WINDOW < hop 2's WINDOW; evidence = >= 8 words verbatim from ONE message of that window.
- "link_anchor": ONE token verbatim from hop 1's evidence that occurs in hop 1's WINDOW only; the
  QUESTION MUST contain it; it must NOT appear in hop 2's evidence.
- "gold_anchor": ONE token verbatim from hop 2's evidence that occurs in hop 2's WINDOW only, NOT in hop
  1's evidence; the question must NOT contain it nor the answer.
- "canonical_short_answer": <= 20 words.
- Output JSON only."""
MH_SCHEMA = ('Schema: {"items":[{"question":"...","canonical_short_answer":"...","link_anchor":"...",'
             '"gold_anchor":"...","hops":[{"span_idx":1,"msg_idx":N,"evidence":"..."},'
             '{"span_idx":3,"msg_idx":M,"evidence":"..."}]}]}')


def single_prompt(session: dict, excerpt: str, n: int, used: list[str]) -> str:
    return (f"You write exam items about ONE very long agent session (title: {session.get('title', '')[:120]!r}). "
            "Below is an excerpt; each message starts with [#index role].\n\n" + excerpt +
            f"\n\nProduce EXACTLY {n} items in one JSON object.\n[ANCHORS ALREADY USED] "
            f"{', '.join(sorted(used)[:80]) or '(none)'}\n\n" + RULES + "\n" + SINGLE_SCHEMA)


def mh_prompt(session: dict, ex_a: str, ex_b: str, wa: int, wb: int, n: int, used: list[str]) -> str:
    return (f"You write multi-hop exam items about ONE very long agent session (title: "
            f"{session.get('title', '')[:120]!r}). Two excerpts from two DIFFERENT windows follow (window "
            f"{wa} is EARLIER than window {wb}); messages start with [#index role].\n\n"
            f"{ex_a}\n\n{ex_b}\n\nProduce EXACTLY {n} MULTI-HOP item(s) in one JSON object. "
            f"hop 1 span_idx = {wa}, hop 2 span_idx = {wb}.\n[ANCHORS ALREADY USED] "
            f"{', '.join(sorted(used)[:80]) or '(none)'}\n\n" + MH_RULES + "\n" + MH_SCHEMA)


def parse_items(raw: str) -> list[dict]:
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
        its = j.get("items") if isinstance(j, dict) else None
        return [i for i in its if isinstance(i, dict)] if isinstance(its, list) else []
    return []


# ------------------------------------------------------------------ validation
def validate_single(item: dict, session: dict, windows: list[tuple[int, int]], widx: int,
                    taken: set[str]) -> tuple[bool, list[str], dict | None]:
    """(ok, reasons, evidence span). `widx` is the 0-based window the excerpt came from."""
    mc = _mc()
    why: list[str] = []
    q = mc.norm_ws(item.get("question") or "")
    ans = mc.norm_ws(item.get("answer") or "")
    anc = (item.get("gold_anchor") or "").strip()
    ev = (item.get("evidence") or "").strip()
    lo, hi = windows[widx]
    if len(q.split()) < 5:
        why.append("question_too_short")
    if not ans:
        why.append("empty_answer")
    elif len(ans.split()) > MAX_ANSWER_WORDS:
        why.append(f"answer_too_long:{len(ans.split())}w")
    if len(mc.norm_ws(ev).split()) < MIN_WORDS:
        why.append("evidence_lt_8_words")
    hint = item.get("msg_idx") if isinstance(item.get("msg_idx"), int) else None
    span = locate(session, ev, lo, hi, hint) if ev else None
    if span is None:
        why.append("evidence_not_verbatim_in_one_message")
        return False, why, None
    if len(anc) < 2 or anc.lower() in mc.ANCHOR_BAD or re.fullmatch(r"\d", anc):
        why.append("anchor_bad_form")
        return False, why, span
    arx = mc.anchor_re(anc)
    ev_text = session["messages"][span["msg_idx"]]["text"][span["start"]:span["end"]]
    if not arx.search(ev_text):
        why.append("anchor_not_in_evidence")
    if arx.search(q) or anc.lower() in q.lower():
        why.append("anchor_in_question")
    if len(ans) >= 3 and ans.lower() in q.lower():
        why.append("answer_in_question")
    hits = [i for i, m in enumerate(session["messages"]) if arx.search(m["text"])]
    if len(hits) > MAX_ANCHOR_MSGS:
        why.append(f"anchor_in_{len(hits)}_messages")
    if any(not (lo <= i < hi) for i in hits):
        why.append("anchor_outside_window")
    if anc.lower() in taken:
        why.append("anchor_taken")
    return (not why), why, span


def validate_multi(item: dict, session: dict, windows: list[tuple[int, int]],
                   win_text: dict[int, str], taken: set[str]) -> tuple[bool, list[str], list[dict]]:
    hops = item.get("hops") or []
    ok, why = validate_multihop(item, win_text, taken)
    spans: list[dict] = []
    if ok:
        for h in hops:
            lo, hi = windows[int(h["span_idx"]) - 1]
            s = locate(session, h.get("evidence") or "", lo, hi, h.get("msg_idx"))
            if s is None:
                return False, why + ["hop_evidence_not_in_one_message"], []
            spans.append(s)
    return ok, why, spans


# ------------------------------------------------------------------ driver
def estimate_calls(n_single: int, n_mh: int, per_call: int = 3) -> int:
    return -(-n_single // per_call) + -(-n_mh // 2) + 2


def generate(session: dict, llm, *, window: int, keep_tail: int, reserve: int, use_real: bool,
             n_single: int, n_multihop: int, excerpt_tokens: int, seed: int, per_call: int,
             max_calls: int, budget_usd: float, raw_log: str | None = None) -> tuple[list[dict], dict]:
    rng = random.Random(f"{seed}:{session['sid']}")
    cost0 = llm.cost
    tok = message_tokens(session["messages"])
    wins = windows_for(session, window, keep_tail, reserve, use_real, tok)
    stats = {"sid": session["sid"], "windows": len(wins), "calls": 0, "single_ok": 0, "single_rejected": {},
             "multi_ok": 0, "multi_rejected": {}, "cost_usd": 0.0}
    if not wins:
        return [], stats
    win_text = {k + 1: serialize(session["messages"][lo:hi]) for k, (lo, hi) in enumerate(wins)}
    taken: set[str] = set()
    questions: list[dict] = []

    def call(prompt: str, kind: str) -> list[dict]:
        if stats["calls"] >= max_calls or llm.cost - cost0 >= budget_usd:
            return []
        worst = llm.worst_case(prompt, 4096) if hasattr(llm, "worst_case") else None
        if worst is not None and llm.cost - cost0 + worst > budget_usd + 1e-12:
            print(f"  [qa] session cap ${budget_usd:.4f}: skipping a call with worst-case ${worst:.4f} "
                  f"(spent ${llm.cost - cost0:.4f})", flush=True)
            stats["cap_refused"] = stats.get("cap_refused", 0) + 1
            stats["calls"] = max_calls          # the cap is the stop: every generation loop ends
            return []
        stats["calls"] += 1
        try:
            r = llm.complete(prompt, max_tokens=4096)
        except LLMError as e:
            print(f"  [qa] llm error: {str(e)[:160]}", flush=True)
            return []
        if raw_log:
            with open(raw_log, "a") as f:
                f.write(json.dumps({"sid": session["sid"], "kind": kind, "model": r["model"],
                                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                    "prompt": prompt, "raw": r["text"], "usage": {
                                        "in": r["input_tokens"], "out": r["output_tokens"]},
                                    "cost_usd": r["cost_usd"], "stop": r["stop_reason"]},
                                   ensure_ascii=False) + "\n")
        return parse_items(r["text"])

    def bump(d: dict, reasons: list[str]):
        for w in reasons or ["?"]:
            k = w.split(":")[0]
            d[k] = d.get(k, 0) + 1

    k = 0
    single_cap = max(1, max_calls - (2 * n_multihop + 1 if n_multihop else 0))   # keep calls for multihop
    while stats["single_ok"] < n_single and stats["calls"] < single_cap and llm.cost - cost0 < budget_usd:
        w = k % len(wins)
        k += 1
        s, e = pick_excerpt(session, tok, wins[w][0], wins[w][1], excerpt_tokens, rng)
        ex = render_excerpt(session, s, e, f"window {w + 1}")
        want = min(per_call, n_single - stats["single_ok"])
        for it in call(single_prompt(session, ex, want, sorted(taken)), "single"):
            ok, why, span = validate_single(it, session, wins, w, taken)
            if not ok:
                bump(stats["single_rejected"], why)
                continue
            taken.add(it["gold_anchor"].strip().lower())
            questions.append({
                "qid": f"{session['sid']}_s{stats['single_ok'] + 1}", "qtype": "single",
                "question": it["question"].strip(), "answer": it["answer"].strip(), "answerable": True,
                "evidence": [span], "gold_anchor": it["gold_anchor"].strip(),
                "meta": {"window": w + 1, "generator": llm.model}})
            stats["single_ok"] += 1
        if k > max_calls * 3:
            break
    if len(wins) >= 2:
        tries = 0
        while stats["multi_ok"] < n_multihop and stats["calls"] < max_calls and llm.cost - cost0 < budget_usd \
                and tries < max_calls:
            tries += 1
            a, b = sorted(rng.sample(range(len(wins)), 2))
            sa, ea = pick_excerpt(session, tok, wins[a][0], wins[a][1], excerpt_tokens // 2, rng)
            sb, eb = pick_excerpt(session, tok, wins[b][0], wins[b][1], excerpt_tokens // 2, rng)
            p = mh_prompt(session, render_excerpt(session, sa, ea, f"window {a + 1}"),
                          render_excerpt(session, sb, eb, f"window {b + 1}"), a + 1, b + 1, 1,
                          sorted(taken))
            for it in call(p, "multihop"):
                try:
                    for h in it.get("hops") or []:
                        h["span_idx"] = int(h["span_idx"])
                    ok, why, spans = validate_multi(it, session, wins, win_text, taken)
                except (KeyError, TypeError, ValueError, IndexError) as ex_:
                    ok, why, spans = False, [f"malformed:{type(ex_).__name__}"], []
                if not ok:
                    bump(stats["multi_rejected"], why)
                    continue
                taken.update({it["link_anchor"].strip().lower(), it["gold_anchor"].strip().lower()})
                for sp, role in zip(spans, ("link", "answer")):
                    sp["role"] = role
                questions.append({
                    "qid": f"{session['sid']}_m{stats['multi_ok'] + 1}", "qtype": "multihop",
                    "question": it["question"].strip(), "answer": it["canonical_short_answer"].strip(),
                    "answerable": True, "evidence": spans, "gold_anchor": it["gold_anchor"].strip(),
                    "meta": {"link_anchor": it["link_anchor"].strip(), "generator": llm.model,
                             "windows": [int(h["span_idx"]) for h in it["hops"]]}})
                stats["multi_ok"] += 1
    stats["cost_usd"] = round(llm.cost - cost0, 6)
    return questions, stats


# ================================================================== MENU MODE (default)
# The CODE picks anchors; the model only phrases questions. Free-text generation on a long research
# session wastes calls (identifiers repeat, so post-hoc "anchor occurs in <= 3 messages / only in this
# window" gates reject most items: measured 12 single + 1 multihop from 32 calls on a real 2.5M-token
# session). Here every menu entry is already a verified fact: an anchor token that occurs only in this
# compaction window (<= 3 messages, all inside it), with a verbatim >= 8-word evidence sentence around it.
MENU_K = 12              # candidates kept per window (the single-question menu shows MENU_SHOW of them)
MENU_SHOW = 8
CTX_CHARS = 500          # context kept on each side of the evidence (shown to the model, used for bridges)
EV_MAX_CHARS = 420
MH_ANSWER_WORDS = 20
_KIND_W = {"err": 9.0, "path": 8.0, "numunit": 7.0, "hash": 6.5, "flag": 6.0, "version": 5.5,
           "ident": 5.0, "dotted": 5.0, "num": 3.0}
_ENTITY_KINDS = {"path", "ident", "dotted", "err", "flag"}
_UNIT = (r"ms|s|sec|secs|min|h|MB|GB|KB|TB|MiB|GiB|KiB|%|k|M|B|tokens|USD|MHz|GHz|fps|rps|qps|x|"
         r"px|files|tests|lines|msgs|messages|workers|threads|epochs|steps")
_TOK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("err", re.compile(r"\b(?:[A-Z][a-z]+){1,4}(?:Error|Exception|Warning)\b|\bE[A-Z]{3,9}\b")),
    ("path", re.compile(r"(?<![\w/.])(?:~|\.{1,2})?(?:/[\w.@+-]+)+\.\w{1,8}\b"
                        r"|(?<![\w/.])(?:[\w.@+-]+/)+[\w.@+-]*\w\.\w{1,8}\b")),
    ("hash", re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b")),
    ("numunit", re.compile(rf"(?<![\w.])\d[\d,]*(?:\.\d+)?(?:{_UNIT})(?![\w])|(?<![\w.])\d[\d,]*(?:\.\d+)? "
                           rf"(?:ms|sec|seconds|MB|GB|KB|MiB|GiB|tokens|USD|files|tests|lines|messages|workers|threads|epochs|steps)\b")),
    ("version", re.compile(r"\bv?\d+\.\d+\.\d+(?:[-+][\w.]+)?\b")),
    ("flag", re.compile(r"(?<![\w-])--[a-z][a-z0-9]*(?:-[a-z0-9]+)+|(?<![\w-])--[a-z]{5,}")),
    ("ident", re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"
                         r"|\b[A-Za-z]{2,}\d+[A-Za-z0-9]*\b")),
    ("dotted", re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*){2,}\b")),
    ("num", re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])|(?<![\w.,])\d{4,}(?![\w.,])")),
]
_STOP_TOK = {"e.g", "i.e", "utf_8", "__init__", "__main__", "__name__", "node_modules", "true_false"}


def extract_tokens(text: str) -> list[tuple[int, int, str, str]]:
    """Non-overlapping anchor-shaped spans of `text`: (start, end, kind, token). Kinds are tried in
    priority order (a path wins over the snake_case name inside it)."""
    taken: list[tuple[int, int]] = []
    out: list[tuple[int, int, str, str]] = []
    for kind, pat in _TOK_PATTERNS:
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            if any(s < b and a < e for a, b in taken):
                continue
            tok = m.group(0)
            if not (4 <= len(tok) <= 80) or tok.lower() in _STOP_TOK or tok.lower() in _mc().ANCHOR_BAD:
                continue
            if kind in ("ident", "dotted", "num", "hash") and len(tok) < 5:
                continue
            if kind == "num" and re.fullmatch(r"(19|20)\d\d", tok):
                continue
            if text.startswith("[REDACTED", max(0, s - 1)) or "REDACTED" in tok:
                continue
            taken.append((s, e))
            out.append((s, e, kind, tok))
    out.sort()
    return out


def build_token_index(session: dict) -> tuple[list[list[tuple[int, int, str, str]]], dict[str, list[int]]]:
    """Per-message anchor-shaped tokens and tok(lower) -> message indices (capped at 4 entries: a token
    that occurs in > MAX_ANCHOR_MSGS messages is never a candidate, the exact count does not matter)."""
    toks: list[list[tuple[int, int, str, str]]] = []
    idx: dict[str, list[int]] = {}
    for i, m in enumerate(session["messages"]):
        ts = extract_tokens(m["text"])
        toks.append(ts)
        for t in {x[3].lower() for x in ts}:
            lst = idx.setdefault(t, [])
            if len(lst) <= MAX_ANCHOR_MSGS:
                lst.append(i)
    return toks, idx


def evidence_bounds(text: str, s: int, e: int) -> tuple[int, int] | None:
    """Verbatim sentence-ish span of `text` around [s, e): >= MIN_WORDS words, <= ~EV_MAX_CHARS chars,
    word-aligned. None when the whole message is shorter than MIN_WORDS words."""
    n = len(text)
    lo0, hi0 = max(0, s - 220), min(n, e + 220)
    left = [text.rfind("\n", lo0, s)]
    d = text.rfind(". ", lo0, s)
    if d >= 0:
        left.append(d + 1)
    c = max(left)
    lo = c + 1 if c >= 0 else lo0
    right = []
    nl = text.find("\n", e, hi0)
    if nl >= 0:
        right.append(nl)
    d = text.find(". ", e, hi0)
    if d >= 0:
        right.append(d + 1)
    hi = min(right) if right else hi0
    grow = 0
    while len(text[lo:hi].split()) < MIN_WORDS and grow < 8 and (lo > 0 or hi < n):
        grow += 1
        lo, hi = max(0, lo - 80), min(n, hi + 80)
    while hi - lo > EV_MAX_CHARS and lo < s and hi > e:    # shrink toward the anchor
        if s - lo > hi - e:
            lo = min(s, lo + 20)
        else:
            hi = max(e, hi - 20)
    while 0 < lo < s and not text[lo - 1].isspace():       # word-align
        lo += 1
    while e < hi < n and not text[hi].isspace():
        hi -= 1
    while lo < s and text[lo].isspace():
        lo += 1
    while hi > e and text[hi - 1].isspace():
        hi -= 1
    if lo > s or hi < e or len(text[lo:hi].split()) < MIN_WORDS:
        return None
    return lo, hi


def _prose_ratio(s: str) -> float:
    w = s.split()
    return sum(bool(re.search(r"[A-Za-z]{3,}", x)) for x in w) / max(1, len(w))


def anchor_hits(session: dict, anchor: str) -> list[int]:
    arx = _mc().anchor_re(anchor)
    return [i for i, m in enumerate(session["messages"]) if arx.search(m["text"])]


def verify_candidate(session: dict, wins: list[tuple[int, int]], cand: dict, taken: set[str] | None = None
                     ) -> list[str]:
    """Reasons a menu candidate is NOT usable (empty = ok): evidence verbatim at its span and >= 8 words,
    anchor token-bounded in the evidence, anchor occurs in <= 3 messages and only inside its window."""
    mc = _mc()
    why: list[str] = []
    lo, hi = wins[cand["w"] - 1]
    txt = session["messages"][cand["msg_idx"]]["text"]
    ev = txt[cand["start"]:cand["end"]]
    if not (lo <= cand["msg_idx"] < hi):
        why.append("evidence_outside_window")
    if ev != cand["ev_text"] or len(mc.norm_ws(ev).split()) < MIN_WORDS:
        why.append("evidence_not_verbatim_or_short")
    if not _anchor_form_ok(cand["anchor"]) or not mc.anchor_re(cand["anchor"]).search(ev):
        why.append("anchor_not_in_evidence")
    hits = cand["hits"] if cand.get("hits") is not None else anchor_hits(session, cand["anchor"])
    if len(hits) > MAX_ANCHOR_MSGS:
        why.append(f"anchor_in_{len(hits)}_messages")
    if any(not (lo <= i < hi) for i in hits) or cand["msg_idx"] not in hits:
        why.append("anchor_outside_window")
    if taken and cand["anchor"].lower() in taken:
        why.append("anchor_taken")
    return why


def window_candidates(session: dict, wins: list[tuple[int, int]], toks, idx, *, k: int = MENU_K,
                      seed: int = 0) -> list[list[dict]]:
    """Per window (0-based list) up to k verified candidates, best first, diverse in message and kind."""
    out: list[list[dict]] = []
    mc = _mc()
    for w, (lo, hi) in enumerate(wins, 1):
        pool = []
        for i in range(lo, hi):
            for (s, e, kind, tok) in toks[i]:
                hits = idx.get(tok.lower()) or []
                if not hits or len(hits) > MAX_ANCHOR_MSGS or any(not (lo <= h < hi) for h in hits):
                    continue
                if hits[0] != i:                       # one candidate per token: its first message
                    continue
                if not _anchor_form_ok(tok):
                    continue
                cheap = _KIND_W.get(kind, 1) + (0.8 if len(hits) == 1 else 0.0) \
                    + (0.5 if 6 <= len(tok) <= 40 else 0.0) \
                    + (int(hashlib.md5(f"{seed}:{tok}".encode()).hexdigest()[:4], 16) / 65535) * 1.5
                pool.append((cheap, i, s, e, kind, tok, hits))
        pool.sort(key=lambda x: -x[0])
        picked: list[dict] = []
        per_msg: dict[int, int] = {}
        per_kind: dict[str, int] = {}
        for cheap, i, s, e, kind, tok, hits in pool[:max(400, k * 40)]:
            if len(picked) >= k:
                break
            if per_msg.get(i, 0) >= 1 or per_kind.get(kind, 0) >= max(3, k // 2):
                continue
            text = session["messages"][i]["text"]
            b = evidence_bounds(text, s, e)
            if b is None:
                continue
            ev = text[b[0]:b[1]]
            pr = _prose_ratio(ev)
            if pr < 0.5:
                continue
            cand = {"id": f"W{w}C{len(picked) + 1}", "w": w, "anchor": tok, "kind": kind, "msg_idx": i,
                    "start": b[0], "end": b[1], "ev_text": ev, "hits": hits, "score": round(cheap + 2 * pr, 3),
                    "ctx": text[max(0, b[0] - CTX_CHARS):b[1] + CTX_CHARS]}
            cand["hits"] = anchor_hits(session, tok)             # the REAL uniqueness scan (index is a prefilter)
            if verify_candidate(session, wins, cand):
                continue
            cand["entities"] = sorted({t.lower() for (_, _, kd, t) in extract_tokens(cand["ctx"])
                                       if kd in _ENTITY_KINDS and len(t) >= 5 and t.lower() != tok.lower()})
            picked.append(cand)
            per_msg[i] = per_msg.get(i, 0) + 1
            per_kind[kind] = per_kind.get(kind, 0) + 1
        out.append(picked)
    return out


# ------------------------------------------------------------------ allocation
def allocate_singles(n_windows: int, target: int, per_window_max: int = 4) -> list[int]:
    """Questions per window: even, remainder windows spread over early/mid/late by a golden-ratio
    low-discrepancy order (so a cap below 2 x windows still touches every stratum)."""
    if n_windows <= 0 or target <= 0:
        return [0] * max(0, n_windows)
    base, rem = divmod(target, n_windows)
    counts = [min(per_window_max, base)] * n_windows
    order = sorted(range(n_windows), key=lambda i: ((i + 0.5) * 0.6180339887) % 1.0)
    for i in order[:rem if base < per_window_max else 0]:
        counts[i] = min(per_window_max, counts[i] + 1)
    return counts


def stratum(w: int, n_windows: int) -> str:
    """early / mid / late third of 1-based window w."""
    return ("early", "mid", "late")[min(2, (w - 1) * 3 // max(1, n_windows))]


def find_pairs(cands: list[list[dict]], *, max_pairs: int, seed: int = 0) -> list[dict]:
    """Multihop pairs: (earlier-window candidate c1, later-window candidate c2) linked by a shared entity
    that occurs in the context window of BOTH evidences (code-verified token match). c1.anchor is the
    link (goes into the question, unique to its window), c2.anchor is the gold (unique to its later
    window). Greedy by score with a diversity penalty so windows/strata repeat as little as possible."""
    mc = _mc()
    n = len(cands)
    by_ent: dict[str, dict[int, list[dict]]] = {}
    for ws in cands:
        for c in ws:
            for e in c.get("entities", []):
                by_ent.setdefault(e, {}).setdefault(c["w"], []).append(c)
    raw = []
    for ent, byw in by_ent.items():
        if len(byw) < 2 or (n >= 5 and len(byw) > 0.6 * n):
            continue
        erx = mc.anchor_re(ent)
        loc = []
        ws = sorted(byw)
        for ai, a in enumerate(ws):
            for b in ws[ai + 1:]:
                for c1 in byw[a]:
                    for c2 in byw[b]:
                        if c1["anchor"].lower() == c2["anchor"].lower():
                            continue
                        if ent in (c1["anchor"].lower(), c2["anchor"].lower()):
                            continue
                        if not (erx.search(c1["ctx"]) and erx.search(c2["ctx"])):
                            continue
                        sc = c1["score"] + c2["score"] - 1.2 * math.log(len(byw)) + 0.15 * (b - a)
                        loc.append((sc, ent, c1, c2))
        loc.sort(key=lambda x: (-x[0], x[2]["id"], x[3]["id"]))
        raw.extend(loc[:40])
    raw.sort(key=lambda x: (-x[0], x[1], x[2]["id"], x[3]["id"]))
    out: list[dict] = []
    used_w: dict[int, int] = {}
    used_anchor: set[str] = set()
    used_pair: set[tuple[int, int]] = set()
    while len(out) < max_pairs and raw:
        best, best_sc = None, -1e9
        for sc, ent, c1, c2 in raw[:400]:
            if c1["anchor"].lower() in used_anchor or c2["anchor"].lower() in used_anchor:
                continue
            adj = sc - 2.0 * (used_w.get(c1["w"], 0) + used_w.get(c2["w"], 0)) \
                - (3.0 if (c1["w"], c2["w"]) in used_pair else 0.0)
            if adj > best_sc:
                best, best_sc = (sc, ent, c1, c2), adj
        if best is None:
            break
        sc, ent, c1, c2 = best
        raw = [r for r in raw if r is not best]
        used_anchor.update({c1["anchor"].lower(), c2["anchor"].lower()})
        for c in (c1, c2):
            used_w[c["w"]] = used_w.get(c["w"], 0) + 1
        used_pair.add((c1["w"], c2["w"]))
        out.append({"id": f"P{len(out) + 1}", "entity": ent, "c1": c1, "c2": c2, "score": round(sc, 3)})
    return out


# ------------------------------------------------------------------ prompts (menu)
def _around(c: dict, before: int = 160, after: int = 260) -> str:
    """The text just before/after the evidence (the evidence itself is shown once, not repeated)."""
    txt = c["ctx"]
    i = txt.find(c["ev_text"])
    b = txt[:i][-before:] if i > 0 else ""
    a = txt[i + len(c["ev_text"]):][:after] if i >= 0 else ""
    return (f"BEFORE: ...{_clip(b, before)}\n" if b.strip() else "") + (f"AFTER: {_clip(a, after)}..." if a.strip() else "")


def _clip(t: str, n: int) -> str:
    t = t.replace("\r", "")
    return t if len(t) <= n else t[:n] + " [...]"


MENU_RULES_SINGLE = """RULES
- Choose {n} DIFFERENT entries. For each, write a standalone question whose correct answer is a fact
  stated in that entry's EVIDENCE (usually the ANCHOR itself, or the value/result tied to it).
- The QUESTION MUST NOT contain the ANCHOR, nor the answer, nor copy a run of words from the EVIDENCE.
  Describe the circumstances instead (what the agent was doing, which task/step, what was asked or
  observed) so that someone who saw the session can find the spot, but no one else can answer.
- No "above", "the log", "the excerpt", "window", "entry", no IDs. Plain question of >= 6 words.
- "answer": <= 30 words, the exact checkable fact, taken from the entry (BEFORE/AFTER may help).
- Output JSON only, no prose, no fence."""
MENU_SCHEMA_SINGLE = 'Schema: {"items":[{"id":"W3C2","question":"...","answer":"..."}]}'


def menu_single_prompt(session: dict, w: int, n_windows: int, menu: list[dict], n: int, used: list[str]) -> str:
    lines = []
    for c in menu:
        role = session["messages"][c["msg_idx"]]["role"]
        lines.append(f"[{c['id']}] ({role}) ANCHOR: {c['anchor']}\nEVIDENCE: {_clip(c['ev_text'], EV_MAX_CHARS)}\n"
                     f"{_around(c)}")
    return (f"You write exam items about ONE very long coding-agent session (title: "
            f"{session.get('title', '')[:120]!r}), compacted {n_windows} times. Below is a MENU for compaction "
            f"window {w} of {n_windows} (1 = oldest): each entry has an ID, an ANCHOR token that occurs only in "
            "this window, and the verbatim EVIDENCE sentence around it.\n\n" + "\n\n".join(lines)
            + f"\n\n[ANCHORS ALREADY USED] {', '.join(sorted(used)[:60]) or '(none)'}\n\n"
            + MENU_RULES_SINGLE.format(n=n) + "\n" + MENU_SCHEMA_SINGLE)


MENU_RULES_MH = """RULES
- For EACH pair write ONE multi-hop question needing BOTH facts: hop 1 (earlier) says something about
  the LINK token and the shared ENTITY; hop 2 (later) says what happened to / with that ENTITY afterwards.
- The question MUST contain the literal placeholder {LINK} exactly once, standing for the LINK token of hop 1
  (the code substitutes it), and ask for the OUTCOME stated in hop 2, e.g. "After the agent worked with {LINK},
  what did it later find out about the <entity description>?".
- It must NOT contain hop 2's GOLD token, the answer, or any other identifier only hop 2 contains.
- "answer": <= 20 words, stated in hop 2 (the AFTER/BEFORE text of hop 2 may help). Standalone question; no "above", "hop", "window".
- Output JSON only, no prose, no fence."""
MENU_SCHEMA_MH = 'Schema: {"items":[{"pair":"P1","question":"... {LINK} ...","answer":"..."}]}'


def menu_mh_prompt(session: dict, pairs: list[dict], n_windows: int, used: list[str]) -> str:
    blocks = []
    for p in pairs:
        c1, c2 = p["c1"], p["c2"]
        blocks.append(f"[{p['id']}] shared ENTITY: {p['entity']}\n"
                      f"HOP 1 (window {c1['w']}) LINK: {c1['anchor']}\nEVIDENCE 1: {_clip(c1['ev_text'], EV_MAX_CHARS)}\n"
                      f"{_around(c1)}\n"
                      f"HOP 2 (window {c2['w']}, later) GOLD: {c2['anchor']}\nEVIDENCE 2: {_clip(c2['ev_text'], EV_MAX_CHARS)}\n"
                      f"{_around(c2)}")
    return (f"You write multi-hop exam items about ONE very long coding-agent session (title: "
            f"{session.get('title', '')[:120]!r}), compacted {n_windows} times. Each PAIR below links an EARLIER "
            "and a LATER compaction window through a shared ENTITY that occurs in both contexts.\n\n"
            + "\n\n".join(blocks)
            + f"\n\n[ANCHORS ALREADY USED] {', '.join(sorted(used)[:60]) or '(none)'}\n\n"
            + MENU_RULES_MH + "\n" + MENU_SCHEMA_MH)


PROBE_RULES = """RULES
- Write {n} plausible-sounding questions about SPECIFIC things (a file path, flag, function name, number,
  error name, setting) that fit this session's topic and style but that the session NEVER mentions. A reader
  who has seen the whole session must have to say "that is not in the session".
- Each question contains exactly one such invented specific, copied to "absent_token" (an identifier-like
  string of >= 6 characters: no ordinary English words). The invented token must be unusual enough that
  it cannot have appeared by chance.
- Output JSON only, no prose, no fence."""
PROBE_SCHEMA = 'Schema: {"items":[{"question":"...","absent_token":"..."}]}'


def probe_prompt(session: dict, samples: list[dict], n: int) -> str:
    ex = "\n".join(f"- {c['kind']}: {c['anchor']}  <- {_clip(c['ev_text'], 160)}" for c in samples)
    return (f"You write 'unanswerable' exam probes about ONE very long coding-agent session (title: "
            f"{session.get('title', '')[:120]!r}). Examples of real specifics from it (for style only):\n{ex}\n\n"
            + PROBE_RULES.format(n=n) + "\n" + PROBE_SCHEMA)


# ------------------------------------------------------------------ validation (menu)
def _copies_run(question: str, evidence: str, n: int = 5) -> bool:
    mc = _mc()
    q = [w.lower().strip(".,;:!?\"'`()") for w in mc.norm_ws(question).split()]
    e = " " + " ".join(w.lower().strip(".,;:!?\"'`()") for w in mc.norm_ws(evidence).split()) + " "
    return any(" " + " ".join(q[i:i + n]) + " " in e for i in range(0, max(0, len(q) - n + 1)))


def answer_supported(answer: str, anchor: str, context: str) -> bool:
    """Cheap grounding check: the answer is the anchor, or its concrete tokens / content words occur in
    the evidence context (the model may not invent facts)."""
    mc = _mc()
    a, ctx = mc.norm_ws(answer).lower(), mc.norm_ws(context).lower()
    if anchor.lower() in a:
        return True
    concrete = [t for t in mc.probe_tokens(answer)]
    if concrete:
        return all(t.lower() in ctx for t in concrete)
    words = [w for w in re.findall(r"[a-z]{4,}", a)]
    return bool(words) and sum(w in ctx for w in words) / len(words) >= 0.5


def validate_menu_single(item: dict, cand: dict, taken: set[str]) -> list[str]:
    mc = _mc()
    why: list[str] = []
    q = mc.norm_ws(item.get("question") or "")
    ans = mc.norm_ws(item.get("answer") or "")
    anc = cand["anchor"]
    if len(q.split()) < 5:
        why.append("question_too_short")
    if "{" in q or re.search(r"\b(above|excerpt|the log)\b", q, re.I):
        why.append("question_scaffold")
    if not ans:
        why.append("empty_answer")
    elif len(ans.split()) > MAX_ANSWER_WORDS:
        why.append(f"answer_too_long:{len(ans.split())}w")
    if mc.anchor_re(anc).search(q) or anc.lower() in q.lower():
        why.append("anchor_in_question")
    if len(ans) >= 3 and ans.lower() in q.lower():
        why.append("answer_in_question")
    if _copies_run(q, cand["ev_text"]):
        why.append("question_copies_evidence")
    if ans and not answer_supported(ans, anc, cand["ctx"]):
        why.append("answer_unsupported")
    if anc.lower() in taken:
        why.append("anchor_taken")
    return why


def validate_menu_multi(item: dict, pair: dict, session: dict, wins: list[tuple[int, int]], taken: set[str]
                        ) -> tuple[list[str], str]:
    """(reasons, final question with {LINK} substituted)."""
    mc = _mc()
    c1, c2, ent = pair["c1"], pair["c2"], pair["entity"]
    why: list[str] = []
    raw_q = item.get("question") or ""
    if raw_q.count("{LINK}") != 1:
        return ["link_placeholder"], ""
    q = mc.norm_ws(mc.strip_wrappers(raw_q.replace("{LINK}", c1["anchor"])))
    ans = mc.norm_ws(item.get("answer") or "")
    link, gold = c1["anchor"], c2["anchor"]
    if len(q.split()) < 6:
        why.append("question_too_short")
    if not ans:
        why.append("empty_answer")
    elif len(ans.split()) > MH_ANSWER_WORDS:
        why.append(f"answer_too_long:{len(ans.split())}w")
    if c1["w"] >= c2["w"]:
        why.append("hop_order")
    for n_, c in ((1, c1), (2, c2)):
        for r in verify_candidate(session, wins, c, None):
            why.append(f"hop{n_}_{r}")
    erx = mc.anchor_re(ent)
    if not (erx.search(c1["ctx"]) and erx.search(c2["ctx"])):
        why.append("entity_not_in_both_windows")
    if mc.anchor_re(gold).search(q) or gold.lower() in q.lower():
        why.append("anchor_leaked_in_question")
    if len(ans) >= 3 and ans.lower() in q.lower():
        why.append("answer_in_question")
    if mc.anchor_re(gold).search(c1["ctx"]) or mc.anchor_re(link).search(c2["ctx"]):
        why.append("anchor_in_other_hop")
    if len(ans) >= 3 and ans.lower() in mc.norm_ws(c1["ev_text"]).lower():
        why.append("single_ev1_has_answer")
    if ans and not answer_supported(ans, gold, c2["ctx"]):
        why.append("answer_unsupported")
    if _copies_run(q, c1["ev_text"]) or _copies_run(q, c2["ev_text"]):
        why.append("question_copies_evidence")
    odd = [t for t in mc.probe_tokens(q) if t.lower() != link.lower() and t.lower() != ent
           and mc.anchor_re(t).search(c2["ev_text"]) and not mc.anchor_re(t).search(c1["ev_text"])
           and t.lower() not in link.lower()]
    if odd:
        why.append("single_question_names_ev2_token:" + odd[0])
    if {link.lower(), gold.lower()} & taken:
        why.append("anchor_taken")
    return why, q


def validate_probe(item: dict, full_lower: str) -> list[str]:
    mc = _mc()
    q = mc.norm_ws(item.get("question") or "")
    tok = (item.get("absent_token") or "").strip()
    why: list[str] = []
    if len(q.split()) < 5:
        why.append("question_too_short")
    if len(tok) < 6 or tok.lower() in mc.ANCHOR_BAD:
        why.append("probe_bad_form")
        return why
    if tok.lower() not in q.lower():
        why.append("probe_not_in_question")
    if tok.lower() in full_lower:
        why.append("probe_present_in_session")
    return why


# ------------------------------------------------------------------ budget-aware calls
class Caller:
    """One place for 'call the model': worst-case pre-check against the session cap AND the shared
    providers.Budget (uses LLM.worst_case / Budget.reserve when present, else a local estimate), raw log,
    call counter, per-phase spend limit."""

    def __init__(self, llm, *, cap_usd: float, max_calls: int, raw_log: str | None, sid: str, stats: dict):
        self.llm, self.cap, self.max_calls, self.raw_log, self.sid, self.stats = \
            llm, cap_usd, max_calls, raw_log, sid, stats
        self.cost0 = llm.cost
        self.phase_limit = float("inf")

    @property
    def spent(self) -> float:
        return self.llm.cost - self.cost0

    @property
    def remaining(self) -> float:
        rem = self.cap - self.spent
        b = getattr(self.llm, "budget", None)
        if b is not None:
            rem = min(rem, b.remaining)
        return max(0.0, rem)

    def begin_phase(self, frac: float) -> None:
        """Phase may spend at most `frac` of what is left now (unused money flows to later phases)."""
        self.phase_limit = self.spent + frac * self.remaining

    def worst_case(self, prompt: str, max_tokens: int) -> float:
        w = self.llm.worst_case(prompt, max_tokens) if hasattr(self.llm, "worst_case") else None
        if w is not None:
            return float(w)
        pin, pout = getattr(self.llm, "price", (0.0, 0.0))
        est_in = (len(prompt) / 3.0) + 64                  # pessimistic chars/token, like providers.py
        return (est_in * pin + max_tokens * pout) / 1e6

    def call(self, prompt: str, kind: str, max_tokens: int) -> list[dict] | None:
        """Parsed items, [] when the answer had none, None when the call was NOT made (cap / call limit)."""
        st = self.stats
        if st["calls"] >= self.max_calls:
            st["stopped"] = "max_calls"
            return None
        worst = self.worst_case(prompt, max_tokens)
        if self.spent + worst > min(self.cap, self.phase_limit) + 1e-12 \
                or worst > self.remaining + 1e-12:
            st["cap_refused"] = st.get("cap_refused", 0) + 1
            return None
        st["calls"] += 1
        st["calls_by_kind"][kind] = st["calls_by_kind"].get(kind, 0) + 1
        try:
            r = self.llm.complete(prompt, max_tokens=max_tokens)
        except LLMError as e:
            print(f"  [qa] llm error: {str(e)[:160]}", flush=True)
            st["llm_errors"] = st.get("llm_errors", 0) + 1
            return []
        if self.raw_log:
            with open(self.raw_log, "a") as f:
                f.write(json.dumps({"sid": self.sid, "kind": kind, "mode": "menu", "model": r["model"],
                                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                    "prompt": prompt, "raw": r["text"], "worst_case_usd": round(worst, 6),
                                    "usage": {"in": r["input_tokens"], "out": r["output_tokens"]},
                                    "cost_usd": r["cost_usd"], "stop": r["stop_reason"]},
                                   ensure_ascii=False) + "\n")
        return parse_items(r["text"])


def _bump(d: dict, reasons: list[str]) -> None:
    for w in reasons or ["?"]:
        k = w.split(":")[0]
        d[k] = d.get(k, 0) + 1


def span_of(c: dict) -> dict:
    return {"msg_idx": c["msg_idx"], "start": c["start"], "end": c["end"]}


def plan_menu(session: dict, wins: list[tuple[int, int]], cands: list[list[dict]], n_single: int, n_multihop: int,
              n_unanswerable: int, per_window_max: int = 4) -> dict:
    """What a run would ask for (also used by --dry-run)."""
    live = [i for i, c in enumerate(cands) if c]
    alloc = allocate_singles(len(live), n_single, per_window_max)
    per_win = {live[k]: alloc[k] for k in range(len(live))}
    pairs = find_pairs(cands, max_pairs=max(n_multihop * 2, n_multihop + 3), seed=0) if n_multihop else []
    return {"windows": len(wins), "windows_with_candidates": len(live), "per_window": per_win,
            "n_single": sum(per_win.values()), "pairs": pairs, "n_multihop": min(n_multihop, len(pairs)),
            "n_unanswerable": n_unanswerable if live else 0,
            "est_calls": len([v for v in per_win.values() if v]) + -(-min(n_multihop, len(pairs)) // 2)
            + (1 if n_unanswerable and live else 0)}


def generate_menu(session: dict, llm, *, window: int, keep_tail: int, reserve: int, use_real: bool,
                  n_single: int, n_multihop: int, n_unanswerable: int = 4, seed: int = 1337,
                  max_calls: int = 80, budget_usd: float = 1.0, raw_log: str | None = None,
                  per_window_max: int = 4, mh_per_call: int = 2) -> tuple[list[dict], dict]:
    """Window-targeted generation (see the MENU MODE comment). Same output shape as generate()."""
    mc = _mc()
    tok = message_tokens(session["messages"])
    wins = windows_for(session, window, keep_tail, reserve, use_real, tok)
    stats = {"sid": session["sid"], "mode": "menu", "windows": len(wins), "calls": 0, "calls_by_kind": {},
             "single_ok": 0, "single_rejected": {}, "multi_ok": 0, "multi_rejected": {}, "probe_ok": 0,
             "probe_rejected": {}, "cost_usd": 0.0}
    if not wins:
        return [], stats
    toks, idx = build_token_index(session)
    cands = window_candidates(session, wins, toks, idx, seed=seed)
    plan = plan_menu(session, wins, cands, n_single, n_multihop, n_unanswerable, per_window_max)
    stats.update({"windows_with_candidates": plan["windows_with_candidates"],
                  "candidates": sum(len(c) for c in cands), "planned_single": plan["n_single"],
                  "planned_multihop": plan["n_multihop"], "pairs_found": len(plan["pairs"])})
    stats["precheck"] = "LLM.worst_case" if hasattr(llm, "worst_case") else "local-estimate"
    caller = Caller(llm, cap_usd=budget_usd, max_calls=max_calls, raw_log=raw_log, sid=session["sid"], stats=stats)
    taken: set[str] = set()
    questions: list[dict] = []
    nW = len(wins)

    def add_single(c: dict, it: dict) -> None:
        taken.add(c["anchor"].lower())
        questions.append({
            "qid": f"{session['sid']}_s{stats['single_ok'] + 1}", "qtype": "single",
            "question": mc.norm_ws(it["question"]), "answer": mc.norm_ws(it["answer"]), "answerable": True,
            "evidence": [span_of(c)], "gold_anchor": c["anchor"],
            "meta": {"window": c["w"], "windows_total": nW, "stratum": stratum(c["w"], nW),
                     "anchor_kind": c["kind"], "mode": "menu", "generator": llm.model}})
        stats["single_ok"] += 1

    # ---- phase 1: singles, one menu call per window with a deficit (retry rounds run last, with fresh entries)
    want = dict(plan["per_window"])
    used_ids: set[str] = set()
    reserved = {x["anchor"].lower() for p in plan["pairs"] for x in (p["c1"], p["c2"])}   # kept for multihop

    def singles_round(rnd: int) -> bool:
        progressed = False
        order = sorted((w for w, n in want.items() if n > 0), key=lambda i: ((i + 0.5) * 0.6180339887) % 1.0)
        for w in order:
            free = [c for c in cands[w] if c["id"] not in used_ids and c["anchor"].lower() not in taken
                    and (rnd > 1 or c["anchor"].lower() not in reserved)]
            if not free:
                want[w] = 0
                continue
            menu = free[:MENU_SHOW]
            n_ask = min(len(menu), want[w] + (1 if rnd == 1 else 0))
            items = caller.call(menu_single_prompt(session, w + 1, nW, menu, n_ask, sorted(taken)), "single", 1400)
            if items is None:
                break
            by_id = {c["id"]: c for c in menu}
            got = 0
            for it in items:
                c = by_id.get(str(it.get("id") or "").strip())
                if c is None:
                    _bump(stats["single_rejected"], ["unknown_id"])
                    continue
                if c["id"] in used_ids:
                    _bump(stats["single_rejected"], ["duplicate_id"])
                    continue
                used_ids.add(c["id"])                      # an entry is spent whether or not it was accepted
                why = validate_menu_single(it, c, taken) + verify_candidate(session, wins, c, None)
                if why or got >= want[w]:
                    if why:
                        _bump(stats["single_rejected"], why)
                    continue
                add_single(c, it)
                got += 1
            want[w] -= got
            progressed = progressed or got > 0
        return progressed

    caller.begin_phase(0.55)
    singles_round(1)

    # ---- phase 2: multihop pairs, mh_per_call pairs per call
    if n_multihop and plan["pairs"]:
        caller.begin_phase(0.75)
        pool = list(plan["pairs"])
        while stats["multi_ok"] < n_multihop and pool and not stats.get("stopped"):
            pool = [p for p in pool if p["c1"]["anchor"].lower() not in taken and p["c2"]["anchor"].lower() not in taken]
            batch = pool[:min(mh_per_call, max(1, n_multihop - stats["multi_ok"]))]
            if not batch:
                break
            pool = pool[len(batch):]
            items = caller.call(menu_mh_prompt(session, batch, nW, sorted(taken)), "multihop", 1200)
            if items is None:
                break
            by_id = {p["id"]: p for p in batch}
            for it in items:
                p = by_id.pop(str(it.get("pair") or "").strip(), None)
                if p is None:
                    _bump(stats["multi_rejected"], ["unknown_pair"])
                    continue
                why, q = validate_menu_multi(it, p, session, wins, taken)
                if why:
                    _bump(stats["multi_rejected"], why)
                    continue
                c1, c2 = p["c1"], p["c2"]
                taken.update({c1["anchor"].lower(), c2["anchor"].lower()})
                s1, s2 = span_of(c1), span_of(c2)
                s1["role"], s2["role"] = "link", "answer"
                questions.append({
                    "qid": f"{session['sid']}_m{stats['multi_ok'] + 1}", "qtype": "multihop", "question": q,
                    "answer": mc.norm_ws(it["answer"]), "answerable": True, "evidence": [s1, s2],
                    "gold_anchor": c2["anchor"],
                    "meta": {"link_anchor": c1["anchor"], "shared_entity": p["entity"], "mode": "menu",
                             "generator": llm.model, "windows": [c1["w"], c2["w"]], "windows_total": nW,
                             "stratum": stratum(c1["w"], nW)}})
                stats["multi_ok"] += 1
                if stats["multi_ok"] >= n_multihop:
                    break

    # ---- phase 3: unanswerable probes (plausible, verified absent from the whole session)
    live = [c for ws in cands for c in ws[:1]]
    if n_unanswerable and live and not stats.get("stopped"):
        caller.begin_phase(1.0)
        rng = random.Random(f"{seed}:{session['sid']}:probe")
        samples = rng.sample(live, min(4, len(live)))
        full_lower = "\n".join(m["text"] for m in session["messages"]).lower()
        for _ in range(2):
            need = n_unanswerable - stats["probe_ok"]
            if need <= 0:
                break
            items = caller.call(probe_prompt(session, samples, need + 1), "probe", 900)
            if items is None:
                break
            for it in items:
                why = validate_probe(it, full_lower)
                if why or stats["probe_ok"] >= n_unanswerable:
                    if why:
                        _bump(stats["probe_rejected"], why)
                    continue
                full_lower += "\n" + it["absent_token"].lower()      # no two probes share a token
                questions.append({
                    "qid": f"{session['sid']}_u{stats['probe_ok'] + 1}", "qtype": "single",
                    "question": mc.norm_ws(it["question"]), "answer": "This is not mentioned anywhere in the session.",
                    "answerable": False, "evidence": [],
                    "meta": {"probe_token": it["absent_token"].strip(), "mode": "menu", "generator": llm.model}})
                stats["probe_ok"] += 1

    # ---- phase 4: retry the windows that are still short (fresh menu entries), while money and calls last
    for rnd in (2, 3):
        if not any(n > 0 for n in want.values()) or stats.get("stopped"):
            break
        caller.begin_phase(1.0)
        if not singles_round(rnd):
            break

    ok = stats["single_ok"] + stats["multi_ok"] + stats["probe_ok"]
    stats["cost_usd"] = round(llm.cost - caller.cost0, 6)
    stats["accepted"] = ok
    stats["accept_per_call"] = round(ok / stats["calls"], 3) if stats["calls"] else None
    return questions, stats


def dry_run_menu(session: dict, llm, a, wins: list[tuple[int, int]]) -> float:
    """Offline plan: builds the real menus/prompts (no network), prints expected item counts and the
    worst-case cost of the planned calls. Returns the worst-case total."""
    tok = message_tokens(session["messages"])
    toks, idx = build_token_index(session)
    cands = window_candidates(session, wins, toks, idx, seed=a.seed)
    plan = plan_menu(session, wins, cands, a.n_single, a.n_multihop, a.n_unanswerable, a.per_window_max)
    caller = Caller(llm, cap_usd=float("inf"), max_calls=10 ** 9, raw_log=None, sid=session["sid"],
                    stats={"calls": 0, "calls_by_kind": {}})
    pin, pout = llm.price
    worst = expect = 0.0
    calls = 0
    for w, n in plan["per_window"].items():
        if not n:
            continue
        pr = menu_single_prompt(session, w + 1, len(wins), cands[w][:MENU_SHOW], min(MENU_SHOW, n + 1), [])
        worst += caller.worst_case(pr, 1400)
        expect += (len(pr) / 3.6 * pin + 60 * (n + 1) * pout) / 1e6
        calls += 1
    for i in range(0, len(plan["pairs"][:plan["n_multihop"]]), 2):
        pr = menu_mh_prompt(session, plan["pairs"][i:i + 2], len(wins), [])
        worst += caller.worst_case(pr, 1200)
        expect += (len(pr) / 3.6 * pin + 130 * 2 * pout) / 1e6
        calls += 1
    if plan["n_unanswerable"]:
        pr = probe_prompt(session, [c[0] for c in cands if c][:4], plan["n_unanswerable"] + 1)
        worst += caller.worst_case(pr, 900)
        expect += (len(pr) / 3.6 * pin + 40 * plan["n_unanswerable"] * pout) / 1e6
        calls += 1
    print(f"{session['sid']}: {len(wins)} windows ({plan['windows_with_candidates']} with candidates, "
          f"{sum(len(c) for c in cands)} candidates), planned single={plan['n_single']} "
          f"multihop={plan['n_multihop']} (pairs found {len(plan['pairs'])}) probes={plan['n_unanswerable']}; "
          f"~{calls} first-pass calls (+ retries), expected {fmt_usd(expect)}, worst-case {fmt_usd(worst)} "
          f"({a.model} @ {llm.price} USD/MTok in,out)")
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    P.add_profile_args(ap, roles=("qa_generator",))
    ap.add_argument("--provider", default=None, choices=P.PROFILE_NAMES + ("mock",),
                    help="alias of --profile (also: mock = offline, no network)")
    ap.add_argument("--model", default=None, help="alias of --model-qa-generator")
    ap.add_argument("--key-env", default=None, help="env var holding the API key (never printed)")
    ap.add_argument("--window-tokens", type=int, default=150000)
    ap.add_argument("--keep-tail", type=int, default=20000)
    ap.add_argument("--reserve", type=int, default=1536)
    ap.add_argument("--use-real-compactions", action="store_true")
    ap.add_argument("--mode", choices=("menu", "legacy"), default="menu",
                    help="menu (default): per compaction window the CODE extracts anchors unique to the window "
                         "+ verbatim evidence, the model only phrases questions; legacy: free-text generation "
                         "from random excerpts with post-hoc gates (wasteful on long research sessions)")
    ap.add_argument("--n-single", type=int, default=None, help="menu: 40 (~2 per window), legacy: 12")
    ap.add_argument("--n-multihop", type=int, default=None, help="menu: 10, legacy: 4")
    ap.add_argument("--n-unanswerable", type=int, default=None,
                    help="menu: unanswerable probes (plausible, verified absent), default 4; legacy: 0")
    ap.add_argument("--per-window-max", type=int, default=4, help="menu: cap of single questions per window")
    ap.add_argument("--excerpt-tokens", type=int, default=8000)
    ap.add_argument("--per-call", type=int, default=3)
    ap.add_argument("--max-calls", type=int, default=None,
                    help="per session (default: legacy 60, menu 2 x windows + 3 x n-multihop + 12)")
    P.add_budget_arg(ap)
    ap.add_argument("--session-budget-usd", type=float, default=None,
                    help="optional cap per session (default: the remaining --budget-usd)")
    ap.add_argument("--price-in", type=float, default=None, help="USD/MTok override of the profile price")
    ap.add_argument("--price-out", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--keep-existing", action="store_true", help="append to existing questions")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    menu = a.mode == "menu"
    if a.n_single is None:
        a.n_single = 40 if menu else 12
    if a.n_multihop is None:
        a.n_multihop = 10 if menu else 4
    if a.n_unanswerable is None:
        a.n_unanswerable = 4 if menu else 0
    sessions = load_sessions(a.sessions)
    if not sessions:
        raise SystemExit("no sessions found")
    os.makedirs(a.out, exist_ok=True)
    raw_log = os.path.join(a.out, "qa-raw.jsonl")
    provider = a.profile or a.provider or "anthropic"
    try:
        profile = None if provider == "mock" else P.load_profile(provider, base_url=a.base_url,
                                                                  pi_provider=a.pi_provider)
    except P.ProviderError as e:
        raise SystemExit(f"!!! {e}")
    a.model = a.model_qa_generator or a.model or (profile.default_model("qa_generator") if profile
                                                  else "mock")
    if not a.model:
        raise SystemExit(f"!!! no qa_generator model for profile {provider}: pass --model-qa-generator")
    budget = None
    if not a.dry_run and provider != "mock":
        try:
            budget = P.Budget(a.budget_usd)
        except P.ProviderError as e:
            raise SystemExit(f"!!! {e} (--budget-usd is mandatory for a paid generation run)")
    override = (a.price_in, a.price_out) if (a.price_in is not None and a.price_out is not None) else None
    llm = LLM(provider, a.model, role="qa_generator", profile=profile, budget=budget,
              key_env=a.key_env, price=override)
    if not a.dry_run and llm.key_problem():
        raise SystemExit(llm.key_problem())
    a.provider = provider
    manifest = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": a.model,
                "provider": a.provider, "argv": sys.argv[1:], "sessions": [],
                "budget_usd": a.budget_usd,
                "price_usd_per_mtok": list(llm.price),
                "price_status": ((profile.price(a.model) or {}).get("status") if profile else None)}
    est_total = 0.0
    for s in sessions:
        bad = validate_session(s)
        if bad:
            print(f"[skip] {s.get('sid')}: {bad[:2]}")
            continue
        wins = windows_for(s, a.window_tokens, a.keep_tail, a.reserve, a.use_real_compactions)
        if a.dry_run:
            pin, pout = llm.price
            if menu:
                est_total += dry_run_menu(s, llm, a, wins)
                continue
            calls = estimate_calls(a.n_single, a.n_multihop, a.per_call)
            est = calls * (a.excerpt_tokens * 1.3 * pin + 1500 * pout) / 1e6
            est_total += est
            print(f"{s['sid']}: {len(wins)} windows, ~{calls} calls, est {fmt_usd(est)} "
                  f"({a.model} @ {llm.price} USD/MTok in,out)")
            continue
        sess_cap = min(a.session_budget_usd or float("inf"), budget.remaining if budget else float("inf"))
        if menu:
            qs, st = generate_menu(s, llm, window=a.window_tokens, keep_tail=a.keep_tail, reserve=a.reserve,
                                   use_real=a.use_real_compactions, n_single=a.n_single,
                                   n_multihop=a.n_multihop, n_unanswerable=a.n_unanswerable, seed=a.seed,
                                   max_calls=a.max_calls or (2 * len(wins) + 3 * a.n_multihop + 12),
                                   budget_usd=sess_cap, raw_log=raw_log, per_window_max=a.per_window_max)
            old = s.get("questions") or [] if a.keep_existing else []
            s["questions"] = old + qs
            save_session(s, a.out)
            manifest["sessions"].append(st)
            print(f"{s['sid']}: {st['single_ok']} single + {st['multi_ok']} multihop + {st['probe_ok']} probes "
                  f"accepted ({st['calls']} calls, accept/call {st['accept_per_call']}, {fmt_usd(st['cost_usd'])}); "
                  f"{st['windows_with_candidates']}/{st['windows']} windows with candidates; rejected "
                  f"single={st['single_rejected']} multi={st['multi_rejected']} probe={st['probe_rejected']}"
                  + (f"; cap_refused={st['cap_refused']}" if st.get("cap_refused") else ""))
            continue
        qs, st = generate(s, llm, window=a.window_tokens, keep_tail=a.keep_tail, reserve=a.reserve,
                          use_real=a.use_real_compactions, n_single=a.n_single,
                          n_multihop=a.n_multihop, excerpt_tokens=a.excerpt_tokens, seed=a.seed,
                          per_call=a.per_call, max_calls=a.max_calls or 60,
                          budget_usd=sess_cap, raw_log=raw_log)
        old = s.get("questions") or [] if a.keep_existing else []
        s["questions"] = old + qs
        save_session(s, a.out)
        manifest["sessions"].append(st)
        print(f"{s['sid']}: {st['single_ok']} single + {st['multi_ok']} multihop accepted "
              f"({st['calls']} calls, {fmt_usd(st['cost_usd'])}); rejected single={st['single_rejected']} "
              f"multi={st['multi_rejected']}")
    if a.dry_run:
        cap = f" vs --budget-usd {fmt_usd(a.budget_usd)}" if a.budget_usd else ""
        print(f"DRY RUN total estimate {fmt_usd(est_total)} (menu: worst-case of the first pass; legacy: max calls per session){cap}"
              + ("  [OVER CAP]" if a.budget_usd and est_total > a.budget_usd else ""))
    if not a.dry_run:
        manifest["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["total_cost_usd"] = round(llm.cost, 6)
        manifest["unpriced_calls"] = llm.unpriced_calls
        with open(os.path.join(a.out, "qa-manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
