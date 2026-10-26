#!/usr/bin/env python3
"""QA generator for longsession/1 sessions that carry no questions (e.g. Claude Code imports).

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
  python3 gen_longsession_qa.py --sessions DIR --out OUT --dry-run        # no network
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
                            real=real, final=True)
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
    ap.add_argument("--n-single", type=int, default=12)
    ap.add_argument("--n-multihop", type=int, default=4)
    ap.add_argument("--excerpt-tokens", type=int, default=8000)
    ap.add_argument("--per-call", type=int, default=3)
    ap.add_argument("--max-calls", type=int, default=60, help="per session")
    P.add_budget_arg(ap)
    ap.add_argument("--session-budget-usd", type=float, default=None,
                    help="optional cap per session (default: the remaining --budget-usd)")
    ap.add_argument("--price-in", type=float, default=None, help="USD/MTok override of the profile price")
    ap.add_argument("--price-out", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--keep-existing", action="store_true", help="append to existing questions")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
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
            calls = estimate_calls(a.n_single, a.n_multihop, a.per_call)
            pin, pout = llm.price
            est = calls * (a.excerpt_tokens * 1.3 * pin + 1500 * pout) / 1e6
            est_total += est
            print(f"{s['sid']}: {len(wins)} windows, ~{calls} calls, est {fmt_usd(est)} "
                  f"({a.model} @ {llm.price} USD/MTok in,out)")
            continue
        qs, st = generate(s, llm, window=a.window_tokens, keep_tail=a.keep_tail, reserve=a.reserve,
                          use_real=a.use_real_compactions, n_single=a.n_single,
                          n_multihop=a.n_multihop, excerpt_tokens=a.excerpt_tokens, seed=a.seed,
                          per_call=a.per_call, max_calls=a.max_calls,
                          budget_usd=min(a.session_budget_usd or float("inf"),
                                         budget.remaining if budget else float("inf")),
                          raw_log=raw_log)
        old = s.get("questions") or [] if a.keep_existing else []
        s["questions"] = old + qs
        save_session(s, a.out)
        manifest["sessions"].append(st)
        print(f"{s['sid']}: {st['single_ok']} single + {st['multi_ok']} multihop accepted "
              f"({st['calls']} calls, {fmt_usd(st['cost_usd'])}); rejected single={st['single_rejected']} "
              f"multi={st['multi_rejected']}")
    if a.dry_run:
        cap = f" vs --budget-usd {fmt_usd(a.budget_usd)}" if a.budget_usd else ""
        print(f"DRY RUN total estimate {fmt_usd(est_total)} (upper-ish bound: max calls per session){cap}"
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
