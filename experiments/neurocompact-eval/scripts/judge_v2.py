#!/usr/bin/env python3
"""Blind, evidence-grounded v2 judge for the chained-compaction (lag) experiment.

Input: one JSON/JSONL record per (qid, arm) with
  rid, qid, chain_id, session_id, span_idx, lag, slice, answerable,
  question, gold_anchor, canonical_short_answer, evidence, span_text, answer, arm,
  and optionally transcript (the archived full session text, for the hallucination
  post-pass).

What the judge sees, per call:
  * question, reference answer, candidate answer  (arm/wrapper noise stripped;
    arm-revealing strings -- runner session ids such as cr-...-nc-..., any
    "-nc-"/"-baseline-" id segment, "[memory recall ...]" / <memory> / "Recalled
    session memory" headers, "recalled memory", HARS -- are masked as [masked] in
    the judge-visible answer only; every masking is logged per presentation in
    the mapping file under "answer_masked", plus a "masking" summary)
  * a deterministic +-500 char window of span_text around the evidence
  * for unanswerable items NO excerpt at all: a seeded unrelated excerpt from a
    different chain is substituted, so "no local evidence" is not a give-away.
  * items are presented in a seeded shuffled order; the qid->arm mapping is
    written to a separate file and never enters a prompt.

Strict JSON reply: {"score":0|1|2,"unsupported_claim":bool,"abstained":bool,
"reason":"<=25 words"}.

Python post-pass (judge-independent):
  anchor_hit    = escaped, case-insensitive, token-bounded regex of gold_anchor
                  in the answer.
  hallucination = unsupported_claim AND NOT (anchor_hit AND every concrete
                  token/number of the answer is present in the archived
                  transcript)   -- a true auxiliary detail is not a hallucination.
                  For unanswerable items: any non-abstaining concrete factual
                  answer is a hallucination, regardless of the judge.
  verification  = token-level check of the answer's concrete tokens against the
                  record's `visible_context` (the answer-arm's actually-visible
                  post-C3 context; see visible_context.py). Falls back to the
                  legacy `transcript` field only when visible_context is absent.
                  Emits {supported_tokens, unsupported_tokens, verdict:
                  supported|unsupported|unverifiable} per record; "unverifiable"
                  means no context field was available (the degenerate mode the
                  gate-212839 forensics flagged).

Usage:
  python3 judge_v2.py --in answers.jsonl --out judged.jsonl --mapping map.json
  python3 judge_v2.py --dry-run                 # fixtures, $0, plumbing test
  python3 judge_v2.py --dry-run --calibration   # agreement/kappa plumbing test
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import providers  # noqa: E402  (profiles, prices, family check, hard budget)

JUDGE_MODEL_DEFAULT = "openai/gpt-6-luna"     # legacy default (no --profile): OpenRouter
TRANSPORT = None                               # tests inject a fake HTTP transport
FIXTURES = os.path.join(BASE, "judge_v2_fixtures.json")
WINDOW = 500                     # chars either side of the evidence
UNSUPPORTED_AND_FULL_CREDIT_MAX = 0.05   # calibration gate

# ---------------------------------------------------------------- text helpers

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def anchor_re(anchor: str, flags=re.I):
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(anchor.strip())
                      + r"(?![A-Za-z0-9_])", flags)


CONCRETE_PATTERNS = [
    re.compile(r"(?<![\w-])--?[A-Za-z][A-Za-z0-9-]{1,}"),          # --flag
    re.compile(r"[A-Za-z_][A-Za-z0-9]*(?:[._\-/][A-Za-z0-9]+)+"),  # a.b_c-d/e
    re.compile(r"[A-Za-z_]*\d[A-Za-z0-9_]*"),                      # nx4471, v2
    re.compile(r"\d+(?:[.,]\d+)+"),                                # 28.6
    re.compile(r"\d{2,}"),                                         # 8090
]

ABSTAIN_RE = re.compile(
    r"\bi\s+(?:do not|don't|cannot|can't|am unable)\b"
    r"|\bnot\s+(?:recorded|mentioned|available|present)\b"
    r"|\bin\s+the\s+(?:log|context|session|summary|excerpt)\b(?:[^.]{0,40})\b(?:not|no)\b"
    r"|\bno\s+(?:such|record|information|mention)\b"
    r"|\bunable\s+to\b"
    r"|\bdoes\s+not\s+(?:mention|record|say|appear)\b"
    r"|\bnot\s+sure\b|\bthere\s+is\s+no\s+mention\b|\bdecline\b", re.I)

ARM_LABEL_RE = re.compile(
    r"^\s*(?:\[\s*arm\s*=\s*[^\]\s]+\s*\]|(?:nc-baseline|nc-note-only|nc-chunks|nc-graph|baseline|nc|orig|none)"
    r"\s*(?:[:>\-—]\s*|\s*\)\s*:?\s*)?|(?:answer|assistant|response)\s*"
    r"(?:\([^)]{0,30}\))?\s*:\s*)", re.I)


# M8: arm-revealing strings that can appear ANYWHERE inside an answer (the agent
# quoting its own session id or the injected recall wrapper). Masked only in the
# judge-visible answer; post_pass keeps the unmasked text.
ARM_LEAK_RE = re.compile(
    r"\bcr-[\w./:-]+"                                         # runner session ids
    r"|[\w./:]*-(?:nc-note-only|nc-chunks|nc-graph|nc|baseline)-[\w./:-]*"   # -nc-/-nc-chunks-/-nc-graph-/-baseline- id parts
    r"|\bnc-note-only\b|\bnote-only\b|\bnc-chunks\b|\bnc-graph\b"
    # Amendment C: graph section headers of the injected recall (the KG sections exist ONLY
    # in the nc-graph arm's recall, so they would reveal the arm), the packer's own heading
    # and the chunk-section headers
    r"|Knowledge\s+Graph\s+Data\s*\((?:Entity|Relationship[^)\n]*)\)\s*:?"
    r"|#+\s*Direct\s+relations\s*/\s*entities"
    r"|Document\s+Chunks\s*\([^)\n]*\)\s*:?|Reference\s+Document\s+List\s*\([^)\n]*\)\s*:?"
    r"|\bsession[\s_-]*graph\b"
    r"|\[\s*memory[\s_-]*recall[^\]\n]*\]?"                   # [memory recall · HARS ...]
    r"|harness-injected\s+memory(?:\s+recall)?"
    r"|</?\s*memory\s*>"
    r"|#+\s*recalled\s+session\s+memory[^\n]*"
    r"|\brecalled\s+(?:session\s+)?memor(?:y|ies)\b"
    r"|\b(?:injected|retrieved)\s+(?:session\s+)?memor(?:y|ies)\b"
    r"|\bmemory[\s_]+recall\b"
    r"|\bHARS\b", re.I)
MASK = "[masked]"


def mask_arm_leaks(text: str) -> tuple[str, list[str]]:
    hits = [m.group(0) for m in ARM_LEAK_RE.finditer(text or "")]
    return (ARM_LEAK_RE.sub(MASK, text or "") if hits else (text or "")), hits


def sanitize_answer(ans: str) -> str:
    """Strip arm labels / transport wrappers so the grader cannot see the arm."""
    s = (ans or "").strip()
    for _ in range(5):
        new = re.sub(r"\[\s*arm\s*=\s*[^\]]+\]\s*", "", s, flags=re.I)
        new = ARM_LABEL_RE.sub("", new).strip()
        if new == s:
            break
        s = new
    return s


def concrete_tokens(text: str) -> list[str]:
    out, seen = [], set()
    for pat in CONCRETE_PATTERNS:
        for m in pat.finditer(text or ""):
            t = m.group(0).strip("`'\" ")
            if len(t) < 2 or t.lower() in seen:
                continue
            seen.add(t.lower())
            out.append(t)
    return out


# ------------------------------------------------------------- excerpt logic

def window_around(span: str, evidence: str, width: int = WINDOW) -> tuple[str, str]:
    """Deterministic +-width chars around the evidence; reports how it was found."""
    span = span or ""
    ev = norm_ws(evidence)
    if not ev:
        return span[: 2 * width], "no_evidence"
    candidates = [ev] + [ev[:k] for k in (200, 120, 80, 60)]
    for cand in candidates:
        i = span.find(cand)
        if i >= 0:
            a, b = max(0, i - width), min(len(span), i + len(cand) + width)
            return span[a:b], "evidence_exact" if cand is ev else "evidence_prefix"
    # whitespace-insensitive fallback: locate the first 8 words that do appear
    words = ev.split()
    for n in (16, 12, 10, 8):
        for start in range(0, max(1, len(words) - n) + 1, 4):
            probe = " ".join(words[start:start + n])
            flat = re.sub(r"\s+", " ", span)
            j = flat.find(probe)
            if j >= 0:
                a, b = max(0, j - width), min(len(flat), j + len(probe) + width)
                return flat[a:b], "evidence_normalized"
    return span[: 2 * width], "evidence_not_found"


def window_for(span: str, evidence, width: int = WINDOW) -> tuple[str, str]:
    """window_around for one evidence string, or for a LIST of evidence strings (multi-hop /
    long-session items): the per-evidence windows joined in order, status = the weakest."""
    if not isinstance(evidence, (list, tuple)):
        return window_around(span, evidence, width)
    parts, stats = [], []
    for ev in evidence:
        w, how = window_around(span, ev if isinstance(ev, str) else json.dumps(ev), width)
        if w not in parts:
            parts.append(w)
        stats.append(how)
    if not parts:
        return window_around(span, "", width)
    order = ["evidence_exact", "evidence_prefix", "evidence_normalized", "no_evidence",
             "evidence_not_found"]
    weakest = max(stats, key=lambda x: order.index(x) if x in order else len(order))
    return "\n[...]\n".join(parts), weakest


def unrelated_excerpt(rec: dict, pool: list[dict], rng: random.Random,
                      width: int = WINDOW) -> tuple[str, str, str]:
    """Unanswerable items get a real excerpt from a DIFFERENT chain (never their own)."""
    cands = [r for r in pool
             if r.get("chain_id") != rec.get("chain_id") and (r.get("span_text") or "")]
    cands.sort(key=lambda r: (r.get("chain_id") or "", r.get("rid") or ""))
    if not cands:
        return "", "unrelated_unavailable", ""
    other = cands[rng.randrange(len(cands))]
    span = other["span_text"]
    mid = max(0, len(span) // 2 - width)
    return span[mid: mid + 2 * width], "unrelated_other_chain", other.get("chain_id", "")


# ------------------------------------------------------------------ prompting

PROMPT = """You are a strict, blind grader for a memory experiment. You see one item.

Question: {question}
Reference answer (recorded when the item was written; empty when no fact was recorded): {reference}

Evidence excerpt from the archived session window (a narrow locator, the fact may
lie outside it): <excerpt>
{excerpt}
</excerpt>

Candidate answer: <candidate>
{answer}
</candidate>

Grade the candidate answer:
2 = fully correct: states the recorded fact; if no fact was recorded, 2 also means
    it correctly says the fact is not recorded in the session.
1 = partially correct: right direction but vague, incomplete or hedged.
0 = wrong, evasive, or asserts a different/unrelated specific.
unsupported_claim = true ONLY if the answer asserts a concrete number, identifier,
path, flag, error code or causal rationale that is fabricated or contradicted (not
merely absent from this narrow excerpt).
abstained = true if the answer declines to give the fact.
reason: <= 25 words.

Reply ONLY JSON: {{"score":0|1|2,"unsupported_claim":true|false,"abstained":true|false,"reason":"..."}}"""

# Long-session items that carry the benchmark's own rubric (BEAM): the rubric is passed to the
# judge as the grading standard. Items without a rubric use PROMPT verbatim (chain datasets).
RUBRIC_BLOCK = """Rubric written by the benchmark authors for this question (the reference answer above
is its short form). Grade against it: 2 = the candidate satisfies every rubric item (or all
that matter for the question), 1 = it satisfies some but not all, 0 = it satisfies none or
contradicts them. A candidate that merely restates the question satisfies nothing.
<rubric>
{rubric}
</rubric>

"""


def rubric_text(rec: dict) -> str:
    rb = rec.get("rubric")
    if isinstance(rb, str):
        rb = [rb]
    items = [norm_ws(str(x)) for x in (rb or []) if norm_ws(str(x))]
    return "\n".join(f"- {x}" for x in items)


def build_prompt(rec: dict, excerpt: str) -> str:
    base = _build_prompt(rec, excerpt)
    rt = rubric_text(rec)
    if not rt:
        return base
    return base.replace("Grade the candidate answer:", RUBRIC_BLOCK.format(rubric=rt)
                        + "Grade the candidate answer:", 1)


def _build_prompt(rec: dict, excerpt: str) -> str:
    return PROMPT.format(
        question=rec.get("question", "").strip(),
        reference=(rec.get("canonical_short_answer") or "").strip() or "(none recorded)",
        excerpt=excerpt.strip() or "(no excerpt)",
        answer=mask_arm_leaks(sanitize_answer(rec.get("answer", "")))[0]
        or "(empty answer)",
    )


def parse_verdict(raw: str) -> tuple[dict, bool]:
    """Tolerant: prose before/after, fenced JSON, raw newlines inside strings."""
    if not raw:
        return {"score": None, "unsupported_claim": None, "abstained": None,
                "reason": "empty"}, False
    lo, hi = raw.find("{"), raw.rfind("}")
    if lo < 0 or hi <= lo:
        return {"score": None, "unsupported_claim": None, "abstained": None,
                "reason": "no_json"}, False
    for kwargs in ({}, {"strict": False}):
        try:
            j = json.loads(raw[lo:hi + 1], **kwargs)
        except Exception:
            continue
        score = j.get("score")
        if isinstance(score, str) and score.strip().isdigit():
            score = int(score.strip())
        if score not in (0, 1, 2):
            return {"score": None, "unsupported_claim": None, "abstained": None,
                    "reason": f"bad_score:{score}"}, False
        return {"score": score,
                "unsupported_claim": bool(j.get("unsupported_claim")),
                "abstained": bool(j.get("abstained")),
                "reason": str(j.get("reason", ""))[:200]}, True
    return {"score": None, "unsupported_claim": None, "abstained": None,
            "reason": "unparseable"}, False


# ------------------------------------------------------------------- post-pass

def facts_context(rec: dict) -> tuple[str, str]:
    """Verification text: visible_context when present, else legacy transcript.

    Returns (text, source); source is "visible_context" | "transcript" | "none".
    A visible_context field that is present-but-empty is treated as absent so the
    legacy fallback still applies."""
    vc = rec.get("visible_context")
    if isinstance(vc, str) and vc.strip():
        return vc, "visible_context"
    tr = rec.get("transcript")
    if isinstance(tr, str) and tr.strip():
        return tr, "transcript"
    return "", "none"


def verify_tokens(answer: str, text: str, source: str) -> dict:
    """Token-level verification of the answer against the visible context."""
    toks = concrete_tokens(answer)
    if not text:
        return {"supported_tokens": [],
                "unsupported_tokens": toks[:12],
                "verdict": "unverifiable", "source": source}
    unsupported = [t for t in toks if not anchor_re(t).search(text)]
    supported = [t for t in toks if t not in unsupported]
    return {"supported_tokens": supported[:12],
            "unsupported_tokens": unsupported[:12],
            "verdict": "supported" if not unsupported else "unsupported",
            "source": source}


def post_pass(rec: dict, verdict: dict) -> dict:
    answer = sanitize_answer(rec.get("answer", ""))
    answerable = rec.get("answerable", True)
    anchor = (rec.get("gold_anchor") or "").strip()
    anchor_hit = bool(anchor) and bool(anchor_re(anchor).search(answer))
    # deterministic abstention detection (also overrides a judge that ignores a
    # refusal); the word cap keeps a partial answer from being read as a refusal
    heuristic_abstain = (not answer.strip()) or (
        bool(ABSTAIN_RE.search(answer)) and len(answer.split()) <= 45)
    abstained = bool(verdict.get("abstained")) or heuristic_abstain
    concrete = concrete_tokens(answer)
    text, src = facts_context(rec)
    verification = verify_tokens(answer, text, src)
    facts_ok = bool(text) and verification["verdict"] == "supported"
    out = {
        "anchor_hit": anchor_hit,
        "abstained": abstained,
        "abstained_source": ("heuristic" if heuristic_abstain and not verdict.get("abstained")
                             else "judge" if verdict.get("abstained") else "none"),
        "concrete_tokens": concrete[:12],
        "facts_verified_in_transcript": facts_ok if concrete else None,
        "fact_tokens_missing": verification["unsupported_tokens"][:8],
        "verification": verification,
        "hallucination": None,
        "hallucination_rule": "",
    }
    unsupported = verdict.get("unsupported_claim")
    if not answerable:
        # any non-abstaining concrete factual answer is a hallucination
        hall = (not abstained) and bool(concrete)
        out["hallucination"] = hall
        out["hallucination_rule"] = ("unanswerable_non_abstaining_concrete" if hall
                                     else "unanswerable_ok")
    elif unsupported is None:
        out["hallucination_rule"] = "no_verdict"
    elif not unsupported:
        out["hallucination"] = False
        out["hallucination_rule"] = "judge_clean"
    elif anchor_hit and facts_ok:
        out["hallucination"] = False
        out["hallucination_rule"] = "unsupported_but_verified_aux_detail"
    else:
        out["hallucination"] = True
        # no context at all -> unverifiable; with a context, specific tokens are
        # provably absent from what the model could see (contradicted/fabricated)
        if not facts_ok and verification["source"] == "none":
            out["hallucination_rule"] = "unsupported_and_unverifiable"
        elif not facts_ok:
            out["hallucination_rule"] = "unsupported_absent_from_visible_context"
        else:
            out["hallucination_rule"] = "unsupported_no_anchor"
    return out


# ---------------------------------------------------------------------- model

def call_judge(prompt: str, ctx: dict, spend: dict, max_tokens: int = 700,
               transport=None) -> tuple[str, float]:
    """One judge call through the provider layer. ctx = {profile, model, key_env, budget}.
    Cost = provider-reported or the profile price table; the hard budget is enforced by
    providers.chat (an unpriced model without reported cost raises)."""
    t0 = time.time()
    for attempt in range(3):
        try:
            res = providers.chat("judge", [{"role": "user", "content": prompt}],
                                 profile=ctx["profile"], model=ctx["model"], max_tokens=max_tokens,
                                 temperature=0, budget=ctx["budget"], key_env=ctx.get("key_env"),
                                 retries=1, transport=transport or TRANSPORT)
            spend["usd"] = ctx["budget"].spent
            spend["calls"] = ctx["budget"].calls
            return res.text, time.time() - t0
        except (providers.BudgetExceeded, providers.UnpricedModel):
            raise
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"  judge failed: {e}", flush=True)
                return "", time.time() - t0
            time.sleep(4 * (attempt + 1))
    return "", time.time() - t0


# ------------------------------------------------------------------ reporting

def cohen_kappa(a: list, b: list) -> tuple[float | None, float]:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None, 0.0
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    cats = set(x for x, _ in pairs) | set(y for _, y in pairs)
    pe = sum((sum(1 for x, _ in pairs if x == c) / n) *
             (sum(1 for _, y in pairs if y == c) / n) for c in cats)
    if pe >= 1.0:
        return None, po
    return (po - pe) / (1 - pe), po


def calibration_report(judged: list[dict]) -> dict:
    by_rid: dict[str, dict[int, dict]] = {}
    for j in judged:
        if j.get("pass") is None:
            continue
        by_rid.setdefault(j["rid"], {})[j["pass"]] = j
    rids = sorted(by_rid)
    p0 = [by_rid[r].get(0, {}) for r in rids]
    p1 = [by_rid[r].get(1, {}) for r in rids]
    k_score, po_score = cohen_kappa([x.get("score") for x in p0],
                                    [x.get("score") for x in p1])
    k_uns, po_uns = cohen_kappa([x.get("unsupported_claim") for x in p0],
                                [x.get("unsupported_claim") for x in p1])
    k_abs, po_abs = cohen_kappa([x.get("abstained") for x in p0],
                                [x.get("abstained") for x in p1])
    hall_full = [x for x in judged
                 if x.get("unsupported_claim") and x.get("score") == 2]
    rate = len(hall_full) / len(judged) if judged else 0.0
    return {
        "items": len(rids), "presentations": len(judged),
        "agree_score": round(po_score, 4), "kappa_score": None if k_score is None else round(k_score, 4),
        "agree_unsupported": round(po_uns, 4),
        "kappa_unsupported": None if k_uns is None else round(k_uns, 4),
        "agree_abstained": round(po_abs, 4),
        "kappa_abstained": None if k_abs is None else round(k_abs, 4),
        "p_unsupported_and_score2": round(rate, 4),
        "gate_p_unsupported_and_score2": ("PASS" if rate <= UNSUPPORTED_AND_FULL_CREDIT_MAX
                                          else "FAIL"),
        "gate_rule": f"P(unsupported_claim=1 AND score=2) <= "
                     f"{UNSUPPORTED_AND_FULL_CREDIT_MAX:.0%}",
    }


# ------------------------------------------------------------------- driver

def load_records(path: str) -> list[dict]:
    text = open(path).read()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "records" in obj:
            return list(obj["records"])
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def rid_of(rec: dict) -> str:
    return rec.get("rid") or (f"{rec['qid']}::{rec['arm']}" if rec.get("arm")
                              else rec.get("qid", ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="", help="records jsonl (answers to grade)")
    ap.add_argument("--out", default="", help="judged jsonl (default judged.jsonl)")
    ap.add_argument("--mapping", default="", help="presentation order + arm mapping")
    providers.add_profile_args(ap, roles=("judge",))
    ap.add_argument("--model", default=None,
                    help="alias of --model-judge (default: the profile's judge; legacy "
                         f"{JUDGE_MODEL_DEFAULT} on the openrouter profile)")
    ap.add_argument("--key-env", default=None, metavar="VAR",
                    help="read the API key from env var VAR instead of the profile's variable")
    ap.add_argument("--agent-model", action="append", default=[],
                    help="agent-side model(s) for the family check when the records do not "
                         "carry agent_model/qa_model (repeatable)")
    ap.add_argument("--allow-same-family-judge", action="store_true",
                    help="accept a judge of the SAME family as the agent, only if the judge "
                         "MODEL differs from every agent-side model; loud warning, recorded as "
                         "judge_same_family=true + reason in the mapping and the judged rows")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--calibration", "--calibrate", dest="calibration",
                    action="store_true",
                    help="grade every item twice with independent shuffles "
                         "(--calibrate is the RUNBOOK spelling, kept as alias)")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline plumbing test on bundled fixtures (no API, $0)")
    providers.add_budget_arg(ap)
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the first prompt verbatim (blindness check)")
    args = ap.parse_args()
    prof_name = args.profile or "openrouter"      # no --profile = the historical OpenRouter judge
    try:
        prof = providers.load_profile(prof_name, base_url=args.base_url, pi_provider=args.pi_provider)
    except providers.ProviderError as e:
        raise SystemExit(f"!!! {e}")
    judge_model = (getattr(args, "model_judge", None) or args.model
                   or (JUDGE_MODEL_DEFAULT if prof_name == "openrouter" else prof.default_model("judge")))
    if not judge_model:
        raise SystemExit(f"!!! no judge model for profile {prof_name}: pass --model-judge")
    args.model = judge_model

    if args.dry_run:
        records = load_records(FIXTURES)
        out = args.out or os.path.join(BASE, "judge-dryrun.jsonl")
        mapping_path = args.mapping or os.path.join(BASE, "judge-dryrun-mapping.json")
        model = f"{args.model} (fixtures)"
    else:
        if not args.inp:
            raise SystemExit("--in is required (or use --dry-run)")
        records = load_records(args.inp)
        out = args.out or os.path.join(BASE, "judged.jsonl")
        mapping_path = args.mapping or os.path.join(BASE, "judge-mapping.json")
        model = args.model

    if not records:
        raise SystemExit("no records")
    rids = [rid_of(r) for r in records]
    if len(set(rids)) != len(rids):
        dup = sorted({r for r in rids if rids.count(r) > 1})
        raise SystemExit(f"duplicate rid(s) {dup[:5]}: rid must uniquely identify (qid, arm)")
    for r in records:
        if "span_text" not in r:
            raise SystemExit(f"{rid_of(r)}: span_text is required for the evidence window")

    # ---- judge independence (same rule as run_chainreplay): refuse same family by default
    agent_models = list(args.agent_model)
    for r in records:
        agent_models += [r.get("agent_model"), r.get("qa_model")]
    agent_models = [m for m in dict.fromkeys(agent_models) if m]
    if args.dry_run and not agent_models:
        fam = providers.judge_family_verdict([], model, False)
    elif not agent_models:
        raise SystemExit("!!! cannot verify judge independence: the records carry no "
                         "agent_model/qa_model; pass --agent-model <id> (repeatable)")
    else:
        fam = providers.judge_family_verdict(agent_models, args.model, args.allow_same_family_judge)
    if not fam["ok"]:
        raise SystemExit(f"!!! REFUSING: {fam['error']}")
    if fam["warning"]:
        print("\n" + "!" * 78 + "\n" + fam["warning"] + "\n" + "!" * 78 + "\n",
              file=sys.stderr, flush=True)
    ctx = None
    if not args.dry_run:
        try:
            budget = providers.Budget(args.budget_usd)
        except providers.ProviderError as e:
            raise SystemExit(f"!!! {e} (--budget-usd is mandatory for a paid judge run)")
        kp = providers.key_problem(prof, args.key_env)
        if kp:
            raise SystemExit(f"!!! {kp}; judging is paid, or use --dry-run")
        ctx = {"profile": prof, "model": args.model, "key_env": args.key_env, "budget": budget}

    # narrative excerpt source per record, deterministic
    rng = random.Random(args.seed)
    excerpts = {}     # rid -> (excerpt, status, source_chain)
    for r in records:
        if r.get("answerable", True):
            excerpt, how = window_for(r.get("span_text", ""), r.get("evidence", ""))
            excerpts[rid_of(r)] = (excerpt, how, None)
        else:
            excerpts[rid_of(r)] = unrelated_excerpt(r, records, rng)

    # presentation order: seeded shuffle, arm-blind; mapping written separately
    order = list(records)
    random.Random(args.seed).shuffle(order)
    mapping = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "seed": args.seed, "model": model, "calibration": args.calibration,
               "profile": prof_name, "judge_model": args.model,
               "judge_same_family": fam["judge_same_family"],
               "judge_same_family_reason": fam["reason"],
               "agent_models": fam["agent_models"],
               "order": [], "map": {}}
    presentations = []          # (present_id, record, pass_no)
    for i, r in enumerate(order):
        pid = f"p{i:04d}"
        mapping["order"].append(pid)
        mapping["map"][pid] = {"rid": rid_of(r), "qid": r.get("qid"),
                               "arm": r.get("arm"), "pass": 0}
        presentations.append((pid, r, 0))
    if args.calibration:
        for i, r in enumerate(order):
            pid = f"p{i:04d}b"
            mapping["order"].append(pid)
            mapping["map"][pid] = {"rid": rid_of(r), "qid": r.get("qid"),
                                   "arm": r.get("arm"), "pass": 1}
            presentations.append((pid, r, 1))
        random.Random(args.seed + 1).shuffle(presentations)   # second, independent order

    spend = {"usd": 0.0, "calls": 0}
    judged, parse_fail, arm_in_prompt = [], 0, []
    masked_rids = set()
    for n, (pid, rec, pass_no) in enumerate(presentations, 1):
        excerpt, how, src_chain = excerpts[rid_of(rec)]
        prompt = build_prompt(rec, excerpt)
        _, leaks = mask_arm_leaks(sanitize_answer(rec.get("answer", "")))
        if leaks:
            mapping["map"][pid]["answer_masked"] = [x[:80] for x in leaks]
            masked_rids.add(rid_of(rec))
        if rec.get("arm") and re.search(r"(?<![\w-])" + re.escape(str(rec["arm"]))
                                        + r"(?![\w-])", prompt, re.I):
            arm_in_prompt.append(rid_of(rec))
        if args.show_prompt and n == 1:
            print("---- prompt of presentation 1 (arm must not appear) ----")
            print(prompt)
            print("---- end prompt ----", flush=True)
        if args.dry_run:
            raw = rec.get(f"judge_pass{pass_no}") or rec.get("judge_pass0") or ""
            latency = 0.0
        else:
            try:
                ctx["budget"].check()
                raw, latency = call_judge(prompt, ctx, spend)
            except providers.BudgetExceeded as e:
                raise SystemExit(f"judge budget exhausted after {spend['calls']} calls: {e}")
        verdict, ok = parse_verdict(raw)
        parse_fail += not ok
        post = post_pass(rec, verdict)
        judged.append({
            "present_id": pid, "rid": rid_of(rec), "qid": rec.get("qid"),
            "chain_id": rec.get("chain_id"), "session_id": rec.get("session_id"),
            "span_idx": rec.get("span_idx"), "lag": rec.get("lag"),
            "slice": rec.get("slice"), "answerable": rec.get("answerable", True),
            "pass": pass_no if args.calibration else None,
            "score": verdict["score"], "unsupported_claim": verdict["unsupported_claim"],
            "judge_abstained": verdict["abstained"], "abstained": post["abstained"],
            "abstained_source": post["abstained_source"],
            "reason": verdict["reason"],
            "anchor_hit": post["anchor_hit"],
            "hallucination": post["hallucination"],
            "hallucination_rule": post["hallucination_rule"],
            "facts_verified_in_transcript": post["facts_verified_in_transcript"],
            "fact_tokens_missing": post["fact_tokens_missing"],
            "verification": post["verification"],
            "concrete_tokens": post["concrete_tokens"],
            "excerpt_status": how, "excerpt_chars": len(excerpt),
            "excerpt_source_chain": src_chain,
            "judge_parse_ok": ok, "judge_model": model,
            "judge_same_family": fam["judge_same_family"],
            "judge_latency_s": round(latency, 3),
        })
        if n % 10 == 0 or n == len(presentations):
            print(f"  judged {n}/{len(presentations)}"
                  + (f" (${spend['usd']:.4f})" if not args.dry_run else ""), flush=True)

    with open(out, "w") as w:
        for j in judged:
            w.write(json.dumps(j, ensure_ascii=False) + "\n")
    mapping["masking"] = {"records_masked": len(masked_rids),
                          "rids": sorted(masked_rids), "pattern": ARM_LEAK_RE.pattern,
                          "replacement": MASK}
    json.dump(mapping, open(mapping_path, "w"), ensure_ascii=False, indent=1)

    # ---- report
    print(f"\njudged {len(judged)} presentations over {len(records)} records "
          f"({parse_fail} unparseable judge replies)")
    print(f"out: {out}\nmapping (arm labels, kept separate): {mapping_path}")
    print(f"blindness: arm name present in a prompt for {len(arm_in_prompt)} record(s)" +
          (f" -> {arm_in_prompt[:3]}" if arm_in_prompt else " (clean)"))
    print(f"blindness: arm-revealing strings masked in {len(masked_rids)} answer(s)" +
          (f" -> {sorted(masked_rids)[:3]} (logged in mapping 'answer_masked')"
           if masked_rids else ""))
    print(f"excerpt status: " + ", ".join(
        f"{k}={sum(1 for j in judged if j['excerpt_status'] == k)}"
        for k in sorted({j["excerpt_status"] for j in judged})))
    for arm in sorted({str(r.get("arm")) for r in records if r.get("arm")}):
        rs = [j for j in judged
              if mapping["map"].get(j["present_id"], {}).get("arm") == arm]
        for label, group in (("answerable", [j for j in rs if j.get("answerable")]),
                             ("unanswerable", [j for j in rs if not j.get("answerable")])):
            if not group:
                continue
            scored = [j["score"] for j in group if j["score"] is not None]
            print(f"  {arm:14s} {label:13s} n={len(group):3d} "
                  f"score={sum(scored) / max(1, len(scored)):.2f} "
                  f"anchor_hit={sum(bool(j['anchor_hit']) for j in group) / len(group):.2f} "
                  f"halluc={sum(bool(j['hallucination']) for j in group) / len(group):.2f} "
                  f"abstained={sum(bool(j['abstained']) for j in group) / len(group):.2f}")
    print(f"  hallucination rules: " + ", ".join(
        f"{k}={sum(1 for j in judged if j['hallucination_rule'] == k)}"
        for k in sorted({j["hallucination_rule"] for j in judged})))
    print(f"  verification: " + ", ".join(
        f"{k}={sum(1 for j in judged if (j.get('verification') or {}).get('verdict') == k)}"
        for k in ("supported", "unsupported", "unverifiable")))
    if args.calibration:
        cal = calibration_report(judged)
        print("\ncalibration: " + json.dumps(cal, indent=1))
        json.dump(cal, open(os.path.join(os.path.dirname(out) or ".",
                                         "judge-calibration.json"), "w"), indent=1)
    if not args.dry_run:
        print(f"judge spend: ${spend['usd']:.5f} over {spend['calls']} calls "
              f"(cap ${args.budget_usd:.2f}; profile {prof_name}, model {args.model})")
    else:
        print("dry run: no API calls, $0")
    return 0 if parse_fail == 0 else 0   # parse failures are reported, not fatal


if __name__ == "__main__":
    sys.exit(main())
