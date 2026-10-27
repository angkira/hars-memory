#!/usr/bin/env python3
"""Long-session eval: 1M+ token sessions, 5-100 compactions, baseline vs HARS memory arms.

Input: longsession/1 sessions (see longsession.py docstring; import_beam.py, import_claude_code.py,
gen_longsession_qa.py produce them). Nothing here runs by itself: the OWNER runs it with their
models / embedder / HARS. ``--dry-run`` needs no network at all.

BOUNDARIES (arm-independent, so every arm sees the same history)
  synthetic  after message i joins the live context, if tokens(live) + reserve > W (--window) a
             compaction fires: the last <= T tokens (--keep-tail, whole messages) stay as the TAIL,
             everything before them since the previous boundary is DROPPED (each message is
             summarised exactly once). A forced final boundary (default on) makes "ask at the end"
             = "ask right after a compaction".
  real       --use-real-compactions: at_idx from the session's own compactions (Claude Code);
             the tail is the next <= T tokens after the boundary. No forced final boundary.

ARMS (per session, per memory budget B in --mem-budgets)
  baseline   rolling LLM summary = f(previous summary + dropped span), hard-trimmed to B tokens
             (section trimmer, port of neurocompact.ts trimNote). Nothing is written to HARS.
  nc-chunks  rolling short NOTE (cap = note_cap_formula(B), 640 @ B=1536) + the WHOLE dropped span
             to HARS via memory_compact in ~8k-char pieces cut at message boundaries
             (--piece-chars, --pieces-per-call per call). Ask = note + packed memory_recall
             (session_graph off), recall cap = clamp(B - realized note, floor(B), ceil(B)).
  nc-graph   the SAME HARS session and note as nc-chunks; the only difference is
             ``session_graph: true`` on memory_recall (Amendment C ablation).
  nc-note-only  the same note, NO recall (control: what does recall add beyond the note?).
  orig       the session's own REAL summary at a real boundary (Claude Code); only at checkpoints
             that are real boundaries; not budget-limited (memory_tokens = its size).
  Deviation: across the --mem-budgets sweep ONE HARS session per sid is shared (the stored note is
  the one of the largest B; per-B notes are used in context). --hars-per-budget = strict per-B
  sessions at B x the extraction cost/time.

ASK: at each checkpoint (--checkpoints end|all|every:N|1,5,9) each arm gets: its memory, then the
  same verbatim TAIL (messages [at_idx, trigger_idx]), then ASK_PROMPT + question (+ packed recall
  for nc arms). Questions are eligible once their evidence has happened; ``lag`` = number of
  boundaries between the (oldest) evidence and the ask (1 = summarised once, 0 = still in the tail).

VALIDITY (per sid x arm x B; rows buffered, written once with the final flag)
  invalid: summariser/ingest failure, empty or errored recall, 0 packed content units, degraded
  (lexical_only) recall unless --allow-degraded (then only recorded), nc-graph without a
  session_graph block / status error, HARS status gate (entities, vectors_indexed > 0), unrepaired
  stop=length on > 1/6 of the summaries, note alone busting B.

ARTIFACTS (CLAUDE.md "never lose a run"): <out>/ = /mnt/datasets/ft/artifacts/<UTC>-longsession-<name>/
  samples.jsonl (per ask: answer, usage, cost, metrics, latency, memory text, injected recall),
  records.jsonl (same rows, metrics_v2 / export_judge_input compatible keys), compactions.jsonl
  (per boundary per arm: summariser usage/cost/clip, HARS ingest), summary.json, run.log, inputs/
  (sha256 + canonical path per session file), scripts/ (+ cmdline.txt), recall/ (raw payloads),
  memory/ (per-boundary summaries/notes: resume cache), done.jsonl (resume markers), MANIFEST.json.

COST (see README-longsession.md): printed per arm by --dry-run from token counts only.

Usage
  python3 run_longsession.py --sessions DIR --dry-run --mem-budgets 1536 --window 150000
  ANTHROPIC_API_KEY=... python3 run_longsession.py --sessions DIR --profile anthropic \\
      --agent-model claude-haiku-5-5 --mcp http://127.0.0.1:8307/mcp --name beam1M --budget-usd 30
  (--budget-usd is a HARD cap and mandatory for a live run; it also carries the flagged ESTIMATE of
   the HARS-side extraction per ingested piece; --max-cost-usd / --provider are kept aliases)
  python3 run_longsession.py --sessions DIR --resume /mnt/datasets/ft/artifacts/<dir> ...
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types
import urllib.request
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import longsession as L  # noqa: E402
import providers as P  # noqa: E402  (profiles, the single price source, hard --budget-usd)
import tokcount  # noqa: E402  (ONE token counter: the answering model's native tokens, Amendment C.5)
from longsession import (LLM, LLMError, count_tokens, fmt_usd, note_cap_formula,  # noqa: E402
                         recall_ceil_formula, recall_floor_formula)

def ntok(text: str) -> int:
    """BUDGET-grade token count (Amendment C.5): the answering model's native tokenizer when a counter is
    installed (run_chainreplay.set_counter / tokcount.set_default), else the fast estimator. Every
    realized-token, memory_tokens and cap number in this runner goes through here; plain count_tokens()
    stays the fast estimator for dry-run projections and serialization sizing."""
    return tokcount.count(text)


def tok_label() -> str:
    c = tokcount.get_default()
    return c.tokenizer_id if c is not None else L.TOKENIZER


def tok_native() -> bool:
    c = tokcount.get_default()
    return bool(c is not None and c.native)


def make_counter(a, live: bool):
    """The run's counter (installed as the tokcount default and into run_chainreplay for the packer)."""
    prov = "other" if a.provider == L.MOCK else tokcount.provider_of_profile(a.provider)
    mode = (getattr(a, "token_counter", None) or os.environ.get("NC_TOKCOUNT") or "auto").lower()
    if (not live or a.provider == L.MOCK) and mode in ("auto", "native"):
        mode = "estimate"
    key = None
    if live and prov in tokcount.NATIVE_PROVIDERS and mode in ("auto", "native") and a.profile_obj is not None:
        try:
            key = P.resolve_key(a.profile_obj, a.key_env)
        except P.ProviderError:
            key = None
    ctr = tokcount.Counter(prov, a.qa_model or a.agent_model or "", mode=mode, key=key,
                           concurrency=int(getattr(a, "tokcount_concurrency", 4) or 4))
    if live and prov in tokcount.NATIVE_PROVIDERS and mode in ("auto", "native") and not ctr.native:
        raise SystemExit(f"!!! no native token counter for {prov}/{ctr.model}: set the provider key (count_tokens "
                         "is free) or pass --token-counter tiktoken to accept a NON-native parity measurement")
    if ctr.native:
        ctr.count("tokcount preflight: the quick brown fox jumps over the lazy dog")     # fail before any spend
    tokcount.set_default(ctr)
    rcm = L.rc()
    if rcm is not None and hasattr(rcm, "set_counter"):
        rcm.set_counter(ctr)
    return ctr


def trim_note_native(text: str, cap: int, *, enabled: bool = True, force_repair: bool = False):
    """trim_note() to a NATIVE-token cap: trims in estimator units and verifies with the counter."""
    return tokcount.fit_native(tokcount.get_default(), text, cap,
                               lambda t, ec: trim_note(t, ec, enabled=enabled, force_repair=force_repair))


ARM_BASELINE, ARM_CHUNKS, ARM_GRAPH, ARM_ORIG = "baseline", "nc-chunks", "nc-graph", "orig"
ARM_NOTE = "nc-note-only"
ARM_ALIASES = {"nc": ARM_CHUNKS}
ALL_ARMS = (ARM_BASELINE, ARM_CHUNKS, ARM_GRAPH, ARM_NOTE, ARM_ORIG)
NC_ARMS = (ARM_CHUNKS, ARM_GRAPH)               # arms that recall from HARS
NOTE_ARMS = (ARM_CHUNKS, ARM_GRAPH, ARM_NOTE)   # arms that carry the rolling note
PROMPT_VERSION = "ls-1"
SUM_MAX_TOKENS_SLACK = 1.8        # summariser max_tokens = cap*slack+64: real provider tokens run up to ~1.7x the
                                  # chars/3.6 estimate on code/ids, so max_tokens=cap clips; trim_note enforces cap
FILL = 0.85                       # summariser soft target = 85% of the cap (neurocompact.ts)
ANSWER_OUT_EST = 100              # tokens per answer (dry-run estimate)
ASK_FRAME_TOKENS = 60
EXTRACT_IN_PER_CHUNK = 2700       # HARS extraction estimate per ~1200-token chunk: prompt 1500 +
EXTRACT_OUT_PER_CHUNK = 600       #   chunk 1200 in / 600 out. UNVERIFIED assumption, CLI-overridable
CHUNK_TOKENS = 1200
EXTRACT_S_PER_PIECE = 11.0        # HARS ingest seconds per piece: measured 10-12 s/piece in the 2026-10-26 smoke
                                  # (Haiku extractor, 2.5k-char pieces; 8k-char pieces are slower) - calibrate
CLIP_LIMIT = 1.0 / 6.0
SECTIONS = ("GOAL", "KEY FIGURES & IDENTIFIERS", "DECISIONS & RATIONALE", "ACTIVE STATE")
DEFAULT_ARTIFACT_ROOT = "/mnt/datasets/ft/artifacts"
DEFAULT_RESERVE = 2048            # tokens kept free for the memory artifact in the window; a constant,
                                  # NOT max(--mem-budgets), so boundaries do not move with the sweep


# =============================================================================
# summariser prompts (mirror of pi-agent/extensions/neurocompact.ts; kept in sync by hand)
# =============================================================================
def common_rules(cap: int) -> str:
    soft = max(64, int(cap * FILL))
    return ("Use exactly these four section headers, each on its own line, in this order:\n"
            + "\n".join(SECTIONS) + "\n"
            f"Output budget: the hard output limit is {cap} tokens. Target about {soft} tokens "
            f"(~85% of that) and stop writing when you reach the target; a note that is complete at "
            f"{soft} tokens is better than one that is cut at {cap}.\n"
            "The note is read top-down and its TAIL must always exist, so spend the budget on the "
            "tail, not on the head: DECISIONS & RATIONALE and ACTIVE STATE must always be complete, "
            "and ACTIVE STATE must always be present.\n"
            "Plain text under each header; no tables, no markdown fences.")


def summarizer_prompt(kind: str, cap: int, prev: str, conv: str) -> str:
    """kind = "baseline" (comprehensive, self-sufficient) | "note" (short index; the facts live in
    HARS). Same schema and budget rules in both, only the verbosity instruction differs."""
    rules = common_rules(cap)
    pv = f"\n\nPrevious handoff note (keep its active state):\n{prev}" if prev.strip() else ""
    if kind == "baseline":
        head = ("Write the continuation summary of the conversation below for a colleague who will "
                "continue this work with ONLY this text in context. Be comprehensive: preserve every "
                "number, percentage and metric (with what it measures), every model/dataset/file name, "
                "identifier, path, command, version, decision and its rationale, and the exact current "
                "state.\n")
    else:
        head = ("Write the continuation note of the conversation below. The dropped context and this "
                "note are also stored in a searchable long-term memory that is queried automatically "
                "when a question about earlier detail is asked, so this note is a short INDEX of what "
                "matters, not a copy of the facts and not a self-sufficient record. Write tersely: "
                "short lines, no narration of the conversation, and no exhaustive enumeration (KEY "
                "FIGURES & IDENTIFIERS holds at most five entries; the rest are recallable). Keep only "
                "what a reader must not get wrong: the goal in one line, the identifiers that are easy "
                "to confuse, the decisions already taken and why, and what is in flight. If you are "
                "short of room, compress or drop earlier sections first - ACTIVE STATE is written last "
                "and must always be there and complete.\n")
    return f"{head}{rules}{pv}\n\n<conversation>\n{conv}\n</conversation>"


# =============================================================================
# note trimmer (python port of neurocompact.ts trimNote, simplified)
# =============================================================================
_HDR_STRIP = re.compile(r"^[#*`\s]+|[#*`:\s]+$")
_DANGLING = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*$")
_END_OK = tuple(".!?;:)]}`\"'")


def parse_note(text: str) -> tuple[list[str], list[dict]]:
    pre: list[str] = []
    secs: list[dict] = []
    for ln in text.splitlines():
        name = _HDR_STRIP.sub("", ln).upper()
        if name in SECTIONS and len(ln.strip()) <= len(name) + 8:
            secs.append({"name": name, "hdr": ln, "body": []})
        elif secs:
            secs[-1]["body"].append(ln)
        else:
            pre.append(ln)
    return pre, secs


def render_note(pre: list[str], secs: list[dict]) -> str:
    out = list(pre)
    for s in secs:
        out.append(s["hdr"])
        out += s["body"]
    return "\n".join(out).strip()


def cut_to_last_sentence(line: str) -> str | None:
    t = line.rstrip()
    idx = max((t.rfind(c) for c in ".!?;"), default=-1)
    return t[:idx + 1] if idx > 0 else None


def trim_note(text: str, cap: int, *, enabled: bool = True, force_repair: bool = False) -> tuple[str, dict]:
    """(trimmed text, info). Priority cut (DECISIONS trailing lines, then GOAL, then KEY FIGURES;
    ACTIVE STATE is never touched), then a boundary repair of the tail (ends at a sentence/bullet).
    Notes without parsable sections that exceed the cap are hard-cut by whole lines."""
    info = {"before": count_tokens(text), "trim_applied": False, "boundary_repaired": False,
            "over_cap": False, "unparsed": False, "trimmed_sections": []}
    pre, secs = parse_note(text)
    if not secs:
        if enabled and info["before"] > cap:
            info["unparsed"] = info["trim_applied"] = True
            lines = text.splitlines()
            while lines and count_tokens("\n".join(lines)) > cap:
                lines.pop()
            text = "\n".join(lines)
        info["after"] = count_tokens(text)
        info["over_cap"] = info["after"] > cap
        return text.strip(), info

    def over() -> bool:
        return count_tokens(render_note(pre, secs)) > cap

    def find(name):
        return next((s for s in secs if s["name"] == name), None)

    def drop_trailing(s, keep_min=1) -> bool:
        while len(s["body"]) > keep_min:
            s["body"].pop()
            info["trim_applied"] = True
            if s["name"] not in info["trimmed_sections"]:
                info["trimmed_sections"].append(s["name"])
            while len(s["body"]) > keep_min and not s["body"][-1].strip():
                s["body"].pop()
            if not over():
                return True
        return not over()

    def shorten_last(s) -> bool:
        if not s["body"]:
            return False
        cut = cut_to_last_sentence(s["body"][-1])
        if not cut or cut == s["body"][-1]:
            return False
        s["body"][-1] = cut
        info["trim_applied"] = True
        return not over()

    def drop_section(s) -> bool:
        secs.remove(s)
        info["trim_applied"] = True
        info["trimmed_sections"].append(s["name"])
        return not over()

    if enabled and over():
        done = False
        for name in ("DECISIONS & RATIONALE", "GOAL", "KEY FIGURES & IDENTIFIERS"):
            s = find(name)
            if not done and s:
                done = drop_trailing(s) or shorten_last(s) or drop_section(s)
    for _ in range(400 if enabled else 0):
        last = secs[-1] if secs else None
        if not last or not last["body"]:
            break
        line = last["body"][-1]
        if not line.strip():
            last["body"].pop()
            continue
        if line.rstrip().endswith(_END_OK) and not (force_repair and _DANGLING.match(line)):
            break
        cut = cut_to_last_sentence(line)
        if cut and cut.strip() != line.strip() and not (force_repair and _DANGLING.match(cut)):
            last["body"][-1] = cut
            info["boundary_repaired"] = True
            continue
        if len(last["body"]) > 1 or last["name"] != "ACTIVE STATE":
            last["body"].pop()
            info["boundary_repaired"] = True
            continue
        break
    out = render_note(pre, secs)
    info["after"] = count_tokens(out)
    info["over_cap"] = info["after"] > cap
    return out, info


# =============================================================================
# recall cap, rubric proxy, anchor match
# =============================================================================
def recall_cap(B: int, note_tokens: int) -> dict:
    """clamp(B - realized note, floor(B), ceil(B)) - same arithmetic as run_chainreplay.recall_cap_for."""
    lo, hi = recall_floor_formula(B), recall_ceil_formula(B)
    target = int(B) - int(note_tokens)
    cap = max(lo, min(hi, target))
    return {"mem_budget": int(B), "nc_note_tokens": int(note_tokens), "recall_uncapped_target": target,
            "recall_cap": cap, "recall_cap_clamped_low": target < lo, "recall_cap_clamped_high": target > hi,
            "recall_cap_min": lo, "recall_cap_max": hi,
            "recall_cap_arithmetic": f"clamp({B} - note {note_tokens} = {target}, [{lo}, {hi}]) = {cap}"}


_RUB = re.compile(r"(?i)(?:should|must)\s+(?:state|mention|contain|include|say)\s*:?\s*(.+)$")
_WORD = re.compile(r"[a-z0-9]+")


def rubric_hit_frac(answer: str, rubric: list[str]) -> float | None:
    """Cheap deterministic proxy (NOT a judge): share of rubric phrases whose content words are
    >= 80% present in the answer (exact normalised substring for short phrases)."""
    phrases = []
    for r in rubric or []:
        m = _RUB.search(r)
        phrases.append((m.group(1) if m else r).strip())
    phrases = [p for p in phrases if p]
    if not phrases:
        return None
    a = " ".join(_WORD.findall((answer or "").lower()))
    aw = set(a.split())
    hit = 0
    for p in phrases:
        pw = _WORD.findall(p.lower())
        if not pw:
            continue
        if " ".join(pw) in a or (len(pw) > 3 and sum(w in aw for w in pw) / len(pw) >= 0.8):
            hit += 1
    return round(hit / len(phrases), 4)


def anchor_pattern(anchor: str):
    return re.compile(r"(?<![0-9A-Za-z_])" + re.escape(anchor) + r"(?![0-9A-Za-z_])", re.I)


# =============================================================================
# plan (pure, deterministic): boundaries, checkpoints, questions
# =============================================================================
def parse_budgets(spec: str) -> list[int]:
    out = sorted({int(x) for x in str(spec).split(",") if x.strip()}, reverse=True)
    if not out or any(b < 128 for b in out):
        raise SystemExit(f"bad --mem-budgets {spec!r} (integers >= 128)")
    return out


def resolve_checkpoints(spec: str, n_boundaries: int) -> list[int]:
    if n_boundaries <= 0:
        return []
    js: set[int] = set()
    for part in [p.strip() for p in str(spec).split(",") if p.strip()]:
        if part == "end":
            js.add(n_boundaries)
        elif part == "all":
            js.update(range(1, n_boundaries + 1))
        elif part.startswith("every:"):
            n = max(1, int(part.split(":", 1)[1]))
            js.update(range(n, n_boundaries + 1, n))
            js.add(n_boundaries)
        elif part.isdigit() and 1 <= int(part) <= n_boundaries:
            js.add(int(part))
    return sorted(js) or ([n_boundaries] if n_boundaries else [])


def select_questions(questions: list[dict], boundaries: list[dict], j: int, *, qtypes: set[str],
                     lags: set[int], max_q: int, seed: int) -> list[tuple[dict, int | None]]:
    b = boundaries[j - 1]
    el = []
    for q in questions:
        if qtypes and q.get("qtype") not in qtypes:
            continue
        if not L.question_eligible(q, b):
            continue
        lag = L.question_lag(q, boundaries, j)
        if lags and lag is not None and lag not in lags:
            continue
        el.append((q, lag))
    key = lambda ql: hashlib.sha256(f"{seed}:{ql[0]['qid']}".encode()).hexdigest()  # noqa: E731
    el.sort(key=key)
    if max_q and len(el) > max_q:                 # stratified round-robin over qtype
        by: dict[str, list] = defaultdict(list)
        for ql in el:
            by[ql[0].get("qtype", "?")].append(ql)
        picked: list = []
        while len(picked) < max_q and any(by.values()):
            for t in sorted(by):
                if by[t] and len(picked) < max_q:
                    picked.append(by[t].pop(0))
        el = picked
    return sorted(el, key=lambda ql: ql[0]["qid"])


def build_plan(session: dict, args) -> dict:
    msgs = session["messages"]
    tok = L.message_tokens(msgs)
    real = None
    if args.use_real_compactions:
        real = [c for c in session.get("compactions") or []]
    final = args.final_compaction if args.final_compaction is not None else (False if real is not None else True)
    bs = L.compute_boundaries(tok, window=args.window, keep_tail=args.keep_tail, reserve=args.reserve,
                              real=real, final=final) if (real is None or real) else []
    cps = resolve_checkpoints(args.checkpoints, len(bs))
    qtypes = {t for t in args.qtypes.split(",") if t} if args.qtypes else set()
    lags = {int(x) for x in args.lags.split(",") if x.strip()} if args.lags else set()
    ask = {j: select_questions(session.get("questions") or [], bs, j, qtypes=qtypes, lags=lags,
                               max_q=args.max_questions, seed=args.seed) for j in cps}
    return {"sid": session["sid"], "source": session.get("source"), "n_messages": len(msgs),
            "total_tokens": sum(tok), "tok": tok, "boundaries": bs, "checkpoints": cps, "ask": ask,
            "real": real is not None, "n_real_summaries": sum(1 for b in bs if b.get("original_summary")),
            "session": session}


# =============================================================================
# cost estimate (offline)
# =============================================================================
def usd_for(args, kind: str, tin: float, tout: float) -> float:
    """USD of ONE request of `kind` (sum | qa | ext) with tin prompt / tout completion tokens:
    the profile's tier-aware price (Haiku 5.5: 0.10/0.50, 0.50/2.50 above 100k prompt tokens)
    unless a --price-in/--price-out override is active, else the flat (in, out) tuple."""
    prices = getattr(args, "prices_cache", None) or {}
    flat = prices.get(kind, (0.0, 0.0))
    ov = (args.extractor_price_in, args.extractor_price_out) if kind == "ext" else (args.price_in, args.price_out)
    prof = getattr(args, "profile_obj", None)
    model = {"sum": args.sum_model, "qa": args.qa_model, "ext": args.extractor_model}[kind]
    if prof is not None and None in ov:
        c = prof.cost_usd(model, int(tin), int(tout))
        if c is not None:
            return c
    return (tin * flat[0] + tout * flat[1]) / 1e6


def usd_ask(args, k: int, prefix_tok: float, suffix_tok: float, tout: float) -> float:
    """USD of the k-th answer call of a group that shares `prefix_tok` prompt tokens (memory +
    verbatim tail). Prompt caching ON (cost-only, Amendment C.3): the first call writes the prefix,
    the rest read it; OFF or a price override: everything at plain input price."""
    prof = getattr(args, "profile_obj", None)
    ov = (args.price_in, args.price_out)
    if prof is not None and prof.cache_enabled and None in ov:
        c = prof.cost_usd(args.qa_model, int(suffix_tok), int(tout),
                          cache_read=0 if k == 0 else int(prefix_tok),
                          cache_write=int(prefix_tok) if k == 0 else 0, cache_ttl=str(prof.cache_cfg.get("ttl") or "5m"))
        if c is not None:
            return c
    return usd_for(args, "qa", prefix_tok + suffix_tok, tout)


def estimate_cost(plan: dict, args, prices: dict) -> dict:
    """Per-B, per-arm USD estimate from token counts only (formulas in README-longsession.md)."""
    msgs = plan["session"]["messages"]
    bs = plan["boundaries"]
    ep = prices["ext"]
    out: dict = {"per_budget": {}, "hars": {}}
    ser_chars = [sum(len(L.serialize_message(m)) + 2 for m in msgs[b["drop_start"]:b["at_idx"]]) for b in bs]
    ask_p = count_tokens(L.ask_prompt()) + ASK_FRAME_TOKENS
    # HARS ingest (shared by nc-chunks / nc-graph; once per sid unless --hars-per-budget)
    pieces = [max(1, L.ceil_div(c, args.piece_chars)) for c in ser_chars]
    chunks = [max(1, L.ceil_div(b["dropped_tokens"], CHUNK_TOKENS)) for b in bs]
    calls = [L.ceil_div(p, args.pieces_per_call) for p in pieces]
    ext_usd = sum(chunks) * usd_for(args, "ext", args.extract_in_per_chunk, args.extract_out_per_chunk)
    out["hars"] = {"pieces": sum(pieces), "memory_compact_calls": sum(calls), "extract_chunks": sum(chunks),
                   "extract_usd": ext_usd, "est_ingest_hours": sum(pieces) * args.extract_s_per_piece / 3600.0}
    for B in args.budgets:
        ncap = note_cap_formula(B)
        res: dict[str, float] = {}
        for kind, cap, arm in (("baseline", B, ARM_BASELINE), ("note", ncap, "note")):
            p0 = count_tokens(summarizer_prompt(kind, cap, "", ""))
            tin = tout = usd = 0.0
            prev = 0.0
            for b in bs:
                slices = max(1, L.ceil_div(b["dropped_tokens"], max(1, args.sum_max_input - p0 - cap)))
                tin += slices * (p0 + prev) + b["dropped_tokens"]
                tout += slices * cap * FILL
                per_req_in = p0 + prev + b["dropped_tokens"] / slices      # tier is chosen per request
                usd += slices * usd_for(args, "sum", per_req_in, cap * FILL)
                prev = cap * FILL
            res[arm] = usd
            res[arm + "_tokens_in"] = tin
            res[arm + "_tokens_out"] = tout
        asks = {a: [0, 0.0, 0.0, 0.0] for a in ALL_ARMS}     # arm -> [n asks, input tokens, usd, usd without cache]
        for j, ql in plan["ask"].items():
            b = bs[j - 1]
            for k, (q, _lag) in enumerate(ql):
                qtok = count_tokens(q["question"]) + 4
                base_in = ask_p + b["tail_tokens"] + qtok
                pre = b["tail_tokens"] + ASK_FRAME_TOKENS            # + the memory (B or the orig summary)
                suf = base_in - pre
                for a in (ARM_BASELINE, ARM_CHUNKS, ARM_GRAPH, ARM_NOTE):
                    asks[a][0] += 1
                    asks[a][1] += base_in + B
                    asks[a][2] += usd_ask(args, k, pre + B, suf, ANSWER_OUT_EST)
                    asks[a][3] += usd_for(args, "qa", base_in + B, ANSWER_OUT_EST)
                if b.get("original_summary") and B == max(args.budgets):     # orig runs once (reference B)
                    ot = count_tokens(b["original_summary"])
                    asks[ARM_ORIG][0] += 1
                    asks[ARM_ORIG][1] += base_in + ot
                    asks[ARM_ORIG][2] += usd_ask(args, k, pre + ot, suf, ANSWER_OUT_EST)
                    asks[ARM_ORIG][3] += usd_for(args, "qa", base_in + ot, ANSWER_OUT_EST)
        n_recalls = sum(len(ql) for ql in plan["ask"].values())
        kw = n_recalls * usd_for(args, "ext", rc_kw_in(), rc_kw_out())
        arm_usd = {}
        for a in ALL_ARMS:
            n, tin, ask_usd, _nc = asks[a]
            owner = next((x for x in NOTE_ARMS if x in args.arms_list), None)   # carries the shared note cost
            if a == ARM_BASELINE:
                arm_usd[a] = res["baseline"] + ask_usd
            elif a == ARM_GRAPH:                    # note + HARS are shared with nc-chunks
                arm_usd[a] = ask_usd + kw + (res["note"] if owner == a else 0.0)
            elif a in NOTE_ARMS:
                arm_usd[a] = ask_usd + (res["note"] if owner == a else 0.0)
            else:
                arm_usd[a] = ask_usd
        out["per_budget"][B] = {"usd": arm_usd, "asks": {a: asks[a][0] for a in ALL_ARMS},
                                "ask_tokens_in": {a: asks[a][1] for a in ALL_ARMS},
                                "summarizer": {k: v for k, v in res.items() if k.endswith(("_in", "_out"))},
                                "note_cap": ncap, "graph_keyword_usd": kw,
                                "cache_saving_usd": sum(asks[a][3] - asks[a][2] for a in ALL_ARMS if a in args.arms_list)}
    return out


def rc_kw_in() -> float:
    return float(L.rc_attr("KEYWORD_IN", 450.0))


def rc_kw_out() -> float:
    return float(L.rc_attr("KEYWORD_OUT", 80.0))


def active_arms(args, plan: dict | None = None) -> list[str]:
    return list(args.arms_list)


def total_usd(est: dict, arms: list[str], hars_sessions_factor: int = 1) -> float:
    t = est["hars"]["extract_usd"] * hars_sessions_factor if any(a in NC_ARMS for a in arms) else 0.0
    for B, d in est["per_budget"].items():
        t += sum(d["usd"][a] for a in arms)
    return t


def print_dry_run(plans: list[dict], args, prices: dict, *, as_json: bool = False) -> dict:
    args.prices_cache = prices
    res = {"config": {"window": args.window, "keep_tail": args.keep_tail, "reserve": args.reserve,
                      "budgets": args.budgets, "checkpoints": args.checkpoints, "arms": args.arms_list,
                      "real": bool(args.use_real_compactions), "tokenizer": tok_label(),
                      "prices_usd_per_mtok": {k: list(v) for k, v in prices.items()}},
           "sessions": []}
    grand = 0.0
    lines = [f"DRY RUN (no network). tokenizer={tok_label()} window={args.window} keep_tail={args.keep_tail} "
             f"reserve={args.reserve} budgets={args.budgets} arms={args.arms_list} "
             f"checkpoints={args.checkpoints}",
             f"prices USD/MTok (in,out): sum={prices['sum']} qa={prices['qa']} extractor={prices['ext']}"]
    for p in plans:
        est = estimate_cost(p, args, prices)
        arms = args.arms_list
        hf = len(args.budgets) if args.hars_per_budget else 1
        tot = total_usd(est, arms, hf)
        grand += tot
        dr = [b["dropped_tokens"] for b in p["boundaries"]]
        sess = {"sid": p["sid"], "messages": p["n_messages"], "tokens": p["total_tokens"],
                "boundaries": len(p["boundaries"]),
                "dropped_tokens": {"min": min(dr, default=0), "median": sorted(dr)[len(dr) // 2] if dr else 0,
                                   "max": max(dr, default=0)},
                "checkpoints": p["checkpoints"], "questions_per_checkpoint": {str(j): len(q) for j, q in p["ask"].items()},
                "real_summaries": p["n_real_summaries"], "hars": est["hars"],
                "per_budget": {str(B): {"usd": {a: round(v, 6) for a, v in d["usd"].items()},
                                        "asks": d["asks"], "note_cap": d["note_cap"]}
                               for B, d in est["per_budget"].items()},
                "total_usd": round(tot, 6),
                "cache_saving_usd": round(sum(d["cache_saving_usd"] for d in est["per_budget"].values()), 6)}
        res["sessions"].append(sess)
        lines.append(f"\n[{p['sid']}] {p['n_messages']} msgs, {p['total_tokens']:,} tokens, "
                     f"{len(p['boundaries'])} boundaries ({'real' if p['real'] else 'synthetic'}), "
                     f"dropped tokens/boundary min/med/max = {sess['dropped_tokens']['min']:,}/"
                     f"{sess['dropped_tokens']['median']:,}/{sess['dropped_tokens']['max']:,}")
        if not p["boundaries"]:
            lines.append("  no boundary fires (session shorter than the window): nothing to test")
            continue
        shown = p["boundaries"] if len(p["boundaries"]) <= 6 else p["boundaries"][:3] + [None] + p["boundaries"][-3:]
        for b in shown:
            if b is None:
                lines.append("    ...")
                continue
            lines.append(f"    j={b['j']:<3} trigger_idx={b['trigger_idx']:<6} at_idx={b['at_idx']:<6} "
                         f"dropped={b['dropped_tokens']:>8,}tok tail={b['tail_tokens']:>6,}tok"
                         f"{' real' if b['real'] else ''}{' final' if b['final'] else ''}"
                         f"{' orig' if b.get('original_summary') else ''}")
        lines.append(f"  checkpoints {p['checkpoints'][:12]}{'...' if len(p['checkpoints']) > 12 else ''}; "
                     f"questions asked per checkpoint: "
                     f"{dict(list(sess['questions_per_checkpoint'].items())[:6])}")
        h = est["hars"]
        lines.append(f"  HARS ingest: {h['pieces']:,} pieces in {h['memory_compact_calls']:,} memory_compact calls, "
                     f"{h['extract_chunks']:,} extraction chunks ~{fmt_usd(h['extract_usd'])}"
                     f"{' x' + str(len(args.budgets)) + ' budgets' if args.hars_per_budget else ''}, "
                     f"~{h['est_ingest_hours']:.1f} h (guess {args.extract_s_per_piece}s/piece)")
        for B, d in est["per_budget"].items():
            lines.append(f"  B={B} (note cap {d['note_cap']}): "
                         + " | ".join(f"{a} {fmt_usd(d['usd'][a])} ({d['asks'][a]} asks)"
                                      for a in ALL_ARMS if a in arms)
                         + (f" | graph-keyword {fmt_usd(d['graph_keyword_usd'])}" if ARM_GRAPH in arms else ""))
        lines.append(f"  session total (arms {arms}, HARS extraction included once"
                     f"{' per budget' if args.hars_per_budget else ''}): {fmt_usd(tot)}")
    res["grand_total_usd"] = round(grand, 6)
    res["grand_cache_saving_usd"] = round(sum(x["cache_saving_usd"] for x in res["sessions"]), 6)
    res["prompt_cache"] = bool(getattr(args, "profile_obj", None) and args.profile_obj.cache_enabled)
    lines.append(f"\nGRAND TOTAL {fmt_usd(grand)} over {len(plans)} session(s)"
                 " (includes the flagged estimate of HARS extraction, charged to the cap at run time)")
    lines.append(f"prompt cache {'ON' if res['prompt_cache'] else 'off'} (cost-only, Amendment C.3): answer calls "
                 f"estimated with cache write/read prices, saving vs uncached {fmt_usd(res['grand_cache_saving_usd'])}")
    unpriced = [m for m in {args.sum_model, args.qa_model, args.extractor_model}
                if args.provider != L.MOCK and P.price_tuple(m, getattr(args, "profile_obj", None)) is None
                and args.price_in is None]
    res["unpriced_models"] = sorted(unpriced)
    if unpriced:
        lines.append(f"WARNING unpriced model(s) {sorted(unpriced)}: add them to the profile or pass "
                     "--price-in/--price-out (a live run with --budget-usd refuses an unpriced call)")
    if args.budget_usd:
        res["budget_check"] = {"budget_usd": args.budget_usd, "estimate_usd": round(grand, 6),
                               "ok": grand <= args.budget_usd}
        lines.append(f"BUDGET CHECK: estimate {fmt_usd(grand)} vs --budget-usd {fmt_usd(args.budget_usd)}: "
                     + ("OK" if grand <= args.budget_usd else "OVER THE CAP (the run would stop early)"))
    if as_json:
        print(json.dumps(res, indent=1, sort_keys=True))
    else:
        print("\n".join(lines))
    return res


# =============================================================================
# HARS adapters (real + offline fake)
# =============================================================================
class Hars:
    """Real HARS over MCP (reuses run_chainreplay.mcp_call / hars_recall / hars_status)."""

    def __init__(self, mcp_url: str, compact_sem: threading.Semaphore, recall_sem: threading.Semaphore,
                 compact_timeout: int = 3600, recall_timeout: int = 600):
        self.rcm = L.rc_require("mcp_call", "hars_recall", "hars_status")
        self.url, self.cs, self.rs = mcp_url, compact_sem, recall_sem
        self.ct, self.rt = compact_timeout, recall_timeout
        self.run_tag = ""                      # set from --run-tag (a server lease tag); "" = do not send

    def health(self) -> dict:
        base = self.url.rsplit("/mcp", 1)[0]
        req = urllib.request.Request(base + "/healthz")
        op = self.rcm.opener_for(base) if hasattr(self.rcm, "opener_for") else None
        with (op.open(req, timeout=10) if op is not None else urllib.request.urlopen(req, timeout=10)) as r:
            return json.loads(r.read().decode() or "{}")

    def compact(self, sid: str, summary: str, dropped: list[str], metadata: dict) -> dict:
        with self.cs:
            args = {"session_id": sid, "summary": summary, "dropped_context": dropped, "metadata": metadata}
            if self.run_tag:
                args["run_tag"] = self.run_tag
            return self.rcm.mcp_call(self.url, "memory_compact", args, timeout=self.ct)

    def recall(self, sid: str, question: str, graph: bool) -> dict:
        with self.rs:
            return self.rcm.hars_recall(self.url, sid, question, timeout=self.rt, graph=graph)

    def status(self, sid: str) -> dict:
        return self.rcm.hars_status(self.url, sid)


class FakeHars:
    """Deterministic in-memory stand-in (tests / --fake-hars): stores pieces, recalls by token
    overlap, renders the LightRAG-style sections the run_chainreplay packer parses."""

    def __init__(self):
        self.s: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.compact_calls = 0
        self.recall_calls = 0
        self.fail_recall = False
        self.degraded = False

    def health(self) -> dict:
        return {"status": "ok", "fake": True}

    def compact(self, sid, summary, dropped, metadata):
        with self.lock:
            st = self.s.setdefault(sid, {"seq": 0, "docs": [], "ents": set()})
            st["seq"] += 1
            self.compact_calls += 1
            for d in [summary] + list(dropped):
                st["docs"].append(d)
                st["ents"].update(re.findall(r"[A-Za-z_][\w./-]*\d[\w./-]*", d))
            return {"ok": True, "seq": st["seq"], "session_id": sid}

    def status(self, sid):
        st = self.s.get(sid)
        if not st:
            return {"compactions": 0}
        return {"compactions": st["seq"], "entities": len(st["ents"]), "relations": len(st["ents"]) // 2,
                "vectors_indexed": len(st["docs"])}

    def recall(self, sid, question, graph):
        with self.lock:
            self.recall_calls += 1
            st = self.s.get(sid)
        if self.fail_recall:
            return {"context": "", "error": "fake recall failure"}
        if not st:
            return {"ok": True, "context": "", "session": {"status": "unknown_session"}}
        qw = set(re.findall(r"\w{3,}", question.lower()))
        ranked = sorted(st["docs"], key=lambda d: -len(qw & set(re.findall(r"\w{3,}", d.lower()))))[:8]
        chunks = ['{"reference_id": "%d", "content": %s, "section": ""}' % (i + 1, json.dumps(c))
                  for i, c in enumerate(ranked)]
        raw = ["Document Chunks (each has a reference_id):", "```json", *chunks, "```", ""]
        ents = sorted(e for e in st["ents"] if e.lower() in question.lower() or any(
            w in e.lower() for w in qw))[:5]
        if graph and ents:
            raw += ["Knowledge Graph Data (Entity):", "```json",
                    *[json.dumps({"entity": e, "type": "id", "description": f"identifier {e}"}) for e in ents],
                    "```", ""]
        raw += ["Reference Document List (for reference):", "```", f"[1] {sid}.md", "```"]
        payload = {"ok": True, "context": "\n".join(raw),
                   "session": {"status": "degraded" if self.degraded else "ok",
                               "vectors_indexed": len(st["docs"]), "entities": len(st["ents"]),
                               **({"degraded": "lexical_only"} if self.degraded else {})}}
        if graph:
            payload["session_graph"] = {"status": "ok" if ents else "empty", "keyword_source": "question_fallback",
                                        "entities": len(ents), "relations": 0, "latency_ms": 1.0}
        return payload


# =============================================================================
# run context, artifacts
# =============================================================================
class Tee:
    def __init__(self, stream, path):
        self.stream, self.f, self.lock = stream, open(path, "a", buffering=1), threading.Lock()

    def write(self, s):
        with self.lock:
            self.stream.write(s)
            self.f.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def isatty(self):
        return False


class Ctx:
    """Shared run state: models, HARS handle, artifact dir, locks, done set."""

    def __init__(self, args, out: str, hars, sum_llm, qa_llm, run_id: str, budget=None):
        self.args, self.out, self.hars, self.sum_llm, self.qa_llm, self.run_id = args, out, hars, sum_llm, qa_llm, run_id
        self.budget = budget                  # providers.Budget (hard cap) or None (mock/dry)
        self.est_hars_usd = 0.0               # flagged ESTIMATE of the HARS-side extraction
        self.lock = threading.Lock()
        self.done: set[tuple[str, str, int]] = set()
        self.totals = Counter()

    def path(self, *p) -> str:
        return os.path.join(self.out, *p)

    def append(self, name: str, row: dict):
        with self.lock, open(self.path(name), "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load_done(self):
        p = self.path("done.jsonl")
        if os.path.exists(p):
            for line in open(p):
                if line.strip():
                    d = json.loads(line)
                    self.done.add((d["sid"], d["arm"], int(d["B"])))


def atomic_write(path: str, data: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(data)
    os.replace(tmp, path)


class JsonCache:
    def __init__(self, path: str):
        self.path, self.lock = path, threading.Lock()
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            atomic_write(self.path, json.dumps(self.d))


# =============================================================================
# per-session execution
# =============================================================================
class BudgetExceeded(RuntimeError):
    pass


class SessionRun:
    def check_budget(self):
        b = self.c.budget
        if b is not None and b.spent >= b.limit:
            raise BudgetExceeded(f"spend ${b.spent:.4f} reached --budget-usd ${b.limit:.4f}")

    def __init__(self, ctx: Ctx, plan: dict):
        self.c, self.plan, self.a = ctx, plan, ctx.args
        self.s = plan["session"]
        self.sid = plan["sid"]
        self.msgs = self.s["messages"]
        self.mem = JsonCache(ctx.path("memory", f"{self.sid}.json"))
        self.partial_path = ctx.path("partial", f"{self.sid}.jsonl")
        self.partial: dict[str, dict] = {}
        if os.path.exists(self.partial_path):
            for line in open(self.partial_path):
                if line.strip():
                    r = json.loads(line)
                    self.partial[r["rid"]] = r
        self.arms = [a for a in self.a.arms_list]
        ref_b = max(self.a.budgets)          # orig is budget-independent: run it once (reference B)
        self.todo = {(a, B) for a in self.arms for B in self.a.budgets
                     if (self.sid, a, B) not in ctx.done and (a != ARM_ORIG or B == ref_b)}
        self.rows: dict[tuple[str, int], list[dict]] = defaultdict(list)
        self.reasons: dict[tuple[str, int], list[str]] = defaultdict(list)
        self.clip = defaultdict(lambda: [0, 0])      # (kind,B) -> [summaries, unrepaired clipped]
        self.base_text: dict[int, str] = {}
        self.note_text: dict[int, str] = {}
        self.ingest_ok = True
        self.ingest_err = ""
        self.hars_keys: dict[object, str] = {}

    # ------------------------------------------------------------- helpers
    def need(self, arm: str, B: int | None = None) -> bool:
        return any(a == arm and (B is None or b == B) for a, b in self.todo)

    def need_nc(self, B: int | None = None) -> bool:           # needs HARS ingest + recall
        return self.need(ARM_CHUNKS, B) or self.need(ARM_GRAPH, B)

    def need_note(self, B: int | None = None) -> bool:         # needs the rolling note
        return self.need_nc(B) or self.need(ARM_NOTE, B)

    def hars_sid(self, B: int) -> str:
        key = B if self.a.hars_per_budget else None
        suffix = f"-b{B}" if self.a.hars_per_budget else ""
        return f"ls-{self.c.run_id}-{self.sid}{suffix}"[:128]

    def note_budget_for_store(self) -> int:
        bs = [B for B in self.a.budgets if self.need_nc(B)] or self.a.budgets
        return max(bs)

    def log_comp(self, row: dict):
        self.c.append("compactions.jsonl", {"run_id": self.c.run_id, "sid": self.sid, **row})

    # ------------------------------------------------------------- summarisers
    def summarize(self, kind: str, B: int, j: int, prev: str, dropped: list[dict]) -> dict:
        cap = B if kind == "baseline" else note_cap_formula(B)
        conv_full = L.serialize(dropped)
        sha = hashlib.sha256(json.dumps([prev, hashlib.sha256(conv_full.encode()).hexdigest(), PROMPT_VERSION,
                                         self.c.sum_llm.model, cap, self.a.sum_max_input, tok_label()]).encode()).hexdigest()
        key = f"{kind}|B{B}|j{j}"
        hit = self.mem.get(key)
        if hit and hit.get("sha") == sha:
            self.log_comp({"j": j, "arm": kind, "B": B, "cached": True, "realized_tokens": hit["tokens"]})
            return hit
        self.check_budget()
        p0 = count_tokens(summarizer_prompt(kind, cap, "", ""))
        budget_in = max(2000, self.a.sum_max_input - p0 - cap)
        slices, cur, cur_t = [], [], 0
        for m, t in zip(dropped, L.message_tokens(dropped)):
            if cur and cur_t + t > budget_in:
                slices.append(cur)
                cur, cur_t = [], 0
            cur.append(m)
            cur_t += t
        if cur:
            slices.append(cur)
        text, tin, tout, cost, lat, clipped, repaired = prev, 0, 0, 0.0, 0.0, False, False
        info: dict = {}
        last_raw, prompt_shas = "", []
        for sl in slices or [[]]:
            prompt = summarizer_prompt(kind, cap, text, L.serialize(sl))
            prompt_shas.append(L.sha256_text(prompt))
            r = self.c.sum_llm.complete(prompt, max_tokens=int(cap * self.a.sum_max_tokens_slack) + 64)
            last_raw = r["text"]
            if not r["text"].strip():
                raise LLMError(f"empty {kind} summary at boundary {j} (stop={r['stop_reason']})")
            text, info = trim_note_native(r["text"], cap, enabled=True, force_repair=r["stop_reason"] == "length")
            tin += r["input_tokens"]
            tout += r["output_tokens"]
            cost += r["cost_usd"]
            lat += r["latency_s"]
            clipped |= r["stop_reason"] == "length"
            repaired |= info["boundary_repaired"]
        entry = {"sha": sha, "text": text, "raw": last_raw, "prompt_sha256": prompt_shas, "tokens": ntok(text), "tokenizer": tok_label(), "cap": cap, "slices": len(slices),
                 "input_tokens": tin, "output_tokens": tout, "cost_usd": cost, "latency_s": round(lat, 3),
                 "clipped": clipped, "clipped_repaired": clipped and repaired,
                 "trim": info, "over_cap": info.get("over_cap", False)}
        self.mem.put(key, entry)
        self.c.totals["sum_cost"] += cost
        self.log_comp({"j": j, "arm": kind, "B": B, "cached": False, **{k: v for k, v in entry.items()
                                                                       if k not in ("text", "sha", "raw")},
                       "realized_tokens": entry["tokens"], "summary_sha256": L.sha256_text(text)})
        return entry

    # ------------------------------------------------------------- HARS ingest
    def ingest(self, b: dict, dropped: list[dict], note: str, hsid: str, tag: str):
        key = f"ingest|{tag}|j{b['j']}"
        done = self.mem.get(key)
        pieces = L.split_pieces(dropped, self.a.piece_chars)
        n_calls = max(1, L.ceil_div(len(pieces), self.a.pieces_per_call))
        if done and done.get("ok"):
            self.log_comp({"j": b["j"], "arm": "hars", "cached": True, "calls": done["calls"]})
            return
        t0 = time.time()
        seqs = []
        for c in range(n_calls):
            chunk = pieces[c * self.a.pieces_per_call:(c + 1) * self.a.pieces_per_call]
            summary = note if c == 0 else f"Boundary {b['j']} continuation {c + 1}/{n_calls}: dropped context of the same compaction."
            meta = {"source": "longsession", "sid": self.sid, "boundary": b["j"], "call": c + 1,
                    "calls": n_calls, "pieces": len(chunk), "drop_chars": sum(len(x) for x in chunk)}
            err = ""
            for attempt in (0, 1):
                try:
                    r = self.c.hars.compact(hsid, summary, chunk, meta)
                    if isinstance(r, dict) and r.get("ok") is not False:
                        seqs.append(r.get("seq"))
                        err = ""
                        break
                    err = f"memory_compact ok=false: {str(r)[:200]}"
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt == 0:
                    time.sleep(2)
            if err:
                self.ingest_ok = False
                self.ingest_err = f"boundary {b['j']} call {c + 1}: {err}"
                self.log_comp({"j": b["j"], "arm": "hars", "ok": False, "error": err})
                return
        self.mem.put(key, {"ok": True, "calls": n_calls, "seqs": seqs})
        self.charge_extraction_estimate(b, dropped)
        self.log_comp({"j": b["j"], "arm": "hars", "ok": True, "cached": False, "pieces": len(pieces),
                       "calls": n_calls, "seconds": round(time.time() - t0, 2),
                       "drop_chars": sum(len(x) for x in pieces)})

    def charge_extraction_estimate(self, b: dict, dropped: list[dict]):
        """The HARS server bills its own extractor calls; this run charges a flagged ESTIMATE
        (extract_in/out_per_chunk x chunks at the extractor model's profile price) to the same
        hard cap so --budget-usd covers the whole run."""
        ep = self.a.prices_cache["ext"]
        chunks = max(1, L.ceil_div(sum(L.message_tokens(dropped)), CHUNK_TOKENS))
        usd = chunks * (self.a.extract_in_per_chunk * ep[0] + self.a.extract_out_per_chunk * ep[1]) / 1e6
        with self.c.lock:
            self.c.est_hars_usd += usd
        if self.c.budget is not None:
            self.c.budget.charge(usd)

    # ------------------------------------------------------------- asking
    def tail_text(self, b: dict) -> str:
        return L.serialize(self.msgs[b["at_idx"]:b["trigger_idx"] + 1])

    def prompt_parts(self, memory: str, tail: str, question: str, injected: str) -> tuple[str, str]:
        """(stable prefix, per-question suffix); prefix + suffix is the whole prompt, byte for byte.
        The prefix (memory + verbatim tail) is shared by every question asked at a boundary, so it
        is the part the provider prompt cache can serve (cost-only, Amendment C.3)."""
        prefix = ("You are continuing a long working session. Your earlier context was compacted; "
                  "what you still have is below.\n\n[Session memory]\n" + (memory.strip() or "(none)")
                  + "\n\n[Recent conversation, verbatim]\n" + (tail.strip() or "(none)") + "\n\n")
        return prefix, L.ask_prompt() + question + injected

    def build_prompt(self, memory: str, tail: str, question: str, injected: str) -> str:
        return "".join(self.prompt_parts(memory, tail, question, injected))

    def recall_payload(self, j: int, q: dict, hsid: str, graph: bool) -> dict:
        p = self.c.path("recall", self.sid, f"j{j}", f"{q['qid']}.{'g' if graph else 'c'}.json")
        if os.path.exists(p):
            return json.load(open(p))
        t0 = time.time()
        payload = self.c.hars.recall(hsid, q["question"], graph)
        payload = dict(payload) if isinstance(payload, dict) else {"context": "", "error": "non-dict payload"}
        payload["_seconds"] = round(time.time() - t0, 2)
        atomic_write(p, json.dumps(payload))
        return payload

    def ask_one(self, b: dict, q: dict, lag, is_end: bool, tail: str, tail_tokens: int) -> list[dict]:
        a, rcm = self.a, L.rc()
        j = b["j"]
        rows: list[dict] = []
        payloads: dict[tuple[bool, int], tuple] = {}
        graph_payload = {}
        for arm, B in sorted(self.todo):
            qid_row = q["qid"] if is_end else f"{q['qid']}@c{j}"     # unique (qid, arm, B) for metrics_v2
            rid = f"{qid_row}::{arm}::b{B}"
            if rid in self.partial:
                rows.append(self.partial[rid])
                continue
            reasons: list[str] = []
            injected, packed, pay, cap_info = "", {}, {}, {}
            recall_s = 0.0
            if arm == ARM_BASELINE:
                memory, note_tokens = self.base_text[B], 0
            elif arm == ARM_ORIG:
                if not b.get("original_summary"):
                    continue
                memory, note_tokens = b["original_summary"], 0
            elif arm == ARM_NOTE:
                memory, note_tokens = self.note_text[B], ntok(self.note_text[B])
            else:
                graph = arm == ARM_GRAPH
                memory = self.note_text[B]
                note_tokens = ntok(memory)
                cap_info = recall_cap(B, note_tokens)
                hs = self.hars_sid(B)
                pay = self.recall_payload(j, q, hs, graph)
                recall_s = float(pay.get("_seconds") or 0.0)
                try:
                    packed = rcm.pack_injection(pay.get("context") or "", hs, budget=cap_info["recall_cap"],
                                                query=q["question"])
                    injected = packed["injected"]
                except Exception as e:  # noqa: BLE001
                    packed = {"tokens": 0, "total_tokens": 0, "text": "", "raw_chars": 0, "packed_chars": 0}
                    reasons.append(f"pack failed: {type(e).__name__}: {str(e)[:120]}")
                reasons += list(rcm.recall_failure_reasons(pay, packed))
                if graph:
                    reasons += list(rcm.graph_failure_reasons(pay))
                    graph_payload = rcm.graph_summary(pay)
                deg = rcm.recall_degraded(pay)
                if deg and not a.allow_degraded:
                    reasons.append(f"recall degraded ({deg})")
                if not self.ingest_ok:
                    reasons.append(f"ingest failed: {self.ingest_err}")
                if cap_info["recall_cap_clamped_low"] and note_tokens > B:
                    reasons.append(f"note alone busts the budget ({note_tokens} > {B})")
            prefix, suffix = self.prompt_parts(memory, tail, q["question"], injected)
            prompt = prefix + suffix
            self.check_budget()
            r = self.c.qa_llm.complete(prompt, max_tokens=a.answer_max_tokens, cache_prefix_chars=len(prefix))
            answer = r["text"] or ""
            anc = (q.get("gold_anchor") or "").strip()
            pat = anchor_pattern(anc) if anc else None
            mem_tokens = ntok(memory) + (ntok(injected) if injected else 0)     # AS INJECTED, native (C.5)
            kg_txt, ch_txt = (rcm.split_packed_sections(packed.get("text", "")) if arm in NC_ARMS else ("", ""))
            row = {
                "run_id": self.c.run_id, "chain_id": self.sid, "sid": self.sid, "cluster_session": self.sid, "session_id": self.hars_sid(B) if arm in NC_ARMS else self.sid,
                "source": self.s.get("source"), "qid": qid_row, "base_qid": q["qid"], "arm": arm, "mem_budget": B, "b_tag": True, "rid": rid,
                "checkpoint": j, "is_end": is_end, "n_boundaries": len(self.plan["boundaries"]),
                "lag": lag, "in_tail": lag == 0, "span_idx": j, "slice": q.get("slice"),
                "qtype": q.get("qtype"), "source_qtype": q.get("source_qtype"), "answerable": q.get("answerable", True),
                "question": q["question"], "canonical_short_answer": q.get("answer"), "gold_anchor": anc,
                "evidence": q.get("evidence"), "rubric": q.get("rubric"),
                "answer": answer, "ok": bool(answer.strip()), "stop_reason": r["stop_reason"],
                "latency_s": r["latency_s"], "prompt_tokens": r["input_tokens"], "completion_tokens": r["output_tokens"],
                "cost_usd": r["cost_usd"],
                "memory_tokens": mem_tokens, "memory_tokens_injected": ntok(injected) if injected else 0,
                "arm_note_tokens": note_tokens, "arm_memory_total": mem_tokens, "tail_tokens": tail_tokens,
                "tail_from": b["at_idx"], "tail_to": b["trigger_idx"],
                "memory_text": memory, "injected": injected, "prompt_sha256": L.sha256_text(prompt),
                "anchor_hit": bool(pat.search(answer)) if pat else None,
                "anchor_in_recall": bool(pat.search(packed.get("text", ""))) if (pat and arm in NC_ARMS) else None,
                "anchor_in_kg_recall": bool(pat.search(kg_txt)) if (pat and arm in NC_ARMS) else None,
                "anchor_in_chunk_recall": bool(pat.search(ch_txt)) if (pat and arm in NC_ARMS) else None,
                "rubric_hit_frac": rubric_hit_frac(answer, q.get("rubric") or []),
                "recall_raw_chars": len(pay.get("context") or "") if arm in NC_ARMS else 0,
                "recall_packed_tokens": packed.get("tokens", 0) if arm in NC_ARMS else 0,
                "recall_packed_chars": packed.get("packed_chars", 0) if arm in NC_ARMS else 0,
                "recall_seconds": recall_s if arm in NC_ARMS else 0.0,
                "recall_error": (str(pay.get("error")) or None) if (arm in NC_ARMS and pay.get("error")) else None,
                "recall_failure_reasons": list(reasons) if arm in NC_ARMS else [],
                "recall_degraded": rcm.recall_degraded(pay) if arm in NC_ARMS else None,
                "recall_session_status": ((pay.get("session") or {}).get("status") if arm in NC_ARMS else None),
                "recall_error_kind": pay.get("error_kind") if arm in NC_ARMS else None,
                "allow_degraded": bool(a.allow_degraded),
                "recall_kg_entities_kept": packed.get("entities_kept") if arm in NC_ARMS else None,
                "recall_kg_relations_kept": packed.get("relations_kept") if arm in NC_ARMS else None,
                "persisted_compaction": (self.ingest_ok if arm in NC_ARMS else (False if arm == ARM_BASELINE else None)),
                "agent_model": self.c.sum_llm.model, "qa_model": self.c.qa_llm.model,
                "agent_provider": a.provider, "tokenizer": tok_label(), "tokenizer_native": tok_native(),
                "_reasons": reasons,
            }
            row.update(cap_info)
            if arm == ARM_GRAPH:
                row.update(graph_payload)
                row["session_graph_mode"] = "hybrid"
            rows.append(row)
        return rows

    def ask_checkpoint(self, b: dict):
        j = b["j"]
        is_end = j == len(self.plan["boundaries"])
        ql = self.plan["ask"].get(j) or []
        if not ql:
            return
        tail = self.tail_text(b)
        tail_tokens = count_tokens(tail)
        if self.need_nc():
            st = self.c.hars.status(self.hars_sid(max(self.a.budgets)))
            rcm = L.rc()
            fails, _warns = rcm.status_gate_problems(st) if rcm and hasattr(rcm, "status_gate_problems") else ([], [])
            for (arm, B) in self.todo:
                if arm in NC_ARMS:
                    self.reasons[(arm, B)] += [f"status gate: {f}" for f in fails]
            self.log_comp({"j": j, "arm": "hars", "status": {k: st.get(k) for k in (
                "compactions", "entities", "relations", "vectors_indexed")}, "gate_fails": fails})
        with cf.ThreadPoolExecutor(max_workers=max(1, self.a.ask_concurrency)) as ex:
            futs = [ex.submit(self.ask_one, b, q, lag, is_end, tail, tail_tokens) for q, lag in ql]
            for f in futs:
                for row in f.result():
                    key = (row["arm"], row["mem_budget"])
                    self.rows[key].append(row)
                    rs = row.pop("_reasons", [])
                    self.reasons[key] += rs
                    if row["rid"] not in self.partial:
                        self.partial[row["rid"]] = row
                        with open(self.partial_path, "a") as pf:
                            pf.write(json.dumps({**row, "_reasons": rs}, ensure_ascii=False) + "\n")
        print(f"  [{self.sid}] checkpoint j={j}: {len(ql)} questions asked "
              f"({'end' if is_end else 'mid'}); sum_cost={fmt_usd(self.c.sum_llm.cost)} "
              f"qa_cost={fmt_usd(self.c.qa_llm.cost)}", flush=True)

    # ------------------------------------------------------------- main loop
    def run(self):
        a, plan = self.a, self.plan
        os.makedirs(os.path.dirname(self.partial_path), exist_ok=True)
        if not self.todo:
            print(f"[{self.sid}] all (arm, B) done - skipped", flush=True)
            return
        print(f"[{self.sid}] {plan['n_messages']} msgs, {plan['total_tokens']:,} tok, "
              f"{len(plan['boundaries'])} boundaries, arms todo {sorted(self.todo)}", flush=True)
        base_prev: dict[int, str] = {B: "" for B in a.budgets}
        note_prev: dict[int, str] = {B: "" for B in a.budgets}
        store_B = self.note_budget_for_store()
        ingested_for: set = set()
        for b in plan["boundaries"]:
            j = b["j"]
            dropped = self.msgs[b["drop_start"]:b["at_idx"]]
            t0 = time.time()
            for B in a.budgets:                       # largest first
                if self.need(ARM_BASELINE, B):
                    e = self.summarize("baseline", B, j, base_prev[B], dropped)
                    base_prev[B] = self.base_text[B] = e["text"]
                    self.clip[("baseline", B)][0] += 1
                    self.clip[("baseline", B)][1] += int(e["clipped"] and not e["clipped_repaired"])
                if self.need_note(B):
                    e = self.summarize("note", B, j, note_prev[B], dropped)
                    note_prev[B] = self.note_text[B] = e["text"]
                    self.clip[("note", B)][0] += 1
                    self.clip[("note", B)][1] += int(e["clipped"] and not e["clipped_repaired"])
            if any(self.need_nc(B) for B in a.budgets) and self.ingest_ok:
                if a.hars_per_budget:
                    for B in a.budgets:
                        if self.need_nc(B):
                            self.ingest(b, dropped, self.note_text[B], self.hars_sid(B), f"b{B}")
                else:
                    self.ingest(b, dropped, self.note_text[store_B], self.hars_sid(store_B), "shared")
            if j in plan["checkpoints"]:
                self.ask_checkpoint(b)
            if j % 10 == 0 or j == len(plan["boundaries"]):
                print(f"  [{self.sid}] boundary {j}/{len(plan['boundaries'])} done in {time.time() - t0:.1f}s",
                      flush=True)
        self.finish()

    def finish(self):
        a = self.a
        for (arm, B) in sorted(self.todo):
            rows = self.rows.get((arm, B), [])
            why = list(dict.fromkeys(self.reasons.get((arm, B), [])))
            kind = "baseline" if arm == ARM_BASELINE else "note"
            n, bad = self.clip[(kind, B)] if arm != ARM_ORIG else (0, 0)
            if n and bad / n > CLIP_LIMIT:
                why.append(f"CLIP FAIL: {bad}/{n} summaries stopped at max_tokens and were not boundary-repaired")
            if arm in NC_ARMS and not self.ingest_ok:
                why.append(f"ingest failed: {self.ingest_err}")
            if arm == ARM_ORIG and not rows:
                continue
            if not rows:
                why.append("no rows (no question was eligible at any checkpoint)")
            valid = not why
            for r in rows:
                r["valid"] = valid
                r["chain_invalid_reasons"] = why
                r["clipped_summaries"] = {"n": n, "unrepaired": bad}
                self.c.append("records.jsonl", r)
                self.c.append("samples.jsonl", r)
            self.c.append("done.jsonl", {"sid": self.sid, "arm": arm, "B": B, "n_rows": len(rows),
                                         "valid": valid, "reasons": why,
                                         "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            self.c.done.add((self.sid, arm, B))
            print(f"[{self.sid}] {arm} B={B}: {len(rows)} rows valid={valid}"
                  f"{' ' + '; '.join(why[:3]) if why else ''}", flush=True)


# =============================================================================
# summary / manifest
# =============================================================================
def summarize_rows(rows: list[dict]) -> dict:
    out: dict = {}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["arm"], r["mem_budget"])].append(r)

    def agg(rs: list[dict]) -> dict:
        v = [r for r in rs if r.get("valid")]
        ah = [r["anchor_hit"] for r in v if r.get("anchor_hit") is not None]
        rh = [r["rubric_hit_frac"] for r in v if r.get("rubric_hit_frac") is not None]
        mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
        return {"n": len(rs), "n_valid": len(v), "anchor_hit_rate": mean([1.0 if x else 0.0 for x in ah]),
                "n_anchor": len(ah), "rubric_hit_mean": mean(rh), "n_rubric": len(rh),
                "memory_tokens_mean": mean([r["memory_tokens"] for r in v]),
                "latency_s_mean": mean([r["latency_s"] + r.get("recall_seconds", 0) for r in v]),
                "cost_usd_asks": round(sum(r["cost_usd"] for r in rs), 6)}

    for (arm, B), rs in sorted(groups.items()):
        d = agg(rs)
        d["by_qtype"] = {t: agg([r for r in rs if r.get("qtype") == t]) for t in sorted({r.get("qtype") for r in rs if r.get("qtype")})}
        d["by_lag"] = {str(l): agg([r for r in rs if r.get("lag") == l]) for l in sorted({r.get("lag") for r in rs if r.get("lag") is not None})}
        d["by_checkpoint"] = {str(c): agg([r for r in rs if r.get("checkpoint") == c]) for c in sorted({r["checkpoint"] for r in rs})}
        out[f"{arm}@B{B}"] = d
    return out


def git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", BASE, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def config_hash(args) -> str:
    keys = ("window", "keep_tail", "reserve", "budgets", "use_real_compactions", "final_compaction",
            "checkpoints", "piece_chars", "pieces_per_call", "agent_model", "sum_model", "qa_model",
            "provider", "hars_per_budget", "sum_max_input", "sum_max_tokens_slack", "seed", "max_questions", "qtypes", "lags")
    return hashlib.sha256(json.dumps({k: getattr(args, k, None) for k in keys}, sort_keys=True,
                                      default=str).encode()).hexdigest()[:16]


SCRIPT_FILES = ("run_longsession.py", "longsession.py", "ls_redact.py", "run_chainreplay.py", "run_pool.py")


def write_inputs_and_scripts(ctx: Ctx, args, sessions: list[dict]):
    inputs = {}
    for fp in sorted({s.get("_path") for s in sessions if s.get("_path")}):
        inputs[os.path.abspath(fp)] = {"sha256": L.sha256_file(fp), "bytes": os.path.getsize(fp)}
        if args.copy_inputs:
            os.makedirs(ctx.path("inputs", "files"), exist_ok=True)
            shutil.copy2(fp, ctx.path("inputs", "files", os.path.basename(fp)))
    atomic_write(ctx.path("inputs", "inputs.json"), json.dumps(
        {"files": inputs, "sessions": [{"sid": s["sid"], "n_messages": len(s["messages"]),
                                         "n_questions": len(s.get("questions") or []),
                                         "n_compactions": len(s.get("compactions") or [])} for s in sessions]}, indent=1))
    os.makedirs(ctx.path("scripts"), exist_ok=True)
    for fn in SCRIPT_FILES:
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy2(src, ctx.path("scripts", fn))
    with open(ctx.path("scripts", "cmdline.txt"), "a") as f:
        f.write(" ".join([sys.executable] + [shlex_quote(x) for x in sys.argv]) + "\n")


def shlex_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'" if re.search(r"[^\w@%+=:,./-]", s) else s


def preflight(ctx: Ctx, args) -> list[str]:
    """HARS healthz + one memory_recall (and key checks). Returns failures ([] = go)."""
    fails: list[str] = []
    for name, llm in (("summariser", ctx.sum_llm), ("answer", ctx.qa_llm)):
        p = llm.key_problem()
        if p:
            fails.append(f"{name} model: {p}")
    if not any(a in NC_ARMS for a in args.arms_list):
        return fails
    try:
        h = ctx.hars.health()
        print(f"[preflight] HARS healthz ok: {str(h)[:120]}", flush=True)
    except Exception as e:  # noqa: BLE001
        return fails + [f"HARS healthz failed: {type(e).__name__}: {e}"]
    rcm = L.rc_require("preflight_verdict")
    pf = ctx.hars.recall(args.preflight_session or f"ls-preflight-{ctx.run_id}", "preflight probe", False)
    f2, w2 = rcm.preflight_verdict(pf, args.preflight_session, args.allow_degraded)
    for w in w2:
        print(f"[preflight] warning: {w}", flush=True)
    return fails + list(f2)


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--name", default="run")
    ap.add_argument("--out", default=None, help=f"artifact dir (default {DEFAULT_ARTIFACT_ROOT}/<UTC>-longsession-<name>)")
    ap.add_argument("--resume", default=None, help="existing artifact dir to continue (skips done sid/arm/B)")
    ap.add_argument("--force-config", action="store_true", help="resume even if the config hash differs")
    ap.add_argument("--arms", default=None, help="comma list of baseline,nc-chunks,nc-graph,nc-note-only,orig (nc = nc-chunks); "
                                                  "default baseline,nc-chunks,nc-graph,nc-note-only (+orig with --use-real-compactions)")
    ap.add_argument("--window", "--window-tokens", dest="window", type=int, default=150000)
    ap.add_argument("--keep-tail", type=int, default=20000)
    ap.add_argument("--reserve", type=int, default=DEFAULT_RESERVE,
                    help="tokens reserved for the memory artifact in the window (constant, independent of "
                         "--mem-budgets, so boundaries are identical across sweeps)")
    ap.add_argument("--use-real-compactions", action="store_true")
    fg = ap.add_mutually_exclusive_group()
    fg.add_argument("--final-compaction", dest="final_compaction", action="store_true", default=None)
    fg.add_argument("--no-final-compaction", dest="final_compaction", action="store_false", default=None)
    ap.add_argument("--mem-budgets", default="1536")
    ap.add_argument("--hars-per-budget", action="store_true")
    ap.add_argument("--checkpoints", default="end", help="end | all | every:N | comma list of boundary numbers")
    ap.add_argument("--lags", default="", help="only ask questions whose lag is in this comma list")
    ap.add_argument("--qtypes", default="", help="only these qtypes (comma list)")
    ap.add_argument("--max-questions", type=int, default=0, help="per session per checkpoint (0 = all)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit", type=int, default=0, help="max sessions")
    P.add_profile_args(ap, roles=("agent", "answer", "summarizer", "extractor"))
    ap.add_argument("--provider", dest="provider_legacy", default=None,
                    choices=tuple(P.PROFILE_NAMES) + ("mock",),
                    help="alias of --profile (also: mock = offline LLM, no network). Default profile: "
                         "anthropic (Amendment C.1)")
    ap.add_argument("--agent-model", default=None,
                    help="alias of --model-agent: summariser + answer model unless "
                         "--sum-model/--qa-model (or --model-summarizer/--model-answer) are given")
    ap.add_argument("--sum-model", default=None, help="alias of --model-summarizer")
    ap.add_argument("--qa-model", default=None, help="alias of --model-answer (default = agent model)")
    ap.add_argument("--extractor-model", default=None,
                    help="alias of --model-extractor: HARS extractor, RECORDED + priced only "
                         "(the server is configured by `providers.py hars-env`)")
    ap.add_argument("--key-env", default=None, help="env var holding the API key (never printed)")
    ap.add_argument("--token-counter", default=None, choices=("auto", "native", "tiktoken", "estimate"),
                    help="Amendment C.5: parity/budget tokens in the answering model's NATIVE tokenizer "
                         "(count_tokens API; anthropic, gemini). auto falls back to tiktoken cl100k only for other "
                         "providers (native=False, flagged); a live anthropic/gemini run without native counting is "
                         "refused unless this is `tiktoken`. Dry-run/mock use `estimate`.")
    ap.add_argument("--tokcount-concurrency", type=int, default=4, help="max in-flight count_tokens requests")
    ap.add_argument("--reasoning-effort", default=None, help="openrouter reasoning.effort passthrough")
    ap.add_argument("--anthropic-thinking", choices=("disabled", "default"), default="disabled",
                    help="disabled sends thinking={type:disabled} on the anthropic endpoint: adaptive "
                         "thinking otherwise eats the summariser's max_tokens (stop=length, empty text)")
    ap.add_argument("--price-in", type=float, default=None,
                    help="USD/MTok override of the profile price (agent models; the profile tables "
                         "in profiles/*.json are the default and only table)")
    ap.add_argument("--price-out", type=float, default=None)
    ap.add_argument("--extractor-price-in", type=float, default=None)
    ap.add_argument("--extractor-price-out", type=float, default=None)
    ap.add_argument("--extract-in-per-chunk", type=float, default=EXTRACT_IN_PER_CHUNK)
    ap.add_argument("--extract-out-per-chunk", type=float, default=EXTRACT_OUT_PER_CHUNK)
    ap.add_argument("--extract-s-per-piece", type=float, default=EXTRACT_S_PER_PIECE)
    ap.add_argument("--mcp", default=L.rc_attr("DEFAULT_MCP", "http://127.0.0.1:8307/mcp"))
    ap.add_argument("--piece-chars", type=int, default=8000)
    ap.add_argument("--pieces-per-call", type=int, default=12)
    ap.add_argument("--sum-max-input", type=int, default=120000, help="max summariser input tokens (folded above)")
    ap.add_argument("--sum-max-tokens-slack", type=float, default=SUM_MAX_TOKENS_SLACK,
                    help="summariser max_tokens = cap * slack + 64 (the cap itself is enforced by the trimmer)")
    ap.add_argument("--answer-max-tokens", type=int, default=400)
    P.add_budget_arg(ap)
    ap.add_argument("--max-cost-usd", dest="budget_usd", type=float, default=None,
                    help="alias of --budget-usd (HARD cap: summariser + answers + flagged estimate of "
                         "the HARS extraction; the run stops at the cap, resume with a higher one)")
    ap.add_argument("--allow-degraded", action="store_true")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--preflight-session", default=None)
    ap.add_argument("--session-concurrency", type=int, default=1)
    ap.add_argument("--compact-concurrency", type=int, default=1)
    ap.add_argument("--recall-concurrency", type=int, default=2)
    ap.add_argument("--ask-concurrency", type=int, default=2)
    ap.add_argument("--copy-inputs", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true", help="dry-run: print the plan as JSON")
    ap.add_argument("--run-tag", default="", help="HARS lease tag (home_run passes the campaign id when the server "
                    "exposes memory_lease_acquire); sent as run_tag on memory_compact writes; empty = not sent")
    ap.add_argument("--fake-hars", action="store_true", help="offline HARS stand-in (tests / wiring checks)")
    a = ap.parse_args(argv)
    a.budgets = parse_budgets(a.mem_budgets)
    arms = [x.strip() for x in (a.arms or ("baseline,nc-chunks,nc-graph,nc-note-only" + (",orig" if a.use_real_compactions else ""))).split(",") if x.strip()]
    arms = [ARM_ALIASES.get(x, x) for x in arms]
    bad = [x for x in arms if x not in ALL_ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; known {ALL_ARMS}")
    a.arms_list = [x for x in ALL_ARMS if x in arms]
    if ARM_ORIG in a.arms_list and not a.use_real_compactions:
        raise SystemExit("arm 'orig' needs --use-real-compactions (real summaries)")
    if a.keep_tail + a.reserve >= a.window and not a.use_real_compactions:
        raise SystemExit(f"--keep-tail {a.keep_tail} + --reserve {a.reserve} must be < --window {a.window}")
    resolve_models_into(a)
    return a


def resolve_models_into(a) -> None:
    """Profile + per-role models (agent -> summariser/answer fallback), profile object, budget."""
    a.provider = a.profile or a.provider_legacy or "anthropic"
    a.profile_obj = None
    if a.provider != L.MOCK:
        try:
            a.profile_obj = P.load_profile(a.provider, base_url=a.base_url, pi_provider=a.pi_provider,
                                           cache=P.cache_override(a))
        except P.ProviderError as e:
            raise SystemExit(f"!!! {e}")
    dflt = (lambda r: a.profile_obj.default_model(r)) if a.profile_obj else (lambda r: "mock")
    ag = a.model_agent or a.agent_model
    a.agent_model = ag or dflt("agent")
    a.sum_model = a.model_summarizer or a.sum_model or ag or dflt("summarizer")
    a.qa_model = a.model_answer or a.qa_model or ag or dflt("answer")
    a.extractor_model = a.model_extractor or a.extractor_model or dflt("extractor") or "claude-haiku-5-5"
    missing = [n for n, m in (("agent", a.agent_model), ("summarizer", a.sum_model), ("answer", a.qa_model)) if not m]
    if missing:
        raise SystemExit(f"!!! no model for role(s) {missing} in profile {a.provider}: pass --model-<role>")


def model_prices(a) -> dict:
    override = (a.price_in, a.price_out) if (a.price_in is not None and a.price_out is not None) else None
    ext = ((a.extractor_price_in, a.extractor_price_out)
           if (a.extractor_price_in is not None and a.extractor_price_out is not None) else None)
    return {"sum": tuple(L.price_for(a.sum_model, override)), "qa": tuple(L.price_for(a.qa_model, override)),
            "ext": tuple(L.price_for(a.extractor_model, ext))}


def load_plans(a) -> tuple[list[dict], list[dict]]:
    sessions = L.load_sessions(a.sessions)
    ok, plans = [], []
    seen = set()
    for s in sessions:
        bad = L.validate_session(s)
        if bad:
            print(f"[skip] {s.get('sid')}: invalid session: {bad[:2]}", file=sys.stderr)
            continue
        if s["sid"] in seen:
            print(f"[skip] duplicate sid {s['sid']}", file=sys.stderr)
            continue
        seen.add(s["sid"])
        if a.use_real_compactions and not s.get("compactions"):
            print(f"[skip] {s['sid']}: --use-real-compactions but the session has none", file=sys.stderr)
            continue
        ok.append(s)
        if a.limit and len(ok) >= a.limit:
            break
    for s in ok:
        plans.append(build_plan(s, a))
    return ok, plans


def main(argv=None) -> int:
    a = parse_args(argv)
    os.environ.setdefault("NC_PACK_QUIET", "1")      # run_chainreplay.pack_recall logs one line per pack otherwise
    sessions, plans = load_plans(a)
    L.rc()                                              # import run_chainreplay once, before any thread
    if not plans:
        print("no usable sessions", file=sys.stderr)
        return 2
    prices = model_prices(a)
    a.prices_cache = prices
    if a.dry_run:
        make_counter(a, live=False)
        print_dry_run(plans, a, prices, as_json=a.json)
        return 0
    if not a.fake_hars:
        L.rc_require("mcp_call", "hars_recall", "hars_status", "preflight_verdict")
    if any(x in NC_ARMS for x in a.arms_list):
        L.rc_require("pack_injection", "recall_failure_reasons", "recall_degraded", "graph_failure_reasons",
                     "graph_summary", "split_packed_sections", "status_gate_problems")
    run_ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = a.resume or a.out or os.path.join(DEFAULT_ARTIFACT_ROOT, f"{run_ts}-longsession-{a.name}")
    resuming = bool(a.resume)
    if resuming and not os.path.isdir(out):
        raise SystemExit(f"--resume: no such dir {out}")
    if not resuming and os.path.exists(os.path.join(out, "MANIFEST.json")):
        raise SystemExit(f"{out} already holds a run; use --resume {out}")
    os.makedirs(out, exist_ok=True)
    cfg = config_hash(a)
    cfg_path = os.path.join(out, "config.json")
    if resuming and os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        if old.get("hash") != cfg and not a.force_config:
            raise SystemExit(f"config changed since the run started ({old.get('hash')} vs {cfg}); "
                             "use --force-config to override")
    run_id = (json.load(open(cfg_path))["run_id"] if resuming and os.path.exists(cfg_path)
              else f"ls{run_ts}"[:24])
    atomic_write(cfg_path, json.dumps({"hash": cfg, "run_id": run_id, "argv": sys.argv[1:]}))
    orig_streams = (sys.stdout, sys.stderr)
    tee_out = Tee(sys.stdout, os.path.join(out, "run.log"))
    sys.stdout, sys.stderr = tee_out, Tee(sys.stderr, os.path.join(out, "run.log"))
    try:
        return _run(a, sessions, plans, prices, out, resuming, cfg, run_id)
    finally:
        sys.stdout, sys.stderr = orig_streams


def _run(a, sessions, plans, prices, out, resuming, cfg, run_id) -> int:
    compact_sem, recall_sem = threading.Semaphore(a.compact_concurrency), threading.Semaphore(a.recall_concurrency)
    hars = FakeHars() if a.fake_hars else Hars(a.mcp, compact_sem, recall_sem)
    if not a.fake_hars:
        hars.run_tag = a.run_tag or ""
    budget = None
    if a.provider != L.MOCK and a.budget_usd:
        budget = P.Budget(a.budget_usd)
    override = (a.price_in, a.price_out) if (a.price_in is not None and a.price_out is not None) else None
    mk = lambda role, model: LLM(a.provider, model, role=role, profile=a.profile_obj, budget=budget,  # noqa: E731
                                 key_env=a.key_env, price=override, reasoning_effort=a.reasoning_effort,
                                 thinking=a.anthropic_thinking)
    sum_llm, qa_llm = mk("summarizer", a.sum_model), mk("answer", a.qa_model)
    make_counter(a, live=True)
    ctx = Ctx(a, out, hars, sum_llm, qa_llm, run_id, budget)
    ctx.load_done()
    print(f"run {run_id} -> {out} (tokenizer {tok_label()} native={tok_native()}, resume={resuming}, sessions={len(plans)})", flush=True)
    write_inputs_and_scripts(ctx, a, sessions)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    fails = []
    if a.provider != L.MOCK and not a.budget_usd:
        fails.append("--budget-usd is mandatory for a paid run (hard cap; Amendment C.1-5); "
                     "`--dry-run` prints the estimate to pick it from")
    if not a.no_preflight:
        fails += preflight(ctx, a)
    if fails:
        for f in fails:
            print(f"[preflight] FAIL: {f}", file=sys.stderr, flush=True)
        print("preflight failed - nothing spent. Fix the above (the budget check cannot be skipped).",
              file=sys.stderr)
        return 3
    failed: list[str] = []

    def run_one(plan):
        sr = SessionRun(ctx, plan)
        try:
            sr.run()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            failed.append(f"{plan['sid']}: {type(e).__name__}: {str(e)[:200]}")

    try:
        with cf.ThreadPoolExecutor(max_workers=max(1, a.session_concurrency)) as ex:
            list(ex.map(run_one, [p for p in plans if p["boundaries"]]))
    finally:
        rows = [json.loads(l) for l in open(ctx.path("records.jsonl"))] if os.path.exists(ctx.path("records.jsonl")) else []
        comps = ([json.loads(l) for l in open(ctx.path("compactions.jsonl")) if l.strip()]
                 if os.path.exists(ctx.path("compactions.jsonl")) else [])
        total_cost = {"summariser": round(sum(c.get("cost_usd") or 0.0 for c in comps if not c.get("cached")), 6),
                      "answer": round(sum(r.get("cost_usd") or 0.0 for r in rows), 6)}
        summary = {"run_id": run_id, "rows": len(rows), "by_arm_budget": summarize_rows(rows),
                   "failed_sessions": failed, "llm_cost_usd": {"summariser": round(sum_llm.cost, 6),
                                                              "answer": round(qa_llm.cost, 6)},
                   "prompt_cache": {"summariser": sum_llm.cache_stats(), "answer": qa_llm.cache_stats()},
                   "llm_cost_usd_all_invocations": total_cost, "tokenizer": tok_label(),
                   "tokenizer_native": tok_native(),
                   "token_counter": (tokcount.get_default().manifest() if tokcount.get_default() else None),
                   "hars_extraction_estimated_usd": round(ctx.est_hars_usd, 6),
                   "budget_usd": a.budget_usd}
        atomic_write(ctx.path("summary.json"), json.dumps(summary, indent=1))
        rcm = L.rc()
        manifest = {"run_id": run_id, "started": started, "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "machine": {"node": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
                                "cpus": os.cpu_count()},
                    "models": {"summariser": a.sum_model, "answer": a.qa_model, "extractor_assumed": a.extractor_model,
                               "agent": a.agent_model, "provider": a.provider},
                    "profile": ({"name": a.profile_obj.name, "base_url": a.profile_obj.base_url,
                                 "file": os.path.join(P.PROFILES_DIR, f"{a.profile_obj.name}.json"),
                                 "price_status": {m: (a.profile_obj.price(m) or {}).get("status", "unpriced")
                                                  for m in {a.sum_model, a.qa_model, a.extractor_model}}}
                                if a.profile_obj else None),
                    "budget_usd": a.budget_usd,
                    "prompt_cache": {"summariser": sum_llm.cache_stats(), "answer": qa_llm.cache_stats(),
                                     "note": "cost-only (Amendment C.3): identical model input; "
                                             "prefix = memory + verbatim tail"},
                    "spend_usd": {"llm_measured": round(sum_llm.cost + qa_llm.cost, 6),
                                  "hars_extraction_estimated": round(ctx.est_hars_usd, 6),
                                  "unpriced_calls": sum_llm.unpriced_calls + qa_llm.unpriced_calls,
                                  "budget_spent_total": round(budget.spent, 6) if budget else None},
                    "prices_usd_per_mtok": {k: list(v) for k, v in prices.items()},
                    "params": {k: v for k, v in vars(a).items() if k not in ("key_env", "profile_obj", "prices_cache")},
                    "cost_usd": summary["llm_cost_usd"], "git_head": git_rev(),
                    "run_chainreplay": {"importable": rcm is not None, "error": L.rc_error(),
                                        "sha256": L.sha256_file(os.path.join(BASE, "run_chainreplay.py"))
                                        if os.path.exists(os.path.join(BASE, "run_chainreplay.py")) else None},
                    "config_hash": cfg, "failed_sessions": failed, "resumed": resuming,
                    "hars": {"mcp": None if a.fake_hars else a.mcp, "fake": bool(a.fake_hars)}}
        mp = ctx.path("MANIFEST.json")
        if os.path.exists(mp):                       # resumed run: keep the earlier invocations
            try:
                old = json.load(open(mp))
                manifest["started"] = old.get("started", manifest["started"])
                manifest["previous_invocations"] = old.get("previous_invocations", []) + [
                    {"started": old.get("started"), "ended": old.get("ended"), "cost_usd": old.get("cost_usd"),
                     "spend_usd": old.get("spend_usd"), "failed_sessions": old.get("failed_sessions")}]
            except (OSError, ValueError):
                pass
        manifest["cost_usd_all_invocations"] = total_cost
        prev_spent = sum(((i.get("spend_usd") or {}).get("budget_spent_total") or 0.0)
                         for i in manifest.get("previous_invocations", []))
        manifest["budget_spent_all_invocations"] = round(
            prev_spent + ((manifest["spend_usd"] or {}).get("budget_spent_total") or 0.0), 6)
        atomic_write(mp, json.dumps(manifest, indent=1, default=str))
    print(f"done: {len(rows)} rows; cost summariser {fmt_usd(sum_llm.cost)} answers {fmt_usd(qa_llm.cost)}; "
          f"failed sessions {len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
