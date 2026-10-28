#!/usr/bin/env python3
"""Long-session eval: 1M+ token sessions, 5-100 compactions, baseline vs HARS memory arms.

Input: longsession/1 sessions (see longsession.py docstring; import_beam.py, import_claude_code.py,
gen_longsession_qa.py produce them). Nothing here runs by itself: the OWNER runs it with their
models / embedder / HARS. ``--dry-run`` needs no network at all.

HARS LAYOUT (Amendment C.8, --hars-layout): `project` (default for nc-enriched) writes each session's dropped context with
memory_upsert_document into its own project `nc-<run id>-<sid>` of the same server and reads it with memory_recall(project=...,
NO session_id) = the memory used like the global memory. `split-session` (memory_compact + memory_recall(session_id)) only exercises
the server's session SIDE CHANNEL - DIAGNOSTIC ONLY; it is the layout of the question-time arms nc-chunks / nc-graph.

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
  nc-enriched  (Amendment C.7, primary treatment) the same note + a block of facts recalled from HARS AT
             COMPACTION time: at each checkpoint boundary the summariser model writes K short retrieval
             queries from the note + the kept recent context ONLY (never from the questions), each is sent
             to memory_recall (top_k 8, session_graph off), chunks are deduped, packed to B - realized note
             tokens, and folded into the post-compaction context. NO recall at ask time. The block is
             computed once per (sid, boundary, B) and persisted under enrich/.
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
ARM_ENRICHED = "nc-enriched"                    # Amendment C.7: recall at COMPACTION time, no memory at ask time
ARM_ALIASES = {"nc": ARM_CHUNKS}
ALL_ARMS = (ARM_BASELINE, ARM_CHUNKS, ARM_GRAPH, ARM_ENRICHED, ARM_NOTE, ARM_ORIG)
DEFAULT_ARMS = (ARM_BASELINE, ARM_ENRICHED, ARM_NOTE)     # C.7; nc-chunks / nc-graph stay available via --arms
NC_ARMS = (ARM_CHUNKS, ARM_GRAPH)               # arms that recall from HARS AT ASK TIME
NOTE_ARMS = (ARM_CHUNKS, ARM_GRAPH, ARM_ENRICHED, ARM_NOTE)   # arms that carry the rolling note
STORE_ARMS = (ARM_CHUNKS, ARM_GRAPH, ARM_ENRICHED)            # arms that need the HARS session (ingest or reuse)
ENRICH_TOP_K = 8                                # frozen nc-chunks call shape: top_k = 8, no session_graph
ENRICH_QUERY_MAX_CHARS = 300
LAYOUT_SPLIT, LAYOUT_PROJECT = "split-session", "project"      # Amendment C.8: split-session = DIAGNOSTIC ONLY
RECALL_RAW_FILE = "recall_raw.jsonl"            # sidecar: raw recall text per rid (never inline in records)
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
# Amendment C.7: nc-enriched helpers (pure, offline-testable)
# =============================================================================
ENRICH_PROMPT_VERSION = "enrich-1"


def enrich_query_prompt(note: str, recent: str, k: int) -> str:
    """The ONLY input of query generation: the rolling note and the kept recent context. There is deliberately
    no question parameter: a question can never reach this prompt (tests assert it end to end)."""
    return (f"A long working session was just compacted. Below are the compaction note and the most recent "
            f"verbatim messages. From now on the assistant will NOT see the dropped conversation: it will only "
            f"have this note plus whatever facts are retrieved right now from a searchable archive "
            f"(hybrid vector + keyword + knowledge-graph search) of the dropped conversation.\n\n"
            f"Write {k} short retrieval queries that would fetch from that archive the details most likely to "
            f"be needed later: exact identifiers, file paths, commands, numbers, error messages, decisions and "
            f"their reasons, and anything the note mentions only in passing. Make the queries cover different "
            f"topics. Output exactly {k} lines and nothing else; each line is one JSON object:\n"
            f'{{"query": "<natural-language search query, 4-14 words>", "ll_keywords": ["<exact identifiers, '
            f'file paths, function names, error strings, numbers, copied VERBATIM>"], "hl_keywords": ["<2-3 '
            f'topics or concepts>"]}}\n'
            f"Write the queries and keywords in the language of the conversation.\n\n"
            f"<note>\n{(note or '').strip() or '(none)'}\n</note>\n\n"
            f"<recent_context>\n{(recent or '').strip() or '(none)'}\n</recent_context>")


def probe_keyword_prompt(question: str) -> str:
    """Probe only (the caller model of an ordinary ask): keywords for a memory search from the question."""
    return ("Produce keywords for a memory search that will answer the question below. Output ONE JSON object "
            'and nothing else: {"ll_keywords": ["<exact identifiers, file paths, function names, error '
            'strings, numbers from the question, VERBATIM>"], "hl_keywords": ["<2-3 topics or concepts>"]}. '
            "Use the language of the question.\n\n<question>\n" + question + "\n</question>")


def _kw_list(v) -> list[str]:
    if isinstance(v, str):
        v = [v]
    return [str(x).strip() for x in (v or []) if str(x).strip()][:12] if isinstance(v, list) else []


def parse_keywords(text: str) -> dict:
    """Probe keyword output -> {"ll_keywords", "hl_keywords", "source": "llm"|"none"}."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except ValueError:
        d = {}
    ll, hl = _kw_list(d.get("ll_keywords")), _kw_list(d.get("hl_keywords"))
    return {"ll_keywords": ll, "hl_keywords": hl, "source": "llm" if (ll or hl) else "none"}


_Q_BULLET = re.compile(r"^\s*(?:[-*\u2022]+|\d+[.)])\s*")


def parse_queries(text: str, k: int) -> list[dict]:
    """Model output -> at most k distinct queries [{query, ll_keywords, hl_keywords, keywords_source}].
    JSON lines (or one JSON array) are the contract; a plain-text line is kept as a query WITHOUT keywords
    (keywords_source "none": the server then degrades to naive mode, which the run records and flags)."""
    items: list[dict] = []
    t = (text or "").strip()
    if t.startswith("["):
        try:
            arr = json.loads(t)
            items = [x for x in arr if isinstance(x, (dict, str))]
        except ValueError:
            items = []
    if not items:
        for ln in t.splitlines():
            ln = _Q_BULLET.sub("", ln).strip()
            if ln.startswith("{"):
                try:
                    items.append(json.loads(ln))
                    continue
                except ValueError:
                    pass
            items.append(ln)
    out: list[dict] = []
    seen: set[str] = set()
    for it in items:
        if isinstance(it, dict):
            q = str(it.get("query") or it.get("question") or "").strip()
            ll, hl = _kw_list(it.get("ll_keywords")), _kw_list(it.get("hl_keywords"))
        else:
            q, ll, hl = str(it).strip().strip("`\"' ").strip(), [], []
        if len(q) < 3 or (q.lower().startswith(("here are", "queries")) and q.endswith(":")) or q in ("```", "```json"):
            continue
        q = q[:ENRICH_QUERY_MAX_CHARS]
        key = re.sub(r"\s+", " ", q.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"query": q, "ll_keywords": ll, "hl_keywords": hl, "keywords_source": "llm" if (ll or hl) else "none"})
        if len(out) >= k:
            break
    return out


def enrich_wrapper(sid: str, body: str, budget: int = 0) -> str:
    """The block handed to the model (wrapper included, counted in the budget). Neutral: the abstention
    instruction lives in ASK_PROMPT, identical for every arm."""
    return ("Facts recalled from long-term memory at compaction time "
            f"(HARS session {sid}, packed to <={budget} tokens):\n<recalled-facts>\n{body}\n</recalled-facts>")


def _chunk_key(content: str) -> str:
    return re.sub(r"\s+", " ", str(content or "")).strip().lower()


def merge_recall_payloads(payloads: list[dict]) -> tuple[dict, dict]:
    """K memory_recall payloads -> ONE payload (Document Chunks section only), chunks deduped across queries
    (exact normalised content), interleaved by rank (rank 1 of every query first) so no query starves the others.
    Returns (merged payload, stats). KG sections (absent with session_graph off) are ignored and counted."""
    rcm = L.rc_require("_parse_docs", "_HEADER_CHUNKS")
    per_q: list[list[dict]] = []
    ignored_kg = 0
    for p in payloads:
        ctx = str((p or {}).get("context") or "") if isinstance(p, dict) else ""
        per_q.append(rcm._parse_docs(ctx, rcm._HEADER_CHUNKS) if ctx else [])
        ignored_kg += int("Knowledge Graph Data" in ctx)
    seen: dict[str, int] = {}
    kept: list[dict] = []
    sources: list[int] = []
    dup = 0
    for r in range(max((len(d) for d in per_q), default=0)):
        for qi, docs in enumerate(per_q):
            if r >= len(docs):
                continue
            key = _chunk_key(docs[r].get("content"))
            if not key:
                continue
            if key in seen:
                dup += 1
                continue
            seen[key] = len(kept)
            kept.append(docs[r])
            sources.append(qi)
    lines = [json.dumps({"reference_id": str(i + 1), "content": d.get("content"), "section": d.get("section", "")},
                        ensure_ascii=False) for i, d in enumerate(kept)]
    ctx = ("Document Chunks (each has a reference_id):\n```json\n" + "\n".join(lines) + "\n```\n\n"
           "Reference Document List (for reference):\n```\n[1] merged-recall\n```") if kept else ""
    sessions = [p.get("session") for p in payloads if isinstance(p, dict) and isinstance(p.get("session"), dict)]
    sess = next((x for x in sessions if x.get("status") not in ("error", "unknown_session")), None) \
        or (sessions[0] if sessions else None)
    merged: dict = {"ok": bool(kept), "context": ctx, "_docs": kept}
    if sess is not None:
        merged["session"] = sess
    if not kept:
        merged["error"] = "no chunk from any enrichment query"
    stats = {"queries": len(payloads), "chunks_per_query": [len(d) for d in per_q],
             "chunks_total": sum(len(d) for d in per_q), "chunks_unique": len(kept),
             "duplicates_removed": dup, "kept_from_query": sources, "ignored_kg_payloads": ignored_kg}
    return merged, stats


def read_jsonl_file(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def load_reuse(a, plans: list[dict]) -> dict:
    """--reuse-run DIR (a previous run_longsession output dir): the run id whose HARS sessions are reused, the
    stored final note per (sid, B), the rows of the arms copied, and every problem found (the caller fails
    loudly on any).

    NOTE SOURCES: (1) <DIR>/memory/<sid>.json, the resume cache of the rolling summaries (sha-checked against
    the current config when used); (2) the final note carried verbatim by the `memory_text` of the old run's
    nc-note-only rows (identical across the sid's rows; hashed against the summary_sha256 of the note entry of
    compactions.jsonl when that entry exists). (2) is what survives in a bundle without memory/; it covers the
    LAST boundary only, i.e. end-only checkpoints."""
    d = os.path.abspath(a.reuse_run)
    info: dict = {"dir": d, "problems": [], "warnings": [], "notes": {}, "memory_cache": {}, "run_id": None,
                  "arms": [], "rows": {}, "done": {}, "hars_calls": {}}
    if not os.path.isdir(d):
        info["problems"].append(f"--reuse-run: no such dir {d}")
        return info
    try:
        info["run_id"] = str(json.load(open(os.path.join(d, "config.json")))["run_id"])
    except (OSError, ValueError, KeyError) as e:
        info["problems"].append(f"{d}/config.json unreadable ({type(e).__name__}): cannot recover the run id")
        return info
    try:
        old_p = json.load(open(os.path.join(d, "MANIFEST.json"))).get("params") or {}
    except (OSError, ValueError):
        old_p = {}
        info["warnings"].append("no MANIFEST.json in the reused run (params not cross-checked)")
    if old_p and bool(old_p.get("hars_per_budget")) != bool(a.hars_per_budget):
        info["problems"].append("hars_per_budget differs from the reused run (its HARS session ids would not match)")
    recs = read_jsonl_file(os.path.join(d, "records.jsonl"))
    comps = read_jsonl_file(os.path.join(d, "compactions.jsonl"))
    info["done"] = {(x["sid"], x["arm"], int(x["B"])): x for x in read_jsonl_file(os.path.join(d, "done.jsonl"))}
    for c in comps:
        if c.get("arm") == "hars" and c.get("ok") is True and not c.get("cached"):
            info["hars_calls"][c["sid"]] = info["hars_calls"].get(c["sid"], 0) + int(c.get("calls") or 0)
    plans = [p for p in plans if p["boundaries"]]
    try:                          # the reused run must have consumed the very same session files
        old_in = json.load(open(os.path.join(d, "inputs", "inputs.json")))
        old_sha = {v["sha256"] for v in (old_in.get("files") or {}).values()}
        for p in plans:
            fp = p["session"].get("_path")
            if fp and old_sha and L.sha256_file(fp) not in old_sha:
                info["problems"].append(f"session file of {p['sid']} ({os.path.basename(fp)}) has another sha256 than "
                                        "the reused run's inputs: its HARS sessions hold other data")
    except (OSError, ValueError):
        info["warnings"].append("no inputs/inputs.json in the reused run (input identity not verified)")
    for p in plans:
        sid = p["sid"]
        mp = os.path.join(d, "memory", f"{sid}.json")
        if os.path.isfile(mp):
            info["memory_cache"][sid] = mp
        for B in a.budgets:
            rows = [r for r in recs if str(r.get("sid") or r.get("chain_id")) == sid and r.get("arm") == ARM_NOTE
                    and int(r.get("mem_budget") or 0) == int(B)]
            texts = {r.get("memory_text") for r in rows if (r.get("memory_text") or "").strip()}
            if len(texts) > 1:
                info["problems"].append(f"{sid}@B{B}: the nc-note-only rows carry {len(texts)} different notes")
            elif len(texts) == 1:
                txt = next(iter(texts))
                shas = [c.get("summary_sha256") for c in comps if c.get("sid") == sid and c.get("arm") == "note"
                        and int(c.get("B") or 0) == int(B) and c.get("j") == len(p["boundaries"])
                        and c.get("summary_sha256")]
                if shas and L.sha256_text(txt) not in shas:
                    info["problems"].append(f"note of {sid}@B{B} (nc-note-only memory_text) does not hash to the "
                                            "summary_sha256 of its compactions.jsonl entry")
                elif not shas:
                    info["warnings"].append(f"note of {sid}@B{B}: no compactions.jsonl summary_sha256 to verify against")
                info["notes"][(sid, B)] = {"text": txt, "source": "records.nc-note-only.memory_text",
                                           "sha256": L.sha256_text(txt), "verified_vs_compactions": bool(shas)}
            elif sid not in info["memory_cache"]:
                info["problems"].append(f"{sid}@B{B}: the reused run stored no note (no memory/{sid}.json, "
                                        "no nc-note-only row)")
    arms = [x.strip() for x in (a.reuse_arms or "").split(",") if x.strip()]
    for arm in arms:
        for p in plans:
            for B in a.budgets:
                dn = info["done"].get((p["sid"], arm, B))
                if dn is None:
                    info["problems"].append(f"--reuse-arms {arm}: the reused run has no finished ({p['sid']}, {arm}, B{B}) cell")
                elif dn.get("valid") is not True:
                    info["problems"].append(f"--reuse-arms {arm}: cell ({p['sid']}, {arm}, B{B}) of the reused run is "
                                            f"INVALID ({'; '.join(map(str, dn.get('reasons') or []))[:120]})")
                else:
                    info["rows"][(p["sid"], arm, B)] = [
                        r for r in recs if str(r.get("sid") or r.get("chain_id")) == p["sid"] and r.get("arm") == arm
                        and int(r.get("mem_budget") or 0) == int(B)]
    info["arms"] = arms
    return info


def expected_qids(plan: dict) -> set[str]:
    out = set()
    nb = len(plan["boundaries"])
    for j, ql in plan["ask"].items():
        for q, _lag in ql:
            out.add(q["qid"] if j == nb else f"{q['qid']}@c{j}")
    return out


def parity_enriched(rows: list[dict], budgets: list[int], lo: float = 0.80, hi: float = 1.05) -> dict:
    """Realized ask-time memory (native tokens as injected) nc-enriched / baseline per B: mean over sessions of
    the per-session run-mean ratio = the quantity metrics_v2's win_mem_token_ratio gates (same window)."""
    out: dict = {}
    for B in budgets:
        rb = [r for r in rows if int(r.get("mem_budget") or 0) == int(B)]
        sids = sorted({str(r.get("chain_id")) for r in rb})

        def per_sid(arm):
            res = {}
            for sid in sids:
                xs = [float(r["memory_tokens"]) for r in rb if r["arm"] == arm and str(r["chain_id"]) == sid
                      and r.get("memory_tokens") is not None]
                res[sid] = sum(xs) / len(xs) if xs else None
            return res
        base, enr, note = per_sid(ARM_BASELINE), per_sid(ARM_ENRICHED), per_sid(ARM_NOTE)
        rs = [enr[s_] / base[s_] for s_ in sids if enr.get(s_) and base.get(s_)]
        ratio = round(sum(rs) / len(rs), 4) if rs else None
        mean = lambda d: (round(sum(v for v in d.values() if v) / max(1, len([v for v in d.values() if v])), 1)  # noqa: E731
                          if any(d.values()) else None)
        out[str(B)] = {"ratio_enriched_vs_baseline": ratio, "window": [lo, hi],
                       "in_window": bool(ratio is not None and lo <= ratio <= hi), "sessions": len(rs),
                       "baseline_memory_tokens_mean": mean(base), "enriched_memory_tokens_mean": mean(enr),
                       "note_only_memory_tokens_mean": mean(note)}
    return out


def chars_per_token(sample: str, lo: float = 1.5, hi: float = 6.0) -> float:
    """Measured chars/token of the run tokenizer on a text sample (the corpus), bounded; 3.6 when no sample."""
    sample = sample or ""
    if len(sample) < 400:
        return 3.6
    return round(min(hi, max(lo, len(sample) / max(1, ntok(sample)))), 3)


def doc_line(d: dict, i: int) -> str:
    return json.dumps({"reference_id": str(i + 1), "content": d.get("content"), "section": d.get("section", "")},
                      ensure_ascii=False)


def render_chunks_ctx(docs: list[dict]) -> str:
    return ("Document Chunks (each has a reference_id):\n```json\n"
            + "\n".join(doc_line(d, i) for i, d in enumerate(docs)) + "\n```") if docs else ""


def fill_union(docs: list[dict], sid: str, cap: int, wrapper=None) -> tuple[list[dict], str, int]:
    """Whole-chunk greedy fill of the cross-query union (rank-interleaved order): a chunk that no longer fits is
    SKIPPED (like the server's max_chars assembly), the first chunks in rank order always win. No sentence
    salience, no rewriting: the chunks are the server's own. Returns (selected docs, injected block, native tokens)."""
    wrapper = wrapper or enrich_wrapper
    base = ntok(wrapper(sid, "", cap))
    sel: list[dict] = []
    used, misses = base, 0
    for d in docs:
        cost = ntok(doc_line(d, len(sel))) + 1
        if used + cost <= cap:
            sel.append(d)
            used += cost
        else:
            misses += 1
            if misses >= 8:
                break
    inj = wrapper(sid, render_chunks_ctx(sel), cap)
    tot = ntok(inj)
    while sel and tot > cap:                 # the incremental sum is an estimate: verify, shrink until it fits
        sel.pop()
        inj = wrapper(sid, render_chunks_ctx(sel), cap)
        tot = ntok(inj)
    return sel, inj, tot


def budget_recall(recall_fn, sid: str, cap: int, cpt: float, wrapper=None, tries: int = 4):
    """memory_recall(max_chars=...) so that the injected block (wrapper included) fits `cap` NATIVE tokens.
    The server owns the selection (best-scored whole chunks first, never a blind cut); we only convert the
    token cap into its char budget with the run's measured chars/token and verify, shrinking max_chars when the
    block still overshoots. Returns (payload, injected, tokens, attempts)."""
    wrapper = wrapper or enrich_wrapper
    scaffold = ntok(wrapper(sid, "", cap))
    mc = max(200, int(cap * cpt))
    best, attempts = None, []
    for _ in range(tries):
        pay = recall_fn(mc)
        inj = wrapper(sid, str(pay.get("context") or ""), cap)
        tot = ntok(inj)
        attempts.append({"max_chars": mc, "tokens": tot})
        if tot <= cap and (best is None or tot > best[2]):
            best = (pay, inj, tot)
        if pay.get("error") or (tot <= cap and tot >= 0.85 * cap):
            break
        if tot > cap:
            mc = max(200, int(mc * max(1, cap - scaffold) / max(1, tot - scaffold) * 0.97))
        else:
            if len(attempts) >= 2 and attempts[-2]["tokens"] > cap:
                break                         # the next size up was a known overshoot
            mc = int(mc * max(1, cap - scaffold) / max(1, tot - scaffold) * 0.98)
    if best is None:
        raise RuntimeError(f"recall cannot be budgeted into {cap} native tokens via max_chars (tried {attempts})")
    return best[0], best[1], best[2], attempts


def graph_branch_of(pay: dict) -> dict:
    gb = (pay or {}).get("graph_branch") if isinstance(pay, dict) else None
    gb = gb if isinstance(gb, dict) else {}
    return {"graph_branch_enabled": gb.get("enabled"), "graph_branch_reason": gb.get("reason"),
            "lightrag_mode": (pay or {}).get("lightrag_mode") if isinstance(pay, dict) else None,
            "mode_fallback": (pay or {}).get("mode_fallback") if isinstance(pay, dict) else None}


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
    reuse = getattr(args, "reuse_info", None)
    reused_arms = set(reuse["arms"]) if reuse else set()
    out["hars"]["layout"] = getattr(args, "hars_layout", LAYOUT_SPLIT)
    if reuse and (getattr(args, "hars_layout", LAYOUT_SPLIT) == LAYOUT_SPLIT or getattr(args, "hars_ingested", False)):
        # split reuse / projects already ingested: the data exists - no ingest, no extraction, no write calls
        out["hars"] = {"pieces": 0, "memory_compact_calls": 0, "extract_chunks": 0, "extract_usd": 0.0,
                       "est_ingest_hours": 0.0, "reused_run": reuse["run_id"], "layout": out["hars"]["layout"]}
    elif not reuse and getattr(args, "hars_ingested", False):
        out["hars"] = {"pieces": 0, "memory_compact_calls": 0, "extract_chunks": 0, "extract_usd": 0.0, "est_ingest_hours": 0.0,
                       "layout": out["hars"]["layout"], "already_ingested": True}
    for B in args.budgets:
        ncap = note_cap_formula(B)
        res: dict[str, float] = {}
        for kind, cap, arm in (("baseline", B, ARM_BASELINE), ("note", ncap, "note")):
            if reuse and (kind == "note" or ARM_BASELINE in reused_arms):   # notes / baseline summaries come from the old run
                res[arm] = 0.0
                res[arm + "_tokens_in"] = res[arm + "_tokens_out"] = 0.0
                continue
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
                for a in (ARM_BASELINE, ARM_CHUNKS, ARM_GRAPH, ARM_ENRICHED, ARM_NOTE):
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
        # nc-enriched: ONE query-generation call per checkpoint boundary (note + kept recent context in, K JSON queries out)
        enrich_usd = 0.0
        for j, ql in plan["ask"].items():
            if ql:
                tail_t = min(bs[j - 1]["tail_tokens"], args.enrich_tail_tokens)
                tin = count_tokens(enrich_query_prompt("", "", args.enrich_queries)) + ncap * FILL + tail_t
                enrich_usd += usd_for(args, "sum", tin, args.enrich_queries * 70)
        arm_usd = {}
        for a in ALL_ARMS:
            n, tin, ask_usd, _nc = asks[a]
            owner = next((x for x in NOTE_ARMS if x in args.arms_list and x not in reused_arms), None)   # carries the shared note cost
            if a in reused_arms:
                arm_usd[a] = 0.0                         # copied from the reused run: nothing is paid
            elif a == ARM_BASELINE:
                arm_usd[a] = res["baseline"] + ask_usd
            elif a == ARM_ENRICHED:
                arm_usd[a] = ask_usd + enrich_usd + (res["note"] if owner == a else 0.0)
            elif a == ARM_GRAPH:                    # note + HARS are shared with nc-chunks
                arm_usd[a] = ask_usd + kw + (res["note"] if owner == a else 0.0)
            elif a in NOTE_ARMS:
                arm_usd[a] = ask_usd + (res["note"] if owner == a else 0.0)
            else:
                arm_usd[a] = ask_usd
        out["per_budget"][B] = {"usd": arm_usd, "asks": {a: asks[a][0] for a in ALL_ARMS},
                                "ask_tokens_in": {a: asks[a][1] for a in ALL_ARMS},
                                "summarizer": {k: v for k, v in res.items() if k.endswith(("_in", "_out"))},
                                "note_cap": ncap, "graph_keyword_usd": kw, "enrich_usd": enrich_usd,
                                "cache_saving_usd": sum(asks[a][3] - asks[a][2] for a in ALL_ARMS if a in args.arms_list)}
    return out


def rc_kw_in() -> float:
    return float(L.rc_attr("KEYWORD_IN", 450.0))


def rc_kw_out() -> float:
    return float(L.rc_attr("KEYWORD_OUT", 80.0))


def active_arms(args, plan: dict | None = None) -> list[str]:
    return list(args.arms_list)


def total_usd(est: dict, arms: list[str], hars_sessions_factor: int = 1) -> float:
    t = est["hars"]["extract_usd"] * hars_sessions_factor if any(a in STORE_ARMS for a in arms) else 0.0
    for B, d in est["per_budget"].items():
        t += sum(d["usd"][a] for a in arms)
    return t


def print_dry_run(plans: list[dict], args, prices: dict, *, as_json: bool = False) -> dict:
    args.prices_cache = prices
    res = {"config": {"window": args.window, "keep_tail": args.keep_tail, "reserve": args.reserve,
                      "budgets": args.budgets, "checkpoints": args.checkpoints, "arms": args.arms_list,
                      "real": bool(args.use_real_compactions), "tokenizer": tok_label(),
                      "reuse_run": (args.reuse_info or {}).get("dir") if getattr(args, "reuse_info", None) else None,
                      "reuse_arms": (args.reuse_info or {}).get("arms") if getattr(args, "reuse_info", None) else None,
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
        if getattr(args, "ingest_only", False):       # only the server-side extraction is paid (notes are reused)
            tot = est["hars"]["extract_usd"] * hf
        grand += tot
        dr = [b["dropped_tokens"] for b in p["boundaries"]]
        sess = {"sid": p["sid"], "messages": p["n_messages"], "tokens": p["total_tokens"],
                "boundaries": len(p["boundaries"]),
                "dropped_tokens": {"min": min(dr, default=0), "median": sorted(dr)[len(dr) // 2] if dr else 0,
                                   "max": max(dr, default=0)},
                "checkpoints": p["checkpoints"], "questions_per_checkpoint": {str(j): len(q) for j, q in p["ask"].items()},
                "real_summaries": p["n_real_summaries"], "hars": est["hars"],
                "per_budget": {str(B): {"usd": {a: round(v, 6) for a, v in d["usd"].items()},
                                        "asks": {a: n for a, n in d["asks"].items() if a in arms}, "note_cap": d["note_cap"]}
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
        if h.get("already_ingested"):
            lines.append("  HARS ingest: NONE (--hars-ingested: the projects were written by oh-ingest)")
        elif h.get("reused_run"):
            lines.append(f"  HARS ingest: NONE (--reuse-run: HARS sessions of run {h['reused_run']} are reused; "
                         "no memory_compact, no extraction, no ingest wall-clock)")
        else:
            lines.append(f"  HARS ingest ({'memory_upsert_document into project nc-*' if h.get('layout') == LAYOUT_PROJECT else 'memory_compact, split-session = DIAGNOSTIC'}): "
                         f"{h['pieces']:,} pieces in {h['memory_compact_calls']:,} write calls, "
                         f"{h['extract_chunks']:,} extraction chunks ~{fmt_usd(h['extract_usd'])}"
                         f"{' x' + str(len(args.budgets)) + ' budgets' if args.hars_per_budget else ''}, "
                         f"~{h['est_ingest_hours']:.1f} h (guess {args.extract_s_per_piece}s/piece)")
        for B, d in est["per_budget"].items():
            lines.append(f"  B={B} (note cap {d['note_cap']}): "
                         + " | ".join(f"{a} {fmt_usd(d['usd'][a])} ({d['asks'][a]} asks)"
                                      for a in ALL_ARMS if a in arms)
                         + (f" | graph-keyword {fmt_usd(d['graph_keyword_usd'])}" if ARM_GRAPH in arms else "")
                         + (f" | enrich query-gen {fmt_usd(d['enrich_usd'])} ({args.enrich_queries} queries/boundary)"
                            if ARM_ENRICHED in arms else ""))
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

    def recall(self, sid: str, question: str, graph: bool, project: str | None = None) -> dict:
        with self.rs:
            return self.rcm.hars_recall(self.url, sid, question, timeout=self.rt, graph=graph, project=project)

    def recall_correct(self, sid: str, query: str, ll: list, hl: list, max_chars: int | None,
                       project: str | None = None) -> dict:
        """The C.7 shape: query + ll/hl keywords, mode hybrid, server default top_k, server max_chars. With `project`
        (Amendment C.8) the call goes to that project's main index and carries NO session_id."""
        with self.rs:
            return self.rcm.hars_recall_correct(self.url, sid, query, ll, hl, max_chars, timeout=self.rt, project=project)

    def upsert(self, project: str, path: str, content: str) -> dict:
        """Amendment C.8 write path: memory_upsert_document(project, virtual file_path, content) - the tool the global
        memory is written with (full LightRAG extraction into the project's MAIN index and Qdrant collection)."""
        check_project_writable(project)
        with self.cs:
            return self.rcm.mcp_call(self.url, "memory_upsert_document",
                                     {"file_path": path, "content": content, "project": project}, timeout=self.ct)

    def project_state(self, project: str) -> dict:
        """memory_status(project): registered?, graph size, vectors (Qdrant points / NanoVectorDB files)."""
        try:
            st = self.rcm.mcp_call(self.url, "memory_status", {"project": project}, timeout=120)
        except Exception as e:  # noqa: BLE001
            return {"registered": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        return project_state_of(st, project)

    def status(self, sid: str) -> dict:
        return self.rcm.hars_status(self.url, sid)


def check_project_writable(project) -> None:
    """Amendment C.8: the owner's global memory (`default`) is never written by an eval."""
    if not str(project or "").strip() or str(project).strip() == "default":
        raise ValueError(f"refusing to write into project {project!r}: the eval writes only its own nc-* projects")


def project_state_of(st: dict, project: str) -> dict:
    """Normalise a memory_status payload: registered (ok and the server answered for THIS project), node/edge counts,
    vectors (Qdrant: points in the project's workspace; NanoVectorDB: number of vdb_* files)."""
    st = st if isinstance(st, dict) else {}
    ok = st.get("ok") is not False and str(st.get("project") or "") == project
    vec = ((st.get("storage") or {}).get("vector") or {}) if ok else {}
    vectors = vec.get("points_count") if vec.get("points_count") is not None else vec.get("file_count")
    return {"registered": ok, "error": None if ok else str(st.get("error") or "project not answered")[:160],
            "nodes": st.get("node_count"), "edges": st.get("edge_count"), "vectors": vectors,
            "vector_backend": vec.get("backend"), "index_exists": st.get("index_exists")}


def project_doc_path(pid: str, j: int, call: int) -> str:
    """Virtual (never opened) absolute file path of one ingest document: stable per (project, boundary, call), so a
    re-run is a fingerprint no-op and a changed document is purged and re-inserted by the server."""
    return f"/nc-eval/{pid}/j{j:03d}-c{call:02d}.md"


def project_doc_content(hsid: str, j: int, call: int, calls: int, summary: str, pieces: list[str]) -> str:
    """The compaction document of the split layout, as plain document text (same sections; pieces are the SAME
    piece splitting, newlines collapsed like the server's compaction does)."""
    facts = [re.sub(r"\s*\n\s*", " ", x).strip() for x in pieces]
    lines = [f"Session: {hsid}", f"Boundary: {j}", f"Call: {call}/{calls}", "Tags: session, compaction", "",
             "## Summary", summary.strip(), "", "## Dropped context facts", *(f"- {x}" for x in facts if x)]
    return "\n".join(lines).rstrip() + "\n"


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
        self.correct_calls: list[dict] = []        # tests: the exact C.7 call arguments
        # Amendment C.8 project layout: neighbour projects of the SAME server, each with its own docs / entities
        self.projects: dict[str, dict] = {}
        self.auto_register = True                  # False: an unknown project answers 404 like the real server
        self.default_kb: list[str] = []            # chunks of the owner's global KB (the `default` project)
        self.upsert_calls: list[dict] = []
        self.recall_args: list[dict] = []
        self.honour_cap = True                     # False: behaves like a server older than the C.10 hard cap (tests)
        self.report_cap = True                     # False: no `context_cap` field in the recall payload (old server)

    def register_project(self, pid: str) -> None:
        self.projects.setdefault(pid, {"docs": {}, "ents": set()})

    def _proj(self, pid: str):
        if pid not in self.projects and self.auto_register:
            self.register_project(pid)
        return self.projects.get(pid)

    def upsert(self, project, path, content):
        check_project_writable(project)
        with self.lock:
            self.upsert_calls.append({"project": project, "path": path, "chars": len(content)})
            st = self._proj(project)
            if st is None:
                return {"ok": False, "error": f"Project '{project}' not found", "code": 404}
            action = "unchanged" if st["docs"].get(path) == content else ("updated" if path in st["docs"] else "inserted")
            for k in [k for k in st["docs"] if k.startswith(path + "#")]:
                del st["docs"][k]
            for i, ch in enumerate(content.split("\n- ")):      # the server chunks a document; one chunk per fact here
                st["docs"][f"{path}#{i}"] = ch
            st["docs"][path] = content
            st["ents"].update(re.findall(r"[A-Za-z_][\w./-]*\d[\w./-]*", content))
            out = {"ok": True, "action": action, "file_path": path}
            if action != "unchanged":              # like the server: an unchanged document runs no extraction
                n = max(1, len(content) // 2000)
                out["extract_usage"] = {"calls": n, "prompt_tokens": n * 1000, "completion_tokens": n * 100,
                                        "total_tokens": n * 1100, "calls_estimated": 0, "estimated": False,
                                        "scope": "this_call"}
            return out

    def project_state(self, project):
        st = self._proj(project)
        if st is None:
            return {"registered": False, "error": "project not found"}
        return {"registered": True, "error": None, "nodes": len(st["ents"]), "edges": len(st["ents"]) // 2,
                "vectors": len(st["docs"]), "vector_backend": "fake", "index_exists": bool(st["docs"])}

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

    def recall_correct(self, sid, query, ll, hl, max_chars, project=None):
        """Offline twin of memory_recall in the C.7 shape: ranks ALL docs (server default top_k 20), renders
        whole chunks under max_chars (a chunk that no longer fits is skipped), reports graph_branch like the
        server (enabled only with keywords). With `project` (C.8) it searches ONLY that project's main index and the
        session_id is not sent; without it the session's own chunks are scored as the server's side channel does and
        the owner's global KB (`default_kb`) is searched too."""
        with self.lock:
            self.recall_calls += 1
            self.correct_calls.append({"sid": None if project else sid, "project": project, "query": query,
                                       "ll_keywords": list(ll), "hl_keywords": list(hl), "max_chars": max_chars})
            if project:
                pst = self._proj(project)
                st = ({"docs": [v for k, v in pst["docs"].items() if "#" in k], "ents": pst["ents"], "seq": 1} if pst else None)
            else:
                st = self.s.get(sid)
                if st is not None and self.default_kb:
                    st = {**st, "docs": list(st["docs"]) + list(self.default_kb)}
        if self.fail_recall:
            return {"context": "", "error": "fake recall failure"}
        kw = bool(ll or hl)
        gb = {"enabled": kw, "reason": "graph mode 'hybrid' requested with keywords" if kw else "no graph signal"}
        if not st and project:
            return {"ok": False, "context": "", "error": f"Project '{project}' not found", "code": 404}
        if not st:
            return {"ok": True, "context": "", "session": {"status": "unknown_session"}, "graph_branch": gb,
                    "lightrag_mode": "hybrid" if kw else "skipped"}
        qw = set(re.findall(r"\w{3,}", " ".join([query, *ll, *hl]).lower()))
        ranked = sorted(st["docs"], key=lambda d: -len(qw & set(re.findall(r"\w{3,}", d.lower()))))[:20]
        kept, used = [], 0
        # the real server (C.10) bounds the WHOLE rendered context: reserve the fixed section text
        overhead = len("\n".join(["Document Chunks (each has a reference_id):", "```json", "```", "",
                                   "Reference Document List (for reference):", "```", f"[1] {sid}.md", "```"])) + 4
        for c in ranked:
            line = json.dumps({"reference_id": str(len(kept) + 1), "content": c, "section": ""})
            if max_chars and (used + len(line) + 1 > max_chars - overhead if self.honour_cap else used + len(line) > max_chars):
                continue
            kept.append(line)
            used += len(line) + (1 if self.honour_cap else 0)
        raw = ["Document Chunks (each has a reference_id):", "```json", *kept, "```", "",
               "Reference Document List (for reference):", "```", f"[1] {sid}.md", "```"]
        ctx_text = "\n".join(raw)
        out = {"ok": True, "context": ctx_text, "graph_branch": gb,
               "lightrag_mode": "hybrid" if kw else "skipped", "mode_fallback": None}
        if max_chars and self.report_cap:
            out["context_cap"] = {"max_chars": max_chars, "chars_before": len(ctx_text), "chars_after": len(ctx_text),
                                  "applied": False, "cut": {}}
        if not project:                 # the real server adds the session block only for a session_id call
            out["session"] = {"status": "ok", "vectors_indexed": len(st["docs"]), "entities": len(st["ents"])}
        return out

    def recall(self, sid, question, graph, project=None):
        with self.lock:
            self.recall_calls += 1
            if project:
                pst = self._proj(project)
                st = ({"docs": [v for k, v in pst["docs"].items() if "#" in k], "ents": pst["ents"], "seq": 1} if pst else None)
            else:
                st = self.s.get(sid)
        if self.fail_recall:
            return {"context": "", "error": "fake recall failure"}
        if not st and project:
            return {"ok": False, "context": "", "error": f"Project '{project}' not found", "code": 404}
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


def new_extract_totals() -> dict:
    return {"documents": 0, "documents_measured": 0, "documents_unmeasured": 0, "calls": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "calls_estimated": 0}


def add_extract_usage(tot: dict, eu) -> None:
    """Fold one memory_upsert_document `extract_usage` (None = the server did not report it) into `tot`."""
    tot["documents"] += 1
    if not isinstance(eu, dict):
        tot["documents_unmeasured"] += 1
        return
    tot["documents_measured"] += 1
    for k in ("calls", "prompt_tokens", "completion_tokens", "calls_estimated"):
        tot[k] += int(eu.get(k) or 0)


def extract_totals_view(tot: dict, ext_price: tuple | None = None) -> dict:
    """The totals as written to ingest.jsonl / summary.json: tokens, an `estimated` flag (any call without provider
    usage, or any document the server did not report on) and, with the extractor's price (USD per Mtok in, out), USD."""
    out = dict(tot)
    out["total_tokens"] = tot["prompt_tokens"] + tot["completion_tokens"]
    out["estimated"] = bool(tot["calls_estimated"] or tot["documents_unmeasured"])
    out["complete"] = tot["documents_unmeasured"] == 0 and tot["documents"] > 0
    if ext_price:
        out["usd"] = round((tot["prompt_tokens"] * ext_price[0] + tot["completion_tokens"] * ext_price[1]) / 1e6, 6)
    return out


class Ctx:
    """Shared run state: models, HARS handle, artifact dir, locks, done set."""

    def __init__(self, args, out: str, hars, sum_llm, qa_llm, run_id: str, budget=None):
        self.args, self.out, self.hars, self.sum_llm, self.qa_llm, self.run_id = args, out, hars, sum_llm, qa_llm, run_id
        self.budget = budget                  # providers.Budget (hard cap) or None (mock/dry)
        self.est_hars_usd = 0.0               # flagged ESTIMATE of the HARS-side extraction
        self.extract = new_extract_totals()   # MEASURED extractor usage reported by the server (C.10), project layout
        self.lock = threading.Lock()
        self.done: set[tuple[str, str, int]] = set()
        self.totals = Counter()
        self.reuse: dict | None = None        # --reuse-run info (load_reuse), None = a fresh run
        self.hars_run_id = run_id             # run id in the HARS session ids (the reused run's id under --reuse-run)

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


class ReuseError(RuntimeError):
    pass


def make_hars_sid(run_id: str, sid: str, B: int, per_budget: bool) -> str:
    return f"ls-{run_id}-{sid}{f'-b{B}' if per_budget else ''}"[:128]


def reuse_problems(plans: list[dict], reuse: dict, a_=None) -> list[str]:
    """Static checks of a --reuse-run (no network): load_reuse problems + end-only checkpoints + the copied arms'
    question sets. Any problem is fatal BEFORE anything is spent."""
    why = list(reuse["problems"])
    for p in plans:
        if not p["boundaries"]:
            continue
        if (getattr(a_, "hars_layout", None) == LAYOUT_PROJECT and not getattr(a_, "hars_ingested", False)
                and p["sid"] not in reuse["memory_cache"]):
            why.append(f"{p['sid']}: the project layout re-ingests EVERY boundary with its original rolling note, which needs "
                       f"{reuse['dir']}/memory/{p['sid']}.json (absent)")
        n = len(p["boundaries"])
        if p["checkpoints"] != [n]:
            why.append(f"{p['sid']}: --reuse-run needs end-only checkpoints (got {p['checkpoints']}): the reused HARS session "
                       "holds ALL boundaries, so a recall at an earlier checkpoint would see the future")
        want = expected_qids(p)
        for (sid, arm, B), rows in reuse["rows"].items():
            if sid != p["sid"]:
                continue
            got = {r["qid"] for r in rows}
            if got != want:
                why.append(f"{sid}: reused arm {arm}@B{B} answered {len(got)} question(s), this run asks {len(want)} "
                           f"(differs on {sorted(got ^ want)[:3]}): --max-questions/--qtypes/--lags/--seed must match the old run")
    return why


def reuse_hars_gate(hars, plans: list[dict], a, reuse: dict) -> list[str]:
    """HARS status per reused session: compactions > 0, vectors_indexed present and > 0 (+ the M4 gate)."""
    rcm = L.rc()
    why: list[str] = []
    for p in plans:
        if not p["boundaries"]:
            continue
        for B in (a.budgets if a.hars_per_budget else [max(a.budgets)]):
            hs = make_hars_sid(reuse["run_id"], p["sid"], B, a.hars_per_budget)
            st = hars.status(hs)
            f1, _w = rcm.status_gate_problems(st) if rcm and hasattr(rcm, "status_gate_problems") else ([], [])
            why += [f"{hs}: {x}" for x in f1]
            if int(st.get("compactions") or 0) <= 0:
                why.append(f"{hs}: memory_session_status.compactions == 0 (session absent on this server? wrong HARS data dir?)")
            if "vectors_indexed" not in st or not int(st.get("vectors_indexed") or 0):
                why.append(f"{hs}: memory_session_status.vectors_indexed missing or 0")
            exp = reuse["hars_calls"].get(p["sid"])
            if exp and int(st.get("compactions") or 0) < exp:
                print(f"[reuse] warning: {hs} reports {st.get('compactions')} compaction(s), the old run ingested in {exp} "
                      "call(s) (counting may differ; recall gate decides)", flush=True)
    return list(dict.fromkeys(why))


def copy_reused_rows(ctx, plans: list[dict], reuse: dict) -> int:
    """Copy the finished, valid rows of the --reuse-arms cells (fields kept, + reused_from)."""
    n = 0
    for (sid, arm, B), rows in sorted(reuse["rows"].items()):
        if (sid, arm, B) in ctx.done:
            continue
        for r in rows:
            row = {**r, "reused_from": reuse["dir"], "reused_run_id": reuse["run_id"]}
            ctx.append("records.jsonl", row)
            ctx.append("samples.jsonl", row)
            n += 1
        ctx.append("done.jsonl", {"sid": sid, "arm": arm, "B": B, "n_rows": len(rows), "valid": True, "reasons": [],
                                  "reused_from": reuse["dir"], "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        ctx.done.add((sid, arm, B))
    return n


def post_project_gate(hars, pid: str | None, expected_calls: int = 0) -> list[str]:
    """After ingest: the project exists on the server and its MAIN index holds vectors and entities."""
    if not pid:
        return ["no project id"]
    try:
        check_project_writable(pid)
    except ValueError as e:
        return [str(e)]
    st = hars.project_state(pid)
    why: list[str] = []
    if not st.get("registered"):
        return [f"{pid}: not registered on the server ({st.get('error')}): provision it and restart the server once"]
    if not int(st.get("nodes") or 0):
        why.append(f"{pid}: memory_status.node_count == {st.get('nodes')} (no entities extracted: extractor dead or misrouted?)")
    if not int(st.get("vectors") or 0):
        why.append(f"{pid}: no vectors in the project's index (memory_status vector storage: {st.get('vector_backend')})")
    return why


def pre_project_gate(hars, pid: str, fresh: bool) -> list[str]:
    """Before any spend: the project is registered on the server, is not `default`, and (fresh ingest) is EMPTY."""
    try:
        check_project_writable(pid)
    except ValueError as e:
        return [str(e)]
    st = hars.project_state(pid)
    if not st.get("registered"):
        return [f"{pid}: not registered on the server ({st.get('error')}). Projects are read at server start only: run "
                "`home_run.py oh-provision --resume <campaign>`, restart the server once, retry"]
    if fresh and (int(st.get("nodes") or 0) or int(st.get("vectors") or 0)):
        return [f"{pid}: the project is not empty (nodes={st.get('nodes')}, vectors={st.get('vectors')}): a fresh ingest needs "
                "an empty project (wipe <projects dir>/<pid>/ and restart the server, or resume the run that wrote it)"]
    return []


def layout_gate(ctx, a, plans: list[dict]) -> list[str]:
    """Project layout, run before anything is spent (not skippable by --no-preflight)."""
    why: list[str] = []
    for p in plans:
        if not p["boundaries"]:
            continue
        for B in (a.budgets if a.hars_per_budget else [max(a.budgets)]):
            pid = L.make_project_id(ctx.hars_run_id, p["sid"], B, a.hars_per_budget)
            mem = JsonCache(ctx.path("memory", f"{p['sid']}.json"))
            resumed = any(str(k).startswith("ingest|") for k in mem.d)
            if a.hars_ingested:
                why += post_project_gate(ctx.hars, pid)
            else:
                why += pre_project_gate(ctx.hars, pid, fresh=not resumed)
    return list(dict.fromkeys(why))


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
        self.extract = new_extract_totals()        # this session's measured extractor usage (all boundaries, from the cache)
        self.hars_keys: dict[object, str] = {}
        self.enriched: dict[int, dict] = {}
        self.old_mem: dict | None = None            # --reuse-run: the old run's memory/<sid>.json (read-only)
        mp = (ctx.reuse or {}).get("memory_cache", {}).get(self.sid)
        if mp:
            self.old_mem = json.load(open(mp))

    # ------------------------------------------------------------- helpers
    def need(self, arm: str, B: int | None = None) -> bool:
        return any(a == arm and (B is None or b == B) for a, b in self.todo)

    def need_nc(self, B: int | None = None) -> bool:           # recalls at ASK time (question-time arms)
        return self.need(ARM_CHUNKS, B) or self.need(ARM_GRAPH, B)

    def need_store(self, B: int | None = None) -> bool:        # needs the HARS session (ingest, or reuse)
        return self.need_nc(B) or self.need(ARM_ENRICHED, B)

    def need_note(self, B: int | None = None) -> bool:         # needs the rolling note
        return self.need_store(B) or self.need(ARM_NOTE, B)

    def hars_sid(self, B: int) -> str:
        return make_hars_sid(self.c.hars_run_id, self.sid, B, self.a.hars_per_budget)

    def project_of(self, B: int) -> str | None:
        """Amendment C.8: the isolated HARS project of (sid[, B]) in the project layout, else None."""
        if self.a.hars_layout != LAYOUT_PROJECT:
            return None
        return L.make_project_id(self.c.hars_run_id, self.sid, B, self.a.hars_per_budget)

    def expected_calls(self) -> int:
        """memory_upsert_document calls of the WHOLE session (all boundaries) at this run's piece parameters."""
        n = 0
        for b in self.plan["boundaries"]:
            pieces = L.split_pieces(self.msgs[b["drop_start"]:b["at_idx"]], self.a.piece_chars)
            n += max(1, L.ceil_div(len(pieces), self.a.pieces_per_call))
        return n

    def ingest_marks(self) -> int:
        """memory_upsert_document calls this run's cache records as finished (resume bookkeeping)."""
        return sum(int(v.get("calls") or 0) for k, v in self.mem.d.items()
                   if str(k).startswith("ingest|") and isinstance(v, dict) and v.get("ok"))

    def note_budget_for_store(self) -> int:
        bs = [B for B in self.a.budgets if self.need_store(B)] or self.a.budgets
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
        if self.old_mem is not None:                     # --reuse-run: the old run's stored summary, byte for byte
            old = self.old_mem.get(key)
            if old and old.get("sha") == sha:
                entry = {**old, "reused_from": self.c.reuse["dir"]}
                self.mem.put(key, entry)
                self.log_comp({"j": j, "arm": kind, "B": B, "cached": True, "reused": True,
                               "realized_tokens": entry["tokens"], "summary_sha256": L.sha256_text(entry["text"])})
                return entry
            if not self.a.reuse_regenerate_notes:
                raise ReuseError(f"{self.sid}: the reused run's {key} {'is missing' if not old else 'was made under another config (sha differs)'}; "
                                 "refusing to regenerate a summary (the reused arms were judged against the old one). "
                                 "Use the original model/params, or --reuse-regenerate-notes to accept a new summary.")
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
    def ingest(self, b: dict, dropped: list[dict], note: str, hsid: str, tag: str, project: str | None = None):
        key = f"ingest|{tag}|j{b['j']}"
        done = self.mem.get(key)
        pieces = L.split_pieces(dropped, self.a.piece_chars)
        n_calls = max(1, L.ceil_div(len(pieces), self.a.pieces_per_call))
        if done and done.get("ok"):
            self.log_comp({"j": b["j"], "arm": "hars", "cached": True, "calls": done["calls"]})
            self.fold_cached_usage(done)
            return
        t0 = time.time()
        seqs = []
        bt = new_extract_totals()                # this boundary's measured extractor usage (the server's extract_usage)
        for c in range(n_calls):
            chunk = pieces[c * self.a.pieces_per_call:(c + 1) * self.a.pieces_per_call]
            summary = note if c == 0 else f"Boundary {b['j']} continuation {c + 1}/{n_calls}: dropped context of the same compaction."
            meta = {"source": "longsession", "sid": self.sid, "boundary": b["j"], "call": c + 1,
                    "calls": n_calls, "pieces": len(chunk), "drop_chars": sum(len(x) for x in chunk)}
            err = ""
            for attempt in (0, 1):
                try:
                    if project:        # C.8: the global memory's own write tool, into the session's neighbour project
                        r = self.c.hars.upsert(project, project_doc_path(project, b["j"], c + 1),
                                               project_doc_content(hsid, b["j"], c + 1, n_calls, summary, chunk))
                    else:
                        r = self.c.hars.compact(hsid, summary, chunk, meta)
                    if isinstance(r, dict) and r.get("ok") is not False:
                        seqs.append(r.get("seq") or r.get("action"))
                        if project:
                            add_extract_usage(bt, r.get("extract_usage"))
                        err = ""
                        break
                    err = f"{'memory_upsert_document' if project else 'memory_compact'} ok=false: {str(r)[:200]}"
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt == 0:
                    time.sleep(2)
            if err:
                self.ingest_ok = False
                self.ingest_err = f"boundary {b['j']} call {c + 1}: {err}"
                self.log_comp({"j": b["j"], "arm": "hars", "ok": False, "error": err})
                return
        self.mem.put(key, {"ok": True, "calls": n_calls, "seqs": seqs, **({"extract_usage": bt} if project else {})})
        if project:
            self.fold_usage(bt)
        self.charge_extraction_estimate(b, dropped)
        self.log_comp({"j": b["j"], "arm": "hars", "ok": True, "cached": False, "pieces": len(pieces),
                       "project": project, "calls": n_calls, "seconds": round(time.time() - t0, 2),
                       "drop_chars": sum(len(x) for x in pieces)})

    def fold_usage(self, bt: dict) -> None:
        for k, v in bt.items():
            self.extract[k] += v
        with self.c.lock:
            for k, v in bt.items():
                self.c.extract[k] += v

    def fold_cached_usage(self, done: dict) -> None:
        """A resumed run: the boundary was ingested by an earlier invocation; its usage is in the cache entry."""
        bt = done.get("extract_usage")
        if isinstance(bt, dict) and all(k in bt for k in self.extract):
            self.fold_usage(bt)

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

    # ------------------------------------------------------------- reuse / enrichment (Amendment C.7)
    def reused_note(self, B: int, j: int, last: int) -> dict | None:
        """--reuse-run: the exact note the old nc-note-only arm used, at the LAST boundary (verified against
        compactions.jsonl by load_reuse). Earlier boundaries are skipped when the old run's rolling cache
        (memory/<sid>.json) is not available (end-only checkpoints never read them)."""
        r = self.c.reuse
        if not r:
            return None
        n = r["notes"].get((self.sid, B))
        if n and j == last:
            self.log_comp({"j": j, "arm": "note", "B": B, "cached": True, "reused": True, "from": n["source"],
                           "realized_tokens": ntok(n["text"]), "summary_sha256": n["sha256"]})
            return {"text": n["text"], "tokens": ntok(n["text"]), "clipped": False, "clipped_repaired": False,
                    "reused_from": r["dir"]}
        if j < last and self.old_mem is None:
            return {"text": "", "tokens": 0, "skipped": True, "clipped": False, "clipped_repaired": False}
        return None

    def sidecar_raw(self, rid: str, query: str, payload: dict, **extra):
        """Raw recall text goes to the recall_raw.jsonl sidecar (never inline in records.jsonl)."""
        self.c.append(RECALL_RAW_FILE, {"rid": rid, "sid": self.sid, "query": query,
                                        "raw": str((payload or {}).get("context") or ""),
                                        "raw_chars": len(str((payload or {}).get("context") or "")),
                                        "error": (payload or {}).get("error"), **extra})

    def enrich(self, b: dict, B: int, tail: str, n_questions: int) -> dict:
        """The post-compaction memory block of nc-enriched for (sid, boundary, B), computed ONCE and persisted
        under enrich/. Inputs: the rolling note and the kept recent context ONLY (never a question)."""
        a, rcm = self.a, L.rc()
        j = b["j"]
        hs = self.hars_sid(B)
        note = self.note_text[B]
        note_tokens = ntok(note)
        cap_info = recall_cap(B, note_tokens)
        cap = cap_info["recall_cap"]
        recent = tail[-int(a.enrich_tail_tokens * 3.6):] if tail else ""
        prompt = enrich_query_prompt(note, recent, a.enrich_queries)
        sample = L.serialize(self.msgs[:b["at_idx"]])[:60000]
        cpt = chars_per_token(sample)
        pid = self.project_of(B)
        sha = hashlib.sha256(json.dumps([prompt, hs, cap, a.packer, ENRICH_PROMPT_VERSION, self.c.sum_llm.model,
                                         tok_label(), a.enrich_queries, cpt, pid]).encode()).hexdigest()
        path = self.c.path("enrich", self.sid, f"j{j}.b{B}.json")
        if os.path.exists(path):
            old = json.load(open(path))
            if old.get("sha") == sha:
                return old
        self.check_budget()
        t0 = time.time()
        r = self.c.sum_llm.complete(prompt, max_tokens=a.enrich_queries * 160 + 160)
        qs = parse_queries(r["text"], a.enrich_queries)
        max_chars = None if a.packer == "legacy" else int(cap * cpt)

        def one(i_q):
            i, q = i_q
            t1 = time.time()
            try:
                pay = (self.c.hars.recall_correct(hs, q["query"], q["ll_keywords"], q["hl_keywords"], max_chars, project=pid)
                       if pid else self.c.hars.recall_correct(hs, q["query"], q["ll_keywords"], q["hl_keywords"], max_chars))
            except Exception as e:  # noqa: BLE001
                pay = {"context": "", "error": f"{type(e).__name__}: {str(e)[:200]}"}
            pay = dict(pay) if isinstance(pay, dict) else {"context": "", "error": "non-dict payload"}
            pay["_seconds"] = round(time.time() - t1, 2)
            return pay

        with cf.ThreadPoolExecutor(max_workers=max(1, a.recall_concurrency)) as ex:
            pays = list(ex.map(one, list(enumerate(qs))))
        per_q = []
        for i, (q, pay) in enumerate(zip(qs, pays)):
            pf = self.c.path("recall", self.sid, f"j{j}", f"enrich.b{B}.q{i}.json")
            atomic_write(pf, json.dumps(pay))
            self.sidecar_raw(f"{self.sid}::enrich::j{j}::b{B}::q{i}", q["query"], pay, kind="enrich_query", j=j, B=B,
                             ll_keywords=q["ll_keywords"], hl_keywords=q["hl_keywords"])
            per_q.append({**q, "seconds": pay.get("_seconds"), "error": pay.get("error"), "max_chars": max_chars,
                          "raw_chars": len(str(pay.get("context") or "")),
                          "degraded": rcm.recall_degraded(pay), "payload_file": os.path.relpath(pf, self.c.out),
                          **graph_branch_of(pay)})
        merged, mstats = merge_recall_payloads(pays)
        docs = merged.pop("_docs", [])
        reasons: list[str] = []
        if not qs:
            reasons.append("query generation produced no usable query")
        failed = sum(1 for x in per_q if x["error"] or not x["raw_chars"])
        if qs and failed / len(qs) > a.enrich_max_failed_frac:
            reasons.append(f"{failed}/{len(qs)} enrichment recalls failed or came back empty")
        nokw = sum(1 for x in per_q if not (x["ll_keywords"] or x["hl_keywords"]))
        if qs and nokw / len(qs) > 0.5:
            reasons.append(f"keywords missing on {nokw}/{len(qs)} queries (the server degrades to naive mode)")
        gb_off = sum(1 for x in per_q if (x["ll_keywords"] or x["hl_keywords"]) and x["graph_branch_enabled"] is False)
        if qs and gb_off / len(qs) > 0.5 and not a.allow_graph_branch_off:
            reasons.append(f"server graph branch disabled on {gb_off}/{len(qs)} keyworded recalls "
                           "(HARS_MEMORY_GRAPH_BRANCH=off?): not the intended configuration")
        deg = sorted({x["degraded"] for x in per_q if x["degraded"]})
        if deg and not a.allow_degraded:
            reasons.append(f"recall degraded ({','.join(deg)})")
        if a.packer == "legacy":
            try:
                packed = rcm.pack_injection(merged.get("context") or "", hs, budget=cap, query=" ".join(q["query"] for q in qs),
                                            wrapper=enrich_wrapper)
                injected, body, tokens = packed["injected"], packed["text"], packed["total_tokens"]
                units = int(packed.get("units_kept") or 0)
            except Exception as e:  # noqa: BLE001
                injected, body, tokens, units = "", "", 0, 0
                reasons.append(f"pack failed: {type(e).__name__}: {str(e)[:120]}")
        elif docs:
            sel, injected, tokens = fill_union(docs, hs, cap)
            body, units = render_chunks_ctx(sel), len(sel)
        else:
            injected, body, tokens, units = "", "", 0, 0
        reasons += list(rcm.recall_failure_reasons(merged, {"units_kept": units, "relations_kept": 0, "entities_kept": 0}))
        if cap_info["recall_cap_clamped_low"] and note_tokens > B:
            reasons.append(f"note alone busts the budget ({note_tokens} > {B})")
        raw_union = "\n".join(str((p or {}).get("context") or "") for p in pays)
        en = {"sha": sha, "sid": self.sid, "j": j, "B": B, "hars_sid": hs, "hars_project": pid, "hars_layout": a.hars_layout,
              "n_questions": n_questions,
              "note_sha256": L.sha256_text(note), "note_tokens": note_tokens, "cap_info": cap_info, "chars_per_token": cpt,
              "packer": a.packer, "top_k_sent": None, "recall_shape": ("C.8 project memory: project=<pid>, NO session_id, " if pid else "") + "C.7 correct usage: query + ll/hl keywords, "
              "mode hybrid, server-default top_k, no session_graph, budget via max_chars" if a.packer != "legacy" else
              "C.7 shape without max_chars, then the legacy packer",
              "query_gen": {"model": self.c.sum_llm.model, "raw": r["text"], "prompt_sha256": L.sha256_text(prompt),
                            "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                            "cost_usd": r["cost_usd"], "latency_s": r["latency_s"], "stop_reason": r["stop_reason"],
                            "recent_context_chars": len(recent)},
              "queries": per_q, "merge": mstats, "packed_text": body, "injected": injected, "block_tokens": tokens,
              "units_kept": units, "reasons": list(dict.fromkeys(reasons)), "raw_union": raw_union,
              "seconds": round(time.time() - t0, 2), "cost_usd": r["cost_usd"],
              "graph_branch_enabled_rate": (round(sum(1 for x in per_q if x["graph_branch_enabled"]) / len(per_q), 4) if per_q else None),
              "keywords_rate": (round(1 - nokw / len(qs), 4) if qs else None)}
        atomic_write(path, json.dumps(en, ensure_ascii=False))
        self.c.totals["sum_cost"] += r["cost_usd"]
        self.log_comp({"j": j, "arm": "enrich", "B": B, "queries": len(qs), "chunks_unique": mstats["chunks_unique"],
                       "duplicates_removed": mstats["duplicates_removed"], "block_tokens": tokens, "cap": cap,
                       "cost_usd": r["cost_usd"], "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                       "latency_s": en["seconds"], "failed_recalls": failed, "reasons": en["reasons"]})
        return en

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
            elif arm == ARM_ENRICHED:
                en = self.enriched[B]
                memory, note_tokens = self.note_text[B], ntok(self.note_text[B])
                injected, cap_info = en["injected"], en["cap_info"]
                reasons += list(en["reasons"])
                if not self.ingest_ok:
                    reasons.append(f"ingest failed: {self.ingest_err}")
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
            if arm == ARM_ENRICHED:       # the recalled facts are part of the post-compaction MEMORY: no recall at ask time
                prefix, suffix = self.prompt_parts(memory + "\n\n" + injected, tail, q["question"], "")
            else:
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
                "run_id": self.c.run_id, "chain_id": self.sid, "sid": self.sid, "cluster_session": self.sid, "session_id": self.hars_sid(B) if arm in STORE_ARMS else self.sid,
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
                "recall_failure_reasons": list(reasons) if arm in NC_ARMS + (ARM_ENRICHED,) else [],
                "recall_degraded": rcm.recall_degraded(pay) if arm in NC_ARMS else None,
                "recall_session_status": ((pay.get("session") or {}).get("status") if arm in NC_ARMS else None),
                "recall_error_kind": pay.get("error_kind") if arm in NC_ARMS else None,
                "allow_degraded": bool(a.allow_degraded),
                "recall_kg_entities_kept": packed.get("entities_kept") if arm in NC_ARMS else None,
                "recall_kg_relations_kept": packed.get("relations_kept") if arm in NC_ARMS else None,
                "persisted_compaction": (self.ingest_ok if arm in STORE_ARMS else (False if arm == ARM_BASELINE else None)),
                "agent_model": self.c.sum_llm.model, "qa_model": self.c.qa_llm.model,
                "agent_provider": a.provider, "tokenizer": tok_label(), "tokenizer_native": tok_native(),
                "hars_layout": a.hars_layout, "hars_project": self.project_of(B) if arm in STORE_ARMS else None,
                "_reasons": reasons,
            }
            row.update(cap_info)
            if arm in NC_ARMS:         # raw recall text -> sidecar; was the anchor lost in HARS ranking or in the packer?
                row["anchor_in_recall_raw"] = bool(pat.search(pay.get("context") or "")) if pat else None
                self.sidecar_raw(rid, q["question"], pay, kind="ask_recall", arm=arm, graph=arm == ARM_GRAPH,
                                 recall_packed_tokens=packed.get("tokens", 0))
            if arm == ARM_ENRICHED:
                ap = pat.search(en["packed_text"]) if pat else None
                ar = pat.search(en["raw_union"]) if pat else None
                row.update({
                    "anchor_in_enriched": bool(ap) if pat else None, "anchor_in_enriched_raw": bool(ar) if pat else None,
                    "anchor_in_recall": bool(ap) if pat else None, "anchor_in_recall_raw": bool(ar) if pat else None,
                    "recall_packed_tokens": en["block_tokens"], "recall_packed_chars": len(en["packed_text"]),
                    "recall_raw_chars": len(en["raw_union"]),
                    "enrich_cost_usd": round(en["cost_usd"] / max(1, en["n_questions"]), 8),
                    "enrich_seconds": en["seconds"], "enrich_queries": len(en["queries"]),
                    "enrich_chunks_unique": en["merge"]["chunks_unique"],
                    "enrich_duplicates_removed": en["merge"]["duplicates_removed"],
                    "enrich_graph_branch_enabled_rate": en["graph_branch_enabled_rate"],
                    "enrich_keywords_rate": en["keywords_rate"], "enrich_packer": en["packer"],
                    "enrich_artifact": f"enrich/{self.sid}/j{j}.b{B}.json",
                    "recall_degraded": None, "recall_error": None})
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
        if self.need_store() and self.a.hars_layout == LAYOUT_PROJECT:
            fails = []
            for B in self.a.budgets:
                if self.need_store(B):
                    fails += post_project_gate(self.c.hars, self.project_of(B), self.expected_calls())
            for (arm, B) in self.todo:
                if arm in STORE_ARMS:
                    self.reasons[(arm, B)] += [f"project gate: {f}" for f in fails]
            self.log_comp({"j": j, "arm": "hars", "project_gate_fails": fails,
                           "projects": [self.project_of(B) for B in self.a.budgets if self.need_store(B)]})
        elif self.need_store():
            st = self.c.hars.status(self.hars_sid(max(self.a.budgets)))
            rcm = L.rc()
            fails, _warns = rcm.status_gate_problems(st) if rcm and hasattr(rcm, "status_gate_problems") else ([], [])
            for (arm, B) in self.todo:
                if arm in STORE_ARMS:
                    self.reasons[(arm, B)] += [f"status gate: {f}" for f in fails]
            self.log_comp({"j": j, "arm": "hars", "status": {k: st.get(k) for k in (
                "compactions", "entities", "relations", "vectors_indexed")}, "gate_fails": fails})
        for B in self.a.budgets:       # nc-enriched: ONE enrichment per (sid, boundary, B); no recall at ask time
            if self.need(ARM_ENRICHED, B):
                self.enriched[B] = self.enrich(b, B, tail, len(ql))
                e_ = self.enriched[B]
                print(f"  [{self.sid}] j={j} B={B} enrich: {len(e_['queries'])} queries, {e_['merge']['chunks_unique']} unique chunks "
                      f"({e_['merge']['duplicates_removed']} dup), block {e_['block_tokens']}/{e_['cap_info']['recall_cap']} tok"
                      f"{' REASONS ' + '; '.join(e_['reasons'][:2]) if e_['reasons'] else ''}", flush=True)
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
                    e = self.reused_note(B, j, len(plan["boundaries"])) or self.summarize("note", B, j, note_prev[B], dropped)
                    note_prev[B] = self.note_text[B] = e["text"]
                    if not e.get("skipped"):
                        self.clip[("note", B)][0] += 1
                        self.clip[("note", B)][1] += int(e["clipped"] and not e["clipped_repaired"])
            reuse_ingested = bool(self.c.reuse) and a.hars_layout == LAYOUT_SPLIT     # split reuse: sessions exist
            if (any(self.need_store(B) for B in a.budgets) and self.ingest_ok and not reuse_ingested
                    and not a.hars_ingested):
                if a.hars_per_budget:
                    for B in a.budgets:
                        if self.need_store(B):
                            self.ingest(b, dropped, self.note_text[B], self.hars_sid(B), f"b{B}", self.project_of(B))
                else:
                    self.ingest(b, dropped, self.note_text[store_B], self.hars_sid(store_B), "shared", self.project_of(store_B))
            if j in plan["checkpoints"] and not a.ingest_only:
                self.ask_checkpoint(b)
            if j % 10 == 0 or j == len(plan["boundaries"]):
                print(f"  [{self.sid}] boundary {j}/{len(plan['boundaries'])} done in {time.time() - t0:.1f}s",
                      flush=True)
        self.finish()

    def finish(self):
        a = self.a
        if a.ingest_only:
            st = post_project_gate(self.c.hars, self.project_of(self.note_budget_for_store()), self.expected_calls()) \
                if self.ingest_ok else [f"ingest failed: {self.ingest_err}"]
            ep = (getattr(self.a, "prices_cache", None) or {}).get("ext")
            self.c.append("ingest.jsonl", {"sid": self.sid, "project": self.project_of(self.note_budget_for_store()),
                                           "hars_sid": self.hars_sid(self.note_budget_for_store()), "ok": not st,
                                           "problems": st, "calls": self.expected_calls(),
                                           "extract_usage": extract_totals_view(self.extract, tuple(ep) if ep else None),
                                           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            print(f"[{self.sid}] ingest-only: project {self.project_of(self.note_budget_for_store())} "
                  f"{'ok' if not st else 'PROBLEMS ' + '; '.join(st[:2])}", flush=True)
            if st:
                raise RuntimeError("; ".join(st))
            return
        for (arm, B) in sorted(self.todo):
            rows = self.rows.get((arm, B), [])
            why = list(dict.fromkeys(self.reasons.get((arm, B), [])))
            kind = "baseline" if arm == ARM_BASELINE else "note"
            n, bad = self.clip[(kind, B)] if arm != ARM_ORIG else (0, 0)
            if n and bad / n > CLIP_LIMIT:
                why.append(f"CLIP FAIL: {bad}/{n} summaries stopped at max_tokens and were not boundary-repaired")
            if arm in STORE_ARMS and not self.ingest_ok:
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
# recall probe (Amendment C.7 diagnostic): no answer model, no judge
# =============================================================================
PROBE_COLS_BASE = ("A_raw", "A_packed", "B_raw")


def probe_dry_run(plans: list[dict], a, prices: dict) -> dict:
    """--recall-probe --dry-run: the only paid part is ONE keyword-writing call per anchored answerable question
    (summariser/extractor-class model); every HARS recall, packing and BM25 step costs nothing."""
    a.prices_cache = prices
    sess, tot = [], 0.0
    for p in plans:
        if not p["boundaries"]:
            continue
        j = p["boundaries"][-1]["j"]
        n = sum(1 for q, _l in p["ask"].get(j, []) if q.get("answerable", True) and (q.get("gold_anchor") or "").strip())
        usd = n * usd_for(a, "sum", 320, 70)
        tot += usd
        sess.append({"sid": p["sid"], "boundaries": len(p["boundaries"]), "probe_questions": n,
                     "per_budget": {str(max(a.budgets)): {"asks": {}, "usd": {"probe": usd}}}, "hars": {"est_ingest_hours": 0.0}})
    res = {"mode": "recall-probe", "sessions": sess, "grand_total_usd": round(tot, 6),
           "probe": {"questions": sum(x["probe_questions"] for x in sess), "usd": round(tot, 6)},
           "reuse_run": (a.reuse_info or {}).get("dir") if a.reuse_info else None}
    if a.json:
        print(json.dumps(res, indent=1, sort_keys=True))
    else:
        print(f"RECALL PROBE DRY RUN: {res['probe']['questions']} anchored answerable question(s), "
              f"{len(sess)} session(s); paid: one keyword call each ~{fmt_usd(tot)}; HARS recalls / BM25 are free")
    return res


def probe_aggregate(rows: list[dict], caps: list[int]) -> dict:
    """Per column: anchor presence over the answerable questions that carry a gold anchor (n), and over the
    RECOVERABLE subset (the anchor occurs in the dropped context at all = the ceiling any retrieval could reach)."""
    cols = list(PROBE_COLS_BASE) + [f"C_{c}" for c in caps] + [f"D_{c}" for c in caps] + ["C_old_cap", "D_old_cap", "anchor_in_note"]

    def rate(sel, col):
        v = [r["hits"].get(col) for r in sel if isinstance(r["hits"].get(col), bool)]
        return {"k": sum(v), "n": len(v), "rate": round(sum(v) / len(v), 4) if v else None}
    rec = [r for r in rows if r.get("anchor_in_dropped") is True]
    out = {"n_questions": len(rows), "n_recoverable": len(rec),
           "all": {c: rate(rows, c) for c in cols}, "recoverable": {c: rate(rec, c) for c in cols}}
    out["anchor_in_dropped"] = {"k": len(rec), "n": len(rows)}
    gb = [r["B_graph_branch_enabled"] for r in rows if isinstance(r.get("B_graph_branch_enabled"), bool)]
    out["B_graph_branch_enabled"] = {"k": sum(gb), "n": len(gb)}
    out["keywords_rate"] = (round(sum(1 for r in rows if r.get("keywords")) / len(rows), 4) if rows else None)
    sh = [r["session_chunk_share_B"] for r in rows if r.get("session_chunk_share_B") is not None]
    out["session_chunk_share_B_mean"] = round(sum(sh) / len(sh), 4) if sh else None
    shA = [r["session_chunk_share_A"] for r in rows if r.get("session_chunk_share_A") is not None]
    out["session_chunk_share_A_mean"] = round(sum(shA) / len(shA), 4) if shA else None
    for c in ["A_raw", "A_packed", "B_raw"]:
        out[f"raw_chars_{c}_median"] = None
    for key, name in (("A_raw_chars", "A"), ("B_raw_chars", "B")):
        xs = sorted(r[key] for r in rows if r.get(key) is not None)
        out[f"raw_chars_{name}_median"] = xs[len(xs) // 2] if xs else None
    return out


def probe_report(summary: dict, caps: list[int], meta: dict) -> str:
    L_ = ["# Recall probe (Amendment C.7): where does the anchor get lost?", "",
          f"run {meta['run_id']} reusing HARS sessions of {meta['reused_run']} | B={meta['B']} | old cap per session "
          f"(B - note): {meta['old_caps']} | {summary['n_questions']} answerable question(s) with a gold anchor, "
          f"{summary['n_recoverable']} of them recoverable (anchor present in the dropped context)", "",
          "A = the OLD frozen shape (no keywords, top_k=8) cut by the legacy packer at the old cap; B = the C.7 correct "
          "shape (query + ll/hl keywords, mode hybrid, server top_k, no cap); C = the correct shape budgeted by the "
          "server's own max_chars at a token cap; D = BM25 over the full dropped context at the same cap. "
          "Anchor presence is a retrieval ceiling, not an answer score.", ""]
    rows = []

    def line(label, col):
        a_, r_ = summary["all"][col], summary["recoverable"][col]
        f = lambda x: "n/a" if x["rate"] is None else f"{x['rate']:.3f} ({x['k']}/{x['n']})"  # noqa: E731
        rows.append([label, f(a_), f(r_)])
    line("anchor in the note (no memory)", "anchor_in_note")
    line("(A) old shape, raw", "A_raw")
    line("(A) old shape + legacy packer @ old cap", "A_packed")
    line("(B) correct shape, raw (server default)", "B_raw")
    line("(C) correct shape, max_chars @ old cap", "C_old_cap")
    line("(D) BM25 @ old cap", "D_old_cap")
    for c in caps:
        line(f"(C) correct shape, max_chars @ {c} tok", f"C_{c}")
        line(f"(D) BM25 @ {c} tok", f"D_{c}")
    L_ += table(["column", "all anchored questions", "recoverable subset"], rows)
    L_ += ["", f"anchor present in the dropped context (ceiling): {summary['anchor_in_dropped']['k']}/{summary['anchor_in_dropped']['n']}",
           f"(B) graph_branch.enabled: {summary['B_graph_branch_enabled']['k']}/{summary['B_graph_branch_enabled']['n']} recalls; "
           f"keywords produced for {summary['keywords_rate']} of the questions",
           f"share of recalled chunks that belong to the session itself: old shape {summary['session_chunk_share_A_mean']}, "
           f"correct shape {summary['session_chunk_share_B_mean']} (the rest is the project knowledge base)",
           f"median raw chars: old shape {summary['raw_chars_A_median']}, correct shape {summary['raw_chars_B_median']}"]
    return "\n".join(L_) + "\n"


def table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return out


def run_recall_probe(ctx, a, sessions, plans) -> int:
    rcm = L.rc()
    import funnel_chains as FC
    reuse = ctx.reuse
    project_layout = a.hars_layout == LAYOUT_PROJECT
    gate = [] if project_layout else reuse_hars_gate(ctx.hars, plans, a, reuse)     # project layout: layout_gate ran in _run
    if gate:
        for g in gate:
            print(f"[probe] HARS gate FAIL: {g}", file=sys.stderr, flush=True)
        print("probe refused: the reused HARS sessions are not usable on this server.", file=sys.stderr)
        return 3
    B = max(a.budgets)
    caps = sorted(set(a.probe_caps))
    rows: list[dict] = []
    old_caps: dict[str, int] = {}
    problems: list[str] = []
    t_start = time.time()
    for plan in plans:
        if not plan["boundaries"]:
            continue
        sid, msgs = plan["sid"], plan["session"]["messages"]
        b = plan["boundaries"][-1]
        j = b["j"]
        hs = make_hars_sid(ctx.hars_run_id if project_layout else reuse["run_id"], sid, B, a.hars_per_budget)
        pid = L.make_project_id(ctx.hars_run_id, sid, B, a.hars_per_budget) if project_layout else None
        n = reuse["notes"].get((sid, B))
        note = n["text"] if n else ""
        if not note and sid in reuse["memory_cache"]:
            note = (json.load(open(reuse["memory_cache"][sid])).get(f"note|B{B}|j{j}") or {}).get("text", "")
        if not note:
            problems.append(f"{sid}: no stored note (cap unknown)")
            continue
        old_cap = recall_cap(B, ntok(note))["recall_cap"]
        old_caps[sid] = old_cap
        dropped = L.serialize(msgs[:b["at_idx"]])
        sample = dropped[:60000]
        cpt = chars_per_token(sample)
        est_scale = ntok(sample[:20000]) / max(1, count_tokens(sample[:20000]))
        bm = FC.BM25(FC.chunk_text(dropped, 128, count_tokens))
        qs = [(q, lag) for q, lag in plan["ask"].get(j, []) if q.get("answerable", True) and (q.get("gold_anchor") or "").strip()]

        def one(ql):
            q, lag = ql
            anc = q["gold_anchor"].strip()
            pat = anchor_pattern(anc)
            has = lambda t: bool(pat.search(t or ""))  # noqa: E731
            r: dict = {"sid": sid, "qid": q["qid"], "qtype": q.get("qtype"), "lag": lag, "gold_anchor": anc, "hars_sid": hs, "hars_project": pid,
                       "old_cap": old_cap, "hits": {"anchor_in_note": has(note)}, "anchor_in_dropped": has(dropped)}
            # (A) the old frozen shape + the legacy packer at the old cap
            pa = ctx.hars.recall(hs, q["question"], False, project=pid) if pid else ctx.hars.recall(hs, q["question"], False)
            ctxA = str((pa or {}).get("context") or "")
            ctx.append(RECALL_RAW_FILE, {"rid": f"{sid}::probe::{q['qid']}::A", "sid": sid, "query": q["question"], "raw": ctxA,
                                         "raw_chars": len(ctxA), "error": (pa or {}).get("error"), "kind": "probe_A"})
            r["A_error"], r["A_raw_chars"] = (pa or {}).get("error"), len(ctxA)
            r["hits"]["A_raw"] = None if (pa or {}).get("error") else has(ctxA)
            r["session_chunk_share_A"] = session_share(ctxA, hs)
            try:
                pk = rcm.pack_injection(ctxA, hs, budget=old_cap, query=q["question"])
                r["hits"]["A_packed"] = has(pk["text"])
                r["A_packed_tokens"] = pk["tokens"]
            except Exception as e:  # noqa: BLE001
                r["hits"]["A_packed"] = None
                r["A_pack_error"] = f"{type(e).__name__}: {str(e)[:100]}"
            # the caller model's keywords (one cheap call per question)
            self_check_budget(ctx)
            kr = ctx.sum_llm.complete(probe_keyword_prompt(q["question"]), max_tokens=300)
            kw = parse_keywords(kr["text"])
            r.update({"keywords": bool(kw["ll_keywords"] or kw["hl_keywords"]), "ll_keywords": kw["ll_keywords"],
                      "hl_keywords": kw["hl_keywords"], "keyword_cost_usd": kr["cost_usd"]})
            # (B) the correct shape, raw
            pb = (ctx.hars.recall_correct(hs, q["question"], kw["ll_keywords"], kw["hl_keywords"], None, project=pid) if pid
                  else ctx.hars.recall_correct(hs, q["question"], kw["ll_keywords"], kw["hl_keywords"], None))
            ctxB = str((pb or {}).get("context") or "")
            ctx.append(RECALL_RAW_FILE, {"rid": f"{sid}::probe::{q['qid']}::B", "sid": sid, "query": q["question"], "raw": ctxB,
                                         "raw_chars": len(ctxB), "error": (pb or {}).get("error"), "kind": "probe_B",
                                         "ll_keywords": kw["ll_keywords"], "hl_keywords": kw["hl_keywords"]})
            r["B_error"], r["B_raw_chars"] = (pb or {}).get("error"), len(ctxB)
            r["hits"]["B_raw"] = None if (pb or {}).get("error") else has(ctxB)
            gbB = graph_branch_of(pb)
            r["B_graph_branch_enabled"] = gbB["graph_branch_enabled"]
            r["B_lightrag_mode"], r["B_mode_fallback"] = gbB["lightrag_mode"], gbB["mode_fallback"]
            r["session_chunk_share_B"] = session_share(ctxB, hs)
            # (C) correct shape via the server's max_chars at the old cap and the sweep; (D) BM25 at the same caps
            r["C"], r["D"] = {}, {}
            for cap, key in [(old_cap, "old_cap")] + [(c, str(c)) for c in caps]:
                try:
                    pay, inj, tot, att = budget_recall(
                        lambda mc, kw=kw: (ctx.hars.recall_correct(hs, q["question"], kw["ll_keywords"], kw["hl_keywords"], mc, project=pid)
                                           if pid else ctx.hars.recall_correct(hs, q["question"], kw["ll_keywords"], kw["hl_keywords"], mc)),
                        hs, cap, cpt)
                    r["hits"][f"C_{key}"] = None if pay.get("error") else has(str(pay.get("context") or ""))
                    r["C"][key] = {"cap": cap, "tokens": tot, "attempts": att, "error": pay.get("error")}
                except Exception as e:  # noqa: BLE001
                    r["hits"][f"C_{key}"] = None
                    r["C"][key] = {"cap": cap, "error": f"{type(e).__name__}: {str(e)[:120]}"}
                sel = bm.fill(q["question"], max(1, int(cap / est_scale)), count_tokens)
                r["hits"][f"D_{key}"] = has("\n".join(sel))
                r["D"][key] = {"cap": cap, "chunks": len(sel)}
            return r

        with cf.ThreadPoolExecutor(max_workers=max(1, a.ask_concurrency)) as ex:
            for r in ex.map(one, qs):
                rows.append(r)
                ctx.append("probe.jsonl", r)
        print(f"  [{sid}] probed {len(qs)} anchored answerable question(s) (old cap {old_cap}, {cpt} chars/token)", flush=True)
    summ = probe_aggregate(rows, caps)
    summ["problems"] = problems
    summ["keyword_cost_usd"] = round(sum(r.get("keyword_cost_usd") or 0.0 for r in rows), 6)
    meta = {"run_id": ctx.run_id, "reused_run": reuse["run_id"], "B": B, "old_caps": old_caps, "hars_layout": a.hars_layout}
    atomic_write(ctx.path("probe-summary.json"), json.dumps({"meta": meta, "caps": caps, **summ}, indent=1))
    md = probe_report(summ, caps, meta)
    atomic_write(ctx.path("REPORT-probe.md"), md)
    atomic_write(ctx.path("MANIFEST.json"), json.dumps({
        "run_id": ctx.run_id, "mode": "recall-probe", "started_s": round(t_start), "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {"keywords": a.sum_model}, "cost_usd": {"keywords": round(ctx.sum_llm.cost, 6)},
        "budget_spent_all_invocations": round(ctx.budget.spent, 6) if ctx.budget else 0.0,
        "reuse": {"dir": reuse["dir"], "run_id": reuse["run_id"]}, "git_head": git_rev(), "params": {
            k: v for k, v in vars(a).items() if k not in ("key_env", "profile_obj", "prices_cache", "reuse_info")}}, indent=1, default=str))
    print(md)
    return 1 if problems else 0


def session_share(ctx_text: str, hsid: str) -> float | None:
    """Share of the recalled chunks that are chunks of the session itself (their document header names the
    HARS session id); the rest is the project knowledge base the same recall also searches."""
    rcm = L.rc()
    docs = rcm._parse_docs(ctx_text or "", rcm._HEADER_CHUNKS) if ctx_text else []
    if not docs:
        return None
    return round(sum(1 for d in docs if hsid in str(d.get("content") or "")) / len(docs), 4)


def self_check_budget(ctx):
    b = ctx.budget
    if b is not None and b.spent >= b.limit:
        raise BudgetExceeded(f"spend ${b.spent:.4f} reached --budget-usd ${b.limit:.4f}")


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
            "provider", "hars_per_budget", "sum_max_input", "sum_max_tokens_slack", "seed", "max_questions", "qtypes", "lags",
            "enrich_queries", "enrich_tail_tokens", "packer", "reuse_run", "reuse_arms")
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
    if not any(a in STORE_ARMS for a in args.arms_list):
        return fails
    try:
        h = ctx.hars.health()
        print(f"[preflight] HARS healthz ok: {str(h)[:120]}", flush=True)
    except Exception as e:  # noqa: BLE001
        return fails + [f"HARS healthz failed: {type(e).__name__}: {e}"]
    if args.hars_layout == LAYOUT_PROJECT:     # no session-recall probe: layout_gate checks the projects themselves
        return fails
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
    ap.add_argument("--arms", default=None, help="comma list of baseline,nc-enriched,nc-note-only,nc-chunks,nc-graph,orig (nc = nc-chunks); "
                                                  "default baseline,nc-enriched,nc-note-only (+orig with --use-real-compactions). "
                                                  "Amendment C.7: nc-chunks / nc-graph (recall AT ASK TIME) are exploratory, opt-in only")
    ap.add_argument("--enrich-queries", type=int, default=8, help="nc-enriched: K retrieval queries (with ll/hl keywords) "
                    "the summariser model writes per boundary from the note + recent context ONLY (default 8)")
    ap.add_argument("--enrich-tail-tokens", type=int, default=12000,
                    help="nc-enriched: at most this many tokens of the kept recent context go into the query prompt (newest kept)")
    ap.add_argument("--enrich-max-failed-frac", type=float, default=0.25,
                    help="nc-enriched: the (sid, boundary, B) block is INVALID when more than this share of its recalls errors/is empty")
    ap.add_argument("--packer", choices=("server", "legacy"), default="server",
                    help="how the recall is cut to the token cap. server (default, Amendment C.7): the server's own max_chars "
                         "(from the cap and the measured chars/token) + whole-chunk fill of the cross-query union; legacy: no "
                         "max_chars, then run_chainreplay's packing v2 (comparison only)")
    ap.add_argument("--allow-graph-branch-off", action="store_true",
                    help="nc-enriched: do not invalidate a block whose recalls report graph_branch.enabled=false")
    ap.add_argument("--reuse-run", default=None, metavar="DIR",
                    help="Amendment C.7: a previous run_longsession output dir whose HARS sessions (ls-<its run id>-<sid>) and notes "
                         "are reused: NO memory_compact, HARS status gate (vectors_indexed > 0, compactions > 0), the exact old note, "
                         "arms in --reuse-arms copied (rows keep their fields + reused_from). End-only checkpoints required.")
    ap.add_argument("--reuse-arms", default=None,
                    help="with --reuse-run: arms whose finished, valid rows are copied from the old run (default baseline,nc-note-only; "
                         "'' = copy nothing)")
    ap.add_argument("--reuse-regenerate-notes", action="store_true",
                    help="with --reuse-run: accept a summary that has to be regenerated because the old run's stored one does not match")
    ap.add_argument("--hars-layout", choices=(LAYOUT_PROJECT, LAYOUT_SPLIT), default=None,
                    help="Amendment C.8. project (default for nc-enriched): the session's dropped context is written with "
                         "memory_upsert_document into its OWN project on the same server (`nc-<run id>-<sid>`, provisioned by "
                         "`home_run.py oh-provision`) and read back with memory_recall(project=..., NO session_id), exactly like "
                         "the global memory. split-session: memory_compact + memory_recall(session_id) = the server's session "
                         "SIDE CHANNEL (cosine/F1 on the question over session chunks), NOT the memory - DIAGNOSTIC ONLY, and the "
                         "only layout of the question-time arms nc-chunks / nc-graph")
    ap.add_argument("--hars-run-id", default=None, metavar="ID",
                    help="project layout: the id in the project names (provisioned up front; default: this run's id)")
    ap.add_argument("--hars-ingested", action="store_true",
                    help="project layout: the projects were ingested by an earlier run with the same --hars-run-id (oh-ingest): "
                         "no ingest here, gate = projects registered and non-empty")
    ap.add_argument("--ingest-only", action="store_true",
                    help="project layout: write the dropped spans (+ the reused notes) into the projects and stop: no queries, "
                         "no asks, no answer model; spends only the server-side extraction (estimated)")
    ap.add_argument("--recall-probe", action="store_true",
                    help="Amendment C.7 diagnostic, NO answer model and NO judge: with --reuse-run, for every answerable question compare "
                         "(A) the old frozen recall shape + legacy packer, (B) the correct shape raw, (C) the correct shape via max_chars at "
                         "the old cap and at --probe-caps, (D) BM25 over the full dropped context at the same caps (anchor presence)")
    ap.add_argument("--probe-caps", default="800,1536,3000,6000", help="recall-probe: token caps of the (C)/(D) sweep")
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
    arms = [x.strip() for x in (a.arms or (",".join(DEFAULT_ARMS) + (",orig" if a.use_real_compactions else ""))).split(",") if x.strip()]
    arms = [ARM_ALIASES.get(x, x) for x in arms]
    bad = [x for x in arms if x not in ALL_ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; known {ALL_ARMS}")
    a.arms_list = [x for x in ALL_ARMS if x in arms]
    if ARM_ORIG in a.arms_list and not a.use_real_compactions:
        raise SystemExit("arm 'orig' needs --use-real-compactions (real summaries)")
    if a.keep_tail + a.reserve >= a.window and not a.use_real_compactions:
        raise SystemExit(f"--keep-tail {a.keep_tail} + --reserve {a.reserve} must be < --window {a.window}")
    if a.reuse_run is None:
        a.reuse_arms = ""
        if a.recall_probe:
            raise SystemExit("--recall-probe needs --reuse-run (it probes the existing HARS sessions)")
    else:
        a.reuse_arms = "baseline,nc-note-only" if a.reuse_arms is None else a.reuse_arms
        bad_r = [x for x in a.reuse_arms.split(",") if x.strip() and x.strip() not in a.arms_list]
        if bad_r:
            raise SystemExit(f"--reuse-arms {bad_r} not in --arms {a.arms_list}")
        if ARM_ENRICHED in [x.strip() for x in a.reuse_arms.split(",")]:
            raise SystemExit("--reuse-arms must not contain nc-enriched: that arm is the one being paid for")
    a.probe_caps = [int(x) for x in str(a.probe_caps).split(",") if x.strip()]
    question_time = [x for x in a.arms_list if x in (ARM_CHUNKS, ARM_GRAPH)]
    if a.hars_layout is None:
        a.hars_layout = LAYOUT_SPLIT if question_time else LAYOUT_PROJECT
    if a.hars_layout == LAYOUT_PROJECT and question_time:
        raise SystemExit(f"--hars-layout project cannot run the question-time arms {question_time}: they recall with a session_id "
                         "(split-session, diagnostic only)")
    if a.hars_layout == LAYOUT_SPLIT and (a.hars_ingested or a.ingest_only or a.hars_run_id):
        raise SystemExit("--hars-ingested / --ingest-only / --hars-run-id belong to --hars-layout project")
    if a.ingest_only:
        a.arms_list = [ARM_ENRICHED]
        a.reuse_arms = ""
        if a.recall_probe or a.hars_ingested:
            raise SystemExit("--ingest-only excludes --recall-probe / --hars-ingested")
    if a.recall_probe and a.hars_layout == LAYOUT_PROJECT and not a.hars_ingested:
        raise SystemExit("--recall-probe in the project layout probes ingested projects: pass --hars-ingested (run oh-ingest first)")
    a.reuse_info = None
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
    reuse_why: list[str] = []
    if a.reuse_run:
        a.reuse_info = load_reuse(a, plans)
        reuse_why = reuse_problems(plans, a.reuse_info, a)
        for w in a.reuse_info["warnings"]:
            print(f"[reuse] warning: {w}", file=sys.stderr)
        for w in reuse_why:
            print(f"[reuse] PROBLEM: {w}", file=sys.stderr)
    if a.dry_run:
        make_counter(a, live=False)
        if a.recall_probe:
            probe_dry_run(plans, a, prices)
        else:
            print_dry_run(plans, a, prices, as_json=a.json)
        if reuse_why:
            print(f"!!! --reuse-run has {len(reuse_why)} problem(s) (above): the real run would refuse", file=sys.stderr)
            return 3
        return 0
    if reuse_why:
        print("--reuse-run refused: nothing was spent. Fix the problems above.", file=sys.stderr)
        return 3
    if not a.fake_hars:
        L.rc_require("mcp_call", "hars_recall", "hars_recall_correct", "hars_status", "preflight_verdict")
    if any(x in STORE_ARMS for x in a.arms_list) or a.recall_probe:
        L.rc_require("pack_injection", "recall_failure_reasons", "recall_degraded", "graph_failure_reasons",
                     "graph_summary", "split_packed_sections", "status_gate_problems", "_parse_docs")
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
    if a.reuse_info:
        ctx.reuse = a.reuse_info
        if a.hars_layout == LAYOUT_SPLIT:
            ctx.hars_run_id = a.reuse_info["run_id"]
    if a.hars_layout == LAYOUT_PROJECT and a.hars_run_id:
        ctx.hars_run_id = a.hars_run_id
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
    if a.hars_layout == LAYOUT_PROJECT and (any(x in STORE_ARMS for x in a.arms_list) or a.recall_probe):
        lg = layout_gate(ctx, a, plans)
        if lg:
            for g in lg:
                print(f"[layout] GATE FAIL: {g}", file=sys.stderr, flush=True)
            print("project-layout gate failed - nothing spent.", file=sys.stderr)
            return 3
    elif a.hars_layout == LAYOUT_SPLIT and (any(x in STORE_ARMS for x in a.arms_list) or a.recall_probe):
        print("[layout] split-session is DIAGNOSTIC ONLY: the recall is the server's session side channel, not the memory "
              "(REPORT-PROTOCOL.md Amendment C.8)", flush=True)
    if a.recall_probe:
        return run_recall_probe(ctx, a, sessions, plans)
    if ctx.reuse:
        if a.hars_layout == LAYOUT_SPLIT:
            gate = reuse_hars_gate(hars, plans, a, ctx.reuse)
            if gate:
                for g in gate:
                    print(f"[reuse] HARS gate FAIL: {g}", file=sys.stderr, flush=True)
                print("reuse gate failed - nothing spent (the reused HARS sessions are not usable on this server).", file=sys.stderr)
                return 3
        nrows = 0 if a.ingest_only else copy_reused_rows(ctx, plans, ctx.reuse)
        print(f"[reuse] run {ctx.reuse['run_id']}: {len([p for p in plans if p['boundaries']])} session(s); "
              f"{nrows} row(s) of arms {ctx.reuse['arms']} copied (reused_from set); "
              + ("HARS sessions reused, no memory_compact will be called" if a.hars_layout == LAYOUT_SPLIT else
                 "notes reused; the dropped spans are written into the nc-* projects (data is NOT reusable across layouts)"),
              flush=True)
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
        parity = parity_enriched(rows, a.budgets) if ARM_ENRICHED in a.arms_list else None
        for B_, pb_ in (parity or {}).items():
            tag_ = "in window" if pb_["in_window"] else "OUTSIDE the window"
            print(f"[parity] B={B_}: nc-enriched / baseline ask-time memory = {pb_['ratio_enriched_vs_baseline']} "
                  f"({pb_['enriched_memory_tokens_mean']} / {pb_['baseline_memory_tokens_mean']} native tok) {tag_} {pb_['window']}",
                  flush=True)
        summary = {"run_id": run_id, "rows": len(rows), "by_arm_budget": summarize_rows(rows),
                   "parity_enriched": parity,
                   "reuse": ({"dir": a.reuse_info["dir"], "run_id": a.reuse_info["run_id"], "arms": a.reuse_info["arms"],
                              "note_sources": {f"{k[0]}@B{k[1]}": v["source"] for k, v in a.reuse_info["notes"].items()}}
                             if a.reuse_info else None),
                   "failed_sessions": failed, "llm_cost_usd": {"summariser": round(sum_llm.cost, 6),
                                                              "answer": round(qa_llm.cost, 6)},
                   "prompt_cache": {"summariser": sum_llm.cache_stats(), "answer": qa_llm.cache_stats()},
                   "llm_cost_usd_all_invocations": total_cost, "tokenizer": tok_label(),
                   "tokenizer_native": tok_native(),
                   "token_counter": (tokcount.get_default().manifest() if tokcount.get_default() else None),
                   "hars_extraction_estimated_usd": round(ctx.est_hars_usd, 6),
                   "hars_extraction_measured": extract_totals_view(
                       ctx.extract, tuple((getattr(a, "prices_cache", None) or {}).get("ext") or ()) or None),
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
                                  "hars_extraction_measured_usd": summary["hars_extraction_measured"].get("usd"),
                                  "unpriced_calls": sum_llm.unpriced_calls + qa_llm.unpriced_calls,
                                  "budget_spent_total": round(budget.spent, 6) if budget else None},
                    "prices_usd_per_mtok": {k: list(v) for k, v in prices.items()},
                    "params": {k: v for k, v in vars(a).items() if k not in ("key_env", "profile_obj", "prices_cache", "reuse_info")},
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
