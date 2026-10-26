#!/usr/bin/env python3
"""Chained-compaction QA chains: C=3 consecutive real compaction boundaries.

A chain is one session cut at three consecutive compaction boundaries, giving
three spans and a known re-summarization lag per span:

    span 1  (boundary k-1 .. k)      lag 3   re-summarized twice more
    span 2  (boundary k   .. k+1)    lag 2   re-summarized once more
    span 3  (boundary k+1 .. k+2)    lag 1   never re-summarized

Per chain exactly 7 QA items are asked from ONE generator call (JSON batch):
2 per span (one whose fact sits in the span's first half -> slice HEAD, one in
its second half -> slice TAIL) plus 1 adversarial UNANSWERABLE item whose
invented sibling detail must not occur anywhere in the chain or the session.

Span construction is byte-identical to make_longreplay_dataset.py: the same
`split_segments` / `replay_window` are imported and a role-aware re-implementation
is asserted equal to them on every span (--dry-run prints the verdict), so a
question here is grounded in exactly the text run_longreplay.py replays.

Usage:
  python3 make_chains.py --dry-run --max-chains 10          # spans only, $0
  OPENROUTER_API_KEY=... python3 make_chains.py --limit 2    # paid pilot

Outputs (defaults): chains.jsonl, chains-manifest.json

Independence options (all OPT-IN; without them the output is byte-identical to the
pre-2026-10-26 builder, see tests/test_make_chains.py):
  --disjoint        chains of one session never share a span: boundaries are tiled in
                    non-overlapping blocks 0-2, 3-5, 6-8, ... (a 2nd chain per session
                    only when it has >= 6 boundaries). Default (overlapping) enumeration
                    makes c0 = boundaries 0-2 and c1 = 1-3, i.e. 2 of 3 spans shared.
  --unique-probes   unanswerable probes are drawn WITHOUT replacement from a pool of
                    distinct phantom identifiers (PROBE_POOL_*; NC_PHANTOM_TTL_MS kept);
                    exits if the pool is smaller than the unanswerables requested, and
                    rejects (-> repair) an unanswerable that reuses an earlier probe.
  --dedup-anchors   rejects (-> repair) an answerable item whose gold_anchor is already
                    the gold_anchor of another chain of the SAME session; rejections are
                    recorded per chain in the manifest (`rejections`, `independence_rejections`).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
# single source of truth for segmentation + replay window (do not diverge)
from make_longreplay_dataset import (  # noqa: E402
    CHARS_PER_TOKEN, TOKEN_BUDGET, DEFAULT_SEGS, replay_window, split_segments,
)

SESS_DIR = os.path.join(BASE, "longreplay", "sessions")
CHAIN_LEN = 3                  # C: consecutive boundaries per chain
ITEMS_PER_SPAN = 2             # HEAD + TAIL
QA_MODEL_DEFAULT = "z-ai/glm-5.3-flash"
AGENT_MODEL = "qwen/qwen3.8-flash"   # arm under test; never used for generation
MIN_SPAN_CHARS = 8000          # same floor as make_longreplay_dataset.py
MIN_EVIDENCE_WORDS = 8
MAX_ANCHOR_HITS_OWN_SPAN = 3
MAX_Q_WORDS = 30               # canonical_short_answer word budget
MAX_RETRIES_PER_SLOT = 2       # batch repair rounds; a retry round covers all failing slots
PROMPT_HALF_CHARS = 24000      # per HEAD/TAIL half shown to the generator (cost guard)
URL = "https://openrouter.ai/api/v1/chat/completions"

# OpenRouter list prices (USD / token) for the estimate printout; overridable.
PRICE_IN = float(os.environ.get("NC_PRICE_IN", "0.00000015"))
PRICE_OUT = float(os.environ.get("NC_PRICE_OUT", "0.0000005"))

ANCHOR_BAD = {"", "true", "false", "yes", "no", "null", "none", "n/a"}

# ---------------------------------------------------------------- text helpers

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def strip_wrappers(s: str) -> str:
    """Remove prompt scaffolding a model tends to echo back (not content)."""
    s = (s or "").strip()
    for _ in range(3):
        before = s
        s = re.sub(r"^\[?\s*(SPAN|HEAD|TAIL|EVIDENCE|ANSWER|Q)\s*\d*\]?\s*[:\-]?\s*",
                   "", s, flags=re.I)
        s = s.strip().strip("`").strip()
        if s[:1] in "\"'" and s[-1:] == s[:1] and len(s) > 2:
            s = s[1:-1].strip()
        if s == before:
            break
    return s


def anchor_re(anchor: str, flags=re.I):
    """Escaped, token-bounded matcher: 28.6 matches 28.6, NX-4471 matches NX-4471,
    and `port` never matches inside `transport`."""
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(anchor.strip())
                      + r"(?![A-Za-z0-9_])", flags)


PROBE_PATTERNS = [
    re.compile(r"(?<![\w-])--?[A-Za-z][A-Za-z0-9-]{1,}"),                    # --flag
    re.compile(r"[A-Za-z_][A-Za-z0-9]*(?:[._\-/][A-Za-z0-9]+)+"),            # a.b_c-d/e
    re.compile(r"[A-Za-z_]*\d[A-Za-z0-9_]*"),                                # nx4471, v2
    re.compile(r"\d+(?:[.,]\d+)+"),                                          # 28.6
    re.compile(r"\d{2,}"),                                                   # 8090
]


def probe_tokens(question: str) -> list[str]:
    """Concrete identifiers/numbers in a question -> the unanswerable probe set."""
    found, seen = [], set()
    for pat in PROBE_PATTERNS:
        for m in pat.finditer(question or ""):
            t = m.group(0).strip("`'\" ")
            if len(t) < 2 or t.lower() in ANCHOR_BAD or t.lower() in seen:
                continue
            seen.add(t.lower())
            found.append(t)
    return found


def has_verbatim(span_text: str, evidence: str) -> bool:
    ev = norm_ws(evidence)
    return bool(ev) and ev in norm_ws(span_text)


# ------------------------------------------------------------ span construction

def session_spans(s: dict) -> list[list[dict]]:
    """Span events per boundary, using the exact prev/idx walk of the original
    generator: span = events[prev : boundaries[i].idx], prev = idx + 1."""
    ev, bounds = s["events"], s["boundaries"]
    out, prev = [], 0
    for b in bounds:
        out.append(ev[prev:b["idx"]])
        prev = b["idx"] + 1
    return out


def segments_struct(span_events: list[dict], chunk: int = 3500) -> list[list[dict]]:
    """Role-aware twin of split_segments(): identical cut points, same ^CHUNK
    accumulation on turn edges. Each segment is the list of events in it."""
    segs, cur, cur_len = [], [], 0
    for e in span_events:
        t = (e.get("text") or "").strip()
        if not t:
            continue
        piece = f"[{e['role']}] {t}\n\n"
        if cur_len + len(piece) > chunk and cur:
            segs.append(cur)
            cur, cur_len = [e], len(piece)
        else:
            if not cur:
                cur, cur_len = [], 0
            cur.append(e)
            cur_len += len(piece)
    if cur:
        segs.append(cur)
    return segs


def seg_text(seg: list[dict]) -> str:
    return "".join(f"[{e['role']}] {(e.get('text') or '').strip()}\n\n" for e in seg)


def window_struct(segs: list[list[dict]], tail_n: int) -> tuple[list[list[dict]], int]:
    """Twin of replay_window(): same tail_n cut and the same TOKEN_BUDGET pop loop."""
    win = list(segs[-tail_n:]) if tail_n else list(segs)
    dropped = len(segs) - len(win)
    while len(win) > 1 and sum(len(seg_text(s)) for s in win) / CHARS_PER_TOKEN >= TOKEN_BUDGET:
        win.pop(0)
        dropped += 1
    return win, dropped


def selfcheck_span(span_events: list[dict], tail_n: int) -> bool:
    """Prove segmentation equality with the imported original on one span."""
    a = split_segments(span_events)
    b = [seg_text(s) for s in segments_struct(span_events)]
    if a != b:
        return False
    wa, _ = replay_window(a, tail_n)
    wb, _ = window_struct(segments_struct(span_events), tail_n)
    return wa == [seg_text(s) for s in wb]


def build_chain(s: dict, chain_idx: int, tail_n: int, check: bool) -> dict:
    """Assemble one chain's 3 spans (tail_n segments each, TOKEN_BUDGET capped)."""
    all_spans = session_spans(s)
    spans, reasons = [], []
    for k in range(CHAIN_LEN):
        bi = chain_idx + k
        events = all_spans[bi]
        if check and not selfcheck_span(events, tail_n):
            reasons.append(f"segmentation_mismatch:boundary{bi}")
        segs = segments_struct(events)
        win, dropped = window_struct(segs, tail_n)
        texts = [seg_text(x) for x in win]
        text = "\n\n".join(texts)
        chars = sum(len(t) for t in texts)
        if len(win) < 2 or chars < MIN_SPAN_CHARS:
            reasons.append(f"thin_span:span{k + 1}:{chars}chars/{len(win)}segs")
        spans.append({
            "span_idx": k + 1,
            "lag": CHAIN_LEN - k,
            "boundary_idx": s["boundaries"][bi].get("idx"),
            "segments": [{"role": e.get("role", ""), "text": (e.get("text") or "")}
                         for seg in win for e in seg],
            "chars": chars,
            "est_tokens": int(chars / CHARS_PER_TOKEN),
            "dropped_segments": dropped,
            "text": text,
        })
    return {
        "session_id": s["sid"],
        "title": (s.get("title") or "").strip(),
        "chain_idx": chain_idx,
        "spans": spans,
        "absent_probes": absent_probe_candidates([sp["text"] for sp in spans],
                                                 transcript_of(s), salt=s["sid"]),
        "reasons": reasons,
    }


def span_fp(spans: list[dict]) -> str:
    """Content fingerprint over the FORMATTED segment text, computable identically
    from freshly built spans and from chains.jsonl records (which store the same
    flat segments). The old resume path hashed raw segment texts while generation
    hashed the joined span text, so the two never matched: a chain whose content
    duplicated an already-stored chain was paid for a second time after a resume
    (yUemS6tx_c0 == gcKDB2Q7_c0, 2026-10-25)."""
    return hashlib.sha1("\x00\x00".join(
        f"[{e.get('role', '')}] {(e.get('text') or '').strip()}\n\n"
        for sp in spans for e in sp["segments"]).encode()).hexdigest()[:16]


def chain_starts(n_boundaries: int, disjoint: bool = False) -> range:
    """Start boundaries of the chains a session yields. Overlapping (default): every
    start 0..n-C, so chain c and c+1 share C-1 spans. Disjoint: starts 0, C, 2C, ...
    (span ranges [k, k+C-1] never overlap)."""
    return range(0, n_boundaries - CHAIN_LEN + 1, CHAIN_LEN if disjoint else 1)


def enumerate_chains(sessions: list[dict], disjoint: bool = False) -> list[tuple[int, int]]:
    """Deterministic round-robin: chain 0 of every session, then chain 1, ...
    Spreads the pilot across sessions instead of taking 7 near-duplicate chains
    from one long session (bootstrap is clustered on chains).

    The second element is the chain's START boundary (= chain_idx). With
    disjoint=True only non-overlapping boundary blocks are emitted."""
    per = [(i, chain_starts(len(s["boundaries"]), disjoint))
           for i, s in enumerate(sessions) if len(s["boundaries"]) >= CHAIN_LEN]
    order, depth = [], 0
    while True:
        row = [(i, c[depth]) for i, c in per if depth < len(c)]
        if not row:
            return order
        order.extend(row)
        depth += 1


# --------------------------------------------------------------------- model

class Spend:
    def __init__(self, budget: float):
        self.budget, self.spent, self.calls = budget, 0.0, 0

    def add(self, usage: dict):
        cost = usage.get("cost") or usage.get("total_cost")
        if cost is None:
            cost = ((usage.get("prompt_tokens") or 0) * PRICE_IN
                    + (usage.get("completion_tokens") or 0) * PRICE_OUT)
        self.spent += float(cost)
        self.calls += 1
        print(f"  [spend] call {self.calls}: +${float(cost):.5f} "
              f"(in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')}) "
              f"total ${self.spent:.5f} / ${self.budget:.2f}", flush=True)

    def can_afford(self, est: float) -> bool:
        return self.spent + est <= self.budget


RAW_LOG = {"path": "", "n": 0}
BLOCKED = {"n": 0}        # consecutive 401s in this process: the endpoint then
                          # refuses every later call, retrying only burns time


def log_raw(entry: dict):
    if not RAW_LOG["path"]:
        return
    RAW_LOG["n"] += 1
    entry = dict(entry, call=RAW_LOG["n"])
    with open(RAW_LOG["path"], "a") as w:
        w.write(json.dumps(entry, ensure_ascii=False) + "\n")


def http_post(body: dict, key: str, timeout: int = 300) -> tuple[int, str]:
    """POST to OpenRouter. `requests` when available (it handles connection reuse and
    exposes redirect history), urllib otherwise. Returns (status, text)."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
               "User-Agent": "neurocompact-eval/2.0"}
    try:
        import requests  # noqa: PLC0415 - optional dependency
        r = requests.post(URL, headers=headers, json=body, timeout=timeout)
        return r.status_code, r.text
    except ImportError:
        pass
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def llm(prompt: str, key: str, model: str, spend: Spend, max_tokens: int = 10000,
        est_cost: float = 0.02, temperature: float = 0.2,
        retry_waits: tuple[int, ...] = (10, 30)) -> tuple[str, dict]:
    """One generation call. Same request shape as make_longreplay_dataset.llm
    (reasoning.max_tokens=300 is mandatory for glm-5.3-flash: it rejects requests
    with reasoning disabled), plus usage capture and a hard budget gate.

    max_tokens must leave room for the answer: reasoning tokens count against it,
    and a 6144 cap truncated the 7-item JSON batch on 2026-10-25.
    """
    if not spend.can_afford(est_cost):
        print(f"  [spend] refused: est ${est_cost:.4f} would exceed ${spend.budget:.2f}")
        return "", {}
    if not key.startswith("sk-or-"):
        # 2026-10-25: the repair-mapping loop reused `key` as a slot variable, so every
        # third call sent "Authorization: Bearer ('answerable', 3, 'TAIL')". OpenRouter
        # answers that with 401 "Missing Authentication header", which looked like an
        # endpoint flake. Fail loudly instead of retrying.
        raise SystemExit(f"llm(): API key looks shadowed/corrupt "
                         f"(len={len(key)}, repr={key!r:.60})")
    body = {
        "model": model, "max_tokens": max_tokens, "temperature": temperature,
        "reasoning": {"max_tokens": 300},
        "usage": {"include": True},
        "messages": [{"role": "user", "content": prompt}],
    }
    if BLOCKED["n"] >= 2:
        print("  llm skipped: this process already saw repeated 401s "
              "(endpoint refuses further calls; rerun later)", flush=True)
        return "", {}
    last = ""
    for attempt in range(len(retry_waits) + 1):
        try:
            status, text = http_post(body, key)
        except Exception as e:  # noqa: BLE001 - transport failure, retried
            status, text = 0, f"{type(e).__name__}: {e}"
        if status == 200:
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                status, text = 0, text[:200]
            else:
                usage = raw.get("usage") or {}
                spend.add(usage)
                msg = raw["choices"][0]["message"]
                content = msg.get("content") or ""
                log_raw({"kind": "llm", "model": model, "prompt_chars": len(prompt),
                         "prompt": prompt, "raw": content,
                         "reasoning_chars": len(msg.get("reasoning") or ""),
                         "finish_reason": raw["choices"][0].get("finish_reason"),
                         "usage": usage})
                if not content:
                    print(f"  llm returned no content "
                          f"(reasoning={len(msg.get('reasoning') or '')} chars, "
                          f"finish={raw['choices'][0].get('finish_reason')})", flush=True)
                return content, usage
        last = f"HTTP {status}: {text[:400]}"
        if status == 401:
            BLOCKED["n"] += 1
        print(f"  llm HTTP {status} (attempt {attempt + 1}/{len(retry_waits) + 1}): "
              f"{text[:160]}", flush=True)
        if attempt < len(retry_waits):
            time.sleep(retry_waits[attempt])
    log_raw({"kind": "llm_failed", "model": model, "prompt_chars": len(prompt),
             "prompt": prompt, "error": last})
    print(f"  llm failed: {last[:200]}", flush=True)
    return "", {}


def parse_items(raw: str) -> list[dict]:
    """Tolerant JSON extraction (fenced / raw newlines / trailing prose)."""
    if not raw:
        return []
    lo, hi = raw.find("{"), raw.rfind("}")
    if lo < 0 or hi <= lo:
        return []
    blob = raw[lo:hi + 1]
    for kwargs in ({}, {"strict": False}):
        try:
            j = json.loads(blob, **kwargs)
        except Exception:
            continue
        items = j.get("items") or j.get("questions") or []
        out = []
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            kind = (it.get("kind") or ("unanswerable" if it.get("answerable") is False
                                       else "answerable")).strip().lower()
            out.append({
                "kind": "unanswerable" if kind.startswith("unans") else "answerable",
                "span_idx": it.get("span_idx"),
                "slice": (it.get("slice") or "").strip().upper() or None,
                "question": strip_wrappers(it.get("question") or it.get("q") or ""),
                "canonical_short_answer": strip_wrappers(
                    it.get("canonical_short_answer") or it.get("a") or it.get("answer") or ""),
                "gold_anchor": strip_wrappers(it.get("gold_anchor") or it.get("anchor") or ""),
                "evidence": strip_wrappers(it.get("evidence") or ""),
            })
        return out
    return []


# ----------------------------------------------------------------- prompting

def halves(text: str) -> tuple[str, str]:
    mid = len(text) // 2
    return text[:mid], text[mid:]


def evidence_windows(text: str, want: int = 8) -> list[str]:
    """Guaranteed-verbatim, >= 8-word quote candidates: groups of 2-4 CONSECUTIVE
    lines (a config block, a log burst, a sentence that spans lines).

    Handing these to the generator is what makes the 8-consecutive-words evidence
    gate cheap instead of a retry loop: a span full of terse config lines has no
    8-word sentence, and models kept quoting single lines or joining distant ones
    (observed 2026-10-25). Line groups keep the quote readable."""
    lines = text.split("\n")
    groups, i = [], 0
    while i < len(lines):
        if len(lines[i].split()) < 1:
            i += 1
            continue
        grp, j = [], i
        while j < len(lines) and len("\n".join(grp).split()) < 10 \
                and len("\n".join(grp)) < 500:
            grp.append(lines[j])
            j += 1
        block = "\n".join(grp).strip()
        if len(block.split()) >= MIN_EVIDENCE_WORDS and len(block) <= 700:
            groups.append(block)
        i = max(j, i + 1)
    uniq, seen = [], set()
    for g in groups:
        key = norm_ws(g).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(g)
    if not uniq:
        return []
    if len(uniq) <= want:
        return uniq
    step = (len(uniq) - 1) / (want - 1)
    return [uniq[round(k * step)] for k in range(want)]


def suggested_anchors(windows: list[str], span_text: str, other_texts: list[str],
                      limit: int = 6, exclude: set[str] | None = None) -> list[tuple[int, str]]:
    """(window_no, token) pairs that already pass the anchor gates by construction:
    exactly one occurrence in this span, zero in the other two, >= 3 chars, and
    inside a candidate evidence window of the same half. Removes the most common
    generator failures (anchor repeated 4+ times, anchor absent, anchor in
    another span) without changing the protocol."""
    out, seen = [], set()
    for n, w in enumerate(windows, 1):
        for t in probe_tokens(w):
            tl = t.lower()
            if tl in seen or len(t) < 3 or re.fullmatch(r"\d{1,2}", t):
                continue
            if exclude and tl in exclude:     # --dedup-anchors: used by a sibling chain
                continue
            if len(anchor_re(t).findall(span_text)) != 1:
                continue
            if any(anchor_re(t).search(o) for o in other_texts):
                continue
            seen.add(tl)
            out.append((n, t))
    out.sort(key=lambda nt: (0 if re.search(r"[\d_.\-/:]", nt[1]) else 1, -len(nt[1])))
    return out[:limit]


def span_block(span: dict, half_chars: int, other_texts: list[str] | None = None,
               exclude: set[str] | None = None) -> str:
    head, tail = halves(span["text"])
    if len(head) > half_chars:
        head = head[:half_chars] + "\n[...HEAD truncated...]"
    if len(tail) > half_chars:
        tail = "[...TAIL truncated...]\n" + tail[-half_chars:]
    lines = [f"SPAN {span['span_idx']} (lag {span['lag']}: compacted after this span "
             f"{span['lag'] - 1} more time(s); {span['chars']} chars)",
             f"[HEAD]\n{head}", f"[TAIL]\n{tail}"]
    for sl, txt in (("HEAD", head), ("TAIL", tail)):
        wins = evidence_windows(txt)
        if wins:
            lines.append(f"[CANDIDATE EVIDENCE {sl} of SPAN {span['span_idx']} — "
                         f"quote one of them verbatim]")
            lines += [f"E{span['span_idx']}{sl[0]}{n + 1} ({len(w.split())} words): {w}"
                      for n, w in enumerate(wins)]
            anchors = suggested_anchors(wins, span["text"], other_texts or [],
                                        exclude=exclude)
            if anchors:
                lines.append(f"[SUGGESTED ANCHORS {sl} of SPAN {span['span_idx']} — copy "
                             f"ONE verbatim; each occurs once here and never in the other spans]")
                lines.append(", ".join(f"E{span['span_idx']}{sl[0]}{n}->{t}"
                                       for n, t in anchors))
    return "\n".join(lines)


def absent_probe_candidates(span_texts: list[str], transcript: str,
                            limit: int = 5, salt: str = "") -> list[str]:
    """Invented env-var names that are verified to occur NOWHERE in the session, so the
    unanswerable item's probe gate passes by construction. The generator kept inventing
    names that really exist in this repo's session (NINJA_*, ninja-coder, ninja-mcp.env),
    which the gate correctly rejected three rounds in a row (2026-10-25).

    `salt` (session id) rotates the base/tail order so consecutive chains do not all
    bait with the same NC_PHANTOM_* token."""
    bases = ("PHANTOM", "GHOST", "ZEPHYR", "OBSIDIAN", "QUARTZ", "VECTOR")
    tails = ("RETENTION_HOURS", "TTL_MS", "QUOTA_FACTOR", "FLUSH_INTERVAL_S",
             "MAX_DEPTH", "WINDOW_SECONDS")
    if salt:
        off = int(hashlib.sha1(salt.encode()).hexdigest(), 16)
        bases = bases[off % len(bases):] + bases[:off % len(bases)]
        tails = tails[off % len(tails):] + tails[:off % len(tails)]
    out = []
    for b in bases:
        for t in tails:
            cand = f"NC_{b}_{t}"
            if anchor_re(cand).search(transcript) or any(anchor_re(cand).search(x)
                                                        for x in span_texts):
                continue
            out.append(cand)
            if len(out) >= limit:
                return out
    return out


# --unique-probes pool: the 6x6 grid above plus extra bases/tails -> 12 x 10 = 120
# distinct identifiers. NC_PHANTOM_TTL_MS stays in the pool (compatibility with the
# existing dataset, where it is the probe of 7/10 unanswerables).
PROBE_POOL_BASES = ("PHANTOM", "GHOST", "ZEPHYR", "OBSIDIAN", "QUARTZ", "VECTOR",
                    "NEBULA", "COBALT", "TUNDRA", "MERIDIAN", "HALCYON", "SPECTRE")
PROBE_POOL_TAILS = ("RETENTION_HOURS", "TTL_MS", "QUOTA_FACTOR", "FLUSH_INTERVAL_S",
                    "MAX_DEPTH", "WINDOW_SECONDS", "BACKOFF_JITTER_MS", "SHARD_COUNT",
                    "LEASE_SECONDS", "PREFETCH_DEPTH")


def probe_pool(seed: int) -> list[str]:
    """All distinct pool identifiers, in a seed-determined draw order."""
    pool = [f"NC_{b}_{t}" for b in PROBE_POOL_BASES for t in PROBE_POOL_TAILS]
    assert len(set(pool)) == len(pool) and "NC_PHANTOM_TTL_MS" in pool
    random.Random(seed).shuffle(pool)
    return pool


def draw_unique_probes(pool: list[str], used: set[str], span_texts: list[str],
                       transcript: str, limit: int = 5) -> list[str]:
    """Next `limit` pool identifiers that no accepted chain used yet (without
    replacement) and that occur nowhere in this session. Raises SystemExit when the
    pool is exhausted instead of silently reusing a probe."""
    out = []
    for cand in pool:
        if cand.lower() in used:
            continue
        rx = anchor_re(cand)
        if rx.search(transcript) or any(rx.search(x) for x in span_texts):
            continue
        out.append(cand)
        if len(out) >= limit:
            break
    if not out:
        raise SystemExit(f"--unique-probes: probe pool exhausted ({len(pool)} identifiers, "
                         f"{len(used)} used); extend PROBE_POOL_BASES/PROBE_POOL_TAILS")
    return out


RULES = """HARD RULES
- "question": standalone; a reader with no other context must understand it. Never
  write "the log", "the session above", "span 2". Ask about the artefact itself
  (file, flag, port, metric, error, decision, version, count).
- "canonical_short_answer": <= 30 words, the exact fact.
- "gold_anchor": ONE token copied verbatim from the excerpt that proves the answer:
  an identifier, path, flag, error code, version or number WITH its unit.
  Never a bare single digit, never a boolean, never a generic word, never a word
  that also occurs in the OTHER spans. It MUST be different from every other
  item's anchor. Copy it character-for-character from the window you quoted as
  evidence (no paraphrase, no case change) and prefer a token that occurs exactly
  ONCE in that span: an id, port, version, filename, flag, error code or value
  with a unit. A token that repeats 4+ times in the span is rejected as
  non-discriminating.
- "evidence": a CONTIGUOUS run of at least 8 words (typically 60+ characters)
  copied CHARACTER-FOR-CHARACTER from ONE place in the excerpt. If the fact is a
  short config line or a single value, extend the quote to the neighbouring line(s)
  of the SAME block until it has 8+ words. Never join distant lines with commas,
  never reformat YAML/JSON, never drop backticks, quotes or punctuation, never add
  a label like [HEAD].
- The question MUST NOT contain the gold_anchor token verbatim (or the answer is
  given away) and must not contain the answer text itself.
- Answers must be unambiguous, stated once, and not derivable from general
  knowledge. Prefer concrete engineering facts over opinions.
- slice=HEAD items must be grounded in the [HEAD] part of that span, slice=TAIL
  items in the [TAIL] part.
- Output JSON only. No prose, no markdown fence.
- Prefer facts whose surrounding text naturally carries 8+ words; if a candidate
  fact cannot be supported by an 8-word contiguous quote, pick another fact from
  the same half of the same span.
- Exactly 7 items, in this order: span 1 HEAD, span 1 TAIL, span 2 HEAD,
  span 2 TAIL, span 3 HEAD, span 3 TAIL, then the unanswerable item. No extras.
- The unanswerable item: "canonical_short_answer", "gold_anchor" and "evidence"
  MUST be the empty string "" (not null, not a placeholder).
- The unanswerable item's question must contain EXACTLY ONE concrete token (no
  version numbers, no second identifier): copy one [UNANSWERABLE STEMS] line
  verbatim. Those names were checked against this session and occur nowhere, so the
  probe gate passes; do not invent your own name (invented names keep turning out
  to exist in the session, which is why this item was rejected three times).
- Use a [SUGGESTED ANCHORS] token when one is listed for that span half: copy it
  exactly as the gold_anchor and quote its window ExYz verbatim as evidence. Those
  tokens are already verified to occur once in that span and never in the others.
- Every span block also lists [CANDIDATE EVIDENCE HEAD/TAIL] windows: each already
  has 12+ words and is verbatim. For each item, COPY ONE window of the correct
  span and half into "evidence" exactly, from its FIRST character to its LAST
  (the "(N words)" label states how many whitespace-separated tokens it has). Do
  not stop early, do not drop trailing words, do not add anything before or after
  it, and take the gold_anchor from inside that same window. A truncated copy is
  the single most common rejection: six words were returned when sixteen were
  listed (2026-10-25). Paths, flags and bare numbers each count as a word.
- Tool output in these spans carries grep-style line-number prefixes such as
  `/path/to/driver.py-79-`. They are part of the excerpt: if your window contains
  them, copy them too. Do not clean up, re-indent or re-wrap anything.
- Before you answer, COUNT the words in every "evidence" value. Fewer than 8 words
  means the item will be rejected and you will be asked again.

WORKED EXAMPLE (format only, not from this session)
Excerpt: "deploy.sh run --rm --publish 8081:8080 ghcr.io/acme/harness:v0.9.3 --workers 6 --timeout-ms 4500 in the CI rehearsal"
{"kind":"answerable","span_idx":1,"slice":"TAIL","question":"Which host port did the CI rehearsal publish for the harness container?","canonical_short_answer":"Host port 8081","gold_anchor":"8081:8080","evidence":"run --rm --publish 8081:8080 ghcr.io/acme/harness:v0.9.3 --workers 6 --timeout-ms 4500"}
(11 contiguous words, anchor absent from the question, anchor unique to that span.)
{"kind":"unanswerable","span_idx":0,"slice":"NONE","question":"What value does the harness --worker-shard-ttl flag default to?","canonical_short_answer":"","gold_anchor":"","evidence":""}
"""


def absent_block(chain: dict) -> str:
    toks = chain.get("absent_probes") or []
    if not toks:
        return "[VERIFIED-ABSENT TOKENS] (none available: invent one plain flag name)"
    stems = [f"What does {t} default to when it is unset?" for t in toks[:3]]
    return ("[VERIFIED-ABSENT TOKENS — each occurs nowhere in this session; use exactly "
            "one of them in the unanswerable question]\n" + ", ".join(toks) +
            "\n[UNANSWERABLE STEMS — copy ONE of these lines verbatim as the question; "
            "do not add, rename or rewrite any token in it]\n" + "\n".join(stems))


def forbidden_block(chain: dict) -> str:
    """--dedup-anchors only: anchors already used by another chain of this session.
    Empty (no prompt change) when the flag is off."""
    fb = chain.get("forbidden_anchors") or []
    if not fb:
        return ""
    return ("[ANCHORS ALREADY USED BY ANOTHER CHAIN OF THIS SESSION — never use any of "
            "these as gold_anchor]\n" + ", ".join(sorted(fb)) + "\n\n")


def chain_prompt(chain: dict, half_chars: int) -> str:
    excl = {a.lower() for a in chain.get("forbidden_anchors") or []} or None
    blocks = "\n\n".join(
        span_block(s, half_chars,
                   [o["text"] for o in chain["spans"] if o["span_idx"] != s["span_idx"]],
                   exclude=excl)
        for s in chain["spans"])
    return (
        "You write exam items about ONE long coding session that was compacted "
        f"{CHAIN_LEN} times in a row. Below are the three spans between those "
        "boundaries, oldest first; each span is shown as [HEAD] (start) and [TAIL] "
        "(end, adjacent to the compaction boundary).\n\n"
        f"{blocks}\n\n"
        f"Produce EXACTLY 7 items in one JSON object:\n"
        '- 6 "kind":"answerable": 2 per span, one with "slice":"HEAD", one with "slice":"TAIL";\n'
        '- 1 "kind":"unanswerable": a plausible SIBLING detail that looks like it '
        "belongs to this session (an adjacent CLI flag, port, metric, config key, "
        "symbol or error code) but is NOT mentioned anywhere above. Ask for the value "
        "of ONE token from [VERIFIED-ABSENT TOKENS]. Its canonical_short_answer, "
        "gold_anchor and evidence are empty strings.\n\n"
        f"{absent_block(chain)}\n\n"
        f"{forbidden_block(chain)}"
        f"{RULES}\n\n"
        'Schema: {"items":[{"kind":"answerable","span_idx":1,"slice":"HEAD",'
        '"question":"...","canonical_short_answer":"...","gold_anchor":"...",'
        '"evidence":"..."}, ..., {"kind":"unanswerable","span_idx":0,"slice":"NONE",'
        '"question":"...","canonical_short_answer":"","gold_anchor":"","evidence":""}]}\n\n'
        f"Session title: {chain['title'][:200] or chain['session_id']}"
    )


def repair_prompt(chain: dict, slots: list[dict], half_chars: int) -> str:
    """Re-ask only the slots that failed validation (one round = one retry per slot)."""
    asks, blocks = [], []
    for n, sl in enumerate(slots, 1):
        asks.append(f"[SLOT {n}] kind={sl['kind']} span_idx={sl['span_idx']} "
                    f"slice={sl['slice']} :: previous attempt rejected because: {sl['why']}")
    if any(sl["kind"] == "unanswerable" for sl in slots):
        blocks.append("(All three spans are shown: the token must occur in NONE of "
                      "them, and in none of the session's other turns.)\n"
                      + absent_block(chain))
    shown = sorted({sl["span_idx"] for sl in slots if sl["kind"] == "answerable"})
    excl = {a.lower() for a in chain.get("forbidden_anchors") or []} or None
    if excl:
        blocks.append(forbidden_block(chain).rstrip())
    for s in chain["spans"]:
        if s["span_idx"] in shown or any(sl["kind"] == "unanswerable" for sl in slots):
            blocks.append(span_block(s, half_chars,
                                     [o["text"] for o in chain["spans"]
                                      if o["span_idx"] != s["span_idx"]],
                                     exclude=excl))
    return (
        f"You previously wrote exam items that failed validation. Return EXACTLY "
        f"{len(slots)} item(s), one per slot, in the order listed, and obey every rule "
        "below literally. The most common failures are: evidence that is not a "
        "contiguous 8+ word quote (short config lines must be extended to the "
        "neighbouring lines of the same block, copied character-for-character), a "
        "gold_anchor that also occurs in another span, and a question that contains "
        "the gold_anchor.\n\n"
        + "\n".join(asks) + "\n\n"
        + "\n\n".join(blocks) + "\n\n" + RULES + "\n\n"
        'Reply JSON only: {"items":[{"kind":"answerable","span_idx":N,"slice":"HEAD|TAIL",'
        '"question":"...","canonical_short_answer":"...","gold_anchor":"...",'
        '"evidence":"..."}]}'
    )


# --------------------------------------------------------------- validation

def validate(item: dict, spans: list[dict], transcript: str) -> tuple[bool, list[str], dict]:
    """The 5 deterministic gates from the protocol (+1 soft slice report)."""
    why, checks = [], {}
    span_text = {s["span_idx"]: s["text"] for s in spans}
    q = norm_ws(item["question"])
    if len(q.split()) < 5:
        why.append("question_too_short")
    checks["question"] = len(q.split()) >= 5

    if item["kind"] == "unanswerable":
        probes = probe_tokens(item["question"])
        hits = []
        for t in probes:
            rx = anchor_re(t)
            in_chain = any(rx.search(txt) for txt in span_text.values())
            in_tr = bool(rx.search(transcript))
            if in_chain or in_tr:
                hits.append({"token": t, "in_chain": in_chain, "in_transcript": in_tr})
        checks["probe_tokens"] = len(probes) > 0
        checks["probe_zero_hit"] = not hits
        if not probes:
            why.append("no_probe_tokens")
        if hits:
            why.append("probe_hit:" + ",".join(h["token"] for h in hits[:3]))
        if item["gold_anchor"] or item["evidence"] or item["canonical_short_answer"]:
            why.append("unanswerable_must_be_empty_fields")
        return (not why), why, checks

    own = span_text.get(item["span_idx"])
    if own is None:
        return False, ["bad_span_idx"], checks
    ans, anchor, ev = norm_ws(item["canonical_short_answer"]), item["gold_anchor"].strip(), item["evidence"]
    if not ans:
        why.append("empty_answer")
    elif len(ans.split()) > MAX_Q_WORDS:
        why.append(f"answer_too_long:{len(ans.split())}w")
    checks["answer_len"] = bool(ans) and len(ans.split()) <= MAX_Q_WORDS

    # 1. evidence verbatim (whitespace-normalized, case-sensitive)
    checks["evidence_verbatim"] = has_verbatim(own, ev)
    checks["evidence_words"] = len(norm_ws(ev).split()) >= MIN_EVIDENCE_WORDS
    if not checks["evidence_verbatim"]:
        why.append("evidence_not_verbatim")
    if not checks["evidence_words"]:
        why.append(f"evidence_lt_8_words:{len(norm_ws(ev).split())}w")

    # 2. anchor form + presence in its own span
    checks["anchor_form"] = (len(anchor) >= 2 and not re.fullmatch(r"\d", anchor)
                             and anchor.lower() not in ANCHOR_BAD)
    if not checks["anchor_form"]:
        why.append(f"anchor_bad_form:{anchor!r}")
    rx = anchor_re(anchor) if anchor else None
    own_hits = len(rx.findall(own)) if rx else 0
    checks["anchor_in_span"] = own_hits > 0
    checks["anchor_unique_in_span"] = 0 < own_hits <= MAX_ANCHOR_HITS_OWN_SPAN
    if own_hits == 0:
        why.append("anchor_not_in_span")
    elif own_hits > MAX_ANCHOR_HITS_OWN_SPAN:
        why.append(f"anchor_{own_hits}x_in_span")

    # 3. anchor must discriminate this span from the other two
    others = [v for k, v in span_text.items() if k != item["span_idx"]]
    in_other = [n for n, txt in zip([k for k in span_text if k != item["span_idx"]], others)
                if rx and rx.search(txt)]
    checks["anchor_not_in_other_spans"] = not in_other
    if in_other:
        why.append("anchor_in_other_span:" + ",".join(map(str, in_other)))

    # 5. no answer leak in the question
    leaked = bool(anchor) and norm_ws(anchor).lower() in q.lower()
    checks["no_anchor_leak"] = not leaked
    if leaked:
        why.append("anchor_leaked_in_question")

    # soft (reported, not a rejection): does the evidence sit in the labelled half?
    head, tail = halves(own)
    checks["slice_evidence_match"] = has_verbatim(tail if item["slice"] == "TAIL" else head, ev)
    checks["slice"] = item["slice"]
    return (not why), why, checks


# ------------------------------------------------------------------- driver

def load_sessions() -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(SESS_DIR, "*.json"))):
        try:
            s = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            print(f"skip unreadable {f}: {e}")
            continue
        if s.get("sid") and len(s.get("boundaries") or []) >= CHAIN_LEN:
            out.append(s)
    out.sort(key=lambda s: s["sid"])
    return out


def transcript_of(s: dict) -> str:
    return "\n\n".join((e.get("text") or "") for e in s.get("events", []))


def qid_for(cid: str, span_idx: int, n: int) -> str:
    """qid embeds the chain id: two chains of one session share span_idx/n (their
    spans overlap by construction), and qid is the pairing key for judging and
    metrics -- a collision silently merges two different questions downstream."""
    return f"{cid}_{span_idx}_{n}"


def chain_id_for(sid8: str, chain_idx: int) -> str:
    return f"chain_{sid8}_c{chain_idx}"


def slots_plan() -> list[dict]:
    return ([{"kind": "answerable", "span_idx": k + 1, "slice": sl, "n": n + 1}
             for k in range(CHAIN_LEN) for n, sl in enumerate(("HEAD", "TAIL"))]
            + [{"kind": "unanswerable", "span_idx": 0, "slice": "NONE", "n": 1}])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0,
                    help="chains to build (0 = use --max-chains)")
    ap.add_argument("--max-chains", type=int, default=10, help="hard ceiling (default 10)")
    ap.add_argument("--model", default=QA_MODEL_DEFAULT, help="generator model (never the agent model)")
    ap.add_argument("--out", default=os.path.join(BASE, "chains.jsonl"))
    ap.add_argument("--manifest", default=os.path.join(BASE, "chains-manifest.json"))
    ap.add_argument("--spans-out", default="", help="optional span plan JSONL (works with --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="build spans only: no API, no writes")
    ap.add_argument("--segs", type=int, default=DEFAULT_SEGS, help="tail segments per span (default 8)")
    ap.add_argument("--half-chars", type=int, default=PROMPT_HALF_CHARS,
                    help="chars per HEAD/TAIL half shown to the generator")
    ap.add_argument("--budget-usd", type=float, default=0.30, help="hard spend cap for this run")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dedupe", dest="dedupe", action="store_true", default=True,
                    help="skip a chain whose 3 spans duplicate an already selected "
                         "chain byte-for-byte (imported forks of one session) [default]")
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    ap.add_argument("--save-raw", default="",
                    help="append every prompt/response (JSONL) for audit")
    ap.add_argument("--resume", action="store_true",
                    help="keep chains already in --out and skip them (append mode)")
    ap.add_argument("--no-selfcheck", action="store_true",
                    help="skip the segmentation equality assertion")
    ap.add_argument("--disjoint", action="store_true",
                    help="opt-in: chains of one session never share spans (boundary "
                         "blocks 0-2, 3-5, ...; a 2nd chain only with >= 6 boundaries)")
    ap.add_argument("--unique-probes", action="store_true",
                    help="opt-in: draw unanswerable probes without replacement from "
                         "a pool; exit if the pool is smaller than the request")
    ap.add_argument("--dedup-anchors", action="store_true",
                    help="opt-in: reject (and re-ask) an item whose gold_anchor is "
                         "already a gold_anchor of another chain of the same session")
    args = ap.parse_args()
    build_flags = {"disjoint": args.disjoint, "unique_probes": args.unique_probes,
                   "dedup_anchors": args.dedup_anchors}
    indep_on = any(build_flags.values())

    RAW_LOG["path"] = args.save_raw
    if args.save_raw:
        open(args.save_raw, "w").close()
    if args.model == AGENT_MODEL:
        raise SystemExit("refusing: the QA generator must not be the agent model")

    sessions = load_sessions()
    if not sessions:
        raise SystemExit(f"no sessions with >= {CHAIN_LEN} boundaries in {SESS_DIR}")
    probe_session = sessions[0]
    ok = selfcheck_span(session_spans(probe_session)[0], args.segs)
    print(f"segmentation/window identical to make_longreplay_dataset.py: "
          f"{'OK' if ok else 'MISMATCH'} ({probe_session['sid'][-8:]} span 1)")
    order = enumerate_chains(sessions, disjoint=args.disjoint)
    want = args.max_chains if args.limit <= 0 else min(args.limit, args.max_chains)
    plan = order

    print(f"sessions with >= {CHAIN_LEN} boundaries: {len(sessions)} "
          f"({', '.join(s['sid'][-8:] + ':' + str(len(s['boundaries'])) for s in sessions)})")
    print(f"chains available: {len(order)}, requested: {want}, model: {args.model}"
          f"{' (dry run)' if args.dry_run else ''}\n")

    key = ""
    if not args.dry_run:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise SystemExit("export OPENROUTER_API_KEY (generation is paid), or use --dry-run")

    spend = Spend(args.budget_usd)
    chains, manifest_chains, dropped, span_plan = [], [], [], []
    seen_chain_ids: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    carried = 0            # chains already in --out when resuming (manifest honesty)
    # independence state (only consulted when the matching opt-in flag is set)
    pool = probe_pool(args.seed) if args.unique_probes else []
    used_probes: dict[str, str] = {}                     # probe.lower() -> chain_id
    session_anchors: dict[str, dict[str, tuple[str, str]]] = {}  # sid -> a.lower() -> (a, cid)

    def remember_chain(c: dict):
        for q in c.get("qa", []):
            if q.get("answerable"):
                a = (q.get("gold_anchor") or "").strip()
                if a:
                    session_anchors.setdefault(c["session_id"], {}).setdefault(
                        a.lower(), (a, c["chain_id"]))
            else:
                for t in probe_tokens(q.get("question") or ""):
                    used_probes.setdefault(t.lower(), c["chain_id"])

    def independence_reject(it: dict, sid: str) -> list[str]:
        """Extra (opt-in) rejection reasons on top of validate()."""
        if it["kind"] == "unanswerable":
            if args.unique_probes:
                reused = [t for t in probe_tokens(it["question"]) if t.lower() in used_probes]
                if reused:
                    return [f"probe_reused:{reused[0]}@{used_probes[reused[0].lower()]}"]
            return []
        if args.dedup_anchors:
            a = (it.get("gold_anchor") or "").strip().lower()
            hit = session_anchors.get(sid, {}).get(a)
            if a and hit:
                return [f"anchor_dup_across_chains:{hit[0]}@{hit[1]}"]
        return []

    def write_outputs():
        """Persist after every chain: a network/API failure must not throw away work."""
        if args.dry_run:
            return
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "qa_model": args.model, "agent_model": AGENT_MODEL,
            "protocol": {"chain_len": CHAIN_LEN, "items_per_span": ITEMS_PER_SPAN,
                         "tail_segments": args.segs, "token_budget": TOKEN_BUDGET,
                         "chars_per_token": CHARS_PER_TOKEN, "seed": args.seed,
                         "price_in_per_token": PRICE_IN, "price_out_per_token": PRICE_OUT},
            "sessions_eligible": [{"sid": s["sid"], "boundaries": len(s["boundaries"])}
                                  for s in sessions],
            # chains_requested is CUMULATIVE (carried over + this run): a resume run
            # with --limit 1 used to report chains_requested=1 next to chains_ok=2,
            # which read as a bug (2026-10-25).
            "chains_requested": carried + want,
            "chains_requested_this_run": want,
            "chains_carried_over": carried,
            "chains_available": len(order),
            "chains": manifest_chains, "dropped": dropped,
            "totals": {"chains_ok": len(chains), "chains_dropped": len(dropped),
                       "items": sum(len(c["qa"]) for c in chains),
                       "api_calls": spend.calls, "spend_usd": round(spend.spent, 5),
                       "budget_usd": args.budget_usd},
        }
        if indep_on:      # additive key, absent in the default build
            per_sess: dict[str, list[str]] = {}
            for c in chains:
                per_sess.setdefault(c["session_id"], []).append(c["chain_id"])
            manifest["independence"] = dict(
                build_flags,
                distinct_sessions=len(per_sess),
                chains_per_session=per_sess,
                probe_pool_size=len(pool) if args.unique_probes else None,
                probes_used=sorted(used_probes) if args.unique_probes else None,
            )
        with open(args.out, "w") as w:
            for c in chains:
                w.write(json.dumps(c, ensure_ascii=False) + "\n")
        json.dump(manifest, open(args.manifest, "w"), ensure_ascii=False, indent=1)

    def record_drop(entry: dict):
        """One entry per chain in the report: a retried chain replaces its earlier
        failure instead of listing every attempt."""
        dropped[:] = [x for x in dropped if x["chain_id"] != entry["chain_id"]]
        dropped.append(entry)

    if args.resume and not args.dry_run:
        if os.path.exists(args.out):
            chains = [json.loads(l) for l in open(args.out) if l.strip()]
            carried = len(chains)
            for c in chains:
                seen_chain_ids[c["chain_id"]] = c["session_id"]
                fingerprints[span_fp(c["spans"])] = c["chain_id"]
                remember_chain(c)
            print(f"resume: {len(chains)} chain(s) already in {args.out}")
        if os.path.exists(args.manifest):
            try:
                old = json.load(open(args.manifest))
                manifest_chains = [m for m in old.get("chains", [])
                                   if m["chain_id"] in seen_chain_ids]
                dropped = [d for d in old.get("dropped", [])
                           if d["chain_id"] not in seen_chain_ids]
                prior = old.get("totals") or {}
                spend.spent = float(prior.get("spend_usd") or 0.0)
                spend.calls = int(prior.get("api_calls") or 0)
                if spend.spent:
                    print(f"resume: carrying ${spend.spent:.5f} / {spend.calls} calls "
                          f"of spend from the previous run (global cap ${spend.budget:.2f})")
            except Exception:  # noqa: BLE001 - a broken manifest must not block resume
                manifest_chains, dropped = [], []

    if args.unique_probes:
        free = len([p for p in pool if p.lower() not in used_probes])
        if free < want:
            raise SystemExit(f"--unique-probes: pool has {free} unused identifier(s) "
                             f"({len(pool)} total, {len(used_probes)} used) but {want} "
                             f"unanswerable(s) requested; extend PROBE_POOL_* or lower --limit")
        print(f"unique probes: pool {len(pool)}, unused {free}, requested {want}")

    attempted = 0
    for pos, (si, ci) in enumerate(plan, 1):
        s = sessions[si]
        if attempted >= want:
            print(f"reached the chain budget ({want}); stopping")
            break
        sid8 = s["sid"][-8:]
        cid = chain_id_for(sid8, ci)
        if cid in seen_chain_ids:
            if args.resume:
                print(f"  {cid} already present, skipped")
                continue
            cid = f"{cid}_{s['sid'][-12:]}"   # collision guard, reported below
            print(f"  ! chain_id collision on {sid8}; using {cid}")
        if args.resume and any(d["chain_id"] == cid
                               and (d["reason"].startswith("unrecoverable_slots")
                                    or d["reason"].startswith("duplicate_content_of"))
                               for d in dropped):
            # a chain that already burned all repair rounds twice (e.g. span full of
            # JSON-in-JSON quotes the generator cannot copy verbatim) is skipped on
            # resume instead of paying for a third identical fight (2026-10-25:
            # chain_w20P3or1_c0 slot 3/HEAD failed 6 rounds across 2 runs); a chain
            # already recorded as a byte-for-byte duplicate is likewise not re-paid.
            print(f"  {cid} skipped: previously dropped ({next(d['reason'] for d in dropped if d['chain_id'] == cid)[:60]}), not retrying")
            continue
        seen_chain_ids[cid] = s["sid"]

        chain = build_chain(s, ci, args.segs, check=not args.no_selfcheck)
        spans = chain["spans"]
        fp = span_fp(spans)
        if args.dedupe and fp in fingerprints:
            record_drop({"chain_id": cid, "session_id": s["sid"], "chain_idx": ci,
                         "reason": f"duplicate_content_of:{fingerprints[fp]}"})
            print(f"  {cid} skipped: span content identical to {fingerprints[fp]}")
            continue
        fingerprints[fp] = cid
        attempted += 1
        head = (f"[{attempted}/{want}] {cid} session={s['sid']} "
                f"boundaries {ci}..{ci + CHAIN_LEN - 1}")
        if args.dry_run:
            print(head)
            for sp in spans:
                print(f"    span {sp['span_idx']} lag={sp['lag']} tail≤{args.segs} segs: "
                      f"{len(sp['segments'])} events, {sp['chars']} chars "
                      f"~{sp['est_tokens']} tok, {sp['dropped_segments']} segs dropped")
            span_plan.append({"chain_id": cid, "session_id": s["sid"], "chain_idx": ci,
                              "spans": [{k: v for k, v in sp.items() if k != "text"}
                                        for sp in spans],
                              "reasons": chain["reasons"]})
            if chain["reasons"]:
                print(f"    ! {', '.join(chain['reasons'])}")
            continue

        if chain["reasons"]:
            record_drop({"chain_id": cid, "session_id": s["sid"], "chain_idx": ci,
                         "reason": ";".join(chain["reasons"])})
            print(f"{head}\n  dropped before generation: {', '.join(chain['reasons'])}")
            continue

        print(head)
        for sp in spans:
            print(f"    span {sp['span_idx']} lag={sp['lag']}: {sp['chars']} chars "
                  f"~{sp['est_tokens']} tok")
        if args.unique_probes:
            chain["absent_probes"] = draw_unique_probes(
                pool, set(used_probes), [sp["text"] for sp in spans], transcript_of(s))
        if args.dedup_anchors:
            chain["forbidden_anchors"] = sorted(
                a for a, _ in session_anchors.get(s["sid"], {}).values())
        indep_rejections = []
        predicted_in = sum(sp["chars"] for sp in spans) + 2500
        est = predicted_in / CHARS_PER_TOKEN * PRICE_IN + 2000 * PRICE_OUT + 0.004
        raw, _ = llm(chain_prompt(chain, args.half_chars), key, args.model, spend,
                     max_tokens=10000, est_cost=est, temperature=args.temperature)
        attempts_raw = [raw]
        items = parse_items(raw)
        by_slot = {}
        for it in items:
            if it["kind"] == "unanswerable":
                by_slot.setdefault("unanswerable", []).append(it)
            elif it.get("span_idx") in (1, 2, 3):
                sl = (it.get("slice") or "").upper()
                if sl in ("HEAD", "TAIL"):
                    by_slot.setdefault((int(it["span_idx"]), sl), []).append(it)

        accepted, slot_reports, rejection_counts = {}, [], {}
        failed_now = False      # NOT `any(d.chain_id == cid for d in dropped)`: resume
                                # keeps earlier attempts' drops, and that made a chain
                                # that finally passed look dropped (lost 2026-10-25).
        transcript = transcript_of(s)
        # ---- round 0: consume the batch, then up to MAX_RETRIES_PER_SLOT repairs
        pending = slots_plan()
        for rnd in range(MAX_RETRIES_PER_SLOT + 1):
            still = []
            for slot in pending:
                k = "unanswerable" if slot["kind"] == "unanswerable" else (slot["span_idx"], slot["slice"])
                cand = (by_slot.pop(k, []) if rnd == 0 else by_slot.pop(k, [])) or []
                if isinstance(cand, dict):
                    cand = [cand]
                ok_item, why_all = None, []
                for it in cand:
                    it = dict(it)
                    if it["kind"] == "answerable" and not it.get("slice"):
                        it["slice"] = slot["slice"]
                    it["span_idx"] = 0 if it["kind"] == "unanswerable" else slot["span_idx"]
                    ok, why, checks = validate(it, spans, transcript)
                    if ok and indep_on:
                        extra = independence_reject(it, s["sid"])
                        if extra:
                            ok, why = False, extra
                            indep_rejections.append({"slot": str(k), "round": rnd,
                                                     "reason": extra[0]})
                    slot_reports.append({"slot": k, "round": rnd, "ok": ok, "why": why,
                                         "checks": checks})
                    if ok:
                        ok_item = (it, checks)
                        break
                    why_all = why
                if ok_item:
                    accepted[k] = ok_item
                else:
                    for w in why_all:
                        rejection_counts[w.split(":")[0]] = rejection_counts.get(w.split(":")[0], 0) + 1
                    if not cand:
                        why_all = why_all or ["missing_slot"]
                        rejection_counts["missing_slot"] = rejection_counts.get("missing_slot", 0) + 1
                    still.append(dict(slot, why="; ".join(why_all) or "missing_slot"))
            if not still or rnd == MAX_RETRIES_PER_SLOT:
                if still:
                    failed_now = True
                    record_drop({"chain_id": cid, "session_id": s["sid"], "chain_idx": ci,
                                 "reason": "unrecoverable_slots:" +
                                           ",".join(f"{x['kind']}@{x['span_idx']}/{x['slice']}"
                                                    for x in still),
                                 "raw_response": attempts_raw[-1][:4000]})
                    print(f"  DROPPED after {rnd} repair round(s): "
                          + "; ".join(f"{x['slot'] if 'slot' in x else x['span_idx']}"
                                      f"{x['why'][:80]}" for x in still))
                break
            # within-chain anchor uniqueness (protocol: every item's gold_anchor
            # MUST differ): a later item that reused an earlier item's anchor is
            # pulled out of `accepted` and re-asked in the next repair round.
            seen_anchor: dict[str, str] = {}
            dups = []
            for slot in pending:
                k = ("unanswerable" if slot["kind"] == "unanswerable"
                     else (slot["span_idx"], slot["slice"]))
                if k not in accepted or accepted[k][0]["kind"] != "answerable":
                    continue
                a = accepted[k][0]["gold_anchor"].strip().lower()
                if a and a in seen_anchor:
                    del accepted[k]
                    why = "anchor_duplicated_within_chain"
                    dups.append(f"{k}={a}")
                    still.append(dict(slot, why=why))
                elif a:
                    seen_anchor[a] = k
            if dups:
                rejection_counts["anchor_duplicated_within_chain"] = \
                    rejection_counts.get("anchor_duplicated_within_chain", 0) + len(dups)
                print(f"  anchor reuse inside the chain -> re-asking: {', '.join(dups)}")
            print(f"  repair round {rnd + 1}: {len(still)} slot(s) rejected -> re-asking")
            for x in still:
                print(f"    rejected {x['kind']}@{x['span_idx']}/{x['slice']}: "
                      f"{x['why'][:160]}")
            raw, _ = llm(repair_prompt(chain, still, args.half_chars), key, args.model, spend,
                         max_tokens=4000, est_cost=0.012, temperature=args.temperature)
            attempts_raw.append(raw)
            # map repairs back to slots: exact (kind, span, slice) first, then
            # leftover items fill leftover slots in the order the slots were asked
            by_slot, left = {}, list(still)
            for it in parse_items(raw):
                k = ("unanswerable" if it["kind"] == "unanswerable"
                     else (it.get("span_idx"), (it.get("slice") or "").upper()))
                match = next((sl for sl in left
                              if ("unanswerable" if sl["kind"] == "unanswerable"
                                  else (sl["span_idx"], sl["slice"])) == k), None)
                if match is None:
                    same_kind = [sl for sl in left
                                 if (sl["kind"] == "unanswerable")
                                 == (it["kind"] == "unanswerable")]
                    match = same_kind[0] if same_kind else (left[0] if left else None)
                if match is None:
                    break
                left.remove(match)
                slot_key = ("unanswerable" if match["kind"] == "unanswerable"
                            else (match["span_idx"], match["slice"]))
                by_slot.setdefault(slot_key, []).append(it)
            pending = still

        if failed_now:
            if indep_rejections:
                dropped[-1]["independence_rejections"] = indep_rejections
            write_outputs()
            continue

        dropped[:] = [x for x in dropped if x["chain_id"] != cid]   # this attempt passed

        qa = []
        for slot in slots_plan():
            k = "unanswerable" if slot["kind"] == "unanswerable" else (slot["span_idx"], slot["slice"])
            it, checks = accepted[k]
            qa.append({
                "qid": qid_for(cid, slot["span_idx"], slot["n"]),
                "chain_id": cid, "session_id": s["sid"],
                "span_idx": slot["span_idx"],
                "lag": (CHAIN_LEN - slot["span_idx"] + 1) if slot["span_idx"] else 0,
                "slice": slot["slice"],
                "answerable": it["kind"] == "answerable",
                "must_abstain": it["kind"] != "answerable",
                "question": it["question"],
                "canonical_short_answer": it["canonical_short_answer"] if it["kind"] == "answerable" else "",
                "gold_anchor": it["gold_anchor"] if it["kind"] == "answerable" else "",
                "evidence": it["evidence"] if it["kind"] == "answerable" else "",
                "checks": checks,
            })
        # rates are over the FINAL accepted answerable items: the slot_reports of
        # round 0 include the unanswerable slot (no anchor/evidence checks) in the
        # denominator and exclude items accepted in a repair round, which made a
        # 12/12-clean dataset report anchor_pass_rate 0.80/0.833 (2026-10-25).
        checks_flat = [q["checks"] for q in qa if q["answerable"]]
        rate = lambda f: round(sum(1 for c in checks_flat if c.get(f)) / max(1, len(checks_flat)), 3)  # noqa: E731
        manifest_chains.append({
            "chain_id": cid, "session_id": s["sid"], "chain_idx": ci, "valid": True,
            "spans": [{"span_idx": sp["span_idx"], "lag": sp["lag"], "chars": sp["chars"],
                       "est_tokens": sp["est_tokens"], "events": len(sp["segments"]),
                       "dropped_segments": sp["dropped_segments"]} for sp in spans],
            "qa_count": len(qa),
            "slot_rounds": [{"slot": str(r["slot"]), "round": r["round"], "ok": r["ok"],
                             "why": r["why"]} for r in slot_reports],
            "anchor_pass_rate": rate("anchor_in_span"),
            "anchor_discrimination_rate": rate("anchor_not_in_other_spans"),
            "evidence_verbatim_rate": rate("evidence_verbatim"),
            "no_leak_rate": rate("no_anchor_leak"),
            "slice_evidence_match_rate": rate("slice_evidence_match"),
            "rejections": rejection_counts,
        })
        if indep_on:
            manifest_chains[-1]["independence_rejections"] = indep_rejections
        chains.append({
            "chain_id": cid, "session_id": s["sid"], "chain_idx": ci,
            "spans": [{"span_idx": sp["span_idx"], "lag": sp["lag"],
                       "segments": sp["segments"], "chars": sp["chars"],
                       "est_tokens": sp["est_tokens"]} for sp in spans],
            "qa": qa, "agent_model": AGENT_MODEL, "qa_model": args.model,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "valid": True,
        })
        if indep_on:      # additive field, absent in the default build
            chains[-1]["build_flags"] = dict(build_flags)
        remember_chain(chains[-1])
        print(f"  accepted {len(qa)}/7 items; spend so far ${spend.spent:.4f}")
        write_outputs()

    if args.dry_run:
        print(f"\ndry run: {len(span_plan)} chains built, no API calls, no files written")
        print(f"projected spend for generation: ~${len(span_plan) * 0.012:.3f} "
              f"(cap ${args.budget_usd:.2f})")
        if args.spans_out:
            with open(args.spans_out, "w") as w:
                for r in span_plan:
                    w.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"span plan -> {args.spans_out}")
        return 0

    write_outputs()
    print(f"\nchains: {args.out} ({len(chains)} chains, "
          f"{sum(len(c['qa']) for c in chains)} items)")
    print(f"manifest: {args.manifest} ({len(dropped)} dropped)")
    print(f"spend: ${spend.spent:.5f} over {spend.calls} calls (cap ${args.budget_usd:.2f})")
    for d in dropped:
        print(f"  dropped {d['chain_id']}: {d['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
