#!/usr/bin/env python3
"""Per-record answer-arm visible context (the post-C3 state the model answered from).

Fixes the defect from the gate-212839 forensics: exported judge records carried no
transcript, so the hallucination post-pass ran degenerate ("everything unsupported
becomes unverifiable"). This module rebuilds, from the archived pi session JSONL,
exactly what the answer-arm could see when it answered:

  baseline       C3 (last) compaction summary + post-compaction messages
  nc             C3 note + the injected <memory> recall block + post-compaction messages
  nc-note-only   C3 note + post-compaction messages (no recall, by protocol)

Sources: runs/<run>/sessions/<chain_id>/asks/<arm>/<qid>/*.jsonl (pi session JSONL:
`compaction` entries carry the summary; the final user message carries the question
and -- for nc -- the injected `<memory>` block). The question text and the model's
own answer are deliberately EXCLUDED: visible_context must stay untainted by the
question so "anchor absent from context" can expose question-primed reconstruction.

Order: compaction/note FIRST, then recall, then post-compaction messages; line-level
dedup; total char-capped (~12k default).

Used by export_judge_input.py (writer) and metrics_v2.py semantics via the exported
`visible_context` field; importable for one-off forensics.
"""
from __future__ import annotations

import glob
import json
import os
import re

MEM_RE = re.compile(r"<memory>(.*?)</memory>", re.S)
DEFAULT_CAP = 12000


def message_text(msg: dict) -> str:
    """Plain text of a pi message (text parts only, thinking dropped)."""
    parts = msg.get("content")
    if isinstance(parts, list):
        return "\n".join(p.get("text") or "" for p in parts
                         if isinstance(p, dict) and p.get("type") == "text")
    return str(parts or "")


def load_entries(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def find_ask_session(sessions_root: str, chain_id: str, arm: str, qid: str):
    """sessions/<chain>/asks/<arm>/<qid>/*.jsonl -> deterministic first match."""
    pat = os.path.join(sessions_root, str(chain_id), "asks", str(arm), str(qid), "*.jsonl")
    files = sorted(glob.glob(pat))
    return files[0] if files else None


def parse_ask_session(path: str, arm: str, recall_arms: tuple = ("nc",)) -> dict:
    """Split an ask session into (C3 summary, recall block, post-compaction messages).

    Returns dict with:
      summary            last compaction entry's summary text ("" when none)
      recall_block       injected <memory> block (" when absent or arm not in recall_arms)
      recall_in_session  whether the session carried a <memory> block at all
      post_messages      [(role, text)] after the last compaction, excluding the
                         final question message and the assistant answer after it
    """
    entries = load_entries(path)
    comp_idx = [i for i, e in enumerate(entries) if e.get("type") == "compaction"]
    last_comp = comp_idx[-1] if comp_idx else -1
    summary = (entries[last_comp].get("summary") or "") if last_comp >= 0 else ""

    # last user message after the final compaction == the ask (question [+ recall])
    tail = entries[last_comp + 1:]
    ask_i = None
    for i, e in enumerate(tail):
        if e.get("type") == "message" and (e.get("message") or {}).get("role") == "user":
            ask_i = i
    ask_txt = message_text((tail[ask_i]["message"] if ask_i is not None else {}))

    m = MEM_RE.search(ask_txt)
    recall_in_session = m is not None
    recall_block = ""
    if m is not None and arm in recall_arms:
        recall_block = (m.group(1) or "").strip()

    post = []
    for i, e in enumerate(tail):
        if e.get("type") != "message" or i == ask_i:
            continue
        if ask_i is not None and i > ask_i:
            break                      # assistant answer after the question: stop
        msg = e.get("message") or {}
        txt = message_text(msg).strip()
        if txt:
            post.append((msg.get("role") or "?", txt))
    return {"summary": summary, "recall_block": recall_block,
            "recall_in_session": recall_in_session, "post_messages": post,
            "n_compactions": len(comp_idx)}


def assemble(summary: str, recall: str, post_messages, cap: int = DEFAULT_CAP):
    """compaction/note FIRST, then recall, then post-messages; dedup lines; cap chars."""
    seen: set[str] = set()
    out: list[str] = []
    total, truncated, dropped = 0, False, []

    def add_block(name: str, text: str):
        nonlocal total, truncated
        if truncated or not text:
            if text:
                dropped.append(name)
            return
        for line in text.splitlines():
            key = line.strip()
            if key and key in seen:
                continue
            need = len(line) + (1 if out else 0)
            if total + need > cap:
                truncated = True
                dropped.append(name)
                return
            if key:
                seen.add(key)
            out.append(line)
            total += need

    add_block("compaction_summary", summary or "")
    add_block("recall_block", recall or "")
    add_block("post_messages", "\n".join(f"[{role}] {txt}" for role, txt in post_messages))
    return "\n".join(out), {"chars": total, "truncated": truncated, "dropped": dropped}


def build_visible_context(sessions_root: str, chain_id: str, arm: str, qid: str,
                          cap: int = DEFAULT_CAP) -> tuple[str | None, dict]:
    """(visible_context, meta). visible_context is None when no ask session exists."""
    path = find_ask_session(sessions_root, chain_id, arm, qid)
    meta: dict = {"ask_session": None, "cap": cap}
    if path is None:
        meta["error"] = "ask_session_not_found"
        return None, meta
    meta["ask_session"] = os.path.relpath(path, sessions_root)
    p = parse_ask_session(path, arm)
    text, a = assemble(p["summary"], p["recall_block"], p["post_messages"], cap=cap)
    meta.update({
        "n_compactions": p["n_compactions"],
        "has_compaction_summary": bool(p["summary"]),
        "compaction_summary_chars": len(p["summary"]),
        "recall_block_in_session": p["recall_in_session"],
        "recall_in_context": bool(p["recall_block"]),
        "recall_chars": len(p["recall_block"]),
        "n_post_messages": len(p["post_messages"]),
        "post_chars": sum(len(t) for _, t in p["post_messages"]),
        "visible_context_chars": a["chars"],
        "truncated": a["truncated"],
        "dropped": a["dropped"],
    })
    return text, meta
