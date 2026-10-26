#!/usr/bin/env python3
"""Import Claude Code transcripts (~/.claude/projects/<proj>/<uuid>.jsonl) into longsession/1.

Transcript structure (learned from a live transcript's key/type layout, 2026-10; content is
never copied into this repo). One JSON object per line; the types that matter:

  user      {uuid, parentUuid, isSidechain, isMeta?, isCompactSummary?, timestamp, sessionId,
             message:{role:"user", content: str | [ {type:"text",text} |
                      {type:"tool_result", tool_use_id, content: str | [{type:"text",text}|{type:"image"}],
                       is_error?} ]}}
  assistant {uuid, parentUuid, isSidechain, timestamp, message:{id, role:"assistant",
             content:[ {type:"text"} | {type:"thinking"} | {type:"tool_use", id, name, input} ]}}
             -- ONE API message is split over several lines (one per content block, same message.id)
  system    {subtype:"compact_boundary", compactMetadata:{trigger:"auto"|"manual", preTokens},
             parentUuid:null, logicalParentUuid:<last uuid before the compaction>}   (also
             stop_hook_summary & co: ignored)
  attachment / queue-operation / last-prompt / ai-title / summary / file-history-snapshot ...: ignored
             (a legacy {"type":"summary"} line is a TITLE, not a compaction summary)
  Compaction: the compact_boundary line, then a user line with isCompactSummary=true whose text is
  "This session is being continued from a previous conversation ... Summary: ...". The transcript
  keeps the FULL pre-compaction history, so the dropped messages are all present.
  Subagents live in <session>/subagents/agent-*.jsonl (isSidechain=true, own parent chain).

Mapping to longsession/1
  * Ordering: follow parentUuid (logicalParentUuid across a compact boundary) from the LAST
    message back to the root, so rewound/forked dead branches are dropped (counted in
    meta.dead_branch_entries). A broken link falls back to file order. ``--order file`` = plain
    file order.
  * Resumed sessions (several files sharing uuids) are merged into ONE session (union by uuid,
    files ordered by first timestamp); ``--session-per-file`` disables the merge.
  * messages: user text -> user; assistant text blocks (same message.id, consecutive) -> one
    assistant message; thinking blocks dropped; tool_use + its tool_result -> ONE ``tool`` message
    in the import_opencode format ``[tool NAME] in: {args<=300ch}\\n[out] {output<=600ch}``
    (emitted where the result appears; an unanswered tool_use gets ``[out] (no result)``).
    isMeta user lines and ``<system-reminder>`` blocks are removed (``--include-meta`` /
    ``--keep-system-reminders``). Sidechains/subagents are EXCLUDED unless ``--include-sidechains``
    (then merged by timestamp with a ``[subagent ID]`` prefix).
  * compactions: one per compact_boundary; ``at_idx`` = number of messages before it,
    ``original_summary`` = the following isCompactSummary text (kept in full), ``auto`` and
    ``pre_tokens`` from compactMetadata.
  * REDACTION is ON by default (ls_redact.py: sk-, sk-ant-, sk-or-v1, ghp_, glpat-, AWS keys, secret-keyed high-entropy assignments,
    PRIVATE KEY blocks). The per-kind count report is printed and
    stored in meta.redaction. ``--no-redact`` turns it off (the output is then as private as the
    transcript). Review the report, keep the output out of git.
  * No questions: run gen_longsession_qa.py on the output.

Usage
  python3 import_claude_code.py PATH [PATH...] --out OUT_DIR [--min-tokens 0] [--limit N]
  PATH = transcript .jsonl, or a project dir / ~/.claude/projects (searched for *.jsonl)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from longsession import (FORMAT, SID_RE, count_tokens, message_tokens, save_session,  # noqa: E402
                         validate_session)
from ls_redact import format_report, redact_session  # noqa: E402

MSG_TYPES = ("user", "assistant")
ARGS_CAP = 300
OUT_CAP = 600
SUMMARY_LEAD = "This session is being continued from a previous conversation"
_SYS_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>\s*", re.S)


# ---------------------------------------------------------------- reading
def read_entries(path: str) -> tuple[list[dict], int]:
    out, bad = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(j, dict):
                j["_file"], j["_line"] = path, i
                out.append(j)
    return out, bad


def discover(paths: list[str], include_subagents: bool = False) -> list[str]:
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, fns in os.walk(p):
                if not include_subagents:
                    dirs[:] = [d for d in dirs if d != "subagents"]
                files += [os.path.join(root, f) for f in fns if f.endswith(".jsonl")]
        else:
            files.append(p)
    return sorted(set(files))


def _first_ts(entries: list[dict]) -> str:
    return next((e["timestamp"] for e in entries if isinstance(e.get("timestamp"), str)), "")


def group_files(file_entries: dict[str, list[dict]]) -> list[list[str]]:
    """Union files that share any uuid (a resumed session copies the earlier history)."""
    parent = {f: f for f in file_entries}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[str, str] = {}
    for f, es in file_entries.items():
        for e in es:
            u = e.get("uuid")
            if isinstance(u, str) and e.get("type") in MSG_TYPES and not e.get("isSidechain"):
                if u in owner:
                    parent[find(f)] = find(owner[u])
                else:
                    owner[u] = f
    groups: dict[str, list[str]] = {}
    for f in file_entries:
        groups.setdefault(find(f), []).append(f)
    return [sorted(g, key=lambda f: (_first_ts(file_entries[f]), f)) for g in groups.values()]


# ---------------------------------------------------------------- ordering
def order_entries(entries: list[dict], mode: str = "parent") -> tuple[list[dict], int]:
    """(ordered main-chain entries, #dead-branch message entries). `entries` are in file order,
    deduped by uuid, sidechains already removed."""
    carrier = [e for e in entries if e.get("type") in MSG_TYPES
               or (e.get("type") == "system" and e.get("subtype") == "compact_boundary")]
    if mode == "file" or not carrier:
        return carrier, 0
    pos = {id(e): i for i, e in enumerate(entries)}
    nodes = {e["uuid"]: e for e in entries if isinstance(e.get("uuid"), str)}
    leaf = carrier[-1]
    path: list[dict] = []
    seen: set[int] = set()
    cur = leaf
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        path.append(cur)
        par = cur.get("parentUuid") or cur.get("logicalParentUuid")
        if par is None:
            cur = None
        elif par in nodes:
            cur = nodes[par]
        else:   # broken link: continue from the nearest earlier unvisited carrier in file order
            earlier = [e for e in carrier if pos[id(e)] < pos[id(cur)] and id(e) not in seen]
            cur = earlier[-1] if earlier else None
    path.reverse()
    keep = {id(e) for e in carrier}
    path = [e for e in path if id(e) in keep]
    # Parallel tool calls: every tool_result hangs off ITS OWN tool_use entry, so all but the last
    # are side leaves of the walk. Rescue them (not real dead branches): a user entry made only of
    # tool_result blocks whose tool_use ids were issued by an assistant entry on the path.
    used = {b.get("id") for e in path if e.get("type") == "assistant"
            for b in ((e.get("message") or {}).get("content") or []) if isinstance(b, dict)
            and b.get("type") == "tool_use"}
    chosen = {id(e) for e in path}
    for e in carrier:
        if id(e) in chosen or e.get("type") != "user":
            continue
        c = (e.get("message") or {}).get("content")
        if isinstance(c, list) and c and all(isinstance(b, dict) and b.get("type") == "tool_result"
                                              and b.get("tool_use_id") in used for b in c):
            i = max((k for k, x in enumerate(path) if pos[id(x)] < pos[id(e)]), default=-1)
            path.insert(i + 1, e)
            chosen.add(id(e))
    dead = sum(1 for e in carrier if id(e) not in chosen)
    return path, dead


# ---------------------------------------------------------------- text helpers
def _clean(text: str, keep_reminders: bool) -> str:
    return (text if keep_reminders else _SYS_REMINDER.sub("", text)).strip()


def _block_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if isinstance(b, dict):
            if b.get("type") == "text":
                parts.append(str(b.get("text") or ""))
            elif b.get("type") == "image":
                parts.append("[image]")
    return "\n".join(p for p in parts if p)


def tool_text(name: str, inp, out: str, is_error: bool = False) -> str:
    brief = json.dumps(inp, ensure_ascii=False)[:ARGS_CAP]
    body = (("[error] " if is_error else "") + (out or ""))[:OUT_CAP]
    return f"[tool {name or '?'}] in: {brief}\n[out] {body}"


# ---------------------------------------------------------------- conversion
def build_session(entries: list[dict], *, files: list[str], include_sidechains=False,
                  include_meta=False, keep_reminders=False, order="parent",
                  sid: str | None = None) -> dict:
    stats = Counter()
    title = ""
    main, side = [], []
    for e in entries:
        t = e.get("type")
        if t in ("summary", "ai-title") and not title:
            title = str(e.get("summary") or e.get("aiTitle") or e.get("title") or "")[:120]
        if e.get("isSidechain"):
            stats["sidechain_entries"] += 1
            if include_sidechains and t in MSG_TYPES:
                side.append(e)
            continue
        main.append(e)
    ordered, dead = order_entries(main, order)
    if side:
        merged = sorted(ordered + side, key=lambda e: str(e.get("timestamp") or ""))
        ordered = merged
    msgs: list[dict] = []
    comps: list[dict] = []
    pending: dict[str, tuple[str, object]] = {}
    last_mid: str | None = None

    def push(role: str, text: str, e: dict, mid: str | None = None):
        nonlocal last_mid
        if not text:
            return
        prefix = f"[subagent {str(e.get('agentId') or '?')[:8]}] " if e.get("isSidechain") else ""
        m = {"idx": len(msgs), "role": role, "text": prefix + text}
        if isinstance(e.get("timestamp"), str):
            m["ts"] = e["timestamp"]
        if role == "assistant" and mid and msgs and msgs[-1]["role"] == "assistant" \
                and last_mid == mid:
            msgs[-1]["text"] += "\n" + text
            return
        msgs.append(m)
        last_mid = mid if role == "assistant" else None

    for e in ordered:
        t = e.get("type")
        if t == "system":
            cm = e.get("compactMetadata") or {}
            comps.append({"at_idx": len(msgs), "original_summary": None,
                          "auto": cm.get("trigger") == "auto", "pre_tokens": cm.get("preTokens")})
            continue
        msg = e.get("message") or {}
        content = msg.get("content")
        if t == "user":
            if e.get("isCompactSummary") or (isinstance(content, str) and content.startswith(SUMMARY_LEAD)) \
                    or (isinstance(content, list) and _block_text(content).startswith(SUMMARY_LEAD)):
                summ = _block_text(content)
                if comps and comps[-1]["original_summary"] is None and comps[-1]["at_idx"] == len(msgs):
                    comps[-1]["original_summary"] = summ
                else:
                    comps.append({"at_idx": len(msgs), "original_summary": summ, "auto": None,
                                  "pre_tokens": None})
                continue
            if e.get("isMeta") and not include_meta:
                stats["meta_skipped"] += 1
                continue
            if isinstance(content, str):
                push("user", _clean(content, keep_reminders), e)
                continue
            texts = []
            for b in content or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    name, inp = pending.pop(b.get("tool_use_id"), ("?", {}))
                    out = _clean(_block_text(b.get("content")), keep_reminders)
                    push("tool", tool_text(name, inp, out, bool(b.get("is_error"))), e)
                elif b.get("type") == "text":
                    texts.append(str(b.get("text") or ""))
                elif b.get("type") == "image":
                    texts.append("[image]")
            push("user", _clean("\n".join(texts), keep_reminders), e)
        elif t == "assistant":
            mid = msg.get("id") if isinstance(msg.get("id"), str) else None
            for b in content if isinstance(content, list) else []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    push("assistant", _clean(str(b.get("text") or ""), keep_reminders), e, mid)
                elif b.get("type") == "tool_use":
                    pending[b.get("id")] = (b.get("name"), b.get("input"))
                elif b.get("type") == "thinking":
                    stats["thinking_dropped"] += 1
    for name, inp in pending.values():          # tool_use that never got a result
        push("tool", tool_text(name, inp, "(no result)"), {})
    # a boundary at_idx beyond the messages (compaction at the very end) is clamped/dropped
    n = len(msgs)
    comps = [c for c in comps if 0 < c["at_idx"] <= n]
    seen_at, uniq = set(), []
    for c in comps:
        if c["at_idx"] in seen_at:                  # two boundaries with no message in between
            if uniq and uniq[-1]["original_summary"] is None:
                uniq[-1]["original_summary"] = c["original_summary"]
            continue
        seen_at.add(c["at_idx"])
        uniq.append(c)
    last_file = files[-1]
    sess_id = next((e.get("sessionId") for e in reversed(entries) if e.get("sessionId")), "")
    stem = os.path.splitext(os.path.basename(last_file))[0]
    sid = sid or re.sub(r"[^A-Za-z0-9._-]", "-", f"cc-{(sess_id or stem)[:8]}")[:64]
    if not title:
        title = next((m["text"] for m in msgs if m["role"] == "user"), "")
    title = " ".join(title.split())[:80]
    return {"format": FORMAT, "sid": sid, "source": "claude_code", "title": title,
            "messages": msgs, "compactions": uniq, "questions": [],
            "meta": {"files": [os.path.basename(f) for f in files],
                     "dead_branch_entries": dead,
                     "sidechain_entries": stats["sidechain_entries"],
                     "sidechains_included": bool(include_sidechains),
                     "meta_skipped": stats["meta_skipped"],
                     "thinking_dropped": stats["thinking_dropped"],
                     "unanswered_tool_use": len(pending), "order": order,
                     "n_compactions": len(uniq),
                     "n_summaries": sum(1 for c in uniq if c["original_summary"])}}


def import_paths(paths: list[str], *, redact=True, session_per_file=False, **kw) -> list[dict]:
    files = discover(paths, include_subagents=kw.get("include_sidechains", False))
    fe: dict[str, list[dict]] = {}
    bad = 0
    for f in files:
        es, b = read_entries(f)
        bad += b
        if any(e.get("type") in MSG_TYPES for e in es):
            fe[f] = es
    groups = [[f] for f in sorted(fe)] if session_per_file else group_files(fe)
    out = []
    for g in groups:
        seen, merged = set(), []
        for f in g:
            for e in fe[f]:
                u = e.get("uuid")
                if isinstance(u, str):
                    if u in seen:
                        continue
                    seen.add(u)
                merged.append(e)
        if not any(e.get("type") in MSG_TYPES and not e.get("isSidechain") for e in merged):
            continue
        s = build_session(merged, files=g, **kw)
        s["meta"]["bad_lines"] = bad
        if redact:
            redact_session(s)
        else:
            s["meta"]["redaction"] = {"enabled": False, "counts": {}, "total": 0}
        out.append(s)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-redact", action="store_true", help="DISABLE secret redaction")
    ap.add_argument("--include-sidechains", action="store_true")
    ap.add_argument("--include-meta", action="store_true")
    ap.add_argument("--keep-system-reminders", action="store_true")
    ap.add_argument("--order", choices=("parent", "file"), default="parent")
    ap.add_argument("--session-per-file", action="store_true")
    ap.add_argument("--min-tokens", type=int, default=0, help="skip sessions shorter than this")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    sessions = import_paths(a.paths, redact=not a.no_redact, session_per_file=a.session_per_file,
                            include_sidechains=a.include_sidechains, include_meta=a.include_meta,
                            keep_reminders=a.keep_system_reminders, order=a.order)
    n_out = 0
    total = Counter()
    for s in sessions:
        toks = sum(message_tokens(s["messages"]))
        if toks < a.min_tokens or validate_session(s):
            print(f"[skip] {s['sid']}: {toks} tokens {validate_session(s)[:2]}", file=sys.stderr)
            continue
        if a.limit and n_out >= a.limit:
            break
        save_session(s, a.out)
        n_out += 1
        red = Counter(s["meta"]["redaction"]["counts"])
        total.update(red)
        print(f"{s['sid']}: {len(s['messages'])} msgs ~{toks} tok, {len(s['compactions'])} compactions "
              f"({s['meta']['n_summaries']} with summary), files={len(s['meta']['files'])}, "
              f"dead={s['meta']['dead_branch_entries']}, sidechain_entries={s['meta']['sidechain_entries']}; "
              + (format_report(red) if not a.no_redact else "redaction OFF"))
    print(f"imported {n_out} sessions -> {a.out}; "
          + (format_report(total) if not a.no_redact else "REDACTION DISABLED (--no-redact)"))
    return 0 if n_out else 1


if __name__ == "__main__":
    sys.exit(main())
