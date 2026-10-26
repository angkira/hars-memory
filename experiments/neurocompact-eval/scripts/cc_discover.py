#!/usr/bin/env python3
"""Find Claude Code transcripts (~/.claude/projects/*/*.jsonl) by title / first message / session id.

Used by `home_run.py cc --find TEXT | --session-id ID | --list`; also runnable alone:

  python3 cc_discover.py --list [--root ~/.claude/projects] [--top 15]
  python3 cc_discover.py --find "HARS Experts Research"
  python3 cc_discover.py --session-id 03023d8a-4fa0-4ff8-8345-a67624798daf     (cse_/session_ prefix ok)

Record types used (structure only, learned from real transcripts; no content is stored here):
  user / assistant                 messages (isSidechain=true -> subagent, not counted; isMeta=true skipped
                                   for the first-message title); message.content is str or a block list
  system{subtype:compact_boundary} one per compaction (compactMetadata.trigger auto|manual)
  user{isCompactSummary:true}      the summary Claude Code wrote after a boundary
  summary{summary}                 legacy title record; ai-title{aiTitle}, custom-title{customTitle}
                                   (the /rename title), last-prompt{lastPrompt}: titles
  every other type (attachment, queue-operation, file-history-snapshot, cost-state, ...) is ignored.
A session is identified by the file stem (= sessionId) and by `sessionId` / `uuid` fields inside the records.

The scan is line-oriented and parses JSON only for the few record types it needs, so a multi-hundred-MB
transcript takes about a second.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

DEFAULT_ROOT = "~/.claude/projects"
TITLE_KEYS = ("summary", "aiTitle", "customTitle", "title", "lastPrompt")
_TS = re.compile(r'"timestamp"\s*:\s*"([^"]+)"')
_TYPE = re.compile(r'"type"\s*:\s*"([A-Za-z_-]+)"')
_SIDE = re.compile(r'"isSidechain"\s*:\s*true')
_FIRST_USER_LIMIT = 3          # first N user-text messages are searchable by --find
_TEXT_CAP = 2000


def norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def norm_id(s: str) -> str:
    s = (s or "").strip()
    for p in ("cse_", "session_", "cse-", "session-"):
        if s.lower().startswith(p):
            s = s[len(p):]
    return s.lower()


def _user_text(rec: dict) -> str:
    c = (rec.get("message") or {}).get("content")
    if isinstance(c, str):
        return c
    parts = []
    for b in c or []:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(str(b.get("text") or ""))
    return "\n".join(parts)


def scan_session(path: str, session_id: str | None = None) -> dict:
    """Facts about one transcript file. `session_id` (normalised) is looked for in the raw lines."""
    want = norm_id(session_id) if session_id else ""
    info = {"path": path, "file_id": os.path.splitext(os.path.basename(path))[0],
            "project": os.path.basename(os.path.dirname(path)), "size": os.path.getsize(path),
            "first_ts": "", "last_ts": "", "messages": 0, "user_messages": 0, "compactions": 0,
            "compact_summaries": 0, "titles": [], "first_user": [], "session_ids": [], "id_hit": ""}
    sids: set[str] = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            m = _TYPE.search(line)
            typ = m.group(1) if m else ""
            tm = _TS.search(line)
            if tm:
                if not info["first_ts"]:
                    info["first_ts"] = tm.group(1)
                info["last_ts"] = tm.group(1)
            if want and not info["id_hit"] and want in line:
                info["id_hit"] = "record"
            side = bool(_SIDE.search(line))
            if typ in ("user", "assistant"):
                if side:
                    continue
                info["messages"] += 1
                if typ == "user":
                    info["user_messages"] += 1
                    if "isCompactSummary" in line and re.search(r'"isCompactSummary"\s*:\s*true', line):
                        info["compact_summaries"] += 1
                        continue
                    if len(info["first_user"]) < _FIRST_USER_LIMIT and not re.search(r'"isMeta"\s*:\s*true', line):
                        try:
                            t = _user_text(json.loads(line)).strip()
                        except ValueError:
                            t = ""
                        if t and not t.startswith(("<command-", "<local-command", "Caveat:")):
                            info["first_user"].append(t[:_TEXT_CAP])
                continue
            if typ == "system":
                if "compact_boundary" in line:
                    info["compactions"] += 1
                continue
            if typ in ("summary", "ai-title", "custom-title", "last-prompt", "title", "agent-name"):
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                for k in TITLE_KEYS + ("agentName",):
                    v = rec.get(k)
                    if isinstance(v, str) and v.strip() and v.strip() not in info["titles"]:
                        info["titles"].append(v.strip()[:_TEXT_CAP])
            if len(sids) < 8:
                sm = re.search(r'"sessionId"\s*:\s*"([^"]+)"', line)
                if sm:
                    sids.add(sm.group(1))
    info["session_ids"] = sorted(sids)
    if want:
        if norm_id(info["file_id"]) == want or (len(want) >= 8 and norm_id(info["file_id"]).startswith(want)):
            info["id_hit"] = "file name"
        elif any(norm_id(s) == want or (len(want) >= 8 and norm_id(s).startswith(want)) for s in sids):
            info["id_hit"] = "sessionId"
    return info


def list_files(root: str) -> list[str]:
    """<root>/*/*.jsonl (project dirs), or root itself when it is a project dir / a single file.
    subagents/ directories are never entered."""
    root = os.path.expanduser(root)
    if os.path.isfile(root):
        return [root]
    out = []
    if not os.path.isdir(root):
        return out
    for e in sorted(os.listdir(root)):
        p = os.path.join(root, e)
        if e.endswith(".jsonl") and os.path.isfile(p):
            out.append(p)
        elif os.path.isdir(p):
            for g in sorted(os.listdir(p)):
                if g.endswith(".jsonl") and os.path.isfile(os.path.join(p, g)):
                    out.append(os.path.join(p, g))
    return out


def match_find(info: dict, needle: str) -> str:
    """Where `needle` (case/space-insensitive substring) occurs: title | first message | file name | ''."""
    n = norm(needle)
    if not n:
        return ""
    if any(n in norm(t) for t in info["titles"]):
        return "title"
    if any(n in norm(t) for t in info["first_user"]):
        return "first message"
    if n in norm(info["file_id"]) or n in norm(info["project"]):
        return "file/project name"
    return ""


def discover(root: str = DEFAULT_ROOT, *, find: str | None = None, session_id: str | None = None,
             list_top: int = 0, scan=scan_session) -> list[dict]:
    """Candidate sessions. find / session_id filter (ranked: id hit, title, first message, name);
    neither -> the `list_top` sessions with the most compactions, then the largest."""
    out = []
    for p in list_files(root):
        try:
            info = scan(p, session_id) if session_id else scan(p)
        except OSError:
            continue
        if session_id:
            if not info["id_hit"]:
                continue
            info["matched"] = f"session id ({info['id_hit']})"
        elif find:
            w = match_find(info, find)
            if not w:
                continue
            info["matched"] = w
        else:
            info["matched"] = ""
        out.append(info)
    if find or session_id:
        rank = {"session id (file name)": 0, "session id (sessionId)": 0, "title": 1, "first message": 2}
        out.sort(key=lambda i: (rank.get(i["matched"], 3), -i["compactions"], -i["size"]))
    else:
        out.sort(key=lambda i: (-i["compactions"], -i["size"]))
        if list_top:
            out = out[:list_top]
    return out


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024
    return str(n)


def label(info: dict, width: int = 64) -> str:
    t = (info["titles"] or info["first_user"] or ["(no title)"])[0]
    return " ".join(t.split())[:width]


def format_candidates(cands: list[dict]) -> str:
    if not cands:
        return "(none)"
    rows = ["  #  session      project                       size     dates                    msgs  compact  title / first message"]
    for i, c in enumerate(cands, 1):
        d0, d1 = c["first_ts"][:10], c["last_ts"][:10]
        rows.append(f" {i:>2}  {c['file_id'][:11]:<11}  {c['project'][-28:]:<28}  {human(c['size']):>7}  "
                    f"{d0}..{d1}  {c['messages']:>5}  {c['compactions']:>7}  {label(c)}"
                    + (f"   [match: {c['matched']}]" if c.get("matched") else ""))
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--find", help="substring of the session title / first user message")
    g.add_argument("--session-id", help="sessionId or message uuid (cse_/session_ prefix accepted)")
    g.add_argument("--list", action="store_true", help="longest sessions with compaction counts")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)
    cands = discover(a.root, find=a.find, session_id=a.session_id, list_top=a.top if a.list else 0)
    print(format_candidates(cands))
    return 0 if cands else 1


if __name__ == "__main__":
    sys.exit(main())
