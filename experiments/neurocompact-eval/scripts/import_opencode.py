#!/usr/bin/env python3
"""Import OpenCode sessions (sqlite part-store) and reconstruct linear histories
with compaction boundaries marked.

Where the compaction summary actually lives (verified, see probe_compaction.py):
  * `part.data` with `type=="compaction"` carries ONLY the marker
    {auto, overflow, tail_start_id} — no text at all.
  * The summary is an assistant message in `message` with
    `data.mode == "compaction"`, `data.summary == true`,
    `data.agent == "compaction"`, whose `data.parentID` is the id of the
    message that owns the compaction marker part. Its `text` parts concatenate
    to the summary.
  * `session_message` (type='compaction') additionally stores the same artifact
    as `data.summary` with `status`/`reason`/`recent`/`time`; used only as a
    time-matched fallback when the summary message has no text parts.

The old heuristic ("first assistant text part after the marker") is wrong for
10/107 markers: when a compaction produced no text, it silently returned the
NEXT ordinary assistant turn (a few hundred to a few thousand chars of the
model's narration, or even a whole serialized transcript).

Boundary detection (idx / auto) is unchanged: idx is the event index of the
marker part, auto is part.data.auto.

Output: longreplay/sessions/<sesid>.json
  { sid, title, n_msgs,
    events: [ {role: user|assistant|tool|compaction, text, ts} ],   # linear order
    boundaries: [ {idx, auto, original_summary, original_summary_chars,
                   orig_summary_valid, orig_summary_source, orig_summary_reject} ] }

Usage: python3 import_opencode.py [--min-span-chars 8000] [--show-shortest 5]
"""
import sqlite3, os, json, argparse, sys, glob, re

DB = os.path.expanduser("~/.local/share/opencode/opencode.db")
STORE = os.path.expanduser("~/.local/share/opencode/storage/session")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "longreplay", "sessions")
SUMMARY_CAP = 12000     # chars kept from the real compaction summary
TITLE_MAX = 80          # fallback title = first user message, this many chars
MIN_VALID_CHARS = 600   # shorter than this is never a usable comprehensive summary
SM_TIME_TOL_MS = 10_000  # session_message fallback: max |delta| vs the marker

# --- validity heuristic ------------------------------------------------------
# A usable compaction summary is a structured stand-in for the dropped window:
# section headers ("## Objective" etc.) or a multi-bullet digest. What is NOT a
# summary: an assistant's step narration ("Let me ... I'll ... Now ..."), and a
# transcript echo ([Command] / [Tool result] / tool-call tokens). Both are
# observed failure modes of OpenCode's auto-compaction (see probe_extract.py).
PREAMBLE_RE = re.compile(
    r"^(let me|let's|let us|i'll|i will|i need|i should|i'm |i am |now\b|next,|"
    r"looking at|wait\b|good\b|ok\b|okay\b|sure\b|here's|here is|"
    r"продолжаю|итак|теперь|сейчас|хорошо)", re.I)
TOOLLOG_RE = re.compile(
    r"(\[Command\]|\[tool |\[Tool result\]|\[Assistant tool call\]|\[Assistant reasoning\]|"
    r"<\|tool_calls\|>|<\|.*?\|>|<｜.*?｜>|Exited with code)", re.I)
HEADER_RE = re.compile(r"^#{1,4}\s+\S", re.M)
BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+\S", re.M)
SECTION_WORD_RE = re.compile(
    r"\b(Objective|Summary|Key Findings|Important Details|Next Steps|Current State|"
    r"Decisions|Open Questions|Constraints|Context|Status)\b")


def summary_valid(text: str) -> tuple[bool, str]:
    """Is this text a summary of the dropped window, or assistant noise?"""
    t = (text or "").strip()
    if not t:
        return False, "missing"
    if len(t) < MIN_VALID_CHARS:
        return False, f"short(<{MIN_VALID_CHARS})"
    if TOOLLOG_RE.search(t):
        return False, "tool_log"
    if PREAMBLE_RE.match(t):
        return False, "preamble"
    structured = (bool(HEADER_RE.search(t))
                  or len(BULLET_RE.findall(t)) >= 3
                  or bool(SECTION_WORD_RE.search(t)))
    if not structured:
        return False, "no_structure"
    # a preamble that also carries headers is still a failed summary
    head = t[:400]
    if PREAMBLE_RE.match(head) and not HEADER_RE.search(head):
        return False, "preamble"
    return True, "ok"


def usable_title(t) -> bool:
    """OpenCode writes 'New session - <ts>' until it auto-titles; not a title."""
    t = (t or "").strip()
    return bool(t) and not t.startswith("New session")


def resolve_title(con, sid: str, first_user: str) -> str:
    """Title from OpenCode storage json, else the sqlite session row, else the
    first user message. Storage only holds a few sessions; sqlite covers the rest."""
    for p in glob.glob(os.path.join(STORE, "*", f"{sid}.json")):
        try:
            t = json.load(open(p)).get("title")
        except Exception:
            t = None
        if usable_title(t):
            return t.strip()
    try:
        row = con.execute("SELECT title FROM session WHERE id = ?", (sid,)).fetchone()
    except sqlite3.Error:
        row = None
    if row and usable_title(row[0]):
        return row[0].strip()
    return " ".join((first_user or "").split())[:TITLE_MAX]


def fetch_parts(cur):
    # part.data has {type, text?, ...}; order within a message by time_created then id
    rows = cur.execute("""
        SELECT p.session_id, p.data, p.time_created, p.id, m.data as mdata, m.id
        FROM part p JOIN message m ON m.id = p.message_id
        ORDER BY p.session_id, m.time_created, p.time_created, p.id
    """)
    for sid, pdata, tcreated, pid, mdata, mid in rows:
        yield sid, json.loads(pdata), tcreated, pid, json.loads(mdata), mid


def load_compaction_summaries(cur, sid: str):
    """Canonical summary lookup for one session.

    Returns (by_parent, sm_rows):
      by_parent: {marker_message_id: summary text} from the compaction-summary
                 assistant message (mode/summary/agent == compaction).
      sm_rows:   [(time_created_ms, summary)] from session_message type='compaction',
                 used only as a fallback for aborted compactions with no text.
    """
    inner = cur.connection.cursor()
    by_parent = {}
    for mid, mdata in list(cur.execute(
            "SELECT id, data FROM message WHERE session_id=? ORDER BY time_created, id", (sid,))):
        m = json.loads(mdata)
        if not (m.get("mode") == "compaction" or m.get("summary") is True):
            continue
        if m.get("summary") is not True or m.get("role") != "assistant":
            continue
        txt = "".join(
            (json.loads(p).get("text") or "")
            for (p,) in inner.execute(
                "SELECT data FROM part WHERE message_id=? ORDER BY time_created, id", (mid,))
            if json.loads(p).get("type") == "text")
        err = m.get("error") or {}
        note = ""
        if err:
            detail = (err.get("data") or {}).get("message") or ""
            note = f"{err.get('name', 'error')}: {str(detail)[:120]}"
        by_parent[m.get("parentID")] = {"text": txt, "note": note}
    sm_rows = []
    for (d,) in list(cur.execute(
            "SELECT data FROM session_message WHERE session_id=? AND type='compaction' "
            "ORDER BY seq", (sid,))):
        j = json.loads(d or "{}")
        t = (j.get("time") or {}).get("created")
        sm_rows.append((t, j.get("summary") or "", j.get("reason") or ""))
    return by_parent, sm_rows


def extract_summary(marker_mid: str, marker_tc: int, by_parent: dict, sm_rows: list):
    """(text, source, note). Prefers the exact parent-linked summary message."""
    pm = by_parent.get(marker_mid)
    if pm and pm["text"].strip():
        return pm["text"], "compaction_message", ""
    note = (pm or {}).get("note") or "no summary text stored"
    if sm_rows and marker_tc:
        cand = min(sm_rows, key=lambda r: abs((r[0] or 0) - marker_tc))
        if cand[1].strip() and abs((cand[0] or 0) - marker_tc) <= SM_TIME_TOL_MS:
            return cand[1], "session_message", f"reason={cand[2]}"
    return "", "none", note


def part_text(d):
    t = d.get("type")
    if t == "text":
        return d.get("text", "")
    if t == "tool":
        st = d.get("state", {}) or {}
        inp = st.get("input") or {}
        out = st.get("output") or ""
        tool = d.get("tool") or "?"
        brief = json.dumps(inp, ensure_ascii=False)[:300]
        return f"[tool {tool}] in: {brief}\n[out] {str(out)[:600]}"
    if t == "reasoning":
        return ""
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-span-chars", type=int, default=8000)
    ap.add_argument("--show-shortest", type=int, default=5,
                    help="print the N shortest extracted summaries (suspicious ones)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cur = con.cursor()

    sessions = {}   # sid -> {title, events, boundaries}
    marker_at = {}  # (sid, event index) -> {mid, mtc}   (marker message link)
    for sid, d, tc, pid, mdata, mid in fetch_parts(cur):
        s = sessions.setdefault(sid, {"sid": sid, "title": "", "events": [], "boundaries": []})
        t = d.get("type")
        role = mdata.get("role", "?")
        if t == "compaction":
            marker_at[(sid, len(s["events"]))] = {
                "mid": mid, "mtc": (mdata.get("time") or {}).get("created") or tc}
            s["events"].append({"role": "compaction", "text": "", "ts": tc,
                                "auto": bool(d.get("auto"))})
            continue
        if t == "text":
            txt = d.get("text", "").strip()
            if not txt:
                continue
            if role == "user" and "summary" in mdata and isinstance(mdata["summary"], dict):
                # user message meta (title/diff summary) — skip, not session content
                continue
            if role == "user" and not s.get("first_user"):
                s["first_user"] = txt
            s["events"].append({"role": role, "text": txt, "ts": tc})
        elif t == "tool":
            txt = part_text(d)
            if txt.strip():
                s["events"].append({"role": "tool", "text": txt, "ts": tc})

    kept = 0
    all_bounds = []          # every boundary kept across sessions (for stats)
    for sid, s in sessions.items():
        ev = s["events"]
        if not ev:
            continue
        by_parent, sm_rows = load_compaction_summaries(cur, sid)
        # boundaries: idx = event index of the marker, auto = part.data.auto
        # (UNCHANGED); the summary now comes from the compaction-summary message.
        bounds = []
        for i, e in enumerate(ev):
            if e["role"] != "compaction":
                continue
            mk = marker_at.get((sid, i), {})
            text, src, note = extract_summary(mk.get("mid"), mk.get("mtc"), by_parent, sm_rows)
            valid, why = summary_valid(text)
            bounds.append({"idx": i, "auto": e.get("auto"),
                           "original_summary": text[:SUMMARY_CAP],
                           "original_summary_chars": len(text),
                           "orig_summary_valid": valid,
                           "orig_summary_source": src,
                           "orig_summary_reject": "" if valid else why,
                           "orig_summary_note": note})
        # span sizes (chars between boundaries)
        spans = []
        prev = 0
        for b in bounds + [{"idx": len(ev)}]:
            spans.append(sum(len(x["text"]) for x in ev[prev:b["idx"]]))
            prev = b["idx"]
        # NOTE: no longer require a non-empty summary here — a boundary whose
        # compaction was aborted is still a valid boundary (the baseline arm can
        # run on it); it is marked orig_summary_valid=false instead of dropped.
        ok = [b for b, sp in zip(bounds, spans[:-1]) if sp >= args.min_span_chars]
        if not ok:
            continue
        s["boundaries"] = ok
        s["span_chars"] = spans[:-1]
        all_bounds.extend((sid, b) for b in ok)
        # title: OpenCode storage json -> sqlite session row -> first user message
        s["title"] = resolve_title(con, sid, s.pop("first_user", ""))
        out = os.path.join(OUT, f"{sid}.json")
        json.dump(s, open(out, "w"), ensure_ascii=False, indent=1)
        kept += 1

    valid = [b for _sid, b in all_bounds if b["orig_summary_valid"]]
    srcs = {}
    rejs = {}
    for _sid, b in all_bounds:
        srcs[b["orig_summary_source"]] = srcs.get(b["orig_summary_source"], 0) + 1
        if not b["orig_summary_valid"]:
            rejs[b["orig_summary_reject"]] = rejs.get(b["orig_summary_reject"], 0) + 1
    print(f"imported {kept} sessions with usable boundaries -> {OUT}")
    print(f"boundaries: {len(all_bounds)}  orig_summary_valid: {len(valid)}  "
          f"invalid: {len(all_bounds) - len(valid)}")
    print(f"summary sources: {json.dumps(srcs, sort_keys=True)}")
    if rejs:
        print(f"reject reasons:  {json.dumps(rejs, sort_keys=True)}")
    if args.show_shortest:
        print(f"\n{args.show_shortest} shortest NON-EMPTY summaries (suspicious):")
        nonempty = [sb for sb in all_bounds if sb[1]["original_summary_chars"] > 0]
        for sid, b in sorted(nonempty, key=lambda sb: sb[1]["original_summary_chars"])[:args.show_shortest]:
            prev = " ".join((b["original_summary"] or "").split())[:120]
            print(f"  {b['original_summary_chars']:6d} valid={str(b['orig_summary_valid']):5s} "
                  f"{b['orig_summary_reject'] or '-':11s} {sid[-10:]}b{b['idx']}: {prev!r}")
        empty = [sb for sb in all_bounds if sb[1]["original_summary_chars"] == 0]
        print(f"\n{len(empty)} boundaries with NO stored summary text:")
        for sid, b in sorted(empty, key=lambda sb: (sb[0], sb[1]["idx"])):
            print(f"  {sid[-10:]}b{b['idx']:<5d} auto={str(b['auto']):5s} {b['orig_summary_note'][:110]}")


if __name__ == "__main__":
    main()
