#!/usr/bin/env python3
"""Import OpenHands SWE-bench trajectories as long coding-agent sessions.

Public replacement for the 5 private OpenCode sessions (protocol issue B4: metrics_v2
needs >= 8 DISTINCT sessions). Emits EXACTLY the session structure that
import_opencode.py writes, so make_chains.py / make_longreplay_dataset.py /
verify_chains.py consume it unchanged:

  longreplay-oh/sessions/<sid>.json
    { sid, title, events: [ {role: user|assistant|tool|compaction, text, ts} ],
      boundaries: [ {idx, auto, original_summary, original_summary_chars,
                     orig_summary_valid, orig_summary_source, orig_summary_reject,
                     orig_summary_note} ],
      span_chars: [...],
      + extras (ignored downstream): instance_id, repo, source, import }

Source format (OpenHands trajs in the public swe-bench-submissions bucket): JSON list of
OpenAI-chat messages (system / user / assistant[tool_calls] / tool[tool_call_id]).
Input is UNTRUSTED DATA: it is only json-parsed here, never executed.

Mapping
  * system prompt                   dropped (shared OpenHands boilerplate).
  * first user message              the OpenHands task template (phases 1-8, "do not
                                    modify tests", ...) is stripped; only the
                                    <issue_description> body is kept as the user event.
  * assistant text                  -> assistant event (empty text dropped).
  * assistant tool call + its tool  -> ONE tool event in import_opencode's part_text()
    message                            format:  "[tool NAME] in: <json input, capped>\\n[out] <output>"
                                    tool names/keys are mapped to the OpenCode vocabulary
                                    (execute_bash->bash, str_replace_editor view/create/
                                    str_replace/insert->read/write/edit).
  * `think` tool calls              dropped (the analogue of OpenCode `reasoning` parts,
                                    which import_opencode also drops).
  * tool output boilerplate         "[The command completed with exit code N.]",
                                    "[Python interpreter: ...]", the repeated
                                    "[Current working directory: <repo root>]", and the
                                    editor's "Review the changes ..." tail are removed; the
                                    exit code is kept as "[exit N]".
  * long outputs                    capped like OpenCode (out[:600]) but head+tail with an
                                    elision marker, because the tail of a traceback is where
                                    the error is.  Caps: --in-cap (default 400),
                                    --out-cap (default 1200).  OpenCode used 300/600.

Compaction boundaries.  These sessions contain NO real compactions.  Synthetic boundaries
(`auto: true`, `synthetic: true`, original_summary "" and orig_summary_valid false) are inserted
so that every span the chain builder replays is exactly the replay window it computes:
make_chains/make_longreplay_dataset replay the last DEFAULT_SEGS (8) segments of ~CHUNK
(3500) chars of each span, capped at TOKEN_BUDGET (20000 tok at 2.4 chars/tok = 48k chars).
Spans 2..N are therefore cut after exactly `--span-segs` (8) segments (nothing dropped, nothing
unreplayed), using the same greedy turn-edge packing as split_segments().  Span 1 starts at
the beginning of the session; the head is a lead-in outside the replay window, so the first
replayed window starts roughly `--start-frac` (default 0.25) into the session instead of in
the initial repo exploration.  `--n-boundaries` (default 3 = one chain per session; use 6 with
make_chains --disjoint for two non-overlapping chains).

The baseline (in-context) summary is NOT taken from the session: make_chains never reads
`original_summary`, and run_chainreplay pops NC_ORIG_SUMMARY_FILE and has the pi
summarizer generate every arm's summary itself.  The empty original_summary is by design.

Usage:
  python3 -I import_openhands.py --out-dir DIR --src FILE [FILE ...]
  python3 -I import_openhands.py --out-dir DIR --manifest selection.json --src-dir TRAJ_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from make_longreplay_dataset import CHUNK, DEFAULT_SEGS, TOKEN_BUDGET, CHARS_PER_TOKEN  # noqa: E402

IN_CAP = 400
OUT_CAP = 1200
START_FRAC = 0.25
N_BOUNDARIES = 3

ISSUE_RE = re.compile(r"<issue_description>\n?(.*?)\n?</issue_description>", re.S)
EXIT_RE = re.compile(r"\n?\[Command finished with exit code (-?\d+)\]\s*$")
COMPLETED_RE = re.compile(r"\n?\[The command completed with exit code -?\d+\.\]")
INTERP_RE = re.compile(r"\n?\[Python interpreter: [^\]]*\]")
CWD_RE = re.compile(r"\n?\[Current working directory: ([^\]]*)\]")
EDIT_TAIL_RE = re.compile(r"\s*Review the changes and make sure they are as expected.*$", re.S)


# --------------------------------------------------------------------- helpers

def clip(s: str, cap: int) -> str:
    """Head+tail truncation to ~cap chars (60/40) with an elision marker."""
    s = s or ""
    if cap <= 0 or len(s) <= cap:
        return s
    head = int(cap * 0.6)
    tail = cap - head
    return f"{s[:head]}\n...[{len(s) - cap} chars elided]...\n{s[-tail:]}"


def msg_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type", "text") == "text")
    return ""


def repo_of(instance_id: str) -> str:
    """'django__django-11099' -> 'django/django'; 'sphinx-doc__sphinx-8595' -> 'sphinx-doc/sphinx'."""
    owner, _, rest = instance_id.partition("__")
    name = re.sub(r"-\d+$", "", rest)
    return f"{owner}/{name}"


def sid_of(instance_id: str) -> str:
    """Hyphen-free session id: make_chains uses sid[-8:] in chain ids and the judge masks
    '-nc-'/'-baseline-' substrings of session ids, so hyphens are avoided."""
    return instance_id.replace("-", "_")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def ensure_not_tracked_path(path: str, is_dir: bool = False) -> None:
    """Refuse to write data inside the git work tree unless git ignores the path."""
    p = os.path.abspath(path)
    d = p if os.path.isdir(p) else os.path.dirname(p)
    while d and not os.path.isdir(d):
        d = os.path.dirname(d)
    try:
        top = subprocess.run(["git", "-C", d, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return
    if top.returncode != 0:
        return                      # not inside a work tree
    ig = subprocess.run(["git", "-C", d, "check-ignore", "-q", "--no-index",
                         p + "/" if is_dir else p],
                        capture_output=True, timeout=10)
    if ig.returncode != 0:
        raise SystemExit(f"refusing to write data to {p}: inside git work tree {top.stdout.strip()} "
                         f"and not gitignored (data must stay out of git)")


# ------------------------------------------------------------- tool rendering

def _clean_bash_output(out: str, repo_root: str) -> str:
    exit_code = None
    m = EXIT_RE.search(out)
    if m:
        exit_code = m.group(1)
        out = out[:m.start()]
    out = COMPLETED_RE.sub("", out)
    out = INTERP_RE.sub("", out)
    cwd = None

    def _cwd(mm):
        nonlocal cwd
        cwd = mm.group(1).strip()
        return ""
    out = CWD_RE.sub(_cwd, out).rstrip()
    tail = []
    if cwd and cwd.rstrip("/") != repo_root.rstrip("/"):
        tail.append(f"[cwd {cwd}]")
    if exit_code is not None:
        tail.append(f"[exit {exit_code}]")
    return out + ("\n" + " ".join(tail) if tail else "")


def render_call(name: str, args: dict) -> tuple[str, dict]:
    """(opencode tool name, opencode-style input dict)."""
    if name == "execute_bash":
        d = {"command": args.get("command", "")}
        if args.get("is_input"):
            d["is_input"] = args["is_input"]
        return "bash", d
    if name == "str_replace_editor":
        cmd = args.get("command", "")
        path = args.get("path", "")
        if cmd == "view":
            d = {"filePath": path}
            if args.get("view_range"):
                d["viewRange"] = args["view_range"]
            return "read", d
        if cmd == "create":
            return "write", {"filePath": path, "content": args.get("file_text", "")}
        d = {"filePath": path}
        if cmd == "insert":
            d.update({"insertLine": args.get("insert_line"), "newString": args.get("new_str", "")})
        elif cmd == "str_replace":
            d.update({"oldString": args.get("old_str", ""), "newString": args.get("new_str", "")})
        else:                       # undo_edit and anything unknown
            d["command"] = cmd
        return "edit", d
    return name or "?", args


def tool_event_text(name: str, args: dict, output: str, repo_root: str,
                    in_cap: int, out_cap: int) -> str:
    """Same layout as import_opencode.part_text(): '[tool N] in: <brief>\\n[out] <output>'."""
    oc_name, inp = render_call(name, args)
    out = output or ""
    if name == "execute_bash":
        out = _clean_bash_output(out, repo_root)
    else:
        out = EDIT_TAIL_RE.sub("", out).strip()
    brief = clip(json.dumps(inp, ensure_ascii=False), in_cap)
    return f"[tool {oc_name}] in: {brief}\n[out] {clip(out, out_cap)}"


# ------------------------------------------------------------------- conversion

def extract_events(msgs: list[dict], in_cap: int = IN_CAP, out_cap: int = OUT_CAP) -> tuple[list[dict], dict]:
    """Linear events (no compaction markers yet) + a few counters."""
    events: list[dict] = []
    stats = {"messages": len(msgs), "tool_calls": 0, "think_dropped": 0, "unmatched_calls": 0,
             "issue_found": False}
    repo_root = ""
    by_id = {m.get("tool_call_id"): m for m in msgs if m.get("role") == "tool"}
    first_user_done = False
    for m in msgs:
        role = m.get("role")
        if role == "system" or role == "tool":
            continue
        if role == "user":
            t = msg_text(m)
            if not first_user_done:
                first_user_done = True
                up = re.search(r"<uploaded_files>\n?(.*?)\n?</uploaded_files>", t, re.S)
                repo_root = (up.group(1).strip() if up else "")
                im = ISSUE_RE.search(t)
                stats["issue_found"] = bool(im)
                t = im.group(1) if im else t
            t = t.strip()
            if t:
                events.append({"role": "user", "text": t, "ts": len(events)})
            continue
        if role != "assistant":
            continue
        txt = msg_text(m).strip()
        if txt:
            events.append({"role": "assistant", "text": txt, "ts": len(events)})
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name") or "?"
            if name == "think":
                stats["think_dropped"] += 1
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {"arguments": args}
            except (ValueError, TypeError):
                args = {"arguments": str(fn.get("arguments"))[:in_cap]}
            res = by_id.get(tc.get("id"))
            if res is None:
                stats["unmatched_calls"] += 1
            out = msg_text(res) if res is not None else ""
            stats["tool_calls"] += 1
            events.append({"role": "tool",
                           "text": tool_event_text(name, args, out, repo_root, in_cap, out_cap),
                           "ts": len(events)})
    stats["repo_root"] = repo_root
    return events, stats


def _piece_len(e: dict) -> int:
    return len(f"[{e['role']}] {e['text'].strip()}\n\n")


def place_boundaries(events: list[dict], n_boundaries: int = N_BOUNDARIES,
                     span_segs: int = DEFAULT_SEGS, start_frac: float = START_FRAC,
                     chunk: int = CHUNK) -> list[int]:
    """Event indices (into `events`, before marker insertion) at which a compaction marker is
    inserted.  Uses split_segments()'s greedy packing (a segment closes when the next piece
    would overflow `chunk`), so span k > 1 holds exactly `span_segs` segments.  Raises
    ValueError when the session is too short."""
    n = len(events)
    pl = [_piece_len(e) if e["text"].strip() else 0 for e in events]
    total = sum(pl)

    def cut_after_segs(start: int, k: int) -> int | None:
        """Index of the event that would open segment k+1 when packing from `start`,
        or None if the session ends first."""
        segs, cur = 0, 0
        for i in range(start, n):
            if not pl[i]:
                continue
            if cur and cur + pl[i] > chunk:
                segs += 1
                if segs == k:
                    return i
                cur = pl[i]
            else:
                cur += pl[i]
        return None

    def first_cut(target_chars: float) -> int | None:
        """First cut (after >= span_segs segments from event 0) such that the replay window
        (the last span_segs segments) starts at or after `target_chars`."""
        segs, cur, ends = 0, 0, []   # ends[j] = (char offset at end of seg j, next event idx)
        off = 0
        for i in range(n):
            if not pl[i]:
                continue
            if cur and cur + pl[i] > chunk:
                ends.append((off, i))
                segs += 1
                cur = pl[i]
            else:
                cur += pl[i]
            off += pl[i]
        # window of cut j (j segs before the cut) starts at ends[j - span_segs - 1].off
        for j in range(span_segs, len(ends) + 1):
            if j == span_segs:
                start_off = 0
            else:
                start_off = ends[j - span_segs - 1][0]
            if start_off >= target_chars and j - 1 < len(ends):
                return ends[j - 1][1]
        return None

    for frac in [start_frac] + [f for f in (0.2, 0.15, 0.1, 0.05, 0.0) if f < start_frac]:
        first = first_cut(frac * total)
        if first is None:
            continue
        cuts, ok = [first], True
        for _ in range(n_boundaries - 1):
            nxt = cut_after_segs(cuts[-1], span_segs)
            if nxt is None:
                ok = False
                break
            cuts.append(nxt)
        if ok:
            return cuts
    raise ValueError(f"session too short for {n_boundaries} boundaries of {span_segs} segments "
                     f"({total} chars)")


def insert_boundaries(events: list[dict], cuts: list[int]) -> tuple[list[dict], list[dict]]:
    out, bounds, prev = [], [], 0
    for c in cuts:
        out.extend(events[prev:c])
        idx = len(out)
        out.append({"role": "compaction", "text": "", "ts": events[c]["ts"] if c < len(events) else 0,
                    "auto": True, "synthetic": True})
        bounds.append({"idx": idx, "auto": True,
                       "original_summary": "", "original_summary_chars": 0,
                       "orig_summary_valid": False, "orig_summary_source": "none",
                       "orig_summary_reject": "missing",
                       "orig_summary_note": "synthetic boundary: no real compaction in source",
                       "synthetic": True})
        prev = c
    out.extend(events[prev:])
    return out, bounds


def convert(msgs: list[dict], instance_id: str, *, in_cap: int = IN_CAP, out_cap: int = OUT_CAP,
            n_boundaries: int = N_BOUNDARIES, span_segs: int = DEFAULT_SEGS,
            start_frac: float = START_FRAC, source: dict | None = None) -> dict:
    events, stats = extract_events(msgs, in_cap, out_cap)
    cuts = place_boundaries(events, n_boundaries, span_segs, start_frac)
    all_events, bounds = insert_boundaries(events, cuts)
    spans, prev = [], 0
    for b in bounds + [{"idx": len(all_events)}]:
        spans.append(sum(len(x["text"]) for x in all_events[prev:b["idx"]]))
        prev = b["idx"]
    repo = repo_of(instance_id)
    return {
        "sid": sid_of(instance_id),
        "title": f"OpenHands coding session in {repo}",   # no issue title: avoids priming QA
        "n_msgs": len(msgs),
        "events": all_events,
        "boundaries": bounds,
        "span_chars": spans[:-1],
        "instance_id": instance_id,
        "repo": repo,
        "source": source or {},
        "import": {"importer": "import_openhands.py", "in_cap": in_cap, "out_cap": out_cap,
                   "n_boundaries": n_boundaries, "span_segs": span_segs, "start_frac": start_frac,
                   "chunk": CHUNK, "token_budget": TOKEN_BUDGET, "chars_per_token": CHARS_PER_TOKEN,
                   "extracted_chars": sum(len(e["text"]) for e in all_events),
                   "think_dropped": stats["think_dropped"], "tool_calls": stats["tool_calls"],
                   "unmatched_calls": stats["unmatched_calls"], "issue_found": stats["issue_found"]},
    }


def check_session(s: dict, span_segs: int = DEFAULT_SEGS) -> list[str]:
    """Structural self-check against make_chains' own span builder; returns problems."""
    import make_chains as mc  # noqa: PLC0415 - local: heavy module, optional here
    probs = []
    ev = s["events"]
    for b in s["boundaries"]:
        if ev[b["idx"]]["role"] != "compaction":
            probs.append(f"boundary idx {b['idx']} is not a compaction marker")
    spans = mc.session_spans(s)
    for k, sp in enumerate(spans[:len(s["boundaries"])]):
        segs = mc.segments_struct(sp)
        win, dropped = mc.window_struct(segs, span_segs)
        if not mc.selfcheck_span(sp, span_segs):
            probs.append(f"span {k}: segmentation mismatch vs make_longreplay_dataset")
        if k > 0 and dropped:
            probs.append(f"span {k}: {dropped} segment(s) dropped from the replay window")
        if len(win) < 2 or sum(len(mc.seg_text(x)) for x in win) < mc.MIN_SPAN_CHARS:
            probs.append(f"span {k}: thin window")
    return probs


# --------------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True, help="sessions output dir (must be git-ignored/outside git)")
    ap.add_argument("--src", nargs="*", default=[], help="trajectory .json files")
    ap.add_argument("--src-dir", default="", help="directory of <instance_id>.json trajectories")
    ap.add_argument("--manifest", default="", help="select_openhands.py manifest: import only its instances")
    ap.add_argument("--in-cap", type=int, default=IN_CAP)
    ap.add_argument("--out-cap", type=int, default=OUT_CAP)
    ap.add_argument("--n-boundaries", type=int, default=N_BOUNDARIES)
    ap.add_argument("--span-segs", type=int, default=DEFAULT_SEGS)
    ap.add_argument("--start-frac", type=float, default=START_FRAC)
    args = ap.parse_args()
    ensure_not_tracked_path(args.out_dir, is_dir=True)

    files = list(args.src)
    if args.manifest:
        man = json.load(open(args.manifest))
        if not args.src_dir:
            raise SystemExit("--manifest needs --src-dir")
        files += [os.path.join(args.src_dir, f"{e['instance_id']}.json") for e in man["selected"]]
    elif args.src_dir:
        files += sorted(os.path.join(args.src_dir, f) for f in os.listdir(args.src_dir) if f.endswith(".json"))
    if not files:
        raise SystemExit("no input trajectories")
    os.makedirs(args.out_dir, exist_ok=True)

    seen8: dict[str, str] = {}
    n_ok = 0
    for f in files:
        iid = os.path.basename(f)[:-5]
        try:
            with open(f) as fh:
                msgs = json.load(fh)
            s = convert(msgs, iid, in_cap=args.in_cap, out_cap=args.out_cap,
                        n_boundaries=args.n_boundaries, span_segs=args.span_segs,
                        start_frac=args.start_frac,
                        source={"file": os.path.basename(f), "sha256": sha256_file(f),
                                "bytes": os.path.getsize(f)})
        except (ValueError, OSError, KeyError, TypeError) as e:
            print(f"SKIP {iid}: {type(e).__name__}: {e}")
            continue
        probs = check_session(s, args.span_segs)
        if probs:
            print(f"SKIP {iid}: " + "; ".join(probs))
            continue
        k8 = s["sid"][-8:]
        if k8 in seen8:
            print(f"WARN {iid}: sid[-8:] {k8!r} collides with {seen8[k8]} (chain ids get a suffix)")
        seen8[k8] = iid
        with open(os.path.join(args.out_dir, f"{s['sid']}.json"), "w") as w:
            json.dump(s, w, ensure_ascii=False, indent=1)
        n_ok += 1
        print(f"{iid}: {len(s['events'])} events, extracted {s['import']['extracted_chars']} chars, "
              f"boundaries {[b['idx'] for b in s['boundaries']]}, span_chars {s['span_chars']}")
    print(f"imported {n_ok}/{len(files)} sessions -> {args.out_dir}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
