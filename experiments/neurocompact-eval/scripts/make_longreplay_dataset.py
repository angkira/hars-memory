#!/usr/bin/env python3
"""Build the long-replay dataset from imported OpenCode sessions.

For every (session, compaction boundary): the span between the previous boundary
and this one becomes the replay material; QA is generated over the span; the
original compaction summary is kept as the baseline branch's context artifact.

The replay WINDOW is decided first and QA is generated over that window only:
only the tail `--segs N` segments (default 8, 0 = all span segments) are ever
replayed by run_longreplay.py, so QA built from the rest of the span would be
structurally unanswerable (smoke 2026-10-25: 4 of 6 questions referred to
content outside the replayed tail). The window is additionally capped to
TOKEN_BUDGET: earliest segments are dropped until the estimate fits.

Output:
  longreplay/dataset-longreplay.jsonl   q/a/evidence records (session_id, span_idx)
  longreplay/longreplay-manifest.json  per-boundary: replay_segments, token estimates,
                                        original summary, window/dropped segment counts

QA generation uses the cheap judge-class model via OpenRouter (env OPENROUTER_API_KEY).
"""
import json, os, glob, argparse, urllib.request, time

BASE = os.path.dirname(os.path.abspath(__file__))
SESS = os.path.join(BASE, "longreplay", "sessions")
OUTD = os.path.join(BASE, "longreplay")
CHUNK = 3500              # chars per replay segment
CHARS_PER_TOKEN = 2.4     # conservative; code/tool logs run ~2.5-3 chars/token
TOKEN_BUDGET = 20000      # replayed window must stay under this
DEFAULT_SEGS = 8          # tail segments replayed when --segs is not given
MAX_QA_PER_SPAN = int(os.environ.get("NC_MAX_QA", "8"))
GEN_MODEL = os.environ.get("NC_GEN_MODEL", "z-ai/glm-5.3-flash")

def llm(prompt: str, key: str, max_tokens: int = 4096) -> str:
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({"model": GEN_MODEL, "max_tokens": max_tokens,
                         # glm-flash sometimes spends the whole completion on hidden
                         # reasoning and returns prose (or nothing) instead of the JSON
                         # contract; capping it keeps the budget for the answer.
                         "reasoning": {"max_tokens": 300},
                         "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 "User-Agent": "neurocompact-eval/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                return json.load(r)["choices"][0]["message"].get("content") or ""
        except Exception as e:
            if attempt == 2:
                print(f"  llm failed: {e}")
                return ""
            time.sleep(5 * (attempt + 1))

def parse_qa(raw: str) -> list[dict]:
    """Tolerant JSON extraction: the model may fence the object or emit raw newlines."""
    if not raw:
        return []
    m = raw[raw.find("{"): raw.rfind("}") + 1]
    if not m:
        return []
    for kwargs in ({}, {"strict": False}):
        try:
            j = json.loads(m, **kwargs)
        except Exception:
            continue
        out = []
        for q in j.get("questions", []):
            if q.get("q") and q.get("a") and q.get("evidence"):
                out.append({"q": q["q"], "a": q["a"], "evidence": q["evidence"],
                            "pos": q.get("pos", "middle")})
        return out
    return []

def split_segments(span_events):
    """Serialize span events into ordered text segments of ~CHUNK chars, cut on turn edges."""
    segs, cur = [], ""
    for e in span_events:
        t = e["text"].strip()
        if not t:
            continue
        piece = f"[{e['role']}] {t}\n\n"
        if len(cur) + len(piece) > CHUNK and cur:
            segs.append(cur)
            cur = piece
        else:
            cur += piece
    if cur.strip():
        segs.append(cur)
    return segs

def replay_window(segs, tail_n):
    """The exact segments run_longreplay.py will replay: tail_n newest, then the
    earliest ones dropped until the token estimate fits TOKEN_BUDGET."""
    win = list(segs[-tail_n:]) if tail_n else list(segs)
    dropped = len(segs) - len(win)
    while len(win) > 1 and sum(len(s) for s in win) / CHARS_PER_TOKEN >= TOKEN_BUDGET:
        win.pop(0)
        dropped += 1
    return win, dropped

def gen_qa(key, title, segs, max_qa=None):
    # sample across the WINDOW (never the wider span): head/mid/tail slices —
    # the "distance to boundary" axis is kept, but every slice is replayed text
    joined = "\n\n".join(segs)
    n = len(joined)
    parts = []
    for name, a, b in (("HEAD", 0, 8000), ("MIDDLE", n // 2 - 4000, n // 2 + 4000),
                       ("TAIL", n - 8000, n)):
        a, b = max(0, a), min(n, b)
        if b - a > 500:
            parts.append(f"[{name}]\n{joined[a:b]}")
    span = "\n\n[...]\n\n".join(parts)[:24000]
    prompt = (
        "You build an exam from a working session log. The log is given in three slices: "
        "[HEAD] (oldest), [MIDDLE], [TAIL] (closest to the end). From all slices, write "
        f"{max_qa or MAX_QA_PER_SPAN} factual questions that test retention of SPECIFIC facts: "
        "numbers/percentages, model or file names, decisions and their parameters. "
        "Prefer facts stated once, deep in the log. Distribute questions across the three "
        "slices (at least 2 per slice). Each question must be answerable in under 15 words "
        "from the log alone.\n"
        '"evidence" must be copied VERBATIM from the log: 8 or more consecutive words '
        "exactly as they appear (no paraphrase, no ellipsis).\n"
        'Reply ONLY JSON: {"questions":[{"q":"...","a":"...","evidence":"exact log sentence",'
        '"pos":"head|middle|tail"}]}\n\n'
        f"Session: {title}\n\n{span}"
    )
    qs = parse_qa(llm(prompt, key))
    if len(qs) < 3:
        # one retry with an explicit no-prose instruction; a silent skip here
        # biases the dataset toward sessions the model happens to handle
        qs = parse_qa(llm(prompt + "\n\nOutput the JSON object only. No analysis, no prose.", key))
    return qs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8, help="max boundaries total")
    ap.add_argument("--max-qa", type=int, default=MAX_QA_PER_SPAN, help="questions per boundary")
    ap.add_argument("--segs", type=int, default=DEFAULT_SEGS,
                    help="tail segments replayed (0 = all span segments); the window is "
                         "capped so sum(len)/%.1f < %d tokens" % (CHARS_PER_TOKEN, TOKEN_BUDGET))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned window per boundary, no LLM calls, no writes")
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and not args.dry_run:
        raise SystemExit("export OPENROUTER_API_KEY first (QA generation is paid)")
    os.makedirs(OUTD, exist_ok=True)
    records, manifest = [], {"boundaries": []}
    n = 0
    for f in sorted(glob.glob(os.path.join(SESS, "*.json"))):
        s = json.load(open(f))
        ev = s["events"]
        bounds = s["boundaries"]
        prev = 0
        for bi, b in enumerate(bounds + [{"idx": len(ev)}]):
            if bi >= len(bounds):
                break
            span = ev[prev:bounds[bi]["idx"]]
            prev = bounds[bi]["idx"] + 1
            all_segs = split_segments(span)
            segs, dropped = replay_window(all_segs, args.segs)
            span_chars = sum(len(x) for x in segs)
            est_tokens = int(span_chars / CHARS_PER_TOKEN)
            if len(segs) < 2 or span_chars < 8000:
                continue
            n += 1
            if n > args.limit:
                print(f"limit {args.limit} reached; rerun with --limit to extend")
                break
            sid = s["sid"]
            title = s.get("title", "") or sid
            print(f"[{n}] {sid} boundary {bi}: window {len(segs)}/{len(all_segs)} segs "
                  f"({dropped} dropped), {span_chars} chars ~{est_tokens} tok")
            if args.dry_run:
                continue
            qa = gen_qa(key, title, segs, args.max_qa)
            if len(qa) < 3:
                print("  qa generation weak, skipping boundary")
                n -= 1
                continue
            sid_short = f"{sid[-8:]}b{bi}"
            span_id = f"span_{sid_short}"
            for i, q in enumerate(qa):
                records.append({"q": q["q"], "a": q["a"], "evidence": q["evidence"],
                                "pos": q.get("pos", "middle"),
                                "session_id": span_id, "span_idx": i,
                                "session_title": title})
            manifest["boundaries"].append({
                "span_id": span_id, "session": sid, "title": title,
                # the ONLY segments a replay may use, in order, verbatim
                "replay_segments": segs,
                "span_chars": span_chars, "est_tokens": est_tokens,
                # reference artifact + whether it is a real summary; run_longreplay
                # skips the orig arm when this is false (see import_opencode.py)
                "original_summary": b["original_summary"],
                "original_summary_chars": b.get("original_summary_chars",
                                                len(b["original_summary"])),
                "orig_summary_valid": bool(b.get("orig_summary_valid")),
                "orig_summary_source": b.get("orig_summary_source", ""),
                "orig_summary_reject": b.get("orig_summary_reject", ""),
                "boundary_idx": b.get("idx"),
                "window_segments": len(segs), "dropped_segments": dropped,
            })
            json.dump(manifest, open(os.path.join(OUTD, "longreplay-manifest.json"), "w"),
                      ensure_ascii=False, indent=1)
        if n > args.limit:
            break
    if args.dry_run:
        print("dry run: no manifest or dataset written")
        return
    with open(os.path.join(OUTD, "dataset-longreplay.jsonl"), "w") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"dataset: {len(records)} questions over {len(manifest['boundaries'])} boundaries")

if __name__ == "__main__":
    import sys
    main()
