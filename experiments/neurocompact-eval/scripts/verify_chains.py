#!/usr/bin/env python3
"""Independent end-to-end re-validation of chains.jsonl.

Does NOT trust the stored `checks`: rebuilds every chain byte-for-byte from the
archived session files (same code path as make_chains.build_chain), asserts the
stored segments match the rebuild, then re-runs every deterministic gate from
scratch:

  per answerable item
    EVID  evidence is verbatim (whitespace-normalised, case-sensitive) in its span
    EWORD evidence has >= 8 words
    ANCH  gold_anchor regex-present in its own span
    DISC  gold_anchor absent from the other two spans
    LEAK  question does not contain the gold_anchor
    FORM  anchor is a usable token (>= 2 chars, not a bare digit / bad word)
    HALF  (soft) evidence sits in the labelled HEAD/TAIL half
  per unanswerable item
    PROB  question carries >= 1 concrete probe token
    ZERO  zero probe hits across all 3 spans AND the full session transcript
    EMPTY answer / gold_anchor / evidence are all empty

Usage: python3 verify_chains.py [--chains chains.jsonl] [--out verify_chains.report.md]
Exit 0 iff no hard-gate failure.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from make_chains import (  # noqa: E402
    ANCHOR_BAD, CHAIN_LEN, anchor_re, build_chain, has_verbatim, halves,
    load_sessions, norm_ws, probe_tokens, transcript_of,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", default=os.path.join(BASE, "chains.jsonl"))
    ap.add_argument("--segs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(BASE, "verify_chains.report.md"))
    args = ap.parse_args()

    sessions = load_sessions()
    rebuilt = {}
    for s in sessions:
        for ci in range(len(s["boundaries"]) - CHAIN_LEN + 1):
            rebuilt[(s["sid"], ci)] = (s, build_chain(s, ci, args.segs, check=True))

    chains = [json.loads(l) for l in open(args.chains) if l.strip()]
    rows, fails, span_rows = [], [], []
    stats = {}
    seen_qids = {}          # qid is the pairing key: must be unique dataset-wide

    def bump(gate, ok):
        n, p = stats.get(gate, (0, 0))
        stats[gate] = (n + 1, p + (1 if ok else 0))

    for c in chains:
        chain_anchors = {}
        key = (c["session_id"], c["chain_idx"])
        if key not in rebuilt:
            fails.append((c["chain_id"], "REBUILD", "session/chain_idx not reconstructable"))
            continue
        s, chain = rebuilt[key]
        span_texts = {sp["span_idx"]: sp["text"] for sp in chain["spans"]}
        transcript = transcript_of(s)

        # 0) stored segments byte-identical to the rebuild
        seg_match, char_match = True, True
        for stored, fresh in zip(c["spans"], chain["spans"]):
            if [(g["role"], g["text"]) for g in stored["segments"]] != \
               [(e.get("role", ""), (e.get("text") or "")) for e in fresh["segments"]]:
                seg_match = False
            if stored["chars"] != fresh["chars"] or stored["est_tokens"] != fresh["est_tokens"]:
                char_match = False
        span_rows.append((c["chain_id"], seg_match, char_match,
                          [sp["chars"] for sp in chain["spans"]],
                          [sp["est_tokens"] for sp in chain["spans"]]))
        bump("SEGS==rebuild", seg_match)
        if not seg_match:
            fails.append((c["chain_id"], "SEGMENTS", "stored segments differ from rebuild"))

        for qa in c["qa"]:
            if qa["qid"] in seen_qids:
                fails.append((c["chain_id"], qa["qid"],
                              f"qid collision with {seen_qids[qa['qid']]}"))
            seen_qids[qa["qid"]] = c["chain_id"]
            q = norm_ws(qa["question"])
            hard = {}
            soft_half = None
            detail = ""
            if qa["answerable"]:
                own = span_texts.get(qa["span_idx"])
                anchor = (qa["gold_anchor"] or "").strip()
                ev = qa["evidence"] or ""
                rx = anchor_re(anchor) if anchor else None
                own_hits = len(rx.findall(own)) if (rx and own) else 0
                hard["EVID"] = bool(own) and has_verbatim(own, ev)
                hard["EWORD"] = len(norm_ws(ev).split()) >= 8
                hard["ANCH"] = own_hits > 0
                hard["DISC"] = bool(rx) and not any(
                    rx.search(t) for k, t in span_texts.items() if k != qa["span_idx"])
                hard["LEAK"] = bool(anchor) and anchor.lower() not in q.lower()
                # FORM mirrors make_chains' rule exactly: only a bare SINGLE digit
                # (or a stopword) is bad-form; long numbers like unix timestamps
                # are legitimate "number" anchors when unique to the span.
                hard["FORM"] = (len(anchor) >= 2
                                and not re.fullmatch(r"\d", anchor)
                                and anchor.lower() not in ANCHOR_BAD)
                if own is not None:
                    head, tail = halves(own)
                    soft_half = has_verbatim(tail if qa["slice"] == "TAIL" else head, ev)
                detail = (f"anchor={anchor!r} hits_own={own_hits} "
                          f"ev_words={len(norm_ws(ev).split())}")
            else:
                probes = probe_tokens(q)
                hits = []
                for t in probes:
                    rx = anchor_re(t)
                    if any(rx.search(t2) for t2 in span_texts.values()) or rx.search(transcript):
                        hits.append(t)
                hard["PROB"] = len(probes) > 0
                hard["ZERO"] = not hits
                hard["EMPTY"] = not (qa["canonical_short_answer"] or qa["gold_anchor"]
                                     or qa["evidence"])
                detail = f"probes={probes[:3]} hits={hits[:3]}"

            for g, ok in hard.items():
                bump(g, ok)
            if soft_half is not None:
                bump("HALF(soft)", soft_half)
            if qa["answerable"]:
                a = (qa["gold_anchor"] or "").strip().lower()
                if a:
                    bump("ANCHOR-UNIQ(chain)", a not in chain_anchors)
                    if a in chain_anchors:
                        fails.append((c["chain_id"], qa["qid"],
                                      f"anchor {a!r} reused from {chain_anchors[a]}"))
                    else:
                        chain_anchors[a] = qa["qid"]
            hard_ok = all(hard.values())
            if not hard_ok:
                fails.append((c["chain_id"], qa["qid"],
                              ",".join(g for g, v in hard.items() if not v)))
            rows.append((c["chain_id"], qa["qid"], "A" if qa["answerable"] else "U",
                         qa["lag"], qa["span_idx"], qa["slice"],
                         "PASS" if hard_ok else "FAIL",
                         ("in-half" if soft_half else "OUT-OF-HALF") if soft_half is not None else "-",
                         detail))

    # ---- report
    L = ["# verify_chains: independent re-validation", ""]
    L.append(f"chains: {len(chains)} | items: {len(rows)} "
             f"(answerable {sum(1 for r in rows if r[2] == 'A')}, "
             f"unanswerable {sum(1 for r in rows if r[2] == 'U')})")
    L.append("")
    L.append("## Span rebuild check (stored vs recomputed from sessions)")
    L.append("")
    L.append("| chain | segments==rebuild | chars/tok==rebuild | span chars s1/s2/s3 | est tok s1/s2/s3 |")
    L.append("|---|---|---|---|---|")
    for cid, m, cm, ch, tk in span_rows:
        L.append(f"| {cid} | {'yes' if m else 'NO'} | {'yes' if cm else 'NO'} | "
                 f"{'/'.join(map(str, ch))} | {'/'.join(map(str, tk))} |")
    L.append("")
    L.append("## Gate rates")
    L.append("")
    L.append("| gate | pass | applicable | rate |")
    L.append("|---|---|---|---|")
    for g, (n, p) in stats.items():
        L.append(f"| {g} | {p} | {n} | {p / n:.3f} |" if n else f"| {g} | - | 0 | - |")
    L.append("")
    L.append("## Per-item table")
    L.append("")
    L.append("| chain | qid | kind | lag | span | slice | verdict | half | detail |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    L.append("")
    L.append(f"hard-gate failures: {len(fails)}")
    for f in fails:
        L.append(f"- {f[0]} {f[1]}: {f[2]}")
    report = "\n".join(L) + "\n"
    print(report)
    with open(args.out, "w") as w:
        w.write(report)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
