#!/usr/bin/env python3
"""run_longsession records -> judge_v2 input (blind, evidence-grounded, rubric-aware).

The long-session twin of export_judge_input.py (which is chain-shaped and needs chains.jsonl).
Joins each record of `records.jsonl` with its session file (the evidence messages) and writes
the fields judge_v2.py requires:

  question, canonical_short_answer, answer, gold_anchor, answerable, rubric (BEAM rubric: passed
  to the judge when present), span_text (the evidence messages, role-tagged), evidence (list of
  verbatim snippets -> one +-500-char window each), visible_context (what the arm actually saw:
  note/summary + injected recall (nc-enriched: the recalled-facts block of the compaction) + the verbatim post-boundary tail; the question and the answer
  are excluded, so judge_v2's transcript verification and metrics_v2's primed_hit stay honest).

Blindness is the chains' machinery: judge_v2 masks arm-revealing strings in the judge-visible
answer and logs it in the mapping; this exporter additionally drops memory_text/injected (the
arm's memory channel) from the file and puts nothing arm-specific into the prompt fields.
Validity is metrics_v2.validate_rows (same filter as the chains: invalid per-(sid, B) cells and
incomplete arm sets are dropped, duplicates exit 2; the descriptive `orig` arm is kept).

Usage:
  python3 export_longsession_judge.py --records runs/R/records.jsonl --sessions data/beam1M \
      --out runs/R/judge-input.jsonl [--context-cap 150000] [--include-sid '^beam-1M-']
  python3 judge_v2.py --in runs/R/judge-input.jsonl --out runs/R/judged.jsonl \
      --mapping runs/R/judge-mapping.json --profile anthropic --allow-same-family-judge --budget-usd N
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import longsession as L  # noqa: E402
from metrics_v2 import filter_sids, pick, raw_rid, validate_rows, validity_lines  # noqa: E402

DEFAULT_CONTEXT_CAP = 150000      # chars of visible_context (the tail is trimmed from its start)
EVIDENCE_SNIPPET = 300            # chars of one evidence snippet handed to the judge window
MSG_CAP = 6000                    # chars of one evidence message kept in span_text
DROP_KEYS = ("memory_text", "injected")


def evidence_parts(session: dict, q: dict) -> tuple[str, list[str]]:
    """(span_text, [evidence snippets]) of a question from its session messages."""
    msgs = session["messages"]
    span, snippets, seen = [], [], set()
    for ev in q.get("evidence") or []:
        i = ev.get("msg_idx")
        if not isinstance(i, int) or not (0 <= i < len(msgs)) or i in seen:
            continue
        seen.add(i)
        m = msgs[i]
        text = m.get("text") or ""
        st, en = ev.get("start"), ev.get("end")
        if isinstance(st, int) and isinstance(en, int) and 0 <= st < en <= len(text):
            lo = max(0, st - 2000)
            body = text[lo:min(len(text), en + 2000)]
            snip = L.norm_ws(text[st:en])[:EVIDENCE_SNIPPET]
        else:
            body = text[:MSG_CAP]
            snip = L.norm_ws(text)[:EVIDENCE_SNIPPET]
        span.append(f"[{m.get('role', '')}] {body.strip()}\n\n")
        if snip:
            snippets.append(snip)
    return "".join(span), snippets


def visible_context(rec: dict, session: dict | None, cap: int) -> tuple[str | None, dict]:
    """memory (note / summary) + injected recall + verbatim tail, question/answer excluded."""
    head = [x.strip() for x in (rec.get("memory_text"), rec.get("injected")) if (x or "").strip()]
    tail = ""
    if session is not None and isinstance(rec.get("tail_from"), int) and isinstance(rec.get("tail_to"), int):
        tail = L.serialize(session["messages"][rec["tail_from"]:rec["tail_to"] + 1])
    meta = {"has_memory": bool(head), "has_tail": bool(tail), "truncated": False}
    if not head and not tail:
        return None, meta
    h = "\n\n".join(head)
    room = max(0, cap - len(h) - 2)
    if len(tail) > room:
        tail, meta["truncated"] = tail[len(tail) - room:], True      # keep the newest part
    ctx = "\n\n".join(x for x in (h, tail) if x)
    if len(ctx) > cap:
        ctx, meta["truncated"] = ctx[:cap], True
    return ctx, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, help="run_longsession records.jsonl (glob / comma list)")
    ap.add_argument("--sessions", nargs="+", required=True, help="longsession/1 files or dirs (the runs' inputs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context-cap", type=int, default=DEFAULT_CONTEXT_CAP)
    ap.add_argument("--include-sid", default="", metavar="REGEX")
    ap.add_argument("--exclude-sid", default="", metavar="REGEX")
    ap.add_argument("--skip-judged", default="", metavar="JUDGED.jsonl[,..]",
                    help="Amendment C.7 reuse: do not export rows whose rid already carries a verdict in these judge_v2 "
                         "outputs (validity is still decided on ALL rows; only the judge input shrinks)")
    a = ap.parse_args(argv)

    paths = sorted(p for pat in a.records.split(",") for p in glob.glob(pat))
    if not paths:
        print(f"no records files match: {a.records}", file=sys.stderr)
        return 2
    rows = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    n_raw = len(rows)
    rows, flt = filter_sids(rows, a.include_sid, a.exclude_sid)
    rows, validity = validate_rows(rows)
    for line in validity_lines(validity):
        print(f"[validity] {line}", file=sys.stderr)
    if not rows:
        print("no valid rows left after the validity filter", file=sys.stderr)
        return 1
    skip = set()
    for pth in [x for x in a.skip_judged.split(",") if x.strip()]:
        for l in open(pth):
            if l.strip():
                j = json.loads(l)
                if j.get("rid") and j.get("score") is not None:
                    skip.add(j["rid"])
    n_skip_total = 0
    sessions = {s["sid"]: s for s in L.load_sessions(a.sessions)}
    qa = {sid: {q["qid"]: q for q in s.get("questions") or []} for sid, s in sessions.items()}

    missing, n_out, ctx_present, truncated = set(), 0, 0, 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fo:
        for r in rows:
            if raw_rid(r) in skip:
                n_skip_total += 1
                continue
            sid = str(pick(r, "chain_id"))
            s = sessions.get(sid)
            q = qa.get(sid, {}).get(r.get("base_qid") or str(r.get("qid")).split("@c")[0])
            if s is None or q is None:
                missing.add(f"{sid}/{r.get('qid')}")
                continue
            span_text, snippets = evidence_parts(s, q)
            if not span_text:       # unanswerable probe / no evidence: filler like the chains export
                span_text = f"[{s['messages'][0]['role']}] {s['messages'][0]['text'][:MSG_CAP]}\n\n" \
                    if s["messages"] else ""
            ctx, meta = visible_context(r, s, a.context_cap)
            out = {k: v for k, v in r.items() if k not in DROP_KEYS}
            out.update({"rid": raw_rid(r), "question": q["question"], "span_text": span_text,
                        "evidence": snippets, "rubric": q.get("rubric") or [],
                        "canonical_short_answer": q.get("answer"),
                        "visible_context": ctx, "visible_context_meta": meta})
            ctx_present += ctx is not None
            truncated += bool(meta["truncated"])
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
    if missing:
        print(f"WARNING unmatched records (dropped): {sorted(missing)[:5]} ...", file=sys.stderr)
    print(f"exported {n_out}/{len(rows) - n_skip_total} valid records to judge ({n_raw} read, {n_skip_total} already judged and skipped) -> {a.out}; "
          f"visible_context present={ctx_present} truncated={truncated}; "
          f"rubric on {sum(1 for l in open(a.out) if json.loads(l).get('rubric'))} record(s)")
    return 0 if n_out == len(rows) - n_skip_total else 1


if __name__ == "__main__":
    sys.exit(main())
