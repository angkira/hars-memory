#!/usr/bin/env python3
"""Join run_chainreplay records.jsonl with chains.jsonl into judge_v2 input.

run_chainreplay's records.jsonl lacks the `question` and `span_text` fields
judge_v2.py requires for its evidence windows. This adds them from the dataset:
  - question:  qa.question (verbatim)
  - span_text: the dataset span's stored text (role-tagged, exactly what
    make_chains validated evidence against). Unanswerable probes (span_idx 0)
    carry no span; they get span 1's text as a filler because judge_v2 only
    uses OTHER records' span_text for them (unrelated_excerpt) — same fallback
    as make_calibration_items.py.

It also attaches, per record, the answer-arm's ACTUALLY-VISIBLE CONTEXT — the
post-C3 state the model answered from (see visible_context.py): compaction
summary/note first, then the nc-injected <memory> recall block, then
post-compaction messages; line-dedup'd, ~12k char cap. The question text and
the answer itself are excluded so transcript verification (judge_v2 post-pass)
and the question-priming check (metrics_v2 primed_hit) stay honest. When no ask
session exists the record still exports, with `visible_context` null.

Data validity (B2, same filter as metrics_v2.validate_rows): only rows with
``valid is True`` are exported; a chain with any invalid row or any recall_error is
dropped whole; qids without the full (baseline, nc, nc-note-only) triplet are
dropped (Amendment C: four arms baseline / nc-chunks / nc-graph / nc-note-only, per
mem_budget cell; a run with no nc-graph row is a legacy 3-arm run and `nc` reads as
nc-chunks); duplicate (qid, arm, mem_budget) or duplicate rid across ALL input files is a
hard error (exit 2). Drop counts and reasons are printed.

Usage:
  python3 export_judge_input.py --records "runs/gate-*/records.jsonl" \
      --chains chains.jsonl --out runs/gate-*/judge-input.jsonl
  # sessions root defaults to <dirname of each records file>/sessions;
  # override with --sessions-root when sessions live elsewhere.
"""
import argparse
import glob
import json
import os
import sys

from metrics_v2 import canon_arm, raw_rid, validate_rows, validity_lines
from visible_context import DEFAULT_CAP, build_visible_context

BASE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="glob of records.jsonl files")
    ap.add_argument("--chains", default=os.path.join(BASE, "chains.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions-root", default="",
                    help="sessions dir (default: <records dir>/sessions per file)")
    ap.add_argument("--context-cap", type=int, default=DEFAULT_CAP,
                    help=f"visible_context char cap (default {DEFAULT_CAP})")
    args = ap.parse_args()

    chains = {}
    with open(args.chains) as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                chains[c["chain_id"]] = c

    paths = sorted(p for pat in args.records.split(",") for p in glob.glob(pat))
    if not paths:
        print(f"no records files match: {args.records}", file=sys.stderr)
        return 2

    # load every file first: duplicates and chain validity are judged across the
    # whole merged input, not per file
    rows, row_root = [], {}
    for path in paths:
        sessions_root = args.sessions_root or os.path.join(
            os.path.dirname(os.path.abspath(path)), "sessions")
        with open(path) as fi:
            for line in fi:
                if line.strip():
                    rows.append(json.loads(line))
                    row_root[id(rows[-1])] = sessions_root
    n_raw = len(rows)
    rows, validity = validate_rows(rows)
    for line in validity_lines(validity):
        print(f"[validity] {line}", file=sys.stderr)
    if not rows:
        print("no valid rows left after the validity filter", file=sys.stderr)
        return 1

    n_in, n_out, missing = 0, 0, set()
    ctx_stats = {"present": 0, "missing_session": 0, "recall_in_ctx": 0,
                 "recall_possible": 0, "nc-chunks": 0, "nc-graph": 0, "nc-note-only": 0,
                 "baseline": 0, "compaction_summary": 0, "truncated": 0}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fo:
        for r in rows:
            sessions_root = row_root[id(r)]
            n_in += 1
            c = chains.get(r.get("chain_id"))
            if c is None:
                missing.add(r.get("chain_id"))
                continue
            qa = next((q for q in c["qa"] if q["qid"] == r.get("qid")), None)
            if qa is None:
                missing.add(f"{r.get('chain_id')}/{r.get('qid')}")
                continue
            idx = int(qa.get("span_idx") or 0)
            span = next((sp for sp in c["spans"] if int(sp["span_idx"]) == idx),
                        c["spans"][0])
            # chains.jsonl spans store only (role, text) segments; the
            # role-tagged join is the canonical span text (same shape as
            # make_chains.seg_text and run_chainreplay.span_text).
            r["question"] = qa["question"]
            if qa.get("qtype") and not r.get("qtype"):
                r["qtype"] = qa["qtype"]       # multi-hop stratum tag (Amendment C)
            r["span_text"] = "".join(
                f"[{s.get('role', '')}] {str(s.get('text') or '').strip()}\n\n"
                for s in span["segments"])
            r.setdefault("rid", raw_rid(r))
            # answer-arm visible context (post-C3 state the model answered from)
            ctx, meta = build_visible_context(
                sessions_root, r.get("chain_id"), r.get("arm"), r.get("qid"),
                cap=args.context_cap,
                budget=(r.get("mem_budget") if r.get("b_tag") else None))
            r["visible_context"] = ctx
            r["visible_context_meta"] = meta
            ctx_stats["present"] += ctx is not None
            ctx_stats["missing_session"] += ctx is None
            ctx_stats["truncated"] += bool(meta.get("truncated"))
            ctx_stats["compaction_summary"] += bool(meta.get("has_compaction_summary"))
            arm = canon_arm(str(r.get("arm")))
            if arm in ("baseline", "nc-chunks", "nc-graph", "nc-note-only"):
                ctx_stats[arm] += 1
                if arm in ("nc-chunks", "nc-graph"):
                    ctx_stats["recall_possible"] += 1
                    ctx_stats["recall_in_ctx"] += bool(meta.get("recall_in_context"))
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1
    if missing:
        print(f"WARNING unmatched records (dropped): {sorted(missing)[:5]} ...", file=sys.stderr)
    print(f"exported {n_out}/{n_in} valid records ({n_raw} read, "
          f"{n_raw - n_in} dropped by the validity filter) -> {args.out}")
    print(f"visible_context: present={ctx_stats['present']} "
          f"missing_ask_session={ctx_stats['missing_session']} "
          f"compaction_summary={ctx_stats['compaction_summary']} "
          f"truncated={ctx_stats['truncated']} | arms: "
          f"baseline={ctx_stats['baseline']} nc-chunks={ctx_stats['nc-chunks']} "
          f"nc-graph={ctx_stats['nc-graph']} "
          f"nc-note-only={ctx_stats['nc-note-only']} | "
          f"recall-in-context(nc-chunks+nc-graph): "
          f"{ctx_stats['recall_in_ctx']}/{ctx_stats['recall_possible']}")
    return 0 if n_out == n_in and n_out else 1


if __name__ == "__main__":
    sys.exit(main())
