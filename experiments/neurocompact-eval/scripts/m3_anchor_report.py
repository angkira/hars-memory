#!/usr/bin/env python3
"""M3 size report ($0, offline): how many gold anchors reach the persisted compaction documents.

M3 (HANDOFF.md OPEN PROTOCOL ISSUES; Protocol Amendment C item 8): before the fix,
neurocompact.ts sent only the first and last 12 000 chars of a dropped span (> 24 000 chars)
to memory_compact, while the baseline summarizer saw up to RAW_CAP = 60 000. This script
measures, per answerable item, whether the gold anchor

  in_span          is in the dataset span it was drawn from (always true by the EVID gate;
                   the denominator)
  in_compaction_docs  is in the PERSISTED compaction documents of the nc session
                   (`kv_store_full_docs.json`, doc ids `session:<sid>:c<seq>`), i.e. what
                   HARS actually ingested. This is the ground truth.
  in_headtail_sim  would survive the OLD head+tail cut, simulated on the role-tagged span
                   text (APPROXIMATE: the live dropped_context also carries the replay
                   wrappers and tool-result truncation of serializeConversation)

Run it on the archived session dirs of a pre-fix run to size M3, and on a post-fix run to
confirm the fix (in_compaction_docs ~ in_span). Needs no network, no API key.

Usage:
  python3 m3_anchor_report.py --chains chains.jsonl \
      --sessions-root /tmp/opencode/nc-eval/hars-data-chain-glm/sessions \
      [--run runs/<RUN>]            # maps chain -> nc session id via compactions.jsonl
      [--out m3_report.md] [--json m3_report.json]
  python3 m3_anchor_report.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from judge_v2 import anchor_re  # noqa: E402  (identical anchor semantics as the metrics)

DROP_HALF = 12000
NC_ARMS = ("nc-chunks", "nc")          # `nc` = the pre-amendment name
DOC_ID_RE = re.compile(r"^session:(?P<sid>.+):c(?P<seq>\d+)$")


def span_text(span: dict) -> str:
    """Role-tagged span text (same join as run_chainreplay.span_text / make_chains)."""
    return "".join(f"[{s.get('role', '')}] {str(s.get('text') or '').strip()}\n\n"
                   for s in span["segments"])


def headtail(text: str) -> str:
    """The pre-fix persistence window of one span (simulated)."""
    return text if len(text) <= 2 * DROP_HALF else text[:DROP_HALF] + "\n" + text[-DROP_HALF:]


def load_docs(session_dir: str) -> dict[str, str]:
    """doc_id -> content of the session's persisted compaction documents."""
    p = os.path.join(session_dir, "kv_store_full_docs.json")
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8", errors="replace") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError:
            return {}
    out: dict[str, str] = {}
    for k, v in (raw.items() if isinstance(raw, dict) else []):
        content = v.get("content") if isinstance(v, dict) else v
        if isinstance(content, str):
            out[str(k)] = content
    return out


def nc_session_ids(run: str | None) -> dict[str, str]:
    """chain_id -> nc(-chunks) HARS session id from a run dir's compactions.jsonl."""
    out: dict[str, str] = {}
    if not run:
        return out
    p = os.path.join(run, "compactions.jsonl")
    if not os.path.isfile(p):
        return out
    with open(p) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("arm") in NC_ARMS and r.get("session_id") and r.get("chain_id"):
                key = str(r["chain_id"]) + (f"@b{r['mem_budget']}" if r.get("b_tag") else "")
                out.setdefault(str(r["chain_id"]), str(r["session_id"]))
                out.setdefault(key, str(r["session_id"]))
    return out


def find_session_dir(root: str, chain_id: str, sids: dict[str, str]) -> tuple[str | None, str]:
    sid = sids.get(chain_id)
    if sid and os.path.isdir(os.path.join(root, sid)):
        return os.path.join(root, sid), sid
    cands = sorted(glob.glob(os.path.join(root, f"*{chain_id}*nc*")))
    if cands:
        return cands[0], os.path.basename(cands[0])
    return None, sid or ""


def analyse(chains: list[dict], sessions_root: str, run: str | None = None) -> dict:
    sids = nc_session_ids(run)
    rows, per_chain = [], {}
    for c in chains:
        cid = c["chain_id"]
        sdir, sid = find_session_dir(sessions_root, cid, sids)
        docs = load_docs(sdir) if sdir else {}
        blob = "\n".join(docs.values())
        spans = {int(sp["span_idx"]): span_text(sp) for sp in c["spans"]}
        per_chain[cid] = {"session_id": sid, "session_dir": sdir, "n_docs": len(docs),
                          "doc_chars": len(blob), "found": bool(docs)}
        for q in c.get("qa", []):
            anchor = str(q.get("gold_anchor") or "").strip()
            if not anchor or q.get("answerable") is False:
                continue
            idx = int(q.get("span_idx") or 0)
            txt = spans.get(idx, "")
            pat = anchor_re(anchor)
            pos = pat.search(txt)
            rows.append({
                "chain_id": cid, "qid": q["qid"], "span_idx": idx, "slice": q.get("slice"),
                "qtype": q.get("qtype"), "anchor": anchor, "span_chars": len(txt),
                "anchor_offset": pos.start() if pos else None,
                "in_span": pos is not None,
                "in_headtail_sim": bool(pat.search(headtail(txt))) if pos else None,
                "in_compaction_docs": (bool(pat.search(blob)) if docs else None),
            })

    def rate(key, only=None):
        xs = [r[key] for r in rows if r[key] is not None and (only is None or only(r))]
        return (sum(1 for x in xs if x) / len(xs), len(xs)) if xs else (None, 0)

    long_span = lambda r: r["span_chars"] > 2 * DROP_HALF   # noqa: E731
    return {
        "rows": rows, "per_chain": per_chain,
        "summary": {
            "n_items": len(rows),
            "n_chains_with_docs": sum(1 for v in per_chain.values() if v["found"]),
            "in_span": rate("in_span"),
            "in_headtail_sim": rate("in_headtail_sim"),
            "in_compaction_docs": rate("in_compaction_docs"),
            "in_headtail_sim_long_spans": rate("in_headtail_sim", long_span),
            "in_compaction_docs_long_spans": rate("in_compaction_docs", long_span),
            "n_long_spans_items": sum(1 for r in rows if long_span(r)),
        },
    }


def render(res: dict) -> str:
    s = res["summary"]

    def f(t):
        v, n = t
        return "n/a" if v is None else f"{v:.3f} ({round(v * n)}/{n})"

    L = ["# M3 size report: gold anchors in the persisted compaction documents", "",
         f"items: {s['n_items']} | chains with persisted docs: {s['n_chains_with_docs']}/"
         f"{len(res['per_chain'])} | items on spans > {2 * DROP_HALF} chars: "
         f"{s['n_long_spans_items']}", "",
         "| measure | all items | spans > 24k chars |", "|---|---|---|",
         f"| anchor in the dataset span (denominator) | {f(s['in_span'])} | |",
         f"| anchor survives the OLD head+tail cut (simulation, approximate) | "
         f"{f(s['in_headtail_sim'])} | {f(s['in_headtail_sim_long_spans'])} |",
         f"| anchor in the persisted compaction docs (ground truth) | "
         f"{f(s['in_compaction_docs'])} | {f(s['in_compaction_docs_long_spans'])} |", "",
         "## per chain", "", "| chain | session | docs | doc chars |", "|---|---|---|---|"]
    for cid, v in res["per_chain"].items():
        L.append(f"| {cid} | {v['session_id'] or '-'} | {v['n_docs'] if v['found'] else 'NOT FOUND'}"
                 f" | {v['doc_chars']} |")
    miss = [r for r in res["rows"] if r["in_compaction_docs"] is False]
    L += ["", f"## anchors absent from the persisted docs ({len(miss)})", "",
          "| qid | span | slice | span chars | anchor offset | in head+tail sim | anchor |",
          "|---|---|---|---|---|---|---|"]
    for r in miss:
        L.append(f"| {r['qid']} | {r['span_idx']} | {r['slice']} | {r['span_chars']} | "
                 f"{r['anchor_offset']} | {r['in_headtail_sim']} | {r['anchor'][:40]} |")
    return "\n".join(L) + "\n"


def selftest() -> int:
    """Offline fixture: a long span with a mid-span anchor, a head anchor, and a doc store
    holding only head+tail (pre-fix) vs everything (post-fix)."""
    filler = lambda n, tag: (f"{tag} lorem ipsum dolor sit amet. " * 400)[:n]   # noqa: E731
    head, mid, tail = "HEADANCH_aa11", "MIDANCH_bb22", "TAILANCH_cc33"
    text = f"{head} {filler(14000, 'a')} {mid} {filler(14000, 'b')} {tail}"
    seg = {"role": "assistant", "text": text}
    chain = {"chain_id": "chain_T_c0", "session_id": "T", "chain_idx": 0,
             "spans": [{"span_idx": i, "lag": 4 - i, "segments": [seg]} for i in (1, 2, 3)],
             "qa": [{"qid": f"q_{n}", "chain_id": "chain_T_c0", "span_idx": 1, "slice": sl,
                     "gold_anchor": a, "answerable": True}
                    for n, (a, sl) in enumerate(((head, "HEAD"), (mid, "MID"), (tail, "TAIL")))]}
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for tag, content in (("pre", headtail(span_text(chain["spans"][0]))),
                             ("post", span_text(chain["spans"][0]))):
            sd = os.path.join(td, tag, "cr-x-chain_T_c0-nc-chunks-r1")
            os.makedirs(sd)
            json.dump({"session:cr-x-chain_T_c0-nc-chunks-r1:c1": {"content": content}},
                      open(os.path.join(sd, "kv_store_full_docs.json"), "w"))
            res = analyse([chain], os.path.join(td, tag))
            got = {r["anchor"]: r["in_compaction_docs"] for r in res["rows"]}
            sim = {r["anchor"]: r["in_headtail_sim"] for r in res["rows"]}
            exp = ({head: True, mid: False, tail: True} if tag == "pre"
                   else {head: True, mid: True, tail: True})
            good = got == exp and sim == {head: True, mid: False, tail: True}
            ok &= good
            print(f"[m3-selftest] {'PASS' if good else 'FAIL'} {tag}: docs={got} sim={sim}")
        txt = render(res)
        ok &= "NOT FOUND" not in txt
    print(f"[m3-selftest] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains")
    ap.add_argument("--sessions-root", default="/tmp/opencode/nc-eval/hars-data-chain-glm/sessions")
    ap.add_argument("--run", default=None, help="run dir with compactions.jsonl (chain -> nc sid)")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.chains:
        ap.error("--chains is required")
    with open(a.chains) as f:
        chains = [json.loads(x) for x in f if x.strip()]
    res = analyse(chains, a.sessions_root, a.run)
    md = render(res)
    print(md)
    if a.out:
        open(a.out, "w").write(md)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
