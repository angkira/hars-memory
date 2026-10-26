#!/usr/bin/env python3
"""Kill/blame funnel + BM25 comparison for a (merged) chained-replay run dir.

Offline, $0, stdlib only (tiktoken is used only if the run's tokenizer was
tiktoken AND it is importable). Generalizes the gate-212839 funnel
(REPORT-forensics-gate212839.md; funnel_gate212839.py is not in the repo) to
any run_chainreplay.py --out dir or merge_runs.py output:

  python3 funnel_chains.py --run runs/merged-<TS> --chains chains.jsonl \\
      [--graph-store /tmp/opencode/nc-eval/hars-data-chain-glm/sessions]

Per ANSWERABLE question (records.jsonl rows with answerable=true and a
gold_anchor) and per arm, each stage is a token-bounded, case-insensitive anchor
match (the runner's own regex, run_chainreplay.run_asks):

  anchor_in_dropped_context  anchor in the span's dropped context as rebuilt
                             from chains.jsonl (see --framing), cut to
                             --raw-cap chars (neurocompact.ts RAW_CAP=60000)
  anchor_in_dropped_payload  anchor in what memory_compact actually received:
                             the first --head-chars and last --tail-chars of the
                             dropped context when it is longer than head+tail
                             (neurocompact.ts DROP_HALF=12000, slices at
                             convText.slice(0, DROP_HALF) / slice(-DROP_HALF))
  anchor_in_summary          baseline arm: the baseline C3 summary
  anchor_in_note             nc / nc-note-only: the nc C3 note (the fork's note)
  anchor_in_graph            nc / nc-note-only: HARS session store of the nc
                             session_id -- graphml + kv_store_full_entities/
                             relations ("graph"); kv_store_text_chunks/full_docs
                             reported separately as anchor_in_store_chunks
  anchor_in_recall           nc: the records' own anchor_in_recall (computed at
                             run time on the packed injection; authoritative --
                             HARS recall is nondeterministic, never refetch)
  bm25_in_payload            BM25 over the chain's dropped PAYLOADS (all 3
                             spans, i.e. the same dropped_context HARS got),
                             query = question text, filled to the SAME token
                             budget as that qid's packed nc recall (--bm25-budget)
  bm25_in_context            same, over the full dropped contexts (secondary)
  anchor_hit                 the records' anchor_hit (answer contains anchor)

A stage whose input is not in the files prints "n/a (missing: <what>)"; an
arm a stage does not apply to prints "-".

Rules (RUNBOOK-gate.md "Kill/blame funnel"):
  kill graph         HARS anchor_in_recall <= BM25 + 5 pp  (paired over the nc
                     qids where both are defined; BM25 = bm25_in_payload)
  kill architecture  baseline anchor_in_summary >= 85% OR
                     P(anchor_hit | anchor_in_recall) < 70%
  blame extractor    anchor_in_summary <= 60% AND P(hit|recall) >= 85% AND
                     anchor_in_graph < 35% AND chunk-only recall >= 75%
                     (chunk-only recall is not in run files -> n/a unless an
                     available condition already decides NO)

Validity: by default the metrics_v2 B2 filter is applied (chains with any
valid!=true row or any recall_error are dropped whole, then qids without the
full arm triplet); --include-invalid disables it.

Outputs: markdown table (stdout + --out-md, default <run>/funnel.md) and JSON
(--out-json, default <run>/funnel.json) with per-arm / per-arm-lag aggregates,
conditionals, rule verdicts and per-record booleans.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import html
import io
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict

ARMS = ("baseline", "nc", "nc-note-only")
STAGES = ("anchor_in_dropped_context", "anchor_in_dropped_payload", "anchor_in_summary",
          "anchor_in_note", "anchor_in_graph", "anchor_in_store_chunks",
          "anchor_in_recall", "bm25_in_payload", "bm25_in_context", "anchor_hit")
# which arms a stage applies to (others print "-")
APPLIES = {
    "anchor_in_summary": {"baseline"},
    "anchor_in_note": {"nc", "nc-note-only"},
    "anchor_in_graph": {"nc", "nc-note-only"},
    "anchor_in_store_chunks": {"nc", "nc-note-only"},
    "anchor_in_recall": {"nc"},
}
GRAPH_FILES = ("graph_chunk_entity_relation.graphml", "kv_store_full_entities.json",
               "kv_store_full_relations.json")
CHUNK_FILES = ("kv_store_text_chunks.json", "kv_store_full_docs.json")
# run_chainreplay.run_compactions replay prompt (pi turn per segment)
REPLAY_PROMPT = ("SESSION LOG replay - part {i}/{n}. Absorb it as part of your working "
                 "session. Reply with exactly one word: received\n\n<log>\n{text}\n</log>")
STOP = set("""a an and are as at be by did do does for from had has have how in is it its
of on or that the their them then there these this to was were what when where which who
why with you your i we our they he she not no yes""".split())


# ------------------------------------------------------------------ helpers

def anchor_pat(anchor: str):
    """run_chainreplay.run_asks anchor regex (token-bounded, case-insensitive)."""
    return re.compile(r"(?<![0-9A-Za-z_])" + re.escape(anchor) + r"(?![0-9A-Za-z_])", re.I)


def has(anchor: str, texts) -> bool:
    pat = anchor_pat(anchor)
    return any(pat.search(t or "") for t in texts)


def load_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def make_counter(tokenizer: str | None):
    """Same tokenizer the run used when possible (run_chainreplay TOKENIZER)."""
    if tokenizer and tokenizer.startswith("tiktoken"):
        try:
            import tiktoken  # optional, only to match the run's own counts
            enc = tiktoken.get_encoding(tokenizer.split(":", 1)[-1] or "cl100k_base")
            return (lambda t: len(enc.encode(t or ""))), tokenizer, None
        except Exception as e:  # noqa: BLE001
            note = (f"run tokenizer {tokenizer} not importable here ({type(e).__name__}); "
                    "BM25 budget counted with estimator:chars/3.6")
            return (lambda t: int(round(len(t or "") / 3.6))), "estimator:chars/3.6", note
    return (lambda t: int(round(len(t or "") / 3.6))), "estimator:chars/3.6", None


def strings_in(obj) -> list[str]:
    out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            out += strings_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out += strings_in(v)
    return out


def read_store_file(path: str) -> str:
    raw = open(path, encoding="utf-8", errors="replace").read()
    if path.endswith(".graphml"):
        return html.unescape(raw)
    try:
        return "\n".join(strings_in(json.loads(raw)))
    except json.JSONDecodeError:
        return raw


def last_compaction_summary(d: str) -> str:
    """run_chainreplay.last_compaction_summary (pi session file, last compaction)."""
    files = sorted(glob.glob(os.path.join(d, "*.jsonl")))
    if not files:
        return ""
    summary = ""
    with open(files[-1], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "compaction":
                summary = rec.get("summary") or ""
    return summary


# ------------------------------------------------------------------ dropped context

def dropped_context(span: dict, framing: str, raw_cap: int) -> str:
    segs = span.get("segments") or []
    if framing == "raw":
        # run_chainreplay.span_text: the role-tagged join the dataset was validated on
        text = "".join(f"[{s.get('role', '')}] {str(s.get('text') or '').strip()}\n\n"
                       for s in segs)
    else:
        # replay framing: one user turn per segment + the agent's "received"; the
        # exact pi serializeConversation framing is not reproduced (approximation:
        # offsets shift by the per-turn wrapper only)
        n = len(segs)
        text = "".join(
            "[User]: " + REPLAY_PROMPT.format(i=i + 1, n=n, text=s.get("text") or "")
            + "\n\n[Assistant]: received\n\n" for i, s in enumerate(segs))
    return text[:raw_cap] if raw_cap > 0 else text


def dropped_payload(ctx: str, head: int, tail: int) -> list[str]:
    """neurocompact.ts: len > 2*DROP_HALF ? [head, tail] : [whole]."""
    if len(ctx) > head + tail:
        return [ctx[:head], ctx[-tail:] if tail > 0 else ""]
    return [ctx]


# ------------------------------------------------------------------ BM25

def bm25_tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9_]+", (text or "").lower())
    return [t for t in toks if t not in STOP and len(t) > 1]


def chunk_text(text: str, chunk_tokens: int, count) -> list[str]:
    """Line-greedy chunks of <= chunk_tokens; over-long lines are char-split."""
    max_chars = max(64, int(chunk_tokens * 3.6))
    pieces: list[str] = []
    for line in text.splitlines(keepends=True):
        while len(line) > max_chars:
            pieces.append(line[:max_chars])
            line = line[max_chars:]
        if line:
            pieces.append(line)
    chunks, cur, cur_tok = [], "", 0
    for p in pieces:
        t = count(p)
        if cur and cur_tok + t > chunk_tokens:
            chunks.append(cur)
            cur, cur_tok = "", 0
        cur += p
        cur_tok += t
    if cur.strip():
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tf = [Counter(bm25_tokens(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = bm25_tokens(query)
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1)))
            out.append(s)
        return out

    def fill(self, query: str, budget: int, count) -> list[str]:
        """Top-ranked chunks (score > 0) until `budget` tokens; last one truncated."""
        sc = self.scores(query)
        order = sorted((i for i, s in enumerate(sc) if s > 0), key=lambda i: (-sc[i], i))
        out, used = [], 0
        for i in order:
            if used >= budget:
                break
            d = self.docs[i]
            t = count(d)
            if used + t > budget:
                rem = budget - used
                d = d[:max(0, int(len(d) * rem / max(t, 1)))]
                t = rem
            out.append(d)
            used += t
        return out


# ------------------------------------------------------------------ validity

def validate_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """Mirror of metrics_v2.validate_rows (B2) without the duplicate hard-fail
    (duplicates are reported, first row wins -- merge_runs.py refuses them)."""
    bad: dict = defaultdict(set)
    for r in rows:
        cid = str(r.get("chain_id"))
        if r.get("valid") is not True:
            bad[cid].add(f"valid={json.dumps(r.get('valid'))}")
        if r.get("recall_error"):
            bad[cid].add(f"recall_error={str(r['recall_error'])[:40]}")
    kept = [r for r in rows if str(r.get("chain_id")) not in bad]
    have: dict = defaultdict(set)
    for r in kept:
        have[r.get("qid")].add(r.get("arm"))
    incomplete = {q for q, a in have.items() if set(ARMS) - a}
    final = [r for r in kept if r.get("qid") not in incomplete]
    return final, {"rows_in": len(rows), "rows_kept": len(final),
                   "dropped_chains": {c: sorted(w) for c, w in sorted(bad.items())},
                   "incomplete_qids": sorted(map(str, incomplete))}


# ------------------------------------------------------------------ funnel

def na(missing: str) -> dict:
    return {"na": f"missing: {missing}"}


def build(args) -> dict:
    run = os.path.abspath(args.run)
    rows = load_jsonl(os.path.join(run, "records.jsonl"))
    if not rows:
        raise SystemExit(f"no records in {run}/records.jsonl")
    comps = load_jsonl(os.path.join(run, "compactions.jsonl"))
    budgets = load_jsonl(os.path.join(run, "chain_budget.jsonl"))
    dups = [k for k, v in Counter((r.get("qid"), r.get("arm")) for r in rows).items() if v > 1]
    validity = None
    if not args.include_invalid:
        rows, validity = validate_rows(rows)
    seen, uniq = set(), []
    for r in rows:
        k = (r.get("qid"), r.get("arm"))
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    rows = uniq
    tok_name = next((b.get("tokenizer") for b in budgets if b.get("tokenizer")), None) \
        or next((r.get("tokenizer") for r in rows if r.get("tokenizer")), None)
    count, tok_used, tok_note = make_counter(tok_name)

    chains = {}
    if args.chains:
        for c in load_jsonl(args.chains):
            chains[str(c.get("chain_id"))] = c
    questions = {str(q.get("qid")): q for c in chains.values() for q in c.get("qa") or []}

    # ---- per chain: dropped contexts/payloads, summaries/notes, graph, BM25 ----
    per_chain: dict[str, dict] = {}
    for cid in sorted({str(r.get("chain_id")) for r in rows}):
        info: dict = {"missing": {}}
        ch = chains.get(cid)
        if ch:
            spans = {int(s["span_idx"]): s for s in ch.get("spans") or []}
            info["ctx"] = {k: dropped_context(s, args.framing, args.raw_cap)
                           for k, s in spans.items()}
            info["payload"] = {k: dropped_payload(v, args.head_chars, args.tail_chars)
                               for k, v in info["ctx"].items()}
            pay_chunks = [c for k in sorted(info["payload"]) for p in info["payload"][k]
                          for c in chunk_text(p, args.bm25_chunk_tokens, count)]
            ctx_chunks = [c for k in sorted(info["ctx"])
                          for c in chunk_text(info["ctx"][k], args.bm25_chunk_tokens, count)]
            info["bm25_payload"] = BM25(pay_chunks, args.bm25_k1, args.bm25_b)
            info["bm25_context"] = BM25(ctx_chunks, args.bm25_k1, args.bm25_b)
        else:
            why = (f"chains.jsonl entry for {cid}" if args.chains
                   else "--chains (span segments)")
            for st in ("anchor_in_dropped_context", "anchor_in_dropped_payload",
                       "bm25_in_payload", "bm25_in_context"):
                info["missing"][st] = why
        # C3 artifacts: compactions.jsonl summary_path -> file; else pi session file
        for arm, key in (("baseline", "summary"), ("nc", "note")):
            c3 = next((c for c in comps if str(c.get("chain_id")) == cid
                       and c.get("arm") == arm and int(c.get("span_idx") or 0) == 3), None)
            text, tried = None, []
            if c3 and c3.get("summary_path"):
                p = os.path.join(run, c3["summary_path"])
                tried.append(c3["summary_path"])
                if os.path.isfile(p):
                    text = open(p, encoding="utf-8", errors="replace").read()
            elif not c3:
                tried.append(f"compactions.jsonl {arm} span_idx=3 row")
            if text is None:
                sd = os.path.join(run, "sessions", cid, arm)
                tried.append(f"sessions/{cid}/{arm}/*.jsonl")
                s = last_compaction_summary(sd) if os.path.isdir(sd) else ""
                text = s or None
            info[key] = text
            if text is None:
                info["missing"]["anchor_in_" + key] = " and ".join(tried)
        # graph: nc session id -> HARS session dir
        nc_sid = next((r.get("session_id") for r in rows if str(r.get("chain_id")) == cid
                       and r.get("arm") == "nc"), None) or next(
            (c.get("session_id") for c in comps if str(c.get("chain_id")) == cid
             and c.get("arm") == "nc"), None)
        cand = [os.path.join(g, nc_sid) for g in args.graph_store if nc_sid]
        cand += [os.path.join(run, "sessions", nc_sid)] if nc_sid else []
        cand += [os.path.join(run, "sessions", cid, "nc")]
        gfiles = [os.path.join(d, f) for d in cand for f in GRAPH_FILES
                  if os.path.isfile(os.path.join(d, f))]
        cfiles = [os.path.join(d, f) for d in cand for f in CHUNK_FILES
                  if os.path.isfile(os.path.join(d, f))]
        info["nc_session_id"] = nc_sid
        info["graph_files"] = gfiles
        info["chunk_files"] = cfiles
        info["graph_text"] = [read_store_file(p) for p in gfiles]
        info["chunk_text"] = [read_store_file(p) for p in cfiles]
        hint = (f"HARS session store for {nc_sid or '(no nc session_id)'} "
                f"({'/'.join(GRAPH_FILES[:1])} etc.; pass --graph-store <hars-data>/sessions)")
        if not gfiles:
            info["missing"]["anchor_in_graph"] = hint
        if not cfiles:
            info["missing"]["anchor_in_store_chunks"] = hint.replace(
                GRAPH_FILES[0], CHUNK_FILES[0])
        per_chain[cid] = info

    nc_by_qid = {str(r.get("qid")): r for r in rows if r.get("arm") == "nc"}

    # ---- per record ----
    recs = []
    for r in rows:
        if r.get("answerable") is not True or not str(r.get("gold_anchor") or "").strip():
            continue
        arm, cid, qid = r.get("arm"), str(r.get("chain_id")), str(r.get("qid"))
        anchor = str(r["gold_anchor"]).strip()
        info = per_chain[cid]
        miss = info["missing"]
        sp = int(r.get("span_idx") or 0)
        st: dict = {}

        def put(name, fn):
            if name in APPLIES and arm not in APPLIES[name]:
                st[name] = "-"
            elif name in miss:
                st[name] = na(miss[name])
            else:
                st[name] = fn()

        put("anchor_in_dropped_context",
            lambda: has(anchor, [info["ctx"].get(sp, "")]) if sp in info["ctx"]
            else na(f"span {sp} in chains.jsonl"))
        put("anchor_in_dropped_payload",
            lambda: has(anchor, info["payload"].get(sp, [])) if sp in info["payload"]
            else na(f"span {sp} in chains.jsonl"))
        put("anchor_in_summary", lambda: has(anchor, [info["summary"]]))
        put("anchor_in_note", lambda: has(anchor, [info["note"]]))
        put("anchor_in_graph", lambda: has(anchor, info["graph_text"]))
        put("anchor_in_store_chunks", lambda: has(anchor, info["chunk_text"]))
        put("anchor_in_recall",
            lambda: r["anchor_in_recall"] if isinstance(r.get("anchor_in_recall"), bool)
            else na("records.anchor_in_recall"))
        ncr = nc_by_qid.get(qid) or {}
        budget = None
        if args.bm25_budget == "packed":
            budget = ncr.get("recall_packed_tokens") or ncr.get("recall_cap")
        elif args.bm25_budget == "cap":
            budget = ncr.get("recall_cap") or ncr.get("recall_packed_tokens")
        else:
            budget = int(args.bm25_budget)
        qtext = (questions.get(qid) or {}).get("question")
        for name, key in (("bm25_in_payload", "bm25_payload"),
                          ("bm25_in_context", "bm25_context")):
            if name in miss:
                st[name] = na(miss[name])
            elif not qtext:
                st[name] = na(f"question text for {qid} in chains.jsonl")
            elif not budget:
                st[name] = na(f"nc recall_packed_tokens/recall_cap for {qid}")
            else:
                sel = info[key].fill(qtext, int(budget), count)
                st[name] = has(anchor, sel)
        st["anchor_hit"] = (r["anchor_hit"] if isinstance(r.get("anchor_hit"), bool)
                            else na("records.anchor_hit"))
        recs.append({"qid": qid, "chain_id": cid, "arm": arm, "lag": r.get("lag"),
                     "span_idx": sp, "gold_anchor": anchor, "bm25_budget": budget,
                     "stages": st})

    # ---- aggregates ----
    def agg(sel: list[dict]) -> dict:
        out = {}
        for name in STAGES:
            vals = [x["stages"][name] for x in sel]
            bools = [v for v in vals if isinstance(v, bool)]
            nas = sorted({v["na"] for v in vals if isinstance(v, dict)})
            if bools:
                out[name] = {"k": sum(bools), "n": len(bools),
                             "rate": round(sum(bools) / len(bools), 4),
                             **({"partial_na": nas} if nas else {})}
            elif nas:
                out[name] = {"na": "; ".join(nas)}
            else:
                out[name] = "-"
        return out

    per_arm = {a: dict(n=len([x for x in recs if x["arm"] == a]),
                       **agg([x for x in recs if x["arm"] == a]))
               for a in ARMS if any(x["arm"] == a for x in recs)}
    per_arm_lag = {}
    for a in per_arm:
        for lag in sorted({x["lag"] for x in recs if x["arm"] == a}, key=str):
            sel = [x for x in recs if x["arm"] == a and x["lag"] == lag]
            per_arm_lag[f"{a}|lag{lag}"] = dict(n=len(sel), **agg(sel))

    def cond(arm, given, target, val=True):
        sel = [x for x in recs if x["arm"] == arm
               and isinstance(x["stages"].get(given), bool)
               and x["stages"][given] is val and isinstance(x["stages"].get(target), bool)]
        k = sum(1 for x in sel if x["stages"][target])
        return {"k": k, "n": len(sel), "rate": round(k / len(sel), 4) if sel else None}

    conditionals = {
        "nc P(anchor_hit | anchor_in_recall)": cond("nc", "anchor_in_recall", "anchor_hit"),
        "nc P(anchor_hit | NOT anchor_in_recall)": cond("nc", "anchor_in_recall",
                                                        "anchor_hit", False),
        "nc P(anchor_in_recall | anchor_in_dropped_payload)": cond(
            "nc", "anchor_in_dropped_payload", "anchor_in_recall"),
        "nc P(anchor_in_recall | anchor_in_graph)": cond("nc", "anchor_in_graph",
                                                          "anchor_in_recall"),
        "baseline P(anchor_hit | anchor_in_summary)": cond("baseline", "anchor_in_summary",
                                                           "anchor_hit"),
        "baseline P(anchor_hit | NOT anchor_in_summary)": cond(
            "baseline", "anchor_in_summary", "anchor_hit", False),
    }

    # ---- rules ----
    rules = {}
    margin = args.margin_pp / 100.0
    for bm in ("bm25_in_payload", "bm25_in_context"):
        pairs = [(x["stages"]["anchor_in_recall"], x["stages"][bm]) for x in recs
                 if x["arm"] == "nc" and isinstance(x["stages"].get("anchor_in_recall"), bool)
                 and isinstance(x["stages"].get(bm), bool)]
        if not pairs:
            nc = per_arm.get("nc", {})
            why = "; ".join(v["na"] for v in (nc.get("anchor_in_recall"), nc.get(bm))
                            if isinstance(v, dict) and "na" in v) or "no nc answerable rows"
            rules[f"kill_graph[{bm}]"] = {"verdict": "n/a", "why": why}
            continue
        h = sum(p[0] for p in pairs) / len(pairs)
        b = sum(p[1] for p in pairs) / len(pairs)
        kill = h <= b + margin + 1e-12
        rules[f"kill_graph[{bm}]"] = {
            "verdict": "KILL" if kill else "KEEP", "n_paired": len(pairs),
            "hars_anchor_in_recall": round(h, 4), "bm25": round(b, 4),
            "threshold": round(b + margin, 4),
            "both": sum(1 for p in pairs if p[0] and p[1]),
            "hars_only": sum(1 for p in pairs if p[0] and not p[1]),
            "bm25_only": sum(1 for p in pairs if p[1] and not p[0]),
            "neither": sum(1 for p in pairs if not p[0] and not p[1]),
            "rule": f"HARS anchor_in_recall <= BM25 + {args.margin_pp:g} pp"}

    def rate(arm, stage):
        v = per_arm.get(arm, {}).get(stage)
        return v.get("rate") if isinstance(v, dict) else None

    s_sum = rate("baseline", "anchor_in_summary")
    p_hr = conditionals["nc P(anchor_hit | anchor_in_recall)"]["rate"]
    g = rate("nc", "anchor_in_graph")
    ka = [(s_sum, lambda v: v >= 0.85, "anchor_in_summary >= 85%"),
          (p_hr, lambda v: v < 0.70, "P(hit|recall) < 70%")]
    known = [(f(v), t) for v, f, t in ka if v is not None]
    if any(k for k, _ in known):
        rules["kill_architecture"] = {"verdict": "KILL",
                                      "why": [t for k, t in known if k]}
    elif len(known) == len(ka):
        rules["kill_architecture"] = {"verdict": "KEEP", "why": "neither condition holds"}
    else:
        rules["kill_architecture"] = {"verdict": "n/a" if not known else "KEEP?",
                                      "why": "undefined: " + ", ".join(
                                          t for v, _, t in ka if v is None)}
    be = [(s_sum, lambda v: v <= 0.60, "anchor_in_summary <= 60%"),
          (p_hr, lambda v: v >= 0.85, "P(hit|recall) >= 85%"),
          (g, lambda v: v < 0.35, "anchor_in_graph < 35%"),
          (None, None, "chunk-only recall >= 75% (missing: chunk-only recall run)")]
    falses = [t for v, f, t in be if v is not None and not f(v)]
    rules["blame_extractor"] = (
        {"verdict": "NO", "why": "fails: " + ", ".join(falses)} if falses else
        {"verdict": "n/a", "why": "undefined: " + ", ".join(
            t for v, _, t in be if v is None)})

    meta = {"run": run, "chains_file": os.path.abspath(args.chains) if args.chains else None,
            "framing": args.framing, "raw_cap": args.raw_cap, "head_chars": args.head_chars,
            "tail_chars": args.tail_chars, "bm25": {"budget": args.bm25_budget,
                                                     "chunk_tokens": args.bm25_chunk_tokens,
                                                     "k1": args.bm25_k1, "b": args.bm25_b},
            "tokenizer_run": tok_name, "tokenizer_used": tok_used, "tokenizer_note": tok_note,
            "validity": validity, "duplicate_qid_arm": [list(map(str, d)) for d in dups],
            "graph_store": args.graph_store,
            "chains": {c: {"nc_session_id": i["nc_session_id"],
                           "graph_files": i["graph_files"], "chunk_files": i["chunk_files"],
                           "missing": i["missing"]} for c, i in per_chain.items()},
            "answerable_records": len(recs)}
    return {"meta": meta, "per_arm": per_arm, "per_arm_lag": per_arm_lag,
            "conditionals": conditionals, "rules": rules, "records": recs}


# ------------------------------------------------------------------ render

def cell(v) -> str:
    if v == "-" or v is None:
        return "-"
    if "na" in v:
        return f"n/a ({v['na']})"
    s = f"{v['k']}/{v['n']} ({100 * v['rate']:.0f}%)"
    return s + (" +n/a" if v.get("partial_na") else "")


def render(res: dict) -> str:
    m = res["meta"]
    L = ["# Kill/blame funnel + BM25", "",
         f"run: `{m['run']}`  chains: `{m['chains_file']}`  answerable records: "
         f"{m['answerable_records']}",
         f"dropped context: framing={m['framing']} raw_cap={m['raw_cap']} | payload head="
         f"{m['head_chars']} tail={m['tail_chars']} chars | BM25 budget={m['bm25']['budget']} "
         f"chunk={m['bm25']['chunk_tokens']}tok k1={m['bm25']['k1']} b={m['bm25']['b']} | "
         f"tokenizer {m['tokenizer_used']} (run: {m['tokenizer_run']})", ""]
    if m["tokenizer_note"]:
        L += [f"NOTE {m['tokenizer_note']}", ""]
    if m["validity"]:
        v = m["validity"]
        L += [f"validity (metrics_v2 B2 filter): rows kept {v['rows_kept']}/{v['rows_in']}; "
              f"dropped chains: {', '.join(v['dropped_chains']) or 'none'}; "
              f"incomplete qids: {len(v['incomplete_qids'])}", ""]
    if m["duplicate_qid_arm"]:
        L += [f"WARNING duplicate (qid, arm) rows (first kept): {m['duplicate_qid_arm'][:5]}",
              ""]
    head = ["arm", "n"] + list(STAGES)
    L += ["## Per arm (answerable)", "", "| " + " | ".join(head) + " |",
          "|" + "---|" * len(head)]
    for a, d in res["per_arm"].items():
        L.append("| " + " | ".join([a, str(d["n"])] + [cell(d[s]) for s in STAGES]) + " |")
    L += ["", "## Per arm x lag", "", "| " + " | ".join(["arm|lag"] + head[1:]) + " |",
          "|" + "---|" * len(head)]
    for k, d in res["per_arm_lag"].items():
        L.append("| " + " | ".join([k, str(d["n"])] + [cell(d[s]) for s in STAGES]) + " |")
    L += ["", "## Conditionals", ""]
    for k, v in res["conditionals"].items():
        L.append(f"- {k} = {v['k']}/{v['n']}"
                 + (f" ({100 * v['rate']:.0f}%)" if v["rate"] is not None else " (n/a)"))
    L += ["", "## Rules", ""]
    for k, v in res["rules"].items():
        L.append(f"- {k}: **{v['verdict']}** " + json.dumps(
            {x: y for x, y in v.items() if x != "verdict"}, ensure_ascii=False))
    kg = res["rules"].get("kill_graph[bm25_in_payload]", {})
    if kg.get("verdict") in ("KILL", "KEEP"):
        line = (f"VERDICT kill-graph: {kg['verdict']} -- HARS anchor_in_recall "
                f"{100 * kg['hars_anchor_in_recall']:.1f}% "
                f"{'<=' if kg['verdict'] == 'KILL' else '>'} BM25(payload) "
                f"{100 * kg['bm25']:.1f}% + {res['meta'].get('margin_pp', 5):g} pp = "
                f"{100 * kg['threshold']:.1f}% (n={kg['n_paired']} paired nc qids; "
                f"both {kg['both']}, HARS-only {kg['hars_only']}, BM25-only "
                f"{kg['bm25_only']}, neither {kg['neither']})")
    else:
        line = f"VERDICT kill-graph: n/a ({kg.get('why', 'not computed')})"
    L += ["", line]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ selftest

def _selftest_fixture(T: str) -> tuple[str, str, str]:
    """1 chain, 3 spans; anchors: A1 head of span1, A2 middle of span2 (outside the
    head/tail payload), A3 tail of span3. Baseline summary has A1; nc note has A3;
    graph has A1+A3; recall flags A1+A3 (record field); BM25 finds A1 + A3 (the
    question shares terms with the anchor line) but not A2 (outside payload)."""
    run, gs = os.path.join(T, "run"), os.path.join(T, "hars-sessions")
    filler = "\n".join(f"routine log line {i} nothing notable happened here" for i in range(600))
    s1 = "the frobnicator module chosen was zeta_frob_module for probing\n" + filler
    s2 = filler + "\nthe beacon tunnel port was beacon_port_8842 configured\n" + filler
    s3 = filler + "\nthe gizmo checkpoint saved under gizmo_ckpt_tail ok\n"
    spans = [{"span_idx": i + 1, "lag": 3 - i, "segments": [{"role": "out", "text": t}]}
             for i, t in enumerate((s1, s2, s3))]
    qa = [{"qid": "q1", "span_idx": 1, "lag": 3, "question": "Which frobnicator module was chosen for probing?",
           "gold_anchor": "zeta_frob_module", "answerable": True},
          {"qid": "q2", "span_idx": 2, "lag": 2, "question": "What beacon tunnel port was configured?",
           "gold_anchor": "beacon_port_8842", "answerable": True},
          {"qid": "q3", "span_idx": 3, "lag": 1, "question": "Where was the gizmo checkpoint saved?",
           "gold_anchor": "gizmo_ckpt_tail", "answerable": True},
          {"qid": "q0", "span_idx": 0, "lag": None, "question": "What was the colour of X?",
           "gold_anchor": "", "answerable": False}]
    chains = os.path.join(T, "chains.jsonl")
    with open(chains, "w") as f:
        f.write(json.dumps({"chain_id": "cX", "spans": spans, "qa": qa}) + "\n")
    os.makedirs(os.path.join(run, "logs", "cX"))
    open(os.path.join(run, "logs", "cX", "baseline-span3.summary.txt"), "w").write(
        "summary: module zeta_frob_module chosen")
    open(os.path.join(run, "logs", "cX", "nc-span3.summary.txt"), "w").write(
        "note: checkpoint gizmo_ckpt_tail")
    comps = [{"chain_id": "cX", "arm": a, "span_idx": 3, "session_id": f"sid-{a}",
              "summary_path": f"logs/cX/{a}-span3.summary.txt"} for a in ("baseline", "nc")]
    with open(os.path.join(run, "compactions.jsonl"), "w") as f:
        f.write("".join(json.dumps(c) + "\n" for c in comps))
    os.makedirs(os.path.join(gs, "sid-nc"))
    open(os.path.join(gs, "sid-nc", "graph_chunk_entity_relation.graphml"), "w").write(
        "<graphml><node id=\"ZETA_FROB_MODULE\"><data>&quot;zeta_frob_module&quot; and "
        "gizmo_ckpt_tail</data></node></graphml>")
    json.dump({"c1": {"content": "the beacon tunnel port was beacon_port_8842"}},
              open(os.path.join(gs, "sid-nc", "kv_store_text_chunks.json"), "w"))
    recs = []
    in_recall = {"q1": True, "q2": False, "q3": True}
    hit = {("baseline", "q1"): True, ("nc", "q1"): True, ("nc", "q3"): True}
    for q in qa:
        for a in ARMS:
            recs.append({"chain_id": "cX", "qid": q["qid"], "arm": a, "lag": q["lag"],
                         "span_idx": q["span_idx"], "answerable": q["answerable"],
                         "gold_anchor": q["gold_anchor"], "valid": True,
                         "recall_error": None, "session_id": f"sid-{'baseline' if a == 'baseline' else 'nc'}",
                         "anchor_hit": hit.get((a, q["qid"]), False) if q["answerable"] else None,
                         "anchor_in_recall": in_recall.get(q["qid"]) if a == "nc" else None,
                         "recall_packed_tokens": 200 if a == "nc" else 0,
                         "tokenizer": "estimator:chars/3.6"})
    with open(os.path.join(run, "records.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in recs))
    return run, chains, gs


def _main_capture(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def selftest() -> int:
    fails = 0

    def check(name, cond, extra=""):
        nonlocal fails
        print(f"[selftest] {'PASS' if cond else 'FAIL'} {name}")
        if not cond:
            fails += 1
            print("    " + str(extra)[-3000:].replace("\n", "\n    "))

    with tempfile.TemporaryDirectory(prefix="funnel_selftest_") as T:
        run, chains, gs = _selftest_fixture(T)
        oj = os.path.join(T, "f.json")
        rc, out = _main_capture(["--run", run, "--chains", chains, "--graph-store", gs,
                                 "--out-json", oj, "--out-md", os.path.join(T, "f.md")])
        check("exit 0", rc == 0, out)
        res = json.load(open(oj)) if rc == 0 else {}
        st = {(r["arm"], r["qid"]): r["stages"] for r in res.get("records", [])}
        pa = res.get("per_arm", {})
        check("unanswerable excluded (3 answerable x 3 arms)",
              res.get("meta", {}).get("answerable_records") == 9, out)
        check("dropped_context has all 3 anchors",
              pa.get("nc", {}).get("anchor_in_dropped_context", {}).get("k") == 3, out)
        check("payload (12k head/tail) loses the mid-span anchor q2",
              st.get(("nc", "q2"), {}).get("anchor_in_dropped_payload") is False
              and st.get(("nc", "q1"), {}).get("anchor_in_dropped_payload") is True
              and st.get(("nc", "q3"), {}).get("anchor_in_dropped_payload") is True, out)
        check("summary only for baseline (q1 hit, '-' on nc)",
              st.get(("baseline", "q1"), {}).get("anchor_in_summary") is True
              and st.get(("nc", "q1"), {}).get("anchor_in_summary") == "-", out)
        check("note = nc C3 note for nc and nc-note-only (q3)",
              st.get(("nc-note-only", "q3"), {}).get("anchor_in_note") is True
              and st.get(("nc", "q1"), {}).get("anchor_in_note") is False, out)
        check("graph from graphml (q1,q3), chunks store separately (q2)",
              pa.get("nc", {}).get("anchor_in_graph", {}).get("k") == 2
              and st.get(("nc", "q2"), {}).get("anchor_in_store_chunks") is True, out)
        check("recall from records field (2/3)",
              pa.get("nc", {}).get("anchor_in_recall", {}).get("k") == 2, out)
        check("BM25 payload finds q1,q3 not q2; context finds all 3",
              [st.get(("nc", q), {}).get("bm25_in_payload") for q in ("q1", "q2", "q3")]
              == [True, False, True]
              and pa.get("nc", {}).get("bm25_in_context", {}).get("k") == 3, out)
        kg = res.get("rules", {}).get("kill_graph[bm25_in_payload]", {})
        check("kill-graph KILL when HARS 67% <= BM25 67% + 5pp",
              kg.get("verdict") == "KILL" and kg.get("n_paired") == 3, kg)
        check("verdict line printed", "VERDICT kill-graph: KILL" in out, out)
        check("P(hit|recall) = 2/2",
              res.get("conditionals", {}).get("nc P(anchor_hit | anchor_in_recall)",
                                              {}).get("k") == 2, out)

        # missing inputs -> n/a with reasons, no crash
        rc, out2 = _main_capture(["--run", run, "--out-json", oj,
                                  "--out-md", os.path.join(T, "g.md")])
        res2 = json.load(open(oj)) if rc == 0 else {}
        nc2 = res2.get("per_arm", {}).get("nc", {})
        check("no --chains / --graph-store -> n/a (missing: ...)", rc == 0
              and "missing: --chains" in str(nc2.get("anchor_in_dropped_context"))
              and "missing: HARS session store" in str(nc2.get("anchor_in_graph"))
              and "VERDICT kill-graph: n/a" in out2, out2)

        # validity filter drops a recall_error chain
        rows = load_jsonl(os.path.join(run, "records.jsonl"))
        rows[1]["recall_error"] = "530"
        with open(os.path.join(run, "records.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        rc, out3 = _main_capture(["--run", run, "--chains", chains, "--out-json", oj,
                                  "--out-md", os.path.join(T, "h.md")])
        check("recall_error chain dropped by the validity filter", rc == 2
              and "no answerable records" in out3, out3)
    print(f"[selftest] {'OK' if not fails else f'{fails} FAILURE(S)'}")
    return 1 if fails else 0


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run dir (run_chainreplay --out or merge_runs --out)")
    ap.add_argument("--chains", default="",
                    help="chains.jsonl (span segments + question text); without it the "
                         "dropped-context and BM25 stages are n/a")
    ap.add_argument("--graph-store", action="append", default=[],
                    help="HARS sessions dir holding <session_id>/graph_chunk_entity_"
                         "relation.graphml + kv_store_*.json (repeatable; <run>/sessions "
                         "is always searched too)")
    ap.add_argument("--head-chars", type=int, default=12000,
                    help="payload head chars sent to memory_compact (neurocompact.ts "
                         "DROP_HALF, default 12000)")
    ap.add_argument("--tail-chars", type=int, default=12000,
                    help="payload tail chars (default 12000)")
    ap.add_argument("--raw-cap", type=int, default=60000,
                    help="dropped-context char cap (neurocompact.ts RAW_CAP, default "
                         "60000; 0 = none)")
    ap.add_argument("--framing", choices=("replay", "raw"), default="replay",
                    help="rebuild dropped context with the runner's replay-turn wrapper "
                         "(default) or as the role-tagged span_text join")
    ap.add_argument("--bm25-budget", default="packed",
                    help="BM25 fill budget per qid: 'packed' = that qid's nc "
                         "recall_packed_tokens (default), 'cap' = recall_cap, or an int")
    ap.add_argument("--bm25-chunk-tokens", type=int, default=128)
    ap.add_argument("--bm25-k1", type=float, default=1.2)
    ap.add_argument("--bm25-b", type=float, default=0.75)
    ap.add_argument("--margin-pp", type=float, default=5.0,
                    help="kill-graph margin in percentage points (RUNBOOK: 5)")
    ap.add_argument("--include-invalid", action="store_true",
                    help="skip the metrics_v2 validity filter")
    ap.add_argument("--out-md", default="", help="default <run>/funnel.md")
    ap.add_argument("--out-json", default="", help="default <run>/funnel.json")
    ap.add_argument("--selftest", action="store_true", help="synthetic self test")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.run:
        ap.error("--run is required")
    if args.bm25_budget not in ("packed", "cap"):
        try:
            int(args.bm25_budget)
        except ValueError:
            ap.error("--bm25-budget must be packed, cap or an integer")
    res = build(args)
    res["meta"]["margin_pp"] = args.margin_pp
    if not res["records"]:
        print("!!! no answerable records left (validity filter / no gold anchors)",
              file=sys.stderr)
        print(json.dumps(res["meta"].get("validity"), indent=1), file=sys.stderr)
        return 2
    md = render(res)
    print(md)
    out_md = args.out_md or os.path.join(args.run, "funnel.md")
    out_json = args.out_json or os.path.join(args.run, "funnel.json")
    with open(out_md, "w", encoding="utf-8") as w:
        w.write(md)
    with open(out_json, "w", encoding="utf-8") as w:
        json.dump(res, w, ensure_ascii=False, indent=1)
    print(f"[written] {out_md} and {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
