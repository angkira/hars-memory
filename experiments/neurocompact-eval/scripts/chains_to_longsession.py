#!/usr/bin/env python3
"""OpenHands chains.jsonl -> longsession/1 sessions (Amendment C.6).

Why: the OpenHands track no longer replays a chain through a pi agent turn by turn
(run_chainreplay). It runs through `run_longsession.py` exactly like BEAM and Claude Code:
at every compaction boundary the baseline summariser, and HARS memory_compact + the short
note (nc arms), see the dropped span; the questions are asked at the END with each arm's
memory (+ the same verbatim tail, which is empty here, as in the chain replay).

INPUT  only `chains.jsonl` (make_chains.py format). A chain record already carries the raw
       text: `spans[k].segments = [{role, text}]` are the original session EVENTS of the
       replayed window of span k (the very text the old replay sent to the agent and the
       text the questions were verified against). The imported OpenHands session JSONs
       (import_openhands output, `sessions/` of the dataset) are therefore NOT needed and
       are not read; nothing outside `chains.jsonl` enters the conversion.

OUTPUT one `longsession/1` file per VALID chain (`<out>/<chain_id>.json`) + `<out>/.conversion.json`:
  sid          = chain_id (so qids, chain ids, exclusion regexes and judge/metrics/funnel joins
                 stay exactly the chain track's)
  messages     = the events of span 1, then span 2, then span 3 (empty texts skipped like
                 make_longreplay_dataset.split_segments; text stripped; roles user / assistant /
                 tool / system kept, anything else -> `tool` when it looks like a tool block
                 else `assistant`)
  compactions  = the chain's (synthetic) boundaries, one per span end: at_idx = first message of
                 the next span, and at_idx = len(messages) for C3 (nothing survives: no tail).
                 `original_summary: null` (OpenHands has no real summaries; no `orig` arm).
                 Run with `--use-real-compactions --keep-tail 0` so these ARE the boundaries.
                 After C3 the lags are exactly the chain's: span 1 -> 3, span 2 -> 2, span 3 -> 1.
  questions    = every qa item, qid unchanged:
                 qtype  canon_qtype(stored): stored `multihop` -> `xspan_locator` (Amendment C.2),
                        `source_qtype` keeps the stored label;
                 answer = canonical_short_answer (unanswerable probes: the standard
                        "not mentioned" sentence), answerable / gold_anchor passed through;
                 evidence = char-exact positions of the item's verbatim evidence snippet(s) inside
                        the message(s) that carry them: [{msg_idx, start, end}] (one entry per
                        overlapping message when a snippet crosses event edges); multihop items
                        use `evidence_spans` (link hop + answer hop, each in its own span).
                        A snippet that cannot be located falls back to the messages of its span
                        that contain the gold / link anchor (noted in .conversion.json), and
                        to a conversion ERROR if even that fails (never a silent drop).
                 `slice` (HEAD/TAIL/NONE) is kept top level (records and metrics slices read it);
                 the rest of the chain item (span_idx, lag, hop_lags, span_idxs, link_anchor, ...)
                 is under `meta`.

Deterministic: the same chains.jsonl always yields byte-identical files (sorted keys, no clock).
`.conversion.json` carries sha256 of the input and of every output file.

Usage:
  python3 chains_to_longsession.py --chains <oh artifact>/chains.jsonl --out <dir> [--include CH,CH] [--exclude CH]
  python3 run_longsession.py --sessions <dir> --use-real-compactions --keep-tail 0 --checkpoints end ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import longsession as L  # noqa: E402
import qtypes as QT  # noqa: E402

CONVERTER_VERSION = "c6.1"
# hidden: longsession.load_sessions() reads every non-hidden *.json of a directory as a session
MANIFEST_NAME = ".conversion.json"
NOT_MENTIONED = "This is not mentioned anywhere in the session."
SOURCE = "openhands-chain"


class ConversionError(ValueError):
    """A chain that cannot be converted faithfully (message names chain and item)."""


def _role(e: dict) -> str:
    r = str(e.get("role") or "").strip().lower()
    if r in L.ROLES:
        return r
    return "tool" if str(e.get("text") or "").lstrip().startswith("[tool") else "assistant"


def _ws_regex(snippet: str) -> re.Pattern | None:
    toks = (snippet or "").split()
    if not toks:
        return None
    return re.compile(r"\s+".join(re.escape(t) for t in toks))


def _anchor_regex(anchor: str) -> re.Pattern | None:
    a = (anchor or "").strip()
    if not a:
        return None
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(a) + r"(?![A-Za-z0-9_])", re.I)


def build_messages(chain: dict) -> tuple[list[dict], list[dict]]:
    """(messages, span_table). span_table[k] = {span_idx, lag, start, end} message index range [start, end)."""
    msgs: list[dict] = []
    table: list[dict] = []
    spans = sorted(chain.get("spans") or [], key=lambda s: int(s["span_idx"]))
    if not spans:
        raise ConversionError(f"{chain.get('chain_id')}: no spans")
    for sp in spans:
        start = len(msgs)
        for e in sp.get("segments") or []:
            t = (e.get("text") or "").strip()
            if not t:
                continue
            msgs.append({"idx": len(msgs), "role": _role(e), "text": t})
        if len(msgs) == start:
            raise ConversionError(f"{chain.get('chain_id')}: span {sp['span_idx']} has no text")
        table.append({"span_idx": int(sp["span_idx"]), "lag": sp.get("lag"), "start": start, "end": len(msgs)})
    return msgs, table


def locate(msgs: list[dict], lo: int, hi: int, snippet: str):
    """Evidence entries for `snippet` inside messages[lo:hi]: [{msg_idx, start, end}] or None.
    The snippet is matched whitespace-insensitively on the role-tagged, concatenated text (the form
    make_chains verified it against), then cut back to the per-message character ranges."""
    rx = _ws_regex(snippet)
    if rx is None:
        return None
    parts, offs, pos = [], [], 0
    for m in msgs[lo:hi]:
        head = f"[{m['role']}] "
        offs.append(pos + len(head))                      # where this message's TEXT starts in `joined`
        piece = head + m["text"] + "\n\n"
        parts.append(piece)
        pos += len(piece)
    joined = "".join(parts)
    hit = rx.search(joined)
    if not hit:
        return None
    a, b = hit.span()
    out = []
    for k, m in enumerate(msgs[lo:hi]):
        t0, t1 = offs[k], offs[k] + len(m["text"])
        s, e = max(a, t0), min(b, t1)
        if e > s:
            out.append({"msg_idx": lo + k, "start": s - t0, "end": e - t0})
    return out or None


def _fallback(msgs, lo, hi, anchor):
    rx = _anchor_regex(anchor)
    if rx is None:
        return None
    for k in range(lo, hi):
        mt = rx.search(msgs[k]["text"])
        if mt:
            return [{"msg_idx": k, "start": mt.start(), "end": mt.end()}]
    return None


def _evidence_for(chain, item, msgs, table, notes: list[str]) -> list[dict]:
    cid, qid = chain["chain_id"], item["qid"]
    by_span = {t["span_idx"]: t for t in table}
    hops = item.get("evidence_spans")
    if not hops:
        hops = [{"span_idx": item.get("span_idx"), "evidence": item.get("evidence"),
                 "anchor": item.get("gold_anchor"), "role": "answer"}]
    ev: list[dict] = []
    for h in hops:
        t = by_span.get(int(h.get("span_idx") or 0))
        if t is None:
            raise ConversionError(f"{cid}/{qid}: evidence span_idx {h.get('span_idx')!r} not in the chain")
        got = locate(msgs, t["start"], t["end"], h.get("evidence") or "")
        if got is None:
            for t2 in table:                                 # the snippet may sit in another span of the chain
                got = locate(msgs, t2["start"], t2["end"], h.get("evidence") or "")
                if got:
                    notes.append(f"{qid}: evidence found in span {t2['span_idx']} not {t['span_idx']}")
                    break
        if got is None:
            got = _fallback(msgs, t["start"], t["end"], h.get("anchor") or "")
            if got:
                notes.append(f"{qid}: evidence_fallback(anchor {h.get('anchor')!r}, span {t['span_idx']})")
        if got is None:
            raise ConversionError(f"{cid}/{qid}: evidence of hop {h.get('role')!r} (span {h.get('span_idx')}) "
                                  "is neither located verbatim nor by its anchor")
        ev.extend(got)
    seen, out = set(), []
    for e in ev:                                             # dedup, keep order
        k = (e["msg_idx"], e["start"], e["end"])
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def convert_question(chain, item, msgs, table, notes) -> dict:
    qid = item["qid"]
    answerable = item.get("answerable", True) is not False
    stored = item.get("qtype") or "single"
    q = {"qid": qid, "qtype": QT.canon_qtype(stored), "source_qtype": stored,
         "question": (item.get("question") or "").strip(), "answerable": answerable}
    if answerable:
        q["answer"] = (item.get("canonical_short_answer") or "").strip()
        if not q["answer"]:
            raise ConversionError(f"{chain['chain_id']}/{qid}: answerable item without canonical_short_answer")
        q["evidence"] = _evidence_for(chain, item, msgs, table, notes)
        if (item.get("gold_anchor") or "").strip():
            q["gold_anchor"] = item["gold_anchor"].strip()
    else:
        q["answer"] = (item.get("canonical_short_answer") or "").strip() or NOT_MENTIONED
        q["evidence"] = []
    if item.get("slice"):
        q["slice"] = item["slice"]
    meta = {k: item[k] for k in ("span_idx", "lag", "hop_lags", "span_idxs", "link_anchor", "must_abstain",
                                 "mh_version") if k in item}
    meta["stored_qtype"] = stored
    q["meta"] = meta
    return q


def convert_chain(chain: dict) -> tuple[dict, list[str]]:
    """One valid chain record -> (longsession/1 session, notes). Raises ConversionError."""
    cid = chain.get("chain_id")
    if not isinstance(cid, str) or not L.SID_RE.match(cid):
        raise ConversionError(f"bad chain_id {cid!r} (need [A-Za-z0-9._-]{{1,64}})")
    msgs, table = build_messages(chain)
    notes: list[str] = []
    n = len(msgs)
    comps = []
    for k, t in enumerate(table):
        at = table[k + 1]["start"] if k + 1 < len(table) else n
        comps.append({"at_idx": at, "original_summary": None, "auto": False, "synthetic": True,
                      "span_idx": t["span_idx"]})
    qs = [convert_question(chain, it, msgs, table, notes) for it in chain.get("qa") or []]
    qids = [q["qid"] for q in qs]
    if len(set(qids)) != len(qids):
        raise ConversionError(f"{cid}: duplicate qids")
    sess = {"format": L.FORMAT, "sid": cid, "source": SOURCE,
            "title": f"{chain.get('session_id')} chain {chain.get('chain_idx')}",
            "meta": {"origin": "chains_to_longsession", "converter": CONVERTER_VERSION,
                     "session_id": chain.get("session_id"), "chain_idx": chain.get("chain_idx"),
                     "spans": [dict(t) for t in table], "synthetic_boundaries": True,
                     "qa_model": chain.get("qa_model"), "build_flags": chain.get("build_flags") or {}},
            "messages": msgs, "compactions": comps, "questions": qs}
    bad = L.validate_session(sess)
    if bad:
        raise ConversionError(f"{cid}: converted session is invalid: {bad[:3]}")
    return sess, notes


def read_chains(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as e:
                    raise ConversionError(f"{path}:{ln}: not JSON ({e})") from e
    return rows


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_session(sess: dict, out_dir: str) -> str:
    """Deterministic file (sorted keys, no clock): unlike L.save_session, byte-stable across runs."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{sess['sid']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sess, f, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return path


def convert_file(chains_path: str, out_dir: str, include=None, exclude=None) -> dict:
    chains = read_chains(chains_path)
    inc, exc = set(include or []), set(exclude or [])
    manifest = {"converter": CONVERTER_VERSION, "chains_file": os.path.abspath(chains_path),
                "chains_sha256": sha256_file(chains_path), "chains_total": len(chains), "sessions": [],
                "skipped": [], "errors": [], "include": sorted(inc), "exclude": sorted(exc)}
    seen_sessions: dict[str, list[str]] = {}
    for c in chains:
        cid = c.get("chain_id")
        if inc and cid not in inc:
            manifest["skipped"].append({"chain_id": cid, "why": "not in --include"})
            continue
        if cid in exc:
            manifest["skipped"].append({"chain_id": cid, "why": "excluded"})
            continue
        if c.get("valid") is False:
            manifest["skipped"].append({"chain_id": cid, "why": "chain valid=false"})
            continue
        try:
            sess, notes = convert_chain(c)
        except ConversionError as e:
            manifest["errors"].append({"chain_id": cid, "error": str(e)})
            continue
        path = write_session(sess, out_dir)
        seen_sessions.setdefault(str(c.get("session_id")), []).append(cid)
        tok = L.message_tokens(sess["messages"])
        qs = sess["questions"]
        manifest["sessions"].append({
            "sid": cid, "file": os.path.basename(path), "sha256": sha256_file(path),
            "messages": len(sess["messages"]), "tokens": sum(tok), "boundaries": [x["at_idx"] for x in sess["compactions"]],
            "questions": len(qs), "answerable": sum(1 for q in qs if q["answerable"]),
            "unanswerable": sum(1 for q in qs if not q["answerable"]),
            "qtypes": {t: sum(1 for q in qs if q["qtype"] == t) for t in sorted({q["qtype"] for q in qs})},
            "notes": notes})
    manifest["shared_session_chains"] = {s: v for s, v in seen_sessions.items() if len(v) > 1}
    manifest["tokenizer"] = L.TOKENIZER
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", required=True, help="make_chains / OH artifact chains.jsonl")
    ap.add_argument("--out", required=True, help="output dir for the longsession/1 files")
    ap.add_argument("--include", default="", help="comma list of chain ids (default: all)")
    ap.add_argument("--exclude", default="", help="comma list of chain ids to leave out")
    a = ap.parse_args(argv)
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    try:
        m = convert_file(a.chains, a.out, split(a.include), split(a.exclude))
    except (ConversionError, OSError) as e:
        print(f"!!! {e}", file=sys.stderr)
        return 2
    for e in m["errors"]:
        print(f"[error] {e['chain_id']}: {e['error']}", file=sys.stderr)
    for s, v in m["shared_session_chains"].items():
        print(f"WARNING chains {v} share OpenHands session {s}: they are clustered separately (cluster = sid)",
              file=sys.stderr)
    n_q = sum(s["questions"] for s in m["sessions"])
    print(f"converted {len(m['sessions'])}/{m['chains_total']} chain(s), {n_q} question(s) -> {a.out} "
          f"(skipped {len(m['skipped'])}, errors {len(m['errors'])}; tokenizer {m['tokenizer']})")
    return 1 if m["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
