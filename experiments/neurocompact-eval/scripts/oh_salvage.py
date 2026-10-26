#!/usr/bin/env python3
"""Salvage assembly of single-hop chains from the raw generator logs of dropped `make` runs.

make_chains.main() drops a WHOLE chain when any one of its 7 slots stays invalid after the
repair rounds, and throws the 3-6 valid items away (2026-10-26, OpenHands set: 14/14 chains
dropped with 1-5 bad slots each, three attempts, $0.63).  This tool keeps every generator
output that was ever produced (--raw files from `make --save-raw`), re-validates each
candidate with the SAME gates (make_chains.validate + the oh_pipeline contamination guard +
unique-probe / anchor-uniqueness rules), fills the 7 slots of each chain from the valid
candidates and only pays for the still-empty slots (make_chains.repair_prompt, <= --rounds
repair calls per chain).  Nothing is relaxed: a salvaged item satisfies exactly the gates of a
directly accepted one; the per-item provenance (raw file + call number, or repair call) is
stored in the `salvage` key of the manifest.

Slot labelling: an answerable candidate fills the (span, HEAD|TAIL) slot of the half its
evidence ACTUALLY sits in (the generator's own slice label is not trusted), so the slice
label is true by construction.

Usage (through oh_pipeline.py, which installs the guard, probes and the Anthropic adapter):
  python3 -I oh_pipeline.py --qa-provider anthropic --qa-thinking disabled --sessions-dir DIR \\
      salvage --raw raw1.jsonl --raw raw2.jsonl --out chains-single.jsonl \\
      --manifest chains-single-manifest.json --budget-usd 0.15 --save-raw raw-salvage.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import make_chains as mc  # noqa: E402
import multihop  # noqa: E402
import py_probes  # noqa: E402   (window_table: shared id -> verbatim window resolution)

SLOTS = [(k, sl) for k in (1, 2, 3) for sl in ("HEAD", "TAIL")] + ["U"]


def half_of(span_text: str, evidence: str) -> str | None:
    head, tail = mc.halves(span_text)
    if mc.has_verbatim(tail, evidence):
        return "TAIL"
    if mc.has_verbatim(head, evidence):
        return "HEAD"
    return None


def try_fill(chain: dict, spans: list, transcript: str, filled: dict, it: dict,
             used_probes: dict, anchors_seen: dict, prov: dict) -> str | None:
    """Validate candidate `it` against `chain`; store it in `filled` if it fits an open slot.
    Returns the slot key or None."""
    if it["kind"] == "unanswerable":
        if "U" in filled:
            return None
        it = dict(it, span_idx=0, slice="NONE")
        ok, why, checks = mc.validate(it, spans, transcript)
        toks = mc.probe_tokens(it["question"])
        if not ok or any(t.lower() in used_probes for t in toks) \
                or not (len(toks) == 1 and py_probes.fits(transcript, toks[0])):
            return None
        filled["U"] = (it, checks, prov)
        return "U"
    k = it.get("span_idx")
    if k not in (1, 2, 3):
        return None
    span = next(sp for sp in spans if sp["span_idx"] == k)
    sl = half_of(span["text"], it["evidence"] or "")
    if sl is None or (k, sl) in filled:
        return None
    it = dict(it, slice=sl)
    ok, why, checks = mc.validate(it, spans, transcript)
    a = it["gold_anchor"].strip().lower()
    if not ok or a in anchors_seen:
        return None
    filled[(k, sl)] = (it, checks, prov)
    anchors_seen[a] = (k, sl)
    return (k, sl)


ID_RULES = """HARD RULES
- You do NOT type evidence or anchors freely: pick a window by ID from the [CANDIDATE EVIDENCE]
  lists (e.g. "E1T4") and take "anchor" from the "E1T4->token" line of [SUGGESTED ANCHORS] for
  THAT window. Pick only windows that have a suggested anchor. The code copies the window text.
- The window must be in the span AND half (HEAD/TAIL) named by the slot: E<span>H<n> is HEAD,
  E<span>T<n> is TAIL.
- "question": standalone, never "the log"/"span 2"; ask about the artefact in the window; it
  MUST NOT contain the anchor token nor the answer. "canonical_short_answer": <= 20 words.
- Different slots must use different anchors, and none of [ANCHORS ALREADY USED].
- Transcript-local specifics only (paths the agent opened, commands, numbers in outputs,
  intermediate errors); never the issue statement, the root cause or the final patch.
- Output JSON only."""


def id_prompt(chain: dict, miss: list, half_chars: int, used: list) -> str:
    spans = sorted({k[0] for k in miss})
    blocks = "\n\n".join(
        mc.span_block(sp, half_chars, [o["text"] for o in chain["spans"]
                                       if o["span_idx"] != sp["span_idx"]],
                      exclude={u.lower() for u in used} or None)
        for sp in chain["spans"] if sp["span_idx"] in spans)
    asks = "\n".join(f"[SLOT {n}] span {k[0]}, {k[1]} half" for n, k in enumerate(miss, 1))
    return (f"You write exam items about one long coding session. Fill EXACTLY {len(miss)} "
            f"slot(s):\n{asks}\n\n{blocks}\n\n[ANCHORS ALREADY USED]\n"
            f"{', '.join(sorted(used)) or '(none)'}\n\n{ID_RULES}\n\n"
            + mc.RULES.split("HARD RULES")[0] + "\n"
            'Schema: {"items":[{"slot":1,"window":"E1T4","anchor":"...","question":"...",'
            '"canonical_short_answer":"..."}]}\n\n'
            f"Session title: {chain['title'][:200] or chain['session_id']}")


def id_candidates(raw: str, miss: list, table: dict) -> list[dict]:
    """Parse an id-mode reply into answerable candidate items (evidence copied by code)."""
    lo, hi = raw.find("{"), raw.rfind("}")
    out = []
    try:
        items = json.loads(raw[lo:hi + 1], strict=False).get("items") or []
    except Exception:  # noqa: BLE001
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        w = table.get(str(it.get("window") or "").strip())
        if w is None:
            continue
        out.append({"kind": "answerable", "span_idx": w["span_idx"], "slice": w["slice"],
                    "question": mc.strip_wrappers(it.get("question") or ""),
                    "canonical_short_answer": mc.strip_wrappers(
                        it.get("canonical_short_answer") or ""),
                    "gold_anchor": mc.strip_wrappers(it.get("anchor") or ""),
                    "evidence": w["text"]})
    return out


def reprobe(argv: list[str]) -> int:
    """Replace every chain's unanswerable item by a deterministic template probe (no LLM):
    the first verified-absent repo-style candidate of py_probes for that session, distinct
    across chains, question = py_probes.stem_for(token) (the line the generator was told to
    copy verbatim anyway).  Needed because a harvested unanswerable can belong to ANOTHER
    session's repo (the generator copied the prompt's worked example, or a stem of a
    different chain) and still pass the absence gate."""
    ap = argparse.ArgumentParser(prog="oh_pipeline.py ... reprobe")
    ap.add_argument("--chains", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-in", default="")
    ap.add_argument("--manifest-out", default="")
    a = ap.parse_args(argv)
    sessions = {s["sid"]: s for s in mc.load_sessions()}
    chains = [json.loads(l) for l in open(a.chains) if l.strip()]
    used: dict[str, str] = {}
    changed = []
    for c in chains:
        s = sessions[c["session_id"]]
        ch = mc.build_chain(s, c["chain_idx"], 8, False)
        spans, tr = ch["spans"], mc.transcript_of(s)
        for i, q in enumerate(c["qa"]):
            if q["answerable"]:
                continue
            tok = py_probes.probe_candidates(tr, [x["text"] for x in spans], 1,
                                             used=set(used))[0]
            item = {"kind": "unanswerable", "span_idx": 0, "slice": "NONE",
                    "question": py_probes.stem_for(tok), "canonical_short_answer": "",
                    "gold_anchor": "", "evidence": ""}
            ok, why, checks = mc.validate(item, spans, tr)
            assert ok, (c["chain_id"], why)
            old = q["question"]
            c["qa"][i] = dict(q, question=item["question"], checks=checks,
                              provenance={"kind": "template", "was": old})
            used[tok.lower()] = c["chain_id"]
            changed.append((c["chain_id"], old, item["question"]))
    with open(a.out, "w") as w:
        for c in chains:
            w.write(json.dumps(c, ensure_ascii=False) + "\n")
    if a.manifest_in and a.manifest_out:
        m = json.load(open(a.manifest_in))
        m.setdefault("independence", {})["probes_used"] = sorted(used)
        m["reprobe"] = {"chains": len(changed), "method": "py_probes template, no LLM"}
        json.dump(m, open(a.manifest_out, "w"), indent=1, ensure_ascii=False)
    for cid, old, new in changed:
        print(f"{cid}: {old!r} -> {new!r}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="oh_pipeline.py ... salvage")
    ap.add_argument("--raw", action="append", required=True,
                    help="make --save-raw JSONL (repeatable, oldest first)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--rounds", type=int, default=3, help="paid repair calls per chain")
    ap.add_argument("--segs", type=int, default=8)
    ap.add_argument("--half-chars", type=int, default=mc.PROMPT_HALF_CHARS)
    ap.add_argument("--budget-usd", type=float, default=0.15)
    ap.add_argument("--save-raw", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--offline", action="store_true", help="harvest only, no API calls")
    a = ap.parse_args(argv)

    mc.RAW_LOG["path"] = a.save_raw
    if a.save_raw:
        open(a.save_raw, "w").close()
    spend = mc.Spend(a.budget_usd)
    key = os.environ.get("OPENROUTER_API_KEY", "")
    sessions = mc.load_sessions()
    cands = []        # (provenance, item)
    for path in a.raw:
        for n, line in enumerate(open(path), 1):
            e = json.loads(line)
            if e.get("kind") != "llm":
                continue
            for it in mc.parse_items(e.get("raw") or ""):
                cands.append(({"raw": os.path.basename(path), "line": n, "kind": "harvest"}, it))
    print(f"harvested {len(cands)} candidate items from {len(a.raw)} raw file(s)", flush=True)

    pool_used: dict[str, str] = {}      # probe.lower() -> chain_id
    anchors_global: dict[str, str] = {}
    out_chains, man_chains, dropped = [], [], []
    for s in sessions:
        ci = 0
        cid = mc.chain_id_for(s["sid"][-8:], ci)
        chain = mc.build_chain(s, ci, a.segs, check=False)
        spans, transcript = chain["spans"], mc.transcript_of(s)
        if chain["reasons"]:
            dropped.append({"chain_id": cid, "reason": ";".join(chain["reasons"])})
            continue
        filled, anchors_seen = {}, {}
        for prov, it in cands:
            try_fill(chain, spans, transcript, filled, it, pool_used, anchors_seen, prov)
        # a harvested anchor already used by another chain makes verify warn: drop those
        for k in [k for k, v in filled.items() if k != "U"
                  and v[0]["gold_anchor"].strip().lower() in anchors_global]:
            del filled[k]
            anchors_seen = {x: y for x, y in anchors_seen.items() if y != k}
        have0 = len(filled)
        rounds = 0
        while len(filled) < 7 and rounds < a.rounds and not a.offline:
            rounds += 1
            miss = [k for k in SLOTS if k not in filled]
            ans_miss = [k for k in miss if k != "U"]
            used = sorted(set(anchors_seen) | set(anchors_global))
            got = 0
            prov = {"raw": os.path.basename(a.save_raw), "kind": f"repair{rounds}"}
            if ans_miss:           # id mode: the model selects windows, the code copies them
                table = multihop.window_table(chain, a.half_chars, set(used))
                est = sum(sp["chars"] for sp in spans) / mc.CHARS_PER_TOKEN * mc.PRICE_IN \
                    + 2000 * mc.PRICE_OUT + 0.002
                raw, _ = mc.llm(id_prompt(chain, ans_miss, a.half_chars, used), key, a.model,
                                spend, max_tokens=4000, est_cost=est)
                prov["call"] = mc.RAW_LOG["n"]
                got += sum(1 for it in id_candidates(raw, ans_miss, table)
                           if try_fill(chain, spans, transcript, filled, it, pool_used,
                                       anchors_seen, dict(prov, mode="id")))
            if "U" in miss:
                chain["absent_probes"] = mc.draw_unique_probes(
                    [], set(pool_used), [sp["text"] for sp in spans], transcript)
                slots = [{"kind": "unanswerable", "span_idx": 0, "slice": "NONE",
                          "why": "slot is still empty"}]
                raw, _ = mc.llm(mc.repair_prompt(chain, slots, a.half_chars), key, a.model,
                                spend, max_tokens=2000, est_cost=0.006)
                prov = dict(prov, call=mc.RAW_LOG["n"])
                got += sum(1 for it in mc.parse_items(raw)
                           if try_fill(chain, spans, transcript, filled, it, pool_used,
                                       anchors_seen, prov))
            print(f"  {cid} repair {rounds}: +{got} -> {len(filled)}/7", flush=True)
        print(f"{cid}: {len(filled)}/7 ({have0} harvested)", flush=True)
        if len(filled) < 7:
            dropped.append({"chain_id": cid, "reason": "salvage_incomplete:" + ",".join(
                str(k) for k in SLOTS if k not in filled)})
            continue
        qa = []
        for k in SLOTS:
            it, checks, prov = filled[k]
            span_idx = 0 if k == "U" else k[0]
            n = 1 if k == "U" else (1 if k[1] == "HEAD" else 2)
            ans = it["kind"] == "answerable"
            qa.append({"qid": mc.qid_for(cid, span_idx, n), "chain_id": cid,
                       "session_id": s["sid"], "span_idx": span_idx,
                       "lag": (mc.CHAIN_LEN - span_idx + 1) if span_idx else 0,
                       "slice": "NONE" if k == "U" else k[1], "answerable": ans,
                       "must_abstain": not ans, "question": it["question"],
                       "canonical_short_answer": it["canonical_short_answer"] if ans else "",
                       "gold_anchor": it["gold_anchor"] if ans else "",
                       "evidence": it["evidence"] if ans else "", "checks": checks,
                       "provenance": prov})
            if ans:
                anchors_global[it["gold_anchor"].strip().lower()] = cid
            else:
                for t in mc.probe_tokens(it["question"]):
                    pool_used[t.lower()] = cid
        out_chains.append({
            "chain_id": cid, "session_id": s["sid"], "chain_idx": ci,
            "spans": [{"span_idx": sp["span_idx"], "lag": sp["lag"], "segments": sp["segments"],
                       "chars": sp["chars"], "est_tokens": sp["est_tokens"]} for sp in spans],
            "qa": qa, "agent_model": mc.AGENT_MODEL, "qa_model": a.model,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "valid": True, "salvaged": True,
            "build_flags": {"disjoint": True, "unique_probes": True, "dedup_anchors": True}})
        man_chains.append({"chain_id": cid, "session_id": s["sid"], "chain_idx": ci,
                           "valid": True, "qa_count": len(qa), "harvested_slots": have0,
                           "paid_repair_rounds": rounds})
    with open(a.out, "w") as w:
        for c in out_chains:
            w.write(json.dumps(c, ensure_ascii=False) + "\n")
    man = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "qa_model": a.model, "agent_model": mc.AGENT_MODEL,
           "chains": man_chains, "dropped": dropped,
           "independence": {"disjoint": True, "unique_probes": True, "dedup_anchors": True,
                            "probes_used": sorted(pool_used)},
           "salvage": {"raw_files": [os.path.basename(p) for p in a.raw],
                       "harvested_candidates": len(cands)},
           "totals": {"chains_ok": len(out_chains), "chains_dropped": len(dropped),
                      "items": sum(len(c["qa"]) for c in out_chains),
                      "api_calls": spend.calls, "spend_usd": round(spend.spent, 5),
                      "budget_usd": a.budget_usd}}
    json.dump(man, open(a.manifest, "w"), indent=1, ensure_ascii=False)
    print(f"chains ok {len(out_chains)}, dropped {len(dropped)}, spend ${spend.spent:.4f}")
    for d in dropped:
        print("  dropped", d["chain_id"], d["reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
