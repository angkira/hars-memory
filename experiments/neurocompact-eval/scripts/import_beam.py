#!/usr/bin/env python3
"""Import BEAM ("Beyond a Million Tokens") chats into the longsession/1 format.

BEAM repo layout (parsed as DATA ONLY: json, never the *.pickle files; run this with
``python3 -I`` against a checkout you did not write):

  chats/<size>/<n>/chat.json
      100K / 500K / 1M : list of batches  {batch_number, time_anchor, turns: [[turn, turn], ...]}
      10M              : list of {"plan-N": [batch, ...]}  (10 plans x 10 batches)
      turn = {role: user|assistant, id: int, content: str, index?: "1,3", question_type?: str,
              time_anchor?: "March-01-2024"}     # `id` is global within the chat file
  chats/<size>/<n>/probing_questions/probing_questions.json
      {question_type: [ {question, answer|ideal_answer|ideal_response|ideal_summary, rubric,
                         source_chat_ids: [ids] | {first_statement: [..], second_statement: [..]}
                                          | {original_info, updated_info} | {first_event, second_event},
                         ...type specific...} ]}

Mapping
  * message idx = position in the flattened turn order (verified equal to ``id`` on the shipped
    data; the id -> idx map is used anyway). ``ts`` = latest time_anchor seen. BEAM has no real
    compactions: ``compactions`` is empty (the runner computes synthetic boundaries).
  * ``meta.batches`` records {start_idx, plan, batch_number, time_anchor} (BEAM's own sessions).
  * qtype: information_extraction -> ``single``; multi_session_reasoning -> ``multihop``;
    knowledge_update / temporal_reasoning / contradiction_resolution keep their name.
    The other BEAM types (abstention, event_ordering, instruction_following,
    preference_following, summarization) are DROPPED by default and only counted
    (``meta.dropped_types``); ``--types all`` or ``--types abstention,summarization`` keeps them.
    abstention is imported with ``answerable=false`` (gold = "no information").
  * evidence = ``source_chat_ids`` mapped to ``{"msg_idx", "role"}`` (role = the dict key for
    two-statement types). Ids that do not resolve are dropped and counted in
    ``question.meta.unmapped_evidence``. Questions without ids (abstention, a few
    summarization/knowledge_update items) keep ``evidence=[]``.
  * ``gold_anchor`` is set only when the (short, <=8 words) gold answer occurs verbatim
    (case-insensitive, token-bounded) in one of the evidence messages; otherwise it is absent
    and the runner scores that question by rubric/judge only.
  * ``rubric`` is BEAM's own rubric list.

Usage
  python3 -I import_beam.py --repo /path/to/beam-repo --size 1M --out OUT_DIR [--chats 1,2|all]
                            [--types information_extraction,knowledge_update|all] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from longsession import FORMAT, save_session, validate_session  # noqa: E402

DEFAULT_TYPES = ("information_extraction", "knowledge_update", "temporal_reasoning",
                 "multi_session_reasoning", "contradiction_resolution")
ALL_TYPES = DEFAULT_TYPES + ("abstention", "event_ordering", "instruction_following",
                             "preference_following", "summarization")
QTYPE_MAP = {"information_extraction": "single", "multi_session_reasoning": "multihop"}
ABBR = {"information_extraction": "ie", "knowledge_update": "ku", "temporal_reasoning": "tr",
        "multi_session_reasoning": "ms", "contradiction_resolution": "cr", "abstention": "ab",
        "event_ordering": "eo", "instruction_following": "if", "preference_following": "pf",
        "summarization": "su"}
ANSWER_KEYS = ("answer", "ideal_answer", "ideal_response", "ideal_summary", "expected_compliance")
MAX_ANCHOR_WORDS = 8


def _batches(chat) -> list[tuple[str, dict]]:
    """[(plan_name, batch)] for both layouts."""
    out: list[tuple[str, dict]] = []
    for item in chat:
        if isinstance(item, dict) and "turns" in item:
            out.append(("", item))
        elif isinstance(item, dict):                      # 10M: {"plan-N": [batches]}
            for plan, bl in item.items():
                for b in bl:
                    out.append((plan, b))
    return out


def flatten_chat(chat) -> tuple[list[dict], dict[int, int], list[dict]]:
    """(messages, turn_id -> idx, batch markers)."""
    msgs: list[dict] = []
    id2idx: dict[int, int] = {}
    batches: list[dict] = []
    ts = None
    for plan, b in _batches(chat):
        if b.get("time_anchor"):
            ts = b["time_anchor"]
        batches.append({"start_idx": len(msgs), "plan": plan, "batch_number": b.get("batch_number"),
                        "time_anchor": b.get("time_anchor")})
        for group in b.get("turns", []):
            for t in group:
                if t.get("time_anchor"):
                    ts = t["time_anchor"]
                role = t.get("role") if t.get("role") in ("user", "assistant") else "user"
                m = {"idx": len(msgs), "role": role, "text": str(t.get("content") or "")}
                if ts:
                    m["ts"] = ts
                if isinstance(t.get("id"), int):
                    id2idx.setdefault(t["id"], len(msgs))
                msgs.append(m)
    return msgs, id2idx, batches


def _anchor_in(answer: str, texts: list[str]) -> str | None:
    a = (answer or "").strip().strip(".")
    if not a or len(a.split()) > MAX_ANCHOR_WORDS or len(a) < 2:
        return None
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(a) + r"(?![A-Za-z0-9_])", re.I)
    return a if any(pat.search(t) for t in texts) else None


def convert_questions(pq: dict, sid: str, msgs: list[dict], id2idx: dict[int, int],
                      types: tuple[str, ...]) -> tuple[list[dict], dict]:
    qs: list[dict] = []
    stats = {"dropped_types": Counter(), "kept": Counter(), "anchors": 0, "unmapped_evidence": 0}
    for typ in sorted(pq):
        items = pq[typ]
        if typ not in types:
            stats["dropped_types"][typ] += len(items) if isinstance(items, list) else 0
            continue
        for n, it in enumerate(items, 1):
            if not isinstance(it, dict) or not it.get("question"):
                continue
            ans = next((str(it[k]) for k in ANSWER_KEYS if it.get(k)), "")
            if not ans:
                continue
            src = it.get("source_chat_ids")
            ev: list[dict] = []
            unm = 0
            groups = ([(None, src)] if isinstance(src, list)
                      else list(src.items()) if isinstance(src, dict) else [])
            for role, ids in groups:
                for cid in ids if isinstance(ids, list) else []:
                    if isinstance(cid, int) and cid in id2idx:
                        e = {"msg_idx": id2idx[cid]}
                        if role:
                            e["role"] = role
                        if e not in ev:
                            ev.append(e)
                    else:
                        unm += 1
            stats["unmapped_evidence"] += unm
            q = {"qid": f"{sid}_{ABBR.get(typ, typ[:2])}{n}",
                 "qtype": QTYPE_MAP.get(typ, typ), "source_qtype": typ,
                 "question": str(it["question"]).strip(), "answer": ans.strip(),
                 "answerable": typ != "abstention", "evidence": ev,
                 "rubric": [str(r) for r in (it.get("rubric") or [])],
                 "meta": {"difficulty": it.get("difficulty"), "unmapped_evidence": unm}}
            if typ == "contradiction_resolution":
                q["meta"]["expects_contradiction_flag"] = True
            if typ in ("information_extraction", "knowledge_update", "temporal_reasoning",
                       "multi_session_reasoning") and ev:
                a = _anchor_in(ans, [msgs[e["msg_idx"]]["text"] for e in ev])
                if a:
                    q["gold_anchor"] = a
                    stats["anchors"] += 1
            qs.append(q)
            stats["kept"][typ] += 1
    return qs, stats


def import_chat(repo: str, size: str, n: str, types: tuple[str, ...]) -> dict:
    d = os.path.join(repo, "chats", size, n)
    with open(os.path.join(d, "chat.json"), encoding="utf-8") as f:
        chat = json.load(f)
    msgs, id2idx, batches = flatten_chat(chat)
    pqp = os.path.join(d, "probing_questions", "probing_questions.json")
    pq = {}
    if os.path.exists(pqp):
        with open(pqp, encoding="utf-8") as f:
            pq = json.load(f)
    sid = f"beam-{size}-{n}"
    qs, st = convert_questions(pq, sid, msgs, id2idx, types)
    topic = ""
    tp = os.path.join(d, "topic.json")
    if os.path.exists(tp):
        try:
            with open(tp, encoding="utf-8") as f:
                t = json.load(f)
            topic = (t.get("title") or t.get("topic") or "") if isinstance(t, dict) else str(t)
        except (OSError, ValueError):
            topic = ""
    return {"format": FORMAT, "sid": sid, "source": "beam",
            "title": topic or f"BEAM {size} chat {n}", "messages": msgs, "compactions": [],
            "questions": qs,
            "meta": {"size": size, "chat": n, "batches": batches,
                     "dropped_types": dict(st["dropped_types"]), "kept_types": dict(st["kept"]),
                     "anchors": st["anchors"], "unmapped_evidence": st["unmapped_evidence"]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True, help="BEAM checkout (contains chats/)")
    ap.add_argument("--size", required=True, choices=("100K", "500K", "1M", "10M"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--chats", default="all", help="comma list of chat numbers or 'all'")
    ap.add_argument("--types", default=",".join(DEFAULT_TYPES),
                    help="BEAM question types to keep, comma list or 'all'")
    ap.add_argument("--limit", type=int, default=0, help="max chats (0 = all)")
    a = ap.parse_args(argv)
    types = ALL_TYPES if a.types == "all" else tuple(t.strip() for t in a.types.split(",") if t.strip())
    bad = [t for t in types if t not in ALL_TYPES]
    if bad:
        raise SystemExit(f"unknown question types {bad}; known: {', '.join(ALL_TYPES)}")
    root = os.path.join(a.repo, "chats", a.size)
    if not os.path.isdir(root):
        raise SystemExit(f"no such dir: {root}")
    ns = sorted((x for x in os.listdir(root) if os.path.isfile(os.path.join(root, x, "chat.json"))),
                key=lambda x: int(x) if x.isdigit() else 10 ** 9)
    if a.chats != "all":
        want = {c.strip() for c in a.chats.split(",")}
        ns = [x for x in ns if x in want]
    if a.limit:
        ns = ns[:a.limit]
    tot = Counter()
    for n in ns:
        s = import_chat(a.repo, a.size, n, types)
        problems = validate_session(s)
        if problems:
            print(f"[skip] {s['sid']}: {problems[:3]}", file=sys.stderr)
            continue
        save_session(s, a.out)
        tot["sessions"] += 1
        tot["messages"] += len(s["messages"])
        tot["questions"] += len(s["questions"])
        tot["anchors"] += s["meta"]["anchors"]
        tot["unmapped_evidence"] += s["meta"]["unmapped_evidence"]
        print(f"{s['sid']}: {len(s['messages'])} msgs, {len(s['questions'])} questions "
              f"({dict(s['meta']['kept_types'])}), dropped {dict(s['meta']['dropped_types'])}")
    print(f"imported {tot['sessions']} sessions, {tot['messages']} messages, "
          f"{tot['questions']} questions ({tot['anchors']} with gold_anchor, "
          f"{tot['unmapped_evidence']} unmapped evidence ids) -> {a.out}")
    return 0 if tot["sessions"] else 1


if __name__ == "__main__":
    sys.exit(main())
