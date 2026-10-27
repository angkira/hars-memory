"""Synthetic OpenHands-shaped chain records (make_chains.py format) for the Amendment C.6 tests.
No OpenHands text: events are generated here. Deterministic."""
from __future__ import annotations

import json
import os

FILLER = "filler words about nothing in particular " * 8


def _events(tag: str, n: int, base: int):
    """n events alternating tool / assistant; event k carries a unique token NEEDLE_<tag>_<k>."""
    out = []
    for k in range(n):
        role = "tool" if k % 2 == 0 else "assistant"
        head = (f"[tool read] in: {{\"filePath\": \"/w/{tag}_{k}.py\"}}\n[out] " if role == "tool" else "")
        out.append({"role": role, "text": f"{head}line {base + k}: value NEEDLE_{tag}_{k} = {base + 7 * k}\n{FILLER}"})
    return out


def make_chain(cid: str = "chain_syn_c0", session: str = "syn__1", n_events=(6, 5, 4), valid: bool = True,
               unlocatable: bool = False) -> dict:
    spans = []
    for si, n in enumerate(n_events, 1):
        spans.append({"span_idx": si, "lag": 4 - si, "segments": _events(f"s{si}", n, 100 * si),
                      "chars": 1000 * si, "est_tokens": 100 * si})
    ev = lambda si, k: spans[si - 1]["segments"][k]["text"]            # noqa: E731
    qa = [
        {"qid": f"{cid}_1_1", "chain_id": cid, "session_id": session, "span_idx": 1, "lag": 3, "slice": "HEAD",
         "answerable": True, "must_abstain": False, "question": "What value did NEEDLE_s1_1 take?",
         "canonical_short_answer": "107", "gold_anchor": "NEEDLE_s1_1",
         "evidence": f"line 101: value NEEDLE_s1_1 = 107", "qtype": "single"},
        # evidence that crosses two events (tail of event 2 + head of event 3 of span 2), role-tag form
        {"qid": f"{cid}_2_2", "chain_id": cid, "session_id": session, "span_idx": 2, "lag": 2, "slice": "TAIL",
         "answerable": True, "must_abstain": False, "question": "What follows NEEDLE_s2_2?",
         "canonical_short_answer": "NEEDLE_s2_3", "gold_anchor": "NEEDLE_s2_3",
         "evidence": "NEEDLE_s2_2 = 214\n" + FILLER.strip() + "\n\n[assistant] line 203: value NEEDLE_s2_3", "qtype": "single"},
        {"qid": f"{cid}_3_1", "chain_id": cid, "session_id": session, "span_idx": 3, "lag": 1, "slice": "HEAD",
         "answerable": True, "must_abstain": False, "question": "Value of NEEDLE_s3_0?",
         "canonical_short_answer": "300", "gold_anchor": "NEEDLE_s3_0",
         "evidence": "line 300: value NEEDLE_s3_0 = 300", "qtype": "single"},
        {"qid": f"{cid}_0_1", "chain_id": cid, "session_id": session, "span_idx": 0, "lag": 0, "slice": "NONE",
         "answerable": False, "must_abstain": True, "question": "Which function does ghost.py define first?",
         "canonical_short_answer": "", "gold_anchor": "", "evidence": "", "qtype": "single"},
        {"qid": f"{cid}_mh_1", "chain_id": cid, "session_id": session, "qtype": "multihop", "span_idx": 3, "lag": 3,
         "hop_lags": [3, 1], "span_idxs": [1, 3], "slice": "TAIL", "answerable": True, "must_abstain": False,
         "question": "After NEEDLE_s1_2, which value did NEEDLE_s3_2 take?", "canonical_short_answer": "314",
         "gold_anchor": "NEEDLE_s3_2", "link_anchor": "NEEDLE_s1_2",
         "evidence": "line 102: value NEEDLE_s1_2 = 114\n[...]\nline 302: value NEEDLE_s3_2 = 314",
         "evidence_spans": [
             {"role": "link", "span_idx": 1, "anchor": "NEEDLE_s1_2", "evidence": "line 102: value NEEDLE_s1_2 = 114",
              "slice": "HEAD"},
             {"role": "answer", "span_idx": 3, "anchor": "NEEDLE_s3_2", "evidence": "line 302: value NEEDLE_s3_2 = 314",
              "slice": "TAIL"}], "mh_version": 1},
    ]
    if unlocatable:
        qa[0]["evidence"] = "this text occurs nowhere"
        qa[0]["gold_anchor"] = "NOT_ANYWHERE"
    return {"chain_id": cid, "session_id": session, "chain_idx": 0, "spans": spans, "qa": qa,
            "agent_model": "x", "qa_model": "claude-haiku-5-5", "valid": valid,
            "build_flags": {"disjoint": True}}


def write_chains(path: str, chains: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for c in chains:
            f.write(json.dumps(c) + "\n")


def make_artifact(d: str, n: int = 3) -> list[dict]:
    chains = [make_chain(f"chain_s{i}_c0", f"sess{i}", n_events=(6 + i, 5, 4)) for i in range(n)]
    write_chains(os.path.join(d, "chains.jsonl"), chains)
    open(os.path.join(d, "verify-report.md"), "w").write("verify ok\n")
    return chains
