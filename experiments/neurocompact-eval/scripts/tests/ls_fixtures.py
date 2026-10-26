"""Synthetic fixtures for the long-session tests. Everything is generated here: no BEAM text, no
real transcript content. Secret-looking strings are assembled from parts at runtime."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

FILLER = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut "
          "labore et dolore magna aliqua ")


def make_session(sid: str = "syn-1", n: int = 120, chars: int = 1800, real: bool = False,
                 questions: bool = True) -> dict:
    msgs = []
    for i in range(n):
        role = "user" if i % 4 == 0 else ("tool" if i % 4 == 2 else "assistant")
        head = (f"step {i}: job JOB{i:04d} finished with metric_{i} = {1000 + 7 * i} units in "
                f"{(i % 9) + 1}.{i % 7}s; artifact path /data/run{i:03d}/out.bin ")
        text = (head + FILLER * 40)[:chars]
        msgs.append({"idx": i, "role": role, "text": text})
    s = {"format": "longsession/1", "sid": sid, "source": "synthetic", "title": "synthetic session",
         "messages": msgs, "compactions": [], "questions": [], "meta": {}}
    if real:
        for k, at in enumerate((30, 70, 100), 1):
            s["compactions"].append({"at_idx": at, "original_summary": f"GOAL\nreal summary {k}\n"
                                     f"KEY FIGURES & IDENTIFIERS\n- JOB{at - 1:04d}\nDECISIONS & RATIONALE\n- x.\nACTIVE STATE\n- y."})
    if questions:
        for k, idx in enumerate((5, 20, 40, 65), 1):
            s["questions"].append({
                "qid": f"{sid}_q{k}", "qtype": "single", "question": f"What was the value of metric_{idx} reported by job JOB{idx:04d}?",
                "answer": str(1000 + 7 * idx), "answerable": True, "evidence": [{"msg_idx": idx}],
                "gold_anchor": str(1000 + 7 * idx), "rubric": [f"LLM response should state: {1000 + 7 * idx}"]})
        s["questions"].append({
            "qid": f"{sid}_m1", "qtype": "multihop",
            "question": "What metric did the job that ran after JOB0010 report in step 30?", "answer": str(1000 + 7 * 30),
            "answerable": True, "evidence": [{"msg_idx": 10, "role": "link"}, {"msg_idx": 30, "role": "answer"}],
            "gold_anchor": str(1000 + 7 * 30)})
        s["questions"].append({"qid": f"{sid}_u1", "qtype": "single", "question": "What is the codename of the unreleased tool X?",
                               "answer": "no information", "answerable": False, "evidence": []})
    return s


def write_session(dirpath: str, session: dict) -> str:
    os.makedirs(dirpath, exist_ok=True)
    p = os.path.join(dirpath, f"{session['sid']}.json")
    with open(p, "w") as f:
        json.dump(session, f)
    return p


# small-window parameters that give several boundaries on make_session()
RUN_ARGS = ["--window", "14000", "--keep-tail", "3000", "--mem-budgets", "512", "--provider", "mock",
            "--agent-model", "mock", "--fake-hars", "--no-preflight", "--piece-chars", "3000",
            "--pieces-per-call", "4"]
