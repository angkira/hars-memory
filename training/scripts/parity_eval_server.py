#!/usr/bin/env python
"""Parity gate against a RUNNING OpenAI-compatible extractor endpoint.

Evaluates exactly the stack that will serve production extraction (base GGUF +
LoRA via llama-server), instead of an in-process transformers model.

    python parity_eval_server.py --base-url http://127.0.0.1:8084/v1 \
        --parity-jsonl ../outputs/data/parity_heldout.jsonl \
        --out-dir ../outputs/parity_server [--limit N]

Gates: entity F1 >= 0.85, zero format errors, hallucination rate < 0.05.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import aggregate, evaluate_pair, load_jsonl

GATES = {"entity_f1_min": 0.85, "format_errors_max": 0, "hallucination_rate_max": 0.05}


def chat(base_url: str, model: str, prompt: str, max_tokens: int, timeout: int = 900) -> tuple[str, dict]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    text = data["choices"][0]["message"].get("content") or ""
    return text, data.get("usage", {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8084/v1")
    ap.add_argument("--model", default="qwen3.5-9b-extractor")
    ap.add_argument("--parity-jsonl", default="../outputs/data/parity_heldout.jsonl")
    ap.add_argument("--out-dir", default="../outputs/parity_server")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="server")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    pp = args.parity_jsonl if os.path.isabs(args.parity_jsonl) else os.path.normpath(os.path.join(here, args.parity_jsonl))
    od = args.out_dir if os.path.isabs(args.out_dir) else os.path.normpath(os.path.join(here, args.out_dir))
    os.makedirs(od, exist_ok=True)

    rows = load_jsonl(pp)
    if args.limit:
        rows = rows[: args.limit]
    print(f"parity: {len(rows)} prompts -> {args.base_url} ({args.model})", flush=True)

    results = []
    t_all = time.time()
    for i, r in enumerate(rows):
        t0 = time.time()
        pred, usage = chat(args.base_url, args.model, r["input"], args.max_new_tokens)
        dt = time.time() - t0
        m = evaluate_pair(r["output"], pred, r["input"])
        rec = {
            "id": r["id"],
            "metrics": {k: v for k, v in m.items() if k not in ("gold", "pred")},
            "gen_seconds": round(dt, 1),
            "usage": usage,
            "pred_text": pred,
        }
        results.append(rec)
        print(
            f"[{i+1}/{len(rows)}] {r['id']} entF1={m['entity_f1']:.3f} relF1={m['relation_f1']:.3f} "
            f"halluc={m['hallucination_rate']:.3f} fmt_err={m['n_format_errors']} complete={m['complete']} "
            f"{dt:.0f}s ({usage.get('completion_tokens', 0)} tok)",
            flush=True,
        )
        with open(os.path.join(od, "predictions.jsonl"), "w") as f:
            for x in results:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    agg = aggregate([
        {k: x["metrics"][k] for k in (
            "entity_precision", "entity_recall", "entity_f1",
            "relation_precision", "relation_recall", "relation_f1",
            "hallucination_rate", "n_format_errors", "complete")}
        for x in results
    ])
    report = {
        "tag": args.tag,
        "base_url": args.base_url,
        "model": args.model,
        "n_samples": len(results),
        "entity_precision": round(agg.entity_precision, 4),
        "entity_recall": round(agg.entity_recall, 4),
        "entity_f1": round(agg.entity_f1, 4),
        "relation_precision": round(agg.relation_precision, 4),
        "relation_recall": round(agg.relation_recall, 4),
        "relation_f1": round(agg.relation_f1, 4),
        "hallucination_rate": round(agg.hallucination_rate, 4),
        "format_error_rate": round(agg.format_error_rate, 4),
        "complete_rate": round(agg.complete_rate, 4),
        "mean_seconds_per_sample": round(sum(x["gen_seconds"] for x in results) / len(results), 1),
        "total_wall_minutes": round((time.time() - t_all) / 60, 1),
        "gates": GATES,
        "gate_entity_f1_pass": agg.entity_f1 >= GATES["entity_f1_min"],
        "gate_format_pass": agg.format_error_rate == 0.0,
        "gate_hallucination_pass": agg.hallucination_rate < GATES["hallucination_rate_max"],
    }
    report["ship_gate_pass"] = all(
        report[k] for k in ("gate_entity_f1_pass", "gate_format_pass", "gate_hallucination_pass")
    )
    with open(os.path.join(od, "parity_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
