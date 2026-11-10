#!/usr/bin/env python
"""Parity gate: generate extraction outputs with the fine-tuned adapter on the
20 held-out prompts and compare against the cached 27B-teacher outputs.

Metrics per sample and aggregate:
- entity F1 (normalized name+type pair match)
- relation F1 (normalized src+tgt pair match)
- format validity (record grammar, delimiter counts, COMPLETE marker)
- hallucination rate (predicted entity names absent from the source text)
- generation speed (tok/s, sec/sample)

Ship gate: entity F1 >= 0.85 x teacher-teacher F1 is not applicable here
(teacher IS the gold) — the gate is: entity F1 >= 0.85 vs the teacher's own
cached outputs, zero format errors, hallucination rate < 5%.

Usage:
  python parity_eval.py --adapter ../outputs/run_9b_fp8_r32/adapter \
                        --parity-jsonl ../outputs/data/parity_heldout.jsonl \
                        --out-dir ../outputs/run_9b_fp8_r32/parity
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import aggregate, evaluate_pair, load_jsonl

GATES = {"entity_f1_min": 0.85, "format_errors_max": 0, "hallucination_rate_max": 0.05}


def load_model(adapter: str | None, base: str, mode: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    kwargs = dict(dtype="auto", device_map={"": "cuda:0"}, attn_implementation="sdpa")
    if mode == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForImageTextToText.from_pretrained(base, **kwargs)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        log = f"adapter {adapter}"
    else:
        log = "BASE (no adapter)"
    model.config.use_cache = True
    model.eval()
    print(f"loaded {log}")
    return tok, model


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="path to LoRA adapter; omit for base model")
    ap.add_argument("--base", default=os.path.expanduser("~/models/qwen3.5-9b-fp8"))
    ap.add_argument("--mode", default="fp8", choices=["fp8", "bf16", "nf4"])
    ap.add_argument("--parity-jsonl", default="../outputs/data/parity_heldout.jsonl")
    ap.add_argument("--out-dir", default="../outputs/parity")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    parity_path = args.parity_jsonl if os.path.isabs(args.parity_jsonl) else os.path.normpath(os.path.join(here, args.parity_jsonl))
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.normpath(os.path.join(here, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    tok, model = load_model(args.adapter, args.base, args.mode)
    rows = load_jsonl(parity_path)
    print(f"evaluating {len(rows)} held-out prompts")

    eos_ids = {tok.eos_token_id}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int):
        eos_ids.add(im_end)

    results = []
    t_all = time.time()
    for i, r in enumerate(rows):
        from common import render_prompt

        prompt_text = render_prompt(tok, r["input"])
        enc = tok(prompt_text, add_special_tokens=False, return_tensors="pt").to("cuda:0")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                eos_token_id=sorted(eos_ids),
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        dt = time.time() - t0
        new_tokens = out.shape[1] - enc["input_ids"].shape[1]
        pred_text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        m = evaluate_pair(r["output"], pred_text, r["input"])
        rec = {
            "id": r["id"],
            "metrics": {k: v for k, v in m.items() if k not in ("gold", "pred")},
            "gen_seconds": round(dt, 1),
            "gen_tokens": int(new_tokens),
            "tokens_per_sec": round(new_tokens / dt, 1),
            "pred_text": pred_text,
        }
        results.append(rec)
        print(f"[{i+1}/{len(rows)}] {r['id']}  entF1={m['entity_f1']:.3f} relF1={m['relation_f1']:.3f} "
              f"halluc={m['hallucination_rate']:.3f} fmt_err={m['n_format_errors']} "
              f"complete={m['complete']} {new_tokens}tok {dt:.0f}s ({new_tokens/dt:.1f} tok/s)", flush=True)
        with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    agg = aggregate([{"entity_precision": r["metrics"]["entity_precision"],
                      "entity_recall": r["metrics"]["entity_recall"],
                      "entity_f1": r["metrics"]["entity_f1"],
                      "relation_precision": r["metrics"]["relation_precision"],
                      "relation_recall": r["metrics"]["relation_recall"],
                      "relation_f1": r["metrics"]["relation_f1"],
                      "hallucination_rate": r["metrics"]["hallucination_rate"],
                      "n_format_errors": r["metrics"]["n_format_errors"],
                      "complete": r["metrics"]["complete"]} for r in results])

    mean_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    mean_sec = sum(r["gen_seconds"] for r in results) / len(results)
    report = {
        "adapter": args.adapter,
        "base": args.base,
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
        "mean_tokens_per_sec": round(mean_tps, 1),
        "mean_seconds_per_sample": round(mean_sec, 1),
        "total_wall_minutes": round((time.time() - t_all) / 60, 1),
        "gates": GATES,
        "gate_entity_f1_pass": agg.entity_f1 >= GATES["entity_f1_min"],
        "gate_format_pass": agg.format_error_rate == 0.0,
        "gate_hallucination_pass": agg.hallucination_rate < GATES["hallucination_rate_max"],
    }
    report["ship_gate_pass"] = all(
        report[k] for k in ("gate_entity_f1_pass", "gate_format_pass", "gate_hallucination_pass")
    )
    with open(os.path.join(out_dir, "parity_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
