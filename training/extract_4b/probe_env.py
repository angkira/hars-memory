#!/usr/bin/env python3
"""Pre-flight probe on the Colab VM. No weights download (meta device).

Answers, with printed evidence:
  1. GPU / CUDA / library versions
  2. chat template render for Qwen3.5 (enable_thinking=False): does the
     generation prompt prefill the empty <think></think> block?
  3. eos / pad tokens and generation_config eos ids
  4. Qwen3_5 module tree -> candidate LoRA target module names (language_model
     Linears only; vision tower and projector excluded)
  5. token-length stats of prepared train data -> validate MAX_SEQ_LEN
Writes probe_report.json next to itself.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-4B"
DATA_DIR = Path(__file__).resolve().parent / "data"
REPORT = Path(__file__).resolve().parent / "probe_report.json"

msgs = [
    {"role": "system", "content": "SYS-CONTENT-HERE."},
    {"role": "user", "content": "USER-CONTENT-HERE."},
]

rep: dict = {}
rep["nvidia_smi"] = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip()
rep["versions"] = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
}
rep["gpu_ok"] = torch.cuda.is_available()

tok = AutoTokenizer.from_pretrained(MODEL_ID)
rep["tokens"] = {
    "eos_token": tok.eos_token,
    "eos_token_id": tok.eos_token_id,
    "pad_token": tok.pad_token,
    "pad_token_id": getattr(tok, "pad_token_id", None),
    "vocab_size": len(tok),
}

# --- chat template behavior -------------------------------------------------
renders = {}
for kwargs, name in [
    (dict(add_generation_prompt=True), "gen_default"),
    (dict(add_generation_prompt=True, enable_thinking=False), "gen_nothink"),
    (dict(add_generation_prompt=True, enable_thinking=True), "gen_think"),
]:
    try:
        renders[name] = tok.apply_chat_template(msgs, tokenize=False, **kwargs)
    except Exception as e:  # template may not accept the kwarg
        renders[name] = f"ERROR: {type(e).__name__}: {e}"
rep["renders"] = {k: repr(v)[:400] for k, v in renders.items()}

# assistant-message render (continue_final_message) for training-side check
try:
    full = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": "ASSISTANT-OUT"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
        continue_final_message=True)
    gen = renders.get("gen_nothink", "")
    rep["assistant_render"] = {
        "startswith_gen_prompt": full.startswith(gen) if not gen.startswith("ERROR") else None,
        "repr": repr(full)[:500],
    }
except Exception as e:
    rep["assistant_render"] = {"error": f"{type(e).__name__}: {e}"}

# --- model class + meta-device module dump ----------------------------------
cfg = AutoConfig.from_pretrained(MODEL_ID)
rep["architectures"] = cfg.architectures
rep["model_type"] = cfg.model_type
try:
    from transformers import AutoModelForImageTextToText
    with torch.device("meta"):
        m = AutoModelForImageTextToText.from_config(cfg)
    names = [n for n, mod in m.named_modules()
             if type(mod).__name__.lower().startswith("linear")]
    lm = [n for n in names if "language_model" in n and "lm_head" not in n]
    vis = [n for n in names if "language_model" not in n]
    rep["linear_module_counts"] = {"total": len(names), "language_model": len(lm), "other(vision+proj)": len(vis)}
    rep["lm_linear_suffixes"] = sorted({n.rsplit(".", 1)[-1] for n in lm})
    rep["lm_linear_sample"] = lm[:6]
    rep["vision_linear_sample"] = vis[:6]
    del m
except Exception as e:
    rep["meta_init_error"] = f"{type(e).__name__}: {e}"

# --- token length stats on prepared data ------------------------------------
lens = []
for split in ["train.jsonl", "val.jsonl", "parity_holdout.jsonl"]:
    p = DATA_DIR / split
    if not p.exists():
        continue
    rows = [json.loads(line) for line in p.open()]
    for r in rows:
        n = len(tok(r["system"] + r["user"] + r["output"]).input_ids)
        lens.append((split, n))
by = {}
for split in ["train.jsonl", "val.jsonl", "parity_holdout.jsonl"]:
    vals = [n for s, n in lens if s == split]
    if vals:
        vals.sort()
        by[split] = {"n": len(vals), "min": vals[0], "p50": vals[len(vals)//2],
                     "p95": vals[int(len(vals)*0.95)], "max": vals[-1]}
rep["token_lens"] = by

REPORT.write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
