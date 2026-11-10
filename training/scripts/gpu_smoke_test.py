#!/usr/bin/env python
"""GPU smoke test for the R9700 (gfx1201) before committing to a training run.

Tries to load the base model in the requested mode, attach LoRA, and run a
few forward/backward steps on a real training sample. Reports VRAM peak,
step time, and loss sanity. Exits non-zero on failure.

Modes:
  fp8   — keep the checkpoint's blockwise-FP8 weights (transformers
          FineGrainedFP8), train bf16 LoRA on top. ~12GB weights.
  bf16  — dequantize everything to bf16. ~19GB weights (needs a free GPU).
  nf4   — bitsandbytes 4-bit QLoRA (requires bnb with ROCm backend).

Usage: python gpu_smoke_test.py --mode fp8 [--seq-len 4096]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BASE = os.path.expanduser("~/models/qwen3.5-9b-fp8")


def vram_info() -> tuple:
    import torch

    free, total = torch.cuda.mem_get_info(0)
    return free / 2**30, total / 2**30


def log(msg: str) -> None:
    free, total = vram_info()
    print(f"[{time.strftime('%H:%M:%S')}] {msg}  (torch free {free:.1f}/{total:.1f} GiB)", flush=True)


def load_model(mode: str, base: str, max_seq: int):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    kwargs = dict(dtype="auto", device_map={"": "cuda:0"}, attn_implementation="sdpa")
    if mode == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif mode == "bf16":
        from transformers import FineGrainedFP8Config

        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    elif mode != "fp8":
        raise ValueError(mode)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForImageTextToText.from_pretrained(base, **kwargs)
    model.config.use_cache = False
    log(f"model loaded in {time.time() - t0:.0f}s mode={mode}")
    return tok, model


def attach_lora(model, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def build_batch(tok, sample: dict, max_seq: int):
    import torch

    from common import encode_completion_only

    ids, labels = encode_completion_only(tok, sample["input"], sample["output"])
    if len(ids) > max_seq:
        # truncate the *input* tail, keep the completion intact
        overflow = len(ids) - max_seq
        # prompt = ids[:-len(completion_part)]; recompute split
        n_out = sum(1 for x in labels if x != -100)
        ids = ids[: len(ids) - n_out - overflow] + ids[len(ids) - n_out :]
        labels = [-100] * (len(ids) - n_out) + labels[len(labels) - n_out :]
    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")
    lab = torch.tensor([labels], dtype=torch.long, device="cuda:0")
    return input_ids, lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fp8", choices=["fp8", "bf16", "nf4"])
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--train-jsonl", default=os.path.join(os.path.dirname(__file__), "..", "outputs", "data", "train.jsonl"))
    ap.add_argument("--seq-len", type=int, default=6144)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--fp8-patch", default="auto", choices=["auto", "on", "off"],
                    help="on: force pure-torch fp8 dequant; auto: use it only if the triton kernel fails")
    args = ap.parse_args()

    import torch
    from common import load_jsonl
    from torch_utils import chunked_ce_loss

    free, total = vram_info()
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "free_gib": round(free, 1), "total_gib": round(total, 1)}))
    if free < 16 and args.mode != "nf4":
        print(f"REFUSING: only {free:.1f} GiB free and other jobs are on the GPU")
        sys.exit(2)

    tok, model = load_model(args.mode, args.base, args.seq_len)
    if args.mode == "fp8" and args.fp8_patch != "off":
        # The community triton fp8 kernel loads on ROCm but registers no
        # autograd formula — training requires the pure-torch dequant path.
        from torch_utils import patch_fp8_linear_torch

        patch_fp8_linear_torch()
    rows = load_jsonl(args.train_jsonl)
    sample = rows[0]
    model = attach_lora(model, args.lora_r, args.lora_alpha, 0.05)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=0.0)

    input_ids, labels = build_batch(tok, sample, args.seq_len)
    log(f"batch built: {input_ids.shape[1]} tokens")
    import torch as _t

    attn = _t.ones_like(input_ids)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(args.steps):
        loss_sum, n_items = chunked_ce_loss(model, input_ids, attn, labels)
        (loss_sum / n_items).backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(0) / 2**30
        log(f"step {i}: loss={loss_sum.item()/n_items.item():.4f} step_time={time.time() - t0:.1f}s peak_vram={peak:.1f} GiB")
        t0 = time.time()

    # generation sanity (does the fp8 path also generate?)
    model.eval()
    model.config.use_cache = True
    with torch.no_grad():
        gen = model.generate(input_ids=input_ids[:, :512], max_new_tokens=24, do_sample=False)
    log(f"generate OK: {tok.decode(gen[0, 512:], skip_special_tokens=True)[:80]!r}")

    peak = torch.cuda.max_memory_allocated(0) / 2**30
    print(json.dumps({"status": "OK", "mode": args.mode, "peak_vram_gib": round(peak, 1)}))


if __name__ == "__main__":
    main()
