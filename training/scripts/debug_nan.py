#!/usr/bin/env python
"""NaN debugger: locate where the fp8/LoRA training path diverges.

Stages:
  fwd3      — 3 forward-only passes (is forward stable?)
  bwdo      — forward+backward, NO optimizer step (are grads finite?)
  full      — forward+backward+step (reproduce NaN, then find first NaN param)
  anomaly   — forward+backward+step with torch.autograd.set_detect_anomaly
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_jsonl
from gpu_smoke_test import build_batch, load_model
from torch_utils import chunked_ce_loss, patch_fp8_linear_torch


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="bwdo", choices=["fwd3", "bwdo", "full", "anomaly"])
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--mode", default="fp8")
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    if args.mode == "fp8":
        patch_fp8_linear_torch()
    tok, model = load_model(args.mode, os.path.expanduser("~/models/qwen3.5-9b-fp8"), args.seq_len)
    rows = load_jsonl("../outputs/data/train.jsonl")
    model = model  # attach lora like smoke test
    from gpu_smoke_test import attach_lora

    model = attach_lora(model, 32, 64, 0.05)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95))

    ids, labels = build_batch(tok, rows[0], args.seq_len)
    attn = torch.ones_like(ids)

    def fwd():
        loss_sum, n = chunked_ce_loss(model, ids, attn, labels)
        return loss_sum / n

    if args.stage == "fwd3":
        for i in range(3):
            with torch.no_grad():
                print(f"fwd{i}: loss={fwd().item():.4f}")
        return

    if args.stage == "anomaly":
        torch.autograd.set_detect_anomaly(True)

    loss = fwd()
    print(f"loss before bwd: {loss.item():.4f}")
    loss.backward()

    bad = [(n, p.grad.abs().max().item()) for n, p in model.named_parameters()
           if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
    print(f"non-finite grads: {bad[:8]} (of {len(params)})")
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    print(f"total grad norm: {gn.item() if torch.is_tensor(gn) else gn}")

    if args.stage in ("full", "anomaly"):
        opt.step()
        opt.zero_grad(set_to_none=True)
        bad_p = [(n, p.abs().max().item()) for n, p in model.named_parameters()
                 if p.requires_grad and not torch.isfinite(p).all()]
        print(f"non-finite params after step: {bad_p[:8]}")
        with torch.no_grad():
            print(f"loss after step: {fwd().item():.4f}")


if __name__ == "__main__":
    main()
