#!/usr/bin/env python
"""Trace where non-finite gradients first appear in backward order.

Runs in fp8-patch mode so it fits alongside the perception job (~17 GiB free).
Fires backward hooks on every decoder layer + the final norm + lm_head input,
reporting hooks in backward-fire order (top layer first).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_jsonl
from gpu_smoke_test import attach_lora, build_batch, load_model
from torch_utils import chunked_ce_loss, patch_fp8_linear_torch


def main() -> None:
    import torch

    patch_fp8_linear_torch()
    tok, model = load_model("fp8", os.path.expanduser("~/models/qwen3.5-9b-fp8"), 3072)
    model = attach_lora(model, 32, 64, 0.05)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    hs = model.base_model.model.model.language_model
    records = []

    def grad_hook(name):
        def hook(mod, gin, gout):
            vals = [g for g in gin if isinstance(g, torch.Tensor)]
            if not vals:
                return
            finite = all(torch.isfinite(g).all().item() for g in vals)
            try:
                mx = max(float(g.abs().max()) for g in vals)
            except Exception:
                mx = float("nan")
            records.append((name, finite, mx))
        return hook

    for i, layer in enumerate(hs.layers):
        layer.register_full_backward_hook(grad_hook(f"layer{i:02d}"))
    hs.norm.register_full_backward_hook(grad_hook("final_norm"))
    hs.embed_tokens.register_full_backward_hook(grad_hook("embed_tokens"))

    rows = load_jsonl("../outputs/data/train.jsonl")
    ids, labels = build_batch(tok, rows[0], 3072)
    attn = torch.ones_like(ids)
    loss_sum, n = chunked_ce_loss(model, ids, attn, labels)
    print(f"loss={loss_sum.item() / n.item():.4f}")
    (loss_sum / n).backward()

    print(f"{'hook (backward order)':28s} {'finite':8s} max|grad|")
    for name, finite, mx in records:
        flag = "OK" if finite else "NaN/Inf"
        print(f"{name:28s} {flag:8s} {mx:.4e}")
    first_bad = next((r for r in records if not r[1]), None)
    print(f"\nFIRST non-finite grad at: {first_bad}")


if __name__ == "__main__":
    main()
