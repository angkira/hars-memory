"""Torch-side helpers shared by train.py / gpu_smoke_test.py (GPU required)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def get_text_decoder_and_head(peft_model):
    """Return (text_decoder, lm_head) underneath a PeftModel wrapping
    Qwen3_5ForConditionalGeneration. Calling the text decoder directly skips
    full-vocab logits materialization so the loss can be chunked."""
    m = peft_model.base_model.model  # Qwen3_5ForConditionalGeneration
    text_model = m.model.language_model
    lm_head = m.lm_head
    return text_model, lm_head


def patch_fp8_linear_torch() -> None:
    """Replace transformers' triton finegrained-fp8 linear with a pure-torch
    blockwise-dequant implementation (the community triton kernel loads on
    ROCm but registers no autograd formula, so backward fails). Keeps weights
    in fp8; dequantizes per call — VRAM overhead is one layer's bf16 copy."""
    import torch.nn.functional as F
    import transformers.integrations.finegrained_fp8 as ffp8

    def dequant(weight, scale, block_size, out_dtype):
        bn, bk = (block_size or [128, 128])[:2]
        n, k = weight.shape
        s = scale.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)[:n, :k].to(out_dtype)
        return weight.to(out_dtype) * s

    def linear(input, weight, weight_scale_inv, block_size=None, bias=None,
               activation_scale=None, output_dtype=None, allow_deepgemm=True):
        w = dequant(weight, weight_scale_inv, block_size, input.dtype)
        return F.linear(input, w, bias)

    ffp8.finegrained_fp8_linear = linear
    ffp8.fp8_linear = linear
    ffp8.load_finegrained_fp8_kernel = lambda: None
    print("[patch] fp8 linear -> pure-torch blockwise dequant", flush=True)


@torch.no_grad()
def _resolve(model):
    return get_text_decoder_and_head(model)


def chunked_ce_loss(model, input_ids, attention_mask, labels, chunk: int = 512):
    """Completion-only cross entropy computed in hidden-state chunks.

    Avoids materializing [B, T, 248k] logits (≈9 GiB fp32 at T=6144); the
    lm_head+CE run per `chunk` positions instead (~0.8 GiB transient).

    Returns (loss_sum, n_items): use loss_sum / total_items_in_group for
    HF-style num_items_in_batch normalization across grad accumulation.
    """
    text_model, lm_head = _resolve(model)
    hidden = text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    shift_h = hidden[:, :-1, :]
    shift_l = labels[:, 1:]
    n_items = (shift_l != -100).sum()
    loss_sum = hidden.new_zeros((), dtype=torch.float32)
    T = shift_h.shape[1]
    for s in range(0, T, chunk):
        h = shift_h[:, s : s + chunk]
        y = shift_l[:, s : s + chunk]
        if (y != -100).sum() == 0:
            continue
        logits = lm_head(h).float()
        loss_sum = loss_sum + F.cross_entropy(
            logits.transpose(1, 2), y, ignore_index=-100, reduction="sum"
        )
    return loss_sum, n_items


def micro_batches(data, batch_size: int, seed: int, epoch: int):
    """Shuffled, length-grouped micro-batches.

    Yields (input_ids, labels, attn, n_tokens, n_label_items).
    """
    order = list(range(len(data)))
    rng = __import__("random").Random(seed + epoch)
    rng.shuffle(order)
    pool = batch_size * 64
    for s in range(0, len(order), pool):
        chunk = sorted(order[s : s + pool], key=lambda i: len(data[i][0]))
        for b in range(0, len(chunk), batch_size):
            idxs = chunk[b : b + batch_size]
            maxlen = max(len(data[i][0]) for i in idxs)
            input_ids = torch.full((len(idxs), maxlen), 0, dtype=torch.long)
            labels = torch.full((len(idxs), maxlen), -100, dtype=torch.long)
            attn = torch.zeros((len(idxs), maxlen), dtype=torch.long)
            for j, i in enumerate(idxs):
                n = len(data[i][0])
                input_ids[j, :n] = torch.tensor(data[i][0])
                labels[j, :n] = torch.tensor(data[i][1])
                attn[j, :n] = 1
            n_items = int((labels != -100).sum().item())
            yield (
                input_ids.cuda(), labels.cuda(), attn.cuda(),
                int(attn.sum().item()), n_items,
            )
