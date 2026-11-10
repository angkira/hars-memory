#!/usr/bin/env python
"""LoRA SFT for structured extraction on the R9700 (ROCm).

Trains a LoRA adapter on the verified teacher extraction pairs with
completion-only loss (the prompt is fully masked). The base checkpoint keeps
its blockwise-FP8 weights (default) — only bf16 LoRA params + activations
are trainable memory.

Safety:
- Refuses to start if another process holds too much VRAM.
- Watches device-wide free VRAM every step (torch.cuda.mem_get_info sees
  *all* processes on the GPU); below --guard-free-gib it checkpoints and
  exits cleanly with status CONTENTION_STOP.

Example:
  python train.py --config ../configs/train_9b_fp8.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import encode_completion_only, load_jsonl
from torch_utils import chunked_ce_loss, micro_batches

DEFAULT_BASE = os.path.expanduser("~/models/qwen3.5-9b-fp8")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    import torch

    torch.manual_seed(seed)


def load_config(path: str | None, overrides: dict) -> dict:
    cfg = {
        "base_model": DEFAULT_BASE,
        "mode": "fp8",  # fp8 | bf16 | nf4
        "train_jsonl": "../outputs/data/train.jsonl",
        "val_jsonl": "../outputs/data/val.jsonl",
        "out_dir": "../outputs/run_9b_fp8_r32",
        "max_seq_len": 8192,
        "drop_longer_than": 12288,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "epochs": 2,
        "batch_size": 1,
        "grad_accum": 8,
        "lr": 1e-4,
        "min_lr_ratio": 0.1,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": 42,
        "log_every": 5,
        "eval_every": 150,
        "eval_samples": 50,
        "guard_free_gib": 2.5,
        "min_free_gib_to_start": 15.0,
        "resume": None,
    }
    if path:
        with open(path) as f:
            cfg.update(json.load(f))
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def device_free_gib() -> float:
    import torch

    free, _ = torch.cuda.mem_get_info(0)
    return free / 2**30


def load_model(cfg: dict):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    kwargs = dict(dtype="auto", device_map={"": "cuda:0"}, attn_implementation="sdpa")
    if cfg["mode"] == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    elif cfg["mode"] == "bf16":
        from transformers import FineGrainedFP8Config

        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(cfg["base_model"], **kwargs)
    model.config.use_cache = False
    log(f"base model loaded in {time.time() - t0:.0f}s (mode={cfg['mode']})")
    return tok, model


def attach_lora(cfg: dict, model):
    from peft import LoraConfig, get_peft_model

    model.enable_input_require_grads()
    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=cfg["target_modules"],
    )
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    return model


def tokenize_rows(tok, rows, cfg, desc: str, limit: int | None = None):
    """-> list of (ids, labels); drops rows over drop_longer_than."""
    data, dropped = [], 0
    rows = rows if limit is None else rows[:limit]
    for i, r in enumerate(rows):
        ids, labels = encode_completion_only(tok, r["input"], r["output"])
        if len(ids) > cfg["drop_longer_than"]:
            dropped += 1
            continue
        data.append((ids, labels))
    lens = sorted(len(x[0]) for x in data)
    log(f"{desc}: {len(data)} rows ({dropped} dropped > {cfg['drop_longer_than']} tok), "
        f"len med={lens[len(lens)//2]} p90={lens[int(0.9*len(lens))]} max={lens[-1]}")
    return data


def lr_lambda_factory(cfg: dict, total_steps: int):
    warmup = max(1, int(cfg["warmup_ratio"] * total_steps))
    min_ratio = cfg["min_lr_ratio"]

    def f(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        prog = min(1.0, prog)
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return min_ratio + (1 - min_ratio) * cos

    return f


def save_checkpoint(model, tok, cfg, out_dir: dict | None = None):
    import os

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-model", dest="base_model", default=None)
    ap.add_argument("--out-dir", dest="out_dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--mode", default=None, choices=["fp8", "bf16", "nf4"])
    ap.add_argument("--lora-r", dest="lora_r", type=int, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, {
        "base_model": args.base_model, "out_dir": args.out_dir, "epochs": args.epochs,
        "mode": args.mode, "lora_r": args.lora_r, "resume": args.resume,
    })
    here = os.path.dirname(os.path.abspath(__file__))
    for k in ("train_jsonl", "val_jsonl", "out_dir"):
        if not os.path.isabs(cfg[k]):
            cfg[k] = os.path.normpath(os.path.join(here, cfg[k]))

    os.makedirs(cfg["out_dir"], exist_ok=True)
    log_path = os.path.join(cfg["out_dir"], "train_log.jsonl")

    free = device_free_gib()
    log(f"device free VRAM at start: {free:.1f} GiB")
    if free < cfg["min_free_gib_to_start"]:
        log(f"ABORT: only {free:.1f} GiB free (< {cfg['min_free_gib_to_start']}); another job may own the GPU")
        with open(log_path, "a") as f:
            f.write(json.dumps({"event": "ABORT_NO_VRAM", "free_gib": free}) + "\n")
        sys.exit(2)

    set_seed(cfg["seed"])
    tok, model = load_model(cfg)
    model = attach_lora(cfg, model)
    model.train()

    train_rows = load_jsonl(cfg["train_jsonl"])
    val_rows = load_jsonl(cfg["val_jsonl"])
    train_data = tokenize_rows(tok, train_rows, cfg, "train")
    val_data = tokenize_rows(tok, val_rows, cfg, "val", limit=cfg["eval_samples"])

    steps_per_epoch = math.ceil(len(train_data) / (cfg["batch_size"] * cfg["grad_accum"]))
    total_steps = steps_per_epoch * cfg["epochs"]
    log(f"plan: {cfg['epochs']} epochs x {steps_per_epoch} optim steps = {total_steps}")

    params = [p for p in model.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in params)
    log(f"trainable params: {n_lora/1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(cfg, total_steps))

    start_epoch, start_step, gstep = 0, 0, 0
    if cfg["resume"]:
        state = torch.load(os.path.join(cfg["resume"], "trainer_state.pt"), map_location="cuda:0", weights_only=False)

        model.load_adapter(cfg["resume"], adapter_name="default")
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch, start_step, gstep = state["epoch"], state["step"], state["gstep"]
        log(f"resumed from {cfg['resume']} at epoch {start_epoch} step {start_step}")

    def guard_and_log(event: str, extra: dict | None = None):
        free = device_free_gib()
        rec = {"event": event, "free_gib": round(free, 2),
               "alloc_gib": round(torch.cuda.max_memory_allocated(0) / 2**30, 2)}
        if extra:
            rec.update(extra)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return free

    def run_eval():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for ids, labels, attn, _, items in micro_batches(val_data, cfg["batch_size"], cfg["seed"], 0):
                loss_sum, n_items = chunked_ce_loss(model, ids, attn, labels)
                tot += loss_sum.item()
                n += n_items.item()
        model.train()
        val_loss = tot / max(1, n)
        log(f"val_loss={val_loss:.4f} (n={len(val_data)})")
        guard_and_log("eval", {"val_loss": val_loss})
        return val_loss

    status = "DONE"
    t_start = time.time()
    for epoch in range(start_epoch, cfg["epochs"]):
        recent_loss = 0.0
        recent_micro = 0
        t_epoch = time.time()
        micro_iter = micro_batches(train_data, cfg["batch_size"], cfg["seed"], epoch)
        skip = start_step * cfg["grad_accum"] if epoch == start_epoch else 0
        step_in_epoch = 0
        done = skip >= len(train_data)
        while not done:
            group = []
            for _ in range(cfg["grad_accum"]):
                if skip > 0:
                    skip -= 1
                    try:
                        next(micro_iter)
                    except StopIteration:
                        pass
                    continue
                try:
                    group.append(next(micro_iter))
                except StopIteration:
                    done = True
                    break
            if not group:
                break
            group_total = sum(g[4] for g in group)
            group_loss = 0.0
            for ids, labels, attn, ntok, items in group:
                loss_sum, n_items = chunked_ce_loss(model, ids, attn, labels)
                (loss_sum / group_total).backward()
                group_loss += loss_sum.item()
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            gstep += 1
            step_in_epoch += 1
            recent_loss += group_loss / group_total
            recent_micro += 1
            if gstep % cfg["log_every"] == 0:
                lr_now = sched.get_last_lr()[0]
                dt = time.time() - t_epoch
                free = device_free_gib()
                log(f"ep{epoch} step {step_in_epoch}/{steps_per_epoch} (g{gstep}) "
                    f"loss={recent_loss / max(1, recent_micro):.4f} "
                    f"lr={lr_now:.2e} {dt/cfg['log_every']:.1f}s/step free={free:.1f}GiB")
                guard_and_log("train", {"step": gstep, "epoch": epoch, "loss": recent_loss / max(1, recent_micro), "lr": lr_now})
                if free < cfg["guard_free_gib"]:
                    log(f"VRAM guard tripped ({free:.1f} GiB free) — checkpointing and stopping")
                    status = "CONTENTION_STOP"
                    break
                recent_loss = 0.0
                recent_micro = 0
                t_epoch = time.time()
            if gstep % cfg["eval_every"] == 0:
                run_eval()
            if status != "DONE":
                break
        if status != "DONE":
            break
        ckpt = os.path.join(cfg["out_dir"], f"checkpoint-ep{epoch + 1}")
        save_checkpoint(model, tok, cfg, ckpt)
        torch.save({"epoch": epoch + 1, "step": 0, "gstep": gstep, "opt": opt.state_dict(), "sched": sched.state_dict()},
                   os.path.join(ckpt, "trainer_state.pt"))
        log(f"saved checkpoint {ckpt}")
        run_eval()

    if status == "DONE":
        final = os.path.join(cfg["out_dir"], "adapter")
        save_checkpoint(model, tok, cfg, final)
        log(f"saved final adapter {final}")
    elif status == "CONTENTION_STOP":
        ckpt = os.path.join(cfg["out_dir"], "checkpoint-contention")
        save_checkpoint(model, tok, cfg, ckpt)
        torch.save({"epoch": epoch, "step": step_in_epoch, "gstep": gstep,
                    "opt": opt.state_dict(), "sched": sched.state_dict()},
                   os.path.join(ckpt, "trainer_state.pt"))
        log(f"saved contention checkpoint {ckpt} (resume with --resume {ckpt})")
    guard_and_log("end", {"status": status, "wall_hours": round((time.time() - t_start) / 3600, 2)})
    print(json.dumps({"status": status, "out_dir": cfg["out_dir"], "gsteps": gstep,
                      "wall_hours": round((time.time() - t_start) / 3600, 2)}))
    sys.exit(0 if status == "DONE" else 3)


if __name__ == "__main__":
    main()
