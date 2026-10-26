#!/usr/bin/env python3
"""QLoRA fine-tune of Qwen/Qwen3.5-4B for structured entity/relation extraction.

Distills the Qwen3.6-27B teacher data (eval_agentic/rca/sft_extraction.jsonl,
prepared splits in ./data). Runs on a Colab L4 (24GB).

Design notes
------------
* Plain transformers + peft + bitsandbytes (arch `qwen3_5` is brand new; the
  HF-native path is more reliable than unsloth for day-one archs).
* QLoRA: NF4 double-quant base, LoRA r=32 alpha=32 on ALL Linear modules under
  `language_model` (full-attn q/k/v/o + GDN linear-attn projections + MLP),
  vision tower / projector excluded (text-only task).
* Think-safety: Qwen3.5 small models are non-thinking natives; the chat
  template may or may not prefill an empty `<think>\n\n</think>` block in the
  generation prompt. We handle both renders adaptively and train 10% of
  examples with the empty block inside the completion so the model never
  emits an open <think> in either serving mode (prior pipeline died of
  think-leak; see eval_agentic/rca/small_model_eval.md §D).
* Labels: prompt masked (-100); loss on assistant output + template close + eos.
* MAX_SEQ_LEN 12288 with middle-truncation of the fenced input text only —
  outputs are never truncated.
* Wall-clock guard: stops training gracefully at MAX_SECONDS so the CU budget
  holds; adapter is saved regardless.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

# ----------------------------- config ---------------------------------------
MODEL_ID = "Qwen/Qwen3.5-4B"
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "runs" / "qlora-v1"
ADAPTER_DIR = OUT_DIR / "adapter"
MERGED_DIR = OUT_DIR / "merged"

MAX_SEQ_LEN = 12288
NUM_EPOCHS = 2
PER_DEVICE_BS = 2
GRAD_ACCUM = 4
LR = 2e-4
WARMUP_RATIO = 0.1
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_SECONDS = 13800          # 3h50m wall-clock guard for the train phase
THINK_VARIANT_FRAC = 0.10    # fraction of examples with empty-think in completion
SEED = 42
EVAL_STEPS = 75
SAMPLE_N = 5                 # qualitative generations from the parity holdout
MAX_NEW_TOKENS = 2816

THINK_BLOCK = "<think>\n\n</think>\n\n"
LOG_EVERY = 5


def log(msg: str) -> None:
    print(f"[train {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------- tokenizer / template -------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)

_msgs_probe = [
    {"role": "system", "content": "S."},
    {"role": "user", "content": "U."},
]
P = tok.apply_chat_template(_msgs_probe, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False)
PREFILL_THINK = bool(re.search(r"<think>\s*</think>\s*\n*$", P))
if "<think>" in P:
    m = re.search(r"<think>\s*</think>(\s*\n*)$", P)
    THINK_BLOCK = m.group(0)
log(f"template: generation prompt {'PREFILLS' if PREFILL_THINK else 'does NOT prefill'} "
    f"empty think block; block literal = {THINK_BLOCK!r}")

# assistant closing suffix via continue_final_message diff
_full_open = tok.apply_chat_template(
    _msgs_probe + [{"role": "assistant", "content": "OUT"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
    continue_final_message=True)
_full_closed = tok.apply_chat_template(
    _msgs_probe + [{"role": "assistant", "content": "OUT"}],
    tokenize=False, add_generation_prompt=False, enable_thinking=False)
CLOSE_SUFFIX = _full_closed[len(_full_open):] if _full_closed.startswith(_full_open) else ""
log(f"assistant close suffix: {CLOSE_SUFFIX!r}")
EOS_ID = tok.eos_token_id
PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else EOS_ID


# ----------------------------- dataset --------------------------------------
def middle_truncate_user(user: str, drop_tokens_est: int) -> str:
    """Middle-truncate the fenced input text block of the user prompt."""
    m = re.search(r"```\n(.*?)\n```", user, re.S)
    if not m:
        return user
    body = m.group(1)
    drop_chars = int(drop_tokens_est * 3.6)
    if drop_chars >= len(body) - 200:
        drop_chars = max(0, len(body) - 200)
    half = drop_chars // 2
    new_body = body[: len(body) // 2 - half] + "\n[...TRUNCATED...]\n" + body[len(body) // 2 + half:]
    return user[: m.start()] + "```\n" + new_body + "\n```" + user[m.end():]


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.items = []
        n_think_variant = 0
        n_trunc = 0
        for i, r in enumerate(rows):
            # --- prompt ids (template-rendered) ---
            msgs = [{"role": "system", "content": r["system"]},
                    {"role": "user", "content": r["user"]}]
            prompt_text = tok.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True,
                                                  enable_thinking=False)
            # --- completion via template (continue_final_message) ---
            open_text = tok.apply_chat_template(
                msgs + [{"role": "assistant", "content": r["output"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
                continue_final_message=True)
            completion_text = open_text[len(prompt_text):] if open_text.startswith(prompt_text) else r["output"]
            # strip a template-inserted empty think block from completion; we
            # re-add it on the chosen side of the 90/10 split below
            m = re.match(r"^<think>\s*</think>(\s*\n*)", completion_text)
            if m:
                completion_text = completion_text[m.end():]

            p_ids = tok(prompt_text).input_ids
            c_ids = tok(completion_text).input_ids
            total = len(p_ids) + len(c_ids) + len(tok(CLOSE_SUFFIX).input_ids) + 1
            if total > MAX_SEQ_LEN:
                n_trunc += 1
                user2 = r["user"]
                for _ in range(3):
                    user2 = middle_truncate_user(user2, total - MAX_SEQ_LEN)
                    pt = tok.apply_chat_template(
                        [{"role": "system", "content": r["system"]},
                         {"role": "user", "content": user2}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    total = len(tok(pt).input_ids) + len(c_ids) + len(tok(CLOSE_SUFFIX).input_ids) + 1
                    p_ids = tok(pt).input_ids
                    if total <= MAX_SEQ_LEN:
                        break

            # think-variant placement (never an open <think>)
            use_variant = (i / max(1, len(rows))) < THINK_VARIANT_FRAC and not PREFILL_THINK
            if PREFILL_THINK:
                # empty block already inside prompt; completion is pure records
                pass
            elif use_variant:
                c_ids = tok(THINK_BLOCK).input_ids + c_ids
                n_think_variant += 1
            else:
                p_ids = p_ids + tok(THINK_BLOCK).input_ids

            close_ids = tok(CLOSE_SUFFIX).input_ids if CLOSE_SUFFIX else []
            input_ids = p_ids + c_ids + close_ids + [EOS_ID]
            labels = [-100] * len(p_ids) + c_ids + close_ids + [EOS_ID]
            self.items.append({
                "input_ids": input_ids,
                "labels": labels,
                "length": len(input_ids),
            })
        log(f"dataset: {len(self.items)} rows, {n_trunc} truncated, "
            f"{n_think_variant} think-in-completion variants")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    maxlen = (maxlen + 511) // 512 * 512  # pad to 512 multiples
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [PAD_ID] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def load_rows(name: str) -> list[dict]:
    with (DATA_DIR / name).open() as f:
        return [json.loads(line) for line in f]


# ----------------------------- model ----------------------------------------
log("loading tokenizer/data/model ...")
train_rows, val_rows = load_rows("train.jsonl"), load_rows("val.jsonl")
train_ds = SFTDataset(train_rows)
val_ds = SFTDataset(val_rows)

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, quantization_config=bnb, dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model.config.use_cache = False

# LoRA targets: every Linear under language_model, excluding lm_head
linear_names = [
    n for n, mod in model.named_modules()
    if type(mod).__name__.lower().startswith("linear")
    and "language_model" in n and "lm_head" not in n
]
log(f"LoRA targets: {len(linear_names)} Linear modules under language_model; "
    f"suffixes={sorted({n.rsplit('.', 1)[-1] for n in linear_names})}")
assert linear_names, "no LoRA target modules found"

model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
lcfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="none", task_type="CAUSAL_LM", target_modules=linear_names,
)
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()

# smoke test: one fwd+bwd on the longest sample (fail fast before Trainer)
longest = max(train_ds.items, key=lambda x: x["length"])
batch = collate([longest])
batch = {k: v.cuda() for k, v in batch.items()}
out = model(**batch)
out.loss.backward()
model.zero_grad(set_to_none=True)
torch.cuda.empty_cache()
log(f"smoke test OK: seq_len={longest['length']} loss={out.loss.item():.4f} "
    f"peak_mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB")

# ----------------------------- callbacks -------------------------------------
T0 = time.time()
prog_path = OUT_DIR / "progress.json"
loss_log = OUT_DIR / "loss_log.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class GuardAndProgress(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        rec = {"step": state.global_step, "elapsed_s": round(time.time() - T0, 1), **{
            k: v for k, v in logs.items() if isinstance(v, (int, float))}}
        with loss_log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if state.max_steps:
            rate = state.global_step / max(1e-9, time.time() - T0)
            prog_path.write_text(json.dumps({
                "step": state.global_step, "max_steps": state.max_steps,
                "elapsed_s": round(time.time() - T0),
                "eta_s": round((state.max_steps - state.global_step) / max(rate, 1e-9)),
                **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
            }, indent=2))
        if "loss" in logs:
            log(f"step {state.global_step}/{state.max_steps} loss={logs['loss']:.4f} "
                f"lr={logs.get('learning_rate', 0):.2e} elapsed={(time.time()-T0)/60:.1f}min")

    def on_step_end(self, args, state, control, **kw):
        if time.time() - T0 > MAX_SECONDS:
            log(f"wall-clock guard tripped at step {state.global_step}; stopping")
            control.should_training_stop = True
        return control


args = TrainingArguments(
    output_dir=str(OUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BS,
    per_device_eval_batch_size=PER_DEVICE_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.0,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",
    logging_steps=LOG_EVERY,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="epoch",
    save_total_limit=2,
    group_by_length=False,
    length_column_name="length",
    remove_unused_columns=False,
    dataloader_num_workers=2,
    report_to=[],
    seed=SEED,
    save_safetensors=True,
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=collate, callbacks=[GuardAndProgress()],
)

log(f"train start: {len(train_ds)} train / {len(val_ds)} val rows, "
    f"{NUM_EPOCHS} epochs, effective bs {PER_DEVICE_BS*GRAD_ACCUM}, lr {LR}")
trainer.train()

# ----------------------------- save ------------------------------------------
log("saving adapter ...")
model.save_pretrained(ADAPTER_DIR)
tok.save_pretrained(ADAPTER_DIR)

hist = trainer.state.log_history
curve = [
    {k: v for k, v in h.items() if k in ("step", "loss", "eval_loss", "learning_rate", "epoch")}
    for h in hist if "loss" in h or "eval_loss" in h
]
(OUT_DIR / "loss_curve.json").write_text(json.dumps(curve, indent=2))
train_loss = [h["loss"] for h in hist if "loss" in h]
eval_loss = [h["eval_loss"] for h in hist if "eval_loss" in h]
summary = {
    "final_train_loss": train_loss[-1] if train_loss else None,
    "min_eval_loss": min(eval_loss) if eval_loss else None,
    "eval_loss_series": eval_loss,
    "total_steps": hist[-1]["step"] if hist else None,
    "train_seconds": round(time.time() - T0),
    "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
}
(OUT_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))
log(f"train done: {json.dumps(summary)}")

# ----------------------------- merge -----------------------------------------
log("merging LoRA into bf16 base ...")
del model, trainer
torch.cuda.empty_cache()
base = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0},
)
peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR)
merged = peft_model.merge_and_unload()
MERGED_DIR.mkdir(parents=True, exist_ok=True)
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tok.save_pretrained(MERGED_DIR)
try:
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    proc.save_pretrained(MERGED_DIR)
except Exception as e:
    log(f"processor save skipped: {e}")
log(f"merged model saved to {MERGED_DIR}")

# ----------------------------- qualitative samples ---------------------------
log("generating qualitative samples from parity holdout ...")
parity = load_rows("parity_holdout.jsonl")
merged.eval()
samples = []
with torch.no_grad():
    for r in parity[:SAMPLE_N]:
        msgs = [{"role": "system", "content": r["system"]},
                {"role": "user", "content": r["user"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
        if not PREFILL_THINK:
            prompt = prompt + THINK_BLOCK
        ids = tok(prompt, return_tensors="pt").to("cuda")
        t0 = time.time()
        out_ids = merged.generate(
            **ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            repetition_penalty=1.0,
            pad_token_id=PAD_ID,
        )
        gen = out_ids[0][ids["input_ids"].shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True)
        dt = time.time() - t0
        samples.append({
            "id": r["id"],
            "gen_seconds": round(dt, 1),
            "tokens": int(gen.shape[0]),
            "tokens_per_s": round(gen.shape[0] / dt, 1),
            "prediction": text,
            "reference": r["output"],
        })
        log(f"sample {r['id']}: {gen.shape[0]} tok in {dt:.0f}s "
            f"({gen.shape[0]/dt:.0f} tok/s), think_leak={'<think>' in text}")
(OUT_DIR / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False))
log("ALL DONE — merged model + samples ready")
