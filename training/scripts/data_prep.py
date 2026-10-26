#!/usr/bin/env python
"""Split the verified teacher SFT set into train / val / parity-held-out.

- 20 rows are held out for the parity gate (never trained on).
- 100 rows are used as a small validation split (loss tracking only).
- The rest (~2333) is the training split.

Deterministic given --seed. Writes jsonl to --out-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_jsonl, dump_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="eval_agentic/rca/sft_extraction.jsonl")
    ap.add_argument("--out-dir", default="training/outputs/data")
    ap.add_argument("--parity-size", type=int, default=20)
    ap.add_argument("--val-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    print(f"loaded {len(rows)} rows from {args.input}")

    # Prefer stable ids for the held-out split so it survives data regen.
    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)

    parity_idx = sorted(idx[: args.parity_size])
    val_idx = sorted(idx[args.parity_size : args.parity_size + args.val_size])
    train_idx = sorted(idx[args.parity_size + args.val_size :])

    def pick(idxs):
        return [rows[i] for i in idxs]

    dump_jsonl(pick(parity_idx), f"{args.out_dir}/parity_heldout.jsonl")
    dump_jsonl(pick(val_idx), f"{args.out_dir}/val.jsonl")
    dump_jsonl(pick(train_idx), f"{args.out_dir}/train.jsonl")

    stats = {
        "input": args.input,
        "seed": args.seed,
        "n_total": len(rows),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_parity": len(parity_idx),
        "parity_ids": [rows[i]["id"] for i in parity_idx],
    }
    with open(f"{args.out_dir}/split_meta.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "parity_ids"}, indent=2))
    print(f"parity ids written to {args.out_dir}/split_meta.json")


if __name__ == "__main__":
    main()
