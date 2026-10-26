#!/usr/bin/env python3
"""Prepare SFT splits for the extraction fine-tune.

Input : eval_agentic/rca/sft_extraction.jsonl  (2453 verified teacher pairs)
        Each `input` field is LightRAG's cache-key concatenation
        `user_prompt + "\n" + system_prompt` (verified in sft_stats.md), so we
        split each sample at the LAST "\n---Role---" to recover the exact
        (system, user) messages used by the 27B teacher. The corpus contains 3
        system-prompt variants (pipeline schema eras); we keep each sample's
        own variant — zero reconstruction risk.

Output: train.jsonl / val.jsonl / parity_holdout.jsonl + prep_report.json
        Split: 20 parity holdout, 30 val, remainder train. Seed 42.

Also normalizes outputs:
  - strips trailing whitespace, ensures exactly one trailing "<|COMPLETE|>"
  - if COMPLETE is missing (truncated teacher gen), drops the trailing partial
    record line and appends "<|COMPLETE|>"
  - drops samples whose non-empty output lines are <90% well-formed records
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/home/angkira/Project/software/hars-longterm-memory/eval_agentic/rca/sft_extraction.jsonl"
)
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "data"
N_PARITY = 20
N_VAL = 30
SEED = 42

RECORD_RE = re.compile(r"^(entity|relation)<\|#\|>")
COMPLETE = "<|COMPLETE|>"


def clean_output(out: str) -> tuple[str, bool]:
    """Return (cleaned_output, ok). ok=False -> drop sample."""
    lines = out.rstrip().splitlines()
    # quality gate: >=90% of non-empty lines must be well-formed records
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return "", False
    good = sum(1 for ln in nonempty if RECORD_RE.match(ln))
    if good / len(nonempty) < 0.90:
        return "", False
    # ensure exactly one trailing COMPLETE
    body = [ln for ln in lines if ln.strip() and ln.strip() != COMPLETE]
    # drop trailing partial record (teacher truncation): last line without a
    # delimiter is either a stray COMPLETE fragment or a cut record
    if body and not RECORD_RE.match(body[-1]):
        body = body[:-1]
    if not body:
        return "", False
    return "\n".join(body) + "\n" + COMPLETE, True


def split_user_system(inp: str) -> tuple[str, str]:
    idx = inp.rfind("\n---Role---")
    if idx < 0:
        raise ValueError("no system prompt marker found")
    return inp[idx + 1:], inp[:idx]  # (system, user)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples, dropped = [], {"malformed_output": 0, "no_system_marker": 0}
    sys_variants: dict[int, int] = {}
    with SRC.open() as f:
        for line in f:
            d = json.loads(line)
            try:
                system, user = split_user_system(d["input"])
            except ValueError:
                dropped["no_system_marker"] += 1
                continue
            out, ok = clean_output(d["output"])
            if not ok:
                dropped["malformed_output"] += 1
                continue
            samples.append({
                "id": d["id"],
                "system": system,
                "user": user,
                "output": out,
            })
            sys_variants[len(system)] = sys_variants.get(len(system), 0) + 1

    rng = random.Random(SEED)
    rng.shuffle(samples)
    parity, val, train = samples[:N_PARITY], samples[N_PARITY:N_PARITY + N_VAL], samples[N_PARITY + N_VAL:]

    def dump(name: str, rows: list[dict]) -> None:
        with (OUT_DIR / name).open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("train.jsonl", train)
    dump("val.jsonl", val)
    dump("parity_holdout.jsonl", parity)

    report = {
        "source": str(SRC),
        "total_in": sum(dropped.values()) + len(samples),
        "kept": len(samples),
        "dropped": dropped,
        "system_prompt_variants_by_len": sys_variants,
        "splits": {"train": len(train), "val": len(val), "parity": len(parity)},
        "seed": SEED,
    }
    (OUT_DIR / "prep_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
