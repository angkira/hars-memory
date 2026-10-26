#!/usr/bin/env python3
"""Evaluate M2 profiles with the overridden Qwen sidecar."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ML = Path(__file__).resolve().parent
EXP = ML.parent
REPO = ML.parents[2]
for path in (REPO, EXP, ML):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cands import SIDECAR_DIR, build_cands  # noqa: E402
from evalml import evaluate_profile, load_chunks  # noqa: E402
from fuse_ml import fuse_prod, wsum_n  # noqa: E402
from run_m1_m4 import load_all_cases  # noqa: E402

TOP_K = 20


def _rank_channel(channel: str):
    return lambda _case, candidates: [chunk_id for chunk_id, _ in candidates[channel][:TOP_K]]


def _rank_hybrid(_case: dict[str, Any], candidates: dict[str, Any]) -> list[str]:
    return fuse_prod(candidates["qwen"], candidates["bm25"], alpha=0.5, pool=15, top_k=TOP_K)


def _rank_three_way(_case: dict[str, Any], candidates: dict[str, Any]) -> list[str]:
    return wsum_n(
        {channel: candidates[channel] for channel in ("gemma", "qwen", "bm25")},
        (0.2, 0.4, 0.4),
        pool=20,
        norm="minmax",
        top_k=TOP_K,
    )


def main() -> None:
    if not os.environ.get("ML_SIDECAR_DIR"):
        raise RuntimeError("Set ML_SIDECAR_DIR to the Qwen sidecar directory before running M2T")
    meta_path = SIDECAR_DIR / "meta.json"
    metadata = json.loads(meta_path.read_text("utf-8"))
    chunks = load_chunks()
    if int(metadata.get("n", -1)) != len(chunks):
        raise ValueError(
            f"Sidecar must cover all {len(chunks)} chunks; metadata reports n={metadata.get('n')}"
        )
    if "elapsed_seconds" not in metadata:
        raise ValueError(f"Sidecar metadata is missing elapsed_seconds: {meta_path}")

    cases = load_all_cases()
    queries = [str(case["question"]) for case in cases]
    candidates = build_cands(queries, tag="all_cases")
    profiles = (
        ("qwen1024_dense", _rank_channel("qwen")),
        ("qwen1024_hybrid", _rank_hybrid),
        ("qwen1024_3way", _rank_three_way),
    )
    for profile, ranker in profiles:
        evaluate_profile("M2T", profile, cases, candidates, chunks, ranker)
        print(f"Saved M2T/{profile}")


if __name__ == "__main__":
    main()