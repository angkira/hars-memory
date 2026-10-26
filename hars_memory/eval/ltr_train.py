#!/usr/bin/env python3
"""Train the learned channel-fusion ranker and pick the winning backend.

Trains BOTH a LightGBM LambdaRank model and a tiny torch-CPU MLP on the same
query-level split, evaluates each on the held-out queries, and writes the
artifact (manifest + model file) for whichever scores higher on held-out
NDCG@10. Also writes the loser's model file next to it so the comparison stays
reproducible.

The split is BY QUERY (see hars_memory/ranking/train.py). Held-out query sets
passed via ``--test-features`` are NEVER used for model selection or early
stopping — they are reported only.

Usage:
    python -m hars_memory.eval.ltr_train \
        --features eval_agentic-data/ltr_train.jsonl \
        --out hars_memory/ranking/artifacts/ranker.json \
        --metrics-json eval_agentic-data/ltr_train_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from hars_memory.ranking.artifact import FORMAT_LIGHTGBM_TXT, FORMAT_MLP_JSON
from hars_memory.ranking.train import (
    DEFAULT_K,
    QueryGroup,
    TrainLtrRankerError,
    evaluate_groups,
    lightgbm_predictor,
    load_groups,
    mlp_predictor,
    split_groups,
    train_lightgbm,
    train_mlp,
    write_manifest,
)

logger = logging.getLogger("memory.eval.ltr_train")


def _scorer_from_mlp(payload: dict) -> "callable":
    return mlp_predictor(payload)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True, help="Training JSONL from ltr_dataset.")
    parser.add_argument("--out", type=Path, required=True, help="Manifest JSON path to write.")
    parser.add_argument("--valid-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--backends", nargs="+", default=["lightgbm", "mlp"], choices=["lightgbm", "mlp"])
    parser.add_argument("--num-boost-round", type=int, default=400)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--mlp-epochs", type=int, default=150)
    parser.add_argument(
        "--test-features",
        type=Path,
        action="append",
        default=None,
        help="Extra fully held-out JSONL dumps, reported only (repeatable).",
    )
    parser.add_argument("--metrics-json", type=Path, default=None)
    return parser.parse_args(argv)


def _group_stats(groups: list[QueryGroup]) -> dict[str, float]:
    rows = sum(len(g.labels) for g in groups)
    pos = sum(1 for g in groups for label in g.labels if label > 0)
    return {"queries": float(len(groups)), "rows": float(rows), "positives": float(pos)}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    groups, feature_names = load_groups(args.features)
    train_groups, valid_groups = split_groups(groups, valid_fraction=args.valid_fraction, seed=args.seed)
    logger.info(
        "loaded %d queries (%d train / %d valid), %d features",
        len(groups), len(train_groups), len(valid_groups), len(feature_names),
    )

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    scorers: dict[str, "callable"] = {}

    if "lightgbm" in args.backends:
        ranker = train_lightgbm(
            train_groups,
            valid_groups,
            seed=args.seed,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        model_file = "ranker.lgb.txt"
        ranker.booster_.save_model(str(out_dir / model_file))
        best_iteration = getattr(ranker, "best_iteration_", None)
        lgbm_predict = lightgbm_predictor(ranker)
        metrics = {
            "valid": evaluate_groups(valid_groups, lgbm_predict, args.k),
            "train": evaluate_groups(train_groups, lgbm_predict, args.k),
        }
        results["lightgbm"] = {
            "format": FORMAT_LIGHTGBM_TXT,
            "model_file": model_file,
            "metrics": metrics,
            "hyperparams": {
                "num_boost_round": args.num_boost_round,
                "early_stopping_rounds": args.early_stopping_rounds,
                "best_iteration": best_iteration,
                "objective": "lambdarank",
                "num_leaves": 15,
                "learning_rate": 0.05,
            },
        }
        scorers["lightgbm"] = lgbm_predict

    if "mlp" in args.backends:
        payload = train_mlp(train_groups, valid_groups, seed=args.seed, epochs=args.mlp_epochs)
        mlp_file = "ranker_mlp.json"
        (out_dir / mlp_file).write_text(json.dumps(payload), encoding="utf-8")
        predictor = _scorer_from_mlp(payload)
        metrics = {
            "valid": evaluate_groups(valid_groups, predictor, args.k),
            "train": evaluate_groups(train_groups, predictor, args.k),
        }
        results["mlp"] = {
            "format": FORMAT_MLP_JSON,
            "model_file": mlp_file,
            "metrics": metrics,
            "hyperparams": {"hidden": payload.get("hidden"), "epochs": args.mlp_epochs, "objective": "ranknet"},
        }
        scorers["mlp"] = predictor

    # Optional extra held-out sets: reported ONLY, never used for selection.
    for test_path in args.test_features or ():
        test_groups, _ = load_groups(test_path)
        for backend, scorer in scorers.items():
            results[backend]["metrics"].setdefault("holdout", {})[test_path.name] = evaluate_groups(
                test_groups, scorer, args.k
            )

    key = f"ndcg@{args.k}"
    winner = max(results, key=lambda name: results[name]["metrics"]["valid"][key])
    logger.info("winner: %s (valid %s=%.4f)", winner, key, results[winner]["metrics"]["valid"][key])

    write_manifest(
        args.out,
        model_format=results[winner]["format"],
        model_file=results[winner]["model_file"],
        metrics={name: results[name]["metrics"] for name in results},
        hyperparams={
            "winner": winner,
            "seed": args.seed,
            "valid_fraction": args.valid_fraction,
            "k": args.k,
            **results[winner]["hyperparams"],
        },
        dataset={
            "features_file": str(args.features),
            "train": _group_stats(train_groups),
            "valid": _group_stats(valid_groups),
            "feature_names": list(feature_names),
        },
    )

    report = {
        "winner": winner,
        "features": str(args.features),
        "train_queries": len(train_groups),
        "valid_queries": len(valid_groups),
        "results": {name: {k: v for k, v in info.items()} for name, info in results.items()},
    }
    print(json.dumps(report, indent=2, default=str))
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainLtrRankerError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
