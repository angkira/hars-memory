"""Train the learned channel-fusion ranker (LightGBM LambdaRank + tiny MLP).

Input is the JSONL written by :mod:`hars_memory.eval.ltr_dataset`: an optional
header line carrying ``feature_names``, then one object per (query, candidate)
with ``query_id``, ``features`` and a graded ``label`` (2 = evidence chunk of a
gold document, 1 = other chunk of a gold document, 0 = irrelevant).

THE SPLIT IS BY QUERY, NEVER BY ROW
-----------------------------------
Candidates of one query are correlated with each other, so a row-level split
leaks the answer into the training half and the held-out score becomes a
description of the leak. The query-level split mirrors code-diver's H-91a
ranker and is what makes the reported test numbers meaningful.

Backends: LightGBM ``lambdarank`` (the right, list-aware tool) and a tiny
torch-CPU MLP trained with a RankNet pairwise loss. Both are compared on the
same held-out split; see :mod:`hars_memory.eval.ltr_train`.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from hars_memory.ranking.artifact import (
    FORMAT_LIGHTGBM_TXT,
    FORMAT_MLP_JSON,
    LTR_MANIFEST_VERSION,
)
from hars_memory.ranking.features import LTR_FEATURE_NAMES, FEATURE_DOCS

logger = logging.getLogger(__name__)

DEFAULT_K = 10


class TrainLtrRankerError(RuntimeError):
    pass


@dataclass(slots=True)
class QueryGroup:
    query_id: str
    features: list[list[float]]
    labels: list[int]


# ---------------------------------------------------------------------------
# Dataset loading / splitting
# ---------------------------------------------------------------------------


def load_groups(path: Path | str) -> tuple[list[QueryGroup], tuple[str, ...]]:
    feature_names: tuple[str, ...] = ()
    by_query: dict[str, QueryGroup] = {}
    order: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise TrainLtrRankerError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if "feature_names" in payload and "features" not in payload:
                feature_names = tuple(str(name) for name in payload["feature_names"])
                continue
            query_id = str(payload["query_id"])
            group = by_query.get(query_id)
            if group is None:
                group = QueryGroup(query_id=query_id, features=[], labels=[])
                by_query[query_id] = group
                order.append(query_id)
            group.features.append([float(value) for value in payload["features"]])
            group.labels.append(int(payload["label"]))
    if not by_query:
        raise TrainLtrRankerError(f"{path} contains no feature rows.")
    if feature_names and feature_names != LTR_FEATURE_NAMES:
        raise TrainLtrRankerError(
            f"{path} was exported with a different feature list than this checkout defines. "
            "Re-export the features before training."
        )
    return [by_query[query_id] for query_id in order], feature_names or LTR_FEATURE_NAMES


def split_groups(
    groups: Sequence[QueryGroup], *, valid_fraction: float, seed: int
) -> tuple[list[QueryGroup], list[QueryGroup]]:
    """Deterministic query-level split into (train, valid)."""
    ids = [group.query_id for group in groups]
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_valid = max(1, int(round(len(shuffled) * valid_fraction))) if len(shuffled) > 1 else 0
    valid_ids = set(shuffled[:n_valid])
    train = [g for g in groups if g.query_id not in valid_ids]
    valid = [g for g in groups if g.query_id in valid_ids]
    if not train:
        raise TrainLtrRankerError("query-level split left an empty training set; dataset is too small.")
    return train, valid


# ---------------------------------------------------------------------------
# Metrics (graded, used for training diagnostics)
# ---------------------------------------------------------------------------


def _dcg(labels_in_rank_order: Sequence[int], k: int) -> float:
    return sum(
        label / math.log2(rank + 1)
        for rank, label in enumerate(labels_in_rank_order[:k], start=1)
    )


def ndcg_at_k(labels_in_rank_order: Sequence[int], k: int) -> float:
    dcg = _dcg(labels_in_rank_order, k)
    ideal = _dcg(sorted(labels_in_rank_order, reverse=True), k)
    return dcg / ideal if ideal else 0.0


def hit_rate_at_k(labels_in_rank_order: Sequence[int], k: int) -> float:
    return 1.0 if any(label > 0 for label in labels_in_rank_order[:k]) else 0.0


def reciprocal_rank(labels_in_rank_order: Sequence[int]) -> float:
    for rank, label in enumerate(labels_in_rank_order, start=1):
        if label > 0:
            return 1.0 / rank
    return 0.0


def evaluate_groups(
    groups: Sequence[QueryGroup],
    scorer: Callable[[list[list[float]]], list[float]],
    k: int,
) -> dict[str, float]:
    """Mean NDCG@k / hit@k / MRR for a scorer, plus the same for the exported
    base order (the hand-tuned fused order the rows were dumped in)."""
    if not groups:
        return {"queries": 0.0}
    model_ndcg: list[float] = []
    model_hit: list[float] = []
    model_mrr: list[float] = []
    base_ndcg: list[float] = []
    base_hit: list[float] = []
    base_mrr: list[float] = []
    for group in groups:
        scores = scorer(group.features)
        ordered = [
            label
            for _, label in sorted(zip(scores, group.labels, strict=True), key=lambda entry: entry[0], reverse=True)
        ]
        model_ndcg.append(ndcg_at_k(ordered, k))
        model_hit.append(hit_rate_at_k(ordered, k))
        model_mrr.append(reciprocal_rank(ordered))
        base_ndcg.append(ndcg_at_k(group.labels, k))
        base_hit.append(hit_rate_at_k(group.labels, k))
        base_mrr.append(reciprocal_rank(group.labels))
    n = len(groups)
    return {
        "queries": float(n),
        f"ndcg@{k}": sum(model_ndcg) / n,
        f"hit@{k}": sum(model_hit) / n,
        "mrr": sum(model_mrr) / n,
        f"base_ndcg@{k}": sum(base_ndcg) / n,
        f"base_hit@{k}": sum(base_hit) / n,
        "base_mrr": sum(base_mrr) / n,
    }


# ---------------------------------------------------------------------------
# LightGBM LambdaRank
# ---------------------------------------------------------------------------


def train_lightgbm(
    train_groups: Sequence[QueryGroup],
    valid_groups: Sequence[QueryGroup],
    *,
    seed: int,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 400,
    early_stopping_rounds: int = 40,
):
    """Fit an ``LGBMRanker`` with query-level early stopping on NDCG@10."""
    import lightgbm
    import numpy as np

    merged = {
        "objective": "lambdarank",
        "n_estimators": num_boost_round,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 10,
        "random_state": seed,
        "verbose": -1,
        "label_gain": [0, 1, 3],
    }
    if params:
        merged.update(params)
    ranker = lightgbm.LGBMRanker(**merged)

    train_features = np.asarray([row for group in train_groups for row in group.features], dtype=np.float64)
    train_labels = [label for group in train_groups for label in group.labels]
    train_group_sizes = [len(group.labels) for group in train_groups]
    fit_kwargs: dict[str, Any] = {"group": train_group_sizes}
    if valid_groups:
        valid_features = np.asarray(
            [row for group in valid_groups for row in group.features], dtype=np.float64
        )
        valid_labels = [label for group in valid_groups for label in group.labels]
        valid_group_sizes = [len(group.labels) for group in valid_groups]
        fit_kwargs.update(eval_set=[(valid_features, valid_labels)], eval_group=[valid_group_sizes], eval_at=[10])
        fit_kwargs["callbacks"] = [lightgbm.early_stopping(early_stopping_rounds, verbose=False)]
    ranker.fit(train_features, train_labels, **fit_kwargs)
    return ranker


def save_lightgbm(ranker: Any, path: Path) -> None:
    ranker.booster_.save_model(str(path))


def lightgbm_predictor(ranker: Any) -> Callable[[list[list[float]]], list[float]]:
    """A list-of-lists scorer around a fitted LGBMRanker (LightGBM 4.7 rejects
    plain Python lists, so the conversion to a float64 array happens here)."""
    import numpy as np

    def predict(rows: list[list[float]]) -> list[float]:
        matrix = np.asarray([list(row) for row in rows], dtype=np.float64)
        return [float(value) for value in ranker.predict(matrix)]

    return predict


# ---------------------------------------------------------------------------
# Tiny MLP (torch CPU, RankNet pairwise loss) -- exported to dependency-light JSON
# ---------------------------------------------------------------------------


def train_mlp(
    train_groups: Sequence[QueryGroup],
    valid_groups: Sequence[QueryGroup],
    *,
    seed: int,
    hidden: Sequence[int] = (16, 8),
    epochs: int = 120,
    learning_rate: float = 0.01,
    activation: str = "relu",
    max_pairs_per_query: int = 40,
) -> dict[str, Any]:
    """Train a small MLP with a RankNet pairwise logistic loss on within-query
    (higher-label, lower-label) pairs. Returns a JSON-serializable payload."""
    import torch

    torch.manual_seed(seed)
    n_features = len(LTR_FEATURE_NAMES)
    dims = [n_features, *hidden, 1]

    def _activation_module() -> Any:
        if activation == "tanh":
            return torch.nn.Tanh()
        if activation == "sigmoid":
            return torch.nn.Sigmoid()
        return torch.nn.ReLU()

    modules: list[Any] = []
    for i in range(len(dims) - 1):
        modules.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            modules.append(_activation_module())
    model = torch.nn.Sequential(*modules)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def _pairs(group: QueryGroup) -> list[tuple[int, int]]:
        pos = [i for i, label in enumerate(group.labels) if label > 0]
        neg = [i for i, label in enumerate(group.labels) if label == 0]
        pairs: list[tuple[int, int]] = []
        for p in pos:
            for n in neg:
                pairs.append((p, n))
        if len(pairs) > max_pairs_per_query:
            rng = random.Random(seed + hash(group.query_id) % 10_000)
            rng.shuffle(pairs)
            pairs = pairs[:max_pairs_per_query]
        return pairs

    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(list(train_groups))
        model.train()
        total_loss = 0.0
        batches = 0
        for group in train_groups:
            pairs = _pairs(group)
            if not pairs:
                continue
            x = torch.tensor(group.features, dtype=torch.float32)
            scores = model(x).squeeze(-1)
            pos_idx = torch.tensor([p for p, _ in pairs], dtype=torch.long)
            neg_idx = torch.tensor([n for _, n in pairs], dtype=torch.long)
            diff = scores[pos_idx] - scores[neg_idx]
            loss = torch.nn.functional.softplus(-diff).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        if epoch % 20 == 0:
            logger.info("mlp epoch %d loss %.4f", epoch, total_loss / max(batches, 1))

    model.eval()
    layers: list[dict[str, Any]] = []
    with torch.no_grad():
        linear_layers = [m for m in model if isinstance(m, torch.nn.Linear)]
        for layer in linear_layers:
            layers.append({
                "w": layer.weight.detach().cpu().tolist(),
                "b": layer.bias.detach().cpu().tolist(),
            })
    return {"activation": activation, "hidden": list(hidden), "layers": layers}


def mlp_predictor(payload: dict[str, Any]) -> Callable[[list[list[float]]], list[float]]:
    """Reference predictor for a trained MLP payload (numpy-free fallback)."""
    layers = [(layer["w"], layer["b"]) for layer in payload["layers"]]
    activation = str(payload.get("activation") or "relu")

    def _act(values: list[float]) -> list[float]:
        if activation == "tanh":
            return [math.tanh(v) for v in values]
        if activation == "sigmoid":
            return [1.0 / (1.0 + math.exp(-v)) for v in values]
        return [v if v > 0 else 0.0 for v in values]

    def predict(rows: list[list[float]]) -> list[float]:
        out: list[float] = []
        for row in rows:
            current = list(row)
            for index, (weights, bias) in enumerate(layers):
                nxt = []
                for unit in range(len(weights)):
                    total = bias[unit]
                    for j, weight in enumerate(weights[unit]):
                        total += weight * current[j]
                    nxt.append(total)
                current = nxt if index == len(layers) - 1 else _act(nxt)
            out.append(float(current[0]))
        return out

    return predict


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------


def write_manifest(
    path: Path,
    *,
    model_format: str,
    model_file: str,
    metrics: dict[str, Any],
    hyperparams: dict[str, Any],
    dataset: dict[str, Any] | None = None,
) -> None:
    manifest = {
        "manifest_version": LTR_MANIFEST_VERSION,
        "format": model_format,
        "model_file": model_file,
        "feature_names": list(LTR_FEATURE_NAMES),
        "feature_docs": [FEATURE_DOCS[name] for name in LTR_FEATURE_NAMES],
        "hyperparams": hyperparams,
        "metrics": metrics,
        "dataset": dataset or {},
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_K",
    "QueryGroup",
    "TrainLtrRankerError",
    "evaluate_groups",
    "hit_rate_at_k",
    "load_groups",
    "mlp_predictor",
    "ndcg_at_k",
    "reciprocal_rank",
    "save_lightgbm",
    "split_groups",
    "train_lightgbm",
    "train_mlp",
    "write_manifest",
    "FORMAT_LIGHTGBM_TXT",
    "FORMAT_MLP_JSON",
]
