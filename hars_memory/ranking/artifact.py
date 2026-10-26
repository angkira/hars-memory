"""Load and score a trained channel-fusion ranker from disk, fail-soft.

Every failure mode -- no artifact, unreadable manifest, feature-list drift, a
backend library that is not installed -- resolves to :meth:`LtrRankerModel.load`
returning ``None``, so the caller keeps its hand-tuned fused ordering. A
learned ranker that cannot load must never be the difference between a working
and a broken search (the same contract code-diver's H-91a ranker uses).

Supported ``format`` values:

* ``lightgbm_txt``  -- a LightGBM ``Booster`` saved as text (needs ``lightgbm``).
* ``mlp_json``      -- a tiny CPU MLP whose weights live in a small JSON
  sidecar; scored with numpy (already a project dependency), so no torch is
  needed at inference time.
* ``linear``        -- weights embedded directly in the manifest (tests / a
  dependency-free fallback).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hars_memory.ranking.features import LTR_FEATURE_NAMES, LtrFeatureRow

logger = logging.getLogger(__name__)

LTR_MANIFEST_VERSION = 1

FORMAT_LINEAR = "linear"
FORMAT_SKLEARN_JOBLIB = "sklearn_joblib"
FORMAT_LIGHTGBM_TXT = "lightgbm_txt"
FORMAT_MLP_JSON = "mlp_json"


@dataclass(slots=True)
class LtrRankerModel:
    """A trained ranker loaded from disk."""

    feature_names: tuple[str, ...]
    predict: Callable[[Sequence[Sequence[float]]], list[float]]
    manifest: dict[str, Any]

    @classmethod
    def load(cls, path: Path | str | None) -> LtrRankerModel | None:
        if not path:
            return None
        manifest_path = Path(path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.debug("no LTR model artifact at %s, keeping hand-tuned ranking", manifest_path)
            return None
        except Exception:
            logger.warning("unreadable LTR model artifact at %s, keeping hand-tuned ranking", manifest_path)
            return None
        return cls._from_manifest(manifest, manifest_path)

    @classmethod
    def _from_manifest(cls, manifest: object, manifest_path: Path) -> LtrRankerModel | None:
        if not isinstance(manifest, dict):
            logger.warning("LTR manifest %s is not a mapping, keeping hand-tuned ranking", manifest_path)
            return None
        version = manifest.get("manifest_version")
        if version is not None and int(version) != LTR_MANIFEST_VERSION:
            logger.warning(
                "LTR manifest %s has version %s, expected %s, keeping hand-tuned ranking",
                manifest_path, version, LTR_MANIFEST_VERSION,
            )
            return None
        feature_names = tuple(str(name) for name in manifest.get("feature_names") or ())
        if feature_names != LTR_FEATURE_NAMES:
            logger.warning(
                "LTR artifact %s was trained on a different feature list "
                "(%d names vs this checkout's %d), keeping hand-tuned ranking",
                manifest_path, len(feature_names), len(LTR_FEATURE_NAMES),
            )
            return None
        model_format = str(manifest.get("format") or "")
        try:
            predict = cls._predictor(model_format, manifest, manifest_path)
        except Exception:
            logger.warning("failed to load LTR model %s, keeping hand-tuned ranking", manifest_path, exc_info=True)
            return None
        if predict is None:
            return None
        return cls(feature_names=feature_names, predict=predict, manifest=manifest)

    @classmethod
    def _predictor(
        cls,
        model_format: str,
        manifest: dict[str, Any],
        manifest_path: Path,
    ) -> Callable[[Sequence[Sequence[float]]], list[float]] | None:
        if model_format == FORMAT_LINEAR:
            weights = [float(value) for value in manifest.get("weights") or ()]
            bias = float(manifest.get("bias") or 0.0)
            if len(weights) != len(LTR_FEATURE_NAMES):
                logger.warning("LTR linear artifact %s has a mismatched weight count", manifest_path)
                return None

            def predict_linear(rows: Sequence[Sequence[float]]) -> list[float]:
                return [bias + sum(w * f for w, f in zip(weights, row, strict=True)) for row in rows]

            return predict_linear

        model_file = manifest.get("model_file")
        if not model_file:
            logger.warning("LTR artifact %s declares format %r but no model_file", manifest_path, model_format)
            return None
        model_path = manifest_path.parent / str(model_file)

        if model_format == FORMAT_SKLEARN_JOBLIB:
            import joblib

            estimator = joblib.load(model_path)

            def predict_sklearn(rows: Sequence[Sequence[float]]) -> list[float]:
                return [float(value) for value in estimator.predict([list(row) for row in rows])]

            return predict_sklearn

        if model_format == FORMAT_LIGHTGBM_TXT:
            import lightgbm
            import numpy as np

            booster = lightgbm.Booster(model_file=str(model_path))

            def predict_lightgbm(rows: Sequence[Sequence[float]]) -> list[float]:
                # LightGBM 4.7's Dataset/Booster path rejects plain
                # list-of-lists ("Data list can only be of ndarray or
                # Sequence"); a float64 array is always accepted.
                matrix = np.asarray([list(row) for row in rows], dtype=np.float64)
                return [float(value) for value in booster.predict(matrix)]

            return predict_lightgbm

        if model_format == FORMAT_MLP_JSON:
            return _load_mlp(model_path)

        logger.warning("unknown LTR model format %r in %s", model_format, manifest_path)
        return None

    def score(self, rows: Sequence[LtrFeatureRow]) -> list[float] | None:
        """Model score per row, or ``None`` when scoring fails (caller keeps
        its own order)."""
        if not rows:
            return []
        try:
            scores = self.predict([row.features for row in rows])
        except Exception:
            logger.warning("LTR model scoring failed, keeping hand-tuned ranking", exc_info=True)
            return None
        if len(scores) != len(rows):
            logger.warning("LTR model returned %d scores for %d rows", len(scores), len(rows))
            return None
        return [float(score) for score in scores]


def _load_mlp(model_path: Path) -> Callable[[Sequence[Sequence[float]]], list[float]] | None:
    """Load a tiny MLP stored as ``{"activation", "layers": [{"w": [[...]], "b": [...]}, ...]}``."""
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("unreadable LTR MLP artifact %s", model_path)
        return None
    layers = payload.get("layers") if isinstance(payload, dict) else None
    if not isinstance(layers, list) or not layers:
        logger.warning("LTR MLP artifact %s has no layers", model_path)
        return None
    activation = str(payload.get("activation") or "relu")
    parsed: list[tuple[list[list[float]], list[float]]] = []
    for layer in layers:
        weights = [[float(v) for v in row] for row in layer.get("w") or []]
        bias = [float(v) for v in layer.get("b") or []]
        if not weights or len(bias) != len(weights):
            logger.warning("LTR MLP artifact %s has a malformed layer", model_path)
            return None
        parsed.append((weights, bias))

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency of this project
        np = None

    def _activate(values: list[float]) -> list[float]:
        if activation == "tanh":
            return [math.tanh(v) for v in values]
        if activation == "sigmoid":
            return [1.0 / (1.0 + math.exp(-v)) for v in values]
        return [v if v > 0.0 else 0.0 for v in values]

    def predict_mlp_py(rows: Sequence[Sequence[float]]) -> list[float]:
        out: list[float] = []
        for row in rows:
            current = [float(v) for v in row]
            for layer_index, (weights, bias) in enumerate(parsed):
                next_values = []
                for unit in range(len(weights)):
                    total = bias[unit]
                    weights_row = weights[unit]
                    for j, weight in enumerate(weights_row):
                        total += weight * current[j]
                    next_values.append(total)
                current = next_values if layer_index == len(parsed) - 1 else _activate(next_values)
            out.append(float(current[0]))
        return out

    if np is None:
        return predict_mlp_py

    np_layers = [(np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64)) for w, b in parsed]

    def predict_mlp_np(rows: Sequence[Sequence[float]]) -> list[float]:
        matrix = np.asarray(rows, dtype=np.float64)
        current = matrix
        for layer_index, (weights, bias) in enumerate(np_layers):
            current = current @ weights.T + bias
            if layer_index != len(np_layers) - 1:
                if activation == "tanh":
                    current = np.tanh(current)
                elif activation == "sigmoid":
                    current = 1.0 / (1.0 + np.exp(-current))
                else:
                    current = np.maximum(current, 0.0)
        return [float(v) for v in current[:, 0]]

    return predict_mlp_np


__all__ = [
    "FORMAT_LIGHTGBM_TXT",
    "FORMAT_LINEAR",
    "FORMAT_MLP_JSON",
    "FORMAT_SKLEARN_JOBLIB",
    "LTR_MANIFEST_VERSION",
    "LtrRankerModel",
]
