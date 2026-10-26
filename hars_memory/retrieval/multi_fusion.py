"""Weighted min-max fusion for an arbitrary number of retrieval channels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def wsum_scores(
    channels: list[dict[str, float]], weights: Sequence[float], pool: int
) -> dict[str, float]:
    """Return weighted, per-channel min-max scores for the first ``pool`` hits."""
    if pool < 0:
        raise ValueError("pool must be non-negative")
    if len(channels) != len(weights):
        raise ValueError("A weight is required for every channel")
    numeric_weights = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) for weight in numeric_weights):
        raise ValueError("weights must be finite")

    fused: dict[str, float] = {}
    for channel, weight in zip(channels, numeric_weights):
        candidates = [(str(chunk_id), float(score)) for chunk_id, score in list(channel.items())[:pool]]
        if any(not math.isfinite(score) for _, score in candidates):
            raise ValueError("channel scores must be finite")
        if not candidates:
            continue
        values = [score for _, score in candidates]
        low, high = min(values), max(values)
        for chunk_id, score in candidates:
            normalized = (score - low) / (high - low) if high > low else 0.5
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight * normalized
    return fused


def wsum_n(
    channels: list[dict[str, float]], weights: Sequence[float] | Mapping[str, float], pool: int
) -> list[str]:
    """Rank candidates by weighted min-max score; ties break by chunk ID.

    ``weights`` may be a sequence aligned with ``channels`` or a mapping with
    numeric string indices (``"0"``, ``"1"``, …) for convenience.
    """
    if isinstance(weights, Mapping):
        weights = [float(weights.get(str(index), 0.0)) for index in range(len(channels))]
    scores = wsum_scores(channels, weights, pool)
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))


__all__ = ["wsum_n", "wsum_scores"]