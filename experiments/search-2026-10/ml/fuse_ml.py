"""Pure candidate-list fusion functions for M1–M4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hars_memory.retrieval.fusion import ChannelHit, fuse

Channel = Sequence[tuple[str, float]]


def fuse_prod(
    dense: Channel, sparse: Channel, alpha: float = 0.5, pool: int = 15, top_k: int = 20
) -> list[str]:
    """Use production dense+sparse fusion semantics over the first ``pool`` hits."""
    if pool < 0 or top_k < 0:
        raise ValueError("pool and top_k must be non-negative")
    dense_hits = {
        str(chunk_id): ChannelHit(score=float(score), content="", file_path="")
        for chunk_id, score in dense[:pool]
    }
    sparse_hits = {
        str(chunk_id): ChannelHit(score=float(score), content="", file_path="")
        for chunk_id, score in sparse[:pool]
    }
    return [hit.chunk_id for hit in fuse(dense_hits, sparse_hits, alpha)[:top_k]]


def _weights_for(
    channels: Mapping[str, Channel], weights: Mapping[str, float] | Sequence[float] | None
) -> dict[str, float]:
    names = list(channels)
    if weights is None:
        return {name: 1.0 for name in names}
    if isinstance(weights, Mapping):
        return {name: float(weights.get(name, 0.0)) for name in names}
    if len(weights) != len(names):
        raise ValueError("A weight is required for every channel")
    return {name: float(weight) for name, weight in zip(names, weights)}


def rrf_n(
    channels: Mapping[str, Channel],
    weights: Mapping[str, float] | Sequence[float] | None = None,
    k: int = 60,
    pool: int = 20,
    top_k: int = 20,
) -> list[str]:
    """Weighted reciprocal-rank fusion over any number of channels."""
    if k < 0 or pool < 0 or top_k < 0:
        raise ValueError("k, pool, and top_k must be non-negative")
    channel_weights = _weights_for(channels, weights)
    scores: dict[str, float] = {}
    for name, hits in channels.items():
        weight = channel_weights[name]
        for rank, (chunk_id, _) in enumerate(hits[:pool], start=1):
            chunk_id = str(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]


def wsum_n(
    channels: Mapping[str, Channel],
    weights: Mapping[str, float] | Sequence[float] | None = None,
    pool: int = 20,
    norm: str = "minmax",
    top_k: int = 20,
) -> list[str]:
    """Weighted score fusion with per-channel min-max normalization."""
    if pool < 0 or top_k < 0:
        raise ValueError("pool and top_k must be non-negative")
    if norm != "minmax":
        raise ValueError("norm must be 'minmax'")
    channel_weights = _weights_for(channels, weights)
    scores: dict[str, float] = {}
    for name, hits in channels.items():
        pool_hits = [(str(chunk_id), float(score)) for chunk_id, score in hits[:pool]]
        if not pool_hits:
            continue
        values = [score for _, score in pool_hits]
        lo, hi = min(values), max(values)
        for chunk_id, score in pool_hits:
            normalized = (score - lo) / (hi - lo) if hi > lo else 0.5
            scores[chunk_id] = scores.get(chunk_id, 0.0) + channel_weights[name] * normalized
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]


__all__ = ["fuse_prod", "rrf_n", "wsum_n"]