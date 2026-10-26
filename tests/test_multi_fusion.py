from __future__ import annotations

import pytest

from hars_memory.retrieval.multi_fusion import wsum_n, wsum_scores


def test_weighted_minmax_fusion_ranks_and_breaks_ties_by_id() -> None:
    channels = [
        {"c": 3.0, "b": 2.0, "a": 1.0},
        {"a": 4.0, "b": 4.0},
    ]
    assert wsum_n(channels, [0.5, 0.5], pool=3) == ["b", "c", "a"]
    scores = wsum_scores(channels, [0.5, 0.5], pool=3)
    assert scores["a"] == pytest.approx(0.25)
    assert scores["b"] == pytest.approx(0.5)
    assert scores["c"] == pytest.approx(0.5)


def test_pool_limits_each_channel_and_constant_score_is_neutral() -> None:
    channels = [{"z": 2.0, "x": 2.0, "ignored": 99.0}, {}]
    assert wsum_n(channels, [1.0, 0.0], pool=2) == ["x", "z"]
    assert wsum_scores(channels, [1.0, 0.0], pool=2) == {"z": 0.5, "x": 0.5}


@pytest.mark.parametrize(
    ("channels", "weights", "pool"),
    [([{}], [], 1), ([{}], [1.0], -1)],
)
def test_invalid_fusion_inputs_raise(
    channels: list[dict[str, float]], weights: list[float], pool: int
) -> None:
    with pytest.raises(ValueError):
        wsum_n(channels, weights, pool)