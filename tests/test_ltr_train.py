"""Unit tests for the ranker training library: dataset loading, query-level
split, metrics, and an end-to-end LightGBM fit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hars_memory.ranking.features import LTR_FEATURE_NAMES
from hars_memory.ranking.train import (
    QueryGroup,
    TrainLtrRankerError,
    evaluate_groups,
    hit_rate_at_k,
    load_groups,
    mlp_predictor,
    ndcg_at_k,
    reciprocal_rank,
    split_groups,
    train_mlp,
    write_manifest,
)


def _row(query_id: str, features: list[float], label: int) -> dict:
    return {"query_id": query_id, "features": features, "label": label}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"feature_names": list(LTR_FEATURE_NAMES)}) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class TestLoadGroups:
    def test_groups_by_query(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        _write_jsonl(path, [
            _row("q1", [0.0] * len(LTR_FEATURE_NAMES), 1),
            _row("q1", [1.0] * len(LTR_FEATURE_NAMES), 0),
            _row("q2", [0.5] * len(LTR_FEATURE_NAMES), 0),
        ])
        groups, names = load_groups(path)
        assert names == LTR_FEATURE_NAMES
        assert [g.query_id for g in groups] == ["q1", "q2"]
        assert len(groups[0].features) == 2

    def test_feature_mismatch_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text(
            json.dumps({"feature_names": ["only_one"]}) + "\n"
            + json.dumps({"query_id": "q", "features": [1.0], "label": 0}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(TrainLtrRankerError):
            load_groups(path)

    def test_empty_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text(json.dumps({"feature_names": list(LTR_FEATURE_NAMES)}) + "\n", encoding="utf-8")
        with pytest.raises(TrainLtrRankerError):
            load_groups(path)


class TestSplit:
    def test_query_level_split_no_leak(self) -> None:
        groups = [QueryGroup(query_id=f"q{i}", features=[[0.0]], labels=[0]) for i in range(10)]
        train, valid = split_groups(groups, valid_fraction=0.3, seed=0)
        train_ids = {g.query_id for g in train}
        valid_ids = {g.query_id for g in valid}
        assert train_ids.isdisjoint(valid_ids)
        assert train_ids | valid_ids == {f"q{i}" for i in range(10)}

    def test_reproducible(self) -> None:
        groups = [QueryGroup(query_id=f"q{i}", features=[[0.0]], labels=[0]) for i in range(10)]
        a = split_groups(groups, valid_fraction=0.3, seed=7)
        b = split_groups(groups, valid_fraction=0.3, seed=7)
        assert [g.query_id for g in a[1]] == [g.query_id for g in b[1]]


class TestMetrics:
    def test_ndcg_rewards_early_relevant(self) -> None:
        assert ndcg_at_k([1, 0, 0], 10) > ndcg_at_k([0, 0, 1], 10)

    def test_hit_rate_and_rr(self) -> None:
        assert hit_rate_at_k([0, 1, 0], 2) == 1.0
        assert hit_rate_at_k([0, 0, 1], 2) == 0.0
        assert reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)

    def test_evaluate_groups_reports_base(self) -> None:
        # features [[1.0],[0.0]], labels [1,0]: scorer keys on feature 0, so the
        # model keeps the given order and reproduces the base metrics.
        groups = [QueryGroup(query_id="q", features=[[1.0], [0.0]], labels=[1, 0])]
        metrics = evaluate_groups(groups, lambda rows: [float(row[0]) for row in rows], 10)
        assert metrics["queries"] == 1
        assert metrics["ndcg@10"] == 1.0
        assert metrics["base_ndcg@10"] == 1.0


class TestMlp:
    def test_mlp_trains_and_exports(self) -> None:
        pytest.importorskip("torch")
        groups = []
        for i in range(6):
            high = [0.0] * len(LTR_FEATURE_NAMES)
            high[0] = 1.0
            low = [0.0] * len(LTR_FEATURE_NAMES)
            low[0] = 0.0
            groups.append(QueryGroup(query_id=f"q{i}", features=[high, low], labels=[1, 0]))
        payload = train_mlp(groups, groups[:2], seed=0, epochs=30, hidden=(4,))
        assert payload["layers"]
        predictor = mlp_predictor(payload)
        scores = predictor([[1.0] + [0.0] * (len(LTR_FEATURE_NAMES) - 1), [0.0] * len(LTR_FEATURE_NAMES)])
        assert scores[0] > scores[1]


class TestLightGbm:
    def test_end_to_end_fit_and_manifest(self, tmp_path: Path) -> None:
        pytest.importorskip("lightgbm")
        from hars_memory.ranking.train import lightgbm_predictor, train_lightgbm

        groups = []
        for i in range(12):
            strong = [0.0] * len(LTR_FEATURE_NAMES)
            strong[0] = 1.0
            weak = [0.0] * len(LTR_FEATURE_NAMES)
            weak[0] = 0.1
            groups.append(QueryGroup(query_id=f"q{i}", features=[strong, weak], labels=[2, 0]))
        # No eval_set here: with a handful of queries early stopping can halt at
        # iteration 0 and return a constant model, which would make the
        # ordering assertion meaningless.
        ranker = train_lightgbm(groups, [], seed=0, num_boost_round=30)
        scores = lightgbm_predictor(ranker)(groups[0].features)
        assert len(scores) == 2
        assert scores[0] >= scores[1]

        manifest_path = tmp_path / "ranker.json"
        ranker.booster_.save_model(str(tmp_path / "ranker.lgb.txt"))
        write_manifest(
            manifest_path,
            model_format="lightgbm_txt",
            model_file="ranker.lgb.txt",
            metrics={"lightgbm": {"valid": {"ndcg@10": 1.0}}},
            hyperparams={"winner": "lightgbm"},
        )
        from hars_memory.ranking.artifact import LtrRankerModel

        model = LtrRankerModel.load(manifest_path)
        assert model is not None
        scores = model.score([])
        assert scores == []
