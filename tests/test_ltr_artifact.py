"""Unit tests for the ranker artifact loader (manifest guard) and rerank hook."""

from __future__ import annotations

import json
from pathlib import Path

from hars_memory.ranking.artifact import (
    FORMAT_LINEAR,
    FORMAT_MLP_JSON,
    LTR_MANIFEST_VERSION,
    LtrRankerModel,
)
from hars_memory.ranking.features import LTR_FEATURE_NAMES, LtrFeatureRow
from hars_memory.ranking.rerank import (
    apply_ltr_ranker,
    clear_ranker_cache,
    load_default_ranker,
    ltr_ranker_enabled,
)
from hars_memory.retrieval.fusion import FusedChunk


def _row(item_id: str, value: float = 1.0) -> LtrFeatureRow:
    # first feature == value, everything else 0.0
    features = tuple([value] + [0.0] * (len(LTR_FEATURE_NAMES) - 1))
    return LtrFeatureRow(item_id=item_id, path=item_id, features=features)


def _write_linear_manifest(path: Path, *, feature_names=None, weights=None, bias=0.0) -> None:
    manifest = {
        "manifest_version": LTR_MANIFEST_VERSION,
        "format": FORMAT_LINEAR,
        "feature_names": list(feature_names or LTR_FEATURE_NAMES),
        "weights": weights if weights is not None else [1.0] + [0.0] * (len(LTR_FEATURE_NAMES) - 1),
        "bias": bias,
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


class TestManifestGuard:
    def test_missing_artifact_returns_none(self, tmp_path: Path) -> None:
        assert LtrRankerModel.load(tmp_path / "nope.json") is None
        assert LtrRankerModel.load(None) is None

    def test_feature_list_mismatch_is_refused(self, tmp_path: Path) -> None:
        manifest = tmp_path / "ranker.json"
        _write_linear_manifest(manifest, feature_names=LTR_FEATURE_NAMES[:-1])
        assert LtrRankerModel.load(manifest) is None

    def test_unreadable_manifest_is_refused(self, tmp_path: Path) -> None:
        manifest = tmp_path / "ranker.json"
        manifest.write_text("{not json", encoding="utf-8")
        assert LtrRankerModel.load(manifest) is None

    def test_version_mismatch_is_refused(self, tmp_path: Path) -> None:
        manifest = tmp_path / "ranker.json"
        manifest.write_text(
            json.dumps({"manifest_version": 999, "format": FORMAT_LINEAR, "feature_names": list(LTR_FEATURE_NAMES)}),
            encoding="utf-8",
        )
        assert LtrRankerModel.load(manifest) is None

    def test_unknown_format_is_refused(self, tmp_path: Path) -> None:
        manifest = tmp_path / "ranker.json"
        manifest.write_text(
            json.dumps(
                {"manifest_version": LTR_MANIFEST_VERSION, "format": "nope", "feature_names": list(LTR_FEATURE_NAMES)}
            ),
            encoding="utf-8",
        )
        assert LtrRankerModel.load(manifest) is None


class TestLinearPredictor:
    def test_scores_follow_weights(self, tmp_path: Path) -> None:
        manifest = tmp_path / "ranker.json"
        _write_linear_manifest(manifest, weights=[2.0] + [0.0] * (len(LTR_FEATURE_NAMES) - 1), bias=1.0)
        model = LtrRankerModel.load(manifest)
        assert model is not None
        scores = model.score([_row("a", 3.0), _row("b", 0.0)])
        assert scores == [7.0, 1.0]


class TestMlpPredictor:
    def test_mlp_json_round_trip(self, tmp_path: Path) -> None:
        layers = [
            {"w": [[1.0] * len(LTR_FEATURE_NAMES)], "b": [0.0]},
        ]
        payload = {"activation": "relu", "layers": layers}
        (tmp_path / "mlp.json").write_text(json.dumps(payload), encoding="utf-8")
        manifest = tmp_path / "ranker.json"
        manifest.write_text(
            json.dumps(
                {
                    "manifest_version": LTR_MANIFEST_VERSION,
                    "format": FORMAT_MLP_JSON,
                    "model_file": "mlp.json",
                    "feature_names": list(LTR_FEATURE_NAMES),
                }
            ),
            encoding="utf-8",
        )
        model = LtrRankerModel.load(manifest)
        assert model is not None
        (score,) = model.score([_row("a", 1.0)])
        # 20 features, one 1.0 -> relu(sum) == 1.0
        assert score == 1.0


def _fused(chunk_id: str, fused_score: float, *, dense: float | None = None, dense_norm: float = 0.0) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id,
        fused_score=fused_score,
        dense_score=dense,
        sparse_score=None,
        dense_norm=dense_norm,
        sparse_norm=0.0,
        content=f"content {chunk_id}",
        file_path=f"{chunk_id}.md",
    )


class TestRerankHook:
    def setup_method(self) -> None:
        clear_ranker_cache()

    def test_no_artifact_is_passthrough(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("HARS_MEMORY_LTR_RANKER", str(tmp_path / "missing.json"))
        pool = [_fused("a", 0.9), _fused("b", 0.4)]
        out = apply_ltr_ranker(pool, "q", top_k=2)
        assert [c.chunk_id for c in out] == ["a", "b"]

    def test_env_zero_disables_ranker(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("HARS_MEMORY_LTR_RANKER", "0")
        assert load_default_ranker() is None
        assert ltr_ranker_enabled() is False

    def test_reorders_by_model_score(self, tmp_path: Path, monkeypatch) -> None:
        # weight only the base_rank_reciprocal column -> a model that inverts rank
        idx = LTR_FEATURE_NAMES.index("base_rank_reciprocal")
        weights = [0.0] * len(LTR_FEATURE_NAMES)
        weights[idx] = 1.0
        manifest = tmp_path / "ranker.json"
        _write_linear_manifest(manifest, weights=weights)
        monkeypatch.setenv("HARS_MEMORY_LTR_RANKER", str(manifest))
        pool = [_fused("a", 0.9, dense=0.9, dense_norm=1.0), _fused("b", 0.4, dense=0.4, dense_norm=0.1)]
        # base_rank_reciprocal is 1.0 for a and 0.5 for b, so this should NOT flip.
        out = apply_ltr_ranker(pool, "q", top_k=2)
        assert [c.chunk_id for c in out] == ["a", "b"]

    def test_truncates_to_top_k(self, tmp_path: Path, monkeypatch) -> None:
        manifest = tmp_path / "ranker.json"
        _write_linear_manifest(manifest)
        monkeypatch.setenv("HARS_MEMORY_LTR_RANKER", str(manifest))
        pool = [_fused(str(i), 1.0 - i / 10) for i in range(5)]
        out = apply_ltr_ranker(pool, "q", top_k=2)
        assert len(out) == 2

    def test_return_rows_matches_order(self, tmp_path: Path, monkeypatch) -> None:
        manifest = tmp_path / "ranker.json"
        _write_linear_manifest(manifest)
        monkeypatch.setenv("HARS_MEMORY_LTR_RANKER", str(manifest))
        pool = [_fused("a", 0.9), _fused("b", 0.4)]
        ranked, rows = apply_ltr_ranker(pool, "q", top_k=2, return_rows=True)  # type: ignore[misc]
        assert [c.chunk_id for c in ranked] == [row.item_id for row in rows]

    def test_empty_pool(self, tmp_path: Path, monkeypatch) -> None:
        manifest = tmp_path / "ranker.json"
        _write_linear_manifest(manifest)
        monkeypatch.setenv("HARS_MEMORY_LTR_RANKER", str(manifest))
        assert apply_ltr_ranker([], "q", top_k=5) == []
