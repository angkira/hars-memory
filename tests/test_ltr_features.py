"""Unit tests for the learned-ranker feature extractor.

These fix the artifact CONTRACT (the feature tuple), the documented
"missing value" encodings, and the corpus-context lookups — the parts a
silent regression would be hardest to notice from the end metrics.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from hars_memory.ranking.features import (
    FEATURE_DOCS,
    LTR_FEATURE_NAMES,
    LtrCorpusContext,
    QueryFeatures,
    extract_pool_features,
    feature_row_from_columns,
    load_cached_context,
    zero_graph_columns,
)
from hars_memory.retrieval.fusion import ChannelHit, FusedChunk, fuse


def _chunk(
    chunk_id: str,
    *,
    fused_score: float,
    dense_score: float | None = None,
    sparse_score: float | None = None,
    dense_norm: float = 0.0,
    sparse_norm: float = 0.0,
    content: str = "some text",
    file_path: str = "a.md",
    ripgrep_score: float | None = None,
    flat_dense_score: float | None = None,
) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id,
        fused_score=fused_score,
        dense_score=dense_score,
        sparse_score=sparse_score,
        dense_norm=dense_norm,
        sparse_norm=sparse_norm,
        content=content,
        file_path=file_path,
        ripgrep_score=ripgrep_score,
        flat_dense_score=flat_dense_score,
    )


class TestFeatureContract:
    def test_names_unique_and_documented(self) -> None:
        assert len(LTR_FEATURE_NAMES) == len(set(LTR_FEATURE_NAMES))
        assert set(FEATURE_DOCS) == set(LTR_FEATURE_NAMES)
        assert len(LTR_FEATURE_NAMES) >= 15

    def test_row_has_one_value_per_name(self) -> None:
        pool = [_chunk("c1", fused_score=0.9, dense_score=0.8, dense_norm=1.0)]
        (row,) = extract_pool_features(pool, "what is X")
        assert len(row.features) == len(LTR_FEATURE_NAMES)
        assert row.rank == 1

    def test_feature_row_from_columns_orders_by_manifest(self) -> None:
        columns = {name: float(i) for i, name in enumerate(LTR_FEATURE_NAMES)}
        row = feature_row_from_columns(columns, item_id="x")
        assert row.features[0] == 0.0
        assert row.features[-1] == float(len(LTR_FEATURE_NAMES) - 1)


class TestPoolFeatures:
    def test_pool_relative_values(self) -> None:
        top = _chunk("top", fused_score=0.9, dense_score=0.9, dense_norm=1.0)
        second = _chunk("second", fused_score=0.4, dense_score=0.4, dense_norm=0.2)
        rows = extract_pool_features([top, second], "query")

        top_row, second_row = rows
        name_to_idx = {name: i for i, name in enumerate(LTR_FEATURE_NAMES)}
        # gap-to-top: 0 for the top candidate, positive for the rest.
        assert top_row.features[name_to_idx["fused_gap_to_top"]] == 0.0
        assert second_row.features[name_to_idx["fused_gap_to_top"]] == 0.9 - 0.4
        # reciprocal rank
        assert top_row.features[name_to_idx["base_rank_reciprocal"]] == 1.0
        assert second_row.features[name_to_idx["base_rank_reciprocal"]] == 0.5
        # channel_count: top has dense only
        assert top_row.features[name_to_idx["channel_count"]] == 1.0

    def test_ripgrep_norm_matches_min_max_convention(self) -> None:
        a = _chunk("a", fused_score=0.5, ripgrep_score=2.0)
        b = _chunk("b", fused_score=0.4, ripgrep_score=6.0)
        rows = extract_pool_features([a, b], "q")
        idx = LTR_FEATURE_NAMES.index("ripgrep_norm")
        assert rows[0].features[idx] == 0.0
        assert rows[1].features[idx] == 1.0

    def test_single_matched_candidate_norm_is_neutral_half(self) -> None:
        only = _chunk("only", fused_score=0.5, ripgrep_score=3.0)
        (row,) = extract_pool_features([only], "q")
        assert row.features[LTR_FEATURE_NAMES.index("ripgrep_norm")] == 0.5

    def test_exact_identifier_match(self) -> None:
        hit = _chunk("c", fused_score=0.5, content="the Q7X42 run failed")
        miss = _chunk("d", fused_score=0.4, content="unrelated prose")
        rows = extract_pool_features([hit, miss], "how did Q7X42 fail?")
        idx = LTR_FEATURE_NAMES.index("exact_identifier_match")
        assert rows[0].features[idx] == 1.0
        assert rows[1].features[idx] == 0.0

    def test_missing_values_are_encoded_not_zero_faked(self) -> None:
        # No context -> doc-age must be the -1 sentinel (undated), not 0.0.
        (row,) = extract_pool_features([_chunk("c", fused_score=0.5)], "q")
        assert row.features[LTR_FEATURE_NAMES.index("doc_age_days_log")] == -1.0
        assert row.features[LTR_FEATURE_NAMES.index("doc_len_log")] == 0.0
        assert row.features[LTR_FEATURE_NAMES.index("graph_entity_fanin_log")] == 0.0

    def test_zero_graph_columns_ablation(self) -> None:
        rows = extract_pool_features([_chunk("c", fused_score=0.5)], "q")
        # build a context so graph columns could be non-zero
        base = rows[0]
        bumped = base.__class__(
            item_id=base.item_id, path=base.path, rank=base.rank,
            features=tuple(
                5.0 if name in {"graph_entity_fanin_log", "graph_entity_match"} else v
                for name, v in zip(LTR_FEATURE_NAMES, base.features, strict=True)
            ),
        )
        (zeroed,) = zero_graph_columns([bumped])
        assert zeroed.features[LTR_FEATURE_NAMES.index("graph_entity_fanin_log")] == 0.0
        assert zeroed.features[LTR_FEATURE_NAMES.index("graph_entity_match")] == 0.0
        # non-graph columns preserved
        assert zeroed.features[LTR_FEATURE_NAMES.index("fused_score")] == base.features[
            LTR_FEATURE_NAMES.index("fused_score")
        ]


class TestQueryFeatures:
    def test_content_terms_exclude_identifiers_and_stopwords(self) -> None:
        q = QueryFeatures.from_question("What configuration file is the Q7X42 resume plan?")
        assert "Q7X42" in q.identifier_terms
        assert "what" not in q.content_terms
        assert all(term.lower() not in {"q7x42"} for term in q.content_terms)


class TestCorpusContext:
    def _write_index(self, tmp_path: Path) -> Path:
        index = tmp_path / "index"
        index.mkdir()
        chunks = {
            "chunk-0001": {
                "file_path": "docs-2026-07-15_alpha.md",
                "content": "[Document: x | Section: docs | Date: 2026-07-15]\n\n# Title\nbody",
                "chunk_order_index": 0,
                "heading_path": "# Title",
            },
            "chunk-0002": {
                "file_path": "docs-2026-07-15_alpha.md",
                "content": "second chunk " * 20,
                "chunk_order_index": 1,
            },
        }
        entities = {
            "Q7X42": {"chunk_ids": ["chunk-0001"], "count": 7},
            "val/thing": {"chunk_ids": ["chunk-0002"], "count": 2},
        }
        (index / "kv_store_text_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
        (index / "kv_store_entity_chunks.json").write_text(json.dumps(entities), encoding="utf-8")
        return index

    def test_load_and_lookups(self, tmp_path: Path) -> None:
        index = self._write_index(tmp_path)
        ctx = LtrCorpusContext.load(index)
        assert ctx is not None
        assert ctx.doc_length("docs-2026-07-15_alpha.md") > 0
        assert ctx.doc_age_days("docs-2026-07-15_alpha.md") == (date(2026, 10, 17) - date(2026, 7, 15)).days
        assert ctx.doc_age_days("missing.md") is None
        assert ctx.chunk_heading_depth("chunk-0001", "irrelevant") == 1
        assert ctx.entity_fanin_max("chunk-0001") == 7
        assert ctx.entity_match_fraction("chunk-0001", ["Q7X42"]) == 1.0
        assert ctx.entity_match_fraction("chunk-0002", ["Q7X42"]) == 0.0

    def test_features_with_context_populate_doc_and_graph_columns(self, tmp_path: Path) -> None:
        index = self._write_index(tmp_path)
        ctx = LtrCorpusContext.load(index)
        chunk = _chunk("chunk-0001", fused_score=0.5, file_path="docs-2026-07-15_alpha.md")
        (row,) = extract_pool_features([chunk], "what about Q7X42", context=ctx)
        assert row.features[LTR_FEATURE_NAMES.index("doc_age_days_log")] > 0.0
        assert row.features[LTR_FEATURE_NAMES.index("graph_entity_fanin_log")] > 0.0
        assert row.features[LTR_FEATURE_NAMES.index("graph_entity_match")] == 1.0

    def test_cached_context_none_for_missing_store(self, tmp_path: Path) -> None:
        assert load_cached_context(tmp_path / "nope") is None
        assert LtrCorpusContext.load(tmp_path / "nope") is None


class TestFuseIntegration:
    """The extractor must work on the REAL fuse() output, not a hand fixture."""

    def test_extract_over_real_fuse_pool(self) -> None:
        dense = {
            "c1": ChannelHit(score=0.9, content="alpha Q7X42", file_path="a.md"),
            "c2": ChannelHit(score=0.5, content="beta", file_path="b.md"),
        }
        sparse = {
            "c2": ChannelHit(score=4.0, content="beta", file_path="b.md"),
            "c3": ChannelHit(score=2.0, content="gamma", file_path="c.md"),
        }
        pool = fuse(dense, sparse, 0.5)
        assert pool
        rows = extract_pool_features(pool, "Q7X42 beta")
        assert len(rows) == len(pool)
        for row in rows:
            assert len(row.features) == len(LTR_FEATURE_NAMES)
            assert all(isinstance(v, float) for v in row.features)
