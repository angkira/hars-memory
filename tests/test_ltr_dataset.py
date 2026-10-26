"""Unit tests for dataset export: query loading, gold matching, graded labels."""

from __future__ import annotations

import json
from pathlib import Path

from hars_memory.eval.ltr_dataset import (
    GRADE_EVIDENCE,
    GRADE_IRRELEVANT,
    GRADE_SAME_DOC,
    QueryRecord,
    build_rows,
    grade_candidate,
    load_query_records,
    matches_gold,
    write_jsonl,
)
from hars_memory.ranking.features import LTR_FEATURE_NAMES
from hars_memory.retrieval.fusion import FusedChunk


class TestGoldMatching:
    def test_flattened_basename_match(self) -> None:
        assert matches_gold(".plans-2026-06-20_plan.md", ".plans-2026-06-20_plan.md")
        assert matches_gold("staging/some/2026-01-01_x.md", "2026-01-01_x.md")

    def test_suffix_match_for_separator_gold(self) -> None:
        assert matches_gold("a/b/c/target.md", "b/c/target.md")
        assert not matches_gold("a/b/c/target.md", "other/target.md")

    def test_non_match(self) -> None:
        assert not matches_gold("x.md", "y.md")


class TestGrading:
    def test_evidence_chunk_is_two(self) -> None:
        label = grade_candidate("g.md", "the answer is 0.0132 here", ["g.md"], "0.0132")
        assert label == GRADE_EVIDENCE

    def test_same_doc_without_span_is_one(self) -> None:
        label = grade_candidate("g.md", "no span here", ["g.md"], "0.0132")
        assert label == GRADE_SAME_DOC

    def test_non_gold_is_zero(self) -> None:
        assert grade_candidate("x.md", "0.0132", ["g.md"], "0.0132") == GRADE_IRRELEVANT

    def test_span_less_query_makes_all_gold_chunks_two(self) -> None:
        assert grade_candidate("g.md", "anything", ["g.md"], None) == GRADE_EVIDENCE


class TestQueryLoading:
    def test_ab_bench_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "q.yaml"
        path.write_text(
            "queries:\n- id: q1\n  type: fact\n  question: What?\n  gold_docs: [a.md]\n",
            encoding="utf-8",
        )
        (record,) = load_query_records(path)
        assert record.query_id == "q1"
        assert record.gold_docs == ("a.md",)
        assert record.answer_span is None

    def test_relational_schema_dict_gold(self, tmp_path: Path) -> None:
        path = tmp_path / "r.yaml"
        path.write_text(
            "questions:\n- id: R1\n  question: Which?\n  gold_docs:\n  - doc: a.md\n    entity: X\n  - doc: b.md\n",
            encoding="utf-8",
        )
        (record,) = load_query_records(path)
        assert record.gold_docs == ("a.md", "b.md")

    def test_synthetic_schema_with_span(self, tmp_path: Path) -> None:
        path = tmp_path / "s.yaml"
        path.write_text(
            'queries:\n- id: s1\n  type: identifier\n  question: Which file?\n'
            '  answer_span: config.yaml\n  gold_docs: ["note.md"]\n',
            encoding="utf-8",
        )
        (record,) = load_query_records(path)
        assert record.answer_span == "config.yaml"

    def test_empty_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "e.yaml"
        path.write_text("queries: []\n", encoding="utf-8")
        try:
            load_query_records(path)
        except ValueError:
            return
        raise AssertionError("expected ValueError")


class TestBuildRows:
    def _pool(self) -> list[FusedChunk]:
        def chunk(cid: str, path: str, content: str, score: float) -> FusedChunk:
            return FusedChunk(
                chunk_id=cid, fused_score=score, dense_score=score, sparse_score=None,
                dense_norm=score, sparse_norm=0.0, content=content, file_path=path,
            )

        return [
            chunk("c1", "gold.md", "the value is 0.0132", 0.9),
            chunk("c2", "gold.md", "other chunk", 0.5),
            chunk("c3", "other.md", "unrelated", 0.3),
        ]

    def test_labels_are_graded(self) -> None:
        record = QueryRecord(
            query_id="q", question="value?", qtype="fact",
            gold_docs=("gold.md",), answer_span="0.0132",
        )
        rows = build_rows(record, self._pool(), None)
        labels = [row["label"] for row in rows]
        assert labels == [GRADE_EVIDENCE, GRADE_SAME_DOC, GRADE_IRRELEVANT]
        assert all(len(row["features"]) == len(LTR_FEATURE_NAMES) for row in rows)

    def test_write_jsonl_has_manifest_header(self, tmp_path: Path) -> None:
        record = QueryRecord(query_id="q", question="x", qtype="fact", gold_docs=("gold.md",))
        rows = build_rows(record, self._pool(), None)
        out = tmp_path / "d.jsonl"
        write_jsonl(rows, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        assert header["feature_names"] == list(LTR_FEATURE_NAMES)
        assert len(lines) == len(rows) + 1
