"""Unit tests for hars_memory/eval/relational_eval.py.

The functions under test are pure (scoring, name/doc matching, gold loading) or
fully offline (trace parsing from a synthetic envelope). Nothing here touches
the live service or the live index — the live run is exercised by the module's
`run`/`validate` subcommands, which are read-only by construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hars_memory.eval.relational_eval import (
    RelationalQuestion,
    aggregate,
    canonicalize_docs,
    doc_recall,
    entity_scores,
    load_gold,
    match_doc,
    normalize_name,
    relation_scores,
    score_trace,
)


class TestNormalizeName:
    def test_casefolds_and_collapses_whitespace(self) -> None:
        # The live graph stores both `A2S124A` and `A2S124a` for one entity.
        assert normalize_name("A2S124A") == normalize_name("A2S124a")

    def test_strips_and_collapses(self) -> None:
        assert normalize_name("  shortcut   hypothesis ") == "shortcut hypothesis"

    def test_none_is_empty(self) -> None:
        assert normalize_name(None) == ""


class TestMatchDoc:
    def test_exact_case_insensitive(self) -> None:
        assert match_doc("Baseline8K.md", "baseline8k.md")

    def test_namespaced_prefix_suffix_match(self) -> None:
        # Service emits `hypotheses-hypotheses-0123.md`; graph records the bare name.
        assert match_doc("hypotheses-hypotheses-0123.md", "hypotheses-0123.md")

    def test_unrelated_docs_do_not_match(self) -> None:
        assert not match_doc("2026-09-27_full_research_report.md", "2026-09-12_baseline8k-failure.md")


class TestCanonicalizeDocs:
    def test_maps_served_to_gold_spelling(self) -> None:
        served = ["hypotheses-hypotheses-0123.md", "unrelated.md"]
        gold = ["hypotheses-0123.md", "docB.md"]
        assert canonicalize_docs(served, gold) == ["hypotheses-0123.md", "unrelated.md"]


class TestEntityScores:
    def test_full_recall(self) -> None:
        s = entity_scores(["A", "B"], ["a", "b", "c"])
        assert s["recall"] == 1.0 and s["hit_any"] == 1.0 and s["missing"] == []

    def test_partial_recall_lists_missing_bridge(self) -> None:
        s = entity_scores(["A", "Bridge", "C"], ["A", "C"])
        assert s["recall"] == pytest.approx(2 / 3)
        assert s["missing"] == ["Bridge"]
        assert s["hit_any"] == 1.0

    def test_zero_recall(self) -> None:
        s = entity_scores(["A"], ["Z"])
        assert s["recall"] == 0.0 and s["hit_any"] == 0.0


class TestRelationScores:
    def test_unordered_pair_match(self) -> None:
        exp = [frozenset({"a", "m"}), frozenset({"m", "c"})]
        # traced pairs are unordered: (m, a) counts as (a, m)
        found = [frozenset({"m", "a"}), frozenset({"m", "c"})]
        assert relation_scores(exp, found)["recall"] == 1.0

    def test_missing_edge_reported(self) -> None:
        exp = [frozenset({"a", "m"}), frozenset({"m", "c"})]
        found = [frozenset({"a", "m"})]
        s = relation_scores(exp, found)
        assert s["recall"] == 0.5
        assert s["missing"] == [["c", "m"]]


class TestDocRecall:
    def test_recall_and_mrr(self) -> None:
        r = doc_recall(["x.md", "gold.md", "y.md"], ["gold.md"])
        assert r["recall"] == 1.0
        assert r["first_rank"] == 2
        assert r["mrr"] == pytest.approx(0.5)

    def test_respects_cutoff(self) -> None:
        r = doc_recall(["x.md", "gold.md"], ["gold.md"], k=1)
        assert r["recall"] == 0.0 and r["first_rank"] == 2

    def test_empty_gold_rejected(self) -> None:
        with pytest.raises(ValueError):
            doc_recall(["x.md"], [])


def _make_question() -> RelationalQuestion:
    return load_gold(_write_gold())[0]


def _write_gold() -> Path:
    doc = {
        "questions": [
            {
                "id": "T01",
                "question": "How is A connected to B?",
                "ll_keywords": ["A", "B"],
                "expected_entities": [
                    {"name": "A", "type": "hypothesis"},
                    {"name": "M", "type": "metric"},
                    {"name": "B", "type": "experiment"},
                ],
                "expected_relations": [
                    {"source": "A", "target": "M", "relation": ["evaluated_on"]},
                    {"source": "M", "target": "B", "relation": ["produces"]},
                ],
                "gold_docs": [{"doc": "docA.md"}, {"doc": "docB.md"}],
                "rationale": "no single doc mentions both",
            }
        ]
    }
    path = Path("/tmp/opencode/_test_gold.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


class TestLoadGold:
    def test_round_trip(self) -> None:
        q = _make_question()
        assert q.id == "T01"
        assert q.entity_names == ["A", "M", "B"]
        assert q.doc_names == ["docA.md", "docB.md"]
        assert q.ll_keywords == ("A", "B")
        assert [sorted(p) for p in q.relation_pairs] == [["a", "m"], ["b", "m"]]


def _envelope() -> dict:
    return {
        "ok": True,
        "events": [
            {"event": "trace_start", "data": {"lightrag_mode": "mix"}},
            {"event": "bm25_hits", "data": {"latency_ms": 1.0}},
            {"event": "vector_hits", "data": {"latency_ms": 2.0}},
            {"event": "rerank", "data": {"latency_ms": None}},
            {
                "event": "graph_expand",
                "data": {
                    "entity_count": 1,
                    "relation_count": 2,
                    "latency_ms": 10.0,
                    "entities": [{"name": "A", "type": "hypothesis"}],
                    "relations": [
                        {"entity1": "A", "entity2": "M"},
                        {"entity1": "M", "entity2": "B"},
                    ],
                },
            },
            {
                "event": "context",
                "data": {
                    "chars": 1234,
                    "source_count": 2,
                    "entity_count": 1,
                    "relation_count": 2,
                    "sources": [{"doc": "docA.md"}, {"doc": "docB.md"}],
                },
            },
            {
                "event": "result",
                "data": {"hybrid": {"fused_chunks": [{"file_path": "docA.md"}, {"file_path": "other.md"}]}},
            },
            {"event": "done", "data": {"total_ms": 42.0}},
        ],
    }


class TestScoreTrace:
    def test_both_branches_scored(self) -> None:
        row = score_trace(_make_question(), _envelope())
        # entities surfaced = entity list + relation endpoints = {A, M, B}
        assert row["graph"]["entity_recall"] == 1.0
        # only A was in the direct entity list; M and B only appear as relation endpoints
        assert row["graph"]["entity_recall_direct"] == pytest.approx(1 / 3)
        assert row["graph"]["relation_recall"] == 1.0
        assert row["graph"]["doc_recall"] == 1.0
        assert row["fusion_only"]["doc_recall"] == 0.5
        # graph-only docs (context - fusion) = {docB.md}: half the gold set
        assert row["graph_delta"]["doc_recall"] == 0.5
        assert row["latency_ms"]["total"] == 42.0
        assert row["latency_ms"]["graph_expand"] == 10.0


class TestAggregate:
    def test_counts_and_means(self) -> None:
        rows = [score_trace(_make_question(), _envelope())]
        agg = aggregate(rows)
        assert agg["n_questions"] == 1
        assert agg["graph_relation_recall"] == 1.0
        assert agg["fusion_doc_recall"] == 0.5
        assert agg["graph_delta_doc_recall"] == 0.5
        assert agg["graph_adds_docs"] == 1
        assert agg["graph_doc_solved"] == 1
        assert agg["fusion_doc_solved"] == 0

    def test_empty(self) -> None:
        assert aggregate([]) == {}
