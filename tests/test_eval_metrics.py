"""Hand-computed regression tests for tools/memory/eval/metrics.py.

Per metrics.py's own module docstring: "A bug in this module invalidates
every future retrieval decision made from ab_bench.py's output silently."
Every test below asserts a value computed by hand (shown in a comment),
not merely "does not crash" or "returns something plausible" — a buggy
metric that always returns a plausible-looking number is exactly the
failure mode this suite exists to catch.
"""

from __future__ import annotations

import math

import pytest

from hars_memory.eval.metrics import (
    chunk_coverage,
    dedupe_preserve_order,
    file_recall_at_k,
    first_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    no_answer_hit_rate,
    recall_at_k,
    reciprocal_rank,
    supersession_error_rate,
    supersession_violated,
)


class TestDedupePreserveOrder:
    def test_removes_repeats_keeping_first_occurrence_rank(self) -> None:
        assert dedupe_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_empty_list(self) -> None:
        assert dedupe_preserve_order([]) == []

    def test_no_duplicates_is_unchanged(self) -> None:
        assert dedupe_preserve_order(["x", "y", "z"]) == ["x", "y", "z"]


class TestRecallAtK:
    def test_single_gold_doc_hit_within_k(self) -> None:
        # gold doc "b" is at rank 2 <= k=3 -> full recall (1.0)
        assert recall_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0

    def test_single_gold_doc_miss_outside_k(self) -> None:
        # gold doc "b" is at rank 2, but k=1 only looks at ["a"] -> 0.0
        assert recall_at_k(["a", "b", "c"], {"b"}, k=1) == 0.0

    def test_multi_gold_partial_hit_is_fraction_found(self) -> None:
        # gold = {b, d}; top-3 = {a, b, c} -> only "b" found -> 1/2 = 0.5
        assert recall_at_k(["a", "b", "c", "d"], {"b", "d"}, k=3) == pytest.approx(0.5)

    def test_multi_gold_full_hit(self) -> None:
        # gold = {a, c}; top-3 = {a, b, c} -> both found -> 2/2 = 1.0
        assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0

    def test_empty_ranked_list_is_zero(self) -> None:
        assert recall_at_k([], {"a"}, k=5) == 0.0

    def test_empty_gold_docs_raises(self) -> None:
        with pytest.raises(ValueError, match="gold_docs"):
            recall_at_k(["a"], set(), k=1)

    def test_k_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            recall_at_k(["a"], {"a"}, k=0)


class TestNdcgAtK:
    def test_single_relevant_doc_at_rank_1_is_perfect_score(self) -> None:
        # DCG = 1/log2(2) = 1.0; IDCG (1 relevant doc, ideal at rank 1) = 1.0
        # NDCG = 1.0 / 1.0 = 1.0
        assert ndcg_at_k(["a", "b", "c"], {"a"}, k=3) == pytest.approx(1.0)

    def test_single_relevant_doc_at_rank_2_is_discounted(self) -> None:
        # DCG = 1/log2(3); IDCG = 1/log2(2) = 1.0
        expected = (1.0 / math.log2(3)) / 1.0
        assert ndcg_at_k(["x", "a", "b"], {"a"}, k=3) == pytest.approx(expected)
        assert 0.0 < ndcg_at_k(["x", "a", "b"], {"a"}, k=3) < 1.0

    def test_relevant_doc_missing_entirely_is_zero(self) -> None:
        assert ndcg_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_two_relevant_docs_ideal_order_is_perfect(self) -> None:
        # gold={a,b}; ranked=[a,b,c] places both relevant docs at the top,
        # matching the ideal ranking exactly -> NDCG = 1.0
        assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3) == pytest.approx(1.0)

    def test_two_relevant_docs_worse_order_is_less_than_one(self) -> None:
        # gold={a,b}; ranked=[c,a,b] -> DCG = 1/log2(3) + 1/log2(4)
        # IDCG (ideal: both relevant at ranks 1,2) = 1/log2(2) + 1/log2(3) = 1 + 1/log2(3)
        dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert ndcg_at_k(["c", "a", "b"], {"a", "b"}, k=3) == pytest.approx(dcg / idcg)
        assert ndcg_at_k(["c", "a", "b"], {"a", "b"}, k=3) < 1.0

    def test_ideal_hits_capped_at_k_not_at_gold_doc_count(self) -> None:
        # gold has 3 docs but k=1: IDCG must use min(3, 1)=1 ideal position,
        # not 3 — otherwise IDCG would be inflated relative to what k=1 can
        # ever achieve, making NDCG@1 artificially small.
        # ranked=[a, x, x]; gold={a,b,c} -> DCG@1 = 1/log2(2) = 1.0
        # IDCG@1 (ideal_hits=min(3,1)=1) = 1/log2(2) = 1.0 -> NDCG = 1.0
        assert ndcg_at_k(["a", "x", "x"], {"a", "b", "c"}, k=1) == pytest.approx(1.0)

    def test_empty_gold_docs_raises(self) -> None:
        with pytest.raises(ValueError, match="gold_docs"):
            ndcg_at_k(["a"], set(), k=1)

    def test_k_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            ndcg_at_k(["a"], {"a"}, k=0)


class TestFirstRank:
    def test_returns_1_indexed_rank_of_first_match(self) -> None:
        assert first_rank(["x", "y", "z"], {"y"}) == 2

    def test_returns_none_when_absent(self) -> None:
        assert first_rank(["x", "y"], {"q"}) is None

    def test_returns_earliest_match_among_multiple_targets(self) -> None:
        assert first_rank(["x", "y", "z"], {"z", "y"}) == 2

    def test_empty_ranked_list_returns_none(self) -> None:
        assert first_rank([], {"a"}) is None


class TestReciprocalRank:
    def test_first_position_is_1_0(self) -> None:
        assert reciprocal_rank(["a", "b"], {"a"}) == pytest.approx(1.0)

    def test_third_position_is_one_third(self) -> None:
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1.0 / 3.0)

    def test_absent_is_zero(self) -> None:
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_empty_gold_docs_raises(self) -> None:
        with pytest.raises(ValueError, match="gold_docs"):
            reciprocal_rank(["a"], set())


class TestMeanReciprocalRank:
    def test_averages_per_query_reciprocal_ranks(self) -> None:
        # (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert mean_reciprocal_rank([1.0, 0.5, 0.0]) == pytest.approx(0.5)

    def test_empty_list_is_zero(self) -> None:
        assert mean_reciprocal_rank([]) == 0.0


class TestSupersessionViolated:
    def test_correct_doc_outranks_superseded_is_not_violated(self) -> None:
        # correct at rank 1, superseded at rank 2 -> not violated
        assert supersession_violated(["good", "bad"], {"good"}, {"bad"}) is False

    def test_superseded_doc_outranks_correct_is_violated(self) -> None:
        # superseded at rank 1, correct at rank 2 -> violated
        assert supersession_violated(["bad", "good"], {"good"}, {"bad"}) is True

    def test_superseded_present_correct_entirely_absent_is_violated(self) -> None:
        # worst case: only the outdated doc shows up at all
        assert supersession_violated(["bad", "x"], {"good"}, {"bad"}) is True

    def test_superseded_absent_entirely_is_not_violated(self) -> None:
        # superseded doc doesn't even appear -> cannot outrank anything
        assert supersession_violated(["good", "x"], {"good"}, {"bad"}) is False

    def test_neither_doc_present_is_not_violated(self) -> None:
        assert supersession_violated(["x", "y"], {"good"}, {"bad"}) is False

    def test_tied_absent_case_both_missing_from_short_list(self) -> None:
        assert supersession_violated([], {"good"}, {"bad"}) is False

    def test_empty_correct_docs_raises(self) -> None:
        with pytest.raises(ValueError, match="correct_docs"):
            supersession_violated(["a"], set(), {"bad"})

    def test_empty_superseded_docs_raises(self) -> None:
        with pytest.raises(ValueError, match="superseded_docs"):
            supersession_violated(["a"], {"good"}, set())


class TestSupersessionErrorRate:
    def test_fraction_of_true_violations(self) -> None:
        # 2 violated out of 4 -> 0.5
        assert supersession_error_rate([True, False, True, False]) == pytest.approx(0.5)

    def test_no_violations_is_zero(self) -> None:
        assert supersession_error_rate([False, False]) == 0.0

    def test_all_violations_is_one(self) -> None:
        assert supersession_error_rate([True, True]) == 1.0

    def test_empty_list_is_zero(self) -> None:
        assert supersession_error_rate([]) == 0.0


class TestNoAnswerHitRate:
    def test_fraction_that_returned_any_hit(self) -> None:
        # 1 out of 3 returned a hit -> 1/3
        assert no_answer_hit_rate([True, False, False]) == pytest.approx(1.0 / 3.0)

    def test_all_silent_is_zero(self) -> None:
        assert no_answer_hit_rate([False, False]) == 0.0

    def test_all_returned_hits_is_one(self) -> None:
        assert no_answer_hit_rate([True, True]) == 1.0

    def test_empty_list_is_zero(self) -> None:
        assert no_answer_hit_rate([]) == 0.0


class TestFileRecallAtK:
    def test_delegates_to_recall_at_k_with_same_contract(self) -> None:
        # gold doc "b" at rank 2, k=3 -> full recall (1.0)
        assert file_recall_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0

    def test_single_gold_file_miss_outside_k(self) -> None:
        assert file_recall_at_k(["a", "b", "c"], {"b"}, k=1) == 0.0

    def test_multi_gold_partial_hit_is_fraction(self) -> None:
        # gold = {b, d}; top-3 = {a, b, c} -> only "b" found -> 1/2 = 0.5
        assert file_recall_at_k(["a", "b", "c", "d"], {"b", "d"}, k=3) == pytest.approx(0.5)

    def test_empty_gold_docs_raises(self) -> None:
        with pytest.raises(ValueError, match="gold_docs"):
            file_recall_at_k(["a"], set(), k=1)

    def test_k_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            file_recall_at_k(["a"], {"a"}, k=0)


class TestChunkCoverage:
    """Chunk coverage is the fraction of expected (file, chunk_id) pairs that
    appear among the retrieved fused chunks.  It can also be computed at the
    file-only level when no chunk-level ground truth is available."""

    def test_all_chunks_found(self) -> None:
        found = [
            {"file_path": "a.md", "chunk_id": "chunk-1"},
            {"file_path": "a.md", "chunk_id": "chunk-2"},
            {"file_path": "b.md", "chunk_id": "chunk-1"},
        ]
        # both specified: expected_pairs = { (a.md,chunk-1), (a.md,chunk-2) }
        assert chunk_coverage(found, {"a.md"}, {"chunk-1", "chunk-2"}) == pytest.approx(1.0)

    def test_half_chunks_found(self) -> None:
        found = [
            {"file_path": "a.md", "chunk_id": "chunk-1"},
            {"file_path": "b.md", "chunk_id": "chunk-1"},
        ]
        # expected_pairs = { (a.md,chunk-1), (a.md,chunk-2) }
        # found_pairs = { (a.md,chunk-1), (b.md,chunk-1) }
        # intersection = { (a.md,chunk-1) } -> 1/2 = 0.5
        assert chunk_coverage(found, {"a.md"}, {"chunk-1", "chunk-2"}) == pytest.approx(0.5)

    def test_no_chunks_found(self) -> None:
        found = [{"file_path": "c.md", "chunk_id": "chunk-9"}]
        assert chunk_coverage(found, {"a.md"}, {"chunk-1"}) == pytest.approx(0.0)

    def test_empty_found_returns_zero(self) -> None:
        assert chunk_coverage([], {"a.md"}, {"chunk-1"}) == pytest.approx(0.0)

    def test_file_only_coverage_all_files_found(self) -> None:
        found = [
            {"file_path": "a.md", "chunk_id": "chunk-1"},
            {"file_path": "b.md", "chunk_id": "chunk-2"},
        ]
        # Only expected_files provided (no expected_chunks): count fraction of
        # expected files that have at least one chunk in found.
        assert chunk_coverage(found, {"a.md", "b.md"}) == pytest.approx(1.0)

    def test_file_only_coverage_half_files_found(self) -> None:
        found = [{"file_path": "a.md", "chunk_id": "chunk-1"}]
        assert chunk_coverage(found, {"a.md", "b.md"}) == pytest.approx(0.5)

    def test_no_ground_truth_returns_none(self) -> None:
        # Both expected_files and expected_chunks are None/empty
        found = [{"file_path": "a.md", "chunk_id": "chunk-1"}]
        assert chunk_coverage(found) is None

    def test_chunks_without_files_returns_none(self) -> None:
        # Only expected_chunks, no expected_files -> None
        found = [{"file_path": "a.md", "chunk_id": "chunk-1"}]
        assert chunk_coverage(found, expected_chunks={"chunk-1"}) is None

    def test_skips_chunks_without_file_path_or_chunk_id(self) -> None:
        found = [
            {"file_path": "", "chunk_id": "chunk-1"},  # empty file_path
            {"file_path": "a.md", "chunk_id": ""},  # empty chunk_id
            {"file_path": "a.md", "chunk_id": "chunk-2"},  # valid
        ]
        # Only (a.md, chunk-2) is a valid pair; expected_pairs = {(a.md, chunk-1), (a.md, chunk-2)}
        # -> 1/2 = 0.5
        assert chunk_coverage(found, {"a.md"}, {"chunk-1", "chunk-2"}) == pytest.approx(0.5)


class TestHardQueryFixture:
    """Validates the hard-100 fixture data integrity and backward
    compatibility — ensures the ground-truth contract is practically usable,
    not just syntactically valid."""

    # ------------------------------------------------------------------ #
    # Dataset size and shape
    # ------------------------------------------------------------------ #

    def test_exactly_100_entries(self) -> None:
        from tests.e2e.fixtures.kb_queries_hard_100 import KB_QUERIES_HARD_100

        assert len(KB_QUERIES_HARD_100) == 100

    def test_all_entries_are_3tuple_or_4tuple(self) -> None:
        from tests.e2e.fixtures.kb_queries_hard_100 import KB_QUERIES_HARD_100

        for raw in KB_QUERIES_HARD_100:
            assert len(raw) in (3, 4), f"Unexpected tuple length {len(raw)}"

    def test_98_entries_are_3tuple_2_entries_are_4tuple(self) -> None:
        from tests.e2e.fixtures.kb_queries_hard_100 import KB_QUERIES_HARD_100

        count_3 = sum(1 for q in KB_QUERIES_HARD_100 if len(q) == 3)
        count_4 = sum(1 for q in KB_QUERIES_HARD_100 if len(q) == 4)
        assert count_3 == 98
        assert count_4 == 2

    def test_no_duplicate_ood_entries(self) -> None:
        from tests.e2e.fixtures.kb_queries_hard_100 import KB_QUERIES_HARD_100

        domains = [q[2] for q in KB_QUERIES_HARD_100]
        assert domains.count("ood-biology-crispr") == 1

    # ------------------------------------------------------------------ #
    # Ground-truth paths point to real files
    # ------------------------------------------------------------------ #

    def test_expected_docs_paths_exist_on_disk(self) -> None:
        from tests.e2e.fixtures.kb_queries_hard_100 import KB_QUERIES_HARD_100
        import os

        kb_root = os.environ.get("HARS_MEMORY_EVAL_KB_ROOT")
        if not kb_root:
            pytest.skip("HARS_MEMORY_EVAL_KB_ROOT is not configured")
        for raw in KB_QUERIES_HARD_100:
            if len(raw) == 4:
                _, _, _, docs = raw
                for d in docs:
                    assert os.path.isfile(os.path.join(kb_root, d)), (
                        f"Ground-truth path does not exist: {d}"
                    )

    def test_expected_docs_paths_are_kb_relative(self) -> None:
        from tests.e2e.fixtures.kb_queries_hard_100 import KB_QUERIES_HARD_100

        for raw in KB_QUERIES_HARD_100:
            if len(raw) == 4:
                _, _, _, docs = raw
                for d in docs:
                    assert d.startswith("kb/"), f"Expected doc path should start with kb/: {d}"

    # ------------------------------------------------------------------ #
    # Path canonicalization
    # ------------------------------------------------------------------ #

    def test_canonicalize_strips_project_root(self) -> None:
        from tests.e2e.bench_compare import _canonicalize_retrieved_path

        result = _canonicalize_retrieved_path(
            "/Users/iurii.medvedev/Work/sre-support-pier/kb/k8s/credential-rotation-controller/readme.md"
        )
        assert result == "kb/k8s/credential-rotation-controller/readme.md"

    def test_canonicalize_passes_flat_name_through(self) -> None:
        from tests.e2e.bench_compare import _canonicalize_retrieved_path

        result = _canonicalize_retrieved_path("k8s-credential-rotation-controller-readme.md")
        assert result == "k8s-credential-rotation-controller-readme.md"

    def test_canonicalize_matches_fixture_path(self) -> None:
        """After canonicalization, a retrieved absolute path should match
        the fixture's kb/-relative expected_doc path."""
        from tests.e2e.bench_compare import _canonicalize_retrieved_path

        fixture_path = "kb/k8s/credential-rotation-controller/readme.md"
        retrieved_path = "/Users/iurii.medvedev/Work/sre-support-pier/kb/k8s/credential-rotation-controller/readme.md"
        assert _canonicalize_retrieved_path(retrieved_path) == fixture_path

    # ------------------------------------------------------------------ #
    # normalize_query backward compatibility
    # ------------------------------------------------------------------ #

    def test_normalize_query_3tuple(self) -> None:
        from tests.e2e.bench_compare import normalize_query

        q = normalize_query(("question", ["kw"], "domain"))
        assert q.question == "question"
        assert q.expected_keywords == ["kw"]
        assert q.domain == "domain"
        assert q.expected_docs is None

    def test_normalize_query_4tuple(self) -> None:
        from tests.e2e.bench_compare import normalize_query

        q = normalize_query(("question", ["kw"], "domain", ["kb/doc.md"]))
        assert q.question == "question"
        assert q.expected_docs == ["kb/doc.md"]

    def test_normalize_query_4tuple_with_empty_docs(self) -> None:
        from tests.e2e.bench_compare import normalize_query

        q = normalize_query(("question", ["kw"], "domain", []))
        assert q.expected_docs == []

    def test_normalize_query_all_100_entries(self) -> None:
        from tests.e2e.fixtures.kb_queries_hard_100 import KB_QUERIES_HARD_100
        from tests.e2e.bench_compare import normalize_query

        for raw in KB_QUERIES_HARD_100:
            q = normalize_query(raw)
            assert q.question
            assert isinstance(q.expected_keywords, list)
            assert q.domain
            if len(raw) == 4:
                assert q.expected_docs is not None
