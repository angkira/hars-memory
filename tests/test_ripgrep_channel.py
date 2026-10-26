"""Unit tests for tools/memory/retrieval/ripgrep_channel.py.

GPU-free, no LLM calls, no network, and — per this session's constraints —
never touches `/home/demo/.local/share/hars-graphrag/index_v4`
(the live, actively-mutating LightRAG index): every test here either runs
`rg` against synthetic `tmp_path` fixtures or, for the availability-gating
tests, monkeypatches `shutil.which`/PATH rather than depending on whether
the real system binary happens to be installed.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from hars_memory.retrieval import ripgrep_channel as rgc
from hars_memory.retrieval.ripgrep_channel import (
    HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV,
    HARS_MEMORY_RIPGREP_ROOTS_ENV,
    RipgrepSearchHit,
    check_availability,
    default_roots,
    extract_terms,
    search,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestExtractTerms:
    """Item: identifier-term extraction from a natural-language question."""

    def test_extracts_identifiers_from_question(self) -> None:
        terms = extract_terms(
            "What happened in phase_x1_adapter_safe and does Q7X42 relate to vma_native?"
        )
        assert terms == ["phase_x1_adapter_safe", "Q7X42", "vma_native"]

    def test_plain_question_yields_no_terms(self) -> None:
        assert extract_terms("How is training going in general?") == []

    def test_ll_keywords_contribute_identifier_shaped_terms(self) -> None:
        terms = extract_terms("How is training going?", ll_keywords=["Q7X42"])
        assert terms == ["Q7X42"]

    def test_ll_keywords_plain_words_are_rejected(self) -> None:
        # A plain-English keyword must not sneak a noisy substring search in
        # through the keyword side-channel — the whole point of this
        # channel is exact identifiers, not keyword recall.
        terms = extract_terms("How is training going?", ll_keywords=["training", "results"])
        assert terms == []

    def test_question_and_keywords_are_merged_and_deduped(self) -> None:
        terms = extract_terms(
            "What about Q7X42?", ll_keywords=["q7x42", "phase_x1_adapter_safe"]
        )
        assert terms == ["Q7X42", "phase_x1_adapter_safe"]

    def test_empty_question_no_keywords_yields_no_terms(self) -> None:
        assert extract_terms("") == []

    def test_identifiers_supplied_only_via_ll_keywords_are_extracted(self) -> None:
        # Reproduces the live "Call A" regression exactly: the question text
        # carries no identifier at all, and ll_keywords mixes identifier-
        # shaped entries ("qdrant_transplant", "full_scan_threshold") with
        # plain-English ones ("Qdrant", "retrieval parity") — only the
        # identifier-shaped entries must survive.
        terms = extract_terms(
            "What was the outcome of the Qdrant vector storage migration "
            "executed on 2026-07-30, and what was the retrieval parity result?",
            ll_keywords=["Qdrant", "qdrant_transplant", "full_scan_threshold", "retrieval parity"],
        )
        assert terms == ["qdrant_transplant", "full_scan_threshold"]

    def test_terms_are_capped_at_max_query_terms(self) -> None:
        many_keywords = [f"identifier_{i}" for i in range(rgc.MAX_QUERY_TERMS + 5)]
        terms = extract_terms("plain question with no identifiers", ll_keywords=many_keywords)
        assert len(terms) == rgc.MAX_QUERY_TERMS
        assert terms == many_keywords[: rgc.MAX_QUERY_TERMS]

    def test_question_terms_survive_the_cap_ahead_of_keyword_overflow(self) -> None:
        # Question-derived terms are appended first (see extract_terms'
        # docstring), so when the combined list would exceed MAX_QUERY_TERMS,
        # the question's own identifier must never be the one truncated away.
        many_keywords = [f"identifier_{i}" for i in range(rgc.MAX_QUERY_TERMS + 5)]
        terms = extract_terms("What about Q7X42?", ll_keywords=many_keywords)
        assert terms[0] == "Q7X42"
        assert len(terms) == rgc.MAX_QUERY_TERMS


class TestAvailabilityFailSoft:
    """Item: the rg-missing fail-soft path — never an exception."""

    def test_available_when_binary_present(self) -> None:
        availability = check_availability()
        # ripgrep is a hard prerequisite of this dev environment; assert the
        # *shape* of a positive result rather than hardcoding a path.
        if availability.available:
            assert availability.binary_path
            assert availability.reason is None

    def test_unavailable_when_binary_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rgc.shutil, "which", lambda _name: None)
        availability = check_availability()
        assert availability.available is False
        assert availability.binary_path is None
        assert "not found on PATH" in availability.reason

    def test_search_returns_unavailable_result_not_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path / "note.md", "Contains Q7X42 somewhere.\n")
        monkeypatch.setattr(rgc.shutil, "which", lambda _name: None)

        result = search("What about Q7X42?", roots=[tmp_path])

        assert result.available is False
        assert result.hits == []
        assert result.unavailable_reason is not None
        assert result.query_terms == ("Q7X42",)

    def test_search_never_raises_on_binary_vanishing_mid_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate the race: available at check_availability() time, but the
        # exec itself raises FileNotFoundError.
        _write(tmp_path / "note.md", "Q7X42 mentioned here.\n")

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise FileNotFoundError("rg vanished")

        monkeypatch.setattr(rgc.subprocess, "run", _boom)

        result = search("What about Q7X42?", roots=[tmp_path])

        assert result.available is True  # binary WAS found at the gate check
        assert result.hits == []


class TestNoIdentifierTermsReturnsNothing:
    """Item: 'no identifier terms -> return nothing' behaviour."""

    def test_plain_question_returns_no_hits_without_invoking_rg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path / "note.md", "Some prose about training results.\n")

        def _fail_if_called(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("rg must not be invoked when there are no identifier terms")

        monkeypatch.setattr(rgc.subprocess, "run", _fail_if_called)

        result = search("How is training going in general today?", roots=[tmp_path])

        assert result.hits == []
        assert result.query_terms == ()

    def test_non_identifier_ll_keywords_alone_returns_no_hits_without_invoking_rg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Plain-English ll_keywords (no identifier shape) must not sneak a
        # noisy substring search in through the keyword side-channel either
        # — this channel exists for exact identifiers, not keyword recall.
        _write(tmp_path / "note.md", "Some prose about Qdrant retrieval parity.\n")

        def _fail_if_called(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("rg must not be invoked when no term is identifier-shaped")

        monkeypatch.setattr(rgc.subprocess, "run", _fail_if_called)

        result = search(
            "What was the outcome of the Qdrant migration?",
            ll_keywords=["Qdrant", "retrieval parity"],
            roots=[tmp_path],
        )

        assert result.hits == []
        assert result.query_terms == ()


@pytest.mark.skipif(
    check_availability().available is False, reason="ripgrep (rg) not installed"
)
class TestSearchAgainstSyntheticTree:
    """End-to-end `search()` against a real, synthetic tmp_path tree — the
    real `rg` binary is exercised here, but never against the live index.
    """

    def _corpus(self, tmp_path: Path) -> Path:
        _write(
            tmp_path / ".reports" / "report_a.md",
            "# Report A\n\nExperiment Q7X42 stabilized the gate at step 80.\n"
            "Q7X42 also appears again here for emphasis.\n",
        )
        _write(
            tmp_path / ".session" / "session_b.md",
            "# Session B\n\nWe discussed phase_x1_adapter_safe briefly, "
            "no relation to q7x42-suffix tokens.\n",
        )
        _write(
            tmp_path / "docs" / "guide.md",
            "# Guide\n\nGeneral training guidance, no identifiers here.\n",
        )
        return tmp_path

    def test_finds_whole_token_identifier_hit(self, tmp_path: Path) -> None:
        corpus = self._corpus(tmp_path)
        result = search("What happened with Q7X42?", roots=[corpus], top_k=5)

        assert result.available is True
        assert result.hits, "expected at least one hit for Q7X42"
        top = result.hits[0]
        assert top.file_path.endswith("report_a.md")
        assert top.term_hits.get("Q7X42", 0) >= 2

    def test_substring_hit_scores_lower_than_whole_token_hit(self, tmp_path: Path) -> None:
        corpus = self._corpus(tmp_path)
        result = search("Anything about Q7X42?", roots=[corpus], top_k=10)

        by_file = {Path(h.file_path).name: h for h in result.hits}
        assert "report_a.md" in by_file
        assert "session_b.md" in by_file
        # report_a.md has two WHOLE-TOKEN "Q7X42" hits; session_b.md has one
        # SUBSTRING hit ("q7x42-suffix"). Whole-token must outrank substring.
        assert by_file["report_a.md"].score > by_file["session_b.md"].score

    def test_no_match_returns_empty_hits_not_error(self, tmp_path: Path) -> None:
        corpus = self._corpus(tmp_path)
        result = search("What about ZZZ99notfound?", roots=[corpus], top_k=5)
        assert result.available is True
        assert result.hits == []

    def test_result_hit_shape_mirrors_bm25_hit_fields(self, tmp_path: Path) -> None:
        corpus = self._corpus(tmp_path)
        result = search("What happened with Q7X42?", roots=[corpus], top_k=5)
        assert result.hits
        hit = result.hits[0]
        assert isinstance(hit, RipgrepSearchHit)
        # Same first-four-field shape as bm25_index.BM25SearchHit.
        assert isinstance(hit.chunk_id, str) and hit.chunk_id.startswith("file:")
        assert isinstance(hit.score, float)
        assert isinstance(hit.content, str)
        assert isinstance(hit.file_path, str)

    def test_timeout_returns_empty_result_not_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus = self._corpus(tmp_path)

        import subprocess as _subprocess

        def _timeout(*_args: object, **_kwargs: object) -> None:
            raise _subprocess.TimeoutExpired(cmd="rg", timeout=0.001)

        monkeypatch.setattr(rgc.subprocess, "run", _timeout)

        result = search("What happened with Q7X42?", roots=[corpus], timeout_seconds=0.001)
        assert result.hits == []
        assert result.timed_out is True

    def test_co_occurrence_of_multiple_terms_boosts_score(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "single.md",
            "Q7X42 appears once here, nothing else related.\n",
        )
        _write(
            tmp_path / "both.md",
            "Q7X42 and vma_native both appear together in this note.\n",
        )
        result = search(
            "How do Q7X42 and vma_native relate?", roots=[tmp_path], top_k=5
        )
        by_name = {Path(h.file_path).name: h for h in result.hits}
        assert by_name["both.md"].score > by_name["single.md"].score

    def test_ignores_memoryignore_excluded_files(self, tmp_path: Path) -> None:
        _write(tmp_path / "keep.md", "Q7X42 lives here.\n")
        _write(tmp_path / "secret_q7x42.md", "Q7X42 also lives here, but secretly.\n")
        _write(tmp_path / ".memoryignore", "secret_*\n")

        result = search("What about Q7X42?", roots=[tmp_path], top_k=10)

        matched_names = {Path(h.file_path).name for h in result.hits}
        assert "keep.md" in matched_names
        assert "secret_q7x42.md" not in matched_names

    def test_legacy_graphragignore_warns_and_is_not_applied(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write(tmp_path / "keep.md", "Q7X42 lives here.\n")
        _write(tmp_path / "secret_q7x42.md", "Q7X42 also lives here.\n")
        _write(tmp_path / ".graphragignore", "secret_*\n")

        with caplog.at_level("WARNING"):
            result = search("What about Q7X42?", roots=[tmp_path], top_k=10)

        matched_names = {Path(h.file_path).name for h in result.hits}
        # Legacy ignore file's pattern is NOT applied -> the "secret" file is
        # still found (loud warning, not a silent exclusion).
        assert "secret_q7x42.md" in matched_names
        assert any("legacy" in record.message.lower() for record in caplog.records)

    def test_excludes_venv_and_outputs_directories(self, tmp_path: Path) -> None:
        _write(tmp_path / "real.md", "Q7X42 in the real corpus.\n")
        _write(tmp_path / ".venv" / "lib" / "site.md", "Q7X42 in a venv artifact.\n")
        _write(tmp_path / "outputs" / "run1" / "log.md", "Q7X42 in build output.\n")

        result = search("What about Q7X42?", roots=[tmp_path], top_k=10)

        matched_names = {Path(h.file_path).name for h in result.hits}
        assert matched_names == {"real.md"}

    def test_identifiers_supplied_only_via_ll_keywords_find_real_hits(
        self, tmp_path: Path
    ) -> None:
        """Real, non-monkeypatched end-to-end reproduction of the live "Call
        A" regression: an identifier-shaped term arrives ONLY through
        ll_keywords, the question text is clean prose. `rg` must still be
        invoked and must find the file."""
        _write(
            tmp_path / ".session" / "2026-07-30_qdrant-migration-execution.md",
            "# Qdrant migration\n\nqdrant_transplant.py ran the migration; "
            "full_scan_threshold stayed at its configured default throughout.\n",
        )
        result = search(
            "What was the outcome of the Qdrant vector storage migration "
            "executed on 2026-07-30, and what was the retrieval parity result?",
            ll_keywords=["Qdrant", "qdrant_transplant", "full_scan_threshold", "retrieval parity"],
            roots=[tmp_path],
            top_k=5,
        )

        assert result.available is True
        assert result.query_terms == ("qdrant_transplant", "full_scan_threshold")
        assert result.hits, "expected the ll_keywords-only identifiers to find the file"
        assert result.hits[0].file_path.endswith("2026-07-30_qdrant-migration-execution.md")

    def test_measured_latency_is_reported_and_reasonable(self, tmp_path: Path) -> None:
        corpus = self._corpus(tmp_path)
        result = search("What happened with Q7X42?", roots=[corpus], top_k=5)
        assert result.latency_seconds >= 0.0
        assert result.latency_seconds < 2.0  # generous upper bound for a tiny fixture


class TestScoreFormula:
    """Item: the score formula — component-level checks against
    ripgrep_channel's internal helpers (documented in the module docstring).
    """

    def test_recency_multiplier_is_one_at_zero_age(self) -> None:
        now = time.time()
        assert rgc._recency_multiplier(now, now=now) == pytest.approx(1.0)

    def test_recency_multiplier_floors_for_old_files(self) -> None:
        now = time.time()
        very_old = now - rgc.RECENCY_HORIZON_DAYS * 86_400.0 * 10
        assert rgc._recency_multiplier(very_old, now=now) == pytest.approx(rgc.RECENCY_FLOOR)

    def test_recency_multiplier_is_monotonic_in_age(self) -> None:
        now = time.time()
        fresher = rgc._recency_multiplier(now - 1 * 86_400.0, now=now)
        older = rgc._recency_multiplier(now - 100 * 86_400.0, now=now)
        assert fresher > older

    def test_whole_token_match_detection(self) -> None:
        line = "Experiment Q7X42 stabilized the gate."
        start = line.index("Q7X42")
        end = start + len("Q7X42")
        assert rgc._is_whole_token_match(line, start, end) is True

    def test_substring_match_detection(self) -> None:
        # The match is embedded inside a longer word on both sides, so it is
        # a substring hit, not a whole-token hit.
        line = "prefixq7x42suffix"
        start = line.index("q7x42")
        end = start + len("q7x42")
        assert rgc._is_whole_token_match(line, start, end) is False

    def test_path_type_weight_prioritizes_curated_sections(self, tmp_path: Path) -> None:
        session_file = tmp_path / ".session" / "note.md"
        code_file = tmp_path / "tools" / "memory" / "thing.py"
        roots = [tmp_path]
        assert rgc._path_type_weight(session_file, roots) > rgc._path_type_weight(
            code_file, roots
        )

    def test_path_type_weight_for_memory_dir_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_root = tmp_path / "project"
        memory_dir = tmp_path / "claude_memory"
        memory_file = memory_dir / "note.md"
        roots = [project_root, memory_dir]
        monkeypatch.setenv(rgc.HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV, str(memory_dir))
        assert rgc._path_type_weight(memory_file, roots) == pytest.approx(
            rgc._MEMORY_DIR_PATH_WEIGHT
        )

    def test_extra_manifest_roots_do_not_get_the_memory_dir_bonus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a knowledge-source manifest `roots` holds many curated roots;
        only the configured memory dir may claim the memory-dir weight."""
        kb_root = tmp_path / "kb"
        learning_root = tmp_path / "learning"
        monkeypatch.delenv(rgc.HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV, raising=False)
        assert rgc._path_type_weight(
            learning_root / "note.md", [kb_root, learning_root]
        ) == pytest.approx(rgc._DEFAULT_PATH_WEIGHT)


class TestGlobConventionsMatchWalker:
    """Drift guard: this module intentionally DUPLICATES (not imports)
    walker.py's private include/exclude glob constants, translated into
    rg's glob syntax (see ripgrep_channel.py's module docstring). This test
    asserts the two stay semantically equivalent — a future edit to
    walker.py's globs that isn't mirrored here will fail this test rather
    than silently diverging the two channels' file universes.
    """

    def test_include_globs_cover_the_same_extensions(self) -> None:
        from hars_memory.ingest import walker

        walker_extensions = {g.removeprefix("**/*") for g in walker._DEFAULT_INCLUDE_GLOBS}
        rg_extensions = {g.removeprefix("*") for g in rgc._INCLUDE_GLOBS_RG}
        assert walker_extensions == rg_extensions

    def test_exclude_globs_cover_the_same_names(self) -> None:
        from hars_memory.ingest import walker

        def _normalize(pattern: str) -> str:
            return pattern.strip("*/").removeprefix(".")

        walker_names = {_normalize(g) for g in walker._DEFAULT_EXCLUDE_GLOBS}
        rg_names = {_normalize(g) for g in rgc._EXCLUDE_GLOBS_RG}
        # rg's set is a superset (it adds .venv/ and outputs/ — see module
        # docstring "Search roots and filters" for why).
        assert walker_names <= rg_names

    def test_ignore_file_name_matches_walker(self) -> None:
        from hars_memory.ingest import walker

        assert rgc._DEFAULT_IGNORE_FILE == walker._DEFAULT_IGNORE_FILE
        assert rgc._LEGACY_IGNORE_FILE == walker._LEGACY_IGNORE_FILE


class TestDefaultRoots:
    """C1 fix #3: HARS_MEMORY_RIPGREP_ROOTS_ENV replaces the old hardcoded
    `_PROJECT_ROOT` positional argument — default_roots() now reads its
    root(s) from that env var (comma-separated), with no meaningful default."""

    def test_roots_env_unset_yields_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(HARS_MEMORY_RIPGREP_ROOTS_ENV, raising=False)
        monkeypatch.delenv(HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV, raising=False)
        assert default_roots() == []

    def test_roots_env_set_single_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(HARS_MEMORY_RIPGREP_ROOTS_ENV, str(tmp_path))
        monkeypatch.delenv(HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV, raising=False)
        roots = default_roots()
        assert roots == [tmp_path.resolve()]

    def test_roots_env_set_comma_separated_multiple_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        monkeypatch.setenv(HARS_MEMORY_RIPGREP_ROOTS_ENV, f"{root_a}, {root_b}")
        monkeypatch.delenv(HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV, raising=False)
        roots = default_roots()
        assert roots == [root_a.resolve(), root_b.resolve()]

    def test_memory_dir_env_set_is_appended(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        monkeypatch.setenv(HARS_MEMORY_RIPGREP_ROOTS_ENV, str(tmp_path / "project"))
        monkeypatch.setenv(HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV, str(memory_dir))
        roots = default_roots()
        assert roots == [(tmp_path / "project").resolve(), memory_dir.resolve()]

    def test_memory_dir_not_duplicated_if_same_as_configured_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(HARS_MEMORY_RIPGREP_ROOTS_ENV, str(tmp_path))
        monkeypatch.setenv(HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV, str(tmp_path))
        roots = default_roots()
        assert roots == [tmp_path.resolve()]

    def test_memory_dir_appended_even_when_roots_env_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Legitimate pre-existing behaviour preserved: the memory dir is
        appended unconditionally when configured, independent of whether the
        primary roots env var is set."""
        monkeypatch.delenv(HARS_MEMORY_RIPGREP_ROOTS_ENV, raising=False)
        monkeypatch.setenv(HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV, str(tmp_path))
        roots = default_roots()
        assert roots == [tmp_path.resolve()]


class TestSearchUnavailableWithoutConfiguredRoots:
    """C1 fix #3: search() must report unavailable (not silently search cwd)
    when handed an empty roots list — the shape default_roots() returns when
    HARS_MEMORY_RIPGREP_ROOTS is unset."""

    def test_empty_roots_reports_unavailable_not_a_silent_search(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rgc, "check_availability", lambda: rgc.RipgrepAvailability(
            available=True, reason=None, binary_path="/usr/bin/rg"
        ))
        result = search("what is Q7X42?", roots=[])
        assert result.available is False
        assert result.hits == []
        assert HARS_MEMORY_RIPGREP_ROOTS_ENV in (result.unavailable_reason or "")

    def test_configured_roots_are_actually_searched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(HARS_MEMORY_RIPGREP_ROOTS_ENV, raising=False)
        _write(tmp_path / "docs" / "note.md", "Q7X42 is the identifier under test.\n")
        monkeypatch.setenv(HARS_MEMORY_RIPGREP_ROOTS_ENV, str(tmp_path))
        roots = default_roots()
        result = search("Q7X42", roots=roots)
        assert result.available is True
        assert any("note.md" in hit.file_path for hit in result.hits)


class TestAsearchDoesNotBlockEventLoop:
    """`asearch()` must offload the blocking `rg` subprocess to a worker
    thread: while it runs, other coroutines on the same loop must keep being
    scheduled. Regression guard for calling `search()` inline in a coroutine.
    """

    _BLOCKING_SECONDS = 0.3
    _TICK_SECONDS = 0.01

    def test_concurrent_coroutine_keeps_ticking_during_search(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rgc, "check_availability", lambda: rgc.RipgrepAvailability(
            available=True, reason=None, binary_path="/usr/bin/rg"
        ))

        def _blocking_run_rg(cmd: list[str], *, timeout_seconds: float) -> tuple[str, bool]:
            time.sleep(self._BLOCKING_SECONDS)  # stands in for a slow `rg`
            return "", False

        monkeypatch.setattr(rgc, "_run_rg", _blocking_run_rg)

        async def scenario() -> tuple[int, object]:
            ticks = 0
            done = False

            async def ticker() -> int:
                nonlocal ticks
                while not done:
                    await asyncio.sleep(self._TICK_SECONDS)
                    ticks += 1
                return ticks

            ticker_task = asyncio.create_task(ticker())
            result = await rgc.asearch("Q7X42", roots=[tmp_path])
            done = True
            await ticker_task
            return ticks, result

        ticks, result = asyncio.run(scenario())
        assert result.available is True
        # A blocked loop yields ~0 ticks; a free one yields ~30 at 10ms.
        assert ticks >= 10, f"event loop appeared blocked: only {ticks} ticks"

    def test_asearch_matches_search_result(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(HARS_MEMORY_RIPGREP_ROOTS_ENV, raising=False)
        _write(tmp_path / "docs" / "note.md", "Q7X42 is the identifier under test.\n")
        sync_result = search("Q7X42", roots=[tmp_path])
        async_result = asyncio.run(rgc.asearch("Q7X42", roots=[tmp_path]))
        assert [h.file_path for h in async_result.hits] == [h.file_path for h in sync_result.hits]
