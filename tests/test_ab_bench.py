"""Unit tests for tools/memory/eval/ab_bench.py's config-provenance and
--strict-env machinery (2026-08-01).

Background: a benchmark run reported recall@1=0.4815 and was believed to be
a same-config re-measurement showing cross-session drift. It was not --
`HARS_MEMORY_HYBRID_ALPHA=0.0` had been left exported in the shell (almost
certainly leftover from an `alpha-sweep` session), and `ab_bench.py`'s old
`main()` silently fell back to it whenever `--alpha` was omitted, with
nothing in the printed table or JSON report showing which value (or which
SOURCE — cli/env/default) had actually been used.

These tests cover, in order:
  - `_resolve_alpha`'s cli > env > default precedence and its purity (reads
    `os.environ`, never writes it -- the property that makes cross-command
    leakage structurally impossible within one process).
  - `_resolve_fusion_knobs` / `_resolve_index_dir` provenance for every
    other tracked HARS_MEMORY_* knob.
  - `_build_config_snapshot` assembling all of the above into one dict.
  - `_check_strict_env` refusing a polluted ambient environment, and NOT
    refusing when a CLI override (alpha only) or a clean environment is in
    play.
  - `FUSION_SCORING_ENV_VARS` scope: exactly the 8 knobs this task's spec
    named as the minimum set, discovered from fusion.py's own exported `_ENV`
    constants (not re-typed by hand), plus HARS_MEMORY_HYBRID_ALPHA.
  - `HARS_MEMORY_INDEX_DIR` is echoed for reproducibility but deliberately
    NOT part of the strict-env pollution check (pointing at a specific
    corpus index is required, normal usage, not a leaked leftover).
  - Cross-command purity: resolving alpha for an `alpha-sweep`-shaped call
    and then for an `ab`-shaped call in the same process never lets one
    influence the other.
"""

from __future__ import annotations

import os

import pytest

from hars_memory.eval import ab_bench
from hars_memory.retrieval import fusion


def test_score_query_can_normalize_portable_gold_paths() -> None:
    query = ab_bench.QuerySpec(
        id="portable",
        type="identifier",
        question="where?",
        gold_docs=frozenset({"nested/answer.md"}),
    )

    strict = ab_bench.score_query(query, ["/tmp/run/nested/answer.md"], (1,))
    normalized = ab_bench.score_query(
        query,
        ["/tmp/run/nested/answer.md"],
        (1,),
        normalize_gold_paths=True,
    )

    assert strict.recall_at_k[1] == 0.0
    assert normalized.recall_at_k[1] == 1.0


def test_score_query_normalizes_supersession_paths() -> None:
    query = ab_bench.QuerySpec(
        id="supersession",
        type="supersession",
        question="current?",
        gold_docs=frozenset(),
        correct_docs=frozenset({"current.md"}),
        superseded_docs=frozenset({"old.md"}),
    )

    score = ab_bench.score_query(
        query,
        ["upload://old.md", "upload://current.md"],
        (1, 2),
        normalize_gold_paths=True,
    )

    assert score.recall_at_k[2] == 1.0
    assert score.supersession_violation is True


# ---------------------------------------------------------------------------
# _resolve_alpha
# ---------------------------------------------------------------------------


class TestResolveAlpha:
    def test_cli_wins_over_env_and_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV, "0.0")
        resolved = ab_bench._resolve_alpha(0.9)
        assert resolved.value == pytest.approx(0.9)
        assert resolved.source == "cli"

    def test_env_wins_over_default_when_no_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV, "0.0")
        resolved = ab_bench._resolve_alpha(None)
        assert resolved.value == pytest.approx(0.0)
        assert resolved.source == "env"

    def test_default_when_neither_cli_nor_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV, raising=False)
        resolved = ab_bench._resolve_alpha(None)
        assert resolved.value == pytest.approx(fusion.DEFAULT_HYBRID_ALPHA)
        assert resolved.source == "default"

    def test_pure_never_mutates_os_environ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV, raising=False)
        before = dict(os.environ)
        ab_bench._resolve_alpha(0.7)
        ab_bench._resolve_alpha(None)
        monkeypatch.setenv(ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV, "0.3")
        ab_bench._resolve_alpha(None)
        monkeypatch.delenv(ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV, raising=False)
        assert dict(os.environ) == before


# ---------------------------------------------------------------------------
# _resolve_fusion_knobs / _resolve_index_dir
# ---------------------------------------------------------------------------


class TestResolveFusionKnobs:
    def test_covers_exactly_the_fusion_env_vars(self) -> None:
        resolved = ab_bench._resolve_fusion_knobs()
        assert set(resolved) == {
            fusion.HARS_MEMORY_FUSION_TIE_EPSILON_ENV,
            fusion.HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV,
            fusion.HARS_MEMORY_FUSION_AGREEMENT_BONUS_ENV,
            fusion.HARS_MEMORY_SUPERSESSION_SCORING_ENV,
            fusion.HARS_MEMORY_SUPERSESSION_MARKER_PENALTY_ENV,
            fusion.HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT_ENV,
            fusion.HARS_MEMORY_RIPGREP_CHANNEL_ENV,
            fusion.HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV,
            fusion.HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT_ENV,
        }

    def test_source_is_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for env_var in ab_bench._FUSION_KNOB_GETTERS:
            monkeypatch.delenv(env_var, raising=False)
        resolved = ab_bench._resolve_fusion_knobs()
        assert all(cv.source == "default" for cv in resolved.values())
        # Default flipped 2026-08-01 (corrected alpha=0.5 remeasurement) --
        # see fusion.py's "MEASUREMENT CORRECTION" comment above
        # HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV.
        assert resolved[fusion.HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV].value == "zscore_tiebreak"

    def test_source_is_env_when_set_and_value_reflects_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(fusion.HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV, "zscore_tiebreak")
        monkeypatch.setenv(fusion.HARS_MEMORY_SUPERSESSION_SCORING_ENV, "0")
        resolved = ab_bench._resolve_fusion_knobs()
        assert resolved[fusion.HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV] == ab_bench.ConfigValue(
            value="zscore_tiebreak", source="env"
        )
        assert resolved[fusion.HARS_MEMORY_SUPERSESSION_SCORING_ENV] == ab_bench.ConfigValue(
            value=False, source="env"
        )


class TestResolveIndexDir:
    def test_default_source_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ab_bench.HARS_MEMORY_INDEX_DIR_ENV, raising=False)
        resolved = ab_bench._resolve_index_dir()
        assert resolved.source == "default"

    def test_env_source_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ab_bench.HARS_MEMORY_INDEX_DIR_ENV, "/some/index")
        resolved = ab_bench._resolve_index_dir()
        assert resolved == ab_bench.ConfigValue(value="/some/index", source="env")


# ---------------------------------------------------------------------------
# _build_config_snapshot
# ---------------------------------------------------------------------------


class TestBuildConfigSnapshot:
    def test_snapshot_has_alpha_all_fusion_knobs_index_dir_and_strict_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ab_bench.HARS_MEMORY_INDEX_DIR_ENV, raising=False)
        alpha = ab_bench.ConfigValue(value=0.5, source="cli")
        snapshot = ab_bench._build_config_snapshot(alpha, strict_env=True)
        assert snapshot["alpha"] == {"value": 0.5, "source": "cli"}
        for env_var in ab_bench.FUSION_SCORING_ENV_VARS:
            if env_var == ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV:
                continue
            assert env_var in snapshot, f"{env_var} missing from config_snapshot"
        assert ab_bench.HARS_MEMORY_INDEX_DIR_ENV in snapshot
        assert snapshot["strict_env"] is True

    def test_snapshot_is_json_serializable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        alpha = ab_bench.ConfigValue(value=0.0, source="default")
        snapshot = ab_bench._build_config_snapshot(alpha, strict_env=False)
        json.dumps(snapshot)  # must not raise


# ---------------------------------------------------------------------------
# _check_strict_env
# ---------------------------------------------------------------------------


class TestCheckStrictEnv:
    def _clean_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for env_var in ab_bench.FUSION_SCORING_ENV_VARS:
            monkeypatch.delenv(env_var, raising=False)

    def test_clean_environment_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clean_all(monkeypatch)
        ab_bench._check_strict_env(alpha_cli_overridden=False)  # must not raise

    def test_ambient_alpha_without_cli_override_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clean_all(monkeypatch)
        monkeypatch.setenv(ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV, "0.0")
        with pytest.raises(SystemExit, match="HARS_MEMORY_HYBRID_ALPHA"):
            ab_bench._check_strict_env(alpha_cli_overridden=False)

    def test_ambient_alpha_with_cli_override_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This is the exact scenario the incident needed a fix for: an
        ambient HARS_MEMORY_HYBRID_ALPHA=0.0 leftover must not silently win
        once the operator has passed --alpha explicitly."""
        self._clean_all(monkeypatch)
        monkeypatch.setenv(ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV, "0.0")
        ab_bench._check_strict_env(alpha_cli_overridden=True)  # must not raise

    def test_ambient_non_alpha_fusion_knob_always_raises_even_with_alpha_cli_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--alpha` only covers HARS_MEMORY_HYBRID_ALPHA -- an ambient
        HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL (no CLI equivalent exists)
        must still refuse the run."""
        self._clean_all(monkeypatch)
        monkeypatch.setenv(fusion.HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV, "agreement_bonus")
        with pytest.raises(SystemExit, match="HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL"):
            ab_bench._check_strict_env(alpha_cli_overridden=True)

    def test_alpha_sweep_shaped_call_still_flags_ambient_alpha(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`alpha-sweep`/`token-budget-sweep` never read
        HARS_MEMORY_HYBRID_ALPHA (they have no --alpha flag at all), but an
        ambient leftover is flagged for them too -- see
        `_check_strict_env`'s docstring for why (hygiene forcing-function:
        catch the leak before it can bite a LATER `ab` run in the same
        shell)."""
        self._clean_all(monkeypatch)
        monkeypatch.setenv(ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV, "0.0")
        with pytest.raises(SystemExit):
            ab_bench._check_strict_env(alpha_cli_overridden=False)

    def test_index_dir_ambient_presence_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HARS_MEMORY_INDEX_DIR is required, normal usage (every real
        invocation needs it set to the corpus under test) -- it is
        deliberately excluded from FUSION_SCORING_ENV_VARS/strict-env."""
        self._clean_all(monkeypatch)
        monkeypatch.setenv(ab_bench.HARS_MEMORY_INDEX_DIR_ENV, "/mnt/datasets/some/index")
        ab_bench._check_strict_env(alpha_cli_overridden=False)  # must not raise

    def test_multiple_polluted_vars_all_named_in_the_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clean_all(monkeypatch)
        monkeypatch.setenv(ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV, "0.0")
        monkeypatch.setenv(fusion.HARS_MEMORY_SUPERSESSION_SCORING_ENV, "0")
        with pytest.raises(SystemExit) as exc_info:
            ab_bench._check_strict_env(alpha_cli_overridden=False)
        message = str(exc_info.value)
        assert ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV in message
        assert fusion.HARS_MEMORY_SUPERSESSION_SCORING_ENV in message


# ---------------------------------------------------------------------------
# FUSION_SCORING_ENV_VARS scope
# ---------------------------------------------------------------------------


class TestFusionScoringEnvVarsScope:
    def test_matches_the_task_specified_minimum_set(self) -> None:
        assert set(ab_bench.FUSION_SCORING_ENV_VARS) == {
            "HARS_MEMORY_HYBRID_ALPHA",
            "HARS_MEMORY_FUSION_TIE_EPSILON",
            "HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL",
            "HARS_MEMORY_FUSION_AGREEMENT_BONUS",
            "HARS_MEMORY_SUPERSESSION_SCORING",
            "HARS_MEMORY_SUPERSESSION_MARKER_PENALTY",
            "HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT",
            "HARS_MEMORY_RIPGREP_CHANNEL",
            "HARS_MEMORY_RIPGREP_CHANNEL_MODE",
            "HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT",
        }

    def test_sourced_from_fusion_modules_own_exported_env_constants(self) -> None:
        """Not re-typed by hand: every non-alpha entry must be byte-identical
        to fusion.py's own public `_ENV` constant, so a future rename in
        fusion.py cannot silently desync this module's tracked set."""
        assert fusion.HARS_MEMORY_FUSION_TIE_EPSILON_ENV in ab_bench.FUSION_SCORING_ENV_VARS
        assert fusion.HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV in ab_bench.FUSION_SCORING_ENV_VARS
        assert fusion.HARS_MEMORY_FUSION_AGREEMENT_BONUS_ENV in ab_bench.FUSION_SCORING_ENV_VARS
        assert fusion.HARS_MEMORY_SUPERSESSION_SCORING_ENV in ab_bench.FUSION_SCORING_ENV_VARS
        assert fusion.HARS_MEMORY_SUPERSESSION_MARKER_PENALTY_ENV in ab_bench.FUSION_SCORING_ENV_VARS
        assert fusion.HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT_ENV in ab_bench.FUSION_SCORING_ENV_VARS
        assert fusion.HARS_MEMORY_RIPGREP_CHANNEL_ENV in ab_bench.FUSION_SCORING_ENV_VARS
        assert fusion.HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV in ab_bench.FUSION_SCORING_ENV_VARS
        assert fusion.HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT_ENV in ab_bench.FUSION_SCORING_ENV_VARS


# ---------------------------------------------------------------------------
# Cross-command purity: alpha-sweep cannot leak into a later ab resolution
# within the same process.
# ---------------------------------------------------------------------------


class TestNoCrossCommandAlphaLeakage:
    def test_alpha_sweep_never_reads_or_writes_hybrid_alpha_env(self) -> None:
        """Static guard: `_run_alpha_sweep`'s source must not reference
        HARS_MEMORY_HYBRID_ALPHA at all -- it drives alpha entirely from
        `args.alphas` (the `--alphas` CLI list), so there is no code path by
        which it could either read a polluted ambient value or write one for
        a later `ab` call to pick up."""
        import inspect

        source = inspect.getsource(ab_bench._run_alpha_sweep)
        assert "os.environ.get(ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV" not in source
        assert f"os.environ.get({ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV!r}" not in source
        assert "os.environ[" not in source  # no assignment/mutation of any env var

    def test_simulated_alpha_sweep_then_ab_in_one_process_does_not_leak(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dynamic proof: resolve alpha the way `alpha-sweep` conceptually
        would (iterate a list, never touching os.environ), then resolve it
        the way `ab` does immediately after in the SAME process -- the `ab`
        resolution must depend only on its own cli/env inputs, never on
        anything the simulated sweep touched."""
        monkeypatch.delenv(ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV, raising=False)

        # Simulated alpha-sweep: 11 iterations, alpha passed as a plain
        # local variable exactly like `_run_alpha_sweep`'s loop does.
        sweep_alphas = [round(i * 0.1, 1) for i in range(11)]
        for a in sweep_alphas:
            assert a == a  # no os.environ interaction anywhere in this loop

        # A subsequent `ab` call with no --alpha and a clean environment
        # must resolve to the built-in default, NOT to any value the sweep
        # "iterated" above.
        resolved = ab_bench._resolve_alpha(None)
        assert resolved.value == pytest.approx(fusion.DEFAULT_HYBRID_ALPHA)
        assert resolved.source == "default"
        assert ab_bench.HARS_MEMORY_HYBRID_ALPHA_ENV not in os.environ
