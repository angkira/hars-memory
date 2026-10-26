"""Knowledge-source manifest: parsing, flag semantics and the two consumers
(indexer ingest roots, ripgrep search roots).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hars_memory.ingest import sources
from hars_memory.retrieval import ripgrep_channel as rgc
from hars_memory.server import index as index_mod

MANIFEST_ENV = sources.HARS_MEMORY_SOURCES_MANIFEST_ENV
LOCAL_ENV = sources.HARS_MEMORY_SOURCES_MANIFEST_LOCAL_ENV


def _write_manifest(tmp_path: Path, body: str, name: str = "knowledge-sources.yaml") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    monkeypatch.delenv(LOCAL_ENV, raising=False)
    monkeypatch.delenv(rgc.HARS_MEMORY_RIPGREP_ROOTS_ENV, raising=False)
    monkeypatch.delenv(rgc.HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV, raising=False)


class TestLoading:
    def test_no_manifest_configured_returns_nothing(self) -> None:
        assert sources.load_sources() == []
        assert sources.ingest_roots() == []
        assert sources.grep_roots() == []

    def test_flags_split_ingest_and_grep_roots(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "kb").mkdir()
        (tmp_path / "meta").mkdir()
        (tmp_path / "repos").mkdir()
        manifest = _write_manifest(
            tmp_path,
            """
version: 1
sources:
  - path: kb
    index: true
    grep: true
  - path: meta
    index: false
    grep: true
  - path: repos
    index: false
    grep: false
""",
        )
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))

        assert sources.ingest_roots() == [tmp_path / "kb"]
        assert sources.grep_roots() == [tmp_path / "kb", tmp_path / "meta"]

    def test_flags_default_to_true(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "kb").mkdir()
        manifest = _write_manifest(tmp_path, "version: 1\nsources:\n  - path: kb\n")
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))

        assert sources.ingest_roots() == [tmp_path / "kb"]
        assert sources.grep_roots() == [tmp_path / "kb"]

    def test_paths_resolve_relative_to_the_manifest_not_the_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        (repo / "kb").mkdir(parents=True)
        manifest = _write_manifest(repo, "version: 1\nsources:\n  - path: kb\n")
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))
        monkeypatch.chdir(tmp_path)

        assert sources.ingest_roots() == [repo / "kb"]

    def test_absolute_paths_are_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        manifest = _write_manifest(
            tmp_path, f"version: 1\nsources:\n  - path: {outside}\n"
        )
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))

        assert sources.ingest_roots() == [outside]

    def test_missing_path_is_skipped_with_a_warning_not_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        (tmp_path / "kb").mkdir()
        manifest = _write_manifest(
            tmp_path,
            "version: 1\nsources:\n  - path: kb\n  - path: repos/not-initialized\n",
        )
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))

        with caplog.at_level("WARNING"):
            roots = sources.ingest_roots()

        assert roots == [tmp_path / "kb"]
        assert "not-initialized" in caplog.text

    def test_files_are_valid_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "AGENTS.md").write_text("rules", encoding="utf-8")
        manifest = _write_manifest(tmp_path, "version: 1\nsources:\n  - path: AGENTS.md\n")
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))

        assert sources.ingest_roots() == [tmp_path / "AGENTS.md"]


class TestLocalOverride:
    def test_default_local_override_is_appended(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "kb").mkdir()
        (tmp_path / "extra").mkdir()
        (tmp_path / "meta").mkdir()
        manifest = _write_manifest(tmp_path, "version: 1\nsources:\n  - path: kb\n")
        (tmp_path / "meta" / "knowledge-sources.local.yaml").write_text(
            "version: 1\nsources:\n  - path: extra\n", encoding="utf-8"
        )
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))

        assert sources.ingest_roots() == [tmp_path / "kb", tmp_path / "extra"]

    def test_override_wins_on_duplicate_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "kb").mkdir()
        manifest = _write_manifest(
            tmp_path, "version: 1\nsources:\n  - path: kb\n    index: true\n"
        )
        override = _write_manifest(
            tmp_path,
            "version: 1\nsources:\n  - path: kb\n    index: false\n",
            name="local.yaml",
        )
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))
        monkeypatch.setenv(LOCAL_ENV, str(override))

        assert sources.ingest_roots() == []
        assert sources.grep_roots() == [tmp_path / "kb"]


class TestValidation:
    def test_configured_but_missing_manifest_is_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(MANIFEST_ENV, str(tmp_path / "nope.yaml"))
        with pytest.raises(sources.SourcesManifestError):
            sources.load_sources()

    def test_unsupported_version_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = _write_manifest(tmp_path, "version: 99\nsources: []\n")
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))
        with pytest.raises(sources.SourcesManifestError):
            sources.load_sources()

    def test_source_without_path_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = _write_manifest(tmp_path, "version: 1\nsources:\n  - index: true\n")
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))
        with pytest.raises(sources.SourcesManifestError):
            sources.load_sources()

    def test_non_boolean_flag_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = _write_manifest(
            tmp_path, "version: 1\nsources:\n  - path: kb\n    index: sometimes\n"
        )
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))
        with pytest.raises(sources.SourcesManifestError):
            sources.load_sources()


class TestIndexerConsumesManifest:
    def test_omitted_paths_use_the_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "kb").mkdir()
        (tmp_path / "learning").mkdir()
        manifest = _write_manifest(
            tmp_path,
            "version: 1\nsources:\n  - path: kb\n  - path: learning\n    index: false\n",
        )
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))

        resolved = index_mod._resolve_ingest_paths(None, tmp_path / "project")

        assert resolved == [tmp_path / "kb"]

    def test_explicit_paths_win_over_the_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "kb").mkdir()
        manifest = _write_manifest(tmp_path, "version: 1\nsources:\n  - path: kb\n")
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))
        project_root = tmp_path / "project"

        resolved = index_mod._resolve_ingest_paths(["docs"], project_root)

        assert resolved == [project_root / "docs"]

    def test_legacy_default_without_a_manifest(self, tmp_path: Path) -> None:
        resolved = index_mod._resolve_ingest_paths(None, tmp_path)

        assert resolved == [tmp_path / ".plans", tmp_path / "docs"]


class TestRipgrepConsumesManifest:
    def test_manifest_supplies_roots_when_env_is_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "kb").mkdir()
        (tmp_path / "repos").mkdir()
        manifest = _write_manifest(
            tmp_path,
            "version: 1\nsources:\n  - path: kb\n  - path: repos\n    grep: false\n",
        )
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))

        assert rgc.default_roots() == [tmp_path / "kb"]

    def test_explicit_env_roots_win_over_the_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "kb").mkdir()
        (tmp_path / "only-this").mkdir()
        manifest = _write_manifest(tmp_path, "version: 1\nsources:\n  - path: kb\n")
        monkeypatch.setenv(MANIFEST_ENV, str(manifest))
        monkeypatch.setenv(rgc.HARS_MEMORY_RIPGREP_ROOTS_ENV, str(tmp_path / "only-this"))

        assert rgc.default_roots() == [tmp_path / "only-this"]

    def test_broken_manifest_degrades_to_no_roots(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed manifest must not break query serving — the channel
        reports itself unavailable instead."""
        monkeypatch.setenv(MANIFEST_ENV, str(tmp_path / "does-not-exist.yaml"))

        assert rgc.default_roots() == []
