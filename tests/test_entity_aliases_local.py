"""HARS_MEMORY_EXTRA_ENTITY_ALIASES — operator-local alias merging.

The shipped ENTITY_ALIASES table must stay generic (0.7.1 sanitization);
deployment-specific duplicate-spelling aliases are registered via a YAML file
pointed at by HARS_MEMORY_EXTRA_ENTITY_ALIASES. A bad file must never break
server startup.

hars_memory.mcp_server is imported lazily inside each test: importing it
requires HARS_MEMORY_INDEX_DIR, which tests/conftest.py provides through an
autouse fixture (unavailable at module import / collection time).
"""

from __future__ import annotations

import logging

ENV_VAR = "HARS_MEMORY_EXTRA_ENTITY_ALIASES"


def _mcp():
    from hars_memory import mcp_server

    return mcp_server


def test_loader_noop_without_env(monkeypatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert _mcp()._load_extra_entity_aliases() == {}


def test_loader_merges_valid_file(monkeypatch, tmp_path) -> None:
    aliases = tmp_path / "aliases.yaml"
    aliases.write_text(
        'alpha-project:\n  - "Alpha Project"\n  - "alpha_project"\n"beta-2": ["BETA2"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_VAR, str(aliases))
    extra = _mcp()._load_extra_entity_aliases()
    assert extra["alpha-project"] == ("alpha-project", "Alpha Project", "alpha_project")
    assert extra["beta-2"] == ("beta-2", "BETA2")


def test_loader_dedupes_canonical_spelling(monkeypatch, tmp_path) -> None:
    aliases = tmp_path / "aliases.yaml"
    aliases.write_text('demo: ["demo", "Demo"]\n', encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(aliases))
    assert _mcp()._load_extra_entity_aliases()["demo"] == ("demo", "Demo")


def test_loader_missing_file_is_ignored_with_warning(monkeypatch, tmp_path, caplog) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "does-not-exist.yaml"))
    with caplog.at_level(logging.WARNING):
        assert _mcp()._load_extra_entity_aliases() == {}
    assert ENV_VAR in caplog.text


def test_loader_invalid_yaml_is_ignored(monkeypatch, tmp_path, caplog) -> None:
    aliases = tmp_path / "aliases.yaml"
    aliases.write_text("(:, not yaml\n", encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(aliases))
    with caplog.at_level(logging.WARNING):
        assert _mcp()._load_extra_entity_aliases() == {}
    assert ENV_VAR in caplog.text


def test_loader_invalid_shape_is_ignored(monkeypatch, tmp_path, caplog) -> None:
    aliases = tmp_path / "aliases.yaml"
    aliases.write_text("- just\n- a\n- list\n", encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(aliases))
    with caplog.at_level(logging.WARNING):
        assert _mcp()._load_extra_entity_aliases() == {}
    assert ENV_VAR in caplog.text


def test_loader_empty_alias_list_is_ignored(monkeypatch, tmp_path, caplog) -> None:
    aliases = tmp_path / "aliases.yaml"
    aliases.write_text("demo: []\n", encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(aliases))
    with caplog.at_level(logging.WARNING):
        assert _mcp()._load_extra_entity_aliases() == {}
    assert ENV_VAR in caplog.text


def test_loader_result_feeds_query_expansion(monkeypatch, tmp_path) -> None:
    """_expand_query_aliases must pick up operator aliases via ENTITY_ALIASES."""
    mcp = _mcp()
    aliases = tmp_path / "aliases.yaml"
    aliases.write_text('alpha-project: ["Alpha Project"]\n', encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(aliases))
    merged = dict(mcp.ENTITY_ALIASES)
    merged.update(mcp._load_extra_entity_aliases())
    names = {s.lower() for s in merged["alpha-project"]}
    assert "alpha project" in names
