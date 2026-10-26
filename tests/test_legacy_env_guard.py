import pytest

from hars_memory.server.legacy_env_guard import refuse_if_legacy_env


def test_no_op_with_empty_prefix_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHRAG_ANYTHING", "1")
    refuse_if_legacy_env([])  # must not raise — empty list means "check nothing"


def test_raises_on_configured_prefix_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHRAG_FOO", "1")
    with pytest.raises(SystemExit):
        refuse_if_legacy_env(["GRAPHRAG_"])


def test_default_reads_env_var_when_no_arg_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARS_MEMORY_LEGACY_ENV_PREFIXES", "GRAPHRAG_")
    monkeypatch.setenv("GRAPHRAG_FOO", "1")
    with pytest.raises(SystemExit):
        refuse_if_legacy_env()


def test_default_is_empty_when_env_var_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARS_MEMORY_LEGACY_ENV_PREFIXES", raising=False)
    monkeypatch.setenv("GRAPHRAG_FOO", "1")
    refuse_if_legacy_env()  # must not raise — no prefixes configured, package stays quiet
