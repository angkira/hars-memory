"""Test-collection bootstrap for tools/memory/tests.

hars_memory is a real installed package (editable install of the
tools/memory project via `uv sync`), so no sys.path manipulation is
needed here.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_required_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """hars_memory.mcp_server now REQUIRES HARS_MEMORY_INDEX_DIR /
    HARS_MEMORY_STAGING_DIR to be set (I4 of the 2026-08-25 final-review fix
    wave: no machine-specific defaults) — every real test run therefore needs
    them set to *something*. Provide harmless per-test tmp_path-based values
    here so the majority of tests, which don't care about these paths' exact
    value, don't each have to set them individually. Tests that DO care
    already set their own value explicitly (e.g. via
    test_mcp_server.py::_load_mcp_module_with_env), which simply overrides
    what this fixture set.
    """
    monkeypatch.setenv("HARS_MEMORY_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("HARS_MEMORY_STAGING_DIR", str(tmp_path / "staging"))
    # Keep the live-trace SSE listener (mcp_server._serve) OFF for the whole
    # suite. It is read-only and harmless, but tests should never bind a port
    # implicitly; TestTraceEndpoint starts its own listener on an ephemeral
    # port, and the live integration test targets the running service by URL.
    monkeypatch.setenv("HARS_MEMORY_TRACE_HTTP_PORT", "0")
