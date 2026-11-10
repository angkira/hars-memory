"""Terminal UI for hars-longterm-memory (optional ``[tui]`` extra).

This package is a pure out-of-process client: it talks to a running HARS
service over HTTP (health, MCP tools, the recall-trace SSE listener) and reads
local files for things that have no API yet (session directories, eval run
directories). It never imports ``hars_memory.mcp_server``, LightRAG or torch.

Importing :mod:`hars_memory.tui` itself does not import Textual; only
``hars_memory.tui.app`` and the screens do, so the console entry point can print
a clean "install the extra" message when Textual is missing.
"""

from __future__ import annotations

__all__ = ["__doc__"]
