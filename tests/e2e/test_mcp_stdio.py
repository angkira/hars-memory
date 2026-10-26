from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_real_mcp_stdio_transport_is_llm_and_network_free(tmp_path: Path) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    guard = tmp_path / "network_guard"
    source.mkdir()
    staging.mkdir()
    guard.mkdir()
    (source / "fixture.md").write_text("# Offline MCP\n\nToken MCP-77.\n", encoding="utf-8")
    sentinel = tmp_path / "network-attempted"
    loaded = tmp_path / "network-guard-loaded"
    (guard / "sitecustomize.py").write_text(
        "import os, socket\n"
        "from pathlib import Path\n"
        "Path(os.environ['HARS_E2E_GUARD_LOADED']).write_text('loaded')\n"
        "def blocked(*args, **kwargs):\n"
        "    Path(os.environ['HARS_E2E_NETWORK_SENTINEL']).write_text(repr((args, kwargs)))\n"
        "    raise RuntimeError('network forbidden in offline E2E')\n"
        "socket.socket.connect = blocked\n"
        "socket.socket.connect_ex = blocked\n"
        "socket.create_connection = blocked\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(guard),
            "HARS_E2E_NETWORK_SENTINEL": str(sentinel),
            "HARS_E2E_GUARD_LOADED": str(loaded),
            "HARS_MEMORY_INDEX_DIR": str(tmp_path / "graph-index"),
            "HARS_MEMORY_STAGING_DIR": str(staging),
            "HARS_MEMORY_VECTOR_STORAGE": "NanoVectorDBStorage",
            "HARS_MEMORY_GRAPH_STORAGE": "NetworkXStorage",
            "HARS_MEMORY_HYBRID_ENABLED": "0",
            "HARS_MEMORY_EMBED_LOCAL_FILES_ONLY": "1",
            "HARS_MEMORY_EXTRACTOR_BASE_URL": "http://127.0.0.1:9/v1",
            "HARS_MEMORY_QUERY_BASE_URL": "http://127.0.0.1:9/v1",
            "HARS_MEMORY_LOG_FILE": str(tmp_path / "memory.log"),
            "HARS_MEMORY_EVENT_LOG_FILE": str(tmp_path / "events.jsonl"),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    async def journey() -> None:
        executable = Path(sys.executable).parent / "hars-longterm-memory-mcp"
        params = StdioServerParameters(command=str(executable), env=env, cwd=str(tmp_path))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                assert initialized.serverInfo.name == "hars-longterm-memory"

                names = {tool.name for tool in (await session.list_tools()).tools}
                assert names == {
                    "memory_recall",
                    "memory_remember",
                    "memory_entities",
                    "memory_inspect_entity",
                    "memory_related",
                    "memory_status",
                    "memory_upsert_document",
                    "memory_delete_document",
                    "memory_sync_status",
                    "memory_list_projects",
                    "memory_consolidate",
                    "memory_forget",
                    "memory_compact",
                    "memory_merge_session",
                    "memory_session_status",
                }

                status = json.loads(
                    (await session.call_tool("memory_status", {})).content[0].text
                )
                assert status["ok"] is True
                assert status["index_exists"] is False

                remembered = json.loads(
                    (
                        await session.call_tool(
                            "memory_remember",
                            {
                                "title": "offline-mcp-e2e",
                                "content": "Written through real MCP stdio. Unicode: grüß.",
                                "tags": ["e2e", "offline"],
                                "importance": "low",
                            },
                        )
                    ).content[0].text
                )
                assert remembered["ok"] is True
                assert Path(remembered["saved"]).is_relative_to(staging)

                consolidated = json.loads(
                    (
                        await session.call_tool(
                            "memory_consolidate",
                            {"paths": [str(source)], "dry_run": True},
                        )
                    ).content[0].text
                )
                assert consolidated["ok"] is True
                assert consolidated["returncode"] == 0
                assert "1 documents accepted" in consolidated["stderr"]
                assert "no LLM extraction performed" in consolidated["stderr"]

    anyio.run(journey)
    assert loaded.read_text(encoding="utf-8") == "loaded"
    assert not sentinel.exists(), sentinel.read_text(encoding="utf-8") if sentinel.exists() else ""
