from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _offline_env(guard: Path, sentinel: Path, loaded: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HARS_MEMORY_EMBED_LOCAL_FILES_ONLY": "1",
            "HARS_MEMORY_EXTRACTOR_BASE_URL": "http://127.0.0.1:9/v1",
            "HARS_MEMORY_QUERY_BASE_URL": "http://127.0.0.1:9/v1",
            "PYTHONPATH": str(guard),
            "HARS_E2E_NETWORK_SENTINEL": str(sentinel),
            "HARS_E2E_GUARD_LOADED": str(loaded),
        }
    )
    return env


def _run(
    *args: str, cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    memory = Path(sys.executable).parent / "memory"
    return subprocess.run(
        [str(memory), *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )


def test_installed_cli_build_query_and_incremental_replace(tmp_path: Path) -> None:
    source = tmp_path / "source"
    index = tmp_path / "index"
    source.mkdir()
    guard = tmp_path / "network_guard"
    guard.mkdir()
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
    env = _offline_env(guard, sentinel, loaded)
    (source / "alpha.md").write_text(
        "# Alpha\n\nExact offline token ZXQ-741.\n", encoding="utf-8"
    )
    (source / "beta.txt").write_text("Original beta.\n", encoding="utf-8")

    built = _run(
        "build", "--paths", str(source), "--index-dir", str(index), cwd=tmp_path, env=env
    )
    assert "Build complete: 2 documents" in built.stdout
    assert "added=2 changed=0 unchanged=0 deleted=0" in built.stdout

    status = _run("status", "--index-dir", str(index), cwd=tmp_path, env=env)
    assert "documents:          2" in status.stdout

    query = _run(
        "query",
        "--index-dir",
        str(index),
        "--question",
        "ZXQ-741",
        "--mode",
        "sparse",
        "--top-k",
        "3",
        cwd=tmp_path,
        env=env,
    )
    assert "alpha.md" in query.stdout
    assert "ZXQ-741" in query.stdout

    (source / "alpha.md").write_text(
        "# Alpha changed\n\nExact offline token ZXQ-741.\n", encoding="utf-8"
    )
    (source / "beta.txt").unlink()
    (source / "gamma.json").write_text('{"new": "document"}\n', encoding="utf-8")
    rebuilt = _run(
        "build", "--paths", str(source), "--index-dir", str(index), cwd=tmp_path, env=env
    )
    assert "added=1 changed=1 unchanged=0 deleted=1" in rebuilt.stdout

    chunks = json.loads((index / "kv_store_text_chunks.json").read_text(encoding="utf-8"))
    assert all("beta.txt" not in str(entry["file_path"]) for entry in chunks.values())
    assert loaded.read_text(encoding="utf-8") == "loaded"
    assert not sentinel.exists(), sentinel.read_text(encoding="utf-8") if sentinel.exists() else ""
