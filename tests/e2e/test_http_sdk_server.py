from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from hars_memory.sdk import HarsMemoryAPIError, HarsMemoryClient


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_healthy(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, _ = process.communicate()
            raise AssertionError(f"server exited early ({process.returncode}):\n{stdout}")
        try:
            if httpx.get(f"{base_url}/health", timeout=0.25).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise AssertionError("server did not become healthy")


def _start_server(tmp_path: Path, port: int) -> tuple[subprocess.Popen[str], Path]:
    guard = tmp_path / "network_guard"
    guard.mkdir(exist_ok=True)
    sentinel = tmp_path / "outbound-network-attempted"
    loaded = tmp_path / "network-guard-loaded"
    (guard / "sitecustomize.py").write_text(
        "import os, socket\n"
        "from pathlib import Path\n"
        "Path(os.environ['HARS_E2E_GUARD_LOADED']).write_text('loaded')\n"
        "def blocked(*args, **kwargs):\n"
        "    Path(os.environ['HARS_E2E_NETWORK_SENTINEL']).write_text(repr((args, kwargs)))\n"
        "    raise RuntimeError('outbound network forbidden in corpus E2E')\n"
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
            "HARS_MEMORY_API_KEYS_JSON": json.dumps(
                {"tenant-a-key": "tenant-a", "tenant-b-key": "tenant-b"}
            ),
            "HARS_MEMORY_SERVICE_DATABASE_URL": f"sqlite:///{tmp_path / 'service.sqlite3'}",
            "HARS_MEMORY_ARTIFACT_STORE_URL": (tmp_path / "artifacts").resolve().as_uri(),
            "HARS_MEMORY_WORKER_SCRATCH_DIR": str(tmp_path / "scratch"),
            "HARS_MEMORY_WORKER_POLL_SECONDS": "0.02",
            "HARS_MEMORY_SERVICE_HOST": "127.0.0.1",
            "HARS_MEMORY_SERVICE_PORT": str(port),
            "HARS_MEMORY_EXTRACTOR_BASE_URL": "http://127.0.0.1:9/v1",
            "HARS_MEMORY_QUERY_BASE_URL": "http://127.0.0.1:9/v1",
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    executable = Path(sys.executable).parent / "hars-longterm-memory-server"
    process = subprocess.Popen(
        [str(executable)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _wait_healthy(f"http://127.0.0.1:{port}", process)
    assert loaded.read_text(encoding="utf-8") == "loaded"
    return process, sentinel


def _stop_server(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)


def test_real_server_sdk_create_extend_download_tenant_and_restart(tmp_path: Path) -> None:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    process, sentinel = _start_server(tmp_path, port)
    try:
        with HarsMemoryClient(base_url, "tenant-a-key") as client:
            first = client.create_index(
                {"alpha.md": "# Alpha\n\nExact API token CLOUD-42.\n"},
                idempotency_key="create-cloud-42",
            )
            duplicate = client.create_index(
                {"alpha.md": "# Alpha\n\nExact API token CLOUD-42.\n"},
                idempotency_key="create-cloud-42",
            )
            assert duplicate.id == first.id
            completed = client.wait_job(first.id, timeout=15, poll_interval=0.02)
            assert completed.status == "succeeded", completed.raw
            assert completed.index_id is not None
            terminal_retry = client.create_index(
                {"alpha.md": "# Alpha\n\nExact API token CLOUD-42.\n"},
                idempotency_key="create-cloud-42",
            )
            assert terminal_retry.id == first.id
            assert terminal_retry.status == "succeeded"

            v1 = client.get_latest_index(completed.index_id)
            assert v1.version == 1
            bundle_v1 = client.download_artifact(
                completed.index_id, tmp_path / "index-v1.tar.gz"
            )
            assert bundle_v1.stat().st_size > 0

            extended = client.extend_index(
                completed.index_id,
                {"beta.txt": "Second immutable version.\n"},
                expected_version=1,
                idempotency_key="extend-cloud-42",
            )
            extended_done = client.wait_job(extended.id, timeout=15, poll_interval=0.02)
            assert extended_done.status == "succeeded", extended_done.raw
            v2 = client.get_latest_index(completed.index_id)
            assert v2.version == 2
            assert v2.raw["manifest"]["document_count"] == 2

            with HarsMemoryClient(base_url, "tenant-b-key") as other:
                with pytest.raises(HarsMemoryAPIError) as denied:
                    other.get_latest_index(completed.index_id)
                assert denied.value.status_code == 404
    finally:
        _stop_server(process)

    assert not sentinel.exists(), sentinel.read_text(encoding="utf-8") if sentinel.exists() else ""

    restart_port = _free_port()
    restarted, restart_sentinel = _start_server(tmp_path, restart_port)
    try:
        with HarsMemoryClient(f"http://127.0.0.1:{restart_port}", "tenant-a-key") as client:
            latest = client.get_latest_index(completed.index_id)
            assert latest.version == 2
    finally:
        _stop_server(restarted)
    assert not restart_sentinel.exists()
