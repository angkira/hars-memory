"""Opt-in real-model journey. Never runs unless HARS_RUN_LIVE_LLM_E2E=1."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tarfile
import time
from functools import partial
from pathlib import Path

import anyio
import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from hars_memory.sdk import HarsMemoryClient
from hars_memory.strategies import IndexStrategy, SearchStrategy, load_strategy_matrix

pytestmark = pytest.mark.live_llm


def _enabled() -> bool:
    return os.environ.get("HARS_RUN_LIVE_LLM_E2E", "").lower() in {"1", "true", "yes"}


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _require_model_endpoint(base_url: str, model: str) -> None:
    headers: dict[str, str] = {}
    api_key = os.environ.get("HARS_MEMORY_LLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/models", headers=headers, timeout=10
        )
        response.raise_for_status()
        model_ids = {
            str(item.get("id"))
            for item in response.json().get("data", [])
            if isinstance(item, dict)
        }
    except Exception as exc:  # noqa: BLE001 -- operator-facing preflight
        pytest.fail(f"LLM endpoint preflight failed for {base_url}: {exc}")
    if model not in model_ids:
        pytest.fail(f"model {model!r} is not served by {base_url}; available={sorted(model_ids)}")


def _wait_healthy(base_url: str, process: subprocess.Popen[object], log_path: Path) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"index service exited early; see {log_path}")
        try:
            if httpx.get(f"{base_url}/health", timeout=0.5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    pytest.fail(f"index service did not become healthy; see {log_path}")


def _extract_bundle(bundle: Path, destination: Path) -> Path:
    destination.mkdir()
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(destination, filter="data")
    return destination / "index"


async def _recall(
    index_dir: Path,
    env: dict[str, str],
    search: SearchStrategy,
    *,
    context_only: bool,
) -> dict[str, object]:
    executable = Path(sys.executable).parent / "hars-longterm-memory-mcp"
    mcp_env = dict(env)
    mcp_env.update(
        {
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(index_dir.parent / "staging"),
            "HARS_MEMORY_BM25_CACHE_DIR": str(index_dir.parent / "bm25"),
            "HARS_MEMORY_HYBRID_ENABLED": "0",
            "HARS_MEMORY_RIPGREP_CHANNEL": "0",
        }
    )
    params = StdioServerParameters(
        command=str(executable), env=mcp_env, cwd=str(index_dir.parent)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.call_tool(
                "memory_recall",
                {
                    "question": (
                        "After the rollback, which queue must Aurora Relay use and "
                        "which verification gate must pass?"
                    ),
                    "mode": search.mode,
                    "top_k": search.top_k,
                    "fetch_top_k": search.chunk_top_k or search.top_k,
                    "ll_keywords": ["Aurora Relay", "GREEN-COMET-52", "Atlas Queue"],
                    "context_only": context_only,
                },
            )
            return json.loads(response.content[0].text)


@pytest.mark.skipif(not _enabled(), reason="set HARS_RUN_LIVE_LLM_E2E=1")
def test_live_lightrag_sdk_service_versions_and_mcp_recall(tmp_path: Path) -> None:
    required = {
        "HARS_MEMORY_EXTRACTOR_BASE_URL": os.environ.get("HARS_MEMORY_EXTRACTOR_BASE_URL"),
        "HARS_MEMORY_EXTRACTOR_MODEL": os.environ.get("HARS_MEMORY_EXTRACTOR_MODEL"),
        "HARS_MEMORY_QUERY_BASE_URL": os.environ.get("HARS_MEMORY_QUERY_BASE_URL"),
        "HARS_MEMORY_QUERY_MODEL": os.environ.get("HARS_MEMORY_QUERY_MODEL"),
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        pytest.fail(f"missing live LLM configuration: {', '.join(missing)}")
    _require_model_endpoint(
        str(required["HARS_MEMORY_EXTRACTOR_BASE_URL"]),
        str(required["HARS_MEMORY_EXTRACTOR_MODEL"]),
    )
    _require_model_endpoint(
        str(required["HARS_MEMORY_QUERY_BASE_URL"]),
        str(required["HARS_MEMORY_QUERY_MODEL"]),
    )

    fixture = Path(__file__).parent / "fixtures" / "live_llm"
    matrix = load_strategy_matrix(fixture / "strategy-matrix.yaml")
    strategy = next(item for item in matrix.index_strategies if item.engine == "lightrag")
    search = next(item for item in matrix.search_strategies if item.name == "lightrag-naive-k3")
    assert isinstance(strategy, IndexStrategy)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "HARS_MEMORY_API_KEYS_JSON": json.dumps({"live-test-key": "live-test-tenant"}),
            "HARS_MEMORY_SERVICE_DATABASE_URL": f"sqlite:///{tmp_path / 'jobs.sqlite3'}",
            "HARS_MEMORY_ARTIFACT_STORE_URL": (tmp_path / "artifacts").resolve().as_uri(),
            "HARS_MEMORY_WORKER_SCRATCH_DIR": str(tmp_path / "scratch"),
            "HARS_MEMORY_WORKER_POLL_SECONDS": "0.05",
            "HARS_MEMORY_SERVICE_HOST": "127.0.0.1",
            "HARS_MEMORY_SERVICE_PORT": str(port),
            "HARS_MEMORY_VECTOR_STORAGE": "NanoVectorDBStorage",
            "HARS_MEMORY_GRAPH_STORAGE": "NetworkXStorage",
            "HARS_MEMORY_EMBED_LOCAL_FILES_ONLY": "1",
        }
    )
    env.pop("HARS_MEMORY_GPU_GUARD_SCRIPT_PATH", None)
    log_path = tmp_path / "service.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(Path(sys.executable).parent / "hars-longterm-memory-server")],
            cwd=tmp_path,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_healthy(base_url, process, log_path)
            timeout = float(os.environ.get("HARS_LIVE_LLM_JOB_TIMEOUT", "1200"))
            corpus = {
                path.name: path.read_text(encoding="utf-8")
                for path in sorted((fixture / "corpus").glob("*.md"))
            }
            with HarsMemoryClient(base_url, "live-test-key", timeout=60) as client:
                created = client.create_index(
                    corpus,
                    engine="lightrag",
                    strategy=strategy,
                    idempotency_key="live-small-v1",
                )
                v1_job = client.wait_job(created.id, timeout=timeout, poll_interval=1)
                assert v1_job.status == "succeeded", v1_job.raw
                assert v1_job.index_id is not None
                v1 = client.get_latest_index(v1_job.index_id)
                assert v1.version == 1
                assert v1.raw["manifest"]["index_strategy_sha256"] == strategy.fingerprint
                bundle_v1 = client.download_artifact(
                    v1_job.index_id, tmp_path / "v1.tar.gz"
                )

                rollback = (fixture / "extension" / "rollback.md").read_text(
                    encoding="utf-8"
                )
                extended = client.extend_index(
                    v1_job.index_id,
                    {"rollback.md": rollback},
                    engine="lightrag",
                    strategy=strategy,
                    expected_version=1,
                    idempotency_key="live-small-v2",
                )
                v2_job = client.wait_job(extended.id, timeout=timeout, poll_interval=1)
                assert v2_job.status == "succeeded", v2_job.raw
                v2 = client.get_latest_index(v1_job.index_id)
                assert v2.version == 2
                bundle_v2 = client.download_artifact(
                    v1_job.index_id, tmp_path / "v2.tar.gz"
                )
                old_again = client.download_artifact(
                    v1_job.index_id, tmp_path / "v1-again.tar.gz", version=1
                )
                assert old_again.read_bytes() == bundle_v1.read_bytes()

            v1_index = _extract_bundle(bundle_v1, tmp_path / "unpacked-v1")
            v2_index = _extract_bundle(bundle_v2, tmp_path / "unpacked-v2")
            v1_context = anyio.run(
                partial(_recall, v1_index, env, search, context_only=True)
            )
            assert v1_context["ok"] is True
            assert "GREEN-COMET-52" not in str(v1_context.get("context", ""))
            v2_context = anyio.run(
                partial(_recall, v2_index, env, search, context_only=True)
            )
            assert v2_context["ok"] is True
            assert "GREEN-COMET-52" in str(v2_context.get("context", ""))
            answer = anyio.run(
                partial(_recall, v2_index, env, search, context_only=False)
            )
            assert answer["ok"] is True
            answer_text = str(answer.get("answer", ""))
            assert "GREEN-COMET-52" in answer_text
            assert "Atlas Queue" in answer_text
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
