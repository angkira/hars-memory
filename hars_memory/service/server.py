"""Environment-wired entrypoint for the index-job HTTP service."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Final

from fastapi import FastAPI

from hars_memory.service.api import create_app
from hars_memory.service.artifacts import create_artifact_store
from hars_memory.service.auth import APIKeyAuthenticator
from hars_memory.service.database import ServiceDatabase
from hars_memory.service.worker import DurableWorker, WorkerConfig

DATABASE_URL_ENV: Final[str] = "HARS_MEMORY_SERVICE_DATABASE_URL"
ARTIFACT_STORE_URL_ENV: Final[str] = "HARS_MEMORY_ARTIFACT_STORE_URL"


class ThreadedWorker:
    """Lifespan adapter for the worker's blocking polling loop."""

    def __init__(self, worker: DurableWorker) -> None:
        self._worker = worker
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._worker.run_forever, name="hars-index-worker", daemon=True
        )
        self._thread.start()

    def wake(self) -> None:
        # The worker polls at a short bounded interval; this hook deliberately
        # keeps the API independent of its internal synchronization primitive.
        return

    def stop(self) -> None:
        self._worker.stop()
        if self._thread is not None:
            self._thread.join(timeout=10)


def create_default_app() -> FastAPI:
    database_url = os.environ.get(
        DATABASE_URL_ENV, "sqlite:///./hars-memory-service.sqlite3"
    )
    artifact_store_url = os.environ.get(
        ARTIFACT_STORE_URL_ENV,
        Path("./hars-memory-artifacts").resolve().as_uri(),
    )
    repository = ServiceDatabase(database_url)
    repository.create_schema()
    artifact_store = create_artifact_store(
        artifact_store_url,
        s3_endpoint_url=os.environ.get("HARS_MEMORY_S3_ENDPOINT_URL"),
        s3_region_name=os.environ.get("HARS_MEMORY_S3_REGION"),
    )
    worker = ThreadedWorker(
        DurableWorker(
            repository,  # type: ignore[arg-type] -- concrete repository implements worker contract
            artifact_store,  # type: ignore[arg-type] -- aliases bridge the artifact protocol
            config=WorkerConfig(
                scratch_root=(
                    Path(value)
                    if (value := os.environ.get("HARS_MEMORY_WORKER_SCRATCH_DIR"))
                    else None
                ),
                poll_seconds=float(os.environ.get("HARS_MEMORY_WORKER_POLL_SECONDS", "1")),
            ),
        )
    )
    return create_app(
        repository,
        artifact_store,
        worker,
        APIKeyAuthenticator.from_env(),
    )


def main() -> None:
    import uvicorn

    host = os.environ.get("HARS_MEMORY_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("HARS_MEMORY_SERVICE_PORT", "8787"))
    uvicorn.run(create_default_app(), host=host, port=port)


__all__ = ["ThreadedWorker", "create_default_app", "main"]
