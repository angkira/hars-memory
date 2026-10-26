from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from hars_memory.service.artifacts import LocalArtifactStore
from hars_memory.service.database import LeaseLostError, ServiceDatabase
from hars_memory.service.engines import EngineRequest, EngineResult
from hars_memory.service.models import JobKind, JobStatus
from hars_memory.service.worker import DurableWorker, WorkerConfig


@dataclass
class Job:
    id: str = "job-1"
    tenant_id: str = "tenant-1"
    operation: str = "create"
    engine: str = "fake"
    index_id: str = "index-1"
    input_uri: str = "input://one"
    base_version: int | None = None
    cancel_requested: bool = False


class FakeRepository:
    def __init__(self, jobs: list[Job], latest: object | None = None) -> None:
        self.jobs = jobs
        self.latest = latest
        self.requeued = 0
        self.succeeded: list[tuple[object, ...]] = []
        self.failed: list[tuple[str, str]] = []

    def requeue_running_jobs(self) -> int:
        self.requeued += 1
        return 2

    def claim_next_job(self) -> Job | None:
        return self.jobs.pop(0) if self.jobs else None

    def get_latest_index_version(self, tenant_id: str, index_id: str) -> object | None:
        return self.latest

    def mark_job_succeeded(self, *args: object) -> None:
        self.succeeded.append(args)

    def mark_job_failed(self, job_id: str, error: str) -> None:
        self.failed.append((job_id, error))


class FakeStore:
    def __init__(self, source: Path, base: Path | None = None) -> None:
        self.source = source
        self.base = base
        self.published: list[tuple[str, int, bytes]] = []

    def materialize_input(self, uri: str, destination: Path) -> Path:
        shutil.copytree(self.source, destination)
        return destination

    def materialize_result(self, uri: str, destination: Path) -> Path:
        assert self.base is not None
        shutil.copyfile(self.base, destination)
        return destination

    def put_result(self, index_id: str, version: int, path: Path) -> str:
        self.published.append((index_id, version, path.read_bytes()))
        return f"result://{index_id}/{version}"


class FakeEngine:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[EngineRequest] = []

    def build(self, request: EngineRequest) -> EngineResult:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("engine exploded")
        request.output_path.write_bytes(b"immutable-result")
        return EngineResult(request.output_path, {"document_count": 1, "chunk_count": 1})


def test_worker_recovers_builds_stores_then_commits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.md").write_text("one")
    repository = FakeRepository([Job()])
    store = FakeStore(source)
    engine = FakeEngine()
    worker = DurableWorker(
        repository,
        store,
        engines={"fake": engine},
        config=WorkerConfig(scratch_root=tmp_path / "scratch"),
    )

    assert worker.run_once() is True
    assert repository.requeued == 1
    assert store.published == [("index-1", 1, b"immutable-result")]
    assert repository.succeeded[0][:3] == ("job-1", 1, "result://index-1/1")
    assert not repository.failed
    assert worker.run_once() is False


def test_worker_extend_uses_latest_immutable_version(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "two.md").write_text("two")
    base = tmp_path / "base.tar.gz"
    base.write_bytes(b"base")
    latest = {"version": 3, "artifact_uri": "result://index-1/3"}
    repository = FakeRepository(
        [Job(operation="extend", base_version=3)], latest=latest
    )
    engine = FakeEngine()
    worker = DurableWorker(
        repository,
        FakeStore(source, base),
        engines={"fake": engine},
        config=WorkerConfig(scratch_root=tmp_path / "scratch", recover_on_start=False),
    )

    worker.run_once()

    assert engine.requests[0].version == 4
    assert engine.requests[0].base_artifact is not None
    assert repository.succeeded[0][1] == 4


def test_worker_failure_does_not_publish_version(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.md").write_text("one")
    repository = FakeRepository([Job()])
    store = FakeStore(source)
    worker = DurableWorker(
        repository,
        store,
        engines={"fake": FakeEngine(fail=True)},
        config=WorkerConfig(scratch_root=tmp_path / "scratch", recover_on_start=False),
    )

    worker.run_once()

    assert not store.published
    assert not repository.succeeded
    assert repository.failed == [("job-1", "engine exploded")]


def test_worker_rejects_stale_extend_base(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.md").write_text("one")
    repository = FakeRepository(
        [Job(operation="extend", base_version=2)],
        latest={"version": 3, "artifact_uri": "result://index-1/3"},
    )
    store = FakeStore(source)
    worker = DurableWorker(
        repository,
        store,
        engines={"fake": FakeEngine()},
        config=WorkerConfig(scratch_root=tmp_path / "scratch", recover_on_start=False),
    )

    worker.run_once()

    assert not store.published
    assert "base version conflict" in repository.failed[0][1]


class _LeaseRepository(FakeRepository):
    def __init__(self, jobs: list[Job]) -> None:
        super().__init__(jobs)
        self.renewals = 0

    def renew_lease(self, **_kwargs: object) -> None:
        self.renewals += 1

    def publish_version(self, **_kwargs: object) -> None:
        raise LeaseLostError("reclaimed")


class _SlowStore(FakeStore):
    def materialize_input(self, uri: str, destination: Path) -> Path:
        time.sleep(0.05)
        return super().materialize_input(uri, destination)


def test_heartbeat_covers_input_materialization_and_lease_loss_is_not_terminalized(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.md").write_text("one")
    repository = _LeaseRepository([Job(id="job-1"), Job(id="job-2")])
    worker = DurableWorker(
        repository,
        _SlowStore(source),
        engines={"fake": FakeEngine()},
        config=WorkerConfig(
            scratch_root=tmp_path / "scratch",
            recover_on_start=False,
            heartbeat_seconds=0.01,
        ),
    )

    assert worker.run_once() is True
    assert repository.renewals >= 1
    assert not repository.failed
    assert worker.run_once() is True
    assert not repository.failed


def test_failed_extend_preserves_last_published_version(tmp_path: Path) -> None:
    database = ServiceDatabase(f"sqlite:///{tmp_path / 'service.sqlite3'}")
    database.create_schema()
    store = LocalArtifactStore(tmp_path / "artifacts")
    input_file = tmp_path / "input.tar.gz"
    input_file.write_bytes(b"input")
    input_ref = store.put_file("inputs/one.tar.gz", input_file)
    base_file = tmp_path / "base.tar.gz"
    base_file.write_bytes(b"last-good-version")
    base_ref = store.put_file("indexes/base.tar.gz", base_file)
    index = database.create_index(tenant_id="tenant-1", name="docs")
    initial = database.create_job(
        tenant_id="tenant-1",
        index_id=index.id,
        kind=JobKind.CREATE,
        idempotency_key="create",
        request={"engine": "fake"},
        input_artifact_uri=input_ref.uri,
        input_artifact_sha256=input_ref.sha256,
    )
    database.claim_next_job(worker_id="publisher")
    v1 = database.publish_version(
        job_id=initial.id,
        worker_id="publisher",
        artifact_uri=base_ref.uri,
        artifact_sha256=base_ref.sha256,
        artifact_size_bytes=base_ref.size_bytes,
        manifest={"document_count": 1},
    )
    extension = database.create_job(
        tenant_id="tenant-1",
        index_id=index.id,
        kind=JobKind.EXTEND,
        idempotency_key="extend",
        request={"engine": "fake"},
        input_artifact_uri=input_ref.uri,
        input_artifact_sha256=input_ref.sha256,
        base_version_id=v1.id,
    )
    worker = DurableWorker(
        database,
        store,
        engines={"fake": FakeEngine(fail=True)},
        config=WorkerConfig(scratch_root=tmp_path / "scratch", recover_on_start=False),
    )

    assert worker.run_once() is True
    current = database.get_index(tenant_id="tenant-1", index_id=index.id)
    failed = database.get_job(tenant_id="tenant-1", job_id=extension.id)
    assert current is not None and current.active_version_id == v1.id
    assert failed is not None and failed.status == JobStatus.FAILED
    assert database.get_version_by_number(
        tenant_id="tenant-1", index_id=index.id, version_number=2
    ) is None
    database.close()
