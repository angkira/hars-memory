"""Index engine adapters and the immutable service artifact contract.

The HTTP layer persists uploads before an engine sees them.  Engines always
work in a private scratch directory and emit one self-contained tar bundle;
they never mutate an already-published version.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Final, Mapping, Protocol

from hars_memory.corpus.build import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MANIFEST_FILENAME,
)
from hars_memory.ingest.change_detection import compute_fingerprint
from hars_memory.ingest.chunker import chunk_text
from hars_memory.retrieval.bm25_index import CHUNKS_FILENAME
from hars_memory.strategies import IndexStrategy, Scalar

BUNDLE_MANIFEST: Final[str] = "index-bundle.json"
BUNDLE_FORMAT: Final[str] = "hars-index-bundle.v1"
SOURCES_DIR: Final[str] = "sources"
INDEX_DIR: Final[str] = "index"
ALLOWED_SUFFIXES: Final[frozenset[str]] = frozenset({".md", ".txt", ".json", ".py"})


class EngineError(RuntimeError):
    """A build failed without publishing a partial index."""


class UnsafeBundleError(EngineError):
    """A source/index bundle contains an unsafe path or entry type."""


class JobCancelled(EngineError):
    """The durable job was cancelled while its engine was running."""


@dataclass(frozen=True, slots=True)
class EngineRequest:
    job_id: str
    tenant_id: str
    index_id: str
    operation: str
    input_path: Path
    output_path: Path
    workspace: Path
    base_artifact: Path | None = None
    version: int = 1
    strategy: Mapping[str, Scalar] = field(default_factory=dict)
    cancel_requested: Callable[[], bool] = field(default=lambda: False, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class EngineResult:
    artifact_path: Path
    manifest: dict[str, object]


class IndexEngine(Protocol):
    def build(self, request: EngineRequest) -> EngineResult: ...


def _request_strategy(request: EngineRequest, engine: str) -> IndexStrategy:
    values = dict(request.strategy)
    name = str(values.pop("name", "default"))
    return IndexStrategy(name=name, engine=engine, options=values)


def _safe_relative(name: str) -> Path:
    pure = PurePosixPath(name.replace("\\", "/"))
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise UnsafeBundleError(f"unsafe bundle path: {name!r}")
    return Path(*pure.parts)


def _extract_tar_safely(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            _safe_relative(member.name)
            if not (member.isdir() or member.isfile()):
                raise UnsafeBundleError(f"unsupported tar entry type: {member.name!r}")
        tar.extractall(destination, members=members, filter="data")


def _copy_regular_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise EngineError(f"source artifact is neither a tar file nor directory: {source}")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        _safe_relative(relative.as_posix())
        if path.is_symlink():
            raise UnsafeBundleError(f"symlinks are not accepted: {relative}")
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def _materialize_tree(source: Path, destination: Path) -> None:
    if source.is_file() and tarfile.is_tarfile(source):
        _extract_tar_safely(source, destination)
    else:
        _copy_regular_tree(source, destination)


def _find_sources_root(materialized: Path) -> Path:
    nested = materialized / SOURCES_DIR
    return nested if nested.is_dir() else materialized


def _merge_sources(request: EngineRequest) -> tuple[Path, Path]:
    """Return ``(sources, index)`` in a new workspace snapshot."""
    snapshot = request.workspace / "snapshot"
    sources = snapshot / SOURCES_DIR
    index = snapshot / INDEX_DIR
    sources.mkdir(parents=True, exist_ok=False)

    if request.base_artifact is not None:
        base = request.workspace / "base"
        _materialize_tree(request.base_artifact, base)
        base_sources = base / SOURCES_DIR
        if not base_sources.is_dir():
            raise EngineError("base index artifact has no sources/ snapshot")
        _copy_regular_tree(base_sources, sources)
        base_index = base / INDEX_DIR
        if base_index.is_dir():
            _copy_regular_tree(base_index, index)

    incoming = request.workspace / "incoming"
    _materialize_tree(request.input_path, incoming)
    incoming_sources = _find_sources_root(incoming)
    accepted = 0
    for path in sorted(incoming_sources.rglob("*")):
        if path.is_dir():
            continue
        if path.is_symlink():
            raise UnsafeBundleError(f"symlinks are not accepted: {path}")
        relative = path.relative_to(incoming_sources)
        _safe_relative(relative.as_posix())
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise EngineError(f"unsupported uploaded file: {relative}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise EngineError(f"uploaded file is not UTF-8 text: {relative}") from exc
        if not content.strip():
            raise EngineError(f"uploaded file is empty: {relative}")
        target = sources / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        accepted += 1
    if accepted == 0:
        raise EngineError("input artifact contains no accepted text files")
    return sources, index


def _source_id(relative: str) -> str:
    return f"upload:{hashlib.sha256(relative.encode('utf-8')).hexdigest()[:20]}"


def _build_corpus_index(
    sources: Path,
    index: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[int, int]:
    """Build the existing chunk-store contract with upload-stable identities."""
    index.mkdir(parents=True, exist_ok=True)
    chunks: dict[str, dict[str, object]] = {}
    documents: dict[str, dict[str, object]] = {}
    fingerprint_pairs: list[tuple[str, str]] = []
    for source in sorted(path for path in sources.rglob("*") if path.is_file()):
        relative = source.relative_to(sources).as_posix()
        content = source.read_text(encoding="utf-8")
        doc_id = _source_id(relative)
        content_hash = compute_fingerprint(content)
        ids: list[str] = []
        for chunk in chunk_text(content, doc_id, chunk_size, chunk_overlap):
            chunk_id = "chunk-" + hashlib.sha256(
                f"{doc_id}:{chunk.chunk_index}".encode("utf-8")
            ).hexdigest()[:16]
            ids.append(chunk_id)
            chunks[chunk_id] = {
                "content": chunk.text,
                "file_path": f"upload://{relative}",
                "chunk_index": chunk.chunk_index,
                "source_doc_id": doc_id,
            }
        documents[doc_id] = {
            "source_path": f"upload://{relative}",
            "file_stable_id": doc_id,
            "content_sha256": content_hash,
            "size_bytes": source.stat().st_size,
            "mtime": 0.0,
            "chunk_count": len(ids),
            "chunk_ids": ids,
        }
        fingerprint_pairs.append((relative, content_hash))

    chunk_json = json.dumps(chunks, sort_keys=True, indent=2)
    corpus_fingerprint = hashlib.sha256(
        "\n".join(f"{p}:{h}" for p, h in fingerprint_pairs).encode("utf-8")
    ).hexdigest()
    manifest = {
        "build": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "include_globs": sorted(f"**/*{suffix}" for suffix in ALLOWED_SUFFIXES),
            "exclude_globs": None,
            "tool_version": "service-1.0.0",
            "corpus_fingerprint": corpus_fingerprint,
            "chunk_store_sha256": hashlib.sha256(chunk_json.encode("utf-8")).hexdigest(),
            "chunk_id_scheme": 'chunk-{sha256(f"{doc_id}:{chunk_index}")[:16]}',
            "paths": ["upload://"],
        },
        "documents": documents,
    }
    (index / CHUNKS_FILENAME).write_text(chunk_json, encoding="utf-8")
    (index / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    return len(documents), len(chunks)


def _count_lightrag(index: Path, sources: Path) -> tuple[int, int]:
    document_count = sum(1 for path in sources.rglob("*") if path.is_file())
    chunks_path = index / CHUNKS_FILENAME
    if not chunks_path.is_file():
        raise EngineError(f"LightRAG completed without {CHUNKS_FILENAME}")
    try:
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineError(f"invalid LightRAG chunk store: {exc}") from exc
    return document_count, len(chunks)


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _write_bundle(
    request: EngineRequest,
    snapshot: Path,
    *,
    engine: str,
    document_count: int,
    chunk_count: int,
    external_state: dict[str, object] | None = None,
    strategy: IndexStrategy | None = None,
) -> EngineResult:
    external_state = external_state or {}
    manifest: dict[str, object] = {
        "format": BUNDLE_FORMAT,
        "engine": engine,
        "operation": request.operation,
        "index_id": request.index_id,
        "version": request.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_count": document_count,
        "chunk_count": chunk_count,
        "content_sha256": _tree_sha256(snapshot),
        "portable": not bool(external_state),
        "external_state": external_state,
        "index_strategy": strategy.to_dict() if strategy is not None else None,
        "index_strategy_sha256": strategy.fingerprint if strategy is not None else None,
    }
    (snapshot / BUNDLE_MANIFEST).write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    with request.output_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as tar:
                for path in sorted(snapshot.rglob("*")):
                    relative = path.relative_to(snapshot).as_posix()
                    info = tar.gettarinfo(str(path), arcname=relative)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    if path.is_file():
                        with path.open("rb") as handle:
                            tar.addfile(info, handle)
                    elif path.is_dir():
                        tar.addfile(info)
    manifest["artifact_sha256"] = hashlib.sha256(request.output_path.read_bytes()).hexdigest()
    manifest["artifact_size_bytes"] = request.output_path.stat().st_size
    return EngineResult(request.output_path, manifest)


class CorpusEngine:
    """CPU-only engine used by CI and local/no-LLM deployments."""

    name = "corpus"

    def build(self, request: EngineRequest) -> EngineResult:
        if request.cancel_requested():
            raise JobCancelled("job cancelled before corpus build")
        sources, index = _merge_sources(request)
        strategy = _request_strategy(request, self.name)
        document_count, chunk_count = _build_corpus_index(
            sources,
            index,
            chunk_size=int(strategy.options.get("chunk_size", DEFAULT_CHUNK_SIZE)),
            chunk_overlap=int(
                strategy.options.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
            ),
        )
        if request.cancel_requested():
            raise JobCancelled("job cancelled before artifact publication")
        return _write_bundle(
            request,
            sources.parent,
            engine=self.name,
            document_count=document_count,
            chunk_count=chunk_count,
            strategy=strategy,
        )


class LightRAGEngine:
    """Production adapter around the existing, independently runnable indexer."""

    name = "lightrag"

    def __init__(self, *, timeout_seconds: float = 24 * 60 * 60, poll_seconds: float = 0.25):
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds

    def build(self, request: EngineRequest) -> EngineResult:
        if request.cancel_requested():
            raise JobCancelled("job cancelled before LightRAG build")
        sources, index = _merge_sources(request)
        strategy = _request_strategy(request, self.name)
        env = os.environ.copy()
        env.update(strategy.environment())
        vector_storage = env.get("HARS_MEMORY_VECTOR_STORAGE", "NanoVectorDBStorage")
        external_state: dict[str, object] = {}
        if "qdrant" in vector_storage.casefold():
            # Qdrant point IDs are collection-global, so sharing a collection
            # prefix between immutable versions would make an extend mutate
            # the old version's remote state. Give every version its own
            # collections/workspace and rebuild from the complete source
            # snapshot. This costs extraction time but preserves the service's
            # strongest contract: the last published version never changes.
            base_prefix = env.get("HARS_MEMORY_QDRANT_COLLECTION_PREFIX", "").strip()
            if not base_prefix:
                raise EngineError(
                    "HARS_MEMORY_QDRANT_COLLECTION_PREFIX is required for "
                    "Qdrant-backed service jobs"
                )
            # Scope by durable job ID, not predicted version number: two
            # concurrent extend candidates can both target vN, but must never
            # write into each other's collections before one wins publication.
            suffix = hashlib.sha256(
                f"{request.tenant_id}:{request.index_id}:{request.job_id}".encode("utf-8")
            ).hexdigest()[:12]
            base_workspace = env.get(
                "HARS_MEMORY_QDRANT_COLLECTION", "hars_longterm_memory"
            ).strip()
            workspace = f"{base_workspace}_{suffix}"
            env["HARS_MEMORY_QDRANT_COLLECTION"] = workspace
            # lightrag_init only derives this when it is absent. Explicitly
            # override a parent process's value so the child cannot silently
            # write to a different workspace than the manifest reports.
            env["QDRANT_WORKSPACE"] = workspace
            if index.exists():
                shutil.rmtree(index)
            external_state = {
                "kind": "qdrant",
                "workspace": workspace,
                "included_in_artifact": False,
                "immutable_version_scope": True,
            }
        index.mkdir(parents=True, exist_ok=True)
        env["HARS_MEMORY_INDEX_DIR"] = str(index)
        command = [
            sys.executable,
            "-m",
            "hars_memory.server.index",
            "--paths",
            str(sources),
            "--refresh-changed",
        ]
        log_path = request.workspace / "lightrag-indexer.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=request.workspace,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            started = time.monotonic()
            while process.poll() is None:
                if request.cancel_requested():
                    _terminate_process(process)
                    raise JobCancelled("job cancelled during LightRAG build")
                if time.monotonic() - started > self.timeout_seconds:
                    _terminate_process(process)
                    raise EngineError(
                        f"LightRAG build timed out after {self.timeout_seconds:g}s"
                    )
                time.sleep(self.poll_seconds)
        output = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        if process.returncode != 0:
            raise EngineError(
                f"LightRAG indexer exited {process.returncode}: {output[-4000:]}"
            )
        document_count, chunk_count = _count_lightrag(index, sources)
        return _write_bundle(
            request,
            sources.parent,
            engine=self.name,
            document_count=document_count,
            chunk_count=chunk_count,
            external_state=external_state,
            strategy=strategy,
        )


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Stop an indexer and prove it is dead before scratch cleanup."""
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_MANIFEST",
    "CorpusEngine",
    "EngineError",
    "EngineRequest",
    "EngineResult",
    "IndexEngine",
    "JobCancelled",
    "LightRAGEngine",
    "UnsafeBundleError",
]
