"""Build and query an isolated dense embedding sidecar for a LightRAG index."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

DEFAULT_SIDECAR_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)
CHUNKS_FILENAME = "kv_store_text_chunks.json"


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_.-") or "model"


def _normalise(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _load_model(model: str, device: str, max_seq_length: int) -> Any:
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    local_files_only = os.environ.get("HF_HUB_OFFLINE") is not None
    encoder = SentenceTransformer(model, device=device, local_files_only=local_files_only)
    encoder.max_seq_length = max_seq_length
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    return encoder


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_sidecar(
    index_dir: str | Path,
    model: str = DEFAULT_SIDECAR_MODEL,
    out_dir: str | Path | None = None,
    max_seq_length: int = 512,
    batch_size: int = 16,
    device: str = "auto",
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    *,
    embedder: Any | None = None,
    allow_external_out_dir: bool = False,
) -> dict[str, Any]:
    """Embed chunk content to ``dense_sidecar/<model-slug>`` without touching index files.

    ``embedder`` is an injectable SentenceTransformer-compatible object,
    primarily for deterministic CPU-only tests.
    """
    index_path = Path(index_dir).expanduser().resolve()
    chunks_path = index_path / CHUNKS_FILENAME
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunk store not found: {chunks_path}")
    if max_seq_length <= 0 or batch_size <= 0:
        raise ValueError("max_seq_length and batch_size must be positive")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    output_path = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else (index_path / "dense_sidecar" / _slug(model)).resolve()
    )
    if not allow_external_out_dir and not output_path.is_relative_to(index_path):
        raise ValueError(f"out_dir must be inside index_dir: {output_path}")

    source_bytes = chunks_path.read_bytes()
    raw_chunks = json.loads(source_bytes)
    if not isinstance(raw_chunks, dict):
        raise ValueError(f"Expected chunk-ID mapping in {chunks_path}")
    for chunk_id, record in raw_chunks.items():
        if not isinstance(record, dict) or not isinstance(record.get("content"), str):
            raise ValueError(f"Chunk {chunk_id!r} has no string content")
    chunk_ids = sorted(raw_chunks, key=lambda chunk_id: (len(raw_chunks[chunk_id]["content"]), str(chunk_id)))
    if limit is not None:
        chunk_ids = chunk_ids[:limit]

    encoder = embedder if embedder is not None else _load_model(model, device, max_seq_length)
    output_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    batches: list[np.ndarray] = []
    dimension: int | None = None
    for start in range(0, len(chunk_ids), batch_size):
        ids = chunk_ids[start : start + batch_size]
        texts = [raw_chunks[chunk_id]["content"] for chunk_id in ids]
        vectors = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        matrix = _normalise(np.asarray(vectors, dtype=np.float32))
        if matrix.shape[0] != len(ids):
            raise ValueError("embedder returned a different number of vectors than input texts")
        if dimension is not None and matrix.shape[1] != dimension:
            raise ValueError("embedder returned inconsistent vector dimensions")
        dimension = matrix.shape[1]
        batches.append(matrix.astype(np.float16))
        if progress is not None:
            progress(start + len(ids), len(chunk_ids))

    vecs = np.concatenate(batches, axis=0) if batches else np.empty((0, 0), dtype=np.float16)
    seconds = time.perf_counter() - started
    metadata = {
        "model": model,
        "dim": int(vecs.shape[1]) if vecs.ndim == 2 else 0,
        "n": len(chunk_ids),
        "max_seq_length": max_seq_length,
        "dtype": "float16",
        "built_at": datetime.now(UTC).isoformat(),
        "seconds": seconds,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "chunk_id_count": len(raw_chunks),
    }
    _atomic_npy(output_path / "vecs.npy", vecs)
    _atomic_json(output_path / "ids.json", [str(chunk_id) for chunk_id in chunk_ids])
    _atomic_json(output_path / "meta.json", metadata)
    return {**metadata, "out_dir": str(output_path), "chunks_per_second": len(chunk_ids) / seconds if seconds else 0.0}


@dataclass(frozen=True)
class DenseSidecar:
    ids: list[str]
    vecs: np.ndarray
    meta: dict[str, Any]

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k <= 0 or not self.ids:
            return []
        matrix = np.asarray(self.vecs, dtype=np.float32)
        query = np.asarray(qvec, dtype=np.float32).reshape(-1)
        if matrix.ndim != 2 or query.size != matrix.shape[1]:
            raise ValueError(f"Query dimension {query.size} does not match sidecar dimension {matrix.shape[1] if matrix.ndim == 2 else 'invalid'}")
        norm = float(np.linalg.norm(query))
        query = query / norm if norm else np.zeros_like(query)
        scores = matrix @ query
        order = np.argsort(-scores, kind="stable")[: min(k, len(self.ids))]
        return [(self.ids[int(index)], float(scores[index])) for index in order]


def _sidecar_signature(directory: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (name, (directory / name).stat().st_mtime_ns, (directory / name).stat().st_size)
        for name in ("ids.json", "vecs.npy", "meta.json")
    )


@lru_cache(maxsize=16)
def _load_sidecar_cached(directory: str, _signature: tuple[tuple[str, int, int], ...]) -> DenseSidecar:
    path = Path(directory)
    ids = json.loads((path / "ids.json").read_text(encoding="utf-8"))
    if not isinstance(ids, list) or any(not isinstance(chunk_id, str) for chunk_id in ids):
        raise ValueError(f"Expected string chunk IDs in {path / 'ids.json'}")
    vecs = np.load(path / "vecs.npy", allow_pickle=False)
    if vecs.ndim != 2 or vecs.shape[0] != len(ids):
        raise ValueError(f"Sidecar IDs/vectors do not align: {len(ids)} IDs, shape {vecs.shape}")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"Expected sidecar metadata object in {path / 'meta.json'}")
    return DenseSidecar(ids=ids, vecs=vecs, meta=meta)


def load_sidecar(directory: str | Path) -> DenseSidecar:
    path = Path(directory).expanduser().resolve()
    return _load_sidecar_cached(str(path), _sidecar_signature(path))


def coverage(sidecar: DenseSidecar, chunk_ids: list[str] | set[str]) -> float:
    """Return the fraction of current chunk IDs represented in ``sidecar``."""
    expected = {str(chunk_id) for chunk_id in chunk_ids}
    if not expected:
        return 1.0
    return len(expected.intersection(sidecar.ids)) / len(expected)


class SidecarQueryEmbedder:
    """Lazy local SentenceTransformer for instruction-prefixed Qwen query vectors."""

    def __init__(
        self,
        model: str = DEFAULT_SIDECAR_MODEL,
        *,
        device: str = "auto",
        max_seq_length: int = 512,
        embedder: Any | None = None,
    ) -> None:
        self.model_name = model
        self.device = device
        self.max_seq_length = max_seq_length
        self._model = embedder

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = _load_model(self.model_name, self.device, self.max_seq_length)
        return self._model

    def embed(self, query: str) -> np.ndarray:
        model = self._get_model()
        vectors = model.encode(
            [QWEN_QUERY_PREFIX + query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        matrix = _normalise(np.asarray(vectors, dtype=np.float32))
        if matrix.shape[0] != 1:
            raise ValueError("query embedder must return exactly one vector")
        return matrix[0]


__all__ = [
    "DEFAULT_SIDECAR_MODEL", "DenseSidecar", "SidecarQueryEmbedder", "build_sidecar",
    "coverage", "load_sidecar",
]