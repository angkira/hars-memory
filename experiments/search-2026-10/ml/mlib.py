"""Small helpers for the 2026-10 local dense-search experiments."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_INSTRUCTION = (
    "Instruct: Given a technical question about an SRE knowledge base, "
    "retrieve the passage that answers it\nQuery: "
)
QWEN_EMBEDDINGS_URL = "http://127.0.0.1:8001/v1/embeddings"
MAX_SEQ_LENGTH = 512

T = TypeVar("T")


def _normalise(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _device_name(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_model(
    model_name: str,
    device: str,
    max_seq_length: int,
    dtype: Any | None = None,
) -> Any:
    from sentence_transformers import SentenceTransformer

    model_kwargs = {"torch_dtype": dtype} if dtype is not None else None
    model = SentenceTransformer(
        model_name,
        device=device,
        local_files_only=True,
        model_kwargs=model_kwargs,
    )
    model.max_seq_length = max_seq_length
    model.tokenizer.padding_side = "left"
    return model


def load_sidecar(directory: str | Path) -> tuple[list[str], np.ndarray]:
    """Load sidecar IDs and vectors, converting vectors to float32."""
    path = Path(directory)
    ids = json.loads((path / "ids.json").read_text(encoding="utf-8"))
    if not isinstance(ids, list) or any(not isinstance(chunk_id, str) for chunk_id in ids):
        raise ValueError(f"Expected a string ID list in {path / 'ids.json'}")
    vectors = np.load(path / "vecs.npy", allow_pickle=False).astype(np.float32, copy=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise ValueError(f"Sidecar IDs/vectors do not align: {len(ids)} IDs, shape {vectors.shape}")
    return ids, vectors


class QwenQueryEmbedder:
    """Lazily load the local Qwen3 model and embed instruction-prefixed queries."""

    def __init__(
        self,
        model: str = QWEN_MODEL,
        device: str = "auto",
        max_seq_length: int = MAX_SEQ_LENGTH,
        fallback_url: str | None = None,
    ) -> None:
        self.model_name = model
        self.device = _device_name(device)
        self.max_seq_length = max_seq_length
        self.fallback_url = fallback_url
        self._model: Any | None = None

    def _local_model(self) -> Any:
        if self._model is None:
            self._model = _load_model(self.model_name, self.device, self.max_seq_length)
        return self._model

    def _fallback(self, texts: list[str]) -> np.ndarray:
        if not self.fallback_url:
            raise RuntimeError("No Qwen embedding fallback URL is configured")
        models_url = self.fallback_url.rsplit("/", 1)[0] + "/models"
        with urllib.request.urlopen(models_url, timeout=10) as response:
            models = json.loads(response.read().decode("utf-8")).get("data", [])
        if not models or not models[0].get("id"):
            raise RuntimeError(f"No embedding model returned by {models_url}")
        payload = json.dumps(
            {
                "model": str(models[0]["id"]),
                "input": [QWEN_INSTRUCTION + text for text in texts],
                "truncate_prompt_tokens": self.max_seq_length,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.fallback_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8")).get("data", [])
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError(f"Expected {len(texts)} fallback embeddings")
        data.sort(key=lambda row: int(row.get("index", 0)))
        return _normalise(np.asarray([row["embedding"] for row in data], dtype=np.float32))

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        try:
            vectors = self._local_model().encode(
                [QWEN_INSTRUCTION + text for text in texts],
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return _normalise(np.asarray(vectors, dtype=np.float32))
        except Exception:
            if not self.fallback_url:
                raise
            return self._fallback(texts)


def dense_search(
    vecs: np.ndarray, ids: list[str], qvec: np.ndarray, k: int
) -> list[tuple[str, float]]:
    """Return the top-k chunk IDs by cosine similarity."""
    matrix = np.asarray(vecs, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise ValueError(f"IDs/vectors do not align: {len(ids)} IDs, shape {matrix.shape}")
    if k <= 0 or not ids:
        return []
    query = np.asarray(qvec, dtype=np.float32).reshape(-1)
    if query.shape[0] != matrix.shape[1]:
        raise ValueError(f"Query dimension {query.shape[0]} does not match vectors {matrix.shape[1]}")
    norm = float(np.linalg.norm(query))
    query = query / norm if norm > 0 else np.zeros_like(query)
    scores = matrix @ query
    order = np.argsort(-scores, kind="stable")[: min(k, len(ids))]
    return [(ids[int(index)], float(scores[index])) for index in order]


def timed_call(function: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
    """Call a function and return its result with elapsed wall time in seconds."""
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


__all__ = ["QwenQueryEmbedder", "dense_search", "load_sidecar", "timed_call"]