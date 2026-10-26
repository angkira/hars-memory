"""Build offline embedding and dense-candidate caches for H3/H4 variants."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from common import CLONE, EXP, load_cases
from retr import _load_live_env

VARIANTS = ("gemma_plain", "gemma_prefix", "gemma_prefix2", "qwen3", "qwen3_prefix")
CHUNK_BATCH_SIZE = 32
QWEN_BATCH_SIZE = 32
QWEN_MAX_CHARS = 6000
QWEN_MAX_TOKENS = 512
QWEN_REQUEST_TIMEOUT = 5
QWEN_URL = os.environ.get("H34_QWEN_EMBEDDINGS_URL", "http://127.0.0.1:8001/v1/embeddings")
QWEN_INSTRUCTION = (
    "Instruct: Given a technical question about an SRE knowledge base, "
    "retrieve the passage that answers it\nQuery: "
)


def _parse_heading_path(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value.strip()]
    if isinstance(parsed, (list, tuple)):
        return [str(part).strip() for part in parsed if str(part).strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []


def _document_text(variant: str, chunk: dict[str, Any]) -> str:
    content = str(chunk.get("content", ""))
    if variant in {"gemma_plain", "qwen3"}:
        return content

    source_path = str(chunk.get("source_path", ""))
    headings = _parse_heading_path(chunk.get("heading_path"))
    if variant in {"gemma_prefix", "qwen3_prefix"}:
        prefix = " > ".join([part for part in (source_path, *headings) if part])
    elif variant == "gemma_prefix2":
        stem = Path(source_path).stem if source_path else ""
        prefix = " > ".join([part for part in (stem, *headings) if part])
    else:
        raise ValueError(f"Unknown embedding variant: {variant}")
    return f"{prefix}\n{content}" if prefix else content


def _load_base_candidates() -> dict[str, Any]:
    path = EXP / "cache" / "cands_base.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required baseline candidate cache not found: {path}")
    data = json.loads(path.read_text("utf-8"))
    return data.get("cases", data)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as fh:
        np.save(fh, value, allow_pickle=False)
    temporary.replace(path)


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def _request_json(
    url: str, payload: dict[str, Any] | None = None, timeout: float = 300
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body is not None else {}
    request = urllib.request.Request(url, data=body, headers=headers, method="GET" if body is None else "POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return result


class QwenEmbedder:
    def __init__(self) -> None:
        models_url = QWEN_URL.rsplit("/", 1)[0] + "/models"
        result = _request_json(models_url)
        models = result.get("data")
        if not isinstance(models, list) or not models or not isinstance(models[0], dict):
            raise RuntimeError(f"No served embedding model returned by GET {models_url}")
        self.model = str(models[0].get("id") or "")
        if not self.model:
            raise RuntimeError(f"Served model has no id: {models[0]!r}")
        print(f"Qwen endpoint model: {self.model}", flush=True)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        truncated = [text[:QWEN_MAX_CHARS] for text in texts]
        payload = {
            "model": self.model,
            "input": truncated,
            "truncate_prompt_tokens": QWEN_MAX_TOKENS,
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                result = _request_json(QWEN_URL, payload, timeout=QWEN_REQUEST_TIMEOUT)
                data = result.get("data")
                if not isinstance(data, list) or len(data) != len(texts):
                    raise ValueError(f"Expected {len(texts)} embeddings, got {len(data) if isinstance(data, list) else data!r}")
                data.sort(key=lambda item: int(item.get("index", 0)))
                vectors = np.asarray([item["embedding"] for item in data], dtype=np.float32)
                if vectors.ndim != 2 or vectors.shape[0] != len(texts):
                    raise ValueError(f"Unexpected embedding matrix shape: {vectors.shape}")
                return _normalise(vectors)
            except (OSError, urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
                if isinstance(exc, urllib.error.HTTPError):
                    detail = exc.read().decode("utf-8", "replace")[:1000]
                    last_error = RuntimeError(f"{exc}: {detail}")
                else:
                    last_error = exc
                timed_out = isinstance(exc, (TimeoutError, socket.timeout)) or (
                    isinstance(exc, urllib.error.URLError)
                    and isinstance(exc.reason, (TimeoutError, socket.timeout))
                )
                if timed_out and len(truncated) > 1:
                    midpoint = len(truncated) // 2
                    print(
                        f"Qwen batch of {len(truncated)} timed out; splitting into "
                        f"{midpoint}+{len(truncated) - midpoint}",
                        flush=True,
                    )
                    return np.concatenate(
                        [self.embed(truncated[:midpoint]), self.embed(truncated[midpoint:])],
                        axis=0,
                    )
                if attempt < 2:
                    delay = min(2**attempt, 16)
                    print(f"Qwen request failed ({last_error}); retrying in {delay}s ({attempt + 1}/3)", flush=True)
                    time.sleep(delay)
        raise RuntimeError(f"Qwen embeddings failed after 3 attempts: {last_error}") from last_error


class GemmaEmbedder:
    def __init__(self) -> None:
        _load_live_env()
        os.environ["HARS_MEMORY_EMBED_DEVICE"] = "cpu"
        os.environ["HF_HUB_OFFLINE"] = "1"
        from hars_memory.server.embedder import make_embedding_func

        self._embed = make_embedding_func(
            model_name="unsloth/embeddinggemma-300m",
            hf_cache_dir=os.environ.get("HF_HOME", "/mnt/datasets/models/.hf_home"),
            batch_size=CHUNK_BATCH_SIZE,
        )

    async def embed(self, texts: list[str], context: str) -> np.ndarray:
        result = await self._embed(texts, context=context)
        return np.asarray(result, dtype=np.float32)


def _embedding_paths(variant: str) -> tuple[Path, Path, Path]:
    cache = EXP / "cache"
    return (
        cache / f"emb_{variant}.npy",
        cache / f"ids_{variant}.json",
        cache / f"emb_{variant}_meta.json",
    )


def _resume_state(
    variant: str, chunk_ids: list[str]
) -> tuple[np.ndarray | None, list[str], dict[str, Any], Path, Path, Path]:
    matrix_path, ids_path, meta_path = _embedding_paths(variant)
    ids: list[str] = []
    matrix: np.ndarray | None = None
    metadata: dict[str, Any] = {}
    if ids_path.is_file() and matrix_path.is_file():
        try:
            ids = json.loads(ids_path.read_text("utf-8"))
            matrix = np.load(matrix_path, allow_pickle=False)
            if not isinstance(ids, list) or ids != chunk_ids[: len(ids)] or matrix.ndim != 2 or matrix.shape[0] != len(ids):
                raise ValueError("saved ids/matrix are not a valid prefix of the current corpus")
            if len(ids) > len(chunk_ids):
                raise ValueError("saved embeddings exceed current corpus length")
            if len(ids) == len(chunk_ids) and matrix.shape[1] > 0:
                if meta_path.is_file():
                    metadata = json.loads(meta_path.read_text("utf-8"))
                return matrix, ids, metadata, matrix_path, ids_path, meta_path
        except (OSError, ValueError, json.JSONDecodeError):
            print(f"Ignoring incompatible partial embedding cache for {variant}", flush=True)
            ids, matrix, metadata = [], None, {}
    if meta_path.is_file() and matrix is not None:
        try:
            metadata = json.loads(meta_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    return matrix, ids, metadata, matrix_path, ids_path, meta_path


async def _run(variant: str) -> None:
    cache = EXP / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    chunks_path = CLONE / "kv_store_text_chunks.json"
    chunks = json.loads(chunks_path.read_text("utf-8"))
    if not isinstance(chunks, dict):
        raise ValueError(f"Expected chunk-id mapping in {chunks_path}")
    chunk_ids = list(chunks)
    if not chunk_ids:
        raise RuntimeError(f"No chunks found in {chunks_path}")
    cases = load_cases()
    base_candidates = _load_base_candidates()
    embedder: GemmaEmbedder | QwenEmbedder = QwenEmbedder() if variant.startswith("qwen3") else GemmaEmbedder()
    matrix, completed_ids, metadata, matrix_path, ids_path, meta_path = _resume_state(variant, chunk_ids)
    start = len(completed_ids)
    if matrix is not None:
        embedded: list[np.ndarray] = [matrix]
    else:
        embedded = []
    chunk_embedding_seconds = float(metadata.get("chunk_embedding_seconds", 0.0))
    print(f"{variant}: {start}/{len(chunk_ids)} chunks already embedded", flush=True)
    next_progress = (start // 500 + 1) * 500

    def checkpoint(rows: list[np.ndarray], completed: list[str], elapsed: float) -> None:
        combined = np.concatenate(rows, axis=0) if rows else np.empty((0, 0), dtype=np.float32)
        _atomic_npy(matrix_path, combined.astype(np.float32, copy=False))
        _atomic_json(ids_path, completed)
        _atomic_json(
            meta_path,
            {
                "variant": variant,
                "model": embedder.model if isinstance(embedder, QwenEmbedder) else "unsloth/embeddinggemma-300m",
                "chunk_count": len(completed),
                "chunk_embedding_seconds": elapsed,
            },
        )

    if start < len(chunk_ids):
        for batch_start in range(start, len(chunk_ids), CHUNK_BATCH_SIZE):
            batch_end = min(batch_start + CHUNK_BATCH_SIZE, len(chunk_ids))
            texts = [_document_text(variant, chunks[chunk_id]) for chunk_id in chunk_ids[batch_start:batch_end]]
            before = time.perf_counter()
            if isinstance(embedder, QwenEmbedder):
                print(f"{variant}: requesting chunks {batch_start + 1}-{batch_end}", flush=True)
            if isinstance(embedder, GemmaEmbedder):
                vectors = await embedder.embed(texts, context="document")
            else:
                vectors = embedder.embed(texts)
            chunk_embedding_seconds += time.perf_counter() - before
            if isinstance(embedder, QwenEmbedder):
                print(f"{variant}: received chunks {batch_start + 1}-{batch_end}", flush=True)
            if vectors.shape[0] != len(texts):
                raise RuntimeError(f"{variant} returned {vectors.shape[0]} embeddings for {len(texts)} chunks")
            embedded.append(vectors)
            completed = chunk_ids[:batch_end]
            if batch_end % 128 < CHUNK_BATCH_SIZE or batch_end == len(chunk_ids):
                checkpoint(embedded, completed, chunk_embedding_seconds)
            if batch_end >= next_progress or batch_end == len(chunk_ids):
                rate = batch_end / max(chunk_embedding_seconds, 1e-9)
                print(f"{variant}: embedded {batch_end}/{len(chunk_ids)} chunks ({rate:.1f} chunks/s)", flush=True)
                while next_progress <= batch_end:
                    next_progress += 500

    matrix = np.concatenate(embedded, axis=0) if embedded else np.empty((0, 0), dtype=np.float32)
    matrix = _normalise(matrix)
    if matrix.shape[0] != len(chunk_ids):
        raise RuntimeError(f"Incomplete document embeddings: {matrix.shape[0]}/{len(chunk_ids)}")
    checkpoint([matrix], chunk_ids, chunk_embedding_seconds)
    metadata = json.loads(meta_path.read_text("utf-8"))

    question_texts = [str(case.get("question", "")) for case in cases]
    query_started = time.perf_counter()
    if isinstance(embedder, GemmaEmbedder):
        query_vectors = await embedder.embed(question_texts, context="query")
    else:
        query_vectors = np.concatenate(
            [
                embedder.embed([QWEN_INSTRUCTION + question for question in question_texts[i : i + QWEN_BATCH_SIZE]])
                for i in range(0, len(question_texts), QWEN_BATCH_SIZE)
            ],
            axis=0,
        )
    query_embedding_seconds = time.perf_counter() - query_started
    query_vectors = _normalise(query_vectors)
    if query_vectors.shape[0] != len(cases):
        raise RuntimeError(f"{variant} returned {query_vectors.shape[0]} query embeddings for {len(cases)} cases")

    output_cases: dict[str, dict[str, Any]] = {}
    per_query_seconds = query_embedding_seconds / max(len(cases), 1)
    for case_index, case in enumerate(cases):
        case_id = str(case["id"])
        scores = matrix @ query_vectors[case_index]
        top_indices = np.argsort(-scores, kind="stable")[:100]
        base_record = base_candidates.get(case_id)
        if not isinstance(base_record, dict):
            raise KeyError(f"No baseline candidate record for case {case_id}")
        sparse = base_record.get("sparse", base_record.get("bm25", []))
        dense = [(chunk_ids[int(idx)], float(scores[int(idx)])) for idx in top_indices]
        output_cases[case_id] = {
            "dense": dense,
            "bm25": sparse,
            "sparse": sparse,
            "query_embedding_seconds": per_query_seconds,
        }

    cache_path = cache / f"cands_{variant}.json"
    _atomic_json(
        cache_path,
        {
            "pool": 100,
            "variant": variant,
            "model": metadata.get("model"),
            "chunk_embedding_seconds": chunk_embedding_seconds,
            "chunk_embedding_throughput": len(chunk_ids) / chunk_embedding_seconds if chunk_embedding_seconds > 0 else 0.0,
            "query_embedding_seconds_total": query_embedding_seconds,
            "query_embedding_seconds_mean": per_query_seconds,
            "cases": output_cases,
        },
    )

    if variant == "gemma_plain":
        matches = 0
        comparable = 0
        for case in cases:
            case_id = str(case["id"])
            base_dense = base_candidates.get(case_id, {}).get("dense", [])
            new_dense = output_cases[case_id]["dense"]
            if base_dense and new_dense:
                comparable += 1
                matches += str(base_dense[0][0]) == str(new_dense[0][0])
        agreement = matches / comparable if comparable else 0.0
        print(f"gemma_plain dense top-1 agreement: {matches}/{comparable} = {agreement:.1%}", flush=True)
        if agreement < 0.95:
            raise RuntimeError(f"gemma_plain sanity check failed: top-1 agreement {agreement:.1%} is below 95%")

    throughput = len(chunk_ids) / chunk_embedding_seconds if chunk_embedding_seconds > 0 else 0.0
    print(
        f"{variant}: wrote {matrix_path.name}, {ids_path.name}, {cache_path.name}; "
        f"{len(chunk_ids)} chunks, {throughput:.2f} chunks/s, "
        f"query embedding {query_embedding_seconds:.2f}s total ({per_query_seconds:.4f}s/case)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    args = parser.parse_args()
    asyncio.run(_run(args.variant))


if __name__ == "__main__":
    main()