#!/usr/bin/env python3
"""Build a resumable Qwen3 embedding sidecar for a cloned LightRAG index."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

_ML_DIR = Path(__file__).resolve().parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from mlib import (  # noqa: E402
    MAX_SEQ_LENGTH,
    QWEN_EMBEDDINGS_URL,
    QWEN_INSTRUCTION,
    QWEN_MODEL,
    _device_name,
    _load_model,
    _normalise,
    dense_search,
)

CHECKPOINT_EVERY = 500
BENCHMARK_CHUNKS = 64


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def _read_chunks(index_dir: Path) -> tuple[dict[str, dict[str, Any]], Path, str]:
    chunks_path = index_dir / "kv_store_text_chunks.json"
    content = chunks_path.read_bytes()
    chunks = json.loads(content)
    if not isinstance(chunks, dict):
        raise ValueError(f"Expected a chunk-ID mapping in {chunks_path}")
    for chunk_id, record in chunks.items():
        if not isinstance(record, dict) or not isinstance(record.get("content"), str):
            raise ValueError(f"Chunk {chunk_id!r} has no string content")
    return chunks, chunks_path, hashlib.sha256(content).hexdigest()


def _ordered_ids(chunks: dict[str, dict[str, Any]], limit: int | None) -> list[str]:
    ordered = sorted(chunks, key=lambda chunk_id: (len(chunks[chunk_id]["content"]), str(chunk_id)))
    return ordered if limit is None else ordered[:limit]


def _resume(
    out_dir: Path,
    chunk_ids: list[str],
    model_name: str,
    max_seq_length: int,
    source_sha: str,
) -> tuple[list[str], np.ndarray | None, dict[str, Any]]:
    ids_path = out_dir / "ids.json"
    vectors_path = out_dir / "vecs.npy"
    meta_path = out_dir / "meta.json"
    if not ids_path.is_file() or not vectors_path.is_file():
        return [], None, {}
    try:
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        vectors = np.load(vectors_path, allow_pickle=False)
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        if (
            not isinstance(ids, list)
            or ids != chunk_ids[: len(ids)]
            or vectors.ndim != 2
            or vectors.shape[0] != len(ids)
            or len(ids) > len(chunk_ids)
            or metadata.get("model") != model_name
            or int(metadata.get("max_seq_length", -1)) != max_seq_length
            or metadata.get("source_sha256") != source_sha
        ):
            raise ValueError("saved sidecar is not a compatible corpus prefix")
        return ids, vectors.astype(np.float32), metadata
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        print("Ignoring incompatible partial sidecar; starting from the first chunk.", flush=True)
        return [], None, {}


def _encode(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return _normalise(np.asarray(vectors, dtype=np.float32))


def _benchmark(model: Any, texts: list[str], batch_size: int, label: str) -> float:
    count = min(BENCHMARK_CHUNKS, len(texts))
    sample_indices = [int((index + 0.5) * len(texts) / count) for index in range(count)]
    sample = [texts[index] for index in sample_indices]
    if not sample:
        raise ValueError("Cannot benchmark an empty corpus")
    started = time.perf_counter()
    _encode(model, sample, batch_size)
    seconds = time.perf_counter() - started
    rate = len(sample) / max(seconds, 1e-9)
    print(f"{len(sample)}-chunk benchmark ({label}): {rate:.2f} chunks/s ({seconds:.2f}s)", flush=True)
    return rate


def _select_model(
    model_name: str, requested_device: str, max_seq_length: int, batch_size: int, texts: list[str]
) -> tuple[Any, str, int, float]:
    import torch

    device = _device_name(requested_device)
    model = _load_model(model_name, device, max_seq_length)
    best_rate = _benchmark(model, texts, batch_size, f"{device}/float32/batch={batch_size}")
    selected_model, selected_device, selected_batch = model, device, batch_size

    if device == "mps" and best_rate < 3.0:
        candidates: list[tuple[Any, str, int, float]] = []
        try:
            half_model = _load_model(model_name, "mps", max_seq_length, dtype=torch.float16)
            half_batch = max(batch_size * 2, 32)
            half_rate = _benchmark(half_model, texts, half_batch, f"mps/float16/batch={half_batch}")
            candidates.append((half_model, "mps", half_batch, half_rate))
        except Exception as exc:
            print(f"MPS float16 benchmark unavailable ({type(exc).__name__}).", flush=True)
        if best_rate < 3.0 and all(rate < 3.0 for _, _, _, rate in candidates):
            try:
                cpu_model = _load_model(model_name, "cpu", max_seq_length)
                cpu_rate = _benchmark(cpu_model, texts, batch_size, f"cpu/float32/batch={batch_size}")
                candidates.append((cpu_model, "cpu", batch_size, cpu_rate))
            except Exception as exc:
                print(f"CPU benchmark unavailable ({type(exc).__name__}).", flush=True)
        if candidates:
            candidate = max(candidates, key=lambda item: item[3])
            if candidate[3] > best_rate:
                selected_model, selected_device, selected_batch, best_rate = candidate
                del model
            else:
                for candidate_model, _, _, _ in candidates:
                    if candidate_model is not model:
                        del candidate_model
        print(f"Selected {selected_device}, batch={selected_batch} ({best_rate:.2f} chunks/s).", flush=True)
    return selected_model, selected_device, selected_batch, best_rate


def _token_overflow_count(model: Any, texts: list[str], max_seq_length: int) -> int:
    tokenizer = model.tokenizer
    count = 0
    for start in range(0, len(texts), 64):
        encoded = tokenizer(
            texts[start : start + 64],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )["input_ids"]
        count += sum(len(input_ids) > max_seq_length for input_ids in encoded)
    return count


def _request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Embedding endpoint returned a non-object JSON response")
    return result


def _endpoint_sanity(
    model: Any,
    chunks: dict[str, dict[str, Any]],
    model_name: str,
    max_seq_length: int,
    seed: int = 20261002,
) -> None:
    endpoint = QWEN_EMBEDDINGS_URL
    models_url = endpoint.rsplit("/", 1)[0] + "/models"
    try:
        served = _request_json(models_url).get("data")
        if not isinstance(served, list) or not served or not served[0].get("id"):
            print("Endpoint cosine sanity check skipped: no model reported by local server.", flush=True)
            return
        model_id = str(served[0]["id"])
        sample_ids = random.Random(seed).sample(list(chunks), min(20, len(chunks)))
        texts = [str(chunks[chunk_id]["content"]) for chunk_id in sample_ids]
        local = _encode(model, texts, 16)
        response = _request_json(
            endpoint,
            {"model": model_id, "input": texts, "truncate_prompt_tokens": max_seq_length},
        )
        rows = response.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ValueError("Endpoint returned an unexpected number of embeddings")
        rows.sort(key=lambda row: int(row.get("index", 0)))
        remote = _normalise(np.asarray([row["embedding"] for row in rows], dtype=np.float32))
        if remote.shape != local.shape:
            raise ValueError(f"Embedding dimensions differ: local {local.shape}, endpoint {remote.shape}")
        cosines = np.sum(local * remote, axis=1)
        print(
            f"Endpoint sanity: model={model_id}, n={len(cosines)}, "
            f"mean cosine={float(np.mean(cosines)):.4f} (local model={model_name}).",
            flush=True,
        )
    except Exception as exc:
        print(f"Endpoint cosine sanity check skipped ({type(exc).__name__}).", flush=True)


def _first_sentence(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith(("#", "[Document:", "```", "|")):
            continue
        candidate = re.sub(r"^[-*+>]\s+", "", candidate)
        match = re.search(r"^(.+?[.!?])(?:\s|$)", candidate)
        if not match:
            continue
        sentence = match.group(1).strip()
        if len(sentence) >= 200 and len(re.findall(r"\b\w+\b", sentence)) >= 25:
            return sentence
    return ""


def _self_test(model: Any, chunks: dict[str, dict[str, Any]], ids: list[str], vectors: np.ndarray) -> tuple[int, int]:
    eligible_ids = [chunk_id for chunk_id in ids if _first_sentence(str(chunks[chunk_id]["content"]))]
    count = min(10, len(eligible_ids))
    if count == 0:
        raise RuntimeError("No chunks have a complete, substantive first sentence for self-test")
    sample_ids = random.Random(17).sample(eligible_ids, count)
    queries = [_first_sentence(str(chunks[chunk_id]["content"])) for chunk_id in sample_ids]
    query_vectors = _encode(model, [QWEN_INSTRUCTION + query for query in queries], 16)
    hits = 0
    for chunk_id, query_vector in zip(sample_ids, query_vectors):
        hits += chunk_id in {candidate for candidate, _ in dense_search(vectors, ids, query_vector, 5)}
    print(
        f"Self-test: {hits}/{count} complete first-sentence queries retrieved their chunk in top-5 "
        "(sentence >=200 chars and >=25 words).",
        flush=True,
    )
    return hits, count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("/tmp/search-exp/ml-clone"))
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/search-exp/ml/sidecar_qwen3"))
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, or a torch device")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch < 1 or args.max_seq_length < 1 or (args.limit is not None and args.limit < 1):
        raise SystemExit("--batch, --max-seq-length, and --limit must be positive")

    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    process_started = time.perf_counter()
    index_dir = args.index_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir == index_dir or index_dir in out_dir.parents:
        raise SystemExit("--out-dir must be outside --index-dir; the index is read-only")
    chunks, chunks_path, source_sha = _read_chunks(index_dir)
    chunk_ids = _ordered_ids(chunks, args.limit)
    if not chunk_ids:
        raise SystemExit("No chunks found in the source index")
    texts = [chunks[chunk_id]["content"] for chunk_id in chunk_ids]
    out_dir.mkdir(parents=True, exist_ok=True)

    model, device, batch_size, _ = _select_model(
        args.model, args.device, args.max_seq_length, args.batch, texts
    )
    too_long = _token_overflow_count(model, texts, args.max_seq_length)
    print(
        f"Chunks exceeding max_seq_length={args.max_seq_length}: {too_long}/{len(texts)} "
        f"(tokenized with the local model tokenizer).",
        flush=True,
    )

    completed_ids, saved_vectors, saved_meta = _resume(
        out_dir, chunk_ids, args.model, args.max_seq_length, source_sha
    )
    rows = [saved_vectors] if saved_vectors is not None else []
    seconds = float(saved_meta.get("seconds", 0.0))
    start = len(completed_ids)
    elapsed_before_resume = float(saved_meta.get("elapsed_seconds", 0.0))
    print(f"Sidecar progress: {start}/{len(chunk_ids)} chunks already checkpointed.", flush=True)
    next_checkpoint = ((start // CHECKPOINT_EVERY) + 1) * CHECKPOINT_EVERY

    def checkpoint(completed: list[str]) -> np.ndarray:
        matrix = np.concatenate(rows, axis=0) if rows else np.empty((0, 0), dtype=np.float32)
        if matrix.shape[0] != len(completed):
            raise RuntimeError(f"Checkpoint mismatch: {matrix.shape[0]} vectors for {len(completed)} IDs")
        _atomic_npy(out_dir / "vecs.npy", matrix.astype(np.float16))
        _atomic_json(out_dir / "ids.json", completed)
        embedding_seconds = seconds
        n = len(completed)
        _atomic_json(
            out_dir / "meta.json",
            {
                "model": args.model,
                "dim": int(matrix.shape[1]),
                "n": n,
                "device": device,
                "dtype": "float16",
                "max_seq_length": args.max_seq_length,
                "seconds": embedding_seconds,
                "chunks_per_s": n / max(embedding_seconds, 1e-9),
                "source_index_dir": str(index_dir),
                "source_sha256": source_sha,
                "source_chunks_path": str(chunks_path),
                "chunks_exceed_max_seq_length": too_long,
                "elapsed_seconds": elapsed_before_resume + time.perf_counter() - process_started,
            },
        )
        return matrix

    for batch_start in range(start, len(chunk_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(chunk_ids))
        before = time.perf_counter()
        vectors = _encode(model, texts[batch_start:batch_end], batch_size)
        seconds += time.perf_counter() - before
        rows.append(vectors)
        if batch_end >= next_checkpoint or batch_end == len(chunk_ids):
            completed = chunk_ids[:batch_end]
            checkpoint(completed)
            print(
                f"Embedded {batch_end}/{len(chunk_ids)} chunks "
                f"({batch_end / max(seconds, 1e-9):.2f} chunks/s).",
                flush=True,
            )
            while next_checkpoint <= batch_end:
                next_checkpoint += CHECKPOINT_EVERY

    final_vectors = np.concatenate(rows, axis=0) if rows else np.empty((0, 0), dtype=np.float32)
    if final_vectors.shape[0] != len(chunk_ids):
        raise RuntimeError(f"Incomplete sidecar: {final_vectors.shape[0]}/{len(chunk_ids)} vectors")
    final_vectors = _normalise(final_vectors)
    rows = [final_vectors]
    checkpoint(chunk_ids)
    total_elapsed = elapsed_before_resume + time.perf_counter() - process_started
    throughput = len(chunk_ids) / max(seconds, 1e-9)
    print(
        f"Build complete: {len(chunk_ids)} chunks, {seconds:.2f}s embedding time, "
        f"{total_elapsed:.2f}s build wall time, {throughput:.2f} chunks/s; "
        f"vectors saved as float16 to {out_dir / 'vecs.npy'}.",
        flush=True,
    )

    _endpoint_sanity(model, chunks, args.model, args.max_seq_length)
    hits, count = _self_test(model, chunks, chunk_ids, final_vectors)
    if len(chunk_ids) == len(chunks) and (hits < 8 or count < 10):
        raise RuntimeError(f"Required self-test threshold not met: {hits}/{count}, expected at least 8/10")


if __name__ == "__main__":
    main()