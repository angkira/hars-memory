"""Compare Russian-corpus chunking strategies using offline hybrid retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import bm25s
import numpy as np
import tiktoken
from lightrag.operate import chunking_by_token_size

EXP_DIR = Path(__file__).resolve().parents[1]
RU_DIR = Path(__file__).resolve().parent
ML_DIR = EXP_DIR / "ml"
for _path in (EXP_DIR, ML_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import boot_ci, load_cases, ws  # noqa: E402
from embed_variants import (  # noqa: E402
    CHUNK_BATCH_SIZE,
    GemmaEmbedder,
    _atomic_json,
    _atomic_npy,
    _document_text,
    _normalise,
)
from fuse_ml import fuse_prod  # noqa: E402
from hars_memory.ingest.chunker import lightrag_chunking_func  # noqa: E402
from hars_memory.retrieval.tokenizer import tokenize_identifiers  # noqa: E402
from hars_memory.server.lightrag_init import resolve_tokenizer  # noqa: E402

INDEX_DIR = Path("/tmp/ru-exp/index-token")
DOCS_PATH = INDEX_DIR / "kv_store_full_docs.json"
CHUNKS_PATH = INDEX_DIR / "kv_store_text_chunks.json"
CASES_PATH = RU_DIR / "cases_ru.yaml"
STATS_PATH = Path("/tmp/ru-exp/step1_stats.json")
RESULTS_PATH = RU_DIR / "results" / "r6_chunking.json"
CACHE_DIR = RU_DIR / "cache"

EXPECTED_BASE_CHUNKS = 186
RETRIEVAL_TOP_K = 100
FUSION_POOL = 15
FUSION_TOP_K = 20
BOOTSTRAP_SAMPLES = 2000
MODEL_NAME = "unsloth/embeddinggemma-300m"
CELLS = ("A", "B", "C", "D")
VARIANTS = (
    ("token_512_64", "token", 512, 64),
    ("token_256_32", "token", 256, 32),
    ("token_1024_128", "token", 1024, 128),
    ("token_768_96", "token", 768, 96),
    ("char_1800_200", "char", 1800, 200),
    ("char_1300_150", "char", 1300, 150),
    ("markdown_512_min0", "markdown", 512, 0),
    ("markdown_512_min200", "markdown", 512, 200),
)
METRIC_KEYS = ("fact@1", "fact@3", "fact@5", "fact_rr")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _source_paths(chunks: dict[str, dict[str, Any]]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for record in chunks.values():
        doc_id = str(record.get("full_doc_id") or "")
        source_path = str(record.get("source_path") or record.get("file_path") or "")
        if not doc_id or not source_path:
            continue
        previous = paths.setdefault(doc_id, source_path)
        if previous != source_path:
            raise ValueError(
                f"full_doc_id {doc_id!r} maps to conflicting paths: {previous!r} and {source_path!r}"
            )
    return paths


def _load_documents() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    stored_chunks = _read_json(CHUNKS_PATH)
    documents = _read_json(DOCS_PATH)
    if not isinstance(stored_chunks, dict) or not isinstance(documents, dict):
        raise ValueError("Expected mapping objects in the read-only index stores")

    path_by_doc = _source_paths(stored_chunks)
    normalised: dict[str, dict[str, Any]] = {}
    for raw_id, raw_document in documents.items():
        doc_id = str(raw_id)
        if not isinstance(raw_document, dict):
            raise ValueError(f"Document {doc_id!r} is not an object")
        content = str(raw_document.get("content") or "")
        if not content.strip():
            continue
        source_path = path_by_doc.get(doc_id) or str(
            raw_document.get("source_path") or raw_document.get("file_path") or ""
        )
        if not source_path:
            raise ValueError(f"No source_path/file_path found for full_doc_id {doc_id!r}")
        normalised[doc_id] = {"content": content, "source_path": source_path}
    if not normalised:
        raise ValueError(f"No non-empty documents loaded from {DOCS_PATH}")
    return normalised, path_by_doc


def _add_chunk(
    chunks: dict[str, dict[str, Any]],
    text: str,
    source_path: str,
    doc_id: str,
    order: int,
) -> None:
    text = text.strip()
    if not text:
        return
    chunk_id = "chunk-" + hashlib.md5(text.encode("utf-8")).hexdigest()
    existing = chunks.get(chunk_id)
    if existing is not None:
        if existing["content"] != text:
            raise RuntimeError(f"MD5 collision for chunk {chunk_id}")
        return
    chunks[chunk_id] = {
        "id": chunk_id,
        "content": text,
        "source_path": source_path,
        "full_doc_id": doc_id,
        "chunk_order_index": order,
    }


def _build_token_chunks(
    documents: dict[str, dict[str, Any]], size: int, overlap: int
) -> dict[str, dict[str, Any]]:
    _, tokenizer, actual_size, actual_overlap = resolve_tokenizer(size, overlap)
    chunks: dict[str, dict[str, Any]] = {}
    for doc_id, document in documents.items():
        records = chunking_by_token_size(
            tokenizer,
            document["content"],
            chunk_token_size=actual_size,
            chunk_overlap_token_size=actual_overlap,
        )
        for record in records:
            _add_chunk(
                chunks,
                str(record.get("content") or ""),
                document["source_path"],
                doc_id,
                int(record.get("chunk_order_index", 0)),
            )
    if size == 512 and overlap == 64:
        if len(chunks) != EXPECTED_BASE_CHUNKS:
            print(
                f"WARNING: token_512_64 produced {len(chunks)} unique chunks; "
                f"the index has {EXPECTED_BASE_CHUNKS}",
                flush=True,
            )
    return chunks


def _build_char_chunks(
    documents: dict[str, dict[str, Any]], size: int, overlap: int
) -> dict[str, dict[str, Any]]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError(f"Invalid character window: size={size}, overlap={overlap}")
    chunks: dict[str, dict[str, Any]] = {}
    stride = size - overlap
    for doc_id, document in documents.items():
        content = document["content"]
        for order, start in enumerate(range(0, len(content), stride)):
            _add_chunk(
                chunks,
                content[start : start + size],
                document["source_path"],
                doc_id,
                order,
            )
    return chunks


def _build_markdown_chunks(
    documents: dict[str, dict[str, Any]],
    tokenizer: Any,
    size: int,
    overlap: int,
    min_tokens: int,
) -> dict[str, dict[str, Any]]:
    _, _, actual_size, actual_overlap = resolve_tokenizer(size, overlap)
    prior_min = os.environ.get("HARS_MEMORY_CHUNK_MIN_TOKENS")
    os.environ["HARS_MEMORY_CHUNK_MIN_TOKENS"] = str(min_tokens)
    chunks: dict[str, dict[str, Any]] = {}
    try:
        for doc_id, document in documents.items():
            records = lightrag_chunking_func(
                tokenizer,
                document["content"],
                chunk_overlap_token_size=actual_overlap,
                chunk_token_size=actual_size,
            )
            for record in records:
                _add_chunk(
                    chunks,
                    str(record.get("content") or ""),
                    document["source_path"],
                    doc_id,
                    int(record.get("chunk_order_index", 0)),
                )
    finally:
        if prior_min is None:
            os.environ.pop("HARS_MEMORY_CHUNK_MIN_TOKENS", None)
        else:
            os.environ["HARS_MEMORY_CHUNK_MIN_TOKENS"] = prior_min
    return chunks


def _build_variants(
    documents: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], str, Any]:
    tokenizer_name, tokenizer, _, _ = resolve_tokenizer(512, 64)
    chunk_sets: dict[str, dict[str, dict[str, Any]]] = {}
    for name, kind, size, parameter in VARIANTS:
        if kind == "token":
            chunks = _build_token_chunks(documents, size, parameter)
        elif kind == "char":
            chunks = _build_char_chunks(documents, size, parameter)
        else:
            chunks = _build_markdown_chunks(documents, tokenizer, size, 64, parameter)
        chunk_sets[name] = chunks
        print(f"chunked {name}: {len(chunks)} unique chunks", flush=True)
    return chunk_sets, tokenizer_name, tokenizer


def _cache_paths(label: str) -> tuple[Path, Path, Path]:
    return (
        CACHE_DIR / f"r6_emb_{label}.npy",
        CACHE_DIR / f"r6_ids_{label}.json",
        CACHE_DIR / f"r6_emb_{label}_meta.json",
    )


async def _embed_items(
    label: str,
    item_ids: list[str],
    texts: list[str],
    embedder: GemmaEmbedder,
    context: str,
) -> np.ndarray:
    if len(item_ids) != len(texts):
        raise ValueError(f"{label}: ids/texts do not align")
    matrix_path, ids_path, meta_path = _cache_paths(label)
    matrix: np.ndarray | None = None
    completed_ids: list[str] = []
    if matrix_path.is_file() and ids_path.is_file():
        try:
            saved_ids = _read_json(ids_path)
            saved_matrix = np.load(matrix_path, allow_pickle=False)
            if (
                isinstance(saved_ids, list)
                and saved_ids == item_ids[: len(saved_ids)]
                and saved_matrix.ndim == 2
                and saved_matrix.shape[0] == len(saved_ids)
                and len(saved_ids) <= len(item_ids)
            ):
                matrix = saved_matrix.astype(np.float32, copy=False)
                completed_ids = saved_ids
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            matrix = None
            completed_ids = []

    if matrix is None:
        vectors: list[np.ndarray] = []
    elif len(completed_ids) == len(item_ids):
        return _normalise(matrix)
    else:
        vectors = [matrix]

    elapsed = 0.0
    start = len(completed_ids)
    for batch_start in range(start, len(item_ids), CHUNK_BATCH_SIZE):
        batch_end = min(batch_start + CHUNK_BATCH_SIZE, len(item_ids))
        before = time.perf_counter()
        batch = await embedder.embed(texts[batch_start:batch_end], context=context)
        elapsed += time.perf_counter() - before
        if batch.ndim != 2 or batch.shape[0] != batch_end - batch_start:
            raise RuntimeError(
                f"{label}: expected {batch_end - batch_start} embedding rows, got {batch.shape}"
            )
        vectors.append(_normalise(batch))
        matrix = np.concatenate(vectors, axis=0)
        completed_ids = item_ids[:batch_end]
        _atomic_npy(matrix_path, matrix.astype(np.float32, copy=False))
        _atomic_json(ids_path, completed_ids)
        _atomic_json(
            meta_path,
            {"label": label, "model": MODEL_NAME, "context": context, "seconds": elapsed},
        )

    if not item_ids:
        raise ValueError(f"{label}: cannot embed an empty set")
    if matrix is None or matrix.shape[0] != len(item_ids):
        raise RuntimeError(
            f"{label}: incomplete embeddings {0 if matrix is None else matrix.shape[0]}/{len(item_ids)}"
        )
    return _normalise(matrix)


def _dense_rankings(
    chunk_ids: list[str],
    chunk_vectors: np.ndarray,
    query_vectors: np.ndarray,
) -> list[list[tuple[str, float]]]:
    if chunk_vectors.shape[0] != len(chunk_ids):
        raise ValueError("Chunk IDs and document embedding rows are misaligned")
    rankings: list[list[tuple[str, float]]] = []
    for query_vector in query_vectors:
        scores = chunk_vectors @ query_vector
        indices = np.argsort(-scores, kind="stable")[:RETRIEVAL_TOP_K]
        rankings.append([(chunk_ids[int(index)], float(scores[int(index)])) for index in indices])
    return rankings


def _bm25_rankings(
    chunks: dict[str, dict[str, Any]], cases: list[dict[str, Any]]
) -> list[list[tuple[str, float]]]:
    chunk_ids = sorted(chunks)
    corpus = [tokenize_identifiers(str(chunks[cid]["content"])) for cid in chunk_ids]
    retriever = bm25s.BM25(k1=1.5, b=0.75)
    retriever.index(corpus, show_progress=False)
    rankings: list[list[tuple[str, float]]] = []
    for case in cases:
        query_tokens = tokenize_identifiers(str(case.get("question") or ""))
        if not query_tokens or not chunk_ids:
            rankings.append([])
            continue
        indices, scores = retriever.retrieve(
            [query_tokens],
            k=min(RETRIEVAL_TOP_K, len(chunk_ids)),
            show_progress=False,
            sorted=True,
        )
        rankings.append(
            [
                (chunk_ids[int(index)], float(score))
                for index, score in zip(indices[0], scores[0])
                if float(score) > 0.0
            ]
        )
    return rankings


def _fact_present(case: dict[str, Any], chunk: dict[str, Any]) -> bool:
    fact = ws(str(case.get("gold_fact") or ""))
    content = ws(str(chunk.get("content") or ""))
    return bool(fact and fact in content)


def _score_case(
    case: dict[str, Any],
    ranked_ids: list[str],
    chunks: dict[str, dict[str, Any]],
    fact_covered: bool,
) -> dict[str, Any]:
    contains_fact = [
        _fact_present(case, chunks[chunk_id]) for chunk_id in ranked_ids if chunk_id in chunks
    ]
    result: dict[str, Any] = {
        "case_id": str(case["id"]),
        "cell": str(case["cell"]),
        "fact_covered": int(fact_covered),
    }
    for k in (1, 3, 5):
        result[f"fact@{k}"] = int(any(contains_fact[:k]))
    fact_rank = next(
        (rank for rank, hit in enumerate(contains_fact[:FUSION_TOP_K], start=1) if hit),
        None,
    )
    result["fact_rr"] = 1.0 / fact_rank if fact_rank is not None else 0.0
    result["fact@5_covered"] = result["fact@5"] if fact_covered else None
    return result


def _metric_summary(values: list[float]) -> dict[str, float | int]:
    lo, hi = boot_ci(values, n=BOOTSTRAP_SAMPLES, seed=1)
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else 0.0,
        "lo": lo,
        "hi": hi,
    }


def _summarize_cases(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {cell: [] for cell in CELLS}
    groups["overall"] = list(rows)
    for row in rows:
        groups[str(row["cell"])].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for group_name, group_rows in groups.items():
        covered_rows = [row for row in group_rows if row["fact_covered"]]
        metrics: dict[str, dict[str, float | int]] = {
            key: _metric_summary([float(row[key]) for row in group_rows]) for key in METRIC_KEYS
        }
        metrics["fact_covered"] = _metric_summary(
            [float(row["fact_covered"]) for row in group_rows]
        )
        metrics["fact@5_covered"] = _metric_summary(
            [float(row["fact@5_covered"]) for row in covered_rows]
        )
        summaries[group_name] = {"n_cases": len(group_rows), **metrics}
    return summaries


def _token_stats(chunks: dict[str, dict[str, Any]], encoding: Any) -> dict[str, float | int]:
    lengths = [len(encoding.encode(str(record["content"]))) for record in chunks.values()]
    return {
        "chunk_count": len(chunks),
        "mean_chunk_tokens_cl100k": statistics.fmean(lengths) if lengths else 0.0,
    }


def _case_coverage(case: dict[str, Any], chunks: dict[str, dict[str, Any]]) -> bool:
    return any(_fact_present(case, chunk) for chunk in chunks.values())


def _format_metric(stats: dict[str, Any]) -> str:
    return f"{float(stats['mean']):.3f} [{float(stats['lo']):.3f}, {float(stats['hi']):.3f}]"


def _print_table(results: dict[str, Any]) -> None:
    print(
        "\n| Variant | Cell | n | Chunks | Mean tok | Fact covered [95% CI] | "
        "Fact@5 covered [95% CI] | Fact@1 [95% CI] | Fact@3 [95% CI] | "
        "Fact@5 [95% CI] | Fact RR mean |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for variant_name, variant in results["variants"].items():
        stats = variant["chunk_stats"]
        for cell in (*CELLS, "overall"):
            summary = variant["by_cell"][cell]
            n = int(summary["n_cases"])
            cells = [
                variant_name,
                cell,
                str(n),
                str(stats["chunk_count"]),
                f"{float(stats['mean_chunk_tokens_cl100k']):.1f}",
                _format_metric(summary["fact_covered"]),
                _format_metric(summary["fact@5_covered"]),
                _format_metric(summary["fact@1"]),
                _format_metric(summary["fact@3"]),
                _format_metric(summary["fact@5"]),
                f"{float(summary['fact_rr']['mean']):.3f}",
            ]
            print("| " + " | ".join(cells) + " |")


async def _run() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases(CASES_PATH)
    if not cases:
        raise ValueError(f"No evaluation cases found in {CASES_PATH}")
    seen_case_ids: set[str] = set()
    for case in cases:
        missing = {"id", "cell", "question", "gold_fact"} - set(case)
        if missing:
            raise ValueError(f"Case is missing required fields: {sorted(missing)}")
        case_id = str(case["id"])
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen_case_ids.add(case_id)
        if str(case["cell"]) not in CELLS:
            raise ValueError(f"Unexpected case cell {case['cell']!r} for case {case_id}")

    documents, _ = _load_documents()
    chunk_sets, tokenizer_name, _ = _build_variants(documents)
    encoding = tiktoken.get_encoding("cl100k_base")
    embedder = GemmaEmbedder()

    query_ids = [
        f"{case['id']}:{hashlib.sha256(str(case['question']).encode('utf-8')).hexdigest()}"
        for case in cases
    ]
    query_vectors = await _embed_items(
        "queries",
        query_ids,
        [str(case["question"]) for case in cases],
        embedder,
        context="query",
    )

    result: dict[str, Any] = {
        "experiment": "R6 Russian chunking retrieval comparison",
        "input": {
            "index_dir": str(INDEX_DIR),
            "cases_path": str(CASES_PATH),
            "case_count": len(cases),
            "tokenizer": tokenizer_name,
            "retrieval": {
                "dense_top_k": RETRIEVAL_TOP_K,
                "bm25_top_k": RETRIEVAL_TOP_K,
                "fusion": "fuse_prod",
                "alpha": 0.5,
                "pool": FUSION_POOL,
                "top_k": FUSION_TOP_K,
            },
            "embedding_model": MODEL_NAME,
            "fact_match": "whitespace-normalised chunk content only",
        },
        "tokenizer_inflation": {},
        "variants": {},
    }

    if not STATS_PATH.is_file():
        raise FileNotFoundError(f"Missing tokenizer-density baseline: {STATS_PATH}")
    stats = _read_json(STATS_PATH)
    russian_density = float(stats["strata"]["ru"]["token_density"]["tokens_per_1000_chars"])
    english_density = float(stats["strata"]["en"]["token_density"]["tokens_per_1000_chars"])
    result["tokenizer_inflation"] = {
        "tokenizer": str(stats.get("tokenizer", "")),
        "russian_tokens_per_1000_chars": russian_density,
        "english_tokens_per_1000_chars": english_density,
        "absolute_difference": russian_density - english_density,
        "ratio_ru_to_en": russian_density / english_density if english_density else 0.0,
        "percent_over_english": (russian_density / english_density - 1.0) * 100.0
        if english_density
        else 0.0,
    }
    print(
        "Tokenizer density: RU "
        f"{russian_density:.1f} vs EN {english_density:.1f} tokens/1,000 chars "
        f"({result['tokenizer_inflation']['percent_over_english']:+.1f}%)",
        flush=True,
    )

    for variant_name, chunks in chunk_sets.items():
        chunk_ids = list(chunks)
        print(f"retrieving {variant_name} ({len(chunk_ids)} chunks)", flush=True)
        vectors = await _embed_items(
            variant_name,
            chunk_ids,
            [_document_text("gemma_plain", chunks[chunk_id]) for chunk_id in chunk_ids],
            embedder,
            context="document",
        )
        dense = _dense_rankings(chunk_ids, vectors, query_vectors)
        sparse = _bm25_rankings(chunks, cases)
        case_rows: list[dict[str, Any]] = []
        for case_index, case in enumerate(cases):
            fused_ids = fuse_prod(
                dense[case_index],
                sparse[case_index],
                alpha=0.5,
                pool=FUSION_POOL,
                top_k=FUSION_TOP_K,
            )
            covered = _case_coverage(case, chunks)
            case_rows.append(_score_case(case, fused_ids, chunks, covered))

        result["variants"][variant_name] = {
            "chunk_stats": _token_stats(chunks, encoding),
            "by_cell": _summarize_cases(case_rows),
            "case_scores": case_rows,
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(RESULTS_PATH, result)
    _print_table(result)
    print(f"Results: {RESULTS_PATH}")
    print("DONE")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
