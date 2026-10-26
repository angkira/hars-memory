"""Evaluate markdown chunking on the read-only search experiment clone."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import Any

import bm25s
import numpy as np

from common import CLONE, EXP, load_cases, ws
from embed_variants import (
    CHUNK_BATCH_SIZE,
    GemmaEmbedder,
    _atomic_json,
    _atomic_npy,
    _document_text,
    _normalise,
)
from hars_memory.ingest.chunker import lightrag_chunking_func
from hars_memory.retrieval.tokenizer import tokenize_identifiers
from hars_memory.server.lightrag_init import resolve_tokenizer
from metrics import breakdown, load_chunks, save_results, table
from retr import fuse_hybrid, rank_bm25, rank_dense
from run_baseline import METRIC_KEYS, TOP_K

logger = logging.getLogger("h9_chunking")

DOCS_PATH = CLONE / "kv_store_full_docs.json"
CHUNK_TOKEN_SIZE = 512
CHUNK_OVERLAP_TOKENS = 64
MIN_TOKEN_SETTINGS = {"md": 0, "mdmerge": 200}
RETRIEVAL_TOP_K = 100
RESULT_KS = (1, 3, 5)
CHECKPOINT_CHUNKS = 256


def _setup_logging() -> None:
    cache = EXP / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(cache / "h9_progress.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(stream)
    logger.addHandler(file_handler)


def _source_paths(chunks: dict[str, dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in chunks.values():
        doc_id = str(record.get("full_doc_id") or "")
        source_path = str(record.get("source_path") or record.get("file_path") or "")
        if not doc_id or not source_path:
            continue
        previous = result.setdefault(doc_id, source_path)
        if previous != source_path:
            raise ValueError(
                f"full_doc_id {doc_id!r} maps to conflicting paths: {previous!r} and {source_path!r}"
            )
    return result


def _load_documents() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    token_chunks = load_chunks()
    path_by_doc = _source_paths(token_chunks)
    documents = json.loads(DOCS_PATH.read_text("utf-8"))
    if not isinstance(documents, dict):
        raise ValueError(f"Expected doc-id mapping in {DOCS_PATH}")
    if len(documents) != 1057:
        logger.warning("Clone contains %d full docs (task expected 1057)", len(documents))
    return documents, path_by_doc


def _chunk_set(
    label: str,
    min_tokens: int,
    documents: dict[str, dict[str, Any]],
    path_by_doc: dict[str, str],
    tokenizer: Any,
    tokenizer_name: str,
    chunk_size: int,
    overlap: int,
) -> dict[str, dict[str, Any]]:
    logger.info(
        "%s: chunking %d documents (tokenizer=%s chunk=%d merge_min=%d)",
        label,
        len(documents),
        tokenizer_name,
        CHUNK_TOKEN_SIZE,
        min_tokens,
    )
    prior_min = os.environ.get("HARS_MEMORY_CHUNK_MIN_TOKENS")
    os.environ["HARS_MEMORY_CHUNK_MIN_TOKENS"] = str(min_tokens)
    chunks: dict[str, dict[str, Any]] = {}
    duplicate_ids = 0
    try:
        for doc_index, (doc_id_raw, document) in enumerate(documents.items(), start=1):
            doc_id = str(doc_id_raw)
            if not isinstance(document, dict):
                raise ValueError(f"Document {doc_id!r} is not an object")
            content = str(document.get("content") or "")
            if not content.strip():
                continue
            source_path = path_by_doc.get(doc_id) or str(
                document.get("source_path") or document.get("file_path") or ""
            )
            if not source_path:
                raise ValueError(f"No source_path/file_path found for full_doc_id {doc_id!r}")

            records = lightrag_chunking_func(
                tokenizer,
                content,
                chunk_overlap_token_size=overlap,
                chunk_token_size=chunk_size,
            )
            for record in records:
                text = str(record.get("content") or "")
                if not text.strip():
                    continue
                chunk_id = "chunk-" + hashlib.md5(text.encode("utf-8")).hexdigest()
                chunk = {
                    "id": chunk_id,
                    "content": text,
                    "source_path": source_path,
                    "heading_path": record.get("heading_path") or [],
                    "chunk_order_index": int(record.get("chunk_order_index", len(chunks))),
                    "full_doc_id": doc_id,
                }
                existing = chunks.get(chunk_id)
                if existing is not None:
                    if existing["content"] != text:
                        raise RuntimeError(f"MD5 collision for chunk {chunk_id}")
                    duplicate_ids += 1
                    continue
                chunks[chunk_id] = chunk
            if doc_index % 100 == 0 or doc_index == len(documents):
                logger.info("%s: chunked %d/%d docs; unique chunks=%d", label, doc_index, len(documents), len(chunks))
    finally:
        if prior_min is None:
            os.environ.pop("HARS_MEMORY_CHUNK_MIN_TOKENS", None)
        else:
            os.environ["HARS_MEMORY_CHUNK_MIN_TOKENS"] = prior_min

    if duplicate_ids:
        logger.info("%s: collapsed %d duplicate content-addressed chunk IDs", label, duplicate_ids)
    path = EXP / "cache" / f"h9_chunks_{label}.json"
    _atomic_json(path, chunks)
    return chunks


def _token_chunks() -> dict[str, dict[str, Any]]:
    source = load_chunks()
    by_doc = _source_paths(source)
    output: dict[str, dict[str, Any]] = {}
    for chunk_id, record in source.items():
        item = dict(record)
        item["id"] = str(chunk_id)
        item["source_path"] = str(item.get("source_path") or item.get("file_path") or "")
        item["full_doc_id"] = str(item.get("full_doc_id") or "")
        item["heading_path"] = item.get("heading_path") or []
        item["chunk_order_index"] = int(item.get("chunk_order_index", 0) or 0)
        if not item["source_path"] and item["full_doc_id"]:
            item["source_path"] = by_doc.get(item["full_doc_id"], "")
        output[str(chunk_id)] = item
    return output


def _chunk_stats(
    label: str,
    chunks: dict[str, dict[str, Any]],
    tokenizer: Any,
) -> dict[str, float | int]:
    chars = [len(str(record.get("content", ""))) for record in chunks.values()]
    tokens = [len(tokenizer.encode(str(record.get("content", "")))) for record in chunks.values()]
    if not chars:
        return {"set": label, "chunks": 0, "docs": 0, "mean_chars": 0, "median_chars": 0, "p95_chars": 0,
                "mean_tokens": 0, "median_tokens": 0, "p95_tokens": 0}
    return {
        "set": label,
        "chunks": len(chunks),
        "docs": len({str(record.get("full_doc_id", "")) for record in chunks.values()}),
        "mean_chars": statistics.fmean(chars),
        "median_chars": float(np.median(chars)),
        "p95_chars": float(np.percentile(chars, 95)),
        "mean_tokens": statistics.fmean(tokens),
        "median_tokens": float(np.median(tokens)),
        "p95_tokens": float(np.percentile(tokens, 95)),
    }


def _print_stats(stats: list[dict[str, float | int]]) -> None:
    print("\nChunk set size and length (cl100k token counts):")
    print("| Set | Chunks | Docs | Mean chars | Median chars | P95 chars | Mean tokens | Median tokens | P95 tokens |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in stats:
        print(
            f"| {row['set']} | {row['chunks']} | {row['docs']} | {row['mean_chars']:.0f} | "
            f"{row['median_chars']:.0f} | {row['p95_chars']:.0f} | {row['mean_tokens']:.0f} | "
            f"{row['median_tokens']:.0f} | {row['p95_tokens']:.0f} |"
        )
    token = stats[0]
    print(
        f"\nToken baseline reference: {token['chunks']} chunks, mean {token['mean_chars']:.0f} chars "
        "(task estimate: 4168 chunks, ~1.7k chars)."
    )


def _cache_paths(label: str) -> tuple[Path, Path, Path]:
    cache = EXP / "cache"
    return (
        cache / f"emb_{label}.npy",
        cache / f"ids_{label}.json",
        cache / f"emb_{label}_meta.json",
    )


async def _embed_corpus(
    label: str,
    chunks: dict[str, dict[str, Any]],
    embedder: GemmaEmbedder,
) -> np.ndarray:
    chunk_ids = list(chunks)
    matrix_path, ids_path, meta_path = _cache_paths(label)
    completed_ids: list[str] = []
    vectors: list[np.ndarray] = []
    elapsed = 0.0
    if matrix_path.is_file() and ids_path.is_file():
        try:
            saved_ids = json.loads(ids_path.read_text("utf-8"))
            saved_matrix = np.load(matrix_path, allow_pickle=False)
            if (
                isinstance(saved_ids, list)
                and saved_ids == chunk_ids[: len(saved_ids)]
                and saved_matrix.ndim == 2
                and saved_matrix.shape[0] == len(saved_ids)
                and len(saved_ids) <= len(chunk_ids)
            ):
                completed_ids = saved_ids
                vectors = [saved_matrix]
                if meta_path.is_file():
                    elapsed = float(json.loads(meta_path.read_text("utf-8")).get("seconds", 0.0))
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            logger.warning("%s: ignoring incompatible partial embedding cache", label)
            completed_ids, vectors, elapsed = [], [], 0.0

    start = len(completed_ids)
    logger.info("%s: %d/%d chunk embeddings cached", label, start, len(chunk_ids))
    for batch_start in range(start, len(chunk_ids), CHUNK_BATCH_SIZE):
        batch_end = min(batch_start + CHUNK_BATCH_SIZE, len(chunk_ids))
        texts = [
            _document_text("gemma_plain", chunks[chunk_id])
            for chunk_id in chunk_ids[batch_start:batch_end]
        ]
        before = time.perf_counter()
        batch = await embedder.embed(texts, context="document")
        elapsed += time.perf_counter() - before
        if batch.ndim != 2 or batch.shape[0] != len(texts):
            raise RuntimeError(f"{label}: expected {len(texts)} embedding rows, got {batch.shape}")
        vectors.append(_normalise(batch))
        completed_ids = chunk_ids[:batch_end]
        if batch_end % CHECKPOINT_CHUNKS < CHUNK_BATCH_SIZE or batch_end == len(chunk_ids):
            matrix = np.concatenate(vectors, axis=0)
            _atomic_npy(matrix_path, matrix.astype(np.float32, copy=False))
            _atomic_json(ids_path, completed_ids)
            _atomic_json(meta_path, {"label": label, "model": "unsloth/embeddinggemma-300m", "seconds": elapsed})
            logger.info(
                "%s: embedded %d/%d chunks (%.2f chunks/s)",
                label,
                batch_end,
                len(chunk_ids),
                batch_end / max(elapsed, 1e-9),
            )

    if not vectors:
        raise RuntimeError(f"{label}: no chunk embeddings available")
    matrix = _normalise(np.concatenate(vectors, axis=0))
    if matrix.shape[0] != len(chunk_ids):
        raise RuntimeError(f"{label}: incomplete document embeddings {matrix.shape[0]}/{len(chunk_ids)}")
    _atomic_npy(matrix_path, matrix.astype(np.float32, copy=False))
    _atomic_json(ids_path, chunk_ids)
    _atomic_json(meta_path, {"label": label, "model": "unsloth/embeddinggemma-300m", "seconds": elapsed})
    return matrix


async def _query_embeddings(cases: list[dict[str, Any]], embedder: GemmaEmbedder) -> np.ndarray:
    query_ids = [str(case["id"]) for case in cases]
    matrix_path = EXP / "cache" / "emb_h9_queries.npy"
    ids_path = EXP / "cache" / "ids_h9_queries.json"
    if matrix_path.is_file() and ids_path.is_file():
        try:
            ids = json.loads(ids_path.read_text("utf-8"))
            matrix = np.load(matrix_path, allow_pickle=False)
            if ids == query_ids and matrix.ndim == 2 and matrix.shape[0] == len(query_ids):
                return _normalise(matrix)
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring incompatible cached query embeddings")
    texts = [str(case.get("question", "")) for case in cases]
    logger.info("Embedding %d queries with embeddinggemma context=query", len(texts))
    matrix = _normalise(await embedder.embed(texts, context="query"))
    if matrix.ndim != 2 or matrix.shape[0] != len(query_ids):
        raise RuntimeError(f"Expected {len(query_ids)} query embeddings, got {matrix.shape}")
    _atomic_npy(matrix_path, matrix.astype(np.float32, copy=False))
    _atomic_json(ids_path, query_ids)
    return matrix


def _bm25_rankings(
    chunks: dict[str, dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, list[tuple[str, float]]]:
    chunk_ids = sorted(chunks)
    corpus = [tokenize_identifiers(str(chunks[chunk_id].get("content", ""))) for chunk_id in chunk_ids]
    retriever = bm25s.BM25(k1=1.5, b=0.75)
    retriever.index(corpus, show_progress=False)
    rankings: dict[str, list[tuple[str, float]]] = {}
    for case in cases:
        case_id = str(case["id"])
        query_tokens = tokenize_identifiers(str(case.get("question", "")))
        if not query_tokens or not chunk_ids:
            rankings[case_id] = []
            continue
        indices, scores = retriever.retrieve(
            [query_tokens],
            k=min(RETRIEVAL_TOP_K, len(chunk_ids)),
            show_progress=False,
            sorted=True,
        )
        rankings[case_id] = [
            (chunk_ids[int(index)], float(score))
            for index, score in zip(indices[0], scores[0])
            if float(score) > 0.0
        ]
    return rankings


def _dense_rankings(
    chunks: dict[str, dict[str, Any]], matrix: np.ndarray, query_vectors: np.ndarray, cases: list[dict[str, Any]]
) -> dict[str, list[tuple[str, float]]]:
    chunk_ids = list(chunks)
    if matrix.shape[0] != len(chunk_ids):
        raise ValueError(f"Embedding/chunk mismatch: {matrix.shape[0]} vs {len(chunk_ids)}")
    rankings: dict[str, list[tuple[str, float]]] = {}
    for case_index, case in enumerate(cases):
        scores = matrix @ query_vectors[case_index]
        top_indices = np.argsort(-scores, kind="stable")[:RETRIEVAL_TOP_K]
        rankings[str(case["id"])] = [
            (chunk_ids[int(index)], float(scores[int(index)])) for index in top_indices
        ]
    return rankings


def _fact_present(case: dict[str, Any], chunk: dict[str, Any]) -> bool:
    fact = ws(str(case.get("gold_fact", "")))
    content = ws(str(chunk.get("content", "")))
    return bool(fact and fact in content)


def _score_case(
    case: dict[str, Any], ranked_ids: list[str], chunks: dict[str, dict[str, Any]]
) -> dict[str, float | int]:
    ranked = [str(chunk_id) for chunk_id in ranked_ids]
    records = [chunks.get(chunk_id, {}) for chunk_id in ranked]
    gold_file = str(case.get("gold_file", ""))
    result: dict[str, float | int] = {}
    for k in RESULT_KS:
        prefix = records[:k]
        result[f"file_hit@{k}"] = int(any(str(record.get("source_path", "")) == gold_file for record in prefix))
        result[f"fact@{k}"] = int(any(_fact_present(case, record) for record in prefix))
    fact_rank = next(
        (rank for rank, record in enumerate(records[:20], start=1) if _fact_present(case, record)),
        None,
    )
    result["fact_rr"] = 1.0 / fact_rank if fact_rank is not None else 0.0
    result["chars@5"] = sum(len(str(record.get("content", ""))) for record in records[:5])
    result["n_ranked"] = len(ranked)
    return result


def _score_rankings(
    cases: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    chunks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"case_id": str(case["id"]), **_score_case(case, rankings[str(case["id"])], chunks)}
        for case in cases
    ]


def _split_fact_cases(cases: list[dict[str, Any]], chunks: dict[str, dict[str, Any]]) -> int:
    return sum(
        not any(_fact_present(case, chunk) for chunk in chunks.values())
        for case in cases
    )


def _heading_localisation(
    cases: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    chunks: dict[str, dict[str, Any]],
) -> tuple[int, int, float]:
    located = bearing = 0
    for case in cases:
        heading_path = case.get("heading_path") or []
        if isinstance(heading_path, str):
            heading_path = [heading_path]
        target = ws(str(heading_path[-1])).casefold() if heading_path else ""
        if not target:
            continue
        for chunk_id in rankings[str(case["id"])][:5]:
            chunk = chunks.get(chunk_id, {})
            if not _fact_present(case, chunk):
                continue
            bearing += 1
            headings = chunk.get("heading_path") or []
            if isinstance(headings, str):
                headings_text = headings
            else:
                headings_text = " > ".join(str(part) for part in headings)
            searchable = ws(f"{headings_text} {chunk.get('content', '')}").casefold()
            located += target in searchable
    return located, bearing, located / bearing if bearing else 0.0


def _print_breakdowns(
    arm_name: str,
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> None:
    metric_keys = ("file_hit@1", "fact@5", "fact_rr")
    for field in ("qtype", "detected_kind"):
        groups = breakdown(rows, cases, by=field)
        print(f"\n{arm_name} by {field}:")
        print("| Group | " + " | ".join(metric_keys) + " |")
        print("|---|" + "|".join("---:" for _ in metric_keys) + "|")
        for group, summary in groups.items():
            cells = [f"{float(summary[key]['mean']):.3f}" for key in metric_keys]
            print("| " + group + " | " + " | ".join(cells) + " |")


def _evaluate_set(
    label: str,
    chunks: dict[str, dict[str, Any]],
    matrix: np.ndarray,
    query_vectors: np.ndarray,
    cases: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    logger.info("%s: building in-memory BM25 index and ranking %d queries", label, len(cases))
    dense = _dense_rankings(chunks, matrix, query_vectors, cases)
    sparse_pairs = _bm25_rankings(chunks, cases)
    sparse = {case_id: rank_bm25(hits, TOP_K) for case_id, hits in sparse_pairs.items()}
    dense_top = {case_id: rank_dense(hits, TOP_K) for case_id, hits in dense.items()}
    hybrid = {
        str(case["id"]): fuse_hybrid(
            dense[str(case["id"])], sparse_pairs[str(case["id"])], alpha=0.5, pool=15, top_k=TOP_K
        )
        for case in cases
    }
    return {
        "dense": {"rankings": dense_top, "rows": _score_rankings(cases, dense_top, chunks)},
        "bm25": {"rankings": sparse, "rows": _score_rankings(cases, sparse, chunks)},
        "hybrid": {"rankings": hybrid, "rows": _score_rankings(cases, hybrid, chunks)},
    }


def _print_reports(
    cases: list[dict[str, Any]],
    chunks_by_name: dict[str, dict[str, dict[str, Any]]],
    results: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
) -> None:
    arms = ("token", "md", "mdmerge")
    names = {arm: f"h9_{arm}" for arm in arms}
    for arm in arms:
        for method in ("hybrid", "dense", "bm25"):
            result_name = names[arm] if method == "hybrid" else f"{names[arm]}_{method}"
            save_results(result_name, results[arm][method]["rows"])

    print("\nHybrid metrics (means and bootstrap 95% CIs; paired Δ vs h9_token, * = CI excludes 0):")
    print(table([names["token"], names["md"], names["mdmerge"]], METRIC_KEYS))
    for method in ("dense", "bm25"):
        ref = f"{names['token']}_{method}"
        print(f"\n{method.upper()} metrics (paired Δ vs {ref}; * = CI excludes 0):")
        print(table([ref, f"{names['md']}_{method}", f"{names['mdmerge']}_{method}"], METRIC_KEYS))

    print("\nWhole-fact split rate:")
    for arm in arms:
        misses = _split_fact_cases(cases, chunks_by_name[arm])
        print(f"- {arm}: {misses}/{len(cases)} ({misses / max(len(cases), 1):.1%}) cases have no chunk containing the whole gold fact")

    print("\nHeading localisation among top-5 fact-bearing chunks:")
    print("| Set | Method | Localised / fact-bearing | Share |")
    print("|---|---|---:|---:|")
    for arm in arms:
        for method in ("hybrid", "dense", "bm25"):
            rankings = results[arm][method]["rankings"]
            located, bearing, share = _heading_localisation(cases, rankings, chunks_by_name[arm])
            print(f"| {arm} | {method} | {located}/{bearing} | {share:.3f} |")

    for arm in arms:
        _print_breakdowns(names[arm], results[arm]["hybrid"]["rows"], cases)


async def _run(prepare_only: bool = False) -> None:
    _setup_logging()
    cases = load_cases()
    if len(cases) != 120:
        logger.warning("Loaded %d cases (task expected 120)", len(cases))
    documents, path_by_doc = _load_documents()
    tokenizer_name, tokenizer, chunk_size, overlap = resolve_tokenizer(CHUNK_TOKEN_SIZE, CHUNK_OVERLAP_TOKENS)
    logger.info(
        "Using tokenizer=%s nominal_chunk_size=%d chunk_size=%d overlap=%d",
        tokenizer_name,
        CHUNK_TOKEN_SIZE,
        chunk_size,
        overlap,
    )

    token_chunks = _token_chunks()
    chunks_by_name: dict[str, dict[str, dict[str, Any]]] = {"token": token_chunks}
    chunks_by_name["md"] = _chunk_set(
        "md", MIN_TOKEN_SETTINGS["md"], documents, path_by_doc, tokenizer, tokenizer_name, chunk_size, overlap
    )
    chunks_by_name["mdmerge"] = _chunk_set(
        "mdmerge", MIN_TOKEN_SETTINGS["mdmerge"], documents, path_by_doc, tokenizer, tokenizer_name, chunk_size, overlap
    )
    stats = [_chunk_stats(name, chunks_by_name[name], tokenizer) for name in ("token", "md", "mdmerge")]
    _print_stats(stats)
    if prepare_only:
        logger.info("Prepared chunk sets; skipping embedding and retrieval (--prepare-only)")
        return

    embedder = GemmaEmbedder()
    query_vectors = await _query_embeddings(cases, embedder)
    results: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for arm in ("token", "md", "mdmerge"):
        matrix = await _embed_corpus(arm, chunks_by_name[arm], embedder)
        results[arm] = _evaluate_set(arm, chunks_by_name[arm], matrix, query_vectors, cases)
    _print_reports(cases, chunks_by_name, results)
    logger.info("H9 completed; all outputs are under %s", EXP)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="re-chunk and print corpus statistics without loading the embedding model",
    )
    args = parser.parse_args()
    asyncio.run(_run(prepare_only=args.prepare_only))


if __name__ == "__main__":
    main()