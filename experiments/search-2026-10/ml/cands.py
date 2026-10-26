"""Offline candidate retrieval for the multilingual search experiments."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
ML = Path(__file__).resolve().parent
CLONE = Path("/tmp/search-exp/ml-clone")
BM25_CACHE = Path("/tmp/search-exp/ml/bm25_clone")
_DEFAULT_SIDECAR_DIR = Path("/tmp/search-exp/ml/sidecar_qwen3")
_SIDECAR_OVERRIDE = os.environ.get("ML_SIDECAR_DIR")
SIDECAR_DIR = Path(_SIDECAR_OVERRIDE) if _SIDECAR_OVERRIDE else _DEFAULT_SIDECAR_DIR
_SIDECAR_CACHE_SUFFIX = ""
if _SIDECAR_OVERRIDE:
    _sidecar_name = re.sub(r"[^A-Za-z0-9_-]+", "_", SIDECAR_DIR.name).strip("_") or "custom"
    _sidecar_hash = hashlib.sha256(str(SIDECAR_DIR).encode("utf-8")).hexdigest()[:8]
    _SIDECAR_CACHE_SUFFIX = f"_{_sidecar_name}_{_sidecar_hash}"
CANDIDATE_POOL = 100

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from hars_memory.retrieval.tokenizer import tokenize_identifiers  # noqa: E402

_RAG: Any | None = None
_BM25: Any | None = None
_BM25RU: Any | None = None
_QWEN: tuple[Any, Any, list[str], Any] | None = None
_TOKENIZER_LOCK = threading.RLock()
_CYRILLIC_WORD = re.compile(r"[А-Яа-яЁё]+")

_RU_SUFFIXES = tuple(
    sorted(
        {
            "иями", "ями", "ами", "его", "ого", "ему", "ому", "ими", "ыми",
            "ать", "ять", "ить", "еть", "уть", "ыть", "ешь", "ишь", "ете",
            "ите", "ут", "ют", "ат", "ят", "ет", "ит", "ал", "ял", "ил",
            "ла", "ли", "ло", "ет", "ют", "ешь", "ишь", "ый", "ий", "ой",
            "ая", "яя", "ое", "ее", "ые", "ие", "ого", "ему", "ыми", "ими",
            "ому", "ую", "юю", "ых", "их", "ым", "им", "ом", "ем", "а", "я",
            "ы", "и", "у", "ю", "е", "о", "ь", "й",
        },
        key=len,
        reverse=True,
    )
)


def russian_stem(word: str) -> str:
    """Apply a small suffix-stripping stemmer to a Cyrillic word."""
    normalized = word.lower().replace("ё", "е")
    for suffix in _RU_SUFFIXES:
        stem = normalized[: -len(suffix)] if suffix else normalized
        if normalized.endswith(suffix) and len(stem) >= 3:
            return stem
    return normalized


def tokenize_russian(text: str) -> list[str]:
    """Preserve production Latin/identifier tokens and stem Cyrillic words."""
    latin_text = _CYRILLIC_WORD.sub(" ", text)
    tokens = tokenize_identifiers(latin_text)
    tokens.extend(
        stem
        for word in _CYRILLIC_WORD.findall(text)
        if len(stem := russian_stem(word)) >= 5
    )
    return tokens


def _load_optional_env() -> None:
    env_file = REPO / "config" / "live-llm-e2e.env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text("utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _configure_runtime() -> None:
    _load_optional_env()
    os.environ["HARS_MEMORY_INDEX_DIR"] = str(CLONE)
    os.environ["HARS_MEMORY_EMBED_MODEL"] = "unsloth/embeddinggemma-300m"
    os.environ["HF_HUB_OFFLINE"] = "1"
    if not (CLONE / "kv_store_text_chunks.json").is_file():
        raise FileNotFoundError(f"Experiment clone is missing chunks: {CLONE}")


def _load_chunks() -> dict[str, dict[str, Any]]:
    path = CLONE / "kv_store_text_chunks.json"
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a chunk-id mapping in {path}")
    return data


async def _load_live_env_and_rag() -> tuple[Any, Any]:
    global _RAG, _BM25
    if _RAG is not None and _BM25 is not None:
        return _RAG, _BM25
    _configure_runtime()
    from hars_memory.retrieval.bm25_index import get_or_build_index
    from hars_memory.server.lightrag_init import create_lightrag

    rag = create_lightrag(working_dir=str(CLONE))
    await rag.initialize_storages()
    bm25, _ = await asyncio.to_thread(get_or_build_index, str(CLONE), str(BM25_CACHE))
    _RAG, _BM25 = rag, bm25
    return _RAG, _BM25


def _build_bm25ru() -> Any:
    global _BM25RU
    if _BM25RU is not None:
        return _BM25RU
    import bm25s
    import hars_memory.retrieval.bm25_index as bm25_module

    chunks_path = CLONE / "kv_store_text_chunks.json"
    chunks = _load_chunks()
    chunk_ids = sorted(chunks)
    with _TOKENIZER_LOCK:
        original = bm25_module.tokenize_identifiers
        try:
            bm25_module.tokenize_identifiers = tokenize_russian
            corpus_tokens = [tokenize_russian(str(chunks[cid].get("content", ""))) for cid in chunk_ids]
        finally:
            bm25_module.tokenize_identifiers = original
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    metadata = {
        str(cid): {
            "content": str(chunks[cid].get("content", "")),
            "file_path": str(chunks[cid].get("source_path", chunks[cid].get("file_path", ""))),
        }
        for cid in chunk_ids
    }
    _BM25RU = bm25_module.BM25SparseIndex(
        retriever=retriever,
        chunk_ids=chunk_ids,
        chunk_meta=metadata,
        source_mtime=chunks_path.stat().st_mtime,
    )
    return _BM25RU


def _bm25ru_search(index: Any, query: str, top_k: int) -> list[Any]:
    import hars_memory.retrieval.bm25_index as bm25_module

    with _TOKENIZER_LOCK:
        original = bm25_module.tokenize_identifiers
        try:
            bm25_module.tokenize_identifiers = tokenize_russian
            return index.search(query, top_k)
        finally:
            bm25_module.tokenize_identifiers = original


def _load_qwen() -> tuple[Any, list[str], Any, Any]:
    global _QWEN
    if _QWEN is not None:
        return _QWEN
    if str(ML) not in sys.path:
        sys.path.insert(0, str(ML))
    from mlib import QwenQueryEmbedder, dense_search, load_sidecar

    ids, vecs = load_sidecar(SIDECAR_DIR)
    metadata = json.loads((SIDECAR_DIR / "meta.json").read_text("utf-8"))
    expected_count = metadata.get("n")
    if expected_count is not None and int(expected_count) != len(ids):
        raise ValueError(
            f"Sidecar metadata count {expected_count} does not match {len(ids)} IDs in {SIDECAR_DIR}"
        )
    max_seq_length = int(metadata.get("max_seq_length", 512))
    if max_seq_length < 1:
        raise ValueError(f"Invalid max_seq_length={max_seq_length} in {SIDECAR_DIR / 'meta.json'}")
    embedder = QwenQueryEmbedder(max_seq_length=max_seq_length)
    _QWEN = (embedder, [str(chunk_id) for chunk_id in ids], vecs, dense_search)
    return _QWEN


async def _qwen_search(query: str, top_k: int = CANDIDATE_POOL) -> list[tuple[str, float]]:
    embedder, ids, vecs, dense_search = _load_qwen()
    query_vectors = await asyncio.to_thread(embedder.embed, [query])
    hits = dense_search(vecs, ids, query_vectors[0], k=top_k)
    return [(str(chunk_id), float(score)) for chunk_id, score in hits[:top_k]]


def _load_cache(tag: str) -> tuple[Path, dict[str, Any]]:
    path = ML / "cache" / f"cands_{tag}{_SIDECAR_CACHE_SUFFIX}.json"
    if not path.is_file():
        return path, {}
    data = json.loads(path.read_text("utf-8"))
    records = data.get("queries", {}) if isinstance(data, dict) else {}
    return path, records if isinstance(records, dict) else {}


def _save_cache(path: Path, records: dict[str, Any], tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"tag": tag, "queries": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _as_pairs(values: list[Any]) -> list[tuple[str, float]]:
    return [(str(chunk_id), float(score)) for chunk_id, score in values]


async def _build_candidates_async(queries: list[str], tag: str) -> dict[str, dict[str, Any]]:
    cache_path, cached = _load_cache(tag)
    channels = ("gemma", "qwen", "bm25", "bm25ru")
    pending = {
        query: {channel for channel in channels if channel not in cached.get(query, {})}
        for query in dict.fromkeys(queries)
    }
    pending = {query: missing for query, missing in pending.items() if missing}
    if pending:
        rag, bm25 = await _load_live_env_and_rag()
        bm25ru = _build_bm25ru() if any("bm25ru" in missing for missing in pending.values()) else None
        for query, missing in pending.items():
            record = cached.setdefault(query, {})
            if "gemma" in missing:
                before = time.perf_counter()
                gemma_raw = await rag.chunks_vdb.query(query, top_k=CANDIDATE_POOL)
                record["gemma"] = [
                    (str(hit["id"]), float(hit.get("distance", 0.0)))
                    for hit in gemma_raw
                    if hit.get("id") is not None
                ][:CANDIDATE_POOL]
                record["t_gemma_ms"] = (time.perf_counter() - before) * 1000
            if "qwen" in missing:
                before = time.perf_counter()
                record["qwen"] = await _qwen_search(query)
                record["t_qwen_ms"] = (time.perf_counter() - before) * 1000
            if "bm25" in missing:
                before = time.perf_counter()
                bm25_raw = await asyncio.to_thread(bm25.search, query, CANDIDATE_POOL)
                record["bm25"] = [
                    (hit.chunk_id, float(hit.score)) for hit in bm25_raw
                ][:CANDIDATE_POOL]
                record["t_bm25_ms"] = (time.perf_counter() - before) * 1000
            if "bm25ru" in missing:
                if bm25ru is None:
                    raise RuntimeError("Russian BM25 index was not initialized")
                before = time.perf_counter()
                bm25ru_raw = await asyncio.to_thread(_bm25ru_search, bm25ru, query, CANDIDATE_POOL)
                record["bm25ru"] = [
                    (hit.chunk_id, float(hit.score)) for hit in bm25ru_raw
                ][:CANDIDATE_POOL]
                record["t_bm25ru_ms"] = (time.perf_counter() - before) * 1000
            _save_cache(cache_path, cached, tag)

    return {
        query: {
            **cached[query],
            **{
                channel: _as_pairs(cached[query].get(channel, []))
                for channel in ("gemma", "qwen", "bm25", "bm25ru")
            },
        }
        for query in dict.fromkeys(queries)
    }


def build_cands(queries: list[str], tag: str) -> dict[str, dict[str, Any]]:
    """Return top-100 candidates per exact query, incrementally disk-cached."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_build_candidates_async(queries, tag))
    raise RuntimeError("build_cands is synchronous and cannot run inside an active event loop")


__all__ = ["build_cands", "russian_stem", "tokenize_russian"]