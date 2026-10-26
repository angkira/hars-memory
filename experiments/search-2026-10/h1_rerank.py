"""H1: Qwen3 local reranker on hybrid candidate pools."""

from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from typing import Any

from common import EXP, load_cases
from metrics import breakdown, load_chunks, save_results, score_case, table
from retr import build_candidates, fuse_hybrid

METRIC_KEYS = (
    "file_hit@1",
    "file_hit@3",
    "file_hit@5",
    "fact@1",
    "fact@3",
    "fact@5",
    "fact_rr",
    "chars@5",
)
RERANK_URL = "http://127.0.0.1:8081/v1/rerank"
DOC_CHAR_LIMIT = 2500
SCORE_CACHE_PATH = EXP / "cache" / "h1_scores.json"
INSTRUCT_PREFIX = (
    "<Instruct>: Given a technical question about an SRE knowledge base, "
    "retrieve the passage that answers it\n<Query>: "
)
TOP_K = 20
ALPHA = 0.5
CACHE_VERSION = 1


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _latency_stats(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    ordered = sorted(latencies)
    return {
        "n": float(len(ordered)),
        "mean": statistics.fmean(ordered),
        "median": float(statistics.median(ordered)),
        "p95": _percentile(ordered, 0.95),
    }


def _empty_cache() -> dict[str, Any]:
    return {"version": CACHE_VERSION, "variants": {}}


def _load_score_cache() -> dict[str, Any]:
    if not SCORE_CACHE_PATH.is_file():
        return _empty_cache()
    data = json.loads(SCORE_CACHE_PATH.read_text("utf-8"))
    if not isinstance(data, dict):
        return _empty_cache()

    # Canonical shape: {version, variants: {name: {case_id: entry}}}
    if isinstance(data.get("variants"), dict):
        cache = {
            "version": int(data.get("version") or CACHE_VERSION),
            "variants": {
                str(name): dict(entries)
                for name, entries in data["variants"].items()
                if isinstance(entries, dict)
            },
        }
    else:
        cache = _empty_cache()

    # Migrate flat top-level variant buckets written by earlier drafts.
    for key, value in data.items():
        if key in {"version", "variants"} or not isinstance(value, dict):
            continue
        bucket = cache["variants"].setdefault(str(key), {})
        for case_id, entry in value.items():
            if case_id in bucket or not isinstance(entry, dict):
                continue
            if "scores" in entry and ("chunk_ids" in entry or "signature" in entry):
                bucket[str(case_id)] = entry
    return cache


def _save_score_cache(cache: dict[str, Any]) -> None:
    SCORE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": int(cache.get("version") or CACHE_VERSION),
        "variants": cache.get("variants") or {},
    }
    SCORE_CACHE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _doc_text(chunks: dict[str, dict[str, Any]], chunk_id: str) -> str:
    content = str(chunks.get(chunk_id, {}).get("content", "") or "")
    return content[:DOC_CHAR_LIMIT]


def _input_signature(query: str, documents: list[str]) -> str:
    encoded = json.dumps(
        [query, documents], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scores_to_list(raw: Any, n: int) -> list[float] | None:
    if isinstance(raw, list) and len(raw) == n:
        return [float(x) for x in raw]
    if isinstance(raw, dict):
        try:
            by_index = {int(k): float(v) for k, v in raw.items()}
        except (TypeError, ValueError):
            return None
        if set(by_index) == set(range(n)):
            return [by_index[i] for i in range(n)]
    return None


def _post_rerank(query: str, documents: list[str], retries: int = 4) -> tuple[list[float], float]:
    payload = json.dumps({"model": "x", "query": query, "documents": documents}).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(retries):
        started = time.perf_counter()
        req = urllib.request.Request(
            RERANK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            latency = time.perf_counter() - started
            scores = [0.0] * len(documents)
            for item in body.get("results", []) or []:
                idx = int(item["index"])
                if 0 <= idx < len(scores):
                    scores[idx] = float(item.get("relevance_score", 0.0))
            return scores, latency
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            last_err = exc
            time.sleep(min(8.0, 0.75 * (2**attempt)))
    raise RuntimeError(f"rerank failed after {retries} attempts: {last_err}")


def _rerank_order(chunk_ids: list[str], scores: list[float]) -> list[str]:
    order = sorted(range(len(chunk_ids)), key=lambda i: (-scores[i], i))
    return [chunk_ids[i] for i in order]


def _full_ranked_list(original: list[str], reranked_head: list[str], top_k: int = TOP_K) -> list[str]:
    """Reranked head first, then remaining original order; truncate/pad to top_k."""
    seen: set[str] = set()
    out: list[str] = []
    for chunk_id in list(reranked_head) + list(original):
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        out.append(chunk_id)
        if len(out) >= top_k:
            break
    return out


def _get_or_score(
    cache: dict[str, Any],
    variant: str,
    case_id: str,
    query: str,
    chunk_ids: list[str],
    chunks: dict[str, dict[str, Any]],
) -> tuple[list[float], float, bool]:
    """Return (scores aligned to chunk_ids, latency_s, from_cache)."""
    documents = [_doc_text(chunks, cid) for cid in chunk_ids]
    signature = _input_signature(query, documents)
    bucket = cache.setdefault("variants", {}).setdefault(variant, {})
    entry = bucket.get(case_id)

    if isinstance(entry, dict):
        scores = _scores_to_list(entry.get("scores"), len(chunk_ids))
        same_sig = entry.get("signature") == signature
        same_ids = entry.get("chunk_ids") == chunk_ids
        if scores is not None and (same_sig or same_ids):
            latency = float(entry.get("latency_seconds", entry.get("latency_s", 0.0)) or 0.0)
            # Normalize entry shape while preserving the hit.
            bucket[case_id] = {
                "signature": signature,
                "chunk_ids": chunk_ids,
                "scores": {str(i): scores[i] for i in range(len(scores))},
                "latency_seconds": latency,
            }
            return scores, latency, True

    scores, latency = _post_rerank(query, documents)
    bucket[case_id] = {
        "signature": signature,
        "chunk_ids": chunk_ids,
        "scores": {str(i): scores[i] for i in range(len(scores))},
        "latency_seconds": latency,
    }
    return scores, latency, False


def _print_breakdown(name: str, rows: list[dict[str, Any]], cases: list[dict[str, Any]], by: str) -> None:
    groups = breakdown(rows, cases, by=by)
    keys = ("file_hit@1", "fact@5", "fact_rr")
    print(f"\n{name} by {by}:")
    print("| Group | " + " | ".join(keys) + " |")
    print("|---|" + "|".join("---:" for _ in keys) + "|")
    for group, stats in groups.items():
        cells = [f"{float(stats[key]['mean']):.3f}" for key in keys]
        print("| " + group + " | " + " | ".join(cells) + " |")


def _best_variant(rows_by_name: dict[str, list[dict[str, Any]]], exclude: set[str]) -> str:
    best_name = ""
    best_key = (-1.0, -1.0, -1.0)
    for name, rows in rows_by_name.items():
        if name in exclude:
            continue
        n = len(rows) or 1
        fact5 = sum(float(r.get("fact@5", 0.0)) for r in rows) / n
        fact_rr = sum(float(r.get("fact_rr", 0.0)) for r in rows) / n
        hit1 = sum(float(r.get("file_hit@1", 0.0)) for r in rows) / n
        key = (fact5, fact_rr, hit1)
        if key > best_key:
            best_key = key
            best_name = name
    return best_name


async def _run() -> None:
    cases = load_cases()
    if not cases:
        raise RuntimeError("No cases found in experiments/search-2026-10/cases.yaml")

    chunks = load_chunks()
    candidates = await build_candidates(cases, pool=100, tag="base")
    score_cache = _load_score_cache()

    rows_by_method: dict[str, list[dict[str, Any]]] = {
        "h1_none": [],
        "h1_qwen3_top20": [],
        "h1_qwen3_top20_instr": [],
        "h1_qwen3_top10": [],
        "h1_qwen3_top30": [],
    }
    latencies: dict[str, list[float]] = {
        "h1_qwen3_top20": [],
        "h1_qwen3_top20_instr": [],
        "h1_qwen3_top10": [],
        "h1_qwen3_top30": [],
    }
    cache_hits = {name: 0 for name in latencies}
    cache_misses = {name: 0 for name in latencies}

    # Task: MiniLM not available offline.
    print("MiniLM unavailable offline", flush=True)

    dirty = False
    started_all = time.perf_counter()
    for idx, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        question = str(case["question"])
        record = candidates[case_id]
        dense = record["dense"]
        sparse = record["bm25"]

        hybrid20 = fuse_hybrid(dense, sparse, alpha=ALPHA, pool=20, top_k=20)

        # (a) none = hybrid ranking of the same candidate set (full top-20 list).
        rows_by_method["h1_none"].append(
            {"case_id": case_id, **score_case(case, hybrid20, chunks)}
        )

        variants: list[tuple[str, list[str], str, list[str]]] = [
            # name, pool_ids to score, query, original list used for tail fill
            ("h1_qwen3_top20", hybrid20, question, hybrid20),
            ("h1_qwen3_top10", hybrid20[:10], question, hybrid20),
        ]

        for name, pool_ids, query, original in variants:
            scores, latency, from_cache = _get_or_score(
                score_cache, name, case_id, query, pool_ids, chunks
            )
            if from_cache:
                cache_hits[name] += 1
            else:
                cache_misses[name] += 1
                dirty = True
            latencies[name].append(latency)
            reranked = _rerank_order(pool_ids, scores)
            ranked = _full_ranked_list(original, reranked, top_k=TOP_K)
            rows_by_method[name].append(
                {"case_id": case_id, **score_case(case, ranked, chunks)}
            )

        if dirty:
            _save_score_cache(score_cache)
            dirty = False

        if idx % 5 == 0 or idx == len(cases):
            elapsed = time.perf_counter() - started_all
            miss_total = sum(cache_misses.values())
            hit_total = sum(cache_hits.values())
            print(
                f"progress {idx}/{len(cases)} elapsed={elapsed:.0f}s "
                f"cache_hit={hit_total} cache_miss={miss_total}",
                flush=True,
            )

    if dirty:
        _save_score_cache(score_cache)

    for name, rows in rows_by_method.items():
        save_results(name, rows)

    names = [
        "base_hybrid",
        "h1_none",
        "h1_qwen3_top20",
        "h1_qwen3_top10",
    ]
    print(
        "\nH1 rerank metrics (means with bootstrap 95% CIs; "
        "Δ rows compare against base_hybrid):",
        flush=True,
    )
    print(table(names, METRIC_KEYS), flush=True)

    print("\nQwen3 rerank latency (seconds per query):", flush=True)
    print("| Variant | n | mean | median | p95 | cache_hit | cache_miss |", flush=True)
    print("|---|---:|---:|---:|---:|---:|---:|", flush=True)
    for name in (
        "h1_qwen3_top20",
        "h1_qwen3_top10",
    ):
        stats = _latency_stats(latencies[name])
        print(
            f"| {name} | {int(stats['n'])} | {stats['mean']:.3f} | "
            f"{stats['median']:.3f} | {stats['p95']:.3f} | "
            f"{cache_hits[name]} | {cache_misses[name]} |",
            flush=True,
        )

    best = _best_variant(rows_by_method, exclude={"h1_none"}) or "h1_qwen3_top20"
    print(f"\nBest Qwen variant by fact@5/fact_rr/file_hit@1: {best}", flush=True)
    _print_breakdown(best, rows_by_method[best], cases, "lang")
    _print_breakdown(best, rows_by_method[best], cases, "qtype")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
