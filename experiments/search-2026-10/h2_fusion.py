"""Fusion and BM25 tokenization experiments using cached H2 candidates."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import bm25s

from common import EXP, load_cases
from hars_memory.retrieval.tokenizer import tokenize_identifiers
from metrics import load_chunks, save_results, score_case, summarize, table
from retr import fuse_hybrid, fuse_rrf, rank_bm25

TOP_K = 20
CANDIDATE_K = 100
METRIC_KEYS = ("file_hit@1", "fact@3", "fact@5", "fact_rr")
POOL_SIZES = (15, 30, 50)
ALPHAS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
STOPLIST = frozenset("the a of to is in and for on that this it with as are be by".split())
WHOLE_RUN_RE = re.compile(r"[A-Za-z0-9_\-:./]+")
WORD_RE = re.compile(r"\w+", re.UNICODE)
EDGE_STRIP_CHARS = "_-:./"


def _read_candidates(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    path = EXP / "cache" / "cands_base.json"
    data = json.loads(path.read_text("utf-8"))
    by_case = data.get("cases", data)
    if not isinstance(by_case, dict):
        raise ValueError(f"Expected a case-id mapping in {path}")
    candidates: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case["id"])
        if case_id not in by_case:
            raise KeyError(f"No cached retrieval candidates for case {case_id!r}")
        record = by_case[case_id]
        if "dense" not in record or "bm25" not in record:
            raise ValueError(f"Cached candidates for {case_id!r} need dense and bm25 lists")
        candidates[case_id] = {
            "dense": [(str(cid), float(score)) for cid, score in record["dense"]],
            "bm25": [(str(cid), float(score)) for cid, score in record["bm25"]],
        }
    return candidates


def _score_rankings(
    cases: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    chunks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"case_id": str(case["id"]), **score_case(case, rankings[str(case["id"])], chunks)}
        for case in cases
    ]


def _alpha_name(alpha: float, pool: int) -> str:
    value = f"{alpha:g}".replace(".", "p")
    return f"h2_alpha_{value}_p{pool}"


def _weight_name(weight: float) -> str:
    return f"{weight:g}".replace(".", "p")


def _weighted_rrf(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    dense_weight: float,
    bm25_weight: float,
    k: int,
    pool: int,
) -> list[str]:
    scores: dict[str, float] = {}
    for weight, hits in (
        (dense_weight, dense[:pool]),
        (bm25_weight, sparse[:pool]),
    ):
        for rank, (chunk_id, _) in enumerate(hits, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:TOP_K]


def _tokenizers() -> dict[str, Callable[[str], list[str]]]:
    def plain_words(text: str) -> list[str]:
        return [word.casefold() for word in WORD_RE.findall(text)]

    def whole_runs(text: str) -> list[str]:
        return [
            run.casefold()
            for raw in WHOLE_RUN_RE.findall(text)
            if (run := raw.strip(EDGE_STRIP_CHARS))
        ]

    def with_char4(text: str) -> list[str]:
        tokens = tokenize_identifiers(text)
        for word in WORD_RE.findall(text.casefold()):
            if len(word) >= 4:
                tokens.extend(f"char4:{word[i : i + 4]}" for i in range(len(word) - 3))
        return tokens

    def without_stopwords(text: str) -> list[str]:
        return [token for token in tokenize_identifiers(text) if token not in STOPLIST]

    return {
        "current": tokenize_identifiers,
        "plain_words": plain_words,
        "whole_runs": whole_runs,
        "current_char4": with_char4,
        "current_nostop": without_stopwords,
    }


def _build_bm25(
    chunks: dict[str, dict[str, Any]],
    tokenizer: Callable[[str], list[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[bm25s.BM25, list[str]]:
    chunk_ids = sorted(chunks)
    corpus = [tokenizer(str(chunks[cid].get("content", ""))) for cid in chunk_ids]
    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(corpus, show_progress=False)
    return retriever, chunk_ids


def _search_bm25(
    retriever: bm25s.BM25,
    chunk_ids: list[str],
    query: str,
    tokenizer: Callable[[str], list[str]],
) -> list[tuple[str, float]]:
    query_tokens = tokenizer(query)
    if not query_tokens or not chunk_ids:
        return []
    k = min(CANDIDATE_K, len(chunk_ids))
    indices, scores = retriever.retrieve([query_tokens], k=k, show_progress=False, sorted=True)
    return [
        (chunk_ids[int(index)], float(score))
        for index, score in zip(indices[0], scores[0])
        if float(score) > 0.0
    ]


def _run_tokenizer_variant(
    name: str,
    cases: list[dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    tokenizer: Callable[[str], list[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[str, str]:
    retriever, chunk_ids = _build_bm25(chunks, tokenizer, k1, b)
    bm25_rankings: dict[str, list[str]] = {}
    hybrid_rankings: dict[str, list[str]] = {}
    for case in cases:
        case_id = str(case["id"])
        sparse = _search_bm25(retriever, chunk_ids, str(case.get("question", "")), tokenizer)
        bm25_rankings[case_id] = rank_bm25(sparse, TOP_K)
        hybrid_rankings[case_id] = fuse_hybrid(
            candidates[case_id]["dense"], sparse, alpha=0.5, pool=15, top_k=TOP_K
        )

    bm25_name = f"h2_tok_{name}_bm25"
    hybrid_name = f"h2_tok_{name}_hybrid"
    save_results(bm25_name, _score_rankings(cases, bm25_rankings, chunks))
    save_results(hybrid_name, _score_rankings(cases, hybrid_rankings, chunks))
    return bm25_name, hybrid_name


def main() -> None:
    cases = load_cases()
    chunks = load_chunks()
    candidates = _read_candidates(cases)

    fusion_rows: dict[str, list[dict[str, Any]]] = {}
    alpha_rows: dict[tuple[float, int], list[dict[str, Any]]] = {}

    for alpha in ALPHAS:
        for pool in POOL_SIZES:
            name = _alpha_name(alpha, pool)
            rankings = {
                str(case["id"]): fuse_hybrid(
                    candidates[str(case["id"])]["dense"],
                    candidates[str(case["id"])]["bm25"],
                    alpha=alpha,
                    pool=pool,
                    top_k=TOP_K,
                )
                for case in cases
            }
            rows = _score_rankings(cases, rankings, chunks)
            save_results(name, rows)
            fusion_rows[name] = rows
            alpha_rows[(alpha, pool)] = rows

    for k in (60, 20):
        for pool in POOL_SIZES:
            name = f"h2_rrf_k{k}_p{pool}"
            rankings = {
                str(case["id"]): fuse_rrf(
                    candidates[str(case["id"])]["dense"],
                    candidates[str(case["id"])]["bm25"],
                    k=k,
                    pool=pool,
                    top_k=TOP_K,
                )
                for case in cases
            }
            rows = _score_rankings(cases, rankings, chunks)
            save_results(name, rows)
            fusion_rows[name] = rows

    weighted_configs = ((0.5, 1.0), (1.0, 0.5), (0.7, 1.0), (1.0, 0.7))
    for dense_weight, bm25_weight in weighted_configs:
        for k in (20, 60):
            for pool in POOL_SIZES:
                name = (
                    f"h2_wrrf_d{_weight_name(dense_weight)}"
                    f"_b{_weight_name(bm25_weight)}_k{k}_p{pool}"
                )
                rankings = {
                    str(case["id"]): _weighted_rrf(
                        candidates[str(case["id"])]["dense"],
                        candidates[str(case["id"])]["bm25"],
                        dense_weight,
                        bm25_weight,
                        k,
                        pool,
                    )
                    for case in cases
                }
                rows = _score_rankings(cases, rankings, chunks)
                save_results(name, rows)
                fusion_rows[name] = rows

    top_fusion_names = sorted(
        fusion_rows,
        key=lambda name: (
            -float(summarize(fusion_rows[name]).get("fact_rr", {}).get("mean", 0.0)),
            name,
        ),
    )[:10]
    print("### Fusion: ten highest fact_rr variants (paired Δ vs baseline; * = CI excludes 0)")
    print(table(["base_hybrid", *top_fusion_names], METRIC_KEYS))

    case_positions = {str(case["id"]): index for index, case in enumerate(cases)}
    cv_rows: list[dict[str, Any]] = []
    cv_picks: list[str] = []
    for train_parity, test_parity in ((0, 1), (1, 0)):
        train_cases = {
            str(case["id"]) for case in cases if case_positions[str(case["id"])] % 2 == train_parity
        }
        test_cases = {
            str(case["id"]) for case in cases if case_positions[str(case["id"])] % 2 == test_parity
        }
        best_key = max(
            alpha_rows,
            key=lambda key: (
                sum(
                    float(row["fact_rr"])
                    for row in alpha_rows[key]
                    if str(row["case_id"]) in train_cases
                )
                / max(1, len(train_cases)),
                -ALPHAS.index(key[0]),
                -POOL_SIZES.index(key[1]),
            ),
        )
        selected_name = _alpha_name(*best_key)
        cv_picks.append(
            f"{('even' if train_parity == 0 else 'odd')}→"
            f"{('odd' if test_parity == 1 else 'even')}: {selected_name}"
        )
        cv_rows.extend(row for row in alpha_rows[best_key] if str(row["case_id"]) in test_cases)

    save_results("h2_cv_alpha", cv_rows)
    print("### Alpha/pool 2-fold cross-validation")
    print("Picks: " + "; ".join(cv_picks))
    print(table(["base_hybrid", "h2_cv_alpha"], METRIC_KEYS))

    tokenizer_names: list[str] = []
    hybrid_names: list[str] = []
    for name, tokenizer in _tokenizers().items():
        bm25_name, hybrid_name = _run_tokenizer_variant(name, cases, chunks, candidates, tokenizer)
        tokenizer_names.append(bm25_name)
        hybrid_names.append(hybrid_name)

    for k1, b in ((1.5, 0.75), (1.2, 0.75), (0.9, 0.4), (2.0, 0.75), (1.2, 0.3)):
        name = f"k1_{_weight_name(k1)}_b{_weight_name(b)}"
        bm25_name, hybrid_name = _run_tokenizer_variant(
            name, cases, chunks, candidates, tokenize_identifiers, k1, b
        )
        tokenizer_names.append(bm25_name)
        hybrid_names.append(hybrid_name)

    print("### BM25 tokenization and parameters — sparse (paired Δ vs base_bm25)")
    print(table(["base_bm25", *tokenizer_names], METRIC_KEYS))
    print("### BM25 tokenization and parameters — hybrid (paired Δ vs base_hybrid)")
    print(table(["base_hybrid", *hybrid_names], METRIC_KEYS))


if __name__ == "__main__":
    main()
