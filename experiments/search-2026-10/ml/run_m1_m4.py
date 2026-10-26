#!/usr/bin/env python3
"""Run offline multilingual retrieval experiments M1–M4."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ML = Path(__file__).resolve().parent
EXP = ML.parent
REPO = ML.parents[2]
OLD_CASES_PATH = EXP / "cases.yaml"
NEW_CASES_PATH = ML / "cases_ml.yaml"
SIDECAR_META = Path("/tmp/search-exp/ml/sidecar_qwen3/meta.json")
POLL_SECONDS = 30
MAX_WAIT_SECONDS = 45 * 60
TOP_K = 20
WEIGHT_OPTIONS = (
    (1 / 3, 1 / 3, 1 / 3),
    (0.25, 0.25, 0.5),
    (0.2, 0.4, 0.4),
    (0.0, 0.5, 0.5),
    (0.15, 0.35, 0.5),
)

for path in (REPO, EXP, ML):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cands import build_cands, russian_stem, tokenize_russian  # noqa: E402
from evalml import evaluate_profile, load_chunks  # noqa: E402
from fuse_ml import fuse_prod, rrf_n, wsum_n  # noqa: E402
from report_md import render_report  # noqa: E402


def _wait_for_inputs() -> None:
    _wait_for_paths((NEW_CASES_PATH, SIDECAR_META, ML / "mlib.py"))


def _wait_for_paths(required: tuple[Path, ...]) -> None:
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for concurrent inputs: " + ", ".join(missing))
        print(f"Waiting for concurrent inputs ({remaining / 60:.1f} min remain): {', '.join(missing)}")
        time.sleep(min(POLL_SECONDS, remaining))


def _load_case_file(path: Path) -> list[dict[str, Any]]:
    loaded = yaml.safe_load(path.read_text("utf-8"))
    cases = loaded.get("cases", []) if isinstance(loaded, dict) else loaded
    if not isinstance(cases, list):
        raise ValueError(f"Expected list of cases in {path}")
    return [dict(case) for case in cases]


def load_all_cases() -> list[dict[str, Any]]:
    new_cases = _load_case_file(NEW_CASES_PATH)
    old_cases = _load_case_file(OLD_CASES_PATH)
    for case in new_cases:
        if case.get("cell") not in {"a", "b", "c", "d", "e"}:
            raise ValueError(f"Unexpected multilingual case cell in {case.get('id')}: {case.get('cell')}")
    for case in old_cases:
        lang = str(case.get("lang", "")).lower()
        case["cell"] = "old_ru" if lang == "ru" else "old_en"
    cases = new_cases + old_cases
    case_ids = [str(case["id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Case IDs must be unique across old and multilingual sets")
    return cases


def _sanity_tokenizer() -> None:
    samples = ("машинами", "настройками", "работающего", "обновления")
    print("Russian tokenizer sanity:")
    for word in samples:
        print(f"  {word} -> {russian_stem(word)}")
    identifier = "phase_c1_lora_safe A2S32"
    production = __import__("hars_memory.retrieval.tokenizer", fromlist=["tokenize_identifiers"])
    base_tokens = production.tokenize_identifiers(identifier)
    ru_tokens = tokenize_russian(identifier)
    print(f"  Latin/identifiers unchanged: {base_tokens == ru_tokens} ({identifier})")
    if base_tokens != ru_tokens:
        raise AssertionError("Russian tokenization changed the production Latin/identifier tokens")


def _rank_channel(channel: str):
    return lambda _case, cands: [chunk_id for chunk_id, _ in cands[channel][:TOP_K]]


def _prod_rank(dense_key: str, sparse_key: str):
    return lambda _case, cands: fuse_prod(
        cands[dense_key], cands[sparse_key], alpha=0.5, pool=15, top_k=TOP_K
    )


def _three_way(channels: tuple[str, str, str], weights: tuple[float, float, float]):
    names = ("gemma", "qwen", "bm25")
    return lambda _case, cands: wsum_n(
        {name: cands[channel] for name, channel in zip(names, channels)},
        dict(zip(names, weights)),
        pool=20,
        norm="minmax",
        top_k=TOP_K,
    )


def _mean(rows: list[dict[str, Any]], metric: str) -> float:
    return sum(float(row.get(metric, 0.0)) for row in rows) / max(len(rows), 1)


def _make_fold_split(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, case in enumerate(cases):
        key = str(case.get("pair_id") or f"case:{case['id']}:{index}")
        groups.setdefault(key, []).append(case)
    keys = sorted(groups)
    random.Random(20261002).shuffle(keys)
    folds: tuple[list[dict[str, Any]], list[dict[str, Any]]] = ([], [])
    for index, key in enumerate(keys):
        folds[index % 2].extend(groups[key])
    return folds


def _rank_weight_candidates(
    cases: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    channels: tuple[str, str, str] = ("gemma", "qwen", "bm25"),
) -> tuple[tuple[float, float, float], list[dict[str, Any]]]:
    scores: list[tuple[float, tuple[float, float, float], list[dict[str, Any]]]] = []
    for weights in WEIGHT_OPTIONS:
        rows = []
        ranker = _three_way(channels, weights)
        for case in cases:
            query_cands = candidates[str(case["question"])]
            ranked = ranker(case, query_cands)
            from metrics import score_case

            rows.append(score_case(case, ranked, chunks))
        scores.append((_mean(rows, "fact_rr"), weights, rows))
    return max(scores, key=lambda item: (item[0], tuple(-v for v in item[1])))[1:]


def _print_cv(
    cases: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
) -> tuple[float, float, float]:
    fold_a, fold_b = _make_fold_split(cases)
    selected: list[tuple[float, float, float]] = []
    heldout_scores: list[float] = []
    for train, test, label in ((fold_a, fold_b, "A→B"), (fold_b, fold_a, "B→A")):
        weights, _ = _rank_weight_candidates(train, candidates, chunks)
        test_rows = []
        ranker = _three_way(("gemma", "qwen", "bm25"), weights)
        for case in test:
            ranked = ranker(case, candidates[str(case["question"])])
            from metrics import score_case

            test_rows.append(score_case(case, ranked, chunks))
        heldout = _mean(test_rows, "fact_rr")
        selected.append(weights)
        heldout_scores.append(heldout)
        print(f"  M3 CV {label}: train weights={weights}, held-out fact_rr={heldout:.3f} (n={len(test_rows)})")
    count = {weights: selected.count(weights) for weights in WEIGHT_OPTIONS}
    selected_best = max(
        WEIGHT_OPTIONS,
        key=lambda weights: (
            count[weights],
            sum(heldout_scores[i] for i, selected_weights in enumerate(selected) if selected_weights == weights),
            weights,
        ),
    )
    print(f"  M3 cross-validated M4 weights: {selected_best}")
    return selected_best


def _profile(
    experiment: str,
    profile: str,
    cases: list[dict[str, Any]],
    cands: dict[str, dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    ranker: Any,
    baseline: str | None,
    baseline_data: dict[str, Any] | None,
) -> dict[str, Any]:
    result = evaluate_profile(
        experiment,
        profile,
        cases,
        cands,
        chunks,
        ranker,
        baseline_name=baseline,
        baseline_rows=(baseline_data or {}).get("rows"),
    )
    return result


def _run_experiment(
    experiment: str,
    definitions: list[tuple[str, Any]],
    cases: list[dict[str, Any]],
    cands: dict[str, dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    baseline_profile: str,
    baseline_data: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    started = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    for profile, ranker in definitions:
        results[profile] = _profile(
            experiment,
            profile,
            cases,
            cands,
            chunks,
            ranker,
            baseline_profile if profile != baseline_profile else None,
            baseline_data if profile != baseline_profile else None,
        )
        if profile == baseline_profile:
            baseline_data = results[profile]
    print(render_report(experiment))
    print(f"{experiment} runtime: {time.perf_counter() - started:.1f}s")
    return results


def run(cases: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    _sanity_tokenizer()
    chunks = load_chunks()
    queries = [str(case["question"]) for case in cases]
    cands = build_cands(queries, tag="all_cases")
    for channel in ("gemma", "qwen", "bm25", "bm25ru"):
        timing_key = f"t_{channel}_ms"
        average = sum(float(cands[query].get(timing_key, 0.0)) for query in set(queries)) / max(len(set(queries)), 1)
        print(f"Candidate retrieval {channel}: {average:.1f}ms/query average")

    m1_defs = [
        ("base_hybrid", _prod_rank("gemma", "bm25")),
        ("gemma_dense", _rank_channel("gemma")),
        ("bm25", _rank_channel("bm25")),
        ("qwen_dense", _rank_channel("qwen")),
        (
            "oracle_union",
            lambda _case, candidate: list(
                dict.fromkeys(
                    [
                        chunk_id
                        for channel in ("gemma", "qwen", "bm25")
                        for chunk_id, _ in candidate[channel]
                    ]
                )
            ),
        ),
    ]
    m1 = _run_experiment("M1", m1_defs, cases, cands, chunks, "base_hybrid")

    m2_defs = [
        ("gemma_dense", _rank_channel("gemma")),
        ("qwen_dense", _rank_channel("qwen")),
        ("qwen_hybrid", _prod_rank("qwen", "bm25")),
    ]
    m2 = _run_experiment("M2", m2_defs, cases, cands, chunks, "gemma_dense")

    m3_defs: list[tuple[str, Any]] = [
        ("rrf_k60", lambda _case, c: rrf_n({k: c[k] for k in ("gemma", "qwen", "bm25")}, {k: 1 / 3 for k in ("gemma", "qwen", "bm25")}, k=60, pool=20, top_k=TOP_K)),
        ("rrf_k20", lambda _case, c: rrf_n({k: c[k] for k in ("gemma", "qwen", "bm25")}, {k: 1 / 3 for k in ("gemma", "qwen", "bm25")}, k=20, pool=20, top_k=TOP_K)),
    ]
    for weights in WEIGHT_OPTIONS:
        profile = "wsum_" + "_".join(f"{weight:.2f}" for weight in weights)
        m3_defs.append((profile, _three_way(("gemma", "qwen", "bm25"), weights)))
    m3 = _run_experiment("M3", m3_defs, cases, cands, chunks, "wsum_0.33_0.33_0.33")
    best_weights = _print_cv(cases, cands, chunks)
    best_profile = "best_3way_cv"
    best_ranker = _three_way(("gemma", "qwen", "bm25"), best_weights)
    m3[best_profile] = _profile(
        "M3", best_profile, cases, cands, chunks, best_ranker, "wsum_0.33_0.33_0.33", m3["wsum_0.33_0.33_0.33"]
    )
    print(f"M3 cross-validated best 3-way profile: {best_profile} weights={best_weights}")
    print(render_report("M3"))

    # M4 compares the Russian-aware tokenizer with production BM25, and uses
    # the cross-validated M3 weights for its three-channel transfer.
    m4_defs = [
        ("bm25", _rank_channel("bm25")),
        ("bm25ru", _rank_channel("bm25ru")),
        ("base_hybrid_ru", _prod_rank("gemma", "bm25ru")),
        ("qwen_hybrid_ru", _prod_rank("qwen", "bm25ru")),
        ("three_way_ru", _three_way(("gemma", "qwen", "bm25ru"), best_weights)),
    ]
    _run_experiment("M4", m4_defs, cases, cands, chunks, "bm25")
    print(f"Total runner runtime: {time.perf_counter() - started:.1f}s")
    return {"M1": m1, "M2": m2, "M3": m3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-only", action="store_true", help="interim run on cases.yaml only")
    args = parser.parse_args()
    if args.old_only:
        _wait_for_paths((SIDECAR_META, ML / "mlib.py"))
        old_cases = _load_case_file(OLD_CASES_PATH)
        for case in old_cases:
            case["cell"] = "old_ru" if str(case.get("lang", "")).lower() == "ru" else "old_en"
        run(old_cases)
    else:
        _wait_for_inputs()
        run(load_all_cases())


if __name__ == "__main__":
    main()