"""Russian retrieval experiment helpers for the October 2026 search study."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

RU_DIR = Path(__file__).resolve().parent
EXP_DIR = RU_DIR.parent
ML_DIR = EXP_DIR / "ml"
REPO_DIR = RU_DIR.parents[2]
SOURCE_INDEX = Path("/tmp/ru-exp/index-token")
WORK_INDEX = Path("/tmp/ru-exp/index-token-work")
BM25_CACHE = Path("/tmp/ru-exp/bm25_token")
SIDECAR_DIR = SOURCE_INDEX / "dense_sidecar" / "Qwen_Qwen3-Embedding-0.6B"
CASES_FILE = RU_DIR / "cases_ru.yaml"
RESULTS_DIR = RU_DIR / "results"
CACHE_DIR = RU_DIR / "cache"
CELLS = ("A", "B", "C", "D")
METRICS = ("fact@1", "fact@3", "fact@5", "fact_rr")
ALL_CHANNELS = ("gemma", "qwen", "bm25", "bm25ru")

for _path in (REPO_DIR, EXP_DIR, ML_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HARS_MEMORY_LOG_LEVEL"] = "ERROR"
os.environ["HARS_MEMORY_LOG_FILE"] = "/tmp/ru-exp/hars-memory.log"
os.environ["HARS_MEMORY_EVENT_LOG_FILE"] = "/tmp/ru-exp/hars-memory-events.jsonl"

# cands imports mlib by its historical top-level name; load it from ml first.
importlib.import_module("mlib")
cands = importlib.import_module("cands")
common = importlib.import_module("common")
metrics = importlib.import_module("metrics")
fuse_ml = importlib.import_module("fuse_ml")

cands.CLONE = WORK_INDEX
cands.BM25_CACHE = BM25_CACHE
cands.SIDECAR_DIR = SIDECAR_DIR
cands.ML = RU_DIR
cands._SIDECAR_CACHE_SUFFIX = ""
common.COST_FILE = RU_DIR / "cost.json"
common.BUDGET_USD = 2.0


def load_cases() -> list[dict[str, Any]]:
    cases = common.load_cases(CASES_FILE)
    if not cases:
        raise ValueError(f"No evaluation cases in {CASES_FILE}")
    unknown = sorted({str(case.get("cell", "")) for case in cases} - set(CELLS))
    if unknown:
        raise ValueError(f"Unexpected case cells: {unknown}")
    return cases


def load_chunks() -> dict[str, dict[str, Any]]:
    path = cands.CLONE / "kv_store_text_chunks.json"
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a chunk-id mapping in {path}")
    return data


def build_candidates(queries: list[str], tag: str = "ru_token") -> dict[str, dict[str, Any]]:
    if cands.CLONE == SOURCE_INDEX:
        raise RuntimeError("Refusing to initialize LightRAG against the read-only source index")
    if not (cands.CLONE / "kv_store_text_chunks.json").is_file():
        raise FileNotFoundError(
            f"Working index copy is missing: {cands.CLONE}; copy {SOURCE_INDEX} first"
        )
    logging.disable(logging.INFO)
    try:
        transformers_logging = importlib.import_module("transformers.utils.logging")
        transformers_logging.disable_progress_bar()
    except ImportError:
        pass
    return cands.build_cands(queries, tag=tag)


def unique_queries(cases: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(str(case["question"]) for case in cases))


def fact_hit(case: dict[str, Any], ranked_ids: list[str], chunks: dict[str, Any]) -> bool:
    clean_case = {**case, "gold_chunk_ids": []}
    count = max(1, len(ranked_ids))
    scored = metrics.score_case(clean_case, ranked_ids, chunks, ks=(count,))
    return bool(scored.get(f"fact@{count}", 0))


def oracle_rows(
    cases: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    chunks: dict[str, Any],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for case in cases:
        candidate = candidates[str(case["question"])]
        union = list(
            dict.fromkeys(
                str(chunk_id)
                for channel in ALL_CHANNELS
                for chunk_id, _ in candidate.get(channel, [])[:100]
            )
        )
        result[str(case["id"])] = int(fact_hit(case, union, chunks))
    return result


def evaluate_profile(
    profile: str,
    cases: list[dict[str, Any]],
    chunks: dict[str, Any],
    rankings: dict[str, list[str]],
    oracle: dict[str, int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        if case_id not in rankings:
            raise KeyError(f"Missing ranking for case {case_id}")
        ranked_ids = [str(chunk_id) for chunk_id in rankings[case_id]]
        # The experiment defines a fact hit by normalized gold-fact text only.
        clean_case = {**case, "gold_chunk_ids": []}
        scored = metrics.score_case(clean_case, ranked_ids, chunks, ks=(1, 3, 5))
        rows.append(
            {
                "case_id": case_id,
                "cell": str(case["cell"]),
                "pair_id": str(case.get("pair_id") or ""),
                "q_lang": str(case.get("q_lang") or ""),
                **{metric: float(scored[metric]) for metric in METRICS},
                "oracle_recall@100": int(oracle[case_id]),
                "ranked_ids": ranked_ids,
            }
        )
    return {"rows": rows, "summary": summarize_rows(rows)}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("cell") or "(missing)")].append(row)
    result = {cell: summarize_cell(grouped.get(cell, [])) for cell in CELLS}
    result["overall"] = summarize_cell(rows)
    return result


def summarize_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(rows)}
    for metric in (*METRICS, "oracle_recall@100"):
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        lo, hi = common.boot_ci(values)
        summary[metric] = {
            "mean": sum(values) / len(values) if values else 0.0,
            "lo": lo,
            "hi": hi,
        }
    return summary


def paired_diff(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    keys: tuple[str, ...] = METRICS,
    group_by_cell: bool = True,
) -> dict[str, dict[str, dict[str, Any]]]:
    b_by_id = {str(row["case_id"]): row for row in rows_b}
    paired: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row_a in rows_a:
        row_b = b_by_id.get(str(row_a["case_id"]))
        if row_b is not None:
            group = str(row_a.get("cell") or "overall") if group_by_cell else "overall"
            paired[group].append((row_a, row_b))
            paired["overall"].append((row_a, row_b)) if group_by_cell else None

    groups = (*CELLS, "overall") if group_by_cell else ("overall",)
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for group in groups:
        pairs = paired.get(group, [])
        result[group] = {}
        for key in keys:
            values_a = [float(a[key]) for a, _ in pairs]
            values_b = [float(b[key]) for _, b in pairs]
            mean, lo, hi = common.paired_diff_ci(values_a, values_b)
            result[group][key] = {
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "n": len(pairs),
                "significant": lo > 0 or hi < 0,
            }
    return result


def pair_id_diff(
    rows: list[dict[str, Any]],
    cell_a: str,
    cell_b: str,
    keys: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        cell = str(row.get("cell") or "")
        if pair_id and cell in {cell_a, cell_b}:
            by_pair[pair_id][cell] = row
    pairs = [value for value in by_pair.values() if cell_a in value and cell_b in value]
    result: dict[str, dict[str, Any]] = {}
    for key in keys:
        values_a = [float(pair[cell_a][key]) for pair in pairs]
        values_b = [float(pair[cell_b][key]) for pair in pairs]
        mean, lo, hi = common.paired_diff_ci(values_a, values_b)
        result[key] = {
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "n": len(pairs),
            "significant": lo > 0 or hi < 0,
        }
    return result


def save_step(step: str, data: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{step.lower()}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def print_table(step: str, data: dict[str, Any]) -> None:
    print(f"\n{step}")
    profiles = data.get("profiles", {})
    if not profiles:
        print(f"Status: {data.get('status', 'no profiles evaluated')}")
        return
    print(
        "| Profile | Cell | n | fact@1 mean [CI] | fact@3 mean [CI] | "
        "fact@5 mean [CI] | fact_rr mean | oracle@100 mean [CI] |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for profile, value in profiles.items():
        for cell in (*CELLS, "overall"):
            stats = value["summary"][cell]

            def fmt(key: str) -> str:
                return f"{stats[key]['mean']:.3f} [{stats[key]['lo']:.3f}, {stats[key]['hi']:.3f}]"

            print(
                f"| {profile} | {cell} | {stats['n']} | {fmt('fact@1')} | {fmt('fact@3')} | "
                f"{fmt('fact@5')} | {stats['fact_rr']['mean']:.3f} | "
                f"{fmt('oracle_recall@100')} |"
            )


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def twin_cache_key(query: str, language: str) -> str:
    return hashlib.sha256(f"{language}\0{query}".encode("utf-8")).hexdigest()


__all__ = [
    "ALL_CHANNELS",
    "BM25_CACHE",
    "CACHE_DIR",
    "CASES_FILE",
    "CELLS",
    "METRICS",
    "RESULTS_DIR",
    "RU_DIR",
    "SIDECAR_DIR",
    "SOURCE_INDEX",
    "WORK_INDEX",
    "atomic_json",
    "build_candidates",
    "cands",
    "common",
    "evaluate_profile",
    "fact_hit",
    "fuse_ml",
    "load_cases",
    "load_chunks",
    "oracle_rows",
    "pair_id_diff",
    "paired_diff",
    "print_table",
    "save_step",
    "summarize_rows",
    "twin_cache_key",
    "unique_queries",
]
