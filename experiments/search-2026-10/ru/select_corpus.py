from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hars_memory.ingest.walker import walk
from hars_memory.server.lightrag_init import resolve_tokenizer


SUPPORT_ROOT = Path.home() / "Work" / "sre-support-pier"
SOURCE_ROOTS = (
    SUPPORT_ROOT / "kb",
    SUPPORT_ROOT / "repos" / "sre-docs" / "docs",
)
OUTPUT_ROOT = Path("/tmp/ru-exp")
CORPUS_ROOT = OUTPUT_ROOT / "corpus"
MANIFEST_PATH = OUTPUT_ROOT / "corpus_manifest.json"
STATS_PATH = OUTPUT_ROOT / "step1_stats.json"

MIN_CHARS = 300
MAX_CHARS = 12_000
MAX_MIXED_DOCS = 15
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


@dataclass(frozen=True)
class Candidate:
    path: Path
    rel: str
    text: str
    stratum: str
    ratio: float
    chars: int


def cyrillic_ratio(text: str) -> float:
    alphabetic = [char for char in text if char.isalpha()]
    if not alphabetic:
        return 0.0
    cyrillic = sum("\u0400" <= char <= "\u052f" and char.isalpha() for char in alphabetic)
    return cyrillic / len(alphabetic)


def classify(ratio: float) -> str:
    if ratio > 0.5:
        return "ru"
    if 0.2 <= ratio <= 0.5:
        return "mixed"
    if ratio < 0.1:
        return "en"
    return "other"


def enumerate_candidates() -> list[Candidate]:
    documents, _ = walk(paths=list(SOURCE_ROOTS))
    candidates: dict[str, Candidate] = {}
    for document in documents:
        path = Path(document.source_path).resolve()
        if path.suffix.lower() != ".md":
            continue
        try:
            rel = path.relative_to(SUPPORT_ROOT.resolve()).as_posix()
        except ValueError:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        chars = len(text)
        if not MIN_CHARS <= chars <= MAX_CHARS:
            continue
        ratio = cyrillic_ratio(text)
        candidates[rel] = Candidate(
            path=path,
            rel=rel,
            text=text,
            stratum=classify(ratio),
            ratio=ratio,
            chars=chars,
        )
    return sorted(candidates.values(), key=lambda candidate: candidate.rel)


def evenly_spread(items: list[Candidate], limit: int) -> list[Candidate]:
    ordered = sorted(items, key=lambda candidate: (candidate.ratio, candidate.rel))
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[(len(ordered) - 1) // 2]]
    last = len(ordered) - 1
    return [ordered[index * last // (limit - 1)] for index in range(limit)]


def token_density(items: list[Candidate], tokenizer: Any) -> dict[str, int | float | None]:
    total_chars = sum(candidate.chars for candidate in items)
    total_tokens = sum(len(tokenizer.encode(candidate.text)) for candidate in items)
    return {
        "tokens": total_tokens,
        "chars": total_chars,
        "tokens_per_1000_chars": (total_tokens * 1000 / total_chars if total_chars else None),
    }


def chunk_stats(items: list[Candidate], tokenizer: Any) -> dict[str, int | float | None]:
    from hars_memory.ingest.chunker import lightrag_chunking_func
    from lightrag.operate import chunking_by_token_size

    token_chunks = 0
    markdown_chunk_tokens: list[int] = []
    for candidate in items:
        token_result = chunking_by_token_size(
            tokenizer,
            candidate.text,
            None,
            False,
            CHUNK_OVERLAP,
            CHUNK_SIZE,
        )
        token_chunks += len(token_result)

        markdown_result = lightrag_chunking_func(
            tokenizer,
            candidate.text,
            None,
            False,
            CHUNK_OVERLAP,
            CHUNK_SIZE,
        )
        markdown_chunk_tokens.extend(int(chunk["tokens"]) for chunk in markdown_result)

    chunk_count = len(markdown_chunk_tokens)
    return {
        "documents": len(items),
        "token_chunks_512_64": token_chunks,
        "markdown_chunks_512_64": chunk_count,
        "markdown_mean_tokens_per_chunk": (
            sum(markdown_chunk_tokens) / chunk_count if chunk_count else None
        ),
        "markdown_chunks_over_512_tokens": sum(
            token_count > CHUNK_SIZE for token_count in markdown_chunk_tokens
        ),
    }


def main() -> None:
    os.environ["HARS_MEMORY_CHUNK_MIN_TOKENS"] = "0"
    candidates = enumerate_candidates()
    by_stratum = {
        stratum: [candidate for candidate in candidates if candidate.stratum == stratum]
        for stratum in ("ru", "mixed", "en", "other")
    }
    selected_mixed = evenly_spread(by_stratum["mixed"], MAX_MIXED_DOCS)
    selected = sorted(
        by_stratum["ru"] + selected_mixed,
        key=lambda candidate: candidate.rel,
    )

    tokenizer_name, tokenizer_impl, _, _ = resolve_tokenizer(CHUNK_SIZE, CHUNK_OVERLAP)
    if tokenizer_name != "cl100k_base":
        raise RuntimeError(f"Expected production cl100k_base tokenizer, got {tokenizer_name!r}")
    from lightrag.utils import Tokenizer

    tokenizer = Tokenizer(tokenizer_name, tokenizer_impl)

    for candidate in selected:
        destination = CORPUS_ROOT / candidate.rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.path, destination)

    manifest = [
        {
            "rel": candidate.rel,
            "stratum": candidate.stratum,
            "ratio": candidate.ratio,
            "chars": candidate.chars,
        }
        for candidate in selected
    ]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    strata_stats: dict[str, dict[str, Any]] = {}
    for stratum, items in by_stratum.items():
        stats: dict[str, Any] = {
            "count": len(items),
            "total_chars": sum(candidate.chars for candidate in items),
        }
        if stratum in {"ru", "mixed", "en"}:
            stats["token_density"] = token_density(items, tokenizer)
        if stratum in {"ru", "mixed"}:
            chunk_items = by_stratum["ru"] if stratum == "ru" else selected_mixed
            stats["chunking"] = chunk_stats(chunk_items, tokenizer)
        strata_stats[stratum] = stats

    stats_output = {
        "tokenizer": tokenizer_name,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "markdown_min_tokens": 0,
        "mixed_selection_limit": MAX_MIXED_DOCS,
        "selected_counts": {
            "ru": len(by_stratum["ru"]),
            "mixed": len(selected_mixed),
            "total": len(selected),
        },
        "strata": strata_stats,
        "unclassified_ratio_range": "0.1 <= ratio < 0.2",
    }
    STATS_PATH.write_text(json.dumps(stats_output, ensure_ascii=False, indent=2) + "\n")

    print("stratum  docs  chars     tok/1k  tok_chunks  md_chunks  md_mean  md>512  copied")
    for stratum, stats in strata_stats.items():
        density = stats.get("token_density")
        chunks = stats.get("chunking")
        if chunks:
            tokens_per_1000 = density["tokens_per_1000_chars"]
            token_density_label = (
                f"{tokens_per_1000:>8.1f}" if tokens_per_1000 is not None else f"{('—'):>8}"
            )
            mean_tokens = chunks["markdown_mean_tokens_per_chunk"]
            markdown_mean_label = (
                f"{mean_tokens:>8.1f}" if mean_tokens is not None else f"{('—'):>8}"
            )
            print(
                f"{stratum:<7} {stats['count']:>5} {stats['total_chars']:>7} "
                f"{token_density_label} "
                f"{chunks['token_chunks_512_64']:>10} "
                f"{chunks['markdown_chunks_512_64']:>9} "
                f"{markdown_mean_label} "
                f"{chunks['markdown_chunks_over_512_tokens']:>7} "
                f"{stats['chunking']['documents']:>7}"
            )
        elif density:
            tokens_per_1000 = density["tokens_per_1000_chars"]
            token_density_label = (
                f"{tokens_per_1000:>8.1f}" if tokens_per_1000 is not None else f"{('—'):>8}"
            )
            print(
                f"{stratum:<7} {stats['count']:>5} {stats['total_chars']:>7} "
                f"{token_density_label} {('—'):>10} {('—'):>9} "
                f"{('—'):>8} {('—'):>7} {0:>7}"
            )
        else:
            print(f"{stratum:<7} {stats['count']:>5} {stats['total_chars']:>7} (uncategorized)")
    print(f"selected corpus: {len(selected)} docs ({len(selected_mixed)} mixed)")
    print(f"manifest: {MANIFEST_PATH}")
    print(f"stats: {STATS_PATH}")


if __name__ == "__main__":
    main()
