"""Property checks for Russian-aware BM25 tokenization on the ML corpus.

The production-corpus check needs the ML clone (``kv_store_text_chunks.json``),
which is not shipped with the repository: point ``SEARCH_EXP_ML_CLONE`` at it and
the test runs, otherwise it skips.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
ML = Path(__file__).resolve().parent
CLONE = Path(os.environ.get("SEARCH_EXP_ML_CLONE", "/tmp/search-exp/ml-clone"))

for path in (REPO, EXP, ML):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cands import russian_stem, tokenize_russian  # noqa: E402
from hars_memory.retrieval.tokenizer import tokenize_identifiers  # noqa: E402

_CYRILLIC_WORD = re.compile(r"[А-Яа-яЁё]+")
_CYRILLIC_TOKEN = re.compile(r"^[а-яё]+$")


def _load_cases(path: Path) -> list[dict[str, Any]]:
    loaded = yaml.safe_load(path.read_text("utf-8"))
    cases = loaded.get("cases", []) if isinstance(loaded, dict) else loaded
    if not isinstance(cases, list):
        raise ValueError(f"Expected a list of cases in {path}")
    return cases


def _latin_tokens(text: str) -> Counter[str]:
    return Counter(token for token in tokenize_russian(text) if not _CYRILLIC_TOKEN.fullmatch(token))


def test_latin_identifier_multiset_matches_production() -> None:
    chunks_path = CLONE / "kv_store_text_chunks.json"
    if not chunks_path.is_file():
        pytest.skip(
            f"ML corpus clone not available at {chunks_path} "
            "(set SEARCH_EXP_ML_CLONE to the clone directory to enable this test)"
        )
    chunks = json.loads(chunks_path.read_text("utf-8"))
    contents = [str(record.get("content", "")) for record in chunks.values()]
    if len(contents) < 200:
        raise AssertionError(f"Expected at least 200 chunks in {chunks_path}, got {len(contents)}")
    sample = random.Random(20261002).sample(contents, 200)
    queries = [
        str(case["question"])
        for path in (EXP / "cases.yaml", ML / "cases_ml.yaml")
        for case in _load_cases(path)
    ]

    for text in (*sample, *queries):
        expected = tokenize_identifiers(_CYRILLIC_WORD.sub(" ", text))
        assert _latin_tokens(text) == Counter(expected), text[:160]


def test_russian_stems_normalize_yo_and_strip_suffixes() -> None:
    assert russian_stem("Берёзами") == "берез"
    assert "берез" in tokenize_russian("Берёзами")
    assert not any(_CYRILLIC_TOKEN.fullmatch(token) and len(token) < 5 for token in tokenize_russian("и в к"))