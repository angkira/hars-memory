"""Contract tests for the deterministic hard_1k retrieval gold set.

The dataset itself lives in ``eval_agentic-data/hard_1k``, which is a generated,
gitignored fixture.  These tests are split into two layers:

* schema / leakage / dedupe checks that run wherever the fixture is present
  (they skip when it is not, e.g. on a clean checkout);
* corpus-resolution and verbatim-snippet checks that are skipped when the
  production index is not present on the machine.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from hars_memory.eval import gen_hard_queries as gen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval_agentic-data" / "hard_1k"
INDEX = gen.DEFAULT_INDEX_DIR
STRATA = set(gen.STRATA)


def _require(file_name: str) -> Path:
    """Path to a dataset file, skipping the test when the fixture is absent.

    ``eval_agentic-data/`` is a generated, gitignored fixture (see
    ``.gitignore``), so a clean checkout has no dataset and the contract tests
    below cannot run.
    """
    path = DATA / file_name
    if not path.is_file():
        pytest.skip(f"hard_1k dataset fixture not present: {path} (generated, gitignored)")
    return path


def _load(name: str) -> dict:
    return yaml.safe_load(_require(name).read_text(encoding="utf-8"))


def _read_json(name: str) -> dict:
    return json.loads(_require(name).read_text(encoding="utf-8"))


def _queries(name: str) -> list[dict]:
    return _load(name)["queries"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


@pytest.fixture(scope="module")
def all_cases() -> list[dict]:
    return _queries("hard_1k.yaml")


def test_split_sizes_and_disjoint_questions(all_cases: list[dict]) -> None:
    train = _queries("hard_1k_train.yaml")
    evaluation = _queries("hard_1k_eval.yaml")
    assert len(train) + len(evaluation) == len(all_cases) == 1000
    assert len(train) / len(all_cases) == pytest.approx(0.70, abs=0.02)
    train_questions = {_norm(case["question"]) for case in train}
    eval_questions = {_norm(case["question"]) for case in evaluation}
    assert train_questions.isdisjoint(eval_questions)
    assert len(train_questions) == len(train)
    assert len(eval_questions) == len(evaluation)


def test_required_fields(all_cases: list[dict]) -> None:
    required = {
        "id",
        "stratum",
        "difficulty",
        "hop_count",
        "question",
        "answer",
        "gold_docs",
        "gold_snippets",
        "rationale",
    }
    ids = set()
    for case in all_cases:
        assert required <= set(case)
        assert case["stratum"] in STRATA
        assert case["difficulty"] in {"medium", "hard", "very_hard"}
        assert case["hop_count"] in {2, 3}
        assert isinstance(case["gold_docs"], list) and len(case["gold_docs"]) >= 2
        assert isinstance(case["gold_snippets"], list) and case["gold_snippets"]
        assert case["question"].strip().endswith("?")
        assert 40 <= len(case["question"]) <= 400
        assert case["rationale"].strip()
        assert case["id"] not in ids
        ids.add(case["id"])


def test_no_answer_leakage(all_cases: list[dict]) -> None:
    for case in all_cases:
        answer = str(case["answer"]).casefold().strip()
        question = case["question"].casefold()
        assert answer not in question, case["id"]
        # Long answers must not be reconstructible from the question's tokens.
        answer_tokens = set(gen._content_tokens(answer))
        if len(answer_tokens) >= 3:
            overlap = len(answer_tokens & set(gen._content_tokens(question))) / len(answer_tokens)
            assert overlap < 0.8, case["id"]


def test_no_internal_duplicates(all_cases: list[dict]) -> None:
    seen: list[dict] = []
    for case in all_cases:
        candidate = gen.Candidate(
            stratum=case["stratum"],
            question=case["question"],
            answer=str(case["answer"]),
            anchor=case["gold_docs"][0],
            targets=case["gold_docs"][1:],
            snippets={entry["doc"]: entry["text"] for entry in case["gold_snippets"]},
            rationale=case["rationale"],
            hop_count=case["hop_count"],
            bridge=case.get("bridge", ""),
            template=case.get("template", ""),
            fact_kind=case.get("fact_kind", "unknown"),
        )
        assert not gen._is_internal_duplicate(candidate, seen), case["id"]
        seen.append(candidate)


def test_deduped_against_existing_gold_sets(all_cases: list[dict]) -> None:
    existing: list[tuple[str, set[str]]] = []
    for rel in gen.EXISTING_FILES:
        existing.extend(gen.load_existing_questions([ROOT / rel]))
    assert existing, "existing gold sets were not found"
    for case in all_cases:
        gold = {path.rsplit("/", 1)[-1] for path in case["gold_docs"]}
        assert not gen._similar_to_existing(case["question"], gold, existing), case["id"]


def test_stratum_and_difficulty_coverage(all_cases: list[dict]) -> None:
    stats = _read_json("stats.json")
    by_stratum = {name: 0 for name in STRATA}
    by_difficulty: dict[str, int] = {}
    hops: dict[str, int] = {}
    for case in all_cases:
        by_stratum[case["stratum"]] += 1
        by_difficulty[case["difficulty"]] = by_difficulty.get(case["difficulty"], 0) + 1
        hops[str(case["hop_count"])] = hops.get(str(case["hop_count"]), 0) + 1
    assert by_stratum == stats["by_stratum"]
    assert by_difficulty == stats["by_difficulty"]
    assert hops == stats["hop_distribution"]
    assert stats["gold_existence_rate"] == 1.0
    assert stats["duplicate_rate_internal"] == 0.0
    assert stats["duplicate_rate_vs_existing"] == 0.0
    assert all(count >= 90 for count in by_stratum.values())


def test_gold_paths_manifest_matches_dataset(all_cases: list[dict]) -> None:
    manifest = _read_json("gold_paths.json")
    paths = set(manifest["paths"])
    gold = {path for case in all_cases for path in case["gold_docs"]}
    assert gold == paths
    assert manifest["count"] == len(paths)


@pytest.mark.skipif(not INDEX.exists(), reason="production index not available")
def test_gold_docs_and_snippets_resolve_in_index(all_cases: list[dict]) -> None:
    statuses = json.loads((INDEX / "kv_store_doc_status.json").read_text(encoding="utf-8"))
    full_docs = json.loads((INDEX / "kv_store_full_docs.json").read_text(encoding="utf-8"))
    resolved: dict[str, str] = {}
    for doc_id, meta in statuses.items():
        if not isinstance(meta, dict) or meta.get("status") != "processed":
            continue
        raw = full_docs.get(doc_id)
        if raw is None:
            continue
        path = str(meta.get("file_path") or "")
        if path:
            resolved[path] = raw.get("content", "") if isinstance(raw, dict) else str(raw)
    for case in all_cases:
        for path in case["gold_docs"]:
            assert path in resolved, f"{case['id']}: unresolved gold {path}"
        for entry in case["gold_snippets"]:
            assert entry["text"] in resolved[entry["doc"]], f"{case['id']}: non-verbatim snippet"


def test_generator_is_deterministic_on_toy_corpus(tmp_path: Path) -> None:
    index = tmp_path / "index"
    index.mkdir()
    shared = "11111111-2222-3333-4444-555555555555"
    statuses = {
        "file:a": {"status": "processed", "file_path": "2026-01-01_anchor.md"},
        "file:b": {"status": "processed", "file_path": "2026-01-02_target.md"},
    }
    full_docs = {
        "file:a": {
            "content": (
                "[Document: 2026-01-01_anchor.md | Section: docs | Date: 2026-01-01]\n\n"
                "# Anchor note\n\n"
                f"The managed job run {shared} was launched to test the anchor branch.\n"
            )
        },
        "file:b": {
            "content": (
                "[Document: 2026-01-02_target.md | Section: docs | Date: 2026-01-02]\n\n"
                "# Target note\n\n"
                f"The managed job run {shared} finished.\nStatus: completed\n"
            )
        },
    }
    (index / "kv_store_doc_status.json").write_text(json.dumps(statuses), encoding="utf-8")
    (index / "kv_store_full_docs.json").write_text(json.dumps(full_docs), encoding="utf-8")

    corpus = gen.Corpus(index)
    assert set(corpus.docs) == {"2026-01-01_anchor.md", "2026-01-02_target.md"}
    cases = gen.mine_shared_ids(corpus, "uuid", "multi_hop", limit=10)
    assert len(cases) == 1
    candidate = cases[0]
    assert candidate.answer == "completed"
    assert candidate.gold_docs == ["2026-01-01_anchor.md", "2026-01-02_target.md"]
    assert gen._validate_candidate(candidate, corpus) is None

    # Determinism: a second load yields byte-identical question/answer.
    again = gen.mine_shared_ids(gen.Corpus(index), "uuid", "multi_hop", limit=10)
    assert [(c.question, c.answer) for c in cases] == [(c.question, c.answer) for c in again]


def test_validate_rejects_missing_gold(tmp_path: Path) -> None:
    index = tmp_path / "index"
    index.mkdir()
    statuses = {"file:a": {"status": "processed", "file_path": "a.md"}}
    full_docs = {"file:a": {"content": "# A\n\nText."}}
    (index / "kv_store_doc_status.json").write_text(json.dumps(statuses), encoding="utf-8")
    (index / "kv_store_full_docs.json").write_text(json.dumps(full_docs), encoding="utf-8")
    corpus = gen.Corpus(index)
    candidate = gen.Candidate(
        stratum="multi_hop",
        question="What happened in the missing record?",
        answer="x",
        anchor="a.md",
        targets=["missing.md"],
        snippets={"a.md": "Text.", "missing.md": "x"},
        rationale="test",
        hop_count=2,
        bridge="x",
        template="t0",
        fact_kind="status",
    )
    assert "missing" in (gen._validate_candidate(candidate, corpus) or "")
