"""Regression coverage for oversized table rows and front-matter handling."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from hars_memory.ingest.chunker import chunk_markdown_structured, lightrag_chunking_func
from hars_memory.server.lightrag_init import _TiktokenTokenizer

HEADER = "| Parameter | Value | Description |"
SEPARATOR = "|---|---|---|"
ROW_KEY = "`iceberg.synthetic.long-cell`"


@pytest.fixture(scope="module")
def tok() -> _TiktokenTokenizer:
    return _TiktokenTokenizer()


def _tokens(tok: _TiktokenTokenizer):
    return lambda text: len(tok.encode(text))


def _oversized_row_text() -> tuple[str, str]:
    sentences = [
        f"Sentence {i:03d} explains the complete configuration value and preserves every original word boundary."
        for i in range(60)
    ]
    description = "<br/>".join(
        " ".join(sentences[start : start + 2]) for start in range(0, len(sentences), 2)
    )
    text = (
        "# Configuration\n\n## Parameters\n\n"
        + HEADER
        + "\n|---|---|---|\n"
        + f"| {ROW_KEY} | enabled | {description} |\n"
    )
    return text, description


@pytest.mark.parametrize("chunk_size", [512, 128])
def test_oversized_table_row_repeats_key_and_keeps_all_words(tok, chunk_size) -> None:
    text, description = _oversized_row_text()
    measure = _tokens(tok)

    chunks = chunk_markdown_structured(text, "row", chunk_size, measure)
    pieces = [chunk for chunk in chunks if ROW_KEY in chunk.text]

    assert len(pieces) > 1
    assert all(measure(chunk.text) <= chunk_size for chunk in chunks)
    descriptions: list[str] = []
    for chunk in pieces:
        assert HEADER in chunk.text
        assert SEPARATOR in chunk.text
        row = next(line for line in chunk.text.splitlines() if ROW_KEY in line)
        descriptions.append(row.rsplit(" |", 2)[1].removeprefix(" "))

    emitted = "".join(descriptions)
    assert emitted == description
    assert re.findall(r"[\w.-]+", emitted) == re.findall(r"[\w.-]+", description)
    assert [int(n) for n in re.findall(r"Sentence (\d{3})", emitted)] == list(range(60))
    assert [(chunk.part_index, chunk.part_count) for chunk in pieces] == [
        (n, len(pieces)) for n in range(1, len(pieces) + 1)
    ]
    assert chunks == chunk_markdown_structured(text, "row", chunk_size, measure)


def test_header_is_dropped_when_key_and_header_exceed_half_budget(tok) -> None:
    key = f"`{' '.join(['parameter'] * 40)}`"
    description = "A description sentence remains complete. " * 40
    text = f"# Configuration\n\n{HEADER}\n|---|---|---|\n| {key} | enabled | {description} |\n"
    measure = _tokens(tok)

    chunks = chunk_markdown_structured(text, "long-key", 128, measure)
    pieces = [chunk for chunk in chunks if key in chunk.text]

    assert len(pieces) > 1
    assert all(key in chunk.text for chunk in pieces)
    assert all(HEADER not in chunk.text for chunk in pieces)
    assert all(measure(chunk.text) <= 128 for chunk in chunks)


def test_trimmed_t4_iceberg_fixture_keeps_parameter_key_with_values(tok) -> None:
    fixture = Path(__file__).parent / "fixtures/iceberg-sink-configuration-reference.md"
    text = fixture.read_text()
    measure = _tokens(tok)
    chunks = chunk_markdown_structured(text, "iceberg", 512, measure)

    assert all(measure(chunk.text) <= 512 for chunk in chunks)
    interval = next(chunk for chunk in chunks if "| 60000 | 300000 |" in chunk.text)
    assert "`iceberg.control.commit.interval-ms`" in interval.text
    assert "| Parameter | Value | Code default | Description |" in interval.text
    synthetic = [chunk for chunk in chunks if "`iceberg.synthetic.long-cell`" in chunk.text]
    assert len(synthetic) > 1
    assert all("|---|---|---|---|" in chunk.text for chunk in synthetic)


def test_table_row_splitting_respects_escaped_and_code_span_pipes(tok) -> None:
    description = ("Sentence uses `sample|value` as one complete code span. " * 45).rstrip()
    row = f"| `configuration.key` | plain\\|escaped | {description} |"
    text = f"# Configuration\n\n{HEADER}\n|---|---|---|\n{row}\n"
    measure = _tokens(tok)

    chunks = chunk_markdown_structured(text, "pipes", 128, measure)
    pieces = [chunk for chunk in chunks if "`configuration.key`" in chunk.text]
    fragments = []
    for chunk in pieces:
        assert "|---|---|---|" in chunk.text
        line = next(line for line in chunk.text.splitlines() if "`configuration.key`" in line)
        if "Sentence uses" in line:
            fragments.append(line.rsplit(" |", 2)[1].removeprefix(" "))

    emitted = "".join(fragments)
    assert any("plain\\|escaped" in chunk.text for chunk in pieces)
    assert emitted == description
    assert all(fragment.count("`") % 2 == 0 for fragment in fragments)
    assert all(measure(chunk.text) <= 128 for chunk in chunks)


@pytest.mark.parametrize("min_size", [0, 200])
def test_front_matter_folds_into_first_content_chunk(min_size) -> None:
    text = (
        "---\narticle_id: note-1\ntitle: Note\n---\n\n"
        "[Document: note.md | Section: notes | Date: 2026-01-01]\n\n"
        "# Note\n\nA short first content paragraph.\n"
    )

    chunks = chunk_markdown_structured(text, "note", 400, min_size=min_size)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("---\narticle_id: note-1")
    assert "[Document: note.md" in chunks[0].text
    assert "A short first content paragraph." in chunks[0].text
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 10)


@pytest.mark.parametrize("min_size", [0, 200])
def test_front_matter_is_dropped_if_it_would_overflow_but_doc_header_remains(min_size) -> None:
    text = (
        "---\narticle_id: too-large\nmetadata: "
        + ("x" * 240)
        + "\n---\n\n"
        "[Document: oversized.md | Section: notes | Date: 2026-01-01]\n\n"
        "# Note\n\n"
        + ("Content words that remain available for retrieval. " * 18)
    )

    chunks = chunk_markdown_structured(text, "note", 220, min_size=min_size)

    assert chunks
    assert all(len(chunk.text) <= 220 for chunk in chunks)
    assert not any("article_id:" in chunk.text for chunk in chunks)
    assert "[Document: oversized.md | Section: notes | Date: 2026-01-01]" in chunks[0].text
    assert chunks[0].start_line == 6
    assert any("Content words" in chunk.text for chunk in chunks)


@pytest.mark.parametrize("min_size", [0, 200])
def test_front_matter_overflow_respects_token_budget(tok, min_size) -> None:
    text = (
        "---\narticle_id: too-large\nmetadata: "
        + ("x" * 700)
        + "\n---\n\n"
        "[Document: token-budget.md | Section: notes | Date: 2026-01-01]\n\n"
        "# Note\n\n"
        + ("Content words remain available for retrieval. " * 30)
    )
    measure = _tokens(tok)

    chunks = chunk_markdown_structured(text, "note", 128, measure, min_size=min_size)

    assert chunks
    assert all(measure(chunk.text) <= 128 for chunk in chunks)
    assert not any("article_id:" in chunk.text for chunk in chunks)
    assert "[Document: token-budget.md | Section: notes | Date: 2026-01-01]" in chunks[0].text


@pytest.mark.parametrize("min_size", [0, 200])
def test_front_matter_only_document_stays_a_single_chunk(min_size) -> None:
    text = "---\narticle_id: only-metadata\ntitle: Metadata only\n---\n"

    chunks = chunk_markdown_structured(text, "metadata", 400, min_size=min_size)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("---\narticle_id: only-metadata")


def test_non_markdown_token_mode_matches_prechange_golden(tok) -> None:
    texts = (
        "[Document: tool.py | Section: all | Date: 2026-01-01]\n\n# comment\nx = 1\nprint(x)\n",
        "Plain prose remains a token-window input. " * 45,
        '[Document: settings.json | Section: all | Date: 2026-01-01]\n\n{"enabled": true, "items": [1, 2, 3]}\n',
    )
    expected = (
        "a394c76592d4583ed495f84256a830417cf22e9053d31491f87f13f08d718b3b",
        "1ccb0310ed6ad745c6a8132dac6bcd2d8ed8b4910b7334470948561a7abee424",
        "2eb7642a281f569f52c9469fa9b1649e4c10b5fbe7bea05be92a6430095cb39c",
    )

    actual = []
    for text in texts:
        records = lightrag_chunking_func(tok, text, None, False, 4, 16)
        payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        actual.append(hashlib.sha256(payload.encode()).hexdigest())

    assert tuple(actual) == expected
