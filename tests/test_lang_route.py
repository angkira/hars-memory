from __future__ import annotations

from hars_memory.retrieval.lang_route import detect_lang


def test_detect_lang_uses_cyrillic_ratio_and_word_count() -> None:
    assert detect_lang("Q7X42 deployment checks") == "en"
    assert detect_lang("привет") == "ru"
    assert detect_lang("где посмотреть статус") == "ru"


def test_detect_lang_keeps_latin_identifiers_in_russian_query() -> None:
    assert detect_lang("как проверить Qdrant collection status") == "ru"


def test_detect_lang_handles_empty_and_identifier_only_text() -> None:
    assert detect_lang("") == "en"
    assert detect_lang("R9700 Q7X42 qdrant_transplant") == "en"