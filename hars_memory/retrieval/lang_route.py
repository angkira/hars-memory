"""Dependency-free query language detection for multilingual retrieval."""

from __future__ import annotations


def _is_cyrillic(char: str) -> bool:
    code = ord(char)
    return 0x0400 <= code <= 0x052F or char in "Ёё"


def detect_lang(text: str) -> str:
    """Return ``ru`` when Cyrillic is substantial, otherwise ``en``.

    A query is Russian when at least 25% of its Latin/Cyrillic letters are
    Cyrillic or it contains at least three Cyrillic words. Latin identifiers
    embedded in Russian text therefore do not interfere with the word rule.
    """
    cyrillic_letters = 0
    latin_letters = 0
    cyrillic_words = 0
    in_cyrillic_word = False
    for char in text:
        if _is_cyrillic(char):
            cyrillic_letters += 1
            if not in_cyrillic_word:
                cyrillic_words += 1
                in_cyrillic_word = True
        else:
            in_cyrillic_word = False
            if "A" <= char <= "Z" or "a" <= char <= "z":
                latin_letters += 1
    total_script_letters = cyrillic_letters + latin_letters
    if total_script_letters and cyrillic_letters / total_script_letters >= 0.25:
        return "ru"
    if cyrillic_words >= 3:
        return "ru"
    return "en"


__all__ = ["detect_lang"]