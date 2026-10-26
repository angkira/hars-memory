"""Query-language detection for multilingual retrieval routing.

Pure, dependency-free helper intended for later packaging. Detection is based
only on letter-script counts (Cyrillic vs Latin) and Cyrillic word counts —
no models, no third-party imports.

``detect_lang`` returns one of:

- ``ru`` — predominantly Russian / Cyrillic query (including mixed queries
  that still clear the Cyrillic thresholds; Latin identifiers inside an
  otherwise Russian question stay classified as ``ru``).
- ``en`` — Latin-only (or below-threshold Cyrillic) query.
- ``mixed-ru`` — both scripts are present and the Cyrillic share is
  non-trivial but below the ``ru`` thresholds (rare; callers that only need a
  binary route should treat ``mixed-ru`` like ``ru``).
"""

from __future__ import annotations

_CYRILLIC_START = 0x0400
_CYRILLIC_END = 0x052F


def _is_cyrillic(char: str) -> bool:
    code = ord(char)
    return _CYRILLIC_START <= code <= _CYRILLIC_END or char in "Ёё"


def _is_latin(char: str) -> bool:
    return ("A" <= char <= "Z") or ("a" <= char <= "z")


def detect_lang(text: str) -> str:
    """Detect query language as ``ru``, ``en``, or ``mixed-ru``.

    Rules (letter = Unicode alphabetic character):

    1. Count Cyrillic letters vs Latin letters; ignore digits/punctuation.
    2. Count Cyrillic *words* (maximal runs of Cyrillic letters).
    3. Return ``ru`` if Cyrillic letters are >= 25% of all letters **or**
       there are >= 3 Cyrillic words.
    4. Else if both scripts appear and there is at least one Cyrillic letter,
       return ``mixed-ru``.
    5. Else return ``en``.

    Latin identifiers embedded in Russian text do not flip the label away from
    ``ru`` once the thresholds above are met.
    """
    if not text:
        return "en"

    cyrillic_letters = 0
    latin_letters = 0
    cyrillic_words = 0
    in_cyr_word = False

    for char in text:
        if _is_cyrillic(char):
            cyrillic_letters += 1
            if not in_cyr_word:
                cyrillic_words += 1
                in_cyr_word = True
            continue
        in_cyr_word = False
        if _is_latin(char):
            latin_letters += 1

    total_letters = cyrillic_letters + latin_letters
    if total_letters == 0:
        return "en"

    cyr_share = cyrillic_letters / total_letters
    if cyr_share >= 0.25 or cyrillic_words >= 3:
        return "ru"
    if cyrillic_letters > 0 and latin_letters > 0:
        return "mixed-ru"
    return "en"


def route_lang(text: str) -> str:
    """Binary route label: map ``mixed-ru`` → ``ru``."""
    label = detect_lang(text)
    return "ru" if label in {"ru", "mixed-ru"} else "en"


__all__ = ["detect_lang", "route_lang"]
