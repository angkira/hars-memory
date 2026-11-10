"""Identifier-preserving tokenizer for BM25 sparse retrieval.

This corpus is queried with exact identifiers: `Q7X42`, `phase_x1_adapter_safe`,
`vma_native`, `hyp:1a2b3c4d`. A tokenizer that stems, strips underscores/digits,
or splits inside alphanumeric codes destroys exactly the signal BM25 exists to
provide here (CORE-Bench / arXiv:2606.11864 names this failure mode explicitly
for dense retrieval; the whole point of adding a sparse channel is to not repeat
it in the sparse one too).

Design (see `tokenize_identifiers` for the full algorithm):

1. Extract identifier-shaped runs directly with a character-class regex
   (`[A-Za-z0-9_\\-:./]+`) rather than splitting on whitespace-then-stripping —
   this means punctuation like commas, parentheses, and quotes act as natural
   token boundaries for free, without needing a stopword/punctuation list.
2. Emit the **whole run** as one token, casefolded. This is what makes
   `Q7X42` / `phase_x1_adapter_safe` / `hyp:1a2b3c4d` survive intact and match
   exactly at query time (query text goes through the identical function).
3. ALSO emit sub-tokens: split on `_ - : . /` delimiters, then split again on
   lowercase→uppercase (camelCase) boundaries. This lets a partial query like
   "adapter safe" or "q7" still surface the chunk containing the full identifier,
   without which BM25's exact-term matching would be too brittle for queries
   that don't reproduce an identifier verbatim.
4. Casefold (not aggressive stemming/normalization) is applied identically to
   both indexed text and queries, so it does not lose information relative to
   an un-casefolded index — it only removes a distinction (upper vs lower)
   that would otherwise force a caller to reproduce a code's exact casing to
   find it, which callers of this MCP server routinely do not do.

Known, deliberate limitation: the camelCase sub-token split in step 3 uses
case information (a lowercase/digit -> uppercase transition), so it can only
fire on tokens that still carry that transition. "Q7X42" splits into "q7" /
"x42"; an already-all-lowercase "q7x42" cannot — the transition it would be
based on does not exist in an all-lowercase string. This affects sub-token
*recall* only; the whole-token exact match (point 2, the primary guarantee
this module exists for) is unaffected and fully case-symmetric either way.

No stopword removal, no stemming: BM25's IDF term already down-weights common
words, and any stopword/stemmer list is exactly the kind of aggressive
normalization the identifier-retrieval evidence warns against. Kept out
deliberately, not as an oversight.

Cyrillic and other non-ASCII text: the run regex above is deliberately
ASCII-only, so by DEFAULT (`HARS_MEMORY_SPARSE_STEMMER` unset / "off")
Cyrillic prose yields NO tokens at all — the sparse channel contributes
nothing on a non-ASCII corpus. That is the shipped, byte-identical default;
measured 2026-11-01 on a 5-sentence Russian corpus: 4/5 documents tokenized
to zero tokens and every Russian query returned 0 hits (see
`experiments/search-2026-10/ml/cands.py` for the prior offline experiment
that first worked around this).

`HARS_MEMORY_SPARSE_STEMMER` (default "off"; one of "off" | "ru" | "en",
read fresh on every call like the other HARS_MEMORY_* flags in this package)
opts one language's morphology into the sparse channel. Whatever mode is
active MUST be applied identically to indexed text and queries — bm25s
matches literal token strings, so a build-time/query-time mode mismatch
produces empty matches rather than an error:

- "ru": the ASCII tokenization above is emitted UNCHANGED, plus each Cyrillic
  word (`[А-Яа-яЁё]+`) contributes its PyStemmer Snowball-'russian' stem
  (casefolded; stems shorter than 5 chars dropped as noise, matching the
  filter the prior experiment validated) and — when the stem is longer than
  9 chars — ALSO its first 9 characters. The extra prefix token exists
  because Snowball 'russian' does NOT converge noun-vs-infinitive pairs:
  "делегирование" -> "делегирован" vs "делегировать" -> "делегирова"
  (measured, PyStemmer 3.1.0), while the shared prefix "делегиров" matches
  both. Pairs that already converge under the bare stem (задачами/задач,
  полномочиями/полномочия, тестированию/тестирование) keep matching.
- "en": plain-word ASCII runs — NOT identifier-shaped ones, so the whole-run
  identifier guarantee above stays intact — emit their Snowball-'english'
  stem instead of the raw token ("delegation" / "delegate" / "delegating"
  -> "deleg").

The stemmer itself comes from the OPTIONAL PyStemmer dependency (extras
group `sparse-stem`; PyPI distribution name "PyStemmer", import name
"Stemmer" — the bare PyPI name "stemmer" is an unrelated package, verified
2026-11-01). It is imported lazily, only when the mode is not "off", so the
default path never touches it. `retrieval/bm25_index.py` records the active
mode in its on-disk cache sidecar and rebuilds when it changes, which is
what enforces the identical-application requirement across the persisted
index and live queries.
"""

from __future__ import annotations

import os
import re

# Matches an identifier-shaped run: letters, digits, and the punctuation that
# commonly appears *inside* identifiers/paths/codes (underscore, hyphen, colon,
# dot, slash). Anything else (whitespace, commas, parens, quotes, ...) is a
# natural token boundary and is never included.
_IDENTIFIER_RUN_RE = re.compile(r"[A-Za-z0-9_\-:./]+")

# Delimiter characters that separate sub-tokens *inside* an identifier run,
# e.g. "phase_x1_adapter_safe" -> "phase", "x1", "adapter", "safe";
#      "hyp:1a2b3c4d"       -> "hyp", "1a2b3c4d".
_DELIMITER_RE = re.compile(r"[_\-:./]+")

# camelCase boundary: a lowercase letter or digit immediately followed by an
# uppercase letter, e.g. "VmaNative" -> "Vma", "Native"; "Q7X42" -> "Q7", "X42"
# (the digit '7' before 'X' counts as the lowercase-or-digit side).
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Characters that only carry meaning *between* other identifier characters,
# not at the edges of a run (e.g. a sentence-final period after a word, or a
# leading "--" on a CLI flag). Stripped only from the edges, never internally,
# so "e.g." -> "e.g" but "phase_x1_adapter_safe" is untouched.
_EDGE_STRIP_CHARS = "_-:./"

# An identifier-shaped token: contains a colon-prefixed hex-ish suffix
# (`hyp:1a2b3c4d`), an underscore (`phase_x1_adapter_safe`), a camelCase boundary,
# or a mix of letters AND digits at least 3 chars long (`Q7X42`). Used to
# decide when a query token should trigger the dedicated exact-identifier
# lookup path rather than relying on fusion weighting alone.
_PREFIXED_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*:[0-9A-Za-z]{4,}$")
_ALNUM_CODE_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*[0-9])[A-Za-z0-9]{3,}$")

# ---------------------------------------------------------------------------
# Optional sparse-stemmer mode (see the module docstring section above for the
# full design rationale and the measured morphology evidence behind each
# constant). Default "off" = the byte-identical pre-stemmer behaviour.
HARS_MEMORY_SPARSE_STEMMER_ENV = "HARS_MEMORY_SPARSE_STEMMER"
_SPARSE_STEMMER_OFF = "off"
_SPARSE_STEMMER_RU = "ru"
_SPARSE_STEMMER_EN = "en"
_VALID_SPARSE_STEMMERS: frozenset[str] = frozenset(
    {_SPARSE_STEMMER_OFF, _SPARSE_STEMMER_RU, _SPARSE_STEMMER_EN}
)
_DEFAULT_SPARSE_STEMMER = _SPARSE_STEMMER_OFF

# Cyrillic word runs. Deliberately separate from `_IDENTIFIER_RUN_RE`: a
# Cyrillic word can never be an identifier on this corpus's conventions, so
# stemmer-mode tokens never interact with the exact-identifier guarantee.
_CYRILLIC_RUN_RE = re.compile(r"[А-Яа-яЁё]+")

# Minimum stem length to emit (prior experiment
# experiments/search-2026-10/ml/cands.py::tokenize_russian used the same 5):
# shorter Snowball outputs are function-word noise ("и", "в", "не").
_RU_MIN_STEM_LEN = 5
# Prefix length emitted IN ADDITION to the stem. Snowball 'russian' does not
# converge noun-vs-infinitive pairs ("делегирован" vs "делегирова" — measured);
# their shared prefix is 10 chars, so 9 keeps that pair (and longer families
# like автоматизация/автоматизировала -> "автоматиз") matching while staying
# long enough that unrelated words essentially never collide on 9 Cyrillic
# characters.
_RU_STEM_PREFIX_LEN = 9

# Language -> constructed PyStemmer stemmer, populated lazily. Snowball
# programs are cheap to construct but not free; one per process per language.
_stemmers: dict[str, object] = {}


def sparse_stemmer_mode() -> str:
    """Current effective `HARS_MEMORY_SPARSE_STEMMER` mode ("off" by default),
    read fresh on every call (same convention as fusion.py's env accessors),
    validated before any optional import happens so an invalid value fails
    fast even on hosts without PyStemmer installed.
    """
    raw = os.environ.get(HARS_MEMORY_SPARSE_STEMMER_ENV, _DEFAULT_SPARSE_STEMMER)
    mode = raw.strip().lower()
    if mode not in _VALID_SPARSE_STEMMERS:
        raise ValueError(
            f"{HARS_MEMORY_SPARSE_STEMMER_ENV} must be one of "
            f"{sorted(_VALID_SPARSE_STEMMERS)}, got {mode!r}"
        )
    return mode


def _get_stemmer(language: str):
    """Lazily import PyStemmer and return its Snowball stemmer for `language`.

    The import lives here — not at module top — so the default ("off") path
    never touches the optional dependency at all.
    """
    stemmer = _stemmers.get(language)
    if stemmer is None:
        try:
            import Stemmer  # optional dep; extras group `sparse-stem`
        except ImportError as exc:  # pragma: no cover - exercised only w/o PyStemmer
            raise ImportError(
                f"{HARS_MEMORY_SPARSE_STEMMER_ENV} is enabled but PyStemmer is not "
                "installed — install it with `uv sync --extra sparse-stem` (PyPI "
                "distribution 'PyStemmer'; import name 'Stemmer')"
            ) from exc
        stemmer = Stemmer.Stemmer(language)
        _stemmers[language] = stemmer
    return stemmer


def _ru_stem_tokens(word: str) -> list[str]:
    """Token list one Cyrillic word contributes in "ru" mode: [stem, prefix?]."""
    stem = _get_stemmer("russian").stemWord(word.casefold())
    if len(stem) < _RU_MIN_STEM_LEN:
        return []
    tokens = [stem]
    if len(stem) > _RU_STEM_PREFIX_LEN:
        tokens.append(stem[:_RU_STEM_PREFIX_LEN])
    return tokens



def _split_identifier(token: str) -> list[str]:
    """Split one identifier-shaped token into sub-tokens (delimiter + camelCase).

    Operates on the *original-case* token, since camelCase detection requires
    case information; callers casefold the resulting sub-tokens themselves.
    """
    parts: list[str] = []
    for delim_part in _DELIMITER_RE.split(token):
        if not delim_part:
            continue
        camel_parts = [p for p in _CAMEL_BOUNDARY_RE.split(delim_part) if p]
        parts.extend(camel_parts if camel_parts else [delim_part])
    return parts


def tokenize_identifiers(text: str) -> list[str]:
    """Tokenize `text` for BM25 indexing/querying, preserving whole identifiers.

    Returns a flat list of tokens (whole runs + their sub-tokens, casefolded,
    duplicates within a single run removed). Order is not significant to BM25
    but is kept deterministic for testability.

    `HARS_MEMORY_SPARSE_STEMMER` (module docstring) layers optional morphology
    on top: "ru" adds Snowball-Russian stem (+prefix) tokens for Cyrillic
    words; "en" replaces plain-word (non-identifier) run tokens with their
    Snowball-English stems; "off" — the default — returns exactly the
    pre-stemmer output. The SAME mode must have been active when the BM25
    index was built (bm25_index.py records and enforces this).
    """
    if not text:
        return []
    mode = sparse_stemmer_mode()
    tokens: list[str] = []
    for raw_run in _IDENTIFIER_RUN_RE.findall(text):
        stripped = raw_run.strip(_EDGE_STRIP_CHARS)
        if not stripped:
            continue
        whole = stripped.casefold()
        if mode == _SPARSE_STEMMER_EN and not looks_like_identifier(stripped):
            # Plain prose word: emit its stem instead of the raw token.
            # Identifier-shaped runs keep the untouched whole+sub-token
            # treatment below (that guarantee is the point of this module).
            tokens.append(_get_stemmer("english").stemWord(whole))
            continue
        tokens.append(whole)
        for sub in _split_identifier(stripped):
            sub_cf = sub.casefold()
            if sub_cf and sub_cf != whole:
                tokens.append(sub_cf)
    if mode == _SPARSE_STEMMER_RU:
        # The ASCII pass above is byte-identical to "off" mode; Cyrillic
        # morphology is purely additive on top of it.
        for word in _CYRILLIC_RUN_RE.findall(text):
            tokens.extend(_ru_stem_tokens(word))
    return tokens


def looks_like_identifier(token: str) -> bool:
    """True if `token` has the shape of a code identifier rather than a plain word.

    Matches: `hyp:1a2b3c4d`-style prefixed IDs, `snake_case` names, camelCase
    names, and alphanumeric codes mixing letters and digits (`Q7X42`). Used to
    decide when a query should trigger the dedicated exact-identifier lookup
    (see `retrieval.fusion`), not to gate normal tokenization.
    """
    if not token:
        return False
    if _PREFIXED_ID_RE.match(token):
        return True
    if "_" in token and len(token.strip("_")) > 0:
        return True
    if _CAMEL_BOUNDARY_RE.search(token) is not None:
        return True
    if _ALNUM_CODE_RE.match(token):
        return True
    return False


def extract_identifier_terms(text: str) -> list[str]:
    """Return the identifier-shaped runs found in `text`, original case, deduped
    in first-seen order. Used to detect "this question is asking about a
    specific code/entity" for the exact-identifier lookup path.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for raw_run in _IDENTIFIER_RUN_RE.findall(text):
        stripped = raw_run.strip(_EDGE_STRIP_CHARS)
        if not stripped or not looks_like_identifier(stripped):
            continue
        key = stripped.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(stripped)
    return terms


__all__ = [
    "tokenize_identifiers",
    "looks_like_identifier",
    "extract_identifier_terms",
    "sparse_stemmer_mode",
]
