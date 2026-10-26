"""Unit tests for the token-based tokenizer used by LightRAG.

GPU-free and network-free: only the tokenizer wrapper is exercised, no
LightRAG instance is built and no LLM/embedder is loaded.
"""

from __future__ import annotations

import pytest

from hars_memory.server.lightrag_init import (
    BYTE_FALLBACK_TOKEN_RATIO,
    TIKTOKEN_ENCODING,
    _ByteTokenizer,
    _TiktokenTokenizer,
    resolve_tokenizer,
)

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64


@pytest.fixture()
def tokenizer() -> _TiktokenTokenizer:
    return _TiktokenTokenizer()


class TestResolveTokenizer:
    def test_prefers_tiktoken_and_passes_budgets_through(self) -> None:
        name, impl, chunk, overlap = resolve_tokenizer(CHUNK_TOKENS, OVERLAP_TOKENS)

        assert name == TIKTOKEN_ENCODING
        assert isinstance(impl, _TiktokenTokenizer)
        # tiktoken counts real tokens, so budgets must NOT be rescaled.
        assert (chunk, overlap) == (CHUNK_TOKENS, OVERLAP_TOKENS)

    def test_byte_fallback_rescales_budgets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import hars_memory.server.lightrag_init as mod

        def boom(*_args: object, **_kwargs: object) -> None:
            raise ImportError("no tiktoken here")

        monkeypatch.setattr(mod, "_TiktokenTokenizer", boom)

        name, impl, chunk, overlap = resolve_tokenizer(CHUNK_TOKENS, OVERLAP_TOKENS)

        assert name == "utf8-byte"
        assert isinstance(impl, _ByteTokenizer)
        assert chunk == CHUNK_TOKENS * BYTE_FALLBACK_TOKEN_RATIO
        assert overlap == OVERLAP_TOKENS * BYTE_FALLBACK_TOKEN_RATIO


class TestTokenizerContract:
    def test_encode_decode_roundtrip(self, tokenizer: _TiktokenTokenizer) -> None:
        text = "Kubernetes контроллеры: kubectl get pods -n monitoring 🚀"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_empty_and_none_safe(self, tokenizer: _TiktokenTokenizer) -> None:
        assert tokenizer.encode("") == []
        assert tokenizer.encode(None) == []  # type: ignore[arg-type]
        assert tokenizer.decode([]) == ""

    def test_counts_tokens_not_bytes(self, tokenizer: _TiktokenTokenizer) -> None:
        text = "kubernetes " * 200
        n_tokens = len(tokenizer.encode(text))
        n_bytes = len(text.encode("utf-8"))
        # Real tokens must be dramatically fewer than bytes (~4 bytes/token for
        # English prose) — this is the whole point of the switch.
        assert n_tokens < n_bytes / 2


class TestChunkBoundaries:
    """A 512-token budget must yield ~512-token chunks with intact Unicode."""

    def _text(self) -> str:
        # ~10k chars of mixed ASCII / Cyrillic / emoji, i.e. multi-byte chars
        # right where naive byte slicing would cut a character in half.
        line = (
            "The controller reconciles the namespace quota. "
            "Контроллер сверяет квоту пространства имён. 🚀\n"
        )
        return line * 120

    def test_chunk_count_and_size(self, tokenizer: _TiktokenTokenizer) -> None:
        tokens = tokenizer.encode(self._text())
        chunks = [
            tokens[i : i + CHUNK_TOKENS] for i in range(0, len(tokens), CHUNK_TOKENS)
        ]

        assert all(len(c) <= CHUNK_TOKENS for c in chunks)
        assert all(len(c) == CHUNK_TOKENS for c in chunks[:-1])
        # Sanity: a real tokenizer keeps the chunk count an order of magnitude
        # below what byte-granularity chunking would produce for the same text.
        byte_chunk_count = -(-len(self._text().encode("utf-8")) // CHUNK_TOKENS)
        assert len(chunks) < byte_chunk_count / 2

    def test_no_broken_unicode_at_boundaries(self, tokenizer: _TiktokenTokenizer) -> None:
        text = self._text()
        tokens = tokenizer.encode(text)
        decoded = ""
        for i in range(0, len(tokens), CHUNK_TOKENS):
            piece = tokenizer.decode(tokens[i : i + CHUNK_TOKENS])
            assert "\ufffd" not in piece  # no U+FFFD replacement characters
            decoded += piece

        assert decoded == text

    def test_byte_fallback_would_break_unicode(self) -> None:
        """Documents WHY the fallback is a fallback: byte slicing mangles text."""
        byte_tokenizer = _ByteTokenizer()
        tokens = byte_tokenizer.encode("квота")
        # Cutting mid-character drops the incomplete sequence entirely.
        assert byte_tokenizer.decode(tokens[:3]) != "квота"[:2]
