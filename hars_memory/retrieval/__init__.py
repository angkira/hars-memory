"""Sparse (BM25) retrieval + dense/sparse hybrid fusion for HARS long-term memory.

Why a separate top-level package instead of `tools/memory/server/`:

`server/` wires up the LightRAG *runtime* (embedder, LLM functions, vector/graph
storage lifecycle) — everything in it either constructs or feeds a `LightRAG`
instance. BM25 retrieval and dense/sparse fusion are a different concern: they
read the same on-disk chunk text LightRAG already produced
(`kv_store_text_chunks.json`) and combine it with LightRAG's dense channel at
query time, but they never construct or configure a `LightRAG` instance and
have zero dependency on `server/lightrag_init.py` or `server/embedder.py`.
Keeping this in its own package also means it can be developed, tested, and
reasoned about independently while another change concurrently lands in
`server/lightrag_init.py` (a CPU cross-encoder reranker) without either change
touching the other's files.

Modules
-------
tokenizer
    Identifier-preserving tokenization (whole tokens + sub-tokens) and
    identifier-shape detection, shared by indexing and querying so both sides
    tokenize identically.
bm25_index
    Build / persist / load a `bm25s.BM25` index over LightRAG's text-chunk KV
    store, with mtime-based cache invalidation. No GPU, no LLM, no re-embedding.
fusion
    Score normalization and tuned convex-combination (`alpha`) fusion of a
    dense channel's chunk hits with the BM25 channel's chunk hits.
"""

from __future__ import annotations
