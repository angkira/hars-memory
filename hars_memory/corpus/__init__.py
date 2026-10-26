"""LLM-free, CPU-only corpus build + query subsystem.

Point this at a set of directories and get a queryable BM25/flat-dense index
back — no entity extraction, no LLM call, no GPU. This is a *sibling* channel
to the LightRAG graph index built by ``server/index.py``: it emits the same
``kv_store_text_chunks.json`` chunk-store contract that
``retrieval/bm25_index.py`` and ``retrieval/flat_index.py`` already read, so
both existing retrieval channels work against a corpus-built index
unmodified.

Modules
-------
``build``
    ``build_corpus()`` — walk + chunk + emit chunk-store + provenance
    manifest, with atomic build isolation and incremental-rebuild reporting.
``query``
    ``search()`` — LLM-free retrieval (sparse/dense/fusion) over an index
    built by ``build``.
"""

from __future__ import annotations
