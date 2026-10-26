#!/usr/bin/env python3
"""Chunk-level localization check (no LLM, CPU embedder only).

For every query in a queries YAML that carries ``fact`` (regex) and ``heading``
(text) keys, retrieve the top-k chunks with the dense and the dense+BM25 channel
from the index in ``$HARS_MEMORY_INDEX_DIR`` and report

  fact@k     some retrieved chunk contains the fact
  located@k  ... and that SAME chunk also contains the heading text, i.e. the
             chunk says which section it belongs to (breadcrumb / own heading)

Run it once per index and compare the two outputs.  Point it at a COPY of a live
index (``cp -cR``) so that LightRAG start-up can never write into the original.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import yaml

from hars_memory.eval.channels import dense_only, hybrid_bm25


async def _main(args: argparse.Namespace) -> int:
    from hars_memory.retrieval.bm25_index import get_or_build_index
    from hars_memory.server.lightrag_init import create_lightrag

    queries = [q for q in yaml.safe_load(args.queries.read_text("utf-8"))["queries"] if q.get("fact")]
    rag = create_lightrag()
    await rag.initialize_storages()
    bm25 = (await asyncio.to_thread(get_or_build_index, os.environ["HARS_MEMORY_INDEX_DIR"], args.bm25_cache_dir))[0]

    summary: dict[str, dict[str, int]] = {}
    rows = []
    for channel in ("dense_only", "hybrid_bm25"):
        found = located = 0
        for q in queries:
            if channel == "dense_only":
                hits = await dense_only(rag, q["question"], args.top_k)
            else:
                hits = await hybrid_bm25(rag, bm25, q["question"], args.top_k, args.alpha)
            fact_rank = loc_rank = None
            for hit in hits:
                rec = await rag.text_chunks.get_by_id(hit.chunk_id) if hit.chunk_id else None
                text = (rec or {}).get("content", "")
                if re.search(q["fact"], text):
                    fact_rank = fact_rank or hit.rank
                    if q.get("heading", "") in text:
                        loc_rank = loc_rank or hit.rank
            found += fact_rank is not None
            located += loc_rank is not None
            rows.append({"channel": channel, "id": q["id"], "fact_rank": fact_rank, "located_rank": loc_rank})
        summary[channel] = {"queries": len(queries), "fact@k": found, "located@k": located}
    print(json.dumps({"top_k": args.top_k, "summary": summary, "rows": rows}, indent=1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries", type=Path, default=Path("config/eval/sre-kb-localization-queries.yaml"))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--bm25-cache-dir", default="/tmp/hars_memory_bm25_eval")
    return asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
