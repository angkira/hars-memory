"""Offline exporter: UMAP projection for the Memory Travel cosmos view.

Writes ``coords_cache.json`` INTO the index dir (the only file it ever writes —
everything else is read-only), where the :8098 export endpoint
(``hars_memory/graph_export.py``) serves it.

Why offline: the service venv has no numpy/umap-learn, and a minutes-long
compute does not belong inside an HTTP request. Run this once per index (and
again whenever the index changes — the endpoint detects staleness via the
stored vdb mtimes and answers 503 with this script's name until you do):

    uv run --with numpy --with umap-learn python scripts/export_cosmos.py \
        --index-dir /path/to/index

CPU-only, fixed params + random_state=42 for reproducibility (~minutes for a
47k-entity index). The output is normalised to [-1,1] per axis with xyz (3D,
n_components=3) exactly like the historical memory-travel snapshot, so the UI
needs no changes between live and demo data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hars_memory.graph_export import (  # noqa: E402
    COORDS_CACHE_NAME,
    compute_coords,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--index-dir",
        default=os.environ.get("HARS_MEMORY_INDEX_DIR"),
        help="Index directory (default: $HARS_MEMORY_INDEX_DIR)",
    )
    parser.add_argument("--seed", type=int, default=42, help="UMAP random_state")
    args = parser.parse_args()

    if not args.index_dir:
        parser.error("no index directory (use --index-dir or set HARS_MEMORY_INDEX_DIR)")
    idx = Path(args.index_dir).expanduser().resolve()
    if not idx.is_dir():
        print(f"error: no such index dir: {idx}", file=sys.stderr)
        return 2

    print(f"computing UMAP projection for {idx.name} (this can take minutes)...", flush=True)
    cache = compute_coords(idx, dim=3, seed=args.seed)
    out = idx / COORDS_CACHE_NAME
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cache, f, separators=(",", ":"))
    print(
        f"wrote {out} — entities: {len(cache['entities']['ids'])}, "
        f"chunks: {len(cache['chunks'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
