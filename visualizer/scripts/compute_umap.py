"""Offline UMAP layout over STORED vectors (no live calls).

Inputs (read-only): vdb_entities.json / vdb_chunks.json matrices (base64
float32, 768-dim) + node_index.json (entity order from export_graph.py).

Outputs (into ../public/data/):
  coords_entities.json  {ids:[...], xy:[[x,y]...], xyz:[[x,y,z]...]}  (47k, UMAP cosine)
  coords_chunks.json    {chunk_id: [x,y,z]}                           (2.8k, separate fit)

3D (n_components=3) since the cosmos is now a THREE.js starfield; `xy` is
kept as the first two dims for backward compat with old traces that baked
2D positions. Per-axis normalised to [-1,1]; the client flattens z slightly.

Entity order MUST match vdb `data` order intersected with node_index.json;
we emit ids + coords so the client joins by id (robust to order drift).
UMAP params fixed + random_state=42 for reproducibility.

CPU-only (no GPU jobs on this box). ~minutes for 47k points.
"""
import base64
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

INDEX = Path("/home/angkira/.local/share/hars-graphrag/index_compact_20260903_imported")
DATADIR = Path(__file__).resolve().parent.parent / "app" / "public" / "data"


def load_matrix(name):
    d = json.load(open(INDEX / name))
    raw = base64.b64decode(d["matrix"])
    n = len(d["data"])
    dim = d.get("embedding_dim", 768)
    arr = np.frombuffer(raw, dtype=np.float32, count=n * dim).reshape(n, dim).copy()
    return d["data"], arr


def fit_umap(X, seed=42, dim=3):
    import umap

    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.35, n_components=dim, metric="cosine", random_state=seed
    )
    return reducer.fit_transform(X)


def norm(xy):
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    xy = (xy - mins) / np.maximum(maxs - mins, 1e-9)  # [0,1]
    return (xy * 2 - 1).astype(np.float64)  # [-1,1]


def main():
    if importlib.util.find_spec("umap") is None:
        raise SystemExit("umap-learn is required (uv pip install umap-learn)")

    print("loading entity vectors...", flush=True)
    edata, X = load_matrix("vdb_entities.json")
    print("entities:", X.shape, flush=True)
    print("fitting UMAP 3D (entities)...", flush=True)
    Xe = norm(fit_umap(X, seed=42, dim=3))
    # vdb entity ids are ent-... hashes, NOT graphml names. Join key instead:
    # data[].entity_name <-> graphml entity_id. Emit name-keyed coords.
    names = [r.get("entity_name", "") for r in edata]
    xyz = [[round(float(a), 4) for a in row] for row in Xe]
    out = {
        "ids": names,
        "xy": [[row[0], row[1]] for row in xyz],  # first two dims: old-trace compat
        "xyz": xyz,
    }
    with open(DATADIR / "coords_entities.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote coords_entities.json", flush=True)

    print("loading chunk vectors...", flush=True)
    cdata, C = load_matrix("vdb_chunks.json")
    print("chunks:", C.shape, flush=True)
    print("fitting UMAP 3D (chunks)...", flush=True)
    Xc = norm(fit_umap(C, seed=43, dim=3))
    cids = [r.get("__id__", "") for r in cdata]
    outc = {
        cid: [round(float(a), 4) for a in row]
        for cid, row in zip(cids, Xc)
    }
    with open(DATADIR / "coords_chunks.json", "w") as f:
        json.dump(outc, f, separators=(",", ":"))
    print("wrote coords_chunks.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
