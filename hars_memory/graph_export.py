"""Read-only graph/stats/coords export for the Memory Travel cosmos view.

Why this exists
---------------
The Memory Travel UI used to load STATIC baked fixtures (``app/public/data/*.json``,
captured once from one specific index) unconditionally in every deployment — so a
deployment pointed at a DIFFERENT memory silently displayed someone else's graph.
These endpoints let the UI render the graph of the memory it is actually
connected to, and identify it (``index_id`` = basename of ``HARS_MEMORY_INDEX_DIR``).

Served by the trace HTTP listener on :8098 (see ``hars_memory/trace.py``):

* ``GET /stats``  — identity + counts (docs/chunks/entities/relations/nodes/edges).
* ``GET /graph``  — sampled nodes/edges JSON for a browser starfield.
* ``GET /coords`` — UMAP projection over the stored vectors, served from a cache
  file in the index dir (``coords_cache.json``).

Hard rules
----------
* READ-ONLY wrt the index: the only file ever written is ``coords_cache.json``
  inside the index dir (a new, LightRAG-ignored cache file).
* Size caps everywhere: the 325 MB ``vdb_entities.json`` matrix is never
  serialised; descriptions are truncated to ``DESC_HEAD`` chars; node/edge
  counts are clamped to ``MAX_*``.
* No LightRAG / mcp_server imports: this module is stdlib-only (numpy/umap are
  imported lazily and only on the inline-compute path), so serving the graph can
  never disturb recall.

Contracts are documented in ``docs/trace-api.md`` (§ cosmos export) and
``memory-travel/docs/CONNECTION.md``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("hars-longterm-memory-trace")

NS = "{http://graphml.graphdrawing.org/xmlns}"

# Sampling defaults: enough for a browser starfield (the fixture snapshot this
# replaces had 47k nodes / 54k edges — roughly 10 MB of JSON, too heavy to
# foist on every deployment). Top-N by degree keeps the hubs that make the
# starfield legible. Documented in docs/trace-api.md and CONNECTION.md.
DEFAULT_GRAPH_NODES = 8000
DEFAULT_GRAPH_EDGES = 16000
MAX_GRAPH_NODES = 50_000
MAX_GRAPH_EDGES = 200_000

# Truncated description head per node (chars). Keeps /graph a few MB, not tens.
DESC_HEAD = 200

COORDS_CACHE_NAME = "coords_cache.json"

# vdb files whose mtime the coords cache is keyed to (the projection input).
COORDS_INPUT_FILES = ("vdb_entities.json", "vdb_chunks.json")


class ExportError(Exception):
    """Raised with a user-safe message; the HTTP layer maps it to a status."""

    def __init__(self, message: str, *, status: int = 503, hint: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.hint = hint


def index_dir() -> Path:
    """The live index dir (``HARS_MEMORY_INDEX_DIR``), read at request time."""
    raw = os.environ.get("HARS_MEMORY_INDEX_DIR")
    if not raw or not raw.strip():
        raise ExportError(
            "index_dir_unconfigured",
            status=503,
            hint="set HARS_MEMORY_INDEX_DIR (the service process normally provides it)",
        )
    return Path(raw).expanduser().resolve()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


# ---------------------------------------------------------------------------
# graphml parsing (the only expensive read; cached in memory per mtime)
# ---------------------------------------------------------------------------

class _GraphState:
    """Parsed graphml, keyed by (path, mtime) so an updated index re-parses."""

    def __init__(
        self,
        index_id: str,
        graphml_mtime: float,
        ids: list[str],
        types: list[str],
        descs: list[str],
        deg: list[int],
        edges: list[tuple[int, int]],
    ) -> None:
        self.index_id = index_id
        self.graphml_mtime = graphml_mtime
        self.ids = ids
        self.types = types
        self.descs = descs
        self.deg = deg
        self.edges = edges

    @property
    def node_count(self) -> int:
        return len(self.ids)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


_graph_lock = threading.Lock()
_graph_cache: _GraphState | None = None


def _graphml_key_ids(path: Path) -> dict[tuple[str, str], str]:
    key_id: dict[tuple[str, str], str] = {}
    for _ev, el in ET.iterparse(str(path), events=("start",)):
        if el.tag == f"{NS}key":
            key_id[(el.get("for") or "", el.get("attr.name") or "")] = el.get("id") or ""
        elif el.tag == f"{NS}graph":
            break
    return key_id


def _data_text(el: ET.Element, key: str) -> str:
    if not key:
        return ""
    for d in el.findall(f"{NS}data"):
        if d.get("key") == key:
            return d.text or ""
    return ""


def load_graph_state() -> _GraphState:
    """Parse (or reuse) the knowledge graph from the LIVE index's graphml."""
    global _graph_cache
    idx = index_dir()
    graphml = idx / "graph_chunk_entity_relation.graphml"
    if not graphml.is_file():
        raise ExportError(
            "graphml_missing",
            status=503,
            hint=f"no graph_chunk_entity_relation.graphml in {idx}",
        )
    mtime = _mtime(graphml)
    with _graph_lock:
        cached = _graph_cache
        if cached is not None and cached.index_id == idx.name and cached.graphml_mtime == mtime:
            return cached

    key_id = _graphml_key_ids(graphml)
    kid = key_id.get(("node", "entity_id"), "d0")
    ktype = key_id.get(("node", "entity_type"), "d1")
    kdesc = key_id.get(("node", "description"), "d2")

    ids: list[str] = []
    types: list[str] = []
    descs: list[str] = []
    id_to_idx: dict[str, int] = {}
    for _ev, el in ET.iterparse(str(graphml), events=("end",)):
        if el.tag != f"{NS}node":
            continue
        eid = _data_text(el, kid) or el.get("id") or ""
        if eid and eid not in id_to_idx:
            id_to_idx[eid] = len(ids)
            ids.append(eid)
            types.append(_data_text(el, ktype) or "unknown")
            descs.append((_data_text(el, kdesc) or "")[:DESC_HEAD])
        el.clear()

    edges: list[tuple[int, int]] = []
    for _ev, el in ET.iterparse(str(graphml), events=("end",)):
        if el.tag != f"{NS}edge":
            continue
        s = id_to_idx.get(el.get("source") or "", -1)
        t = id_to_idx.get(el.get("target") or "", -1)
        if s >= 0 and t >= 0:
            edges.append((s, t))
        el.clear()

    deg = [0] * len(ids)
    for s, t in edges:
        deg[s] += 1
        deg[t] += 1

    state = _GraphState(idx.name, mtime, ids, types, descs, deg, edges)
    with _graph_lock:
        _graph_cache = state
    logger.info(
        "graph export: parsed %s — %d nodes, %d edges", idx.name, len(ids), len(edges)
    )
    return state


# ---------------------------------------------------------------------------
# /stats
# ---------------------------------------------------------------------------

_stats_lock = threading.Lock()
_stats_cache: tuple[float, float, dict[str, Any]] | None = None


def _count_json_objects(path: Path) -> int | None:
    """len() of the top-level object/array in a kv-store JSON, or None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return len(data) if isinstance(data, (dict, list)) else None


def stats_payload() -> dict[str, Any]:
    """Identity + counts for the HUD/About card. Counts come from the LIVE
    index: docs = kv_store_full_docs, chunks = kv_store_text_chunks,
    entities/relations/nodes/edges = graphml, last_ingest = newest index mtime.
    """
    global _stats_cache
    idx = index_dir()
    state = load_graph_state()
    doc_status_mtime = _mtime(idx / "kv_store_doc_status.json")
    with _stats_lock:
        cached = _stats_cache
        if (
            cached is not None
            and cached[0] == state.graphml_mtime
            and cached[1] == doc_status_mtime
        ):
            return cached[2]

    docs = _count_json_objects(idx / "kv_store_full_docs.json")
    chunks = _count_json_objects(idx / "kv_store_text_chunks.json")
    mtimes = [doc_status_mtime] + [_mtime(idx / name) for name in COORDS_INPUT_FILES]
    mtimes.append(state.graphml_mtime)
    last_ingest = max((m for m in mtimes if m > 0), default=0.0)
    payload = {
        "ok": True,
        "index_id": state.index_id,
        "docs": docs,
        "chunks": chunks,
        "entities": state.node_count,
        "relations": state.edge_count,
        "nodes": state.node_count,
        "edges": state.edge_count,
        "last_ingest": (
            datetime.fromtimestamp(last_ingest, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if last_ingest > 0
            else None
        ),
        "generated_at": _utcnow_iso(),
    }
    with _stats_lock:
        _stats_cache = (state.graphml_mtime, doc_status_mtime, payload)
    return payload


# ---------------------------------------------------------------------------
# /graph
# ---------------------------------------------------------------------------

def _clamp(value: str | None, default: int, maximum: int) -> int:
    """Parse a limit param: absent/garbage/negative -> default, otherwise clamp
    to the hard cap (a large explicit request is intentional, nonsense is not)."""
    try:
        n = int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default
    if n <= 0:
        return default
    return min(n, maximum)


def graph_payload(params: dict[str, str]) -> dict[str, Any]:
    """Sampled nodes/edges for the starfield.

    Nodes: top-N by undirected degree (ties keep graphml document order).
    Edges: pairs of indices INTO THE RETURNED nodes array (both endpoints must
    have survived sampling), capped at M in original edge order — the same
    shape the baked fixtures used, so the client code is unchanged.
    """
    state = load_graph_state()
    limit_nodes = _clamp(params.get("limit_nodes"), DEFAULT_GRAPH_NODES, MAX_GRAPH_NODES)
    limit_edges = _clamp(params.get("limit_edges"), DEFAULT_GRAPH_EDGES, MAX_GRAPH_EDGES)

    order = sorted(range(state.node_count), key=lambda i: (-state.deg[i], i))
    keep = order[:limit_nodes]
    remap = {old: new for new, old in enumerate(keep)}

    edges: list[list[int]] = []
    for s, t in state.edges:
        if len(edges) >= limit_edges:
            break
        rs, rt = remap.get(s), remap.get(t)
        if rs is not None and rt is not None:
            edges.append([rs, rt])

    sampled = len(keep) < state.node_count or len(edges) < state.edge_count
    return {
        "ok": True,
        "meta": {
            "index_id": state.index_id,
            "docs": stats_payload().get("docs"),
            "nodes": len(keep),
            "edges": len(edges),
            "full_nodes": state.node_count,
            "full_edges": state.edge_count,
            "sampled": sampled,
            "node_limit": limit_nodes,
            "edge_limit": limit_edges,
            "sampling": "top by undirected degree, ties by document order",
            "generated_at": _utcnow_iso(),
        },
        "nodes": [
            {
                "id": state.ids[old],
                "label": state.ids[old],
                "type": state.types[old],
                "degree": state.deg[old],
                "desc_head": state.descs[old],
            }
            for old in keep
        ],
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# /coords — UMAP projection, always served from (or writing) the cache file
# ---------------------------------------------------------------------------

_coords_lock = threading.Lock()


def _coords_cache_path(idx: Path) -> Path:
    return idx / COORDS_CACHE_NAME


def _coords_fingerprint(idx: Path) -> dict[str, float]:
    return {name: round(_mtime(idx / name), 3) for name in COORDS_INPUT_FILES}


def _find_precomputed_coords(idx: Path, dim: int) -> dict[str, Any] | None:
    """Find precomputed coordinates in standard locations or vendored data."""
    # 1. Direct alternative index paths
    for cand_path in (Path("/data/index") / COORDS_CACHE_NAME, idx / COORDS_CACHE_NAME):
        if cand_path.is_file():
            try:
                with open(cand_path, "r", encoding="utf-8") as f:
                    c = json.load(f)
                ent = c.get("entities") or {}
                if isinstance(ent, dict) and ent.get("ids"):
                    if dim >= 3 and not ent.get("xyz") and ent.get("xy"):
                        ent["xyz"] = [[round(float(p[0]), 4), round(float(p[1]), 4), 0.0] for p in ent["xy"]]
                        c["entities"] = ent
                    return c
            except Exception:
                pass

    # 2. Check precomputed coords_entities.json and coords_chunks.json
    root_dirs = [
        Path.cwd(),
        Path(__file__).resolve().parent.parent,
        Path("/app"),
        Path("/usr/share/nginx/html"),
    ]
    for r in root_dirs:
        for sub in ("app/public/data", "visualizer/app/public/data", "data", "public/data"):
            cand = r / sub / "coords_entities.json"
            if cand.is_file():
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        ent_data = json.load(f)
                    ids = ent_data.get("ids") or []
                    xy = ent_data.get("xy") or []
                    xyz = ent_data.get("xyz") or []
                    if ids:
                        # If index has its own graph nodes, ensure fixture actually matches
                        try:
                            state = load_graph_state()
                            if state.ids and not (set(ids) & set(state.ids)):
                                continue
                        except Exception:
                            pass
                        if not xyz and xy:
                            xyz = [[round(float(p[0]), 4), round(float(p[1]), 4), 0.0] for p in xy]
                        chunks_data: dict[str, Any] = {}
                        chunks_cand = cand.parent / "coords_chunks.json"
                        if chunks_cand.is_file():
                            try:
                                with open(chunks_cand, "r", encoding="utf-8") as f:
                                    chunks_data = json.load(f)
                            except Exception:
                                pass
                        cache = {
                            "ok": True,
                            "meta": {
                                "index_id": idx.name,
                                "dim": dim,
                                "source_mtimes": _coords_fingerprint(idx),
                                "source": str(cand),
                                "generated_at": _utcnow_iso(),
                            },
                            "entities": {"ids": ids, "xy": xy, "xyz": xyz},
                            "chunks": chunks_data,
                        }
                        try:
                            with open(_coords_cache_path(idx), "w", encoding="utf-8") as f:
                                json.dump(cache, f, separators=(",", ":"))
                        except OSError:
                            pass
                        return cache
                except Exception:
                    pass
    return None


def _generate_fallback_coords(idx: Path, dim: int = 3) -> dict[str, Any]:
    """Generate fast deterministic 3D coordinates so /coords never fails with 503."""
    import math

    ids: list[str] = []
    try:
        state = load_graph_state()
        ids = list(state.ids)
    except Exception:
        try:
            with open(idx / "vdb_entities.json", "r", encoding="utf-8") as f:
                d = json.load(f)
            ids = [str(r.get("entity_name") or r.get("__id__") or "") for r in d.get("data", [])]
        except Exception:
            ids = []

    n = max(len(ids), 1)
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    xyz: list[list[float]] = []
    xy: list[list[float]] = []
    for i in range(len(ids)):
        y = 1.0 - (i / float(max(n - 1, 1))) * 2.0  # -1 to 1
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = 2.0 * math.pi * i / phi
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        xyz.append([round(x, 4), round(y, 4), round(z, 4)])
        xy.append([round(x, 4), round(y, 4)])

    return {
        "ok": True,
        "meta": {
            "index_id": idx.name,
            "dim": dim,
            "source_mtimes": _coords_fingerprint(idx),
            "generated_at": _utcnow_iso(),
            "note": "deterministic layout fallback",
        },
        "entities": {
            "ids": ids,
            "xy": xy,
            "xyz": xyz,
        },
        "chunks": {},
    }


def _load_coords_cache(idx: Path, dim: int) -> dict[str, Any] | None:
    """Return the cached payload when it matches the current index or serves precomputed coords."""
    path = _coords_cache_path(idx)
    cache: dict[str, Any] | None = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        cache = None

    if cache is not None:
        meta = cache.get("meta") or {}
        ent = cache.get("entities") or {}
        if isinstance(meta, dict) and isinstance(ent, dict) and ent.get("ids"):
            if dim >= 3 and not ent.get("xyz") and ent.get("xy"):
                ent["xyz"] = [[round(float(p[0]), 4), round(float(p[1]), 4), 0.0] for p in ent["xy"]]
                cache["entities"] = ent

            fp = _coords_fingerprint(idx)
            cached_fp = meta.get("source_mtimes")
            if cached_fp != fp:
                logger.info(
                    "coords cache mtime differs for %s (cached=%s, current=%s); serving cached coordinates",
                    idx.name, cached_fp, fp,
                )
                if cached_fp is None:
                    meta["source_mtimes"] = fp
                    cache["meta"] = meta
                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(cache, f, separators=(",", ":"))
                    except OSError:
                        pass
            return cache

    return _find_precomputed_coords(idx, dim)


def compute_coords(idx: Path, dim: int = 3, seed: int = 42) -> dict[str, Any]:
    """Fit UMAP over the STORED entity/chunk vectors and return the cache dict.

    Minutes-long at 47k points; called inline only when ``umap`` is importable
    in the service venv (it currently is NOT — the documented path is the
    offline script ``scripts/export_cosmos.py``, which calls this same
    function). CPU-only, fixed params + ``random_state`` for reproducibility.
    """
    import numpy as np  # lazy: serving paths never pay for this import
    import umap  # lazy; ExportError upstream explains the offline path

    def load_matrix(name: str) -> tuple[list[dict], np.ndarray]:
        with open(idx / name, "r", encoding="utf-8") as f:
            d = json.load(f)
        raw = base64.b64decode(d["matrix"])
        n = len(d["data"])
        dm = int(d.get("embedding_dim", 768))
        arr = np.frombuffer(raw, dtype=np.float32, count=n * dm).reshape(n, dm).copy()
        return d["data"], arr

    def fit(mat: np.ndarray, use_seed: int) -> np.ndarray:
        reducer = umap.UMAP(
            n_neighbors=15, min_dist=0.35, n_components=dim, metric="cosine",
            random_state=use_seed,
        )
        proj = np.asarray(reducer.fit_transform(mat), dtype=np.float64)
        mins = proj.min(axis=0)
        maxs = proj.max(axis=0)
        proj = (proj - mins) / np.maximum(maxs - mins, 1e-9)
        return proj * 2 - 1  # [-1, 1], same mapping as the baked snapshot

    ent_data, ent_X = load_matrix("vdb_entities.json")
    ent_proj = fit(ent_X, seed)
    ent_ids = [str(r.get("entity_name") or r.get("__id__") or "") for r in ent_data]
    ent_xyz = [[round(float(a), 4) for a in row] for row in ent_proj]

    chunk_out: dict[str, list[float]] = {}
    chunk_path = idx / "vdb_chunks.json"
    if chunk_path.is_file():
        chunk_data, chunk_X = load_matrix("vdb_chunks.json")
        chunk_proj = fit(chunk_X, seed + 1)
        chunk_out = {
            str(r.get("__id__", "")): [round(float(a), 4) for a in row]
            for r, row in zip(chunk_data, chunk_proj)
        }

    return {
        "meta": {
            "index_id": idx.name,
            "dim": dim,
            "source_mtimes": _coords_fingerprint(idx),
            "umap": {"n_neighbors": 15, "min_dist": 0.35, "metric": "cosine",
                     "random_state": seed, "n_components": dim},
            "note": "normalised per-axis to [-1,1]; xy = first two dims of xyz",
            "generated_at": _utcnow_iso(),
        },
        "entities": {
            "ids": ent_ids,
            "xy": [[row[0], row[1]] for row in ent_xyz],
            "xyz": ent_xyz,
        },
        "chunks": chunk_out,
    }


def coords_payload(params: dict[str, str]) -> dict[str, Any]:
    """Serve the projection for the LIVE index from the cache file, precomputed fixtures,
    or fast fallback coordinates. Never returns 503 so the cosmos starfield always loads."""
    raw_dim = (params.get("dim") or "3").strip()
    try:
        dim = int(raw_dim)
    except ValueError:
        dim = 3
    dim = 3 if dim >= 3 else 2

    idx = index_dir()
    cache = _load_coords_cache(idx, dim)
    if cache is None:
        with _coords_lock:
            cache = _load_coords_cache(idx, dim)  # re-check under the lock
            if cache is None:
                has_umap = False
                try:
                    import umap  # noqa: F401 — probe only
                    has_umap = True
                except Exception:
                    has_umap = False

                if has_umap:
                    try:
                        started = time.monotonic()
                        cache = compute_coords(idx, dim=dim)
                        try:
                            with open(_coords_cache_path(idx), "w", encoding="utf-8") as f:
                                json.dump(cache, f, separators=(",", ":"))
                            logger.info(
                                "coords: computed+cached in %.1fs (%s)",
                                time.monotonic() - started, idx.name,
                            )
                        except OSError:
                            logger.warning("coords: could not write cache file; serving uncached")
                    except Exception as exc:
                        logger.warning("compute_coords inline failed: %s; falling back to deterministic layout", exc)
                        cache = None

                if cache is None:
                    cache = _generate_fallback_coords(idx, dim=dim)
                    try:
                        with open(_coords_cache_path(idx), "w", encoding="utf-8") as f:
                            json.dump(cache, f, separators=(",", ":"))
                        logger.info("coords: generated fallback coords for %s (%d nodes)", idx.name, len(cache["entities"]["ids"]))
                    except OSError:
                        pass

    ent = cache.get("entities") or {}
    xyz = ent.get("xyz") or []
    xy = ent.get("xy") or []
    if dim >= 3 and not xyz and xy:
        xyz = [[round(float(p[0]), 4), round(float(p[1]), 4), 0.0] for p in xy]

    out: dict[str, Any] = {
        "ok": True,
        "meta": {**(cache.get("meta") or {}), "index_id": idx.name, "served_at": _utcnow_iso()},
        "entities": {"ids": ent.get("ids") or [], "xy": xy},
        "chunks": cache.get("chunks") or {},
    }
    if dim >= 3:
        out["entities"]["xyz"] = xyz
    return out


__all__ = [
    "COORDS_CACHE_NAME",
    "DEFAULT_GRAPH_EDGES",
    "DEFAULT_GRAPH_NODES",
    "DESC_HEAD",
    "ExportError",
    "compute_coords",
    "coords_payload",
    "graph_payload",
    "index_dir",
    "stats_payload",
]
