"""R7: Russian KB graph quality + graph-mode recall vs naive."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

RU = Path(__file__).resolve().parent
EXP = RU.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(EXP))
import common  # noqa: E402

# --- paths / budget (patch BEFORE any LLM use) ---
common.COST_FILE = RU / "cost.json"
common.BUDGET_USD = 2.0
TASK_BUDGET_USD = 0.30
TASK_TAG_PREFIX = "r7:"
HARD_STOP_USD = 1.9

INDEX_RO = Path("/tmp/ru-exp/index-token")  # read-only quality analysis
INDEX_GRAPH = Path("/tmp/ru-exp/index-graph")  # recall copy
CASES_PATH = RU / "cases_ru.yaml"
CACHE_DIR = RU / "cache"
RESULTS_DIR = RU / "results"
RESULTS_JSON = RESULTS_DIR / "r7_graph.json"
EYE_MD = RESULTS_DIR / "r7_eye.md"
KW_CACHE = CACHE_DIR / "r7_keywords.json"
DUP_CACHE = CACHE_DIR / "r7_dup_judge.json"
JUDGE_CACHE = CACHE_DIR / "r7_halluc_judge.json"
RECALL_CACHE = CACHE_DIR / "r7_recall.json"

SEP = "<SEP>"
CYR = re.compile(r"[А-Яа-яЁё]")
LAT = re.compile(r"[A-Za-z]")
LETTER = re.compile(r"[A-Za-zА-Яа-яЁё]")
PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)

# MCP enum modes; hybrid maps to LightRAG mix internally. Also try mix explicitly.
MODES = ("naive", "local", "global", "hybrid", "mix")
CASE_PICK = {"A": 10, "C": 10, "B": 5, "D": 5}  # total 30
MAX_DUP_PAIRS = 60
N_DOCS_JUDGE = 25
N_ENT_PER_DOC = 3
N_REL_PER_DOC = 2
EYE_SAMPLES = 10
TOP_K = 5


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text("utf-8"))


def _task_spend() -> float:
    data = common._read_cost()
    total = 0.0
    for tag, value in (data.get("by_tag") or {}).items():
        if str(tag).startswith(TASK_TAG_PREFIX):
            total += float(value.get("usd") or 0.0)
    return total


def _guard_call(prompt: str, max_out: int) -> None:
    if common.cost_usd() >= HARD_STOP_USD:
        raise common.BudgetExceeded(
            f"hard stop: total ${common.cost_usd():.4f} >= ${HARD_STOP_USD}"
        )
    est_in = (len(prompt.encode("utf-8")) + 64) / 2
    estimate = (est_in * common.PRICE_IN + max_out * common.PRICE_OUT) / 1_000_000
    if _task_spend() + estimate > TASK_BUDGET_USD:
        raise common.BudgetExceeded(
            f"R7 task budget: spent=${_task_spend():.4f} +est=${estimate:.4f} > ${TASK_BUDGET_USD}"
        )
    if common.cost_usd() + estimate > HARD_STOP_USD:
        raise common.BudgetExceeded(
            f"would exceed hard stop: ${common.cost_usd():.4f}+${estimate:.4f}"
        )


def _cache_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


async def _llm_cached(
    client: Any,
    prompt: str,
    tag: str,
    cache: dict[str, Any],
    cache_path: Path,
    max_out: int = 80,
) -> str:
    key = _cache_key(tag, prompt)
    hit = cache.get(key)
    if isinstance(hit, dict) and isinstance(hit.get("text"), str):
        return hit["text"]
    _guard_call(prompt, max_out)
    text = await common.llm(client, prompt, tag, max_out=max_out)
    cache[key] = {"text": text, "tag": tag}
    _write_json(cache_path, cache)
    return text


def _yes_no(text: str) -> str | None:
    t = (text or "").strip().upper()
    # first YES/NO token
    m = re.search(r"\b(YES|NO)\b", t)
    if m:
        return m.group(1)
    if t.startswith("Y"):
        return "YES"
    if t.startswith("N"):
        return "NO"
    return None


def _norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKC", name or "")
    s = s.replace("ё", "е").replace("Ё", "Е")
    s = s.casefold()
    s = PUNCT_RE.sub("", s)
    return s


def _script_class(name: str) -> str:
    has_c = bool(CYR.search(name or ""))
    has_l = bool(LAT.search(name or ""))
    if has_c and has_l:
        return "mixed"
    if has_c:
        return "cyrillic_only"
    if has_l:
        return "latin_only"
    return "other"


def _cyr_ratio(text: str) -> float:
    letters = LETTER.findall(text or "")
    if not letters:
        return 0.0
    cyr = sum(1 for ch in letters if CYR.match(ch))
    return cyr / len(letters)


def _split_ids(raw: str) -> list[str]:
    if not raw:
        return []
    return [p.strip() for p in str(raw).split(SEP) if p.strip()]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, (centre + margin) / denom)


def _mean_ci(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    lo, hi = common.boot_ci(vals)
    return {"mean": sum(vals) / len(vals), "lo": lo, "hi": hi, "n": len(vals)}


def load_cases() -> list[dict[str, Any]]:
    data = yaml.safe_load(CASES_PATH.read_text("utf-8"))
    if isinstance(data, dict):
        return list(data.get("cases") or [])
    return list(data or [])


def pick_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        by_cell[str(c.get("cell"))].append(c)
    for cell in by_cell:
        by_cell[cell].sort(key=lambda x: str(x["id"]))
    picked: list[dict[str, Any]] = []
    for cell, n in CASE_PICK.items():
        pool = by_cell.get(cell) or []
        if len(pool) < n:
            raise RuntimeError(f"cell {cell}: need {n}, have {len(pool)}")
        # deterministic stride sample
        step = max(1, len(pool) // n)
        idxs = [(i * step) % len(pool) for i in range(n)]
        # unique preserve order
        seen: set[int] = set()
        chosen: list[dict[str, Any]] = []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                chosen.append(pool[i])
        # fill if collisions
        j = 0
        while len(chosen) < n:
            if j not in seen:
                seen.add(j)
                chosen.append(pool[j])
            j += 1
        picked.extend(chosen)
    picked.sort(key=lambda c: str(c["id"]))
    return picked


def load_chunks(index: Path) -> dict[str, dict[str, Any]]:
    return json.loads((index / "kv_store_text_chunks.json").read_text("utf-8"))


# ---------------------------------------------------------------------------
# Part 1 — graph quality
# ---------------------------------------------------------------------------


def graph_stats(index: Path) -> dict[str, Any]:
    gpath = index / "graph_chunk_entity_relation.graphml"
    G = nx.read_graphml(str(gpath))
    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()

    type_dist: Counter[str] = Counter()
    script_dist: Counter[str] = Counter()
    desc_cyr_ratios: list[float] = []
    ent_per_chunk: Counter[str] = Counter()
    rel_per_chunk: Counter[str] = Counter()

    nodes_meta: list[dict[str, Any]] = []
    for node, data in G.nodes(data=True):
        name = str(data.get("entity_id") or node)
        et = str(data.get("entity_type") or "UNKNOWN").lower()
        type_dist[et] += 1
        sc = _script_class(name)
        script_dist[sc] += 1
        desc = str(data.get("description") or "")
        # descriptions may be SEP-joined multi-source
        for part in _split_ids(desc) if SEP in desc else [desc]:
            if LETTER.search(part or ""):
                desc_cyr_ratios.append(_cyr_ratio(part))
        srcs = _split_ids(str(data.get("source_id") or ""))
        for sid in srcs:
            ent_per_chunk[sid] += 1
        nodes_meta.append(
            {
                "name": name,
                "entity_type": et,
                "script": sc,
                "description": desc,
                "source_ids": srcs,
                "file_path": str(data.get("file_path") or ""),
            }
        )

    edges_meta: list[dict[str, Any]] = []
    for u, v, data in G.edges(data=True):
        desc = str(data.get("description") or "")
        srcs = _split_ids(str(data.get("source_id") or ""))
        for sid in srcs:
            rel_per_chunk[sid] += 1
        edges_meta.append(
            {
                "src": str(u),
                "tgt": str(v),
                "description": desc,
                "keywords": str(data.get("keywords") or ""),
                "source_ids": srcs,
                "file_path": str(data.get("file_path") or ""),
            }
        )

    # exact-after-normalisation duplicates
    by_norm: dict[str, list[str]] = defaultdict(list)
    for m in nodes_meta:
        by_norm[_norm_name(m["name"])].append(m["name"])
    exact_dup_groups = {k: sorted(set(v)) for k, v in by_norm.items() if k and len(set(v)) > 1}
    exact_dup_extra = sum(len(v) - 1 for v in exact_dup_groups.values())

    # entities/relations per chunk stats (only chunks that appear)
    ent_counts = list(ent_per_chunk.values()) or [0]
    rel_counts = list(rel_per_chunk.values()) or [0]

    # description language share
    if desc_cyr_ratios:
        ru_share = sum(1 for r in desc_cyr_ratios if r >= 0.3) / len(desc_cyr_ratios)
        en_share = sum(1 for r in desc_cyr_ratios if r < 0.1) / len(desc_cyr_ratios)
        mean_cyr = statistics.mean(desc_cyr_ratios)
    else:
        ru_share = en_share = mean_cyr = 0.0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "entity_type_distribution": dict(type_dist.most_common()),
        "entity_name_script_share": {
            k: v / n_nodes if n_nodes else 0.0 for k, v in script_dist.items()
        },
        "entity_name_script_counts": dict(script_dist),
        "description_language": {
            "mean_cyrillic_letter_ratio": mean_cyr,
            "share_mostly_russian_cyr_ge_0.3": ru_share,
            "share_mostly_english_cyr_lt_0.1": en_share,
            "n_description_parts": len(desc_cyr_ratios),
        },
        "entities_per_chunk": {
            "mean": statistics.mean(ent_counts),
            "median": statistics.median(ent_counts),
            "max": max(ent_counts),
            "chunks_with_entities": len(ent_per_chunk),
        },
        "relations_per_chunk": {
            "mean": statistics.mean(rel_counts),
            "median": statistics.median(rel_counts),
            "max": max(rel_counts),
            "chunks_with_relations": len(rel_per_chunk),
        },
        "exact_norm_duplicates": {
            "n_groups": len(exact_dup_groups),
            "n_extra_nodes": exact_dup_extra,
            "examples": [
                {"norm": k, "names": v}
                for k, v in sorted(exact_dup_groups.items(), key=lambda kv: -len(kv[1]))[:15]
            ],
        },
        "_nodes": nodes_meta,
        "_edges": edges_meta,
        "_ent_per_chunk": dict(ent_per_chunk),
        "_rel_per_chunk": dict(rel_per_chunk),
    }


def cross_lingual_candidates(nodes: list[dict[str, Any]]) -> list[tuple[dict, dict, str]]:
    """Pairs (cyr_entity, lat_entity) sharing a source chunk."""
    by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in nodes:
        for sid in m["source_ids"]:
            by_chunk[sid].append(m)

    pairs: list[tuple[dict, dict, str]] = []
    seen: set[tuple[str, str]] = set()
    for sid, ents in by_chunk.items():
        cyr = [e for e in ents if e["script"] == "cyrillic_only"]
        lat = [e for e in ents if e["script"] == "latin_only"]
        for a in cyr:
            for b in lat:
                key = tuple(sorted([a["name"], b["name"]]))
                if key in seen:
                    continue
                # cheap filter: description token overlap OR shared file
                if not _desc_overlap(a["description"], b["description"]) and not (
                    set(_split_ids(a["file_path"])) & set(_split_ids(b["file_path"]))
                ):
                    # still keep if same chunk (already true) — always candidate
                    pass
                seen.add(key)
                pairs.append((a, b, sid))

    # rank by description overlap then name fuzzy
    def score(pair: tuple[dict, dict, str]) -> float:
        a, b, _ = pair
        return _desc_overlap(a["description"], b["description"]) + 0.1 * _token_jaccard(
            a["name"], b["name"]
        )

    pairs.sort(key=score, reverse=True)
    return pairs


def _tokens(text: str) -> set[str]:
    t = _norm_name(text)
    # split alnum runs from original lower
    parts = re.findall(r"[a-z0-9а-я]+", (text or "").casefold().replace("ё", "е"))
    return set(parts) | ({t} if t else set())


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _desc_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9а-я]{3,}", (a or "").casefold().replace("ё", "е")))
    tb = set(re.findall(r"[a-z0-9а-я]{3,}", (b or "").casefold().replace("ё", "е")))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


async def judge_duplicates(
    client: Any, pairs: list[tuple[dict, dict, str]], chunks: dict[str, dict]
) -> dict[str, Any]:
    cache = _read_json(DUP_CACHE, {})
    if not isinstance(cache, dict):
        cache = {}
    results: list[dict[str, Any]] = []
    confirmed = 0
    limited = pairs[:MAX_DUP_PAIRS]
    for i, (a, b, sid) in enumerate(limited):
        chunk_txt = str((chunks.get(sid) or {}).get("content") or "")[:600]
        prompt = (
            "Are these two names the SAME real-world entity (aliases / translation / "
            "transliteration), based on the descriptions and source excerpt?\n"
            "Answer YES or NO on the first line, then one short reason.\n"
            f"Name A (Cyrillic): {a['name']}\n"
            f"Type A: {a['entity_type']}\n"
            f"Desc A: {(a['description'] or '')[:400]}\n"
            f"Name B (Latin): {b['name']}\n"
            f"Type B: {b['entity_type']}\n"
            f"Desc B: {(b['description'] or '')[:400]}\n"
            f"Source chunk excerpt: {chunk_txt}\n"
        )
        try:
            raw = await _llm_cached(client, prompt, "r7:dup", cache, DUP_CACHE, max_out=700)
        except common.BudgetExceeded as exc:
            results.append({"error": str(exc), "stopped_at": i})
            break
        yn = _yes_no(raw) or "UNKNOWN"
        if yn == "YES":
            confirmed += 1
        results.append(
            {
                "a": a["name"],
                "b": b["name"],
                "source_id": sid,
                "verdict": yn,
                "raw": raw[:200],
            }
        )
    return {
        "n_candidates_total": len(pairs),
        "n_judged": sum(1 for r in results if "verdict" in r),
        "n_confirmed_same_entity": confirmed,
        "pairs": results,
    }


def sample_docs_for_judge(chunks: dict[str, dict[str, Any]], n: int = N_DOCS_JUDGE) -> list[str]:
    """Deterministic document sample spread over folder prefixes."""
    by_doc: dict[str, list[str]] = defaultdict(list)
    for cid, meta in chunks.items():
        sp = str(meta.get("source_path") or meta.get("file_path") or "unknown")
        by_doc[sp].append(cid)
    docs = sorted(by_doc.keys())
    # group by parent folder (deeper path)
    by_folder: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        parts = Path(d).parts
        folder = "/".join(parts[:-1]) if len(parts) > 1 else d
        by_folder[folder].append(d)
    folders = sorted(by_folder.keys())
    rnd = random.Random(7)
    # round-robin folders
    chosen: list[str] = []
    ptr = {f: 0 for f in folders}
    # shuffle docs inside folder deterministically
    for f in folders:
        by_folder[f] = sorted(by_folder[f])
        rnd.shuffle(by_folder[f])
    while len(chosen) < min(n, len(docs)):
        progress = False
        for f in folders:
            i = ptr[f]
            if i < len(by_folder[f]):
                chosen.append(by_folder[f][i])
                ptr[f] = i + 1
                progress = True
                if len(chosen) >= n:
                    break
        if not progress:
            break
    return chosen[:n]


async def hallucination_judge(
    client: Any,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cache = _read_json(JUDGE_CACHE, {})
    if not isinstance(cache, dict):
        cache = {}

    docs = sample_docs_for_judge(chunks)
    # map doc -> chunk ids
    doc_chunks: dict[str, list[str]] = defaultdict(list)
    for cid, meta in chunks.items():
        sp = str(meta.get("source_path") or meta.get("file_path") or "")
        doc_chunks[sp].append(cid)

    # index entities/rels by primary source chunk
    ents_by_chunk: dict[str, list[dict]] = defaultdict(list)
    for m in nodes:
        for sid in m["source_ids"][:1]:  # primary
            ents_by_chunk[sid].append(m)
    rels_by_chunk: dict[str, list[dict]] = defaultdict(list)
    for m in edges:
        for sid in m["source_ids"][:1]:
            rels_by_chunk[sid].append(m)

    rnd = random.Random(11)
    items: list[dict[str, Any]] = []
    for doc in docs:
        cids = sorted(doc_chunks.get(doc) or [])
        # gather ents/rels for this doc's chunks
        ents: list[dict] = []
        rels: list[dict] = []
        for cid in cids:
            ents.extend(ents_by_chunk.get(cid) or [])
            rels.extend(rels_by_chunk.get(cid) or [])
        # unique by name/pair
        seen_e: set[str] = set()
        uniq_e = []
        for e in ents:
            if e["name"] not in seen_e:
                seen_e.add(e["name"])
                uniq_e.append(e)
        seen_r: set[tuple[str, str]] = set()
        uniq_r = []
        for r in rels:
            key = (r["src"], r["tgt"])
            if key not in seen_r:
                seen_r.add(key)
                uniq_r.append(r)
        rnd.shuffle(uniq_e)
        rnd.shuffle(uniq_r)
        for e in uniq_e[:N_ENT_PER_DOC]:
            sid = (e["source_ids"] or [None])[0]
            items.append(
                {
                    "kind": "entity",
                    "doc": doc,
                    "name": e["name"],
                    "entity_type": e["entity_type"],
                    "description": e["description"],
                    "source_id": sid,
                }
            )
        for r in uniq_r[:N_REL_PER_DOC]:
            sid = (r["source_ids"] or [None])[0]
            items.append(
                {
                    "kind": "relation",
                    "doc": doc,
                    "src": r["src"],
                    "tgt": r["tgt"],
                    "description": r["description"],
                    "source_id": sid,
                }
            )

    judgments: list[dict[str, Any]] = []
    nos = 0
    for i, it in enumerate(items):
        sid = it.get("source_id")
        chunk_txt = str((chunks.get(sid) or {}).get("content") or "")[:4000]
        if it["kind"] == "entity":
            subject = f"Entity name: {it['name']}\nType: {it['entity_type']}\nDescription: {(it['description'] or '')[:500]}"
        else:
            subject = (
                f"Relation: {it['src']} -> {it['tgt']}\n"
                f"Description: {(it['description'] or '')[:500]}"
            )
        prompt = (
            "Is this entity/relation description faithful to (supported by) the source text?\n"
            "Answer YES or NO on the first line, then one short reason.\n\n"
            f"{subject}\n\nSource chunk text:\n{chunk_txt}\n"
        )
        try:
            raw = await _llm_cached(client, prompt, "r7:judge", cache, JUDGE_CACHE, max_out=700)
        except common.BudgetExceeded as exc:
            judgments.append({"error": str(exc), "stopped_at": i})
            break
        yn = _yes_no(raw) or "UNKNOWN"
        if yn == "NO":
            nos += 1
        judgments.append({**it, "verdict": yn, "raw": raw[:240], "chunk_excerpt": chunk_txt[:400]})

    judged = [j for j in judgments if "verdict" in j]
    n = len(judged)
    k_no = sum(1 for j in judged if j["verdict"] == "NO")
    rate = k_no / n if n else 0.0
    wlo, whi = _wilson(k_no, n)
    boot_vals = [1.0 if j["verdict"] == "NO" else 0.0 for j in judged]
    blo, bhi = common.boot_ci(boot_vals) if boot_vals else (0.0, 0.0)

    # eye samples
    eye_pool = [j for j in judged if j.get("chunk_excerpt")]
    rnd2 = random.Random(13)
    eye = eye_pool[:]
    rnd2.shuffle(eye)
    eye = eye[:EYE_SAMPLES]

    return {
        "n_docs_sampled": len(docs),
        "docs": docs,
        "n_items": len(items),
        "n_judged": n,
        "n_no": k_no,
        "n_yes": sum(1 for j in judged if j["verdict"] == "YES"),
        "n_unknown": sum(1 for j in judged if j["verdict"] == "UNKNOWN"),
        "hallucination_rate": rate,
        "wilson_ci": {"lo": wlo, "hi": whi},
        "bootstrap_ci": {"lo": blo, "hi": bhi},
        "judgments": judgments,
        "eye_samples": eye,
    }


def write_eye_md(eye: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# R7 eye samples (entity/relation vs source chunk)",
        "",
        "Manual inspection set — 10 random judged items.",
        "",
    ]
    for i, j in enumerate(eye, 1):
        lines.append(f"## {i}. {j.get('kind')} — verdict **{j.get('verdict')}**")
        if j.get("kind") == "entity":
            lines.append(f"- name: `{j.get('name')}` ({j.get('entity_type')})")
        else:
            lines.append(f"- relation: `{j.get('src')}` → `{j.get('tgt')}`")
        lines.append(f"- doc: `{j.get('doc')}`")
        lines.append(f"- source_id: `{j.get('source_id')}`")
        lines.append(f"- description: {(j.get('description') or '')[:500]}")
        lines.append(f"- judge: {(j.get('raw') or '')[:200]}")
        lines.append("")
        lines.append("```")
        lines.append((j.get("chunk_excerpt") or "")[:400])
        lines.append("```")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Part 2 — graph modes vs naive
# ---------------------------------------------------------------------------


def _configure_runtime() -> None:
    common.load_env_file()
    os.environ.update(
        {
            "HARS_MEMORY_INDEX_DIR": str(INDEX_GRAPH),
            "HARS_MEMORY_STAGING_DIR": "/tmp/ru-exp/staging",
            "HARS_MEMORY_RIPGREP_CHANNEL": "0",
            "HARS_MEMORY_EMBED_MODEL": "unsloth/embeddinggemma-300m",
            "HARS_MEMORY_EMBED_LOCAL_FILES_ONLY": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_HOME": str(Path.home() / ".cache" / "huggingface"),
            "HARS_MEMORY_RERANK_BACKEND": "off",
        }
    )


def _string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        t = str(item).strip()
        if t and t not in out:
            out.append(t)
        if len(out) >= limit:
            break
    return out


async def load_keywords(
    cases: list[dict[str, Any]], client: Any
) -> dict[str, dict[str, list[str]]]:
    data = _read_json(KW_CACHE, {})
    if not isinstance(data, dict):
        data = {}
    for case in cases:
        cid = str(case["id"])
        cached = data.get(cid)
        if isinstance(cached, dict) and cached.get("ll_keywords") and cached.get("hl_keywords"):
            continue
        prompt = (
            "Извлеки поисковые ключевые слова для вопроса к базе знаний. "
            "Ответь ТОЛЬКО JSON: "
            '{"ll_keywords":[2-4 конкретных сущности/идентификатора], '
            '"hl_keywords":[2-4 тематических фразы]}. '
            "Ключевые слова на русском (идентификаторы латиницей сохрани как есть). "
            "Не выдумывай сущности.\n"
            f"Вопрос: {case['question']}\n"
            f"Gold fact (только для дисамбигуации): {case.get('gold_fact', '')}"
        )
        try:
            _guard_call(prompt, 400)
            raw = await common.llm(client, prompt, f"r7:kw:{cid}", max_out=400)
        except common.BudgetExceeded:
            # fallback heuristic keywords from question tokens
            toks = re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9_-]{2,}", case["question"])
            data[cid] = {
                "ll_keywords": toks[:4] or [case["question"][:40]],
                "hl_keywords": toks[:3] or ["тема"],
                "fallback": True,
            }
            _write_json(KW_CACHE, data)
            continue
        parsed = common.parse_json(raw) if raw.strip() else None
        if not isinstance(parsed, dict):
            toks = re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9_-]{2,}", case["question"])
            data[cid] = {
                "ll_keywords": toks[:4] or [str(case["question"])[:40]],
                "hl_keywords": toks[:3] or ["тема"],
                "fallback": True,
            }
        else:
            data[cid] = {
                "ll_keywords": _string_list(parsed.get("ll_keywords"), 4),
                "hl_keywords": _string_list(parsed.get("hl_keywords"), 4),
            }
            if not data[cid]["ll_keywords"]:
                data[cid]["ll_keywords"] = [str(case["question"])[:40]]
            if not data[cid]["hl_keywords"]:
                data[cid]["hl_keywords"] = data[cid]["ll_keywords"][:2]
        _write_json(KW_CACHE, data)
    return data  # type: ignore[return-value]


async def run_recalls(
    cases: list[dict[str, Any]],
    keywords: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    from hars_memory import mcp_server  # after env configured

    cache = _read_json(RECALL_CACHE, {})
    if not isinstance(cache, dict):
        cache = {}

    for case in cases:
        cid = str(case["id"])
        case_cache = cache.setdefault(cid, {})
        kw = keywords.get(cid) or {}
        for mode in MODES:
            if mode in case_cache and isinstance(case_cache[mode].get("response"), str):
                continue
            args: dict[str, Any] = {
                "question": str(case["question"]),
                "top_k": TOP_K,
                "context_only": True,
                "mode": mode,
                "view": "full",
            }
            if mode != "naive":
                args["ll_keywords"] = list(kw.get("ll_keywords") or [])
                args["hl_keywords"] = list(kw.get("hl_keywords") or [])
            t0 = time.perf_counter()
            err = None
            raw = ""
            try:
                result = await mcp_server.call_tool("memory_recall", args)
                if not result:
                    raise RuntimeError("empty result")
                text = getattr(result[0], "text", None)
                if not isinstance(text, str):
                    raise TypeError("no text")
                raw = text
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                raw = json.dumps({"ok": False, "error": err}, ensure_ascii=False)
            latency = time.perf_counter() - t0
            fact = common.ws(str(case.get("gold_fact") or ""))
            hit = bool(fact and fact in common.ws(raw))
            case_cache[mode] = {
                "arguments": {k: v for k, v in args.items()},
                "response": raw,
                "latency_s": latency,
                "chars": len(raw),
                "hit": hit,
                "error": err,
            }
            _write_json(RECALL_CACHE, cache)
            print(
                f"  recall {cid} {mode}: hit={int(hit)} {latency:.2f}s chars={len(raw)}"
                + (f" ERR={err}" if err else ""),
                file=sys.stderr,
            )
    return cache  # type: ignore[return-value]


def summarize_modes(
    cases: list[dict[str, Any]],
    cache: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    per_mode: dict[str, Any] = {}
    hit_vectors: dict[str, list[float]] = {}
    for mode in MODES:
        hits: list[float] = []
        lats: list[float] = []
        chars: list[float] = []
        errs = 0
        by_cell: dict[str, list[float]] = defaultdict(list)
        for case in cases:
            rec = (cache.get(str(case["id"])) or {}).get(mode) or {}
            h = 1.0 if rec.get("hit") else 0.0
            hits.append(h)
            lats.append(float(rec.get("latency_s") or 0.0))
            chars.append(float(rec.get("chars") or 0.0))
            if rec.get("error"):
                errs += 1
            by_cell[str(case.get("cell"))].append(h)
        hit_vectors[mode] = hits
        cell_sum = {cell: _mean_ci(vals) for cell, vals in sorted(by_cell.items())}
        per_mode[mode] = {
            "n": len(hits),
            "hit_rate": _mean_ci(hits),
            "latency_s": _mean_ci(lats),
            "chars": _mean_ci(chars),
            "n_errors": errs,
            "by_cell": cell_sum,
        }
    paired: dict[str, Any] = {}
    base = hit_vectors.get("naive") or []
    for mode in MODES:
        if mode == "naive":
            continue
        md, lo, hi = common.paired_diff_ci(hit_vectors[mode], base)
        paired[f"{mode}_minus_naive"] = {
            "mean_diff": md,
            "lo": lo,
            "hi": hi,
            "note": "CI containing 0 => noise",
        }
    return {"per_mode": per_mode, "paired_vs_naive": paired}


def fmt_ci(d: dict[str, float]) -> str:
    return f"{d['mean']:.3f} [{d['lo']:.3f},{d['hi']:.3f}]"


def print_tables(quality: dict[str, Any], modes: dict[str, Any]) -> None:
    print("\n### Graph quality")
    print("| metric | value |")
    print("|---|---|")
    print(f"| nodes | {quality['n_nodes']} |")
    print(f"| edges | {quality['n_edges']} |")
    sc = quality["entity_name_script_share"]
    print(
        f"| name script cyr/lat/mix | "
        f"{sc.get('cyrillic_only', 0):.2f}/"
        f"{sc.get('latin_only', 0):.2f}/"
        f"{sc.get('mixed', 0):.2f} |"
    )
    dl = quality["description_language"]
    print(f"| desc mean cyr ratio | {dl['mean_cyrillic_letter_ratio']:.3f} |")
    print(f"| desc mostly RU share | {dl['share_mostly_russian_cyr_ge_0.3']:.3f} |")
    print(f"| ents/chunk mean | {quality['entities_per_chunk']['mean']:.2f} |")
    print(f"| rels/chunk mean | {quality['relations_per_chunk']['mean']:.2f} |")
    ed = quality["exact_norm_duplicates"]
    print(f"| exact-norm dup groups (extra) | {ed['n_groups']} ({ed['n_extra_nodes']}) |")
    xd = quality.get("cross_lingual_duplicates") or {}
    print(
        f"| xl dup judged/confirmed | {xd.get('n_judged', 0)}/{xd.get('n_confirmed_same_entity', 0)} |"
    )
    hj = quality.get("hallucination") or {}
    print(
        f"| halluc rate (n={hj.get('n_judged', 0)}) | "
        f"{hj.get('hallucination_rate', 0):.3f} "
        f"W[{hj.get('wilson_ci', {}).get('lo', 0):.3f},"
        f"{hj.get('wilson_ci', {}).get('hi', 0):.3f}] |"
    )
    types = quality["entity_type_distribution"]
    top_t = ", ".join(f"{k}:{v}" for k, v in list(types.items())[:6])
    print(f"| entity types (top) | {top_t} |")

    print("\n### Modes vs naive (gold_fact in response)")
    print("| mode | n | hit_rate mean [CI] | lat_s | chars | err | Δ vs naive [CI] |")
    print("|---|---:|---|---:|---:|---:|---|")
    paired = modes.get("paired_vs_naive") or {}
    for mode, row in (modes.get("per_mode") or {}).items():
        hr = row["hit_rate"]
        delta = "—"
        if mode != "naive":
            p = paired.get(f"{mode}_minus_naive") or {}
            delta = f"{p.get('mean_diff', 0):+.3f} [{p.get('lo', 0):.3f},{p.get('hi', 0):.3f}]"
        print(
            f"| {mode} | {row['n']} | {fmt_ci(hr)} | "
            f"{row['latency_s']['mean']:.2f} | {row['chars']['mean']:.0f} | "
            f"{row['n_errors']} | {delta} |"
        )
    print("\nCaveat: context_only=True (no answer LLM). hybrid MCP mode maps to LightRAG `mix`.")
    print(
        f"LLM task spend ${_task_spend():.4f} / ${TASK_BUDGET_USD:.2f}; total ${common.cost_usd():.4f}"
    )


async def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if common.cost_usd() >= HARD_STOP_USD:
        print(f"ABORT: cost already ${common.cost_usd():.4f} >= {HARD_STOP_USD}", file=sys.stderr)
        return 2

    # ensure graph index copy
    if not INDEX_GRAPH.is_dir():
        import shutil

        shutil.copytree(INDEX_RO, INDEX_GRAPH)

    print("== Part 1: graph quality ==", file=sys.stderr)
    quality = graph_stats(INDEX_RO)
    chunks = load_chunks(INDEX_RO)
    nodes = quality.pop("_nodes")
    edges = quality.pop("_edges")
    quality.pop("_ent_per_chunk", None)
    quality.pop("_rel_per_chunk", None)

    client = common.make_client()

    pairs = cross_lingual_candidates(nodes)
    print(f"  cross-lingual candidate pairs: {len(pairs)}", file=sys.stderr)
    try:
        xl = await judge_duplicates(client, pairs, chunks)
    except common.BudgetExceeded as exc:
        xl = {"error": str(exc), "n_candidates_total": len(pairs)}
    quality["cross_lingual_duplicates"] = xl

    print("  hallucination judge…", file=sys.stderr)
    try:
        hall = await hallucination_judge(client, nodes, edges, chunks)
    except common.BudgetExceeded as exc:
        hall = {"error": str(exc)}
    quality["hallucination"] = {
        k: v for k, v in hall.items() if k not in ("judgments", "eye_samples")
    }
    write_eye_md(hall.get("eye_samples") or [], EYE_MD)
    # keep compact judgment summary counts only in main json; full optional
    quality["hallucination"]["judgment_verdicts"] = dict(
        Counter(j.get("verdict") for j in (hall.get("judgments") or []) if "verdict" in j)
    )
    # drop heavy pair details from main result (keep counts + examples)
    if isinstance(quality.get("cross_lingual_duplicates"), dict):
        xl = quality["cross_lingual_duplicates"]
        pairs = xl.get("pairs") or []
        xl["pair_examples"] = [p for p in pairs if p.get("verdict") == "YES"][:10]
        xl["pairs"] = pairs  # keep for cache re-use transparency; modest size

    print("== Part 2: modes vs naive ==", file=sys.stderr)
    _configure_runtime()
    # import mcp after env
    cases_all = load_cases()
    cases = pick_cases(cases_all)
    print(
        f"  picked {len(cases)} cases: "
        + ", ".join(f"{k}={sum(1 for c in cases if c.get('cell') == k)}" for k in CASE_PICK),
        file=sys.stderr,
    )
    keywords = await load_keywords(cases, client)
    cache = await run_recalls(cases, keywords)
    modes_sum = summarize_modes(cases, cache)

    out = {
        "task": "r7_graph",
        "index_ro": str(INDEX_RO),
        "index_graph": str(INDEX_GRAPH),
        "budget": {
            "task_usd_cap": TASK_BUDGET_USD,
            "task_usd_spent": _task_spend(),
            "total_usd": common.cost_usd(),
        },
        "caveat": (
            "memory_recall used context_only=True so no answer-generation LLM; "
            "hl/ll keywords pre-supplied to skip query-time keyword LLM. "
            "MCP mode 'hybrid' maps to LightRAG mode 'mix' via _lightrag_mode; "
            "mode 'mix' is passed through if accepted."
        ),
        "graph_quality": quality,
        "cases_selected": [
            {"id": c["id"], "cell": c["cell"], "question": c["question"]} for c in cases
        ],
        "keywords": keywords,
        "modes": modes_sum,
        "per_case_hits": {
            cid: {m: (rec.get(m) or {}).get("hit") for m in MODES}
            for cid, rec in cache.items()
            if cid in {str(c["id"]) for c in cases}
        },
    }
    # strip heavy judgment lists already summarized
    _write_json(RESULTS_JSON, out)
    print_tables(quality, modes_sum)
    print("DONE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except common.BudgetExceeded as exc:
        print(f"BUDGET: {exc}", file=sys.stderr)
        print("DONE")
        raise SystemExit(3)
