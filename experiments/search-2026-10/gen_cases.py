#!/usr/bin/env python3
"""Generate ~120 stratified retrieval eval cases for the 2026-10 search experiment.

Usage:
  uv run python experiments/search-2026-10/gen_cases.py
"""

from __future__ import annotations

import asyncio
import difflib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from common import (  # noqa: E402
    CLONE,
    EXP,
    SOURCE_ROOT,
    BudgetExceeded,
    Doc,
    cost_usd,
    flat_name,
    inventory,
    llm,
    load_env_file,
    make_client,
    parse_json,
    ws,
)

CHUNKS_PATH = CLONE / "kv_store_text_chunks.json"
OUT_PATH = EXP / "cases.yaml"
CONCURRENCY = 6
TARGET_DOCS = 62
TARGET_CASES = 120
SEED_SAMPLE = 11
MIN_CHARS, MAX_CHARS = 600, 12000

PREFIXES = ("kb/", "repos/sre-docs/docs/")
AREA_PREFIXES = (
    "kb/k8s/",
    "kb/services/",
    "kb/aws/",
    "kb/gcp/",
    "kb/monitoring/",
    "kb/tools/",
    "repos/sre-docs/docs/",
)

QTYPES = ("identifier", "table", "code", "list", "deep_heading", "vague", "conceptual")

GEN_PROMPT = """You write retrieval-evaluation questions for a documentation search system.

Below is ONE markdown document (do NOT put the file path or file name into any question).
Write {n} question/answer-evidence items about it as a JSON list.

Hard rules:
- gold_fact MUST be copied CHARACTER FOR CHARACTER from the document: a contiguous substring
  of 40-400 characters. Keep markdown (pipes, backticks, bullets, newlines) exactly. Never paraphrase.
- question is what a human engineer would type into search. Self-contained. Do NOT quote gold_fact.
  Do NOT mention the file name or path. It must be answerable from gold_fact alone.
- Each item needs a different gold_fact.
- qtype is one of: identifier, table, code, list, deep_heading, vague, conceptual.
  identifier = the question MUST contain an exact identifier / flag / resource name / command
    taken from the document (copy it verbatim into the question).
  table = answer lives in a markdown table row; gold_fact should include the full "| ... |" row.
  code = answer is inside a fenced code block or is a shell/config snippet.
  list = answer is a bullet or numbered list item.
  deep_heading = fact sits under a heading of level H4 or deeper.
  vague = short under-specified natural question a human would ask the KB (still answerable by gold_fact).
  conceptual = higher-level "how/why/what is" question grounded in a specific passage.
- lang is "ru" or "en". About 35% of items should use Russian phrasing for the question
  (asking in Russian about English doc content is fine). Keep identifiers/commands/resource names
  in the original English spelling inside Russian questions.

Structure present in this document (prefer covering available structure):
{inventory}

Requested mix for these {n} items (adapt if structure missing): {mix_hint}

Return ONLY a JSON array:
[{{"question": "...", "gold_fact": "...", "qtype": "...", "lang": "en"|"ru"}}, ...]

DOCUMENT:
<<<
{doc}
>>>"""


def _area_of(rel: str) -> str:
    for p in AREA_PREFIXES:
        if rel.startswith(p):
            return p.rstrip("/")
    if rel.startswith("kb/"):
        parts = rel.split("/")
        if len(parts) >= 2:
            return f"kb/{parts[1]}"
    return "other"


def _doc_features(text: str, d: Doc) -> dict[str, bool]:
    tbl = any(x.lstrip().startswith("|") for x in d.lines)
    fen = any(d.fence_line)
    lis = sum(1 for x in d.lines if re.match(r"^\s*([-*+]|\d+[.)])\s+", x)) >= 6
    deep = any(lv >= 4 for _, lv, _ in d.headings)
    return {"table": tbl, "code": fen, "list": lis, "deep": deep}


def load_index() -> tuple[dict[str, dict[str, Any]], dict[str, list[tuple[str, dict[str, Any]]]]]:
    raw = json.loads(CHUNKS_PATH.read_text("utf-8"))
    by_src: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for cid, meta in raw.items():
        if not isinstance(meta, dict):
            continue
        sp = meta.get("source_path") or meta.get("file_path") or ""
        if not sp:
            continue
        by_src[str(sp)].append((str(cid), meta))
    return raw, by_src


def candidate_docs(
    by_src: dict[str, list[tuple[str, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []
    for sp, chunks in by_src.items():
        if not (sp.startswith("kb/") or sp.startswith("repos/sre-docs/docs/")):
            continue
        path = SOURCE_ROOT / sp
        if not path.is_file():
            continue
        try:
            text = path.read_text("utf-8")
        except OSError:
            continue
        n = len(text)
        if n < MIN_CHARS or n > MAX_CHARS:
            continue
        d = Doc(text)
        feats = _doc_features(text, d)
        cands.append(
            {
                "source_path": sp,
                "text": text,
                "doc": d,
                "chunks": chunks,
                "features": feats,
                "area": _area_of(sp),
                "basename": Path(sp).name.lower(),
                "head": text[:1500],
            }
        )
    return cands


def build_dup_map(cands: list[dict[str, Any]]) -> dict[str, list[str]]:
    """source_path -> other source_paths with same basename or quick_ratio>0.8 on first 1500 chars."""
    by_base: dict[str, list[str]] = defaultdict(list)
    for c in cands:
        by_base[c["basename"]].append(c["source_path"])

    out: dict[str, list[str]] = {c["source_path"]: [] for c in cands}

    for _base, group in by_base.items():
        if len(group) < 2:
            continue
        for p in group:
            out[p].extend(x for x in group if x != p)

    # Content near-dups: only compare within same top-level area and similar length.
    # Use quick_ratio only (spec); skip pairs already linked by basename.
    by_area: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cands:
        by_area[c["area"]].append(c)

    linked: set[tuple[str, str]] = set()
    for group in by_area.values():
        m = len(group)
        for i in range(m):
            pi, hi = group[i]["source_path"], group[i]["head"]
            li = len(hi)
            known_i = set(out[pi])
            for j in range(i + 1, m):
                pj, hj = group[j]["source_path"], group[j]["head"]
                if pj in known_i:
                    continue
                lj = len(hj)
                if li and lj:
                    if min(li, lj) / max(li, lj) < 0.7:
                        continue
                if difflib.SequenceMatcher(None, hi, hj).quick_ratio() > 0.8:
                    out[pi].append(pj)
                    out[pj].append(pi)
                    known_i.add(pj)
                    linked.add((pi, pj))

    for p in out:
        out[p] = sorted(set(out[p]))
    return out


def stratified_sample(cands: list[dict[str, Any]], dup_map: dict[str, list[str]], k: int = TARGET_DOCS) -> list[dict[str, Any]]:
    rnd = random.Random(SEED_SAMPLE)
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()

    def take(pool: list[dict[str, Any]], n: int) -> None:
        pool = [c for c in pool if c["source_path"] not in seen]
        rnd.shuffle(pool)
        for c in pool[:n]:
            seen.add(c["source_path"])
            picked.append(c)

    tables = [c for c in cands if c["features"]["table"]]
    codes = [c for c in cands if c["features"]["code"]]
    lists = [c for c in cands if c["features"]["list"]]
    deeps = [c for c in cands if c["features"]["deep"]]
    dups = [c for c in cands if dup_map.get(c["source_path"])]

    # ensure coverage of structural strata
    take(tables, 10)
    take(codes, 10)
    take(lists, 8)
    take(deeps, 8)
    take(dups, 10)

    remaining = k - len(picked)
    if remaining > 0:
        # proportional across areas among remaining candidates
        rest = [c for c in cands if c["source_path"] not in seen]
        by_area: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c in rest:
            by_area[c["area"]].append(c)
        areas = sorted(by_area.keys())
        if areas:
            # allocate roughly proportional to pool sizes
            total = sum(len(by_area[a]) for a in areas) or 1
            alloc: dict[str, int] = {}
            assigned = 0
            for a in areas:
                n = max(1, round(remaining * len(by_area[a]) / total)) if by_area[a] else 0
                alloc[a] = n
                assigned += n
            # fix rounding
            while assigned > remaining:
                for a in sorted(alloc, key=lambda x: -alloc[x]):
                    if alloc[a] > 0 and assigned > remaining:
                        alloc[a] -= 1
                        assigned -= 1
            while assigned < remaining:
                for a in sorted(areas, key=lambda x: -len(by_area[x])):
                    if assigned >= remaining:
                        break
                    if by_area[a]:
                        alloc[a] = alloc.get(a, 0) + 1
                        assigned += 1
            for a in areas:
                take(by_area[a], alloc.get(a, 0))

    # fill any shortfall randomly
    if len(picked) < k:
        take([c for c in cands if c["source_path"] not in seen], k - len(picked))

    rnd.shuffle(picked)
    return picked[:k]


def gold_chunk_ids(fact: str, chunks: list[tuple[str, dict[str, Any]]]) -> list[str]:
    nf = ws(fact)
    if not nf:
        return []
    hits: list[str] = []
    for cid, meta in chunks:
        content = str(meta.get("content") or "")
        if nf in ws(content):
            hits.append(cid)
    return hits


def detected_kind(doc: Doc, fact: str, text: str) -> str:
    off = text.find(fact)
    if off < 0:
        # try whitespace-flexible locate: fall back to prose
        return "prose"
    kinds = doc.detect(off, off + len(fact))
    prio = ["table_row", "code_block", "list", "duplicate_heading", "deep_heading", "cross_chunk", "prose"]
    for k in prio:
        if k in kinds:
            return k
    return "prose"


def heading_path_for_fact(chunks: list[tuple[str, dict[str, Any]]], fact: str, doc: Doc, text: str) -> list[str]:
    nf = ws(fact)
    for _cid, meta in chunks:
        content = str(meta.get("content") or "")
        if nf in ws(content):
            hp = meta.get("heading_path")
            if isinstance(hp, list) and hp:
                return [str(x) for x in hp]
            if isinstance(hp, str) and hp.strip():
                return [hp]
    off = text.find(fact)
    if off >= 0:
        return doc.path_at(doc.line_of(off))
    return []


def mix_hint_for(feats: dict[str, bool]) -> str:
    parts = []
    if feats["table"]:
        parts.append("1 table")
    if feats["code"]:
        parts.append("1 code")
    if feats["list"]:
        parts.append("1 list")
    if feats["deep"]:
        parts.append("1 deep_heading")
    parts.append("1 identifier")
    parts.append("optional vague/conceptual")
    return ", ".join(parts)


async def gen_for_doc(
    client: Any,
    sem: asyncio.Semaphore,
    c: dict[str, Any],
    dup_map: dict[str, list[str]],
    seen_facts: set[str],
    lock: asyncio.Lock,
) -> list[dict[str, Any]]:
    sp = c["source_path"]
    text = c["text"]
    doc: Doc = c["doc"]
    chunks = c["chunks"]
    name = flat_name(sp)
    # keep prompt size reasonable
    doc_for_prompt = text if len(text) <= 14000 else text[:14000]
    n = 3
    prompt = GEN_PROMPT.format(
        n=n,
        inventory=inventory(doc),
        mix_hint=mix_hint_for(c["features"]),
        doc=doc_for_prompt,
    )
    kept: list[dict[str, Any]] = []

    async with sem:
        for attempt in range(3):
            try:
                raw = await llm(client, prompt, tag=f"gen:{name}:{attempt}")
                items = parse_json(raw)
                if not isinstance(items, list):
                    raise ValueError("not a list")
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"  {sp}: attempt {attempt + 1} failed ({exc!r}), retry", file=sys.stderr)
                await asyncio.sleep(1.5 * (attempt + 1))
                continue

            local: list[dict[str, Any]] = []
            for it in items if isinstance(items, list) else []:
                if not isinstance(it, dict):
                    continue
                fact = str(it.get("gold_fact") or "")
                q = str(it.get("question") or "").strip()
                qtype = str(it.get("qtype") or "conceptual").strip().lower()
                lang = str(it.get("lang") or "en").strip().lower()
                if lang not in ("ru", "en"):
                    lang = "en"
                if qtype not in QTYPES:
                    qtype = "conceptual"
                if not q or not fact:
                    continue
                if len(fact) < 40 or len(fact) > 400:
                    # allow slight slack if still a substring
                    if len(fact) < 30 or len(fact) > 500:
                        continue
                if fact not in text:
                    continue
                # do not leak path/name into question
                low_q = q.lower()
                if name.lower() in low_q or sp.lower() in low_q:
                    # strip-reject if filename appears
                    continue
                gids = gold_chunk_ids(fact, chunks)
                if not gids:
                    continue
                nf = ws(fact)
                async with lock:
                    if nf in seen_facts:
                        continue
                    seen_facts.add(nf)
                local.append(
                    {
                        "question": q,
                        "gold_file": sp,
                        "gold_fact": fact,
                        "gold_chunk_ids": gids,
                        "qtype": qtype,
                        "lang": lang,
                        "detected_kind": detected_kind(doc, fact, text),
                        "heading_path": heading_path_for_fact(chunks, fact, doc, text),
                        "dup_group": list(dup_map.get(sp) or []),
                    }
                )
            kept.extend(local)
            print(
                f"  {sp}: attempt {attempt + 1}: got {len(items)} raw, {len(local)} kept (doc total {len(kept)})",
                file=sys.stderr,
            )
            if len(kept) >= 2:
                break
            await asyncio.sleep(0.5)

    return kept[:3]


def balance_cases(cases: list[dict[str, Any]], limit: int = TARGET_CASES) -> list[dict[str, Any]]:
    """Keep up to `limit` cases balancing qtype and lang."""
    if len(cases) <= limit:
        rnd = random.Random(SEED_SAMPLE)
        ordered = list(cases)
        rnd.shuffle(ordered)
        return ordered

    rnd = random.Random(SEED_SAMPLE)
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        by_key[(c.get("qtype") or "conceptual", c.get("lang") or "en")].append(c)
    for v in by_key.values():
        rnd.shuffle(v)

    # target ~35% Russian
    target_ru = int(round(limit * 0.35))
    per_qtype = max(1, limit // len(QTYPES))

    selected: list[dict[str, Any]] = []
    used: set[int] = set()

    def add_from(pool: list[dict[str, Any]], n: int) -> int:
        got = 0
        for c in pool:
            if got >= n:
                break
            i = id(c)
            if i in used:
                continue
            used.add(i)
            selected.append(c)
            got += 1
        return got

    # first pass: per qtype quotas, split lang
    for qt in QTYPES:
        ru_pool = by_key.get((qt, "ru"), [])
        en_pool = by_key.get((qt, "en"), [])
        want = per_qtype
        want_ru = max(1, int(round(want * 0.35))) if ru_pool else 0
        want_en = want - want_ru
        add_from(ru_pool, want_ru)
        add_from(en_pool, want_en)
        # fill remainder of qtype from either
        need = want - sum(1 for c in selected if c["qtype"] == qt)
        add_from(ru_pool + en_pool, need)

    # adjust global lang balance
    def count_lang(lang: str) -> int:
        return sum(1 for c in selected if c.get("lang") == lang)

    # fill up to limit from leftovers preferring underrepresented lang/qtype
    leftovers = [c for c in cases if id(c) not in used]
    rnd.shuffle(leftovers)

    while len(selected) < limit and leftovers:
        need_ru = count_lang("ru") < target_ru
        pick_i = None
        for i, c in enumerate(leftovers):
            if need_ru and c.get("lang") == "ru":
                pick_i = i
                break
            if not need_ru and c.get("lang") == "en":
                pick_i = i
                break
        if pick_i is None:
            pick_i = 0
        c = leftovers.pop(pick_i)
        used.add(id(c))
        selected.append(c)

    # if still over lang, swap is optional; trim is enough
    rnd.shuffle(selected)
    selected = selected[:limit]

    # rebalance slightly if RU far off
    ru_n = count_lang("ru")
    if ru_n < target_ru - 5:
        extras_ru = [c for c in cases if id(c) not in {id(x) for x in selected} and c.get("lang") == "ru"]
        for i, c in enumerate(list(selected)):
            if c.get("lang") == "en" and extras_ru and count_lang("ru") < target_ru:
                selected[i] = extras_ru.pop(0)

    return selected[:limit]


async def main() -> int:
    load_env_file()
    if not CHUNKS_PATH.is_file():
        print(f"missing chunks index: {CHUNKS_PATH}", file=sys.stderr)
        return 1

    print("Loading chunk index…", file=sys.stderr)
    _raw, by_src = load_index()
    print(f"  {len(by_src)} distinct source_path values", file=sys.stderr)

    print("Building candidates…", file=sys.stderr)
    cands = candidate_docs(by_src)
    print(f"  {len(cands)} candidates (exists under SOURCE_ROOT, prefix, size)", file=sys.stderr)
    if len(cands) < 20:
        print("too few candidates", file=sys.stderr)
        return 1

    print("Computing near-duplicate groups…", file=sys.stderr)
    dup_map = build_dup_map(cands)
    n_dup = sum(1 for c in cands if dup_map.get(c["source_path"]))
    print(f"  {n_dup} docs have at least one dup peer", file=sys.stderr)

    sampled = stratified_sample(cands, dup_map, TARGET_DOCS)
    print(f"Sampled {len(sampled)} docs", file=sys.stderr)
    feat_counts = Counter()
    area_counts = Counter()
    for c in sampled:
        area_counts[c["area"]] += 1
        for k, v in c["features"].items():
            if v:
                feat_counts[k] += 1
        if dup_map.get(c["source_path"]):
            feat_counts["dup"] += 1
    print(f"  features: {dict(feat_counts)}", file=sys.stderr)
    print(f"  areas: {dict(area_counts)}", file=sys.stderr)

    client = make_client()
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    seen_facts: set[str] = set()

    tasks = [gen_for_doc(client, sem, c, dup_map, seen_facts, lock) for c in sampled]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_cases: list[dict[str, Any]] = []
    for sp, res in zip((c["source_path"] for c in sampled), results):
        if isinstance(res, BudgetExceeded):
            print(f"BUDGET: {res}", file=sys.stderr)
            break
        if isinstance(res, Exception):
            print(f"  {sp}: ERROR {res!r}", file=sys.stderr)
            continue
        all_cases.extend(res)

    print(f"Raw verified cases: {len(all_cases)}", file=sys.stderr)
    balanced = balance_cases(all_cases, TARGET_CASES)
    # assign ids
    for i, c in enumerate(balanced, start=1):
        c["id"] = f"q{i:03d}"

    # final field order
    ordered = []
    for c in balanced:
        ordered.append(
            {
                "id": c["id"],
                "question": c["question"],
                "gold_file": c["gold_file"],
                "gold_fact": c["gold_fact"],
                "gold_chunk_ids": c["gold_chunk_ids"],
                "qtype": c["qtype"],
                "lang": c["lang"],
                "detected_kind": c["detected_kind"],
                "heading_path": c["heading_path"],
                "dup_group": c["dup_group"],
            }
        )

    header = (
        "# Auto-generated by experiments/search-2026-10/gen_cases.py\n"
        "# Model: gpt-6-luna temp=1.0; gold_fact verified as file substring AND whitespace-normalised chunk substring.\n"
        f"# Source index: {CHUNKS_PATH}\n"
        f"# SOURCE_ROOT: {SOURCE_ROOT}\n"
    )
    body = yaml.safe_dump({"cases": ordered}, allow_unicode=True, width=120, sort_keys=False)
    OUT_PATH.write_text(header + body, encoding="utf-8")

    # stats
    qt = Counter(c["qtype"] for c in ordered)
    lang = Counter(c["lang"] for c in ordered)
    kind = Counter(c["detected_kind"] for c in ordered)
    n_with_dup = sum(1 for c in ordered if c.get("dup_group"))
    usd = cost_usd()
    print("=== gen_cases summary ===")
    print(f"total: {len(ordered)}")
    print(f"by qtype: {dict(qt)}")
    print(f"by lang: {dict(lang)}")
    print(f"by detected_kind: {dict(kind)}")
    print(f"with dup_group: {n_with_dup}")
    print(f"usd spent: ${usd:.4f}")
    print(f"wrote: {OUT_PATH}")

    if len(ordered) < 100:
        print(f"WARNING: only {len(ordered)} cases (<100)", file=sys.stderr)
        return 2
    if usd >= 0.5:
        print(f"WARNING: cost ${usd:.4f} >= 0.5", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
