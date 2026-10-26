#!/usr/bin/env python3
"""3-way chunking eval on the 10 smoke documents (token vs markdown vs markdown+merge).

Sub-commands (all CPU retrieval; the only LLM use is case generation + answerability judge,
through the OpenAI-compatible proxy in config/live-llm-e2e.env, key never printed):

  gen-cases  read each source doc, ask a cheap model for ~6 questions per doc, validate that
             every gold_fact is a verbatim substring, derive heading path/kind from the doc
             itself, write config/eval/smoke10-cases.yaml
  run        one arm: open an index CLONE (never the original), retrieve per case/mode/top-k,
             compute the metrics, judge answerability of the top-3 chunks, dump JSON
  report     combine per-arm JSONs into tables per arm / per kind with bootstrap CIs

The retrieval path mirrors production (hars_memory.eval.channels): dense_only, hybrid_bm25
(dense + BM25 fusion, alpha 0.5), naive (LightRAG naive context). The cross-encoder reranker is
off for every arm (production default; the configured model is not cached offline).

Run each arm on an `cp -cR` clone, e.g. (see .session/2026-10-01_smoke3-chunking-eval.md):
  HARS_MEMORY_INDEX_DIR=/tmp/eval-smoke/token uv run python scripts/smoke_eval.py run --arm token
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "config/eval/smoke10-cases.yaml"
OUT_DIR = Path("/tmp/eval-smoke")
COST_FILE = OUT_DIR / "cost.json"
MODEL = "openai/gpt-6-luna"
# USD per 1M tokens, sync price (batch price x2, see AGENTS.md / ab batch defaults 0.05/0.25).
PRICE_IN, PRICE_OUT = 0.10, 0.50
KINDS = ["prose", "table_row", "code_block", "list", "deep_heading", "duplicate_heading", "cross_chunk"]
STRUCTURAL = {"table_row", "code_block", "list", "deep_heading", "duplicate_heading", "cross_chunk"}
SOURCE_ROOT = Path("/Users/iurii.medvedev/Work/sre-support-pier")
DOCS = [
    "repos/sre-docs/docs/kubernetes/object-storage-controller/aws-s3-buckets-operations/configuring-s3-bucket-policy.md",
    "kb/services/backstage/troubleshooting/repo-not-in-catalog.md",
    "repos/sre-docs/docs/platform/internal/to-delete/migration-services-from-one-cluster-to-another.md",
    "kb/k8s/object-storage-controller/troubleshooting/s3-china-endpoint.md",
    "kb/k8s/object-storage-controller/troubleshooting/irsa-role-cannot-assume-other-roles.md",
    "kb/aws/accounts/account-id-map.md",
    "repos/sre-docs/docs/guides/sre-team-collaboration-and-request-guidelines.md",
    "kb/tools/github/permission-model.md",
    "kb/readme.md",
    "repos/sre-docs/README.md",
]

# --------------------------------------------------------------------------- helpers


def ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm_head(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def load_env_file() -> None:
    p = ROOT / "config/live-llm-e2e.env"
    for line in p.read_text("utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cost_add(tag: str, tin: int, tout: int) -> float:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(COST_FILE.read_text()) if COST_FILE.exists() else {"calls": 0, "in": 0, "out": 0, "by_tag": {}}
    data["calls"] += 1
    data["in"] += tin
    data["out"] += tout
    t = data["by_tag"].setdefault(tag, {"calls": 0, "in": 0, "out": 0})
    t["calls"] += 1
    t["in"] += tin
    t["out"] += tout
    data["usd"] = round(data["in"] / 1e6 * PRICE_IN + data["out"] / 1e6 * PRICE_OUT, 4)
    COST_FILE.write_text(json.dumps(data, indent=1))
    return data["usd"]


async def llm(client: Any, prompt: str, tag: str, max_out: int = 8000) -> str:
    r = await client.chat.completions.create(
        model=MODEL.split("/", 1)[1],
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_completion_tokens=max_out,
    )
    u = r.usage
    usd = cost_add(tag, u.prompt_tokens, u.completion_tokens)
    print(f"  [cost] {tag}: +{u.prompt_tokens}/{u.completion_tokens} tok, running eval tally ${usd:.4f}", file=sys.stderr)
    return r.choices[0].message.content or ""


def make_client() -> Any:
    from openai import AsyncOpenAI

    load_env_file()
    return AsyncOpenAI(
        base_url=os.environ["HARS_MEMORY_EXTRACTOR_BASE_URL"],
        api_key=os.environ["HARS_MEMORY_LLM_API_KEY"],
        timeout=300,
    )


def parse_json(text: str) -> Any:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    i = min([x for x in (text.find("["), text.find("{")) if x >= 0], default=0)
    return json.loads(text[i:])


# --------------------------------------------------------------------------- doc structure


class Doc:
    """Source markdown with headings (fence-aware), fences, tables, lists."""

    def __init__(self, text: str):
        self.text = text
        self.lines = text.split("\n")
        self.offsets = []
        o = 0
        for ln in self.lines:
            self.offsets.append(o)
            o += len(ln) + 1
        self.fence_line: list[bool] = []
        self.headings: list[tuple[int, int, str]] = []  # (line, level, text)
        inside = False
        for i, ln in enumerate(self.lines):
            if ln.lstrip().startswith("```") or ln.lstrip().startswith("~~~"):
                self.fence_line.append(True)
                inside = not inside
                continue
            self.fence_line.append(inside)
            m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", ln)
            if m and not inside:
                self.headings.append((i, len(m.group(1)), m.group(2)))
        counts: dict[str, int] = {}
        for _, _, t in self.headings:
            counts[norm_head(t)] = counts.get(norm_head(t), 0) + 1
        self.dup = {k for k, v in counts.items() if v > 1}

    def line_of(self, off: int) -> int:
        lo = 0
        for i, o in enumerate(self.offsets):
            if o <= off:
                lo = i
            else:
                break
        return lo

    def path_at(self, line: int) -> list[str]:
        stack: list[tuple[int, str]] = []
        for hl, lvl, t in self.headings:
            if hl > line:
                break
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, t))
        return [t for _, t in stack]

    def level_at(self, line: int) -> int:
        lvl = 0
        for hl, level, _ in self.headings:
            if hl > line:
                break
            lvl = level
        return lvl

    def detect(self, start: int, end: int) -> set[str]:
        sl, el = self.line_of(start), self.line_of(max(start, end - 1))
        span = self.lines[sl : el + 1]
        kinds: set[str] = set()
        if any(self.fence_line[sl : el + 1]):
            kinds.add("code_block")
        body = [x for x in span if x.strip()]
        if body and all(x.lstrip().startswith("|") for x in body):
            kinds.add("table_row")
        if body and re.match(r"^\s*([-*+]|\d+[.)])\s+", body[0]):
            kinds.add("list")
        path_line = sl
        if self.level_at(path_line) >= 4:
            kinds.add("deep_heading")
        p = self.path_at(path_line)
        if p and norm_head(p[-1]) in self.dup:
            kinds.add("duplicate_heading")
        if end - start >= 500 and any(not x.strip() for x in span):
            kinds.add("cross_chunk")
        if not kinds:
            kinds.add("prose")
        return kinds


def flat_name(rel: str) -> str:
    return Path(rel).name


# --------------------------------------------------------------------------- gen-cases

GEN_PROMPT = """You write retrieval-evaluation questions for a documentation search system.

Below is ONE markdown document ({name}). Write {n} question/answer-evidence items about it.

Hard rules:
- gold_fact must be copied CHARACTER FOR CHARACTER from the document (a contiguous substring, keep markdown such as pipes/backticks/bullets and newlines exactly). \
Short: one sentence, one table row (the full "| a | b |" line), one command / one or two code lines, one list item. Never paraphrase it.
- question is what a human engineer would type into a search box to find that fact. Self-contained, do NOT quote the gold_fact, \
do NOT mention the file name. Same language as the document passage. It must be answerable by the gold_fact alone.
- Each item has a different gold_fact.
- kind is one of: prose, table_row, code_block, list, deep_heading, duplicate_heading, cross_chunk.
  table_row = fact is a table row; code_block = fact is inside a ``` fence; list = a bullet/numbered item;
  deep_heading = fact sits under a heading of level H4 or deeper; duplicate_heading = fact sits under a heading whose text also appears elsewhere in this doc;
  cross_chunk = a LONG contiguous passage of 600-1200 characters (several paragraphs/steps incl. blank lines) that the question needs as a whole (question about the whole procedure/explanation); prose = plain paragraph.
Structure present in this document (use it, cover as many of these as available):
{inventory}
Distribution target: at least {n_struct} of the {n} items must be structural kinds that actually exist here (not prose); the rest prose.

Return ONLY a JSON array: [{{"question": "...", "gold_fact": "...", "kind": "..."}}, ...]

DOCUMENT:
<<<
{doc}
>>>"""


def inventory(d: Doc) -> str:
    lines = []
    tbl = sum(1 for x in d.lines if x.lstrip().startswith("|"))
    fen = sum(1 for i, x in enumerate(d.lines) if d.fence_line[i] and not x.lstrip().startswith("```"))
    lis = sum(1 for x in d.lines if re.match(r"^\s*([-*+]|\d+[.)])\s+", x))
    lines.append(f"- table rows: {tbl}; code-fence lines: {fen}; list items: {lis}")
    deep = [t for _, lv, t in d.headings if lv >= 4]
    lines.append(f"- headings H4+: {deep or 'none'}")
    dup = sorted({t for _, _, t in d.headings if norm_head(t) in d.dup})
    lines.append(f"- headings whose text is duplicated in the doc: {dup or 'none'}")
    lines.append("- all headings: " + "; ".join(f"H{lv} {t}" for _, lv, t in d.headings))
    return "\n".join(lines)


async def gen_cases(args: argparse.Namespace) -> int:
    client = make_client()
    out: list[dict[str, Any]] = []
    rnd = random.Random(7)
    for di, rel in enumerate(DOCS, start=1):
        src = SOURCE_ROOT / rel
        text = src.read_text("utf-8")
        d = Doc(text)
        name = flat_name(rel)
        kept: list[dict[str, Any]] = []
        seen: set[str] = set()
        for attempt in range(3):
            want = 8 if attempt == 0 else 5
            prompt = GEN_PROMPT.format(
                name=name, n=want, n_struct=max(3, want // 2), inventory=inventory(d), doc=text
            )
            try:
                items = parse_json(await llm(client, prompt, f"gen:{name}"))
            except Exception as exc:  # noqa: BLE001
                print(f"  {name}: bad JSON ({exc}), retry", file=sys.stderr)
                continue
            bad = 0
            for it in items:
                fact = str(it.get("gold_fact", ""))
                q = str(it.get("question", "")).strip()
                off = text.find(fact) if fact.strip() else -1
                if off < 0 or not q or fact in seen or len(fact) < 12:
                    bad += 1
                    continue
                seen.add(fact)
                detected = d.detect(off, off + len(fact))
                claimed = str(it.get("kind", "prose"))
                prio = ["table_row", "code_block", "list", "duplicate_heading", "deep_heading", "cross_chunk", "prose"]
                kind = claimed if claimed in detected else next(k for k in prio if k in detected)
                line = d.line_of(off)
                kept.append(
                    {
                        "question": q,
                        "gold_file": name,
                        "gold_fact": fact,
                        "gold_heading_path": d.path_at(line),
                        "kind": kind,
                        "claimed_kind": claimed,
                        "start_line": line + 1,
                    }
                )
            print(f"  {name}: attempt {attempt + 1}: {len(items)} items, {bad} rejected, {len(kept)} kept", file=sys.stderr)
            if len(kept) >= 6:
                break
        # trim to 7, keep structural ones first
        rnd.shuffle(kept)
        kept.sort(key=lambda c: c["kind"] == "prose")
        kept = kept[:7]
        for c in kept:
            c["_doc"] = di
        out.extend(kept)
    for i, c in enumerate(out, start=1):
        c["id"] = f"c{i:02d}"
        c.pop("_doc", None)
    ordered = [
        {k: c[k] for k in ("id", "question", "gold_file", "gold_fact", "gold_heading_path", "kind", "claimed_kind", "start_line")}
        for c in out
    ]
    header = (
        "# Auto-generated by scripts/smoke_eval.py gen-cases (gpt-6-luna, temp 1.0) over the 10 smoke docs.\n"
        "# gold_fact is verified verbatim against the source; gold_heading_path and kind are derived from the doc itself.\n"
    )
    CASES_PATH.write_text(header + yaml.safe_dump({"cases": ordered}, allow_unicode=True, sort_keys=False, width=120), "utf-8")
    verify_cases()
    return 0


def verify_cases() -> list[dict[str, Any]]:
    cases = yaml.safe_load(CASES_PATH.read_text("utf-8"))["cases"]
    by_name = {flat_name(r): (SOURCE_ROOT / r).read_text("utf-8") for r in DOCS}
    for c in cases:
        if c["gold_fact"] not in by_name[c["gold_file"]]:
            raise SystemExit(f"{c['id']}: gold_fact is NOT a verbatim substring of {c['gold_file']}")
    from collections import Counter

    print(f"{len(cases)} cases, all gold_facts verbatim. kinds: {dict(Counter(c['kind'] for c in cases))}")
    print(f"structural (non-prose): {sum(c['kind'] != 'prose' for c in cases)}")
    return cases


# --------------------------------------------------------------------------- run (one arm)

JUDGE_PROMPT = """You grade a retrieval system. You get a question, the reference fact that answers it, and the \
top retrieved text chunks (the ONLY context the system would hand to an answering model).

Decide whether, using ONLY these chunks, the question can be answered correctly and completely, consistent with the reference fact.
- YES: the chunks contain the information of the reference fact (answer is fully supported).
- PARTIAL: the chunks contain only part of the needed information (e.g. a table row without its header/context, a truncated step list, an ambiguous section).
- NO: the chunks do not contain what is needed.

Question: {q}

Reference fact: {fact}

Chunks:
{chunks}

Reply with exactly one word: YES, PARTIAL or NO."""

CRUMB_RE = re.compile(r"^#{1,6} .+ > #{1,6} ")


def text_heading_status(chunk_text: str, fact_norm: str, gold_path: list[str]) -> str:
    """correct / wrong / none: last heading in the chunk text at or before the fact."""
    pos = ws(chunk_text).find(fact_norm)
    if pos < 0:
        return "n/a"
    last: str | None = None
    inside = False
    consumed = 0
    for ln in chunk_text.split("\n"):
        stripped = ln.lstrip()
        if stripped.startswith("```"):
            inside = not inside
        if not inside and re.match(r"^#{1,6}\s", ln):
            # breadcrumb line "# A > ## B": the leaf is the last segment
            seg = re.split(r"\s>\s", ln)[-1] if CRUMB_RE.match(ln) else ln
            head = re.sub(r"^#{1,6}\s*", "", seg).strip()
            before = len(ws(chunk_text[: consumed]))
            if before <= pos:
                last = head
        consumed += len(ln) + 1
    if last is None:
        return "none"
    return "correct" if norm_head(last) == norm_head(gold_path[-1] if gold_path else "") else "wrong"


def meta_heading_ok(rec: dict[str, Any], gold_line: int, gold_path: list[str]) -> bool | None:
    """Heading metadata of the chunk, member-aware for merged chunks. None = chunk has no metadata."""
    if not rec.get("heading_path"):
        return None
    path = rec["heading_path"]
    for m in rec.get("merged_from") or []:
        if m["start_line"] <= gold_line <= m["end_line"]:
            path = m["heading_path"]
            break
    return [norm_head(x) for x in path] == [norm_head(x) for x in gold_path]


async def retrieve(channel: str, rag: Any, bm25: Any, q: str, k: int) -> list[str]:
    from hars_memory.eval import channels as ch

    if channel == "dense_only":
        return [h.chunk_id for h in await ch.dense_only(rag, q, k) if h.chunk_id]
    if channel == "hybrid_bm25":
        return [h.chunk_id for h in await ch.hybrid_bm25(rag, bm25, q, k, 0.5) if h.chunk_id]
    raise ValueError(channel)


async def naive_chunks(rag: Any, q: str, k: int, by_content: dict[str, str]) -> list[str]:
    from lightrag import QueryParam

    ctx = await rag.aquery(q, param=QueryParam(mode="naive", top_k=k, chunk_top_k=k, only_need_context=True))
    if not isinstance(ctx, str):
        return []
    from hars_memory.eval.channels import _extract_fenced_block

    block = _extract_fenced_block(ctx, "Document Chunks (") or ""
    ids = []
    for line in block.splitlines():
        try:
            c = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        cid = by_content.get(ws(str(c.get("content", ""))))
        if cid:
            ids.append(cid)
    return ids


async def run_arm(args: argparse.Namespace) -> int:
    from hars_memory.retrieval.bm25_index import get_or_build_index
    from hars_memory.server.lightrag_init import create_lightrag

    idx = os.environ["HARS_MEMORY_INDEX_DIR"]
    if not idx.startswith("/tmp/eval-smoke/"):
        raise SystemExit("refusing: HARS_MEMORY_INDEX_DIR must be a clone under /tmp/eval-smoke/")
    os.environ.pop("HARS_MEMORY_RERANK_MODEL", None)  # reranker off for every arm (see module docstring)
    cases = verify_cases()
    rag = create_lightrag()
    await rag.initialize_storages()
    bm25 = (await asyncio.to_thread(get_or_build_index, idx, f"/tmp/eval-smoke/bm25-{args.arm}"))[0]
    raw = json.loads((Path(idx) / "kv_store_text_chunks.json").read_text("utf-8"))
    recs: dict[str, dict[str, Any]] = {k: v for k, v in raw.items()}
    by_content = {ws(v.get("content", "")): k for k, v in recs.items()}
    by_file: dict[str, list[str]] = {}
    for k, v in recs.items():
        by_file.setdefault(v.get("file_path", ""), []).append(k)
    client = None if args.no_judge else make_client()

    rows = []
    for c in cases:
        fact = ws(c["gold_fact"])
        gold_line = c["start_line"]
        gpath = c["gold_heading_path"]
        file_chunks = by_file.get(c["gold_file"], [])
        whole = [cid for cid in file_chunks if fact in ws(recs[cid]["content"])]
        row: dict[str, Any] = {
            "id": c["id"],
            "kind": c["kind"],
            "question": c["question"],
            "gold_file": c["gold_file"],
            "fact_whole_in_one_chunk": bool(whole),
            "n_chunks_file": len(file_chunks),
            "channels": {},
        }
        for channel in ("hybrid_bm25", "dense_only", "naive"):
            for k in (3, 5):
                if channel == "naive":
                    ids = await naive_chunks(rag, c["question"], k, by_content)
                else:
                    ids = await retrieve(channel, rag, bm25, c["question"], k)
                ids = ids[:k]
                texts = [recs[i]["content"] for i in ids if i in recs]
                fact_ids = [i for i in ids if i in recs and fact in ws(recs[i]["content"])]
                res: dict[str, Any] = {
                    "files": [recs[i].get("file_path") for i in ids if i in recs],
                    "file_hit": any(recs[i].get("file_path") == c["gold_file"] for i in ids if i in recs),
                    "fact_in_topk": bool(fact_ids),
                    "chars": sum(len(t) for t in texts),
                    "chunk_ids": ids,
                }
                if fact_ids:
                    top = fact_ids[0]
                    res["fact_rank"] = ids.index(top) + 1
                    res["meta_heading_ok"] = meta_heading_ok(recs[top], gold_line, gpath)
                    res["text_heading"] = text_heading_status(recs[top]["content"], fact, gpath)
                    res["top_chunk_head"] = recs[top].get("heading_path")
                row["channels"][f"{channel}@{k}"] = res
                if channel == "hybrid_bm25" and k == 3 and client is not None:
                    chunks = "\n\n".join(f"[chunk {n}]\n{t}" for n, t in enumerate(texts, 1))
                    ans = await llm(
                        client, JUDGE_PROMPT.format(q=c["question"], fact=c["gold_fact"], chunks=chunks), f"judge:{args.arm}", 2000
                    )
                    m = re.search(r"\b(YES|PARTIAL|NO)\b", ans.upper())
                    res["answerable"] = m.group(1) if m else "NO"
                    res["retrieved_preview"] = [t[:160].replace("\n", " ") for t in texts]
        rows.append(row)
    await rag.finalize_storages()
    sizes = [len(v.get("content", "")) for v in recs.values()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "arm": args.arm,
        "index": idx,
        "n_chunks": len(recs),
        "chunk_chars_mean": round(statistics.mean(sizes), 1),
        "chunk_tokens_mean": round(statistics.mean(v.get("tokens", 0) for v in recs.values()), 1),
        "rows": rows,
    }
    (OUT_DIR / f"results-{args.arm}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {OUT_DIR}/results-{args.arm}.json ({len(rows)} cases)")
    return 0


# --------------------------------------------------------------------------- report


def boot_ci(vals: list[float], n: int = 2000, seed: int = 1) -> tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    means = sorted(statistics.mean(rnd.choices(vals, k=len(vals))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def pct(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    lo, hi = boot_ci(vals)
    return f"{100 * statistics.mean(vals):.0f} [{100 * lo:.0f}-{100 * hi:.0f}]"


def metric_vals(rows: list[dict[str, Any]], key: str, metric: str) -> list[float]:
    out = []
    for r in rows:
        ch = r["channels"][key]
        if metric == "file_hit":
            out.append(float(ch["file_hit"]))
        elif metric == "fact":
            out.append(float(ch["fact_in_topk"]))
        elif metric == "fact1":
            out.append(float(ch.get("fact_rank") == 1))
        elif metric == "mrr":
            out.append(1.0 / ch["fact_rank"] if ch.get("fact_rank") else 0.0)
        elif metric == "orphan":
            if ch["fact_in_topk"] and ch.get("text_heading") in ("correct", "wrong", "none"):
                out.append(float(ch["text_heading"] == "none"))
        elif metric == "ans_yes":
            out.append(float(ch.get("answerable") == "YES"))
        elif metric == "ans_yes_partial":
            out.append(float(ch.get("answerable") in ("YES", "PARTIAL")))
        elif metric == "chars":
            out.append(float(ch["chars"]))
        elif metric == "meta_head":
            if ch["fact_in_topk"] and ch.get("meta_heading_ok") is not None:
                out.append(float(ch["meta_heading_ok"]))
        elif metric == "text_head":
            if ch["fact_in_topk"] and ch.get("text_heading") in ("correct", "wrong", "none"):
                out.append(float(ch["text_heading"] == "correct"))
        elif metric == "text_head_wrong":
            if ch["fact_in_topk"] and ch.get("text_heading") in ("correct", "wrong", "none"):
                out.append(float(ch["text_heading"] == "wrong"))
    return out


def report(args: argparse.Namespace) -> int:
    arms = {a: json.loads((OUT_DIR / f"results-{a}.json").read_text()) for a in args.arms}
    lines: list[str] = []
    n = len(next(iter(arms.values()))["rows"])
    lines.append(f"n = {n} cases per arm. Values are % with 95% bootstrap CI; at this n differences below ~8-10 points are noise.\n")
    for key in ("hybrid_bm25@3", "hybrid_bm25@5", "dense_only@3", "dense_only@5", "naive@3", "naive@5"):
        lines.append(f"### {key}")
        extra = " answerable YES | YES+PARTIAL |" if key == "hybrid_bm25@3" else ""
        lines.append(
            "| arm | chunks (mean tok) | file hit | fact in top-k | fact@1 | MRR | heading metadata ok | "
            "text heading correct | orphaned (no heading in chunk before fact) | wrong heading | chars/query |" + extra
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|" + ("---|---|" if extra else ""))
        for a, d in arms.items():
            rows = d["rows"]
            mv = metric_vals(rows, key, "meta_head")
            meta = "n/a (no native metadata)" if a == "token" else f"{pct(mv)} n={len(mv)}"
            cells = [
                a,
                f"{d['n_chunks']} ({d['chunk_tokens_mean']:.0f})",
                pct(metric_vals(rows, key, "file_hit")),
                pct(metric_vals(rows, key, "fact")),
                pct(metric_vals(rows, key, "fact1")),
                f"{statistics.mean(metric_vals(rows, key, 'mrr')):.2f}",
                meta,
                pct(metric_vals(rows, key, "text_head")),
                pct(metric_vals(rows, key, "orphan")),
                pct(metric_vals(rows, key, "text_head_wrong")),
                f"{statistics.mean(metric_vals(rows, key, 'chars')):.0f}",
            ]
            if extra:
                cells += [pct(metric_vals(rows, key, "ans_yes")), pct(metric_vals(rows, key, "ans_yes_partial"))]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("### chunk integrity: % of gold facts NOT contained whole in any single chunk of their file")
    lines.append("| arm | split | n |\n|---|---|---|")
    for a, d in arms.items():
        v = [float(not r["fact_whole_in_one_chunk"]) for r in d["rows"]]
        lines.append(f"| {a} | {pct(v)} | {len(v)} |")
    lines.append("")
    kinds = sorted({r["kind"] for d in arms.values() for r in d["rows"]}, key=KINDS.index)
    for key, metric, title in (
        ("hybrid_bm25@3", "fact", "fact in top-3"),
        ("hybrid_bm25@3", "ans_yes", "answerable YES (top-3)"),
        ("hybrid_bm25@5", "fact", "fact in top-5"),
    ):
        lines.append(f"### by kind: hybrid, {title}  (cell = % (n))")
        lines.append("| kind | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
        for kd in kinds:
            cells = []
            for d in arms.values():
                rows = [r for r in d["rows"] if r["kind"] == kd]
                v = metric_vals(rows, key, metric)
                cells.append(f"{100 * statistics.mean(v):.0f} ({len(v)})" if v else "n/a")
            lines.append(f"| {kd} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("### by kind: chunk integrity (% split) (n)")
    lines.append("| kind | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
    for kd in kinds:
        cells = []
        for d in arms.values():
            v = [float(not r["fact_whole_in_one_chunk"]) for r in d["rows"] if r["kind"] == kd]
            cells.append(f"{100 * statistics.mean(v):.0f} ({len(v)})")
        lines.append(f"| {kd} | " + " | ".join(cells) + " |")
    lines.append("")
    # paired bootstrap differences on the primary metric
    names = list(arms)
    lines.append("### paired bootstrap differences (hybrid@3 fact-in-topk, and answerable YES), 95% CI of A-B in points")
    for m in ("fact", "ans_yes"):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = metric_vals(arms[names[i]]["rows"], "hybrid_bm25@3", m)
                b = metric_vals(arms[names[j]]["rows"], "hybrid_bm25@3", m)
                diffs = [x - y for x, y in zip(a, b)]
                lo, hi = boot_ci(diffs)
                lines.append(f"- {m}: {names[i]} - {names[j]} = {100 * statistics.mean(diffs):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    lines.append("")
    # failures: rank by severity
    lines.append("### most telling failures per arm (hybrid@3)")
    for a, d in arms.items():
        lines.append(f"**{a}**")
        scored = []
        for r in d["rows"]:
            ch = r["channels"]["hybrid_bm25@3"]
            why = []
            sev = 0
            if not ch["file_hit"]:
                why.append("wrong file")
                sev += 4
            if not ch["fact_in_topk"]:
                why.append("fact not whole in any top-3 chunk" + (" (fact split across chunks)" if not r["fact_whole_in_one_chunk"] else ""))
                sev += 3
            if ch.get("answerable") not in (None, "YES"):
                why.append(f"judge={ch.get('answerable')}")
                sev += 3
            if ch["fact_in_topk"] and ch.get("text_heading") == "none":
                why.append("fact chunk is orphaned from its heading")
                sev += 1
            if ch.get("fact_rank", 1) > 1:
                why.append(f"fact chunk only rank {ch['fact_rank']}")
                sev += 1
            if sev:
                scored.append((sev, r, ch, why))
        scored.sort(key=lambda t: -t[0])
        lines.append(f"({len(scored)} of {len(d['rows'])} cases show any issue)")
        for sev, r, ch, why in scored[:5]:
            lines.append(f"- {r['id']} [{r['kind']}] Q: {r['question'][:120]}")
            lines.append(f"    why: {'; '.join(why)}; retrieved: " + " || ".join(p[:90] for p in ch.get("retrieved_preview", [])))
        lines.append("")
    text = "\n".join(lines)
    (OUT_DIR / "report.md").write_text(text)
    print(text)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen-cases")
    sub.add_parser("verify")
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--no-judge", action="store_true")
    rep = sub.add_parser("report")
    rep.add_argument("--arms", nargs="+", default=["token", "md", "md-merge"])
    args = p.parse_args()
    if args.cmd == "gen-cases":
        return asyncio.run(gen_cases(args))
    if args.cmd == "verify":
        verify_cases()
        return 0
    if args.cmd == "run":
        return asyncio.run(run_arm(args))
    return report(args)


if __name__ == "__main__":
    raise SystemExit(main())
