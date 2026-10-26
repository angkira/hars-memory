"""Repeatable LLM extractor benchmark (methodology ported from /tmp/bench, Oct 2026).

Compares candidate extractor LLMs (GGUFs served on llama-server) on fixed
documents using the REAL extraction prompts: ``lightrag.prompt.PROMPTS``
formatted exactly as ``lightrag/operate.py::_process_single_content`` does,
plus the schema guidance append from ``server/lightrag_init.py``
(``PROMPTS["entity_extraction_system_prompt"] += schema.guidance``).
No paraphrase, no hand-copied prompts.

Key finding encoded here: thinking MUST be off via
``chat_template_kwargs={"enable_thinking": False}``. A top-level
``enable_thinking`` flag is silently ignored by llama-server; without the
nested knob every thinking-capable model burns its token budget on thought
traces and emits no payload. ``run`` therefore asserts the knob took
effect: any response with non-empty ``reasoning_content`` fails the run
unless ``--allow-thinking-leak`` is passed.

Subcommands:
    run    --model <gguf-path-or-alias> --docs <dir> --out <dir>
           [--base-url URL] [--allow-thinking-leak]
    score  --run-dir <dir> --reference <json>
    report --runs <dir> [<dir> ...] [--out <path>]

Fixtures (versioned, content-copied, never symlinked):
    hars_memory/eval/fixtures/extractor_bench_v1/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

FIXTURE_VERSION = "extractor_bench_v1"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / FIXTURE_VERSION
DEFAULT_REFERENCE = FIXTURE_DIR / "reference_entities.json"

TUPLE_DELIMITER = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"
MAX_TOKENS = 4096

# LightRAG 1.5.x removed ``lightrag.constants.DEFAULT_ENTITY_TYPES`` (the
# 1.4.x list of generic entity types).  1.5 renders the vocabulary as a prose
# ``entity_types_guidance`` block in the extraction prompt profile listing the
# same 11 types; this list preserves the 1.4.16 value so the bench's ``meta``
# block stays byte-comparable across the upgrade (see ``build_prompts``).
DEFAULT_ENTITY_TYPES: list[str] = [
    "Person",
    "Creature",
    "Organization",
    "Location",
    "Event",
    "Concept",
    "Method",
    "Content",
    "Data",
    "Artifact",
    "NaturalObject",
]

# Default model aliases: exact GGUF paths from the original bench report.
MODEL_ALIASES: dict[str, str] = {
    "gemma-12b": (
        "/data/hf_cache/hub/models--unsloth--gemma-4-12b-it-GGUF/"
        "snapshots/3249fa54d5efa384afc552cc6700ad091efd5c39/"
        "gemma-4-12b-it-Q4_K_M.gguf"
    ),
    "qwen3.5-9b": "/home/demo/models/qwen3.5-9b-gguf/Qwen3.5-9B-Q4_K_M.gguf",
    "qwen3.8-27b": "/home/demo/models/qwen3.8-27b-gguf/Qwen3.8-27B-Q4_K_M.gguf",
    "muse-glimmer-30b": (
        "/home/demo/models/muse-glimmer-30b-gguf/"
        "muse-glimmer-30B-kquant-dynamic.gguf"
    ),
}

SCHEMA_ENV_VAR = "HARS_MEMORY_ENTITY_SCHEMA_PATH"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class ExtractorBenchError(RuntimeError):
    """The bench could not produce trustworthy comparable results."""


# ---------------------------------------------------------------------------
# Prompt construction (mirrors /tmp/bench/build_prompts.py exactly)
# ---------------------------------------------------------------------------


def resolve_schema_path(explicit: Path | None = None) -> Path:
    """Packaged default schema unless overridden (arg wins, then env var).

    The original bench used a consumer-specific schema file; this tool uses
    the generic packaged schema so it stays consumer-agnostic. Any consumer
    needing the original vocabulary passes its own schema path.
    """
    if explicit is not None:
        return explicit
    env = os.environ.get(SCHEMA_ENV_VAR, "").strip()
    if env:
        return Path(env).expanduser()
    from hars_memory.schema.loader import DEFAULT_SCHEMA_PATH

    return DEFAULT_SCHEMA_PATH


def build_prompts(
    docs: dict[str, str], schema_path: Path | None = None
) -> dict[str, Any]:
    """Build extraction prompts via the REAL PROMPTS dict + REAL schema loader.

    Mirrors ``lightrag.operate._process_single_content`` for lightrag-hku 1.5.7:
    the entity vocabulary is the resolved prompt profile's prose
    ``entity_types_guidance`` (1.5 removed ``DEFAULT_ENTITY_TYPES`` and the
    ``{entity_types}`` template slot), and the system/user prompts also carry
    the per-response row caps. The bench extracts whole documents, so the
    optional per-chunk section-context block stays empty — the documented
    no-context user-prompt path.
    """
    from lightrag.constants import (
        DEFAULT_MAX_EXTRACTION_ENTITIES,
        DEFAULT_MAX_EXTRACTION_RECORDS,
        DEFAULT_SUMMARY_LANGUAGE,
    )
    from lightrag.prompt import PROMPTS, get_default_entity_extraction_prompt_profile

    from hars_memory.schema.loader import load_schema

    schema = load_schema(resolve_schema_path(schema_path))
    system_tmpl = PROMPTS["entity_extraction_system_prompt"]
    if schema.guidance and schema.guidance not in system_tmpl:
        system_tmpl = system_tmpl + schema.guidance

    entity_types = list(DEFAULT_ENTITY_TYPES)
    language = DEFAULT_SUMMARY_LANGUAGE
    profile = get_default_entity_extraction_prompt_profile()
    entity_types_guidance = profile["entity_types_guidance"]
    examples = "\n".join(profile["entity_extraction_examples"]).format(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types_guidance=entity_types_guidance,
        language=language,
    )
    context_base = {
        "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        "entity_types_guidance": entity_types_guidance,
        "examples": examples,
        "language": language,
        "max_total_records": DEFAULT_MAX_EXTRACTION_RECORDS,
        "max_entity_records": DEFAULT_MAX_EXTRACTION_ENTITIES,
    }
    user_tmpl = PROMPTS["entity_extraction_user_prompt"]
    return {
        "meta": {
            "entity_types": entity_types,
            "language": language,
            "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            "guidance_appended": bool(schema.guidance),
            "schema_path": str(resolve_schema_path(schema_path)),
            "source": "real PROMPTS import + real schema loader (no paraphrase)",
        },
        "system": system_tmpl.format(**context_base),
        "users": {
            name: user_tmpl.format(
                **{**context_base, "input_text": text, "heading_context_block": ""}
            )
            for name, text in docs.items()
        },
    }


def load_docs(docs_dir: Path) -> dict[str, str]:
    """Read ``*.md`` docs sorted by filename stem (stable cross-run order)."""
    docs = {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(docs_dir.glob("*.md"))
    }
    if not docs:
        raise ExtractorBenchError(f"no *.md docs in {docs_dir}")
    return docs


# ---------------------------------------------------------------------------
# Output parsing + scoring (mirrors /tmp/bench/score1.py + score2.py)
# ---------------------------------------------------------------------------


def parse_output(text: str) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str], bool]:
    """Parse LightRAG tuple-format output.

    Returns (entities, relations, malformed_lines, has_complete). A line is
    malformed unless it is exactly ``entity<SEP>x4`` or ``relation<SEP>x5``
    (tag match is case-insensitive, backtick-tolerant, as in score1.py).
    """
    entities: list[dict[str, str]] = []
    relations: list[dict[str, str]] = []
    malformed: list[str] = []
    has_complete = COMPLETION_DELIMITER in text
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == COMPLETION_DELIMITER:
            continue
        parts = stripped.split(TUPLE_DELIMITER)
        tag = parts[0].strip().lower().strip("`")
        if tag == "entity" and len(parts) == 4:
            entities.append(
                {"name": parts[1].strip(), "type": parts[2].strip(), "desc": parts[3].strip()}
            )
        elif tag == "relation" and len(parts) == 5:
            relations.append(
                {
                    "src": parts[1].strip(),
                    "dst": parts[2].strip(),
                    "kw": parts[3].strip(),
                    "desc": parts[4].strip(),
                }
            )
        else:
            malformed.append(stripped[:120])
    return entities, relations, malformed, has_complete


def normalize(name: str) -> str:
    """Lowercase alphanumeric-and-spaces fold (score2.py parity)."""
    return "".join(c if (c.isalnum() or c == " ") else " " for c in name.lower())


def _matches(entity_name: str, aliases: list[str]) -> bool:
    return any(a in entity_name or entity_name in a for a in aliases)


def load_reference(path: Path) -> dict[str, list[list[str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def score_doc(
    text: str, reference: list[list[str]], completion_tokens: int
) -> dict[str, Any]:
    """Format stats + reference-based entity P/R + dangling-relation count."""
    entities, relations, malformed, has_complete = parse_output(text)
    names = {e["name"].lower() for e in entities}
    dangling = sum(
        1 for r in relations if r["src"].lower() not in names or r["dst"].lower() not in names
    )
    normed = [normalize(e["name"]) for e in entities]
    matched_ref: set[int] = set()
    matched_ent = 0
    for norm_name in normed:
        hit = False
        for i, aliases in enumerate(reference):
            if _matches(norm_name, aliases):
                matched_ref.add(i)
                hit = True
        if hit:
            matched_ent += 1
    precision = matched_ent / len(entities) if entities else 0.0
    recall = len(matched_ref) / len(reference) if reference else 0.0
    return {
        "entities": len(entities),
        "relations": len(relations),
        "malformed_lines": len(malformed),
        "complete": has_complete,
        "format_ok": has_complete and not malformed,
        "truncated": completion_tokens >= MAX_TOKENS,
        "dangling": dangling,
        "dangling_rate": (dangling / len(relations)) if relations else 0.0,
        "ref_hit": f"{len(matched_ref)}/{len(reference)}",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": matched_ent,
        "fp": len(entities) - matched_ent,
        "fn": len(reference) - len(matched_ref),
    }


def score_run(
    results: dict[str, dict[str, Any]], reference: dict[str, list[list[str]]]
) -> dict[str, Any]:
    """Per-doc scores plus pooled entity P/R and mean tok/s over scored docs."""
    per_doc: dict[str, dict[str, Any]] = {}
    tp = fp = fn = dangling = relations = 0
    tps_values: list[float] = []
    format_ok_count = 0
    scored = 0
    for doc, record in results.items():
        if "error" in record:
            per_doc[doc] = {"error": record["error"]}
            continue
        if doc not in reference:
            per_doc[doc] = {"error": f"no reference list for {doc!r}"}
            continue
        scored += 1
        doc_score = score_doc(record["text"], reference[doc], record.get("completion_tokens", 0))
        doc_score["seconds"] = record.get("seconds", 0)
        doc_score["tok_per_s"] = record.get("tok_per_s", 0)
        doc_score["thinking_leak"] = bool(record.get("reasoning_chars", 0))
        per_doc[doc] = doc_score
        tp += doc_score["tp"]
        fp += doc_score["fp"]
        fn += doc_score["fn"]
        dangling += doc_score["dangling"]
        relations += doc_score["relations"]
        tps_values.append(float(record.get("tok_per_s", 0)))
        format_ok_count += 1 if doc_score["format_ok"] else 0
    pooled_p = tp / (tp + fp) if (tp + fp) else 0.0
    pooled_r = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "per_doc": per_doc,
        "pooled": {
            "precision": round(pooled_p, 4),
            "recall": round(pooled_r, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "format_ok_rate": round(format_ok_count / scored, 4) if scored else 0.0,
        "dangling_total": dangling,
        "dangling_rate": round(dangling / relations, 4) if relations else 0.0,
        "mean_tok_per_s": round(sum(tps_values) / len(tps_values), 2) if tps_values else 0.0,
        "docs_scored": scored,
    }


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible /v1, stdlib only)
# ---------------------------------------------------------------------------


def chat_once(
    base_url: str, system: str, user: str, timeout: float = 600.0
) -> dict[str, Any]:
    """One extraction call. Returns text + token/timing stats.

    Raises ExtractorBenchError if ``reasoning_content`` is non-empty: the
    ``chat_template_kwargs.enable_thinking=false`` knob (nested — a top-level
    flag is silently ignored by llama-server) must take effect, otherwise
    the model burned budget on thought traces and the comparison is invalid.
    """
    body = json.dumps(
        {
            "model": "bench",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False, "reasoning_strength": "low"},
        }
    ).encode()
    started = time.time()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except Exception as exc:
        raise ExtractorBenchError(f"chat completion failed: {exc}") from exc
    elapsed = time.time() - started
    message = payload["choices"][0]["message"]
    # Some servers nest quirks: accept both reasoning_content and reasoning.
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if isinstance(reasoning, list):
        reasoning = "".join(str(p) for p in reasoning)
    reasoning = str(reasoning)
    content = str(message.get("content") or "")
    used_fallback = False
    if not content.strip() and reasoning.strip():
        # Pipeline parity (server/lightrag_init.py): payload leaked into
        # reasoning_content is the answer, not a discardable trace.
        content = reasoning
        used_fallback = True
    # Inline <think> blocks are stripped the same way the pipeline does.
    content = _THINK_RE.sub("", content)
    usage = payload.get("usage", {}) or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    return {
        "text": content,
        "seconds": round(elapsed, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tok_per_s": round(completion_tokens / elapsed, 2) if elapsed > 0 else 0,
        "reasoning_fallback": used_fallback,
        "reasoning_chars": len(reasoning),
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(base_url: str, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=5):
                return
        except Exception:
            time.sleep(2.0)
    raise ExtractorBenchError(f"server at {base_url} not ready within {timeout}s")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def resolve_model(value: str) -> str:
    """Alias → GGUF path; anything else must be an existing .gguf file."""
    if value in MODEL_ALIASES:
        return MODEL_ALIASES[value]
    path = Path(value).expanduser()
    if path.suffix == ".gguf" and path.is_file():
        return str(path)
    raise ExtractorBenchError(
        f"unknown model {value!r}: use an alias ({', '.join(sorted(MODEL_ALIASES))}) "
        "or a path to an existing .gguf file"
    )


def cmd_run(args: argparse.Namespace) -> int:
    docs = load_docs(args.docs)
    prompts = build_prompts(docs, schema_path=args.schema)
    model_path = resolve_model(args.model)
    label = next((a for a, p in MODEL_ALIASES.items() if p == model_path), Path(model_path).stem)

    server: subprocess.Popen | None = None
    base_url = args.base_url
    args.out.mkdir(parents=True, exist_ok=True)
    if base_url is None:
        binary = shutil.which("llama-server")
        if binary is None:
            print("error: --base-url not given and llama-server not on PATH", file=sys.stderr)
            return 1
        port = args.port or _free_port()
        base_url = f"http://127.0.0.1:{port}/v1"
        print(f"serving {model_path} on {base_url} ...", flush=True)
        server = subprocess.Popen(
            [binary, "-m", model_path, "--port", str(port), "--ctx-size", str(args.ctx_size)],
            stdout=open(args.out / f"llama_server_{label}.log", "w"),  # noqa: PTH123
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_ready(base_url, timeout=args.serve_timeout)
        except ExtractorBenchError as exc:
            server.terminate()
            print(f"error: {exc}", file=sys.stderr)
            return 1

    try:
        results: dict[str, dict[str, Any]] = {}
        for name, user in prompts["users"].items():
            print(f"[{label}] {name} ...", flush=True)
            try:
                record = chat_once(base_url, prompts["system"], user, timeout=args.timeout)
            except ExtractorBenchError as exc:
                results[name] = {"error": str(exc)}
                print(f"  ERROR {exc}", flush=True)
                continue
            results[name] = record
            print(
                f"  {record['seconds']}s "
                f"tok={record['prompt_tokens']}/{record['completion_tokens']} "
                f"{record['tok_per_s']}/s fallback={record['reasoning_fallback']} "
                f"out_chars={len(record['text'])}",
                flush=True,
            )
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=60)

    leaks = sorted(n for n, r in results.items() if r.get("reasoning_chars", 0))
    if leaks and not args.allow_thinking_leak:
        print(
            f"error: thinking knob did not take effect (non-empty reasoning_content "
            f"for {', '.join(leaks)}); re-run with --allow-thinking-leak to keep "
            "these outputs anyway",
            file=sys.stderr,
        )
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"raw_{label}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    meta = {
        "model_label": label,
        "model_path": model_path,
        "base_url": base_url,
        "fixture": FIXTURE_VERSION,
        "docs": sorted(docs),
        "prompts_meta": prompts["meta"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thinking_leak_docs": leaks,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"saved {args.out}/raw_{label}.json")
    return 0


def format_score_table(model: str, report: dict[str, Any]) -> str:
    lines = [
        f"model: {model}",
        f"{'doc':8s} {'ent':>4s} {'rel':>4s} {'mal':>4s} "
        f"{'complete':>8s} {'dangling':>8s} {'P':>6s} {'R':>6s} {'tok/s':>7s}",
    ]
    for doc, score in report["per_doc"].items():
        if "error" in score:
            lines.append(f"{doc:8s} ERROR {score['error']}")
            continue
        lines.append(
            f"{doc:8s} {score['entities']:4d} {score['relations']:4d} "
            f"{score['malformed_lines']:4d} {str(score['complete']):>8s} "
            f"{score['dangling']:8d} {score['precision']:6.2f} {score['recall']:6.2f} "
            f"{score['tok_per_s']:7.2f}"
        )
    pooled = report["pooled"]
    lines.append(
        f"POOLED P={pooled['precision']:.2f} R={pooled['recall']:.2f} "
        f"(tp={pooled['tp']} fp={pooled['fp']} fn={pooled['fn']}) "
        f"format_ok={report['format_ok_rate']:.0%} "
        f"dangling={report['dangling_rate']:.1%} "
        f"mean_tok/s={report['mean_tok_per_s']}"
    )
    return "\n".join(lines)


def _load_run(run_dir: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    raw_files = sorted(run_dir.glob("raw_*.json"))
    if not raw_files:
        raise ExtractorBenchError(f"no raw_*.json in {run_dir}")
    if len(raw_files) > 1:
        raise ExtractorBenchError(f"multiple raw_*.json in {run_dir}: {raw_files}")
    label = raw_files[0].stem.removeprefix("raw_")
    return label, json.loads(raw_files[0].read_text(encoding="utf-8"))


def cmd_score(args: argparse.Namespace) -> int:
    try:
        label, results = _load_run(args.run_dir)
        reference = load_reference(args.reference)
        report = score_run(results, reference)
    except (ExtractorBenchError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"error: cannot score {args.run_dir}: {exc}", file=sys.stderr)
        return 1
    report["model"] = label
    report["run_dir"] = str(args.run_dir)
    out_path = args.run_dir / "scores.json"
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(format_score_table(label, report))
    print(f"wrote {out_path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    try:
        reference = load_reference(args.reference) if args.reference else None
        rows: list[dict[str, Any]] = []
        for run_dir in args.runs:
            label, results = _load_run(run_dir)
            scores_path = run_dir / "scores.json"
            if scores_path.is_file() and reference is None:
                report = json.loads(scores_path.read_text(encoding="utf-8"))
            else:
                report = score_run(results, reference or load_reference(DEFAULT_REFERENCE))
            pooled = report["pooled"]
            rows.append(
                {
                    "model": label,
                    "P": pooled["precision"],
                    "R": pooled["recall"],
                    "format_ok": report["format_ok_rate"],
                    "dangling": report["dangling_rate"],
                    "tok/s": report["mean_tok_per_s"],
                }
            )
    except (ExtractorBenchError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"error: cannot build report: {exc}", file=sys.stderr)
        return 1
    header = f"{'model':20s} {'P':>6s} {'R':>6s} {'format_ok':>9s} {'dangling':>8s} {'tok/s':>7s}"
    lines = [header]
    for row in sorted(rows, key=lambda r: (r["R"], r["P"]), reverse=True):
        lines.append(
            f"{row['model']:20s} {row['P']:6.2f} {row['R']:6.2f} "
            f"{row['format_ok']:9.0%} {row['dangling']:8.1%} {row['tok/s']:7.2f}"
        )
    table = "\n".join(lines)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory-extractor-bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run extraction on every doc via an OpenAI-compatible endpoint.")
    run_p.add_argument("--model", required=True, help=f"GGUF path or alias ({', '.join(sorted(MODEL_ALIASES))})")
    run_p.add_argument("--docs", type=Path, default=FIXTURE_DIR, help="Doc dir (default: versioned fixtures)")
    run_p.add_argument("--out", type=Path, required=True, help="Output run dir")
    run_p.add_argument("--base-url", default=None, help="Existing endpoint base URL (default: serve --model)")
    run_p.add_argument("--port", type=int, default=0, help="Port for spawned llama-server (0 = free port)")
    run_p.add_argument("--ctx-size", type=int, default=16384)
    run_p.add_argument("--timeout", type=float, default=600.0, help="Per-doc request timeout (s)")
    run_p.add_argument("--serve-timeout", type=float, default=300.0, help="Server readiness wait (s)")
    run_p.add_argument("--schema", type=Path, default=None, help="Schema YAML (default: packaged schema)")
    run_p.add_argument(
        "--allow-thinking-leak",
        action="store_true",
        help="Keep outputs even when reasoning_content is non-empty (default: exit 2)",
    )
    run_p.set_defaults(func=cmd_run)

    score_p = sub.add_parser("score", help="Score one run dir against reference entity lists.")
    score_p.add_argument("--run-dir", type=Path, required=True)
    score_p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    score_p.set_defaults(func=cmd_score)

    report_p = sub.add_parser("report", help="Comparison table across multiple run dirs.")
    report_p.add_argument("--runs", nargs="+", type=Path, required=True)
    report_p.add_argument("--reference", type=Path, default=None)
    report_p.add_argument("--out", type=Path, default=None)
    report_p.set_defaults(func=cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()


__all__ = [
    "COMPLETION_DELIMITER",
    "DEFAULT_REFERENCE",
    "FIXTURE_DIR",
    "FIXTURE_VERSION",
    "MAX_TOKENS",
    "MODEL_ALIASES",
    "TUPLE_DELIMITER",
    "ExtractorBenchError",
    "build_prompts",
    "chat_once",
    "load_docs",
    "load_reference",
    "normalize",
    "parse_output",
    "resolve_model",
    "resolve_schema_path",
    "score_doc",
    "score_run",
]
