"""Eval results reader: parse what the eval tooling already wrote, recompute NOTHING.

Understands the NeuroCompact run layouts without importing anything from
``experiments/`` (not part of the wheel):

* ``run_manifest.json`` (run_chainreplay.py) or ``MANIFEST.json`` (run_longsession /
  the CLAUDE.md artifact layout): profile, models, embedder, spend;
* a ``metrics_v2.py --json`` output (``{"meta","summary","deltas","gates","slices",...}``);
* ``summary.json`` of the long-session runner (``by_arm_budget`` with ``by_qtype``);
* ``chain_budget.jsonl`` / ``done.jsonl`` / ``records.jsonl`` for invalid chains + reasons.

Judge text and arm-blinding material are never loaded (REPORT-PROTOCOL blindness rule).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_NAMES = ("run_manifest.json", "MANIFEST.json")
MARKERS = (*MANIFEST_NAMES, "summary.json", "records.jsonl", "chain_budget.jsonl", "done.jsonl", "run_budget.json")
# Flat gate exports prefix their budget files (``gate6-013036_chain_budget.jsonl``); match both spellings.
CHAIN_BUDGET_GLOB = "*chain_budget.jsonl"
RUN_BUDGET_GLOB = "*run_budget.json"
MAX_JSONL_ROWS = 300_000

_SECRET_FLAG = re.compile(r"(key|token|secret|passw|auth)", re.I)


@dataclass
class Table:
    columns: list[str]
    rows: list[list[str]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.rows)


@dataclass
class RunInfo:
    path: Path
    name: str
    markers: list[str]
    metrics_files: list[str]
    mtime: float


@dataclass
class RunDetail:
    path: Path
    manifest: dict[str, Any] | None = None
    manifest_name: str = ""
    metrics: dict[str, Any] | None = None
    metrics_name: str = ""
    summary: dict[str, Any] | None = None
    run_budgets: dict[str, dict[str, Any]] = field(default_factory=dict)
    chain_budget: list[dict[str, Any]] = field(default_factory=list)
    done_rows: list[dict[str, Any]] = field(default_factory=list)
    record_invalid: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# -- generic helpers -----------------------------------------------------------

def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path, limit: int = MAX_JSONL_ROWS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except OSError:
        pass
    return rows


def fmt(value: Any, digits: int = 3) -> str:
    """Display formatting only (never alters the stored number)."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(fmt(v, digits) for v in value)
    return str(value)


def redact_argv(argv: list[Any]) -> list[str]:
    """Mask values of flags that look like secrets (``--api-key X`` / ``--token=X``)."""
    out: list[str] = []
    mask_next = False
    for item in argv:
        text = str(item)
        if mask_next:
            out.append("***")
            mask_next = False
            continue
        if text.startswith("--") and _SECRET_FLAG.search(text.split("=", 1)[0]):
            # `--key-env NAME` names an env var, which is not a secret; mask real values only.
            base = text.split("=", 1)[0]
            if base.endswith("-env") or base.endswith("_env"):
                out.append(text)
            elif "=" in text:
                out.append(base + "=***")
            else:
                out.append(text)
                mask_next = True
            continue
        out.append(text)
    return out


# -- discovery -----------------------------------------------------------------

def _metrics_candidates(path: Path) -> list[str]:
    names = []
    for p in sorted(path.glob("*.json")):
        if p.name in MARKERS:
            continue
        data = _read_json(p)
        if isinstance(data, dict) and "meta" in data and "summary" in data and "gates" in data:
            names.append(p.name)
    return names


def _inspect(path: Path) -> RunInfo | None:
    if not path.is_dir():
        return None
    markers = [m for m in MARKERS if (path / m).exists()]
    for glob, label in ((CHAIN_BUDGET_GLOB, "chain_budget.jsonl"), (RUN_BUDGET_GLOB, "run_budget.json")):
        if label not in markers and any(path.glob(glob)):
            markers.append(label)
    metrics = _metrics_candidates(path)
    if not markers and not metrics:
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return RunInfo(path, path.name, markers, metrics, mtime)


def scan_runs(root: Path | None, max_depth: int = 2) -> list[RunInfo]:
    """The root itself (if it is a run) plus run directories up to ``max_depth`` below it."""
    if root is None or not root.is_dir():
        return []
    found: list[RunInfo] = []
    seen: set[Path] = set()

    def walk(directory: Path, depth: int) -> None:
        info = _inspect(directory)
        if info and directory not in seen:
            seen.add(directory)
            found.append(info)
        if depth >= max_depth:
            return
        try:
            children = sorted(p for p in directory.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            return
        for child in children:
            walk(child, depth + 1)

    walk(root, 0)
    found.sort(key=lambda r: (-r.mtime, r.name))
    return found


# -- loading -------------------------------------------------------------------

def load_run(path: Path, metrics_name: str | None = None) -> RunDetail:
    detail = RunDetail(path=path)
    for name in MANIFEST_NAMES:
        data = _read_json(path / name)
        if isinstance(data, dict):
            detail.manifest, detail.manifest_name = data, name
            break
    candidates = _metrics_candidates(path)
    pick = metrics_name if metrics_name in candidates else (candidates[0] if candidates else "")
    if pick:
        data = _read_json(path / pick)
        if isinstance(data, dict):
            detail.metrics, detail.metrics_name = data, pick
    data = _read_json(path / "summary.json")
    if isinstance(data, dict) and not ("meta" in data and "gates" in data):
        detail.summary = data
    for p in sorted(path.glob(RUN_BUDGET_GLOB)):
        data = _read_json(p)
        if isinstance(data, dict):
            detail.run_budgets[p.name] = data
    for p in sorted(path.glob(CHAIN_BUDGET_GLOB)):
        for row in _read_jsonl(p):
            row["_file"] = p.name
            detail.chain_budget.append(row)
    detail.done_rows = _read_jsonl(path / "done.jsonl")
    if not detail.chain_budget and not detail.done_rows and (path / "records.jsonl").exists():
        for row in _read_jsonl(path / "records.jsonl"):
            reasons = row.get("chain_invalid_reasons")
            if row.get("valid") is False or reasons:
                cid = str(row.get("chain_id") or row.get("sid") or "?")
                bucket = detail.record_invalid.setdefault(cid, [])
                for r in reasons or ["valid=false"]:
                    if str(r) not in bucket:
                        bucket.append(str(r))
    if not any((detail.manifest, detail.metrics, detail.summary, detail.run_budgets, detail.chain_budget)):
        detail.notes.append("no manifest / metrics / summary found in this directory")
    return detail


# -- views ---------------------------------------------------------------------

def verdict(detail: RunDetail) -> str:
    gates = (detail.metrics or {}).get("gates") or {}
    return str(gates.get("verdict") or "-")


def manifest_lines(detail: RunDetail) -> list[tuple[str, str]]:
    m = detail.manifest or {}
    out: list[tuple[str, str]] = []
    if not m:
        return [("manifest", "none (no run_manifest.json / MANIFEST.json)")]
    out.append(("file", detail.manifest_name))
    for key in ("run_id", "tool", "profile"):
        if m.get(key):
            out.append((key, fmt(m[key])))
    started = m.get("started_utc") or m.get("started")
    ended = m.get("ended_utc") or m.get("ended")
    if started or ended:
        out.append(("window", f"{fmt(started)} -> {fmt(ended)}"))
    models = m.get("models")
    if isinstance(models, dict):
        out.append(("models", ", ".join(f"{k}={v}" for k, v in models.items() if v)))
    emb = m.get("embedder")
    if isinstance(emb, dict):
        flags = emb.get("flags") or []
        out.append(("embedder", f"{emb.get('id')} [source: {emb.get('source')}]" + (f" FLAGS: {'; '.join(map(str, flags))}" if flags else "")))
    elif emb:
        out.append(("embedder", fmt(emb)))
    if "judge_same_family" in m:
        out.append(("judge same family", fmt(m.get("judge_same_family")) + (f" ({m['judge_same_family_reason']})" if m.get("judge_same_family_reason") else "")))
    if m.get("arms"):
        out.append(("arms", fmt(m["arms"])))
    if m.get("mem_budgets"):
        out.append(("mem budgets", fmt(m["mem_budgets"])))
    params = m.get("params")
    if isinstance(params, dict) and params.get("provider"):
        out.append(("provider", fmt(params["provider"])))
    if isinstance(m.get("machine"), dict):
        out.append(("machine", ", ".join(f"{k}={v}" for k, v in m["machine"].items())))
    if m.get("git_head"):
        out.append(("git head", str(m["git_head"])[:12]))
    if isinstance(m.get("argv"), list):
        out.append(("command", " ".join(redact_argv(m["argv"]))))
    return out


def spend_lines(detail: RunDetail) -> list[tuple[str, str]]:
    m = detail.manifest or {}
    spend = m.get("spend") if isinstance(m.get("spend"), dict) else None
    out: list[tuple[str, str]] = []
    if spend:
        for key, value in spend.items():
            if key != "note":
                out.append((key, fmt(value, 4)))
        if spend.get("note"):
            out.append(("note", str(spend["note"])))
    elif isinstance(m.get("cost_usd"), dict):
        out += [(k, fmt(v, 4)) for k, v in m["cost_usd"].items()]
    elif detail.summary and isinstance(detail.summary.get("llm_cost_usd"), dict):
        out += [(k, fmt(v, 4)) for k, v in detail.summary["llm_cost_usd"].items()]
    return out or [("spend", "not recorded")]


_METRIC_COLS = (
    ("n", "n"), ("n_answerable", "n_ans"), ("score", "score"), ("anchor_hit", "anchor"),
    ("clean_anchor_hit", "clean_anchor"), ("hallucination", "halluc"), ("abstained", "abstained"),
    ("abstain_on_unanswerable", "abstain@UNANS"), ("latency_median", "lat_med_s"),
    ("memory_tokens_median", "mem_tok_med"), ("cost_per_question", "cost/q"),
)
_SUMMARY_COLS = (
    ("n", "n"), ("n_valid", "n_valid"), ("anchor_hit_rate", "anchor_hit"), ("rubric_hit_mean", "rubric"),
    ("memory_tokens_mean", "mem_tok"), ("latency_s_mean", "lat_s"), ("cost_usd_asks", "cost_asks"),
)


def arms_table(detail: RunDetail) -> Table:
    """Per-arm table straight from metrics_v2's ``summary`` (or the runner's ``by_arm_budget``)."""
    if detail.metrics and isinstance(detail.metrics.get("summary"), dict):
        summ = detail.metrics["summary"]
        present = [(k, h) for k, h in _METRIC_COLS if any(isinstance(v, dict) and k in v for v in summ.values())]
        table = Table(["arm", *[h for _, h in present]])
        for arm in sorted(summ):
            v = summ[arm]
            if isinstance(v, dict):
                table.rows.append([arm, *[fmt(v.get(k)) for k, _ in present]])
        return table
    if detail.summary and isinstance(detail.summary.get("by_arm_budget"), dict):
        by = detail.summary["by_arm_budget"]
        table = Table(["arm@budget", *[h for _, h in _SUMMARY_COLS]])
        for key in sorted(by):
            table.rows.append([key, *[fmt(by[key].get(k)) for k, _ in _SUMMARY_COLS]])
        return table
    return Table(["arm"])


def deltas_table(detail: RunDetail) -> Table:
    table = Table(["pair", "metric", "delta", "CI lo", "CI hi", "chains", "wilcoxon p"])
    for pair, d in ((detail.metrics or {}).get("deltas") or {}).items():
        for metric, m in (d.get("metrics") or {}).items():
            table.rows.append([pair, metric, fmt(m.get("point")), fmt(m.get("ci_lo")), fmt(m.get("ci_hi")),
                               fmt(m.get("n_chains")), fmt(m.get("wilcoxon_p"))])
    return table


def gates_table(detail: RunDetail) -> Table:
    gates = (detail.metrics or {}).get("gates") or {}
    table = Table(["rule set", "rule", "ok", "definition", "observed"])
    for label, key in (("WIN", "win_rules"), ("FAIL", "fail_rules")):
        for rule in gates.get(key) or []:
            if isinstance(rule, dict):
                table.rows.append([label, str(rule.get("name")), fmt(rule.get("ok")),
                                   str(rule.get("rule", "")), str(rule.get("observed", ""))])
    return table


def strata_table(detail: RunDetail) -> Table:
    """Per-qtype (and per-slice) strata exactly as the producing tool wrote them."""
    table = Table(["source", "stratum", "arm", "n", "score", "anchor / clean_anchor", "other"])
    metrics = detail.metrics or {}
    ls = metrics.get("longsession") if isinstance(metrics.get("longsession"), dict) else None
    for qt, v in ((ls or {}).get("strata") or {}).items():
        label = f"{qt} [{v.get('declared')}]" if v.get("declared") else qt
        for arm, g in (v.get("per_arm") or {}).items():
            if isinstance(g, dict) and g.get("n"):
                table.rows.append(["metrics_v2 longsession", label, arm, fmt(g.get("n")), fmt(g.get("score")),
                                   f"{fmt(g.get('anchor_hit'))} / {fmt(g.get('clean_anchor_hit'))}",
                                   f"rubric={fmt(g.get('rubric_hit_frac'))}"])
    for arm, slices in (metrics.get("slices") or {}).items():
        if isinstance(slices, dict):
            for name, g in slices.items():
                if isinstance(g, dict):
                    table.rows.append(["metrics_v2 slices", name, arm, fmt(g.get("n")), fmt(g.get("score")),
                                       f"{fmt(g.get('anchor_hit'))} / {fmt(g.get('clean_anchor_hit'))}", ""])
    by = (detail.summary or {}).get("by_arm_budget") or {}
    for key in sorted(by):
        for qt, g in (by[key].get("by_qtype") or {}).items():
            table.rows.append(["summary.json", qt, key, fmt(g.get("n")), "-",
                               fmt(g.get("anchor_hit_rate")), f"rubric={fmt(g.get('rubric_hit_mean'))} lat={fmt(g.get('latency_s_mean'))}"])
    return table


def invalid_table(detail: RunDetail) -> Table:
    """Invalid chains / sessions with the reasons the tools recorded."""
    table = Table(["chain / session", "scope", "source", "reasons"])
    for row in detail.chain_budget:
        if row.get("valid") is False or row.get("invalid_reasons"):
            table.rows.append([str(row.get("chain_id")), f"B={row.get('mem_budget', '-')}", str(row.get("_file") or "chain_budget.jsonl"),
                               "; ".join(map(str, row.get("invalid_reasons") or ["valid=false"]))])
    for row in detail.done_rows:
        if row.get("valid") is False:
            table.rows.append([str(row.get("sid") or row.get("chain_id")), f"{row.get('arm', '-')} B={row.get('B', '-')}",
                               "done.jsonl", "; ".join(map(str, row.get("reasons") or ["valid=false"]))])
    validity = ((detail.metrics or {}).get("meta") or {}).get("validity") or {}
    for chain, why in (validity.get("dropped_chains") or {}).items():
        table.rows.append([str(chain), "dropped from analysis", f"{detail.metrics_name} meta.validity", "; ".join(map(str, why))])
    for qid, missing in (validity.get("incomplete_qids") or {}).items():
        table.rows.append([str(qid), "incomplete arm triplet", f"{detail.metrics_name} meta.validity", "missing " + ", ".join(map(str, missing))])
    for cid, reasons in detail.record_invalid.items():
        table.rows.append([cid, "records.jsonl", "records.jsonl", "; ".join(reasons)])
    for cid in (detail.summary or {}).get("failed_sessions") or []:
        table.rows.append([str(cid), "failed session", "summary.json", "listed in failed_sessions"])
    return table


def meta_lines(detail: RunDetail) -> list[tuple[str, str]]:
    meta = (detail.metrics or {}).get("meta") or {}
    out = []
    for key in ("records", "chains", "sessions", "questions", "primary_b", "treatment", "judge_model", "boot_n", "seed", "qtype_mapping"):
        if key in meta and meta[key] is not None:
            out.append((key, fmt(meta[key])))
    if meta.get("judge_same_family"):
        out.append(("LIMITATION", "same-family judge (self-preference not controlled)"))
    return out


def budget_lines(detail: RunDetail) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name, rb in detail.run_budgets.items():
        out.append((name, ", ".join(f"{k}={fmt(v)}" for k, v in rb.items())))
    return out
