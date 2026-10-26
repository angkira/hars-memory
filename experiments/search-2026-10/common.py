"""Shared helpers for the 2026-10 search experiment (cost-capped LLM, CIs, cases I/O)."""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
SOURCE_ROOT = Path("/Users/iurii.medvedev/Work/sre-support-pier")
CLONE = Path("/tmp/search-exp/base")

COST_FILE = EXP / "cost.json"
BUDGET_USD = 4.0
PRICE_IN, PRICE_OUT = 0.10, 0.50  # USD per 1M tokens
MODEL = "openai/gpt-6-luna"

# Import smoke_eval helpers without polluting package layout.
sys.path.insert(0, str(REPO / "scripts"))
import smoke_eval  # noqa: E402

load_env_file = smoke_eval.load_env_file
make_client = smoke_eval.make_client
parse_json = smoke_eval.parse_json
Doc = smoke_eval.Doc
flat_name = smoke_eval.flat_name
inventory = smoke_eval.inventory

last_usage: dict[str, Any] = {}


class BudgetExceeded(RuntimeError):
    """Raised when cumulative LLM spend would exceed BUDGET_USD."""


def ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _read_cost() -> dict[str, Any]:
    if COST_FILE.exists():
        return json.loads(COST_FILE.read_text("utf-8"))
    return {"calls": 0, "in": 0, "out": 0, "usd": 0.0, "by_tag": {}}


def cost_usd() -> float:
    return float(_read_cost().get("usd") or 0.0)


def cost_add(tag: str, tin: int, tout: int) -> float:
    EXP.mkdir(parents=True, exist_ok=True)
    data = _read_cost()
    data["calls"] = int(data.get("calls") or 0) + 1
    data["in"] = int(data.get("in") or 0) + int(tin)
    data["out"] = int(data.get("out") or 0) + int(tout)
    by_tag = data.setdefault("by_tag", {})
    t = by_tag.setdefault(tag, {"calls": 0, "in": 0, "out": 0, "usd": 0.0})
    t["calls"] = int(t.get("calls") or 0) + 1
    t["in"] = int(t.get("in") or 0) + int(tin)
    t["out"] = int(t.get("out") or 0) + int(tout)
    tag_usd = t["in"] / 1e6 * PRICE_IN + t["out"] / 1e6 * PRICE_OUT
    t["usd"] = round(tag_usd, 6)
    data["usd"] = round(data["in"] / 1e6 * PRICE_IN + data["out"] / 1e6 * PRICE_OUT, 6)
    COST_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return float(data["usd"])


async def llm(client: Any, prompt: str, tag: str, max_out: int = 4000) -> str:
    """Call gpt-6-luna; enforce budget; record usage in cost.json and last_usage."""
    global last_usage
    current = cost_usd()
    if current >= BUDGET_USD:
        raise BudgetExceeded(f"budget exhausted: ${current:.4f} >= ${BUDGET_USD:.2f} before tag={tag}")

    model = MODEL.split("/", 1)[1] if "/" in MODEL else MODEL
    r = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_completion_tokens=max_out,
    )
    u = r.usage
    tin = int(getattr(u, "prompt_tokens", 0) or 0)
    tout = int(getattr(u, "completion_tokens", 0) or 0)
    usd = cost_add(tag, tin, tout)
    last_usage = {
        "tag": tag,
        "prompt_tokens": tin,
        "completion_tokens": tout,
        "usd_running": usd,
    }
    print(
        f"  [cost] {tag}: +{tin}/{tout} tok, running experiment tally ${usd:.4f}",
        file=sys.stderr,
    )
    return r.choices[0].message.content or ""


def boot_ci(vals: list[float], n: int = 2000, seed: int = 1) -> tuple[float, float]:
    """Bootstrap 95% CI of the mean. Returns (lo, hi)."""
    if not vals:
        return (0.0, 0.0)
    rnd = random.Random(seed)
    m = len(vals)
    means: list[float] = []
    for _ in range(n):
        sample = [vals[rnd.randrange(m)] for _ in range(m)]
        means.append(sum(sample) / m)
    means.sort()
    lo_i = max(0, int(0.025 * n) - 1)
    hi_i = min(n - 1, int(0.975 * n))
    return (means[lo_i], means[hi_i])


def paired_diff_ci(
    a: list[float], b: list[float], n: int = 2000, seed: int = 1
) -> tuple[float, float, float]:
    """Bootstrap 95% CI of mean(a - b). Returns (mean_diff, lo, hi)."""
    if not a or not b or len(a) != len(b):
        return (0.0, 0.0, 0.0)
    diffs = [x - y for x, y in zip(a, b)]
    mean_diff = sum(diffs) / len(diffs)
    lo, hi = boot_ci(diffs, n=n, seed=seed)
    return (mean_diff, lo, hi)


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or (EXP / "cases.yaml")
    data = yaml.safe_load(p.read_text("utf-8"))
    if isinstance(data, dict):
        return list(data.get("cases") or [])
    if isinstance(data, list):
        return data
    return []
