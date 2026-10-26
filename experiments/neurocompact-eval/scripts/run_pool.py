"""Minimal ``run_pool`` helper for run_chainreplay.py: parse pi ``--mode json`` stdout.

Only ``extract`` lives here (the original run_pool.py with ``run_pi``/``judge``/
``hars_recall`` stayed on the owner's machine; run_longreplay.py / judge_backfill.py
still need that fuller version).

Event schema (``@earendil-works/pi-coding-agent`` 1.4.0, one JSON object per stdout
line, verified against the installed package and a captured run):

* ``{"type":"message_end","message":{"role":"assistant","content":[{"type":"text",
  "text":...}|{"type":"thinking",...}|{"type":"toolCall",...}],"usage":{"input",
  "output","cacheRead","cacheWrite","totalTokens",...},"stopReason":...}}``
* ``{"type":"compaction_end","reason","result","aborted","willRetry"}``

``message_update`` / ``turn_end`` / ``agent_end`` repeat the same assistant message
and are ignored so nothing is double counted.

Returned ``usage``:

* ``input``  = context size of the LAST assistant turn = ``input + cacheRead +
  cacheWrite``. Anthropic reports cached prompt tokens separately (``input`` is
  only the uncached remainder, e.g. 4 for a 2661-token prompt), so the three must
  be summed. Providers without caching report cacheRead/cacheWrite as 0.
* ``output`` = output tokens summed over all assistant turns of the call.
"""
from __future__ import annotations

import json


def _num(v) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "".join(parts)


def extract(stdout: str) -> dict:
    """pi ``--mode json`` stdout -> {"text", "usage": {"input","output"}, "compactions"}."""
    text = ""
    last_input = 0
    out_total = 0
    compactions = 0
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        kind = ev.get("type")
        if kind == "compaction_end":
            compactions += 1
        elif kind == "message_end":
            msg = ev.get("message")
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            text = _text(msg)
            u = msg.get("usage") or {}
            last_input = _num(u.get("input")) + _num(u.get("cacheRead")) + _num(u.get("cacheWrite"))
            out_total += _num(u.get("output"))
    return {"text": text, "usage": {"input": last_input, "output": out_total},
            "compactions": compactions}


def extract_spend(stdout: str) -> dict:
    """Per-assistant-turn token split + pi's own catalog-priced cost, for ``--budget-usd``.

    Separate from :func:`extract` (whose return shape is pinned by tests). Same event rules:
    only ``message_end`` assistant messages count. ``turns`` is a list of
    ``{"input","output","cacheRead","cacheWrite"}`` (``input`` = uncached remainder), so a
    price tier can be picked per request; ``pi_cost_usd`` = sum of ``usage.cost.total``.
    """
    turns: list[dict] = []
    pi_cost = 0.0
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict) or ev.get("type") != "message_end":
            continue
        msg = ev.get("message")
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        u = msg.get("usage") or {}
        c = u.get("cost") if isinstance(u.get("cost"), dict) else {}
        t = c.get("total")
        pi_cost += float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) else 0.0
        turns.append({"input": _num(u.get("input")), "output": _num(u.get("output")),
                      "cacheRead": _num(u.get("cacheRead")), "cacheWrite": _num(u.get("cacheWrite"))})
    return {"turns": turns, "pi_cost_usd": pi_cost}
