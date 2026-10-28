#!/usr/bin/env python3
"""THE anchor matcher (Amendment C.9). One function, used for every "is the gold anchor in this text" question of the
agentic track: the tool results (workspace and memory, separately), the summary, the dropped context (the
memory-relevant subset), the final answer. The methodology review of the C.6 bundle found two matchers over two
corpora giving different rates for the same quantity; nothing in run_agentic / metrics_agentic may use another one.

Semantics (identical to judge_v2.anchor_re and run_longsession.anchor_pattern, pinned by a test): the anchor, stripped,
escaped, case-insensitive, bounded on both sides by a non-[A-Za-z0-9_] character (token-bounded)."""
from __future__ import annotations

import functools
import re


@functools.lru_cache(maxsize=8192)
def _pattern(anchor: str):
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(anchor) + r"(?![A-Za-z0-9_])", re.I)


def anchor_found(text, anchor) -> bool:
    """True iff `anchor` occurs in `text` (see the module docstring). Empty anchor or text -> False."""
    a = (anchor or "").strip() if isinstance(anchor, str) else ""
    if not a or not isinstance(text, str) or not text:
        return False
    return _pattern(a).search(text) is not None
