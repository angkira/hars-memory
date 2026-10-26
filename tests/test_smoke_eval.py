"""Unit tests for scripts/smoke_eval.py helpers (no LLM, no index)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "smoke_eval", Path(__file__).resolve().parents[1] / "scripts" / "smoke_eval.py"
)
assert _SPEC and _SPEC.loader
smoke_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke_eval)

DOC = """# Title

## Setup

### Setup

| a | b |
|---|---|
| 1 | 2 |

```bash
# not a heading
run --now
```

- item one
"""


def _span(fact: str) -> tuple[int, int]:
    i = DOC.index(fact)
    return i, i + len(fact)


def test_heading_path_ignores_fenced_hash_lines() -> None:
    d = smoke_eval.Doc(DOC)
    assert d.path_at(d.line_of(DOC.index("run --now"))) == ["Title", "Setup", "Setup"]
    assert len(d.headings) == 3


@pytest.mark.parametrize(
    ("fact", "kind"),
    [("| 1 | 2 |", "table_row"), ("run --now", "code_block"), ("- item one", "list")],
)
def test_detect_kinds(fact: str, kind: str) -> None:
    assert kind in smoke_eval.Doc(DOC).detect(*_span(fact))


def test_duplicate_heading_detected() -> None:
    assert "duplicate_heading" in smoke_eval.Doc(DOC).detect(*_span("| 1 | 2 |"))


def test_text_heading_status() -> None:
    chunk = "# Title > ## Setup\n\nfoo bar baz"
    assert smoke_eval.text_heading_status(chunk, "foo bar", ["Title", "Setup"]) == "correct"
    assert smoke_eval.text_heading_status("foo bar baz", "foo bar", ["Title", "Setup"]) == "none"
    assert smoke_eval.text_heading_status("## Other\n\nfoo bar", "foo bar", ["Title", "Setup"]) == "wrong"


def test_meta_heading_ok_uses_merged_member() -> None:
    rec = {
        "heading_path": ["T", "A"],
        "merged_from": [
            {"heading_path": ["T", "A"], "start_line": 1, "end_line": 5},
            {"heading_path": ["T", "B"], "start_line": 7, "end_line": 9},
        ],
    }
    assert smoke_eval.meta_heading_ok(rec, 8, ["T", "B"]) is True
    assert smoke_eval.meta_heading_ok(rec, 3, ["T", "B"]) is False
    assert smoke_eval.meta_heading_ok({}, 3, ["T"]) is None
