"""Tiny local state: the recall query history ring buffer (never stored in the KB)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

HISTORY_MAX = 50


def default_state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "hars-memory-tui"


class HistoryStore:
    def __init__(self, directory: Path | None = None, max_items: int = HISTORY_MAX) -> None:
        self.path = (directory or default_state_dir()) / "recall-history.json"
        self.max_items = max_items
        self.items: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [d for d in data if isinstance(d, dict) and d.get("question")][-self.max_items:] if isinstance(data, list) else []

    def add(self, entry: dict[str, Any]) -> None:
        # Never persist credentials, even if a caller passes them in.
        entry = {k: v for k, v in entry.items() if k != "access_token"}
        self.items = [e for e in self.items if e != entry] + [entry]
        self.items = self.items[-self.max_items:]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.items, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass  # history is a convenience; never fail the UI over it
