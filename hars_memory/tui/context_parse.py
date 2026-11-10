"""Split a ``memory_recall`` ``context`` string into its LightRAG sections.

Derived from the rendered text (like the server's own ``parse_context_blocks``),
so it is independent of the server package. Missing or garbled lines are skipped,
never guessed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_HEADER_RE = re.compile(
    r"^(?P<kind>Knowledge Graph Data \((?P<sub>Entity|Relationship)\)|Document Chunks|Reference Document List).*:\s*$"
)
_REF_RE = re.compile(r"^\[(?P<ref>[^\]]+)\]\s*(?P<path>.+)$")


@dataclass
class ContextSections:
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    references: dict[str, str] = field(default_factory=dict)
    chars: int = 0

    @property
    def has_content(self) -> bool:
        return bool(self.entities or self.relations or self.chunks)


def parse_context(context: str | None) -> ContextSections:
    out = ContextSections(chars=len(context or ""))
    section: str | None = None
    for raw in (context or "").splitlines():
        line = raw.strip()
        header = _HEADER_RE.match(line)
        if header:
            kind = header.group("kind")
            section = ("entity" if header.group("sub") == "Entity" else "relation") if header.group("sub") else (
                "chunks" if kind == "Document Chunks" else "refs")
            continue
        if not line or line.startswith("```") or section is None:
            continue
        if section == "refs":
            m = _REF_RE.match(line)
            if m:
                out.references[m.group("ref")] = m.group("path").strip()
            continue
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        {"entity": out.entities, "relation": out.relations, "chunks": out.chunks}[section].append(obj)
    return out
