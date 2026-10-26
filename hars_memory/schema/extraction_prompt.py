"""Extraction prompt helpers for LightRAG.

The entity/relation type vocabulary and the domain extraction guidance text
used to be a hardcoded f-string built from the ``EntityType``/``RelationType``
enums. Both now live in a loadable YAML ``Schema`` (see
``hars_memory.schema.loader``) — the functions below derive the
LightRAG-facing strings from a loaded ``Schema`` instead of hardcoding them.
"""

from __future__ import annotations

from hars_memory.schema.loader import Schema


def entity_types_prompt(schema: Schema) -> str:
    """Comma-separated entity type list injected into LightRAG's extraction prompt."""
    return ", ".join(schema.entity_types)


def relation_types_prompt(schema: Schema) -> str:
    """Comma-separated relation type list."""
    return ", ".join(schema.relation_types)


def domain_extraction_guidance(schema: Schema) -> str:
    """Text appended verbatim to LightRAG's ``entity_extraction_system_prompt``."""
    return schema.guidance


__all__ = [
    "entity_types_prompt",
    "relation_types_prompt",
    "domain_extraction_guidance",
]
