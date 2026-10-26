"""Loads an entity/relation extraction schema from a YAML config file.

Generic by default (see default_schema.yaml, shipped with the package).
A consuming project supplies its own schema file via
HARS_MEMORY_ENTITY_SCHEMA_PATH to override the entity/relation vocabulary
and extraction guidance without touching package code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml

DEFAULT_SCHEMA_PATH = Path(__file__).parent / "default_schema.yaml"

_FORBIDDEN_CHARS = ("/", "|")


class SchemaValidationError(ValueError):
    """Raised when a loaded schema's entity/relation types violate a hard
    LightRAG constraint (no '/' or '|' — LightRAG silently drops entities
    of a type value containing either)."""


@dataclass(slots=True)
class Schema:
    entity_types: list[str]
    relation_types: list[str]
    guidance: str
    entity_id_prefixes: dict[str, str] = field(default_factory=dict)


def _validate_no_forbidden_chars(values: list[str], *, kind: str) -> None:
    for value in values:
        for char in _FORBIDDEN_CHARS:
            if char in value:
                raise SchemaValidationError(
                    f"{kind} type '{value}' contains forbidden character '{char}' — "
                    "LightRAG will silently drop all entities of this type."
                )


def load_schema(path: Path) -> Schema:
    """Load and validate a schema YAML file. Raises SchemaValidationError
    on any entity/relation type containing '/' or '|'."""
    raw = yaml.safe_load(path.read_text())
    entity_types = raw["entity_types"]
    relation_types = raw["relation_types"]
    _validate_no_forbidden_chars(entity_types, kind="Entity")
    _validate_no_forbidden_chars(relation_types, kind="Relation")
    return Schema(
        entity_types=entity_types,
        relation_types=relation_types,
        guidance=raw.get("guidance", ""),
        entity_id_prefixes=raw.get("entity_id_prefixes", {}),
    )


def _member_name(value: str) -> str:
    """Convert a schema type value (CamelCase or snake_case) into a valid
    UPPER_SNAKE_CASE Python Enum member name, e.g. 'JobRun' -> 'JOB_RUN'."""
    spaced = re.sub(r"(?<!^)(?=[A-Z])", "_", value)
    sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", spaced)
    return sanitized.upper().strip("_")


def load_entity_types(schema_path: Path | None = None) -> tuple[type[Enum], type[Enum]]:
    """Load a schema and build ``EntityType``/``RelationType`` enums from it.

    Backward-compatible replacement for the previously hardcoded
    ``EntityType``/``RelationType`` enums in ``entity_types.py``. Defaults to
    the generic package schema (``DEFAULT_SCHEMA_PATH``) when no path is given.
    """
    schema = load_schema(schema_path or DEFAULT_SCHEMA_PATH)
    entity_type = Enum(
        "EntityType",
        {_member_name(value): value for value in schema.entity_types},
        type=str,
    )
    relation_type = Enum(
        "RelationType",
        {_member_name(value): value for value in schema.relation_types},
        type=str,
    )
    return entity_type, relation_type
