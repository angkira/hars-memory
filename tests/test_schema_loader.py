from pathlib import Path

import pytest
import yaml

from hars_memory.schema.loader import Schema, SchemaValidationError, load_schema


def _write_schema(tmp_path: Path, entity_types: list[str], relation_types: list[str] | None = None) -> Path:
    p = tmp_path / "schema.yaml"
    p.write_text(
        yaml.dump(
            {
                "entity_types": entity_types,
                "relation_types": relation_types or ["relates_to"],
                "guidance": "Generic extraction guidance.",
            }
        )
    )
    return p


def test_load_valid_schema(tmp_path: Path) -> None:
    path = _write_schema(tmp_path, ["Concept", "Document"])
    schema = load_schema(path)
    assert isinstance(schema, Schema)
    assert schema.entity_types == ["Concept", "Document"]
    assert schema.relation_types == ["relates_to"]
    assert schema.guidance == "Generic extraction guidance."


def test_rejects_entity_type_with_slash(tmp_path: Path) -> None:
    path = _write_schema(tmp_path, ["Model/Backbone"])
    with pytest.raises(SchemaValidationError, match="/"):
        load_schema(path)


def test_rejects_entity_type_with_pipe(tmp_path: Path) -> None:
    path = _write_schema(tmp_path, ["Model|Backbone"])
    with pytest.raises(SchemaValidationError, match=r"\|"):
        load_schema(path)


def test_rejects_relation_type_with_slash(tmp_path: Path) -> None:
    path = _write_schema(tmp_path, ["Concept"], relation_types=["uses/depends"])
    with pytest.raises(SchemaValidationError, match="/"):
        load_schema(path)


def test_rejects_relation_type_with_pipe(tmp_path: Path) -> None:
    path = _write_schema(tmp_path, ["Concept"], relation_types=["uses|depends"])
    with pytest.raises(SchemaValidationError, match=r"\|"):
        load_schema(path)


def test_default_schema_ships_with_package_and_loads_clean() -> None:
    from hars_memory.schema.loader import DEFAULT_SCHEMA_PATH

    schema = load_schema(DEFAULT_SCHEMA_PATH)
    assert len(schema.entity_types) > 0
    assert "Hypothesis" not in schema.entity_types  # generic default carries no HARS vocabulary
