"""Pin the LightRAG version decision (index-cutover blocker, audit 2026-10-17).

The serving venv must be able to read the store it serves: a 1.5.7-built
``kv_store_doc_status.json`` is not faithfully readable by 1.4.16 (rows carry
``content_hash`` → ``TypeError``; the ``parsing``/``analyzing`` statuses are
unknown to 1.4.16's ``DocStatus`` → ``get_status_counts()`` raises
``KeyError``). Reader and builder are therefore aligned on the exact builder
version, ``lightrag-hku==1.5.7``.

This test pins the *landed* decision: the declared specifier must select
exactly 1.5.7, and the 1.5-era import migration must be in place:

* ``chunking_by_token_size`` — imported from ``lightrag.chunker`` (moved out of
  ``lightrag.operate``); used by ``ingest/chunker.py`` and
  ``ingest/md_structure.py``.
* ``DEFAULT_ENTITY_TYPES`` — removed from ``lightrag.constants`` in 1.5; a
  local constant preserves the value in ``eval/extractor_bench.py``.

See ``docs/VERSION_PIN.md`` for the evidence, the migration, and the
upgrade/rollback procedure.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet

_REPO_ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_VERSION = "1.5.7"


def _declared_specifier() -> str:
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for dep in data["project"]["dependencies"]:
        if dep.replace(" ", "").startswith("lightrag-hku"):
            return dep.removeprefix("lightrag-hku")
    raise AssertionError("lightrag-hku is not a declared dependency")


def test_lightrag_specifier_pins_the_builder_version() -> None:
    spec = SpecifierSet(_declared_specifier())
    assert spec.contains(_REQUIRED_VERSION), (
        "lightrag-hku must admit the builder's version "
        f"{_REQUIRED_VERSION} so the serving reader can read a "
        f"{_REQUIRED_VERSION}-built store"
    )
    # Fail-safe: no OTHER version may resolve onto the migrated code, which
    # targets the 1.5.7 import/prompt contract exactly.
    assert not spec.contains("1.6.0"), "the pin must stay exact (==1.5.7)"
    assert not spec.contains("1.4.16"), "the superseded 1.4 line must not resolve"


def test_migrated_imports_are_in_place() -> None:
    """The exact symbols 1.5 moved/removed are imported from their 1.5 homes."""
    from lightrag.chunker import chunking_by_token_size  # noqa: F401  (1.5 home)

    import lightrag.constants as constants

    assert not hasattr(constants, "DEFAULT_ENTITY_TYPES"), (
        "lightrag.constants.DEFAULT_ENTITY_TYPES unexpectedly exists again — "
        "the local shim in eval/extractor_bench.py can be reconsidered"
    )

    from hars_memory.eval import extractor_bench

    assert extractor_bench.DEFAULT_ENTITY_TYPES, (
        "eval/extractor_bench.py must carry the local DEFAULT_ENTITY_TYPES list"
    )
