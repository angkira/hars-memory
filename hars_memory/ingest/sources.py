"""Knowledge-source manifest: the single maintained list of directories the
memory engine draws knowledge from.

Knowledge does not live in one directory. Before this module the two pipelines
each learned about their inputs separately — the indexer from ``memory-index
--paths`` and the literal search channel from ``HARS_MEMORY_RIPGREP_ROOTS`` —
so a newly added knowledge directory had to be remembered in two places and
was, in practice, remembered in neither.

A manifest is a YAML file owned by the knowledge repository (not by this
engine), pointed at with ``HARS_MEMORY_SOURCES_MANIFEST``::

    version: 1
    sources:
      - path: kb
        index: true
        grep: true
        note: Team knowledge base.

``path`` is resolved relative to the base manifest's directory (also for the
local override below) unless absolute or ``~``-prefixed.
``index`` (default true) selects ingest roots, ``grep``
(default true) selects literal-search roots; an entry may enable either, both
or neither. A path that does not exist is skipped with a warning rather than
failing the run — entries legitimately point at uninitialized submodules or
machine-local directories.

A sibling ``*.local.yaml`` override (``HARS_MEMORY_SOURCES_MANIFEST_LOCAL``, or
``meta/knowledge-sources.local.yaml`` next to the manifest by default) is
appended when present, so machine-specific roots stay out of version control.
Both env vars unset means "no manifest" — every caller keeps its previous
behaviour untouched.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

HARS_MEMORY_SOURCES_MANIFEST_ENV: Final[str] = "HARS_MEMORY_SOURCES_MANIFEST"
HARS_MEMORY_SOURCES_MANIFEST_LOCAL_ENV: Final[str] = "HARS_MEMORY_SOURCES_MANIFEST_LOCAL"

_DEFAULT_LOCAL_OVERRIDE: Final[str] = "meta/knowledge-sources.local.yaml"
_SUPPORTED_VERSIONS: Final[frozenset[int]] = frozenset({1})


class SourcesManifestError(RuntimeError):
    """The manifest exists but cannot be used as configured."""


@dataclass(frozen=True)
class KnowledgeSource:
    path: Path
    index: bool
    grep: bool
    note: str = ""


def _coerce_flag(raw: Any, *, field: str, entry: str) -> bool:
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    raise SourcesManifestError(
        f"source {entry!r}: {field} must be a boolean, got {type(raw).__name__}"
    )


def _resolve(raw_path: str, base_dir: Path) -> Path:
    expanded = Path(raw_path).expanduser()
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return expanded.resolve()


def _parse(document: Any, manifest_path: Path, base_dir: Path) -> list[KnowledgeSource]:
    if document is None:
        return []
    if not isinstance(document, dict):
        raise SourcesManifestError(
            f"{manifest_path}: top level must be a mapping, got {type(document).__name__}"
        )

    version = document.get("version", 1)
    if version not in _SUPPORTED_VERSIONS:
        raise SourcesManifestError(
            f"{manifest_path}: unsupported manifest version {version!r} "
            f"(supported: {sorted(_SUPPORTED_VERSIONS)})"
        )

    raw_sources = document.get("sources") or []
    if not isinstance(raw_sources, list):
        raise SourcesManifestError(
            f"{manifest_path}: `sources` must be a list, got {type(raw_sources).__name__}"
        )

    sources: list[KnowledgeSource] = []
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise SourcesManifestError(
                f"{manifest_path}: each source must be a mapping, got {type(raw).__name__}"
            )
        raw_path = raw.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise SourcesManifestError(f"{manifest_path}: every source needs a non-empty `path`")
        sources.append(
            KnowledgeSource(
                path=_resolve(raw_path.strip(), base_dir),
                index=_coerce_flag(raw.get("index"), field="index", entry=raw_path),
                grep=_coerce_flag(raw.get("grep"), field="grep", entry=raw_path),
                note=str(raw.get("note") or ""),
            )
        )
    return sources


def _load_file(manifest_path: Path, base_dir: Path) -> list[KnowledgeSource]:
    import yaml

    try:
        document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SourcesManifestError(f"{manifest_path}: invalid YAML — {exc}") from exc
    return _parse(document, manifest_path, base_dir)


def manifest_path() -> Path | None:
    """Configured manifest path, or None when the feature is not in use.

    A configured-but-missing manifest is an error, not a silent no-op: it
    almost certainly means a typo'd path, and silently indexing nothing is the
    failure mode this module exists to prevent.
    """
    raw = os.environ.get(HARS_MEMORY_SOURCES_MANIFEST_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise SourcesManifestError(
            f"{HARS_MEMORY_SOURCES_MANIFEST_ENV}={raw} does not point at a readable file"
        )
    return path


def _local_override_path(base_manifest: Path) -> Path | None:
    raw = os.environ.get(HARS_MEMORY_SOURCES_MANIFEST_LOCAL_ENV, "").strip()
    path = (
        Path(raw).expanduser().resolve()
        if raw
        else (base_manifest.parent / _DEFAULT_LOCAL_OVERRIDE)
    )
    return path if path.is_file() else None


def load_sources() -> list[KnowledgeSource]:
    """All declared sources (base manifest + optional local override), deduped
    by resolved path with the later declaration winning. Returns `[]` when no
    manifest is configured.
    """
    base = manifest_path()
    if base is None:
        return []

    base_dir = base.parent
    sources = _load_file(base, base_dir)
    override = _local_override_path(base)
    if override is not None:
        logger.info("Applying local knowledge-source override: %s", override)
        # Relative paths in the override resolve against the BASE manifest's
        # directory (the knowledge repository root), not the override's own
        # location — the override typically sits in a `meta/` subdirectory and
        # `- path: kb` there must mean the same directory it means in the base.
        sources += _load_file(override, base_dir)

    deduped: dict[Path, KnowledgeSource] = {}
    for source in sources:
        deduped[source.path] = source
    return list(deduped.values())


def _existing(sources: list[KnowledgeSource], kind: str) -> list[Path]:
    roots: list[Path] = []
    for source in sources:
        if source.path.exists():
            roots.append(source.path)
        else:
            logger.warning(
                "Knowledge source declared for %s does not exist, skipping: %s", kind, source.path
            )
    return roots


def ingest_roots() -> list[Path]:
    """Existing paths declared with `index: true`, in manifest order."""
    return _existing([s for s in load_sources() if s.index], "indexing")


def grep_roots() -> list[Path]:
    """Existing paths declared with `grep: true`, in manifest order."""
    return _existing([s for s in load_sources() if s.grep], "literal search")


def read_manifest(manifest_path: Path) -> list[KnowledgeSource]:
    """Read a knowledge-source manifest file and return KnowledgeSource entries."""
    path = Path(manifest_path).expanduser().resolve()
    return _load_file(path, path.parent)


__all__ = [
    "HARS_MEMORY_SOURCES_MANIFEST_ENV",
    "HARS_MEMORY_SOURCES_MANIFEST_LOCAL_ENV",
    "KnowledgeSource",
    "SourcesManifestError",
    "grep_roots",
    "ingest_roots",
    "load_sources",
    "manifest_path",
    "read_manifest",
]
