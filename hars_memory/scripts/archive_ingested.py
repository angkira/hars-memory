#!/usr/bin/env python3
"""Archive staged notes into ``<staging>/ingested/`` ONLY if their ingest succeeded.

``update_kb.sh`` used to ``mv`` every staged ``*.md`` after the indexer exited,
without checking per-document outcome, so a note whose LightRAG ingest failed
(or never ran) was silently moved to an ignored directory and lost. This helper
looks up each note's LightRAG ``doc_status`` (``kv_store_doc_status.json`` in the
index dir, keyed by :func:`hars_memory.ingest.document.file_stable_id`) and moves
only notes whose status is ``processed``. Everything else stays in staging and is
reported; the exit code is non-zero if any note was left behind.

Usage:
    python -m hars_memory.scripts.archive_ingested --index-dir DIR --staging DIR [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from hars_memory.ingest.document import file_stable_id

DOC_STATUS_FILENAME: Final[str] = "kv_store_doc_status.json"
STATUS_PROCESSED: Final[str] = "processed"
STATUS_ABSENT: Final[str] = "absent"
ARCHIVE_DIRNAME: Final[str] = "ingested"


class DocStatusError(RuntimeError):
    """The doc-status store is missing or unreadable (never guess success)."""


@dataclass(frozen=True, slots=True)
class ArchiveResult:
    archived: list[Path] = field(default_factory=list)
    left_behind: dict[Path, str] = field(default_factory=dict)  # note -> status


def load_doc_status(index_dir: Path) -> dict[str, dict[str, object]]:
    path = index_dir / DOC_STATUS_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DocStatusError(f"cannot read doc status store {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DocStatusError(f"doc status store {path} is not a JSON object")
    return data


def note_status(note: Path, doc_status: dict[str, dict[str, object]]) -> str:
    record = doc_status.get(file_stable_id(note))
    if not isinstance(record, dict):
        return STATUS_ABSENT
    return str(record.get("status", STATUS_ABSENT))


def archive_succeeded(
    index_dir: Path, staging_dir: Path, *, dry_run: bool = False
) -> ArchiveResult:
    """Move notes whose doc_status is ``processed`` into ``staging_dir/ingested``."""
    doc_status = load_doc_status(index_dir)
    archive_dir = staging_dir / ARCHIVE_DIRNAME
    archived: list[Path] = []
    left_behind: dict[Path, str] = {}
    for note in sorted(staging_dir.glob("*.md")):
        status = note_status(note, doc_status)
        if status != STATUS_PROCESSED:
            left_behind[note] = status
            continue
        if not dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(note), str(archive_dir / note.name))
        archived.append(note)
    return ArchiveResult(archived=archived, left_behind=left_behind)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = archive_succeeded(args.index_dir, args.staging, dry_run=args.dry_run)
    except DocStatusError as exc:
        print(f"ERROR: {exc}; nothing archived", file=sys.stderr)
        return 2
    verb = "would archive" if args.dry_run else "archived"
    print(f"{verb} {len(result.archived)} ingested note(s); left {len(result.left_behind)} in staging")
    for note, status in result.left_behind.items():
        print(f"NOT ARCHIVED (status={status}): {note.name}", file=sys.stderr)
    return 1 if result.left_behind else 0


if __name__ == "__main__":
    raise SystemExit(main())
