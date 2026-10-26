"""Tests for the staging-archive exclusion convention used by
scripts/update_kb.sh (see that script's ".memoryignore" generation step and
ingest/document.py's file_stable_id() docstring for the full bug writeup).

update_kb.sh moves every successfully-indexed staging note into
staging/ingested/ after each run, then (idempotently) ensures
staging/.memoryignore contains an "ingested/" exclude line. This uses
walker.py's existing, already-tested .memoryignore mechanism (see
test_memory_dir_ingest.py::test_memoryignore_respected_in_memory_dir) --
no walker.py code change was needed, only a generated config file at the
walked root, which is exercised here with a synthetic tmp_path staging dir
shaped like the real one.
"""

from __future__ import annotations

from pathlib import Path

from hars_memory.ingest.document import file_stable_id
from hars_memory.ingest.walker import walk


class TestIngestedArchiveExcludedViaMemoryignore:
    def test_ingested_notes_are_not_enumerated(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / ".memoryignore").write_text("ingested/\n")
        (staging / "new_note.md").write_text("fresh content")
        archive = staging / "ingested"
        archive.mkdir()
        (archive / "old_note.md").write_text("already-indexed content")

        docs, stats = walk([staging], dry_run=True)

        assert stats.files_accepted == 1
        assert docs[0].source_path.endswith("new_note.md")

    def test_without_memoryignore_ingested_notes_leak_back_in(self, tmp_path: Path) -> None:
        """Negative control: confirms the exclusion comes from the
        generated .memoryignore, not some other default -- proving the
        walker really would re-walk the archive without it (the original
        bug's mechanism)."""
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "new_note.md").write_text("fresh content")
        archive = staging / "ingested"
        archive.mkdir()
        (archive / "old_note.md").write_text("already-indexed content")

        docs, stats = walk([staging], dry_run=True)

        assert stats.files_accepted == 2

    def test_archived_note_id_matches_what_it_would_have_been_pre_archival(
        self, tmp_path: Path
    ) -> None:
        """Even though the walker no longer visits staging/ingested/ at all
        (belt), file_stable_id() itself is independently archive-invariant
        (suspenders) -- see test_ingest_document.py for the dedicated unit
        tests. Sanity-checked here against the same synthetic staging shape
        used by this module's other tests."""
        staging = tmp_path / "staging"
        staging.mkdir()
        archive = staging / "ingested"
        archive.mkdir()
        note_path = archive / "old_note.md"
        note_path.write_text("already-indexed content")

        pre_archival_path = staging / "old_note.md"
        assert file_stable_id(note_path) == file_stable_id(pre_archival_path)
