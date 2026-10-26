"""Tests for ingest/document.py's file_stable_id() -- specifically the fix
for the archival-path-hash bug (see .session note / task report):
scripts/update_kb.sh archives every successfully-indexed staging note into
staging/ingested/ after a run; file_stable_id() must be invariant to that
move so LightRAG never sees an archived file as "new".
"""

from __future__ import annotations

import re
from pathlib import Path

from hars_memory.ingest.document import file_stable_id


class TestFileStableIdArchiveInvariance:
    def test_archived_file_keeps_the_same_id_as_before_archival(self) -> None:
        staged = Path("/mnt/datasets/graphrag/staging/note.md")
        archived = Path("/mnt/datasets/graphrag/staging/ingested/note.md")

        assert file_stable_id(staged) == file_stable_id(archived)

    def test_only_a_literal_ingested_path_component_is_stripped(self) -> None:
        """A filename or directory that merely CONTAINS "ingested" as a
        substring (not a standalone path component) must not be affected --
        only an exact "ingested" component, matching update_kb.sh's
        mkdir -p "$STAGING/ingested" archive convention."""
        plain = Path("/mnt/datasets/graphrag/staging/pre-ingested-notes/note.md")
        archived = Path("/mnt/datasets/graphrag/staging/ingested/note.md")

        assert file_stable_id(plain) != file_stable_id(archived)

    def test_multiple_archival_round_trips_are_idempotent(self) -> None:
        """Even a pathologically-nested 'ingested/ingested/note.md' (which
        update_kb.sh's mkdir -p + single-level mv never produces, but which
        must not silently mis-hash if it ever occurred) strips every literal
        'ingested' component, not just the first."""
        staged = Path("/mnt/datasets/graphrag/staging/note.md")
        double_archived = Path("/mnt/datasets/graphrag/staging/ingested/ingested/note.md")

        assert file_stable_id(staged) == file_stable_id(double_archived)

    def test_id_format_unchanged(self) -> None:
        doc_id = file_stable_id(Path("/some/path/report.md"))
        assert re.match(r"^file:[0-9a-f]{12}$", doc_id)

    def test_different_files_still_get_different_ids(self) -> None:
        a = file_stable_id(Path("/mnt/datasets/graphrag/staging/note-a.md"))
        b = file_stable_id(Path("/mnt/datasets/graphrag/staging/note-b.md"))
        assert a != b


class TestFileStableIdReproducesRealHistoricalIds:
    """Regression fixture reproducing the exact real-world case that
    motivated this fix: a file physically archived into staging/ingested/
    whose doc_id was minted (under the OLD, buggy scheme) back when it
    still lived directly in staging/, and is recorded as such in the live
    kv_store_doc_status.json. The fixed file_stable_id() must reproduce
    that original id from the file's current (archived) location, not mint
    a new one.

    The concrete id/path pair below is copied verbatim from the live store
    (kv_store_doc_status.json key "file:0cdae3d9d402", file_path
    ".../staging/2026-07-12_extractor-bench-verdict_120000.md") -- see the
    task report for the live verification this was drawn from.
    """

    def test_reproduces_a_known_historical_doc_id(self) -> None:
        historical_id = "file:0cdae3d9d402"
        original_staging_path = Path(
            "/mnt/datasets/graphrag/staging/2026-07-12_extractor-bench-verdict_120000.md"
        )
        archived_path = Path(
            "/mnt/datasets/graphrag/staging/ingested/2026-07-12_extractor-bench-verdict_120000.md"
        )

        assert file_stable_id(original_staging_path) == historical_id
        assert file_stable_id(archived_path) == historical_id
