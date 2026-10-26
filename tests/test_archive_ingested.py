"""Staged notes are archived only after a successful (processed) ingest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hars_memory.ingest.document import file_stable_id
from hars_memory.scripts.archive_ingested import DocStatusError, archive_succeeded, main


def _setup(tmp_path: Path, statuses: dict[str, str | None]) -> tuple[Path, Path]:
    staging = tmp_path / "staging"
    index = tmp_path / "index"
    staging.mkdir()
    index.mkdir()
    status_store: dict[str, dict[str, str]] = {}
    for name, status in statuses.items():
        note = staging / name
        note.write_text(f"note {name}", encoding="utf-8")
        if status is not None:  # None = absent from doc_status
            status_store[file_stable_id(note)] = {"status": status}
    (index / "kv_store_doc_status.json").write_text(json.dumps(status_store), encoding="utf-8")
    return index, staging


def test_only_processed_notes_are_moved(tmp_path: Path) -> None:
    index, staging = _setup(
        tmp_path,
        {"ok.md": "processed", "bad.md": "failed", "wait.md": "pending", "never.md": None},
    )
    result = archive_succeeded(index, staging)
    assert [p.name for p in result.archived] == ["ok.md"]
    assert {p.name: s for p, s in result.left_behind.items()} == {
        "bad.md": "failed",
        "wait.md": "pending",
        "never.md": "absent",
    }
    assert (staging / "ingested" / "ok.md").exists() and not (staging / "ok.md").exists()
    for name in ("bad.md", "wait.md", "never.md"):
        assert (staging / name).exists()


def test_dry_run_moves_nothing(tmp_path: Path) -> None:
    index, staging = _setup(tmp_path, {"ok.md": "processed"})
    result = archive_succeeded(index, staging, dry_run=True)
    assert len(result.archived) == 1 and (staging / "ok.md").exists()


def test_missing_status_store_is_explicit_error_and_moves_nothing(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "a.md").write_text("x", encoding="utf-8")
    with pytest.raises(DocStatusError, match="kv_store_doc_status.json"):
        archive_succeeded(tmp_path / "no-index", staging)
    assert (staging / "a.md").exists()


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index, staging = _setup(tmp_path, {"ok.md": "processed", "bad.md": "failed"})
    assert main(["--index-dir", str(index), "--staging", str(staging)]) == 1
    assert "NOT ARCHIVED (status=failed): bad.md" in capsys.readouterr().err
    assert main(["--index-dir", str(tmp_path / "nope"), "--staging", str(staging)]) == 2
    (staging / "bad.md").unlink()
    assert main(["--index-dir", str(index), "--staging", str(staging)]) == 0
