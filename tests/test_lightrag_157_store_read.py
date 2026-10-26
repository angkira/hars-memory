"""Decisive proof that lightrag-hku 1.5.7 reads a 1.5.7-built store.

The Phase-D index swap serves `index_full_mdstruct_20261014`, which was BUILT by
lightrag-hku 1.5.7.  The old 1.4.16 serving reader could not read it:

* ``DocProcessingStatus(**row)`` raised ``TypeError: unexpected keyword
  argument 'content_hash'`` on **1050 / 1340** ``kv_store_doc_status.json`` rows;
* ``get_status_counts()`` raised ``KeyError: 'parsing'`` (the ``parsing`` /
  ``analyzing`` enum values are unknown to 1.4.16).

This test loads the *actual* 2026-10-17 snapshot with the *installed* reader
and asserts that every row constructs and the status counts are complete.  It
is the migration's acceptance test for "the upgrade unblocks the swap".

The snapshot is an external artifact (not part of the repo), so the test
**skips** when it is absent — the CI suite stays green on hosts without it.  It
runs in a subprocess so the snapshot storage never leaks into the shared
LightRAG namespace used by the rest of the suite.

Point it at another copy with ``HARS_LIGHTRAG_157_SNAPSHOT=<dir containing
kv_store_doc_status.json>``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Expected cluster for the 2026-10-17 in-flight snapshot (see the snapshot's
# VM_COUNTS.json).  Kept explicit so a schema drift that silently drops rows
# (e.g. the old 1.4.16 per-row ``except TypeError: continue``) fails the test
# instead of passing vacuously.
_EXPECTED_TOTAL = 1340
_EXPECTED_COUNTS = {"processed": 958, "failed": 290, "parsing": 75, "pending": 17}

_ENV_VAR = "HARS_LIGHTRAG_157_SNAPSHOT"

_CANDIDATE_SNAPSHOTS = (
    Path("/home/demo/mem-probe/scratch_157/index_full_mdstruct_20261014"),
    Path(
        "/home/demo/mem-probe/index_snaps/20261017T132719Z/"
        "index_full_mdstruct_20261014"
    ),
)


def _snapshot_dir() -> Path | None:
    env = os.environ.get(_ENV_VAR)
    candidates = [Path(env).expanduser()] if env else list(_CANDIDATE_SNAPSHOTS)
    for candidate in candidates:
        if (candidate / "kv_store_doc_status.json").is_file():
            return candidate
    return None


# Runs in a subprocess (isolated shared-storage namespace); prints one JSON line.
_PROOF_SNIPPET = textwrap.dedent(
    """
    import asyncio, json, os, sys

    from lightrag.base import DocProcessingStatus, DocStatus
    from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
    from lightrag.kg.shared_storage import initialize_share_data

    snap = sys.argv[1]
    with open(os.path.join(snap, "kv_store_doc_status.json"), encoding="utf-8") as fh:
        rows = json.load(fh)

    failures = []
    for key, row in rows.items():
        try:
            DocProcessingStatus(**row)
        except Exception as exc:  # noqa: BLE001 - the point is to catch ANY failure
            failures.append({"id": key, "error": f"{type(exc).__name__}: {exc}"})

    known_statuses = {s.value for s in DocStatus}

    async def _counts():
        initialize_share_data(workers=1)
        storage = JsonDocStatusStorage(
            namespace="doc_status",
            workspace="",
            global_config={"working_dir": snap},
            embedding_func=None,
        )
        await storage.initialize()
        return await storage.get_status_counts()

    counts = asyncio.run(_counts())
    print(json.dumps({
        "total": len(rows),
        "failures": len(failures),
        "sample": failures[:3],
        "counts": counts,
        "known_statuses": sorted(known_statuses),
    }))
    """
)


@pytest.mark.skipif(
    _snapshot_dir() is None,
    reason=(
        "1.5.7 index snapshot not present (external artifact); set "
        f"{_ENV_VAR} to run this proof"
    ),
)
def test_157_reader_reads_the_157_built_store() -> None:
    snapshot = _snapshot_dir()
    assert snapshot is not None  # narrowed by skipif

    proc = subprocess.run(
        [sys.executable, "-c", _PROOF_SNIPPET, str(snapshot)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"proof subprocess failed:\n{proc.stderr}"
    report = json.loads(proc.stdout.strip().splitlines()[-1])

    # Decisive: every row constructs (0 TypeError on content_hash et al.).
    assert report["total"] == _EXPECTED_TOTAL
    assert report["failures"] == 0, report["sample"]

    # Decisive: get_status_counts() does not raise, and no status is unknown.
    counts = report["counts"]
    assert set(_EXPECTED_COUNTS) <= set(report["known_statuses"])
    for status, expected in _EXPECTED_COUNTS.items():
        assert counts.get(status) == expected, (status, counts)
    assert sum(counts.values()) == _EXPECTED_TOTAL
