"""Refuses to start if a configured set of legacy environment-variable
prefixes is present.

Generic by default (an empty prefix list is a no-op) — a consuming project
supplies its own forbidden prefixes via HARS_MEMORY_LEGACY_ENV_PREFIXES
(comma-separated) if it has legacy env vars from a prior rename or naming
scheme it wants to guard against. Cortex sets this to "GRAPHRAG_" (see
tools/memory-config), guarding against stale GRAPHRAG_* vars left over from
the 2026-07-29 hars-graphrag -> hars-longterm-memory rename.
"""

from __future__ import annotations

import os
import sys


def refuse_if_legacy_env(forbidden_prefixes: list[str] | None = None) -> None:
    """Exit the process if any configured forbidden env-var prefix is present.

    Args:
        forbidden_prefixes: prefixes to check for. Defaults to the
            comma-separated HARS_MEMORY_LEGACY_ENV_PREFIXES env var, or an
            empty list (no-op) if that var is unset.
    """
    if forbidden_prefixes is None:
        raw = os.environ.get("HARS_MEMORY_LEGACY_ENV_PREFIXES", "")
        forbidden_prefixes = [p for p in raw.split(",") if p]

    for prefix in forbidden_prefixes:
        offenders = sorted(k for k in os.environ if k.startswith(prefix))
        if offenders:
            print(
                f"ERROR: legacy environment variable(s) with forbidden prefix "
                f"'{prefix}' detected: {', '.join(offenders)}. "
                "Unset these before starting.",
                file=sys.stderr,
            )
            sys.exit(1)


__all__ = ["refuse_if_legacy_env"]
