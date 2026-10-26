"""Regression guard: the shipped package tree must not carry internal data.

0.7.0 shipped internal experiment identifiers (docstring examples and eval
gold-set package data), an internal host address, and operator home paths.
This test scans every file that ships in the WHEEL (the ``hars_memory/``
tree) for those patterns; the sdist-side classes (operator scripts, agent
docs, internal-ops docs) are covered by the sdist ``exclude`` list in
``pyproject.toml``.

Pattern literals are assembled from fragments so THIS file never contains a
self-match (tests ship in the sdist too).
"""

from __future__ import annotations

import re
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1] / "hars_memory"

# Each pattern is joined from fragments (see module docstring).
_FRAGMENTS: list[list[str]] = [
    ["A2", "S32"],  # internal experiment run id
    ["A2", "S28"],  # sibling run id
    ["A2", "S166"],  # sibling run id
    ["DINO", "v3"],  # internal encoder codename usage
    [r"\bV", r"EA\b"],  # project codename, standalone token
    [r"\bve", r"a\b"],  # project codename, lowercase standalone
    [r"\bVe", r"a\b"],  # project codename, capitalised standalone
    ["ve", "a_"],  # codename-prefixed identifiers / filenames
    ["_ve", "a"],  # codename-suffixed identifiers / filenames
    ["10.", "66.66.99"],  # internal host address
    ["/home/", "angk", "ira"],  # operator home path
    ["/home/", "hmast", "er"],  # operator home path
    ["index_g", "emma_v4"],  # internal index dir name
    ["index_compact_", "20260903"],  # internal index dir name
]

PATTERNS = [re.compile("".join(parts)) for parts in _FRAGMENTS]


def _shipped_files() -> list[Path]:
    return [
        path
        for path in sorted(PKG_DIR.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]


def test_no_internal_identifiers_or_hosts_in_shipped_tree() -> None:
    offenders: list[str] = []
    for path in _shipped_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary assets have no line-oriented identifiers
        for pattern in PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(
                    f"{path.relative_to(PKG_DIR)}:{line}: matched /{pattern.pattern}/"
                )
    assert not offenders, (
        "internal identifiers/hosts leaked into the shipped package tree:\n"
        + "\n".join(offenders)
    )


def test_hygiene_patterns_still_match_their_canonical_shape() -> None:
    """Guard the guard: the fragment-joined patterns must really match the
    canonical leak shapes (so a refactor cannot silently neuter them)."""
    assert PATTERNS[0].search("run " + "A2" + "S32" + " finished")
    assert PATTERNS[3].search("frozen " + "DINO" + "v3" + " tokens")
    assert PATTERNS[4].search("the " + "V" + "EA" + " project")
    assert PATTERNS[9].search("curl http://" + "10." + "66.66.99" + ":8098/healthz")
    assert PATTERNS[10].search("see " + "/home/" + "angk" + "ira" + "/x")
