"""Directory walker — scans paths, respects globs + .memoryignore, yields Documents.

This layer is intentionally LLM-free and GPU-free.
Chunking is the responsibility of the caller (or server/index.py).
"""

from __future__ import annotations

import datetime
import fnmatch
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from hars_memory.ingest.document import (
    HEADER_DATE_UNKNOWN,
    Document,
    SourceKind,
    build_source_header,
    file_stable_id,
    infer_source_kind,
)

logger = logging.getLogger(__name__)

# Sensible hard limits — override via config, not code.
_DEFAULT_INCLUDE_GLOBS: tuple[str, ...] = ("**/*.md", "**/*.txt", "**/*.json", "**/*.py")
_DEFAULT_EXCLUDE_GLOBS: tuple[str, ...] = (
    "**/.env*",
    "**/secrets*",
    "**/*.ckpt",
    "**/*.pt",
    "**/*.safetensors",
    "**/*.bin",
    "**/*.pkl",
    "**/*.zarr",
    "**/__pycache__/**",
    "**/node_modules/**",
    "**/.git/**",
    "**/target/**",
    "**/htmlcov/**",
    "**/unsloth_compiled_cache/**",
)
_DEFAULT_IGNORE_FILE = ".memoryignore"
# Pre-2026-07-29-rename ignore-file name. If found alongside (or instead of)
# .memoryignore, its patterns are NOT loaded — silently dropping exclusion
# rules could ingest secret-bearing paths, so this is a loud warning, not a
# silent no-op. See .plans/2026-07-29_rename-to-hars-longterm-memory.md R2-2.
_LEGACY_IGNORE_FILE = ".graphragignore"
_MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MiB per file; skip larger blobs

# ---------------------------------------------------------------------------
# Attribution header (build_source_header) support
# ---------------------------------------------------------------------------
# Section label chosen from the *walked root's* directory name — this is the
# root the caller (server/index.py --paths / HARS_MEMORY_CLAUDE_MEMORY_DIR) passed in,
# not the file's own directory, so a file nested under .session/foo/bar.md
# still resolves to "session".
_SECTION_BY_ROOT_NAME: dict[str, str] = {
    ".session": "session",
    ".reports": "report",
    ".plans": "plan",
    "docs": "docs",
    "memory": "memory",
}
_SECTION_DEFAULT_CODE = "code"
_SECTION_DEFAULT_TEXT = "docs"
_GIT_LOG_TIMEOUT_SECONDS = 3.0


_FRONTMATTER_FENCE = "---"
# Bounded scan: a real YAML frontmatter block sits at the very top of the
# file. Without a cap, a document whose body happens to contain a lone
# `---` line would make us search to EOF for a closing fence that does not
# exist.
_FRONTMATTER_MAX_LINES = 200


def split_frontmatter(content: str) -> tuple[str, str]:
    """Split *content* into ``(frontmatter_block, body)``.

    ``frontmatter_block`` includes both ``---`` fences and the trailing
    newline, and is ``""`` when *content* does not open with a YAML
    frontmatter block.

    Why this exists: the attribution header used to be prepended ahead of
    everything, which put a ``[Document: ...]`` line BEFORE the opening
    ``---``. That stops the block from being frontmatter at all — every YAML
    parser (and every downstream consumer that reads ``tags:``/``date:`` out
    of it) then sees plain prose, and the fences leak into the indexed text
    as body content. The header must go *after* the closing fence instead.
    """
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != _FRONTMATTER_FENCE:
        return "", content
    for idx in range(1, min(len(lines), _FRONTMATTER_MAX_LINES)):
        if lines[idx].strip() == _FRONTMATTER_FENCE:
            block = "".join(lines[: idx + 1])
            body = "".join(lines[idx + 1 :])
            return block, body.lstrip("\n")
    return "", content


def apply_source_header(content: str, header: str) -> str:
    """Insert *header* into *content*, after any YAML frontmatter block."""
    frontmatter, body = split_frontmatter(content)
    if not frontmatter:
        return header + content
    if not frontmatter.endswith("\n"):
        frontmatter += "\n"
    return f"{frontmatter}\n{header}{body}"


def _infer_section(root: Path, kind: SourceKind) -> str:
    """Pick a ``Section`` label for the attribution header.

    Falls back to a per-SourceKind default when the walked root's directory
    name has no explicit mapping (e.g. an ad-hoc ``--paths`` root).
    """
    mapped = _SECTION_BY_ROOT_NAME.get(root.name)
    if mapped is not None:
        return mapped
    return _SECTION_DEFAULT_CODE if kind is SourceKind.PYTHON else _SECTION_DEFAULT_TEXT


def _git_commit_date(file_path: Path) -> str | None:
    """Return the last commit date (``YYYY-MM-DD``) for *file_path*, or None.

    None means "no signal from git" (untracked file, not a git repo, git not
    installed, or the lookup timed out/failed) — callers should fall back to
    filesystem mtime.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(file_path.parent),
                "log",
                "-1",
                "--format=%ad",
                "--date=short",
                "--",
                file_path.name,
            ],
            capture_output=True,
            text=True,
            timeout=_GIT_LOG_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("git log lookup failed for %s: %s", file_path, exc)
        return None
    if result.returncode != 0:
        return None
    date_str = result.stdout.strip()
    return date_str or None


def _file_date(file_path: Path, mtime: float) -> str:
    """Best-available date for a file: git last-commit date, else mtime, else 'unknown'."""
    git_date = _git_commit_date(file_path)
    if git_date:
        return git_date
    if mtime > 0:
        return datetime.date.fromtimestamp(mtime).isoformat()
    return HEADER_DATE_UNKNOWN


@dataclass
class WalkStats:
    """Counts collected during a dry-run or real walk."""

    files_seen: int = 0
    files_accepted: int = 0
    files_skipped_glob: int = 0
    files_skipped_ignore: int = 0
    files_skipped_size: int = 0
    files_skipped_binary: int = 0
    per_kind: dict[str, int] = field(default_factory=dict)

    def record(self, kind: SourceKind) -> None:
        self.files_accepted += 1
        key = kind.value
        self.per_kind[key] = self.per_kind.get(key, 0) + 1


def _warn_if_legacy_ignore_file(root: Path) -> None:
    """Loudly warn if a pre-rename .graphragignore is present at *root*.

    Its patterns are never loaded (only ignore_file, default .memoryignore,
    is read) — silently dropping exclusion rules could let secret-bearing
    paths get ingested, so this must be visible, not a silent no-op.
    """
    legacy_path = root / _LEGACY_IGNORE_FILE
    if legacy_path.exists():
        logger.warning(
            "Found legacy %s at %s — its patterns are NOT applied (renamed to "
            "%s on 2026-07-29). Rename the file to keep its exclusion rules in "
            "effect: mv %s %s",
            _LEGACY_IGNORE_FILE, legacy_path, _DEFAULT_IGNORE_FILE,
            legacy_path, root / _DEFAULT_IGNORE_FILE,
        )


def _load_ignore_patterns(root: Path, ignore_file: str) -> list[str]:
    """Load patterns from a .memoryignore file at *root*, if present."""
    _warn_if_legacy_ignore_file(root)
    ignore_path = root / ignore_file
    if not ignore_path.exists():
        return []
    patterns: list[str] = []
    for line in ignore_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            patterns.append(stripped)
    return patterns


def _matches_any(rel_path: str, patterns: tuple[str, ...] | list[str]) -> bool:
    """Return True if *rel_path* matches any of *patterns* (fnmatch-style).

    ``fnmatch.fnmatch`` does not handle ``**`` globs (that's pathlib territory).
    We strip the ``**/`` prefix and match against the relative path and the
    basename so that ``**/*.md`` matches both ``report.md`` and
    ``subdir/report.md``.
    """
    rel_name = Path(rel_path).name
    # Normalise path separators
    rel_norm = rel_path.replace("\\", "/")
    rel_parts = Path(rel_path).parts  # e.g. ("subdir", "file.md")

    for pattern in patterns:
        # Direct full-path match (works for simple patterns like "*.ckpt")
        if fnmatch.fnmatch(rel_norm, pattern):
            return True
        # Basename match for extension-only patterns (e.g. "*.md", "*.bin")
        if fnmatch.fnmatch(rel_name, pattern):
            return True

        # Directory-prefix patterns (e.g. "private/" or "private")
        # Strip trailing slash and check if any path component matches
        dir_pattern = pattern.rstrip("/")
        if dir_pattern and "/" not in dir_pattern and not any(c in dir_pattern for c in ("*", "?")):
            if dir_pattern in rel_parts:
                return True

        # Strip leading **/ and re-match against relative path and name
        stripped = pattern.lstrip("*").lstrip("/")
        if stripped and stripped != pattern:
            if fnmatch.fnmatch(rel_norm, stripped):
                return True
            if fnmatch.fnmatch(rel_name, stripped):
                return True

        # Handle **/dir/** patterns — match if any path component matches
        if "/**" in pattern or "**/" in pattern:
            parts = pattern.replace("**/", "*/").replace("/**", "/*")
            if fnmatch.fnmatch(rel_norm, parts):
                return True
            # Check if any directory component of rel_path matches the non-glob part
            raw = pattern.replace("**/", "").replace("/**", "").strip("/")
            if raw and not any(c in raw for c in ("*", "?")):
                if raw in rel_parts:
                    return True

    return False


def _is_likely_binary(path: Path, sample_bytes: int = 512) -> bool:
    """Heuristic: return True if the file looks like a binary blob."""
    try:
        with path.open("rb") as fh:
            chunk = fh.read(sample_bytes)
        return b"\x00" in chunk
    except OSError:
        return True


def walk(
    paths: list[Path],
    *,
    include_globs: tuple[str, ...] | list[str] = _DEFAULT_INCLUDE_GLOBS,
    exclude_globs: tuple[str, ...] | list[str] = _DEFAULT_EXCLUDE_GLOBS,
    ignore_file: str = _DEFAULT_IGNORE_FILE,
    dry_run: bool = False,
) -> tuple[list[Document], WalkStats]:
    """Walk *paths* and yield Document objects for every accepted file.

    Parameters
    ----------
    paths:
        List of directories (or individual files) to walk.
    include_globs:
        Only files matching at least one glob are accepted.
    exclude_globs:
        Files matching any of these globs are excluded (takes precedence).
    ignore_file:
        Name of ignore-pattern file to look for in each directory root.
    dry_run:
        When True, collect stats but return zero-content Document stubs
        (saves file I/O during verification runs).

    Returns
    -------
    (documents, stats):
        Full document list and accumulated walk statistics.
    """
    stats = WalkStats()
    docs: list[Document] = []

    for base_path in paths:
        base_path = base_path.resolve()
        if not base_path.exists():
            logger.warning("Walk path does not exist, skipping: %s", base_path)
            continue

        # Load .memoryignore from this root
        extra_excludes = _load_ignore_patterns(base_path, ignore_file)
        effective_excludes = list(exclude_globs) + extra_excludes

        is_single_file = base_path.is_file()
        candidates: list[Path] = (
            [base_path] if is_single_file else list(base_path.rglob("*"))
        )
        # When the caller passes an individual file, use its parent directory as
        # the base for relative-path and glob computation.  Without this,
        # file_path.relative_to(base_path) resolves to '.' which matches no
        # include glob and silently drops the document.
        rel_base = base_path.parent if is_single_file else base_path

        for file_path in candidates:
            if not file_path.is_file() or file_path.is_symlink():
                continue

            stats.files_seen += 1
            try:
                rel = str(file_path.relative_to(rel_base))
            except ValueError:
                rel = str(file_path)

            # Exclude globs take precedence
            if _matches_any(rel, effective_excludes):
                stats.files_skipped_glob += 1
                logger.debug("Excluded by glob: %s", file_path)
                continue

            # Must match an include glob
            if not _matches_any(rel, include_globs):
                stats.files_skipped_glob += 1
                logger.debug("Not in include globs: %s", file_path)
                continue

            # Size guard
            try:
                size = file_path.stat().st_size
            except OSError:
                size = 0
            if size > _MAX_FILE_BYTES:
                stats.files_skipped_size += 1
                logger.debug("File too large (%d bytes): %s", size, file_path)
                continue

            # Binary guard
            if _is_likely_binary(file_path):
                stats.files_skipped_binary += 1
                logger.debug("Skipped binary: %s", file_path)
                continue

            kind = infer_source_kind(file_path)
            stats.record(kind)

            if dry_run:
                content = f"[dry-run stub: {file_path}]"
            else:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    logger.warning("Cannot read %s: %s", file_path, exc)
                    continue

            doc_id = file_stable_id(file_path)
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                mtime = 0.0

            if not dry_run:
                # The attribution header goes AFTER any YAML frontmatter
                # block, never ahead of it — see `apply_source_header` /
                # `split_frontmatter` for why prepending breaks the
                # frontmatter outright.
                header = build_source_header(
                    document_name=file_path.name,
                    section=_infer_section(base_path, kind),
                    date=_file_date(file_path, mtime),
                )
                content = apply_source_header(content, header)

            docs.append(
                Document(
                    doc_id=doc_id,
                    content=content,
                    source_kind=kind,
                    source_path=str(file_path),
                    metadata={
                        "base_path": str(base_path),
                        "relative_path": rel,
                        "size_bytes": size,
                        "mtime": mtime,
                    },
                )
            )
            logger.debug("Accepted: %s (%s)", file_path, kind.value)

    logger.info(
        "Walk complete: %d seen, %d accepted, %d skipped(glob/size/binary=%d/%d/%d)",
        stats.files_seen,
        stats.files_accepted,
        stats.files_skipped_glob + stats.files_skipped_size + stats.files_skipped_binary,
        stats.files_skipped_glob,
        stats.files_skipped_size,
        stats.files_skipped_binary,
    )
    return docs, stats
