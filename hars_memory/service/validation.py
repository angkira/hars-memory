"""Validation and bounded reading for uploaded text documents."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final, Iterable

from fastapi import UploadFile

ALLOWED_SUFFIXES: Final[frozenset[str]] = frozenset({".md", ".txt", ".json", ".py"})
DEFAULT_MAX_FILES: Final[int] = 100
DEFAULT_MAX_FILE_BYTES: Final[int] = 2 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES: Final[int] = 20 * 1024 * 1024


class UploadValidationError(ValueError):
    """An upload is unsafe or outside configured bounds."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class UploadLimits:
    max_files: int = DEFAULT_MAX_FILES
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES

    @classmethod
    def from_env(cls) -> UploadLimits:
        return cls(
            max_files=int(os.environ.get("HARS_MEMORY_MAX_UPLOAD_FILES", DEFAULT_MAX_FILES)),
            max_file_bytes=int(
                os.environ.get("HARS_MEMORY_MAX_UPLOAD_FILE_BYTES", DEFAULT_MAX_FILE_BYTES)
            ),
            max_total_bytes=int(
                os.environ.get("HARS_MEMORY_MAX_UPLOAD_TOTAL_BYTES", DEFAULT_MAX_TOTAL_BYTES)
            ),
        )


def safe_relative_name(raw_name: str | None) -> str:
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise UploadValidationError("unsafe_filename", "Upload filename is unsafe")
    path = PurePosixPath(raw_name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise UploadValidationError("unsafe_filename", "Upload filename is unsafe")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise UploadValidationError(
            "unsupported_file_type",
            f"Unsupported file type for {raw_name!r}; allowed: {sorted(ALLOWED_SUFFIXES)}",
        )
    return path.as_posix()


def validate_text(name: str, content: bytes) -> None:
    if not content:
        raise UploadValidationError("empty_file", f"Uploaded file {name!r} is empty")
    if b"\x00" in content:
        raise UploadValidationError("binary_file", f"Uploaded file {name!r} is binary")
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UploadValidationError(
            "invalid_utf8", f"Uploaded file {name!r} is not valid UTF-8"
        ) from exc
    if not decoded.strip():
        raise UploadValidationError(
            "empty_file", f"Uploaded file {name!r} contains no text"
        )


async def read_uploads(
    files: Iterable[UploadFile], limits: UploadLimits
) -> dict[str, bytes]:
    uploads = list(files)
    if not uploads:
        raise UploadValidationError("no_files", "At least one text file is required")
    if len(uploads) > limits.max_files:
        raise UploadValidationError(
            "too_many_files", f"At most {limits.max_files} files may be uploaded"
        )

    result: dict[str, bytes] = {}
    total = 0
    for upload in uploads:
        name = safe_relative_name(upload.filename)
        if name in result:
            raise UploadValidationError("duplicate_filename", f"Duplicate filename: {name}")
        content = await upload.read(limits.max_file_bytes + 1)
        if len(content) > limits.max_file_bytes:
            raise UploadValidationError(
                "file_too_large", f"Uploaded file {name!r} exceeds the per-file limit"
            )
        total += len(content)
        if total > limits.max_total_bytes:
            raise UploadValidationError(
                "upload_too_large", "Uploaded files exceed the total request limit"
            )
        validate_text(name, content)
        result[name] = content
    return result


__all__ = [
    "ALLOWED_SUFFIXES",
    "UploadLimits",
    "UploadValidationError",
    "read_uploads",
    "safe_relative_name",
    "validate_text",
]
