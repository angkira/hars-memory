"""Multi-project support for HARS long-term memory."""

from __future__ import annotations

from hars_memory.projects.models import ProjectMetadata
from hars_memory.projects.registry import (
    DEFAULT_PROJECTS_DIR,
    PROJECTS_CONFIG_ENV,
    PROJECTS_DIR_ENV,
    ProjectRegistry,
    get_default_project_registry,
)

__all__ = [
    "DEFAULT_PROJECTS_DIR",
    "PROJECTS_CONFIG_ENV",
    "PROJECTS_DIR_ENV",
    "ProjectMetadata",
    "ProjectRegistry",
    "get_default_project_registry",
]
