"""Project metadata model for multi-project knowledge base isolation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProjectMetadata:
    """Configuration and access metadata for a knowledge base project."""

    project_id: str
    name: str = ""
    index_dir: Path = field(default_factory=lambda: Path("/tmp/hars_memory_lightrag"))
    sources_manifest: Path | str | None = None
    visibility: str = "private"  # "private" | "department" | "public" | "shared_all"
    department: str = ""
    shared_departments: list[str] = field(default_factory=list)
    owner_user_id: str = ""
    description: str = ""
    staging_dir: Path | None = None
    bm25_cache_dir: Path | None = None
    flat_dense_cache_dir: Path | None = None
    qdrant_collection: str | None = None
    qdrant_collection_prefix: str | None = None

    def __post_init__(self) -> None:
        self.project_id = self.project_id.strip()
        if not self.name:
            self.name = self.project_id
        if self.department and self.department not in self.shared_departments:
            self.shared_departments.append(self.department)
        if isinstance(self.index_dir, str):
            self.index_dir = Path(self.index_dir).expanduser().resolve()
        elif isinstance(self.index_dir, Path):
            self.index_dir = self.index_dir.expanduser().resolve()

        if self.staging_dir is None:
            # Default staging to sibling or child of index_dir
            self.staging_dir = self.index_dir.parent / f"staging_{self.project_id}" if self.project_id != "default" else self.index_dir / "staging"
        elif isinstance(self.staging_dir, str):
            self.staging_dir = Path(self.staging_dir).expanduser().resolve()

        if self.bm25_cache_dir is None:
            self.bm25_cache_dir = self.index_dir.parent / f"bm25_cache_{self.project_id}"
        elif isinstance(self.bm25_cache_dir, str):
            self.bm25_cache_dir = Path(self.bm25_cache_dir).expanduser().resolve()

        if self.flat_dense_cache_dir is None:
            self.flat_dense_cache_dir = self.index_dir.parent / f"flat_dense_cache_{self.project_id}"
        elif isinstance(self.flat_dense_cache_dir, str):
            self.flat_dense_cache_dir = Path(self.flat_dense_cache_dir).expanduser().resolve()

        self.visibility = self.visibility.strip().lower()

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_id: str = "") -> ProjectMetadata:
        project_id = str(data.get("project_id") or data.get("id") or default_id).strip()
        name = str(data.get("name") or project_id).strip()
        index_dir_raw = data.get("index_dir") or "/tmp/hars_memory_lightrag"
        index_dir = Path(str(index_dir_raw))

        staging_dir_raw = data.get("staging_dir")
        staging_dir = Path(str(staging_dir_raw)) if staging_dir_raw else None

        bm25_cache_dir_raw = data.get("bm25_cache_dir")
        bm25_cache_dir = Path(str(bm25_cache_dir_raw)) if bm25_cache_dir_raw else None

        flat_dense_cache_dir_raw = data.get("flat_dense_cache_dir")
        flat_dense_cache_dir = (
            Path(str(flat_dense_cache_dir_raw)) if flat_dense_cache_dir_raw else None
        )

        sources_manifest = data.get("sources_manifest")
        visibility = str(data.get("visibility") or "private").strip().lower()
        department = str(data.get("department") or "").strip()

        shared_depts = data.get("shared_departments") or []
        if isinstance(shared_depts, str):
            shared_depts = [d.strip() for d in shared_depts.split(",") if d.strip()]
        else:
            shared_depts = [str(d).strip() for d in shared_depts if str(d).strip()]
        if department and department not in shared_depts:
            shared_depts.append(department)

        owner_user_id = str(data.get("owner_user_id") or "").strip()
        description = str(data.get("description") or "").strip()
        qdrant_coll = data.get("qdrant_collection")
        qdrant_prefix = data.get("qdrant_collection_prefix")

        return cls(
            project_id=project_id,
            name=name,
            index_dir=index_dir,
            sources_manifest=sources_manifest,
            visibility=visibility,
            department=department,
            shared_departments=shared_depts,
            owner_user_id=owner_user_id,
            description=description,
            staging_dir=staging_dir,
            bm25_cache_dir=bm25_cache_dir,
            flat_dense_cache_dir=flat_dense_cache_dir,
            qdrant_collection=qdrant_coll,
            qdrant_collection_prefix=qdrant_prefix,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "index_dir": str(self.index_dir),
            "sources_manifest": str(self.sources_manifest) if self.sources_manifest else None,
            "visibility": self.visibility,
            "shared_departments": list(self.shared_departments),
            "owner_user_id": self.owner_user_id,
            "description": self.description,
            "staging_dir": str(self.staging_dir) if self.staging_dir else None,
            "bm25_cache_dir": str(self.bm25_cache_dir) if self.bm25_cache_dir else None,
            "flat_dense_cache_dir": (
                str(self.flat_dense_cache_dir) if self.flat_dense_cache_dir else None
            ),
            "qdrant_collection": self.qdrant_collection,
        }
