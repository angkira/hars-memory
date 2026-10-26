"""Registry for managing multi-project knowledge base configurations and instances."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Final

import yaml

from hars_memory.auth.models import TokenContext
from hars_memory.auth.policy import check_permission
from hars_memory.projects.models import ProjectMetadata

logger = logging.getLogger(__name__)

DEFAULT_PROJECTS_DIR: Final[Path] = (
    Path.home() / ".local" / "share" / "hars-longterm-memory" / "projects"
)
PROJECTS_CONFIG_ENV: Final[str] = "HARS_MEMORY_PROJECTS_CONFIG"
PROJECTS_DIR_ENV: Final[str] = "HARS_MEMORY_PROJECTS_DIR"


class ProjectRegistry:
    """Manages project definitions, isolation boundaries, and RAG/Graph pools."""

    def __init__(self, auto_load: bool = True) -> None:
        self._projects: dict[str, ProjectMetadata] = {}
        self._rag_pool: dict[str, Any] = {}
        self._rag_lock = asyncio.Lock()
        self._graph_pool: dict[str, tuple[Any, float]] = {}
        self._graph_lock = asyncio.Lock()
        self._bm25_pool: dict[str, tuple[Any, float]] = {}
        self._bm25_lock = asyncio.Lock()
        if auto_load:
            self.reload()

    def clear_caches(self) -> None:
        """Reset cached RAG, GraphML and BM25 instances."""
        self._rag_pool.clear()
        self._graph_pool.clear()
        self._bm25_pool.clear()

    def register_project(self, project: ProjectMetadata) -> None:
        """Register or update a project's metadata."""
        self._projects[project.project_id] = project

    def unregister_project(self, project_id: str) -> bool:
        """Remove a project from the registry."""
        self._rag_pool.pop(project_id, None)
        self._graph_pool.pop(project_id, None)
        self._bm25_pool.pop(project_id, None)
        return self._projects.pop(project_id, None) is not None

    def _get_default_project(self) -> ProjectMetadata:
        """Dynamically compute the default project from current environment variables."""
        index_dir = os.environ.get("HARS_MEMORY_INDEX_DIR", "/tmp/hars_memory_lightrag")
        staging_dir = os.environ.get("HARS_MEMORY_STAGING_DIR")
        sources_manifest = os.environ.get("HARS_MEMORY_SOURCES_MANIFEST")
        bm25_cache_dir = os.environ.get("HARS_MEMORY_BM25_CACHE_DIR")
        flat_dense_cache_dir = os.environ.get("HARS_MEMORY_FLAT_DENSE_CACHE_DIR")
        qdrant_coll = os.environ.get("HARS_MEMORY_QDRANT_COLLECTION")

        return ProjectMetadata(
            project_id="default",
            name="Default Project",
            index_dir=Path(index_dir),
            sources_manifest=sources_manifest,
            visibility="public",
            owner_user_id="system",
            description="Default HARS knowledge base project",
            staging_dir=Path(staging_dir) if staging_dir else None,
            bm25_cache_dir=Path(bm25_cache_dir) if bm25_cache_dir else None,
            flat_dense_cache_dir=Path(flat_dense_cache_dir) if flat_dense_cache_dir else None,
            qdrant_collection=qdrant_coll,
        )

    def get_project(self, project_id: str = "default") -> ProjectMetadata | None:
        """Get project metadata by ID, falling back to env-configured default project."""
        p_id = (project_id or "default").strip()
        if p_id in self._projects:
            return self._projects[p_id]
        if p_id == "default":
            return self._get_default_project()
        return None

    def list_projects(self, department: str | None = None) -> list[ProjectMetadata]:
        """Return all registered projects (including the default project), optionally filtered by department."""
        result = list(self._projects.values())
        if not any(p.project_id == "default" for p in result):
            result.insert(0, self._get_default_project())
        if department:
            dept_lower = department.strip().lower()
            return [
                p
                for p in result
                if p.department.lower() == dept_lower
                or dept_lower in [d.lower() for d in p.shared_departments]
            ]
        return result

    def list_projects_for_context(
        self, context: TokenContext | None, action: str = "read"
    ) -> list[ProjectMetadata]:
        """Return projects accessible to the caller context for the specified action."""
        accessible: list[ProjectMetadata] = []
        for project in self.list_projects():
            if check_permission(context, project.project_id, action, project):
                accessible.append(project)
        return accessible

    def reload(self) -> None:
        """Reload project definitions from configuration and directory discovery."""
        self._projects.clear()
        self.clear_caches()

        # 1. Load from HARS_MEMORY_PROJECTS_CONFIG env
        env_config = os.environ.get(PROJECTS_CONFIG_ENV, "").strip()
        if env_config:
            self._load_from_env_config(env_config)

        # 2. Discover from HARS_MEMORY_PROJECTS_DIR or default projects directory
        projects_dir_raw = os.environ.get(PROJECTS_DIR_ENV, "").strip()
        projects_dir = Path(projects_dir_raw).expanduser() if projects_dir_raw else DEFAULT_PROJECTS_DIR
        if projects_dir.is_dir():
            self._discover_projects_dir(projects_dir)

    def _load_from_env_config(self, content: str) -> None:
        candidate_path = Path(content).expanduser()
        if candidate_path.is_file():
            try:
                data = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
                self._parse_and_register(data)
            except Exception as exc:
                logger.warning("Failed to load projects config from %s: %s", candidate_path, exc)
            return

        try:
            data = yaml.safe_load(content)
            self._parse_and_register(data)
        except Exception as exc:
            logger.warning("Failed to parse %s content: %s", PROJECTS_CONFIG_ENV, exc)

    def _discover_projects_dir(self, projects_dir: Path) -> None:
        try:
            for item in sorted(projects_dir.iterdir()):
                if not item.is_dir():
                    continue
                # Check for explicit config file inside subdirectory
                json_cfg = item / "project.json"
                yaml_cfg = item / "project.yaml"
                yml_cfg = item / "project.yml"

                cfg_file = next((f for f in (json_cfg, yaml_cfg, yml_cfg) if f.is_file()), None)
                if cfg_file:
                    try:
                        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
                        if isinstance(cfg, dict):
                            project = ProjectMetadata.from_dict(cfg, default_id=item.name)
                            self.register_project(project)
                            continue
                    except Exception as exc:
                        logger.warning("Failed to parse project file %s: %s", cfg_file, exc)

                # Default directory discovery convention
                project = ProjectMetadata(
                    project_id=item.name,
                    name=item.name,
                    index_dir=item / "index",
                    staging_dir=item / "staging",
                    visibility="private",
                    description=f"Auto-discovered project {item.name}",
                )
                self.register_project(project)
        except Exception as exc:
            logger.warning("Error discovering projects in %s: %s", projects_dir, exc)

    def _parse_and_register(self, data: Any) -> None:
        if not data:
            return

        if isinstance(data, dict):
            if "projects" in data and isinstance(data["projects"], list):
                for item in data["projects"]:
                    if isinstance(item, dict):
                        p = ProjectMetadata.from_dict(item)
                        if p.project_id:
                            self.register_project(p)
            else:
                for key, val in data.items():
                    if isinstance(val, dict):
                        p = ProjectMetadata.from_dict(val, default_id=key)
                        if p.project_id:
                            self.register_project(p)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    p = ProjectMetadata.from_dict(item)
                    if p.project_id:
                        self.register_project(p)

    async def get_rag(self, project_id: str = "default") -> Any:
        """Obtain or initialize the LightRAG instance isolated for project_id."""
        p_id = (project_id or "default").strip()
        async with self._rag_lock:
            if p_id in self._rag_pool:
                return self._rag_pool[p_id]

            project = self.get_project(p_id)
            if project is None:
                raise ValueError(f"Project '{p_id}' is not registered")

            base_coll = os.environ.get("HARS_MEMORY_QDRANT_COLLECTION", "hars_longterm_memory")
            if project.qdrant_collection:
                coll = project.qdrant_collection
            elif project.project_id == "default":
                coll = base_coll
            else:
                coll = f"{base_coll}_{project.project_id}"

            from hars_memory.server.lightrag_init import create_lightrag

            rag = create_lightrag(
                working_dir=str(project.index_dir),
                qdrant_collection=coll,
            )
            await rag.initialize_storages()  # type: ignore[attr-defined]
            self._rag_pool[p_id] = rag
            return rag

    async def get_graph(self, project_id: str = "default") -> tuple[Any, str]:
        """Return (cached NetworkX graph, cache_status) for project_id.

        Raises FileNotFoundError if graph_chunk_entity_relation.graphml does not exist.
        """
        p_id = (project_id or "default").strip()
        project = self.get_project(p_id)
        if project is None:
            raise ValueError(f"Project '{p_id}' is not registered")

        graph_file = Path(project.index_dir) / "graph_chunk_entity_relation.graphml"
        if not graph_file.exists():
            raise FileNotFoundError(
                f"Index not built yet for project '{p_id}': {graph_file}"
            )

        mtime = graph_file.stat().st_mtime
        async with self._graph_lock:
            cache_status = "hit"
            cached = self._graph_pool.get(p_id)
            if cached is None or cached[1] != mtime:
                import networkx as nx  # type: ignore[import-not-found]

                logger.info(
                    "Loading GraphML index into cache for project '%s': %s", p_id, graph_file
                )
                graph = await asyncio.to_thread(nx.read_graphml, str(graph_file))
                self._graph_pool[p_id] = (graph, mtime)
                cache_status = "rebuild"
            else:
                graph = cached[0]
            return graph, cache_status

    async def get_bm25_index(self, project_id: str = "default") -> tuple[Any, str]:
        """Return (cached BM25SparseIndex, cache_status) for project_id."""
        p_id = (project_id or "default").strip()
        project = self.get_project(p_id)
        if project is None:
            raise ValueError(f"Project '{p_id}' is not registered")

        from hars_memory.retrieval.bm25_index import (
            CHUNKS_FILENAME,
            BM25IndexUnavailableError,
            get_or_build_index,
        )

        chunks_file = Path(project.index_dir) / CHUNKS_FILENAME
        if not chunks_file.exists():
            raise BM25IndexUnavailableError(
                f"No text-chunk store at {chunks_file} for project '{p_id}'"
            )

        mtime = chunks_file.stat().st_mtime
        async with self._bm25_lock:
            cache_status = "hit"
            cached = self._bm25_pool.get(p_id)
            if cached is None or cached[1] != mtime:
                cache_dir = project.bm25_cache_dir or (
                    Path(project.index_dir).parent / f"bm25_cache_{project.project_id}"
                )
                bm25_idx, _ = await asyncio.to_thread(
                    get_or_build_index, str(project.index_dir), str(cache_dir)
                )
                self._bm25_pool[p_id] = (bm25_idx, mtime)
                cache_status = "rebuild"
            else:
                bm25_idx = cached[0]
            return bm25_idx, cache_status

    def invalidate_caches(self, project_id: str) -> None:
        """Invalidate in-memory caches (graph, bm25, rag) for a specific project."""
        p_id = (project_id or "default").strip()
        self._graph_pool.pop(p_id, None)
        self._bm25_pool.pop(p_id, None)
        self._rag_pool.pop(p_id, None)

    def invalidate_all_caches(self) -> None:
        """Invalidate in-memory caches across all projects."""
        self._graph_pool.clear()
        self._bm25_pool.clear()
        self._rag_pool.clear()


_default_registry: ProjectRegistry | None = None


def get_default_project_registry() -> ProjectRegistry:
    """Return the global default ProjectRegistry singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ProjectRegistry()
    return _default_registry


def reset_default_project_registry() -> None:
    """Reset the global default ProjectRegistry singleton."""
    global _default_registry
    _default_registry = None
