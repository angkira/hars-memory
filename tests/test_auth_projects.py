"""Tests for multi-project registry, token authorization, RBAC and department sharing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hars_memory.auth.models import LOCAL_SUPERUSER, AccessToken, TokenContext
from hars_memory.auth.policy import check_permission, is_auth_enabled
from hars_memory.auth.store import TokenStore
from hars_memory.projects.models import ProjectMetadata
from hars_memory.projects.registry import ProjectRegistry


class TestAuthStoreAndPolicy:
    def test_local_fallback_when_auth_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARS_MEMORY_AUTH_ENABLED", "0")
        assert is_auth_enabled() is False

        # Access unconditionally granted when auth is disabled
        assert check_permission(None, "any_project", "admin") is True
        assert check_permission(LOCAL_SUPERUSER, "any_project", "write") is True

    def test_authenticate_and_check_permission(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARS_MEMORY_AUTH_ENABLED", "1")
        assert is_auth_enabled() is True

        tokens = [
            AccessToken(
                token="sec-dev-123",
                user_id="alice",
                departments=["ai-lab"],
                roles=["developer"],
                permissions=["proj-a:write", "proj-b:read"],
            ),
            AccessToken(
                token="sec-guest-456",
                user_id="bob",
                departments=["marketing"],
                roles=["viewer"],
                permissions=["public:*"],
            ),
        ]
        store = TokenStore(tokens=tokens)

        # Unauthenticated / invalid token
        assert store.authenticate("") is None
        assert store.authenticate("wrong-token") is None
        assert check_permission(None, "proj-a", "read") is False

        # Authenticated Alice
        alice = store.authenticate("sec-dev-123")
        assert alice is not None
        assert alice.user_id == "alice"
        assert alice.departments == ["ai-lab"]

        proj_a = ProjectMetadata(project_id="proj-a", name="A", department="ai-lab")
        proj_b = ProjectMetadata(project_id="proj-b", name="B", department="core")
        proj_c = ProjectMetadata(project_id="proj-c", name="C", department="finance")

        # Alice has proj-a:write which covers read & write
        assert check_permission(alice, "proj-a", "write", proj_a) is True
        assert check_permission(alice, "proj-a", "read", proj_a) is True
        assert check_permission(alice, "proj-a", "admin", proj_a) is False

        # Alice has proj-b:read (read only)
        assert check_permission(alice, "proj-b", "read", proj_b) is True
        assert check_permission(alice, "proj-b", "write", proj_b) is False

        # Alice cannot access proj-c
        assert check_permission(alice, "proj-c", "read", proj_c) is False

    def test_department_sharing_and_public_visibility(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARS_MEMORY_AUTH_ENABLED", "1")

        token = AccessToken(
            token="sec-infra-user",
            user_id="charlie",
            departments=["infrastructure"],
            roles=["engineer"],
            permissions=[],
        )
        ctx = TokenContext.from_access_token(token)

        shared_proj = ProjectMetadata(
            project_id="k8s-mon",
            name="K8s Monitoring",
            department="infrastructure",
            shared_departments=["infrastructure", "sre"],
        )
        public_proj = ProjectMetadata(
            project_id="handbook",
            name="Company Handbook",
            department="hr",
            visibility="public",
        )
        secret_proj = ProjectMetadata(
            project_id="salary-db",
            name="Salaries",
            department="finance",
            visibility="private",
        )

        # Access allowed via department sharing (read and write)
        assert check_permission(ctx, "k8s-mon", "read", shared_proj) is True
        assert check_permission(ctx, "k8s-mon", "write", shared_proj) is True
        # But not admin
        assert check_permission(ctx, "k8s-mon", "admin", shared_proj) is False

        # Public project allows read for any authenticated user
        assert check_permission(ctx, "handbook", "read", public_proj) is True
        # But not write
        assert check_permission(ctx, "handbook", "write", public_proj) is False

        # Secret project denied
        assert check_permission(ctx, "salary-db", "read", secret_proj) is False


class TestProjectRegistry:
    def test_register_and_list_projects(self, tmp_path: Path) -> None:
        reg = ProjectRegistry()
        p1 = ProjectMetadata(
            project_id="frontend",
            name="Frontend App",
            index_dir=tmp_path / "frontend_idx",
            department="ui-team",
        )
        p2 = ProjectMetadata(
            project_id="backend",
            name="Backend API",
            index_dir=tmp_path / "backend_idx",
            department="core-team",
        )
        reg.register_project(p1)
        reg.register_project(p2)

        assert reg.get_project("frontend") == p1
        assert reg.get_project("backend") == p2
        assert reg.get_project("unknown") is None

        # Filter by department
        ui_projects = reg.list_projects(department="ui-team")
        assert any(p.project_id == "frontend" for p in ui_projects)
        assert not any(p.project_id == "backend" for p in ui_projects)

        # Invalidation works cleanly
        reg.invalidate_caches("frontend")
        reg.invalidate_all_caches()


class TestMcpServerAuthAndProjects:
    def test_list_projects_tool(self, tmp_path: Path) -> None:
        import hars_memory.mcp_server as mcp
        from hars_memory.projects.registry import get_default_project_registry

        reg = get_default_project_registry()
        reg.register_project(
            ProjectMetadata(
                project_id="proj_alpha",
                name="Project Alpha",
                index_dir=tmp_path / "alpha",
                department="research",
            )
        )

        res = asyncio.run(mcp.call_tool("memory_list_projects", {}))
        data = json.loads(res[0].text)
        assert data["ok"] is True
        project_ids = [p["project_id"] for p in data["projects"]]
        assert "default" in project_ids
        assert "proj_alpha" in project_ids

    def test_auth_rejection_when_enabled(self, tmp_path: Path) -> None:
        from hars_memory.auth.store import reset_default_token_store
        from tests.test_mcp_server import _load_mcp_module_with_env

        tokens_file = tmp_path / "tokens.json"
        tokens_file.write_text(
            json.dumps([
                {
                    "token": "valid-token-123",
                    "user_id": "valid_user",
                    "departments": ["core"],
                    "roles": ["admin"],
                    "permissions": ["*"],
                }
            ]),
            encoding="utf-8",
        )

        reset_default_token_store()
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_AUTH_ENABLED": "1",
            "HARS_MEMORY_TOKENS_CONFIG": str(tokens_file),
        })

        try:
            # Invalid token -> rejected
            res_bad = asyncio.run(
                mod.call_tool("memory_status", {"access_token": "wrong-token"})
            )
            data_bad = json.loads(res_bad[0].text)
            assert data_bad["ok"] is False
            assert data_bad["error_type"] in ("UnauthorizedError", "ForbiddenError")

            # Valid token -> authorized
            res_good = asyncio.run(
                mod.call_tool("memory_status", {"access_token": "valid-token-123"})
            )
            data_good = json.loads(res_good[0].text)
            assert data_good.get("project") == "default" or "index_exists" in data_good
        finally:
            reset_default_token_store()

    def test_project_staging_isolation(self, tmp_path: Path) -> None:
        from hars_memory.projects import ProjectMetadata, get_default_project_registry
        from tests.test_mcp_server import _load_mcp_module_with_env

        projects_root = tmp_path / "projects"
        default_index = tmp_path / "default_index"
        default_staging = tmp_path / "default_staging"
        default_index.mkdir(parents=True)
        default_staging.mkdir(parents=True)

        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(default_index),
            "HARS_MEMORY_STAGING_DIR": str(default_staging),
            "HARS_MEMORY_PROJECTS_DIR": str(projects_root),
            "HARS_MEMORY_AUTH_ENABLED": "0",
        })

        # Register two separate projects in the registry
        reg = get_default_project_registry()
        reg.register_project(
            ProjectMetadata(
                project_id="alpha",
                name="Alpha Project",
                staging_dir=projects_root / "alpha" / "staging",
            )
        )
        reg.register_project(
            ProjectMetadata(
                project_id="beta",
                name="Beta Project",
                staging_dir=projects_root / "beta" / "staging",
            )
        )

        # Remember note in alpha
        res_rem = asyncio.run(
            mod.call_tool("memory_remember", {
                "project": "alpha",
                "title": "Secret in Alpha",
                "content": "Alpha specific data",
                "source": "unit-test",
            })
        )
        assert json.loads(res_rem[0].text)["ok"] is True

        alpha_staging = projects_root / "alpha" / "staging"
        beta_staging = projects_root / "beta" / "staging"

        alpha_files = list(alpha_staging.glob("*.md"))
        beta_files = list(beta_staging.glob("*.md"))
        default_files = list(default_staging.glob("*.md"))

        assert len(alpha_files) == 1
        assert len(beta_files) == 0
        assert len(default_files) == 0
        assert "Alpha specific data" in alpha_files[0].read_text(encoding="utf-8")

    def test_rbac_write_permission_enforcement(self, tmp_path: Path) -> None:
        from hars_memory.auth.store import reset_default_token_store
        from hars_memory.projects import ProjectMetadata, get_default_project_registry
        from tests.test_mcp_server import _load_mcp_module_with_env

        projects_root = tmp_path / "projects"
        tokens_file = tmp_path / "tokens.json"
        tokens_file.write_text(
            json.dumps([
                {
                    "token": "viewer-tok",
                    "user_id": "viewer_user",
                    "departments": ["analytics"],
                    "roles": ["viewer"],
                    "permissions": ["alpha:read"],
                },
                {
                    "token": "writer-tok",
                    "user_id": "writer_user",
                    "departments": ["analytics"],
                    "roles": ["developer"],
                    "permissions": ["alpha:write"],
                },
            ]),
            encoding="utf-8",
        )

        reset_default_token_store()
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_AUTH_ENABLED": "1",
            "HARS_MEMORY_TOKENS_CONFIG": str(tokens_file),
            "HARS_MEMORY_PROJECTS_DIR": str(projects_root),
        })

        try:
            # Register project alpha with admin token or directly
            reg = get_default_project_registry()
            reg.register_project(
                ProjectMetadata(
                    project_id="alpha",
                    name="Alpha",
                    department="analytics",
                )
            )

            # Viewer attempts to write note -> rejected with 403 ForbiddenError
            res_view = asyncio.run(
                mod.call_tool("memory_remember", {
                    "project": "alpha",
                    "title": "Unauthorized Note",
                    "content": "Should fail",
                    "access_token": "viewer-tok",
                })
            )
            data_view = json.loads(res_view[0].text)
            assert data_view["ok"] is False
            assert data_view["code"] == 403
            assert data_view["error_type"] == "ForbiddenError"

            # Writer attempts to write note -> succeeds
            res_write = asyncio.run(
                mod.call_tool("memory_remember", {
                    "project": "alpha",
                    "title": "Authorized Note",
                    "content": "Should succeed",
                    "access_token": "writer-tok",
                })
            )
            data_write = json.loads(res_write[0].text)
            assert data_write["ok"] is True
        finally:
            reset_default_token_store()
