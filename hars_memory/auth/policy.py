"""Permission checking and access control policies."""

from __future__ import annotations

import logging
import os
from typing import Any, Final

from hars_memory.auth.models import LOCAL_SUPERUSER, TokenContext

logger = logging.getLogger(__name__)

AUTH_ENABLED_ENV: Final[str] = "HARS_MEMORY_AUTH_ENABLED"


def is_auth_enabled() -> bool:
    """Return True if token-based authentication is explicitly enabled."""
    return os.environ.get(AUTH_ENABLED_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def check_permission(
    context: TokenContext | None,
    project: str,
    action: str,
    project_metadata: Any = None,
    auth_enabled: bool | None = None,
) -> bool:
    """Evaluate whether the given context has permission to perform action on project.

    Rules:
    1. LOCAL FALLBACK: If authorization is not enabled (HARS_MEMORY_AUTH_ENABLED=0 or unset)
       or context is LOCAL_SUPERUSER, access is unconditionally granted.
    2. Missing context (unauthenticated caller) fails closed when auth is enabled.
    3. Global admin role or superuser permission ('*', '*:*') grants all actions.
    4. Project owner (project_metadata.owner_user_id == context.user_id) has full access.
    5. Token scopes: checks '{project}:{action}', '{project}:*', '*:{action}', '*'.
       Action hierarchy applies: 'admin' covers read/write/admin, 'write' covers read/write.
    6. Department sharing: if user belongs to project_metadata.shared_departments,
       'read' and 'write' actions are granted (unless user has explicit 'viewer' role, which restricts to 'read').
    7. Public projects (visibility in 'public', 'shared_all'): 'read' is allowed for any valid token.
    """
    # 1. Local fallback / superuser bypass
    enabled = auth_enabled if auth_enabled is not None else is_auth_enabled()
    if not enabled:
        return True

    if context is not None and (context is LOCAL_SUPERUSER or getattr(context, "is_superuser", False)):
        return True

    # 2. Unauthenticated caller
    if context is None:
        return False

    act = action.strip().lower()
    proj = project.strip()

    # 3. Global admin roles
    user_roles = [r.lower() for r in context.roles]
    if "admin" in user_roles:
        return True

    # 4. Project owner full access
    if project_metadata is not None:
        owner = getattr(project_metadata, "owner_user_id", "")
        if owner and owner == context.user_id:
            return True

    # 5. Token granular permission scopes
    for perm in context.permissions:
        perm_clean = perm.strip()
        if not perm_clean:
            continue
        if perm_clean in ("*", "*:*"):
            return True

        if ":" in perm_clean:
            scope_proj, scope_act = perm_clean.split(":", 1)
            scope_proj = scope_proj.strip()
            scope_act = scope_act.strip().lower()
        else:
            scope_proj = perm_clean
            scope_act = "*"

        if scope_proj in ("*", proj):
            if scope_act in ("*", act):
                return True
            if scope_act == "admin" and act in ("read", "write", "admin"):
                return True
            if scope_act == "write" and act in ("read", "write"):
                return True

    # 6. Department sharing
    if project_metadata is not None:
        shared_depts = getattr(project_metadata, "shared_departments", []) or []
        if shared_depts:
            user_depts = {d.lower() for d in (context.departments + context.groups)}
            proj_depts = {d.lower() for d in shared_depts}
            if user_depts.intersection(proj_depts):
                if "viewer" in user_roles:
                    if act == "read":
                        return True
                elif act in ("read", "write"):
                    return True

    # 7. Public visibility (read-only for all valid users)
    if project_metadata is not None:
        visibility = str(getattr(project_metadata, "visibility", "private")).strip().lower()
        if visibility in ("public", "shared_all"):
            if act == "read":
                return True

    return False
