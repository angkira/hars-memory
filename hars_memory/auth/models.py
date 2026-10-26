"""Data models for authentication, tokens, and token context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AccessToken:
    """Represents an API access token with assigned identity, roles, and scopes."""

    token: str
    user_id: str
    departments: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_token: str = "") -> AccessToken:
        token = str(data.get("token") or default_token).strip()
        user_id = str(data.get("user_id") or data.get("id") or "anonymous").strip()

        departments = data.get("departments") or []
        if isinstance(departments, str):
            departments = [d.strip() for d in departments.split(",") if d.strip()]
        else:
            departments = [str(d).strip() for d in departments if str(d).strip()]

        groups = data.get("groups") or []
        if isinstance(groups, str):
            groups = [g.strip() for g in groups.split(",") if g.strip()]
        else:
            groups = [str(g).strip() for g in groups if str(g).strip()]

        roles = data.get("roles") or []
        if isinstance(roles, str):
            roles = [r.strip() for r in roles.split(",") if r.strip()]
        else:
            roles = [str(r).strip() for r in roles if str(r).strip()]

        permissions = data.get("permissions") or []
        if isinstance(permissions, str):
            permissions = [p.strip() for p in permissions.split(",") if p.strip()]
        else:
            permissions = [str(p).strip() for p in permissions if str(p).strip()]

        return cls(
            token=token,
            user_id=user_id,
            departments=departments,
            groups=groups,
            roles=roles,
            permissions=permissions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "user_id": self.user_id,
            "departments": list(self.departments),
            "groups": list(self.groups),
            "roles": list(self.roles),
            "permissions": list(self.permissions),
        }


@dataclass
class TokenContext:
    """Verified caller identity and effective permissions."""

    user_id: str
    departments: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    token: str = ""
    is_superuser: bool = False

    @classmethod
    def from_access_token(cls, access_token: AccessToken) -> TokenContext:
        is_superuser = (
            "admin" in [r.lower() for r in access_token.roles]
            or "*" in access_token.permissions
            or "*:*" in access_token.permissions
        )
        # Merge departments and groups into a canonical departments list
        all_depts = list(dict.fromkeys(access_token.departments + access_token.groups))
        return cls(
            user_id=access_token.user_id,
            departments=all_depts,
            groups=list(access_token.groups),
            roles=list(access_token.roles),
            permissions=list(access_token.permissions),
            token=access_token.token,
            is_superuser=is_superuser,
        )


LOCAL_SUPERUSER = TokenContext(
    user_id="local_superuser",
    departments=[],
    groups=[],
    roles=["admin"],
    permissions=["*"],
    token="",
    is_superuser=True,
)
