"""JWT issuance, verification, key generation, and revocation management."""

from __future__ import annotations

import datetime
import logging
import secrets
import time
from pathlib import Path
from typing import Any, Final

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from hars_memory.auth.keystore import EncryptedKeyStore, KeyStoreData
from hars_memory.auth.models import TokenContext

logger = logging.getLogger(__name__)

JWT_ISSUER: Final[str] = "hars-memory"
DEFAULT_ALGORITHM: Final[str] = "EdDSA"


class JWTManager:
    """Manages cryptographic key pairs and JWT token lifecycle."""

    def __init__(
        self,
        keystore: EncryptedKeyStore | None = None,
        keystore_path: Path | str | None = None,
        master_key: bytes | str | None = None,
    ) -> None:
        self.keystore = keystore or EncryptedKeyStore(
            keystore_path=keystore_path, master_key=master_key
        )
        self._cached_data: KeyStoreData | None = None
        self._last_mtime_ns: int | None = None

    def _get_data(self, refresh: bool = False) -> KeyStoreData:
        try:
            current_mtime = (
                self.keystore.keystore_path.stat().st_mtime_ns
                if self.keystore.keystore_path.exists()
                else None
            )
        except OSError:
            current_mtime = None

        if (
            self._cached_data is None
            or refresh
            or (current_mtime is not None and current_mtime != self._last_mtime_ns)
        ):
            self._cached_data = self.keystore.load()
            self._last_mtime_ns = current_mtime
        return self._cached_data

    def _save_data(self, data: KeyStoreData) -> None:
        self.keystore.save(data)
        self._cached_data = data
        try:
            self._last_mtime_ns = (
                self.keystore.keystore_path.stat().st_mtime_ns
                if self.keystore.keystore_path.exists()
                else None
            )
        except OSError:
            self._last_mtime_ns = None

    def init_keys(
        self,
        algorithm: str = DEFAULT_ALGORITHM,
        key_id: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Initialize and persist a new signing key pair in the encrypted keystore."""
        data = self._get_data(refresh=True)
        if data.active_kid and not force:
            active_info = data.keys.get(data.active_kid, {})
            return {
                "kid": data.active_kid,
                "algorithm": active_info.get("algorithm", DEFAULT_ALGORITHM),
                "created": False,
                "public_key_pem": active_info.get("public_key_pem", ""),
            }

        kid = key_id or f"key_{secrets.token_hex(4)}"

        if algorithm.upper() == "EDDSA":
            priv_key = ed25519.Ed25519PrivateKey.generate()
            pub_key = priv_key.public_key()

            priv_pem = priv_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ).decode("utf-8")

            pub_pem = pub_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("utf-8")

            data.keys[kid] = {
                "algorithm": "EdDSA",
                "private_key_pem": priv_pem,
                "public_key_pem": pub_pem,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        elif algorithm.upper() == "HS256":
            secret_hex = secrets.token_hex(32)
            data.keys[kid] = {
                "algorithm": "HS256",
                "secret": secret_hex,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'EdDSA' or 'HS256'.")

        data.active_kid = kid
        self._save_data(data)
        logger.info("Initialized new signing key %s (%s) in encrypted keystore", kid, algorithm)

        return {
            "kid": kid,
            "algorithm": algorithm,
            "created": True,
            "public_key_pem": data.keys[kid].get("public_key_pem", ""),
        }

    def issue_token(
        self,
        user_id: str,
        departments: list[str] | None = None,
        groups: list[str] | None = None,
        roles: list[str] | None = None,
        scopes: list[str] | None = None,
        expires_in_seconds: int | None = 30 * 86400,  # Default 30 days
        jti: str | None = None,
    ) -> str:
        """Issue and sign a new JWT access token."""
        data = self._get_data()
        if not data.active_kid or data.active_kid not in data.keys:
            # Auto-initialize keys if not present
            self.init_keys()
            data = self._get_data(refresh=True)

        kid = data.active_kid
        key_entry = data.keys[kid]
        algorithm = key_entry.get("algorithm", DEFAULT_ALGORITHM)

        now = int(time.time())
        token_id = jti or f"tok_{secrets.token_hex(8)}"

        clean_departments = [d.strip() for d in (departments or []) if d.strip()]
        clean_groups = [g.strip() for g in (groups or []) if g.strip()]
        clean_roles = [r.strip() for r in (roles or []) if r.strip()]
        clean_scopes = [s.strip() for s in (scopes or []) if s.strip()]

        payload: dict[str, Any] = {
            "iss": JWT_ISSUER,
            "sub": str(user_id).strip(),
            "iat": now,
            "jti": token_id,
            "dept": clean_departments,
            "groups": clean_groups,
            "roles": clean_roles,
            "scope": clean_scopes,
        }

        if expires_in_seconds is not None:
            payload["exp"] = now + expires_in_seconds

        headers = {"kid": kid}

        if algorithm == "EdDSA":
            priv_pem = key_entry["private_key_pem"]
            token = jwt.encode(payload, priv_pem, algorithm="EdDSA", headers=headers)
        elif algorithm == "HS256":
            secret = key_entry["secret"]
            token = jwt.encode(payload, secret, algorithm="HS256", headers=headers)
        else:
            raise ValueError(f"Unknown signing algorithm: {algorithm}")

        return token

    def verify_token(self, token_str: str) -> TokenContext | None:
        """Verify JWT signature, expiration, and revocation status.

        Returns TokenContext if valid, or None if invalid.
        """
        if not token_str or not isinstance(token_str, str):
            return None

        clean_token = token_str.strip()
        if clean_token.lower().startswith("bearer "):
            clean_token = clean_token[7:].strip()

        # Quick check for JWT format (three dot-separated segments)
        if clean_token.count(".") != 2:
            return None

        try:
            unverified_headers = jwt.get_unverified_header(clean_token)
        except Exception:
            return None

        kid = unverified_headers.get("kid")
        data = self._get_data()

        # Find key
        key_entry = None
        if kid and kid in data.keys:
            key_entry = data.keys[kid]
        elif data.active_kid and data.active_kid in data.keys:
            key_entry = data.keys[data.active_kid]
        else:
            return None

        algorithm = key_entry.get("algorithm", DEFAULT_ALGORITHM)
        if algorithm == "EdDSA":
            verification_key = key_entry.get("public_key_pem") or key_entry.get("private_key_pem")
            allowed_algorithms = ["EdDSA"]
        elif algorithm == "HS256":
            verification_key = key_entry.get("secret")
            allowed_algorithms = ["HS256"]
        else:
            return None

        try:
            payload = jwt.decode(
                clean_token,
                verification_key,
                algorithms=allowed_algorithms,
                issuer=JWT_ISSUER,
                options={"require": ["sub", "iat"]},
            )
        except jwt.PyJWTError as exc:
            logger.debug("JWT verification failed: %s", exc)
            return None

        jti = payload.get("jti")
        if jti and jti in data.revoked_jtis:
            logger.warning("Rejected revoked JWT jti=%s", jti)
            return None

        user_id = payload.get("sub", "")
        departments = payload.get("dept") or []
        groups = payload.get("groups") or []
        roles = payload.get("roles") or []
        scopes = payload.get("scope") or []

        is_superuser = (
            "admin" in [r.lower() for r in roles]
            or "*" in scopes
            or "*:*" in scopes
        )

        all_depts = list(dict.fromkeys(departments + groups))

        return TokenContext(
            user_id=user_id,
            departments=all_depts,
            groups=groups,
            roles=roles,
            permissions=scopes,
            token=clean_token,
            is_superuser=is_superuser,
        )

    def revoke_token(self, token_or_jti: str) -> bool:
        """Revoke a token by its jti or the raw JWT token string."""
        jti = token_or_jti.strip()
        if jti.count(".") == 2:
            try:
                unverified_payload = jwt.decode(jti, options={"verify_signature": False})
                jti = unverified_payload.get("jti", "")
            except Exception:
                return False

        if not jti:
            return False

        data = self._get_data(refresh=True)
        data.revoked_jtis.add(jti)
        self._save_data(data)
        logger.info("Revoked token jti=%s", jti)
        return True

    def inspect_token(self, token_str: str) -> dict[str, Any]:
        """Decode and inspect token claims and validity status."""
        clean = token_str.strip()
        if clean.lower().startswith("bearer "):
            clean = clean[7:].strip()

        try:
            header = jwt.get_unverified_header(clean)
            payload = jwt.decode(clean, options={"verify_signature": False})
        except Exception as exc:
            return {"valid_structure": False, "error": str(exc)}

        context = self.verify_token(clean)
        data = self._get_data()
        jti = payload.get("jti", "")
        is_revoked = bool(jti and jti in data.revoked_jtis)

        exp = payload.get("exp")
        exp_iso = (
            datetime.datetime.fromtimestamp(exp, tz=datetime.timezone.utc).isoformat()
            if exp
            else "never"
        )
        is_expired = bool(exp and exp < time.time())

        return {
            "valid_structure": True,
            "signature_verified": context is not None,
            "is_revoked": is_revoked,
            "is_expired": is_expired,
            "header": header,
            "payload": payload,
            "expires_at": exp_iso,
            "effective_context": context.__dict__ if context else None,
        }


_default_jwt_manager: JWTManager | None = None


def get_default_jwt_manager() -> JWTManager:
    """Return the global default JWTManager singleton."""
    global _default_jwt_manager
    if _default_jwt_manager is None:
        _default_jwt_manager = JWTManager()
    return _default_jwt_manager


def reset_default_jwt_manager() -> None:
    """Reset the global default JWTManager singleton."""
    global _default_jwt_manager
    _default_jwt_manager = None
