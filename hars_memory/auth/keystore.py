"""Encrypted at-rest keystore for cryptographic signing keys and secrets.

Uses AES-256-GCM for authenticated encryption of private keys, signing secrets,
and revocation state. Master key can be supplied via environment variable,
key file (mode 0600), or derived from a passphrase using PBKDF2.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

DEFAULT_AUTH_DIR: Final[Path] = (
    Path.home() / ".local" / "share" / "hars-longterm-memory" / "auth"
)
DEFAULT_KEYSTORE_PATH: Final[Path] = DEFAULT_AUTH_DIR / "keystore.enc"
DEFAULT_MASTER_KEY_PATH: Final[Path] = DEFAULT_AUTH_DIR / "master.key"

MASTER_KEY_ENV: Final[str] = "HARS_MEMORY_MASTER_KEY"
KEYSTORE_PATH_ENV: Final[str] = "HARS_MEMORY_KEYSTORE_PATH"

PBKDF2_ITERATIONS: Final[int] = 100_000


@dataclass
class KeyStoreData:
    """Decrypted keystore contents."""

    active_kid: str
    keys: dict[str, dict[str, Any]] = field(default_factory=dict)
    revoked_jtis: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_kid": self.active_kid,
            "keys": self.keys,
            "revoked_jtis": sorted(self.revoked_jtis),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeyStoreData:
        return cls(
            active_kid=str(data.get("active_kid", "")),
            keys=dict(data.get("keys", {})),
            revoked_jtis=set(data.get("revoked_jtis", [])),
        )


class EncryptedKeyStore:
    """Manages reading, writing, and encrypting signing keys at rest."""

    def __init__(
        self,
        keystore_path: Path | str | None = None,
        master_key: bytes | str | None = None,
        passphrase: str | None = None,
        auto_create_master_key: bool = True,
    ) -> None:
        self.keystore_path = Path(
            keystore_path
            or os.environ.get(KEYSTORE_PATH_ENV)
            or DEFAULT_KEYSTORE_PATH
        ).expanduser()
        self._master_key = self._resolve_master_key(
            master_key=master_key,
            passphrase=passphrase,
            auto_create=auto_create_master_key,
        )

    @classmethod
    def generate_master_key(cls) -> bytes:
        """Generate a random 32-byte (256-bit) cryptographically strong master key."""
        return secrets.token_bytes(32)

    def _resolve_master_key(
        self,
        master_key: bytes | str | None,
        passphrase: str | None,
        auto_create: bool,
    ) -> bytes | None:
        if isinstance(master_key, bytes) and len(master_key) == 32:
            return master_key

        if isinstance(master_key, str) and master_key.strip():
            return self._parse_key_string(master_key.strip())

        env_key = os.environ.get(MASTER_KEY_ENV, "").strip()
        if env_key:
            return self._parse_key_string(env_key)

        passphrase_val = passphrase or os.environ.get("HARS_MEMORY_AUTH_PASSPHRASE", "").strip()
        if passphrase_val:
            return self._derive_key_from_passphrase(passphrase_val, salt=b"hars_memory_master_salt_static")

        # Check default master key file
        master_key_file = DEFAULT_MASTER_KEY_PATH
        if master_key_file.is_file():
            try:
                content = master_key_file.read_text(encoding="utf-8").strip()
                return self._parse_key_string(content)
            except Exception as exc:
                logger.warning("Could not read master key file %s: %s", master_key_file, exc)

        if auto_create:
            # Generate and persist a local master key file with 0600 permissions
            try:
                new_key = self.generate_master_key()
                master_key_file.parent.mkdir(parents=True, exist_ok=True)
                # Write with 0600 permissions
                hex_key = new_key.hex()
                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                fd = os.open(str(master_key_file), flags, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"{hex_key}\n")
                logger.info("Generated new master key at %s", master_key_file)
                return new_key
            except Exception as exc:
                logger.warning("Could not automatically create master key file: %s", exc)

        return None

    @staticmethod
    def _parse_key_string(key_str: str) -> bytes:
        # Hex (64 chars) or base64 (44 chars) or plain 32 chars
        if len(key_str) == 64:
            try:
                return bytes.fromhex(key_str)
            except ValueError:
                pass
        try:
            raw = base64.b64decode(key_str)
            if len(raw) == 32:
                return raw
        except Exception:
            pass
        # Fallback SHA256 of raw string
        digest = hashes.Hash(hashes.SHA256())
        digest.update(key_str.encode("utf-8"))
        return digest.finalize()

    @staticmethod
    def _derive_key_from_passphrase(passphrase: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=PBKDF2_ITERATIONS,
        )
        return kdf.derive(passphrase.encode("utf-8"))

    def exists(self) -> bool:
        """Check if keystore file exists on disk."""
        return self.keystore_path.is_file()

    def save(self, data: KeyStoreData) -> None:
        """Encrypt and save KeyStoreData to disk using AES-256-GCM."""
        if not self._master_key:
            raise RuntimeError(
                "Cannot save encrypted keystore: master key is not configured or resolved."
            )

        payload_bytes = json.dumps(data.to_dict(), ensure_ascii=False).encode("utf-8")
        aesgcm = AESGCM(self._master_key)
        nonce = secrets.token_bytes(12)  # Standard 96-bit nonce for GCM
        ciphertext = aesgcm.encrypt(nonce, payload_bytes, associated_data=b"hars_memory_keystore_v1")

        container = {
            "version": 1,
            "cipher": "aes-256-gcm",
            "nonce": nonce.hex(),
            "ciphertext": ciphertext.hex(),
        }

        self.keystore_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(str(self.keystore_path), flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(container, f, indent=2)

    def load(self) -> KeyStoreData:
        """Load and decrypt KeyStoreData from disk."""
        if not self.exists():
            return KeyStoreData(active_kid="")

        if not self._master_key:
            raise RuntimeError(
                "Cannot decrypt keystore: master key is not configured or resolved."
            )

        text = self.keystore_path.read_text(encoding="utf-8")
        container = json.loads(text)

        version = container.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported keystore version: {version}")

        nonce = bytes.fromhex(container["nonce"])
        ciphertext = bytes.fromhex(container["ciphertext"])

        aesgcm = AESGCM(self._master_key)
        decrypted_bytes = aesgcm.decrypt(
            nonce, ciphertext, associated_data=b"hars_memory_keystore_v1"
        )
        data_dict = json.loads(decrypted_bytes.decode("utf-8"))
        return KeyStoreData.from_dict(data_dict)
