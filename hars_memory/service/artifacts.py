"""Content-verified local and optional S3-compatible artifact stores."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Protocol
from urllib.parse import unquote, urlparse


class ArtifactError(RuntimeError):
    pass


class ArtifactNotFoundError(ArtifactError):
    pass


class ArtifactIntegrityError(ArtifactError):
    pass


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    uri: str
    sha256: str
    size_bytes: int


class ArtifactStore(Protocol):
    def put_file(self, key: str, source: Path) -> ArtifactRef: ...

    def materialize(self, ref: ArtifactRef, destination: Path) -> Path: ...

    def open(self, ref: ArtifactRef) -> BinaryIO: ...

    def materialize_input(
        self, uri: str, destination: Path, sha256: str | None = None
    ) -> Path: ...

    def materialize_result(
        self, uri: str, destination: Path, sha256: str | None = None
    ) -> Path: ...

    def put_result(self, index_id: str, version: int, path: Path) -> ArtifactRef: ...


def _validated_key(key: str) -> PurePosixPath:
    path = PurePosixPath(key)
    if not key or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactError(f"Unsafe artifact key: {key!r}")
    return path


def _digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _copy_verified(source: BinaryIO, destination: Path, expected_sha256: str) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
                digest.update(block)
                size += len(block)
            target.flush()
            os.fsync(target.fileno())
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ArtifactIntegrityError(
                f"Artifact checksum mismatch: expected {expected_sha256}, got {actual}"
            )
        os.replace(temporary_name, destination)
        return size
    finally:
        Path(temporary_name).unlink(missing_ok=True)


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        relative = _validated_key(key)
        candidate = self.root.joinpath(*relative.parts)
        # Reject a symlinked parent escaping the configured store root.
        resolved_parent = candidate.parent.resolve()
        if resolved_parent != self.root and self.root not in resolved_parent.parents:
            raise ArtifactError(f"Artifact key escapes store root: {key!r}")
        return candidate

    def put_file(self, key: str, source: Path) -> ArtifactRef:
        if not source.is_file():
            raise ArtifactNotFoundError(f"Artifact source does not exist: {source}")
        sha256, size = _digest(source)
        destination = self._path(key)
        if destination.exists():
            existing_sha, existing_size = _digest(destination)
            if (existing_sha, existing_size) != (sha256, size):
                raise ArtifactIntegrityError(f"Immutable artifact key already has other data: {key}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, candidate_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=destination.parent
            )
            candidate = Path(candidate_name)
            try:
                with os.fdopen(fd, "wb") as target, source.open("rb") as stream:
                    shutil.copyfileobj(stream, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                try:
                    # A hard link publishes the fully-written candidate only if
                    # the immutable key is still absent. It cannot expose a
                    # partially-written destination to readers or contenders.
                    os.link(candidate, destination)
                except FileExistsError:
                    existing_sha, existing_size = _digest(destination)
                    if (existing_sha, existing_size) != (sha256, size):
                        raise ArtifactIntegrityError(
                            f"Immutable artifact key already has other data: {key}"
                        )
            finally:
                candidate.unlink(missing_ok=True)
        return ArtifactRef(destination.as_uri(), sha256, size)

    def materialize(self, ref: ArtifactRef, destination: Path) -> Path:
        parsed = urlparse(ref.uri)
        if parsed.scheme != "file":
            raise ArtifactError(f"Local store cannot read URI {ref.uri!r}")
        source = Path(unquote(parsed.path)).resolve()
        if source != self.root and self.root not in source.parents:
            raise ArtifactError("Artifact URI is outside this store")
        if not source.is_file():
            raise ArtifactNotFoundError(f"Artifact does not exist: {ref.uri}")
        with source.open("rb") as stream:
            size = _copy_verified(stream, destination, ref.sha256)
        if size != ref.size_bytes:
            destination.unlink(missing_ok=True)
            raise ArtifactIntegrityError(
                f"Artifact size mismatch: expected {ref.size_bytes}, got {size}"
            )
        return destination

    def open(self, ref: ArtifactRef) -> BinaryIO:
        parsed = urlparse(ref.uri)
        if parsed.scheme != "file":
            raise ArtifactError(f"Local store cannot read URI {ref.uri!r}")
        source = Path(unquote(parsed.path)).resolve()
        if source != self.root and self.root not in source.parents:
            raise ArtifactError("Artifact URI is outside this store")
        if not source.is_file():
            raise ArtifactNotFoundError(f"Artifact does not exist: {ref.uri}")
        actual_sha, actual_size = _digest(source)
        if actual_sha != ref.sha256 or actual_size != ref.size_bytes:
            raise ArtifactIntegrityError(f"Artifact does not match its reference: {ref.uri}")
        return source.open("rb")

    def _ref_for_uri(self, uri: str, sha256: str | None) -> ArtifactRef:
        parsed = urlparse(uri)
        source = Path(unquote(parsed.path)).resolve()
        if parsed.scheme != "file" or (source != self.root and self.root not in source.parents):
            raise ArtifactError(f"Local store cannot read URI {uri!r}")
        if not source.is_file():
            raise ArtifactNotFoundError(f"Artifact does not exist: {uri}")
        actual_sha, size = _digest(source)
        if sha256 is not None and sha256 != actual_sha:
            raise ArtifactIntegrityError(
                f"Artifact checksum mismatch: expected {sha256}, got {actual_sha}"
            )
        return ArtifactRef(uri, sha256 or actual_sha, size)

    def materialize_input(
        self, uri: str, destination: Path, sha256: str | None = None
    ) -> Path:
        return self.materialize(self._ref_for_uri(uri, sha256), destination)

    def materialize_result(
        self, uri: str, destination: Path, sha256: str | None = None
    ) -> Path:
        return self.materialize_input(uri, destination, sha256)

    def put_result(self, index_id: str, version: int, path: Path) -> ArtifactRef:
        _validated_key(index_id)
        if version < 1:
            raise ValueError("version must be positive")
        return self.put_file(f"indexes/{index_id}/v{version}/index.tar.gz", path)


class S3ArtifactStore:
    """S3-compatible store; boto3 is imported only when this adapter is used."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        endpoint_url: str | None = None,
        region_name: str | None = None,
        client: object | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3 bucket must not be empty")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise ArtifactError("S3 artifacts require the optional boto3 dependency") from exc
            client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region_name)
        self.client = client

    def _object_key(self, key: str) -> str:
        value = str(_validated_key(key))
        return f"{self.prefix}/{value}" if self.prefix else value

    def put_file(self, key: str, source: Path) -> ArtifactRef:
        if not source.is_file():
            raise ArtifactNotFoundError(f"Artifact source does not exist: {source}")
        sha256, size = _digest(source)
        object_key = self._object_key(key)
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=object_key)  # type: ignore[attr-defined]
        except Exception as exc:
            response = getattr(exc, "response", {})
            code = str(response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise ArtifactError(f"Cannot inspect s3://{self.bucket}/{object_key}: {exc}") from exc
        else:
            metadata = head.get("Metadata", {})
            if metadata.get("sha256") != sha256 or int(head.get("ContentLength", -1)) != size:
                raise ArtifactIntegrityError(
                    f"Immutable artifact key already has other data: s3://{self.bucket}/{object_key}"
                )
            return ArtifactRef(f"s3://{self.bucket}/{object_key}", sha256, size)
        try:
            with source.open("rb") as body:
                self.client.put_object(  # type: ignore[attr-defined]
                    Bucket=self.bucket,
                    Key=object_key,
                    Body=body,
                    Metadata={"sha256": sha256},
                    IfNoneMatch="*",
                )
        except Exception as exc:
            response = getattr(exc, "response", {})
            code = str(response.get("Error", {}).get("Code", ""))
            if code not in {"409", "412", "ConditionalRequestConflict", "PreconditionFailed"}:
                raise ArtifactError(
                    f"Cannot upload s3://{self.bucket}/{object_key}: {exc}"
                ) from exc
            head = self.client.head_object(Bucket=self.bucket, Key=object_key)  # type: ignore[attr-defined]
            metadata = head.get("Metadata", {})
            if metadata.get("sha256") != sha256 or int(head.get("ContentLength", -1)) != size:
                raise ArtifactIntegrityError(
                    f"Immutable artifact key already has other data: s3://{self.bucket}/{object_key}"
                ) from exc
        return ArtifactRef(f"s3://{self.bucket}/{object_key}", sha256, size)

    def materialize(self, ref: ArtifactRef, destination: Path) -> Path:
        parsed = urlparse(ref.uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ArtifactError(f"S3 store cannot read URI {ref.uri!r}")
        object_key = parsed.path.lstrip("/")
        expected_prefix = f"{self.prefix}/" if self.prefix else ""
        if expected_prefix and not object_key.startswith(expected_prefix):
            raise ArtifactError("Artifact URI is outside this store prefix")
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=object_key)["Body"]  # type: ignore[attr-defined]
            size = _copy_verified(body, destination, ref.sha256)
        except ArtifactIntegrityError:
            raise
        except Exception as exc:
            raise ArtifactNotFoundError(f"Cannot download {ref.uri}: {exc}") from exc
        if size != ref.size_bytes:
            destination.unlink(missing_ok=True)
            raise ArtifactIntegrityError(
                f"Artifact size mismatch: expected {ref.size_bytes}, got {size}"
            )
        return destination

    def open(self, ref: ArtifactRef) -> BinaryIO:
        parsed = urlparse(ref.uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ArtifactError(f"S3 store cannot read URI {ref.uri!r}")
        try:
            response = self.client.get_object(  # type: ignore[attr-defined]
                Bucket=self.bucket, Key=parsed.path.lstrip("/")
            )
        except Exception as exc:
            raise ArtifactNotFoundError(f"Cannot download {ref.uri}: {exc}") from exc
        metadata = response.get("Metadata", {})
        length = int(response.get("ContentLength", ref.size_bytes))
        if metadata.get("sha256") not in {None, ref.sha256} or length != ref.size_bytes:
            response["Body"].close()
            raise ArtifactIntegrityError(f"Artifact does not match its reference: {ref.uri}")
        return response["Body"]

    def _ref_for_uri(self, uri: str, sha256: str | None) -> ArtifactRef:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ArtifactError(f"S3 store cannot read URI {uri!r}")
        key = parsed.path.lstrip("/")
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=key)  # type: ignore[attr-defined]
        except Exception as exc:
            raise ArtifactNotFoundError(f"Cannot inspect {uri}: {exc}") from exc
        stored_sha = head.get("Metadata", {}).get("sha256")
        expected_sha = sha256 or stored_sha
        if not expected_sha:
            raise ArtifactIntegrityError(f"S3 object has no sha256 metadata: {uri}")
        if stored_sha and stored_sha != expected_sha:
            raise ArtifactIntegrityError(
                f"Artifact checksum mismatch: expected {expected_sha}, got {stored_sha}"
            )
        return ArtifactRef(uri, expected_sha, int(head["ContentLength"]))

    def materialize_input(
        self, uri: str, destination: Path, sha256: str | None = None
    ) -> Path:
        return self.materialize(self._ref_for_uri(uri, sha256), destination)

    def materialize_result(
        self, uri: str, destination: Path, sha256: str | None = None
    ) -> Path:
        return self.materialize_input(uri, destination, sha256)

    def put_result(self, index_id: str, version: int, path: Path) -> ArtifactRef:
        _validated_key(index_id)
        if version < 1:
            raise ValueError("version must be positive")
        return self.put_file(f"indexes/{index_id}/v{version}/index.tar.gz", path)


def create_artifact_store(
    url: str,
    *,
    s3_endpoint_url: str | None = None,
    s3_region_name: str | None = None,
) -> ArtifactStore:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return LocalArtifactStore(Path(unquote(parsed.path)))
    if parsed.scheme == "s3":
        return S3ArtifactStore(
            parsed.netloc,
            prefix=parsed.path.strip("/"),
            endpoint_url=s3_endpoint_url,
            region_name=s3_region_name,
        )
    if not parsed.scheme:
        return LocalArtifactStore(Path(url))
    raise ArtifactError(f"Unsupported artifact store URL scheme: {parsed.scheme!r}")


__all__ = [
    "ArtifactError",
    "ArtifactIntegrityError",
    "ArtifactNotFoundError",
    "ArtifactRef",
    "ArtifactStore",
    "LocalArtifactStore",
    "S3ArtifactStore",
    "create_artifact_store",
]
