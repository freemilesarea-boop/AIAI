"""Object-storage abstraction for audio assets.

Business logic addresses audio by *storage key* only. It never builds a
filesystem path, never learns which backend is in use, and never sees a
bucket name or credential. That keeps the local development adapter and
the production S3-compatible adapter interchangeable.

Storage keys are deterministic and derived from the generation id and
asset type, so re-running post-processing for a generation overwrites
its own objects instead of accumulating new ones — collisions between
different generations are impossible because the UUID is in the key.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


class AudioStorageError(Exception):
    """Raised when storing, resolving, reading, or deleting audio fails."""


@dataclass(frozen=True)
class DownloadTarget:
    """How a client should fetch an object.

    Exactly one of the two modes applies:

    * ``url`` set — the backend issues a short-lived signed URL and the
      client fetches the object directly from the storage provider.
    * ``url`` ``None`` — the backend must stream the bytes itself.

    Authorization happens in the API before either mode is used.
    """

    url: str | None
    expires_in_seconds: int | None = None

    @property
    def is_signed_url(self) -> bool:
        return self.url is not None


def master_storage_key(generation_id: UUID, extension: str = "wav") -> str:
    """Deterministic key for a generation's production master."""
    return f"audio/{generation_id}/master.{extension}"


def finished_master_storage_key(generation_id: UUID, extension: str = "wav") -> str:
    """Deterministic key for a generation's finished master.

    A separate key from the raw master, not a variant of it: finishing
    must never be able to write over the only copy of what the model
    produced, and a distinct key makes that impossible rather than merely
    unintended.
    """
    return f"audio/{generation_id}/finished.{extension}"


def preview_storage_key(generation_id: UUID, extension: str = "mp3") -> str:
    """Deterministic key for a generation's preview asset."""
    return f"audio/{generation_id}/preview.{extension}"


def generation_prefix(generation_id: UUID) -> str:
    return f"audio/{generation_id}/"


def _validate_key(storage_key: str) -> str:
    """Reject keys that are absolute, empty, or contain traversal parts.

    Keys are server-generated, but this runs on every access so a
    corrupted or hand-edited database row cannot reach outside the
    storage root.
    """
    if not storage_key or storage_key.strip() != storage_key:
        raise AudioStorageError(f"invalid storage key: {storage_key!r}")
    if storage_key.startswith("/") or storage_key.startswith("\\"):
        # An absolute key would resolve outside the root, so it is
        # reported the same way as a traversal attempt.
        raise AudioStorageError(
            f"storage key escapes storage root (must be relative): {storage_key!r}"
        )
    parts = storage_key.replace("\\", "/").split("/")
    if any(part in ("..", ".", "") for part in parts):
        raise AudioStorageError(f"storage key escapes storage root: {storage_key!r}")
    if "\x00" in storage_key:
        raise AudioStorageError("storage key contains a null byte")
    return storage_key


class AudioStorage(ABC):
    """Backend for storing and serving generation audio."""

    @abstractmethod
    async def put(self, storage_key: str, source: Path) -> str:
        """Store *source* under *storage_key*; return the key."""

    @abstractmethod
    async def open(self, storage_key: str) -> bytes:
        """Read an object's bytes."""

    @abstractmethod
    async def exists(self, storage_key: str) -> bool:
        """Whether an object is present."""

    @abstractmethod
    async def delete(self, storage_key: str) -> None:
        """Remove one object. Idempotent."""

    @abstractmethod
    async def delete_generation_audio(self, generation_id: UUID) -> None:
        """Remove all objects for a generation. Idempotent."""

    @abstractmethod
    async def download_target(
        self, storage_key: str, *, filename: str, content_type: str, expires_in_seconds: int
    ) -> DownloadTarget:
        """How the caller should deliver this object to an authorized client."""

    @abstractmethod
    def local_path(self, storage_key: str) -> Path | None:
        """Local file path when the backend has one, else ``None``.

        Only backend-internal streaming uses this. It is never part of
        any API response.
        """

    async def store_master_wav(self, generation_id: UUID, source: Path) -> str:
        """Convenience wrapper kept for the master-asset call site."""
        return await self.put(master_storage_key(generation_id), source)

    @staticmethod
    async def sha256_of(path: Path) -> str:
        def _digest() -> str:
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        return await asyncio.to_thread(_digest)


class LocalAudioStorage(AudioStorage):
    """Filesystem-backed storage for development and single-node runs.

    Has no signed-URL mechanism, so downloads are always backend-
    mediated — the API applies the same authorization it would apply
    before minting a signed URL in production.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def resolve_path(self, storage_key: str) -> Path:
        """Resolve a key to a path, refusing anything outside the root."""
        _validate_key(storage_key)
        base = self._base_dir.resolve()
        candidate = (base / storage_key).resolve()
        if candidate != base and base not in candidate.parents:
            raise AudioStorageError(f"storage key escapes storage root: {storage_key!r}")
        return candidate

    def local_path(self, storage_key: str) -> Path | None:
        return self.resolve_path(storage_key)

    async def put(self, storage_key: str, source: Path) -> str:
        if not await asyncio.to_thread(source.is_file):
            raise AudioStorageError(f"source audio missing: {source}")
        destination = self.resolve_path(storage_key)
        try:
            await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copyfile, source, destination)
        except OSError as exc:
            raise AudioStorageError(f"failed to store {storage_key}: {exc}") from exc
        return storage_key

    async def open(self, storage_key: str) -> bytes:
        path = self.resolve_path(storage_key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise AudioStorageError(f"failed to read {storage_key}: {exc}") from exc

    async def exists(self, storage_key: str) -> bool:
        return await asyncio.to_thread(self.resolve_path(storage_key).is_file)

    async def delete(self, storage_key: str) -> None:
        path = self.resolve_path(storage_key)
        try:
            await asyncio.to_thread(path.unlink, True)
        except OSError as exc:
            raise AudioStorageError(f"failed to delete {storage_key}: {exc}") from exc

    async def delete_generation_audio(self, generation_id: UUID) -> None:
        directory = self.resolve_path(generation_prefix(generation_id).rstrip("/"))
        try:
            await asyncio.to_thread(shutil.rmtree, directory, True)
        except OSError as exc:
            raise AudioStorageError(f"failed to delete audio for {generation_id}: {exc}") from exc

    async def download_target(
        self, storage_key: str, *, filename: str, content_type: str, expires_in_seconds: int
    ) -> DownloadTarget:
        # No signing available locally: the API streams the bytes after
        # performing the same authorization check.
        _validate_key(storage_key)
        return DownloadTarget(url=None)
