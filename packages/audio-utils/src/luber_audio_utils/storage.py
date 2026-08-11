"""Audio storage abstraction.

Phase 1 ships :class:`LocalAudioStorage` (local filesystem). Phase 4
adds S3-compatible adapters (S3/R2/Supabase) behind the same
:class:`AudioStorage` interface, plus signed download URLs. Services
never copy files with pathlib directly — always through this boundary.
"""

from __future__ import annotations

import asyncio
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from uuid import UUID


class AudioStorageError(Exception):
    """Raised when storing, resolving, or deleting audio fails."""


class AudioStorage(ABC):
    """Storage backend for generation audio assets."""

    @abstractmethod
    async def store_master_wav(self, generation_id: UUID, source: Path) -> str:
        """Persist the master WAV for a generation; return its storage key."""
        raise NotImplementedError

    @abstractmethod
    def resolve_path(self, storage_key: str) -> Path:
        """Local filesystem path for a storage key (local backend only)."""
        raise NotImplementedError

    @abstractmethod
    async def delete_generation_audio(self, generation_id: UUID) -> None:
        """Remove all stored audio for a generation. Idempotent."""
        raise NotImplementedError


class LocalAudioStorage(AudioStorage):
    """Filesystem-backed storage: ``{base_dir}/audio/{generation_id}/master.wav``."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def _key(self, generation_id: UUID) -> str:
        return f"audio/{generation_id}/master.wav"

    async def store_master_wav(self, generation_id: UUID, source: Path) -> str:
        if not await asyncio.to_thread(source.is_file):
            raise AudioStorageError(f"source audio missing: {source}")
        storage_key = self._key(generation_id)
        destination = self._base_dir / storage_key
        try:
            await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copyfile, source, destination)
        except OSError as exc:
            raise AudioStorageError(f"failed to store {storage_key}: {exc}") from exc
        return storage_key

    def resolve_path(self, storage_key: str) -> Path:
        return self._base_dir / storage_key

    async def delete_generation_audio(self, generation_id: UUID) -> None:
        directory = self._base_dir / "audio" / str(generation_id)
        try:
            await asyncio.to_thread(shutil.rmtree, directory, True)
        except OSError as exc:
            raise AudioStorageError(f"failed to delete audio for {generation_id}: {exc}") from exc
