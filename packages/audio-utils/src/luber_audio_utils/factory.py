"""Storage backend selection from configuration.

The only place that decides which adapter is in use. Services call this
once at startup and then hold an :class:`AudioStorage` — nothing
downstream knows or cares whether bytes live on a local disk or in an
object store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from luber_audio_utils.s3 import S3AudioStorage, S3StorageConfig
from luber_audio_utils.storage import AudioStorage, AudioStorageError, LocalAudioStorage


class StorageSettings(Protocol):
    """The settings fields this factory reads (see BaseServiceSettings)."""

    storage_provider: str
    audio_storage_dir: str
    storage_bucket: str
    storage_region: str | None
    storage_endpoint: str | None
    storage_access_key_id: str | None
    storage_secret_access_key: str | None
    storage_force_path_style: bool
    storage_key_prefix: str


def storage_from_settings(settings: StorageSettings) -> AudioStorage:
    """Build the configured storage backend.

    Credentials are read from settings (i.e. the environment) and are
    never logged or echoed back — a misconfiguration reports which field
    is missing, never its value.
    """
    provider = (settings.storage_provider or "local").strip().lower()

    if provider == "local":
        return LocalAudioStorage(Path(settings.audio_storage_dir))

    if provider == "s3":
        if not settings.storage_bucket:
            raise AudioStorageError("storage_provider='s3' requires STORAGE_BUCKET to be set")
        return S3AudioStorage(
            S3StorageConfig(
                bucket=settings.storage_bucket,
                region=settings.storage_region,
                endpoint_url=settings.storage_endpoint,
                access_key_id=settings.storage_access_key_id,
                secret_access_key=settings.storage_secret_access_key,
                force_path_style=settings.storage_force_path_style,
                prefix=settings.storage_key_prefix,
            )
        )

    raise AudioStorageError(f"unknown storage provider: {provider!r} (available: 'local', 's3')")
