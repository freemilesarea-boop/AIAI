"""Post-processing: raw model output → shippable audio assets.

Sits between inference and storage. Takes whatever the provider wrote
and produces the two delivery assets, then hands them to the storage
backend. Nothing here knows which storage backend is configured, and
nothing here alters the sound — see
:mod:`luber_audio_utils.transcode`.

Failure policy: if any required asset cannot be produced, verified, or
stored, the whole step raises. The caller must not mark the generation
COMPLETED. Intermediate files live in a temporary directory that is
removed on both success and failure, so a failed run leaves no partial
files behind. Stored objects use deterministic keys, so a retry
overwrites its own previous attempt rather than accumulating garbage.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from luber_audio_utils import (
    MASTER_FILE_EXTENSION,
    MASTER_FORMAT,
    MASTER_MIME_TYPE,
    PREVIEW_FILE_EXTENSION,
    PREVIEW_FORMAT,
    PREVIEW_MIME_TYPE,
    AudioProbe,
    AudioStorage,
    encode_preview_mp3_async,
    master_storage_key,
    preview_storage_key,
    sha256_file,
    transcode_master_wav_async,
)
from luber_schemas import AssetType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProducedAsset:
    """One delivery asset, ready to be recorded in the database."""

    asset_type: AssetType
    format: str
    mime_type: str
    file_extension: str
    sample_rate: int
    bit_depth: int | None
    bitrate: int | None
    channels: int
    duration: float
    storage_key: str
    sha256: str
    file_size: int


@dataclass(frozen=True)
class PostProcessingResult:
    master: ProducedAsset
    preview: ProducedAsset

    @property
    def assets(self) -> tuple[ProducedAsset, ProducedAsset]:
        return (self.master, self.preview)


def _describe(
    *,
    asset_type: AssetType,
    fmt: str,
    mime_type: str,
    extension: str,
    probe: AudioProbe,
    storage_key: str,
    path: Path,
) -> ProducedAsset:
    return ProducedAsset(
        asset_type=asset_type,
        format=fmt,
        mime_type=mime_type,
        file_extension=extension,
        sample_rate=probe.sample_rate,
        bit_depth=probe.bit_depth,
        bitrate=probe.bitrate_bps,
        channels=probe.channels,
        duration=probe.duration_seconds,
        storage_key=storage_key,
        # Hash the exact bytes that were uploaded, so the recorded digest
        # is the digest of what a client will download.
        sha256=sha256_file(path),
        file_size=path.stat().st_size,
    )


async def produce_delivery_assets(
    generation_id: UUID,
    raw_audio: Path,
    storage: AudioStorage,
) -> PostProcessingResult:
    """Normalize, encode, verify, and store the master and preview.

    Raises on any failure; callers treat that as a failed generation.
    """
    with tempfile.TemporaryDirectory(prefix=f"luber-post-{generation_id}-") as tmp:
        workdir = Path(tmp)
        master_path = workdir / f"master.{MASTER_FILE_EXTENSION}"
        preview_path = workdir / f"preview.{PREVIEW_FILE_EXTENSION}"

        # Format normalization only — verified against the contract
        # inside the transcoder itself.
        master_probe = await transcode_master_wav_async(raw_audio, master_path)
        # The preview is derived from the master, not from the raw file,
        # so both assets describe identical audio.
        preview_probe = await encode_preview_mp3_async(master_path, preview_path)

        master_key = master_storage_key(generation_id, MASTER_FILE_EXTENSION)
        preview_key = preview_storage_key(generation_id, PREVIEW_FILE_EXTENSION)

        master = _describe(
            asset_type=AssetType.MASTER,
            fmt=MASTER_FORMAT,
            mime_type=MASTER_MIME_TYPE,
            extension=MASTER_FILE_EXTENSION,
            probe=master_probe,
            storage_key=master_key,
            path=master_path,
        )
        preview = _describe(
            asset_type=AssetType.PREVIEW,
            fmt=PREVIEW_FORMAT,
            mime_type=PREVIEW_MIME_TYPE,
            extension=PREVIEW_FILE_EXTENSION,
            probe=preview_probe,
            storage_key=preview_key,
            path=preview_path,
        )

        # Upload last: a storage failure leaves no asset rows written by
        # the caller, and the temp dir is cleaned up regardless.
        await storage.put(master_key, master_path)
        await storage.put(preview_key, preview_path)

        logger.info(
            "post-processing produced delivery assets",
            extra={
                "generation_id": str(generation_id),
                "master_sha256": master.sha256,
                "preview_sha256": preview.sha256,
            },
        )
        return PostProcessingResult(master=master, preview=preview)
