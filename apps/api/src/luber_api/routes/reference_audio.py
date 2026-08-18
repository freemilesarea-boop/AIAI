"""Reference-audio ingestion.

The only endpoint in the product that accepts a file from the browser,
which makes it the only place where bytes of unknown provenance enter
the system. Everything it does is shaped by that.

The upload is streamed to a temporary file with a hard byte ceiling
enforced *during* the read, so an oversized or lying ``Content-Length``
cannot make the server buffer an arbitrary amount. It is then decoded
before it is believed, normalised to the canonical form, hashed, and
stored under a server-generated key. The client's filename survives only
as a display label.

What comes back is an opaque identifier. A client never learns a storage
key, never supplies one, and has no route that will serve a reference
back to it — references live outside the ``audio/`` namespace that the
download endpoint resolves against, so there is no asset row that could
name one.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from luber_api.dependencies import get_audio_storage, get_repository
from luber_api.session import enforce_trusted_origin, require_current_user
from luber_audio_utils import (
    AudioProcessingError,
    AudioStorage,
    ReferenceAudioRejected,
    normalize_reference,
    resolve_upload_format,
    safe_display_name,
)
from luber_database import GenerationRepository
from luber_schemas import (
    CANONICAL_REFERENCE_EXTENSION,
    MAX_REFERENCE_DURATION_SECONDS,
    MAX_REFERENCE_FILE_BYTES,
    SUPPORTED_REFERENCE_EXTENSIONS,
    reference_storage_key,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/reference-audio",
    tags=["reference-audio"],
    # Every product route: a session, and for unsafe methods an origin
    # this deployment serves. Applied at the router so a route added
    # later is protected by default rather than by remembering.
    dependencies=[Depends(require_current_user), Depends(enforce_trusted_origin)],
)

#: Read size. Small enough that the ceiling is enforced promptly, large
#: enough not to make a 40 MB upload a million syscalls.
_CHUNK_BYTES = 1024 * 1024


class ReferenceAudioResponse(BaseModel):
    """What a client gets back: an identity and the facts about it.

    Deliberately no storage key and no path. The identifier is the only
    handle, and it is all a generation request needs.
    """

    reference_id: uuid.UUID
    display_name: str | None
    duration_seconds: float
    sample_rate: int
    channels: int
    file_size: int


class ReferenceAudioLimits(BaseModel):
    max_file_bytes: int
    max_duration_seconds: float
    supported_formats: list[str]


@router.get("/limits", response_model=ReferenceAudioLimits)
async def reference_audio_limits() -> ReferenceAudioLimits:
    """What the server will accept, so a client can say so before uploading."""
    return ReferenceAudioLimits(
        max_file_bytes=MAX_REFERENCE_FILE_BYTES,
        max_duration_seconds=MAX_REFERENCE_DURATION_SECONDS,
        supported_formats=list(SUPPORTED_REFERENCE_EXTENSIONS),
    )


async def _spool_upload(upload: UploadFile, destination: Path) -> int:
    """Stream the upload to disk, refusing to exceed the byte ceiling.

    The limit is checked as bytes arrive rather than from the declared
    ``Content-Length``, which is a client assertion. Partial writes need
    no cleanup here: the caller owns a temporary directory that is
    removed on every exit path, success and refusal alike.
    """
    written = 0
    with destination.open("wb") as handle:
        while chunk := await upload.read(_CHUNK_BYTES):
            written += len(chunk)
            if written > MAX_REFERENCE_FILE_BYTES:
                limit_mb = MAX_REFERENCE_FILE_BYTES // (1024 * 1024)
                raise ReferenceAudioRejected(f"That file is larger than {limit_mb} MB.")
            handle.write(chunk)
    if written == 0:
        raise ReferenceAudioRejected("That file is empty.")
    return written


@router.post("", status_code=status.HTTP_201_CREATED, response_model=ReferenceAudioResponse)
async def upload_reference_audio(
    file: Annotated[UploadFile, File(description="Audio file to use as a reference")],
    repository: Annotated[GenerationRepository, Depends(get_repository)],
    storage: Annotated[AudioStorage, Depends(get_audio_storage)],
) -> ReferenceAudioResponse:
    """Validate, normalise and store a reference track.

    Rejections are 400 with a message the user can act on; a failure of
    ours is 500 with none of the detail. The two are kept apart because
    "your file is 90 minutes long" and "ffmpeg is missing" call for very
    different reactions.
    """
    try:
        source_format = resolve_upload_format(file.filename, file.content_type)
    except ReferenceAudioRejected as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    reference_id = uuid.uuid4()
    # The key is built from a server-generated UUID. No part of it comes
    # from the request, so there is nothing for a traversal to travel.
    storage_key = reference_storage_key(reference_id)

    with tempfile.TemporaryDirectory(prefix=f"luber-ref-{reference_id}-") as tmp:
        workdir = Path(tmp)
        raw = workdir / f"upload.{source_format}"
        canonical = workdir / f"reference.{CANONICAL_REFERENCE_EXTENSION}"
        try:
            await _spool_upload(file, raw)
            normalized = normalize_reference(raw, canonical, source_format=source_format)
        except ReferenceAudioRejected as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except AudioProcessingError as exc:
            logger.error("reference normalisation failed", exc_info=exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The reference track could not be processed.",
            ) from exc

        # Object first, row second. A row pointing at nothing is a broken
        # reference that will fail a generation later; an object with no
        # row is unreferenced bytes that nothing will ever select.
        await storage.put(storage_key, normalized.path)
        record = await repository.create_reference_audio(
            reference_id=reference_id,
            storage_key=storage_key,
            sha256=normalized.sha256,
            source_sha256=normalized.source_sha256,
            source_format=normalized.source_format,
            duration_seconds=normalized.duration_seconds,
            sample_rate=normalized.sample_rate,
            channels=normalized.channels,
            file_size=normalized.file_size,
            display_name=safe_display_name(file.filename),
        )

    logger.info(
        "stored reference audio",
        extra={
            "reference_id": str(record.id),
            "sha256": record.sha256,
            "duration_seconds": record.duration_seconds,
        },
    )
    return ReferenceAudioResponse(
        reference_id=record.id,
        display_name=record.display_name,
        duration_seconds=record.duration_seconds,
        sample_rate=record.sample_rate,
        channels=record.channels,
        file_size=record.file_size,
    )
