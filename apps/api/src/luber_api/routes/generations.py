"""Generation endpoints: POST/GET/LIST/DELETE /v1/generations.

Routes contain no business logic — they translate HTTP to repository/
enqueuer calls. Generation execution happens in the worker behind the
queue boundary.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from fastapi.responses import FileResponse
from sqlalchemy.exc import IntegrityError

from luber_api.dependencies import get_audio_storage, get_enqueuer, get_repository
from luber_api.jobs import GenerationEnqueuer
from luber_api.schemas import (
    GenerationCreateRequest,
    GenerationCreateResponse,
    GenerationListResponse,
    GenerationResponse,
)
from luber_audio_utils import AudioStorage, AudioStorageError
from luber_database import GenerationRepository
from luber_generation_client import GENERATION_QUEUE_NAME
from luber_schemas import AssetType, ErrorCode, GenerationStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/generations", tags=["generations"])

IDEMPOTENCY_KEY_MAX_LENGTH = 200

_FILENAME_STRIP = re.compile(r"[^a-z0-9]+")
DOWNLOAD_FILENAME_MAX_LENGTH = 60


def build_download_filename(title: str, generation_id: uuid.UUID) -> str:
    """ASCII-safe ``.wav`` filename derived from a user-supplied title.

    User input never reaches the filesystem — this only labels the
    download. Non-ASCII titles (e.g. Korean) legitimately slug to
    nothing, so a stable generation-derived name is used instead.
    """
    slug = _FILENAME_STRIP.sub("-", title.lower()).strip("-")[:DOWNLOAD_FILENAME_MAX_LENGTH]
    slug = slug.strip("-")
    if not slug:
        slug = f"luber-track-{generation_id.hex[:8]}"
    return f"{slug}.wav"


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=GenerationCreateResponse,
)
async def create_generation(
    payload: GenerationCreateRequest,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
    enqueuer: Annotated[GenerationEnqueuer, Depends(get_enqueuer)],
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> GenerationCreateResponse:
    if idempotency_key is not None and len(idempotency_key) > IDEMPOTENCY_KEY_MAX_LENGTH:
        raise HTTPException(
            status_code=422,
            detail="Idempotency-Key too long (max 200 characters)",
        )

    if idempotency_key is not None:
        existing = await repository.get_by_idempotency_key(idempotency_key)
        if existing is not None:
            return GenerationCreateResponse(
                generation_id=existing.id, status=GenerationStatus(existing.status)
            )

    try:
        generation = await repository.create_generation(
            title=payload.title,
            prompt=payload.prompt,
            lyrics=payload.lyrics,
            vocal_gender=payload.vocal_gender.value,
            duration_requested=payload.duration,
            seed=payload.seed,
            language=payload.language,
            instrumental=payload.instrumental,
            status=GenerationStatus.QUEUED.value,
            idempotency_key=idempotency_key,
        )
    except IntegrityError:
        # Concurrent duplicate: the unique index won the race — return
        # the winner's generation instead of creating a second one.
        if idempotency_key is None:
            raise
        existing = await repository.get_by_idempotency_key(idempotency_key)
        if existing is None:
            raise
        return GenerationCreateResponse(
            generation_id=existing.id, status=GenerationStatus(existing.status)
        )

    await repository.create_job(
        generation.id,
        queue_name=GENERATION_QUEUE_NAME,
        status=GenerationStatus.QUEUED.value,
    )

    try:
        await enqueuer.enqueue(generation.id)
    except Exception as exc:
        logger.exception(
            "failed to enqueue generation",
            extra={"generation_id": str(generation.id)},
        )
        await repository.mark_failed(
            generation.id,
            status=GenerationStatus.FAILED.value,
            error_code=ErrorCode.QUEUE_FAILED.value,
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorCode.QUEUE_FAILED.value,
        ) from exc

    return GenerationCreateResponse(
        generation_id=generation.id,
        status=GenerationStatus(generation.status),
    )


@router.get("/{generation_id}", response_model=GenerationResponse)
async def get_generation(
    generation_id: uuid.UUID,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> GenerationResponse:
    generation = await repository.get_generation(generation_id)
    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="generation not found")
    return GenerationResponse.model_validate(generation)


@router.get("/{generation_id}/audio")
async def get_generation_audio(
    generation_id: uuid.UUID,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
    storage: Annotated[AudioStorage, Depends(get_audio_storage)],
    download: Annotated[bool, Query()] = False,
) -> FileResponse:
    """Stream the MASTER WAV so a browser can play or download it.

    Only the MASTER asset is served. The client addresses audio by
    generation id — storage keys and filesystem paths are never part of
    the contract, so no client-supplied value reaches path resolution.
    """
    generation = await repository.get_generation(generation_id)
    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="generation not found")

    master = next(
        (a for a in generation.audio_assets if a.asset_type == AssetType.MASTER.value),
        None,
    )
    if master is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="master audio not available for this generation",
        )

    try:
        path = storage.resolve_path(master.storage_key)
    except AudioStorageError:
        # A bad storage key is an operator problem; the client learns
        # nothing about paths.
        logger.exception(
            "unsafe storage key for audio asset",
            extra={"generation_id": str(generation_id), "asset_id": str(master.id)},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="master audio not available for this generation",
        ) from None

    if not path.is_file():
        logger.error(
            "master audio missing from storage",
            extra={"generation_id": str(generation_id), "asset_id": str(master.id)},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="master audio not available for this generation",
        )

    # FileResponse streams from disk and sets Content-Length itself; the
    # file is never read into memory in full.
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=build_download_filename(generation.title, generation_id),
        content_disposition_type="attachment" if download else "inline",
    )


@router.get("", response_model=GenerationListResponse)
async def list_generations(
    repository: Annotated[GenerationRepository, Depends(get_repository)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> GenerationListResponse:
    generations, total = await repository.list_generations(limit=limit, offset=offset)
    return GenerationListResponse(
        items=[GenerationResponse.model_validate(g) for g in generations],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.delete("/{generation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_generation(
    generation_id: uuid.UUID,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
    storage: Annotated[AudioStorage, Depends(get_audio_storage)],
) -> Response:
    """Hard-delete a generation, its jobs/asset rows, and stored audio.

    Phase 1 policy (development): immediate hard delete of DB rows and
    local audio files (best-effort, after the DB delete). A
    production-grade retention/irreversible-deletion policy is
    deliberately deferred to a later phase — none is invented here.
    """
    generation = await repository.get_generation(generation_id)
    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="generation not found")

    await repository.delete_generation(generation_id)
    try:
        await storage.delete_generation_audio(generation_id)
    except AudioStorageError:
        # DB rows are gone; orphaned files are logged for cleanup, not
        # surfaced as a client error.
        logger.exception(
            "failed to delete stored audio",
            extra={"generation_id": str(generation_id)},
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
