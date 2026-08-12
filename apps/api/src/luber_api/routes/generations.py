"""Generation endpoints: POST/GET/LIST/DELETE /v1/generations.

Routes contain no business logic — they translate HTTP to repository/
enqueuer calls. Generation execution happens in the worker behind the
queue boundary.
"""

from __future__ import annotations

import logging
import re
import uuid
from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.exc import IntegrityError

from luber_api.dependencies import get_audio_storage, get_enqueuer, get_repository
from luber_api.jobs import GenerationEnqueuer
from luber_api.schemas import (
    GenerationCreateRequest,
    GenerationCreateResponse,
    GenerationListResponse,
    GenerationResponse,
)
from luber_audio_utils import ASSET_FORMAT_CONTRACT, AudioStorage, AudioStorageError
from luber_database import GenerationRepository
from luber_database.models.generation import Generation
from luber_generation_client import GENERATION_QUEUE_NAME
from luber_schemas import AssetType, ErrorCode, GenerationStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/generations", tags=["generations"])

IDEMPOTENCY_KEY_MAX_LENGTH = 200

_FILENAME_STRIP = re.compile(r"[^a-z0-9]+")
DOWNLOAD_FILENAME_MAX_LENGTH = 60
#: Signed download URLs are deliberately short-lived: long enough for a
#: browser to start the transfer, short enough that a leaked URL expires.
SIGNED_URL_TTL_SECONDS = 300


class AudioAssetKind(StrEnum):
    """Which delivery asset a client is asking for."""

    MASTER = "master"
    PREVIEW = "preview"

    @property
    def asset_type(self) -> str:
        return AssetType.MASTER.value if self is AudioAssetKind.MASTER else AssetType.PREVIEW.value


def build_download_filename(title: str, generation_id: uuid.UUID, extension: str = "wav") -> str:
    """ASCII-safe filename derived from a user-supplied title.

    User input never reaches the filesystem — this only labels the
    download. Non-ASCII titles (e.g. Korean) legitimately slug to
    nothing, so a stable generation-derived name is used instead. The
    extension comes from the asset's format contract, never from input.
    """
    slug = _FILENAME_STRIP.sub("-", title.lower()).strip("-")[:DOWNLOAD_FILENAME_MAX_LENGTH]
    slug = slug.strip("-")
    if not slug:
        slug = f"luber-track-{generation_id.hex[:8]}"
    safe_extension = _FILENAME_STRIP.sub("", extension.lower()) or "bin"
    return f"{slug}.{safe_extension}"


def caller_may_access(generation: Generation, caller: uuid.UUID | None) -> bool:
    """Whether *caller* is allowed to read this generation's assets.

    Generations created before authentication exists carry no owner and
    stay readable, which keeps local development and the Phase 3 flow
    working. Once a generation *is* owned, only that owner may read it —
    so the authorization boundary is real and enforced today, and
    swapping in a full auth system later means changing only how the
    caller identity is derived.
    """
    if generation.user_id is None:
        return True
    return caller is not None and caller == generation.user_id


def get_caller_user_id(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> uuid.UUID | None:
    """Identity of the requesting user, if one was supplied.

    A placeholder for the authentication phase: it establishes *where*
    identity enters the request so the ownership check has something to
    compare against. A malformed value is treated as anonymous rather
    than as an error, so it can never be used to probe for valid ids.
    """
    if not x_user_id:
        return None
    try:
        return uuid.UUID(x_user_id)
    except ValueError:
        return None


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
    caller: Annotated[uuid.UUID | None, Depends(get_caller_user_id)],
    asset: Annotated[AudioAssetKind, Query()] = AudioAssetKind.MASTER,
    download: Annotated[bool, Query()] = False,
) -> Response:
    """Deliver one audio asset to an authorized client.

    Clients address audio by generation id and asset role. Storage keys,
    bucket names, and filesystem paths are never part of the request, so
    no client-supplied value reaches path resolution.

    Where the storage backend can sign URLs (production object storage),
    the response is a redirect to a short-lived signed URL. Where it
    cannot (local development), the backend streams the bytes itself.
    Authorization is identical in both cases and always happens first.
    """
    generation = await repository.get_generation(generation_id)
    if generation is None or not caller_may_access(generation, caller):
        # Same response for "absent" and "not yours": ownership is not
        # an existence oracle.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="generation not found")

    asset_type = asset.asset_type
    record = next(
        (a for a in generation.audio_assets if a.asset_type == asset_type),
        None,
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{asset.value} audio not available for this generation",
        )

    # Serve under the media type recorded for the asset, cross-checked
    # against the format contract so a tampered row cannot cause a
    # MIME/extension mismatch.
    expected = ASSET_FORMAT_CONTRACT.get(record.format)
    if expected is None or (record.mime_type, record.file_extension) != expected:
        logger.error(
            "audio asset has a mime/extension mismatch",
            extra={"generation_id": str(generation_id), "asset_id": str(record.id)},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{asset.value} audio not available for this generation",
        )
    content_type, extension = expected
    filename = build_download_filename(generation.title, generation_id, extension)

    try:
        target = await storage.download_target(
            record.storage_key,
            filename=filename,
            content_type=content_type,
            expires_in_seconds=SIGNED_URL_TTL_SECONDS,
        )
        if target.is_signed_url and target.url is not None:
            # Bytes never transit this process in production.
            return RedirectResponse(url=target.url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)
        path = storage.local_path(record.storage_key)
    except AudioStorageError:
        # A bad storage key is an operator problem; the client learns
        # nothing about paths.
        logger.exception(
            "unsafe or unusable storage key for audio asset",
            extra={"generation_id": str(generation_id), "asset_id": str(record.id)},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{asset.value} audio not available for this generation",
        ) from None

    if path is None or not path.is_file():
        logger.error(
            "audio asset missing from storage",
            extra={"generation_id": str(generation_id), "asset_id": str(record.id)},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{asset.value} audio not available for this generation",
        )

    # FileResponse streams from disk and sets Content-Length itself; the
    # file is never read into memory in full.
    return FileResponse(
        path,
        media_type=content_type,
        filename=filename,
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
