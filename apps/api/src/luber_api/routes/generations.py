"""Generation endpoints: POST/GET/LIST/DELETE /v1/generations.

Routes contain no business logic — they translate HTTP to repository/
enqueuer calls. Generation execution happens in the worker behind the
queue boundary.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.exc import IntegrityError

from luber_api.dependencies import get_audio_storage, get_enqueuer, get_repository
from luber_api.jobs import GenerationEnqueuer
from luber_api.schemas import (
    AdvisoryResponse,
    ExpectedLineResponse,
    GenerationCreateRequest,
    GenerationCreateResponse,
    GenerationListResponse,
    GenerationQARequest,
    GenerationQAResponse,
    GenerationResponse,
    LongFormQAResponse,
    LyricLineQAEntry,
    PreflightRequest,
    PreflightResponse,
    SectionSummary,
)
from luber_audio_utils import ASSET_FORMAT_CONTRACT, AudioStorage, AudioStorageError
from luber_database import GenerationRepository
from luber_database.models.generation import Generation, GenerationQA, LyricLineQA
from luber_generation_client import GENERATION_QUEUE_NAME
from luber_schemas import (
    FULL_SONG_THRESHOLD_SECONDS,
    Advisory,
    AssetType,
    ErrorCode,
    GenerationStatus,
    LineVerdict,
    estimate_syllables,
    expected_lyric_lines,
    parse_structure,
    preflight,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/generations", tags=["generations"])

IDEMPOTENCY_KEY_MAX_LENGTH = 200

_FILENAME_STRIP = re.compile(r"[^a-z0-9]+")
DOWNLOAD_FILENAME_MAX_LENGTH = 60
#: Signed download URLs are deliberately short-lived: long enough for a
#: browser to start the transfer, short enough that a leaked URL expires.
SIGNED_URL_TTL_SECONDS = 300

#: Response fields stored as JSON text on the ORM row; decoded rather
#: than read straight off the model.
_DECODED_JSON_FIELDS = frozenset({"advisories", "request_trace"})


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


def decode_advisories(raw: str | None) -> list[AdvisoryResponse]:
    """Parse the stored advisory JSON.

    Tolerant on purpose: advisories are diagnostic, so a row written by
    an older build (or corrupted) degrades to "none recorded" rather
    than turning a readable generation into a 500.
    """
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except ValueError:
        logger.warning("could not decode stored advisories")
        return []
    if not isinstance(decoded, list):
        return []
    parsed: list[AdvisoryResponse] = []
    for item in decoded:
        if not isinstance(item, dict):
            continue
        try:
            parsed.append(AdvisoryResponse.model_validate(item))
        except ValueError:
            continue
    return parsed


def to_advisory_responses(advisories: Iterable[Advisory]) -> list[AdvisoryResponse]:
    """Domain advisories → wire advisories, unchanged in content."""
    return [
        AdvisoryResponse(
            code=advisory.code,
            level=advisory.level.value,
            message=advisory.message,
            detail=dict(advisory.detail),
        )
        for advisory in advisories
    ]


def decode_request_trace(raw: str | None) -> dict[str, object] | None:
    """Parse the stored provider trace, or ``None`` if unavailable.

    The trace is built by the provider's ``describe_request`` and is
    already free of credentials and host details; this only decodes it.
    """
    if not raw:
        return None
    try:
        decoded = json.loads(raw)
    except ValueError:
        logger.warning("could not decode stored request trace")
        return None
    return decoded if isinstance(decoded, dict) else None


def serialize_generation(generation: Generation) -> GenerationResponse:
    """ORM row → wire response, decoding the JSON text columns.

    ``advisories`` and ``request_trace`` are stored as JSON text but
    exposed as structured values, so they are decoded here instead of
    being handed to ``from_attributes`` (which would see the raw string
    and reject it).
    """
    fields = {
        name: getattr(generation, name)
        for name in GenerationResponse.model_fields
        if name not in _DECODED_JSON_FIELDS
    }
    return GenerationResponse.model_validate(
        {
            **fields,
            "advisories": decode_advisories(generation.advisories),
            "request_trace": decode_request_trace(generation.request_trace),
        }
    )


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
    caller: Annotated[uuid.UUID | None, Depends(get_caller_user_id)],
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
                generation_id=existing.id,
                status=GenerationStatus(existing.status),
                advisories=decode_advisories(existing.advisories),
            )

    # Advisory only. Every finding here is recorded and returned, and
    # none of them stops the generation: the user's input is submitted
    # exactly as typed even when the heuristics disagree with it.
    advisories = preflight(
        lyrics=payload.lyrics,
        duration_seconds=payload.duration,
        language=payload.language,
        instrumental=payload.instrumental,
    )

    if payload.parent_generation_id is not None:
        parent = await repository.get_generation(payload.parent_generation_id)
        # Same answer for "no such parent" and "not yours", using the
        # one ownership rule this module already applies to audio reads.
        # Lineage must not become an existence oracle for other people's
        # generations, and it must not let a caller attach their work to
        # a parent they cannot read.
        if parent is None or not caller_may_access(parent, caller):
            raise HTTPException(status_code=422, detail="parent generation not found")

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
            bpm=payload.bpm,
            key_scale=payload.key_scale,
            time_signature=payload.time_signature,
            advisories=json.dumps(
                [advisory.to_dict() for advisory in advisories], ensure_ascii=False
            ),
            parent_generation_id=payload.parent_generation_id,
            variation_label=payload.variation_label,
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
            generation_id=existing.id,
            status=GenerationStatus(existing.status),
            advisories=decode_advisories(existing.advisories),
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
        advisories=to_advisory_responses(advisories),
    )


@router.post("/preflight", response_model=PreflightResponse)
async def preflight_generation(payload: PreflightRequest) -> PreflightResponse:
    """Advisories for a draft, without creating anything.

    Read-only and side-effect free: it touches no database, enqueues no
    job, and never reaches the provider. It exists so the editor can show
    the findings that submitting *would* record, computed by the same
    ``preflight`` the create path uses — one implementation, so the
    editor and the stored advisories can never disagree.

    Nothing here can reject a generation. Findings are advice.
    """
    structure = parse_structure(payload.lyrics)
    advisories = preflight(
        lyrics=payload.lyrics,
        duration_seconds=payload.duration,
        language=payload.language,
        instrumental=payload.instrumental,
    )
    return PreflightResponse(
        advisories=to_advisory_responses(advisories),
        sections=[
            SectionSummary(
                kind=section.kind.value if section.kind is not None else None,
                label=section.label,
                index=section.index,
                line_number=section.line_number,
                line_count=len(section.lines),
                has_content=section.has_content,
                recognised=section.is_recognised,
            )
            for section in structure.sections
        ],
        preamble_line_count=len(structure.preamble),
        estimated_syllables=estimate_syllables(payload.lyrics),
    )


@router.get("/{generation_id}", response_model=GenerationResponse)
async def get_generation(
    generation_id: uuid.UUID,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> GenerationResponse:
    generation = await repository.get_generation(generation_id)
    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="generation not found")
    return serialize_generation(generation)


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
        items=[serialize_generation(g) for g in generations],
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


def _build_qa_response(
    generation: Generation,
    record: GenerationQA | None,
    lines: list[LyricLineQA],
) -> GenerationQAResponse:
    """Assemble the QA view: expected lines plus whatever was recorded.

    ``expected_lines`` is re-derived from the stored lyrics on every
    read, so the review form is always anchored to what was actually
    submitted rather than to a stale snapshot.
    """
    return GenerationQAResponse(
        generation_id=generation.id,
        overall_rating=record.overall_rating if record else None,
        failure_tags=decode_string_list(record.failure_tags) if record else [],
        section_verdicts=decode_string_map(record.section_verdicts) if record else {},
        notes=record.notes if record else None,
        reviewer=record.reviewer if record else None,
        reviewed=record is not None,
        expected_lines=[
            ExpectedLineResponse(index=line.index, section_label=line.section_label, text=line.text)
            for line in expected_lyric_lines(generation.lyrics)
        ],
        lyric_lines=[
            LyricLineQAEntry(
                line_index=row.line_index,
                section_label=row.section_label,
                line_text=row.line_text,
                verdict=LineVerdict(row.verdict),
                note=row.note,
            )
            for row in lines
        ],
    )


@router.get("/{generation_id}/qa", response_model=GenerationQAResponse)
async def get_generation_qa(
    generation_id: uuid.UUID,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> GenerationQAResponse:
    """Human QA for one generation, with the lines to judge against."""
    generation = await repository.get_generation(generation_id)
    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="generation not found")
    record = await repository.get_generation_qa(generation_id)
    lines = await repository.get_lyric_line_qa(generation_id)
    return _build_qa_response(generation, record, lines)


@router.put("/{generation_id}/qa", response_model=GenerationQAResponse)
async def put_generation_qa(
    generation_id: uuid.UUID,
    payload: GenerationQARequest,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> GenerationQAResponse:
    """Record a listener's review. Idempotent: re-reviewing corrects it.

    This endpoint stores judgement, never audio analysis, and it never
    changes the generation itself — a bad review does not alter, hide,
    or reprocess the track it describes.
    """
    generation = await repository.get_generation(generation_id)
    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="generation not found")

    record = await repository.upsert_generation_qa(
        generation_id,
        overall_rating=payload.overall_rating,
        failure_tags=json.dumps([tag.value for tag in payload.failure_tags]),
        section_verdicts=json.dumps(payload.section_verdicts, ensure_ascii=False),
        notes=payload.notes,
        reviewer=payload.reviewer,
    )
    lines = await repository.replace_lyric_line_qa(
        generation_id,
        [
            {
                "line_index": entry.line_index,
                "section_label": entry.section_label,
                "line_text": entry.line_text,
                "verdict": entry.verdict.value,
                "note": entry.note,
            }
            for entry in payload.lyric_lines
        ],
    )
    return _build_qa_response(generation, record, lines)


@router.get("/{generation_id}/longform-qa", response_model=LongFormQAResponse)
async def get_longform_qa(
    generation_id: uuid.UUID,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> LongFormQAResponse:
    """Technical summary for the developer QA view.

    Separate endpoint rather than extra fields on ``GenerationResponse``
    so this diagnostic detail stays out of the listener-facing payload.
    """
    generation = await repository.get_generation(generation_id)
    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="generation not found")

    seconds: float | None = None
    if generation.started_at and generation.completed_at:
        seconds = (generation.completed_at - generation.started_at).total_seconds()
    rtf = (
        seconds / generation.duration_actual
        if seconds is not None and generation.duration_actual
        else None
    )
    structure = parse_structure(generation.lyrics)
    return LongFormQAResponse(
        generation_id=generation.id,
        requested_duration=generation.duration_requested,
        actual_duration=generation.duration_actual,
        sections_requested=len(structure.sections),
        lyric_line_count=len(expected_lyric_lines(generation.lyrics)),
        bpm_requested=generation.bpm,
        key_requested=generation.key_scale,
        time_signature_requested=generation.time_signature,
        generation_seconds=round(seconds, 2) if seconds is not None else None,
        real_time_factor=round(rtf, 3) if rtf is not None else None,
        status=generation.status,
        is_full_song=generation.duration_requested >= FULL_SONG_THRESHOLD_SECONDS,
    )


def decode_string_list(raw: str | None) -> list[str]:
    """Parse a stored JSON list of strings, tolerating a corrupt value."""
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except ValueError:
        logger.warning("could not decode stored string list")
        return []
    return [str(item) for item in decoded] if isinstance(decoded, list) else []


def decode_string_map(raw: str | None) -> dict[str, str]:
    """Parse a stored JSON object of strings, tolerating a corrupt value."""
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except ValueError:
        logger.warning("could not decode stored string map")
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {str(k): str(v) for k, v in decoded.items()}
