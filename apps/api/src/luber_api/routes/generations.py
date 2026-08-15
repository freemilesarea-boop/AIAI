"""Generation endpoints: POST/GET/LIST/DELETE /v1/generations.

Routes contain no business logic — they translate HTTP to repository/
enqueuer calls. Generation execution happens in the worker behind the
queue boundary.
"""

from __future__ import annotations

import json
import logging
import math
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
    COVER_STRENGTH_TO_ADHERENCE,
    ENGINE_LATENT_FRAME_SECONDS,
    EXTENSION_TOTAL_MAX_SECONDS,
    MIN_PRESERVED_SECONDS,
    AdvisoryResponse,
    BulkIdsRequest,
    BulkProjectRequest,
    BulkResultResponse,
    CoverGenerationRequest,
    CreatedGeneration,
    ExpectedLineResponse,
    ExtendGenerationRequest,
    GenerationCreateRequest,
    GenerationCreateResponse,
    GenerationListResponse,
    GenerationQARequest,
    GenerationQAResponse,
    GenerationResponse,
    GenerationUpdateRequest,
    LineageNode,
    LineageResponse,
    LongFormQAResponse,
    LyricLineQAEntry,
    PreflightRequest,
    PreflightResponse,
    ReplaceRangeRequest,
    SectionSummary,
)
from luber_audio_utils import ASSET_FORMAT_CONTRACT, AudioStorage, AudioStorageError
from luber_database import GenerationHasDescendantsError, GenerationRepository
from luber_database.models.generation import AudioAsset, Generation, GenerationQA, LyricLineQA
from luber_generation_client import GENERATION_QUEUE_NAME
from luber_schemas import (
    FULL_SONG_THRESHOLD_SECONDS,
    Advisory,
    AssetType,
    EditKind,
    ErrorCode,
    GenerationStatus,
    LineVerdict,
    classify_operation,
    estimate_syllables,
    expected_lyric_lines,
    parse_structure,
    preflight,
    select_delivery_master,
    select_raw_master,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/generations", tags=["generations"])

IDEMPOTENCY_KEY_MAX_LENGTH = 200

#: Characters no mainstream filesystem accepts in a name, plus control
#: characters. Everything else — including Korean — is kept, because the
#: point of this filename is that a human recognises the track in their
#: downloads folder.
_FILENAME_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]+')
_FILENAME_WHITESPACE = re.compile(r"\s+")
_FILENAME_DOT_RUN = re.compile(r"\.{2,}")
_EXTENSION_STRIP = re.compile(r"[^a-z0-9]+")
DOWNLOAD_FILENAME_PREFIX = "LUBER - "
DOWNLOAD_FILENAME_MAX_LENGTH = 60
#: Signed download URLs are deliberately short-lived: long enough for a
#: browser to start the transfer, short enough that a leaked URL expires.
SIGNED_URL_TTL_SECONDS = 300

#: Response fields stored as JSON text on the ORM row; decoded rather
#: than read straight off the model.
_DECODED_JSON_FIELDS = frozenset({"advisories", "request_trace"})


class AudioAssetKind(StrEnum):
    """Which delivery asset a client is asking for.

    A client asks for "master" and gets whichever master it should have.
    Phase 14B introduced a second one, and the public vocabulary
    deliberately did not change: the finished/raw distinction is an
    internal storage concern, and asking callers to know about it would
    put the choice in the hands of whoever knew least about it.
    """

    MASTER = "master"
    PREVIEW = "preview"


def resolve_requested_asset(assets: list[AudioAsset], kind: AudioAssetKind) -> AudioAsset | None:
    """The stored row backing a requested asset kind.

    ``master`` resolves through the shared delivery selector, so the
    finished master is served when one exists and the raw is served when
    it does not.
    """
    if kind is AudioAssetKind.PREVIEW:
        return next((a for a in assets if a.asset_type == AssetType.PREVIEW.value), None)
    return select_delivery_master(assets)


def build_download_filename(title: str, generation_id: uuid.UUID, extension: str = "wav") -> str:
    """Human-readable download name: ``LUBER - Midnight Window.wav``.

    User input never reaches the filesystem — this only labels the
    download — but it does reach the *user's* filesystem, so every
    character a mainstream OS rejects is removed, runs of dots are
    collapsed (no ``..`` can survive), and the length is bounded.

    Unicode is deliberately preserved. Phase 3 slugged titles to ASCII,
    which turned every Korean title into ``luber-track-1a2b3c4d`` — the
    opposite of recognisable for this product's main audience. Starlette
    emits a ``filename*=utf-8''`` header for non-ASCII names, which every
    current browser decodes. A title that sanitises to nothing still
    falls back to a stable generation-derived name.

    The extension comes from the asset's format contract, never from
    user input, and is scrubbed regardless.
    """
    cleaned = _FILENAME_UNSAFE.sub(" ", title)
    cleaned = _FILENAME_WHITESPACE.sub(" ", cleaned).strip()
    cleaned = _FILENAME_DOT_RUN.sub(".", cleaned).strip(". ")
    cleaned = cleaned[:DOWNLOAD_FILENAME_MAX_LENGTH].strip(". ")
    if not cleaned:
        cleaned = f"track-{generation_id.hex[:8]}"
    safe_extension = _EXTENSION_STRIP.sub("", extension.lower()) or "bin"
    return f"{DOWNLOAD_FILENAME_PREFIX}{cleaned}.{safe_extension}"


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
            return await _replay_response(repository, existing)

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

    advisory_json = json.dumps([advisory.to_dict() for advisory in advisories], ensure_ascii=False)
    # Every result of one CREATE shares this. It is application metadata:
    # the provider is never told, and each sibling below is a fully
    # independent generation — its own row, job, seed, status and asset.
    group_id = uuid.uuid4()
    created: list[Generation] = []

    # Checked before anything is queued. A generation that reaches the
    # worker naming a reference that does not exist can only fail there,
    # after the user has been told it started.
    if payload.reference_audio_id is not None:
        reference = await repository.get_reference_audio(payload.reference_audio_id)
        if reference is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="That reference track does not exist. Upload it again.",
            )

    for index in range(payload.result_count):
        try:
            generation = await repository.create_generation(
                title=payload.title,
                prompt=payload.prompt,
                lyrics=payload.lyrics,
                vocal_gender=payload.vocal_gender.value,
                duration_requested=payload.duration,
                reference_audio_id=payload.reference_audio_id,
                # A pinned seed applies to the first result only. Giving
                # every sibling the same seed would ask the engine for the
                # same song twice, which defeats the purpose of asking for
                # alternatives. The rest get engine-chosen seeds, and the
                # seed each one actually used is recorded on completion.
                seed=payload.seed if index == 0 else None,
                language=payload.language,
                instrumental=payload.instrumental,
                bpm=payload.bpm,
                key_scale=payload.key_scale,
                time_signature=payload.time_signature,
                advisories=advisory_json,
                parent_generation_id=payload.parent_generation_id,
                variation_label=payload.variation_label,
                generation_group_id=group_id,
                status=GenerationStatus.QUEUED.value,
                # Only the first result carries the caller's key. The
                # column is unique, and a replay resolves the whole group
                # through this row — so siblings need no derived key, and
                # no key can be truncated into a collision.
                idempotency_key=idempotency_key if index == 0 else None,
            )
        except IntegrityError:
            # Concurrent duplicate: the unique index won the race — return
            # the winner's generation instead of creating a second one.
            if idempotency_key is None or index > 0:
                raise
            existing = await repository.get_by_idempotency_key(idempotency_key)
            if existing is None:
                raise
            return await _replay_response(repository, existing)

        await repository.create_job(
            generation.id,
            queue_name=GENERATION_QUEUE_NAME,
            status=GenerationStatus.QUEUED.value,
        )
        created.append(generation)

    accepted: list[CreatedGeneration] = []
    for generation in created:
        try:
            await enqueuer.enqueue(generation.id)
            accepted.append(
                CreatedGeneration(
                    generation_id=generation.id,
                    status=GenerationStatus(generation.status),
                    seed=generation.seed,
                )
            )
        except Exception as exc:
            # One sibling failing to reach the queue does not invalidate
            # the other. The failure is recorded on the row it belongs to
            # and reported in place, so the user sees one result working
            # and one broken rather than losing both.
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
            accepted.append(
                CreatedGeneration(
                    generation_id=generation.id,
                    status=GenerationStatus.FAILED,
                    seed=generation.seed,
                )
            )

    if all(item.status is GenerationStatus.FAILED for item in accepted):
        # Nothing was queued at all — this is an outage, not a partial
        # result, and the caller should be told with a status code.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorCode.QUEUE_FAILED.value,
        )

    return GenerationCreateResponse(
        generation_id=accepted[0].generation_id,
        status=accepted[0].status,
        advisories=to_advisory_responses(advisories),
        generation_group_id=group_id,
        generations=accepted,
    )


async def _replay_response(
    repository: GenerationRepository, existing: Generation
) -> GenerationCreateResponse:
    """Answer a repeated Idempotency-Key with what was already created.

    A replayed two-result submission must return *both* songs, not just
    the one that happens to carry the key — otherwise a retrying client
    would lose track of its second result and could submit it again.
    """
    siblings = (
        await repository.list_generations_in_group(existing.generation_group_id)
        if existing.generation_group_id is not None
        else [existing]
    ) or [existing]
    return GenerationCreateResponse(
        generation_id=existing.id,
        status=GenerationStatus(existing.status),
        advisories=decode_advisories(existing.advisories),
        generation_group_id=existing.generation_group_id,
        generations=[
            CreatedGeneration(
                generation_id=row.id,
                status=GenerationStatus(row.status),
                seed=row.seed,
            )
            for row in siblings
        ],
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

    record = resolve_requested_asset(list(generation.audio_assets), asset)
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


async def _editable_parent(
    generation_id: uuid.UUID,
    repository: GenerationRepository,
    storage: AudioStorage,
    caller: uuid.UUID | None,
    *,
    verb: str,
) -> tuple[Generation, float]:
    """Resolve a generation that may serve as the source of an audio edit.

    Shared by every editing route so the preconditions cannot drift apart:
    an edit needs a readable master, and finding out after the user has
    queued and waited is a worse experience than a 409 now.

    Returns the parent and its recorded master duration. That duration is
    a *screening* value only — the boundary the engine receives is
    measured from the audio itself in the worker.
    """
    parent = await repository.get_generation(generation_id)
    # Same answer for "no such generation" and "not yours", matching the
    # rule this module already applies to audio reads and lineage.
    if parent is None or not caller_may_access(parent, caller):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="generation not found")

    if parent.status != GenerationStatus.COMPLETED.value:
        raise HTTPException(status_code=409, detail=f"only a completed song can be {verb}")

    # The raw master: an edit is fed back into the model, and the model
    # must not be given audio the finishing engine has already shaped.
    master = select_raw_master(list(parent.audio_assets))
    if master is None:
        raise HTTPException(status_code=409, detail="this song has no master audio")
    if not await storage.exists(master.storage_key):
        raise HTTPException(status_code=409, detail="this song's audio is unavailable")

    source_seconds = float(master.duration or parent.duration_actual or 0.0)
    if source_seconds <= 0:
        raise HTTPException(status_code=409, detail="this song's audio is unavailable")
    return parent, source_seconds


@router.post(
    "/{generation_id}/cover",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=GenerationCreateResponse,
)
async def cover_generation(
    generation_id: uuid.UUID,
    payload: CoverGenerationRequest,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
    enqueuer: Annotated[GenerationEnqueuer, Depends(get_enqueuer)],
    storage: Annotated[AudioStorage, Depends(get_audio_storage)],
    caller: Annotated[uuid.UUID | None, Depends(get_caller_user_id)],
) -> GenerationCreateResponse:
    """Create a new performance of a song in a different style.

    The engine regenerates the whole performance, steered by a semantic
    sketch of the source — it does not keep the original recording, which
    is why this is a cover and not a remix. Calibration measured the
    result as demonstrably derived from the source (structure agreement
    0.50 against 0.27 for an unrelated song) at the two settings the
    product offers, and at nothing below them.

    Lyrics and musical metadata are inherited; the target style is what
    the user supplies. No engine vocabulary appears in this contract.
    """
    parent, source_seconds = await _editable_parent(
        generation_id, repository, storage, caller, verb="covered"
    )
    adherence = COVER_STRENGTH_TO_ADHERENCE[payload.strength]

    child = await repository.create_generation(
        title=parent.title,
        # The target style is the point of the operation, so it replaces
        # the parent's brief rather than being appended to it.
        prompt=payload.prompt,
        # Inherited verbatim: LUBER has no lyric-to-time alignment and so
        # cannot honestly offer to change the words.
        lyrics=parent.lyrics,
        vocal_gender=parent.vocal_gender,
        # The engine uses the source as its canvas, so the result keeps
        # the source's length.
        duration_requested=math.ceil(source_seconds),
        seed=None,
        language=parent.language,
        instrumental=parent.instrumental,
        bpm=parent.bpm,
        key_scale=parent.key_scale,
        time_signature=parent.time_signature,
        parent_generation_id=parent.id,
        generation_group_id=uuid.uuid4(),
        edit_kind=EditKind.COVER.value,
        # No time range: a cover regenerates everything.
        source_adherence=adherence,
        status=GenerationStatus.QUEUED.value,
    )
    return await _queue_edit_child(child, parent, repository, enqueuer)


@router.post(
    "/{generation_id}/replace-range",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=GenerationCreateResponse,
)
async def replace_generation_range(
    generation_id: uuid.UUID,
    payload: ReplaceRangeRequest,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
    enqueuer: Annotated[GenerationEnqueuer, Depends(get_enqueuer)],
    storage: Annotated[AudioStorage, Depends(get_audio_storage)],
    caller: Annotated[uuid.UUID | None, Depends(get_caller_user_id)],
) -> GenerationCreateResponse:
    """Regenerate one interior span of a song and keep the rest.

    Real inpainting: the worker uploads the parent's master and the engine
    re-imposes the source outside the chosen span at every diffusion step,
    so the audio before and after it is the original recording. The song
    keeps its length.

    Boundaries land on the engine's 0.04s latent grid, so the span is
    approximate to within a frame. Nothing here is described to the client
    in the engine's terms.
    """
    parent, source_seconds = await _editable_parent(
        generation_id, repository, storage, caller, verb="edited"
    )

    if payload.end_seconds > source_seconds + ENGINE_LATENT_FRAME_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the range ends after the song does "
                f"({payload.end_seconds:.2f}s > {source_seconds:.2f}s)"
            ),
        )
    end_seconds = min(payload.end_seconds, source_seconds)
    preserved = source_seconds - (end_seconds - payload.start_seconds)
    if preserved < MIN_PRESERVED_SECONDS:
        # Replacing (almost) everything is a regeneration, not an edit.
        raise HTTPException(
            status_code=422,
            detail=(
                f"a replacement must leave at least {MIN_PRESERVED_SECONDS}s of the original song"
            ),
        )

    child = await repository.create_generation(
        title=parent.title,
        # An optional re-description steers the regenerated span. The
        # parent's own brief is the default, because it is what produced
        # the audio being kept on either side.
        prompt=payload.prompt or parent.prompt,
        # Lyrics are inherited verbatim: LUBER has no lyric-to-time
        # alignment, so it cannot honestly offer to change the words of
        # one section.
        lyrics=parent.lyrics,
        vocal_gender=parent.vocal_gender,
        # The song keeps its length.
        duration_requested=math.ceil(source_seconds),
        seed=None,
        language=parent.language,
        instrumental=parent.instrumental,
        bpm=parent.bpm,
        key_scale=parent.key_scale,
        time_signature=parent.time_signature,
        parent_generation_id=parent.id,
        generation_group_id=uuid.uuid4(),
        edit_kind=EditKind.REPLACE_RANGE.value,
        edit_start_seconds=payload.start_seconds,
        edit_end_seconds=end_seconds,
        status=GenerationStatus.QUEUED.value,
    )
    return await _queue_edit_child(child, parent, repository, enqueuer)


@router.post(
    "/{generation_id}/extend",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=GenerationCreateResponse,
)
async def extend_generation(
    generation_id: uuid.UUID,
    payload: ExtendGenerationRequest,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
    enqueuer: Annotated[GenerationEnqueuer, Depends(get_enqueuer)],
    storage: Annotated[AudioStorage, Depends(get_audio_storage)],
    caller: Annotated[uuid.UUID | None, Depends(get_caller_user_id)],
) -> GenerationCreateResponse:
    """Append newly generated music to the end of an existing song.

    The child is a real audio edit: the worker uploads the parent's master
    to the engine and regenerates only the range past its end, so the
    original audio is preserved by the model rather than spliced on by us.
    Nothing about that mechanism appears in this contract — the client
    asks for seconds.

    The parent's brief, lyrics and musical controls are inherited. Asking
    someone to retype a song's whole description in order to make it
    longer would be a worse product for no gain, and the inherited values
    are exactly what conditioned the audio being continued.
    """
    parent, source_seconds = await _editable_parent(
        generation_id, repository, storage, caller, verb="extended"
    )
    total_seconds = source_seconds + payload.seconds
    if total_seconds > EXTENSION_TOTAL_MAX_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"extending by {payload.seconds}s would exceed the "
                f"{EXTENSION_TOTAL_MAX_SECONDS}s maximum song length"
            ),
        )

    child = await repository.create_generation(
        title=parent.title,
        prompt=parent.prompt,
        lyrics=parent.lyrics,
        vocal_gender=parent.vocal_gender,
        # Rounded up so the requested seconds are never silently short.
        duration_requested=math.ceil(total_seconds),
        # A new seed: reusing the parent's would ask the engine to
        # re-derive the same material for the new section.
        seed=None,
        language=parent.language,
        instrumental=parent.instrumental,
        bpm=parent.bpm,
        key_scale=parent.key_scale,
        time_signature=parent.time_signature,
        parent_generation_id=parent.id,
        # Its own group: a group is the set of results from one CREATE,
        # and an extension is a different action. Lineage is what ties it
        # to the parent.
        generation_group_id=uuid.uuid4(),
        edit_kind=EditKind.EXTEND.value,
        edit_start_seconds=source_seconds,
        edit_end_seconds=total_seconds,
        status=GenerationStatus.QUEUED.value,
    )
    return await _queue_edit_child(child, parent, repository, enqueuer)


async def _queue_edit_child(
    child: Generation,
    parent: Generation,
    repository: GenerationRepository,
    enqueuer: GenerationEnqueuer,
) -> GenerationCreateResponse:
    """Create the job, enqueue it, and answer like any other submission.

    Shared by every editing route: an edit that cannot be queued must be
    marked failed rather than left QUEUED forever, and the client should
    see the same shape a CREATE returns so the existing queue UI needs no
    special case.
    """
    await repository.create_job(
        child.id,
        queue_name=GENERATION_QUEUE_NAME,
        status=GenerationStatus.QUEUED.value,
    )
    try:
        await enqueuer.enqueue(child.id)
    except Exception as exc:
        logger.exception(
            "failed to enqueue audio edit",
            extra={"generation_id": str(child.id), "parent_id": str(parent.id)},
        )
        await repository.mark_failed(
            child.id,
            status=GenerationStatus.FAILED.value,
            error_code=ErrorCode.QUEUE_FAILED.value,
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorCode.QUEUE_FAILED.value,
        ) from exc

    refreshed = await repository.get_generation(child.id)
    assert refreshed is not None
    return GenerationCreateResponse(
        generation_id=child.id,
        status=GenerationStatus(refreshed.status),
        advisories=[],
        generation_group_id=child.generation_group_id,
        generations=[
            CreatedGeneration(
                generation_id=child.id,
                status=GenerationStatus(refreshed.status),
                seed=child.seed,
            )
        ],
    )


@router.get("/groups/{group_id}", response_model=GenerationListResponse)
async def list_group_generations(
    group_id: uuid.UUID,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> GenerationListResponse:
    """Every song produced by one CREATE.

    This is what makes a two-result submission survive a page refresh:
    the client can re-read the whole group from one id instead of holding
    both generation ids in transient state.
    """
    generations = await repository.list_generations_in_group(group_id)
    if not generations:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="group not found")
    return GenerationListResponse(
        items=[serialize_generation(g) for g in generations],
        total=len(generations),
        limit=len(generations),
        offset=0,
    )


@router.patch("/{generation_id}", response_model=GenerationResponse)
async def update_generation(
    generation_id: uuid.UUID,
    payload: GenerationUpdateRequest,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> GenerationResponse:
    """Edit presentation metadata: the title, and whether it is a favourite.

    Nothing else is editable, and the request model forbids unknown keys
    so an attempt to revise the prompt, lyrics, seed or model is a 422
    rather than a silent no-op. Those fields record what was actually
    generated; changing them would make the library disagree with the
    audio it describes. Duplicate the settings and generate again instead.
    """
    try:
        generation = await repository.update_generation_metadata(
            generation_id,
            title=payload.title,
            favorite=payload.favorite,
        )
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="generation not found"
        ) from None
    refreshed = await repository.get_generation(generation.id)
    assert refreshed is not None
    return serialize_generation(refreshed)


@router.post("/bulk-delete", response_model=BulkResultResponse)
async def bulk_delete_generations(
    payload: BulkIdsRequest,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
    storage: Annotated[AudioStorage, Depends(get_audio_storage)],
) -> BulkResultResponse:
    """Delete several generations, reporting how many actually existed.

    Ids that are already gone are skipped rather than failing the whole
    request: the user's intent ("remove these") is satisfied either way,
    and a partially stale selection is normal in a list they have had
    open for a while.
    """
    affected = 0
    blocked = 0
    for generation_id in payload.ids:
        if await repository.get_generation(generation_id) is None:
            continue
        # DB first, storage second — the same order the single delete
        # uses, so a storage failure can never leave a row pointing at
        # audio that has been removed.
        try:
            await repository.delete_generation(generation_id)
        except GenerationHasDescendantsError:
            # Skipped for the same reason a stale id is skipped: the rest
            # of the selection is still deletable, and the count tells the
            # user how many actually went. Silently destroying a lineage
            # to satisfy a bulk action would be the worse answer.
            blocked += 1
            continue
        affected += 1
        try:
            await storage.delete_generation_audio(generation_id)
        except AudioStorageError:
            logger.exception(
                "failed to delete stored audio",
                extra={"generation_id": str(generation_id)},
            )
    return BulkResultResponse(affected=affected, blocked=blocked)


@router.post("/bulk-project", response_model=BulkResultResponse)
async def bulk_assign_project(
    payload: BulkProjectRequest,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> BulkResultResponse:
    """File several generations under a project, or unfile them with ``null``."""
    if payload.project_id is not None and await repository.get_project(payload.project_id) is None:
        raise HTTPException(status_code=422, detail="project not found")
    affected = await repository.bulk_set_project(payload.ids, payload.project_id)
    return BulkResultResponse(affected=affected)


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

    try:
        await repository.delete_generation(generation_id)
    except GenerationHasDescendantsError as exc:
        # Nothing has been touched at this point: the repository checks
        # for descendants before deleting any asset row, so a refusal
        # leaves the generation and its lineage exactly as they were.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": ErrorCode.GENERATION_HAS_DERIVED_VERSIONS.value,
                "message": "This version has derived versions. Delete those first.",
                "derived_count": exc.descendant_count,
            },
        ) from exc
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


def serialize_lineage_node(generation: Generation) -> LineageNode:
    """Project a row down to what version history needs.

    The operation is computed by the shared classifier rather than read
    off ``edit_kind``, so the stored ``REPLACE_RANGE`` becomes
    ``REPLACE_SECTION`` here and nowhere else has to know.
    """
    return LineageNode(
        id=generation.id,
        parent_generation_id=generation.parent_generation_id,
        title=generation.title,
        status=generation.status,
        operation=classify_operation(
            parent_generation_id=generation.parent_generation_id,
            edit_kind=generation.edit_kind,
        ).value,
        created_at=generation.created_at,
        duration_actual=generation.duration_actual,
        cover_art_url=generation.cover_art_url,
        edit_start_seconds=generation.edit_start_seconds,
        edit_end_seconds=generation.edit_end_seconds,
    )


@router.get("/{generation_id}/lineage", response_model=LineageResponse)
async def get_generation_lineage(
    generation_id: uuid.UUID,
    repository: Annotated[GenerationRepository, Depends(get_repository)],
) -> LineageResponse:
    """Where a generation came from and what came from it.

    Deliberately called lineage rather than "variations": on this
    provider path nothing is audio-to-audio, so a child is a
    re-generation that recorded its origin, not a mutation of its
    parent's audio. The UI must not imply otherwise.
    """
    generation = await repository.get_generation(generation_id)
    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="generation not found")

    parent = None
    if generation.parent_generation_id is not None:
        parent_row = await repository.get_generation(generation.parent_generation_id)
        if parent_row is not None:
            parent = serialize_generation(parent_row)

    children = await repository.list_children(generation_id)

    # The whole bounded tree, so version history is one request rather
    # than a walk. Ancestry gives the root; descendants of that root give
    # every sibling branch, which is what makes the current generation
    # locatable inside its family instead of only its own line.
    ancestry = await repository.get_ancestry(generation_id)
    root = ancestry[-1] if ancestry else generation
    nodes = [root, *await repository.get_descendants(root.id)]

    return LineageResponse(
        generation_id=generation_id,
        parent=parent,
        children=[serialize_generation(child) for child in children],
        root_generation_id=root.id,
        current_generation_id=generation_id,
        nodes=[serialize_lineage_node(node) for node in nodes],
    )
