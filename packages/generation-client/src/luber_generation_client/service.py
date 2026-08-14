"""GenerationService — orchestrates the generation lifecycle.

Runs inside the generation worker in production (the API process never
executes providers; it only persists and enqueues). The service walks
the full status lifecycle even though the mock provider is instant, so
the architecture already matches the real GPU path:

    QUEUED → STARTING → GENERATING → POST_PROCESSING → UPLOADING →
    COMPLETED | FAILED

All failures are translated to standard :class:`ErrorCode` values;
raw exception strings never reach clients (they are stored in
``error_message`` for operators only).
"""

from __future__ import annotations

import json
import logging
import tempfile
from contextlib import AsyncExitStack
from pathlib import Path
from uuid import UUID

from luber_audio_utils import (
    AudioProcessingError,
    AudioStorage,
    AudioStorageError,
    WavValidationError,
    inspect_wav,
    probe_audio,
)
from luber_database import GenerationRepository
from luber_database.models.generation import Generation
from luber_generation_client.editing import (
    AudioEditingProvider,
    AudioEditKind,
    AudioEditRequest,
)
from luber_generation_client.errors import GenerationProviderError
from luber_generation_client.postprocess import produce_delivery_assets
from luber_generation_client.provider import GenerationRequest, MusicGenerationProvider
from luber_schemas import AssetType, ErrorCode, GenerationStatus, VocalGender

logger = logging.getLogger(__name__)


class GenerationService:
    def __init__(
        self,
        repository: GenerationRepository,
        provider: MusicGenerationProvider,
        storage: AudioStorage,
    ) -> None:
        self._repository = repository
        self._provider = provider
        self._storage = storage

    async def _resolve_source_audio(self, parent: Generation, stack: AsyncExitStack) -> Path:
        """Get the parent's MASTER onto local disk for the provider.

        Uses the storage backend's local path when there is one, and
        otherwise materialises the object into a temp file that the
        caller's exit stack removes. Object storage has no path, so the
        edit path cannot assume one exists.
        """
        master = next(
            (a for a in parent.audio_assets if a.asset_type == AssetType.MASTER.value), None
        )
        if master is None:
            raise GenerationProviderError(
                "source generation has no master audio",
                error_code=ErrorCode.INVALID_AUDIO,
            )

        local = self._storage.local_path(master.storage_key)
        if local is not None and local.is_file():
            return local

        data = await self._storage.open(master.storage_key)
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        stack.callback(lambda: Path(handle.name).unlink(missing_ok=True))
        with handle:
            handle.write(data)
        return Path(handle.name)

    async def _to_edit_request(
        self, generation: Generation, stack: AsyncExitStack
    ) -> AudioEditRequest:
        """Build the engine-level edit for an audio-edit generation.

        The regenerated range comes from the *measured* parent audio, not
        from any stored or requested duration: the boundary decides which
        latent frames are preserved, so it has to match the file the
        engine is actually given. A stored duration that drifted by a
        frame would silently move the seam.
        """
        if generation.parent_generation_id is None:
            raise GenerationProviderError(
                "audio edit has no parent generation",
                error_code=ErrorCode.UNKNOWN_GENERATION_ERROR,
            )
        parent = await self._repository.get_generation(generation.parent_generation_id)
        if parent is None:
            raise GenerationProviderError(
                "source generation no longer exists",
                error_code=ErrorCode.UNKNOWN_GENERATION_ERROR,
            )

        source_path = await self._resolve_source_audio(parent, stack)
        measured = probe_audio(source_path).duration_seconds
        # The requested total was computed from a measurement taken at
        # submission time. Re-measuring here keeps the range anchored to
        # the bytes actually being uploaded, and the recorded extension
        # length is what the user asked for.
        extension = (generation.edit_end_seconds or 0.0) - (generation.edit_start_seconds or 0.0)
        if extension <= 0:
            raise GenerationProviderError(
                "audio edit has an empty range",
                error_code=ErrorCode.UNKNOWN_GENERATION_ERROR,
            )

        return AudioEditRequest(
            kind=AudioEditKind(generation.edit_kind or AudioEditKind.REGENERATE_RANGE.value),
            source_audio=source_path,
            start_seconds=measured,
            end_seconds=measured + extension,
            # Conditioning is inherited from the child row, which the API
            # populated from the parent — so an edit is described by the
            # same prompt and lyrics that produced the source.
            title=generation.title,
            prompt=generation.prompt,
            lyrics=generation.lyrics,
            vocal_gender=VocalGender(generation.vocal_gender),
            language=generation.language,
            instrumental=generation.instrumental,
            seed=generation.seed,
            bpm=generation.bpm,
            key_scale=generation.key_scale,
            time_signature=generation.time_signature,
        )

    def _to_provider_request(self, generation: Generation) -> GenerationRequest:
        return GenerationRequest(
            title=generation.title,
            prompt=generation.prompt,
            lyrics=generation.lyrics,
            vocal_gender=VocalGender(generation.vocal_gender),
            duration_seconds=generation.duration_requested,
            seed=generation.seed,
            language=generation.language,
            instrumental=generation.instrumental,
            bpm=generation.bpm,
            key_scale=generation.key_scale,
            time_signature=generation.time_signature,
        )

    async def _record_trace(self, generation_id: UUID, request: GenerationRequest) -> None:
        """Persist what the provider is about to send.

        Best-effort by design: a trace is diagnostic metadata, so
        failing to write it must never fail the generation the user
        actually asked for.
        """
        try:
            trace = self._provider.describe_request(request)
            if not trace:
                return
            await self._repository.record_request_trace(
                generation_id, trace=json.dumps(trace, ensure_ascii=False)
            )
        except Exception:
            logger.warning(
                "could not record provider request trace",
                extra={"generation_id": str(generation_id)},
                exc_info=True,
            )

    async def _record_edit_trace(self, generation_id: UUID, request: AudioEditRequest) -> None:
        """Persist what the editing provider is about to send.

        Same best-effort contract as the generation trace. The provider's
        ``describe_edit`` is responsible for keeping the source's path and
        bytes out of it.
        """
        try:
            provider = self._provider
            if not isinstance(provider, AudioEditingProvider):
                return
            trace = provider.describe_edit(request)
            if not trace:
                return
            await self._repository.record_request_trace(
                generation_id, trace=json.dumps(trace, ensure_ascii=False)
            )
        except Exception:
            logger.warning(
                "could not record provider edit trace",
                extra={"generation_id": str(generation_id)},
                exc_info=True,
            )

    async def execute(
        self, generation_id: UUID, *, worker_id: str | None = None
    ) -> GenerationStatus:
        """Run one generation job to a terminal state. Returns that state."""
        repo = self._repository
        generation = await repo.get_generation(generation_id)
        if generation is None:
            raise LookupError(f"generation not found: {generation_id}")

        job = await repo.get_latest_job(generation_id)
        if job is not None:
            await repo.mark_job_started(
                job.id, status=GenerationStatus.STARTING.value, worker_id=worker_id
            )

        try:
            await repo.mark_started(generation_id, status=GenerationStatus.STARTING.value)

            async with AsyncExitStack() as stack:
                if generation.edit_kind is not None:
                    # An audio edit must reach an editing provider or fail.
                    # Falling back to generate() would quietly hand the
                    # user a new, unrelated song in place of the edit they
                    # asked for — the exact substitution this feature
                    # exists to rule out.
                    if not isinstance(self._provider, AudioEditingProvider):
                        raise GenerationProviderError(
                            "configured provider cannot edit audio",
                            error_code=ErrorCode.MODEL_LOAD_FAILED,
                        )
                    edit_request = await self._to_edit_request(generation, stack)
                    await self._record_edit_trace(generation_id, edit_request)
                    await repo.update_status(generation_id, GenerationStatus.GENERATING.value)
                    result = await self._provider.edit(edit_request)
                else:
                    request = self._to_provider_request(generation)
                    await self._record_trace(generation_id, request)
                    await repo.update_status(generation_id, GenerationStatus.GENERATING.value)
                    result = await self._provider.generate(request)

            # POST_PROCESSING and UPLOADING both happen inside
            # produce_delivery_assets; the status is advanced around it
            # so a stuck run is attributable to the right stage.
            await repo.update_status(generation_id, GenerationStatus.POST_PROCESSING.value)
            # Structural check of the raw model output before spending
            # time transcoding it.
            inspect_wav(result.audio_path)

            await repo.update_status(generation_id, GenerationStatus.UPLOADING.value)
            produced = await produce_delivery_assets(
                generation_id, result.audio_path, self._storage
            )

            for asset in produced.assets:
                await repo.create_audio_asset(
                    generation_id,
                    asset_type=asset.asset_type.value,
                    format=asset.format,
                    mime_type=asset.mime_type,
                    file_extension=asset.file_extension,
                    sample_rate=asset.sample_rate,
                    bit_depth=asset.bit_depth,
                    bitrate=asset.bitrate,
                    channels=asset.channels,
                    duration=asset.duration,
                    storage_key=asset.storage_key,
                    sha256=asset.sha256,
                    file_size=asset.file_size,
                )
            # COMPLETED only after every required asset is stored and
            # recorded — a post-processing or upload failure raises above
            # and lands in the FAILED branch instead.
            await repo.mark_completed(
                generation_id,
                status=GenerationStatus.COMPLETED.value,
                duration_actual=produced.master.duration,
                provider=result.provider,
                model_name=result.model_name,
                model_version=result.model_version,
                seed=result.seed_used,
            )
            if job is not None:
                await repo.mark_job_finished(job.id, status=GenerationStatus.COMPLETED.value)
            logger.info(
                "generation completed",
                extra={
                    "generation_id": str(generation_id),
                    "worker_id": worker_id,
                    "model_version": result.model_version,
                },
            )
            return GenerationStatus.COMPLETED
        except Exception as exc:
            error_code = self._translate_error(exc)
            await repo.mark_failed(
                generation_id,
                status=GenerationStatus.FAILED.value,
                error_code=error_code.value,
                error_message=str(exc),
            )
            if job is not None:
                await repo.mark_job_finished(
                    job.id,
                    status=GenerationStatus.FAILED.value,
                    error_code=error_code.value,
                    error_message=str(exc),
                )
            logger.exception(
                "generation failed",
                extra={
                    "generation_id": str(generation_id),
                    "worker_id": worker_id,
                    "error_code": error_code.value,
                },
            )
            return GenerationStatus.FAILED

    @staticmethod
    def _translate_error(exc: Exception) -> ErrorCode:
        if isinstance(exc, GenerationProviderError):
            return exc.error_code
        if isinstance(exc, WavValidationError):
            return ErrorCode.INVALID_AUDIO
        if isinstance(exc, AudioProcessingError):
            return ErrorCode.ENCODING_FAILED
        if isinstance(exc, AudioStorageError):
            return ErrorCode.UPLOAD_FAILED
        return ErrorCode.UNKNOWN_GENERATION_ERROR
