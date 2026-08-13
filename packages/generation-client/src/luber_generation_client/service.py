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
from uuid import UUID

from luber_audio_utils import (
    AudioProcessingError,
    AudioStorage,
    AudioStorageError,
    WavValidationError,
    inspect_wav,
)
from luber_database import GenerationRepository
from luber_database.models.generation import Generation
from luber_generation_client.errors import GenerationProviderError
from luber_generation_client.postprocess import produce_delivery_assets
from luber_generation_client.provider import GenerationRequest, MusicGenerationProvider
from luber_schemas import ErrorCode, GenerationStatus, VocalGender

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
