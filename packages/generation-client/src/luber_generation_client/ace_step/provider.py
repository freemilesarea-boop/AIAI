"""AceStepProvider — real ACE-Step 1.5 engine behind the provider contract.

GenerationService sees only ``generate(request) → GenerationResult``;
release_task/query_result/status integers stay in this module and the
client. Polling is async (``asyncio.sleep``), cancellation-safe, and
bounded by a hard generation timeout — a hung server can never leave a
LUBER generation in GENERATING forever.
"""

from __future__ import annotations

import asyncio
import logging
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx

from luber_generation_client.ace_step.client import AceStepApiError, AceStepClient
from luber_generation_client.ace_step.compiler import AceStepPromptCompiler
from luber_generation_client.ace_step.types import AceStepQueryResult, AceStepTaskStatus
from luber_generation_client.ace_step.version import ACE_STEP_VERSION
from luber_generation_client.errors import GenerationProviderError
from luber_generation_client.provider import (
    GenerationRequest,
    GenerationResult,
    MusicGenerationProvider,
)
from luber_schemas import ErrorCode

logger = logging.getLogger(__name__)

ACE_STEP_PROVIDER_NAME = "ace_step"


@dataclass(frozen=True)
class AceStepProviderConfig:
    base_url: str
    api_key: str | None = None
    model: str = "acestep-v15-turbo"
    request_timeout: float = 60.0
    generation_timeout: float = 600.0
    poll_interval: float = 2.0
    output_dir: Path = Path("data/raw-model-output")
    inference_steps: int = 8
    thinking: bool = False


class AceStepProvider(MusicGenerationProvider):
    name = ACE_STEP_PROVIDER_NAME

    def __init__(
        self,
        config: AceStepProviderConfig,
        *,
        client: AceStepClient | None = None,
        compiler: AceStepPromptCompiler | None = None,
    ) -> None:
        self._config = config
        self._client = client or AceStepClient(
            config.base_url,
            api_key=config.api_key,
            request_timeout=config.request_timeout,
        )
        self._compiler = compiler or AceStepPromptCompiler()

    async def close(self) -> None:
        await self._client.close()

    def _build_payload(self, request: GenerationRequest) -> dict[str, object]:
        compiled = self._compiler.compile(request)
        logger.info(
            "compiled ACE-Step prompt",
            extra={
                "original_prompt": compiled.original_prompt,
                "compiled_prompt": compiled.prompt,
                "vocal_language": compiled.vocal_language,
                "instrumental": compiled.instrumental,
            },
        )
        payload: dict[str, object] = {
            # Verified upstream fields only (docs/ACE_STEP_UPSTREAM_AUDIT.md)
            "prompt": compiled.prompt,
            "lyrics": compiled.lyrics,
            "vocal_language": compiled.vocal_language,
            "audio_duration": float(request.duration_seconds),
            "audio_format": "wav",
            "model": self._config.model,
            "inference_steps": self._config.inference_steps,
            "thinking": self._config.thinking,
            "batch_size": 1,
        }
        if not self._config.thinking:
            # DiT-only mode: keep the LM out of the request path entirely.
            payload["use_cot_caption"] = False
            payload["use_cot_language"] = False
        if request.seed is not None:
            payload["use_random_seed"] = False
            payload["seed"] = request.seed
        # Musical metadata. Verified at the pinned commit to reach the
        # DiT via ``dit_generate_kwargs`` in acestep/inference.py — the
        # ``user_metadata`` block a few lines above it is LM-only, but
        # these three are forwarded whether or not the LM runs. Omitted
        # entirely when unset: upstream treats "" as "not specified",
        # and sending an empty string would be indistinguishable from a
        # deliberate choice in the trace.
        if request.bpm is not None:
            payload["bpm"] = request.bpm
        if request.key_scale:
            payload["key_scale"] = request.key_scale
        if request.time_signature:
            payload["time_signature"] = request.time_signature
        return payload

    def describe_request(self, request: GenerationRequest) -> dict[str, object]:
        """Sanitized trace of the request. No base_url, no api_key."""
        compiled = self._compiler.compile(request)
        payload = self._build_payload(request)
        return {
            "provider": ACE_STEP_PROVIDER_NAME,
            "model": self._config.model,
            "engine_version": ACE_STEP_VERSION,
            "inference_steps": self._config.inference_steps,
            "original_prompt": compiled.original_prompt,
            "compiled_prompt": compiled.prompt,
            "original_lyrics": request.lyrics,
            "compiled_lyrics": compiled.lyrics,
            "vocal_language": compiled.vocal_language,
            "instrumental": compiled.instrumental,
            "added_conditioning": list(compiled.added_conditioning),
            "skipped_conditioning": list(compiled.skipped_conditioning),
            "payload": payload,
        }

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            health = await self._client.health()
        except (AceStepApiError, httpx.HTTPError) as exc:
            raise GenerationProviderError(
                f"ACE-Step server unreachable or unhealthy: {exc}",
                error_code=ErrorCode.MODEL_LOAD_FAILED,
            ) from exc
        if not health.models_initialized:
            raise GenerationProviderError(
                "ACE-Step server has no initialized model",
                error_code=ErrorCode.MODEL_LOAD_FAILED,
            )

        payload = self._build_payload(request)
        try:
            handle = await self._client.submit_generation(payload)
        except (AceStepApiError, httpx.HTTPError) as exc:
            raise GenerationProviderError(
                f"ACE-Step task submission failed: {exc}",
                error_code=self._classify_upstream_error(str(exc)),
            ) from exc

        result = await self._poll_until_terminal(handle.task_id)
        if result.status is AceStepTaskStatus.FAILED:
            message = result.error_message or "ACE-Step task failed"
            raise GenerationProviderError(
                f"ACE-Step generation failed: {message}",
                error_code=self._classify_upstream_error(message),
            )
        if not result.tracks:
            raise GenerationProviderError(
                "ACE-Step task succeeded but returned no audio tracks",
                error_code=ErrorCode.INVALID_AUDIO,
            )

        track = result.tracks[0]
        destination = self._config.output_dir / f"{handle.task_id}.wav"
        try:
            await self._client.download_audio(track.file_url, destination)
        except (AceStepApiError, httpx.HTTPError) as exc:
            raise GenerationProviderError(
                f"ACE-Step audio download failed: {exc}",
                error_code=ErrorCode.INVALID_AUDIO,
            ) from exc

        duration_seconds, sample_rate = self._read_wav_header(destination)
        return GenerationResult(
            audio_path=destination,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            seed_used=track.first_seed() if track.first_seed() is not None else request.seed,
            provider=ACE_STEP_PROVIDER_NAME,
            model_name=track.dit_model or self._config.model,
            model_version=ACE_STEP_VERSION,
        )

    async def _poll_until_terminal(self, task_id: str) -> AceStepQueryResult:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._config.generation_timeout
        while True:
            if loop.time() >= deadline:
                raise GenerationProviderError(
                    f"ACE-Step generation timed out after "
                    f"{self._config.generation_timeout:.0f}s (task {task_id})",
                    error_code=ErrorCode.GENERATION_TIMEOUT,
                )
            try:
                result = await self._client.query_generation(task_id)
            except (AceStepApiError, httpx.HTTPError) as exc:
                raise GenerationProviderError(
                    f"ACE-Step status query failed: {exc}",
                    error_code=ErrorCode.UNKNOWN_GENERATION_ERROR,
                ) from exc
            if result.status is not AceStepTaskStatus.QUEUED_OR_RUNNING:
                return result
            await asyncio.sleep(self._config.poll_interval)

    @staticmethod
    def _read_wav_header(path: Path) -> tuple[float, int]:
        try:
            with wave.open(str(path), "rb") as wav:
                sample_rate = wav.getframerate()
                frames = wav.getnframes()
        except (wave.Error, EOFError, OSError) as exc:
            raise GenerationProviderError(
                f"ACE-Step returned invalid WAV audio: {exc}",
                error_code=ErrorCode.INVALID_AUDIO,
            ) from exc
        if sample_rate <= 0 or frames <= 0:
            raise GenerationProviderError(
                "ACE-Step returned empty WAV audio",
                error_code=ErrorCode.INVALID_AUDIO,
            )
        return frames / sample_rate, sample_rate

    @staticmethod
    def _classify_upstream_error(message: str) -> ErrorCode:
        lowered = message.lower()
        if "out of memory" in lowered or "oom" in lowered or "cuda out" in lowered:
            return ErrorCode.OUT_OF_MEMORY
        if "model" in lowered and ("load" in lowered or "init" in lowered):
            return ErrorCode.MODEL_LOAD_FAILED
        if "timeout" in lowered or "timed out" in lowered:
            return ErrorCode.GENERATION_TIMEOUT
        return ErrorCode.UNKNOWN_GENERATION_ERROR
