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
from luber_generation_client.audio_to_audio import (
    AudioToAudioProvider,
    AudioToAudioRequest,
)
from luber_generation_client.editing import (
    AudioEditingProvider,
    AudioEditKind,
    AudioEditRequest,
)
from luber_generation_client.errors import GenerationProviderError
from luber_generation_client.provider import (
    GenerationRequest,
    GenerationResult,
    MusicGenerationProvider,
)
from luber_schemas import DURATION_MIN, ErrorCode

logger = logging.getLogger(__name__)

ACE_STEP_PROVIDER_NAME = "ace_step"

#: Upstream's task name for masked regeneration.
ACE_STEP_REPAINT_TASK = "repaint"

#: Upstream's task name for source-conditioned regeneration.
ACE_STEP_COVER_TASK = "cover"

#: Checkpoints whose task list includes ``cover`` (``TASK_TYPES_TURBO``).
COVER_CAPABLE_MODELS = frozenset({"acestep-v15-turbo"})

#: The ``audio_cover_strength`` band Phase 13D measured as genuinely
#: source-conditioned on this runtime. At 0.75-1.00 structure agreement
#: against the source is ~0.50 versus ~0.27 for an unrelated song; at 0.50
#: it falls to 0.33, i.e. the operation stops being a cover. Nothing
#: outside this band is offered, and the provider refuses it.
#: See benchmarks/remix_cover/README.md.
COVER_ADHERENCE_MIN = 0.75
COVER_ADHERENCE_MAX = 1.0

#: Checkpoints whose task list includes ``repaint``. Read from
#: ``TASK_TYPES_TURBO`` / ``TASK_TYPES_BASE`` in the pinned build: every
#: shipped checkpoint supports repaint, but extract/lego/complete are
#: base-only, so this set exists to be *narrowed* honestly if a
#: repaint-incapable model is ever configured.
REPAINT_CAPABLE_MODELS = frozenset({"acestep-v15-turbo"})


@dataclass(frozen=True)
class AceStepProviderConfig:
    base_url: str
    api_key: str | None = None
    model: str = "acestep-v15-turbo"
    request_timeout: float = 60.0
    #: Floor for the generation timeout. Duration-aware scaling can only
    #: raise the budget above this, never below it.
    generation_timeout: float = 600.0
    #: Fixed part of the duration-aware budget: model warm-up, text
    #: encode, queue wait, download — the costs that do not scale with
    #: requested audio length.
    timeout_base_seconds: float = 300.0
    #: Seconds of budget per requested audio-second. Measured wall clock
    #: is well under 1.0x for long form; 4.0 is a deliberate ~10x margin.
    timeout_multiplier: float = 4.0
    poll_interval: float = 2.0
    output_dir: Path = Path("data/raw-model-output")
    inference_steps: int = 8
    thinking: bool = False


class AceStepProvider(MusicGenerationProvider, AudioEditingProvider, AudioToAudioProvider):
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
            # Reference provenance: which track, proved by digest, and the
            # transport that carried it. Never the local path — it is a
            # temp file on whichever host the worker happened to run on,
            # and durable records must not contain filesystem locations.
            "reference_audio": (
                None
                if request.reference_audio is None
                else {
                    "reference_id": str(request.reference_audio.reference_id),
                    "sha256": request.reference_audio.sha256,
                    "duration_seconds": request.reference_audio.duration_seconds,
                    "transport": "multipart:ref_audio",
                    "engine_field": "reference_audio_path",
                }
            ),
        }

    async def _require_healthy_server(self) -> None:
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

    async def _collect_result(
        self,
        task_id: str,
        *,
        duration_seconds: float,
        fallback_seed: int | None,
    ) -> GenerationResult:
        """Poll one submitted task to a terminal state and fetch its audio.

        Shared by generation and editing: everything after submission is
        identical, and duplicating it would let the two paths drift in
        timeout, error translation or WAV handling.
        """
        timeout = self.timeout_for(duration_seconds)
        logger.info(
            "polling ACE-Step task",
            extra={
                "task_id": task_id,
                "duration_seconds": duration_seconds,
                "timeout_seconds": round(timeout),
            },
        )
        result = await self._poll_until_terminal(task_id, timeout)
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
        destination = self._config.output_dir / f"{task_id}.wav"
        try:
            await self._client.download_audio(track.file_url, destination)
        except (AceStepApiError, httpx.HTTPError) as exc:
            raise GenerationProviderError(
                f"ACE-Step audio download failed: {exc}",
                error_code=ErrorCode.INVALID_AUDIO,
            ) from exc

        actual_duration, sample_rate = self._read_wav_header(destination)
        return GenerationResult(
            audio_path=destination,
            duration_seconds=actual_duration,
            sample_rate=sample_rate,
            seed_used=track.first_seed() if track.first_seed() is not None else fallback_seed,
            provider=ACE_STEP_PROVIDER_NAME,
            model_name=track.dit_model or self._config.model,
            model_version=ACE_STEP_VERSION,
        )

    @property
    def supports_reference_audio(self) -> bool:
        """Verified against the installed runtime, not inferred.

        ACE-Step 6d467e4b accepts a ``ref_audio`` multipart upload on
        ``/release_task``, saves it, and sets ``reference_audio_path`` on
        ``GenerateMusicRequest``. That feeds the timbre encoder, whose
        output is merged into ``encoder_hidden_states`` before the sampler
        runs — so the conditioning is applied on the MLX path even though
        the reference tensor never appears in the MLX sampler itself.
        Phase 13E measured the effect at roughly twenty times the
        seed-only noise floor, in the correct direction for two opposite
        references.
        """
        return True

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        await self._require_healthy_server()

        payload = self._build_payload(request)
        reference = request.reference_audio
        try:
            if reference is None:
                handle = await self._client.submit_generation(payload)
            else:
                # A missing file here is a server-side fault, not a bad
                # request: the service materialised it from storage a
                # moment ago. Refusing beats generating without it.
                if not reference.audio_path.is_file():
                    raise GenerationProviderError(
                        "reference audio is no longer available",
                        error_code=ErrorCode.REFERENCE_AUDIO_UNAVAILABLE,
                    )
                handle = await self._client.submit_generation_with_reference_audio(
                    payload, reference.audio_path
                )
        except (AceStepApiError, httpx.HTTPError) as exc:
            raise GenerationProviderError(
                f"ACE-Step task submission failed: {exc}",
                error_code=self._classify_upstream_error(str(exc)),
            ) from exc

        return await self._collect_result(
            handle.task_id,
            duration_seconds=request.duration_seconds,
            fallback_seed=request.seed,
        )

    # ── audio editing (Phase 13B) ──────────────────────────────────
    #
    # ACE-Step's repaint task regenerates a masked time range while
    # re-imposing the VAE-encoded source outside it at every diffusion
    # step. A range that runs past the end of the source makes upstream
    # pad the source and generate into the padding, which is how an
    # extension is expressed — there is no separate "extend" task.

    def supports_edit(self, kind: AudioEditKind) -> bool:
        """Only range regeneration, and only on a model that can repaint.

        Answered from the configured checkpoint rather than from
        ACE-Step's feature list: extract/lego/complete exist in the code
        but are base-model only, and the HTTP API accepts them against a
        turbo model without complaint, returning undefined audio.
        """
        if kind is not AudioEditKind.REGENERATE_RANGE:
            return False
        return self._config.model in REPAINT_CAPABLE_MODELS

    def _build_edit_payload(self, request: AudioEditRequest) -> dict[str, object]:
        compiled = self._compiler.compile(
            GenerationRequest(
                title=request.title,
                prompt=request.prompt,
                lyrics=request.lyrics,
                vocal_gender=request.vocal_gender,
                # Only prompt/lyrics/vocal/language are read by the
                # compiler; the duration is carried for shape and clamped
                # to the text-to-music floor because an edit's canvas may
                # legitimately be shorter than a generation is allowed to
                # request.
                duration_seconds=max(DURATION_MIN, round(request.total_seconds)),
                seed=request.seed,
                language=request.language,
                instrumental=request.instrumental,
                bpm=request.bpm,
                key_scale=request.key_scale,
                time_signature=request.time_signature,
            )
        )
        payload: dict[str, object] = {
            "task_type": ACE_STEP_REPAINT_TASK,
            "prompt": compiled.prompt,
            "lyrics": compiled.lyrics,
            "vocal_language": compiled.vocal_language,
            # The canvas upstream must return: the source plus whatever
            # the edit range adds beyond it.
            "audio_duration": float(request.total_seconds),
            # Engine words, and they stop here. The domain says
            # start/end seconds; only this module knows they are called
            # repainting_start/repainting_end on the wire.
            "repainting_start": float(request.start_seconds),
            "repainting_end": float(request.end_seconds),
            # "balanced" is the only mode whose strength dial is read;
            # conservative and aggressive ignore it.
            "repaint_mode": "balanced",
            "repaint_strength": float(1.0 - request.preservation),
            "audio_format": "wav",
            "model": self._config.model,
            "inference_steps": self._config.inference_steps,
            "thinking": self._config.thinking,
            "batch_size": 1,
        }
        if not self._config.thinking:
            payload["use_cot_caption"] = False
            payload["use_cot_language"] = False
        if request.seed is not None:
            payload["use_random_seed"] = False
            payload["seed"] = request.seed
        if request.bpm is not None:
            payload["bpm"] = request.bpm
        if request.key_scale:
            payload["key_scale"] = request.key_scale
        if request.time_signature:
            payload["time_signature"] = request.time_signature
        return payload

    def describe_edit(self, request: AudioEditRequest) -> dict[str, object]:
        """Sanitized trace of the edit. No base_url, no api_key, no paths.

        The source is described by size and format only: its path is a
        transient worker detail and its bytes are not diagnostics.
        """
        payload = self._build_edit_payload(request)
        return {
            "provider": ACE_STEP_PROVIDER_NAME,
            "model": self._config.model,
            "engine_version": ACE_STEP_VERSION,
            "operation": "edit",
            "edit_kind": request.kind.value,
            "start_seconds": request.start_seconds,
            "end_seconds": request.end_seconds,
            "preservation": request.preservation,
            "source_audio_bytes": request.source_audio.stat().st_size,
            "source_audio_format": request.source_audio.suffix.lstrip("."),
            "source_audio_transport": "multipart",
            "payload": payload,
        }

    # ── source-conditioned generation (Phase 13D) ──────────────────
    #
    # A different operation from repaint, not a variant of it. Cover
    # masks nothing and preserves nothing: the source is quantised to a
    # 5Hz semantic sketch and used as context while the whole canvas is
    # regenerated. See docs/ACE_STEP_COVER_AUDIT.md.

    def supports_audio_to_audio(self) -> bool:
        return self._config.model in COVER_CAPABLE_MODELS

    def validated_adherence_range(self) -> tuple[float, float]:
        """The band Phase 13D actually measured as source-conditioned.

        Below the floor the engine's own structure agreement falls to the
        level of an unrelated song — the operation stops being a cover
        without saying so. The product must not be able to ask for that.
        """
        return (COVER_ADHERENCE_MIN, COVER_ADHERENCE_MAX)

    def _build_cover_payload(self, request: AudioToAudioRequest) -> dict[str, object]:
        compiled = self._compiler.compile(
            GenerationRequest(
                title=request.title,
                prompt=request.prompt,
                lyrics=request.lyrics,
                vocal_gender=request.vocal_gender,
                duration_seconds=max(DURATION_MIN, round(request.source_duration_seconds)),
                seed=request.seed,
                language=request.language,
                instrumental=request.instrumental,
                bpm=request.bpm,
                key_scale=request.key_scale,
                time_signature=request.time_signature,
            )
        )
        payload: dict[str, object] = {
            "task_type": ACE_STEP_COVER_TASK,
            "prompt": compiled.prompt,
            "lyrics": compiled.lyrics,
            "vocal_language": compiled.vocal_language,
            # Engine words, and they stop here. ``source_adherence`` and
            # ``audio_cover_strength`` share a direction — higher means
            # closer to the source — so this is a rename, not a flip.
            "audio_cover_strength": float(request.source_adherence),
            # Deliberately absent: cover_noise_strength. It is implemented
            # only in the PyTorch sampler and never reaches the MLX path
            # this deployment runs, so sending it would be a control that
            # silently does nothing.
            "audio_format": "wav",
            "model": self._config.model,
            "inference_steps": self._config.inference_steps,
            "thinking": self._config.thinking,
            "batch_size": 1,
        }
        if not self._config.thinking:
            payload["use_cot_caption"] = False
            payload["use_cot_language"] = False
        if request.seed is not None:
            payload["use_random_seed"] = False
            payload["seed"] = request.seed
        if request.bpm is not None:
            payload["bpm"] = request.bpm
        if request.key_scale:
            payload["key_scale"] = request.key_scale
        if request.time_signature:
            payload["time_signature"] = request.time_signature
        # audio_duration is deliberately omitted: for a cover the engine
        # uses the source itself as the canvas and ignores it.
        return payload

    def describe_audio_to_audio(self, request: AudioToAudioRequest) -> dict[str, object]:
        payload = self._build_cover_payload(request)
        return {
            "provider": ACE_STEP_PROVIDER_NAME,
            "model": self._config.model,
            "engine_version": ACE_STEP_VERSION,
            "operation": "cover",
            "source_adherence": request.source_adherence,
            "source_duration_seconds": request.source_duration_seconds,
            "source_audio_bytes": request.source_audio.stat().st_size,
            "source_audio_format": request.source_audio.suffix.lstrip("."),
            "source_audio_transport": "multipart",
            "payload": payload,
        }

    async def create_from_audio(self, request: AudioToAudioRequest) -> GenerationResult:
        if not self.supports_audio_to_audio():
            raise GenerationProviderError(
                f"model {self._config.model!r} cannot generate from audio",
                error_code=ErrorCode.MODEL_LOAD_FAILED,
            )
        low, high = self.validated_adherence_range()
        if not low <= request.source_adherence <= high:
            # Refused rather than clamped: a silently adjusted setting
            # would make the product's own labels wrong.
            raise GenerationProviderError(
                f"source_adherence {request.source_adherence} is outside the "
                f"validated range {low}-{high}",
                error_code=ErrorCode.UNKNOWN_GENERATION_ERROR,
            )
        await self._require_healthy_server()

        payload = self._build_cover_payload(request)
        try:
            handle = await self._client.submit_generation_with_source_audio(
                payload, request.source_audio
            )
        except (AceStepApiError, httpx.HTTPError) as exc:
            raise GenerationProviderError(
                f"ACE-Step cover submission failed: {exc}",
                error_code=self._classify_upstream_error(str(exc)),
            ) from exc

        return await self._collect_result(
            handle.task_id,
            duration_seconds=request.source_duration_seconds,
            fallback_seed=request.seed,
        )

    async def edit(self, request: AudioEditRequest) -> GenerationResult:
        if not self.supports_edit(request.kind):
            raise GenerationProviderError(
                f"model {self._config.model!r} cannot perform {request.kind.value}",
                error_code=ErrorCode.MODEL_LOAD_FAILED,
            )
        await self._require_healthy_server()

        payload = self._build_edit_payload(request)
        try:
            handle = await self._client.submit_generation_with_source_audio(
                payload, request.source_audio
            )
        except (AceStepApiError, httpx.HTTPError) as exc:
            raise GenerationProviderError(
                f"ACE-Step edit submission failed: {exc}",
                error_code=self._classify_upstream_error(str(exc)),
            ) from exc

        return await self._collect_result(
            handle.task_id,
            duration_seconds=request.total_seconds,
            fallback_seed=request.seed,
        )

    def timeout_for(self, duration_seconds: float) -> float:
        """How long to wait for *duration_seconds* of audio before giving up.

        A timeout exists to tell "the provider is dead or hung" apart
        from "this is taking a while". A single flat value cannot do
        that once requests range from 30s to 240s of audio, so the
        budget is ``base + per_audio_second x duration``, floored at the
        configured flat timeout so this can only ever be *more* generous
        than Phase 8 was — no request that used to fit can start failing.

        The coefficients are deliberately loose. Measured on this
        deployment (docs/PHASE9_LONG_FORM_ENGINE_AUDIT.md §6), wall clock
        is roughly flat at 76-96s across 120-240s of audio, so
        ``timeout_multiplier=4.0`` leaves roughly an order of magnitude
        of headroom. That is the point: the timeout is a liveness
        backstop, not a performance budget, and a slow-but-working
        engine must never be reported as a dead one.
        """
        scaled = self._config.timeout_base_seconds + (
            self._config.timeout_multiplier * max(0, duration_seconds)
        )
        return max(self._config.generation_timeout, scaled)

    async def _poll_until_terminal(self, task_id: str, budget_seconds: float) -> AceStepQueryResult:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget_seconds
        while True:
            if loop.time() >= deadline:
                raise GenerationProviderError(
                    f"ACE-Step generation timed out after {budget_seconds:.0f}s (task {task_id})",
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
