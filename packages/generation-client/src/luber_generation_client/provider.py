"""The MusicGenerationProvider architecture boundary.

Phase 0 establishes this seam so that Phase 1 (MockGenerationProvider)
and Phase 2 (AceStepProvider) plug in without touching business logic.
The request/result models below are the minimal spec-mandated contract;
Phase 1 extends them (seed, language, section parsing, asset metadata)
without breaking this interface.

Rules:
- API and worker business logic depend on this abstract interface only.
- Each concrete provider lives in its own module/service and registers
  behind this contract.
- Providers must not fabricate success: a failed generation raises or
  returns an explicit failure, never a fake audio path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from luber_schemas import (
    BPM_MAX,
    BPM_MIN,
    DURATION_MAX,
    DURATION_MIN,
    VALID_KEY_SCALES,
    VALID_TIME_SIGNATURE_VALUES,
    VocalGender,
)


class GenerationRequest(BaseModel):
    """Provider-agnostic description of one music generation job.

    The musical parameters below (``bpm``, ``key_scale``,
    ``time_signature``) exist because the pinned ACE-Step build accepts
    them and forwards them to the DiT conditioning metadata. Their
    ranges mirror the engine's own validation rather than a house
    style — see ``luber_schemas.songcraft``. A provider that cannot
    honour one of them must ignore it, never approximate it.
    """

    title: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=4000)
    lyrics: str = Field(default="", max_length=20000)
    vocal_gender: VocalGender
    duration_seconds: int = Field(default=180, ge=DURATION_MIN, le=DURATION_MAX)
    seed: int | None = None
    language: str | None = Field(default=None, max_length=16)
    instrumental: bool = False
    bpm: int | None = Field(default=None, ge=BPM_MIN, le=BPM_MAX)
    key_scale: str | None = None
    time_signature: str | None = None

    @field_validator("key_scale")
    @classmethod
    def _known_key_scale(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if value not in VALID_KEY_SCALES:
            raise ValueError(f"unsupported key_scale: {value!r}")
        return value

    @field_validator("time_signature")
    @classmethod
    def _known_time_signature(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if value not in VALID_TIME_SIGNATURE_VALUES:
            raise ValueError(f"unsupported time_signature: {value!r}")
        return value


class GenerationResult(BaseModel):
    """Outcome of a successful provider run.

    ``audio_path`` points at the raw model output on local disk; the
    audio post-processing pipeline (Phase 4) turns it into master
    WAV/MP3 assets. Providers never upload to object storage themselves.
    """

    audio_path: Path
    duration_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    seed_used: int | None = None
    provider: str
    model_name: str
    model_version: str


class MusicGenerationProvider(ABC):
    """Abstract music generation engine.

    Implementations planned:
    - ``MockGenerationProvider`` (Phase 1, CI-safe, returns fixture WAV)
    - ``AceStepProvider``        (Phase 2, real ACE-Step 1.5 engine)
    - future: Stable Audio family, licensed/custom foundation models
    """

    name: str

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run one generation job and return the raw audio result.

        Must raise a provider error (mapped to a standard
        :class:`luber_schemas.ErrorCode`) on failure — never return
        silently invalid audio.
        """
        raise NotImplementedError

    def describe_request(self, request: GenerationRequest) -> dict[str, object]:
        """What this provider *would* send, without sending it.

        Used to persist a request trace before execution, so a failed
        run is as inspectable as a successful one. Implementations must
        keep this free of credentials, hostnames, and local paths: the
        trace is shown to the user.
        """
        return {}
