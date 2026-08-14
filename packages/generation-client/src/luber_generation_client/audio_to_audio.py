"""Provider contract for source-conditioned generation.

Separate from both :mod:`provider` and :mod:`editing`, because this is a
third operation and conflating it with either would be a lie.

- ``MusicGenerationProvider.generate`` has no source at all.
- ``AudioEditingProvider.edit`` regenerates part of a recording and
  *preserves* the rest — the engine re-imposes the source latents at every
  diffusion step, measured at 0.999 correlation in Phase 13C.
- This contract regenerates **everything** while being steered by a source.
  Nothing is preserved.

Phase 13D established that difference from the pinned engine's own code:
the cover path masks nothing, and the source reaches the model as a 5 Hz
semantic sketch used as context rather than as audio. Routing a cover
through ``edit()`` would claim a preservation guarantee that does not
exist here.

Vocabulary is engine-neutral. ``task_type``, ``src_audio`` and
``audio_cover_strength`` stay inside ``ace_step/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from luber_generation_client.editing import SUPPORTED_SOURCE_SUFFIXES
from luber_generation_client.provider import GenerationResult
from luber_schemas import (
    BPM_MAX,
    BPM_MIN,
    VALID_KEY_SCALES,
    VALID_TIME_SIGNATURE_VALUES,
    VocalGender,
)


class AudioToAudioRequest(BaseModel):
    """Generate a new performance steered by an existing recording.

    ``source_adherence`` is the one control, and it is named for what it
    does rather than for how much it transforms: **higher keeps the result
    closer to the source**. The product's "how much should this change"
    labels are the inverse of it, and that inversion is done once, at the
    API boundary, so no layer below has to remember which way round it is.
    """

    model_config = {"arbitrary_types_allowed": True}

    source_audio: Path
    #: Measured length of the source, supplied by the caller, which has
    #: already probed the file.
    source_duration_seconds: float = Field(gt=0.0)

    #: Target description. This is the real transformation lever —
    #: calibration moved pitch-contour similarity from 0.285 to 0.490
    #: across style targets, far more than the strength dial did.
    prompt: str = Field(min_length=1, max_length=4000)
    lyrics: str = Field(default="", max_length=20000)
    title: str = Field(min_length=1, max_length=200)
    vocal_gender: VocalGender
    language: str | None = Field(default=None, max_length=16)
    instrumental: bool = False
    seed: int | None = None
    bpm: int | None = Field(default=None, ge=BPM_MIN, le=BPM_MAX)
    key_scale: str | None = None
    time_signature: str | None = None

    #: How closely to follow the source, 0.0-1.0. Higher is closer.
    #: Callers must pass a value the provider has evidence for; the
    #: provider rejects anything outside its own validated band rather
    #: than silently clamping.
    source_adherence: float = Field(ge=0.0, le=1.0)

    @field_validator("source_audio")
    @classmethod
    def _readable_source(cls, value: Path) -> Path:
        if not value.is_file():
            raise ValueError(f"source audio does not exist: {value}")
        if value.stat().st_size == 0:
            raise ValueError(f"source audio is empty: {value}")
        if value.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
            raise ValueError(f"unsupported source audio format: {value.suffix!r}")
        return value

    @field_validator("key_scale")
    @classmethod
    def _known_key_scale(cls, value: str | None) -> str | None:
        if not value:
            return None
        if value not in VALID_KEY_SCALES:
            raise ValueError(f"unsupported key_scale: {value!r}")
        return value

    @field_validator("time_signature")
    @classmethod
    def _known_time_signature(cls, value: str | None) -> str | None:
        if not value:
            return None
        if value not in VALID_TIME_SIGNATURE_VALUES:
            raise ValueError(f"unsupported time_signature: {value!r}")
        return value


class AudioToAudioProvider(ABC):
    """An engine that can generate a new performance from existing audio.

    Its own ABC, so ``isinstance`` remains a truthful capability test and
    an engine that can edit but not do this stays valid.
    """

    @abstractmethod
    def supports_audio_to_audio(self) -> bool:
        """Whether this provider *and its loaded model* can do this.

        Must answer from the checkpoint, not from the engine's feature
        list: ACE-Step accepts an unsupported task without complaint and
        returns undefined audio.
        """

    @abstractmethod
    def validated_adherence_range(self) -> tuple[float, float]:
        """The ``source_adherence`` band this provider has evidence for.

        Exists so the product cannot offer a setting nobody measured.
        Outside it the operation may quietly stop being source-conditioned
        at all, which is the failure this whole line of work rules out.
        """

    @abstractmethod
    async def create_from_audio(self, request: AudioToAudioRequest) -> GenerationResult:
        """Run one source-conditioned generation."""

    @abstractmethod
    def describe_audio_to_audio(self, request: AudioToAudioRequest) -> dict[str, object]:
        """Sanitized trace. No credentials, no host, no paths, no bytes."""
