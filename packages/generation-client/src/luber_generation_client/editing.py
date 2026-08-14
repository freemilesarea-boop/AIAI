"""Provider contract for source-audio-conditioned editing.

Separate from :mod:`luber_generation_client.provider` because editing is a
genuinely different operation: it takes existing audio as input and the
engine's output is conditioned on it. A text-to-music request has no
source, and widening ``GenerationRequest`` with optional audio fields
would make every caller carry a concept most of them cannot use.

The contract is deliberately as narrow as the evidence. Phase 13A proved
exactly one editing primitive reachable on the pinned ACE-Step turbo
runtime — masked regeneration over a time range, with the unmasked audio
re-imposed at every diffusion step — so exactly one primitive is
modelled here. Cover, reference conditioning, stem extraction and
persona are *not* declared: an abstract method for a capability no
implementation can honour is a promise the product would then be tempted
to make to users.

Vocabulary is engine-neutral on purpose. ``repainting_start`` and
``repaint_strength`` are ACE-Step's words and stay inside
``ace_step/provider.py``; what crosses this boundary is a time range and
a preservation preference, which any diffusion editor could implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from luber_generation_client.provider import GenerationResult
from luber_schemas import (
    BPM_MAX,
    BPM_MIN,
    VALID_KEY_SCALES,
    VALID_TIME_SIGNATURE_VALUES,
    VocalGender,
)

#: Audio containers a provider may be handed as an edit source. LUBER
#: masters are WAV; the list exists so an unreadable format fails at the
#: contract rather than inside an engine.
SUPPORTED_SOURCE_SUFFIXES = frozenset({".wav", ".flac"})


class AudioEditKind(StrEnum):
    """Editing operations LUBER can ask a provider to perform.

    One member. Adding another means proving another primitive first.
    """

    #: Regenerate the audio inside ``[start_seconds, end_seconds)`` while
    #: the rest of the source is preserved. A range that begins at (or
    #: past) the end of the source therefore *extends* it — the engine
    #: pads the source and generates into the padding.
    REGENERATE_RANGE = "REGENERATE_RANGE"


class AudioEditRequest(BaseModel):
    """One source-audio-conditioned edit, described engine-neutrally.

    ``source_audio`` is a path on the worker's local disk — resolved from
    LUBER's storage abstraction before the request is built, never a
    client-supplied path and never a durable value.
    """

    model_config = {"arbitrary_types_allowed": True}

    kind: AudioEditKind = AudioEditKind.REGENERATE_RANGE
    source_audio: Path

    #: Range to regenerate, in seconds from the start of the source.
    #: Phase 13B only produces ranges that start at the source's end, but
    #: the contract permits any interior range because that is what the
    #: engine primitive actually does.
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)

    #: Conditioning, inherited from the source generation.
    title: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=4000)
    lyrics: str = Field(default="", max_length=20000)
    vocal_gender: VocalGender
    language: str | None = Field(default=None, max_length=16)
    instrumental: bool = False
    seed: int | None = None
    bpm: int | None = Field(default=None, ge=BPM_MIN, le=BPM_MAX)
    key_scale: str | None = None
    time_signature: str | None = None

    #: How strongly the source should be preserved, 0.0-1.0. Higher keeps
    #: more of the original. Named for the intent rather than for any
    #: engine's parameter, and a provider that cannot honour it must
    #: ignore it rather than approximate it.
    preservation: float = Field(default=0.5, ge=0.0, le=1.0)

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

    @field_validator("start_seconds", "end_seconds")
    @classmethod
    def _finite(cls, value: float) -> float:
        # Pydantic accepts inf/nan for float by default; a non-finite
        # boundary would silently become a nonsense latent index.
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("edit boundaries must be finite")
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

    @model_validator(mode="after")
    def _range_is_ordered(self) -> AudioEditRequest:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self

    @property
    def total_seconds(self) -> float:
        """Length of the canvas the engine is expected to return."""
        return self.end_seconds


class AudioEditingProvider(ABC):
    """An engine that can regenerate part of existing audio.

    Implemented alongside :class:`MusicGenerationProvider` by engines that
    support both. Kept as its own ABC so a text-only provider stays valid
    and so ``isinstance`` is a truthful capability test.
    """

    @abstractmethod
    def supports_edit(self, kind: AudioEditKind) -> bool:
        """Whether this provider *and its loaded model* can do ``kind``.

        Must reflect the model actually loaded, not the engine's feature
        list. ACE-Step ships extract/lego/complete, but the turbo
        checkpoint cannot run them, and the HTTP API does not reject the
        attempt — it produces undefined output. A capability probe that
        answered from the codebase rather than the checkpoint would turn
        that into a user-visible bug.
        """

    @abstractmethod
    async def edit(self, request: AudioEditRequest) -> GenerationResult:
        """Run one edit and return the engine's full output canvas."""

    @abstractmethod
    def describe_edit(self, request: AudioEditRequest) -> dict[str, object]:
        """Sanitized trace of the edit. No credentials, no host, no paths."""
