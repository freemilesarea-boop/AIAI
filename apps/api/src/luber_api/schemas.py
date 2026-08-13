"""API request/response DTOs for the generation endpoints.

These are the public wire contract — separate from ORM models and from
the provider-level ``GenerationRequest`` so each layer can evolve
independently.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from luber_schemas import (
    BPM_MAX,
    BPM_MIN,
    DURATION_MAX,
    DURATION_MIN,
    QA_SECTIONS,
    VALID_KEY_SCALES,
    VALID_TIME_SIGNATURE_VALUES,
    FailureTag,
    GenerationStatus,
    LineVerdict,
    VocalGender,
)

#: Largest seed the engine accepts without wrapping. Values are stored
#: in a BIGINT column, so this is the API-level guard.
SEED_MAX = 2**63 - 1


class GenerationCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=4000)
    lyrics: str = Field(default="", max_length=20000)
    vocal_gender: VocalGender
    duration: int = Field(default=180, ge=DURATION_MIN, le=DURATION_MAX)
    seed: int | None = Field(default=None, ge=0, le=SEED_MAX)
    language: str | None = Field(default=None, max_length=16)
    instrumental: bool = False

    # Advanced controls. Each is optional and each is validated against
    # the pinned engine's own accepted values, so an unsupported value
    # is rejected at the edge instead of being silently dropped deeper
    # in the stack.
    bpm: int | None = Field(default=None, ge=BPM_MIN, le=BPM_MAX)
    key_scale: str | None = None
    time_signature: str | None = None

    #: Set when this request came from "generate again" or a variation.
    parent_generation_id: uuid.UUID | None = None
    variation_label: str | None = Field(default=None, max_length=50)

    @field_validator("key_scale")
    @classmethod
    def _known_key_scale(cls, value: str | None) -> str | None:
        if not value:
            return None
        if value not in VALID_KEY_SCALES:
            raise ValueError("unsupported key_scale")
        return value

    @field_validator("time_signature")
    @classmethod
    def _known_time_signature(cls, value: str | None) -> str | None:
        if not value:
            return None
        if value not in VALID_TIME_SIGNATURE_VALUES:
            raise ValueError("unsupported time_signature")
        return value

    @field_validator("variation_label")
    @classmethod
    def _clean_variation_label(cls, value: str | None) -> str | None:
        # Whitespace-only is the same as absent: it labels nothing.
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def _sync_instrumental(self) -> GenerationCreateRequest:
        # vocal_gender=instrumental implies the instrumental flag.
        if self.vocal_gender is VocalGender.INSTRUMENTAL:
            self.instrumental = True
        return self

    @model_validator(mode="after")
    def _lineage_is_consistent(self) -> GenerationCreateRequest:
        # A label names this generation's relationship to a parent, so
        # without a parent it names nothing. Rejected rather than
        # silently dropped: the caller believes it is recording lineage.
        if self.variation_label is not None and self.parent_generation_id is None:
            raise ValueError("variation_label requires parent_generation_id")
        return self


class AdvisoryResponse(BaseModel):
    """A non-blocking pre-flight finding. Never prevents generation."""

    code: str
    level: str
    message: str
    detail: dict[str, object] = Field(default_factory=dict)


class GenerationCreateResponse(BaseModel):
    generation_id: uuid.UUID
    status: GenerationStatus
    #: Advisories recorded for this request. Informational: the
    #: generation was accepted regardless of what they say.
    advisories: list[AdvisoryResponse] = Field(default_factory=list)


class PreflightRequest(BaseModel):
    """Inputs the advisory heuristics actually read.

    Deliberately narrower than :class:`GenerationCreateRequest`: this
    endpoint creates nothing, so it asks only for what ``preflight``
    inspects. Title and prompt are not part of any advisory.
    """

    lyrics: str = Field(default="", max_length=20000)
    duration: int = Field(default=180, ge=DURATION_MIN, le=DURATION_MAX)
    language: str | None = Field(default=None, max_length=16)
    instrumental: bool = False


class SectionSummary(BaseModel):
    """One parsed lyric section, described without echoing its text."""

    #: Canonical role, or ``None`` when the tag is not one we recognise.
    kind: str | None
    #: The tag text as written, without brackets.
    label: str
    index: int | None
    line_number: int
    line_count: int
    has_content: bool
    recognised: bool


class PreflightResponse(BaseModel):
    """Advisories for a draft, computed without creating a generation.

    Exists so the editor can show the *same* findings the create call
    would record, rather than a second implementation of the heuristics
    that could disagree with the backend.
    """

    advisories: list[AdvisoryResponse] = Field(default_factory=list)
    sections: list[SectionSummary] = Field(default_factory=list)
    #: Lines before the first section tag. Untagged lyrics are valid.
    preamble_line_count: int = 0
    #: Estimate used by the density advisories, surfaced so the editor
    #: can explain a warning instead of just asserting it.
    estimated_syllables: int = 0


class AudioAssetResponse(BaseModel):
    id: uuid.UUID
    asset_type: str
    format: str
    mime_type: str
    file_extension: str
    sample_rate: int
    bit_depth: int | None
    bitrate: int | None
    channels: int
    duration: float
    #: Relative, UUID-scoped object key — never a filesystem path, bucket
    #: name, or URL. Clients address audio by generation id, not by key.
    storage_key: str
    sha256: str
    file_size: int
    created_at: datetime

    model_config = {"from_attributes": True}


class GenerationResponse(BaseModel):
    id: uuid.UUID
    title: str
    prompt: str
    lyrics: str
    vocal_gender: str
    duration_requested: int
    duration_actual: float | None
    seed: int | None
    language: str | None
    instrumental: bool
    bpm: int | None = None
    key_scale: str | None = None
    time_signature: str | None = None
    parent_generation_id: uuid.UUID | None = None
    variation_label: str | None = None
    #: Pre-flight findings recorded at submission. Empty list means
    #: "none found"; these were persisted as JSON.
    advisories: list[AdvisoryResponse] = Field(default_factory=list)
    #: What was actually sent to the provider, minus credentials and
    #: host details. ``None`` means no trace was recorded — either the
    #: row predates Phase 8 or the run never reached the provider.
    request_trace: dict[str, object] | None = None
    status: str
    provider: str | None
    model_name: str | None
    model_version: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_code: str | None
    error_message: str | None
    audio_assets: list[AudioAssetResponse]

    model_config = {"from_attributes": True}


class GenerationListResponse(BaseModel):
    items: list[GenerationResponse]
    total: int
    limit: int
    offset: int


class ErrorResponse(BaseModel):
    error_code: str
    message: str


class LyricLineQAEntry(BaseModel):
    """One listener verdict about one submitted lyric line."""

    line_index: int = Field(ge=0)
    section_label: str | None = Field(default=None, max_length=50)
    line_text: str
    verdict: LineVerdict
    note: str | None = None


class GenerationQARequest(BaseModel):
    """A human review of one generation.

    Every field is optional. A reviewer who only wants to record "this
    is a 2/10, the vocal sounds trot-like" should not be forced to
    adjudicate forty lyric lines first.
    """

    #: 1-10 triage. ``None`` means "not rated", which is a different
    #: statement from a low score.
    overall_rating: int | None = Field(default=None, ge=1, le=10)
    failure_tags: list[FailureTag] = Field(default_factory=list)
    #: Section name -> free-text verdict, for the full-song view.
    section_verdicts: dict[str, str] = Field(default_factory=dict)
    notes: str | None = Field(default=None, max_length=5000)
    reviewer: str | None = Field(default=None, max_length=100)
    lyric_lines: list[LyricLineQAEntry] = Field(default_factory=list)

    @field_validator("section_verdicts")
    @classmethod
    def _known_sections(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = set(value) - set(QA_SECTIONS)
        if unknown:
            raise ValueError(f"unknown QA section(s): {sorted(unknown)}")
        return value


class ExpectedLineResponse(BaseModel):
    """A line the model was asked to sing, in submission order."""

    index: int
    section_label: str | None
    text: str


class GenerationQAResponse(BaseModel):
    """The QA record, plus the expected lyric lines to judge against.

    ``expected_lines`` is derived from the stored lyrics on every read,
    so the review form is always anchored to what was actually
    submitted. ``lyric_lines`` holds the verdicts recorded so far and is
    empty until somebody reviews.
    """

    generation_id: uuid.UUID
    overall_rating: int | None = None
    failure_tags: list[str] = Field(default_factory=list)
    section_verdicts: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None
    reviewer: str | None = None
    reviewed: bool = False
    expected_lines: list[ExpectedLineResponse] = Field(default_factory=list)
    lyric_lines: list[LyricLineQAEntry] = Field(default_factory=list)


class LongFormQAResponse(BaseModel):
    """Developer-facing technical summary of one generation.

    Deliberately separate from :class:`GenerationResponse`: this is
    diagnostic clutter that must not reach the normal listener-facing
    experience.
    """

    generation_id: uuid.UUID
    requested_duration: int
    actual_duration: float | None
    sections_requested: int
    lyric_line_count: int
    bpm_requested: int | None
    key_requested: str | None
    time_signature_requested: str | None
    generation_seconds: float | None
    real_time_factor: float | None
    status: str
    is_full_song: bool
