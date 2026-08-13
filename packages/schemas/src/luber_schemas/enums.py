"""Domain enums shared across API, workers, and database layers.

These values are part of the persisted contract (stored in PostgreSQL
and returned by the public API), so renames are breaking changes and
must go through a migration.
"""

from __future__ import annotations

from enum import StrEnum


class GenerationStatus(StrEnum):
    """Lifecycle of a music generation job."""

    QUEUED = "QUEUED"
    STARTING = "STARTING"
    GENERATING = "GENERATING"
    POST_PROCESSING = "POST_PROCESSING"
    UPLOADING = "UPLOADING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.CANCELLED,
        }


class VocalGender(StrEnum):
    """User-facing vocal selection. ``INSTRUMENTAL`` means no vocals."""

    FEMALE = "female"
    MALE = "male"
    INSTRUMENTAL = "instrumental"


class AssetType(StrEnum):
    """Kind of audio asset attached to a generation."""

    MASTER = "MASTER"
    PREVIEW = "PREVIEW"
    STEM = "STEM"


class ErrorCode(StrEnum):
    """Standard machine-readable error codes returned to the frontend.

    Raw exception strings are never sent to clients.
    """

    GENERATION_TIMEOUT = "GENERATION_TIMEOUT"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    INVALID_AUDIO = "INVALID_AUDIO"
    UPLOAD_FAILED = "UPLOAD_FAILED"
    ENCODING_FAILED = "ENCODING_FAILED"
    QUEUE_FAILED = "QUEUE_FAILED"
    UNKNOWN_GENERATION_ERROR = "UNKNOWN_GENERATION_ERROR"


class LineVerdict(StrEnum):
    """What a listener heard happen to one submitted lyric line.

    ``UNKNOWN`` is a real answer, not a missing one: on a dense mix a
    listener genuinely cannot always tell, and forcing a guess would
    poison the record this exists to build.
    """

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    SKIPPED = "SKIPPED"
    DUPLICATED = "DUPLICATED"
    UNKNOWN = "UNKNOWN"


class FailureTag(StrEnum):
    """Named failure modes observed in LUBER output.

    These are the defects the human evaluator actually reports, so the
    listening tool offers them as checkboxes rather than asking for free
    text that cannot be aggregated later.
    """

    KOREAN_LINE_OMISSION = "KOREAN_LINE_OMISSION"
    LYRIC_LINE_SKIP = "LYRIC_LINE_SKIP"
    LYRIC_DUPLICATION = "LYRIC_DUPLICATION"
    TROT_LIKE_VOCAL = "TROT_LIKE_VOCAL"
    VOCAL_STYLE_OUTDATED = "VOCAL_STYLE_OUTDATED"
    EXCESSIVE_SIBILANCE = "EXCESSIVE_SIBILANCE"
    HIGH_END_OVERBOOST = "HIGH_END_OVERBOOST"
    INSTRUMENT_FIDELITY_LOW = "INSTRUMENT_FIDELITY_LOW"
    STRUCTURE_COLLAPSE = "STRUCTURE_COLLAPSE"
    MELODY_DRIFT = "MELODY_DRIFT"
    VOCAL_IDENTITY_DRIFT = "VOCAL_IDENTITY_DRIFT"
    ENDING_FAILURE = "ENDING_FAILURE"


#: Sections the full-song QA view asks about, in song order.
QA_SECTIONS: tuple[str, ...] = (
    "intro",
    "verse_1",
    "chorus",
    "verse_2",
    "bridge",
    "final_chorus",
    "outro",
)
