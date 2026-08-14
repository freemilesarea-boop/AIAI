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


class EditKind(StrEnum):
    """How a generation was derived from another generation's audio.

    A *product* vocabulary, not the engine's. Both kinds reach ACE-Step
    as the same primitive — regenerate this time range, preserve the rest
    — and the difference is what the range means:

    ``EXTEND``
        The range begins at the end of the source, so the engine pads the
        source and generates into the padding. The song gets longer.

    ``REPLACE_RANGE``
        The range is interior. The song keeps its length and only that
        span is regenerated.

    The worker needs the distinction because it anchors the two
    differently: an extension is re-anchored to the *measured* end of the
    audio being uploaded, while a replacement uses the absolute times the
    user chose. Storing one value for both would make that routing
    guesswork.
    """

    #: ``COVER`` is a third case and not an edit at all: the engine
    #: regenerates the whole performance steered by a semantic sketch of
    #: the source, preserving none of the recording. It shares this column
    #: because the question the column answers — "how did this come from
    #: its parent?" — is the same one. It carries no time range.
    EXTEND = "EXTEND"
    REPLACE_RANGE = "REPLACE_RANGE"
    COVER = "COVER"

    @property
    def preserves_source_audio(self) -> bool:
        """Whether the parent's recording survives into the result.

        True for the repaint-backed edits, where the engine re-imposes the
        source outside the edited range. False for a cover, which
        regenerates everything. The UI uses this to avoid promising
        preservation it does not get.
        """
        return self in (EditKind.EXTEND, EditKind.REPLACE_RANGE)


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
