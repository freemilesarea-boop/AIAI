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
