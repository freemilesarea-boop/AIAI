"""Turning finished generations into observations, exactly once each.

Two entry points over one function, because backfill and incremental
ingest differ in only one thing: where they start reading. Sharing the
projection means a row written by a backfill and a row written by a
worker are the same row, which is what lets an operator run a backfill
over a period the incremental path already covered and get the same
answer rather than a doubled one.

Idempotence is structural, not defensive. `generation_id` is the
projection's primary key, so ingesting the same generation twice
replaces a row instead of adding one. Nothing here needs to check
whether it has seen something before, which means nothing here can get
that check wrong.

**Late finalisation.** Phase 29 writes its trace as the run proceeds, so
a generation observed mid-flight and re-observed after completion
produces two different projections of the same row. The second replaces
the first: the final state is the true one, and the counts do not
double because the key did not change.

**What is deliberately not read.** The prompt, the lyrics, the title,
the user, and `request_trace` — which looks like a diagnostic blob and
contains the full original prompt and lyrics. The extractor below names
every field it touches, so adding one is a visible act.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from luber_inference_observability.events import InferenceObservation, observe
from luber_inference_observability.storage import to_mapping

#: The only fields ingestion reads off a generation. Written down rather
#: than accessed ad hoc so that adding one is a change somebody reviews —
#: the field a careless addition would reach for is `prompt`, and it is
#: conspicuously not here.
SOURCE_FIELDS: tuple[str, ...] = (
    "id",
    "status",
    "created_at",
    "started_at",
    "completed_at",
    "provider",
    "model_name",
    "model_version",
    "duration_requested",
    "language",
    "instrumental",
    "bpm",
    "key_scale",
    "edit_kind",
    "reference_audio_id",
    "error_code",
    "inference_qc_trace",
    "finishing_trace",
)


class GenerationLike(Protocol):
    """What ingestion needs from a generation row.

    Structural rather than an import of the ORM class: this package does
    not depend on the database package, and a Protocol keeps it that way
    while still type-checking the fields that are read.
    """

    id: Any
    status: str


def _json(raw: Any) -> dict[str, Any] | None:
    """Parse a stored trace, or ``None`` if it is absent or unreadable.

    A trace that will not parse is treated as absent rather than raising.
    One corrupt row must not stop a backfill: the observation is written
    without candidate data and flagged as lacking it, which is both true
    and recoverable.
    """
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def from_generation(
    generation: Any,
    *,
    luber_revision: str | None = None,
    ingested_at: datetime | None = None,
) -> InferenceObservation:
    """Project one generation row into an observation.

    Every value is read by name from a column or a parsed trace. Nothing
    is inferred from text, and there is no path from here to the prompt.
    """
    started = getattr(generation, "started_at", None) or getattr(generation, "created_at", None)
    if started is None:
        raise ValueError(
            f"generation {getattr(generation, 'id', '?')} has neither a start nor a "
            "creation time, so it cannot be placed on a timeline"
        )

    duration = getattr(generation, "duration_requested", None)
    return observe(
        generation_id=str(generation.id),
        status=str(getattr(generation, "status", "UNKNOWN")),
        occurred_at=started,
        completed_at=getattr(generation, "completed_at", None),
        provider=getattr(generation, "provider", None),
        model_name=getattr(generation, "model_name", None),
        model_version=getattr(generation, "model_version", None),
        duration_requested=None if duration is None else float(duration),
        language=getattr(generation, "language", None),
        instrumental=getattr(generation, "instrumental", None),
        bpm=getattr(generation, "bpm", None),
        key_scale=getattr(generation, "key_scale", None),
        edit_kind=getattr(generation, "edit_kind", None),
        has_reference=getattr(generation, "reference_audio_id", None) is not None,
        error_code=getattr(generation, "error_code", None),
        qc_trace=_json(getattr(generation, "inference_qc_trace", None)),
        finishing_trace=_json(getattr(generation, "finishing_trace", None)),
        luber_revision=luber_revision,
        ingested_at=ingested_at or datetime.now(UTC),
    )


@dataclass
class IngestResult:
    """What one ingestion run did, in terms an operator can check."""

    scanned: int = 0
    written: int = 0
    skipped: int = 0
    failed: int = 0
    #: Generations with no Phase 29 trace. Not an error — they predate
    #: it — but counted, because a backfill that was mostly these should
    #: not read as a backfill that worked.
    without_qc_trace: int = 0
    errors: list[str] = None  # type: ignore[assignment]
    watermark: datetime | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "written": self.written,
            "skipped": self.skipped,
            "failed": self.failed,
            "without_qc_trace": self.without_qc_trace,
            "errors": self.errors[:20],
            "error_count": len(self.errors),
            "watermark": self.watermark.isoformat() if self.watermark else None,
        }


def project(
    generations: Iterable[Any],
    *,
    luber_revision: str | None = None,
    ingested_at: datetime | None = None,
) -> tuple[list[InferenceObservation], IngestResult]:
    """Project many generations, surviving the ones that will not.

    A row that cannot be projected is counted and named rather than
    raised. The alternative — stopping the run — means one malformed
    generation from six months ago blocks every backfill until somebody
    deletes it.
    """
    result = IngestResult()
    observations: list[InferenceObservation] = []
    for generation in generations:
        result.scanned += 1
        try:
            observation = from_generation(
                generation, luber_revision=luber_revision, ingested_at=ingested_at
            )
        except Exception as exc:  # one bad row must not stop a backfill
            result.failed += 1
            result.errors.append(f"{getattr(generation, 'id', '?')}: {exc}")
            continue
        if not observation.qc_data_available:
            result.without_qc_trace += 1
        observations.append(observation)
        result.written += 1
        if result.watermark is None or observation.occurred_at > result.watermark:
            result.watermark = observation.occurred_at
    return observations, result


def as_rows(observations: Sequence[InferenceObservation]) -> list[dict[str, Any]]:
    """Observations as row-shaped mappings, for a repository to write."""
    return [to_mapping(observation) for observation in observations]


__all__ = [
    "SOURCE_FIELDS",
    "GenerationLike",
    "IngestResult",
    "as_rows",
    "from_generation",
    "project",
]
