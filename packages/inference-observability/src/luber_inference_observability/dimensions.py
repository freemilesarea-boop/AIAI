"""What a metric may be sliced by, and the limits on slicing.

Two rules run through this module.

**UNKNOWN is a value.** A provider revision nobody recorded is UNKNOWN,
not "default" and not blank. The difference matters the moment somebody
compares two revisions: a bucket labelled UNKNOWN says "these rows
cannot take part in that comparison", and an empty string says nothing
at all while still being grouped.

**Cardinality is bounded on purpose.** Every dimension here has a small,
enumerable range — a task type, a duration bucket, a provider. The one
field that would explode it, `request_sha256`, is deliberately not a
dimension: it is drilldown identity. Grouping by it produces one bucket
per request, which is not an aggregate, and doing it accidentally is how
an analytics table becomes larger than the data it summarises.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

#: What a dimension holds when nothing recorded it. Never blank, never a
#: plausible-looking default.
UNKNOWN = "UNKNOWN"


class TaskType(StrEnum):
    """What kind of generation this was.

    Derived from the row's own routing columns rather than from anything
    a client could set: ``edit_kind`` is what the worker dispatches on,
    and ``reference_audio_id`` is a foreign key. A label a caller could
    influence would make the segmentation forgeable.
    """

    TEXT_TO_MUSIC = "TEXT_TO_MUSIC"
    #: Text-to-music steered by a reference track the user supplied.
    #: Split out from TEXT_TO_MUSIC because it exercises a different
    #: provider path and fails differently.
    REFERENCE_CONDITIONED = "REFERENCE_CONDITIONED"
    EXTEND = "EXTEND"
    REPLACE_RANGE = "REPLACE_RANGE"
    COVER = "COVER"


def task_type(*, edit_kind: str | None, has_reference: bool) -> str:
    """Which task this generation was, from columns that cannot be forged."""
    if edit_kind == "EXTEND":
        return TaskType.EXTEND.value
    if edit_kind == "REPLACE_RANGE":
        return TaskType.REPLACE_RANGE.value
    if edit_kind == "COVER":
        return TaskType.COVER.value
    if edit_kind is not None:
        # A kind this build does not know about. Recorded as unknown
        # rather than folded into TEXT_TO_MUSIC, which would make a new
        # task type silently inflate the oldest bucket.
        return UNKNOWN
    return TaskType.REFERENCE_CONDITIONED.value if has_reference else TaskType.TEXT_TO_MUSIC.value


#: Requested-duration buckets, as (label, lower inclusive, upper inclusive).
#:
#: The boundaries are the brief's, and they are worth keeping rather than
#: rounding: generation failures concentrate by length, and a bucket that
#: straddled 180s — where several models change behaviour — would hide
#: exactly the segment regression this exists to find.
#:
#: Changing these changes what every historical number means, which is
#: why they live beside `AGGREGATION_VERSION` in spirit: a boundary move
#: is a version bump.
DURATION_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0_30", 0.0, 30.0),
    ("31_60", 30.0, 60.0),
    ("61_120", 60.0, 120.0),
    ("121_180", 120.0, 180.0),
    ("181_240", 180.0, 240.0),
    ("241_360", 240.0, 360.0),
    ("360_PLUS", 360.0, float("inf")),
)

DURATION_BUCKET_LABELS: tuple[str, ...] = tuple(name for name, _, _ in DURATION_BUCKETS)


def duration_bucket(seconds: float | None) -> str:
    """Which bucket a requested duration falls in.

    Lower bound exclusive, upper inclusive, so 30s lands in ``0_30`` and
    31s in ``31_60`` — matching how the labels read. A duration nobody
    requested is UNKNOWN rather than being pushed into the first bucket.
    """
    if seconds is None or seconds < 0:
        return UNKNOWN
    for name, low, high in DURATION_BUCKETS:
        if low < seconds <= high or (low == 0.0 and seconds == 0.0):
            return name
    return UNKNOWN


class Dimension(StrEnum):
    """Fields a query may group by.

    An allowlist rather than "any column", because the cost of the wrong
    answer is an aggregation that never finishes. `request_sha256` and
    `generation_id` are absent by construction.
    """

    PROVIDER = "provider"
    PROVIDER_REVISION = "provider_revision"
    MODEL_NAME = "model_name"
    TASK_TYPE = "task_type"
    DURATION_BUCKET = "duration_bucket"
    LANGUAGE = "language"
    INSTRUMENTAL = "instrumental"
    QC_POLICY = "qc_policy"
    QC_ENGINE_VERSION = "qc_engine_version"
    RETRY_POLICY_VERSION = "retry_policy_version"
    FINISHING_VERSION = "finishing_version"
    LUBER_REVISION = "luber_revision"


#: How many dimensions one grouping may combine.
#:
#: Two is enough for every question the runbook asks — "provider by
#: duration", "task by revision" — and the third multiplies the bucket
#: count by the cardinality of another field while dividing the samples
#: in each. A three-way split of a week's traffic mostly produces
#: buckets too small to say anything about, and a regression detector
#: fed tiny buckets either stays silent or lies.
MAX_GROUP_DIMENSIONS = 2


class GroupingTooWide(ValueError):
    """Raised rather than quietly truncating a caller's grouping."""


def validate_grouping(dimensions: tuple[str, ...]) -> tuple[Dimension, ...]:
    """Check a requested grouping, or refuse it with the reason."""
    if len(dimensions) > MAX_GROUP_DIMENSIONS:
        raise GroupingTooWide(
            f"at most {MAX_GROUP_DIMENSIONS} grouping dimensions are supported; "
            f"{len(dimensions)} were requested ({', '.join(dimensions)}). A wider "
            "split divides the samples until no bucket can support a finding."
        )
    known = {item.value for item in Dimension}
    resolved: list[Dimension] = []
    for name in dimensions:
        if name not in known:
            raise GroupingTooWide(
                f"{name!r} is not a grouping dimension. Known: {', '.join(sorted(known))}. "
                "Per-request identifiers are drilldown handles, not dimensions."
            )
        resolved.append(Dimension(name))
    if len(set(resolved)) != len(resolved):
        raise GroupingTooWide("the same dimension was requested twice")
    return tuple(resolved)


@dataclass(frozen=True)
class Segment:
    """A set of dimension filters, as a value that can be a dict key.

    Frozen and sorted so the same filters always produce the same
    fingerprint — which is what stops an incident from being reopened
    under a new identity because a caller passed its filters in a
    different order.
    """

    filters: tuple[tuple[str, str], ...] = ()

    @classmethod
    def of(cls, **filters: str | None) -> Segment:
        return cls(
            tuple(sorted((key, value) for key, value in filters.items() if value is not None))
        )

    def to_dict(self) -> dict[str, str]:
        return dict(self.filters)

    def label(self) -> str:
        if not self.filters:
            return "all traffic"
        return ", ".join(f"{key}={value}" for key, value in self.filters)

    def matches(self, observation: Any) -> bool:
        return all(getattr(observation, key, None) == value for key, value in self.filters)


__all__ = [
    "DURATION_BUCKETS",
    "DURATION_BUCKET_LABELS",
    "MAX_GROUP_DIMENSIONS",
    "UNKNOWN",
    "Dimension",
    "GroupingTooWide",
    "Segment",
    "TaskType",
    "duration_bucket",
    "task_type",
    "validate_grouping",
]
