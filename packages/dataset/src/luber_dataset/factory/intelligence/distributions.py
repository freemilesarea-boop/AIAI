"""Distributions that cannot hide their denominator.

Every share reported here is a share *of the known data*, and every
distribution carries how much data that was. The alternative — dividing
by the dataset size and letting unknowns dilute the percentages — reads
as a statement about the whole corpus and is not one. A genre
distribution built from 10% coverage saying "pop 60%" means 6% of the
dataset is known pop and 90% is a shrug, and those are extremely
different facts to plan a training run around.

Distributions come in two forms and both matter:

*By count*, which is how many tracks.
*By duration*, which is how much the model actually sees.

They diverge constantly. A hundred thirty-second sketches and ten
six-minute pieces are equal by duration and ten-to-one by count, and a
dataset can look balanced one way while being dominated the other.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from luber_dataset.factory.intelligence.schemas import Observation, TrackView


@dataclass
class Bucket:
    """One category, counted and weighed."""

    label: str
    count: int = 0
    hours: float = 0.0
    share_by_count: float = 0.0
    share_by_duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "count": self.count,
            "hours": round(self.hours, 4),
            "share_by_count": round(self.share_by_count, 6),
            "share_by_duration": round(self.share_by_duration, 6),
        }


@dataclass
class CategoricalDistribution:
    """Counts over a dimension, with its coverage stated.

    ``buckets`` describes the *known* subset only. ``unknown_count`` and
    ``unknown_hours`` sit outside it, never folded in as a category, so
    a caller cannot accidentally treat "unknown" as the largest genre.
    """

    dimension: str
    buckets: list[Bucket] = field(default_factory=list)
    known_count: int = 0
    known_hours: float = 0.0
    unknown_count: int = 0
    unknown_hours: float = 0.0
    #: Values below their confidence gate: measured, and not trusted.
    low_confidence_count: int = 0
    #: How each known value was established.
    source_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        return self.known_count + self.unknown_count

    @property
    def coverage(self) -> float:
        """Share of tracks for which this dimension is known."""
        return self.known_count / self.total_count if self.total_count else 0.0

    @property
    def category_count(self) -> int:
        return len(self.buckets)

    def share(self, label: str) -> float:
        for bucket in self.buckets:
            if bucket.label == label:
                return bucket.share_by_count
        return 0.0

    def share_by_duration(self, label: str) -> float:
        for bucket in self.buckets:
            if bucket.label == label:
                return bucket.share_by_duration
        return 0.0

    def top(self, n: int = 1) -> list[Bucket]:
        return self.buckets[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "known_count": self.known_count,
            "known_hours": round(self.known_hours, 4),
            "unknown_count": self.unknown_count,
            "unknown_hours": round(self.unknown_hours, 4),
            "low_confidence_count": self.low_confidence_count,
            "coverage": round(self.coverage, 6),
            "category_count": self.category_count,
            "source_breakdown": dict(sorted(self.source_breakdown.items())),
            "buckets": [bucket.to_dict() for bucket in self.buckets],
        }


def categorical(
    tracks: Sequence[TrackView],
    dimension: str,
    accessor: Callable[[TrackView], Observation],
    *,
    normalise: Callable[[Any], str] | None = None,
) -> CategoricalDistribution:
    """Count a dimension across tracks, keeping unknowns outside.

    Buckets are sorted by count descending then label ascending, so the
    ordering is total and two runs over the same data produce the same
    list — which several downstream digests depend on.
    """
    result = CategoricalDistribution(dimension=dimension)
    counts: dict[str, int] = {}
    hours: dict[str, float] = {}

    for track in tracks:
        observation = accessor(track)
        track_hours = track.hours
        if not observation.known:
            result.unknown_count += 1
            result.unknown_hours += track_hours
            if observation.source == "LOW_CONFIDENCE":
                result.low_confidence_count += 1
            continue
        label = normalise(observation.value) if normalise else str(observation.value)
        counts[label] = counts.get(label, 0) + 1
        hours[label] = hours.get(label, 0.0) + track_hours
        result.known_count += 1
        result.known_hours += track_hours
        result.source_breakdown[observation.source] = (
            result.source_breakdown.get(observation.source, 0) + 1
        )

    for label in sorted(counts, key=lambda name: (-counts[name], name)):
        result.buckets.append(
            Bucket(
                label=label,
                count=counts[label],
                hours=hours[label],
                share_by_count=counts[label] / result.known_count if result.known_count else 0.0,
                share_by_duration=(
                    hours[label] / result.known_hours if result.known_hours > 0 else 0.0
                ),
            )
        )
    return result


@dataclass
class NumericSummary:
    """Quantiles for a measured quantity, over its known values."""

    dimension: str
    known_count: int = 0
    unknown_count: int = 0
    low_confidence_count: int = 0
    minimum: float | None = None
    p10: float | None = None
    p25: float | None = None
    median: float | None = None
    p75: float | None = None
    p90: float | None = None
    maximum: float | None = None
    mean: float | None = None

    @property
    def coverage(self) -> float:
        total = self.known_count + self.unknown_count
        return self.known_count / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "known_count": self.known_count,
            "unknown_count": self.unknown_count,
            "low_confidence_count": self.low_confidence_count,
            "coverage": round(self.coverage, 6),
            "min": self.minimum,
            "p10": self.p10,
            "p25": self.p25,
            "median": self.median,
            "p75": self.p75,
            "p90": self.p90,
            "max": self.maximum,
            "mean": self.mean,
        }


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    """Linear interpolation between order statistics.

    Written out rather than pulled from numpy so the whole intelligence
    layer stays pure-python and importable without an array library —
    it operates on metadata, not audio.
    """
    if not ordered:
        raise ValueError("no values")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def numeric(
    tracks: Sequence[TrackView],
    dimension: str,
    accessor: Callable[[TrackView], Observation],
) -> NumericSummary:
    result = NumericSummary(dimension=dimension)
    values: list[float] = []
    for track in tracks:
        observation = accessor(track)
        if not observation.known or not isinstance(observation.value, (int, float)):
            result.unknown_count += 1
            if observation.source == "LOW_CONFIDENCE":
                result.low_confidence_count += 1
            continue
        values.append(float(observation.value))
    result.known_count = len(values)
    if not values:
        return result

    values.sort()
    result.minimum = round(values[0], 4)
    result.p10 = round(_quantile(values, 0.10), 4)
    result.p25 = round(_quantile(values, 0.25), 4)
    result.median = round(_quantile(values, 0.50), 4)
    result.p75 = round(_quantile(values, 0.75), 4)
    result.p90 = round(_quantile(values, 0.90), 4)
    result.maximum = round(values[-1], 4)
    result.mean = round(sum(values) / len(values), 4)
    return result


# ── bucketing ────────────────────────────────────────────────────────

#: Duration buckets, in seconds. Chosen around how music is actually
#: shaped: a sketch, a short piece, a single, a long single, an extended
#: piece, and everything beyond.
DURATION_BUCKETS: tuple[tuple[float, str], ...] = (
    (30.0, "<30s"),
    (60.0, "30-60s"),
    (120.0, "60-120s"),
    (180.0, "120-180s"),
    (240.0, "180-240s"),
    (360.0, "240-360s"),
)

#: Tempo buckets. Wide enough that a few BPM of estimation error does
#: not move a track between them.
TEMPO_BUCKETS: tuple[tuple[float, str], ...] = (
    (70.0, "<70"),
    (90.0, "70-90"),
    (110.0, "90-110"),
    (130.0, "110-130"),
    (150.0, "130-150"),
    (180.0, "150-180"),
)


def bucket_label(value: float, edges: Iterable[tuple[float, str]], overflow: str) -> str:
    for edge, label in edges:
        if value < edge:
            return label
    return overflow


def duration_bucket(seconds: float) -> str:
    return bucket_label(seconds, DURATION_BUCKETS, ">360s")


def tempo_bucket(bpm: float) -> str:
    return bucket_label(bpm, TEMPO_BUCKETS, ">180")


@dataclass
class CompletenessScore:
    """How much of a dimension the dataset actually knows."""

    dimension: str
    known: int
    unknown: int
    low_confidence: int = 0

    @property
    def total(self) -> int:
        return self.known + self.unknown

    @property
    def completeness(self) -> float:
        return self.known / self.total if self.total else 0.0

    @property
    def missing_percentage(self) -> float:
        return 100.0 * (1.0 - self.completeness)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "known": self.known,
            "unknown": self.unknown,
            "low_confidence": self.low_confidence,
            "total": self.total,
            "completeness": round(self.completeness, 6),
            "missing_percentage": round(self.missing_percentage, 4),
        }
