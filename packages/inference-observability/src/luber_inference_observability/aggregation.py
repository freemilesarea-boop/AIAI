"""Counting, and the three ways a count can lie.

**A percentage without its counts.** "Retry rate 2.86%" is unreadable:
it could be 12 of 420 or 2 of 70, and those call for different responses.
Every rate here carries its numerator and denominator, and its renderer
prints them.

**Zero standing in for nothing.** No generations in a window is NO_DATA.
Reporting 0% failure for an hour when nothing ran is a green light for a
system that was switched off.

**A mean standing in for a distribution.** Latency is reported as
quantiles because the mean of a bimodal distribution describes neither
mode, and the request an operator cares about is at P95.

One more rule, less obvious: rows that predate Phase 29 have no
candidate data, so their retries are *unknown* rather than zero. They
count toward request volume and toward success, and are excluded from
every rate whose numerator depends on the candidate trace — with the
exclusion reported, not silent.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from luber_inference_observability.dimensions import Dimension, Segment, validate_grouping
from luber_inference_observability.events import (
    AVAILABILITY_FINDINGS,
    InferenceObservation,
)
from luber_inference_observability.versions import AGGREGATION_VERSION, version_block
from luber_inference_observability.windows import TimeWindow
from luber_inference_qc.findings import Finding


class MetricStatus(StrEnum):
    """Why a metric has, or does not have, a value."""

    OK = "OK"
    #: The denominator was zero. Distinct from a value of 0.0 — nothing
    #: happened, as opposed to nothing failed.
    NO_DATA = "NO_DATA"
    #: There were samples, but too few for the question being asked.
    #: Only ever set by a comparison, never by counting.
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class Rate:
    """A proportion that cannot be printed without its counts."""

    name: str
    numerator: int
    denominator: int
    #: Rows excluded from the denominator because they could not answer
    #: the question — pre-Phase-29 generations, cancellations. Reported
    #: so a shrinking denominator is visible rather than mysterious.
    excluded: int = 0

    @property
    def status(self) -> str:
        return MetricStatus.NO_DATA.value if self.denominator == 0 else MetricStatus.OK.value

    @property
    def value(self) -> float | None:
        """The ratio, or ``None`` when there is nothing to divide."""
        if self.denominator == 0:
            return None
        return self.numerator / self.denominator

    @property
    def percent(self) -> float | None:
        value = self.value
        return None if value is None else value * 100.0

    def render(self) -> str:
        """The only way this should ever reach a human."""
        if self.denominator == 0:
            return f"{self.name}: NO_DATA (0 samples)"
        return (
            f"{self.name}: {self.numerator}/{self.denominator} "
            f"({self.numerator / self.denominator * 100:.2f}%)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "excluded": self.excluded,
            "value": self.value,
            "percent": None if self.percent is None else round(self.percent, 4),
            "status": self.status,
            "render": self.render(),
        }


@dataclass(frozen=True)
class Average:
    """A mean that carries its sample count, for counts rather than times."""

    name: str
    total: float
    count: int

    @property
    def value(self) -> float | None:
        return None if self.count == 0 else self.total / self.count

    @property
    def status(self) -> str:
        return MetricStatus.NO_DATA.value if self.count == 0 else MetricStatus.OK.value

    def to_dict(self) -> dict[str, Any]:
        value = self.value
        return {
            "name": self.name,
            "value": None if value is None else round(value, 4),
            "count": self.count,
            "status": self.status,
        }


def quantile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank quantile. Deterministic, and it returns a real sample.

    Interpolating between two observations invents a latency nothing
    experienced. Nearest-rank returns a measurement that actually
    happened, which is what an operator wants when they go looking for
    the request that took that long.
    """
    if not values:
        return None
    ordered = sorted(values)
    if fraction <= 0:
        return ordered[0]
    if fraction >= 1:
        return ordered[-1]
    rank = math.ceil(fraction * len(ordered))
    return ordered[min(len(ordered) - 1, max(0, rank - 1))]


@dataclass(frozen=True)
class Distribution:
    """A latency distribution, reported the way latency has to be."""

    name: str
    count: int
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    maximum: float | None = None
    #: Present for completeness. Never reported alone.
    mean: float | None = None

    @property
    def status(self) -> str:
        return MetricStatus.NO_DATA.value if self.count == 0 else MetricStatus.OK.value

    @classmethod
    def of(cls, name: str, values: Iterable[float | None]) -> Distribution:
        samples = [float(value) for value in values if value is not None]
        if not samples:
            return cls(name=name, count=0)
        return cls(
            name=name,
            count=len(samples),
            p50=quantile(samples, 0.50),
            p90=quantile(samples, 0.90),
            p95=quantile(samples, 0.95),
            p99=quantile(samples, 0.99),
            maximum=max(samples),
            mean=sum(samples) / len(samples),
        )

    def at(self, fraction: float) -> float | None:
        return {0.5: self.p50, 0.9: self.p90, 0.95: self.p95, 0.99: self.p99}.get(fraction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "p50": _round(self.p50),
            "p90": _round(self.p90),
            "p95": _round(self.p95),
            "p99": _round(self.p99),
            "max": _round(self.maximum),
            "mean": _round(self.mean),
            "status": self.status,
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


#: Critical findings counted individually, because each sends an
#: operator somewhere different. Soft findings are counted too, but into
#: their own map — a harshness advisory must never be able to look like
#: invalid audio.
COUNTED_FINDINGS: tuple[str, ...] = (
    Finding.INVALID_AUDIO.value,
    Finding.NON_FINITE_SAMPLES.value,
    Finding.SILENT_OUTPUT.value,
    Finding.NEAR_SILENT.value,
    Finding.EARLY_COLLAPSE.value,
    Finding.DURATION_SHORT.value,
    Finding.DURATION_LONG.value,
    Finding.SEVERE_CLIPPING.value,
    Finding.PHASE_UNSAFE.value,
    Finding.SPECTRAL_COLLAPSE.value,
    Finding.CHANNEL_IMBALANCE.value,
    Finding.DC_OFFSET.value,
    Finding.PROVIDER_TIMEOUT.value,
    Finding.PROVIDER_ERROR.value,
    Finding.PROVIDER_MISCONFIGURED.value,
)


@dataclass
class Counters:
    """Raw counts. Everything else in this module is derived from these."""

    generation_requests: int = 0
    completed_generations: int = 0
    failed_generations: int = 0
    cancelled_generations: int = 0
    #: Rows with no Phase 29 trace. Counted so partial history is
    #: visible rather than being averaged in as flawless.
    without_qc_data: int = 0

    provider_calls: int = 0
    candidates_generated: int = 0
    quality_retries: int = 0
    retry_exhaustions: int = 0
    candidate_rejections: int = 0
    first_candidate_accepted: int = 0
    #: Denominator for every candidate-derived rate.
    qc_observed: int = 0

    finding_counts: dict[str, int] = field(default_factory=dict)
    soft_finding_counts: dict[str, int] = field(default_factory=dict)
    failure_code_counts: dict[str, int] = field(default_factory=dict)
    data_quality_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_requests": self.generation_requests,
            "completed_generations": self.completed_generations,
            "failed_generations": self.failed_generations,
            "cancelled_generations": self.cancelled_generations,
            "without_qc_data": self.without_qc_data,
            "provider_calls": self.provider_calls,
            "candidates_generated": self.candidates_generated,
            "quality_retries": self.quality_retries,
            "retry_exhaustions": self.retry_exhaustions,
            "candidate_rejections": self.candidate_rejections,
            "first_candidate_accepted": self.first_candidate_accepted,
            "qc_observed": self.qc_observed,
            "finding_counts": dict(sorted(self.finding_counts.items())),
            "soft_finding_counts": dict(sorted(self.soft_finding_counts.items())),
            "failure_code_counts": dict(sorted(self.failure_code_counts.items())),
            "data_quality_counts": dict(sorted(self.data_quality_counts.items())),
        }


#: Every rate this system knows how to compute, by name. The names are
#: the contract: incidents, baselines and reports all refer to a metric
#: by one of these strings, so adding one is additive and renaming one
#: is an aggregation version bump.
class Metric(StrEnum):
    GENERATION_SUCCESS_RATE = "generation_success_rate"
    GENERATION_FAILURE_RATE = "generation_failure_rate"
    FIRST_CANDIDATE_ACCEPT_RATE = "first_candidate_accept_rate"
    QUALITY_RETRY_RATE = "quality_retry_rate"
    RETRY_EXHAUSTION_RATE = "retry_exhaustion_rate"
    PROVIDER_FAILURE_RATE = "provider_failure_rate"
    PROVIDER_TIMEOUT_RATE = "provider_timeout_rate"
    INVALID_AUDIO_RATE = "invalid_audio_rate"
    EARLY_COLLAPSE_RATE = "early_collapse_rate"
    DURATION_FAILURE_RATE = "duration_failure_rate"
    SEVERE_CLIPPING_RATE = "severe_clipping_rate"
    SILENT_OUTPUT_RATE = "silent_output_rate"
    SPECTRAL_COLLAPSE_RATE = "spectral_collapse_rate"
    AVERAGE_PROVIDER_CALLS = "average_provider_calls_per_generation"
    AVERAGE_CANDIDATES = "average_candidates_per_generation"
    PROVIDER_LATENCY = "provider_latency_seconds"
    QC_LATENCY = "qc_latency_seconds"
    DELIVERY_LATENCY = "delivery_latency_seconds"
    TOTAL_LATENCY = "total_latency_seconds"


#: Metrics where a *rise* is the bad direction. Used by the regression
#: engine so a policy does not have to restate it, and so nobody has to
#: remember that acceptance falling and retries rising are the same news.
RISE_IS_BAD: frozenset[str] = frozenset(
    {
        Metric.GENERATION_FAILURE_RATE.value,
        Metric.QUALITY_RETRY_RATE.value,
        Metric.RETRY_EXHAUSTION_RATE.value,
        Metric.PROVIDER_FAILURE_RATE.value,
        Metric.PROVIDER_TIMEOUT_RATE.value,
        Metric.INVALID_AUDIO_RATE.value,
        Metric.EARLY_COLLAPSE_RATE.value,
        Metric.DURATION_FAILURE_RATE.value,
        Metric.SEVERE_CLIPPING_RATE.value,
        Metric.SILENT_OUTPUT_RATE.value,
        Metric.SPECTRAL_COLLAPSE_RATE.value,
        Metric.AVERAGE_PROVIDER_CALLS.value,
        Metric.AVERAGE_CANDIDATES.value,
        Metric.PROVIDER_LATENCY.value,
        Metric.QC_LATENCY.value,
        Metric.DELIVERY_LATENCY.value,
        Metric.TOTAL_LATENCY.value,
    }
)

#: Metrics where a *fall* is the bad direction.
FALL_IS_BAD: frozenset[str] = frozenset(
    {
        Metric.GENERATION_SUCCESS_RATE.value,
        Metric.FIRST_CANDIDATE_ACCEPT_RATE.value,
    }
)


@dataclass
class Aggregate:
    """Everything counted for one window and one segment."""

    window: TimeWindow
    segment: Segment
    counters: Counters
    rates: dict[str, Rate]
    averages: dict[str, Average]
    distributions: dict[str, Distribution]
    aggregation_version: str = AGGREGATION_VERSION

    @property
    def sample_count(self) -> int:
        return self.counters.generation_requests

    @property
    def partial_history(self) -> bool:
        """Whether some rows in this window could not answer QC questions."""
        return self.counters.without_qc_data > 0

    def rate(self, metric: str) -> Rate:
        return self.rates.get(metric, Rate(name=metric, numerator=0, denominator=0))

    def distribution(self, metric: str) -> Distribution:
        return self.distributions.get(metric, Distribution(name=metric, count=0))

    def to_dict(self) -> dict[str, Any]:
        return {
            **version_block(),
            "window": self.window.to_dict(),
            "segment": self.segment.to_dict(),
            "sample_count": self.sample_count,
            "partial_history": self.partial_history,
            "counters": self.counters.to_dict(),
            "rates": {name: item.to_dict() for name, item in sorted(self.rates.items())},
            "averages": {name: item.to_dict() for name, item in sorted(self.averages.items())},
            "distributions": {
                name: item.to_dict() for name, item in sorted(self.distributions.items())
            },
        }


def count(observations: Iterable[InferenceObservation]) -> Counters:
    """Walk the rows once and count everything."""
    counters = Counters()
    for observation in observations:
        counters.generation_requests += 1
        if observation.completed:
            counters.completed_generations += 1
        elif observation.failed:
            counters.failed_generations += 1
        elif observation.cancelled:
            counters.cancelled_generations += 1

        for issue in observation.data_quality_issues:
            counters.data_quality_counts[issue] = counters.data_quality_counts.get(issue, 0) + 1

        if observation.generation_failure_code:
            code = observation.generation_failure_code
            counters.failure_code_counts[code] = counters.failure_code_counts.get(code, 0) + 1

        if not observation.qc_data_available:
            counters.without_qc_data += 1
            continue

        counters.provider_calls += observation.provider_call_count or 0
        counters.candidates_generated += observation.candidate_count or 0
        counters.quality_retries += observation.quality_retry_count or 0
        counters.candidate_rejections += observation.candidate_rejections or 0

        for code in observation.critical_findings:
            counters.finding_counts[code] = counters.finding_counts.get(code, 0) + 1
        for code in observation.soft_findings:
            counters.soft_finding_counts[code] = counters.soft_finding_counts.get(code, 0) + 1

        # Cancelled runs are excluded from every candidate-derived rate.
        # A user changing their mind is not the model getting worse.
        if observation.cancelled:
            continue
        counters.qc_observed += 1
        if observation.first_candidate_accepted:
            counters.first_candidate_accepted += 1
        if observation.retry_exhausted:
            counters.retry_exhaustions += 1
    return counters


def aggregate(
    observations: Iterable[InferenceObservation],
    *,
    window: TimeWindow,
    segment: Segment | None = None,
) -> Aggregate:
    """Count a window's rows into every metric this system reports."""
    rows = list(observations)
    counters = count(rows)
    where = segment or Segment()

    # Denominators, named so the choice is arguable rather than buried.
    requests = counters.generation_requests
    # Cancellations are excluded from delivery success: a cancelled run
    # neither succeeded nor failed at producing music.
    delivery_denominator = counters.completed_generations + counters.failed_generations
    qc = counters.qc_observed
    qc_excluded = counters.without_qc_data + counters.cancelled_generations

    def finding_rate(name: str, code: str) -> Rate:
        return Rate(
            name=name,
            numerator=counters.finding_counts.get(code, 0),
            denominator=qc,
            excluded=qc_excluded,
        )

    duration_failures = sum(
        1
        for row in rows
        if row.counts_toward_quality
        and (
            row.has_finding(Finding.DURATION_SHORT.value)
            or row.has_finding(Finding.DURATION_LONG.value)
        )
    )
    availability_failures = sum(
        1 for row in rows if row.counts_toward_quality and row.has_availability_failure
    )

    rates: dict[str, Rate] = {
        Metric.GENERATION_SUCCESS_RATE.value: Rate(
            name=Metric.GENERATION_SUCCESS_RATE.value,
            numerator=counters.completed_generations,
            denominator=delivery_denominator,
            excluded=counters.cancelled_generations,
        ),
        Metric.GENERATION_FAILURE_RATE.value: Rate(
            name=Metric.GENERATION_FAILURE_RATE.value,
            numerator=counters.failed_generations,
            denominator=delivery_denominator,
            excluded=counters.cancelled_generations,
        ),
        Metric.FIRST_CANDIDATE_ACCEPT_RATE.value: Rate(
            name=Metric.FIRST_CANDIDATE_ACCEPT_RATE.value,
            numerator=counters.first_candidate_accepted,
            denominator=qc,
            excluded=qc_excluded,
        ),
        Metric.QUALITY_RETRY_RATE.value: Rate(
            name=Metric.QUALITY_RETRY_RATE.value,
            numerator=sum(
                1
                for row in rows
                if row.counts_toward_quality and (row.quality_retry_count or 0) > 0
            ),
            denominator=qc,
            excluded=qc_excluded,
        ),
        Metric.RETRY_EXHAUSTION_RATE.value: Rate(
            name=Metric.RETRY_EXHAUSTION_RATE.value,
            numerator=counters.retry_exhaustions,
            denominator=qc,
            excluded=qc_excluded,
        ),
        Metric.PROVIDER_FAILURE_RATE.value: Rate(
            name=Metric.PROVIDER_FAILURE_RATE.value,
            numerator=availability_failures,
            denominator=qc,
            excluded=qc_excluded,
        ),
        Metric.PROVIDER_TIMEOUT_RATE.value: finding_rate(
            Metric.PROVIDER_TIMEOUT_RATE.value, Finding.PROVIDER_TIMEOUT.value
        ),
        Metric.INVALID_AUDIO_RATE.value: finding_rate(
            Metric.INVALID_AUDIO_RATE.value, Finding.INVALID_AUDIO.value
        ),
        Metric.EARLY_COLLAPSE_RATE.value: finding_rate(
            Metric.EARLY_COLLAPSE_RATE.value, Finding.EARLY_COLLAPSE.value
        ),
        Metric.SEVERE_CLIPPING_RATE.value: finding_rate(
            Metric.SEVERE_CLIPPING_RATE.value, Finding.SEVERE_CLIPPING.value
        ),
        Metric.SILENT_OUTPUT_RATE.value: finding_rate(
            Metric.SILENT_OUTPUT_RATE.value, Finding.SILENT_OUTPUT.value
        ),
        Metric.SPECTRAL_COLLAPSE_RATE.value: finding_rate(
            Metric.SPECTRAL_COLLAPSE_RATE.value, Finding.SPECTRAL_COLLAPSE.value
        ),
        Metric.DURATION_FAILURE_RATE.value: Rate(
            name=Metric.DURATION_FAILURE_RATE.value,
            numerator=duration_failures,
            denominator=qc,
            excluded=qc_excluded,
        ),
    }

    averages = {
        Metric.AVERAGE_PROVIDER_CALLS.value: Average(
            name=Metric.AVERAGE_PROVIDER_CALLS.value,
            total=float(counters.provider_calls),
            count=qc,
        ),
        Metric.AVERAGE_CANDIDATES.value: Average(
            name=Metric.AVERAGE_CANDIDATES.value,
            total=float(counters.candidates_generated),
            count=qc,
        ),
    }

    distributions = {
        Metric.PROVIDER_LATENCY.value: Distribution.of(
            Metric.PROVIDER_LATENCY.value, (row.provider_latency_seconds for row in rows)
        ),
        Metric.QC_LATENCY.value: Distribution.of(
            Metric.QC_LATENCY.value, (row.qc_latency_seconds for row in rows)
        ),
        Metric.DELIVERY_LATENCY.value: Distribution.of(
            Metric.DELIVERY_LATENCY.value, (row.delivery_latency_seconds for row in rows)
        ),
        Metric.TOTAL_LATENCY.value: Distribution.of(
            Metric.TOTAL_LATENCY.value, (row.total_latency_seconds for row in rows)
        ),
    }

    _ = requests  # named above for readability of the denominators
    return Aggregate(
        window=window,
        segment=where,
        counters=counters,
        rates=rates,
        averages=averages,
        distributions=distributions,
    )


def group(
    observations: Iterable[InferenceObservation],
    *,
    window: TimeWindow,
    by: tuple[str, ...],
) -> dict[Segment, Aggregate]:
    """Aggregate separately for each combination of *by*.

    The grouping is validated rather than trusted: a caller asking for
    four dimensions gets a refusal naming the reason, not a query that
    runs for a minute and returns buckets of one.
    """
    dimensions = validate_grouping(by)
    buckets: dict[Segment, list[InferenceObservation]] = {}
    for observation in observations:
        key = Segment.of(
            **{item.value: getattr(observation, item.value, None) for item in dimensions}
        )
        buckets.setdefault(key, []).append(observation)
    return {
        segment: aggregate(rows, window=window, segment=segment)
        for segment, rows in sorted(buckets.items(), key=lambda pair: pair[0].label())
    }


__all__ = [
    "AVAILABILITY_FINDINGS",
    "COUNTED_FINDINGS",
    "FALL_IS_BAD",
    "RISE_IS_BAD",
    "Aggregate",
    "Average",
    "Counters",
    "Dimension",
    "Distribution",
    "Metric",
    "MetricStatus",
    "Rate",
    "aggregate",
    "count",
    "group",
    "quantile",
]
