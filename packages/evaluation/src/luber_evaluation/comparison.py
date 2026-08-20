"""Comparing a candidate against its baseline, metric by metric.

No single scalar. A composite score is available as advisory output and
is never what a gate reads, because the one thing a scalar reliably does
is hide the regression that mattered inside an average of things that
improved.

Two subtleties this module exists to handle.

**Direction is per metric.** A falling failure rate is an improvement; a
falling phase correlation is a regression; a moved spectral centroid is
neither, and treating it as an improvement would reward brightness for
its own sake.

**A difference is not automatically a finding.** Generation is
stochastic, and two draws from the same model differ. A metric is only
called improved or regressed when the movement exceeds what the suite
can resolve — otherwise it is `INCONCLUSIVE`, which is an honest answer
and not a hedge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from luber_evaluation.metrics import (
    CATALOGUE,
    Aggregate,
    MetricDirection,
    MetricStatus,
)
from luber_evaluation.schemas import ComparisonVerdict, RegressionSeverity

COMPARISON_SCHEMA_VERSION = "luber-evaluation-comparison/1"

#: Relative movement below this is treated as noise rather than signal.
#: Deliberately generous: a comparison that called a 1% difference a
#: regression would flag every run, and a gate that fires constantly
#: gets disabled.
DEFAULT_NOISE_FLOOR_RELATIVE = 0.05

#: Rate metrics live in [0, 1], where relative change is misleading —
#: 0.001 to 0.002 is +100% and irrelevant, while 0.02 to 0.08 is +300%
#: and serious. These are compared on absolute movement instead.
RATE_METRICS: frozenset[str] = frozenset(
    {
        "generation_success_rate",
        "generation_failure_rate",
        "generation_timeout_rate",
        "invalid_audio_rate",
        "silent_output_rate",
        "early_collapse_rate",
        "wrong_duration_rate",
        "clipping_sample_ratio",
        "silence_ratio",
    }
)

#: Absolute movement in a rate that counts as a real change.
DEFAULT_RATE_NOISE_FLOOR = 0.02

#: How far a regression has to go before it stops being cosmetic.
#: Applied to whichever scale the metric uses.
SEVERITY_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.30, RegressionSeverity.CRITICAL.value),
    (0.15, RegressionSeverity.MAJOR.value),
    (0.05, RegressionSeverity.MINOR.value),
)


@dataclass
class MetricComparison:
    """One metric, both sides, and what the difference means."""

    metric_name: str
    direction: str
    unit: str = ""
    baseline_value: float | None = None
    candidate_value: float | None = None
    absolute_delta: float | None = None
    relative_delta: float | None = None
    verdict: str = ComparisonVerdict.NOT_MEASURABLE.value
    severity: str = RegressionSeverity.NONE.value
    #: Why a verdict is NOT_MEASURABLE or INCONCLUSIVE.
    detail: str = ""
    baseline_status: str = MetricStatus.NOT_MEASURABLE.value
    candidate_status: str = MetricStatus.NOT_MEASURABLE.value
    #: Failure counts carried through, so a summary cannot bury them.
    baseline_failures: int = 0
    candidate_failures: int = 0

    @property
    def regressed(self) -> bool:
        return self.verdict == ComparisonVerdict.REGRESSED.value

    @property
    def improved(self) -> bool:
        return self.verdict == ComparisonVerdict.IMPROVED.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _severity(magnitude: float) -> str:
    for threshold, severity in SEVERITY_THRESHOLDS:
        if magnitude >= threshold:
            return severity
    return RegressionSeverity.INFO.value


def compare_metric(
    metric_name: str,
    baseline: Aggregate | None,
    candidate: Aggregate | None,
    *,
    noise_floor_relative: float = DEFAULT_NOISE_FLOOR_RELATIVE,
    rate_noise_floor: float = DEFAULT_RATE_NOISE_FLOOR,
) -> MetricComparison:
    """Compare one metric's aggregates. Never invents a number."""
    spec = CATALOGUE.get(metric_name)
    result = MetricComparison(
        metric_name=metric_name,
        direction=spec.direction if spec else MetricDirection.INFORMATIONAL.value,
        unit=spec.unit if spec else "",
    )

    if baseline is None or candidate is None:
        result.detail = "one side produced no aggregate for this metric"
        return result

    result.baseline_status = baseline.status
    result.candidate_status = candidate.status
    result.baseline_failures = baseline.count_failed
    result.candidate_failures = candidate.count_failed

    if baseline.median_value is None or candidate.median_value is None:
        # Not measurable on at least one side. Reported as such, never
        # as a zero on the missing side — which would manufacture an
        # enormous improvement or regression out of an absent detector.
        result.detail = (
            baseline.detail
            or candidate.detail
            or "the metric was not measurable on at least one side"
        )
        return result

    # Median rather than mean: one catastrophic case should move a
    # failure *count*, not silently drag a central tendency.
    result.baseline_value = baseline.median_value
    result.candidate_value = candidate.median_value
    delta = candidate.median_value - baseline.median_value
    result.absolute_delta = round(delta, 6)

    is_rate = metric_name in RATE_METRICS
    if not is_rate and abs(baseline.median_value) > 1e-12:
        result.relative_delta = round(delta / abs(baseline.median_value), 6)

    if spec and spec.direction == MetricDirection.INFORMATIONAL.value:
        result.verdict = ComparisonVerdict.UNCHANGED.value
        result.detail = "informational: recorded, not judged"
        return result

    magnitude = abs(delta) if is_rate else abs(result.relative_delta or 0.0)
    # A metric may declare its own resolution. The shared defaults are
    # right for most things and wrong for a few, and the catalogue is
    # where a metric says which it is.
    if spec is not None and spec.noise_floor is not None:
        floor = spec.noise_floor
    else:
        floor = rate_noise_floor if is_rate else noise_floor_relative

    if magnitude < floor:
        result.verdict = (
            ComparisonVerdict.UNCHANGED.value
            if magnitude == 0.0
            else ComparisonVerdict.INCONCLUSIVE.value
        )
        if magnitude:
            result.detail = f"moved {magnitude:.5f}, inside the {floor:g} the suite can resolve"
        return result

    direction = spec.direction if spec else MetricDirection.INFORMATIONAL.value
    if direction == MetricDirection.HIGHER_BETTER.value:
        better = delta > 0
    elif direction == MetricDirection.LOWER_BETTER.value:
        better = delta < 0
    elif direction == MetricDirection.TARGET_RANGE.value and spec is not None:
        baseline_distance = _distance_from_range(baseline.median_value, spec)
        candidate_distance = _distance_from_range(candidate.median_value, spec)
        better = candidate_distance < baseline_distance
        magnitude = abs(candidate_distance - baseline_distance)
    else:
        result.verdict = ComparisonVerdict.UNCHANGED.value
        return result

    result.verdict = (
        ComparisonVerdict.IMPROVED.value if better else ComparisonVerdict.REGRESSED.value
    )
    if not better:
        result.severity = _severity(magnitude)
    return result


def _distance_from_range(value: float, spec: Any) -> float:
    low = spec.target_low if spec.target_low is not None else float("-inf")
    high = spec.target_high if spec.target_high is not None else float("inf")
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


@dataclass
class CandidateComparison:
    """Every metric, compared, with nothing collapsed away."""

    evaluation_id: str
    baseline_label: str
    candidate_label: str
    metrics: dict[str, MetricComparison] = field(default_factory=dict)
    schema_version: str = COMPARISON_SCHEMA_VERSION

    @property
    def regressions(self) -> list[MetricComparison]:
        return [m for m in self.metrics.values() if m.regressed]

    @property
    def improvements(self) -> list[MetricComparison]:
        return [m for m in self.metrics.values() if m.improved]

    def worst_severity(self) -> str:
        order = [
            RegressionSeverity.CRITICAL.value,
            RegressionSeverity.MAJOR.value,
            RegressionSeverity.MINOR.value,
            RegressionSeverity.INFO.value,
        ]
        severities = {m.severity for m in self.regressions}
        for severity in order:
            if severity in severities:
                return severity
        return RegressionSeverity.NONE.value

    def advisory_score(self) -> dict[str, Any]:
        """A composite, clearly labelled as advisory.

        Provided because people want one number, and withheld from every
        gate because one number cannot express "reliability improved and
        stereo collapsed". The regression list travels with it so the
        score can never be read alone.
        """
        judged = [
            m
            for m in self.metrics.values()
            if m.verdict in (ComparisonVerdict.IMPROVED.value, ComparisonVerdict.REGRESSED.value)
        ]
        if not judged:
            return {
                "score": None,
                "basis": "no metric moved beyond the suite's resolution",
                "improved": 0,
                "regressed": 0,
                "worst_severity": self.worst_severity(),
                "warning": "advisory only; gates read individual metrics, never this",
            }
        improved = sum(1 for m in judged if m.improved)
        return {
            "score": round(improved / len(judged), 4),
            "basis": "share of judged metrics that improved",
            "improved": improved,
            "regressed": len(judged) - improved,
            "worst_severity": self.worst_severity(),
            "warning": "advisory only; gates read individual metrics, never this",
        }

    def pareto_summary(self) -> dict[str, list[str]]:
        """Which metrics moved which way, without netting them off."""
        return {
            "improved": sorted(m.metric_name for m in self.improvements),
            "regressed": sorted(m.metric_name for m in self.regressions),
            "inconclusive": sorted(
                m.metric_name
                for m in self.metrics.values()
                if m.verdict == ComparisonVerdict.INCONCLUSIVE.value
            ),
            "not_measurable": sorted(
                m.metric_name
                for m in self.metrics.values()
                if m.verdict == ComparisonVerdict.NOT_MEASURABLE.value
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "metrics": {
                name: comparison.to_dict() for name, comparison in sorted(self.metrics.items())
            },
            "pareto": self.pareto_summary(),
            "worst_regression_severity": self.worst_severity(),
            "advisory_score": self.advisory_score(),
        }


def compare(
    evaluation_id: str,
    baseline: dict[str, Aggregate],
    candidate: dict[str, Aggregate],
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    noise_floor_relative: float = DEFAULT_NOISE_FLOOR_RELATIVE,
    rate_noise_floor: float = DEFAULT_RATE_NOISE_FLOOR,
) -> CandidateComparison:
    """Compare every metric either side produced."""
    result = CandidateComparison(
        evaluation_id=evaluation_id,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
    )
    for name in sorted(set(baseline) | set(candidate)):
        result.metrics[name] = compare_metric(
            name,
            baseline.get(name),
            candidate.get(name),
            noise_floor_relative=noise_floor_relative,
            rate_noise_floor=rate_noise_floor,
        )
    return result
