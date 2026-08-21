"""Deciding that something got worse, in a way an operator can argue with.

Every finding here is a comparison of two counted values against written
thresholds. There is no model, no learned normal, no score. The reason
is not conservatism: a detector nobody can check is a detector nobody
acts on, and an unactioned alert is worse than no alert because it looks
like coverage.

Four rules, applied in this order, and each one exists because of a
specific way naive detection embarrasses itself.

**Sample size first.** One failure in two requests is two requests, not a
regression. Below the policy's minimum the answer is INSUFFICIENT_DATA —
which is deliberately not NORMAL, because "we cannot tell" and "it is
fine" are different things and only one of them should let somebody stop
looking.

**Absolute delta, not only relative.** 0.1% to 0.2% is a 100% relative
increase and operationally nothing. A policy that fired on relative
change alone would spend its credibility on rounding.

**Direction is per metric.** Acceptance falling and retries rising are
the same news. The direction lives with the metric rather than in each
policy, so a new policy cannot get it backwards.

**Severity is earned.** CRITICAL means an operator should stop what they
are doing. A metric that doubled from a small base has not earned that,
however alarming the ratio looks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from luber_inference_observability.aggregation import (
    FALL_IS_BAD,
    RISE_IS_BAD,
    Aggregate,
    Metric,
    MetricStatus,
)
from luber_inference_observability.baselines import Baseline, BaselineStatus
from luber_inference_observability.dimensions import Segment
from luber_inference_observability.versions import REGRESSION_ENGINE_VERSION, version_block


class FindingType(StrEnum):
    """What kind of regression this is.

    One per metric that has an operator response, rather than one
    generic "metric moved". The type is what routes a finding to a
    runbook section, so a shared type would mean a shared response to
    problems that need different ones.
    """

    FIRST_CANDIDATE_ACCEPTANCE_DROP = "FIRST_CANDIDATE_ACCEPTANCE_DROP"
    QUALITY_RETRY_RATE_INCREASE = "QUALITY_RETRY_RATE_INCREASE"
    RETRY_EXHAUSTION_INCREASE = "RETRY_EXHAUSTION_INCREASE"
    INVALID_AUDIO_INCREASE = "INVALID_AUDIO_INCREASE"
    EARLY_COLLAPSE_INCREASE = "EARLY_COLLAPSE_INCREASE"
    DURATION_FAILURE_INCREASE = "DURATION_FAILURE_INCREASE"
    SEVERE_CLIPPING_INCREASE = "SEVERE_CLIPPING_INCREASE"
    SILENT_OUTPUT_INCREASE = "SILENT_OUTPUT_INCREASE"
    SPECTRAL_COLLAPSE_INCREASE = "SPECTRAL_COLLAPSE_INCREASE"
    GENERATION_FAILURE_INCREASE = "GENERATION_FAILURE_INCREASE"
    PROVIDER_FAILURE_INCREASE = "PROVIDER_FAILURE_INCREASE"
    PROVIDER_TIMEOUT_INCREASE = "PROVIDER_TIMEOUT_INCREASE"
    PROVIDER_CALL_INCREASE = "PROVIDER_CALL_INCREASE"
    LATENCY_REGRESSION = "LATENCY_REGRESSION"


class Category(StrEnum):
    """Availability and quality are different problems with different fixes.

    A provider timing out is not a model producing bad songs. Keeping the
    taxonomy explicit is what stops an operator being sent to inspect
    audio when the machine is simply unreachable — and what stops a
    genuine quality regression being written off as "the provider was
    flaky".
    """

    #: The provider did not answer.
    AVAILABILITY = "AVAILABILITY"
    #: The provider answered and what it produced was worse.
    QUALITY = "QUALITY"
    #: It still worked, and it cost more — time or inferences.
    EFFICIENCY = "EFFICIENCY"


class Severity(StrEnum):
    INFO = "INFO"
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    CRITICAL = "CRITICAL"


class Status(StrEnum):
    """The verdict for one metric in one segment."""

    NORMAL = "NORMAL"
    REGRESSED = "REGRESSED"
    #: Samples exist but not enough of them. Never NORMAL.
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    #: Nothing at all in the current window.
    NO_DATA = "NO_DATA"
    #: The baseline is not ready — a new revision, or a quiet week.
    BASELINE_BUILDING = "BASELINE_BUILDING"


@dataclass(frozen=True)
class Thresholds:
    """When a movement counts, for one metric.

    Both deltas must be crossed, not either. That conjunction is the
    whole defence against alert fatigue: relative alone fires on
    rounding, absolute alone fires on metrics that are legitimately
    large and noisy.
    """

    #: How far the value must move in absolute terms. For a rate this is
    #: a proportion (0.02 = two percentage points); for latency, seconds.
    minimum_absolute_delta: float
    #: And how far in relative terms. 0.5 = a 50% move.
    minimum_relative_delta: float
    #: Escalation points, both applied to the absolute delta.
    major_absolute_delta: float
    critical_absolute_delta: float
    #: Rows needed in the current window before any verdict but
    #: INSUFFICIENT_DATA is possible.
    minimum_current_samples: int = 30
    minimum_baseline_samples: int = 50

    def severity_for(self, absolute_delta: float) -> str:
        if absolute_delta >= self.critical_absolute_delta:
            return Severity.CRITICAL.value
        if absolute_delta >= self.major_absolute_delta:
            return Severity.MAJOR.value
        return Severity.MINOR.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_absolute_delta": self.minimum_absolute_delta,
            "minimum_relative_delta": self.minimum_relative_delta,
            "major_absolute_delta": self.major_absolute_delta,
            "critical_absolute_delta": self.critical_absolute_delta,
            "minimum_current_samples": self.minimum_current_samples,
            "minimum_baseline_samples": self.minimum_baseline_samples,
        }


@dataclass(frozen=True)
class Rule:
    """One metric, watched, with its type and thresholds."""

    metric: str
    finding_type: str
    category: str
    thresholds: Thresholds
    #: Only for latency metrics: which quantile is compared. Comparing
    #: P95 against a baseline's P50 would report a disaster on a healthy
    #: system, so the fraction is part of the rule.
    quantile_fraction: float | None = None

    @property
    def direction(self) -> str:
        if self.metric in FALL_IS_BAD:
            return "fall"
        if self.metric in RISE_IS_BAD:
            return "rise"
        raise ValueError(f"no direction is defined for {self.metric!r}")


#: Rates in this system live between 0 and 1, so an absolute delta of
#: 0.02 is two percentage points. The numbers below were chosen against
#: what Phase 29 actually produces: the corpus rejects nothing, so a
#: healthy deployment sits near zero on every failure rate, and two
#: points of movement is a real change rather than noise.
_QUALITY = Thresholds(
    minimum_absolute_delta=0.02,
    minimum_relative_delta=0.50,
    major_absolute_delta=0.05,
    critical_absolute_delta=0.20,
)

#: Acceptance is a large number that moves slowly, so the same absolute
#: sensitivity applies but the escalation points are further apart —
#: acceptance dropping five points is worth a look, and dropping twenty
#: five means most requests now need a retry.
_ACCEPTANCE = Thresholds(
    minimum_absolute_delta=0.05,
    minimum_relative_delta=0.05,
    major_absolute_delta=0.15,
    critical_absolute_delta=0.25,
)

#: Availability failures should be near zero always, so a smaller
#: absolute movement counts, and the critical point is lower: a provider
#: failing a quarter of requests is an outage.
_AVAILABILITY = Thresholds(
    minimum_absolute_delta=0.02,
    minimum_relative_delta=0.50,
    major_absolute_delta=0.10,
    critical_absolute_delta=0.25,
)

#: Latency deltas are in seconds. Generation takes minutes, so ten
#: seconds at P95 is noise and a minute is not. Relative matters more
#: here than absolute, which is why the relative floor is higher.
_LATENCY = Thresholds(
    minimum_absolute_delta=10.0,
    minimum_relative_delta=0.50,
    major_absolute_delta=60.0,
    critical_absolute_delta=180.0,
    minimum_current_samples=30,
    minimum_baseline_samples=50,
)

#: Provider calls per generation. The default policy allows at most
#: three, so 0.3 of a call per request is a tenth of the ceiling being
#: consumed by retries that were not happening before.
_EFFICIENCY = Thresholds(
    minimum_absolute_delta=0.30,
    minimum_relative_delta=0.20,
    major_absolute_delta=0.60,
    critical_absolute_delta=1.00,
)


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule(
        Metric.FIRST_CANDIDATE_ACCEPT_RATE.value,
        FindingType.FIRST_CANDIDATE_ACCEPTANCE_DROP.value,
        Category.QUALITY.value,
        _ACCEPTANCE,
    ),
    Rule(
        Metric.QUALITY_RETRY_RATE.value,
        FindingType.QUALITY_RETRY_RATE_INCREASE.value,
        Category.QUALITY.value,
        _QUALITY,
    ),
    Rule(
        Metric.RETRY_EXHAUSTION_RATE.value,
        FindingType.RETRY_EXHAUSTION_INCREASE.value,
        Category.QUALITY.value,
        _QUALITY,
    ),
    Rule(
        Metric.GENERATION_FAILURE_RATE.value,
        FindingType.GENERATION_FAILURE_INCREASE.value,
        Category.QUALITY.value,
        _QUALITY,
    ),
    Rule(
        Metric.INVALID_AUDIO_RATE.value,
        FindingType.INVALID_AUDIO_INCREASE.value,
        Category.QUALITY.value,
        _QUALITY,
    ),
    Rule(
        Metric.EARLY_COLLAPSE_RATE.value,
        FindingType.EARLY_COLLAPSE_INCREASE.value,
        Category.QUALITY.value,
        _QUALITY,
    ),
    Rule(
        Metric.DURATION_FAILURE_RATE.value,
        FindingType.DURATION_FAILURE_INCREASE.value,
        Category.QUALITY.value,
        _QUALITY,
    ),
    Rule(
        Metric.SEVERE_CLIPPING_RATE.value,
        FindingType.SEVERE_CLIPPING_INCREASE.value,
        Category.QUALITY.value,
        _QUALITY,
    ),
    Rule(
        Metric.SILENT_OUTPUT_RATE.value,
        FindingType.SILENT_OUTPUT_INCREASE.value,
        Category.QUALITY.value,
        _QUALITY,
    ),
    Rule(
        Metric.SPECTRAL_COLLAPSE_RATE.value,
        FindingType.SPECTRAL_COLLAPSE_INCREASE.value,
        Category.QUALITY.value,
        _QUALITY,
    ),
    Rule(
        Metric.PROVIDER_FAILURE_RATE.value,
        FindingType.PROVIDER_FAILURE_INCREASE.value,
        Category.AVAILABILITY.value,
        _AVAILABILITY,
    ),
    Rule(
        Metric.PROVIDER_TIMEOUT_RATE.value,
        FindingType.PROVIDER_TIMEOUT_INCREASE.value,
        Category.AVAILABILITY.value,
        _AVAILABILITY,
    ),
    Rule(
        Metric.AVERAGE_PROVIDER_CALLS.value,
        FindingType.PROVIDER_CALL_INCREASE.value,
        Category.EFFICIENCY.value,
        _EFFICIENCY,
    ),
    Rule(
        Metric.TOTAL_LATENCY.value,
        FindingType.LATENCY_REGRESSION.value,
        Category.EFFICIENCY.value,
        _LATENCY,
        quantile_fraction=0.95,
    ),
    Rule(
        Metric.PROVIDER_LATENCY.value,
        FindingType.LATENCY_REGRESSION.value,
        Category.EFFICIENCY.value,
        _LATENCY,
        quantile_fraction=0.95,
    ),
)


class Recommendation(StrEnum):
    """What an operator might do. Nothing here is executed.

    Advisory by construction: Phase 30 detects and explains. Disabling a
    provider, lowering a duration cap or changing a QC threshold are all
    decisions with costs this system cannot weigh, and a detector that
    took them would be acting on the same evidence it is asking a human
    to check.
    """

    INVESTIGATE_PROVIDER = "INVESTIGATE_PROVIDER"
    COMPARE_PROVIDER_REVISION = "COMPARE_PROVIDER_REVISION"
    CHECK_RECENT_DEPLOYMENT = "CHECK_RECENT_DEPLOYMENT"
    INSPECT_DURATION_SEGMENT = "INSPECT_DURATION_SEGMENT"
    CHECK_SAMPLE_GENERATIONS = "CHECK_SAMPLE_GENERATIONS"
    CONSIDER_TEMPORARY_OPERATOR_POLICY_CHANGE = "CONSIDER_TEMPORARY_OPERATOR_POLICY_CHANGE"


@dataclass(frozen=True)
class RegressionFinding:
    """One metric, one segment, and everything behind the verdict.

    There is no field for a vague summary. Whatever a reader wants to
    challenge — the counts, the windows, the threshold that was crossed —
    is here, because a finding that cannot be checked will be either
    believed too much or ignored entirely.
    """

    finding_type: str
    category: str
    metric: str
    segment: Segment
    status: str
    severity: str = Severity.INFO.value

    baseline_value: float | None = None
    current_value: float | None = None
    absolute_delta: float | None = None
    relative_delta: float | None = None

    baseline_numerator: int | None = None
    baseline_denominator: int | None = None
    current_numerator: int | None = None
    current_denominator: int | None = None
    baseline_sample_count: int = 0
    current_sample_count: int = 0

    baseline_window: dict[str, Any] = field(default_factory=dict)
    current_window: dict[str, Any] = field(default_factory=dict)
    quantile_fraction: float | None = None
    thresholds: dict[str, Any] = field(default_factory=dict)
    threshold_crossed: str | None = None
    reason: str = ""
    recommendations: tuple[str, ...] = ()
    partial_history: bool = False
    regression_engine_version: str = REGRESSION_ENGINE_VERSION

    @property
    def regressed(self) -> bool:
        return self.status == Status.REGRESSED.value

    def explain(self) -> str:
        """One sentence of plain English, with the numbers in it.

        Deliberately says *what* moved and never *why*. "Early collapse
        rose after the deploy" is a correlation an operator can act on;
        "the deploy caused early collapse" is a claim this system has no
        evidence for.
        """
        where = self.segment.label()
        if self.status == Status.NO_DATA.value:
            return f"{self.metric}: no generations in the current window for {where}."
        if self.status == Status.INSUFFICIENT_DATA.value:
            return (
                f"{self.metric}: {self.current_sample_count} samples for {where} is below the "
                f"{self.thresholds.get('minimum_current_samples')} needed to compare."
            )
        if self.status == Status.BASELINE_BUILDING.value:
            return (
                f"{self.metric}: no comparable history for {where} yet "
                f"({self.baseline_sample_count} baseline samples)."
            )
        if not self.regressed:
            return f"{self.metric}: within normal range for {where}."

        if self.quantile_fraction is not None:
            return (
                f"{self.metric} at P{int(self.quantile_fraction * 100)} rose from "
                f"{_seconds(self.baseline_value)} to {_seconds(self.current_value)} "
                f"for {where} "
                f"({self.baseline_sample_count} → {self.current_sample_count} samples)."
            )
        if self.baseline_denominator is not None and self.current_denominator is not None:
            return (
                f"{self.metric} moved from "
                f"{self.baseline_numerator}/{self.baseline_denominator} "
                f"({_percent(self.baseline_value)}) to "
                f"{self.current_numerator}/{self.current_denominator} "
                f"({_percent(self.current_value)}) for {where}."
            )
        return (
            f"{self.metric} moved from {_number(self.baseline_value)} to "
            f"{_number(self.current_value)} for {where} "
            f"({self.current_sample_count} samples)."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **version_block(),
            "finding_type": self.finding_type,
            "category": self.category,
            "metric": self.metric,
            "segment": self.segment.to_dict(),
            "segment_label": self.segment.label(),
            "status": self.status,
            "severity": self.severity,
            "baseline_value": _r(self.baseline_value),
            "current_value": _r(self.current_value),
            "absolute_delta": _r(self.absolute_delta),
            "relative_delta": _r(self.relative_delta),
            "baseline_numerator": self.baseline_numerator,
            "baseline_denominator": self.baseline_denominator,
            "current_numerator": self.current_numerator,
            "current_denominator": self.current_denominator,
            "baseline_sample_count": self.baseline_sample_count,
            "current_sample_count": self.current_sample_count,
            "baseline_window": self.baseline_window,
            "current_window": self.current_window,
            "quantile_fraction": self.quantile_fraction,
            "thresholds": self.thresholds,
            "threshold_crossed": self.threshold_crossed,
            "reason": self.reason,
            "explanation": self.explain(),
            "recommendations": list(self.recommendations),
            "partial_history": self.partial_history,
        }


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}s"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _recommendations(rule: Rule, segment: Segment) -> tuple[str, ...]:
    """What to look at. Never what to change automatically."""
    out: list[str] = []
    filters = segment.to_dict()
    if rule.category == Category.AVAILABILITY.value:
        out.append(Recommendation.INVESTIGATE_PROVIDER.value)
    else:
        out.append(Recommendation.CHECK_SAMPLE_GENERATIONS.value)
    if "provider_revision" in filters or "model_version" in filters:
        out.append(Recommendation.COMPARE_PROVIDER_REVISION.value)
    if "duration_bucket" in filters:
        out.append(Recommendation.INSPECT_DURATION_SEGMENT.value)
    out.append(Recommendation.CHECK_RECENT_DEPLOYMENT.value)
    return tuple(dict.fromkeys(out))


def compare(
    rule: Rule,
    *,
    baseline: Baseline,
    current: Aggregate,
) -> RegressionFinding:
    """Decide whether *current* is worse than *baseline* for one rule.

    The order of the guards is the point. Sample size is checked before
    any arithmetic, so a tiny window can never produce a delta that then
    has to be argued away.
    """
    thresholds = rule.thresholds
    segment = current.segment
    current_value, current_numerator, current_denominator, current_samples = _current(rule, current)

    def finding(
        status: str,
        *,
        severity: str = Severity.INFO.value,
        reason: str = "",
        include_current: bool = True,
        deltas: tuple[float, float] | None = None,
        threshold_crossed: str | None = None,
        recommendations: tuple[str, ...] = (),
    ) -> RegressionFinding:
        """One constructor, so every branch fills the same fields.

        Written as a closure over the comparison rather than as a dict
        splatted into the dataclass: a `**shared` splat type-checks as
        nothing, and this record's whole job is to be complete.
        """
        return RegressionFinding(
            finding_type=rule.finding_type,
            category=rule.category,
            metric=rule.metric,
            segment=segment,
            status=status,
            severity=severity,
            baseline_value=baseline.value,
            current_value=current_value if include_current else None,
            absolute_delta=deltas[0] if deltas else None,
            relative_delta=deltas[1] if deltas else None,
            baseline_numerator=baseline.numerator,
            baseline_denominator=baseline.denominator,
            current_numerator=current_numerator if include_current else None,
            current_denominator=current_denominator if include_current else None,
            baseline_sample_count=baseline.sample_count,
            current_sample_count=current_samples,
            baseline_window=baseline.window.to_dict(),
            current_window=current.window.to_dict(),
            quantile_fraction=rule.quantile_fraction,
            thresholds=thresholds.to_dict(),
            threshold_crossed=threshold_crossed,
            reason=reason,
            recommendations=recommendations,
            partial_history=current.partial_history,
        )

    if current_samples == 0:
        return finding(Status.NO_DATA.value, include_current=False)

    if current_samples < thresholds.minimum_current_samples:
        return finding(
            Status.INSUFFICIENT_DATA.value,
            reason=(
                f"{current_samples} samples in the current window is below the "
                f"{thresholds.minimum_current_samples} this rule requires"
            ),
        )

    if baseline.status == BaselineStatus.NO_DATA.value or baseline.value is None:
        return finding(
            Status.BASELINE_BUILDING.value,
            reason="no baseline history for this segment yet",
        )

    if baseline.sample_count < thresholds.minimum_baseline_samples:
        return finding(
            Status.BASELINE_BUILDING.value,
            reason=(
                f"the baseline holds {baseline.sample_count} samples, below the "
                f"{thresholds.minimum_baseline_samples} this rule requires"
            ),
        )

    if current_value is None:
        return finding(Status.NO_DATA.value, include_current=False)

    signed = current_value - baseline.value
    # Movement in the bad direction only. A failure rate falling is not a
    # regression, and reporting it as "changed" would bury the ones that
    # matter.
    absolute_delta = signed if rule.direction == "rise" else -signed
    relative_delta = (
        absolute_delta / baseline.value
        if baseline.value
        else float("inf")
        if absolute_delta > 0
        else 0.0
    )
    deltas = (absolute_delta, relative_delta)

    if absolute_delta < thresholds.minimum_absolute_delta:
        return finding(
            Status.NORMAL.value,
            reason=(
                f"moved {absolute_delta:.4f}, below the "
                f"{thresholds.minimum_absolute_delta} absolute minimum"
            ),
            deltas=deltas,
        )

    if relative_delta < thresholds.minimum_relative_delta:
        return finding(
            Status.NORMAL.value,
            reason=(
                f"moved {relative_delta * 100:.1f}% relative, below the "
                f"{thresholds.minimum_relative_delta * 100:.0f}% minimum"
            ),
            deltas=deltas,
        )

    return finding(
        Status.REGRESSED.value,
        severity=thresholds.severity_for(absolute_delta),
        threshold_crossed=(
            f"absolute \u2265 {thresholds.minimum_absolute_delta} and "
            f"relative \u2265 {thresholds.minimum_relative_delta}"
        ),
        reason=(
            f"moved {absolute_delta:.4f} absolute and {relative_delta * 100:.1f}% relative, "
            "crossing both minimums"
        ),
        deltas=deltas,
        recommendations=_recommendations(rule, segment),
    )


def _current(rule: Rule, current: Aggregate) -> tuple[float | None, int | None, int | None, int]:
    """The current value, its counts, and how many samples stand behind it."""
    if rule.metric in current.rates:
        rate = current.rates[rule.metric]
        if rate.status == MetricStatus.NO_DATA.value:
            return None, rate.numerator, rate.denominator, 0
        return rate.value, rate.numerator, rate.denominator, rate.denominator
    if rule.metric in current.distributions:
        distribution = current.distributions[rule.metric]
        fraction = rule.quantile_fraction or 0.95
        return distribution.at(fraction), None, None, distribution.count
    if rule.metric in current.averages:
        average = current.averages[rule.metric]
        return average.value, None, None, average.count
    return None, None, None, 0


def detect(
    *,
    current: Aggregate,
    baselines: dict[str, Baseline],
    rules: tuple[Rule, ...] = DEFAULT_RULES,
) -> list[RegressionFinding]:
    """Run every rule that has a baseline. Returns findings of all statuses.

    Non-regressions are returned rather than filtered because
    "we checked and it is fine" and "we did not check" are different
    answers, and only the caller knows which it needs.
    """
    findings: list[RegressionFinding] = []
    for rule in rules:
        baseline = baselines.get(rule.metric)
        if baseline is None:
            continue
        findings.append(compare(rule, baseline=baseline, current=current))
    return findings


def regressions(findings: list[RegressionFinding]) -> list[RegressionFinding]:
    """Only the ones that crossed, worst first."""
    order = {
        Severity.CRITICAL.value: 0,
        Severity.MAJOR.value: 1,
        Severity.MINOR.value: 2,
        Severity.INFO.value: 3,
    }
    return sorted(
        (item for item in findings if item.regressed),
        key=lambda item: (order.get(item.severity, 9), -(item.absolute_delta or 0.0)),
    )


__all__ = [
    "DEFAULT_RULES",
    "Category",
    "FindingType",
    "Recommendation",
    "RegressionFinding",
    "Rule",
    "Severity",
    "Status",
    "Thresholds",
    "compare",
    "detect",
    "regressions",
]
