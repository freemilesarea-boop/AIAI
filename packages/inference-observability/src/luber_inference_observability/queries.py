"""The questions the dashboard, the CLI and the reports all ask.

One place, so the number on a card and the number in a report are the
same number computed the same way. Two implementations of "retry rate"
eventually disagree, and the one an operator trusts is whichever they
saw first.

Every function here takes a store and a window and returns a plain
structure. Nothing caches, nothing mutates, and nothing here decides
what to do about an answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from luber_inference_observability.aggregation import (
    Aggregate,
    Metric,
    aggregate,
    group,
)
from luber_inference_observability.baselines import (
    DEFAULT_BASELINE_GAP,
    DEFAULT_BASELINE_SPAN,
    Baseline,
    baselines_from,
    rolling_window,
)
from luber_inference_observability.dimensions import Segment
from luber_inference_observability.events import InferenceObservation
from luber_inference_observability.incidents import IncidentLedger
from luber_inference_observability.regressions import (
    DEFAULT_RULES,
    RegressionFinding,
    Rule,
    detect,
    regressions,
)
from luber_inference_observability.storage import ObservationStore
from luber_inference_observability.versions import PHASE29_BOUNDARY_COMMIT, version_block
from luber_inference_observability.windows import TimeWindow, step_for, trend_step

#: Metrics shown on the health overview, in the order they are read.
#: Deliberately short: nine numbers an operator can take in, not every
#: metric the system knows how to compute.
OVERVIEW_METRICS: tuple[str, ...] = (
    Metric.GENERATION_SUCCESS_RATE.value,
    Metric.FIRST_CANDIDATE_ACCEPT_RATE.value,
    Metric.QUALITY_RETRY_RATE.value,
    Metric.RETRY_EXHAUSTION_RATE.value,
    Metric.PROVIDER_FAILURE_RATE.value,
    Metric.EARLY_COLLAPSE_RATE.value,
)


@dataclass(frozen=True)
class TrendPoint:
    """One bucket of a trend line, with its sample count.

    The count travels with the value because a chart without one invites
    the eye to read a spike from three requests as the same event as a
    spike from three hundred.
    """

    window: TimeWindow
    values: dict[str, float | None]
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.window.start.isoformat(),
            "end": self.window.end.isoformat(),
            "sample_count": self.sample_count,
            "values": {name: _r(value) for name, value in sorted(self.values.items())},
        }


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def summary(
    store: ObservationStore,
    *,
    window: TimeWindow,
    segment: Segment | None = None,
) -> dict[str, Any]:
    """The health overview: what happened, over what, in this window."""
    rows = list(store.select(window, segment=segment))
    result = aggregate(rows, window=window, segment=segment or Segment())
    return {
        **version_block(),
        "window": window.to_dict(),
        "segment": (segment or Segment()).to_dict(),
        "sample_count": result.sample_count,
        "counters": result.counters.to_dict(),
        "overview": {
            name: result.rate(name).to_dict() for name in OVERVIEW_METRICS if name in result.rates
        },
        "rates": {name: item.to_dict() for name, item in sorted(result.rates.items())},
        "averages": {name: item.to_dict() for name, item in sorted(result.averages.items())},
        "latency": {name: item.to_dict() for name, item in sorted(result.distributions.items())},
        "findings": {
            "critical": dict(sorted(result.counters.finding_counts.items())),
            "soft": dict(sorted(result.counters.soft_finding_counts.items())),
        },
        "data_quality": dict(sorted(result.counters.data_quality_counts.items())),
        "coverage": coverage(rows, window),
    }


def coverage(rows: Sequence[InferenceObservation], window: TimeWindow) -> dict[str, Any]:
    """How much of this window can answer QC questions at all.

    Generations from before Phase 29 have no candidate trace. Their
    retries are unknown rather than zero, so every rate that needs the
    trace excludes them — and says so here, because a denominator that
    quietly shrank is a number nobody can reconcile.
    """
    total = len(rows)
    with_qc = sum(1 for row in rows if row.qc_data_available)
    return {
        "observations": total,
        "with_qc_data": with_qc,
        "without_qc_data": total - with_qc,
        "complete": total == with_qc,
        "partial": total > 0 and with_qc < total,
        "boundary_commit": PHASE29_BOUNDARY_COMMIT,
        "note": (
            None
            if total == with_qc
            else (
                f"{total - with_qc} of {total} generations in this window predate the "
                f"candidate trace ({PHASE29_BOUNDARY_COMMIT}). Their retry counts are "
                "unknown, not zero, and they are excluded from candidate-derived rates."
            )
        ),
    }


def trend(
    store: ObservationStore,
    *,
    window: TimeWindow,
    metrics: tuple[str, ...],
    segment: Segment | None = None,
    size: str | None = None,
) -> dict[str, Any]:
    """A metric over time, bucketed.

    Buckets with no samples carry `None` rather than 0. A chart that
    drew zero through a quiet night would show a recovery that never
    happened.
    """
    step = trend_step(size) if size else step_for(window)
    rows = list(store.select(window, segment=segment))
    points: list[TrendPoint] = []

    for bucket in window.buckets(step):
        inside = [row for row in rows if bucket.contains(row.occurred_at)]
        result = aggregate(inside, window=bucket, segment=segment or Segment())
        values: dict[str, float | None] = {}
        for metric in metrics:
            if metric in result.rates:
                values[metric] = result.rates[metric].value
            elif metric in result.averages:
                values[metric] = result.averages[metric].value
            elif metric in result.distributions:
                values[metric] = result.distributions[metric].p95
            else:
                values[metric] = None
        points.append(TrendPoint(window=bucket, values=values, sample_count=len(inside)))

    return {
        **version_block(),
        "window": window.to_dict(),
        "segment": (segment or Segment()).to_dict(),
        "metrics": list(metrics),
        "bucket_seconds": step.total_seconds(),
        "points": [point.to_dict() for point in points],
        "has_data": any(point.sample_count for point in points),
    }


def build_baselines(
    store: ObservationStore,
    *,
    current: TimeWindow,
    segment: Segment | None = None,
    metrics: tuple[str, ...] | None = None,
    span: timedelta = DEFAULT_BASELINE_SPAN,
    gap: timedelta = DEFAULT_BASELINE_GAP,
) -> dict[str, Baseline]:
    """Rolling baselines for one segment, measured before *current*."""
    reference = rolling_window(current, span=span, gap=gap)
    rows = list(store.select(reference, segment=segment))
    result = aggregate(rows, window=reference, segment=segment or Segment())
    wanted = metrics or tuple(rule.metric for rule in DEFAULT_RULES)
    return baselines_from(result, metrics=wanted)


def evaluate(
    store: ObservationStore,
    *,
    current: TimeWindow,
    segment: Segment | None = None,
    baselines: dict[str, Baseline] | None = None,
    rules: tuple[Rule, ...] = DEFAULT_RULES,
    span: timedelta = DEFAULT_BASELINE_SPAN,
    gap: timedelta = DEFAULT_BASELINE_GAP,
) -> list[RegressionFinding]:
    """Run every rule for one segment against a rolling or supplied baseline."""
    reference = baselines or build_baselines(
        store, current=current, segment=segment, span=span, gap=gap
    )
    rows = list(store.select(current, segment=segment))
    result = aggregate(rows, window=current, segment=segment or Segment())
    return detect(current=result, baselines=reference, rules=rules)


def evaluate_segments(
    store: ObservationStore,
    *,
    current: TimeWindow,
    by: tuple[str, ...],
    rules: tuple[Rule, ...] = DEFAULT_RULES,
    span: timedelta = DEFAULT_BASELINE_SPAN,
    gap: timedelta = DEFAULT_BASELINE_GAP,
    include_overall: bool = True,
) -> list[RegressionFinding]:
    """Evaluate every segment separately, plus the whole population.

    Both, because they catch different things. A regression confined to
    one duration bucket is invisible in the overall rate; a broad
    degradation is easier to read as one finding than as fifteen.

    A *degenerate* split is skipped. When every observation in the window
    shares one value — a single provider revision, which is the normal
    case — the segment and the whole population are the same rows, and
    evaluating both raises two incidents for one problem. An operator
    reading a list of six when there are three learns to skim it, which
    is the failure this whole module is built to avoid.
    """
    findings: list[RegressionFinding] = []
    rows = list(store.select(current))
    grouped = group(rows, window=current, by=by)
    degenerate = len(grouped) <= 1

    if include_overall:
        findings.extend(evaluate(store, current=current, rules=rules, span=span, gap=gap))

    if degenerate and include_overall:
        return findings

    for segment in grouped:
        findings.extend(
            evaluate(store, current=current, segment=segment, rules=rules, span=span, gap=gap)
        )

    # A revision too new for a historical baseline is judged against its
    # peers instead. Folded in here rather than only into
    # `run_detection` so the dashboard's regression list and the
    # detector's incidents cannot disagree about what is wrong — an
    # operator seeing an incident with no matching finding, or the
    # reverse, has no way to tell which one to believe.
    findings.extend(evaluate_new_revisions(store, current=current, rules=rules))
    return findings


def evaluate_new_revisions(
    store: ObservationStore,
    *,
    current: TimeWindow,
    rules: tuple[Rule, ...] = DEFAULT_RULES,
    minimum_samples: int = 30,
) -> list[RegressionFinding]:
    """Judge a revision that has no history, against its peers right now.

    A revision that shipped this morning has no rolling baseline, so the
    ordinary path answers BASELINE_BUILDING — correct, and useless if the
    new revision is the reason everything is on fire. The rollout is
    exactly when somebody needs to know.

    So a revision without history is compared against **every other
    revision in the same window**. Same period, so a traffic shift or a
    slow afternoon hits both sides equally; different code, which is the
    thing being asked about. That is a weaker claim than a historical
    baseline and it is labelled as one: the finding's baseline window is
    the current window, and the segment names the revision under test.

    It is still not a controlled experiment. Requests are not randomised
    between revisions, so a revision serving a different traffic mix can
    look worse for reasons that have nothing to do with the model. The
    finding says what was measured; deciding what it means is the
    operator's.
    """
    rows = list(store.select(current))
    revisions = {row.provider_revision for row in rows if row.provider_revision != "UNKNOWN"}
    if len(revisions) < 2:
        return []

    findings: list[RegressionFinding] = []
    for revision in sorted(revisions):
        segment = Segment.of(provider_revision=revision)
        subject = [row for row in rows if row.provider_revision == revision]
        peers = [row for row in rows if row.provider_revision != revision]
        if len(subject) < minimum_samples or len(peers) < minimum_samples:
            continue

        # Only for revisions the historical path cannot judge. A revision
        # with history gets the stronger comparison, and running both
        # would raise two incidents for one problem.
        historical = build_baselines(store, current=current, segment=segment)
        if any(baseline.ready for baseline in historical.values()):
            continue

        peer_aggregate = aggregate(
            peers, window=current, segment=Segment.of(provider_revision="OTHER_REVISIONS")
        )
        reference = baselines_from(
            peer_aggregate,
            metrics=tuple(rule.metric for rule in rules),
            minimum_samples=minimum_samples,
        )
        subject_aggregate = aggregate(subject, window=current, segment=segment)
        findings.extend(detect(current=subject_aggregate, baselines=reference, rules=rules))
    return findings


def run_detection(
    store: ObservationStore,
    *,
    current: TimeWindow,
    ledger: IncidentLedger,
    by: tuple[str, ...] = ("provider_revision",),
    rules: tuple[Rule, ...] = DEFAULT_RULES,
    at: datetime | None = None,
    span: timedelta = DEFAULT_BASELINE_SPAN,
    gap: timedelta = DEFAULT_BASELINE_GAP,
) -> dict[str, Any]:
    """One full evaluation: findings in, incidents updated, summary out.

    This is what a scheduled run calls. Running it repeatedly over an
    unchanged window updates one incident rather than creating many —
    that property is the ledger's, and it is asserted by a test that
    runs this fifty times.
    """
    moment = (at or datetime.now(UTC)).astimezone(UTC)
    findings = evaluate_segments(store, current=current, by=by, rules=rules, span=span, gap=gap)
    touched = ledger.apply(findings, at=moment)
    crossed = regressions(findings)
    return {
        **version_block(),
        "evaluated_at": moment.isoformat(),
        "window": current.to_dict(),
        "grouped_by": list(by),
        "findings_evaluated": len(findings),
        "regressions": [item.to_dict() for item in crossed],
        "incidents_touched": [item.incident_id for item in touched],
        "open_incidents": len(ledger.active()),
    }


def top_segments(
    store: ObservationStore,
    *,
    window: TimeWindow,
    by: tuple[str, ...],
    metric: str = Metric.GENERATION_FAILURE_RATE.value,
    minimum_samples: int = 30,
    limit: int = 10,
) -> dict[str, Any]:
    """The worst segments by one metric, among those big enough to rank.

    The minimum is not optional. Ranking by failure rate without one
    puts "1 of 1 failed" at the top of every list, which is a segment of
    one request wearing the shape of a crisis.
    """
    rows = list(store.select(window))
    grouped = group(rows, window=window, by=by)
    ranked: list[dict[str, Any]] = []
    skipped = 0

    for segment, result in grouped.items():
        rate = result.rate(metric)
        if rate.denominator < minimum_samples:
            skipped += 1
            continue
        ranked.append(
            {
                "segment": segment.to_dict(),
                "segment_label": segment.label(),
                "metric": metric,
                "value": _r(rate.value),
                "numerator": rate.numerator,
                "denominator": rate.denominator,
                "render": rate.render(),
                "sample_count": result.sample_count,
            }
        )

    ranked.sort(key=lambda item: (-(item["value"] or 0.0), -item["denominator"]))
    return {
        **version_block(),
        "window": window.to_dict(),
        "grouped_by": list(by),
        "metric": metric,
        "minimum_samples": minimum_samples,
        "segments": ranked[:limit],
        "segments_considered": len(grouped),
        "segments_below_minimum": skipped,
    }


def compare_windows(
    store: ObservationStore,
    *,
    before: TimeWindow,
    after: TimeWindow,
    segment: Segment | None = None,
    metrics: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Two windows side by side, with no claim about what connects them.

    Used for before/after a deployment and for revision A against
    revision B. It reports two measurements and their difference. It does
    not say the deployment caused the difference, because nothing here
    could know that — a rollout and a traffic change on the same
    afternoon look identical from inside this data.
    """
    wanted = metrics or (
        Metric.GENERATION_SUCCESS_RATE.value,
        Metric.FIRST_CANDIDATE_ACCEPT_RATE.value,
        Metric.QUALITY_RETRY_RATE.value,
        Metric.RETRY_EXHAUSTION_RATE.value,
        Metric.DURATION_FAILURE_RATE.value,
        Metric.EARLY_COLLAPSE_RATE.value,
        Metric.PROVIDER_FAILURE_RATE.value,
    )
    left = aggregate(
        list(store.select(before, segment=segment)), window=before, segment=segment or Segment()
    )
    right = aggregate(
        list(store.select(after, segment=segment)), window=after, segment=segment or Segment()
    )
    return {
        **version_block(),
        "before": _side(left, wanted),
        "after": _side(right, wanted),
        "deltas": _deltas(left, right, wanted),
        "caveat": (
            "These are two measurements of two periods. Any change is a correlation with "
            "whatever else happened between them; this system has no evidence of cause."
        ),
    }


def compare_revisions(
    store: ObservationStore,
    *,
    window: TimeWindow,
    left_revision: str,
    right_revision: str,
    metrics: tuple[str, ...] | None = None,
    minimum_samples: int = 30,
) -> dict[str, Any]:
    """Two provider revisions over the same period.

    Same window for both, because comparing one revision's Tuesday to
    another's Saturday compares the days as much as the models.
    """
    wanted = metrics or (
        Metric.GENERATION_SUCCESS_RATE.value,
        Metric.FIRST_CANDIDATE_ACCEPT_RATE.value,
        Metric.QUALITY_RETRY_RATE.value,
        Metric.RETRY_EXHAUSTION_RATE.value,
        Metric.DURATION_FAILURE_RATE.value,
        Metric.EARLY_COLLAPSE_RATE.value,
        Metric.TOTAL_LATENCY.value,
    )
    left_segment = Segment.of(provider_revision=left_revision)
    right_segment = Segment.of(provider_revision=right_revision)
    left = aggregate(
        list(store.select(window, segment=left_segment)), window=window, segment=left_segment
    )
    right = aggregate(
        list(store.select(window, segment=right_segment)), window=window, segment=right_segment
    )

    sufficient = left.sample_count >= minimum_samples and right.sample_count >= minimum_samples
    return {
        **version_block(),
        "window": window.to_dict(),
        "left_revision": left_revision,
        "right_revision": right_revision,
        "minimum_samples": minimum_samples,
        "sufficient_data": sufficient,
        "status": "OK" if sufficient else "INSUFFICIENT_DATA",
        "left": _side(left, wanted),
        "right": _side(right, wanted),
        "deltas": _deltas(left, right, wanted) if sufficient else {},
        "caveat": (
            "Two revisions measured over the same window. Traffic mix may differ between "
            "them; a difference here is a difference in what was observed, not a "
            "controlled experiment."
        ),
    }


def _side(result: Aggregate, metrics: tuple[str, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "window": result.window.to_dict(),
        "segment": result.segment.to_dict(),
        "sample_count": result.sample_count,
        "partial_history": result.partial_history,
        "metrics": {},
    }
    for metric in metrics:
        if metric in result.rates:
            payload["metrics"][metric] = result.rates[metric].to_dict()
        elif metric in result.averages:
            payload["metrics"][metric] = result.averages[metric].to_dict()
        elif metric in result.distributions:
            payload["metrics"][metric] = result.distributions[metric].to_dict()
    return payload


def _deltas(left: Aggregate, right: Aggregate, metrics: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in metrics:
        before = _value(left, metric)
        after = _value(right, metric)
        if before is None or after is None:
            out[metric] = {"status": "NO_DATA", "before": before, "after": after}
            continue
        absolute = after - before
        out[metric] = {
            "status": "OK",
            "before": _r(before),
            "after": _r(after),
            "absolute_delta": _r(absolute),
            "relative_delta": _r(absolute / before) if before else None,
        }
    return out


def _value(result: Aggregate, metric: str) -> float | None:
    if metric in result.rates:
        return result.rates[metric].value
    if metric in result.averages:
        return result.averages[metric].value
    if metric in result.distributions:
        return result.distributions[metric].p95
    return None


__all__ = [
    "OVERVIEW_METRICS",
    "TrendPoint",
    "build_baselines",
    "compare_revisions",
    "compare_windows",
    "coverage",
    "evaluate",
    "evaluate_segments",
    "run_detection",
    "summary",
    "top_segments",
    "trend",
]
