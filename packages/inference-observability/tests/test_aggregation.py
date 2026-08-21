"""Counting, and the three ways a count can lie.

Every assertion here is about a number being unreadable rather than
wrong. A rate without its counts, a zero standing in for an empty
window, a mean standing in for a distribution — none of them is a bug in
the arithmetic, and all three send an operator somewhere useless.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from obs_fixtures import NOW, observation

from luber_inference_observability import TimeWindow, aggregate, group
from luber_inference_observability.aggregation import (
    Distribution,
    Metric,
    MetricStatus,
    Rate,
    quantile,
)
from luber_inference_observability.dimensions import Segment
from luber_inference_observability.windows import WindowSize, duration_of


def window() -> TimeWindow:
    return TimeWindow.ending_at(NOW, "1h")


def rows(count: int, *, start_index: int = 0, **kwargs):
    """*count* observations inside the last hour.

    ``start_index`` keeps generation ids distinct when a test builds two
    populations. The store is keyed on the id, so overlapping ranges
    would silently collapse into one set and the test would be counting
    something other than what it says.
    """
    start = NOW - timedelta(minutes=55)
    return [
        observation(start_index + index, start + timedelta(seconds=index * 3), **kwargs)
        for index in range(count)
    ]


# ── rates carry their counts ─────────────────────────────────────────


def test_a_rate_cannot_be_rendered_without_its_counts():
    rate = Rate(name="quality_retry_rate", numerator=12, denominator=420)
    assert "12/420" in rate.render()
    assert "2.86%" in rate.render()


def test_an_empty_denominator_is_no_data_not_zero():
    rate = Rate(name="quality_retry_rate", numerator=0, denominator=0)
    assert rate.status == MetricStatus.NO_DATA.value
    assert rate.value is None
    assert "NO_DATA" in rate.render()
    # The distinction: nothing happened, not nothing failed.
    assert "0.00%" not in rate.render()


def test_an_empty_window_produces_no_data_everywhere():
    result = aggregate([], window=window())
    assert result.sample_count == 0
    for rate in result.rates.values():
        assert rate.status == MetricStatus.NO_DATA.value
    for distribution in result.distributions.values():
        assert distribution.status == MetricStatus.NO_DATA.value


# ── latency ──────────────────────────────────────────────────────────


def test_a_quantile_returns_a_measurement_that_actually_happened():
    """Interpolating would invent a latency nothing experienced, and an
    operator going to find that request would find nothing."""
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert quantile(values, 0.5) in values
    assert quantile(values, 0.95) in values
    assert quantile(values, 1.0) == 100.0


def test_a_distribution_is_never_reported_as_a_mean_alone():
    distribution = Distribution.of("total", [1.0, 1.0, 1.0, 1.0, 100.0])
    payload = distribution.to_dict()
    assert payload["p50"] == 1.0
    assert payload["p95"] == 100.0
    # The mean is present and is not the headline.
    assert payload["mean"] is not None
    assert payload["count"] == 5


def test_a_distribution_of_nothing_is_no_data():
    assert Distribution.of("total", []).status == MetricStatus.NO_DATA.value
    assert Distribution.of("total", [None, None]).count == 0


# ── denominators ─────────────────────────────────────────────────────


def test_cancellations_are_excluded_from_delivery_success():
    """A cancelled run neither succeeded nor failed at making music."""
    result = aggregate(
        rows(10) + rows(5, status="CANCELLED", start_index=1000),
        window=window(),
    )
    success = result.rate(Metric.GENERATION_SUCCESS_RATE.value)
    assert success.denominator == 10
    assert success.excluded == 5


def test_rows_without_a_trace_are_excluded_and_counted():
    result = aggregate(rows(10) + rows(5, qc_data=False), window=window())
    retry = result.rate(Metric.QUALITY_RETRY_RATE.value)
    assert retry.denominator == 10
    assert retry.excluded == 5
    assert result.counters.without_qc_data == 5
    assert result.partial_history is True


def test_a_soft_finding_is_never_counted_as_a_failure():
    result = aggregate(rows(10, soft=("NARROW_STEREO", "HIGH_HARSHNESS_PROXY")), window=window())
    assert result.counters.soft_finding_counts["NARROW_STEREO"] == 10
    assert result.counters.finding_counts == {}
    assert result.rate(Metric.GENERATION_FAILURE_RATE.value).numerator == 0


def test_a_finding_seen_twice_in_one_generation_counts_once():
    """The observation deduplicates across attempts: one generation with
    a problem is one generation, not two."""
    result = aggregate(rows(1, critical=("EARLY_COLLAPSE",)), window=window())
    assert result.counters.finding_counts["EARLY_COLLAPSE"] == 1


def test_duration_failures_combine_short_and_long():
    result = aggregate(
        rows(5, critical=("DURATION_SHORT",))
        + rows(5, critical=("DURATION_LONG",), start_index=2000),
        window=window(),
    )
    assert result.rate(Metric.DURATION_FAILURE_RATE.value).numerator == 10


def test_availability_failures_are_separate_from_quality_ones():
    result = aggregate(
        rows(5, critical=("PROVIDER_TIMEOUT",))
        + rows(5, critical=("EARLY_COLLAPSE",), start_index=3000),
        window=window(),
    )
    assert result.rate(Metric.PROVIDER_FAILURE_RATE.value).numerator == 5
    assert result.rate(Metric.EARLY_COLLAPSE_RATE.value).numerator == 5


# ── grouping ─────────────────────────────────────────────────────────


def test_grouping_splits_the_population_without_losing_anybody():
    population = rows(10, duration_bucket="61_120") + rows(
        6, duration_bucket="181_240", start_index=4000
    )
    grouped = group(population, window=window(), by=("duration_bucket",))

    assert sum(item.sample_count for item in grouped.values()) == 16
    labels = {segment.to_dict()["duration_bucket"] for segment in grouped}
    assert labels == {"61_120", "181_240"}


def test_a_segment_filter_selects_only_its_rows():
    segment = Segment.of(duration_bucket="181_240")
    population = rows(10) + rows(6, duration_bucket="181_240", start_index=5000)
    matched = [row for row in population if segment.matches(row)]
    assert len(matched) == 6


def test_the_same_filters_in_a_different_order_are_the_same_segment():
    """Otherwise an incident would be reopened under a new identity
    because a caller passed its filters differently."""
    left = Segment.of(provider="ace_step", duration_bucket="61_120")
    right = Segment.of(duration_bucket="61_120", provider="ace_step")
    assert left == right


# ── windows ──────────────────────────────────────────────────────────


def test_windows_are_half_open_so_adjacent_ones_tile():
    first = TimeWindow.ending_at(NOW, "1h")
    second = first.shifted(-duration_of("1h"))
    boundary = first.start
    assert first.contains(boundary)
    assert not second.contains(boundary)


def test_a_window_must_be_timezone_aware():
    from datetime import datetime as naive_datetime

    with pytest.raises(ValueError, match="timezone-aware"):
        TimeWindow(naive_datetime(2026, 8, 21, 11), naive_datetime(2026, 8, 21, 12))


def test_a_window_must_end_after_it_starts():
    with pytest.raises(ValueError, match="must end after"):
        TimeWindow(NOW, NOW - timedelta(hours=1))


def test_bucketing_never_overhangs_the_window():
    """A trend point covering time that has not happened would show a
    cliff at the right edge of every chart."""
    buckets = TimeWindow.ending_at(NOW, "1h").buckets(timedelta(minutes=7))
    assert buckets[-1].end == NOW
    assert all(bucket.end <= NOW for bucket in buckets)


def test_a_baseline_window_stops_short_of_the_current_one():
    """So a live regression is not learned as normal."""
    current = TimeWindow.ending_at(NOW, "1h")
    baseline = current.preceding(timedelta(days=7), gap=timedelta(hours=1))
    assert baseline.end < current.start
    assert baseline.duration == timedelta(days=7)


@pytest.mark.parametrize("size", [item.value for item in WindowSize])
def test_every_named_window_is_supported(size):
    assert duration_of(size).total_seconds() > 0
    assert TimeWindow.ending_at(NOW, size).duration == duration_of(size)
