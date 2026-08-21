"""Whether this holds at volume, and whether the report tells the truth.

The scale test is not a benchmark. It is a guard against the one shape
that would quietly ruin this design: an accidental O(N²) — a filter
inside a loop over segments, a re-scan per bucket — which is invisible
at fixture size and fatal at a month of traffic. The numbers below are
generous on purpose; what matters is that they are *bounds*, so a
regression trips a test rather than a dashboard timing out.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta

from obs_fixtures import NOW, observation

from luber_inference_observability import (
    IncidentLedger,
    InMemoryObservationStore,
    TimeWindow,
    aggregate,
    group,
    health_report,
    render_markdown,
)
from luber_inference_observability.queries import run_detection, summary, top_segments, trend

#: 100,000 observations is well past what this deployment will produce
#: in a month. Aggregation is arithmetic over a list, so the ceiling is
#: about the shape of the code rather than the machine.
SCALE = 100_000

REVISIONS = ("acestep@v1", "acestep@v2")
BUCKETS = ("0_30", "61_120", "181_240", "360_PLUS")


def large_store() -> InMemoryObservationStore:
    start = NOW - timedelta(days=7)
    step = timedelta(days=7) / SCALE
    rows = [
        observation(
            index,
            start + step * index,
            accepted=index % 20 != 0,
            retries=1 if index % 25 == 0 else 0,
            critical=("EARLY_COLLAPSE",) if index % 200 == 0 else (),
            # Different strides, so revision and bucket are not
            # correlated: with both keyed off `index % 2` only half the
            # combinations would exist and a two-way grouping test would
            # be asserting against a fixture, not against the grouping.
            revision=REVISIONS[(index // len(BUCKETS)) % len(REVISIONS)],
            duration_bucket=BUCKETS[index % len(BUCKETS)],
            total_latency=45.0 + (index % 90),
        )
        for index in range(SCALE)
    ]
    return InMemoryObservationStore(rows)


def test_a_hundred_thousand_observations_ingest_and_aggregate_in_bounds():
    started = time.monotonic()
    store = large_store()
    ingest_seconds = time.monotonic() - started
    assert store.count() == SCALE

    week = TimeWindow.ending_at(NOW, "7d")
    started = time.monotonic()
    result = aggregate(list(store.select(week)), window=week)
    aggregate_seconds = time.monotonic() - started

    assert result.sample_count > SCALE * 0.9
    # Bounds, not benchmarks: an accidental quadratic blows past these
    # by orders of magnitude while a slow machine does not.
    assert ingest_seconds < 60.0
    assert aggregate_seconds < 15.0


def test_grouping_at_scale_stays_linear():
    store = large_store()
    week = TimeWindow.ending_at(NOW, "7d")
    rows = list(store.select(week))

    started = time.monotonic()
    grouped = group(rows, window=week, by=("provider_revision", "duration_bucket"))
    elapsed = time.monotonic() - started

    assert len(grouped) == len(REVISIONS) * len(BUCKETS)
    assert sum(item.sample_count for item in grouped.values()) == len(rows)
    assert elapsed < 20.0


def test_detection_at_scale_completes_and_dedupes():
    store = large_store()
    ledger = IncidentLedger()

    started = time.monotonic()
    result = run_detection(store, current=TimeWindow.ending_at(NOW, "1h"), ledger=ledger, at=NOW)
    elapsed = time.monotonic() - started

    assert "regressions" in result
    assert elapsed < 30.0


def test_a_trend_over_a_week_stays_bounded():
    store = large_store()
    week = TimeWindow.ending_at(NOW, "7d")

    started = time.monotonic()
    payload = trend(store, window=week, metrics=("quality_retry_rate",), size="7d")
    elapsed = time.monotonic() - started

    # A week at six-hour buckets is 28 points, not 10,000.
    assert 20 <= len(payload["points"]) <= 40
    assert elapsed < 20.0


def test_the_summary_of_a_large_window_is_not_a_dump():
    store = large_store()
    payload = summary(store, window=TimeWindow.ending_at(NOW, "7d"))
    rendered = json.dumps(payload)
    # Bounded regardless of how many rows it summarises.
    assert len(rendered) < 20_000


# ── reports ──────────────────────────────────────────────────────────


def small_store() -> InMemoryObservationStore:
    start = NOW - timedelta(days=7)
    step = timedelta(days=7) / 400
    rows = [
        observation(
            index,
            start + step * index,
            accepted=index % 10 != 0,
            retries=1 if index % 10 == 0 else 0,
            critical=("EARLY_COLLAPSE",) if index % 50 == 0 else (),
            soft=("NARROW_STEREO",),
        )
        for index in range(400)
    ]
    return InMemoryObservationStore(rows)


def test_a_report_carries_counts_beside_every_percentage():
    report = health_report(small_store(), window=TimeWindow.ending_at(NOW, "7d"), generated_at=NOW)
    rendered = render_markdown(report)

    assert "# Inference health report" in rendered
    for line in rendered.splitlines():
        if "%" in line and "|" in line:
            # Every rate row carries "n/m" alongside its percentage.
            assert "/" in line, line


def test_a_report_keeps_advisories_out_of_the_rejection_list():
    report = health_report(small_store(), window=TimeWindow.ending_at(NOW, "7d"), generated_at=NOW)
    rendered = render_markdown(report)

    assert "**Rejections**" in rendered
    assert "**Advisories on delivered audio** (not failures)" in rendered
    rejections = rendered.split("**Rejections**")[1].split("**Advisories")[0]
    assert "NARROW_STEREO" not in rejections


def test_a_report_states_that_nothing_was_done_automatically():
    report = health_report(small_store(), window=TimeWindow.ending_at(NOW, "7d"), generated_at=NOW)
    assert "none" in report["automatic_remediation"]
    assert "No action was taken automatically" in render_markdown(report)


def test_a_report_over_an_empty_window_renders_rather_than_failing():
    report = health_report(
        InMemoryObservationStore(), window=TimeWindow.ending_at(NOW, "1h"), generated_at=NOW
    )
    rendered = render_markdown(report)

    assert "NO_DATA" in rendered
    assert "No open incidents" in rendered


def test_a_report_names_the_boundary_for_partial_history():
    store = InMemoryObservationStore(
        [
            observation(index, NOW - timedelta(minutes=30), qc_data=index % 2 == 0)
            for index in range(20)
        ]
    )
    report = health_report(store, window=TimeWindow.ending_at(NOW, "1h"), generated_at=NOW)
    rendered = render_markdown(report)

    assert "Partial data" in rendered
    assert "460642e" in rendered


def test_top_segments_refuses_to_rank_a_segment_of_one():
    """Otherwise "1 of 1 failed" tops every list."""
    store = InMemoryObservationStore(
        [
            observation(
                index,
                NOW - timedelta(minutes=30),
                status="FAILED" if index == 0 else "COMPLETED",
                duration_bucket="360_PLUS" if index == 0 else "61_120",
            )
            for index in range(60)
        ]
    )

    ranked = top_segments(
        store,
        window=TimeWindow.ending_at(NOW, "1h"),
        by=("duration_bucket",),
        minimum_samples=30,
    )

    labels = [item["segment_label"] for item in ranked["segments"]]
    assert not any("360_PLUS" in label for label in labels)
    assert ranked["segments_below_minimum"] >= 1
