"""The scenarios this system exists to get right, and the ones to get wrong.

Half of these assert that nothing fires. That ratio is deliberate: a
detector that catches every real regression and also cries wolf twice a
day is a detector nobody reads, and the day it is right is the day
somebody closes it with the rest.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from obs_fixtures import NOW, healthy_week, observation, recent_window

from luber_inference_observability import (
    IncidentLedger,
    InMemoryObservationStore,
    TimeWindow,
)
from luber_inference_observability.queries import (
    evaluate_segments,
    run_detection,
    summary,
)
from luber_inference_observability.regressions import Category, Status, regressions


def store_of(*groups) -> InMemoryObservationStore:
    rows = [row for group in groups for row in group]
    return InMemoryObservationStore(rows)


def hour() -> TimeWindow:
    return TimeWindow.ending_at(NOW, "1h")


def crossed(store, *, by=("provider_revision",)):
    return regressions(evaluate_segments(store, current=hour(), by=by))


def statuses(store, *, by=("provider_revision",)) -> set[str]:
    return {finding.status for finding in evaluate_segments(store, current=hour(), by=by)}


# ── the regression the brief describes ───────────────────────────────


def test_a_broad_regression_is_found_with_its_counts():
    """Baseline 95% accept / 3% retry / 0.5% collapse; current 70/25/12."""
    store = store_of(
        healthy_week(),
        recent_window(
            200,
            accepted=False,
            retries=1,
            critical=("EARLY_COLLAPSE",),
        ),
    )

    found = crossed(store)
    kinds = {finding.finding_type for finding in found}

    assert "FIRST_CANDIDATE_ACCEPTANCE_DROP" in kinds
    assert "QUALITY_RETRY_RATE_INCREASE" in kinds
    assert "EARLY_COLLAPSE_INCREASE" in kinds
    for finding in found:
        # Every finding carries both sides of the comparison.
        assert finding.baseline_sample_count > 0
        assert finding.current_sample_count > 0
        assert finding.threshold_crossed
        assert finding.explain()


def test_a_finding_explains_what_moved_and_never_why():
    store = store_of(
        healthy_week(), recent_window(200, accepted=False, critical=("EARLY_COLLAPSE",))
    )
    for finding in crossed(store):
        text = finding.explain().lower()
        assert "moved" in text or "rose" in text or "%" in text
        for causal in ("caused", "because", "due to"):
            assert causal not in text


# ── the ones that must stay quiet ────────────────────────────────────


def test_two_requests_and_one_failure_is_not_a_regression():
    """It is two requests. Reporting it would teach an operator to
    ignore the list."""
    store = store_of(
        healthy_week(),
        recent_window(2, status="FAILED", accepted=False, critical=("EARLY_COLLAPSE",)),
    )

    assert crossed(store) == []
    assert Status.INSUFFICIENT_DATA.value in statuses(store)
    # Explicitly not NORMAL: "we cannot tell" and "it is fine" are
    # different answers and only one lets somebody stop looking.
    assert Status.NORMAL.value not in statuses(store)


def test_a_doubling_of_a_tiny_rate_is_not_an_incident():
    """0.1% to 0.2% is a 100% relative increase and operationally
    nothing."""
    baseline = [
        observation(
            index,
            NOW - timedelta(days=7) + timedelta(seconds=index * 500),
            critical=("EARLY_COLLAPSE",) if index == 0 else (),
        )
        for index in range(1000)
    ]
    current = [
        observation(
            50_000 + index,
            NOW - timedelta(minutes=55) + timedelta(seconds=index * 3),
            critical=("EARLY_COLLAPSE",) if index < 2 else (),
        )
        for index in range(1000)
    ]

    store = store_of(baseline, current)

    collapse = [
        finding
        for finding in evaluate_segments(store, current=hour(), by=("provider_revision",))
        if finding.metric == "early_collapse_rate"
    ]
    assert collapse
    assert all(finding.status == Status.NORMAL.value for finding in collapse)
    assert all("absolute minimum" in finding.reason for finding in collapse)


def test_an_empty_window_is_no_data_rather_than_perfect():
    store = store_of(healthy_week())
    assert Status.NO_DATA.value in statuses(store)
    assert crossed(store) == []


def test_a_brand_new_deployment_with_no_history_says_so():
    store = store_of(recent_window(200, accepted=False, critical=("EARLY_COLLAPSE",)))
    assert Status.BASELINE_BUILDING.value in statuses(store)
    assert crossed(store) == []


# ── availability against quality ─────────────────────────────────────


def test_a_provider_timeout_spike_is_an_availability_incident():
    store = store_of(
        healthy_week(),
        recent_window(
            200,
            status="FAILED",
            accepted=False,
            critical=("PROVIDER_TIMEOUT",),
            failure_code="GENERATION_TIMEOUT",
        ),
    )

    found = crossed(store)
    availability = [f for f in found if f.category == Category.AVAILABILITY.value]

    assert availability, "a timeout spike is not a quality problem"
    assert any(f.finding_type == "PROVIDER_TIMEOUT_INCREASE" for f in availability)


def test_a_collapse_spike_is_a_quality_incident():
    store = store_of(
        healthy_week(),
        recent_window(200, accepted=False, critical=("EARLY_COLLAPSE",)),
    )

    quality = [f for f in crossed(store) if f.category == Category.QUALITY.value]
    assert any(f.finding_type == "EARLY_COLLAPSE_INCREASE" for f in quality)


def test_latency_doubling_is_reported_against_a_quantile():
    store = store_of(
        healthy_week(total_latency=60.0),
        recent_window(200, total_latency=300.0, provider_latency=290.0),
    )

    latency = [f for f in crossed(store) if f.finding_type == "LATENCY_REGRESSION"]

    assert latency
    for finding in latency:
        # Compared at P95, and the finding says which quantile.
        assert finding.quantile_fraction == 0.95
        assert "P95" in finding.explain()


# ── segmentation ─────────────────────────────────────────────────────


def test_a_regression_confined_to_one_duration_bucket_is_found():
    """The overall rate barely moves; the segment is on fire."""
    baseline = healthy_week(1400, collapse_rate=0.0)
    for index, row in enumerate(baseline):
        row.duration_bucket = "181_240" if index % 7 == 0 else "61_120"

    current = recent_window(700, accepted=True)
    for index, row in enumerate(current):
        long_form = index % 7 == 0
        row.duration_bucket = "181_240" if long_form else "61_120"
        if long_form and (index // 7) % 10 < 4:
            row.critical_findings = ("EARLY_COLLAPSE",)
            row.first_candidate_accepted = False

    store = store_of(baseline, current)
    found = regressions(evaluate_segments(store, current=hour(), by=("duration_bucket",)))

    segmented = [f for f in found if f.segment.to_dict().get("duration_bucket") == "181_240"]
    assert segmented, "the segment regression must be found"
    assert segmented[0].severity == "CRITICAL"
    # And the overall view, if it fires at all, is milder.
    overall = [f for f in found if not f.segment.filters]
    if overall:
        assert overall[0].absolute_delta < segmented[0].absolute_delta


def test_a_bad_revision_does_not_contaminate_a_good_one():
    store = store_of(
        healthy_week(1200, revision="acestep@v1"),
        recent_window(300, revision="acestep@v1", start_index=200_000),
        recent_window(
            300,
            revision="acestep@v2",
            start_index=300_000,
            accepted=False,
            critical=("EARLY_COLLAPSE",),
        ),
    )

    findings = evaluate_segments(store, current=hour(), by=("provider_revision",))
    v1 = [f for f in findings if f.segment.to_dict().get("provider_revision") == "acestep@v1"]
    v2 = [f for f in findings if f.segment.to_dict().get("provider_revision") == "acestep@v2"]

    assert v1 and all(f.status != Status.REGRESSED.value for f in v1)
    assert v2, "the new revision must be evaluated somehow"


def test_a_new_revision_is_judged_against_its_peers_on_the_day_it_ships():
    """It has no history, so the rolling baseline cannot help. Waiting a
    week to notice a bad rollout is not an option."""
    store = store_of(
        healthy_week(1200, revision="acestep@v1"),
        recent_window(300, revision="acestep@v1", start_index=200_000),
        recent_window(
            300,
            revision="acestep@v2",
            start_index=300_000,
            accepted=False,
            critical=("EARLY_COLLAPSE",),
        ),
    )

    found = crossed(store)
    v2 = [f for f in found if f.segment.to_dict().get("provider_revision") == "acestep@v2"]

    assert v2, "a bad new revision must be visible immediately"
    assert any(f.finding_type == "EARLY_COLLAPSE_INCREASE" for f in v2)


def test_a_degenerate_split_does_not_double_report():
    """One revision means the segment and the whole population are the
    same rows. Two incidents for one problem trains an operator to
    skim."""
    store = store_of(
        healthy_week(), recent_window(200, accepted=False, critical=("EARLY_COLLAPSE",))
    )
    found = crossed(store)
    collapse = [f for f in found if f.finding_type == "EARLY_COLLAPSE_INCREASE"]
    assert len(collapse) == 1


# ── cancellation and partial history ─────────────────────────────────


def test_a_cancelled_generation_is_not_a_quality_failure():
    store = store_of(
        healthy_week(),
        recent_window(200, status="CANCELLED", accepted=False, critical=()),
    )

    result = summary(store, window=hour())

    assert result["counters"]["cancelled_generations"] == 200
    # Excluded from every candidate-derived denominator.
    assert result["overview"]["quality_retry_rate"]["denominator"] == 0
    assert result["overview"]["quality_retry_rate"]["status"] == "NO_DATA"
    assert crossed(store) == []


def test_generations_without_a_trace_are_excluded_and_reported():
    """Their retries are unknown, not zero. Averaging them in as
    flawless would hide a regression behind old data."""
    store = store_of(
        healthy_week(),
        recent_window(100, qc_data=False, start_index=400_000),
        recent_window(100, accepted=False, critical=("EARLY_COLLAPSE",), start_index=500_000),
    )

    result = summary(store, window=hour())

    assert result["coverage"]["partial"] is True
    assert result["coverage"]["without_qc_data"] == 100
    assert "460642e" in result["coverage"]["note"]
    assert result["overview"]["early_collapse_rate"]["denominator"] == 100


# ── incidents ────────────────────────────────────────────────────────


def test_running_the_detector_fifty_times_produces_one_incident_each():
    """A detector run every five minutes must update, not multiply."""
    store = store_of(
        healthy_week(), recent_window(200, accepted=False, critical=("EARLY_COLLAPSE",))
    )
    ledger = IncidentLedger()

    for tick in range(50):
        run_detection(store, current=hour(), ledger=ledger, at=NOW + timedelta(minutes=tick))

    assert len(ledger) == len(ledger.active())
    assert len(ledger) < 10, "one incident per distinct problem, not per run"
    for incident in ledger.all():
        assert incident.occurrence_count == 50
        # Evidence is bounded, so a long incident does not grow forever.
        assert len(incident.evidence) <= ledger.policy.evidence_limit


def test_an_incident_needs_a_sustained_recovery_before_it_resolves():
    bad = store_of(healthy_week(), recent_window(200, accepted=False, critical=("EARLY_COLLAPSE",)))
    good = store_of(healthy_week(), recent_window(200))
    ledger = IncidentLedger()

    run_detection(bad, current=hour(), ledger=ledger, at=NOW)
    assert ledger.active()

    run_detection(good, current=hour(), ledger=ledger, at=NOW + timedelta(minutes=5))
    assert ledger.active(), "one clean run is not a recovery"

    for tick in (10, 15):
        run_detection(good, current=hour(), ledger=ledger, at=NOW + timedelta(minutes=tick))

    assert not ledger.active()
    resolved = ledger.all()[0]
    assert resolved.status == "RESOLVED"
    # The history survives.
    assert resolved.occurrence_count >= 1
    assert resolved.evidence


def test_a_metric_oscillating_around_the_threshold_does_not_flap():
    bad = store_of(healthy_week(), recent_window(200, accepted=False, critical=("EARLY_COLLAPSE",)))
    good = store_of(healthy_week(), recent_window(200))
    ledger = IncidentLedger()

    # Alternating: bad, good, bad, good… Without a recovery window this
    # would open and resolve on every other evaluation.
    for tick in range(10):
        source = bad if tick % 2 == 0 else good
        run_detection(source, current=hour(), ledger=ledger, at=NOW + timedelta(minutes=tick * 5))

    incidents = ledger.all()
    assert len(incidents) <= 4, "flapping must not mint a new incident per cycle"
    assert all(item.status != "RESOLVED" for item in incidents)


def test_a_quiet_window_does_not_resolve_an_incident():
    """Traffic going quiet is when a regression is easiest to miss.
    Counting NO_DATA as recovery would close the incident exactly then."""
    bad = store_of(healthy_week(), recent_window(200, accepted=False, critical=("EARLY_COLLAPSE",)))
    quiet = store_of(healthy_week())
    ledger = IncidentLedger()

    run_detection(bad, current=hour(), ledger=ledger, at=NOW)
    for tick in (5, 10, 15, 20):
        run_detection(quiet, current=hour(), ledger=ledger, at=NOW + timedelta(minutes=tick))

    assert ledger.active(), "silence is not recovery"


def test_acknowledging_does_not_stop_the_evidence_accumulating():
    store = store_of(
        healthy_week(), recent_window(200, accepted=False, critical=("EARLY_COLLAPSE",))
    )
    ledger = IncidentLedger()
    run_detection(store, current=hour(), ledger=ledger, at=NOW)
    incident = ledger.active()[0]
    before = len(incident.evidence)

    ledger.acknowledge(incident.incident_id, by="alex", at=NOW)
    run_detection(store, current=hour(), ledger=ledger, at=NOW + timedelta(minutes=5))

    assert incident.status == "ACKNOWLEDGED"
    assert incident.acknowledged_by == "alex"
    assert len(incident.evidence) > before


def test_a_dismissal_needs_a_reason_and_keeps_the_record():
    store = store_of(
        healthy_week(), recent_window(200, accepted=False, critical=("EARLY_COLLAPSE",))
    )
    ledger = IncidentLedger()
    run_detection(store, current=hour(), ledger=ledger, at=NOW)
    incident = ledger.active()[0]

    with pytest.raises(ValueError, match="needs a reason"):
        ledger.dismiss(incident.incident_id, by="alex", reason="   ", at=NOW)

    ledger.dismiss(incident.incident_id, by="alex", reason="known load test", at=NOW)

    assert incident.status == "DISMISSED"
    assert incident.dismissal_reason == "known load test"
    assert incident.evidence, "dismissal must not delete the history"


def test_two_metrics_sharing_a_finding_type_stay_separate_incidents():
    """Total latency and provider latency both raise LATENCY_REGRESSION
    and are different problems: one says the pipeline slowed, the other
    says the model did."""
    store = store_of(
        healthy_week(total_latency=60.0),
        recent_window(200, total_latency=300.0, provider_latency=290.0),
    )
    ledger = IncidentLedger()
    run_detection(store, current=hour(), ledger=ledger, at=NOW)

    latency = [item for item in ledger.all() if item.finding_type == "LATENCY_REGRESSION"]
    assert len({item.metric for item in latency}) == len(latency) > 1
