"""Turning stored observations into what the console shows.

The browser never touches the analytics engine and never touches the
database. Every view is built here from the same functions the CLI
calls, which is what makes the number on a card and the number in a
report the same number — two implementations of "retry rate" eventually
disagree, and the one an operator believes is whichever they saw first.

Three things this layer enforces that a direct read could not.

**Nothing is invented.** A window with no rows renders NO_DATA, a metric
with too few samples renders INSUFFICIENT_DATA, and a revision without
history renders BASELINE_BUILDING. None of them renders zero.

**Nothing user-written can get through.** The response models have no
field a prompt could occupy, and this layer reads from the projection,
which has no column one could be stored in. Two independent reasons, so
neither has to be remembered.

**Reading is bounded.** Lists are paginated server-side and trend
buckets are fixed per window, so a console pointed at a busy month
renders a page rather than a month.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from luber_api.ops.inference_schemas import (
    GenerationAttemptView,
    GenerationListItem,
    GenerationListResponse,
    GenerationTraceResponse,
    IncidentEvidenceView,
    IncidentListResponse,
    IncidentView,
    IngestStatusResponse,
    MarkerView,
    OverviewResponse,
    ProvidersResponse,
    ProviderView,
    RateView,
    SegmentsResponse,
    SummaryResponse,
    TrendResponse,
    WindowView,
)
from luber_inference_observability.aggregation import Metric, aggregate
from luber_inference_observability.baselines import (
    DEFAULT_MINIMUM_BASELINE_SAMPLES,
)
from luber_inference_observability.dimensions import Segment
from luber_inference_observability.markers import derive as derive_markers
from luber_inference_observability.markers import within
from luber_inference_observability.queries import (
    compare_revisions,
    compare_windows,
    top_segments,
)
from luber_inference_observability.queries import (
    summary as summarise,
)
from luber_inference_observability.queries import (
    trend as trend_over,
)
from luber_inference_observability.service import load_store
from luber_inference_observability.storage import InMemoryObservationStore, from_mapping
from luber_inference_observability.windows import DURATIONS, TimeWindow

#: Beyond this the projection is behind enough that an operator should
#: know before trusting a chart. Two hours: long enough that a paused
#: scheduled ingest is not reported as a crisis, short enough that a
#: genuinely stalled pipeline surfaces the same morning.
STALE_AFTER = timedelta(hours=2)

#: Metrics the retry chart draws.
RETRY_TREND_METRICS = (
    Metric.FIRST_CANDIDATE_ACCEPT_RATE.value,
    Metric.QUALITY_RETRY_RATE.value,
    Metric.RETRY_EXHAUSTION_RATE.value,
)

#: Metrics the failure chart draws.
FAILURE_TREND_METRICS = (
    Metric.INVALID_AUDIO_RATE.value,
    Metric.EARLY_COLLAPSE_RATE.value,
    Metric.DURATION_FAILURE_RATE.value,
    Metric.PROVIDER_FAILURE_RATE.value,
)

#: Metrics the latency chart draws, as P95.
LATENCY_TREND_METRICS = (
    Metric.TOTAL_LATENCY.value,
    Metric.PROVIDER_LATENCY.value,
    Metric.QC_LATENCY.value,
)

TREND_SETS: dict[str, tuple[str, ...]] = {
    "retry": RETRY_TREND_METRICS,
    "failure": FAILURE_TREND_METRICS,
    "latency": LATENCY_TREND_METRICS,
}


def window_for(size: str, end: datetime | None = None) -> TimeWindow:
    if size not in DURATIONS:
        raise ValueError(f"unknown window {size!r}. Known: {', '.join(sorted(DURATIONS))}")
    return TimeWindow.ending_at(end or datetime.now(UTC), size)


def segment_for(
    *,
    provider: str | None = None,
    provider_revision: str | None = None,
    task_type: str | None = None,
    duration_bucket: str | None = None,
) -> Segment | None:
    segment = Segment.of(
        provider=provider,
        provider_revision=provider_revision,
        task_type=task_type,
        duration_bucket=duration_bucket,
    )
    return segment if segment.filters else None


class InferenceReadModel:
    """Every view the inference console renders."""

    def __init__(self, repository: Any) -> None:
        self._repository = repository

    # ── overview ─────────────────────────────────────────────────────

    async def overview(
        self, *, window: TimeWindow, segment: Segment | None = None
    ) -> OverviewResponse:
        store = await load_store(self._repository, window=window, segment=segment)
        payload = summarise(store, window=window, segment=segment)
        open_incidents = await self._repository.count_incidents(statuses=["OPEN", "ACKNOWLEDGED"])
        rows = list(store)
        critical = sum(
            1
            for incident in await self._repository.list_incidents(
                statuses=["OPEN", "ACKNOWLEDGED"], limit=200
            )
            if incident.get("severity") == "CRITICAL"
        )
        markers = [
            MarkerView(**marker.to_dict()) for marker in within(derive_markers(rows), window)
        ]
        return OverviewResponse(
            **_versions(payload),
            window=payload["window"],
            summary=SummaryResponse(**payload),
            open_incidents=open_incidents,
            critical_incidents=critical,
            markers=markers,
        )

    async def summary(
        self, *, window: TimeWindow, segment: Segment | None = None
    ) -> SummaryResponse:
        store = await load_store(self._repository, window=window, segment=segment)
        return SummaryResponse(**summarise(store, window=window, segment=segment))

    async def trend(
        self,
        *,
        window: TimeWindow,
        chart: str,
        size: str | None = None,
        segment: Segment | None = None,
    ) -> TrendResponse:
        metrics = TREND_SETS.get(chart)
        if metrics is None:
            raise ValueError(f"unknown chart {chart!r}. Known: {', '.join(sorted(TREND_SETS))}")
        store = await load_store(self._repository, window=window, segment=segment)
        payload = trend_over(store, window=window, metrics=metrics, segment=segment, size=size)
        return TrendResponse(**payload)

    # ── providers ────────────────────────────────────────────────────

    async def providers(self, *, window: TimeWindow) -> ProvidersResponse:
        """Every revision seen in this window, with its own numbers.

        A revision with too few observations is labelled
        BASELINE_BUILDING rather than being given rates that look
        comparable to a revision with a week behind it.
        """
        store = await load_store(self._repository, window=window)
        rows = list(store)
        by_revision: dict[str, list[Any]] = {}
        for row in rows:
            by_revision.setdefault(row.provider_revision, []).append(row)

        views: list[ProviderView] = []
        for revision, group in sorted(by_revision.items()):
            segment = Segment.of(provider_revision=revision)
            result = aggregate(group, window=window, segment=segment)
            views.append(
                ProviderView(
                    provider=group[0].provider,
                    provider_revision=revision,
                    model_name=group[0].model_name,
                    model_version=group[0].model_version,
                    sample_count=len(group),
                    first_seen=min(row.occurred_at for row in group).isoformat(),
                    last_seen=max(row.occurred_at for row in group).isoformat(),
                    baseline_status=(
                        "READY"
                        if len(group) >= DEFAULT_MINIMUM_BASELINE_SAMPLES
                        else "BASELINE_BUILDING"
                    ),
                    rates={
                        name: RateView(**result.rate(name).to_dict())
                        for name in (
                            Metric.GENERATION_SUCCESS_RATE.value,
                            Metric.FIRST_CANDIDATE_ACCEPT_RATE.value,
                            Metric.QUALITY_RETRY_RATE.value,
                            Metric.EARLY_COLLAPSE_RATE.value,
                        )
                    },
                )
            )
        return ProvidersResponse(
            **_versions(summarise(store, window=window)),
            window=WindowView(**window.to_dict()),
            providers=views,
        )

    async def compare(
        self, *, window: TimeWindow, left: str, right: str, minimum_samples: int = 30
    ) -> dict[str, Any]:
        store = await load_store(self._repository, window=window)
        return compare_revisions(
            store,
            window=window,
            left_revision=left,
            right_revision=right,
            minimum_samples=minimum_samples,
        )

    async def deployment(self, *, at: datetime, hours: int = 24) -> dict[str, Any]:
        span = timedelta(hours=hours)
        before = TimeWindow.of(at - span, at)
        after = TimeWindow.of(at, at + span)
        rows = await self._repository.select_observations(start=before.start, end=after.end)
        store = InMemoryObservationStore(from_mapping(row) for row in rows)
        return compare_windows(store, before=before, after=after)

    # ── segments ─────────────────────────────────────────────────────

    async def segments(
        self,
        *,
        window: TimeWindow,
        by: tuple[str, ...],
        metric: str,
        minimum_samples: int = 30,
        limit: int = 10,
    ) -> SegmentsResponse:
        store = await load_store(self._repository, window=window)
        payload = top_segments(
            store,
            window=window,
            by=by,
            metric=metric,
            minimum_samples=minimum_samples,
            limit=limit,
        )
        return SegmentsResponse(**payload)

    # ── incidents ────────────────────────────────────────────────────

    async def incidents(
        self, *, statuses: list[str] | None, limit: int, offset: int
    ) -> IncidentListResponse:
        rows = await self._repository.list_incidents(statuses=statuses, limit=limit, offset=offset)
        total = await self._repository.count_incidents(statuses=statuses)
        return IncidentListResponse(
            total=total,
            limit=limit,
            offset=offset,
            items=[_incident_view(row) for row in rows],
        )

    async def incident(self, incident_id: str) -> IncidentView | None:
        row = await self._repository.get_incident(incident_id)
        return None if row is None else _incident_view(row)

    # ── drilldown ────────────────────────────────────────────────────

    async def generations(
        self,
        *,
        window: TimeWindow,
        segment: Segment | None,
        limit: int,
        offset: int,
        only_failures: bool = False,
    ) -> GenerationListResponse:
        filters = segment.to_dict() if segment else None
        rows = await self._repository.select_observations(
            start=window.start,
            end=window.end,
            filters=filters,
            limit=limit,
            offset=offset,
        )
        total = await self._repository.count_observations(
            start=window.start, end=window.end, filters=filters
        )
        items = [_list_item(row) for row in rows]
        if only_failures:
            items = [item for item in items if item.generation_status != "COMPLETED"]
        return GenerationListResponse(total=total, limit=limit, offset=offset, items=items)

    async def generation(
        self, generation_id: uuid.UUID, *, qc_trace: dict[str, Any] | None = None
    ) -> GenerationTraceResponse | None:
        """One generation's safe trace.

        The attempt list comes from the Phase 29 trace when the caller
        supplies it, and the summary counters come from the projection.
        Neither carries a prompt: the trace has never held one, and the
        projection has no column for one.
        """
        row = await self._repository.get_observation(generation_id)
        if row is None:
            return None
        return _trace_view(row, qc_trace)


# ── translation ──────────────────────────────────────────────────────


def _versions(payload: dict[str, Any]) -> dict[str, str]:
    return {
        key: payload[key]
        for key in (
            "observability_schema_version",
            "aggregation_version",
            "regression_engine_version",
            "incident_policy_version",
        )
    }


def _load(raw: Any, fallback: Any) -> Any:
    if raw is None:
        return fallback
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return fallback
    return raw


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        aware = value if value.tzinfo else value.replace(tzinfo=UTC)
        return aware.isoformat()
    return str(value)


def _incident_view(row: dict[str, Any]) -> IncidentView:
    segment = _load(row.get("segment"), {})
    label = ", ".join(f"{key}={value}" for key, value in sorted(segment.items())) or "all traffic"
    evidence = _load(row.get("evidence"), [])
    from luber_inference_observability.versions import version_block

    return IncidentView(
        **version_block(),
        incident_id=row["incident_id"],
        created_at=_iso(row["created_at"]) or "",
        status=row["status"],
        severity=row["severity"],
        peak_severity=row["peak_severity"],
        finding_type=row["finding_type"],
        category=row["category"],
        metric=row["metric"],
        provider=row.get("provider"),
        provider_version=row.get("provider_revision"),
        affected_dimensions=segment,
        segment_label=label,
        baseline_window=_load(row.get("baseline_window"), {}),
        current_window=_load(row.get("current_window"), {}),
        first_seen=_iso(row.get("first_seen")),
        last_seen=_iso(row.get("last_seen")),
        occurrence_count=row.get("occurrence_count", 0),
        consecutive_clean=row.get("consecutive_clean", 0),
        # Bounded server-side: an incident open for a week carries
        # hundreds of evidence rows and a browser needs the last few.
        evidence=[IncidentEvidenceView(**item) for item in evidence[-20:]],
        evidence_total=len(evidence),
        recommendations=_load(row.get("recommendations"), []),
        summary=_summary_line(row, evidence),
        acknowledged_at=_iso(row.get("acknowledged_at")),
        acknowledged_by=row.get("acknowledged_by"),
        resolved_at=_iso(row.get("resolved_at")),
        dismissed_at=_iso(row.get("dismissed_at")),
        dismissed_by=row.get("dismissed_by"),
        dismissal_reason=row.get("dismissal_reason"),
    )


def _summary_line(row: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    latest = evidence[-1]["explanation"] if evidence else ""
    segment = _load(row.get("segment"), {})
    label = ", ".join(f"{k}={v}" for k, v in sorted(segment.items())) or "all traffic"
    return f"[{row['severity']}] {row['finding_type']} — {label}. {latest}".strip()


def _list_item(row: dict[str, Any]) -> GenerationListItem:
    return GenerationListItem(
        generation_id=str(row["generation_id"]),
        occurred_at=_iso(row["occurred_at"]) or "",
        provider_revision=row.get("provider_revision", "UNKNOWN"),
        task_type=row.get("task_type", "UNKNOWN"),
        duration_bucket=row.get("duration_bucket", "UNKNOWN"),
        generation_status=row.get("generation_status", "UNKNOWN"),
        quality_retry_count=row.get("quality_retry_count"),
        first_candidate_accepted=row.get("first_candidate_accepted"),
        critical_findings=_load(row.get("critical_findings"), []),
        total_latency_seconds=row.get("total_latency_seconds"),
    )


def _trace_view(row: dict[str, Any], qc_trace: dict[str, Any] | None) -> GenerationTraceResponse:
    attempts: list[GenerationAttemptView] = []
    explanation: list[str] = []

    if qc_trace:
        selected = qc_trace.get("selected_candidate_id")
        for attempt in qc_trace.get("attempts", []) or []:
            findings = attempt.get("findings") or []
            critical = [item["code"] for item in findings if item.get("severity") == "CRITICAL"]
            soft = [item["code"] for item in findings if item.get("severity") != "CRITICAL"]
            attempts.append(
                GenerationAttemptView(
                    attempt_index=attempt.get("attempt_index", 0),
                    candidate_id=attempt.get("candidate_id", ""),
                    status=attempt.get("status", "UNKNOWN"),
                    selection_status=attempt.get("selection_status", "UNDECIDED"),
                    attribution=attempt.get("attribution", "UNKNOWN"),
                    seed=attempt.get("seed"),
                    retry_reason=attempt.get("retry_reason"),
                    not_selected_reason=attempt.get("not_selected_reason"),
                    duration_seconds=attempt.get("duration_seconds"),
                    critical_findings=sorted(set(critical)),
                    soft_findings=sorted(set(soft)),
                    provider_seconds=attempt.get("provider_seconds"),
                    qc_seconds=attempt.get("qc_seconds"),
                )
            )
            index = attempt.get("attempt_index", 0)
            if attempt.get("candidate_id") == selected:
                explanation.append(f"Attempt {index + 1} selected.")
            elif critical:
                explanation.append(
                    f"Attempt {index + 1} rejected: {', '.join(sorted(set(critical)))}"
                )
        budget = qc_trace.get("budget") or {}
        explanation.append(
            f"Provider calls: {budget.get('provider_calls_used', '?')}. "
            f"Quality retries: {budget.get('retry_rounds', '?')}."
        )

    return GenerationTraceResponse(
        generation_id=str(row["generation_id"]),
        occurred_at=_iso(row["occurred_at"]) or "",
        provider=row.get("provider", "UNKNOWN"),
        provider_revision=row.get("provider_revision", "UNKNOWN"),
        task_type=row.get("task_type", "UNKNOWN"),
        duration_bucket=row.get("duration_bucket", "UNKNOWN"),
        requested_duration_seconds=row.get("requested_duration_seconds"),
        language=row.get("language", "UNKNOWN"),
        instrumental=row.get("instrumental", "UNKNOWN"),
        generation_status=row.get("generation_status", "UNKNOWN"),
        generation_failure_code=row.get("generation_failure_code"),
        qc_policy=row.get("qc_policy", "UNKNOWN"),
        qc_data_available=bool(row.get("qc_data_available")),
        qc_outcome=row.get("qc_outcome"),
        finishing_outcome=row.get("finishing_outcome"),
        candidate_count=row.get("candidate_count"),
        provider_call_count=row.get("provider_call_count"),
        quality_retry_count=row.get("quality_retry_count"),
        selected_on_attempt=row.get("selected_on_attempt"),
        first_candidate_accepted=row.get("first_candidate_accepted"),
        retry_exhausted=row.get("retry_exhausted"),
        provider_latency_seconds=row.get("provider_latency_seconds"),
        qc_latency_seconds=row.get("qc_latency_seconds"),
        delivery_latency_seconds=row.get("delivery_latency_seconds"),
        total_latency_seconds=row.get("total_latency_seconds"),
        critical_findings=_load(row.get("critical_findings"), []),
        soft_findings=_load(row.get("soft_findings"), []),
        data_quality_issues=_load(row.get("data_quality_issues"), []),
        attempts=attempts,
        explanation=explanation,
    )


async def ingest_status(repository: Any, *, now: datetime | None = None) -> IngestStatusResponse:
    """Whether what the console is showing is current.

    A projection-backed dashboard can be silently stale: every chart
    renders, every rate looks plausible, and the last ingest ran on
    Tuesday. Reporting the lag is what makes that visible instead of
    misleading.
    """
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    total = await repository.count_observations()
    latest = await repository.latest_observed_at()
    if latest is None:
        return IngestStatusResponse(
            observations=total,
            latest_observation_at=None,
            seconds_behind=None,
            stale=total == 0,
            note=(
                "Nothing has been ingested yet. Run `inference-observability backfill` "
                "to project existing generations."
            ),
        )
    aware = latest if latest.tzinfo else latest.replace(tzinfo=UTC)
    behind = (moment - aware).total_seconds()
    stale = behind > STALE_AFTER.total_seconds()
    return IngestStatusResponse(
        observations=total,
        latest_observation_at=aware.isoformat(),
        seconds_behind=round(behind, 1),
        stale=stale,
        note=(
            f"The newest observation is {behind / 3600:.1f} hours old. Charts may be "
            "behind reality; check that ingestion is running."
        )
        if stale
        else None,
    )


__all__ = [
    "FAILURE_TREND_METRICS",
    "LATENCY_TREND_METRICS",
    "RETRY_TREND_METRICS",
    "STALE_AFTER",
    "TREND_SETS",
    "InferenceReadModel",
    "ingest_status",
    "segment_for",
    "window_for",
]
