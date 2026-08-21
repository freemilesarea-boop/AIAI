"""Response models for the inference console.

The same argument the training console's schemas make about credentials
applies here to user content: **no model below has a field a prompt
could occupy.** Not "we strip it" — there is nowhere to put it. A
response cannot leak lyrics if no response model can hold a string that
came from a user.

The other rule these models enforce is that a rate never travels alone.
`RateView` carries its numerator, its denominator and its status, so a
card cannot render "4%" without also being able to render "12 of 300",
and a window with nothing in it renders NO_DATA rather than 0%.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VersionBlock(BaseModel):
    observability_schema_version: str
    aggregation_version: str
    regression_engine_version: str
    incident_policy_version: str


class WindowView(BaseModel):
    start: str
    end: str
    duration_seconds: float


class RateView(BaseModel):
    """A proportion that cannot be displayed without its counts."""

    name: str
    numerator: int
    denominator: int
    excluded: int = 0
    value: float | None = None
    percent: float | None = None
    #: OK, NO_DATA. Never a bare zero standing in for an empty window.
    status: str
    render: str


class AverageView(BaseModel):
    name: str
    value: float | None = None
    count: int
    status: str


class DistributionView(BaseModel):
    """Latency, reported the only way latency may be."""

    name: str
    count: int
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    max: float | None = None
    mean: float | None = None
    status: str


class CoverageView(BaseModel):
    """How much of a window can answer candidate questions at all."""

    observations: int
    with_qc_data: int
    without_qc_data: int
    complete: bool
    partial: bool
    boundary_commit: str
    note: str | None = None


class CountersView(BaseModel):
    generation_requests: int
    completed_generations: int
    failed_generations: int
    cancelled_generations: int
    without_qc_data: int
    provider_calls: int
    candidates_generated: int
    quality_retries: int
    retry_exhaustions: int
    candidate_rejections: int
    first_candidate_accepted: int
    qc_observed: int
    finding_counts: dict[str, int] = Field(default_factory=dict)
    soft_finding_counts: dict[str, int] = Field(default_factory=dict)
    failure_code_counts: dict[str, int] = Field(default_factory=dict)
    data_quality_counts: dict[str, int] = Field(default_factory=dict)


class FindingsView(BaseModel):
    """Rejections and advisories, kept in separate maps.

    One map would let a harshness advisory be read as invalid audio, and
    a dashboard that conflated them would send somebody to investigate a
    correction the finishing engine was about to make anyway.
    """

    critical: dict[str, int] = Field(default_factory=dict)
    soft: dict[str, int] = Field(default_factory=dict)


class SummaryResponse(VersionBlock):
    window: WindowView
    segment: dict[str, str]
    sample_count: int
    counters: CountersView
    overview: dict[str, RateView]
    rates: dict[str, RateView]
    averages: dict[str, AverageView]
    latency: dict[str, DistributionView]
    findings: FindingsView
    data_quality: dict[str, int]
    coverage: CoverageView


class TrendPointView(BaseModel):
    start: str
    end: str
    sample_count: int
    #: `null` where a bucket had no samples. A chart must not draw zero
    #: through a quiet night and imply a recovery.
    values: dict[str, float | None]


class TrendResponse(VersionBlock):
    window: WindowView
    segment: dict[str, str]
    metrics: list[str]
    bucket_seconds: float
    points: list[TrendPointView]
    has_data: bool


class RegressionView(VersionBlock):
    finding_type: str
    category: str
    metric: str
    segment: dict[str, str]
    segment_label: str
    status: str
    severity: str
    baseline_value: float | None = None
    current_value: float | None = None
    absolute_delta: float | None = None
    relative_delta: float | None = None
    baseline_numerator: int | None = None
    baseline_denominator: int | None = None
    current_numerator: int | None = None
    current_denominator: int | None = None
    baseline_sample_count: int
    current_sample_count: int
    baseline_window: dict[str, Any] = Field(default_factory=dict)
    current_window: dict[str, Any] = Field(default_factory=dict)
    quantile_fraction: float | None = None
    thresholds: dict[str, Any] = Field(default_factory=dict)
    threshold_crossed: str | None = None
    reason: str
    explanation: str
    recommendations: list[str] = Field(default_factory=list)
    partial_history: bool = False


class IncidentEvidenceView(BaseModel):
    observed_at: str
    status: str
    severity: str
    baseline_value: float | None = None
    current_value: float | None = None
    absolute_delta: float | None = None
    relative_delta: float | None = None
    current_sample_count: int
    baseline_sample_count: int
    explanation: str


class IncidentView(VersionBlock):
    incident_id: str
    created_at: str
    status: str
    severity: str
    peak_severity: str
    finding_type: str
    category: str
    metric: str
    provider: str | None = None
    provider_version: str | None = None
    affected_dimensions: dict[str, str] = Field(default_factory=dict)
    segment_label: str
    baseline_window: dict[str, Any] = Field(default_factory=dict)
    current_window: dict[str, Any] = Field(default_factory=dict)
    first_seen: str | None = None
    last_seen: str | None = None
    occurrence_count: int
    consecutive_clean: int
    evidence: list[IncidentEvidenceView] = Field(default_factory=list)
    evidence_total: int
    recommendations: list[str] = Field(default_factory=list)
    summary: str
    acknowledged_at: str | None = None
    acknowledged_by: str | None = None
    resolved_at: str | None = None
    dismissed_at: str | None = None
    dismissed_by: str | None = None
    dismissal_reason: str | None = None


class IncidentListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[IncidentView]


class SegmentRankView(BaseModel):
    segment: dict[str, str]
    segment_label: str
    metric: str
    value: float | None = None
    numerator: int
    denominator: int
    render: str
    sample_count: int


class SegmentsResponse(VersionBlock):
    window: WindowView
    grouped_by: list[str]
    metric: str
    minimum_samples: int
    segments: list[SegmentRankView]
    segments_considered: int
    #: How many were dropped for having too few samples. Reported so a
    #: short list is legible as "most segments are small" rather than as
    #: "only these segments exist".
    segments_below_minimum: int


class ProviderView(BaseModel):
    """One provider revision's volume and health in a window."""

    provider: str
    provider_revision: str
    model_name: str
    model_version: str
    sample_count: int
    first_seen: str | None = None
    last_seen: str | None = None
    #: BASELINE_BUILDING while a revision is too new to judge.
    baseline_status: str
    rates: dict[str, RateView] = Field(default_factory=dict)


class ProvidersResponse(VersionBlock):
    window: WindowView
    providers: list[ProviderView]


class MarkerView(BaseModel):
    kind: str
    occurred_at: str
    label: str
    detail: dict[str, Any] = Field(default_factory=dict)
    #: Travels with every marker rather than being documented once. A
    #: marker reaches a tooltip and an export, and the caveat has to
    #: reach them too.
    caveat: str


class GenerationAttemptView(BaseModel):
    """One candidate, as an operator may see it.

    No audio path, no seed material beyond the seed itself, and above
    all no prompt: the model has no field for one.
    """

    attempt_index: int
    candidate_id: str
    status: str
    selection_status: str
    attribution: str
    seed: int | None = None
    retry_reason: str | None = None
    not_selected_reason: str | None = None
    duration_seconds: float | None = None
    critical_findings: list[str] = Field(default_factory=list)
    soft_findings: list[str] = Field(default_factory=list)
    provider_seconds: float | None = None
    qc_seconds: float | None = None


class GenerationTraceResponse(BaseModel):
    """A safe view of one generation's candidate phase.

    Deliberately not the raw Phase 29 trace. That record is safe by
    construction, but it is written for storage, and re-serving a stored
    blob is how a field added for another purpose ends up on a screen.
    This model names what the console shows.
    """

    generation_id: str
    occurred_at: str
    provider: str
    provider_revision: str
    task_type: str
    duration_bucket: str
    requested_duration_seconds: float | None = None
    language: str
    instrumental: str
    generation_status: str
    generation_failure_code: str | None = None
    qc_policy: str
    qc_data_available: bool
    qc_outcome: str | None = None
    finishing_outcome: str | None = None
    candidate_count: int | None = None
    provider_call_count: int | None = None
    quality_retry_count: int | None = None
    selected_on_attempt: int | None = None
    first_candidate_accepted: bool | None = None
    retry_exhausted: bool | None = None
    provider_latency_seconds: float | None = None
    qc_latency_seconds: float | None = None
    delivery_latency_seconds: float | None = None
    total_latency_seconds: float | None = None
    critical_findings: list[str] = Field(default_factory=list)
    soft_findings: list[str] = Field(default_factory=list)
    data_quality_issues: list[str] = Field(default_factory=list)
    attempts: list[GenerationAttemptView] = Field(default_factory=list)
    #: The Phase 29 explanation, rendered: "attempt 1 rejected for X,
    #: attempt 2 selected". Built server-side so the console and the CLI
    #: tell the same story.
    explanation: list[str] = Field(default_factory=list)


class GenerationListItem(BaseModel):
    generation_id: str
    occurred_at: str
    provider_revision: str
    task_type: str
    duration_bucket: str
    generation_status: str
    quality_retry_count: int | None = None
    first_candidate_accepted: bool | None = None
    critical_findings: list[str] = Field(default_factory=list)
    total_latency_seconds: float | None = None


class GenerationListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[GenerationListItem]


class OverviewResponse(VersionBlock):
    """What the landing screen needs, in one request."""

    window: WindowView
    summary: SummaryResponse
    open_incidents: int
    critical_incidents: int
    markers: list[MarkerView] = Field(default_factory=list)
    #: Stated on the response rather than only in the docs: an operator
    #: reading a CRITICAL card should know nothing was done about it.
    automatic_remediation: str = (
        "none — this console detects and explains; every action is an operator's"
    )


class IngestStatusResponse(BaseModel):
    """Whether the numbers on screen are current.

    A dashboard fed by a projection can be silently stale — the charts
    render, the rates look plausible, and the last ingest was three days
    ago. This is how an operator can tell.
    """

    observations: int
    latest_observation_at: str | None = None
    seconds_behind: float | None = None
    stale: bool
    note: str | None = None


class ActionResponse(BaseModel):
    ok: bool
    incident: IncidentView


__all__ = [
    "ActionResponse",
    "AverageView",
    "CountersView",
    "CoverageView",
    "DistributionView",
    "FindingsView",
    "GenerationAttemptView",
    "GenerationListItem",
    "GenerationListResponse",
    "GenerationTraceResponse",
    "IncidentEvidenceView",
    "IncidentListResponse",
    "IncidentView",
    "IngestStatusResponse",
    "MarkerView",
    "OverviewResponse",
    "ProviderView",
    "ProvidersResponse",
    "RateView",
    "RegressionView",
    "SegmentRankView",
    "SegmentsResponse",
    "SummaryResponse",
    "TrendPointView",
    "TrendResponse",
    "VersionBlock",
    "WindowView",
]
