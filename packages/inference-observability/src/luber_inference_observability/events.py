"""One row per generation, holding only what analytics may see.

This is the privacy boundary, and it is a boundary rather than a rule
because of what the source looks like. `generations` carries `prompt`,
`lyrics` and `title` as ordinary columns, and `request_trace` — which
reads like a diagnostic blob — contains `original_prompt` and
`original_lyrics` in full. An analytics layer querying that table
directly would be one `SELECT *` from putting lyrics in a dashboard.

So nothing downstream of this module can reach a prompt. There is no
field here one could occupy, which is a stronger guarantee than
remembering to strip it — the same argument Phase 28 makes about
credentials having nowhere to go in its read model.

What is kept: counts, latencies, outcome codes, and dimensions with
small enumerable ranges. What is kept as an identifier: `generation_id`,
because an operator investigating a spike has to be able to open one,
and `request_sha256`, because two attempts at the same request should be
recognisable as such. No `user_id`: no metric in this phase is
per-user, and a column nobody needs is a column that can leak.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from luber_inference_observability.dimensions import UNKNOWN, duration_bucket, task_type
from luber_inference_observability.versions import OBSERVABILITY_SCHEMA_VERSION
from luber_inference_qc.findings import Finding, Severity

#: Names that must never appear in an observation, a query result, an API
#: response or a report. Asserted by the verifier and by the tests rather
#: than trusted, because the cost of being wrong is somebody's lyrics on
#: an operator's screen.
FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    {
        "prompt",
        "original_prompt",
        "compiled_prompt",
        "lyrics",
        "original_lyrics",
        "compiled_lyrics",
        "title",
        "user_id",
        "user_email",
        "email",
        "reference_audio_path",
        "audio_path",
        "request_trace",
    }
)

#: Findings that mean the candidate could not be delivered. Counted as
#: quality failures. Provider codes are deliberately absent — see
#: `AVAILABILITY_FINDINGS`.
QUALITY_FAILURE_FINDINGS: frozenset[str] = frozenset(
    {
        Finding.INVALID_AUDIO.value,
        Finding.NON_FINITE_SAMPLES.value,
        Finding.SILENT_OUTPUT.value,
        Finding.NEAR_SILENT.value,
        Finding.EARLY_COLLAPSE.value,
        Finding.SEVERE_CLIPPING.value,
        Finding.SPECTRAL_COLLAPSE.value,
        Finding.PHASE_UNSAFE.value,
        Finding.DC_OFFSET.value,
        Finding.CHANNEL_IMBALANCE.value,
        Finding.DURATION_SHORT.value,
        Finding.DURATION_LONG.value,
    }
)

#: Findings that mean the provider did not answer. These are an
#: availability problem, not a quality one: a model that times out is
#: not a model producing bad songs, and mixing them sends an operator to
#: the wrong system.
AVAILABILITY_FINDINGS: frozenset[str] = frozenset(
    {
        Finding.PROVIDER_TIMEOUT.value,
        Finding.PROVIDER_ERROR.value,
        Finding.PROVIDER_MISCONFIGURED.value,
    }
)


def _utc(value: datetime | None) -> datetime | None:
    """Timestamps are stored in UTC, and a naive one is assumed to be UTC.

    SQLite hands back naive datetimes for timezone-aware columns, so a
    strict reading would make every local test row unusable. Assuming
    UTC is right because that is what the column is documented to hold;
    what would be wrong is assuming local time.
    """
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


@dataclass
class InferenceObservation:
    """One generation, reduced to what may be counted.

    Deliberately flat: every field is a scalar a column can hold and an
    index can cover. A nested structure would have to be parsed per row
    per query, which is the thing the projection exists to avoid.
    """

    # ── identity ─────────────────────────────────────────────────────
    generation_id: str
    #: When the generation *started*, not when it was observed. Trend
    #: analysis asks what the system was doing at a time, and ingestion
    #: time would put a backfilled week into a single afternoon.
    occurred_at: datetime
    #: Digest of the request. Drilldown identity — never a grouping
    #: dimension; see `dimensions.Dimension`.
    request_sha256: str | None = None

    # ── provider identity ────────────────────────────────────────────
    provider: str = UNKNOWN
    model_name: str = UNKNOWN
    model_version: str = UNKNOWN
    #: The revision an operator compares. Composed rather than stored
    #: twice, so a query cannot group by a name and a version that
    #: disagree about which release they describe.
    provider_revision: str = UNKNOWN
    #: The repository revision that produced this generation. Written
    #: only by incremental ingestion, where the ingesting process is the
    #: process that produced it. Backfill writes UNKNOWN, because a
    #: backfill running today cannot know which commit ran last week.
    luber_revision: str = UNKNOWN

    # ── request dimensions ───────────────────────────────────────────
    task_type: str = UNKNOWN
    requested_duration_seconds: float | None = None
    duration_bucket: str = UNKNOWN
    #: Explicit request metadata only. Never inferred from prompt text —
    #: a language guessed from a prompt is a guess in a column that reads
    #: like a fact.
    language: str = UNKNOWN
    instrumental: str = UNKNOWN
    bpm_requested: bool = False
    key_requested: bool = False
    reference_conditioned: bool = False

    # ── QC configuration ─────────────────────────────────────────────
    qc_policy: str = UNKNOWN
    qc_schema_version: str = UNKNOWN
    qc_engine_version: str = UNKNOWN
    retry_policy_version: str = UNKNOWN
    finishing_version: str = UNKNOWN

    # ── outcome ──────────────────────────────────────────────────────
    generation_status: str = UNKNOWN
    generation_failure_code: str | None = None
    #: False for generations that predate Phase 29. Their retry counts
    #: are *unknown*, not zero, and every rate that depends on candidate
    #: data excludes them rather than averaging them in as perfect.
    qc_data_available: bool = False
    qc_outcome: str | None = None
    finishing_outcome: str | None = None

    # ── candidate counters ───────────────────────────────────────────
    candidate_count: int | None = None
    provider_call_count: int | None = None
    quality_retry_count: int | None = None
    selected_on_attempt: int | None = None
    first_candidate_accepted: bool | None = None
    retry_exhausted: bool | None = None
    candidate_rejections: int | None = None

    # ── latency ──────────────────────────────────────────────────────
    provider_latency_seconds: float | None = None
    qc_latency_seconds: float | None = None
    delivery_latency_seconds: float | None = None
    total_latency_seconds: float | None = None

    # ── findings ─────────────────────────────────────────────────────
    #: Every critical finding seen across all attempts, deduplicated.
    #: A code appearing twice in one generation is one generation with
    #: that problem, not two.
    critical_findings: tuple[str, ...] = ()
    #: Non-critical findings on the delivered candidate. Kept apart so a
    #: harshness advisory can never be counted as a failure.
    soft_findings: tuple[str, ...] = ()

    # ── bookkeeping ──────────────────────────────────────────────────
    schema_version: str = OBSERVABILITY_SCHEMA_VERSION
    ingested_at: datetime | None = None
    data_quality_issues: tuple[str, ...] = field(default_factory=tuple)

    # ── derived questions the aggregator asks ────────────────────────

    @property
    def completed(self) -> bool:
        return self.generation_status == "COMPLETED"

    @property
    def failed(self) -> bool:
        return self.generation_status == "FAILED"

    @property
    def cancelled(self) -> bool:
        """A user changing their mind is not a model failure.

        Counted separately everywhere and excluded from every quality
        rate, because a spike in cancellations after a UI change would
        otherwise read as the model getting worse.
        """
        return self.generation_status == "CANCELLED"

    @property
    def counts_toward_quality(self) -> bool:
        """Whether this row may take part in quality rates at all."""
        return self.qc_data_available and not self.cancelled

    @property
    def has_availability_failure(self) -> bool:
        return bool(set(self.critical_findings) & AVAILABILITY_FINDINGS)

    @property
    def has_quality_failure(self) -> bool:
        return bool(set(self.critical_findings) & QUALITY_FAILURE_FINDINGS)

    def has_finding(self, code: str) -> bool:
        return code in self.critical_findings

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["occurred_at"] = self.occurred_at.isoformat()
        payload["ingested_at"] = self.ingested_at.isoformat() if self.ingested_at else None
        payload["critical_findings"] = list(self.critical_findings)
        payload["soft_findings"] = list(self.soft_findings)
        payload["data_quality_issues"] = list(self.data_quality_issues)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)


def provider_revision(model_name: str | None, model_version: str | None) -> str:
    """One label for the thing an operator actually compares.

    A revision is a name *and* a version; comparing them separately lets
    a query answer "is v3 worse than v2" across two different models,
    which is not a question anybody asked.
    """
    name = model_name or UNKNOWN
    version = model_version or UNKNOWN
    if name == UNKNOWN and version == UNKNOWN:
        return UNKNOWN
    return f"{name}@{version}"


def observe(
    *,
    generation_id: str,
    status: str,
    occurred_at: datetime,
    completed_at: datetime | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    duration_requested: float | None = None,
    language: str | None = None,
    instrumental: bool | None = None,
    bpm: int | None = None,
    key_scale: str | None = None,
    edit_kind: str | None = None,
    has_reference: bool = False,
    error_code: str | None = None,
    qc_trace: dict[str, Any] | None = None,
    finishing_trace: dict[str, Any] | None = None,
    luber_revision: str | None = None,
    ingested_at: datetime | None = None,
) -> InferenceObservation:
    """Build one observation from a generation row and its traces.

    Every argument is a column or a parsed trace. Nothing is read from
    free text, and no argument accepts one — a function that could be
    handed a prompt is a function that will eventually be handed one.
    """
    started = _utc(occurred_at)
    assert started is not None
    finished = _utc(completed_at)

    observation = InferenceObservation(
        generation_id=str(generation_id),
        occurred_at=started,
        provider=provider or UNKNOWN,
        model_name=model_name or UNKNOWN,
        model_version=model_version or UNKNOWN,
        provider_revision=provider_revision(model_name, model_version),
        luber_revision=luber_revision or UNKNOWN,
        task_type=task_type(edit_kind=edit_kind, has_reference=has_reference),
        requested_duration_seconds=duration_requested,
        duration_bucket=duration_bucket(duration_requested),
        language=language or UNKNOWN,
        instrumental=UNKNOWN if instrumental is None else ("YES" if instrumental else "NO"),
        bpm_requested=bpm is not None,
        key_requested=bool(key_scale),
        reference_conditioned=has_reference,
        generation_status=status,
        generation_failure_code=error_code,
        ingested_at=_utc(ingested_at),
    )

    if finishing_trace:
        observation.finishing_outcome = finishing_trace.get("outcome")
        observation.finishing_version = finishing_trace.get("finishing_version") or UNKNOWN

    if qc_trace:
        _apply_qc_trace(observation, qc_trace)

    if finished is not None:
        observation.total_latency_seconds = max(0.0, (finished - started).total_seconds())
        candidate_phase = (qc_trace or {}).get("timings", {}).get("candidate_phase_seconds")
        if observation.total_latency_seconds is not None and candidate_phase is not None:
            # Post-processing, finishing, encoding and upload, together.
            # Named for what it covers rather than for finishing alone.
            observation.delivery_latency_seconds = max(
                0.0, observation.total_latency_seconds - float(candidate_phase)
            )

    observation.data_quality_issues = tuple(sorted(validate(observation)))
    return observation


def _apply_qc_trace(observation: InferenceObservation, trace: dict[str, Any]) -> None:
    """Copy the counted facts across. No inference, no defaults."""
    observation.qc_data_available = True
    observation.request_sha256 = trace.get("request_sha256")
    observation.qc_schema_version = trace.get("qc_schema_version") or UNKNOWN
    observation.qc_engine_version = trace.get("qc_engine_version") or UNKNOWN
    observation.retry_policy_version = trace.get("retry_policy_version") or UNKNOWN
    observation.qc_policy = (trace.get("policy") or {}).get("name") or UNKNOWN
    observation.qc_outcome = trace.get("outcome")
    observation.retry_exhausted = bool(trace.get("exhausted"))
    if trace.get("finishing_outcome"):
        observation.finishing_outcome = trace["finishing_outcome"]

    budget = trace.get("budget") or {}
    observation.provider_call_count = budget.get("provider_calls_used")
    observation.candidate_count = budget.get("candidates_generated")
    observation.quality_retry_count = budget.get("retry_rounds")

    attempts = trace.get("attempts") or []
    selected_id = trace.get("selected_candidate_id")
    critical: set[str] = set()
    soft: list[str] = []
    rejections = 0

    for attempt in attempts:
        findings = attempt.get("findings") or []
        is_selected = attempt.get("candidate_id") == selected_id
        for finding in findings:
            code = finding.get("code")
            if not code:
                continue
            if finding.get("severity") == Severity.CRITICAL.value:
                critical.add(code)
            elif is_selected:
                # Soft findings are only counted on the candidate that
                # was actually delivered. A rejected attempt's advisory
                # describes audio nobody received.
                soft.append(code)
        if attempt.get("status") == "REJECTED":
            rejections += 1
        if is_selected:
            observation.selected_on_attempt = attempt.get("attempt_index")

    observation.critical_findings = tuple(sorted(critical))
    observation.soft_findings = tuple(sorted(set(soft)))
    observation.candidate_rejections = rejections
    if observation.selected_on_attempt is not None:
        observation.first_candidate_accepted = observation.selected_on_attempt == 0
    elif observation.qc_outcome is not None:
        # Nothing was selected, so no attempt was accepted. That is a
        # fact, not a missing value.
        observation.first_candidate_accepted = False

    timings = trace.get("timings") or {}
    observation.provider_latency_seconds = timings.get("provider_seconds")
    observation.qc_latency_seconds = timings.get("qc_seconds")


# ── data quality ─────────────────────────────────────────────────────
#
# Telemetry can be wrong, and an observability system that crashes on a
# malformed row is one that goes blind exactly when something has broken
# upstream. Each check below names an impossibility; a row carrying one
# is still stored, still counted where it can be, and flagged.


class DataQuality:
    """Names for the impossibilities a row can carry."""

    NEGATIVE_LATENCY = "NEGATIVE_LATENCY"
    NO_CANDIDATES_ON_COMPLETION = "NO_CANDIDATES_ON_COMPLETION"
    CALLS_BELOW_CANDIDATES = "CALLS_BELOW_CANDIDATES"
    IMPOSSIBLE_RETRY_COUNT = "IMPOSSIBLE_RETRY_COUNT"
    MISSING_SCHEMA_VERSION = "MISSING_SCHEMA_VERSION"
    SELECTED_ATTEMPT_OUT_OF_RANGE = "SELECTED_ATTEMPT_OUT_OF_RANGE"


def validate(observation: InferenceObservation) -> list[str]:
    """Every impossibility this row carries. Empty is the normal answer."""
    issues: list[str] = []

    for value in (
        observation.provider_latency_seconds,
        observation.qc_latency_seconds,
        observation.delivery_latency_seconds,
        observation.total_latency_seconds,
    ):
        if value is not None and value < 0:
            issues.append(DataQuality.NEGATIVE_LATENCY)
            break

    if observation.qc_data_available:
        if observation.qc_schema_version == UNKNOWN:
            issues.append(DataQuality.MISSING_SCHEMA_VERSION)
        if observation.completed and not observation.candidate_count:
            issues.append(DataQuality.NO_CANDIDATES_ON_COMPLETION)
        calls = observation.provider_call_count
        candidates = observation.candidate_count
        if calls is not None and candidates is not None and calls < candidates:
            # Every candidate cost a call; a call may produce none.
            issues.append(DataQuality.CALLS_BELOW_CANDIDATES)
        retries = observation.quality_retry_count
        if retries is not None and retries < 0:
            issues.append(DataQuality.IMPOSSIBLE_RETRY_COUNT)
        elif retries is not None and candidates is not None and retries > candidates:
            issues.append(DataQuality.IMPOSSIBLE_RETRY_COUNT)
        selected = observation.selected_on_attempt
        if selected is not None and (
            selected < 0 or (candidates is not None and selected >= max(candidates, 1))
        ):
            issues.append(DataQuality.SELECTED_ATTEMPT_OUT_OF_RANGE)

    return issues


__all__ = [
    "AVAILABILITY_FINDINGS",
    "FORBIDDEN_FIELDS",
    "QUALITY_FAILURE_FINDINGS",
    "DataQuality",
    "InferenceObservation",
    "observe",
    "provider_revision",
    "validate",
]
