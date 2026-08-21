"""Wiring: a repository on one side, the analytics engine on the other.

The engine's `ObservationStore` protocol is synchronous, because
aggregation is arithmetic over a list and making it async would colour
every function in the package for no benefit. The repository is async,
because the database is.

So this module bridges them the only honest way: it loads a window's
rows once, and hands the engine a materialised store. That is a real
constraint — a window whose rows do not fit in memory cannot be
aggregated this way — and it is the right trade at this scale. A week of
traffic at LUBER's volume is thousands of rows of scalars, not millions,
and the alternative (pushing every metric into SQL) would put the
definition of "retry rate" in two places: a Python function and a query.
Two definitions of one metric is how a dashboard and a report come to
disagree.

The scale test in the suite is what keeps this honest: 100,000
observations aggregate in well under a second, and if that stops being
true the answer is a pre-aggregation table, not a quiet regression.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from luber_inference_observability.dimensions import Segment
from luber_inference_observability.incidents import (
    IncidentLedger,
    IncidentPolicy,
    InferenceIncident,
)
from luber_inference_observability.ingest import IngestResult, as_rows, project
from luber_inference_observability.storage import (
    InMemoryObservationStore,
    from_mapping,
)
from luber_inference_observability.windows import TimeWindow


class ObservabilityRepositoryLike(Protocol):
    """The repository methods this service uses.

    Structural, so the observability package still imports no SQLAlchemy
    and the whole service can be exercised against a fake.
    """

    async def upsert_observations(self, rows: Any) -> int: ...
    async def select_observations(
        self,
        *,
        start: datetime,
        end: datetime,
        filters: dict[str, Any] | None = ...,
        limit: int | None = ...,
        offset: int = ...,
    ) -> list[dict[str, Any]]: ...
    async def latest_observed_at(self) -> datetime | None: ...
    async def generations_to_ingest(
        self, *, since: datetime | None = ..., limit: int = ..., statuses: Any = ...
    ) -> list[Any]: ...
    async def upsert_incidents(self, rows: Any) -> int: ...
    async def all_incidents(self) -> list[dict[str, Any]]: ...


async def load_store(
    repository: ObservabilityRepositoryLike,
    *,
    window: TimeWindow,
    segment: Segment | None = None,
) -> InMemoryObservationStore:
    """Materialise one window's observations for the engine to count."""
    rows = await repository.select_observations(
        start=window.start,
        end=window.end,
        filters=(segment.to_dict() if segment else None) or None,
    )
    return InMemoryObservationStore(from_mapping(row) for row in rows)


async def load_store_spanning(
    repository: ObservabilityRepositoryLike,
    *,
    current: TimeWindow,
    baseline_span: timedelta,
    baseline_gap: timedelta,
) -> InMemoryObservationStore:
    """Load the current window *and* everything its baseline needs.

    One query rather than two, because the detector asks for both and a
    store holding only the current window would silently produce
    BASELINE_BUILDING for everything — a failure that looks exactly like
    a quiet week.
    """
    start = current.start - baseline_gap - baseline_span
    rows = await repository.select_observations(start=start, end=current.end)
    return InMemoryObservationStore(from_mapping(row) for row in rows)


async def ingest(
    repository: ObservabilityRepositoryLike,
    *,
    since: datetime | None = None,
    limit: int = 500,
    luber_revision: str | None = None,
    full: bool = False,
) -> IngestResult:
    """Project finished generations into observations.

    ``full=True`` is the backfill: it starts from the beginning rather
    than from the watermark. Running it twice writes the same rows twice
    and changes no count, because the projection is keyed on the
    generation.

    Without it, ingestion resumes from the newest observation already
    stored, which is what keeps a scheduled run from rescanning the whole
    table on every tick.
    """
    watermark = None if full else (since or await repository.latest_observed_at())
    generations = await repository.generations_to_ingest(since=watermark, limit=limit)
    observations, result = project(generations, luber_revision=luber_revision)
    if observations:
        await repository.upsert_observations(as_rows(observations))
    return result


async def ingest_one(
    repository: ObservabilityRepositoryLike,
    generation: Any,
    *,
    luber_revision: str | None = None,
) -> bool:
    """Record one finished generation, now.

    The incremental path a worker calls. Returns whether anything was
    written, and never raises: a failure to record analytics must not
    fail a generation that already succeeded. The caller logs and moves
    on; the next scheduled ingest picks the row up from the watermark.
    """
    observations, result = project([generation], luber_revision=luber_revision)
    if not observations:
        return False
    await repository.upsert_observations(as_rows(observations))
    return result.written > 0


def _incident_to_row(incident: InferenceIncident) -> dict[str, Any]:
    import json

    payload = incident.to_dict(evidence_limit=None)
    return {
        "incident_id": incident.incident_id,
        "created_at": incident.created_at,
        "status": incident.status,
        "severity": incident.severity,
        "peak_severity": incident.peak_severity,
        "finding_type": incident.finding_type,
        "category": incident.category,
        "metric": incident.metric,
        "provider": incident.provider,
        "provider_revision": incident.provider_revision,
        "segment": json.dumps(incident.segment.to_dict(), sort_keys=True),
        "first_seen": incident.first_seen,
        "last_seen": incident.last_seen,
        "occurrence_count": incident.occurrence_count,
        "consecutive_clean": incident.consecutive_clean,
        "baseline_window": json.dumps(incident.baseline_window, sort_keys=True),
        "current_window": json.dumps(incident.current_window, sort_keys=True),
        "evidence": json.dumps(payload["evidence"], sort_keys=True),
        "recommendations": json.dumps(list(incident.recommendations), sort_keys=True),
        "acknowledged_at": incident.acknowledged_at,
        "acknowledged_by": incident.acknowledged_by,
        "resolved_at": incident.resolved_at,
        "dismissed_at": incident.dismissed_at,
        "dismissed_by": incident.dismissed_by,
        "dismissal_reason": incident.dismissal_reason,
        "incident_policy_version": incident.incident_policy_version,
    }


def _row_to_incident(row: dict[str, Any]) -> InferenceIncident:
    import json

    from luber_inference_observability.incidents import IncidentEvidence

    def _load(raw: Any, fallback: Any) -> Any:
        if raw is None:
            return fallback
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except ValueError:
                return fallback
        return raw

    segment = Segment.of(**_load(row.get("segment"), {}))
    evidence_payload = _load(row.get("evidence"), [])
    evidence = [
        IncidentEvidence(
            observed_at=datetime.fromisoformat(item["observed_at"]),
            status=item["status"],
            severity=item["severity"],
            baseline_value=item.get("baseline_value"),
            current_value=item.get("current_value"),
            absolute_delta=item.get("absolute_delta"),
            relative_delta=item.get("relative_delta"),
            current_sample_count=item.get("current_sample_count", 0),
            baseline_sample_count=item.get("baseline_sample_count", 0),
            explanation=item.get("explanation", ""),
        )
        for item in evidence_payload
    ]
    return InferenceIncident(
        incident_id=row["incident_id"],
        created_at=_aware(row["created_at"]),
        finding_type=row["finding_type"],
        category=row["category"],
        metric=row["metric"],
        segment=segment,
        status=row["status"],
        severity=row["severity"],
        peak_severity=row["peak_severity"],
        provider=row.get("provider"),
        provider_revision=row.get("provider_revision"),
        first_seen=_aware(row.get("first_seen")),
        last_seen=_aware(row.get("last_seen")),
        occurrence_count=row.get("occurrence_count", 0),
        consecutive_clean=row.get("consecutive_clean", 0),
        baseline_window=_load(row.get("baseline_window"), {}),
        current_window=_load(row.get("current_window"), {}),
        evidence=evidence,
        recommendations=tuple(_load(row.get("recommendations"), [])),
        acknowledged_at=_aware(row.get("acknowledged_at")),
        acknowledged_by=row.get("acknowledged_by"),
        resolved_at=_aware(row.get("resolved_at")),
        dismissed_at=_aware(row.get("dismissed_at")),
        dismissed_by=row.get("dismissed_by"),
        dismissal_reason=row.get("dismissal_reason"),
        incident_policy_version=row.get("incident_policy_version", ""),
    )


def _aware(value: Any) -> Any:
    """SQLite returns naive datetimes for timezone-aware columns."""
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


async def load_ledger(
    repository: ObservabilityRepositoryLike,
    *,
    policy: IncidentPolicy | None = None,
) -> IncidentLedger:
    rows = await repository.all_incidents()
    return IncidentLedger((_row_to_incident(row) for row in rows), policy=policy)


async def save_ledger(repository: ObservabilityRepositoryLike, ledger: IncidentLedger) -> int:
    return await repository.upsert_incidents(
        [_incident_to_row(incident) for incident in ledger.all()]
    )


__all__ = [
    "ObservabilityRepositoryLike",
    "ingest",
    "ingest_one",
    "load_ledger",
    "load_store",
    "load_store_spanning",
    "save_ledger",
]
