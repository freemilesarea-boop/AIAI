"""The inference observability console's HTTP surface.

One router, the same gate. `require_operator` and
`enforce_operator_origin` are applied at the router rather than per
route, so an endpoint added later is protected by having been added.
There is no second authentication mechanism here and no role: the
console is a deployment switch, exactly as Phase 28 decided, and this
router is mounted under the same condition.

Everything here is a `GET` except two operator actions on incidents.
Nothing in this file can start a generation, change a QC threshold,
switch a policy or disable a provider — Phase 30 detects and explains,
and the strongest thing it will do about a CRITICAL incident is let
somebody write down that they have seen it.

The response models carry no field a prompt could occupy and the data
comes from a projection with no column one could be stored in. Two
independent reasons, so neither has to be remembered.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from luber_api.dependencies import get_session_factory
from luber_api.ops.inference_readmodel import (
    InferenceReadModel,
    ingest_status,
    segment_for,
    window_for,
)
from luber_api.ops.inference_schemas import (
    ActionResponse,
    GenerationListResponse,
    GenerationTraceResponse,
    IncidentListResponse,
    IncidentView,
    IngestStatusResponse,
    OverviewResponse,
    ProvidersResponse,
    RegressionView,
    SegmentsResponse,
    SummaryResponse,
    TrendResponse,
)
from luber_api.ops.security import enforce_operator_origin, require_operator
from luber_api.settings import ApiSettings, get_settings
from luber_database import ObservabilityRepository
from luber_inference_observability.baselines import (
    DEFAULT_BASELINE_GAP,
    DEFAULT_BASELINE_SPAN,
)
from luber_inference_observability.dimensions import GroupingTooWide
from luber_inference_observability.incidents import IncidentStatus
from luber_inference_observability.queries import evaluate_segments
from luber_inference_observability.regressions import regressions as crossed_only
from luber_inference_observability.service import (
    load_ledger,
    load_store_spanning,
    save_ledger,
)
from luber_inference_observability.windows import DURATIONS

router = APIRouter(
    prefix="/v1/ops/inference",
    tags=["operator-inference-console"],
    dependencies=[Depends(require_operator), Depends(enforce_operator_origin)],
)

#: The default window. A day is long enough to have samples in it on a
#: development machine and short enough that an operator opening the
#: console sees today rather than last week.
DEFAULT_WINDOW = "24h"

WindowParam = Annotated[str, Query(pattern="|".join(sorted(DURATIONS)))]


async def _repository(request: Request) -> Any:
    factory: async_sessionmaker[AsyncSession] = get_session_factory(request)
    async with factory() as session:
        yield ObservabilityRepository(session)


Repository = Annotated[Any, Depends(_repository)]
Settings = Annotated[ApiSettings, Depends(get_settings)]


def _read(repository: Any) -> InferenceReadModel:
    return InferenceReadModel(repository)


def _limit(requested: int, settings: ApiSettings) -> int:
    """Bounded server-side, because a client asking for everything gets
    a page rather than a month."""
    return max(1, min(requested, settings.ops_page_size_limit))


def _segment(
    provider: str | None,
    revision: str | None,
    task: str | None,
    duration_bucket: str | None,
) -> Any:
    return segment_for(
        provider=provider,
        provider_revision=revision,
        task_type=task,
        duration_bucket=duration_bucket,
    )


# ── health ───────────────────────────────────────────────────────────


@router.get("/overview", response_model=OverviewResponse)
async def overview(
    repository: Repository,
    window: WindowParam = DEFAULT_WINDOW,
    provider: str | None = None,
    revision: str | None = None,
    task: str | None = None,
    duration_bucket: str | None = None,
) -> OverviewResponse:
    return await _read(repository).overview(
        window=window_for(window),
        segment=_segment(provider, revision, task, duration_bucket),
    )


@router.get("/summary", response_model=SummaryResponse)
async def summary(
    repository: Repository,
    window: WindowParam = DEFAULT_WINDOW,
    provider: str | None = None,
    revision: str | None = None,
    task: str | None = None,
    duration_bucket: str | None = None,
) -> SummaryResponse:
    return await _read(repository).summary(
        window=window_for(window),
        segment=_segment(provider, revision, task, duration_bucket),
    )


@router.get("/trend", response_model=TrendResponse)
async def trend(
    repository: Repository,
    chart: Annotated[str, Query(pattern="retry|failure|latency")],
    window: WindowParam = DEFAULT_WINDOW,
    provider: str | None = None,
    revision: str | None = None,
    task: str | None = None,
    duration_bucket: str | None = None,
) -> TrendResponse:
    try:
        return await _read(repository).trend(
            window=window_for(window),
            chart=chart,
            size=window,
            segment=_segment(provider, revision, task, duration_bucket),
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.get("/ingest-status", response_model=IngestStatusResponse)
async def ingestion(repository: Repository) -> IngestStatusResponse:
    """Whether the numbers on screen are current.

    Its own endpoint rather than a field on the overview: a console that
    could not reach it should still render, with the staleness banner
    absent rather than the whole page failing.
    """
    return await ingest_status(repository)


# ── providers and segments ───────────────────────────────────────────


@router.get("/providers", response_model=ProvidersResponse)
async def providers(
    repository: Repository, window: WindowParam = DEFAULT_WINDOW
) -> ProvidersResponse:
    return await _read(repository).providers(window=window_for(window))


@router.get("/providers/compare")
async def compare_providers(
    repository: Repository,
    left: str,
    right: str,
    window: WindowParam = "7d",
    minimum_samples: int = 30,
) -> dict[str, Any]:
    """Two revisions over the same window.

    Same period for both, because comparing one revision's Tuesday with
    another's Saturday compares the days as much as the models.
    """
    return await _read(repository).compare(
        window=window_for(window),
        left=left,
        right=right,
        minimum_samples=max(1, minimum_samples),
    )


# Named for what it does rather than for one thing that might explain
# it. It compares two windows either side of any moment; calling it
# "deployment" implied this endpoint knows about deployments, and made a
# read-only comparison trip the operator-console safety check that scans
# routes for words like "deploy".
@router.get("/before-after")
async def before_after(
    repository: Repository,
    at: str,
    hours: Annotated[int, Query(ge=1, le=168)] = 24,
) -> dict[str, Any]:
    """Before and after a moment. Correlation, and it says so."""
    # A `+` in a query string decodes to a space, so an ISO timestamp
    # with a UTC offset arrives as "…T12:00:00 00:00" unless the caller
    # encoded it. Both forms are accepted rather than only the pedantic
    # one: refusing a timestamp because of a URL convention would be a
    # correctness argument nobody at 3am wants to have.
    candidate = at.strip()
    try:
        moment = datetime.fromisoformat(candidate)
    except ValueError:
        try:
            moment = datetime.fromisoformat(candidate.replace(" ", "+", 1))
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"{at!r} is not an ISO timestamp"
            ) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return await _read(repository).deployment(at=moment, hours=hours)


@router.get("/segments", response_model=SegmentsResponse)
async def segments(
    repository: Repository,
    settings: Settings,
    window: WindowParam = DEFAULT_WINDOW,
    group_by: str = "provider,duration_bucket",
    metric: str = "generation_failure_rate",
    minimum_samples: int = 30,
    limit: int = 10,
) -> SegmentsResponse:
    try:
        return await _read(repository).segments(
            window=window_for(window),
            by=tuple(item for item in group_by.split(",") if item),
            metric=metric,
            minimum_samples=max(1, minimum_samples),
            limit=_limit(limit, settings),
        )
    except GroupingTooWide as exc:
        # 409 rather than 400: the request was well formed, and the
        # world's answer is that this split cannot support a finding.
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


# ── regressions ──────────────────────────────────────────────────────


@router.get("/regressions", response_model=list[RegressionView])
async def regressions(
    repository: Repository,
    window: WindowParam = DEFAULT_WINDOW,
    group_by: str = "provider_revision",
) -> list[RegressionView]:
    """What crossed a threshold in this window.

    Read-only and stateless: it evaluates and reports, and it does not
    touch the incident ledger. Opening incidents is the scheduled
    detector's job, so refreshing a browser tab cannot mint them.
    """
    current = window_for(window)
    store = await load_store_spanning(
        repository,
        current=current,
        baseline_span=DEFAULT_BASELINE_SPAN,
        baseline_gap=DEFAULT_BASELINE_GAP,
    )
    findings = evaluate_segments(store, current=current, by=(group_by,))
    return [RegressionView(**item.to_dict()) for item in crossed_only(findings)]


# ── incidents ────────────────────────────────────────────────────────


@router.get("/incidents", response_model=IncidentListResponse)
async def incidents(
    repository: Repository,
    settings: Settings,
    include_closed: bool = False,
    limit: int = 25,
    offset: int = 0,
) -> IncidentListResponse:
    statuses = (
        None if include_closed else [IncidentStatus.OPEN.value, IncidentStatus.ACKNOWLEDGED.value]
    )
    return await _read(repository).incidents(
        statuses=statuses, limit=_limit(limit, settings), offset=max(0, offset)
    )


@router.get("/incidents/{incident_id}", response_model=IncidentView)
async def incident(repository: Repository, incident_id: str) -> IncidentView:
    found = await _read(repository).incident(incident_id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such incident.")
    return found


@router.post("/incidents/{incident_id}/acknowledge", response_model=ActionResponse)
async def acknowledge(
    repository: Repository,
    incident_id: str,
    operator: Annotated[str, Query(min_length=1, max_length=100)],
) -> ActionResponse:
    """Record that a human has seen it. Measurement continues.

    Acknowledging does not suppress anything: evidence keeps
    accumulating and an acknowledged incident that worsens escalates.
    """
    ledger = await load_ledger(repository)
    if ledger.get(incident_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such incident.")
    ledger.acknowledge(incident_id, by=operator, at=datetime.now(UTC))
    await save_ledger(repository, ledger)
    updated = await _read(repository).incident(incident_id)
    assert updated is not None
    return ActionResponse(ok=True, incident=updated)


@router.post("/incidents/{incident_id}/dismiss", response_model=ActionResponse)
async def dismiss(
    repository: Repository,
    incident_id: str,
    operator: Annotated[str, Query(min_length=1, max_length=100)],
    reason: Annotated[str, Query(min_length=1, max_length=500)],
) -> ActionResponse:
    """Close it as not worth acting on, with the reason on the record.

    The reason is required rather than optional. Why something was
    ignored is exactly what the next person needs when it comes back,
    and nothing here deletes the history.
    """
    ledger = await load_ledger(repository)
    if ledger.get(incident_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such incident.")
    try:
        ledger.dismiss(incident_id, by=operator, reason=reason, at=datetime.now(UTC))
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    await save_ledger(repository, ledger)
    updated = await _read(repository).incident(incident_id)
    assert updated is not None
    return ActionResponse(ok=True, incident=updated)


# ── drilldown ────────────────────────────────────────────────────────


@router.get("/generations", response_model=GenerationListResponse)
async def generations(
    repository: Repository,
    settings: Settings,
    window: WindowParam = DEFAULT_WINDOW,
    provider: str | None = None,
    revision: str | None = None,
    task: str | None = None,
    duration_bucket: str | None = None,
    only_failures: bool = False,
    limit: int = 25,
    offset: int = 0,
) -> GenerationListResponse:
    return await _read(repository).generations(
        window=window_for(window),
        segment=_segment(provider, revision, task, duration_bucket),
        limit=_limit(limit, settings),
        offset=max(0, offset),
        only_failures=only_failures,
    )


@router.get("/generations/{generation_id}", response_model=GenerationTraceResponse)
async def generation(
    request: Request, repository: Repository, generation_id: uuid.UUID
) -> GenerationTraceResponse:
    """One generation's candidate phase, safely.

    The attempt list comes from the Phase 29 trace, read here rather
    than copied into the projection: it is bulky, it is only ever wanted
    one row at a time, and duplicating it would double the storage of
    the analytics table for a screen nobody opens in bulk.

    The read is a targeted fetch of two columns by primary key. It
    cannot return a prompt because it does not select one, and the
    response model has no field to put one in.
    """
    from sqlalchemy import select

    from luber_database.models.generation import Generation

    factory: async_sessionmaker[AsyncSession] = get_session_factory(request)
    async with factory() as session:
        row = (
            await session.execute(
                select(Generation.inference_qc_trace).where(Generation.id == generation_id)
            )
        ).scalar_one_or_none()

    qc_trace = None
    if row:
        import json

        try:
            parsed = json.loads(row)
            qc_trace = parsed if isinstance(parsed, dict) else None
        except ValueError:
            qc_trace = None

    found = await _read(repository).generation(generation_id, qc_trace=qc_trace)
    if found is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No observation for that generation. It may not have been ingested yet.",
        )
    return found


__all__ = ["router"]
