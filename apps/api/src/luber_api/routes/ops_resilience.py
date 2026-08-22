"""The circuit view: what the resilience layer is doing, and nothing more.

Read-only, deliberately, and the reason is the console's own boundary
rather than caution. `/ops` is a **non-production deployment switch** —
it is refused outright when the environment is production and is not
mounted unless the process was started with it on. An incident that
needs a circuit forced open is by definition happening in production,
where this console does not exist. Putting the override here would
build an incident tool into the one place it can never be used during
an incident, and would make the console the second thing to check when
traffic stops.

So overrides live in `python -m luber_provider_resilience` — a CLI that
runs wherever the database is reachable, including production, and whose
mutations are audited into the same transition table the panel below
reads. See `docs/PROVIDER_INCIDENT_RUNBOOK.md`.

Everything here is a `GET`. Nothing in this file can open a circuit,
close one, change a threshold, or start a generation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from luber_api.dependencies import get_session_factory
from luber_api.ops.resilience_schemas import (
    CapabilityReadinessView,
    CircuitListResponse,
    CircuitView,
    PolicyResponse,
    ProviderReadinessView,
    ReadinessResponse,
    TransitionListResponse,
    TransitionView,
)
from luber_api.ops.security import enforce_operator_origin, require_operator
from luber_api.settings import ApiSettings, get_settings
from luber_database import ResilienceRepository
from luber_generation_client.resilience_factory import profiles_from_settings
from luber_provider_resilience import (
    CircuitPolicy,
    DurableCircuitStore,
    FailoverMode,
    readiness,
)
from luber_provider_resilience.versions import version_block

router = APIRouter(
    prefix="/v1/ops/resilience",
    tags=["operator-resilience-console"],
    dependencies=[Depends(require_operator), Depends(enforce_operator_origin)],
)

Settings = Annotated[ApiSettings, Depends(get_settings)]


async def _repository(request: Request) -> Any:
    factory: async_sessionmaker[AsyncSession] = get_session_factory(request)
    yield ResilienceRepository(factory)


Repository = Annotated[Any, Depends(_repository)]


def _policy() -> CircuitPolicy:
    """The policy the running system uses.

    Constructed from defaults rather than from settings because that is
    where the thresholds live today. When a deployment can tune them,
    this is the one line that changes — and until then the console
    cannot show a threshold the worker is not using.
    """
    return CircuitPolicy()


def _circuit_view(record: Any, *, now: datetime, policy: CircuitPolicy) -> CircuitView:
    return CircuitView(
        circuit_key=record.identity.key(),
        provider=record.identity.provider,
        task_type=record.identity.task_type,
        state=record.state,
        control=record.control,
        consecutive_failures=record.consecutive_failures,
        consecutive_successes=record.consecutive_successes,
        sample_count=record.sample_count(),
        failure_count=record.failure_count(),
        failure_rate=record.failure_rate(policy),
        opened_at=_iso(record.opened_at),
        open_until=_iso(record.open_until),
        open_reason=record.open_reason,
        consecutive_opens=record.consecutive_opens,
        active_probes=record.active_probes(now),
        probe_successes=record.probe_successes,
        last_failure_at=_iso(record.last_failure_at),
        last_failure_category=record.last_failure_category,
        last_success_at=_iso(record.last_success_at),
        last_transition_at=_iso(record.last_transition_at),
        manual_reason=record.manual_reason,
        manual_operator=record.manual_operator,
        revision=record.revision,
    )


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


@router.get("/circuits", response_model=CircuitListResponse)
async def list_circuits(repository: Repository, settings: Settings) -> CircuitListResponse:
    """Every circuit this deployment has ever written.

    Including circuits for providers no longer configured. They are
    named rather than dropped: a circuit left open against a provider
    that has been removed explains nothing on its own, and hiding it
    would make it impossible to explain at all.
    """
    store = DurableCircuitStore(repository)
    policy = _policy()
    now = datetime.now(UTC)
    records = await store.all_circuits()

    configured = {settings.generation_provider}
    unconfigured = sorted(
        {record.identity.provider for record in records} - configured,
    )

    return CircuitListResponse(
        **version_block(),
        at=now.isoformat(),
        circuits=[_circuit_view(record, now=now, policy=policy) for record in records],
        unconfigured_providers=unconfigured,
    )


@router.get("/transitions", response_model=TransitionListResponse)
async def list_transitions(
    repository: Repository,
    settings: Settings,
    circuit_key: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> TransitionListResponse:
    """The audit trail: every state change, newest first.

    Manual actions appear here beside automatic ones, with the operator
    who took them. A circuit that is closed during an outage should be
    explainable afterwards without asking who did it.
    """
    bounded = max(1, min(limit, settings.ops_page_size_limit))
    rows = await repository.transitions(circuit_key=circuit_key, limit=bounded)
    return TransitionListResponse(
        **version_block(),
        transitions=[
            TransitionView(
                id=row["id"],
                circuit_key=row["circuit_key"],
                provider=row["provider"],
                task_type=row["task_type"],
                previous_state=row["previous_state"],
                current_state=row["current_state"],
                occurred_at=_iso(row["occurred_at"]) or "",
                reason=row["reason"],
                automatic=row["automatic"],
                operator=row["operator"],
                circuit_policy_version=row["circuit_policy_version"],
            )
            for row in rows
        ],
    )


@router.get("/readiness", response_model=ReadinessResponse)
async def generation_readiness(repository: Repository, settings: Settings) -> ReadinessResponse:
    """What the service can generate right now, capability by capability.

    Not `/health` and not `/ready`. The API can be up, its dependencies
    fine, and every provider circuit open — this is the third answer,
    and it is derived from providers and circuits rather than stored, so
    it cannot report AVAILABLE about a circuit that opened a minute ago.
    """
    profiles = await profiles_from_settings(settings)
    report = await readiness(profiles, store=DurableCircuitStore(repository))
    return ReadinessResponse(
        **version_block(),
        at=report.at.isoformat(),
        generation_available=report.generation_available,
        degraded=report.degraded,
        summary=report.summary,
        capabilities=[
            CapabilityReadinessView(
                capability=item.capability,
                status=item.status,
                detail=item.detail,
                providers=[
                    ProviderReadinessView(
                        provider=view.provider,
                        revision=view.revision,
                        circuit_state=view.circuit_state,
                        control=view.control,
                        open_reason=view.open_reason,
                        open_until=view.open_until,
                    )
                    for view in item.providers
                ],
            )
            for item in report.capabilities
        ],
        metrics={key: int(value) for key, value in report.metrics.items()},
    )


@router.get("/policy", response_model=PolicyResponse)
async def policy(settings: Settings) -> PolicyResponse:
    """The thresholds in force, so a count can be read against them.

    Also the honest answer about failover: a mode set to
    SAFE_EQUIVALENT_ONLY on a deployment with one provider can never
    move a request, and `failover_possible` says so rather than leaving
    the console implying a redundancy that does not exist.
    """
    profiles = await profiles_from_settings(settings)
    names = sorted(item.name for item in profiles)
    mode = str(settings.provider_failover_mode).upper()
    return PolicyResponse(
        **version_block(),
        resilience_enabled=settings.provider_resilience_enabled,
        failover_mode=mode,
        failover_possible=mode != FailoverMode.DISABLED.value and len(names) > 1,
        routable_providers=names,
        circuit_policy=_policy().to_dict(),
    )


__all__ = ["router"]
