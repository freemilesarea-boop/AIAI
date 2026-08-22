"""What the service can actually do right now, capability by capability.

Three questions get confused constantly, and separating them is most of
this module's value:

**Is the process up?** `/health`. Never touches a dependency.

**Are the dependencies reachable?** `/ready`. PostgreSQL, Redis.

**Can we generate music?** Neither of the above. The API can be alive,
Postgres and Redis fine, and every provider circuit open. A load
balancer must not take the API out of rotation for that — the API is
working; the thing it calls is not.

So generation readiness is a third answer, derived rather than
configured: for each capability, which providers could serve it, and
what are their circuits doing. AVAILABLE when at least one is closed,
DEGRADED when the only ones left are probing their way back, and
UNAVAILABLE when there is nothing.

Degraded mode here means **fewer things work, and we say which**. It
never means quietly producing something different: a request the system
cannot serve is refused, not downgraded. Dropping a reference track and
generating anyway would be the worst possible interpretation of
"degraded" — the user gets a song, believes it is what they asked for,
and nothing anywhere says otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from luber_provider_resilience.capabilities import Capability, ProviderProfile
from luber_provider_resilience.circuit import (
    CircuitIdentity,
    CircuitState,
    ControlMode,
)
from luber_provider_resilience.store import CircuitStore, utcnow
from luber_provider_resilience.versions import version_block


class CapabilityStatus(StrEnum):
    """Whether a capability can be served."""

    #: At least one provider can serve it and its circuit is closed.
    AVAILABLE = "AVAILABLE"
    #: Providers exist but are all mid-recovery. Requests may be
    #: admitted as probes; most will be refused.
    DEGRADED = "DEGRADED"
    #: No provider can serve this capability right now.
    UNAVAILABLE = "UNAVAILABLE"
    #: No configured provider ever could. Distinct from UNAVAILABLE:
    #: nothing is broken, the deployment simply does not offer it.
    NOT_CONFIGURED = "NOT_CONFIGURED"


#: The capabilities a readiness report covers.
#:
#: Request-shaped rather than every internal capability, because this
#: view answers "what can a user ask for", and `LYRICS` is not something
#: a user asks for on its own.
REPORTED_CAPABILITIES: tuple[str, ...] = (
    Capability.TEXT_TO_MUSIC.value,
    Capability.REFERENCE_CONDITIONED.value,
    Capability.EXTEND.value,
    Capability.REPLACE_RANGE.value,
    Capability.COVER.value,
)

#: Which task type's circuit governs each capability. Circuits are keyed
#: by provider *and* task, so asking about the wrong task would report a
#: healthy circuit for a broken path.
_CAPABILITY_TASK: dict[str, str] = {
    Capability.TEXT_TO_MUSIC.value: "TEXT_TO_MUSIC",
    Capability.REFERENCE_CONDITIONED.value: "REFERENCE_CONDITIONED",
    Capability.EXTEND.value: "EXTEND",
    Capability.REPLACE_RANGE.value: "REPLACE_RANGE",
    Capability.COVER.value: "COVER",
}


@dataclass(frozen=True)
class ProviderView:
    """One provider's contribution to one capability."""

    provider: str
    revision: str
    circuit_state: str
    control: str
    open_reason: str | None = None
    open_until: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "revision": self.revision,
            "circuit_state": self.circuit_state,
            "control": self.control,
            "open_reason": self.open_reason,
            "open_until": self.open_until,
        }


@dataclass(frozen=True)
class CapabilityReadiness:
    """Whether one capability can be served, and by whom."""

    capability: str
    status: str
    providers: tuple[ProviderView, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "status": self.status,
            "detail": self.detail,
            "providers": [item.to_dict() for item in self.providers],
        }


@dataclass(frozen=True)
class ReadinessReport:
    """Generation readiness, derived from providers and circuits."""

    at: datetime
    capabilities: tuple[CapabilityReadiness, ...]
    #: True when at least one capability is servable. A service with
    #: text-to-music up and covers down is still generating music.
    generation_available: bool = False
    #: True when something is servable but not everything.
    degraded: bool = False
    summary: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def status_of(self, capability: str) -> str:
        for item in self.capabilities:
            if item.capability == capability:
                return item.status
        return CapabilityStatus.NOT_CONFIGURED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            **version_block(),
            "at": self.at.isoformat(),
            "generation_available": self.generation_available,
            "degraded": self.degraded,
            "summary": self.summary,
            "capabilities": [item.to_dict() for item in self.capabilities],
            "metrics": self.metrics,
        }

    def render(self) -> str:
        """The plain form, for a CLI or a log line."""
        lines = [f"generation: {'AVAILABLE' if self.generation_available else 'UNAVAILABLE'}"]
        if self.degraded:
            lines[0] += " (DEGRADED)"
        for item in self.capabilities:
            lines.append(f"  {item.capability}: {item.status}")
            if item.detail:
                lines.append(f"      {item.detail}")
        return "\n".join(lines)


async def readiness(
    profiles: Sequence[ProviderProfile],
    *,
    store: CircuitStore,
    now: datetime | None = None,
    capabilities: Sequence[str] = REPORTED_CAPABILITIES,
) -> ReadinessReport:
    """Derive what the service can do from providers and circuit state.

    Derived rather than stored, so it cannot go stale. A cached
    readiness view is a view that says AVAILABLE while the circuit that
    contradicts it opened thirty seconds ago.
    """
    moment = now or utcnow()
    reports: list[CapabilityReadiness] = []

    for capability in capabilities:
        task = _CAPABILITY_TASK.get(capability, "ANY")
        able = [item for item in profiles if item.supports(capability)]

        if not able:
            reports.append(
                CapabilityReadiness(
                    capability=capability,
                    status=CapabilityStatus.NOT_CONFIGURED.value,
                    detail="no configured provider offers this",
                )
            )
            continue

        views: list[ProviderView] = []
        closed = 0
        probing = 0
        for profile in able:
            record = await store.load(CircuitIdentity(profile.name, task))
            views.append(
                ProviderView(
                    provider=profile.name,
                    revision=profile.revision,
                    circuit_state=record.state,
                    control=record.control,
                    open_reason=record.open_reason,
                    open_until=(record.open_until.isoformat() if record.open_until else None),
                )
            )
            if record.state == CircuitState.CLOSED.value:
                closed += 1
            elif record.state == CircuitState.HALF_OPEN.value:
                probing += 1

        if closed:
            status = CapabilityStatus.AVAILABLE.value
            detail = f"{closed} of {len(able)} provider(s) serving"
        elif probing:
            status = CapabilityStatus.DEGRADED.value
            detail = (
                f"{probing} provider(s) testing recovery; most requests will be refused "
                "until a probe succeeds"
            )
        else:
            status = CapabilityStatus.UNAVAILABLE.value
            reasons = sorted(
                {view.open_reason for view in views if view.open_reason} or {"circuit open"}
            )
            detail = "; ".join(reasons)

        reports.append(
            CapabilityReadiness(
                capability=capability,
                status=status,
                providers=tuple(views),
                detail=detail,
            )
        )

    all_records = await store.all_circuits()
    servable = [item for item in reports if item.status == CapabilityStatus.AVAILABLE.value]
    configured = [item for item in reports if item.status != CapabilityStatus.NOT_CONFIGURED.value]
    available = bool(servable)
    degraded = available and len(servable) < len(configured)

    if not configured:
        summary = "no providers are configured"
    elif not available:
        summary = "no capability can be served: every provider circuit is open"
    elif degraded:
        unavailable = sorted(item.capability for item in configured if item not in servable)
        summary = "serving with reduced capability; unavailable: " + ", ".join(unavailable)
    else:
        summary = "all configured capabilities are being served"

    return ReadinessReport(
        at=moment,
        capabilities=tuple(reports),
        generation_available=available,
        degraded=degraded,
        summary=summary,
        metrics={
            "capabilities_configured": len(configured),
            "capabilities_available": len(servable),
            "circuits_open": sum(
                1 for record in all_records if record.state == CircuitState.OPEN.value
            ),
            "circuits_manual": sum(
                1 for record in all_records if record.control == ControlMode.MANUAL.value
            ),
        },
    )


__all__ = [
    "REPORTED_CAPABILITIES",
    "CapabilityReadiness",
    "CapabilityStatus",
    "ProviderView",
    "ReadinessReport",
    "readiness",
]
