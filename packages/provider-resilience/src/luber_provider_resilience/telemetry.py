"""Counters and advisory alerts, and what each is for.

Two audiences. The counters answer "how often" for somebody looking at a
week; the alerts answer "something changed, now" for somebody who needs
interrupting.

Neither of them acts. An alert here is a record with a shape, produced
and returned; nothing in this package sends anything anywhere. Phase 30
made the same decision about its own alerts and for the same reason —
where an operator's attention gets interrupted is a decision with its
own consequences, and it belongs to whoever owns the on-call rota rather
than to the code that noticed.

The counters are deliberately about *resilience actions*, not about
generation outcomes. Phase 30 already counts failures, latencies and
retries from its own projection, and a second implementation of "failure
rate" would eventually disagree with the first. What is counted here is
what only this layer knows: circuits opening, requests refused because
one was open, failovers taken and how they ended.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from luber_provider_resilience.circuit import CircuitIdentity, Transition
from luber_provider_resilience.versions import version_block


class Metric(StrEnum):
    """Every counter this layer keeps. The names are the contract."""

    CIRCUIT_OPEN_TOTAL = "circuit_open_total"
    CIRCUIT_HALF_OPEN_TOTAL = "circuit_half_open_total"
    CIRCUIT_CLOSE_TOTAL = "circuit_close_total"
    REQUESTS_REJECTED_CIRCUIT_OPEN = "requests_rejected_circuit_open"
    PROVIDER_FAILOVER_TOTAL = "provider_failover_total"
    PROVIDER_FAILOVER_SUCCESS = "provider_failover_success"
    PROVIDER_FAILOVER_FAILURE = "provider_failover_failure"
    DEGRADED_MODE_REQUESTS = "degraded_mode_requests"
    PROBE_ADMITTED_TOTAL = "probe_admitted_total"
    PROBE_REFUSED_TOTAL = "probe_refused_total"


class Counters:
    """In-process counters, per circuit identity and in total.

    In-process because these are operational rates, not accounting. A
    worker restart resets them, and that is acceptable: the durable
    record of what happened is the transition log and Phase 30's
    projection. Making these durable would mean a write per refused
    request, which is a lot of writes for a number nobody reconciles.
    """

    def __init__(self) -> None:
        self._totals: dict[str, int] = {}
        self._by_circuit: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def increment(
        self, metric: str, *, identity: CircuitIdentity | None = None, amount: int = 1
    ) -> None:
        with self._lock:
            self._totals[metric] = self._totals.get(metric, 0) + amount
            if identity is not None:
                bucket = self._by_circuit.setdefault(identity.key(), {})
                bucket[metric] = bucket.get(metric, 0) + amount

    def total(self, metric: str) -> int:
        with self._lock:
            return self._totals.get(metric, 0)

    def for_circuit(self, identity: CircuitIdentity) -> dict[str, int]:
        with self._lock:
            return dict(self._by_circuit.get(identity.key(), {}))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                **version_block(),
                "totals": dict(sorted(self._totals.items())),
                "by_circuit": {
                    key: dict(sorted(value.items()))
                    for key, value in sorted(self._by_circuit.items())
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._totals.clear()
            self._by_circuit.clear()


class AlertKind(StrEnum):
    """Things worth interrupting somebody for."""

    CIRCUIT_OPENED = "CIRCUIT_OPENED"
    CIRCUIT_RECOVERED = "CIRCUIT_RECOVERED"
    #: A failover happened and the fallback failed too. Worth knowing
    #: separately: it means the alternative is not one.
    FAILOVER_FAILED = "FAILOVER_FAILED"
    #: Nothing can serve a capability. The most serious one here.
    ALL_PROVIDERS_UNAVAILABLE = "ALL_PROVIDERS_UNAVAILABLE"


class AlertSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class ResilienceAlert:
    """The internal contract a future notifier would send.

    Fixed now so the notifier does not also have to invent a shape. Kept
    deliberately close to Phase 30's `Alert`: an operator receiving both
    should not have to learn two vocabularies for "something changed".
    """

    kind: str
    severity: str
    summary: str
    at: datetime
    identity: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **version_block(),
            "kind": self.kind,
            "severity": self.severity,
            "summary": self.summary,
            "at": self.at.isoformat(),
            "identity": self.identity,
            "evidence": self.evidence,
        }


def alert_for_transition(transition: Transition) -> ResilienceAlert | None:
    """The alert a state change deserves, if any.

    HALF_OPEN produces nothing. It is the system testing a hypothesis,
    and interrupting somebody every thirty seconds while a circuit
    probes its way back is how a channel becomes noise.
    """
    from luber_provider_resilience.circuit import CircuitState

    if transition.current == CircuitState.OPEN.value:
        return ResilienceAlert(
            kind=AlertKind.CIRCUIT_OPENED.value,
            severity=(
                AlertSeverity.WARNING.value if transition.automatic else AlertSeverity.INFO.value
            ),
            summary=f"{transition.identity.label()} circuit opened: {transition.reason}",
            at=transition.at,
            identity=transition.identity.to_dict(),
            evidence=transition.evidence,
        )
    if (
        transition.current == CircuitState.CLOSED.value
        and transition.previous == CircuitState.HALF_OPEN.value
    ):
        return ResilienceAlert(
            kind=AlertKind.CIRCUIT_RECOVERED.value,
            severity=AlertSeverity.INFO.value,
            summary=f"{transition.identity.label()} recovered: {transition.reason}",
            at=transition.at,
            identity=transition.identity.to_dict(),
            evidence=transition.evidence,
        )
    return None


def alert_for_readiness(report: Any) -> ResilienceAlert | None:
    """CRITICAL when nothing can be generated at all.

    Only that. A single degraded capability is on the dashboard and in
    the counters; waking somebody for it would spend the channel's
    credibility on a service that is still working.
    """
    if getattr(report, "generation_available", True):
        return None
    return ResilienceAlert(
        kind=AlertKind.ALL_PROVIDERS_UNAVAILABLE.value,
        severity=AlertSeverity.CRITICAL.value,
        summary="no provider can serve any capability",
        at=report.at,
        evidence={"summary": report.summary},
    )


def alerts_for(
    transitions: Iterable[Transition], *, readiness_report: Any | None = None
) -> list[ResilienceAlert]:
    """Everything worth sending from one evaluation."""
    out = [
        alert for alert in (alert_for_transition(item) for item in transitions) if alert is not None
    ]
    if readiness_report is not None:
        readiness_alert = alert_for_readiness(readiness_report)
        if readiness_alert is not None:
            out.append(readiness_alert)
    return out


__all__ = [
    "AlertKind",
    "AlertSeverity",
    "Counters",
    "Metric",
    "ResilienceAlert",
    "alert_for_readiness",
    "alert_for_transition",
    "alerts_for",
]
