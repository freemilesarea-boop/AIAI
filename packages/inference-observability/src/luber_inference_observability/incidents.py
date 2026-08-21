"""Turning a stream of findings into a small number of things to look at.

A regression detector run every five minutes produces a finding every
five minutes. Left alone that is 288 alerts a day for one problem, which
trains an operator to close them without reading — and the one that
mattered closes with the rest.

So a finding becomes an *incident*: one row per logical regression, that
accumulates evidence rather than multiplying. Three mechanisms do the
work, and each exists because of a specific failure mode.

**Fingerprint.** Two findings describe the same problem when they share
a type, a category and a segment. The fingerprint is derived from those
and nothing else — not from the value, which moves, and not from the
window, which always moves. A fingerprint including either would open a
new incident every run, which is the thing this exists to prevent.

**Recovery window.** An incident does not resolve because one run came
back clean. It resolves after the metric has been healthy for a
configured number of consecutive runs, so a regression that oscillates
around the threshold does not close and reopen all afternoon.

**Escalation is one-way within an occurrence.** Severity rises when the
evidence worsens and does not fall on a single better sample, because an
incident that downgrades itself between two alerts tells an operator the
problem went away when it did not.

An acknowledgement is a human saying "I have seen this". It does not
stop measurement: evidence keeps accumulating, and an acknowledged
incident that gets worse escalates and says so. Dismissal records a
reason and a time, and never deletes anything — the history of what was
dismissed and why is exactly what somebody needs when it comes back.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from luber_inference_observability.dimensions import Segment
from luber_inference_observability.regressions import (
    Category,
    RegressionFinding,
    Severity,
    Status,
)
from luber_inference_observability.versions import INCIDENT_POLICY_VERSION, version_block


class IncidentStatus(StrEnum):
    OPEN = "OPEN"
    #: A human has seen it. Measurement continues.
    ACKNOWLEDGED = "ACKNOWLEDGED"
    #: The metric recovered and stayed recovered.
    RESOLVED = "RESOLVED"
    #: A human decided it was not worth acting on, and said why.
    DISMISSED = "DISMISSED"

    @property
    def active(self) -> bool:
        return self in {IncidentStatus.OPEN, IncidentStatus.ACKNOWLEDGED}


#: Consecutive clean evaluations before an incident resolves.
#:
#: Three, not one. A metric sitting near its threshold crosses and
#: uncrosses with ordinary variance, and resolving on the first clean
#: run turns that into an OPEN/RESOLVED cycle that is worse than no
#: signal — an operator watching a list flicker learns to ignore it.
DEFAULT_RECOVERY_EVALUATIONS = 3

#: Consecutive regressed evaluations before an incident opens at all.
#:
#: One. Unlike recovery, opening late has a real cost: the point of the
#: system is to notice quickly. Flapping is controlled at the other end,
#: which is where it is cheap.
DEFAULT_TRIGGER_EVALUATIONS = 1


def fingerprint(finding_type: str, category: str, metric: str, segment: Segment) -> str:
    """Stable identity for a logical regression.

    Derived from what the problem *is*, never from what it currently
    measures. Including the value or the window would mint a new
    identity on every evaluation, which is precisely the duplication
    this function exists to prevent.

    The metric is part of it because a finding type can cover more than
    one: total latency and provider latency both raise
    LATENCY_REGRESSION, and they are different problems — one says the
    whole pipeline slowed, the other says the model did. Without the
    metric they would collapse into a single incident whose evidence
    alternated between two unrelated numbers.
    """
    material = "|".join(
        [
            finding_type,
            category,
            metric,
            ";".join(f"{key}={value}" for key, value in segment.filters),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


@dataclass
class IncidentEvidence:
    """One evaluation's numbers, kept for the timeline."""

    observed_at: datetime
    status: str
    severity: str
    baseline_value: float | None
    current_value: float | None
    absolute_delta: float | None
    relative_delta: float | None
    current_sample_count: int
    baseline_sample_count: int
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observed_at"] = self.observed_at.isoformat()
        return payload


@dataclass
class InferenceIncident:
    """One logical regression, over its whole life."""

    incident_id: str
    created_at: datetime
    finding_type: str
    category: str
    metric: str
    segment: Segment
    status: str = IncidentStatus.OPEN.value
    severity: str = Severity.MINOR.value
    #: The worst it has ever been. Kept separately from `severity` so a
    #: recovering incident still records how bad it got.
    peak_severity: str = Severity.MINOR.value

    provider: str | None = None
    provider_revision: str | None = None

    first_seen: datetime | None = None
    last_seen: datetime | None = None
    occurrence_count: int = 0
    #: Consecutive clean evaluations since the last regressed one.
    consecutive_clean: int = 0

    baseline_window: dict[str, Any] = field(default_factory=dict)
    current_window: dict[str, Any] = field(default_factory=dict)
    evidence: list[IncidentEvidence] = field(default_factory=list)
    recommendations: tuple[str, ...] = ()

    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    resolved_at: datetime | None = None
    dismissed_at: datetime | None = None
    dismissed_by: str | None = None
    dismissal_reason: str | None = None

    incident_policy_version: str = INCIDENT_POLICY_VERSION

    @property
    def active(self) -> bool:
        return IncidentStatus(self.status).active

    @property
    def availability(self) -> bool:
        return self.category == Category.AVAILABILITY.value

    def summary(self) -> str:
        latest = self.evidence[-1].explanation if self.evidence else ""
        return f"[{self.severity}] {self.finding_type} — {self.segment.label()}. {latest}".strip()

    def to_dict(self, *, evidence_limit: int | None = 20) -> dict[str, Any]:
        history = self.evidence if evidence_limit is None else self.evidence[-evidence_limit:]
        return {
            **version_block(),
            "incident_id": self.incident_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "severity": self.severity,
            "peak_severity": self.peak_severity,
            "finding_type": self.finding_type,
            "category": self.category,
            "metric": self.metric,
            "provider": self.provider,
            "provider_version": self.provider_revision,
            "affected_dimensions": self.segment.to_dict(),
            "segment_label": self.segment.label(),
            "baseline_window": self.baseline_window,
            "current_window": self.current_window,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "occurrence_count": self.occurrence_count,
            "consecutive_clean": self.consecutive_clean,
            "evidence": [item.to_dict() for item in history],
            "evidence_total": len(self.evidence),
            "recommendations": list(self.recommendations),
            "summary": self.summary(),
            "acknowledged_at": _iso(self.acknowledged_at),
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": _iso(self.resolved_at),
            "dismissed_at": _iso(self.dismissed_at),
            "dismissed_by": self.dismissed_by,
            "dismissal_reason": self.dismissal_reason,
        }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


_SEVERITY_ORDER = {
    Severity.INFO.value: 0,
    Severity.MINOR.value: 1,
    Severity.MAJOR.value: 2,
    Severity.CRITICAL.value: 3,
}


def worse_of(left: str, right: str) -> str:
    return left if _SEVERITY_ORDER.get(left, 0) >= _SEVERITY_ORDER.get(right, 0) else right


@dataclass(frozen=True)
class IncidentPolicy:
    """How findings become, update and close incidents."""

    recovery_evaluations: int = DEFAULT_RECOVERY_EVALUATIONS
    trigger_evaluations: int = DEFAULT_TRIGGER_EVALUATIONS
    #: How much evidence one incident keeps. Bounded because an incident
    #: open for a week at five-minute evaluations would otherwise carry
    #: two thousand rows nobody reads, in a column somebody has to load.
    evidence_limit: int = 200

    def to_dict(self) -> dict[str, Any]:
        return {
            "recovery_evaluations": self.recovery_evaluations,
            "trigger_evaluations": self.trigger_evaluations,
            "evidence_limit": self.evidence_limit,
            "incident_policy_version": INCIDENT_POLICY_VERSION,
        }


class IncidentLedger:
    """The set of incidents, updated by successive evaluations.

    Deliberately a plain in-memory structure with an explicit load and
    dump: the store decides durability, and keeping the lifecycle logic
    free of storage makes it testable by writing a sequence of findings
    and reading the resulting timeline.
    """

    def __init__(
        self,
        incidents: Iterable[InferenceIncident] = (),
        *,
        policy: IncidentPolicy | None = None,
    ) -> None:
        self.policy = policy or IncidentPolicy()
        self._by_id: dict[str, InferenceIncident] = {
            incident.incident_id: incident for incident in incidents
        }

    def __len__(self) -> int:
        return len(self._by_id)

    def all(self) -> list[InferenceIncident]:
        return sorted(
            self._by_id.values(),
            key=lambda item: item.last_seen or item.created_at,
            reverse=True,
        )

    def active(self) -> list[InferenceIncident]:
        return [item for item in self.all() if item.active]

    def get(self, incident_id: str) -> InferenceIncident | None:
        return self._by_id.get(incident_id)

    # ── the lifecycle ────────────────────────────────────────────────

    def apply(
        self, findings: Iterable[RegressionFinding], *, at: datetime
    ) -> list[InferenceIncident]:
        """Fold one evaluation's findings into the ledger.

        Returns the incidents this evaluation touched — created, updated
        or resolved — so a caller can alert on the change rather than on
        the whole ledger.
        """
        moment = at.astimezone(UTC)
        touched: list[InferenceIncident] = []
        for finding in findings:
            incident = self._apply_one(finding, moment)
            if incident is not None:
                touched.append(incident)
        return touched

    def _apply_one(self, finding: RegressionFinding, moment: datetime) -> InferenceIncident | None:
        identity = fingerprint(
            finding.finding_type, finding.category, finding.metric, finding.segment
        )
        existing = self._by_id.get(identity)

        if finding.status == Status.REGRESSED.value:
            return self._record_regression(identity, existing, finding, moment)

        # Not regressed. Only an active incident cares; a clean metric
        # with no incident is the normal state of the world and must not
        # create a record.
        if existing is None or not existing.active:
            return None
        return self._record_recovery(existing, finding, moment)

    def _record_regression(
        self,
        identity: str,
        existing: InferenceIncident | None,
        finding: RegressionFinding,
        moment: datetime,
    ) -> InferenceIncident:
        evidence = _evidence(finding, moment)

        if existing is None or existing.status in {
            IncidentStatus.RESOLVED.value,
            IncidentStatus.DISMISSED.value,
        }:
            # A resolved incident that recurs opens a *new* one rather
            # than reanimating the old. The old record is the history of
            # what happened last time, and overwriting it would destroy
            # the evidence that this is a recurrence.
            incident = InferenceIncident(
                incident_id=identity if existing is None else _recurrence_id(identity, moment),
                created_at=moment,
                finding_type=finding.finding_type,
                category=finding.category,
                metric=finding.metric,
                segment=finding.segment,
                severity=finding.severity,
                peak_severity=finding.severity,
                provider=finding.segment.to_dict().get("provider"),
                provider_revision=finding.segment.to_dict().get("provider_revision"),
                first_seen=moment,
                last_seen=moment,
                occurrence_count=1,
                baseline_window=finding.baseline_window,
                current_window=finding.current_window,
                evidence=[evidence],
                recommendations=finding.recommendations,
            )
            self._by_id[incident.incident_id] = incident
            return incident

        existing.occurrence_count += 1
        existing.last_seen = moment
        existing.current_window = finding.current_window
        existing.baseline_window = finding.baseline_window
        existing.consecutive_clean = 0
        existing.recommendations = finding.recommendations
        # Severity rises with the evidence and does not fall on a single
        # better sample: an incident that downgrades between two alerts
        # reads as "it is getting better" on one data point.
        existing.severity = worse_of(existing.severity, finding.severity)
        existing.peak_severity = worse_of(existing.peak_severity, finding.severity)
        _append(existing, evidence, self.policy.evidence_limit)
        return existing

    def _record_recovery(
        self, incident: InferenceIncident, finding: RegressionFinding, moment: datetime
    ) -> InferenceIncident | None:
        # An evaluation that could not tell — too few samples, no data —
        # is not evidence of recovery. Counting it would resolve an
        # incident because traffic went quiet, which is exactly when a
        # regression is easiest to miss.
        if finding.status != Status.NORMAL.value:
            _append(incident, _evidence(finding, moment), self.policy.evidence_limit)
            return incident

        incident.consecutive_clean += 1
        _append(incident, _evidence(finding, moment), self.policy.evidence_limit)
        if incident.consecutive_clean >= self.policy.recovery_evaluations:
            incident.status = IncidentStatus.RESOLVED.value
            incident.resolved_at = moment
        return incident

    # ── operator actions ─────────────────────────────────────────────

    def acknowledge(self, incident_id: str, *, by: str, at: datetime) -> InferenceIncident:
        """Record that a human has seen it. Measurement continues."""
        incident = self._require(incident_id)
        if incident.status == IncidentStatus.OPEN.value:
            incident.status = IncidentStatus.ACKNOWLEDGED.value
        incident.acknowledged_at = at.astimezone(UTC)
        incident.acknowledged_by = by
        return incident

    def dismiss(self, incident_id: str, *, by: str, reason: str, at: datetime) -> InferenceIncident:
        """Close it as not worth acting on, with the reason on the record."""
        if not reason.strip():
            raise ValueError(
                "a dismissal needs a reason: the record of why something was ignored is "
                "what the next person needs when it comes back"
            )
        incident = self._require(incident_id)
        incident.status = IncidentStatus.DISMISSED.value
        incident.dismissed_at = at.astimezone(UTC)
        incident.dismissed_by = by
        incident.dismissal_reason = reason.strip()
        return incident

    def _require(self, incident_id: str) -> InferenceIncident:
        incident = self._by_id.get(incident_id)
        if incident is None:
            raise LookupError(f"no incident {incident_id!r}")
        return incident


def _recurrence_id(identity: str, moment: datetime) -> str:
    """A distinct id for a recurrence of a resolved incident."""
    material = f"{identity}:{moment.isoformat()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _evidence(finding: RegressionFinding, moment: datetime) -> IncidentEvidence:
    return IncidentEvidence(
        observed_at=moment,
        status=finding.status,
        severity=finding.severity,
        baseline_value=finding.baseline_value,
        current_value=finding.current_value,
        absolute_delta=finding.absolute_delta,
        relative_delta=finding.relative_delta,
        current_sample_count=finding.current_sample_count,
        baseline_sample_count=finding.baseline_sample_count,
        explanation=finding.explain(),
    )


def _append(incident: InferenceIncident, evidence: IncidentEvidence, limit: int) -> None:
    incident.evidence.append(evidence)
    if len(incident.evidence) > limit:
        # The oldest evidence goes, not the newest: an operator opening
        # an incident wants what is happening now, and the first
        # occurrence is preserved separately in `first_seen`.
        del incident.evidence[: len(incident.evidence) - limit]


# ── alerting ─────────────────────────────────────────────────────────


class AlertReason(StrEnum):
    OPENED = "OPENED"
    ESCALATED = "ESCALATED"
    RESOLVED = "RESOLVED"


@dataclass(frozen=True)
class Alert:
    """The internal contract a future notifier would send.

    Phase 30 produces these and does nothing with them. No Slack, no
    email, no webhook: an integration is a decision about where an
    operator's attention should be interrupted, and that is a phase of
    its own. What this fixes now is the *shape*, so the notifier does not
    also have to invent one.
    """

    incident_id: str
    reason: str
    severity: str
    summary: str
    timestamp: datetime
    finding_type: str
    category: str
    segment: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "reason": self.reason,
            "severity": self.severity,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
            "finding_type": self.finding_type,
            "category": self.category,
            "segment": self.segment,
            "incident_policy_version": INCIDENT_POLICY_VERSION,
        }


def alerts_for(
    touched: Iterable[InferenceIncident],
    *,
    previous_severity: dict[str, str],
    at: datetime,
) -> list[Alert]:
    """Which of the touched incidents are worth interrupting somebody for.

    An incident alerts when it opens, when it gets worse, and when it
    resolves — and not on the hundred evaluations in between. Continuing
    to alert while nothing changes is how a channel becomes noise, and a
    noisy channel is one nobody reads on the day it matters.
    """
    out: list[Alert] = []
    moment = at.astimezone(UTC)
    for incident in touched:
        known = previous_severity.get(incident.incident_id)
        if incident.status == IncidentStatus.RESOLVED.value and known is not None:
            reason = AlertReason.RESOLVED.value
        elif known is None:
            reason = AlertReason.OPENED.value
        elif worse_of(incident.severity, known) == incident.severity and incident.severity != known:
            reason = AlertReason.ESCALATED.value
        else:
            continue
        out.append(
            Alert(
                incident_id=incident.incident_id,
                reason=reason,
                severity=incident.severity,
                summary=incident.summary(),
                timestamp=moment,
                finding_type=incident.finding_type,
                category=incident.category,
                segment=incident.segment.to_dict(),
            )
        )
    return out


__all__ = [
    "DEFAULT_RECOVERY_EVALUATIONS",
    "DEFAULT_TRIGGER_EVALUATIONS",
    "Alert",
    "AlertReason",
    "IncidentEvidence",
    "IncidentLedger",
    "IncidentPolicy",
    "IncidentStatus",
    "InferenceIncident",
    "alerts_for",
    "fingerprint",
    "worse_of",
]
