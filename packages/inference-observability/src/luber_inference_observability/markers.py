"""Things that changed, on the timeline, without a claim that they caused anything.

A marker is a fact with a timestamp: this commit deployed, this provider
revision appeared, this QC policy changed. Drawn on a chart it lets an
operator see that a rise began near a rollout, which is genuinely useful
and is *not* evidence that the rollout caused it. Both statements have
to survive together, and the wording throughout this module is chosen so
the second one cannot quietly fall away.

The distinction is not pedantry. The failure mode is concrete: a
deployment marker sitting next to a spike gets an engineer to roll back
a change that was innocent, while the real cause — a traffic shift, an
upstream model swap, a bad batch of long requests — keeps running.

Provider revision markers are derived rather than declared. The first
time a revision is observed *is* the moment it appeared, and that fact
needs nobody to remember to record it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from luber_inference_observability.events import InferenceObservation
from luber_inference_observability.windows import TimeWindow


class MarkerKind(StrEnum):
    DEPLOYMENT = "DEPLOYMENT"
    PROVIDER_REVISION = "PROVIDER_REVISION"
    QC_POLICY_CHANGE = "QC_POLICY_CHANGE"
    FINISHING_VERSION_CHANGE = "FINISHING_VERSION_CHANGE"


@dataclass(frozen=True)
class Marker:
    """One dated change, and what it was."""

    kind: str
    occurred_at: datetime
    label: str
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "occurred_at": self.occurred_at.astimezone(UTC).isoformat(),
            "label": self.label,
            "detail": self.detail or {},
            # Carried on every marker rather than documented once,
            # because a marker travels to places the documentation does
            # not — a chart tooltip, a JSON export, an incident detail.
            "caveat": "Correlation only. A marker near a change is not a cause.",
        }


def first_seen(
    observations: Iterable[InferenceObservation],
    *,
    field: str,
    kind: str,
) -> list[Marker]:
    """A marker at the first appearance of each distinct value of *field*.

    Derived from the data, so a revision nobody announced still gets its
    marker. That matters: the rollouts most worth spotting on a chart
    are exactly the ones nobody wrote down.
    """
    earliest: dict[str, datetime] = {}
    for observation in observations:
        value = getattr(observation, field, None)
        if not value or value == "UNKNOWN":
            continue
        moment = observation.occurred_at
        if value not in earliest or moment < earliest[value]:
            earliest[value] = moment
    return [
        Marker(
            kind=kind,
            occurred_at=moment,
            label=value,
            detail={"field": field, "first_observation": moment.astimezone(UTC).isoformat()},
        )
        for value, moment in sorted(earliest.items(), key=lambda pair: pair[1])
    ]


def derive(observations: Sequence[InferenceObservation]) -> list[Marker]:
    """Every marker that can be read out of the observations themselves."""
    markers: list[Marker] = []
    markers += first_seen(
        observations, field="provider_revision", kind=MarkerKind.PROVIDER_REVISION.value
    )
    markers += first_seen(observations, field="qc_policy", kind=MarkerKind.QC_POLICY_CHANGE.value)
    markers += first_seen(
        observations,
        field="finishing_version",
        kind=MarkerKind.FINISHING_VERSION_CHANGE.value,
    )
    markers += first_seen(observations, field="luber_revision", kind=MarkerKind.DEPLOYMENT.value)
    return sorted(markers, key=lambda item: (item.occurred_at, item.kind, item.label))


def within(markers: Iterable[Marker], window: TimeWindow) -> list[Marker]:
    return [marker for marker in markers if window.contains(marker.occurred_at)]


def cold_start(
    observations: Sequence[InferenceObservation],
    *,
    revision: str,
    minimum_samples: int,
) -> dict[str, Any]:
    """Whether a revision has enough history to be judged.

    A revision that appeared an hour ago has no baseline, and comparing
    it against the population it is replacing would attribute the
    previous revision's normal to the new one. The honest answer is that
    it is still building.
    """
    rows = [row for row in observations if row.provider_revision == revision]
    return {
        "provider_revision": revision,
        "observations": len(rows),
        "minimum_samples": minimum_samples,
        "status": "READY" if len(rows) >= minimum_samples else "BASELINE_BUILDING",
        "first_seen": min((row.occurred_at for row in rows), default=None),
    }


__all__ = ["Marker", "MarkerKind", "cold_start", "derive", "first_seen", "within"]
