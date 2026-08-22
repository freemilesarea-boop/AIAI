"""A `CircuitStore` backed by a repository, so circuits outlive a process.

The bridge between the pure state machine and the database, and the only
place in this package that knows a repository exists. It still imports
no SQLAlchemy: the repository is taken structurally, so the whole thing
can be exercised against a fake and the seam points one way.

Two translations happen here, and both are more than plumbing.

**Dataclass to mapping.** The circuit's `window` is a list of `Outcome`
objects and the row's is JSON text. Doing it here rather than in the
repository keeps `luber-database` free of any domain package, which is
the property that lets the schema move without the state machine
knowing.

**Missing row to fresh circuit.** The repository answers ``None`` for a
circuit nobody has written. Turning that into a CLOSED record is what
makes adding a provider require no registration step — and it is done
here, once, rather than at each call site that would otherwise have to
remember.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from luber_provider_resilience.circuit import (
    CircuitIdentity,
    CircuitRecord,
    Outcome,
    Transition,
)
from luber_provider_resilience.store import ConcurrentModification
from luber_provider_resilience.versions import CIRCUIT_POLICY_VERSION


class CircuitRepositoryLike(Protocol):
    """The repository methods this store uses."""

    async def load(self, circuit_key: str) -> dict[str, Any] | None: ...
    async def save(self, payload: dict[str, Any], *, expected_revision: int) -> dict[str, Any]: ...
    async def all_circuits(self) -> list[dict[str, Any]]: ...
    async def record_transition(self, payload: dict[str, Any]) -> None: ...
    async def transitions(
        self, *, circuit_key: str | None = ..., limit: int = ...
    ) -> list[dict[str, Any]]: ...


def _aware(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def record_to_payload(record: CircuitRecord) -> dict[str, Any]:
    """A circuit as a row-shaped mapping."""
    return {
        "circuit_key": record.identity.key(),
        "provider": record.identity.provider,
        "task_type": record.identity.task_type,
        "state": record.state,
        "control": record.control,
        "window": [item.to_dict() for item in record.window],
        "consecutive_failures": record.consecutive_failures,
        "consecutive_successes": record.consecutive_successes,
        "opened_at": record.opened_at,
        "open_until": record.open_until,
        "consecutive_opens": record.consecutive_opens,
        "open_reason": record.open_reason,
        "open_evidence": record.open_evidence,
        "probes": {
            token: expires.astimezone(UTC).isoformat() for token, expires in record.probes.items()
        },
        "probe_successes": record.probe_successes,
        "last_failure_at": record.last_failure_at,
        "last_failure_category": record.last_failure_category,
        "last_success_at": record.last_success_at,
        "last_transition_at": record.last_transition_at,
        "last_provider_revision": record.last_provider_revision,
        "manual_reason": record.manual_reason,
        "manual_operator": record.manual_operator,
        "manual_at": record.manual_at,
        "revision": record.revision,
        "circuit_policy_version": record.circuit_policy_version,
    }


def payload_to_record(payload: dict[str, Any]) -> CircuitRecord:
    """A row-shaped mapping as a circuit."""
    identity = CircuitIdentity(
        provider=payload["provider"], task_type=payload.get("task_type", "ANY")
    )
    window: list[Outcome] = []
    for item in payload.get("window", []) or []:
        at = _aware(item.get("at"))
        if at is None:
            continue
        window.append(
            Outcome(
                at=at,
                succeeded=bool(item.get("succeeded")),
                category=item.get("category"),
                latency_seconds=item.get("latency_seconds"),
                provider_revision=item.get("provider_revision"),
            )
        )
    probes: dict[str, datetime] = {}
    for token, expires in (payload.get("probes") or {}).items():
        moment = _aware(expires)
        if moment is not None:
            probes[token] = moment

    return CircuitRecord(
        identity=identity,
        state=payload["state"],
        control=payload["control"],
        window=window,
        consecutive_failures=payload.get("consecutive_failures", 0),
        consecutive_successes=payload.get("consecutive_successes", 0),
        opened_at=_aware(payload.get("opened_at")),
        open_until=_aware(payload.get("open_until")),
        consecutive_opens=payload.get("consecutive_opens", 0),
        open_reason=payload.get("open_reason"),
        open_evidence=payload.get("open_evidence") or {},
        probes=probes,
        probe_successes=payload.get("probe_successes", 0),
        last_failure_at=_aware(payload.get("last_failure_at")),
        last_failure_category=payload.get("last_failure_category"),
        last_success_at=_aware(payload.get("last_success_at")),
        last_transition_at=_aware(payload.get("last_transition_at")),
        last_provider_revision=payload.get("last_provider_revision"),
        manual_reason=payload.get("manual_reason"),
        manual_operator=payload.get("manual_operator"),
        manual_at=_aware(payload.get("manual_at")),
        revision=payload.get("revision", 0),
        circuit_policy_version=payload.get("circuit_policy_version", CIRCUIT_POLICY_VERSION),
    )


class DurableCircuitStore:
    """Circuit state in a database, shared by every worker."""

    def __init__(self, repository: CircuitRepositoryLike) -> None:
        self._repository = repository

    async def load(self, identity: CircuitIdentity) -> CircuitRecord:
        payload = await self._repository.load(identity.key())
        if payload is None:
            # A circuit nobody has written is closed. Lazy creation
            # means adding a provider needs no registration step, and a
            # deployment that adds one cannot forget.
            return CircuitRecord(identity=identity)
        return payload_to_record(payload)

    async def save(self, record: CircuitRecord, *, expected_revision: int) -> CircuitRecord:
        from luber_database.resilience_repository import CircuitConflict

        try:
            await self._repository.save(
                record_to_payload(record), expected_revision=expected_revision
            )
        except CircuitConflict as exc:
            # Translated into this package's own exception so callers
            # depend on one vocabulary, and so `apply_with_retry` can
            # catch it without importing the database package.
            raise ConcurrentModification(record.identity, exc.expected, exc.actual) from exc
        return record

    async def record_transition(self, transition: Transition) -> None:
        await self._repository.record_transition(
            {
                "circuit_key": transition.identity.key(),
                "provider": transition.identity.provider,
                "task_type": transition.identity.task_type,
                "previous_state": transition.previous,
                "current_state": transition.current,
                "occurred_at": transition.at,
                "reason": transition.reason,
                "automatic": transition.automatic,
                "operator": transition.operator,
                "evidence": transition.evidence,
                "circuit_policy_version": CIRCUIT_POLICY_VERSION,
            }
        )

    async def all_circuits(self) -> Sequence[CircuitRecord]:
        return [payload_to_record(item) for item in await self._repository.all_circuits()]

    async def transitions(self, *, limit: int = 50) -> Sequence[Transition]:
        out: list[Transition] = []
        for item in await self._repository.transitions(limit=limit):
            at = _aware(item.get("occurred_at"))
            if at is None:
                continue
            out.append(
                Transition(
                    identity=CircuitIdentity(item["provider"], item["task_type"]),
                    previous=item["previous_state"],
                    current=item["current_state"],
                    at=at,
                    reason=item["reason"],
                    automatic=bool(item.get("automatic", True)),
                    operator=item.get("operator"),
                    evidence=item.get("evidence") or {},
                )
            )
        return out


__all__ = [
    "CircuitRepositoryLike",
    "DurableCircuitStore",
    "payload_to_record",
    "record_to_payload",
]
