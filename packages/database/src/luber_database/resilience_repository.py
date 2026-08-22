"""Circuit persistence, with the compare-and-set that makes it shared.

Not owner-scoped, and safe not to be: a circuit is about a provider name
and a task type. There is no user in it, no prompt, nothing anybody
owns.

The one interesting method is `save`. It writes only if the row still
carries the revision the caller read, and reports rather than raises
when it does not. That is what turns "several workers noticed the same
failure" into one transition instead of a race — the losing writer
re-reads, finds the circuit already open, and stops.

Async, like every other repository here. The alternative — a synchronous
repository over a second engine — would need a synchronous PostgreSQL
driver this project does not install, and would open a second connection
pool to the same database purely so one layer could avoid `await`.

Only the I/O is async. The circuit state machine stays a pure function of
state plus evidence plus a clock, which is what makes a device full of
timeouts testable without sleeping.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from luber_database.models.resilience import (
    ProviderCircuitRow,
    ProviderCircuitTransitionRow,
)

#: Columns copied straight through between row and mapping.
_SCALAR_COLUMNS = (
    "circuit_key",
    "provider",
    "task_type",
    "state",
    "control",
    "consecutive_failures",
    "consecutive_successes",
    "opened_at",
    "open_until",
    "consecutive_opens",
    "open_reason",
    "probe_successes",
    "last_failure_at",
    "last_failure_category",
    "last_success_at",
    "last_transition_at",
    "last_provider_revision",
    "manual_reason",
    "manual_operator",
    "manual_at",
    "revision",
    "circuit_policy_version",
)

#: Columns stored as JSON text.
_JSON_COLUMNS = ("window", "open_evidence", "probes")


class CircuitConflict(RuntimeError):
    """The row moved between the caller's read and its write."""

    def __init__(self, circuit_key: str, expected: int, actual: int) -> None:
        super().__init__(
            f"{circuit_key} was modified concurrently "
            f"(expected revision {expected}, found {actual})"
        )
        self.circuit_key = circuit_key
        self.expected = expected
        self.actual = actual


def _as_utc(value: Any) -> Any:
    """SQLite returns naive datetimes for timezone-aware columns.

    Normalised here rather than defended against downstream: a naive
    value reaching a circuit comparison raises "can't compare
    offset-naive and offset-aware" at routing time, in production, on
    whichever deployment happens to use SQLite.
    """
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _row_to_mapping(row: ProviderCircuitRow) -> dict[str, Any]:
    payload: dict[str, Any] = {name: _as_utc(getattr(row, name)) for name in _SCALAR_COLUMNS}
    for name in _JSON_COLUMNS:
        raw = getattr(row, name)
        try:
            payload[name] = json.loads(raw) if raw else ({} if name != "window" else [])
        except (TypeError, ValueError):
            payload[name] = {} if name != "window" else []
    return payload


class ResilienceRepository:
    """Reads and writes circuit state. No owner scoping, by design."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | AsyncEngine) -> None:
        if isinstance(session_factory, AsyncEngine):
            self._factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
                bind=session_factory, expire_on_commit=False
            )
        else:
            self._factory = session_factory

    # ── circuits ─────────────────────────────────────────────────────

    async def load(self, circuit_key: str) -> dict[str, Any] | None:
        """The row, or ``None`` when this circuit has never been written.

        ``None`` rather than a default record: the caller knows what a
        fresh circuit looks like, and inventing one here would mean two
        places deciding what "new" means.
        """
        async with self._factory() as session:
            row = await session.get(ProviderCircuitRow, circuit_key)
            return None if row is None else _row_to_mapping(row)

    async def save(self, payload: dict[str, Any], *, expected_revision: int) -> dict[str, Any]:
        """Write, but only if nobody else has since the caller read.

        A single conditional `UPDATE … WHERE circuit_key = :k AND
        revision = :expected`, and the answer is `rowcount`.

        The obvious alternative — read the row, compare the revision in
        Python, then write — is not a compare-and-set at all: two
        callers can both read the same revision, both find it matching,
        and both commit. Under real concurrency that produced several
        "the circuit opened" transitions for one opening, which is
        exactly the duplicate this column exists to prevent. The check
        has to happen inside the statement that writes.

        The insert has the same shape. A guarded `INSERT` racing another
        insert of the same key raises `IntegrityError` on the primary
        key, and that is a conflict rather than an error — somebody else
        created the circuit first.
        """
        key = payload["circuit_key"]
        values = {name: payload.get(name) for name in _SCALAR_COLUMNS}
        for name in _JSON_COLUMNS:
            values[name] = json.dumps(payload.get(name, [] if name == "window" else {}))

        async with self._factory() as session:
            if expected_revision == 0:
                # A circuit nobody has written yet. Attempt the insert
                # and let the primary key arbitrate.
                try:
                    session.add(ProviderCircuitRow(**values))
                    await session.commit()
                    return payload
                except IntegrityError:
                    await session.rollback()
                    current = await session.get(ProviderCircuitRow, key)
                    raise CircuitConflict(
                        key, expected_revision, current.revision if current else 0
                    ) from None

            updates = {name: value for name, value in values.items() if name != "circuit_key"}
            result = await session.execute(
                update(ProviderCircuitRow)
                .where(
                    ProviderCircuitRow.circuit_key == key,
                    ProviderCircuitRow.revision == expected_revision,
                )
                .values(**updates)
            )
            # `rowcount` is on CursorResult; an UPDATE always yields one.
            if getattr(result, "rowcount", 0) != 1:
                await session.rollback()
                current = await session.get(ProviderCircuitRow, key)
                raise CircuitConflict(key, expected_revision, current.revision if current else 0)
            await session.commit()
            return payload

    async def all_circuits(self) -> list[dict[str, Any]]:
        async with self._factory() as session:
            rows = await session.execute(
                select(ProviderCircuitRow).order_by(ProviderCircuitRow.circuit_key)
            )
            return [_row_to_mapping(row) for row in rows.scalars().all()]

    async def delete_circuit(self, circuit_key: str) -> bool:
        """Remove a circuit entirely.

        For a provider that has been decommissioned. Its transition
        history is deliberately left behind — the record of what a
        provider did is worth keeping after the provider is gone.
        """
        async with self._factory() as session:
            row = await session.get(ProviderCircuitRow, circuit_key)
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    # ── transitions ──────────────────────────────────────────────────

    async def record_transition(self, payload: dict[str, Any]) -> None:
        async with self._factory() as session:
            session.add(
                ProviderCircuitTransitionRow(
                    id=uuid.uuid4(),
                    circuit_key=payload["circuit_key"],
                    provider=payload["provider"],
                    task_type=payload["task_type"],
                    previous_state=payload["previous_state"],
                    current_state=payload["current_state"],
                    occurred_at=payload["occurred_at"],
                    reason=payload["reason"],
                    automatic=payload.get("automatic", True),
                    operator=payload.get("operator"),
                    evidence=json.dumps(payload.get("evidence", {}), sort_keys=True),
                    latency_seconds=payload.get("latency_seconds"),
                    circuit_policy_version=payload["circuit_policy_version"],
                )
            )
            await session.commit()

    async def transitions(
        self, *, circuit_key: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        async with self._factory() as session:
            query = select(ProviderCircuitTransitionRow).order_by(
                ProviderCircuitTransitionRow.occurred_at.desc()
            )
            if circuit_key is not None:
                query = query.where(ProviderCircuitTransitionRow.circuit_key == circuit_key)
            rows = (await session.execute(query.limit(limit))).scalars().all()
            return [
                {
                    "id": str(row.id),
                    "circuit_key": row.circuit_key,
                    "provider": row.provider,
                    "task_type": row.task_type,
                    "previous_state": row.previous_state,
                    "current_state": row.current_state,
                    "occurred_at": _as_utc(row.occurred_at),
                    "reason": row.reason,
                    "automatic": row.automatic,
                    "operator": row.operator,
                    "evidence": _load_json(row.evidence),
                    "latency_seconds": row.latency_seconds,
                    "circuit_policy_version": row.circuit_policy_version,
                }
                for row in rows
            ]

    async def count_transitions(self, *, circuit_key: str | None = None) -> int:
        from sqlalchemy import func as sa_func

        async with self._factory() as session:
            query = select(sa_func.count()).select_from(ProviderCircuitTransitionRow)
            if circuit_key is not None:
                query = query.where(ProviderCircuitTransitionRow.circuit_key == circuit_key)
            return int((await session.execute(query)).scalar_one())


def _load_json(raw: str | None) -> Any:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        return {}


__all__ = ["CircuitConflict", "ResilienceRepository"]
