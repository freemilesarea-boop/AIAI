"""Reading and writing the Phase 30 projection.

Separate from `GenerationRepository` for one reason that matters:
`GenerationRepository` is owner-scoped, because every generation belongs
to somebody and a query that forgot the owner would leak one user's
music to another. This repository is deliberately *not* owner-scoped —
and it is safe for it not to be, because the rows it touches contain no
prompt, no lyrics, no title and no user id. Nothing here can leak
anything, because nothing here holds anything.

Keeping them apart makes that argument checkable. A single repository
with some owner-scoped methods and some not is one refactor away from a
method losing its scope quietly.

Rows go in and out as plain mappings. The observability package owns the
dataclasses and knows nothing about SQLAlchemy; this module owns the
tables and knows nothing about aggregation. Neither imports the other.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import Select, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from luber_database.models.generation import Generation
from luber_database.models.observability import (
    InferenceIncidentRow,
    InferenceObservationRow,
)

#: Columns an observation row carries, so the writer can build one from a
#: mapping without listing them twice.
_OBSERVATION_COLUMNS = tuple(column.name for column in InferenceObservationRow.__table__.columns)
_INCIDENT_COLUMNS = tuple(column.name for column in InferenceIncidentRow.__table__.columns)


def _row_to_mapping(row: Any, columns: tuple[str, ...]) -> dict[str, Any]:
    return {name: getattr(row, name) for name in columns}


class ObservabilityRepository:
    """Persistence for observations and incidents. No owner scoping, by design."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── observations ─────────────────────────────────────────────────

    async def upsert_observations(self, rows: Iterable[dict[str, Any]]) -> int:
        """Write observations, replacing any that already exist.

        Keyed on `generation_id`, so ingesting the same generation twice
        updates one row. That is what makes both backfill and incremental
        ingest idempotent without either of them having to check first.

        Deliberately a read-then-write rather than a dialect-specific
        upsert: this runs on PostgreSQL in production and SQLite in every
        test, and one code path that behaves identically on both is worth
        more than the round trip it costs at these volumes.
        """
        written = 0
        for payload in rows:
            values = {key: value for key, value in payload.items() if key in _OBSERVATION_COLUMNS}
            identifier = values.get("generation_id")
            if identifier is None:
                raise ValueError("an observation needs a generation_id")
            if isinstance(identifier, str):
                identifier = uuid.UUID(identifier)
                values["generation_id"] = identifier

            existing = await self._session.get(InferenceObservationRow, identifier)
            if existing is None:
                self._session.add(InferenceObservationRow(**values))
            else:
                for key, value in values.items():
                    if key != "generation_id":
                        setattr(existing, key, value)
            written += 1
        await self._session.commit()
        return written

    async def select_observations(
        self,
        *,
        start: datetime,
        end: datetime,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Observations in `[start, end)`, optionally filtered.

        Half-open so adjacent windows tile without double counting the
        instant they share.
        """
        query = self._window_query(start, end, filters)
        query = query.order_by(InferenceObservationRow.occurred_at)
        if offset:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)
        result = await self._session.execute(query)
        return [_row_to_mapping(row, _OBSERVATION_COLUMNS) for row in result.scalars().all()]

    async def count_observations(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        filters: dict[str, Any] | None = None,
    ) -> int:
        query = select(func.count()).select_from(InferenceObservationRow)
        if start is not None:
            query = query.where(InferenceObservationRow.occurred_at >= start)
        if end is not None:
            query = query.where(InferenceObservationRow.occurred_at < end)
        for name, value in (filters or {}).items():
            column = getattr(InferenceObservationRow, name, None)
            if column is not None:
                query = query.where(column == value)
        return int((await self._session.execute(query)).scalar_one())

    async def get_observation(self, generation_id: uuid.UUID) -> dict[str, Any] | None:
        row = await self._session.get(InferenceObservationRow, generation_id)
        return None if row is None else _row_to_mapping(row, _OBSERVATION_COLUMNS)

    async def latest_observed_at(self) -> datetime | None:
        """The most recent generation this projection knows about.

        The ingestion watermark. Incremental ingest asks for generations
        finished since this, which is what keeps a scheduled run from
        rescanning the whole table.
        """
        query = select(func.max(InferenceObservationRow.occurred_at))
        return (await self._session.execute(query)).scalar_one_or_none()

    async def observed_ids(self, ids: Sequence[uuid.UUID]) -> set[uuid.UUID]:
        """Which of these generations already have an observation."""
        if not ids:
            return set()
        query = select(InferenceObservationRow.generation_id).where(
            InferenceObservationRow.generation_id.in_(ids)
        )
        return set((await self._session.execute(query)).scalars().all())

    def _window_query(
        self, start: datetime, end: datetime, filters: dict[str, Any] | None
    ) -> Select[tuple[InferenceObservationRow]]:
        query = select(InferenceObservationRow).where(
            InferenceObservationRow.occurred_at >= start,
            InferenceObservationRow.occurred_at < end,
        )
        for name, value in (filters or {}).items():
            column = getattr(InferenceObservationRow, name, None)
            if column is None:
                # An unknown filter is refused rather than ignored. A
                # silently dropped filter returns more data than the
                # caller asked for, which for analytics means a wrong
                # number rather than an error.
                raise ValueError(f"{name!r} is not an observation column")
            query = query.where(column == value)
        return query

    # ── the source side of ingestion ─────────────────────────────────

    async def generations_to_ingest(
        self,
        *,
        since: datetime | None = None,
        limit: int = 500,
        statuses: Sequence[str] = ("COMPLETED", "FAILED", "CANCELLED"),
    ) -> list[Generation]:
        """Terminal generations worth projecting, oldest first.

        Failures and cancellations are included deliberately. A system
        that only observed successes would report perfect health during
        an outage, because the only rows it counted were the ones that
        worked.
        """
        query = (
            select(Generation)
            .where(Generation.status.in_(statuses))
            .order_by(Generation.created_at)
            .limit(limit)
        )
        if since is not None:
            # `created_at` rather than `completed_at`: it is indexed, it
            # is never null, and the watermark compares against
            # `occurred_at`, which is the generation's start.
            query = query.where(Generation.created_at >= since)
        return list((await self._session.execute(query)).scalars().all())

    # ── incidents ────────────────────────────────────────────────────

    async def upsert_incidents(self, rows: Iterable[dict[str, Any]]) -> int:
        written = 0
        for payload in rows:
            values = {key: value for key, value in payload.items() if key in _INCIDENT_COLUMNS}
            identifier = values.get("incident_id")
            if identifier is None:
                raise ValueError("an incident needs an incident_id")
            existing = await self._session.get(InferenceIncidentRow, identifier)
            if existing is None:
                self._session.add(InferenceIncidentRow(**values))
            else:
                for key, value in values.items():
                    if key != "incident_id":
                        setattr(existing, key, value)
            written += 1
        await self._session.commit()
        return written

    async def list_incidents(
        self,
        *,
        statuses: Sequence[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = select(InferenceIncidentRow).order_by(
            InferenceIncidentRow.last_seen.desc().nullslast(),
            InferenceIncidentRow.created_at.desc(),
        )
        if statuses:
            query = query.where(InferenceIncidentRow.status.in_(statuses))
        result = await self._session.execute(query.offset(offset).limit(limit))
        return [_row_to_mapping(row, _INCIDENT_COLUMNS) for row in result.scalars().all()]

    async def count_incidents(self, *, statuses: Sequence[str] | None = None) -> int:
        query = select(func.count()).select_from(InferenceIncidentRow)
        if statuses:
            query = query.where(InferenceIncidentRow.status.in_(statuses))
        return int((await self._session.execute(query)).scalar_one())

    async def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        row = await self._session.get(InferenceIncidentRow, incident_id)
        return None if row is None else _row_to_mapping(row, _INCIDENT_COLUMNS)

    async def all_incidents(self) -> list[dict[str, Any]]:
        result = await self._session.execute(select(InferenceIncidentRow))
        return [_row_to_mapping(row, _INCIDENT_COLUMNS) for row in result.scalars().all()]

    async def delete_observations(self, *, before: datetime) -> int:
        """Retention, run explicitly and never on a schedule in this phase.

        Exists so a retention policy has something to call. Nothing
        invokes it automatically: deleting analytics history is a
        decision with consequences an operator should make deliberately,
        and a background job that quietly pruned last quarter would make
        a year-on-year comparison impossible without anybody noticing.
        """
        result = await self._session.execute(
            delete(InferenceObservationRow).where(InferenceObservationRow.occurred_at < before)
        )
        await self._session.commit()
        return int(getattr(result, "rowcount", 0) or 0)


__all__ = ["ObservabilityRepository"]
