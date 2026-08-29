"""Deleting raw acquisition data once it is old enough.

A retention period a product cannot perform is a promise, not a policy.
The privacy policy says raw first-party acquisition records are kept for
twelve months; this is the code that makes that true.

**What is deleted.** Visitor and session rows whose *first* contact was
more than twelve months ago. Sessions go with their visitor via the
existing `ON DELETE CASCADE`, so a visitor is never left with orphaned
arrivals and no session outlives the identity it belongs to.

**What is never touched.** Nothing in `billing_payments`,
`subscriptions`, `billing_checkouts` or `support_tickets` — those are
commerce records that 전자상거래법 requires be kept for years, and a
marketing cleanup must not reach them. The FK from
`acquisition_attributions.visitor_id` is `ON DELETE SET NULL`, so a
customer's acquisition *snapshot* survives its visitor: the snapshot is
what the console reports on, and losing it would silently rewrite last
year's figures.

**Cutoff is on `first_seen_at`, not `last_seen_at`.** Otherwise a
visitor who returns once a year is kept forever, which is the opposite
of a retention period.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from luber_database.models.acquisition import AcquisitionSession, AcquisitionVisitor

logger = logging.getLogger(__name__)

#: How long raw acquisition data is kept. Mirrors
#: `ACQUISITION_RETENTION_MONTHS` in the web app's legal config; the two
#: are the same policy stated to two audiences.
RETENTION_DAYS = 365

#: Rows removed per pass.
#:
#: Bounded so a first run against years of history is a series of small
#: transactions rather than one that locks the table and times out. The
#: job is idempotent, so running it again finishes the work.
DEFAULT_BATCH = 500


@dataclass(frozen=True)
class PurgeReport:
    """What a run did, or would do."""

    cutoff: datetime
    visitors_deleted: int
    sessions_deleted: int
    dry_run: bool


def cutoff_for(now: datetime | None = None, *, days: int = RETENTION_DAYS) -> datetime:
    """The instant before which raw acquisition data is expired.

    Exactly `days` before now, so the boundary is a single comparison a
    test can sit either side of.
    """
    return (now or datetime.now(UTC)) - timedelta(days=days)


async def purge_acquisition(
    session: AsyncSession,
    *,
    now: datetime | None = None,
    days: int = RETENTION_DAYS,
    batch: int = DEFAULT_BATCH,
    dry_run: bool = False,
) -> PurgeReport:
    """Delete expired visitors and their sessions.

    Idempotent: a second run finds nothing and reports zero. `dry_run`
    counts without deleting, and is the only mode that should ever be
    pointed at production without reading the count first.
    """
    at = now or datetime.now(UTC)
    cutoff = cutoff_for(at, days=days)

    expired = select(AcquisitionVisitor.id).where(AcquisitionVisitor.first_seen_at < cutoff)

    if dry_run:
        visitors = int(
            (
                await session.execute(select(func.count()).select_from(expired.subquery()))
            ).scalar_one()
        )
        sessions = int(
            (
                await session.execute(
                    select(func.count(AcquisitionSession.id)).where(
                        AcquisitionSession.visitor_id.in_(expired)
                    )
                )
            ).scalar_one()
        )
        # Counts only. Nothing here writes, so a dry run against
        # production is safe by construction rather than by care.
        return PurgeReport(
            cutoff=cutoff, visitors_deleted=visitors, sessions_deleted=sessions, dry_run=True
        )

    visitors_deleted = 0
    sessions_deleted = 0
    while True:
        ids = list(
            (
                await session.execute(
                    select(AcquisitionVisitor.id)
                    .where(AcquisitionVisitor.first_seen_at < cutoff)
                    .limit(batch)
                )
            ).scalars()
        )
        if not ids:
            break

        # Counted before the delete: afterwards there is nothing to count,
        # and the cascade removes them without reporting how many.
        sessions_deleted += int(
            (
                await session.execute(
                    select(func.count(AcquisitionSession.id)).where(
                        AcquisitionSession.visitor_id.in_(ids)
                    )
                )
            ).scalar_one()
        )
        # Sessions explicitly as well as by cascade: SQLite does not
        # enforce `ON DELETE CASCADE` unless foreign keys are switched
        # on, and a retention job that silently leaves rows behind on one
        # dialect is not a retention job.
        await session.execute(
            delete(AcquisitionSession).where(AcquisitionSession.visitor_id.in_(ids))
        )
        result = await session.execute(
            delete(AcquisitionVisitor).where(AcquisitionVisitor.id.in_(ids))
        )
        visitors_deleted += int(getattr(result, "rowcount", 0) or len(ids))
        await session.commit()

    if visitors_deleted:
        # Counts only. A log line naming a campaign or a landing path
        # would put the very data being deleted into a second system.
        logger.info(
            "acquisition retention purge complete",
            extra={
                "visitors_deleted": visitors_deleted,
                "sessions_deleted": sessions_deleted,
                "retention_days": days,
            },
        )

    return PurgeReport(
        cutoff=cutoff,
        visitors_deleted=visitors_deleted,
        sessions_deleted=sessions_deleted,
        dry_run=False,
    )


__all__ = ["DEFAULT_BATCH", "RETENTION_DAYS", "PurgeReport", "cutoff_for", "purge_acquisition"]
