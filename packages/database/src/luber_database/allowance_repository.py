"""Plans and monthly allowance, scoped to one account.

Same discipline as `GenerationRepository`: the owner is bound when the
repository is constructed, not passed to each method, so a caller cannot
read or spend another account's allowance by forgetting an argument.

The reservation protocol is three calls:

    reserve(generation_id)   before the job is queued
    consume(generation_id)   when the generation completes
    release(generation_id)   when it fails

A reservation that is never settled holds its slot, which is the correct
bias: a generation still running has genuinely spent the allowance, and a
crashed worker's row is visible in the ledger rather than silently
refunded.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from luber_database.models.billing import (
    CONSUMED,
    RELEASED,
    RESERVED,
    AllowanceReservation,
    Subscription,
)
from luber_schemas.plans import (
    DEFAULT_PLAN,
    STANDARD_GENERATION_COST,
    Plan,
    PlanId,
    plan_for,
)

#: How many times a reservation retries when it loses a slot race. Each
#: loss means another request took the slot this one aimed at, so the
#: count is recomputed and the next slot tried. Bounded because an
#: unbounded retry under contention is a hang, not resilience.
MAX_SLOT_ATTEMPTS = 8


class AllowanceExhaustedError(RuntimeError):
    """The account has spent its allowance for the current period."""

    def __init__(self, *, limit: int, used: int) -> None:
        super().__init__(f"monthly generation limit reached ({used}/{limit})")
        self.limit = limit
        self.used = used


def month_bounds(moment: datetime) -> tuple[datetime, datetime]:
    """The calendar month containing *moment*, as UTC bounds.

    The default period for an account with no billing cycle of its own.
    Returned as explicit start/end rather than a month number so a
    subscription provider can later supply bounds this code does not
    compute.
    """
    at = moment.astimezone(UTC)
    start = at.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end


@dataclass(frozen=True)
class Entitlement:
    """What one account may do right now, and how much it has left."""

    plan: Plan
    period_start: datetime
    period_end: datetime
    used: int

    @property
    def limit(self) -> int:
        return self.plan.monthly_generation_limit

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def to_dict(self) -> dict[str, Any]:
        """The shape the browser receives.

        Deliberately narrow: a plan, a period, three counts and three
        booleans. No subscription row id, no provider fields, nothing
        about how any of it is stored.
        """
        return {
            "plan": self.plan.to_dict(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "generation_limit": self.limit,
            "generation_used": self.used,
            "generation_remaining": self.remaining,
            "download_mp3": self.plan.download_mp3,
            "download_wav": self.plan.download_wav,
            "commercial_use": self.plan.commercial_use,
        }


class AllowanceRepository:
    """Plan and allowance access for exactly one account."""

    def __init__(self, session: AsyncSession, owner: UUID) -> None:
        self._session = session
        self._owner = owner

    @property
    def owner(self) -> UUID:
        return self._owner

    # ── subscription ───────────────────────────────────────────────

    async def _subscription(self) -> Subscription | None:
        result = await self._session.execute(
            select(Subscription).where(Subscription.user_id == self._owner)
        )
        return result.scalar_one_or_none()

    async def effective_plan(self) -> Plan:
        """The plan in force. Free when there is no row."""
        row = await self._subscription()
        return plan_for(row.plan_id if row else None)

    async def set_plan(self, plan_id: PlanId, *, now: datetime | None = None) -> Subscription:
        """Assign a plan. Development and administration only.

        No product surface calls this: there is no payment provider, and
        an endpoint that let an account choose its own tier would be a
        way to take Creator for nothing.
        """
        at = now or datetime.now(UTC)
        start, end = month_bounds(at)
        row = await self._subscription()
        if row is None:
            row = Subscription(
                user_id=self._owner,
                plan_id=plan_id.value,
                status="ACTIVE",
                period_start=start,
                period_end=end,
            )
            self._session.add(row)
        else:
            row.plan_id = plan_id.value
            row.updated_at = at
            # A plan change keeps the period: switching tiers mid-month
            # must not hand out a fresh allowance.
        await self._session.commit()
        await self._session.refresh(row)
        return row

    async def current_period(self, *, now: datetime | None = None) -> tuple[datetime, datetime]:
        """The allowance window in force.

        A subscription's own bounds win when they still contain *now* —
        that is where a billing provider's cycle will land. Otherwise the
        window rolls to the calendar month containing *now*, which is
        what makes a Free account's allowance reset without anything
        having to run on a schedule.
        """
        at = (now or datetime.now(UTC)).astimezone(UTC)
        row = await self._subscription()
        if row is not None:
            start = row.period_start
            end = row.period_end
            if start.tzinfo is None:
                start = start.replace(tzinfo=UTC)
            if end.tzinfo is None:
                end = end.replace(tzinfo=UTC)
            if start <= at < end:
                return start, end
        return month_bounds(at)

    # ── usage ──────────────────────────────────────────────────────

    async def _next_slot(self, period_start: datetime) -> int:
        """The next free slot index in this period.

        One past the highest index ever issued, including released ones.
        Not `used`: a release lowers the spend but does not vacate its
        index, and reusing that index would collide with the rows above
        it forever. Spend governs the limit; the index only has to be
        unique.
        """
        result = await self._session.execute(
            select(func.max(AllowanceReservation.slot_index)).where(
                AllowanceReservation.user_id == self._owner,
                AllowanceReservation.period_start == period_start,
            )
        )
        highest = result.scalar_one()
        return 0 if highest is None else int(highest) + 1

    async def _used(self, period_start: datetime) -> int:
        """Slots spent or held in this period. RELEASED rows do not count."""
        result = await self._session.execute(
            select(func.coalesce(func.sum(AllowanceReservation.cost), 0)).where(
                AllowanceReservation.user_id == self._owner,
                AllowanceReservation.period_start == period_start,
                AllowanceReservation.state.in_((RESERVED, CONSUMED)),
            )
        )
        return int(result.scalar_one() or 0)

    async def entitlement(self, *, now: datetime | None = None) -> Entitlement:
        plan = await self.effective_plan()
        start, end = await self.current_period(now=now)
        return Entitlement(
            plan=plan, period_start=start, period_end=end, used=await self._used(start)
        )

    # ── reservation protocol ───────────────────────────────────────

    async def reserve(
        self,
        generation_id: UUID,
        *,
        cost: int = STANDARD_GENERATION_COST,
        now: datetime | None = None,
    ) -> AllowanceReservation:
        """Take a slot for a generation, or refuse.

        Raises :class:`AllowanceExhaustedError` when the period is spent.

        Concurrency is handled by the unique constraint rather than by a
        lock: the row claims a specific slot index, and a request that
        loses that slot to a competitor recomputes and tries the next
        one. Ten simultaneous requests against one remaining slot produce
        one winner and nine refusals, on PostgreSQL and on SQLite alike.
        """
        plan = await self.effective_plan()
        start, end = await self.current_period(now=now)
        limit = plan.monthly_generation_limit

        # Re-reserving the same generation is not a second charge.
        existing = await self._session.execute(
            select(AllowanceReservation).where(AllowanceReservation.generation_id == generation_id)
        )
        held = existing.scalar_one_or_none()
        if held is not None:
            return held

        for _ in range(MAX_SLOT_ATTEMPTS):
            used = await self._used(start)
            if used + cost > limit:
                raise AllowanceExhaustedError(limit=limit, used=used)
            slot = await self._next_slot(start)

            row = AllowanceReservation(
                user_id=self._owner,
                generation_id=generation_id,
                plan_id=plan.plan_id.value,
                period_start=start,
                period_end=end,
                slot_index=slot,
                cost=cost,
                state=RESERVED,
            )
            self._session.add(row)
            try:
                await self._session.commit()
            except IntegrityError:
                # Someone else took this slot. Recount and try the next.
                await self._session.rollback()
                continue
            # Re-read rather than refresh: a previous attempt in this loop
            # may have rolled back and detached instances, and refresh on a
            # detached row raises instead of returning the row that landed.
            stored = await self._session.execute(
                select(AllowanceReservation).where(
                    AllowanceReservation.generation_id == generation_id
                )
            )
            return stored.scalar_one()

        # Sustained contention rather than an exhausted account: report
        # it as exhausted anyway, because the caller's only safe move is
        # to stop, and a partial count is the honest number to show.
        raise AllowanceExhaustedError(limit=limit, used=await self._used(start))

    async def _settle(self, generation_id: UUID, state: str) -> bool:
        result = await self._session.execute(
            update(AllowanceReservation)
            .where(
                AllowanceReservation.generation_id == generation_id,
                AllowanceReservation.state == RESERVED,
            )
            .values(state=state, settled_at=datetime.now(UTC))
        )
        await self._session.commit()
        # `rowcount` is on the CursorResult an UPDATE returns; the
        # generic `Result` type that `execute` is annotated with does not
        # advertise it, hence the narrowing rather than an ignore.
        settled = cast("CursorResult[Any]", result).rowcount
        return bool(settled)

    async def consume(self, generation_id: UUID) -> bool:
        """Mark a reservation spent. Called when a generation completes."""
        return await self._settle(generation_id, CONSUMED)

    async def release(self, generation_id: UUID) -> bool:
        """Return a reservation. Called when a generation fails.

        Idempotent: settling an already-settled row changes nothing, so a
        retry or a duplicate failure callback cannot refund twice.
        """
        return await self._settle(generation_id, RELEASED)

    async def reservations_in_period(
        self, *, now: datetime | None = None
    ) -> list[AllowanceReservation]:
        """The ledger for this period — which generations spent what."""
        start, _ = await self.current_period(now=now)
        result = await self._session.execute(
            select(AllowanceReservation)
            .where(
                AllowanceReservation.user_id == self._owner,
                AllowanceReservation.period_start == start,
            )
            .order_by(AllowanceReservation.slot_index)
        )
        return list(result.scalars().all())


def unscoped_allowance(session: AsyncSession, user_id: uuid.UUID) -> AllowanceRepository:
    """For the worker, which settles a reservation the API already made."""
    return AllowanceRepository(session, user_id)


__all__ = [
    "DEFAULT_PLAN",
    "MAX_SLOT_ATTEMPTS",
    "AllowanceExhaustedError",
    "AllowanceRepository",
    "Entitlement",
    "month_bounds",
    "unscoped_allowance",
]
