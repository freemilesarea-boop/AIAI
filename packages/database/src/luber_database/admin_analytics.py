"""Aggregates for the operator console, computed in the database.

Two rules run through this module.

**Nothing is counted in Python.** Every figure here is a `GROUP BY` or a
`count()`. Pulling payment rows to sum them in the application works
until the day it does not, and the day it does not is the day revenue
matters most. The console must never be the reason production is slow.

**Days are Korean days.** Timestamps are stored in UTC and stay that
way; only the bucket boundary is shifted. A payment at 08:00 KST on the
28th is 23:00 UTC on the 27th, so bucketing on the raw UTC date would
file a morning's revenue under yesterday — and the operator comparing
the dashboard against a bank statement would find them disagreeing by a
day, with no obvious reason why.

The shift is done in SQL rather than by fetching rows, which means it
has to be expressed for both dialects the project runs on: PostgreSQL in
production, SQLite in the tests. `_KstDate` below compiles to the right
thing for each, so the tests exercise the same boundary logic that
production uses rather than a Python approximation of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

from sqlalchemy import ColumnElement, Select, and_, case, distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import FunctionElement

from luber_database.models.admin import DownloadEvent
from luber_database.models.billing import Subscription
from luber_database.models.generation import Generation
from luber_database.models.payments import PAYMENT_SUCCEEDED, BillingPayment
from luber_database.models.support import SupportTicket
from luber_database.models.user import User
from luber_schemas.enums import SupportStatus
from luber_schemas.plans import PLAN_ORDER, PlanId

#: BOORDA operates from Korea, and the operator reads these numbers
#: against Korean days. Fixed offset: KST has no daylight saving, so a
#: constant is exact rather than an approximation.
KST_OFFSET = timedelta(hours=9)
KST_OFFSET_SQL = "+9 hours"

Granularity = Literal["day", "week", "month", "year"]


class _KstDate(FunctionElement[Any]):
    """The Korean calendar day a UTC timestamp falls on.

    A function element rather than a helper that builds a string, so it
    composes into `GROUP BY` and `WHERE` like any other expression and
    each dialect renders its own correct form.
    """

    inherit_cache = True
    name = "kst_date"


@compiles(_KstDate)
def _kst_date_default(element: _KstDate, compiler: Any, **kw: Any) -> str:
    # SQLite (and anything else the tests reach for): shift, then take
    # the date part.
    inner = compiler.process(next(iter(element.clauses)), **kw)
    return f"date({inner}, '{KST_OFFSET_SQL}')"


@compiles(_KstDate, "postgresql")
def _kst_date_postgresql(element: _KstDate, compiler: Any, **kw: Any) -> str:
    # `AT TIME ZONE` converts the stored instant into Korean wall-clock
    # time; the cast then takes the calendar day of that.
    inner = compiler.process(next(iter(element.clauses)), **kw)
    return f"(({inner}) AT TIME ZONE 'Asia/Seoul')::date"


def kst_date(column: Any) -> _KstDate:
    """The Korean day of a timestamp column.

    Typed loosely on purpose: callers pass mapped attributes
    (`BillingPayment.paid_at`), which are `InstrumentedAttribute` rather
    than `ColumnElement`, and narrowing the annotation would only be
    accurate about a distinction SQLAlchemy itself does not make here.
    """
    return _KstDate(column)


def kst_day_bounds(day: date) -> tuple[datetime, datetime]:
    """The UTC instants a Korean calendar day starts and ends at.

    Used for range filters, where a half-open interval on the raw column
    lets the index do the work — no function on the left-hand side.
    """
    start = datetime(day.year, day.month, day.day, tzinfo=UTC) - KST_OFFSET
    return start, start + timedelta(days=1)


def kst_today(now: datetime | None = None) -> date:
    return ((now or datetime.now(UTC)).astimezone(UTC) + KST_OFFSET).date()


def period_bounds(
    granularity: Granularity, now: datetime | None = None
) -> tuple[datetime, datetime]:
    """The UTC window covering the current Korean day/week/month/year.

    Weeks start on Monday, which is what a Korean operator means by
    "이번 주".
    """
    today = kst_today(now)
    if granularity == "day":
        start_day = today
    elif granularity == "week":
        start_day = today - timedelta(days=today.weekday())
    elif granularity == "month":
        start_day = today.replace(day=1)
    else:
        start_day = today.replace(month=1, day=1)
    start, _ = kst_day_bounds(start_day)
    _, end = kst_day_bounds(today)
    return start, end


@dataclass(frozen=True)
class Bucket:
    """One point on a chart."""

    day: str
    value: int
    #: A second series where the chart shows two — payment count beside
    #: revenue, failures beside successes.
    secondary: int = 0


def _succeeded() -> ColumnElement[bool]:
    return BillingPayment.status == PAYMENT_SUCCEEDED


async def _scalar(session: AsyncSession, statement: Select[Any]) -> int:
    return int((await session.execute(statement)).scalar_one() or 0)


# ── revenue ──────────────────────────────────────────────────────────


async def revenue_total(
    session: AsyncSession, *, start: datetime, end: datetime
) -> tuple[int, int]:
    """Won and payment count in a window, successful payments only.

    A checkout is not revenue and a failed charge is not revenue. The
    only rows that count are the ones a verified PayApp notification
    wrote as SUCCEEDED — see `docs/PAYAPP_BILLING.md`.
    """
    row = (
        await session.execute(
            select(
                func.coalesce(func.sum(BillingPayment.amount_krw), 0),
                func.count(BillingPayment.id),
            ).where(_succeeded(), BillingPayment.paid_at >= start, BillingPayment.paid_at < end)
        )
    ).one()
    return int(row[0] or 0), int(row[1] or 0)


async def revenue_series(session: AsyncSession, *, start: datetime, end: datetime) -> list[Bucket]:
    """Revenue per Korean day, with the payment count alongside."""
    rows = (
        await session.execute(
            select(
                kst_date(BillingPayment.paid_at).label("day"),
                func.coalesce(func.sum(BillingPayment.amount_krw), 0),
                func.count(BillingPayment.id),
            )
            .where(_succeeded(), BillingPayment.paid_at >= start, BillingPayment.paid_at < end)
            .group_by(kst_date(BillingPayment.paid_at))
            .order_by(kst_date(BillingPayment.paid_at))
        )
    ).all()
    return [Bucket(day=str(r[0]), value=int(r[1] or 0), secondary=int(r[2] or 0)) for r in rows]


async def revenue_split(session: AsyncSession, *, start: datetime, end: datetime) -> dict[str, int]:
    """New versus renewal revenue in a window, in won and in count.

    A payment is "new" when it is the first successful one for its
    subscription. Computed as a correlated count rather than a flag on
    the row, because nothing in the billing path records which it was —
    and inventing a column would mean backfilling a guess for every
    payment already taken.

    A payment with no subscription (an operator comp, a one-off) has no
    earlier sibling and therefore counts as new, which is what it is.
    """
    # Aliased exactly once. Calling `.alias("prior")` per reference makes
    # four different FROM entries that happen to share a name, which
    # SQLite rejects as ambiguous and PostgreSQL would answer wrongly.
    prior = BillingPayment.__table__.alias("prior")
    earlier = (
        select(func.count())
        .select_from(prior)
        .where(
            and_(
                prior.c.subscription_id == BillingPayment.subscription_id,
                prior.c.status == PAYMENT_SUCCEEDED,
                prior.c.paid_at < BillingPayment.paid_at,
            )
        )
        .scalar_subquery()
    )
    rows = (
        await session.execute(
            select(
                earlier.label("prior_count"),
                func.coalesce(func.sum(BillingPayment.amount_krw), 0),
                func.count(BillingPayment.id),
            )
            .where(_succeeded(), BillingPayment.paid_at >= start, BillingPayment.paid_at < end)
            .group_by(earlier)
        )
    ).all()

    first_time = [r for r in rows if int(r[0] or 0) == 0]
    repeat = [r for r in rows if int(r[0] or 0) > 0]
    return {
        "new_krw": sum(int(r[1] or 0) for r in first_time),
        "new_count": sum(int(r[2]) for r in first_time),
        "renewal_krw": sum(int(r[1] or 0) for r in repeat),
        "renewal_count": sum(int(r[2]) for r in repeat),
    }


# ── users ────────────────────────────────────────────────────────────


def _live_user() -> ColumnElement[bool]:
    """Closed accounts are anonymised, not deleted — they must not be
    counted as members."""
    return User.deleted_at.is_(None)


async def user_totals(session: AsyncSession, *, now: datetime | None = None) -> dict[str, int]:
    at = now or datetime.now(UTC)
    total = await _scalar(session, select(func.count(User.id)).where(_live_user()))

    paid = await _scalar(
        session,
        select(func.count(distinct(Subscription.user_id))).where(
            Subscription.status.in_(("ACTIVE", "CANCEL_PENDING")),
            Subscription.period_start <= at,
            Subscription.period_end > at,
            Subscription.plan_id != PlanId.FREE.value,
        ),
    )
    return {"total": total, "paid": paid, "free": max(0, total - paid)}


async def new_users(session: AsyncSession, *, start: datetime, end: datetime) -> int:
    return await _scalar(
        session,
        select(func.count(User.id)).where(
            _live_user(), User.created_at >= start, User.created_at < end
        ),
    )


async def user_series(session: AsyncSession, *, start: datetime, end: datetime) -> list[Bucket]:
    rows = (
        await session.execute(
            select(kst_date(User.created_at), func.count(User.id))
            .where(_live_user(), User.created_at >= start, User.created_at < end)
            .group_by(kst_date(User.created_at))
            .order_by(kst_date(User.created_at))
        )
    ).all()
    return [Bucket(day=str(r[0]), value=int(r[1])) for r in rows]


async def plan_distribution(
    session: AsyncSession, *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """How many accounts sit on each tier right now.

    Free is derived rather than counted: an account with no live paid
    subscription is on Free, and that is one query fewer than asking the
    subscription table about accounts that have no row in it.
    """
    at = now or datetime.now(UTC)
    rows = (
        await session.execute(
            select(Subscription.plan_id, func.count(distinct(Subscription.user_id)))
            .where(
                Subscription.status.in_(("ACTIVE", "CANCEL_PENDING")),
                Subscription.period_start <= at,
                Subscription.period_end > at,
            )
            .group_by(Subscription.plan_id)
        )
    ).all()
    paid_counts = {str(r[0]): int(r[1]) for r in rows if str(r[0]) != PlanId.FREE.value}

    total = await _scalar(session, select(func.count(User.id)).where(_live_user()))
    counts = {PlanId.FREE.value: max(0, total - sum(paid_counts.values())), **paid_counts}

    return [
        {
            "plan_id": plan.value,
            "count": counts.get(plan.value, 0),
            "share": round(counts.get(plan.value, 0) / total, 4) if total else 0.0,
        }
        for plan in PLAN_ORDER
    ]


# ── generations ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenerationTotals:
    """Generation volume in a window.

    A dataclass rather than a dict because the average is a float among
    counts, and a `dict[str, int]` that quietly holds one float is the
    kind of thing that type-checks until it does not.
    """

    requested: int
    completed: int
    failed: int
    creators: int
    average_per_creator: float


async def generation_totals(
    session: AsyncSession, *, start: datetime, end: datetime
) -> GenerationTotals:
    """Requested, completed and failed in a window.

    Zero across the board is a correct answer while generation is
    switched off, and every caller must render it as one rather than as
    a missing chart.
    """
    window = and_(Generation.created_at >= start, Generation.created_at < end)
    requested = await _scalar(session, select(func.count(Generation.id)).where(window))
    completed = await _scalar(
        session, select(func.count(Generation.id)).where(window, Generation.status == "COMPLETED")
    )
    failed = await _scalar(
        session,
        select(func.count(Generation.id)).where(
            window, Generation.status.in_(("FAILED", "CANCELLED"))
        ),
    )
    creators = await _scalar(
        session, select(func.count(distinct(Generation.user_id))).where(window)
    )
    return GenerationTotals(
        requested=requested,
        completed=completed,
        failed=failed,
        creators=creators,
        # Per creator who actually generated, so an idle month does not
        # divide by the whole membership and report a misleading zero.
        average_per_creator=round(requested / creators, 2) if creators else 0.0,
    )


async def generation_series(
    session: AsyncSession, *, start: datetime, end: datetime
) -> list[Bucket]:
    """Requested per Korean day, completed alongside."""
    # `case`, not `func.case`: the latter builds a SQL function literally
    # named "case" and does not accept `else_` at all.
    completed = func.sum(case((Generation.status == "COMPLETED", 1), else_=0))
    rows = (
        await session.execute(
            select(kst_date(Generation.created_at), func.count(Generation.id), completed)
            .where(Generation.created_at >= start, Generation.created_at < end)
            .group_by(kst_date(Generation.created_at))
            .order_by(kst_date(Generation.created_at))
        )
    ).all()
    return [Bucket(day=str(r[0]), value=int(r[1]), secondary=int(r[2] or 0)) for r in rows]


# ── downloads ────────────────────────────────────────────────────────


async def download_total(session: AsyncSession, *, start: datetime, end: datetime) -> int:
    return await _scalar(
        session,
        select(func.count(DownloadEvent.id)).where(
            DownloadEvent.created_at >= start, DownloadEvent.created_at < end
        ),
    )


async def download_series(session: AsyncSession, *, start: datetime, end: datetime) -> list[Bucket]:
    rows = (
        await session.execute(
            select(kst_date(DownloadEvent.created_at), func.count(DownloadEvent.id))
            .where(DownloadEvent.created_at >= start, DownloadEvent.created_at < end)
            .group_by(kst_date(DownloadEvent.created_at))
            .order_by(kst_date(DownloadEvent.created_at))
        )
    ).all()
    return [Bucket(day=str(r[0]), value=int(r[1])) for r in rows]


# ── support ──────────────────────────────────────────────────────────


async def support_counts(session: AsyncSession) -> dict[str, int]:
    rows = (
        await session.execute(
            select(SupportTicket.status, func.count(SupportTicket.id)).group_by(
                SupportTicket.status
            )
        )
    ).all()
    counts = {str(r[0]): int(r[1]) for r in rows}
    return {status.value: counts.get(status.value, 0) for status in SupportStatus}


# ── per-user, for the detail page ────────────────────────────────────


async def user_activity(session: AsyncSession, user_id: UUID) -> dict[str, int]:
    """Lifetime totals for one account. Four counts, no row fetching."""
    return {
        "generations": await _scalar(
            session, select(func.count(Generation.id)).where(Generation.user_id == user_id)
        ),
        "completed": await _scalar(
            session,
            select(func.count(Generation.id)).where(
                Generation.user_id == user_id, Generation.status == "COMPLETED"
            ),
        ),
        "downloads": await _scalar(
            session, select(func.count(DownloadEvent.id)).where(DownloadEvent.user_id == user_id)
        ),
        "payments": await _scalar(
            session,
            select(func.count(BillingPayment.id)).where(
                BillingPayment.user_id == user_id, _succeeded()
            ),
        ),
    }


__all__ = [
    "KST_OFFSET",
    "Bucket",
    "GenerationTotals",
    "Granularity",
    "download_series",
    "download_total",
    "generation_series",
    "generation_totals",
    "kst_date",
    "kst_day_bounds",
    "kst_today",
    "new_users",
    "period_bounds",
    "plan_distribution",
    "revenue_series",
    "revenue_split",
    "revenue_total",
    "support_counts",
    "user_activity",
    "user_series",
    "user_totals",
]
