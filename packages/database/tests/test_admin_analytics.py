"""Korean days, computed in SQL.

Timestamps are stored in UTC and stay that way; only the bucket boundary
shifts. That sounds like a detail until an operator compares the
dashboard against a bank statement and finds them a day apart, with
nothing on the page to explain why.

These tests exist because the shift is done in the database rather than
in Python. Two dialects render it — SQLite here, PostgreSQL in
production — and a boundary that is correct in one and wrong in the
other would pass a suite that only ever checked the Python.

The cases are chosen where the two calendars disagree: a Korean morning
is the previous day in UTC, and a Korean evening is the same day. Both
have to land on the Korean date.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from luber_database import Base, create_session_factory
from luber_database.admin_analytics import (
    DAILY_MAX_DAYS,
    WEEKLY_MAX_DAYS,
    bucketing_for,
    download_series,
    generation_series,
    kst_day_bounds,
    kst_today,
    period_bounds,
    previous_window,
    revenue_series,
    revenue_split,
    revenue_total,
    support_counts,
    user_series,
    user_totals,
)
from luber_database.models.admin import AdminAuditLog, AdminEmailCampaign, DownloadEvent
from luber_database.models.billing import AllowanceReservation, Subscription
from luber_database.models.generation import (
    AudioAsset,
    Generation,
    GenerationJob,
    GenerationQA,
    LyricLineQA,
    Project,
    ReferenceAudio,
)
from luber_database.models.payments import (
    PAYMENT_FAILED,
    PAYMENT_SUCCEEDED,
    BillingAnomaly,
    BillingCheckout,
    BillingEvent,
    BillingPayment,
)
from luber_database.models.support import SupportReply, SupportTicket
from luber_database.models.user import Session, User

ANALYTICS_TABLES = [
    User.__table__,
    Session.__table__,
    ReferenceAudio.__table__,
    Generation.__table__,
    GenerationJob.__table__,
    AudioAsset.__table__,
    GenerationQA.__table__,
    LyricLineQA.__table__,
    Project.__table__,
    Subscription.__table__,
    AllowanceReservation.__table__,
    BillingCheckout.__table__,
    BillingPayment.__table__,
    BillingEvent.__table__,
    BillingAnomaly.__table__,
    SupportTicket.__table__,
    SupportReply.__table__,
    DownloadEvent.__table__,
    AdminAuditLog.__table__,
    AdminEmailCampaign.__table__,
]


@pytest.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(sync_conn, tables=ANALYTICS_TABLES)
        )
    factory = create_session_factory(engine)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _user(session, email: str = "someone@example.com") -> uuid.UUID:
    user = User(id=uuid.uuid4(), email=email, password_hash="x")
    session.add(user)
    await session.commit()
    return user.id


async def _payment(
    session,
    user_id: uuid.UUID,
    *,
    amount: int,
    paid_at: datetime,
    status: str = PAYMENT_SUCCEEDED,
    subscription_id: uuid.UUID | None = None,
) -> None:
    session.add(
        BillingPayment(
            id=uuid.uuid4(),
            user_id=user_id,
            subscription_id=subscription_id,
            provider_payment_id=f"p-{uuid.uuid4().hex[:10]}",
            plan_id="basic",
            amount_krw=amount,
            status=status,
            paid_at=paid_at,
        )
    )
    await session.commit()


# ── the boundary ─────────────────────────────────────────────────────


def test_a_korean_day_starts_nine_hours_before_utc_midnight() -> None:
    from datetime import date

    start, end = kst_day_bounds(date(2026, 8, 28))

    assert start.isoformat() == "2026-08-27T15:00:00+00:00"
    assert end.isoformat() == "2026-08-28T15:00:00+00:00"


def test_the_korean_date_of_a_late_utc_evening_is_tomorrow() -> None:
    """23:00 UTC on the 27th is 08:00 on the 28th in Seoul."""
    assert kst_today(datetime(2026, 8, 27, 23, tzinfo=UTC)).isoformat() == "2026-08-28"


def test_the_korean_date_of_an_early_utc_morning_is_the_same_day() -> None:
    assert kst_today(datetime(2026, 8, 28, 6, tzinfo=UTC)).isoformat() == "2026-08-28"


def test_a_korean_week_starts_on_monday() -> None:
    """What a Korean operator means by "이번 주"."""
    # 2026-08-28 is a Friday; the week began on Monday the 24th.
    start, _ = period_bounds("week", datetime(2026, 8, 28, 3, tzinfo=UTC))

    assert start.isoformat() == "2026-08-23T15:00:00+00:00"  # 24th, 00:00 KST


async def test_a_korean_morning_buckets_under_the_korean_day(session) -> None:
    """The failure this prevents: a morning's revenue filed under
    yesterday, with the dashboard and the bank statement disagreeing."""
    user_id = await _user(session)
    # 08:00 KST on the 28th == 23:00 UTC on the 27th.
    await _payment(session, user_id, amount=19_900, paid_at=datetime(2026, 8, 27, 23, tzinfo=UTC))

    start, end = kst_day_bounds(datetime(2026, 8, 28, tzinfo=UTC).date())
    buckets = await revenue_series(session, start=start, end=end)

    assert [(b.day, b.value) for b in buckets] == [("2026-08-28", 19_900)]


async def test_a_korean_evening_buckets_under_the_same_day(session) -> None:
    user_id = await _user(session)
    # 22:00 KST on the 28th == 13:00 UTC on the 28th.
    await _payment(session, user_id, amount=29_900, paid_at=datetime(2026, 8, 28, 13, tzinfo=UTC))

    start, end = kst_day_bounds(datetime(2026, 8, 28, tzinfo=UTC).date())
    buckets = await revenue_series(session, start=start, end=end)

    assert [(b.day, b.value) for b in buckets] == [("2026-08-28", 29_900)]


async def test_two_payments_either_side_of_midnight_land_on_different_days(
    session,
) -> None:
    user_id = await _user(session)
    # 23:59 KST on the 27th, and 00:01 KST on the 28th.
    await _payment(
        session, user_id, amount=1_000, paid_at=datetime(2026, 8, 27, 14, 59, tzinfo=UTC)
    )
    await _payment(session, user_id, amount=2_000, paid_at=datetime(2026, 8, 27, 15, 1, tzinfo=UTC))

    start, _ = kst_day_bounds(datetime(2026, 8, 27, tzinfo=UTC).date())
    _, end = kst_day_bounds(datetime(2026, 8, 28, tzinfo=UTC).date())
    buckets = await revenue_series(session, start=start, end=end)

    assert [(b.day, b.value) for b in buckets] == [
        ("2026-08-27", 1_000),
        ("2026-08-28", 2_000),
    ]


# ── what counts as revenue ───────────────────────────────────────────


async def test_a_failed_payment_is_not_revenue(session) -> None:
    user_id = await _user(session)
    at = datetime(2026, 8, 28, 3, tzinfo=UTC)
    await _payment(session, user_id, amount=19_900, paid_at=at)
    await _payment(session, user_id, amount=19_900, paid_at=at, status=PAYMENT_FAILED)

    start, end = kst_day_bounds(datetime(2026, 8, 28, tzinfo=UTC).date())
    total, count = await revenue_total(session, start=start, end=end)

    assert (total, count) == (19_900, 1)


async def test_the_first_payment_for_a_subscription_is_new_and_the_next_is_a_renewal(
    session,
) -> None:
    """Split from the payment history, because nothing in the billing
    path records which a payment was — and inventing a column would mean
    backfilling a guess for every payment already taken."""
    user_id = await _user(session)
    subscription = uuid.uuid4()
    session.add(
        Subscription(
            id=subscription,
            user_id=user_id,
            plan_id="basic",
            status="ACTIVE",
            period_start=datetime(2026, 7, 1, tzinfo=UTC),
            period_end=datetime(2026, 10, 1, tzinfo=UTC),
        )
    )
    await session.commit()

    await _payment(
        session,
        user_id,
        amount=19_900,
        paid_at=datetime(2026, 7, 28, 3, tzinfo=UTC),
        subscription_id=subscription,
    )
    await _payment(
        session,
        user_id,
        amount=19_900,
        paid_at=datetime(2026, 8, 28, 3, tzinfo=UTC),
        subscription_id=subscription,
    )

    start, _ = kst_day_bounds(datetime(2026, 7, 1, tzinfo=UTC).date())
    _, end = kst_day_bounds(datetime(2026, 9, 1, tzinfo=UTC).date())
    split = await revenue_split(session, start=start, end=end)

    assert split == {
        "new_krw": 19_900,
        "new_count": 1,
        "renewal_krw": 19_900,
        "renewal_count": 1,
    }


async def test_the_split_is_empty_when_nothing_was_paid(session) -> None:
    start, end = kst_day_bounds(datetime(2026, 8, 28, tzinfo=UTC).date())

    assert await revenue_split(session, start=start, end=end) == {
        "new_krw": 0,
        "new_count": 0,
        "renewal_krw": 0,
        "renewal_count": 0,
    }


# ── members ──────────────────────────────────────────────────────────


async def test_a_closed_account_is_not_a_member(session) -> None:
    """Closing anonymises rather than deletes. The row survives; the
    membership does not."""
    live = await _user(session, "live@example.com")
    closed = await _user(session, "closed@example.com")
    row = await session.get(User, closed)
    row.deleted_at = datetime.now(UTC)
    await session.commit()

    totals = await user_totals(session)

    assert totals["total"] == 1
    assert live is not None


async def test_a_subscription_that_has_lapsed_is_not_a_paid_member(session) -> None:
    user_id = await _user(session)
    session.add(
        Subscription(
            id=uuid.uuid4(),
            user_id=user_id,
            plan_id="basic",
            status="ACTIVE",
            period_start=datetime.now(UTC) - timedelta(days=60),
            period_end=datetime.now(UTC) - timedelta(days=30),
        )
    )
    await session.commit()

    totals = await user_totals(session)

    assert totals["paid"] == 0
    assert totals["free"] == 1


async def test_support_counts_report_every_status_including_zero(session) -> None:
    """A status with no tickets is a zero, not a missing key — the
    console renders a row for each."""
    counts = await support_counts(session)

    assert set(counts) == {"OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"}
    assert all(value == 0 for value in counts.values())


# ── bucketing ────────────────────────────────────────────────────────


def test_the_bucket_size_follows_the_range_length() -> None:
    """Deterministic and total: every range gets exactly one answer.

    A chart reads at roughly 10-60 marks. A quarter of daily bars is 90
    and reads as noise; a year of them is 365 and reads as a smear.
    """
    assert bucketing_for(1) == "day"
    assert bucketing_for(7) == "day"
    assert bucketing_for(DAILY_MAX_DAYS) == "day"
    assert bucketing_for(DAILY_MAX_DAYS + 1) == "week"
    assert bucketing_for(WEEKLY_MAX_DAYS) == "week"
    assert bucketing_for(WEEKLY_MAX_DAYS + 1) == "month"
    assert bucketing_for(365) == "month"


async def test_weekly_buckets_collapse_a_week_onto_its_monday(session) -> None:
    """2026-08-24 is a Monday; the 24th through the 30th are one week."""
    user_id = await _user(session)
    for day, amount in ((24, 1_000), (26, 2_000), (30, 4_000)):
        # 12:00 KST each day, safely inside the Korean day.
        await _payment(
            session, user_id, amount=amount, paid_at=datetime(2026, 8, day, 3, tzinfo=UTC)
        )

    start, _ = kst_day_bounds(date(2026, 8, 24))
    _, end = kst_day_bounds(date(2026, 8, 30))
    buckets = await revenue_series(session, start=start, end=end, bucketing="week")

    assert [(b.day, b.value, b.secondary) for b in buckets] == [("2026-08-24", 7_000, 3)]


async def test_a_sunday_belongs_to_the_week_that_started_that_monday(session) -> None:
    """The case a naive "previous Monday" gets wrong.

    2026-08-30 is a Sunday. Its week began Monday the 24th, not the
    31st, and not the 17th.
    """
    user_id = await _user(session)
    await _payment(session, user_id, amount=5_000, paid_at=datetime(2026, 8, 30, 3, tzinfo=UTC))

    start, _ = kst_day_bounds(date(2026, 8, 1))
    _, end = kst_day_bounds(date(2026, 9, 30))
    buckets = await revenue_series(session, start=start, end=end, bucketing="week")

    assert [b.day for b in buckets] == ["2026-08-24"]


async def test_two_weeks_stay_two_buckets(session) -> None:
    user_id = await _user(session)
    await _payment(session, user_id, amount=1_000, paid_at=datetime(2026, 8, 24, 3, tzinfo=UTC))
    await _payment(session, user_id, amount=2_000, paid_at=datetime(2026, 8, 31, 3, tzinfo=UTC))

    start, _ = kst_day_bounds(date(2026, 8, 24))
    _, end = kst_day_bounds(date(2026, 9, 6))
    buckets = await revenue_series(session, start=start, end=end, bucketing="week")

    assert [(b.day, b.value) for b in buckets] == [("2026-08-24", 1_000), ("2026-08-31", 2_000)]


async def test_monthly_buckets_collapse_a_month_onto_the_first(session) -> None:
    user_id = await _user(session)
    for month, day, amount in ((7, 3, 1_000), (7, 28, 2_000), (8, 15, 4_000)):
        await _payment(
            session, user_id, amount=amount, paid_at=datetime(2026, month, day, 3, tzinfo=UTC)
        )

    start, _ = kst_day_bounds(date(2026, 7, 1))
    _, end = kst_day_bounds(date(2026, 8, 31))
    buckets = await revenue_series(session, start=start, end=end, bucketing="month")

    assert [(b.day, b.value) for b in buckets] == [("2026-07-01", 3_000), ("2026-08-01", 4_000)]


async def test_bucketing_a_korean_morning_keeps_it_in_the_korean_month(session) -> None:
    """23:00 UTC on 31 July is 08:00 KST on 1 August.

    The month boundary has the same trap as the day boundary, one level
    up: bucketing on the raw UTC month files a Korean August morning
    under July.
    """
    user_id = await _user(session)
    await _payment(session, user_id, amount=9_000, paid_at=datetime(2026, 7, 31, 23, tzinfo=UTC))

    start, _ = kst_day_bounds(date(2026, 7, 1))
    _, end = kst_day_bounds(date(2026, 8, 31))
    buckets = await revenue_series(session, start=start, end=end, bucketing="month")

    assert [b.day for b in buckets] == ["2026-08-01"]


async def test_generation_and_user_series_bucket_the_same_way(session) -> None:
    """One bucketing implementation, so the charts cannot disagree."""
    user_id = await _user(session)
    await _payment(session, user_id, amount=1_000, paid_at=datetime(2026, 8, 26, 3, tzinfo=UTC))

    start, _ = kst_day_bounds(date(2026, 8, 24))
    _, end = kst_day_bounds(date(2026, 8, 30))

    # Empty, but the query must compile and group identically for each.
    assert await generation_series(session, start=start, end=end, bucketing="week") == []
    assert await download_series(session, start=start, end=end, bucketing="month") == []
    users = await user_series(session, start=start, end=end, bucketing="week")
    assert all(b.day == "2026-08-24" for b in users)


# ── the previous period ──────────────────────────────────────────────


def test_the_previous_period_is_the_range_immediately_before() -> None:
    """The spec's own example: Aug 15-28 compares against Aug 1-14."""
    assert previous_window(date(2026, 8, 15), date(2026, 8, 28)) == (
        date(2026, 8, 1),
        date(2026, 8, 14),
    )


def test_a_single_day_compares_against_the_day_before() -> None:
    assert previous_window(date(2026, 8, 28), date(2026, 8, 28)) == (
        date(2026, 8, 27),
        date(2026, 8, 27),
    )


def test_the_previous_period_has_the_same_length_and_no_overlap() -> None:
    """Both properties matter: a shorter window understates the
    comparison, and an overlapping one counts days twice."""
    for span in (1, 7, 30, 31, 90, 365):
        first, last = date(2026, 8, 1), date(2026, 8, 1) + timedelta(days=span - 1)
        prev_first, prev_last = previous_window(first, last)

        assert (prev_last - prev_first).days == (last - first).days
        assert prev_last < first


def test_the_previous_period_crosses_a_year_boundary_cleanly() -> None:
    assert previous_window(date(2026, 1, 1), date(2026, 1, 7)) == (
        date(2025, 12, 25),
        date(2025, 12, 31),
    )
