"""The twelve-month retention period, enforced.

The privacy policy states that raw acquisition records are deleted after
twelve months. These tests are what make that a fact rather than a
sentence: they pin the cutoff to the day, prove the job cannot reach
commerce records, and prove a dry run writes nothing.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import create_async_engine

from luber_database import Base, create_session_factory
from luber_database.acquisition_repository import AcquisitionRepository
from luber_database.acquisition_retention import (
    RETENTION_DAYS,
    cutoff_for,
    purge_acquisition,
)
from luber_database.models.acquisition import (
    AcquisitionAttribution,
    AcquisitionSession,
    AcquisitionVisitor,
)
from luber_database.models.payments import PAYMENT_SUCCEEDED, BillingPayment
from luber_database.models.user import User
from luber_schemas.acquisition import Attribution

TABLES = [
    User.__table__,
    BillingPayment.__table__,
    AcquisitionVisitor.__table__,
    AcquisitionSession.__table__,
    AcquisitionAttribution.__table__,
]

NOW = datetime(2026, 8, 29, 12, tzinfo=UTC)
INSTAGRAM = Attribution(source="instagram", medium="paid_social", campaign="summer")


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=TABLES))
    factory = create_session_factory(engine)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _visit(session, at: datetime, key=None):
    return await AcquisitionRepository(session).record_visit(
        visitor_key=key or uuid.uuid4(),
        attribution=INSTAGRAM,
        landing_path="/",
        referrer_host=None,
        now=at,
    )


async def _count(session, model) -> int:
    return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


# ── the cutoff ───────────────────────────────────────────────────────


def test_the_retention_period_is_twelve_months() -> None:
    """The number the privacy policy publishes."""
    assert RETENTION_DAYS == 365


def test_the_cutoff_is_exactly_the_period_before_now() -> None:
    assert cutoff_for(NOW) == NOW - timedelta(days=365)


async def test_a_record_one_day_past_the_cutoff_is_deleted(session) -> None:
    await _visit(session, NOW - timedelta(days=366))

    report = await purge_acquisition(session, now=NOW)

    assert report.visitors_deleted == 1
    assert await _count(session, AcquisitionVisitor) == 0


async def test_a_record_one_day_inside_the_cutoff_is_kept(session) -> None:
    """The boundary, from the other side. A retention period that
    deletes a day early is a different period."""
    await _visit(session, NOW - timedelta(days=364))

    report = await purge_acquisition(session, now=NOW)

    assert report.visitors_deleted == 0
    assert await _count(session, AcquisitionVisitor) == 1


async def test_only_the_expired_half_goes(session) -> None:
    await _visit(session, NOW - timedelta(days=400))
    await _visit(session, NOW - timedelta(days=200))
    await _visit(session, NOW - timedelta(days=1))

    report = await purge_acquisition(session, now=NOW)

    assert report.visitors_deleted == 1
    assert await _count(session, AcquisitionVisitor) == 2


async def test_the_cutoff_reads_first_contact_not_the_latest(session) -> None:
    """Otherwise a visitor who returns once a year is kept forever,
    which is the opposite of a retention period."""
    key = uuid.uuid4()
    await _visit(session, NOW - timedelta(days=400), key=key)
    await _visit(session, NOW - timedelta(days=2), key=key)

    report = await purge_acquisition(session, now=NOW)

    assert report.visitors_deleted == 1


async def test_sessions_go_with_their_visitor(session) -> None:
    key = uuid.uuid4()
    await _visit(session, NOW - timedelta(days=400), key=key)
    await _visit(session, NOW - timedelta(days=399), key=key)
    assert await _count(session, AcquisitionSession) == 2

    report = await purge_acquisition(session, now=NOW)

    assert report.sessions_deleted == 2
    assert await _count(session, AcquisitionSession) == 0


# ── what it must never touch ─────────────────────────────────────────


async def test_billing_records_are_untouched(session) -> None:
    """Commerce records are retained under 전자상거래법 for years. A
    marketing cleanup must not be able to reach them."""
    user = User(id=uuid.uuid4(), email="a@example.com", password_hash="x")
    session.add(user)
    await session.commit()
    session.add(
        BillingPayment(
            id=uuid.uuid4(),
            user_id=user.id,
            subscription_id=None,
            provider_payment_id="p-1",
            plan_id="basic",
            amount_krw=19_900,
            status=PAYMENT_SUCCEEDED,
            paid_at=NOW - timedelta(days=400),
        )
    )
    await session.commit()
    await _visit(session, NOW - timedelta(days=400))

    await purge_acquisition(session, now=NOW)

    assert await _count(session, BillingPayment) == 1, "billing is out of scope"
    assert await _count(session, User) == 1


async def test_a_customers_attribution_snapshot_survives_its_visitor(session) -> None:
    """The snapshot is what the console reports on. Losing it would
    silently rewrite last year's figures."""
    key = uuid.uuid4()
    await _visit(session, NOW - timedelta(days=400), key=key)
    user = User(id=uuid.uuid4(), email="b@example.com", password_hash="x")
    session.add(user)
    await session.commit()
    await AcquisitionRepository(session).bind_signup(visitor_key=key, user_id=user.id)

    await purge_acquisition(session, now=NOW)

    snapshot = await session.get(AcquisitionAttribution, user.id)
    assert snapshot is not None
    assert snapshot.first_source == "instagram"
    assert await _count(session, AcquisitionVisitor) == 0


# ── safety properties ────────────────────────────────────────────────


async def test_a_dry_run_counts_and_writes_nothing(session) -> None:
    await _visit(session, NOW - timedelta(days=400))
    await _visit(session, NOW - timedelta(days=401))

    report = await purge_acquisition(session, now=NOW, dry_run=True)

    assert report.dry_run is True
    assert report.visitors_deleted == 2
    assert await _count(session, AcquisitionVisitor) == 2, "a dry run must not delete"
    assert await _count(session, AcquisitionSession) == 2


async def test_running_twice_changes_nothing_the_second_time(session) -> None:
    await _visit(session, NOW - timedelta(days=400))

    first = await purge_acquisition(session, now=NOW)
    second = await purge_acquisition(session, now=NOW)

    assert first.visitors_deleted == 1
    assert second.visitors_deleted == 0


async def test_an_empty_table_is_a_no_op(session) -> None:
    """What production looks like today: nothing is old enough."""
    report = await purge_acquisition(session, now=NOW)

    assert (report.visitors_deleted, report.sessions_deleted) == (0, 0)


async def test_todays_data_is_never_touched(session) -> None:
    """The three rows currently in production are days old, not months.
    Deploying this must delete none of them."""
    for age in (0, 1, 2):
        await _visit(session, NOW - timedelta(days=age))

    report = await purge_acquisition(session, now=NOW)

    assert report.visitors_deleted == 0
    assert await _count(session, AcquisitionVisitor) == 3


async def test_batching_finishes_the_whole_job(session) -> None:
    """A small batch must not mean a partial purge."""
    for index in range(7):
        await _visit(session, NOW - timedelta(days=400 + index))

    report = await purge_acquisition(session, now=NOW, batch=2)

    assert report.visitors_deleted == 7
    assert await _count(session, AcquisitionVisitor) == 0
