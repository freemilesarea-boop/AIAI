"""The attribution rules, at the repository.

Three properties carry the model, and each is easy to break by writing
the obvious thing instead:

* first touch is written once and never again
* direct traffic never overwrites a known source
* a paid conversion is the *first* successful payment, counted once
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from luber_database import Base, create_session_factory
from luber_database.acquisition_analytics import (
    campaign_breakdown,
    channel_breakdown,
    summary,
    unattributed_users,
)
from luber_database.acquisition_repository import AcquisitionRepository
from luber_database.models.acquisition import (
    AcquisitionAttribution,
    AcquisitionSession,
    AcquisitionVisitor,
)
from luber_database.models.payments import (
    PAYMENT_FAILED,
    PAYMENT_SUCCEEDED,
    BillingPayment,
)
from luber_database.models.user import User
from luber_schemas.acquisition import DIRECT, Attribution

TABLES = [
    User.__table__,
    BillingPayment.__table__,
    AcquisitionVisitor.__table__,
    AcquisitionSession.__table__,
    AcquisitionAttribution.__table__,
]

INSTAGRAM = Attribution(source="instagram", medium="paid_social", campaign="summer")
GOOGLE = Attribution(source="google", medium="organic")
YOUTUBE = Attribution(source="youtube", medium="referral", campaign="august")


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=TABLES))
    factory = create_session_factory(engine)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _visit(session, key, attribution, *, path="/", host=None, at=None):
    return await AcquisitionRepository(session).record_visit(
        visitor_key=key,
        attribution=attribution,
        landing_path=path,
        referrer_host=host,
        now=at,
    )


async def _user(session, email="someone@example.com") -> uuid.UUID:
    user = User(id=uuid.uuid4(), email=email, password_hash="x")
    session.add(user)
    await session.commit()
    return user.id


# ── first touch ──────────────────────────────────────────────────────


async def test_the_first_visit_sets_the_origin(session) -> None:
    key = uuid.uuid4()

    visitor = await _visit(session, key, GOOGLE)

    assert (visitor.first_source, visitor.first_medium) == ("google", "organic")


async def test_first_touch_never_changes(session) -> None:
    """A customer acquired by a search stays acquired by that search,
    however many campaigns they click afterwards."""
    key = uuid.uuid4()
    await _visit(session, key, GOOGLE)

    await _visit(session, key, INSTAGRAM)
    visitor = await _visit(session, key, YOUTUBE)

    assert (visitor.first_source, visitor.first_medium) == ("google", "organic")


async def test_an_attributable_first_visit_is_also_the_last_touch(session) -> None:
    """Otherwise somebody who signs up on their first visit would have
    no last-touch source at all."""
    visitor = await _visit(session, uuid.uuid4(), INSTAGRAM)

    assert visitor.last_source == "instagram"
    assert visitor.last_campaign == "summer"


async def test_a_direct_first_visit_leaves_last_touch_empty(session) -> None:
    """Null, not "direct" — the distinction between "came back on their
    own" and "came back through a campaign" is the point."""
    visitor = await _visit(session, uuid.uuid4(), DIRECT)

    assert visitor.first_source == "direct"
    assert visitor.last_source is None


# ── last non-direct touch ────────────────────────────────────────────


async def test_direct_traffic_never_overwrites_a_known_source(session) -> None:
    """The rule that decides whether paid acquisition looks worthless.

    Somebody arrives from an Instagram ad, leaves, and comes back by
    typing the address. The ad still brought them.
    """
    key = uuid.uuid4()
    await _visit(session, key, INSTAGRAM)

    visitor = await _visit(session, key, DIRECT)

    assert visitor.last_source == "instagram"
    assert visitor.last_campaign == "summer"


async def test_a_later_campaign_does_move_last_touch(session) -> None:
    key = uuid.uuid4()
    await _visit(session, key, GOOGLE)

    visitor = await _visit(session, key, INSTAGRAM)

    assert visitor.first_source == "google"
    assert visitor.last_source == "instagram"


async def test_a_direct_return_still_records_the_visit(session) -> None:
    """Direct updates when they were last seen, and a session row: the
    visit happened, it just says nothing about acquisition."""
    key = uuid.uuid4()
    first = datetime(2026, 8, 1, tzinfo=UTC)
    later = datetime(2026, 8, 20, tzinfo=UTC)
    await _visit(session, key, INSTAGRAM, at=first)

    visitor = await _visit(session, key, DIRECT, at=later)

    # Compared without tzinfo: SQLite drops the offset on round-trip
    # while PostgreSQL keeps it, and the instant is what this asserts.
    assert visitor.last_seen_at.replace(tzinfo=None) == later.replace(tzinfo=None)
    assert visitor.first_seen_at.replace(tzinfo=None) == first.replace(tzinfo=None)


# ── signup binding ───────────────────────────────────────────────────


async def test_signup_snapshots_both_touches(session) -> None:
    key = uuid.uuid4()
    await _visit(session, key, GOOGLE)
    await _visit(session, key, INSTAGRAM)
    user_id = await _user(session)

    attribution = await AcquisitionRepository(session).bind_signup(visitor_key=key, user_id=user_id)

    assert attribution is not None
    assert (attribution.first_source, attribution.last_source) == ("google", "instagram")


async def test_the_snapshot_does_not_move_when_the_visitor_returns(session) -> None:
    """Last month's report has to keep saying what it said."""
    key = uuid.uuid4()
    await _visit(session, key, GOOGLE)
    user_id = await _user(session)
    await AcquisitionRepository(session).bind_signup(visitor_key=key, user_id=user_id)

    await _visit(session, key, INSTAGRAM)

    stored = await session.get(AcquisitionAttribution, user_id)
    assert stored is not None
    assert stored.first_source == "google"
    assert stored.last_source == "google"


async def test_no_cookie_means_no_attribution_and_no_failure(session) -> None:
    """The ordinary case for anyone who blocks cookies. A signup must
    never fail because analytics could not run."""
    user_id = await _user(session)

    assert (
        await AcquisitionRepository(session).bind_signup(visitor_key=None, user_id=user_id) is None
    )


async def test_an_unknown_visitor_key_binds_nothing(session) -> None:
    user_id = await _user(session)

    assert (
        await AcquisitionRepository(session).bind_signup(visitor_key=uuid.uuid4(), user_id=user_id)
        is None
    )


async def test_a_shared_browser_does_not_steal_the_first_account(session) -> None:
    """Two accounts from one laptop is ordinary. The second must not
    take over the first one's visitor."""
    key = uuid.uuid4()
    await _visit(session, key, INSTAGRAM)
    first_user = await _user(session, "first@example.com")
    second_user = await _user(session, "second@example.com")
    repository = AcquisitionRepository(session)

    await repository.bind_signup(visitor_key=key, user_id=first_user)
    await repository.bind_signup(visitor_key=key, user_id=second_user)

    visitor = (
        await session.execute(
            AcquisitionVisitor.__table__.select().where(AcquisitionVisitor.visitor_key == key)
        )
    ).first()
    assert visitor is not None
    assert visitor.user_id == first_user
    # Both accounts still get their own snapshot.
    assert await session.get(AcquisitionAttribution, second_user) is not None


async def test_binding_twice_keeps_the_first_answer(session) -> None:
    key = uuid.uuid4()
    await _visit(session, key, GOOGLE)
    user_id = await _user(session)
    repository = AcquisitionRepository(session)

    first = await repository.bind_signup(visitor_key=key, user_id=user_id)
    await _visit(session, key, INSTAGRAM)
    second = await repository.bind_signup(visitor_key=key, user_id=user_id)

    assert first is not None and second is not None
    assert second.first_source == first.first_source == "google"
    assert second.last_source == "google"


# ── closure ──────────────────────────────────────────────────────────


async def test_closing_an_account_detaches_its_visitors(session) -> None:
    """A browser still carrying the cookie is anonymous again."""
    key = uuid.uuid4()
    await _visit(session, key, INSTAGRAM)
    user_id = await _user(session)
    repository = AcquisitionRepository(session)
    await repository.bind_signup(visitor_key=key, user_id=user_id)

    detached = await repository.unlink_user(user_id)
    await session.commit()

    row = (
        await session.execute(
            AcquisitionVisitor.__table__.select().where(AcquisitionVisitor.visitor_key == key)
        )
    ).first()
    assert detached == 1
    assert row is not None and row.user_id is None
    # The marketing snapshot survives; it names nobody once the account
    # row is anonymised.
    assert await session.get(AcquisitionAttribution, user_id) is not None


# ── reporting ────────────────────────────────────────────────────────


WINDOW_START = datetime(2026, 8, 1, tzinfo=UTC)
WINDOW_END = datetime(2026, 9, 1, tzinfo=UTC)


async def _pay(session, user_id, amount, at, status=PAYMENT_SUCCEEDED):
    session.add(
        BillingPayment(
            id=uuid.uuid4(),
            user_id=user_id,
            subscription_id=None,
            provider_payment_id=f"p-{uuid.uuid4().hex[:10]}",
            plan_id="basic",
            amount_krw=amount,
            status=status,
            paid_at=at,
        )
    )
    await session.commit()


async def test_the_funnel_groups_by_channel(session) -> None:
    key = uuid.uuid4()
    at = datetime(2026, 8, 10, tzinfo=UTC)
    await _visit(session, key, INSTAGRAM, at=at)
    user_id = await _user(session)
    await AcquisitionRepository(session).bind_signup(visitor_key=key, user_id=user_id, now=at)
    await _pay(session, user_id, 19_900, datetime(2026, 8, 12, tzinfo=UTC))

    rows = await channel_breakdown(session, start=WINDOW_START, end=WINDOW_END)

    assert len(rows) == 1
    row = rows[0]
    assert row.key == "instagram_ads"
    assert (row.visitors, row.signups, row.conversions, row.revenue_krw) == (1, 1, 1, 19_900)


async def test_a_failed_payment_is_not_a_conversion(session) -> None:
    """Revenue and conversions come only from verified successful
    payments — the same source of truth the revenue dashboard uses."""
    key = uuid.uuid4()
    at = datetime(2026, 8, 10, tzinfo=UTC)
    await _visit(session, key, INSTAGRAM, at=at)
    user_id = await _user(session)
    await AcquisitionRepository(session).bind_signup(visitor_key=key, user_id=user_id, now=at)
    await _pay(session, user_id, 19_900, datetime(2026, 8, 12, tzinfo=UTC), status=PAYMENT_FAILED)

    rows = await channel_breakdown(session, start=WINDOW_START, end=WINDOW_END)

    assert rows[0].conversions == 0
    assert rows[0].revenue_krw == 0


async def test_a_renewal_does_not_count_as_a_second_conversion(session) -> None:
    """Acquisition counts the first paid conversion. A renewal is
    retention, and inflating acquisition with it would make every
    channel look better the longer it ran."""
    key = uuid.uuid4()
    at = datetime(2026, 8, 1, tzinfo=UTC)
    await _visit(session, key, INSTAGRAM, at=at)
    user_id = await _user(session)
    await AcquisitionRepository(session).bind_signup(visitor_key=key, user_id=user_id, now=at)
    await _pay(session, user_id, 19_900, datetime(2026, 8, 5, tzinfo=UTC))
    await _pay(session, user_id, 19_900, datetime(2026, 8, 25, tzinfo=UTC))

    rows = await channel_breakdown(session, start=WINDOW_START, end=WINDOW_END)

    assert rows[0].conversions == 1, "one customer converted once"
    assert rows[0].revenue_krw == 39_800, "but both payments are revenue"


async def test_first_and_last_touch_credit_different_channels(session) -> None:
    key = uuid.uuid4()
    at = datetime(2026, 8, 10, tzinfo=UTC)
    await _visit(session, key, GOOGLE, at=at)
    await _visit(session, key, INSTAGRAM, at=at + timedelta(days=1))
    user_id = await _user(session)
    await AcquisitionRepository(session).bind_signup(
        visitor_key=key, user_id=user_id, now=at + timedelta(days=2)
    )

    first = await channel_breakdown(session, start=WINDOW_START, end=WINDOW_END, mode="first_touch")
    last = await channel_breakdown(session, start=WINDOW_START, end=WINDOW_END, mode="last_touch")

    assert {r.key: r.signups for r in first}["google_organic"] == 1
    assert {r.key: r.signups for r in last}["instagram_ads"] == 1


async def test_a_direct_only_visitor_is_counted_in_both_modes(session) -> None:
    """Last-touch falls back to first-touch, so the two modes split the
    same population rather than counting different ones."""
    key = uuid.uuid4()
    at = datetime(2026, 8, 10, tzinfo=UTC)
    await _visit(session, key, DIRECT, at=at)
    user_id = await _user(session)
    await AcquisitionRepository(session).bind_signup(visitor_key=key, user_id=user_id, now=at)

    for mode in ("first_touch", "last_touch"):
        totals = await summary(session, start=WINDOW_START, end=WINDOW_END, mode=mode)
        assert totals["signups"] == 1, mode


async def test_a_visit_outside_the_window_is_not_counted(session) -> None:
    await _visit(session, uuid.uuid4(), INSTAGRAM, at=datetime(2026, 7, 1, tzinfo=UTC))

    totals = await summary(session, start=WINDOW_START, end=WINDOW_END)

    assert totals["visitors"] == 0


async def test_rates_are_absent_rather_than_dividing_by_zero(session) -> None:
    totals = await summary(session, start=WINDOW_START, end=WINDOW_END)

    assert totals["visitors"] == 0
    assert totals["signup_rate"] is None
    assert totals["conversion_rate"] is None


async def test_campaigns_break_down_below_source_and_medium(session) -> None:
    at = datetime(2026, 8, 10, tzinfo=UTC)
    await _visit(session, uuid.uuid4(), INSTAGRAM, at=at)
    await _visit(
        session,
        uuid.uuid4(),
        Attribution(source="instagram", medium="paid_social", campaign="creator_01"),
        at=at,
    )

    rows = await campaign_breakdown(session, start=WINDOW_START, end=WINDOW_END, mode="last_touch")

    assert {r.campaign for r in rows} == {"summer", "creator_01"}
    assert all(r.source == "instagram" for r in rows)


async def test_accounts_from_before_this_existed_are_reported_separately(session) -> None:
    """Never folded into 직접 유입: there is no evidence either way, and
    inventing one gives a channel budget it did not earn."""
    await _user(session, "legacy@example.com")

    assert await unattributed_users(session) == 1

    rows = await channel_breakdown(session, start=WINDOW_START, end=WINDOW_END)
    assert rows == []
