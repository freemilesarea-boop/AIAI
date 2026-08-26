"""Monthly allowance: what it counts, and what it refuses.

Three properties carry this file.

**Only successful songs cost anything.** A failure releases its slot, and
the user can generate again. Anything else charges people for the
product not working.

**The limit holds under concurrency.** Ten requests against one remaining
slot must produce one winner. That is asserted by racing them, not by
reading the code and agreeing it looks right — the naive count-then-
insert passes every sequential test ever written.

**An account sees only its own.** The repository is bound to an owner, so
there is no argument to get wrong.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from luber_database import Base, GenerationRepository, create_session_factory
from luber_database.allowance_repository import (
    AllowanceExhaustedError,
    AllowanceRepository,
    month_bounds,
)
from luber_database.models.billing import (
    CONSUMED,
    RELEASED,
    RESERVED,
    AllowanceReservation,
    Subscription,
)
from luber_database.models.generation import (
    AudioAsset,
    Generation,
    GenerationJob,
    GenerationQA,
    LyricLineQA,
    Project,
    ReferenceAudio,
)
from luber_database.models.user import User
from luber_schemas.plans import PLANS, PlanId, plan_for

TABLES = [
    User.__table__,
    ReferenceAudio.__table__,
    Generation.__table__,
    GenerationJob.__table__,
    AudioAsset.__table__,
    GenerationQA.__table__,
    LyricLineQA.__table__,
    Project.__table__,
    Subscription.__table__,
    AllowanceReservation.__table__,
]

USER_A = uuid.UUID("aaaaaaaa-1111-4111-8111-111111111111")
USER_B = uuid.UUID("bbbbbbbb-2222-4222-8222-222222222222")


@pytest.fixture
async def engine():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=TABLES))
    yield engine
    await engine.dispose()


@pytest.fixture
async def factory(engine):
    return create_session_factory(engine)


async def _seed_users(factory) -> None:
    async with factory() as session:
        for uid, email in ((USER_A, "a@boorda.test"), (USER_B, "b@boorda.test")):
            session.add(User(id=uid, email=email, password_hash="x"))
        await session.commit()


async def _allowance(factory, owner=USER_A) -> AllowanceRepository:
    session = factory()
    return AllowanceRepository(await session.__aenter__(), owner)


async def _make_generation(factory, owner=USER_A) -> uuid.UUID:
    """A row to attach a reservation to. Only its id matters here."""
    async with factory() as session:
        repo = GenerationRepository(session, owner=owner)
        generation = await repo.create_generation(
            title="t",
            prompt="p",
            lyrics="",
            vocal_gender="instrumental",
            duration_requested=30,
            language="ko",
            instrumental=True,
            status="QUEUED",
        )
        return generation.id


# ── plan resolution ───────────────────────────────────────────────────


async def test_an_account_with_no_subscription_is_free(factory):
    await _seed_users(factory)
    async with factory() as session:
        plan = await AllowanceRepository(session, USER_A).effective_plan()
    assert plan.plan_id is PlanId.FREE
    assert plan.monthly_generation_limit == 20
    assert plan.download_mp3 is False


async def test_an_unknown_plan_id_falls_back_to_free_rather_than_raising(factory):
    """Least privilege on bad data, never a more generous tier."""
    await _seed_users(factory)
    async with factory() as session:
        session.add(
            Subscription(
                user_id=USER_A,
                plan_id="enterprise-that-does-not-exist",
                period_start=datetime.now(UTC),
                period_end=datetime.now(UTC) + timedelta(days=30),
            )
        )
        await session.commit()
        plan = await AllowanceRepository(session, USER_A).effective_plan()
    assert plan.plan_id is PlanId.FREE


@pytest.mark.parametrize(
    "plan_id,limit",
    [(PlanId.FREE, 20), (PlanId.BASIC, 200), (PlanId.PRO, 500), (PlanId.CREATOR, 1000)],
)
async def test_each_plan_reports_its_own_limit(factory, plan_id, limit):
    await _seed_users(factory)
    async with factory() as session:
        allowance = AllowanceRepository(session, USER_A)
        await allowance.set_plan(plan_id)
        entitlement = await allowance.entitlement()
    assert entitlement.plan.plan_id is plan_id
    assert entitlement.limit == limit
    assert entitlement.remaining == limit


# ── the reservation protocol ──────────────────────────────────────────


async def test_a_reservation_spends_one_song(factory):
    await _seed_users(factory)
    gid = await _make_generation(factory)
    async with factory() as session:
        allowance = AllowanceRepository(session, USER_A)
        before = await allowance.entitlement()
        await allowance.reserve(gid)
        after = await allowance.entitlement()
    assert before.used == 0
    assert after.used == 1
    assert after.remaining == before.remaining - 1


async def test_a_completed_generation_keeps_its_slot(factory):
    await _seed_users(factory)
    gid = await _make_generation(factory)
    async with factory() as session:
        allowance = AllowanceRepository(session, USER_A)
        await allowance.reserve(gid)
        assert await allowance.consume(gid) is True
        assert (await allowance.entitlement()).used == 1


async def test_a_failed_generation_costs_nothing(factory):
    """The property the phase turns on."""
    await _seed_users(factory)
    gid = await _make_generation(factory)
    async with factory() as session:
        allowance = AllowanceRepository(session, USER_A)
        await allowance.reserve(gid)
        assert (await allowance.entitlement()).used == 1
        assert await allowance.release(gid) is True
        assert (await allowance.entitlement()).used == 0
        assert (await allowance.entitlement()).remaining == 20


async def test_releasing_twice_does_not_refund_twice(factory):
    await _seed_users(factory)
    gid = await _make_generation(factory)
    async with factory() as session:
        allowance = AllowanceRepository(session, USER_A)
        await allowance.reserve(gid)
        assert await allowance.release(gid) is True
        assert await allowance.release(gid) is False
        assert (await allowance.entitlement()).used == 0


async def test_a_settled_reservation_cannot_be_consumed_afterwards(factory):
    await _seed_users(factory)
    gid = await _make_generation(factory)
    async with factory() as session:
        allowance = AllowanceRepository(session, USER_A)
        await allowance.reserve(gid)
        await allowance.release(gid)
        assert await allowance.consume(gid) is False


async def test_reserving_the_same_generation_twice_charges_once(factory):
    await _seed_users(factory)
    gid = await _make_generation(factory)
    async with factory() as session:
        allowance = AllowanceRepository(session, USER_A)
        first = await allowance.reserve(gid)
        second = await allowance.reserve(gid)
        assert first.id == second.id
        assert (await allowance.entitlement()).used == 1


# ── the limit ─────────────────────────────────────────────────────────


async def test_the_last_song_is_allowed_and_the_next_is_not(factory):
    await _seed_users(factory)
    async with factory() as session:
        allowance = AllowanceRepository(session, USER_A)
        await allowance.set_plan(PlanId.FREE)
        for _ in range(20):
            await allowance.reserve(await _make_generation(factory))
        entitlement = await allowance.entitlement()
        assert entitlement.used == 20
        assert entitlement.remaining == 0
        assert entitlement.exhausted is True

        with pytest.raises(AllowanceExhaustedError) as raised:
            await allowance.reserve(await _make_generation(factory))
    assert raised.value.limit == 20
    assert raised.value.used == 20


async def test_a_release_frees_the_limit_again(factory):
    await _seed_users(factory)
    ids = []
    async with factory() as session:
        allowance = AllowanceRepository(session, USER_A)
        for _ in range(20):
            gid = await _make_generation(factory)
            ids.append(gid)
            await allowance.reserve(gid)
        await allowance.release(ids[0])
        # One slot back, so one more generation is allowed.
        await allowance.reserve(await _make_generation(factory))
        assert (await allowance.entitlement()).used == 20


# ── concurrency ───────────────────────────────────────────────────────


async def test_ten_racing_requests_cannot_share_one_remaining_slot(tmp_path):
    """The adversarial case. A naive count-then-insert fails here.

    On its own database file rather than the shared in-memory one: a
    StaticPool hands every session the same connection, so a rollback in
    one tears down the others and the race never actually happens. A file
    gives each session its own connection, which is the situation the
    unique constraint exists for.
    """
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/allowance.db")
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=TABLES))
    factory = create_session_factory(engine)
    await _seed_users(factory)
    async with factory() as session:
        allowance = AllowanceRepository(session, USER_A)
        await allowance.set_plan(PlanId.FREE)
        for _ in range(19):
            await allowance.reserve(await _make_generation(factory))
        assert (await allowance.entitlement()).remaining == 1

    contenders = [await _make_generation(factory) for _ in range(10)]

    async def attempt(gid):
        async with factory() as session:
            try:
                await AllowanceRepository(session, USER_A).reserve(gid)
                return True
            except AllowanceExhaustedError:
                return False

    outcomes = await asyncio.gather(*(attempt(g) for g in contenders))

    assert sum(outcomes) == 1, f"expected exactly one winner, got {sum(outcomes)}"
    async with factory() as session:
        entitlement = await AllowanceRepository(session, USER_A).entitlement()
    assert entitlement.used == 20
    assert entitlement.remaining == 0
    assert entitlement.used <= entitlement.limit, "the limit was exceeded"
    await engine.dispose()


async def test_remaining_never_goes_negative(factory):
    await _seed_users(factory)
    async with factory() as session:
        allowance = AllowanceRepository(session, USER_A)
        for _ in range(20):
            await allowance.reserve(await _make_generation(factory))
        for _ in range(5):
            with pytest.raises(AllowanceExhaustedError):
                await allowance.reserve(await _make_generation(factory))
        entitlement = await allowance.entitlement()
    assert entitlement.remaining == 0
    assert entitlement.used == 20


# ── isolation ─────────────────────────────────────────────────────────


async def test_one_accounts_usage_is_invisible_to_another(factory):
    await _seed_users(factory)
    async with factory() as session:
        a = AllowanceRepository(session, USER_A)
        for _ in range(5):
            await a.reserve(await _make_generation(factory, USER_A))

    async with factory() as session:
        b_entitlement = await AllowanceRepository(session, USER_B).entitlement()
        a_entitlement = await AllowanceRepository(session, USER_A).entitlement()
    assert a_entitlement.used == 5
    assert b_entitlement.used == 0, "B must not see A's usage"


async def test_one_accounts_plan_does_not_change_anothers(factory):
    await _seed_users(factory)
    async with factory() as session:
        await AllowanceRepository(session, USER_A).set_plan(PlanId.CREATOR)
        assert (await AllowanceRepository(session, USER_B).effective_plan()).plan_id is PlanId.FREE


# ── period ────────────────────────────────────────────────────────────


def test_a_month_runs_from_the_first_to_the_first():
    start, end = month_bounds(datetime(2026, 3, 17, 9, 30, tzinfo=UTC))
    assert start == datetime(2026, 3, 1, tzinfo=UTC)
    assert end == datetime(2026, 4, 1, tzinfo=UTC)


def test_december_rolls_into_january():
    start, end = month_bounds(datetime(2026, 12, 31, 23, 59, tzinfo=UTC))
    assert start == datetime(2026, 12, 1, tzinfo=UTC)
    assert end == datetime(2027, 1, 1, tzinfo=UTC)


async def test_usage_from_a_past_period_does_not_count_against_this_one(factory):
    await _seed_users(factory)
    last_month = datetime.now(UTC).replace(day=1) - timedelta(days=1)
    gid = await _make_generation(factory)
    async with factory() as session:
        allowance = AllowanceRepository(session, USER_A)
        await allowance.reserve(gid, now=last_month)
        # A new period: the old slot belongs to the old month.
        assert (await allowance.entitlement()).used == 0


async def test_a_subscription_period_wins_while_it_contains_now(factory):
    """Where a billing provider's cycle will land."""
    await _seed_users(factory)
    now = datetime.now(UTC)
    async with factory() as session:
        session.add(
            Subscription(
                user_id=USER_A,
                plan_id=PlanId.PRO.value,
                period_start=now - timedelta(days=3),
                period_end=now + timedelta(days=27),
            )
        )
        await session.commit()
        start, end = await AllowanceRepository(session, USER_A).current_period(now=now)
    assert start == (now - timedelta(days=3))
    assert end == (now + timedelta(days=27))


# ── ledger ────────────────────────────────────────────────────────────


async def test_the_ledger_says_which_generations_spent_the_allowance(factory):
    await _seed_users(factory)
    kept = await _make_generation(factory)
    failed = await _make_generation(factory)
    async with factory() as session:
        allowance = AllowanceRepository(session, USER_A)
        await allowance.reserve(kept)
        await allowance.reserve(failed)
        await allowance.consume(kept)
        await allowance.release(failed)
        rows = await allowance.reservations_in_period()

    by_generation = {r.generation_id: r for r in rows}
    assert by_generation[kept].state == CONSUMED
    assert by_generation[failed].state == RELEASED
    # A failed generation is distinguishable, which is the point.
    assert {r.state for r in rows} == {CONSUMED, RELEASED}


async def test_a_reservation_records_the_plan_in_force_when_it_was_taken(factory):
    await _seed_users(factory)
    gid = await _make_generation(factory)
    async with factory() as session:
        allowance = AllowanceRepository(session, USER_A)
        await allowance.set_plan(PlanId.BASIC)
        await allowance.reserve(gid)
        await allowance.set_plan(PlanId.FREE)
        rows = await allowance.reservations_in_period()
    # History is not rewritten by a later plan change.
    assert rows[0].plan_id == PlanId.BASIC.value
    assert rows[0].state == RESERVED


# ── entitlement payload ───────────────────────────────────────────────


async def test_the_entitlement_payload_carries_no_storage_or_billing_internals(factory):
    await _seed_users(factory)
    async with factory() as session:
        payload = (await AllowanceRepository(session, USER_A).entitlement()).to_dict()
    assert set(payload) == {
        "plan",
        "period_start",
        "period_end",
        "generation_limit",
        "generation_used",
        "generation_remaining",
        "download_mp3",
        "download_wav",
        "commercial_use",
    }
    for leak in ("user_id", "subscription_id", "slot_index", "provider"):
        assert leak not in str(payload)


def test_the_plan_table_matches_the_published_prices():
    """The figures the pricing page is allowed to show."""
    assert PLANS[PlanId.FREE].monthly_price_krw == 0
    assert PLANS[PlanId.BASIC].monthly_price_krw == 19_900
    assert PLANS[PlanId.PRO].monthly_price_krw == 29_900
    assert PLANS[PlanId.CREATOR].monthly_price_krw == 49_900
    assert plan_for("free").download_mp3 is False
    assert plan_for("free").commercial_use is False
    for paid in (PlanId.BASIC, PlanId.PRO, PlanId.CREATOR):
        assert PLANS[paid].download_mp3 and PLANS[paid].download_wav
        assert PLANS[paid].commercial_use is True
