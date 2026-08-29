"""What happens to money when notifications arrive badly.

The unit tests prove a notification can be read and judged. These prove
the part that only exists once there is a database: that the same event
arriving ten times charges once, that a wrong amount grants nothing, that
a failed renewal does not advance a period, and that two simultaneous
Subscribe clicks cannot become two recurring contracts.

Every one of these is a way a real payment system has lost real money.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from luber_billing.anomalies import AnomalyKind
from luber_billing.payapp.fake import feedback_payload
from luber_billing.payapp.notification import parse
from luber_billing.states import CheckoutState, SubscriptionState
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from luber_database import Base, create_session_factory
from luber_database.allowance_repository import AllowanceRepository
from luber_database.billing_reconciliation import reconcile
from luber_database.billing_repository import (
    BillingRepository,
    CheckoutAlreadyOpen,
    NoSubscriptionToCancel,
    NotificationProcessor,
    checkouts_abandoned,
    subscriptions_overdue,
)
from luber_database.models.billing import AllowanceReservation, Subscription
from luber_database.models.payments import (
    EVENT_FAILURE,
    EVENT_FEEDBACK,
    OUTCOME_APPLIED,
    OUTCOME_DUPLICATE,
    OUTCOME_IGNORED,
    OUTCOME_REJECTED,
    PAYMENT_FAILED,
    PAYMENT_SUCCEEDED,
    BillingAnomaly,
    BillingCheckout,
    BillingEvent,
    BillingPayment,
)
from luber_database.models.user import User
from luber_schemas.plans import PlanId

CREDS = {"userid": "boorda", "linkkey": "key-abc", "linkval": "val-xyz"}
PRO_PRICE = 29900

TEST_OWNER = uuid.UUID("21111111-1111-4111-8111-111111111111")
OTHER_OWNER = uuid.UUID("22222222-2222-4222-8222-222222222222")

#: Own fixtures rather than the package conftest's, following
#: `test_allowance.py`: this file needs `users`, `subscriptions` and the
#: four billing tables, and the shared list is scoped to generation.
TABLES = [
    User.__table__,
    Subscription.__table__,
    AllowanceReservation.__table__,
    BillingCheckout.__table__,
    BillingPayment.__table__,
    BillingEvent.__table__,
    BillingAnomaly.__table__,
]


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
async def session(engine):
    factory = create_session_factory(engine)
    async with factory() as session:
        for uid, email in ((TEST_OWNER, "a@boorda.test"), (OTHER_OWNER, "b@boorda.test")):
            session.add(User(id=uid, email=email, password_hash="x"))
        await session.commit()
        yield session


@pytest.fixture
def second_user() -> uuid.UUID:
    return OTHER_OWNER


async def _subscribed(session, owner=TEST_OWNER, plan=PlanId.PRO, rebill_no="900001", now=None):
    """An account that has registered but not paid. The realistic start.

    `now` anchors when the checkout was created. It matters for any test
    about *elapsed* time — abandonment is `created_at < now - window`, so
    a fixture created at the real current moment compared against a
    hardcoded reconciliation date silently changes meaning as the
    calendar advances. Left as None where a test does not care.
    """
    repository = BillingRepository(session, owner)
    checkout = await repository.create_checkout(plan_id=plan, recvphone="01012345678", now=now)
    await repository.mark_registered(
        checkout.id, rebill_no=rebill_no, payurl=f"https://payapp.kr/pay/{rebill_no}"
    )
    return repository, checkout


async def _notify(
    session,
    *,
    rebill_no="900001",
    price=PRO_PRICE,
    pay_state=4,
    mul_no="777",
    correlation_id=None,
    kind=EVENT_FEEDBACK,
    now=None,
):
    processor = NotificationProcessor(session)
    payload = feedback_payload(
        **CREDS,
        rebill_no=rebill_no,
        price=price,
        pay_state=pay_state,
        mul_no=mul_no,
        correlation_id=correlation_id,
    )
    return await processor.process(parse(payload), kind=kind, now=now)


def _utc(value):
    """SQLite hands timestamps back naive; PostgreSQL hands them back aware.

    Normalised here so an assertion about *when* a period started is not
    really an assertion about which database the suite happens to run on.
    """
    return value if value is None or value.tzinfo else value.replace(tzinfo=UTC)


async def _count(session, model) -> int:
    result = await session.execute(select(func.count()).select_from(model))
    return int(result.scalar_one())


# ── A. registration is not payment ───────────────────────────────────


async def test_registration_alone_grants_nothing(session) -> None:
    """The single most important test in the phase.

    `rebillRegist` succeeded, PayApp returned a payurl, and the account
    is still Free. Anything else means closing the tab buys a plan.
    """
    await _subscribed(session)

    plan = await AllowanceRepository(session, TEST_OWNER).effective_plan()

    assert plan.plan_id is PlanId.FREE
    assert plan.monthly_generation_limit == 20


async def test_a_pending_subscription_reports_its_state_honestly(session) -> None:
    repository, _ = await _subscribed(session)

    subscription = await repository.subscription()

    assert subscription is not None
    assert subscription.status == SubscriptionState.PENDING_INITIAL_PAYMENT.value


# ── B. payment activates, exactly once ───────────────────────────────


async def test_a_valid_payment_activates_the_plan(session) -> None:
    await _subscribed(session)

    result = await _notify(session)

    assert result.outcome == OUTCOME_APPLIED
    plan = await AllowanceRepository(session, TEST_OWNER).effective_plan()
    assert plan.plan_id is PlanId.PRO
    assert plan.monthly_generation_limit == 500


async def test_payment_establishes_the_allowance_period(session) -> None:
    await _subscribed(session)
    at = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)

    await _notify(session, now=at)

    entitlement = await AllowanceRepository(session, TEST_OWNER).entitlement(now=at)
    assert _utc(entitlement.period_start) == at
    assert entitlement.limit == 500


async def test_a_payment_writes_exactly_one_payment_row(session) -> None:
    await _subscribed(session)

    await _notify(session)

    assert await _count(session, BillingPayment) == 1


# ── C. idempotency ───────────────────────────────────────────────────


async def test_the_same_notification_ten_times_charges_once(session) -> None:
    """PayApp documents that feedback may be delivered more than once,
    and retries whenever the response is not exactly SUCCESS."""
    await _subscribed(session)

    outcomes = [(await _notify(session)).outcome for _ in range(10)]

    assert outcomes[0] == OUTCOME_APPLIED
    assert set(outcomes[1:]) == {OUTCOME_DUPLICATE}
    assert await _count(session, BillingPayment) == 1
    assert await _count(session, BillingEvent) == 1


async def test_a_duplicate_does_not_move_the_period(session) -> None:
    """The reset that must not happen twice.

    A second allowance period from one payment is a month of free songs
    for every redelivery PayApp makes.
    """
    await _subscribed(session)
    first = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
    await _notify(session, now=first)
    before = (await AllowanceRepository(session, TEST_OWNER).entitlement(now=first)).period_start

    await _notify(session, now=first + timedelta(days=3))

    after = (await AllowanceRepository(session, TEST_OWNER).entitlement(now=first)).period_start
    assert _utc(after) == _utc(before)


async def test_every_duplicate_is_still_acknowledged(session) -> None:
    """A duplicate is not an error. Refusing to acknowledge would make
    PayApp retry forever."""
    await _subscribed(session)
    await _notify(session)

    result = await _notify(session)

    assert result.acknowledge is True


# ── D. amount validation ─────────────────────────────────────────────


async def test_a_short_payment_does_not_activate_anything(session) -> None:
    """The adversarial case: PayApp reports less than the plan costs."""
    await _subscribed(session)

    result = await _notify(session, price=100)

    assert result.outcome == OUTCOME_REJECTED
    assert result.reason == AnomalyKind.AMOUNT_MISMATCH.value
    plan = await AllowanceRepository(session, TEST_OWNER).effective_plan()
    assert plan.plan_id is PlanId.FREE


async def test_an_overpayment_is_also_refused_rather_than_accepted(session) -> None:
    """Not "at least the price". A figure that is not the price is a
    signal that something is wrong, in either direction."""
    await _subscribed(session)

    result = await _notify(session, price=99900)

    assert result.outcome == OUTCOME_REJECTED
    assert (await AllowanceRepository(session, TEST_OWNER).effective_plan()).plan_id is PlanId.FREE


async def test_an_amount_mismatch_records_both_figures(session) -> None:
    """The difference is the whole signal. Overwriting the reported
    number with the expected one would destroy the evidence."""
    await _subscribed(session)

    await _notify(session, price=100)

    anomaly = (
        await session.execute(
            select(BillingAnomaly).where(BillingAnomaly.kind == AnomalyKind.AMOUNT_MISMATCH.value)
        )
    ).scalar_one()
    assert anomaly.detail["expected_krw"] == str(PRO_PRICE)
    assert anomaly.detail["reported_krw"] == "100"


async def test_a_missing_amount_is_not_treated_as_the_right_one(session) -> None:
    await _subscribed(session)
    processor = NotificationProcessor(session)
    payload = feedback_payload(**CREDS, rebill_no="900001", price="")

    result = await processor.process(parse(payload), kind=EVENT_FEEDBACK)

    assert result.outcome == OUTCOME_REJECTED
    assert (await AllowanceRepository(session, TEST_OWNER).effective_plan()).plan_id is PlanId.FREE


async def test_no_payment_row_is_written_for_a_mismatch(session) -> None:
    await _subscribed(session)

    await _notify(session, price=1)

    assert await _count(session, BillingPayment) == 0


# ── G. unknown subscriptions ─────────────────────────────────────────


async def test_a_payment_for_an_unknown_contract_grants_nothing(session) -> None:
    """Otherwise anyone who can post to the endpoint picks their own
    rebill_no and subscribes for free."""
    await _subscribed(session)

    result = await _notify(session, rebill_no="000000", mul_no="999")

    assert result.reason == AnomalyKind.UNKNOWN_REBILL.value
    assert (await AllowanceRepository(session, TEST_OWNER).effective_plan()).plan_id is PlanId.FREE


async def test_an_unknown_contract_is_recorded_as_an_anomaly(session) -> None:
    await _subscribed(session)

    await _notify(session, rebill_no="000000", mul_no="999")

    kinds = [a.kind for a in (await session.execute(select(BillingAnomaly))).scalars().all()]
    assert AnomalyKind.UNKNOWN_REBILL.value in kinds


# ── J/K. renewals ────────────────────────────────────────────────────


async def test_a_renewal_starts_exactly_one_new_period(session) -> None:
    await _subscribed(session)
    first = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=first, mul_no="777")

    renewal = first + timedelta(days=31)
    await _notify(session, now=renewal, mul_no="888")

    entitlement = await AllowanceRepository(session, TEST_OWNER).entitlement(now=renewal)
    assert _utc(entitlement.period_start) == renewal
    assert await _count(session, BillingPayment) == 2


async def test_a_duplicate_renewal_does_not_reset_the_allowance_again(session) -> None:
    await _subscribed(session)
    first = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=first, mul_no="777")
    renewal = first + timedelta(days=31)
    await _notify(session, now=renewal, mul_no="888")

    result = await _notify(session, now=renewal + timedelta(hours=2), mul_no="888")

    assert result.outcome == OUTCOME_DUPLICATE
    entitlement = await AllowanceRepository(session, TEST_OWNER).entitlement(now=renewal)
    assert _utc(entitlement.period_start) == renewal
    assert await _count(session, BillingPayment) == 2


async def test_a_renewal_keeps_the_plan_it_was_on(session) -> None:
    repository, _ = await _subscribed(session)
    first = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=first, mul_no="777")

    await _notify(session, now=first + timedelta(days=31), mul_no="888")

    subscription = await repository.subscription()
    assert subscription is not None
    assert subscription.plan_id == PlanId.PRO.value


# ── L/M. failures ────────────────────────────────────────────────────


async def test_a_failed_renewal_writes_no_successful_payment(session) -> None:
    await _subscribed(session)
    await _notify(session, now=datetime(2026, 8, 27, tzinfo=UTC), mul_no="777")

    await _notify(session, pay_state=99, mul_no="888", kind=EVENT_FAILURE)

    statuses = [p.status for p in (await session.execute(select(BillingPayment))).scalars().all()]
    assert statuses.count(PAYMENT_SUCCEEDED) == 1
    assert statuses.count(PAYMENT_FAILED) == 1


async def test_a_failed_renewal_does_not_advance_the_period(session) -> None:
    """The failure that must never look like a payment."""
    repository, _ = await _subscribed(session)
    first = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=first, mul_no="777")
    before = (await repository.subscription()).period_end  # type: ignore[union-attr]

    await _notify(session, pay_state=99, mul_no="888", kind=EVENT_FAILURE)

    after = (await repository.subscription()).period_end  # type: ignore[union-attr]
    assert _utc(after) == _utc(before)


async def test_a_failed_renewal_moves_the_subscription_to_past_due(session) -> None:
    repository, _ = await _subscribed(session)
    await _notify(session, now=datetime(2026, 8, 27, tzinfo=UTC), mul_no="777")

    await _notify(session, pay_state=99, mul_no="888", kind=EVENT_FAILURE)

    subscription = await repository.subscription()
    assert subscription is not None
    assert subscription.status == SubscriptionState.PAST_DUE.value


async def test_past_due_grants_nothing_even_inside_the_old_period(session) -> None:
    """A failed charge is the absence of a payment, not a payment."""
    await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")

    await _notify(session, pay_state=99, mul_no="888", kind=EVENT_FAILURE, now=at)

    plan = await AllowanceRepository(session, TEST_OWNER).effective_plan(now=at)
    assert plan.plan_id is PlanId.FREE


async def test_a_later_success_recovers_from_past_due_exactly_once(session) -> None:
    """A transient decline must not end the relationship."""
    await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")
    await _notify(session, pay_state=99, mul_no="888", kind=EVENT_FAILURE, now=at)

    recovered = at + timedelta(days=2)
    await _notify(session, now=recovered, mul_no="999")

    plan = await AllowanceRepository(session, TEST_OWNER).effective_plan(now=recovered)
    assert plan.plan_id is PlanId.PRO
    assert await _count(session, BillingPayment) == 3


async def test_a_duplicate_failure_is_recorded_once(session) -> None:
    await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")

    await _notify(session, pay_state=99, mul_no="888", kind=EVENT_FAILURE, now=at)
    result = await _notify(session, pay_state=99, mul_no="888", kind=EVENT_FAILURE, now=at)

    assert result.outcome == OUTCOME_DUPLICATE
    failed = [
        p
        for p in (await session.execute(select(BillingPayment))).scalars().all()
        if p.status == PAYMENT_FAILED
    ]
    assert len(failed) == 1


# ── out-of-order and replay ──────────────────────────────────────────


async def test_an_old_event_cannot_roll_the_subscription_backwards(session) -> None:
    """A notification delivered days late must not undo what came after."""
    repository, _ = await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")
    await repository.mark_canceled(now=at + timedelta(days=40))

    # The period has ended, so cancellation was terminal.
    result = await _notify(session, now=at + timedelta(days=41), mul_no="AAA")

    assert result.reason == AnomalyKind.INVALID_STATE_TRANSITION.value
    plan = await AllowanceRepository(session, TEST_OWNER).effective_plan(
        now=at + timedelta(days=41)
    )
    assert plan.plan_id is PlanId.FREE


async def test_a_replay_after_a_restart_is_still_idempotent(session) -> None:
    """Fingerprints are built from provider identifiers only, so nothing
    about our own process state can make a replay look new."""
    await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")

    # A "restart" is a new processor over the same data.
    replay = await NotificationProcessor(session).process(
        parse(feedback_payload(**CREDS, rebill_no="900001", price=PRO_PRICE, mul_no="777")),
        kind=EVENT_FEEDBACK,
        now=at + timedelta(days=1),
    )

    assert replay.outcome == OUTCOME_DUPLICATE
    assert await _count(session, BillingPayment) == 1


# ── I. one subscription per account ──────────────────────────────────


async def test_a_second_checkout_is_refused_while_one_is_open(session) -> None:
    repository, _ = await _subscribed(session)

    with pytest.raises(CheckoutAlreadyOpen):
        await repository.create_checkout(plan_id=PlanId.BASIC, recvphone="01012345678")


async def test_an_active_subscriber_cannot_open_a_second_contract(session) -> None:
    repository, _ = await _subscribed(session)
    await _notify(session)

    with pytest.raises(CheckoutAlreadyOpen):
        await repository.create_checkout(plan_id=PlanId.CREATOR, recvphone="01012345678")


async def test_two_racing_checkouts_produce_one_contract(tmp_path) -> None:
    """The double-click guard, under a real race.

    On its own database file rather than the shared in-memory one: a
    StaticPool hands every session the same connection, so a rollback in
    one tears down the others and the race never actually happens. A file
    gives each session its own connection, which is the situation the
    partial unique index exists for.
    """
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'race.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync: Base.metadata.create_all(
                sync,
                tables=[
                    User.__table__,
                    Subscription.__table__,
                    AllowanceReservation.__table__,
                    BillingCheckout.__table__,
                ],
            )
        )
    factory = create_session_factory(engine)
    owner = uuid.uuid4()
    async with factory() as setup:
        setup.add(User(id=owner, email=f"{owner}@example.test", password_hash="x"))
        await setup.commit()

    async def attempt() -> bool:
        async with factory() as session:
            try:
                await BillingRepository(session, owner).create_checkout(
                    plan_id=PlanId.PRO, recvphone="01012345678"
                )
                return True
            except CheckoutAlreadyOpen:
                return False

    results = await asyncio.gather(*(attempt() for _ in range(8)), return_exceptions=True)
    await engine.dispose()

    assert sum(1 for r in results if r is True) == 1


# ── N. cancellation ──────────────────────────────────────────────────


async def test_cancelling_keeps_the_period_already_paid_for(session) -> None:
    """PayApp's own semantics: cancellation stops the *next* charge and
    does not reverse the last one. Ending access now would take away
    something the customer paid for."""
    repository, _ = await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")

    await repository.mark_canceled(now=at + timedelta(days=1))

    plan = await AllowanceRepository(session, TEST_OWNER).effective_plan(now=at + timedelta(days=2))
    assert plan.plan_id is PlanId.PRO


async def test_cancelling_turns_off_the_next_charge(session) -> None:
    repository, _ = await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")

    subscription = await repository.mark_canceled(now=at + timedelta(days=1))

    assert subscription.auto_renew is False
    assert subscription.next_renewal_at is None
    assert subscription.status == SubscriptionState.CANCEL_PENDING.value


async def test_access_ends_when_the_cancelled_period_ends(session) -> None:
    repository, _ = await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")
    await repository.mark_canceled(now=at + timedelta(days=1))

    plan = await AllowanceRepository(session, TEST_OWNER).effective_plan(
        now=at + timedelta(days=60)
    )

    assert plan.plan_id is PlanId.FREE


async def test_cancelling_preserves_the_payment_history(session) -> None:
    """Billing history is not deleted when a subscription ends."""
    repository, _ = await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")

    await repository.mark_canceled(now=at + timedelta(days=1))

    assert await _count(session, BillingPayment) == 1


async def test_there_is_nothing_to_cancel_without_a_subscription(session) -> None:
    repository = BillingRepository(session, TEST_OWNER)

    with pytest.raises(NoSubscriptionToCancel):
        await repository.subscription_to_cancel()


# ── isolation ────────────────────────────────────────────────────────


async def test_one_accounts_payment_does_not_reach_another(session, second_user) -> None:
    await _subscribed(session)
    await _notify(session)

    other = AllowanceRepository(session, second_user)

    assert (await other.effective_plan()).plan_id is PlanId.FREE


async def test_payment_history_is_scoped_to_its_owner(session, second_user) -> None:
    await _subscribed(session)
    await _notify(session)

    assert len(await BillingRepository(session, TEST_OWNER).payments()) == 1
    assert await BillingRepository(session, second_user).payments() == []


async def test_a_second_account_cannot_reach_the_first_ones_contract(session, second_user) -> None:
    """The rebill_no is looked up from the caller's own record. There is
    no argument through which a stranger's contract could be named."""
    await _subscribed(session)
    await _notify(session)

    with pytest.raises(NoSubscriptionToCancel):
        await BillingRepository(session, second_user).subscription_to_cancel()


# ── reconciliation ───────────────────────────────────────────────────


async def test_a_missed_renewal_is_detectable(session) -> None:
    """The failure webhooks cannot report: a notification that was never
    delivered leaves no trace anywhere."""
    await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")

    overdue = await subscriptions_overdue(
        session, now=at + timedelta(days=40), grace=timedelta(hours=48)
    )

    assert len(overdue) == 1


async def test_a_subscription_inside_its_period_is_not_overdue(session) -> None:
    await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")

    overdue = await subscriptions_overdue(
        session, now=at + timedelta(days=5), grace=timedelta(hours=48)
    )

    assert overdue == []


async def test_an_unpaid_checkout_becomes_findable_after_the_window(session) -> None:
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _subscribed(session, now=at)

    stale = await checkouts_abandoned(session, now=at + timedelta(days=3))

    assert len(stale) == 1
    assert stale[0].state == CheckoutState.REGISTERED.value


async def test_a_paid_checkout_is_never_abandoned(session) -> None:
    await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")

    assert await checkouts_abandoned(session, now=at + timedelta(days=3)) == []


# ── recording ────────────────────────────────────────────────────────


async def test_every_accepted_notification_leaves_an_event(session) -> None:
    """ "We ignored it" and "we never received it" must not look alike."""
    await _subscribed(session)

    await _notify(session, pay_state=1, mul_no="111")

    event = (await session.execute(select(BillingEvent))).scalars().one()
    assert event.pay_state == 1


async def test_stored_events_carry_no_secrets(session) -> None:
    await _subscribed(session)

    await _notify(session)

    for event in (await session.execute(select(BillingEvent))).scalars().all():
        assert "linkkey" not in event.payload
        assert "linkval" not in event.payload


async def test_a_successful_payment_records_what_it_bought(session) -> None:
    await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)

    await _notify(session, now=at, mul_no="777")

    payment = (await session.execute(select(BillingPayment))).scalars().one()
    assert payment.amount_krw == PRO_PRICE
    assert payment.plan_id == PlanId.PRO.value
    assert payment.provider_payment_id == "777"
    assert _utc(payment.period_start) == at


async def test_a_failed_payment_records_no_period(session) -> None:
    """A failed charge buys nothing. Writing a period here would be the
    most damaging lie this table could tell."""
    await _subscribed(session)
    await _notify(session, now=datetime(2026, 8, 27, tzinfo=UTC), mul_no="777")

    await _notify(session, pay_state=99, mul_no="888", kind=EVENT_FAILURE)

    failed = [
        p
        for p in (await session.execute(select(BillingPayment))).scalars().all()
        if p.status == PAYMENT_FAILED
    ]
    assert failed[0].period_start is None
    assert failed[0].period_end is None


# ── reconciliation ───────────────────────────────────────────────────


async def test_reconciliation_flags_a_renewal_that_never_arrived(session) -> None:
    """The gap webhooks cannot report, found by comparing what we
    expected against what we recorded."""
    await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")

    report = await reconcile(session, now=at + timedelta(days=40))

    assert len(report.missing_renewals) == 1
    kinds = [a.kind for a in (await session.execute(select(BillingAnomaly))).scalars().all()]
    assert AnomalyKind.MISSING_EXPECTED_RENEWAL.value in kinds


async def test_reconciliation_does_not_flag_the_same_gap_twice(session) -> None:
    """A nightly job must not turn one missed renewal into thirty rows —
    an unreadable queue is the same as no queue."""
    await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")
    await reconcile(session, now=at + timedelta(days=40))

    second = await reconcile(session, now=at + timedelta(days=41))

    assert second.missing_renewals == []


async def test_reconciliation_never_grants_anything(session) -> None:
    """The whole point of recording an anomaly is that no entitlement
    decision is made automatically."""
    await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")

    await reconcile(session, now=at + timedelta(days=40))

    plan = await AllowanceRepository(session, TEST_OWNER).effective_plan(
        now=at + timedelta(days=40)
    )
    assert plan.plan_id is PlanId.FREE


async def test_reconciliation_writes_no_payment_rows(session) -> None:
    await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")

    await reconcile(session, now=at + timedelta(days=40))

    assert await _count(session, BillingPayment) == 1


async def test_a_dry_run_changes_nothing(session) -> None:
    await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")

    report = await reconcile(session, now=at + timedelta(days=40), dry_run=True)

    assert len(report.missing_renewals) == 1
    assert await _count(session, BillingAnomaly) == 0


async def test_an_abandoned_checkout_releases_the_slot(session) -> None:
    """Otherwise one attempt nobody paid for blocks every future
    subscription for that account."""
    at = datetime(2026, 8, 27, tzinfo=UTC)
    repository, _ = await _subscribed(session, now=at)

    await reconcile(session, now=at + timedelta(days=3))

    assert await repository.open_checkout() is None
    subscription = await repository.subscription()
    assert subscription is not None
    # EXPIRED, not CANCELED: nobody cancelled anything.
    assert subscription.status == SubscriptionState.EXPIRED.value


async def test_the_account_can_try_again_after_abandoning(session) -> None:
    at = datetime(2026, 8, 27, tzinfo=UTC)
    repository, _ = await _subscribed(session, now=at)
    await reconcile(session, now=at + timedelta(days=3))

    retry = await repository.create_checkout(plan_id=PlanId.PRO, recvphone="01012345678")

    assert retry.state == CheckoutState.CREATED.value


async def test_an_unresolved_past_due_is_flagged(session) -> None:
    await _subscribed(session)
    at = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=at, mul_no="777")
    await _notify(session, pay_state=99, mul_no="888", kind=EVENT_FAILURE, now=at)

    report = await reconcile(session, now=at + timedelta(days=40))

    assert len(report.unresolved_past_due) == 1


# ── failure injection ────────────────────────────────────────────────
#
# Payment systems fail in ugly ways, and the property that matters is
# that the final state is the same however the failure is shaped.


async def test_a_persistence_failure_is_not_acknowledged(session, monkeypatch) -> None:
    """The rule that keeps a payment from being lost.

    If the effect cannot be written, the endpoint must not answer
    SUCCESS — PayApp would never send it again. Refusing means a retry,
    and the retry finds no event row because this transaction rolled
    back, so it applies cleanly.
    """
    await _subscribed(session)
    processor = NotificationProcessor(session)

    async def explode(*args, **kwargs):
        raise RuntimeError("database went away")

    monkeypatch.setattr(processor, "_apply", explode)
    result = await processor.process(
        parse(feedback_payload(**CREDS, rebill_no="900001", price=PRO_PRICE)),
        kind=EVENT_FEEDBACK,
    )

    assert result.acknowledge is False
    assert await _count(session, BillingPayment) == 0


async def test_a_failed_transaction_leaves_no_event_to_swallow_the_retry(
    session, monkeypatch
) -> None:
    """The subtle half of the same rule.

    A design that recorded the event and *then* applied the effect would
    leave the event behind after a crash — and PayApp's retry would
    collide on the fingerprint and be dismissed as a duplicate, losing
    the payment silently. One transaction is what prevents that.
    """
    await _subscribed(session)
    processor = NotificationProcessor(session)

    async def explode(*args, **kwargs):
        raise RuntimeError("database went away")

    monkeypatch.setattr(processor, "_apply", explode)
    await processor.process(
        parse(feedback_payload(**CREDS, rebill_no="900001", price=PRO_PRICE)),
        kind=EVENT_FEEDBACK,
    )

    assert await _count(session, BillingEvent) == 0
    # And the retry now works.
    retry = await _notify(session)
    assert retry.outcome == OUTCOME_APPLIED


async def test_a_callback_arriving_before_the_user_returns_is_fine(session) -> None:
    """The common race. The browser redirect and the server-to-server
    notification are independent, and either may land first."""
    await _subscribed(session)

    await _notify(session)

    # Whatever the return page asks afterwards, the answer is already
    # correct — it reads the database, not the URL.
    assert (await AllowanceRepository(session, TEST_OWNER).effective_plan()).plan_id is PlanId.PRO


async def test_a_callback_after_the_user_closes_the_browser_still_applies(session) -> None:
    """Nothing about activation depends on a browser being open."""
    await _subscribed(session)

    result = await _notify(session, now=datetime(2026, 8, 27, tzinfo=UTC) + timedelta(hours=6))

    assert result.outcome == OUTCOME_APPLIED


async def test_a_cancellation_notification_is_recorded_and_grants_nothing(session) -> None:
    """pay_state=9 is an approval cancellation — a refund. It is not a
    payment, and recording it is how a refund stops being invisible."""
    await _subscribed(session)

    result = await _notify(session, pay_state=9, mul_no="rev1")

    assert result.outcome == OUTCOME_IGNORED
    assert await _count(session, BillingPayment) == 0
    assert (await AllowanceRepository(session, TEST_OWNER).effective_plan()).plan_id is PlanId.FREE


async def test_a_pending_deposit_notification_grants_nothing(session) -> None:
    """pay_state=10 is a virtual account awaiting a deposit. Money has
    not moved."""
    await _subscribed(session)

    await _notify(session, pay_state=10, mul_no="vb1")

    assert (await AllowanceRepository(session, TEST_OWNER).effective_plan()).plan_id is PlanId.FREE


async def test_notifications_arriving_out_of_order_converge(session) -> None:
    """Renewal first, then a late redelivery of the original payment.

    The final state must be the renewal's, not whichever arrived last.
    """
    await _subscribed(session)
    first = datetime(2026, 8, 27, tzinfo=UTC)
    await _notify(session, now=first, mul_no="777")
    renewal = first + timedelta(days=31)
    await _notify(session, now=renewal, mul_no="888")

    # The original, redelivered days late.
    late = await _notify(session, now=renewal + timedelta(hours=1), mul_no="777")

    assert late.outcome == OUTCOME_DUPLICATE
    entitlement = await AllowanceRepository(session, TEST_OWNER).entitlement(now=renewal)
    assert _utc(entitlement.period_start) == renewal
