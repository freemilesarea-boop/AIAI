"""Billing persistence, and the one function that decides about money.

Two halves with very different rules.

The **owner-scoped half** (`BillingRepository`) follows the same
discipline as every other repository here: the account is bound at
construction, never passed per method, so a route cannot read or cancel
another account's subscription by forgetting an argument.

The **provider half** (`NotificationProcessor`) cannot be owner-scoped,
because a PayApp notification arrives with no session and names an
account only indirectly, through identifiers that reached us over the
public internet. It is therefore written to trust nothing: it resolves
the account from *our* records via the provider identifier, checks the
amount against *our* plan table, and refuses anything that does not line
up — recording an anomaly rather than guessing.

**The idempotency design.** Every notification is written to
`billing_events` in the same transaction as its effect, and the unique
constraint on the fingerprint is what makes a redelivery a no-op. The
sequence matters:

    insert event  →  apply effect  →  commit  →  answer SUCCESS

A crash anywhere before the commit rolls back both halves, PayApp retries
(we register with `checkretry=y`), and the retry applies exactly once.
A crash after the commit but before the response also ends in a retry,
which collides on the fingerprint and returns SUCCESS having done
nothing. There is no ordering of failures that produces two charges or a
silent loss, which is the property the whole phase rests on.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from luber_billing.anomalies import AnomalyKind
from luber_billing.payapp.notification import PayAppNotification, event_fingerprint
from luber_billing.policy import CHECKOUT_ABANDON_HOURS, paid_period
from luber_billing.states import (
    OPEN_CHECKOUT_STATES,
    CheckoutState,
    SubscriptionState,
    may_start_new,
    may_transition,
)
from sqlalchemy import and_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from luber_database.models.billing import Subscription
from luber_database.models.payments import (
    EVENT_FAILURE,
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
from luber_schemas.plans import PlanId, plan_for

logger = logging.getLogger(__name__)

PROVIDER_PAYAPP = "payapp"


def _as_utc(value: datetime) -> datetime:
    """Timestamps come back naive from SQLite and aware from PostgreSQL.

    Normalised at every comparison rather than assumed, because the one
    place this bites is a `>` between a stored period end and `now` — and
    getting it wrong there decides whether a cancelling customer keeps
    the month they paid for.
    """
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class CheckoutAlreadyOpen(RuntimeError):
    """The account already has a checkout or subscription in flight.

    Raised rather than silently reusing, so the caller decides whether to
    hand back the existing payurl or refuse. Either is safe; quietly
    starting a second recurring contract is not.
    """

    def __init__(self, checkout: BillingCheckout | None = None) -> None:
        super().__init__("this account already has an open checkout")
        self.checkout = checkout


class NoSubscriptionToCancel(RuntimeError):
    """Nothing of this account's is cancellable."""


def new_correlation_id() -> str:
    """An opaque identifier for one checkout.

    Random rather than derived from the account, because it is sent to
    PayApp as `var1` and comes back through a public endpoint. A
    correlation id that encoded a user id would leak one, and a
    guessable one would let a caller aim a forged notification at a
    specific account. It authorises nothing either way — it only says
    which of our own rows to look at first.
    """
    return f"boorda_{secrets.token_urlsafe(24)}"


@dataclass(frozen=True)
class ProcessingResult:
    """What the endpoint should do, and what actually happened."""

    #: True when the endpoint may answer `SUCCESS`. False only when the
    #: notification could not be safely recorded at all — PayApp then
    #: retries, which is what we want.
    acknowledge: bool
    outcome: str
    reason: str | None = None
    subscription_id: UUID | None = None


class BillingRepository:
    """One account's billing records."""

    def __init__(self, session: AsyncSession, owner: UUID) -> None:
        self._session = session
        self._owner = owner

    @property
    def owner(self) -> UUID:
        return self._owner

    # ── subscription ───────────────────────────────────────────────

    async def subscription(self) -> Subscription | None:
        result = await self._session.execute(
            select(Subscription).where(Subscription.user_id == self._owner)
        )
        return result.scalar_one_or_none()

    async def open_checkout(self) -> BillingCheckout | None:
        result = await self._session.execute(
            select(BillingCheckout)
            .where(
                BillingCheckout.user_id == self._owner,
                BillingCheckout.state.in_([s.value for s in OPEN_CHECKOUT_STATES]),
            )
            .order_by(BillingCheckout.created_at.desc())
        )
        return result.scalars().first()

    # ── checkout ───────────────────────────────────────────────────

    async def create_checkout(
        self, *, plan_id: PlanId, recvphone: str, now: datetime | None = None
    ) -> BillingCheckout:
        """Open a checkout, or refuse because one is already open.

        The row is written *before* PayApp is called. A process that dies
        between here and PayApp's answer leaves a visible CREATED row
        rather than nothing — which matters, because PayApp may have
        registered the contract and lost the response on the way back,
        and reconciliation needs something to find.

        The amount is resolved from the plan table here and stored. The
        browser sent a plan name; it never sent a price, and this is the
        figure the notification's amount will be checked against.
        """
        plan = plan_for(plan_id)
        if not plan.is_paid:
            raise ValueError("free plans do not go through checkout")

        existing_subscription = await self.subscription()
        if (
            existing_subscription is not None
            # A row with no provider contract cannot be double-billed —
            # it is an operator-assigned plan from Phase 6, which is a
            # comp rather than a subscription. Blocking checkout on one
            # would mean an account given a plan by hand could never
            # start paying, and the rule exists to prevent two recurring
            # contracts, not two rows.
            and existing_subscription.provider_subscription_id is not None
            and not may_start_new(SubscriptionState(existing_subscription.status))
        ):
            raise CheckoutAlreadyOpen()

        already = await self.open_checkout()
        if already is not None:
            raise CheckoutAlreadyOpen(already)

        at = now or datetime.now(UTC)
        checkout = BillingCheckout(
            user_id=self._owner,
            correlation_id=new_correlation_id(),
            plan_id=plan.plan_id.value,
            amount_krw=plan.monthly_price_krw,
            state=CheckoutState.CREATED.value,
            provider=PROVIDER_PAYAPP,
            recvphone=recvphone,
            created_at=at,
            updated_at=at,
        )
        self._session.add(checkout)
        try:
            await self._session.commit()
        except IntegrityError as exc:
            # The partial unique index caught a second simultaneous
            # click. The database is the guard here, not the button:
            # two requests can pass the check above at the same instant
            # and only one can land.
            await self._session.rollback()
            raise CheckoutAlreadyOpen(await self.open_checkout()) from exc

        stored = await self._session.execute(
            select(BillingCheckout).where(BillingCheckout.id == checkout.id)
        )
        return stored.scalar_one()

    async def mark_registered(
        self,
        checkout_id: UUID,
        *,
        rebill_no: str,
        payurl: str,
        now: datetime | None = None,
    ) -> Subscription:
        """PayApp accepted the registration. Nobody has paid yet.

        Creates (or reclaims) the account's subscription record in
        PENDING_INITIAL_PAYMENT with no period at all. A period is what
        grants access, and no money has moved — so there is deliberately
        nothing here for the entitlement resolver to find.
        """
        at = now or datetime.now(UTC)
        checkout = await self._owned_checkout(checkout_id)
        checkout.state = CheckoutState.REGISTERED.value
        checkout.provider_subscription_id = rebill_no
        checkout.payurl = payurl
        checkout.updated_at = at

        subscription = await self.subscription()
        if subscription is None:
            subscription = Subscription(
                user_id=self._owner,
                plan_id=checkout.plan_id,
                status=SubscriptionState.PENDING_INITIAL_PAYMENT.value,
                # A zero-width window: syntactically a period, but one
                # that contains no moment, so `current_period` can never
                # mistake it for paid time.
                period_start=at,
                period_end=at,
                provider=PROVIDER_PAYAPP,
                provider_subscription_id=rebill_no,
                auto_renew=True,
            )
            self._session.add(subscription)
        else:
            subscription.plan_id = checkout.plan_id
            subscription.status = SubscriptionState.PENDING_INITIAL_PAYMENT.value
            subscription.provider = PROVIDER_PAYAPP
            subscription.provider_subscription_id = rebill_no
            subscription.auto_renew = True
            subscription.canceled_at = None
            subscription.period_start = at
            subscription.period_end = at
            subscription.updated_at = at

        await self._session.commit()
        refreshed = await self.subscription()
        assert refreshed is not None
        return refreshed

    async def mark_registration_failed(
        self, checkout_id: UUID, *, reason: str, now: datetime | None = None
    ) -> None:
        at = now or datetime.now(UTC)
        checkout = await self._owned_checkout(checkout_id)
        checkout.state = CheckoutState.REGISTRATION_FAILED.value
        # Truncated: provider text is unbounded and this column is for an
        # operator's eyes, not for storing an essay.
        checkout.failure_reason = reason[:500]
        checkout.updated_at = at
        checkout.closed_at = at
        await self._session.commit()

    async def _owned_checkout(self, checkout_id: UUID) -> BillingCheckout:
        result = await self._session.execute(
            select(BillingCheckout).where(
                BillingCheckout.id == checkout_id,
                BillingCheckout.user_id == self._owner,
            )
        )
        checkout = result.scalar_one_or_none()
        if checkout is None:
            raise LookupError("checkout not found for this account")
        return checkout

    # ── cancellation ───────────────────────────────────────────────

    async def subscription_to_cancel(self) -> Subscription:
        """This account's cancellable subscription, with its rebill_no.

        The browser never supplies a `rebill_no`. It asks to cancel *its*
        subscription and the server looks the identifier up here — which
        is the only reason a caller cannot cancel somebody else's
        contract by guessing a number.
        """
        subscription = await self.subscription()
        if subscription is None or subscription.provider_subscription_id is None:
            raise NoSubscriptionToCancel("no provider subscription for this account")
        state = SubscriptionState(subscription.status)
        if state in {SubscriptionState.CANCELED, SubscriptionState.EXPIRED}:
            raise NoSubscriptionToCancel("this subscription is already finished")
        return subscription

    async def mark_canceled(self, *, now: datetime | None = None) -> Subscription:
        """Record that PayApp will not charge again.

        The paid period is left exactly as it is. PayApp's own semantics
        are that cancellation prevents the *next* charge and does not
        reverse the last one, so ending access now would take away
        something the customer paid for.
        """
        at = now or datetime.now(UTC)
        subscription = await self.subscription_to_cancel()
        subscription.auto_renew = False
        subscription.canceled_at = at
        subscription.next_renewal_at = None
        current = SubscriptionState(subscription.status)
        # Still inside the paid period → keep serving it, stop renewing.
        # Otherwise there is nothing left to preserve.
        target = (
            SubscriptionState.CANCEL_PENDING
            if _as_utc(subscription.period_end) > at
            else SubscriptionState.CANCELED
        )
        if may_transition(current, target):
            subscription.status = target.value
        subscription.updated_at = at
        await self._session.commit()
        refreshed = await self.subscription()
        assert refreshed is not None
        return refreshed

    # ── history ────────────────────────────────────────────────────

    async def payments(self, *, limit: int = 50) -> list[BillingPayment]:
        """This account's own payment history. Never another's."""
        result = await self._session.execute(
            select(BillingPayment)
            .where(BillingPayment.user_id == self._owner)
            .order_by(BillingPayment.paid_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


class NotificationProcessor:
    """Applies PayApp notifications. Trusts none of them.

    Not owner-scoped, and cannot be: the caller has no session. Every
    account it touches is resolved from our own records through the
    provider identifier, never from anything the notification asserts.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _subscription_for(self, rebill_no: str | None) -> Subscription | None:
        if not rebill_no:
            return None
        result = await self._session.execute(
            select(Subscription).where(Subscription.provider_subscription_id == rebill_no)
        )
        return result.scalar_one_or_none()

    async def _checkout_for(
        self, *, rebill_no: str | None, correlation_id: str | None
    ) -> BillingCheckout | None:
        clauses = []
        if rebill_no:
            clauses.append(BillingCheckout.provider_subscription_id == rebill_no)
        if correlation_id:
            clauses.append(BillingCheckout.correlation_id == correlation_id)
        if not clauses:
            return None
        result = await self._session.execute(
            select(BillingCheckout).where(or_(*clauses)).order_by(BillingCheckout.created_at.desc())
        )
        return result.scalars().first()

    def _anomaly(
        self,
        kind: AnomalyKind,
        *,
        event: BillingEvent | None = None,
        subscription: Subscription | None = None,
        user_id: UUID | None = None,
        detail: dict[str, Any] | None = None,
        notification: PayAppNotification | None = None,
    ) -> BillingAnomaly:
        anomaly = BillingAnomaly(
            kind=kind.value,
            user_id=user_id or (subscription.user_id if subscription else None),
            subscription_id=subscription.id if subscription else None,
            event_id=event.id if event else None,
            provider_subscription_id=notification.rebill_no if notification else None,
            provider_payment_id=notification.mul_no if notification else None,
            detail={k: str(v) for k, v in (detail or {}).items()},
        )
        self._session.add(anomaly)
        return anomaly

    async def process(
        self,
        notification: PayAppNotification,
        *,
        kind: str,
        now: datetime | None = None,
    ) -> ProcessingResult:
        """Record and apply one notification, exactly once.

        Returns without raising for every outcome the system understands,
        including refusals — a rejected notification is still durably
        recorded, so acknowledging it is honest and a retry would change
        nothing. Only an outright failure to persist propagates, and that
        is deliberate: PayApp must retry rather than be told SUCCESS for
        something we did not write down.
        """
        at = now or datetime.now(UTC)
        fingerprint = event_fingerprint(notification, kind=kind, received_date=at)

        event = BillingEvent(
            provider=PROVIDER_PAYAPP,
            kind=kind,
            fingerprint=fingerprint,
            pay_state=notification.pay_state,
            provider_payment_id=notification.mul_no,
            provider_subscription_id=notification.rebill_no,
            correlation_id=notification.correlation_id,
            amount_krw=notification.price,
            outcome=OUTCOME_IGNORED,
            payload=notification.extra,
            received_at=at,
        )
        self._session.add(event)
        try:
            # Flush rather than commit: the effect below must land in the
            # same transaction as this row, or a crash between them would
            # leave an event recorded with nothing applied — and the
            # retry would then be swallowed as a duplicate.
            await self._session.flush()
        except IntegrityError:
            await self._session.rollback()
            logger.info(
                "payapp notification already processed",
                extra={"payapp_event_kind": kind, "payapp_fingerprint": fingerprint},
            )
            return ProcessingResult(acknowledge=True, outcome=OUTCOME_DUPLICATE)

        try:
            result = await self._apply(notification, event=event, kind=kind, at=at)
            event.outcome = result.outcome
            event.outcome_reason = result.reason
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            logger.exception(
                "failed to process payapp notification",
                extra={"payapp_event_kind": kind, "payapp_fingerprint": fingerprint},
            )
            # Not acknowledged. PayApp retries (we register with
            # checkretry=y), and the retry finds no event row because
            # this transaction rolled back — so it applies cleanly.
            return ProcessingResult(acknowledge=False, outcome=OUTCOME_REJECTED, reason="RETRY")
        return result

    async def _apply(
        self,
        notification: PayAppNotification,
        *,
        event: BillingEvent,
        kind: str,
        at: datetime,
    ) -> ProcessingResult:
        if notification.is_payment_complete:
            return await self._apply_payment(notification, event=event, at=at)
        if notification.is_recurring_failure or kind == EVENT_FAILURE:
            return await self._apply_failure(notification, event=event, at=at)
        # Requests, pending deposits and cancellations are recorded and
        # otherwise left alone. Doing nothing is the correct handling;
        # doing nothing *silently* is not, which is why the row exists.
        return ProcessingResult(
            acknowledge=True, outcome=OUTCOME_IGNORED, reason=f"PAY_STATE_{notification.pay_state}"
        )

    async def _apply_payment(
        self, notification: PayAppNotification, *, event: BillingEvent, at: datetime
    ) -> ProcessingResult:
        subscription = await self._subscription_for(notification.rebill_no)
        checkout = await self._checkout_for(
            rebill_no=notification.rebill_no, correlation_id=notification.correlation_id
        )

        if subscription is None:
            # A payment for a recurring contract we have no record of.
            # Never granted: the alternative is that anyone who can post
            # to this endpoint picks their own rebill_no.
            self._anomaly(
                AnomalyKind.UNKNOWN_REBILL,
                event=event,
                notification=notification,
                user_id=checkout.user_id if checkout else None,
                detail={
                    "rebill_no": notification.rebill_no or "",
                    "mul_no": notification.mul_no or "",
                },
            )
            return ProcessingResult(
                acknowledge=True,
                outcome=OUTCOME_REJECTED,
                reason=AnomalyKind.UNKNOWN_REBILL.value,
            )

        plan = plan_for(subscription.plan_id)
        expected = plan.monthly_price_krw

        # The amount check, before anything is granted and before the
        # payment row exists. A mismatch is never "corrected": the
        # difference is the entire signal, and overwriting the reported
        # figure with the expected one destroys the evidence.
        if notification.price != expected:
            self._anomaly(
                AnomalyKind.AMOUNT_MISMATCH,
                event=event,
                subscription=subscription,
                notification=notification,
                detail={
                    "expected_krw": expected,
                    "reported_krw": notification.price,
                    "plan_id": subscription.plan_id,
                },
            )
            logger.error(
                "payapp reported an unexpected amount; subscription not activated",
                extra={
                    "payapp_rebill_no": notification.rebill_no,
                    "expected_krw": expected,
                    "reported_krw": notification.price,
                },
            )
            return ProcessingResult(
                acknowledge=True,
                outcome=OUTCOME_REJECTED,
                reason=AnomalyKind.AMOUNT_MISMATCH.value,
            )

        current = SubscriptionState(subscription.status)
        if not may_transition(current, SubscriptionState.ACTIVE):
            # A late or replayed event arriving after the user cancelled
            # and the period ended. Recorded, never applied: nothing from
            # the network may revive access somebody ended.
            self._anomaly(
                AnomalyKind.INVALID_STATE_TRANSITION,
                event=event,
                subscription=subscription,
                notification=notification,
                detail={"from": current.value, "to": SubscriptionState.ACTIVE.value},
            )
            return ProcessingResult(
                acknowledge=True,
                outcome=OUTCOME_REJECTED,
                reason=AnomalyKind.INVALID_STATE_TRANSITION.value,
            )

        period_start, period_end = paid_period(at)

        payment = BillingPayment(
            user_id=subscription.user_id,
            subscription_id=subscription.id,
            checkout_id=checkout.id if checkout else None,
            plan_id=subscription.plan_id,
            amount_krw=notification.price,
            status=PAYMENT_SUCCEEDED,
            provider=PROVIDER_PAYAPP,
            provider_payment_id=notification.mul_no,
            provider_subscription_id=notification.rebill_no,
            paid_at=at,
            period_start=period_start,
            period_end=period_end,
        )
        self._session.add(payment)

        subscription.status = SubscriptionState.ACTIVE.value
        subscription.period_start = period_start
        subscription.period_end = period_end
        subscription.last_payment_at = at
        # Only expect another charge if the user has not cancelled.
        subscription.next_renewal_at = period_end if subscription.auto_renew else None
        subscription.updated_at = at

        if checkout is not None and checkout.state in {
            CheckoutState.CREATED.value,
            CheckoutState.REGISTERED.value,
        }:
            checkout.state = CheckoutState.COMPLETED.value
            checkout.updated_at = at
            checkout.closed_at = at

        logger.info(
            "payapp payment confirmed; subscription active",
            extra={
                "payapp_rebill_no": notification.rebill_no,
                "payapp_mul_no": notification.mul_no,
                "plan_id": subscription.plan_id,
                "amount_krw": notification.price,
            },
        )
        return ProcessingResult(
            acknowledge=True, outcome=OUTCOME_APPLIED, subscription_id=subscription.id
        )

    async def _apply_failure(
        self, notification: PayAppNotification, *, event: BillingEvent, at: datetime
    ) -> ProcessingResult:
        """A recurring charge failed.

        Three things must not happen here, and each has cost somebody
        somewhere a great deal of money: no successful payment row, no
        allowance period advance, and no change to the period already
        paid for. A failed renewal is the *absence* of a payment, and the
        customer keeps whatever they last paid for until it runs out.
        """
        subscription = await self._subscription_for(notification.rebill_no)
        if subscription is None:
            self._anomaly(
                AnomalyKind.UNKNOWN_REBILL,
                event=event,
                notification=notification,
                detail={"rebill_no": notification.rebill_no or "", "kind": EVENT_FAILURE},
            )
            return ProcessingResult(
                acknowledge=True,
                outcome=OUTCOME_REJECTED,
                reason=AnomalyKind.UNKNOWN_REBILL.value,
            )

        reason = notification.extra.get("errorMessage") or notification.extra.get("ResultMsg")
        self._session.add(
            BillingPayment(
                user_id=subscription.user_id,
                subscription_id=subscription.id,
                plan_id=subscription.plan_id,
                amount_krw=notification.price or plan_for(subscription.plan_id).monthly_price_krw,
                status=PAYMENT_FAILED,
                provider=PROVIDER_PAYAPP,
                provider_payment_id=notification.mul_no,
                provider_subscription_id=notification.rebill_no,
                paid_at=at,
                # No period. A failed charge buys nothing, and writing a
                # period here would be the most damaging lie this table
                # could tell.
                period_start=None,
                period_end=None,
                failure_reason=(reason or "recurring approval failed")[:500],
            )
        )

        current = SubscriptionState(subscription.status)
        if may_transition(current, SubscriptionState.PAST_DUE):
            subscription.status = SubscriptionState.PAST_DUE.value
            subscription.updated_at = at
            # period_start/period_end deliberately untouched.
        else:
            self._anomaly(
                AnomalyKind.INVALID_STATE_TRANSITION,
                event=event,
                subscription=subscription,
                notification=notification,
                detail={"from": current.value, "to": SubscriptionState.PAST_DUE.value},
            )

        logger.warning(
            "payapp recurring payment failed",
            extra={
                "payapp_rebill_no": notification.rebill_no,
                "plan_id": subscription.plan_id,
            },
        )
        return ProcessingResult(
            acknowledge=True, outcome=OUTCOME_APPLIED, subscription_id=subscription.id
        )

    async def record_rejection(
        self, *, reason: str, payload: dict[str, str], now: datetime | None = None
    ) -> None:
        """A notification that failed authentication.

        Written as an anomaly with no event row: it did not come from
        PayApp as far as we can tell, so recording it as a provider event
        would put an attacker's data in the billing ledger.
        """
        at = now or datetime.now(UTC)
        kind = (
            AnomalyKind.INVALID_USERID
            if reason == AnomalyKind.INVALID_USERID.value
            else AnomalyKind.INVALID_LINKVAL
            if reason == AnomalyKind.INVALID_LINKVAL.value
            else AnomalyKind.MALFORMED_NOTIFICATION
        )
        self._session.add(
            BillingAnomaly(
                kind=kind.value,
                detail={"reason": reason, **{k: v for k, v in list(payload.items())[:10]}},
                detected_at=at,
            )
        )
        await self._session.commit()


async def find_anomalies(
    session: AsyncSession, *, unresolved_only: bool = True, limit: int = 100
) -> list[BillingAnomaly]:
    """Operational query. Not exposed to ordinary users anywhere."""
    statement = select(BillingAnomaly).order_by(BillingAnomaly.detected_at.desc()).limit(limit)
    if unresolved_only:
        statement = statement.where(BillingAnomaly.resolved_at.is_(None))
    result = await session.execute(statement)
    return list(result.scalars().all())


async def subscriptions_overdue(
    session: AsyncSession, *, now: datetime, grace: timedelta
) -> list[Subscription]:
    """Live subscriptions whose renewal should have arrived and did not.

    The check webhooks cannot perform. An undelivered notification leaves
    no trace anywhere, so the only way to notice one is to compare what
    we expected against what we recorded.
    """
    cutoff = now - grace
    result = await session.execute(
        select(Subscription).where(
            Subscription.status == SubscriptionState.ACTIVE.value,
            Subscription.provider_subscription_id.is_not(None),
            Subscription.auto_renew.is_(True),
            Subscription.period_end < cutoff,
        )
    )
    return list(result.scalars().all())


async def checkouts_abandoned(
    session: AsyncSession, *, now: datetime, hours: int = CHECKOUT_ABANDON_HOURS
) -> list[BillingCheckout]:
    """Checkouts nobody ever paid for.

    Closed by reconciliation so the account's single checkout slot is
    released — otherwise one abandoned attempt would block every future
    subscription for that account.
    """
    cutoff = now - timedelta(hours=hours)
    result = await session.execute(
        select(BillingCheckout).where(
            and_(
                BillingCheckout.state.in_([s.value for s in OPEN_CHECKOUT_STATES]),
                BillingCheckout.created_at < cutoff,
            )
        )
    )
    return list(result.scalars().all())


__all__ = [
    "PROVIDER_PAYAPP",
    "BillingRepository",
    "CheckoutAlreadyOpen",
    "NoSubscriptionToCancel",
    "NotificationProcessor",
    "ProcessingResult",
    "checkouts_abandoned",
    "find_anomalies",
    "new_correlation_id",
    "subscriptions_overdue",
]
