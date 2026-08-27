"""Finding the payments that webhooks cannot tell us about.

Webhook delivery is not sufficient for a payment system, and the reason
is simple: a notification that is never delivered leaves no trace
anywhere. Nothing in the database says "PayApp tried to tell us
something and failed". The only way to notice is to have written down
what we expected and then to come back and check.

That is all this does. It compares our own records against our own
expectations and writes anomalies for the gaps. It never grants
entitlement, activates a subscription, or invents a payment.

**What it deliberately cannot do.** PayApp's published API
(https://docs.payapp.kr/dev_center01.html) documents commands to
register, cancel, pause and resume a recurring payment — ``rebillRegist``,
``rebillCancel``, ``rebillStop``, ``rebillStart`` — but no command that
authoritatively answers "what is the status of this rebill_no" or "list
the payments taken against it". So this cannot confirm from the provider
whether a charge happened; it can only notice that we have no record of
one and say so.

That limitation is recorded rather than worked around, because the
alternative — inventing a status-query endpoint and parsing whatever came
back — would produce a reconciliation system that appears authoritative
and is not. If PayApp publishes such an API, this is the one module that
needs to change.

Run it with ``scripts/ops/billing_reconcile.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID

from luber_billing.anomalies import AnomalyKind
from luber_billing.policy import RENEWAL_GRACE_HOURS
from luber_billing.states import CheckoutState, SubscriptionState
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from luber_database.billing_repository import checkouts_abandoned, subscriptions_overdue
from luber_database.models.billing import Subscription
from luber_database.models.payments import PAYMENT_SUCCEEDED, BillingAnomaly, BillingPayment


@dataclass
class ReconciliationReport:
    """What one pass found. Printed, and returned for tests to assert on."""

    missing_renewals: list[str] = field(default_factory=list)
    abandoned_checkouts: list[str] = field(default_factory=list)
    unresolved_past_due: list[str] = field(default_factory=list)
    #: True when nothing was written — either a dry run, or a clean pass.
    dry_run: bool = False

    @property
    def anomaly_count(self) -> int:
        return (
            len(self.missing_renewals)
            + len(self.abandoned_checkouts)
            + len(self.unresolved_past_due)
        )


async def _already_flagged(session: AsyncSession, kind: AnomalyKind, subscription_id: UUID) -> bool:
    """Whether this subscription already has an open anomaly of this kind.

    Without this, a nightly job turns one missed renewal into thirty
    identical rows and the operator queue becomes unreadable — which is
    the same as having no queue.
    """
    result = await session.execute(
        select(BillingAnomaly.id).where(
            BillingAnomaly.kind == kind.value,
            BillingAnomaly.subscription_id == subscription_id,
            BillingAnomaly.resolved_at.is_(None),
        )
    )
    return result.scalars().first() is not None


async def reconcile(
    session: AsyncSession, *, now: datetime | None = None, dry_run: bool = False
) -> ReconciliationReport:
    at = now or datetime.now(UTC)
    report = ReconciliationReport(dry_run=dry_run)
    grace = timedelta(hours=RENEWAL_GRACE_HOURS)

    # ── renewals that should have arrived ────────────────────────────
    for subscription in await subscriptions_overdue(session, now=at, grace=grace):
        # Look for a successful payment covering anything after the
        # period we know about. Its presence would mean the renewal
        # happened and something else is wrong; its absence is the
        # anomaly.
        paid = await session.execute(
            select(BillingPayment.id).where(
                BillingPayment.subscription_id == subscription.id,
                BillingPayment.status == PAYMENT_SUCCEEDED,
                BillingPayment.paid_at > subscription.period_end,
            )
        )
        if paid.scalars().first() is not None:
            continue
        if await _already_flagged(session, AnomalyKind.MISSING_EXPECTED_RENEWAL, subscription.id):
            continue
        report.missing_renewals.append(str(subscription.user_id))
        if not dry_run:
            session.add(
                BillingAnomaly(
                    kind=AnomalyKind.MISSING_EXPECTED_RENEWAL.value,
                    user_id=subscription.user_id,
                    subscription_id=subscription.id,
                    provider_subscription_id=subscription.provider_subscription_id,
                    detail={
                        "period_end": subscription.period_end.isoformat(),
                        "expected_renewal_at": (
                            subscription.next_renewal_at.isoformat()
                            if subscription.next_renewal_at
                            else ""
                        ),
                        "grace_hours": str(RENEWAL_GRACE_HOURS),
                        "note": (
                            "no successful payment recorded after the period ended; "
                            "PayApp publishes no status-query API to confirm from"
                        ),
                    },
                    detected_at=at,
                )
            )

    # ── checkouts nobody paid for ────────────────────────────────────
    for checkout in await checkouts_abandoned(session, now=at):
        report.abandoned_checkouts.append(str(checkout.user_id))
        if dry_run:
            continue
        # Closed, so the account's single checkout slot is released. An
        # abandoned attempt must not block every future subscription for
        # that person.
        checkout.state = CheckoutState.ABANDONED.value
        checkout.closed_at = at
        checkout.updated_at = at
        session.add(
            BillingAnomaly(
                kind=AnomalyKind.ABANDONED_CHECKOUT.value,
                user_id=checkout.user_id,
                provider_subscription_id=checkout.provider_subscription_id,
                detail={
                    "plan_id": checkout.plan_id,
                    "created_at": checkout.created_at.isoformat(),
                    "note": (
                        "registered with PayApp but never paid; first-cycle failure "
                        "is not notified to failurl, so this resolves by timeout"
                    ),
                },
                detected_at=at,
            )
        )
        # The subscription record that was waiting on this payment has
        # nothing to wait for any more. EXPIRED, not CANCELED: nobody
        # cancelled anything, it simply never happened.
        waiting = (
            await session.execute(
                select(Subscription).where(
                    Subscription.user_id == checkout.user_id,
                    Subscription.status == SubscriptionState.PENDING_INITIAL_PAYMENT.value,
                )
            )
        ).scalar_one_or_none()
        if waiting is not None:
            waiting.status = SubscriptionState.EXPIRED.value
            waiting.updated_at = at

    # ── past due that nobody resolved ────────────────────────────────
    stale = await session.execute(
        select(Subscription).where(
            Subscription.status == SubscriptionState.PAST_DUE.value,
            Subscription.period_end < at - grace,
        )
    )
    for subscription in stale.scalars().all():
        if await _already_flagged(session, AnomalyKind.UNRESOLVED_PAST_DUE, subscription.id):
            continue
        report.unresolved_past_due.append(str(subscription.user_id))
        if not dry_run:
            session.add(
                BillingAnomaly(
                    kind=AnomalyKind.UNRESOLVED_PAST_DUE.value,
                    user_id=subscription.user_id,
                    subscription_id=subscription.id,
                    provider_subscription_id=subscription.provider_subscription_id,
                    detail={
                        "plan_id": subscription.plan_id,
                        "period_end": subscription.period_end.isoformat(),
                    },
                    detected_at=at,
                )
            )

    if not dry_run:
        await session.commit()
    return report


__all__ = ["ReconciliationReport", "reconcile"]
