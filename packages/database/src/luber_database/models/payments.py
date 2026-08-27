"""The billing record: checkouts, payments, events, failures, anomalies.

Phase 6 built `subscriptions` and the allowance ledger. This adds what a
real payment provider needs around them, and the constraints are where
most of the safety lives — an invariant enforced by the database holds
under concurrency, after a restart, and against code nobody has written
yet.

Four constraints carry the weight:

``uq_billing_events_provider_fingerprint``
    One row per provider notification. PayApp documents that feedback may
    be delivered more than once, and retries whenever the response is not
    exactly ``SUCCESS``. Insert-first-and-collide is what makes the tenth
    delivery a no-op rather than a tenth charge.

``uq_billing_payments_provider_payment``
    One payment row per PayApp ``mul_no``. Belt to the event table's
    braces: even if two different code paths processed the same money,
    only one payment row can exist.

``uq_one_open_checkout_per_user``
    A partial unique index over open checkout states. Two Subscribe
    clicks in the same instant cannot become two recurring contracts,
    because the second insert has nowhere to go. Partial indexes work on
    PostgreSQL and on the SQLite the tests run against, so the invariant
    is exercised by the suite rather than asserted in a comment.

``uq_subscriptions_provider_subscription``
    One local subscription per PayApp ``rebill_no``.

Nothing here stores card data. PayApp sends ``card_num`` and friends on
card payments and they are dropped at the boundary — we are not a card
processor, and a masked PAN in a billing table is a compliance liability
with no product use.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from luber_database.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class BillingCheckout(Base):
    """One attempt by one account to start a paid subscription.

    Written *before* PayApp is called. A checkout that dies between our
    request and PayApp's answer is then a visible CREATED row rather than
    nothing at all — which matters, because PayApp may have registered
    the contract and lost the response on the way back.
    """

    __tablename__ = "billing_checkouts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: Our own opaque identifier, sent to PayApp as var1 and echoed back
    #: in notifications. A correlation aid, never an authorisation: it
    #: reaches us from the public internet and is treated accordingly.
    correlation_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    #: Resolved server-side from the plan id. The browser sends a plan
    #: name and nothing else; this is the price we will check PayApp's
    #: reported amount against.
    plan_id: Mapped[str] = mapped_column(String(32), nullable=False)
    amount_krw: Mapped[int] = mapped_column(Integer, nullable=False)

    state: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="payapp")
    #: PayApp's recurring registration id, once it answers.
    provider_subscription_id: Mapped[str | None] = mapped_column(String(64), index=True)
    #: The hosted payment URL. Safe to hand to the browser — it is where
    #: PayApp wants the customer, and it carries no credential of ours.
    payurl: Mapped[str | None] = mapped_column(Text)

    #: Required by PayApp's `recvphone`. Collected at checkout and used
    #: for billing only; deliberately not part of the public profile.
    recvphone: Mapped[str | None] = mapped_column(String(20))

    #: Why a registration failed, for the operator. Provider-shaped text,
    #: never shown to the customer unmodified.
    failure_reason: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )
    #: Set when the checkout leaves an open state, so the partial unique
    #: index below stops applying to it.
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_billing_checkouts_user_state", "user_id", "state"),
        Index("ix_billing_checkouts_created", "created_at"),
        # The double-click guard, declared here as well as in the
        # migration. Metadata and migration must agree: a constraint that
        # exists only in the migration is one the test suite creates its
        # tables without, so the invariant ships to production untested
        # and the race passes locally every time.
        Index(
            "uq_one_open_checkout_per_user",
            "user_id",
            unique=True,
            postgresql_where=text("state IN ('CREATED', 'REGISTERED')"),
            sqlite_where=text("state IN ('CREATED', 'REGISTERED')"),
        ),
    )


#: Recorded outcomes for a payment row. There is no PENDING: a payment
#: row exists only once PayApp has told us something definite happened.
PAYMENT_SUCCEEDED = "SUCCEEDED"
PAYMENT_FAILED = "FAILED"


class BillingPayment(Base):
    """One thing PayApp told us happened to money. Append-only.

    Successful payment history is never updated in place and never
    deleted, including when a subscription is cancelled. A correction is
    a new row with its own event, so the record of what we believed and
    when survives the correction — which is the only way to answer a
    chargeback or a customer dispute honestly.
    """

    __tablename__ = "billing_payments"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subscription_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("subscriptions.id", ondelete="SET NULL"), index=True
    )
    checkout_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("billing_checkouts.id", ondelete="SET NULL")
    )

    plan_id: Mapped[str] = mapped_column(String(32), nullable=False)
    amount_krw: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)

    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="payapp")
    #: PayApp's `mul_no`. Null only for failure notifications, which do
    #: not always carry one.
    provider_payment_id: Mapped[str | None] = mapped_column(String(64))
    provider_subscription_id: Mapped[str | None] = mapped_column(String(64), index=True)

    #: When PayApp says the money moved — not when we processed it. A
    #: notification delivered two days late describes a payment that
    #: happened two days ago.
    paid_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    #: The subscription period this payment bought. Null on failures:
    #: a failed charge buys nothing, and writing a period would be the
    #: single most damaging lie this table could tell.
    period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    #: Provider's reason text on a failure, for the operator.
    failure_reason: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )

    __table_args__ = (
        # The financial invariant. One PayApp payment, one row, forever.
        UniqueConstraint(
            "provider", "provider_payment_id", name="uq_billing_payments_provider_payment"
        ),
        Index("ix_billing_payments_user_created", "user_id", "created_at"),
    )


#: Which endpoint an event arrived at. Kept distinct because the same
#: provider identifier can legitimately appear at both.
EVENT_FEEDBACK = "FEEDBACK"
EVENT_FAILURE = "FAILURE"

#: What we did about it — the audit answer to "why did nothing happen?"
OUTCOME_APPLIED = "APPLIED"
OUTCOME_DUPLICATE = "DUPLICATE"
OUTCOME_REJECTED = "REJECTED"
OUTCOME_IGNORED = "IGNORED"


class BillingEvent(Base):
    """Every notification we accepted, and what we did with it.

    The dedupe anchor. A notification is written here in the same
    transaction as its effect, so "recorded" and "applied" cannot come
    apart — a crash between them rolls back both, PayApp retries, and the
    second delivery applies exactly once.
    """

    __tablename__ = "billing_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="payapp")
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    #: Built from provider identifiers only, so a replay after a restart
    #: collapses onto the same row. See `payapp.notification`.
    fingerprint: Mapped[str] = mapped_column(String(200), nullable=False)

    pay_state: Mapped[int] = mapped_column(Integer, nullable=False)
    provider_payment_id: Mapped[str | None] = mapped_column(String(64), index=True)
    provider_subscription_id: Mapped[str | None] = mapped_column(String(64), index=True)
    correlation_id: Mapped[str | None] = mapped_column(String(64), index=True)
    amount_krw: Mapped[int | None] = mapped_column(Integer)

    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    #: Why, when the outcome was not APPLIED. An `AnomalyKind` value or a
    #: short reason — never a stack trace and never provider secrets.
    outcome_reason: Mapped[str | None] = mapped_column(String(64))

    #: The notification with secrets and card fields removed. Enough to
    #: reconstruct an incident; not enough to be a liability.
    payload: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)

    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "provider", "kind", "fingerprint", name="uq_billing_events_provider_fingerprint"
        ),
        Index("ix_billing_events_received", "received_at"),
    )


class BillingAnomaly(Base):
    """Something about money that a person needs to look at.

    Queryable rather than logged, because the questions asked after an
    incident are aggregate ones. Never grants entitlement — an anomaly is
    the record of a decision *not* made automatically.
    """

    __tablename__ = "billing_anomalies"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(48), nullable=False, index=True)

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    subscription_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("subscriptions.id", ondelete="SET NULL")
    )
    event_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("billing_events.id", ondelete="SET NULL")
    )
    provider_subscription_id: Mapped[str | None] = mapped_column(String(64), index=True)
    provider_payment_id: Mapped[str | None] = mapped_column(String(64))

    #: What we expected against what we got. Structured so an operator
    #: can compare without reading prose.
    detail: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )
    #: Set by an operator once the underlying problem is dealt with.
    #: Resolving an anomaly is a record that somebody looked, not an
    #: instruction to the system to do anything.
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_note: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("ix_billing_anomalies_open", "kind", "resolved_at"),)


__all__ = [
    "EVENT_FAILURE",
    "EVENT_FEEDBACK",
    "OUTCOME_APPLIED",
    "OUTCOME_DUPLICATE",
    "OUTCOME_IGNORED",
    "OUTCOME_REJECTED",
    "PAYMENT_FAILED",
    "PAYMENT_SUCCEEDED",
    "BillingAnomaly",
    "BillingCheckout",
    "BillingEvent",
    "BillingPayment",
]
