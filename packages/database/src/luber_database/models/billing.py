"""Which plan an account is on, and what it has spent this period.

Two tables, and the shape of the second is the interesting one.

``subscriptions`` is one row per user naming a plan and the period the
allowance is measured over. It carries no payment provider fields: there
is no payment provider yet, and inventing columns for one would guess at
an integration nobody has chosen.

``allowance_reservations`` is the ledger. Every generation that is
allowed to start takes a row here, and the row records which generation
consumed which slot of which period — so "which songs used A's March
allowance" is a query rather than an archaeology exercise. A failed
generation releases its row, and a released row does not count.

**Why a slot index.** Enforcing a limit by counting and then inserting is
a race: ten concurrent requests all count 199 and all insert. Locking the
user row would fix it on PostgreSQL and quietly do nothing on SQLite,
which is what the tests run on. So the slot is part of the key:
``UNIQUE(user_id, period_start, slot_index)``. Two requests racing for
the same slot cannot both have it — the database refuses the second, the
caller recounts, and the loop ends when the next slot would be past the
limit. The invariant is held by the schema rather than by everyone
remembering to take a lock.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from luber_database.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Subscription(Base):
    """The plan an account is on. One row per user, or none.

    No row means Free: every account that predates plans resolves safely
    without a backfill, and ``plan_for(None)`` returns the least
    privileged tier by design.
    """

    __tablename__ = "subscriptions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    #: A `PlanId` value. Stored as its stable string, never a display name.
    plan_id: Mapped[str] = mapped_column(String(32), nullable=False)
    #: A `SubscriptionState` value. Phase 6 wrote only ACTIVE; Phase 7
    #: turned this into the real state machine, because a payment
    #: provider makes the difference between "registered", "paid",
    #: "renewal failed" and "cancelled but still paid up" the whole
    #: question. See `luber_billing.states`.
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")

    # ── provider linkage (Phase 7) ─────────────────────────────────
    #
    #: Which payment provider holds the recurring contract. Null for a
    #: subscription assigned by an operator script, which is how every
    #: pre-payment account came to have one.
    provider: Mapped[str | None] = mapped_column(String(32))
    #: PayApp's `rebill_no`. Unique: one recurring contract maps to one
    #: local subscription, so a notification can be resolved to exactly
    #: one account and no account can end up sharing a contract.
    #:
    #: Never accepted from a client. The browser asks to cancel *its*
    #: subscription; the server looks this up.
    provider_subscription_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    #: Whether PayApp will charge again. Turned off by cancellation
    #: without ending the period already paid for.
    auto_renew: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    #: When the user asked to stop. Distinct from the period ending:
    #: cancelling on the 3rd of a period paid to the 28th means access
    #: until the 28th, and both dates matter afterwards.
    canceled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: The last confirmed payment, denormalised so "is this account paid
    #: up" is one row rather than a join under load.
    last_payment_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: When PayApp is next expected to charge. What the reconciliation
    #: job compares against to notice a renewal that never arrived —
    #: the failure mode webhooks cannot detect, because an undelivered
    #: notification leaves no trace anywhere.
    next_renewal_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    #: The allowance window. Calendar months for Free today, but stored
    #: as explicit bounds so a billing provider can later define periods
    #: that start on the subscription date instead.
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )


#: A reservation that has not yet been settled. The generation is in
#: flight; the slot is held.
RESERVED = "RESERVED"
#: The generation completed. The slot is spent.
CONSUMED = "CONSUMED"
#: The generation failed. The slot is returned and does not count.
RELEASED = "RELEASED"


class AllowanceReservation(Base):
    """One generation's claim on one slot of one allowance period."""

    __tablename__ = "allowance_reservations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: The generation this slot was taken for. Unique: one generation can
    #: never consume two slots, however many times a retry re-enters the
    #: reservation path.
    generation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("generations.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    #: Denormalised from the subscription so the ledger stays readable
    #: after a plan change: this row records what was true when the slot
    #: was taken, not what is true now.
    plan_id: Mapped[str] = mapped_column(String(32), nullable=False)
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    #: Zero-based position within the period. The uniqueness of
    #: (user, period_start, slot_index) is what makes the limit hold
    #: under concurrency — see the module docstring.
    slot_index: Mapped[int] = mapped_column(Integer, nullable=False)
    #: What this generation cost. One today; a premium model could cost
    #: more without changing the schema.
    cost: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    state: Mapped[str] = mapped_column(String(16), nullable=False, default=RESERVED, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )
    #: When the row moved out of RESERVED. Null while in flight.
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # The invariant. Held by the database, not by callers.
        UniqueConstraint(
            "user_id", "period_start", "slot_index", name="uq_allowance_slot_per_period"
        ),
        # The hot query: how much of this period has this user spent.
        Index("ix_allowance_user_period_state", "user_id", "period_start", "state"),
    )


__all__ = [
    "CONSUMED",
    "RELEASED",
    "RESERVED",
    "AllowanceReservation",
    "Subscription",
]
