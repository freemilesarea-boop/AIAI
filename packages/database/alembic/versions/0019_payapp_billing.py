"""PayApp recurring billing: checkouts, payments, events, anomalies

Phase 7 connects a real payment provider. Phase 6 already had
`subscriptions` and the allowance ledger, and this deliberately extends
them rather than replacing them: the entitlement architecture was right,
it simply had nothing telling it when a period had been paid for.

**Subscription gains provider linkage.** `provider_subscription_id` is
PayApp's `rebill_no` and is UNIQUE — one recurring contract maps to one
account, so a notification resolves to exactly one subscription and two
accounts can never share a contract. `auto_renew`, `canceled_at`,
`last_payment_at` and `next_renewal_at` make the difference between
"cancelled but paid up until the 28th" and "cancelled and finished"
representable, which a single status string could not do.

`next_renewal_at` exists for one reason: webhook delivery is not
sufficient for a payment system. A notification that is never delivered
leaves no trace anywhere, so the only way to notice a renewal that
should have happened is to have written down when we expected it.

**Four new tables, and their constraints are the safety.**

`uq_billing_events_provider_fingerprint` — PayApp documents that feedback
may arrive more than once, and retries whenever the response is not
exactly `SUCCESS`. The endpoint inserts the event and applies the effect
in one transaction; a redelivery collides here and becomes a no-op. This
is why the tenth delivery of a payment is not a tenth charge.

`uq_billing_payments_provider_payment` — one row per PayApp `mul_no`,
independent of the event table. Two constraints for one invariant,
because the cost of getting it wrong is charging a customer twice.

`uq_one_open_checkout_per_user` — a PARTIAL unique index over open
checkout states. Two Subscribe clicks in the same instant cannot become
two recurring contracts: the second insert has nowhere to go. Partial
indexes exist on PostgreSQL and on the SQLite the tests run against, so
this invariant is actually exercised rather than asserted in prose —
the same reasoning that made Phase 6 use a slot index instead of
SELECT FOR UPDATE.

`uq_subscriptions_provider_subscription` — one local subscription per
`rebill_no`.

**No card data anywhere.** PayApp sends `card_num` and similar on card
payments; they are dropped at the parsing boundary and there is no column
here that could hold one. We are not a card processor, and a masked PAN
in a billing table is a compliance liability with no product use.

Additive. Every existing subscription keeps working: `provider` is null
for the operator-assigned rows Phase 6 created, `auto_renew` defaults
true, and nothing reads the new columns unless a provider set them.

Revision ID: 0019
Revises: 0018
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── subscriptions: provider linkage ──────────────────────────────
    op.add_column("subscriptions", sa.Column("provider", sa.String(length=32), nullable=True))
    op.add_column(
        "subscriptions", sa.Column("provider_subscription_id", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "subscriptions",
        sa.Column("auto_renew", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column(
        "subscriptions", sa.Column("canceled_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "subscriptions", sa.Column("last_payment_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "subscriptions", sa.Column("next_renewal_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_unique_constraint(
        "uq_subscriptions_provider_subscription",
        "subscriptions",
        ["provider_subscription_id"],
    )
    # The reconciliation job's query: which live subscriptions are due.
    op.create_index(
        "ix_subscriptions_status_renewal", "subscriptions", ["status", "next_renewal_at"]
    )

    # ── checkouts ────────────────────────────────────────────────────
    op.create_table(
        "billing_checkouts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("correlation_id", sa.String(length=64), nullable=False, unique=True),
        sa.Column("plan_id", sa.String(length=32), nullable=False),
        sa.Column("amount_krw", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False, server_default="payapp"),
        sa.Column("provider_subscription_id", sa.String(length=64), nullable=True),
        sa.Column("payurl", sa.Text(), nullable=True),
        sa.Column("recvphone", sa.String(length=20), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_billing_checkouts_user_id", "billing_checkouts", ["user_id"])
    op.create_index("ix_billing_checkouts_state", "billing_checkouts", ["state"])
    op.create_index(
        "ix_billing_checkouts_provider_subscription_id",
        "billing_checkouts",
        ["provider_subscription_id"],
    )
    op.create_index("ix_billing_checkouts_user_state", "billing_checkouts", ["user_id", "state"])
    op.create_index("ix_billing_checkouts_created", "billing_checkouts", ["created_at"])
    # The double-click guard. One open checkout per account, enforced by
    # the database rather than by a disabled button.
    op.create_index(
        "uq_one_open_checkout_per_user",
        "billing_checkouts",
        ["user_id"],
        unique=True,
        postgresql_where=sa.text("state IN ('CREATED', 'REGISTERED')"),
        sqlite_where=sa.text("state IN ('CREATED', 'REGISTERED')"),
    )

    # ── payments ─────────────────────────────────────────────────────
    op.create_table(
        "billing_payments",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "subscription_id",
            sa.Uuid(),
            sa.ForeignKey("subscriptions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "checkout_id",
            sa.Uuid(),
            sa.ForeignKey("billing_checkouts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("plan_id", sa.String(length=32), nullable=False),
        sa.Column("amount_krw", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False, server_default="payapp"),
        sa.Column("provider_payment_id", sa.String(length=64), nullable=True),
        sa.Column("provider_subscription_id", sa.String(length=64), nullable=True),
        sa.Column("paid_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "provider", "provider_payment_id", name="uq_billing_payments_provider_payment"
        ),
    )
    op.create_index("ix_billing_payments_user_id", "billing_payments", ["user_id"])
    op.create_index("ix_billing_payments_subscription_id", "billing_payments", ["subscription_id"])
    op.create_index("ix_billing_payments_status", "billing_payments", ["status"])
    op.create_index(
        "ix_billing_payments_provider_subscription_id",
        "billing_payments",
        ["provider_subscription_id"],
    )
    op.create_index(
        "ix_billing_payments_user_created", "billing_payments", ["user_id", "created_at"]
    )

    # ── events ───────────────────────────────────────────────────────
    op.create_table(
        "billing_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("provider", sa.String(length=32), nullable=False, server_default="payapp"),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("fingerprint", sa.String(length=200), nullable=False),
        sa.Column("pay_state", sa.Integer(), nullable=False),
        sa.Column("provider_payment_id", sa.String(length=64), nullable=True),
        sa.Column("provider_subscription_id", sa.String(length=64), nullable=True),
        sa.Column("correlation_id", sa.String(length=64), nullable=True),
        sa.Column("amount_krw", sa.Integer(), nullable=True),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("outcome_reason", sa.String(length=64), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "received_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        # The idempotency anchor.
        sa.UniqueConstraint(
            "provider", "kind", "fingerprint", name="uq_billing_events_provider_fingerprint"
        ),
    )
    op.create_index(
        "ix_billing_events_provider_payment_id", "billing_events", ["provider_payment_id"]
    )
    op.create_index(
        "ix_billing_events_provider_subscription_id",
        "billing_events",
        ["provider_subscription_id"],
    )
    op.create_index("ix_billing_events_correlation_id", "billing_events", ["correlation_id"])
    op.create_index("ix_billing_events_received", "billing_events", ["received_at"])

    # ── anomalies ────────────────────────────────────────────────────
    op.create_table(
        "billing_anomalies",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("kind", sa.String(length=48), nullable=False),
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True
        ),
        sa.Column(
            "subscription_id",
            sa.Uuid(),
            sa.ForeignKey("subscriptions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "event_id",
            sa.Uuid(),
            sa.ForeignKey("billing_events.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("provider_subscription_id", sa.String(length=64), nullable=True),
        sa.Column("provider_payment_id", sa.String(length=64), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column(
            "detected_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution_note", sa.Text(), nullable=True),
    )
    op.create_index("ix_billing_anomalies_kind", "billing_anomalies", ["kind"])
    op.create_index("ix_billing_anomalies_user_id", "billing_anomalies", ["user_id"])
    op.create_index(
        "ix_billing_anomalies_provider_subscription_id",
        "billing_anomalies",
        ["provider_subscription_id"],
    )
    op.create_index("ix_billing_anomalies_open", "billing_anomalies", ["kind", "resolved_at"])


def downgrade() -> None:
    op.drop_table("billing_anomalies")
    op.drop_table("billing_events")
    op.drop_table("billing_payments")
    op.drop_index("uq_one_open_checkout_per_user", table_name="billing_checkouts")
    op.drop_table("billing_checkouts")
    op.drop_index("ix_subscriptions_status_renewal", table_name="subscriptions")
    op.drop_constraint("uq_subscriptions_provider_subscription", "subscriptions", type_="unique")
    for column in (
        "next_renewal_at",
        "last_payment_at",
        "canceled_at",
        "auto_renew",
        "provider_subscription_id",
        "provider",
    ):
        op.drop_column("subscriptions", column)
