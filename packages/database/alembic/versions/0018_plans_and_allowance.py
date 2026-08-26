"""subscriptions and the monthly allowance ledger

Phase 6 gives every account a plan and a monthly number of songs, and has
to enforce that number against a user pressing Generate ten times at once.

`subscriptions` is one row per user: a plan id and the window the
allowance is measured over. There are no payment-provider columns —
there is no provider yet, and columns invented for one would guess at an
integration nobody has chosen. A user with no row is Free, which is how
every account that predates this migration resolves without a backfill:
the resolver returns the least-privileged tier for a missing id, so
existing users keep working and none of them is silently upgraded.

`allowance_reservations` is the ledger, and its unique constraint is the
enforcement mechanism rather than a data-quality nicety.

Counting usage and then inserting is a race — ten concurrent requests all
read 199 and all insert, and the account gets 209 songs. Locking the
subscription row would close it on PostgreSQL and do nothing on SQLite,
which is what the test suite runs on, so the bug would ship green. Here
the slot index is part of the key: UNIQUE(user_id, period_start,
slot_index). Two requests racing for slot 199 cannot both hold it. The
loser recounts and tries the next slot, and the loop ends when the next
slot would be past the limit. The invariant lives in the schema, where
forgetting it is not possible.

One row per generation, so the ledger answers "which songs used this
month's allowance" directly. A failed generation moves to RELEASED and
stops counting; a completed one moves to CONSUMED. `plan_id` is copied
onto the row rather than joined, because the row records what was true
when the slot was taken and a later plan change must not rewrite history.

Both tables are additive and empty on creation. No existing row is
touched and no existing behaviour changes until the API begins reserving.

Revision ID: 0018
Revises: 0017
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "subscriptions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("plan_id", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="ACTIVE"),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_subscriptions_user_id", "subscriptions", ["user_id"])

    op.create_table(
        "allowance_reservations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "generation_id",
            sa.Uuid(),
            sa.ForeignKey("generations.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("plan_id", sa.String(length=32), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("slot_index", sa.Integer(), nullable=False),
        sa.Column("cost", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("state", sa.String(length=16), nullable=False, server_default="RESERVED"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_allowance_reservations_user_id", "allowance_reservations", ["user_id"])
    op.create_index("ix_allowance_reservations_state", "allowance_reservations", ["state"])
    op.create_index(
        "ix_allowance_user_period_state",
        "allowance_reservations",
        ["user_id", "period_start", "state"],
    )
    # The constraint the limit rests on.
    op.create_unique_constraint(
        "uq_allowance_slot_per_period",
        "allowance_reservations",
        ["user_id", "period_start", "slot_index"],
    )


def downgrade() -> None:
    op.drop_table("allowance_reservations")
    op.drop_index("ix_subscriptions_user_id", table_name="subscriptions")
    op.drop_table("subscriptions")
