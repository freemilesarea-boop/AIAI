"""first-party acquisition attribution

Three tables, no changes to anything that already exists.

**Nothing is backfilled.** Every account and payment already in
production predates this, and there is no evidence anywhere about where
those people came from. The console reports them as 기존 회원 rather
than inventing a channel — a fabricated source is worse than a missing
one, because somebody will budget against it.

**No conversion table for payments.** A paid conversion is derived at
query time from `billing_payments`, which is already the verified
source of truth and already idempotent against PayApp retries. Writing
a second record of the same money would be a second thing to keep
correct.

**Deletion.** `acquisition_visitors.user_id` and
`acquisition_attributions.user_id` reference `users` with the project's
default (NO ACTION), which is what `generations` and `support_tickets`
do. Closing an account is anonymisation, not deletion — the row
survives — so nothing here can block it. `close_account` additionally
clears the visitor link, so a live cookie stops pointing at a closed
account.

**Retention is unset.** These tables hold no directly identifying data,
but BOORDA has no approved retention period, and inventing one in a
migration is not the place to decide it. The schema supports deletion by
`first_seen_at` / `created_at`, both indexed, whenever a period is
agreed. See docs/ACQUISITION_ANALYTICS.md — POLICY_REQUIRED.

Revision ID: 0023
Revises: 0022
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None

VALUE = 120
PATH = 200


def upgrade() -> None:
    op.create_table(
        "acquisition_visitors",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("visitor_key", sa.Uuid(), nullable=False),
        sa.Column("first_source", sa.String(length=VALUE), nullable=False),
        sa.Column("first_medium", sa.String(length=VALUE), nullable=False),
        sa.Column("first_campaign", sa.String(length=VALUE), nullable=True),
        sa.Column("first_content", sa.String(length=VALUE), nullable=True),
        sa.Column("first_term", sa.String(length=VALUE), nullable=True),
        sa.Column("last_source", sa.String(length=VALUE), nullable=True),
        sa.Column("last_medium", sa.String(length=VALUE), nullable=True),
        sa.Column("last_campaign", sa.String(length=VALUE), nullable=True),
        sa.Column("last_content", sa.String(length=VALUE), nullable=True),
        sa.Column("last_term", sa.String(length=VALUE), nullable=True),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_seen_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=True),
    )
    op.create_index(
        "ix_acquisition_visitors_visitor_key", "acquisition_visitors", ["visitor_key"], unique=True
    )
    op.create_index("ix_acquisition_visitors_user_id", "acquisition_visitors", ["user_id"])
    op.create_index(
        "ix_acquisition_visitors_last_seen_at", "acquisition_visitors", ["last_seen_at"]
    )
    op.create_index("ix_acq_visitors_first_seen", "acquisition_visitors", ["first_seen_at"])

    op.create_table(
        "acquisition_sessions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "visitor_id",
            sa.Uuid(),
            sa.ForeignKey("acquisition_visitors.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("source", sa.String(length=VALUE), nullable=False),
        sa.Column("medium", sa.String(length=VALUE), nullable=False),
        sa.Column("campaign", sa.String(length=VALUE), nullable=True),
        sa.Column("content", sa.String(length=VALUE), nullable=True),
        sa.Column("term", sa.String(length=VALUE), nullable=True),
        sa.Column("landing_path", sa.String(length=PATH), nullable=False, server_default="/"),
        sa.Column("referrer_host", sa.String(length=VALUE), nullable=True),
        sa.Column("is_direct", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_acquisition_sessions_visitor_id", "acquisition_sessions", ["visitor_id"])
    op.create_index("ix_acquisition_sessions_started_at", "acquisition_sessions", ["started_at"])
    op.create_index(
        "ix_acq_sessions_started_source", "acquisition_sessions", ["started_at", "source"]
    )
    op.create_index("ix_acq_sessions_source_medium", "acquisition_sessions", ["source", "medium"])
    op.create_index("ix_acq_sessions_campaign", "acquisition_sessions", ["campaign"])

    op.create_table(
        "acquisition_attributions",
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), primary_key=True),
        sa.Column(
            "visitor_id",
            sa.Uuid(),
            sa.ForeignKey("acquisition_visitors.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("first_source", sa.String(length=VALUE), nullable=False),
        sa.Column("first_medium", sa.String(length=VALUE), nullable=False),
        sa.Column("first_campaign", sa.String(length=VALUE), nullable=True),
        sa.Column("first_content", sa.String(length=VALUE), nullable=True),
        sa.Column("first_term", sa.String(length=VALUE), nullable=True),
        sa.Column("last_source", sa.String(length=VALUE), nullable=True),
        sa.Column("last_medium", sa.String(length=VALUE), nullable=True),
        sa.Column("last_campaign", sa.String(length=VALUE), nullable=True),
        sa.Column("last_content", sa.String(length=VALUE), nullable=True),
        sa.Column("last_term", sa.String(length=VALUE), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index(
        "ix_acquisition_attributions_visitor_id", "acquisition_attributions", ["visitor_id"]
    )
    op.create_index(
        "ix_acquisition_attributions_created_at", "acquisition_attributions", ["created_at"]
    )
    op.create_index(
        "ix_acq_attr_first", "acquisition_attributions", ["first_source", "first_medium"]
    )
    op.create_index("ix_acq_attr_last", "acquisition_attributions", ["last_source", "last_medium"])


def downgrade() -> None:
    op.drop_table("acquisition_attributions")
    op.drop_table("acquisition_sessions")
    op.drop_table("acquisition_visitors")
