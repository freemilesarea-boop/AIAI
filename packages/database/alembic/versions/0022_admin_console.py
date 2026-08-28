"""admin roles, download events, audit log, email campaigns

The operator console needs four things the schema did not have.

**`users.role`.** There was no permission model at all. The alternative
an admin console usually reaches for — comparing the signed-in address
against a constant — is a permission that cannot be revoked, cannot be
audited, and transfers to whoever ends up owning that mailbox. A column
can be changed, and every change goes through `admin_audit_logs`.

Defaults to `USER`, so the migration grants nobody anything. The first
`SUPER_ADMIN` is created deliberately by `scripts/ops/grant_admin.py`,
which needs shell access to the machine holding the database — the same
reasoning as the plan-assignment script.

**`download_events`.** Nothing recorded downloads. The counter has to be
a row rather than an increment because "how many downloads this month"
and "which songs did this account take" are the same question asked two
ways, and a bare counter answers only the first.

**`admin_audit_logs`.** Append-only in practice. An operator action with
no record of who took it is the failure mode this table exists to
prevent.

**`admin_email_campaigns`.** The composed message and the resolved
recipient count. Sending is a separate concern with no provider behind
it yet.

Also `support_tickets.admin_note` — an internal note the customer never
sees, which is why it lives on the ticket rather than in the reply
thread that will eventually be shown to them.

Every change is additive. No existing row is modified, and nothing reads
the new column until the admin routes do.

Revision ID: 0022
Revises: 0021
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── role ─────────────────────────────────────────────────────────
    op.add_column(
        "users",
        sa.Column("role", sa.String(length=24), nullable=False, server_default="USER"),
    )
    # Listing administrators scans by role; they are a tiny fraction of
    # the table, which is exactly when an index earns its place.
    op.create_index("ix_users_role", "users", ["role"])

    # ── downloads ────────────────────────────────────────────────────
    op.create_table(
        "download_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column(
            "generation_id",
            sa.Uuid(),
            sa.ForeignKey("generations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("asset_kind", sa.String(length=32), nullable=False),
        sa.Column("plan_id", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_download_events_user_id", "download_events", ["user_id"])
    op.create_index("ix_download_events_generation_id", "download_events", ["generation_id"])
    op.create_index("ix_download_events_created_at", "download_events", ["created_at"])
    op.create_index("ix_download_events_user_created", "download_events", ["user_id", "created_at"])
    op.create_index(
        "ix_download_events_generation", "download_events", ["generation_id", "created_at"]
    )

    # ── audit ────────────────────────────────────────────────────────
    op.create_table(
        "admin_audit_logs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("actor_user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("target_user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("action", sa.String(length=48), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_admin_audit_logs_actor_user_id", "admin_audit_logs", ["actor_user_id"])
    op.create_index("ix_admin_audit_logs_target_user_id", "admin_audit_logs", ["target_user_id"])
    op.create_index("ix_admin_audit_logs_action", "admin_audit_logs", ["action"])
    op.create_index("ix_admin_audit_logs_created_at", "admin_audit_logs", ["created_at"])
    op.create_index(
        "ix_admin_audit_actor_created", "admin_audit_logs", ["actor_user_id", "created_at"]
    )
    op.create_index("ix_admin_audit_action_created", "admin_audit_logs", ["action", "created_at"])

    # ── email campaigns ──────────────────────────────────────────────
    op.create_table(
        "admin_email_campaigns",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("created_by", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("subject", sa.String(length=200), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("audience_type", sa.String(length=16), nullable=False),
        sa.Column("audience_plan_id", sa.String(length=32), nullable=True),
        sa.Column("recipient_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="DRAFT"),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_admin_email_campaigns_created_by", "admin_email_campaigns", ["created_by"])
    op.create_index("ix_admin_email_campaigns_created_at", "admin_email_campaigns", ["created_at"])
    op.create_index("ix_admin_campaigns_created", "admin_email_campaigns", ["created_at"])

    # ── internal support note ────────────────────────────────────────
    op.add_column("support_tickets", sa.Column("admin_note", sa.Text(), nullable=True))

    # ── aggregation indexes ──────────────────────────────────────────
    #
    # The dashboard groups payments and generations by day. Both tables
    # already index `created_at`; these are the composites the
    # status-filtered aggregates actually scan.
    op.create_index("ix_billing_payments_status_paid", "billing_payments", ["status", "paid_at"])
    op.create_index("ix_generations_status_created", "generations", ["status", "created_at"])
    op.create_index("ix_users_created_at", "users", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_users_created_at", table_name="users")
    op.drop_index("ix_generations_status_created", table_name="generations")
    op.drop_index("ix_billing_payments_status_paid", table_name="billing_payments")
    op.drop_column("support_tickets", "admin_note")
    op.drop_table("admin_email_campaigns")
    op.drop_table("admin_audit_logs")
    op.drop_table("download_events")
    op.drop_index("ix_users_role", table_name="users")
    op.drop_column("users", "role")
