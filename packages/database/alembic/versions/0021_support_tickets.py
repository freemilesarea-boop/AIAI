"""customer support inquiries and their replies

Two tables. `support_tickets` holds what a customer submitted;
`support_replies` holds the conversation on it.

`support_replies` is created empty and nothing writes to it yet — there
is no operator interface. It exists now because adding a reply model
later means migrating live support history, and a schema that assumed
one message per ticket is awkward to reshape once people are waiting on
answers in it. The cost of the empty table is a few kilobytes.

**`reference` is what the customer sees.** A short `SUP-XXXXXXXX` string,
unique, unrelated to the UUID primary key. It is not a substitute for
authorisation — every query is scoped to the owner — but it means a
support address in a shared inbox is not also a list of identifiers to
try, and the primary key never leaves the server.

**`user_id` restricts rather than cascades**, by omitting `ondelete`,
matching `generations` and `projects`. Closing an account is
anonymisation (migration 0020), so the ticket stays attached to a row
that no longer names anyone. Cascading would delete a complaint at the
moment the complainant left, which is when it matters most — and it
would also make account closure depend on support data, which is the
coupling 0020 exists to avoid.

`status` is operator-owned. No customer-facing request carries it.

Revision ID: 0021
Revises: 0020
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "support_tickets",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("reference", sa.String(length=16), nullable=False, unique=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("subject", sa.String(length=200), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("context_url", sa.String(length=500), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="OPEN"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_support_tickets_reference", "support_tickets", ["reference"])
    op.create_index("ix_support_tickets_user_id", "support_tickets", ["user_id"])
    op.create_index("ix_support_tickets_status", "support_tickets", ["status"])
    op.create_index("ix_support_tickets_created_at", "support_tickets", ["created_at"])
    # The list query: this user's tickets, newest first.
    op.create_index("ix_support_tickets_user_created", "support_tickets", ["user_id", "created_at"])
    # The operator queue, once one exists.
    op.create_index(
        "ix_support_tickets_status_created", "support_tickets", ["status", "created_at"]
    )

    op.create_table(
        "support_replies",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "ticket_id",
            sa.Uuid(),
            sa.ForeignKey("support_tickets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("author_type", sa.String(length=16), nullable=False),
        sa.Column("author_user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_support_replies_ticket_id", "support_replies", ["ticket_id"])
    op.create_index("ix_support_replies_created_at", "support_replies", ["created_at"])
    op.create_index(
        "ix_support_replies_ticket_created", "support_replies", ["ticket_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_table("support_replies")
    op.drop_table("support_tickets")
