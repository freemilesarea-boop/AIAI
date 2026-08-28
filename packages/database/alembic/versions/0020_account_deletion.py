"""account deletion by anonymisation

Closing an account cannot be `DELETE FROM users`, and the schema is what
says so.

Three tables reference `users.id` with no `ondelete`, so PostgreSQL
defaults to NO ACTION: `generations`, `projects`, `reference_audio`. A
hard delete against an account that has ever made a song raises a
foreign-key violation and rolls back. The delete would fail for exactly
the users most likely to ask for it.

Six others cascade, and one of those is `billing_payments`. A hard
delete would therefore erase the record of money that changed hands —
the rows needed to answer a chargeback, reconcile a PayApp statement, or
explain a refund. Deleting an account must not delete the evidence that
it paid.

So the account is closed by anonymising it. `deleted_at` marks the row,
the email is replaced with a non-routable placeholder, `password_hash`
and `display_name` are cleared, and every session is removed. What
survives is a user row with no personal data in it, still satisfying the
foreign keys that point at it, still carrying the billing history.

`deleted_at` is what authentication filters on, so a closed account
cannot log in and a retained cookie authenticates nothing. Freeing the
original address also lets someone sign up again with it.

Additive: every existing row gets NULL, which means "not deleted", and
nothing reads the column until the account routes do.

Revision ID: 0020
Revises: 0019
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    # Every authenticated request joins users; the filter has to be cheap.
    op.create_index("ix_users_deleted_at", "users", ["deleted_at"])


def downgrade() -> None:
    op.drop_index("ix_users_deleted_at", table_name="users")
    op.drop_column("users", "deleted_at")
