"""Legacy ownership: internal anchor, ownership FKs, backfill, NOT NULL.

Revision ID: 0014
Revises: 0013

Assigns every pre-authentication row to one internal owner, then makes
ownership mandatory.

The anchor is not an account. Its ``password_hash`` is NULL, which login
refuses outright, and its email is taken, which stops signup claiming
it. Both properties come from Part 1's existing behaviour rather than a
new mechanism; both are covered by tests.

Ordering is the safety property here. The backfill runs before the
constraint, and an explicit check runs between them, so the migration
either completes or fails leaving the database exactly as it was. It
writes one column on three tables and inserts one row: no lineage edge,
storage key, digest or audio byte is touched.

See docs/PHASE20A_LEGACY_OWNERSHIP_PLAN.md.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Deterministic — uuid5(NAMESPACE_DNS, LEGACY_OWNER_EMAIL). Written
#: literally so the value is auditable in the diff and identical on every
#: database that runs this migration.
LEGACY_OWNER_ID = "e3c4d3cd-d86f-52f2-91b7-2b97f5011653"
LEGACY_OWNER_EMAIL = "legacy-system@internal.luber"

#: The three tables a user owns directly. Everything else reaches an
#: owner through ``generation_id``; duplicating the column there would
#: create a second source of truth that can disagree with the first.
OWNED_TABLES = ("generations", "projects", "reference_audio")


def upgrade() -> None:
    connection = op.get_bind()

    # 1. The anchor. ON CONFLICT so re-running is a no-op rather than an
    #    error, which is what makes this migration idempotent.
    connection.execute(
        sa.text(
            """
            INSERT INTO users (id, email, password_hash, display_name)
            VALUES (:id, :email, NULL, :display_name)
            ON CONFLICT (id) DO NOTHING
            """
        ),
        {
            "id": LEGACY_OWNER_ID,
            "email": LEGACY_OWNER_EMAIL,
            "display_name": "Legacy system data (pre-authentication)",
        },
    )

    # 2. reference_audio has no ownership column yet. Nullable first, so
    #    the backfill has somewhere to write.
    op.add_column("reference_audio", sa.Column("user_id", sa.Uuid(), nullable=True))

    # 3. Backfill, only where ownership is absent. An existing owner is
    #    never overwritten — this migration claims orphans, not rows.
    for table in OWNED_TABLES:
        connection.execute(
            sa.text(f"UPDATE {table} SET user_id = :owner WHERE user_id IS NULL"),
            {"owner": LEGACY_OWNER_ID},
        )

    # 4. Prove it before relying on it. Reaching NOT NULL with a NULL
    #    still present would fail anyway; failing here says which table.
    for table in OWNED_TABLES:
        remaining = connection.execute(
            sa.text(f"SELECT count(*) FROM {table} WHERE user_id IS NULL")
        ).scalar_one()
        if remaining:
            raise RuntimeError(
                f"{table} still has {remaining} row(s) without an owner; "
                "refusing to add the NOT NULL constraint"
            )

    # 5. Foreign keys and indexes. The FK is what stops a user_id
    #    pointing at nobody; the index is what every future
    #    ownership-scoped query will use.
    for table in OWNED_TABLES:
        op.create_foreign_key(f"fk_{table}_user_id_users", table, "users", ["user_id"], ["id"])
        op.create_index(f"ix_{table}_user_id", table, ["user_id"])

    # 6. Only now, with every row owned and the FK in place.
    for table in OWNED_TABLES:
        op.alter_column(table, "user_id", existing_type=sa.Uuid(), nullable=False)


def downgrade() -> None:
    """Undo the constraints. Delete no product data.

    The anchor user is deliberately left behind. Removing it would mean
    choosing between cascading — which destroys generations — and
    orphaning rows that still reference it. One unauthenticatable row is
    a cheaper outcome than either.
    """
    for table in OWNED_TABLES:
        op.alter_column(table, "user_id", existing_type=sa.Uuid(), nullable=True)
        op.drop_index(f"ix_{table}_user_id", table_name=table)
        op.drop_constraint(f"fk_{table}_user_id_users", table, type_="foreignkey")

    connection = op.get_bind()
    for table in ("generations", "projects"):
        connection.execute(
            sa.text(f"UPDATE {table} SET user_id = NULL WHERE user_id = :owner"),
            {"owner": LEGACY_OWNER_ID},
        )

    op.drop_column("reference_audio", "user_id")
