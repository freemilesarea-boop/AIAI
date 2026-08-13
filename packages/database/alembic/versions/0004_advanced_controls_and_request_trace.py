"""advanced song controls and provider request trace

Phase 8 adds three musical parameters the pinned ACE-Step build
genuinely accepts (bpm, key_scale, time_signature), the request trace
that records what was actually sent to the provider, the pre-flight
advisories shown to the user, and the parent link used by "generate
again" and variations.

Every column is nullable. Phase 3-7 rows predate all of it, and a
generation with no recorded trace is a real historical state — not
something to invent a default for. Reading code must treat NULL as
"not recorded", never as "empty".

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

PARENT_FK = "fk_generations_parent_generation_id"
PARENT_INDEX = "ix_generations_parent_generation_id"


def upgrade() -> None:
    op.add_column("generations", sa.Column("bpm", sa.Integer(), nullable=True))
    op.add_column("generations", sa.Column("key_scale", sa.String(length=20), nullable=True))
    op.add_column("generations", sa.Column("time_signature", sa.String(length=8), nullable=True))
    # JSON documents kept as Text so the same schema runs on SQLite in
    # unit tests and PostgreSQL in production.
    op.add_column("generations", sa.Column("request_trace", sa.Text(), nullable=True))
    op.add_column("generations", sa.Column("advisories", sa.Text(), nullable=True))
    op.add_column("generations", sa.Column("parent_generation_id", sa.Uuid(), nullable=True))
    op.add_column("generations", sa.Column("variation_label", sa.String(length=50), nullable=True))

    op.create_index(PARENT_INDEX, "generations", ["parent_generation_id"])
    # SET NULL rather than CASCADE: deleting an original must not delete
    # the variations someone made from it.
    op.create_foreign_key(
        PARENT_FK,
        "generations",
        "generations",
        ["parent_generation_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(PARENT_FK, "generations", type_="foreignkey")
    op.drop_index(PARENT_INDEX, table_name="generations")
    op.drop_column("generations", "variation_label")
    op.drop_column("generations", "parent_generation_id")
    op.drop_column("generations", "advisories")
    op.drop_column("generations", "request_trace")
    op.drop_column("generations", "time_signature")
    op.drop_column("generations", "key_scale")
    op.drop_column("generations", "bpm")
