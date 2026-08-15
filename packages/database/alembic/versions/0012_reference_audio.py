"""reference audio as a first-class input

Phase 15R. A reference track is something the user supplies to steer a
generation, which makes it an input and not an asset. Everything in
``audio_assets`` is output: rows are keyed by the generation that
produced them, and the download route resolves objects from them. A
reference exists before any generation, can steer several, and must
never be reachable through a route that serves somebody a master — so it
gets its own table rather than another ``asset_type`` value, which would
have left it one enum away from being downloadable as a finished song.

``generations.reference_audio_id`` is nullable, so every existing row and
every request that names no reference is unchanged.

The foreign key is RESTRICT, not CASCADE. Deleting a reference must not
delete the songs made from it; a generation that silently lost the record
of what conditioned it would be worse than a delete that refuses.

Revision ID: 0012
Revises: 0011
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None

FK = "fk_generations_reference_audio_id"
INDEX = "ix_generations_reference_audio_id"
SHA_INDEX = "ix_reference_audio_sha256"


def upgrade() -> None:
    op.create_table(
        "reference_audio",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_format", sa.String(length=10), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("sample_rate", sa.Integer(), nullable=False),
        sa.Column("channels", sa.Integer(), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index(SHA_INDEX, "reference_audio", ["sha256"])

    op.add_column("generations", sa.Column("reference_audio_id", sa.Uuid(), nullable=True))
    op.create_index(INDEX, "generations", ["reference_audio_id"])
    op.create_foreign_key(
        FK, "generations", "reference_audio", ["reference_audio_id"], ["id"], ondelete="RESTRICT"
    )


def downgrade() -> None:
    op.drop_constraint(FK, "generations", type_="foreignkey")
    op.drop_index(INDEX, table_name="generations")
    op.drop_column("generations", "reference_audio_id")
    op.drop_index(SHA_INDEX, table_name="reference_audio")
    op.drop_table("reference_audio")
