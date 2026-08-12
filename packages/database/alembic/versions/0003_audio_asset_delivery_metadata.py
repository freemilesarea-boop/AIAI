"""audio asset delivery metadata

Adds the delivery fields (mime type, file extension, bitrate) and the
one-asset-per-role guarantee used for idempotent post-processing.

Existing Phase 3 rows are preserved: the new text columns are added
nullable, backfilled from the format already recorded on each row, and
only then made NOT NULL. ``bitrate`` stays nullable because PCM masters
have no meaningful bitrate.

The unique constraint is added last so the backfill cannot trip it. Any
pre-existing duplicate (generation_id, asset_type) rows are collapsed to
the newest first — no Phase 3 generation has duplicates, but the
migration must not fail if one exists.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

UNIQUE_CONSTRAINT = "uq_audio_assets_generation_id_asset_type"


def upgrade() -> None:
    op.add_column("audio_assets", sa.Column("mime_type", sa.String(length=100), nullable=True))
    op.add_column("audio_assets", sa.Column("file_extension", sa.String(length=10), nullable=True))
    op.add_column("audio_assets", sa.Column("bitrate", sa.Integer(), nullable=True))

    # Backfill from the format column that already exists on every row.
    op.execute(
        sa.text(
            """
            UPDATE audio_assets
               SET mime_type = CASE lower(format)
                                   WHEN 'wav' THEN 'audio/wav'
                                   WHEN 'mp3' THEN 'audio/mpeg'
                                   WHEN 'flac' THEN 'audio/flac'
                                   ELSE 'application/octet-stream'
                               END,
                   file_extension = lower(format)
             WHERE mime_type IS NULL OR file_extension IS NULL
            """
        )
    )

    op.alter_column(
        "audio_assets", "mime_type", existing_type=sa.String(length=100), nullable=False
    )
    op.alter_column(
        "audio_assets", "file_extension", existing_type=sa.String(length=10), nullable=False
    )

    # Collapse any duplicates before enforcing uniqueness.
    op.execute(
        sa.text(
            """
            DELETE FROM audio_assets
             WHERE id IN (
                   SELECT id FROM (
                          SELECT id,
                                 ROW_NUMBER() OVER (
                                     PARTITION BY generation_id, asset_type
                                     ORDER BY created_at DESC, id DESC
                                 ) AS rn
                            FROM audio_assets
                   ) ranked
                    WHERE ranked.rn > 1
             )
            """
        )
    )

    op.create_unique_constraint(UNIQUE_CONSTRAINT, "audio_assets", ["generation_id", "asset_type"])


def downgrade() -> None:
    op.drop_constraint(UNIQUE_CONSTRAINT, "audio_assets", type_="unique")
    op.drop_column("audio_assets", "bitrate")
    op.drop_column("audio_assets", "file_extension")
    op.drop_column("audio_assets", "mime_type")
