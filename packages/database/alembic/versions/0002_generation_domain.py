"""create generation domain tables

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-11

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "generations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("lyrics", sa.Text(), nullable=False),
        sa.Column("vocal_gender", sa.String(length=20), nullable=False),
        sa.Column("duration_requested", sa.Integer(), nullable=False),
        sa.Column("duration_actual", sa.Float(), nullable=True),
        sa.Column("seed", sa.BigInteger(), nullable=True),
        sa.Column("language", sa.String(length=16), nullable=True),
        sa.Column("instrumental", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=True),
        sa.Column("model_name", sa.String(length=100), nullable=True),
        sa.Column("model_version", sa.String(length=50), nullable=True),
        sa.Column("idempotency_key", sa.String(length=200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=50), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_generations")),
    )
    op.create_index(
        op.f("ix_generations_idempotency_key"),
        "generations",
        ["idempotency_key"],
        unique=True,
    )
    op.create_index(op.f("ix_generations_status"), "generations", ["status"], unique=False)
    op.create_index(op.f("ix_generations_created_at"), "generations", ["created_at"], unique=False)

    op.create_table(
        "generation_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("generation_id", sa.Uuid(), nullable=False),
        sa.Column("queue_name", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column(
            "enqueued_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("worker_id", sa.String(length=100), nullable=True),
        sa.Column("error_code", sa.String(length=50), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["generation_id"],
            ["generations.id"],
            name=op.f("fk_generation_jobs_generation_id_generations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_generation_jobs")),
    )
    op.create_index(
        op.f("ix_generation_jobs_generation_id"),
        "generation_jobs",
        ["generation_id"],
        unique=False,
    )

    op.create_table(
        "audio_assets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("generation_id", sa.Uuid(), nullable=False),
        sa.Column("asset_type", sa.String(length=20), nullable=False),
        sa.Column("format", sa.String(length=10), nullable=False),
        sa.Column("sample_rate", sa.Integer(), nullable=False),
        sa.Column("bit_depth", sa.Integer(), nullable=True),
        sa.Column("channels", sa.Integer(), nullable=False),
        sa.Column("duration", sa.Float(), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["generation_id"],
            ["generations.id"],
            name=op.f("fk_audio_assets_generation_id_generations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audio_assets")),
    )
    op.create_index(
        op.f("ix_audio_assets_generation_id"),
        "audio_assets",
        ["generation_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_audio_assets_generation_id"), table_name="audio_assets")
    op.drop_table("audio_assets")
    op.drop_index(op.f("ix_generation_jobs_generation_id"), table_name="generation_jobs")
    op.drop_table("generation_jobs")
    op.drop_index(op.f("ix_generations_created_at"), table_name="generations")
    op.drop_index(op.f("ix_generations_status"), table_name="generations")
    op.drop_index(op.f("ix_generations_idempotency_key"), table_name="generations")
    op.drop_table("generations")
