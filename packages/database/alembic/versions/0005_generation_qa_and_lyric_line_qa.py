"""human QA records for generations and individual lyric lines

Phase 9. The most damaging Korean failure the human evaluator reported
is whole lyric lines being skipped. Nothing in this stack can currently
detect that automatically — and this migration does not pretend
otherwise. It creates the place to *record* a human's judgement, line by
line, so the evidence accumulates in a shape a future training phase can
consume.

Two tables:

``generation_qa`` — one row per generation: a 1-10 triage rating,
failure tags, free notes, and per-section verdicts.

``lyric_line_qa`` — one row per lyric line: what was submitted, and what
the listener heard happen to it (COMPLETE / PARTIAL / SKIPPED /
DUPLICATED / UNKNOWN).

Everything is nullable or defaulted. A generation with no QA record is
the normal state, not a missing one — UNKNOWN is a real answer and the
absence of a row means "nobody has listened yet".

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

GENERATION_QA_FK = "fk_generation_qa_generation_id"
LINE_QA_FK = "fk_lyric_line_qa_generation_id"
GENERATION_QA_UNIQUE = "uq_generation_qa_generation_id"
LINE_QA_UNIQUE = "uq_lyric_line_qa_generation_id"


def upgrade() -> None:
    op.create_table(
        "generation_qa",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("generation_id", sa.Uuid(), nullable=False),
        # 1-10 triage. NULL means "not yet rated", which is different
        # from a low score and must stay distinguishable from one.
        sa.Column("overall_rating", sa.Integer(), nullable=True),
        # JSON list of failure tags (KOREAN_LINE_OMISSION, ...). Text so
        # the same schema runs on SQLite in tests and PostgreSQL in prod.
        sa.Column("failure_tags", sa.Text(), nullable=True),
        # JSON object: section name -> verdict, for the full-song view.
        sa.Column("section_verdicts", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("reviewer", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_foreign_key(
        GENERATION_QA_FK,
        "generation_qa",
        "generations",
        ["generation_id"],
        ["id"],
        ondelete="CASCADE",
    )
    # One QA record per generation; re-review updates it in place.
    op.create_unique_constraint(GENERATION_QA_UNIQUE, "generation_qa", ["generation_id"])
    op.create_index("ix_generation_qa_generation_id", "generation_qa", ["generation_id"])

    op.create_table(
        "lyric_line_qa",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("generation_id", sa.Uuid(), nullable=False),
        # Position in the submitted lyric sheet, 0-based, tags excluded.
        sa.Column("line_index", sa.Integer(), nullable=False),
        sa.Column("section_label", sa.String(length=50), nullable=True),
        # The line as submitted, snapshotted so the QA record stays
        # readable even if the generation's lyrics are later edited.
        sa.Column("line_text", sa.Text(), nullable=False),
        # COMPLETE | PARTIAL | SKIPPED | DUPLICATED | UNKNOWN
        sa.Column("verdict", sa.String(length=20), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_foreign_key(
        LINE_QA_FK,
        "lyric_line_qa",
        "generations",
        ["generation_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_unique_constraint(LINE_QA_UNIQUE, "lyric_line_qa", ["generation_id", "line_index"])
    op.create_index("ix_lyric_line_qa_generation_id", "lyric_line_qa", ["generation_id"])
    op.create_index("ix_lyric_line_qa_verdict", "lyric_line_qa", ["verdict"])


def downgrade() -> None:
    op.drop_index("ix_lyric_line_qa_verdict", table_name="lyric_line_qa")
    op.drop_index("ix_lyric_line_qa_generation_id", table_name="lyric_line_qa")
    op.drop_constraint(LINE_QA_UNIQUE, "lyric_line_qa", type_="unique")
    op.drop_constraint(LINE_QA_FK, "lyric_line_qa", type_="foreignkey")
    op.drop_table("lyric_line_qa")

    op.drop_index("ix_generation_qa_generation_id", table_name="generation_qa")
    op.drop_constraint(GENERATION_QA_UNIQUE, "generation_qa", type_="unique")
    op.drop_constraint(GENERATION_QA_FK, "generation_qa", type_="foreignkey")
    op.drop_table("generation_qa")
