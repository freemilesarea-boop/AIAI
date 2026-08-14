"""audio edit provenance: how a generation was produced, and from what range

Phase 13B. A generation can now be produced two ways: text-to-music, or
by regenerating part of an existing generation's audio. Three nullable
columns on ``generations`` record which, and over what source range.

Why these are not derivable from what already exists:

``edit_kind``
    ``parent_generation_id`` already says *which* row this came from, but
    not *how*: Phase 8's "generate again" also sets a parent, and it is a
    fresh text-to-music run rather than an edit. ``variation_label``
    looks like a candidate and is not one — it is client-settable free
    text, so a caller could label an ordinary generation "extend" and the
    two would be indistinguishable. The worker routes on this value, so
    it must not be forgeable through the public API.

``edit_start_seconds`` / ``edit_end_seconds``
    The edited range, measured from the parent's actual master audio
    rather than from any requested duration. ``request_trace`` does carry
    the numbers sent to the engine, but it is documented as best-effort
    diagnostics: it is written outside the transaction, tolerated when it
    fails, and decoded defensively. "Which part of the source was
    edited?" has to be answerable for every edited row, so it gets real
    columns.

All three are nullable with no server default: existing rows are
ordinary generations and must stay that way, and NULL is the honest
representation of "not an edit" rather than a sentinel.

Revision ID: 0008
Revises: 0007
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("generations", sa.Column("edit_kind", sa.String(length=30), nullable=True))
    op.add_column("generations", sa.Column("edit_start_seconds", sa.Float(), nullable=True))
    op.add_column("generations", sa.Column("edit_end_seconds", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("generations", "edit_end_seconds")
    op.drop_column("generations", "edit_start_seconds")
    op.drop_column("generations", "edit_kind")
