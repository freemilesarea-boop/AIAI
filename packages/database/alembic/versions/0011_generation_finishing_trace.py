"""record what the finishing engine decided

Phase 14B integrates the adaptive finishing engine into delivery. Whether
a generation was finished is already visible from its assets — a
FINISHED_MASTER row exists or it does not — but that alone cannot
distinguish three genuinely different states:

* the engine ran and decided nothing needed correcting,
* the engine ran and failed,
* the engine never ran, because the generation predates this phase.

All three look like "no finished master". This column keeps them apart,
and carries the engine version and the decision trail that produced the
result, so a finished master can be explained months later without the
audio being the only evidence.

One nullable Text column holding JSON, matching ``request_trace`` and
``advisories`` from migration 0004: Text rather than JSON so the same
schema runs on SQLite in unit tests and PostgreSQL in production.
Additive and nullable, so older code ignores it and NULL is a real
historical state rather than something to invent a default for.

No asset-type change is needed. ``audio_assets.asset_type`` is
``String(20)`` with no enum type and no check constraint anywhere in this
migration history, so FINISHED_MASTER is an application-level value.
MASTER keeps its existing meaning — the raw generation master — so no
existing row is reinterpreted or rewritten.

Revision ID: 0011
Revises: 0010
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("generations", sa.Column("finishing_trace", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("generations", "finishing_trace")
