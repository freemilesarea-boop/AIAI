"""record what the candidate controller did

Phase 29 generates candidates, measures them, and may spend a second
inference when the first output fails a measurable check. Which of those
happened is invisible from the delivered assets: a generation that took
one call and one that took three both end with exactly one master.

This column is the only durable record of the attempts that did not win.
Their audio is deliberately never uploaded — a rejected candidate must
not be able to reach a library — so without this there would be no way
to answer why a delivery was retried, why one candidate lost, or how
many provider calls a request cost.

One nullable Text column holding JSON, matching ``request_trace`` from
migration 0004 and ``finishing_trace`` from 0011: Text rather than JSON
so the same schema runs on SQLite in unit tests and PostgreSQL in
production. Additive and nullable, so older code ignores it and NULL is
a real historical state — the controller never ran — rather than
something to invent a default for.

No other column changes. ``duration_actual``, ``seed`` and the asset
rows keep their existing meanings: they describe the candidate that was
selected, which is the only one that ever becomes audio a listener can
reach.

Revision ID: 0015
Revises: 0014
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("generations", sa.Column("inference_qc_trace", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("generations", "inference_qc_trace")
