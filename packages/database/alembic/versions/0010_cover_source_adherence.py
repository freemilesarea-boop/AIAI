"""source_adherence: how closely a cover was asked to follow its source

Phase 13D-2. A cover is a third way a generation can come from another
one — the engine regenerates the whole performance steered by a semantic
sketch of the source — and it carries one setting the worker must know:
how closely to follow that source.

Why an existing column could not carry it:

``edit_start_seconds``
    Documented as seconds into the source. Storing a 0-1 ratio there
    would make the recorded provenance untrue, and both values are read
    back for audit.

``variation_label``
    Client-settable free text. The worker routes on this value, so it
    must not be forgeable through the public API — the same reason 0008
    did not use it.

``request_trace``
    Explicitly best-effort diagnostics: written outside the transaction,
    tolerated when it fails, decoded defensively. A value the worker
    depends on cannot live somewhere allowed to be absent.

Nullable with no server default: only covers have one, and NULL is the
honest representation of "not applicable" rather than a sentinel such as
0.0, which would read as "ignore the source entirely".

Revision ID: 0010
Revises: 0009
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("generations", sa.Column("source_adherence", sa.Float(), nullable=True))


def downgrade() -> None:
    # Covers themselves are left in place; only the setting is dropped.
    # Deleting the rows would destroy generated music to undo a schema
    # change, which is never the right trade.
    op.drop_column("generations", "source_adherence")
