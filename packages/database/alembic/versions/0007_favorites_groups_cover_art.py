"""favorites, generation groups, cover art placeholder

Phase 12. Three columns on ``generations``, each read by the product:

``favorite``
    Server-side favourite state. Deliberately not localStorage — a
    favourite that disappears when you open the app on another machine is
    not a favourite.

``generation_group_id``
    One CREATE can produce two songs so the user can compare
    alternatives. The siblings share this id. It is *application*
    metadata: the model provider never sees it, and each sibling remains
    a fully independent generation with its own job, seed, status and
    asset. There is no ``generation_groups`` table on purpose — a group
    is the set of rows carrying the id, and a table would add a lifecycle
    (orphans, cascades) that buys nothing at this size.

``cover_art_url``
    Reserved for generated cover art. Phase 12 never writes it; the UI
    reads it to decide between real artwork and the existing deterministic
    placeholder. Storing a fabricated URL here would be worse than NULL.

``favorite`` carries a server default so the NOT NULL column can be added
to a populated table. Migration 0005 was caught by exactly this: unit
tests build the schema from ORM metadata and so never notice a missing
server default, while real PostgreSQL rejects the ALTER.

Revision ID: 0007
Revises: 0006
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

GROUP_INDEX = "ix_generations_generation_group_id"


def upgrade() -> None:
    op.add_column(
        "generations",
        sa.Column(
            "favorite",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column("generations", sa.Column("generation_group_id", sa.Uuid(), nullable=True))
    op.add_column("generations", sa.Column("cover_art_url", sa.Text(), nullable=True))
    # Reading a group is "give me this song's siblings", which is a
    # lookup by this column on every two-result submission.
    op.create_index(GROUP_INDEX, "generations", ["generation_group_id"])


def downgrade() -> None:
    op.drop_index(GROUP_INDEX, table_name="generations")
    op.drop_column("generations", "cover_art_url")
    op.drop_column("generations", "generation_group_id")
    op.drop_column("generations", "favorite")
