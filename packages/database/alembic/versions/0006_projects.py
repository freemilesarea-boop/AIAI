"""projects: a lightweight workspace grouping for generations

Phase 11. A project is a folder, nothing more: it has a name and it
holds generations. No collaboration, no permissions, no sharing — those
are later product phases and inventing their schema now would guess at
requirements nobody has stated.

Membership is a nullable ``project_id`` on ``generations`` rather than a
join table. One generation belongs to at most one project, which is what
"move it into a folder" means; a join table would model a many-to-many
relationship the product does not have.

``ON DELETE SET NULL``: deleting a project must never delete the music
inside it. The generations survive and return to being unfiled.

Revision ID: 0006
Revises: 0005
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

PROJECT_FK = "fk_generations_project_id"
PROJECT_INDEX = "ix_generations_project_id"


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        # Reserved for the authentication phase. Nullable today because
        # nothing owns anything yet, exactly like generations.user_id.
        sa.Column("user_id", sa.Uuid(), nullable=True),
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
    op.create_index("ix_projects_created_at", "projects", ["created_at"])

    op.add_column("generations", sa.Column("project_id", sa.Uuid(), nullable=True))
    op.create_index(PROJECT_INDEX, "generations", ["project_id"])
    op.create_foreign_key(
        PROJECT_FK, "generations", "projects", ["project_id"], ["id"], ondelete="SET NULL"
    )


def downgrade() -> None:
    op.drop_constraint(PROJECT_FK, "generations", type_="foreignkey")
    op.drop_index(PROJECT_INDEX, table_name="generations")
    op.drop_column("generations", "project_id")
    op.drop_index("ix_projects_created_at", table_name="projects")
    op.drop_table("projects")
