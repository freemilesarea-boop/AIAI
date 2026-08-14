"""edit_kind vocabulary: name the operation, not the engine primitive

Phase 13C. A second kind of audio edit exists now — replacing an interior
range — and it maps to the *same* engine primitive as extending
(regenerate this time range, preserve the rest). What differs is what the
range means, and the worker has to know which:

- ``EXTEND`` re-anchors the range to the measured end of the audio it is
  uploading, so the seam lands at the true end of the recording.
- ``REPLACE_RANGE`` uses the absolute times the user chose.

Phase 13B stored the *engine* primitive's name, ``REGENERATE_RANGE``,
which cannot express that difference. This is a data-only migration: no
column changes, only the vocabulary in the existing rows. Leaving two
vocabularies in a column the worker routes on would reintroduce exactly
the ambiguity 0008 was written to remove.

Every existing ``REGENERATE_RANGE`` row is an extension — it is the only
edit Phase 13B could produce — so the mapping is exact rather than a
guess, and the downgrade restores it precisely.

Revision ID: 0009
Revises: 0008
"""

from __future__ import annotations

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

#: The value Phase 13B wrote: ACE-Step's primitive, not a product concept.
LEGACY_EXTEND = "REGENERATE_RANGE"


def upgrade() -> None:
    op.execute(f"UPDATE generations SET edit_kind = 'EXTEND' WHERE edit_kind = '{LEGACY_EXTEND}'")


def downgrade() -> None:
    # Only extensions existed before this revision, so mapping EXTEND back
    # is lossless. REPLACE_RANGE rows are cleared rather than mislabelled
    # as extensions: the older code cannot represent them, and calling a
    # replacement an extension would make the worker re-anchor its range.
    op.execute(f"UPDATE generations SET edit_kind = '{LEGACY_EXTEND}' WHERE edit_kind = 'EXTEND'")
    op.execute(
        "UPDATE generations SET edit_kind = NULL, edit_start_seconds = NULL, "
        "edit_end_seconds = NULL WHERE edit_kind = 'REPLACE_RANGE'"
    )
