"""What produced a generation, decided from durable provenance alone.

One classifier, used everywhere. The alternative — each route or
component deciding for itself what an ``edit_kind`` of ``REPLACE_RANGE``
means — is how two screens end up disagreeing about the same row.

The rule reads two columns and nothing else:

===================  ==============  ==========================
Operation            ``edit_kind``   ``parent_generation_id``
===================  ==============  ==========================
``ORIGINAL``         NULL            NULL
``GENERATE_AGAIN``   NULL            set
``EXTEND``           ``EXTEND``      set
``REPLACE_SECTION``  ``REPLACE_RANGE``  set
``COVER``            ``COVER``       set
===================  ==============  ==========================

Deliberately *not* consulted: title, prompt, ``generation_group_id``,
``project_id`` and ``reference_audio_id``. The first two are user text
and prove nothing; the group id groups siblings of one CREATE rather
than a lineage; the project is an unrelated filing decision. The
reference matters most: a track uploaded to steer the sound is **input**
provenance, so a reference-conditioned generation with no parent is an
ORIGINAL, not something derived from the reference.
"""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID


class LineageOperation(StrEnum):
    """How a generation came to exist, in product vocabulary.

    ``REPLACE_SECTION`` rather than the stored ``REPLACE_RANGE``: the
    engine-adjacent name never reaches a user, and the boundary where it
    stops is here.
    """

    ORIGINAL = "ORIGINAL"
    GENERATE_AGAIN = "GENERATE_AGAIN"
    EXTEND = "EXTEND"
    REPLACE_SECTION = "REPLACE_SECTION"
    COVER = "COVER"

    @property
    def is_derived(self) -> bool:
        return self is not LineageOperation.ORIGINAL


#: Stored ``edit_kind`` values, mapped to product operations. A value
#: absent from this map is legacy or corrupt and must not crash a page.
_EDIT_KIND_TO_OPERATION: dict[str, LineageOperation] = {
    "EXTEND": LineageOperation.EXTEND,
    "REPLACE_RANGE": LineageOperation.REPLACE_SECTION,
    "COVER": LineageOperation.COVER,
}


def classify_operation(
    *, parent_generation_id: UUID | None, edit_kind: str | None
) -> LineageOperation:
    """The operation a row records, from its durable fields.

    Total by construction. An unrecognised ``edit_kind`` on a row that
    has a parent degrades to ``GENERATE_AGAIN`` — the one thing still
    known to be true is that it was derived from that parent — rather
    than raising. Song Detail must survive imperfect legacy data.
    """
    if parent_generation_id is None:
        # No parent means nothing was derived from, whatever else the row
        # says. A stray edit_kind here is legacy noise, not a lineage.
        return LineageOperation.ORIGINAL
    if not edit_kind:
        return LineageOperation.GENERATE_AGAIN
    return _EDIT_KIND_TO_OPERATION.get(edit_kind, LineageOperation.GENERATE_AGAIN)
