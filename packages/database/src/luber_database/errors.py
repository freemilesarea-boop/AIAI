"""Domain errors the repository raises.

Kept separate from the API layer: the rule they express is about the
data, not about HTTP, and the worker enforces the same one without ever
producing a status code.
"""

from __future__ import annotations

from uuid import UUID


class GenerationHasDescendantsError(Exception):
    """Refuses to delete a generation other generations were derived from.

    Carries the count so a caller can say how many without querying
    again, and the id so a log line can be traced. Neither is shown to a
    user verbatim — the API turns this into product language.
    """

    def __init__(self, generation_id: UUID, descendant_count: int) -> None:
        self.generation_id = generation_id
        self.descendant_count = descendant_count
        super().__init__(f"generation {generation_id} has {descendant_count} derived version(s)")
