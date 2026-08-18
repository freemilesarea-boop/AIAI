"""The internal owner that holds pre-authentication data.

One constant, exported so the migration, the repository and the tests
all name the same row rather than repeating a UUID.

**This is a bridge, and it is meant to be removed.** Until Part 3 takes
the owner from the authenticated session, rows created through
unauthenticated product routes are attributed here — which is truthful,
because that is exactly what they are: data with no user behind it. Part
3 deletes :data:`LEGACY_OWNER_ID` from every call site, and the
remaining references are the checklist for finishing the job.
"""

from __future__ import annotations

import uuid

#: uuid5(NAMESPACE_DNS, LEGACY_OWNER_EMAIL). Deterministic so every
#: database has the same anchor; written out so it is greppable.
LEGACY_OWNER_EMAIL = "legacy-system@internal.luber"
LEGACY_OWNER_ID = uuid.UUID("e3c4d3cd-d86f-52f2-91b7-2b97f5011653")

__all__ = ["LEGACY_OWNER_EMAIL", "LEGACY_OWNER_ID"]
