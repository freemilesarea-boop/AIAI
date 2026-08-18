"""The internal owner that holds pre-authentication data.

One constant, exported so the migration, the repository and the tests
all name the same row rather than repeating a UUID.

Part 3 removed the last product call site. Nothing creates data as this
owner any more: the API takes the owner from the authenticated session,
and an unscoped repository refuses to invent one. What remains is the
identity itself — used by the migration that created it, and by tests
that assert the historical corpus stays unreachable.
"""

from __future__ import annotations

import uuid

#: uuid5(NAMESPACE_DNS, LEGACY_OWNER_EMAIL). Deterministic so every
#: database has the same anchor; written out so it is greppable.
LEGACY_OWNER_EMAIL = "legacy-system@internal.luber"
LEGACY_OWNER_ID = uuid.UUID("e3c4d3cd-d86f-52f2-91b7-2b97f5011653")

__all__ = ["LEGACY_OWNER_EMAIL", "LEGACY_OWNER_ID"]
