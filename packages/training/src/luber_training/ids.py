"""Stable identifiers for every training entity.

Identity is assigned once and never derived from a mutable name. An
experiment can be renamed, a worker can move host, a config can be
edited into a new one — and the run that cites them must still resolve
to the same things a year later.

Prefixed so that an id is self-describing in a log line, a filename and
an error message. ``run_9f3c…`` in a stack trace says what it is without
anyone having to look it up.

The random component is 128 bits of ``secrets`` entropy, hex-encoded and
truncated to 16 characters (64 bits). That is not a security boundary —
ids are not secrets — it is a collision boundary, and 64 bits is far
beyond any plausible number of runs this project will ever create.
"""

from __future__ import annotations

import re
import secrets
from enum import StrEnum

#: Hex characters after the prefix. 64 bits.
ID_ENTROPY_CHARS = 16


class EntityKind(StrEnum):
    """Every entity that owns an identity, with its prefix."""

    MODEL = "mdl"
    EXPERIMENT = "exp"
    RUN = "run"
    CHECKPOINT = "ckpt"
    WORKER = "wrk"
    CANDIDATE = "cand"
    PLAN = "plan"


_PATTERN = re.compile(r"^(mdl|exp|run|ckpt|wrk|cand|plan)_[0-9a-f]{16}$")


def new_id(kind: EntityKind) -> str:
    """A fresh identifier for *kind*."""
    return f"{kind.value}_{secrets.token_hex(ID_ENTROPY_CHARS // 2)}"


def is_valid(identifier: str, kind: EntityKind | None = None) -> bool:
    """Whether a string is a well-formed id, optionally of one kind.

    Used at registry boundaries. An id arriving from a config file or a
    CLI argument is untrusted input, and a malformed one must be
    rejected rather than turned into a filesystem path.
    """
    if not _PATTERN.match(identifier):
        return False
    return kind is None or identifier.startswith(f"{kind.value}_")


def require(identifier: str, kind: EntityKind) -> str:
    """Return the id, or raise if it is not a valid one of *kind*."""
    if not is_valid(identifier, kind):
        raise ValueError(f"{identifier!r} is not a valid {kind.value} identifier")
    return identifier
