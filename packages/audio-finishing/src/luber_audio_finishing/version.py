"""Identity of the finishing engine.

Every finished output must be attributable to the exact code that
produced it, because the whole point of keeping the raw master is being
able to reprocess it later and compare. A finished file whose engine
version is unknown is not comparable to anything.

The version changes when *audible behaviour* changes: thresholds, filter
choices, clamps, the order of the chain. It does not change for
refactoring, docs, or new measurements that drive no decision.
"""

from __future__ import annotations

FINISHING_VERSION = "p14-v1"

#: ffmpeg writes this into the finished file's INFO chunk. It is how a
#: later run recognises audio it has already processed, which is what
#: stops the engine from stacking corrections on corrections.
FINISHING_STAMP_KEY = "comment"
FINISHING_STAMP_PREFIX = "luber_finishing="


def finishing_stamp(version: str = FINISHING_VERSION) -> str:
    """The metadata value written into a finished file."""
    return f"{FINISHING_STAMP_PREFIX}{version}"


def parse_finishing_stamp(value: str | None) -> str | None:
    """Extract the engine version from a stamp, or ``None`` if absent."""
    if value is None:
        return None
    text = value.strip()
    if not text.startswith(FINISHING_STAMP_PREFIX):
        return None
    return text[len(FINISHING_STAMP_PREFIX) :] or None
