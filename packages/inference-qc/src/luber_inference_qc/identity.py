"""What makes two generation requests the same, and how retries pick seeds.

Two things live here because they are the same question twice: what is
this request, and what is *this attempt at* this request.

**The request digest** covers only what changes the audio. Timestamps,
ids and titles are excluded — a request resubmitted a minute later is
the same request, and a digest that disagreed would make every trace
incomparable. Fields the provider ignores are excluded for the same
reason a config field the trainer ignores is refused in Phase 25: it
looks like it means something.

The seed is **not** in the digest. Two attempts differing only in seed
are attempts at the same request, which is exactly what the trace needs
to be able to say.

**Seed derivation is deterministic.** Attempt *n* of a request always
gets the same seed, derived from the base seed and the attempt index by
hashing. An operator reproducing a run needs to be able to reach the
same second attempt as the original, and `random.randint` cannot offer
that.

Hashing rather than `base + n`: adjacent seeds are not guaranteed to
produce unrelated samples in every model, and a retry whose seed is one
away from the failing one is the least useful retry available.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: Request fields that affect the audio. Anything not named here is not
#: part of a request's identity, and adding a field means bumping the QC
#: schema version — a digest whose meaning changed silently would make
#: every earlier trace uncomparable without anyone noticing.
DIGEST_FIELDS: tuple[str, ...] = (
    "prompt",
    "lyrics",
    "vocal_gender",
    "duration_seconds",
    "language",
    "instrumental",
    "bpm",
    "key_scale",
    "time_signature",
    "reference_sha256",
    "task",
    "edit_kind",
    "edit_start_seconds",
    "edit_end_seconds",
    "source_sha256",
)

#: Bound on a seed. Providers take a 32-bit unsigned value.
SEED_MODULUS = 2**32


def _canonical(payload: dict[str, Any]) -> str:
    """Sorted, separator-stable JSON. The same dict always renders the same."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_fields(source: Any) -> dict[str, Any]:
    """Pull the digest-relevant fields off any request-shaped object.

    Duck-typed on purpose: a `GenerationRequest`, an `AudioEditRequest`
    and a plain dict all describe a generation, and the digest should not
    depend on which type the caller happened to hold. A field that is
    absent is absent — never defaulted, because a default would make two
    genuinely different requests hash the same.
    """
    if isinstance(source, dict):
        return {key: source[key] for key in DIGEST_FIELDS if key in source}
    fields: dict[str, Any] = {}
    for key in DIGEST_FIELDS:
        if hasattr(source, key):
            value = getattr(source, key)
            if value is not None:
                fields[key] = value
    return fields


def request_digest(source: Any, *, extra: dict[str, Any] | None = None) -> str:
    """SHA-256 over the canonical form of what was asked for.

    ``extra`` carries anything the caller knows and the request object
    does not — the digest of a resolved reference track, say, which the
    request holds as a path that differs between machines.
    """
    payload = request_fields(source)
    if extra:
        payload.update({key: value for key, value in extra.items() if value is not None})
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def derive_seed(base_seed: int | None, attempt_index: int, request_sha256: str) -> int | None:
    """The seed for attempt *n*, or ``None`` to let the provider choose.

    Attempt 0 is always the seed the user gave, unchanged: their first
    attempt is the one they asked for. Later attempts hash the base seed
    together with the attempt index and the request digest, so the
    sequence is reproducible for this request and does not collide with
    the sequence for another request that happens to share a base seed.

    ``None`` in means the user expressed no preference, and it stays
    ``None`` for every attempt — inventing a seed would take away the
    provider's own randomisation and make every retry of a seedless
    request identical, which is the opposite of what a retry is for.
    """
    if base_seed is None:
        return None
    if attempt_index <= 0:
        return base_seed
    material = f"{base_seed}:{attempt_index}:{request_sha256}".encode()
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:4], "big") % SEED_MODULUS


__all__ = ["DIGEST_FIELDS", "SEED_MODULUS", "derive_seed", "request_digest", "request_fields"]
