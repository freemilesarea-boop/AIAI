"""How observations get in and out, without this package knowing about a database.

Two things live here: the shape of a store, and the translation between
an observation and a plain mapping.

The package deliberately imports no SQLAlchemy. Analytics that could
reach the ORM could reach the `generations` table, and the whole point
of the projection is that there is no path from here to a prompt. A
`Protocol` keeps the boundary honest — the store is whatever the caller
supplies, and the in-memory one below is a complete implementation, not
a stub, which is why the detection tests can run without a database at
all.

The mapping translation is here rather than in the database package for
the same reason in reverse: `luber-database` depends on no domain
package and should not start now. It hands back rows; this turns them
into observations.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from luber_inference_observability.dimensions import Segment
from luber_inference_observability.events import (
    InferenceObservation,
    validate,
)
from luber_inference_observability.versions import OBSERVABILITY_SCHEMA_VERSION
from luber_inference_observability.windows import TimeWindow

#: Fields stored as JSON text rather than as columns. Small, bounded
#: lists — a generation has a handful of findings, not thousands — so a
#: separate table would cost a join on every query to normalise
#: something nothing ever queries independently.
_JSON_FIELDS = ("critical_findings", "soft_findings", "data_quality_issues")


def utcnow() -> datetime:
    """One place, so a test can see every timestamp this package writes."""
    return datetime.now(UTC)


def _as_utc(value: Any) -> datetime | None:
    """A timestamp in UTC, whatever shape the store handed back.

    A stored column is documented to hold UTC, so a naive value is
    assumed to be UTC rather than local. Assuming local would silently
    shift every row by the reader's offset — a bug that produces
    plausible numbers in the wrong buckets.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def to_mapping(observation: InferenceObservation) -> dict[str, Any]:
    """An observation as a row-shaped mapping, ready to be written.

    ``ingested_at`` is stamped here when the observation does not carry
    one. It means "when this row was written", and the writer is the only
    thing that knows that — an observation constructed in a test, or by a
    future caller that builds one directly rather than through `observe`,
    would otherwise reach a NOT NULL column with nothing in it.

    Deliberately not defaulted on the dataclass: `None` there is
    meaningful, and it says this observation has not been persisted.
    """
    payload = observation.to_dict()
    payload["occurred_at"] = observation.occurred_at
    payload["ingested_at"] = observation.ingested_at or utcnow()
    for name in _JSON_FIELDS:
        payload[name] = json.dumps(list(getattr(observation, name)), sort_keys=True)
    return payload


def from_mapping(row: dict[str, Any]) -> InferenceObservation:
    """A row-shaped mapping as an observation.

    Tolerant of a missing optional column and strict about the required
    ones: a row without a generation id or a timestamp is not an
    observation with gaps, it is not an observation.
    """
    payload = dict(row)
    for name in _JSON_FIELDS:
        raw = payload.get(name)
        if isinstance(raw, str):
            try:
                payload[name] = tuple(json.loads(raw))
            except (ValueError, TypeError):
                payload[name] = ()
        elif raw is None:
            payload[name] = ()
        else:
            payload[name] = tuple(raw)

    # The database column is a UUID; the dataclass declares a string,
    # and the in-memory store keys on it. Left as a UUID, a row read back
    # from PostgreSQL and the same row built by the projector would be
    # two different keys holding the same generation.
    if "generation_id" in payload:
        payload["generation_id"] = str(payload["generation_id"])

    # Normalised on the way in rather than defended against everywhere
    # downstream. SQLite hands back naive datetimes for timezone-aware
    # columns, and a naive value reaching a window comparison raises
    # "can't compare offset-naive and offset-aware" — at query time, in
    # production, on whichever deployment happens to use SQLite.
    for name in ("occurred_at", "ingested_at"):
        payload[name] = _as_utc(payload.get(name))

    known = set(InferenceObservation.__dataclass_fields__)
    return InferenceObservation(**{key: value for key, value in payload.items() if key in known})


class ObservationStore(Protocol):
    """What the analytics engine needs from a store, and nothing more.

    Read-shaped on purpose. `select` takes a window and filters and hands
    back observations; there is no method here that could return a
    generation row, which is what keeps the privacy boundary structural
    rather than conventional.
    """

    def select(
        self,
        window: TimeWindow,
        *,
        segment: Segment | None = None,
        limit: int | None = None,
    ) -> Sequence[InferenceObservation]: ...

    def upsert(self, observations: Iterable[InferenceObservation]) -> int: ...

    def latest_occurrence(self) -> datetime | None: ...

    def count(self) -> int: ...


class InMemoryObservationStore:
    """A complete store, used by tests and by the CLI's fixture mode.

    Not a stub. Every detection test runs against this, which is what
    makes the regression engine testable without a database — and what
    stops "it works in the tests" from meaning "it works against the
    fake".

    Keyed by `generation_id`, which is what makes ingestion idempotent
    for free: ingesting the same generation twice replaces the row
    rather than adding one.
    """

    def __init__(self, observations: Iterable[InferenceObservation] = ()) -> None:
        self._rows: dict[str, InferenceObservation] = {}
        self.upsert(observations)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[InferenceObservation]:
        return iter(self._sorted())

    def _sorted(self) -> list[InferenceObservation]:
        return sorted(self._rows.values(), key=lambda item: item.occurred_at)

    def count(self) -> int:
        return len(self._rows)

    def upsert(self, observations: Iterable[InferenceObservation]) -> int:
        written = 0
        for observation in observations:
            self._rows[observation.generation_id] = observation
            written += 1
        return written

    def select(
        self,
        window: TimeWindow,
        *,
        segment: Segment | None = None,
        limit: int | None = None,
    ) -> Sequence[InferenceObservation]:
        where = segment or Segment()
        out = [
            row for row in self._sorted() if window.contains(row.occurred_at) and where.matches(row)
        ]
        return out[:limit] if limit is not None else out

    def latest_occurrence(self) -> datetime | None:
        if not self._rows:
            return None
        return max(row.occurred_at for row in self._rows.values())

    def get(self, generation_id: str) -> InferenceObservation | None:
        return self._rows.get(generation_id)


# ── verification ─────────────────────────────────────────────────────


class VerificationIssue:
    """What a store can be wrong about."""

    DUPLICATE_OBSERVATION = "DUPLICATE_OBSERVATION"
    FORBIDDEN_FIELD_PRESENT = "FORBIDDEN_FIELD_PRESENT"
    SCHEMA_VERSION_MISMATCH = "SCHEMA_VERSION_MISMATCH"
    INVALID_COUNTERS = "INVALID_COUNTERS"
    NAIVE_TIMESTAMP = "NAIVE_TIMESTAMP"
    DUPLICATE_INCIDENT_FINGERPRINT = "DUPLICATE_INCIDENT_FINGERPRINT"


def verify(
    observations: Sequence[InferenceObservation],
    *,
    incidents: Sequence[Any] = (),
) -> dict[str, Any]:
    """Check a store's contents against everything that must be true.

    Returns a report rather than raising. A verifier that stopped at the
    first problem would make an operator fix issues one run at a time,
    and the interesting case is usually "how many rows are like this".
    """
    from luber_inference_observability.events import FORBIDDEN_FIELDS

    issues: dict[str, list[str]] = {}

    def note(kind: str, detail: str) -> None:
        issues.setdefault(kind, []).append(detail)

    seen: set[str] = set()
    for observation in observations:
        if observation.generation_id in seen:
            note(VerificationIssue.DUPLICATE_OBSERVATION, observation.generation_id)
        seen.add(observation.generation_id)

        if observation.schema_version != OBSERVABILITY_SCHEMA_VERSION:
            note(
                VerificationIssue.SCHEMA_VERSION_MISMATCH,
                f"{observation.generation_id}: {observation.schema_version}",
            )

        if observation.occurred_at.tzinfo is None:
            note(VerificationIssue.NAIVE_TIMESTAMP, observation.generation_id)

        problems = validate(observation)
        for problem in problems:
            note(VerificationIssue.INVALID_COUNTERS, f"{observation.generation_id}: {problem}")

        # The privacy check, run against the serialised form rather than
        # the dataclass fields: a value smuggled into a dict-valued
        # column would pass a field-name check and fail this one.
        rendered = observation.to_dict()
        for name in rendered:
            if name in FORBIDDEN_FIELDS:
                note(
                    VerificationIssue.FORBIDDEN_FIELD_PRESENT,
                    f"{observation.generation_id}: {name}",
                )

    fingerprints: set[str] = set()
    for incident in incidents:
        identity = getattr(incident, "incident_id", None)
        if identity is None:
            continue
        if identity in fingerprints:
            note(VerificationIssue.DUPLICATE_INCIDENT_FINGERPRINT, str(identity))
        fingerprints.add(identity)

    return {
        "observations": len(observations),
        "incidents": len(incidents),
        "ok": not issues,
        "issues": {kind: sorted(set(items)) for kind, items in sorted(issues.items())},
        "issue_counts": {kind: len(items) for kind, items in sorted(issues.items())},
        "observability_schema_version": OBSERVABILITY_SCHEMA_VERSION,
    }


__all__ = [
    "InMemoryObservationStore",
    "ObservationStore",
    "VerificationIssue",
    "from_mapping",
    "to_mapping",
    "utcnow",
    "verify",
]
