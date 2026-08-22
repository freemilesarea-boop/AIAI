"""Where circuit state lives, and how two workers avoid disagreeing about it.

`max_jobs = 1` per generation worker means several worker processes is
the normal way to scale. A circuit held in module state would give each
of them a private opinion: worker A gives up on a provider, worker B
keeps calling it for another four minutes, and the operator sees a
circuit that is open and traffic that never stopped.

So state is durable and shared, and every transition is a
compare-and-set on a revision number. Two workers reading the same
record and both deciding to open it produce one write and one loser; the
loser re-reads and finds the circuit already open, which is the right
answer rather than a conflict to resolve.

The `Protocol` is what keeps this package free of SQLAlchemy. The
in-memory implementation below is complete — every state-machine test
runs against it — so the machine is testable without a database, and
"it passed against the fake" cannot mean something different from "it
works".

Only the I/O is async, and only because the database is. The state
machine in `circuit.py` stays a pure synchronous function of state,
evidence and a clock: that is where purity earns its keep, and colouring
it would buy nothing but `await`.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from luber_provider_resilience.circuit import (
    CircuitIdentity,
    CircuitRecord,
    Transition,
)


class ConcurrentModification(RuntimeError):
    """Someone else wrote this circuit first.

    Not an error condition so much as the coordination working. The
    caller re-reads and reconsiders: usually the other writer already
    did what this one was about to.
    """

    def __init__(self, identity: CircuitIdentity, expected: int, actual: int) -> None:
        super().__init__(
            f"{identity.key()} was modified concurrently "
            f"(expected revision {expected}, found {actual})"
        )
        self.identity = identity
        self.expected = expected
        self.actual = actual


class CircuitStore(Protocol):
    """What the router needs from durable storage, and nothing more."""

    async def load(self, identity: CircuitIdentity) -> CircuitRecord: ...

    async def save(self, record: CircuitRecord, *, expected_revision: int) -> CircuitRecord: ...

    async def record_transition(self, transition: Transition) -> None: ...

    async def all_circuits(self) -> Sequence[CircuitRecord]: ...

    async def transitions(self, *, limit: int = 50) -> Sequence[Transition]: ...


class InMemoryCircuitStore:
    """A complete store, for tests and for a single-process deployment.

    Not a stub: the whole state machine is exercised through it. The
    lock is real because a threadpool is real — FastAPI runs sync
    handlers in one, and a store that was only correct under asyncio
    would be correct in exactly the place it is not used.
    """

    def __init__(self, records: Iterable[CircuitRecord] = ()) -> None:
        self._records: dict[str, CircuitRecord] = {
            record.identity.key(): record for record in records
        }
        self._transitions: list[Transition] = []
        self._lock = threading.RLock()

    async def load(self, identity: CircuitIdentity) -> CircuitRecord:
        with self._lock:
            existing = self._records.get(identity.key())
            if existing is not None:
                return existing
            # A circuit nobody has written is closed. Creating it lazily
            # means a new provider needs no registration step, and a
            # deployment that adds one does not have to remember.
            return CircuitRecord(identity=identity)

    async def save(self, record: CircuitRecord, *, expected_revision: int) -> CircuitRecord:
        with self._lock:
            key = record.identity.key()
            current = self._records.get(key)
            actual = current.revision if current is not None else 0
            if actual != expected_revision:
                raise ConcurrentModification(record.identity, expected_revision, actual)
            self._records[key] = record
            return record

    async def record_transition(self, transition: Transition) -> None:
        with self._lock:
            self._transitions.append(transition)
            # Bounded: a flapping circuit over a long weekend would
            # otherwise grow this without limit in a process that never
            # restarts.
            if len(self._transitions) > 500:
                del self._transitions[: len(self._transitions) - 500]

    async def all_circuits(self) -> Sequence[CircuitRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda item: item.identity.key())

    async def transitions(self, *, limit: int = 50) -> Sequence[Transition]:
        with self._lock:
            return list(reversed(self._transitions[-limit:]))

    # ── test and operator conveniences ───────────────────────────────

    def seed(self, record: CircuitRecord) -> None:
        """Place a record directly, ignoring the revision check."""
        with self._lock:
            self._records[record.identity.key()] = record

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._transitions.clear()


async def apply_with_retry(
    store: CircuitStore,
    identity: CircuitIdentity,
    mutate: Any,
    *,
    attempts: int = 8,
    required: bool = True,
) -> tuple[CircuitRecord, Transition | None]:
    """Read, mutate, write — retrying when somebody else got there first.

    The compare-and-set loop, in one place. `mutate` takes a record and
    returns `(record, transition | None)`; if the write loses the race
    the record is re-read and `mutate` runs again against the newer
    state. That re-run is the point: the second attempt usually decides
    differently, because the circuit the first attempt was about to open
    is already open.

    ``required`` is what separates the two kinds of caller, and getting
    it wrong in either direction is a real bug.

    **Operator actions are required.** A manual open that silently did
    not happen is the worst outcome available here: somebody believes
    they stopped traffic and did not. Losing every attempt raises.

    **Recording evidence is not.** A burst of concurrent failures is
    exactly when contention peaks, and it is also when the loser has
    nothing to add — whoever won recorded the same failure against the
    same circuit and already opened it if it needed opening. Raising
    there would push a coordination detail up into a generation that
    otherwise succeeded, so the loser accepts the winner's state and
    returns it.

    Bounded either way. Spinning on a hot row would turn contention into
    a busy loop.
    """
    last_error: ConcurrentModification | None = None
    current = await store.load(identity)
    for _ in range(attempts):
        current = await store.load(identity)
        expected = current.revision
        updated, transition = mutate(current)
        if updated is current or updated.revision == expected:
            # Nothing changed; no write, no transition.
            return current, None
        try:
            saved = await store.save(updated, expected_revision=expected)
        except ConcurrentModification as exc:
            last_error = exc
            continue
        if transition is not None:
            await store.record_transition(transition)
        return saved, transition

    assert last_error is not None
    if required:
        raise last_error
    # Someone else won every round. Their write covers this outcome:
    # same circuit, same moment, same kind of failure.
    return await store.load(identity), None


def utcnow() -> datetime:
    """One place, so a test can see every timestamp this package writes."""
    return datetime.now(UTC)


__all__ = [
    "CircuitStore",
    "ConcurrentModification",
    "InMemoryCircuitStore",
    "apply_with_retry",
    "utcnow",
]
