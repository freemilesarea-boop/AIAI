"""A filesystem registry: small, durable, and hard to corrupt.

No service, no database, no MLflow. This is operator infrastructure for
one project's training runs, and a versioned directory of JSON is
enough — with three properties that are not optional.

**Writes are atomic.** Temp file in the same directory, flush, fsync,
then `os.replace`. A registry half-written by an interrupted CLI is
worse than no registry: it looks readable and is wrong.

**Mutation is locked.** Two CLI processes must not assign the same run
id or interleave a read-modify-write. An exclusive `flock` around the
mutating section is enough for the single-host operator case this is
built for, and the lock file is separate from the data so a stale lock
never destroys anything.

**History is append-only.** Runs are never overwritten to look clean. A
retry writes a new run citing its parent, and the audit log records the
sequence of events rather than the current state.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA_VERSION = "luber-training-registry/1"

#: Sub-directories, one per entity kind.
COLLECTIONS: tuple[str, ...] = (
    "models",
    "experiments",
    "runs",
    "workers",
    "checkpoints",
    "candidates",
)

AUDIT_LOG_NAME = "audit_log.jsonl"
LOCK_NAME = ".registry.lock"


class RegistryError(RuntimeError):
    """Raised when the registry cannot honour a request."""


class ConflictError(RegistryError):
    """Raised when a write would overwrite something that exists."""


def _atomic_write(path: Path, payload: str) -> None:
    """Write via temp file and rename. The rename is the only step that
    must not tear, and on POSIX it does not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), prefix=".tmp-", suffix=".json", delete=False
    )
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class AuditEvent:
    timestamp: str
    event: str
    entity_id: str
    entity_kind: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event": self.event,
            "entity_id": self.entity_id,
            "entity_kind": self.entity_kind,
            "metadata": self.metadata,
        }


# ── audit event names ────────────────────────────────────────────────
BASELINE_REGISTERED = "BASELINE_REGISTERED"
EXPERIMENT_CREATED = "EXPERIMENT_CREATED"
EXPERIMENT_UPDATED = "EXPERIMENT_UPDATED"
RUN_CREATED = "RUN_CREATED"
RUN_VALIDATED = "RUN_VALIDATED"
RUN_BLOCKED = "RUN_BLOCKED"
RUN_QUEUED = "RUN_QUEUED"
RUN_STARTED = "RUN_STARTED"
RUN_COMPLETED = "RUN_COMPLETED"
RUN_FAILED = "RUN_FAILED"
RUN_CANCELLED = "RUN_CANCELLED"
RUN_LOST = "RUN_LOST"
WORKER_REGISTERED = "WORKER_REGISTERED"
WORKER_UPDATED = "WORKER_UPDATED"
CHECKPOINT_REGISTERED = "CHECKPOINT_REGISTERED"
CHECKPOINT_FINALIZED = "CHECKPOINT_FINALIZED"
CANDIDATE_CREATED = "CANDIDATE_CREATED"


class Registry:
    """Durable storage for training entities."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for name in COLLECTIONS:
            (self.root / name).mkdir(exist_ok=True)
        self._lock_path = self.root / LOCK_NAME
        self._lock_depth = 0
        self._lock_handle: Any = None

    # ── locking ──────────────────────────────────────────────────────
    @contextmanager
    def lock(self) -> Iterator[None]:
        """Exclusive registry lock, reentrant within one process.

        Reentrancy matters: `transition_run` locks and then calls
        `save_run`, which would deadlock on a non-reentrant lock. Depth
        counting keeps the flock held once and released once.
        """
        if self._lock_depth > 0:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._lock_path.open("w")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            self._lock_handle = handle
            self._lock_depth = 1
            yield
        finally:
            self._lock_depth = 0
            self._lock_handle = None
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    # ── generic access ───────────────────────────────────────────────
    def _path(self, collection: str, entity_id: str) -> Path:
        if collection not in COLLECTIONS:
            raise RegistryError(f"unknown collection {collection!r}")
        # Ids are validated at the entity layer; this is defence in
        # depth against a crafted id escaping the registry directory.
        if "/" in entity_id or ".." in entity_id or not entity_id:
            raise RegistryError(f"unsafe entity id {entity_id!r}")
        return self.root / collection / f"{entity_id}.json"

    def exists(self, collection: str, entity_id: str) -> bool:
        return self._path(collection, entity_id).is_file()

    def read(self, collection: str, entity_id: str) -> dict[str, Any]:
        path = self._path(collection, entity_id)
        if not path.is_file():
            raise RegistryError(f"{collection[:-1]} {entity_id} is not registered")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RegistryError(f"{path.name} is not a registry record")
        return payload

    def write(
        self, collection: str, entity_id: str, payload: dict[str, Any], *, overwrite: bool = False
    ) -> None:
        with self.lock():
            path = self._path(collection, entity_id)
            if path.exists() and not overwrite:
                raise ConflictError(f"{collection[:-1]} {entity_id} already exists")
            _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))

    def list_ids(self, collection: str) -> list[str]:
        directory = self.root / collection
        if not directory.is_dir():
            return []
        return sorted(p.stem for p in directory.glob("*.json"))

    def list_all(self, collection: str) -> list[dict[str, Any]]:
        """Every record in a collection, sorted by id.

        Reads one file per entity. At the scale this registry is for —
        thousands of runs — that is linear and fast; a query language
        would be a service, and this is not one.
        """
        return [self.read(collection, entity_id) for entity_id in self.list_ids(collection)]

    def find(self, collection: str, **criteria: Any) -> list[dict[str, Any]]:
        """Records matching every supplied field."""
        return [
            record
            for record in self.list_all(collection)
            if all(record.get(key) == value for key, value in criteria.items())
        ]

    # ── audit log ────────────────────────────────────────────────────
    def append_audit(
        self, event: str, entity_id: str, entity_kind: str, **metadata: Any
    ) -> AuditEvent:
        """Append-only. Never rewritten, never compacted here.

        Secrets never reach this: callers pass identifiers and counts.
        The value of the log is that it records what happened in order,
        including the things somebody might later prefer it had not.
        """
        entry = AuditEvent(
            timestamp=datetime.now(UTC).isoformat(),
            event=event,
            entity_id=entity_id,
            entity_kind=entity_kind,
            metadata=metadata,
        )
        with self.lock():
            path = self.root / AUDIT_LOG_NAME
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return entry

    def audit_events(self, entity_id: str | None = None) -> list[dict[str, Any]]:
        path = self.root / AUDIT_LOG_NAME
        if not path.is_file():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # A torn final line from a hard kill. Skip it rather
                # than refusing to read the whole history.
                continue
            if entity_id is None or entry.get("entity_id") == entity_id:
                events.append(entry)
        return events
