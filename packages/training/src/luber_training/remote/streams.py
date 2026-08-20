"""Reading a running job's logs and metrics without reading them twice.

Both problems here are the same problem: something on the worker grows,
the control plane polls it, and each poll must return only what is new.
Re-downloading a five-gigabyte log every thirty seconds would cost more
bandwidth than the training data did.

So both use a cursor. For logs it is a byte offset, returned with every
response and passed back on the next call. For metrics it is a line
count plus a per-event identity — because a byte offset alone cannot
survive a file being rewritten, and metric events must be deduplicated
on their own identity in case they are.

Metric identity is `(run_id, step, metric_name, source)`. That tuple is
what makes a metric one thing: the same step reported twice is the same
measurement, whether it arrived twice because of a retry, an overlapping
poll, or a worker that restarted and replayed its file.

Nothing here parses a metric out of console text when the trainer writes
a structured file. Regex over log lines is how a renamed field silently
becomes a missing metric, and a missing metric that nobody notices is
worse than one that is absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luber_training.metrics import MetricEvent, MetricSource
from luber_training.remote.protocol import now

#: How much log a single poll may return. Bounded so one call cannot
#: pull an unbounded amount into memory on either side.
DEFAULT_LOG_CHUNK_BYTES = 256 * 1024

#: How many metric events one poll may return.
DEFAULT_METRIC_BATCH = 2000


@dataclass
class LogChunk:
    """A slice of one log stream, with the cursor to continue from."""

    stream: str
    offset: int
    next_offset: int
    text: str
    eof: bool
    size_bytes: int
    truncated: bool = False

    @property
    def empty(self) -> bool:
        return not self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "offset": self.offset,
            "next_offset": self.next_offset,
            "text": self.text,
            "eof": self.eof,
            "size_bytes": self.size_bytes,
            "truncated": self.truncated,
        }


def read_log(
    path: Path, *, offset: int = 0, limit: int = DEFAULT_LOG_CHUNK_BYTES, stream: str = "stdout"
) -> LogChunk:
    """Read from *offset*, returning where to continue.

    Two cases have to be handled rather than assumed away. An offset
    past the end of the file means the file was rotated or truncated
    beneath the reader; continuing from that offset would return
    nothing forever, so the read restarts from the beginning and says
    so. An offset landing mid-character is why decoding is lenient —
    a UTF-8 sequence split across a chunk boundary must not raise.
    """
    path = Path(path)
    if not path.is_file():
        return LogChunk(
            stream=stream, offset=offset, next_offset=offset, text="", eof=True, size_bytes=0
        )

    size = path.stat().st_size
    truncated = False
    if offset > size:
        offset = 0
        truncated = True
    if offset < 0:
        offset = 0

    with path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(limit)

    return LogChunk(
        stream=stream,
        offset=offset,
        next_offset=offset + len(raw),
        text=raw.decode("utf-8", errors="replace"),
        eof=offset + len(raw) >= size,
        size_bytes=size,
        truncated=truncated,
    )


@dataclass
class LogCursor:
    """Where a reader has got to in each stream of one run."""

    stdout: int = 0
    stderr: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"stdout": self.stdout, "stderr": self.stderr}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> LogCursor:
        payload = payload or {}
        return cls(stdout=int(payload.get("stdout", 0)), stderr=int(payload.get("stderr", 0)))


def metric_identity(event: MetricEvent) -> tuple[str, int | None, str, str]:
    """What makes a metric event one event.

    The step is part of it and the timestamp is not. Two reports of step
    140's loss are one measurement reported twice; a timestamp would
    make them two, and the run's history would show a loss curve with
    duplicate points wherever a poll overlapped.
    """
    return (event.run_id, event.step, event.metric_name, event.source)


@dataclass
class MetricStream:
    """Reads a worker's metrics file, returning each event once.

    Keeps both a line cursor and the set of identities already emitted.
    The cursor makes the common case cheap; the identity set makes
    correctness independent of it, so a file rewritten from the start —
    which is what a resumed trainer does — replays without duplicating
    anything.
    """

    seen: set[tuple[str, int | None, str, str]] = field(default_factory=set)
    line_cursor: int = 0

    def read(self, path: Path, *, limit: int = DEFAULT_METRIC_BATCH) -> list[MetricEvent]:
        path = Path(path)
        if not path.is_file():
            return []

        events: list[MetricEvent] = []
        lines = path.read_text(encoding="utf-8").splitlines()

        # A file shorter than the cursor was rewritten. Start again; the
        # identity set is what prevents that from producing duplicates.
        start = self.line_cursor if self.line_cursor <= len(lines) else 0
        for index in range(start, len(lines)):
            if len(events) >= limit:
                self.line_cursor = index
                return events
            line = lines[index]
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # A torn final line from a run killed mid-write. Skipped
                # rather than fatal: the events before it are real.
                continue
            event = _event_from(payload)
            if event is None:
                continue
            identity = metric_identity(event)
            if identity in self.seen:
                continue
            self.seen.add(identity)
            events.append(event)

        self.line_cursor = len(lines)
        return events

    def to_dict(self) -> dict[str, Any]:
        return {"line_cursor": self.line_cursor, "seen": len(self.seen)}


def _event_from(payload: dict[str, Any]) -> MetricEvent | None:
    try:
        return MetricEvent(
            run_id=str(payload["run_id"]),
            metric_name=str(payload["metric_name"]),
            value=float(payload["value"]),
            source=str(payload.get("source", MetricSource.TRAINER.value)),
            step=int(payload["step"]) if payload.get("step") is not None else None,
            epoch=int(payload["epoch"]) if payload.get("epoch") is not None else None,
            unit=str(payload.get("unit", "")),
            timestamp=str(payload.get("timestamp", now())),
        )
    except (KeyError, TypeError, ValueError):
        return None


def deduplicate(events: list[MetricEvent]) -> list[MetricEvent]:
    """Collapse repeats within one batch, keeping the first of each."""
    seen: set[tuple[str, int | None, str, str]] = set()
    unique: list[MetricEvent] = []
    for event in events:
        identity = metric_identity(event)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(event)
    return unique


def merge_into(existing: Path, events: list[MetricEvent]) -> int:
    """Append events the file does not already contain.

    Reads the destination's identities first, so a control plane that
    polled twice — or restarted and polled again — records each
    measurement once. Returns how many were actually new.
    """
    existing = Path(existing)
    known: set[tuple[str, int | None, str, str]] = set()
    if existing.is_file():
        for line in existing.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = _event_from(payload)
            if event is not None:
                known.add(metric_identity(event))

    fresh = [event for event in deduplicate(events) if metric_identity(event) not in known]
    if not fresh:
        return 0

    existing.parent.mkdir(parents=True, exist_ok=True)
    with existing.open("a", encoding="utf-8") as handle:
        for event in fresh:
            handle.write(json.dumps(event.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    return len(fresh)


__all__ = [
    "DEFAULT_LOG_CHUNK_BYTES",
    "DEFAULT_METRIC_BATCH",
    "LogChunk",
    "LogCursor",
    "MetricStream",
    "deduplicate",
    "merge_into",
    "metric_identity",
    "read_log",
]
