"""The wire contract between the control plane and a remote worker.

Two programs on two machines have to agree about what a message means,
and they are upgraded at different times. So every interaction carries a
protocol version, and a version this build does not know is a refusal
rather than a best effort. A worker running last month's code that
silently ignored a field it did not recognise would produce a run whose
configuration nobody could reconstruct.

The states here are the *worker's* view of an execution. They are not
`RunStatus` — the control plane's record of a run and a worker's record
of a process are different facts that can legitimately disagree, most
importantly when contact is lost: the worker says RUNNING because the
trainer is running, and the control plane says LOST because it cannot
see it. Collapsing the two into one enum would delete exactly the
distinction that makes recovery possible.

`RUN_STATUS_FOR_WORKER_STATE` maps one to the other where the mapping is
sound, and it maps CANCELLING to RUNNING on purpose: a cancellation that
has been requested but not confirmed has not happened yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from luber_training.entities import FailureCode, RunStatus

#: The protocol this build speaks. Bumped when a message changes shape
#: in a way an older peer would misread.
REMOTE_PROTOCOL_VERSION = "luber-remote/1"

#: Versions this build accepts from a peer. A set rather than a single
#: value so a future build can accept two during a rollout, and so the
#: check is never "starts with luber-remote".
SUPPORTED_PROTOCOL_VERSIONS: frozenset[str] = frozenset({REMOTE_PROTOCOL_VERSION})


def now() -> str:
    return datetime.now(UTC).isoformat()


class ProtocolError(RuntimeError):
    """Raised when a peer speaks a protocol this build cannot honour."""


def check_protocol(version: str | None, *, peer: str = "peer") -> str:
    """Accept a peer's protocol version, or refuse to proceed.

    Refusing is the whole point. A worker whose protocol is unknown may
    have different ideas about what "cancel" does or where checkpoints
    are written, and finding that out during a training run is far worse
    than finding it out here.
    """
    if not version:
        raise ProtocolError(
            f"{peer} reported no remote protocol version; this build speaks "
            f"{REMOTE_PROTOCOL_VERSION} and will not guess at an unversioned peer"
        )
    if version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ProtocolError(
            f"{peer} speaks {version!r}; this build supports "
            f"{', '.join(sorted(SUPPORTED_PROTOCOL_VERSIONS))}. Upgrade one side rather "
            "than attempting execution across an unknown protocol"
        )
    return version


class WorkerState(StrEnum):
    """What a worker believes is happening to one run.

    ``LOST`` is never set by the worker about itself — a process cannot
    report that it has stopped reporting. The control plane derives it
    from silence, and it appears in this enum so that a reconciliation
    result can be expressed in the same vocabulary.
    """

    #: No run assigned. The worker's resting state.
    IDLE = "IDLE"
    RECEIVING = "RECEIVING"
    PREFLIGHT = "PREFLIGHT"
    READY = "READY"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    #: Control-plane-derived, never self-reported.
    LOST = "LOST"


#: Worker states from which nothing further happens on the worker.
TERMINAL_WORKER_STATES: frozenset[str] = frozenset(
    {
        WorkerState.CANCELLED.value,
        WorkerState.COMPLETED.value,
        WorkerState.FAILED.value,
    }
)

#: How a worker's view of an execution maps onto the Phase 25 run state
#: machine. Deliberately not injective: several preparation states all
#: mean "queued" to the control plane, because from its side nothing has
#: started until the trainer is launched.
RUN_STATUS_FOR_WORKER_STATE: dict[str, str] = {
    WorkerState.IDLE.value: RunStatus.QUEUED.value,
    WorkerState.RECEIVING.value: RunStatus.QUEUED.value,
    WorkerState.PREFLIGHT.value: RunStatus.QUEUED.value,
    WorkerState.READY.value: RunStatus.QUEUED.value,
    WorkerState.STARTING.value: RunStatus.STARTING.value,
    WorkerState.RUNNING.value: RunStatus.RUNNING.value,
    # A requested cancellation is not a completed one. The trainer is
    # still running until the worker confirms it stopped, and reporting
    # CANCELLED early would let an operator believe a GPU was free.
    WorkerState.CANCELLING.value: RunStatus.RUNNING.value,
    WorkerState.CANCELLED.value: RunStatus.CANCELLED.value,
    WorkerState.COMPLETED.value: RunStatus.COMPLETED.value,
    WorkerState.FAILED.value: RunStatus.FAILED.value,
    WorkerState.LOST.value: RunStatus.LOST.value,
}


def run_status_for(worker_state: str) -> str:
    """The run status a worker state implies.

    An unknown worker state maps to LOST rather than to anything
    optimistic: this build does not know what the peer meant, and the
    honest reading of "I cannot interpret the worker" is that contact is
    effectively broken.
    """
    return RUN_STATUS_FOR_WORKER_STATE.get(worker_state, RunStatus.LOST.value)


class WorkerHealth(StrEnum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    #: Something is wrong that the worker itself can see.
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


class ReconcileOutcome(StrEnum):
    """What a reconciliation established about a run after lost contact."""

    RUNNING_RECOVERED = "RUNNING_RECOVERED"
    COMPLETED_RECOVERED = "COMPLETED_RECOVERED"
    FAILED_RECOVERED = "FAILED_RECOVERED"
    CANCELLED_RECOVERED = "CANCELLED_RECOVERED"
    #: The worker answered but cannot say what happened. Recorded as
    #: unknown rather than resolved to a guess.
    UNKNOWN = "UNKNOWN"
    #: The worker did not answer at all.
    UNREACHABLE = "UNREACHABLE"
    #: The worker has no record of the run. It was never launched, or
    #: its state was destroyed.
    NOT_PRESENT = "NOT_PRESENT"


#: Which reconcile outcomes justify launching the trainer.
#: NOT_PRESENT is the only one — everything else means a process may
#: exist, and launching a second trainer against one run's checkpoint
#: directory would corrupt both.
LAUNCH_SAFE_OUTCOMES: frozenset[str] = frozenset({ReconcileOutcome.NOT_PRESENT.value})


class RemoteCommand(StrEnum):
    """Worker verbs. A closed set, so an argv can be validated."""

    IDENTITY = "identity"
    PROBE = "probe"
    HEARTBEAT = "heartbeat"
    PREPARE = "prepare"
    RECEIVE = "receive"
    PREFLIGHT = "preflight"
    START = "start"
    STATUS = "status"
    LOGS = "logs"
    METRICS = "metrics"
    CANCEL = "cancel"
    CHECKPOINTS = "checkpoints"
    RESULT = "result"
    CLEANUP = "cleanup"


@dataclass
class Envelope:
    """Every worker reply, in one shape.

    The version travels with the payload rather than being assumed from
    the connection, so a reply that took a different path — read from a
    file, copied out of a log — can still be checked.
    """

    ok: bool
    command: str
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    worker_id: str | None = None
    protocol_version: str = REMOTE_PROTOCOL_VERSION
    timestamp: str = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "ok": self.ok,
            "command": self.command,
            "worker_id": self.worker_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, peer: str = "worker") -> Envelope:
        check_protocol(payload.get("protocol_version"), peer=peer)
        return cls(
            ok=bool(payload.get("ok")),
            command=str(payload.get("command", "")),
            payload=dict(payload.get("payload") or {}),
            error=payload.get("error"),
            worker_id=payload.get("worker_id"),
            protocol_version=str(payload["protocol_version"]),
            timestamp=str(payload.get("timestamp", "")),
        )


#: Which Phase 25 failure code a remote condition maps to. Reused rather
#: than extended: a second taxonomy for remote failures would mean the
#: same problem had two names depending on where it happened.
REMOTE_FAILURE_CODES: dict[str, str] = {
    "PROTOCOL_MISMATCH": FailureCode.ENVIRONMENT_INVALID.value,
    "PREFLIGHT_BLOCKED": FailureCode.ENVIRONMENT_INVALID.value,
    "ARTIFACT_DIGEST_MISMATCH": FailureCode.ENVIRONMENT_INVALID.value,
    "CODE_VERSION_MISMATCH": FailureCode.CODE_VERSION_DIRTY.value,
    "WORKER_UNREACHABLE": FailureCode.WORKER_LOST.value,
    "TRAINER_NONZERO_EXIT": FailureCode.TRAINER_CRASH.value,
    "CUDA_OOM": FailureCode.OOM.value,
    "DISK_FULL": FailureCode.CHECKPOINT_WRITE_FAILED.value,
    "CANCELLED": FailureCode.CANCELLED_BY_OPERATOR.value,
}


__all__ = [
    "LAUNCH_SAFE_OUTCOMES",
    "REMOTE_FAILURE_CODES",
    "REMOTE_PROTOCOL_VERSION",
    "RUN_STATUS_FOR_WORKER_STATE",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "TERMINAL_WORKER_STATES",
    "Envelope",
    "ProtocolError",
    "ReconcileOutcome",
    "RemoteCommand",
    "WorkerHealth",
    "WorkerState",
    "check_protocol",
    "now",
    "run_status_for",
]
