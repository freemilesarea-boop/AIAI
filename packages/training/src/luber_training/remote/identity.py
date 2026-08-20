"""Worker identity, heartbeats, and leases on a run.

Three related problems, all of them about knowing *which* machine and
*which* execution one is talking to.

**Identity.** A hostname is not an identity. Rented GPU boxes are called
`gpu-01` by every provider, DHCP hands the same name to different
machines, and a reprovisioned instance keeps the name and loses
everything else. So a worker mints an id once, writes it into its own
root, and carries a host *fingerprint* — a digest over stable machine
facts — alongside it. If the id matches and the fingerprint does not,
the machine was rebuilt, and that is worth refusing to train on until
someone looks.

**Heartbeat.** A worker writes its state periodically. The control
plane reads timestamps and derives ONLINE / STALE / OFFLINE from them.
The lease is generous on purpose: one missed heartbeat during a
multi-hour run means a hiccup, not a dead machine, and a run failed for
a missed poll is a rented GPU wasted for nothing.

**Run lease.** One run, one worker, one trainer. The lease binds a run
id to a worker and to a *plan hash*, which is what makes redispatch
safe: the same run with the same plan returns the existing execution,
and the same run with a different plan is an integrity violation rather
than a second attempt.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from luber_training.remote.protocol import REMOTE_PROTOCOL_VERSION, WorkerHealth, WorkerState, now

#: Default heartbeat cadence. Frequent enough that a lost worker is
#: noticed within minutes, rare enough that it is not a load source.
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 60.0

#: How long silence is tolerated before a worker is merely STALE.
#: Several missed beats, because networks blink.
DEFAULT_STALE_AFTER_SECONDS = 300.0

#: How long before it is treated as OFFLINE. Note what this does *not*
#: mean: an OFFLINE worker's run does not become FAILED, it becomes
#: LOST. We know we cannot see the trainer; we do not know it stopped.
DEFAULT_OFFLINE_AFTER_SECONDS = 900.0

#: How long a lease stays valid without renewal. Longer than OFFLINE, so
#: a worker that goes quiet and returns still owns its run and does not
#: race a second dispatch.
DEFAULT_LEASE_SECONDS = 3600.0


def _parse(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class Liveness(StrEnum):
    ONLINE = "ONLINE"
    #: Late, but not yet presumed gone.
    STALE = "STALE"
    OFFLINE = "OFFLINE"
    #: Never heard from at all.
    UNKNOWN = "UNKNOWN"


@dataclass
class WorkerIdentity:
    """Who a worker is, stably, across restarts and reboots.

    ``host_fingerprint`` is a digest of machine facts, not of the
    hostname. Two rented instances with identical names produce
    different fingerprints; one instance rebooted produces the same one.
    """

    worker_id: str
    worker_name: str
    backend_type: str
    host_fingerprint: str
    protocol_version: str = REMOTE_PROTOCOL_VERSION
    created_at: str = field(default_factory=now)
    #: The capability signature at the time identity was minted, so a
    #: later probe can show what changed.
    capability_signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorkerIdentity:
        known = {key: value for key, value in payload.items() if key in cls.__annotations__}
        return cls(**known)


def host_fingerprint(report: Any) -> str:
    """A digest over facts that survive a reboot and not a rebuild.

    Includes the machine's architecture, CPU count, memory and — where
    present — GPU UUIDs, which are burned into the hardware. Excludes
    hostname, IP and anything a provider assigns, because those are
    exactly what stays the same when the machine underneath changes.
    """
    facts = {
        "architecture": getattr(report, "architecture", None),
        "os_name": getattr(report, "os_name", None),
        "cpu_count": getattr(report, "cpu_count", None),
        "system_ram_mb": getattr(report, "system_ram_mb", None),
        "total_disk_mb": getattr(report, "total_disk_mb", None),
        "gpu_uuids": sorted(
            device.uuid for device in getattr(report, "gpus", []) or [] if device.uuid
        ),
        "gpu_model": getattr(report, "gpu_model", None),
        "gpu_count": getattr(report, "gpu_count", None),
    }
    return hashlib.sha256(
        json.dumps(facts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def new_worker_id() -> str:
    """A fresh worker id. 64 bits, prefixed like every other id here."""
    return f"wrk_{secrets.token_hex(8)}"


def load_or_create_identity(
    path: Path,
    *,
    worker_name: str,
    backend_type: str,
    fingerprint: str,
    capability_signature: str | None = None,
) -> tuple[WorkerIdentity, bool]:
    """Read this worker's identity, minting one on first run.

    Returns the identity and whether the host fingerprint has changed
    since it was minted. A changed fingerprint is not corrected
    silently: the id still names the same logical worker, and the caller
    decides whether a rebuilt machine may keep it.
    """
    path = Path(path)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = WorkerIdentity.from_dict(payload)
        return identity, identity.host_fingerprint != fingerprint

    identity = WorkerIdentity(
        worker_id=new_worker_id(),
        worker_name=worker_name,
        backend_type=backend_type,
        host_fingerprint=fingerprint,
        capability_signature=capability_signature,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return identity, False


@dataclass
class Heartbeat:
    """One report of life from a worker.

    Carries no secret. It carries the worker's own view of what it is
    doing, which is the only view that can distinguish "the trainer
    died" from "we stopped being able to see the trainer".
    """

    worker_id: str
    timestamp: str = field(default_factory=now)
    protocol_version: str = REMOTE_PROTOCOL_VERSION
    worker_state: str = WorkerState.IDLE.value
    active_run_id: str | None = None
    health: str = WorkerHealth.OK.value
    uptime_seconds: float | None = None
    free_disk_mb: int | None = None
    gpu: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Heartbeat:
        known = {key: value for key, value in payload.items() if key in cls.__annotations__}
        return cls(**known)


@dataclass(frozen=True)
class LivenessPolicy:
    """How long silence means what. Configurable, and documented.

    The defaults are deliberately patient. Training runs for hours on a
    metered machine; declaring a worker dead because one poll timed out,
    and then dispatching the run somewhere else, is how two trainers end
    up writing to one checkpoint directory.
    """

    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS
    offline_after_seconds: float = DEFAULT_OFFLINE_AFTER_SECONDS

    def evaluate(self, last_heartbeat: str | None, *, reference: datetime | None = None) -> str:
        moment = _parse(last_heartbeat)
        if moment is None:
            return Liveness.UNKNOWN.value
        current = reference or datetime.now(UTC)
        age = (current - moment).total_seconds()
        if age >= self.offline_after_seconds:
            return Liveness.OFFLINE.value
        if age >= self.stale_after_seconds:
            return Liveness.STALE.value
        return Liveness.ONLINE.value

    def age_seconds(
        self, last_heartbeat: str | None, *, reference: datetime | None = None
    ) -> float | None:
        moment = _parse(last_heartbeat)
        if moment is None:
            return None
        return ((reference or datetime.now(UTC)) - moment).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeaseError(RuntimeError):
    """Raised when a lease would allow two trainers on one run."""


@dataclass
class RunLease:
    """One worker's claim on one run, bound to one plan.

    The plan hash is the load-bearing field. Without it, a redispatch
    after an edited config would look like the same run and quietly
    continue with different settings; with it, that case is an integrity
    violation and says so.
    """

    lease_id: str
    run_id: str
    worker_id: str
    training_plan_sha256: str
    leased_at: str = field(default_factory=now)
    expires_at: str | None = None
    released_at: str | None = None
    protocol_version: str = REMOTE_PROTOCOL_VERSION

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        worker_id: str,
        training_plan_sha256: str,
        duration_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> RunLease:
        started = datetime.now(UTC)
        return cls(
            lease_id=f"lease_{secrets.token_hex(8)}",
            run_id=run_id,
            worker_id=worker_id,
            training_plan_sha256=training_plan_sha256,
            leased_at=started.isoformat(),
            expires_at=(started + timedelta(seconds=duration_seconds)).isoformat(),
        )

    @property
    def released(self) -> bool:
        return self.released_at is not None

    def expired(self, *, reference: datetime | None = None) -> bool:
        moment = _parse(self.expires_at)
        if moment is None:
            return False
        return (reference or datetime.now(UTC)) >= moment

    def renew(self, duration_seconds: float = DEFAULT_LEASE_SECONDS) -> RunLease:
        self.expires_at = (datetime.now(UTC) + timedelta(seconds=duration_seconds)).isoformat()
        return self

    def release(self) -> RunLease:
        self.released_at = now()
        return self

    def check_compatible(self, *, run_id: str, worker_id: str, plan_sha256: str) -> None:
        """Whether a dispatch may proceed against this existing lease.

        Three outcomes, and the middle one is the reason this exists:

        * a different worker holds the run — refuse, or two machines
          train it;
        * the same worker, a *different* plan — refuse as an integrity
          violation, because the run id no longer means one thing;
        * the same worker, the same plan — permitted, and the caller
          returns the existing execution rather than starting another.
        """
        if self.run_id != run_id:
            raise LeaseError(f"lease {self.lease_id} is for run {self.run_id}, not {run_id}")
        if self.worker_id != worker_id and not self.released:
            raise LeaseError(
                f"run {run_id} is already leased to worker {self.worker_id}; dispatching it "
                f"to {worker_id} would run one training job on two machines"
            )
        if self.training_plan_sha256 != plan_sha256:
            raise LeaseError(
                f"run {run_id} was leased with plan {self.training_plan_sha256[:12]} but the "
                f"dispatch carries {plan_sha256[:12]}; the same run id cannot mean two "
                "different training configurations"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunLease:
        known = {key: value for key, value in payload.items() if key in cls.__annotations__}
        return cls(**known)


__all__ = [
    "DEFAULT_HEARTBEAT_INTERVAL_SECONDS",
    "DEFAULT_LEASE_SECONDS",
    "DEFAULT_OFFLINE_AFTER_SECONDS",
    "DEFAULT_STALE_AFTER_SECONDS",
    "Heartbeat",
    "LeaseError",
    "Liveness",
    "LivenessPolicy",
    "RunLease",
    "WorkerIdentity",
    "host_fingerprint",
    "load_or_create_identity",
    "new_worker_id",
]
