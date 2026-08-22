"""What can actually train right now, target by target.

Three different questions get confused, and separating them is most of
this module's value:

**Is the control plane up?** The API's `/health`. Nothing to do with
hardware.

**Can a workload be placed?** `placement.place()`, per request.

**What are my compute targets doing?** This. Derived from probes rather
than configured, so it cannot report a GPU worker as READY after the
machine was returned to the rental provider.

`NOT_CONNECTED` and `NOT_AVAILABLE` are deliberately different answers.
A remote CUDA worker that has never been registered is not broken —
nobody has rented one — while a Mac that reports no MPS *is* telling
you something is wrong with the installation. Collapsing them would page
somebody about hardware they chose not to buy yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from luber_hardware.capability import UNKNOWN
from luber_hardware.devices import ComputeDevice, ExecutionLocation
from luber_hardware.placement import (
    DEFAULT_POLICY,
    ExecutionTarget,
    PlacementRequest,
    place,
)
from luber_hardware.precision import supported_precisions
from luber_hardware.versions import version_block
from luber_hardware.workloads import WorkloadClass


class TargetStatus(StrEnum):
    """Whether one compute target can be used."""

    #: Probed, and it can hold a tensor.
    READY = "READY"
    #: The machine is here and this device is not. A Mac whose torch has
    #: no Metal backend; a Linux box with no GPU.
    NOT_AVAILABLE = "NOT_AVAILABLE"
    #: No machine of this kind has been registered. Not a fault.
    NOT_CONNECTED = "NOT_CONNECTED"
    #: Registered, but nobody has probed it, or the probe could not
    #: reach an interpreter with torch.
    UNPROBED = "UNPROBED"


@dataclass(frozen=True)
class ComputeTargetView:
    """One row of the operator's compute-targets table."""

    name: str
    location: str
    device: str
    status: str
    detail: str = ""
    memory_mb: int | None = None
    precisions: tuple[str, ...] = ()
    workloads: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    planned: bool = False
    capability_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "location": self.location,
            "device": self.device,
            "status": self.status,
            "detail": self.detail,
            "memory_mb": self.memory_mb,
            "precisions": list(self.precisions),
            "workloads": list(self.workloads),
            "limitations": list(self.limitations),
            "planned": self.planned,
            "capability_digest": self.capability_digest,
        }


@dataclass(frozen=True)
class TrainingExecutionReadiness:
    """Every compute target, and what each can be asked to do."""

    at: datetime
    targets: tuple[ComputeTargetView, ...]
    summary: str = ""

    def status_of(self, name: str, device: str) -> str:
        for view in self.targets:
            if view.name == name and view.device == device:
                return view.status
        return TargetStatus.NOT_CONNECTED.value

    def can_run(self, workload: str) -> bool:
        return any(
            view.status == TargetStatus.READY.value and workload in view.workloads
            for view in self.targets
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **version_block(),
            "at": self.at.isoformat(),
            "summary": self.summary,
            "targets": [view.to_dict() for view in self.targets],
        }

    def render(self) -> str:
        lines = [self.summary]
        for view in self.targets:
            head = f"  {view.location}_{view.device}: {view.status}"
            if view.planned:
                head += "  [PLANNED]"
            lines.append(f"{head}  ({view.name})")
            if view.detail:
                lines.append(f"      {view.detail}")
        return "\n".join(lines)


#: Rows the readiness view always shows, whether or not a machine
#: exists for them. A missing REMOTE_CUDA row would read as "we didn't
#: check"; a row saying NOT_CONNECTED reads as "there isn't one yet",
#: which is the true and more useful statement.
EXPECTED_ROWS: tuple[tuple[str, str], ...] = (
    (ExecutionLocation.LOCAL.value, ComputeDevice.CPU.value),
    (ExecutionLocation.LOCAL.value, ComputeDevice.MPS.value),
    (ExecutionLocation.REMOTE.value, ComputeDevice.CUDA.value),
)


def _workloads_for(target: ExecutionTarget, device: str) -> tuple[str, ...]:
    """Which workload classes this target and device could actually take.

    Answered by running the real placement policy against a
    single-target list rather than by a second table. A readiness view
    that disagreed with placement would be worse than no readiness view,
    because an operator would plan against it.
    """
    out: list[str] = []
    for workload in WorkloadClass:
        request = PlacementRequest(
            workload=workload.value,
            preference=device,
            policy=DEFAULT_POLICY.get(workload.value),
            # Deliberately *not* relaxing `allow_local_fallback`. This
            # row has to answer what an ordinary request would get, and
            # an ordinary heavy-training request is refused locally. A
            # readiness view that showed a Mac as able to take heavy
            # training — because a caller *could* override the policy —
            # would be planning material that contradicts what the
            # scheduler does.
            allow_unverified_precision=True,
        )
        if place(request, [target]).placed:
            out.append(workload.value)
    return tuple(out)


def _status_for(target: ExecutionTarget, device: str) -> tuple[str, str]:
    capability = target.capability
    if capability.has_device(device):
        return TargetStatus.READY.value, ""
    if not capability.torch_installed:
        return (
            TargetStatus.UNPROBED.value,
            "the interpreter that answered has no torch, so this device could not be "
            "verified. Probe the interpreter that runs training.",
        )
    if device == ComputeDevice.MPS.value and capability.mps_built is False:
        return (
            TargetStatus.NOT_AVAILABLE.value,
            "this torch was built without the Metal backend",
        )
    return TargetStatus.NOT_AVAILABLE.value, f"torch reports no {device} on this machine"


def readiness(
    targets: Sequence[ExecutionTarget],
    *,
    now: datetime | None = None,
) -> TrainingExecutionReadiness:
    """Derive what each compute target can be asked to do."""
    moment = now or datetime.now(UTC)
    views: list[ComputeTargetView] = []
    seen: set[tuple[str, str]] = set()

    for target in targets:
        for device in (
            ComputeDevice.CUDA.value,
            ComputeDevice.MPS.value,
            ComputeDevice.CPU.value,
        ):
            # A row per device the target could plausibly offer. CPU
            # always; the accelerators only where the machine's own
            # platform makes them possible, so a Linux worker does not
            # grow an MPS row saying "not available" forever.
            if device == ComputeDevice.MPS.value and not target.capability.apple_silicon:
                continue
            if device == ComputeDevice.CUDA.value and target.capability.cuda_available is not True:
                if target.location != ExecutionLocation.REMOTE.value:
                    continue
            status, detail = _status_for(target, device)
            seen.add((target.location, device))
            views.append(
                ComputeTargetView(
                    name=target.name,
                    location=target.location,
                    device=device,
                    status=status,
                    detail=detail,
                    memory_mb=target.capability.accelerator_memory_mb(device),
                    precisions=supported_precisions(target.capability, device),
                    workloads=(
                        _workloads_for(target, device) if status == TargetStatus.READY.value else ()
                    ),
                    limitations=target.capability.notes,
                    planned=target.capability.planned,
                    capability_digest=target.capability.digest(),
                )
            )

    for location, device in EXPECTED_ROWS:
        if (location, device) in seen:
            continue
        views.append(
            ComputeTargetView(
                name=UNKNOWN,
                location=location,
                device=device,
                status=TargetStatus.NOT_CONNECTED.value,
                detail=(
                    "no NVIDIA worker has been registered with this deployment"
                    if device == ComputeDevice.CUDA.value
                    else f"no {location} machine offering {device} has been probed"
                ),
            )
        )

    ready = [view for view in views if view.status == TargetStatus.READY.value]
    trainable = [view for view in ready if WorkloadClass.HEAVY_TRAINING.value in view.workloads]
    if not ready:
        summary = "no compute target is ready"
    elif not trainable:
        summary = (
            f"{len(ready)} target(s) ready; none can take HEAVY_TRAINING — "
            "no CUDA worker is connected"
        )
    else:
        summary = f"{len(ready)} target(s) ready, {len(trainable)} able to take heavy training"

    return TrainingExecutionReadiness(at=moment, targets=tuple(views), summary=summary)


__all__ = [
    "EXPECTED_ROWS",
    "ComputeTargetView",
    "TargetStatus",
    "TrainingExecutionReadiness",
    "readiness",
]
