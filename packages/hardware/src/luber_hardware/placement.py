"""Where a workload runs, and a decision somebody can read a month later.

Placement answers one question — *which machine, on which device* — and
the answer has to survive being read by somebody who was not there. So a
decision carries its location, its device, its precision, the targets it
considered and rejected, what it does not know, and the version of the
policy that produced it.

The rule that shapes the whole module is **no silent movement**. A job
that needed a GPU and could not get one is BLOCKED. It is not quietly
run on Apple silicon, and it is not quietly run on a CPU. Those would
both produce a model, which is what makes them dangerous: the run
finishes, the checkpoint looks like every other checkpoint, and the only
record that it trained on something else is a wall-clock time nobody
compares.

This is deliberately **not** where provider selection lives. Phase 31's
`ProviderRouter` decides *which generation provider* answers a user's
request. This decides *where a training or evaluation workload
executes*. They are different questions about different subsystems, and
a decision here never influences one there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from luber_hardware.capability import MachineCapability
from luber_hardware.devices import ComputeDevice, ComputePreference, ExecutionLocation, Precision
from luber_hardware.memory import (
    MemoryAssessment,
    MemoryVerdict,
    assess,
    budget_for,
)
from luber_hardware.resolver import DeviceDecision, DeviceRequest, DeviceResolver
from luber_hardware.versions import EXECUTION_PLACEMENT_POLICY_VERSION, version_block
from luber_hardware.workloads import WorkloadClass, is_training, uses_device


class PlacementPolicy(StrEnum):
    """How far a workload may travel, and what it may land on."""

    #: Keep it here if here can do it. Preprocessing, evaluation and
    #: checkpoint inspection: work that a CPU does at the same speed as
    #: anything else, where spending rented GPU-hours is pure waste.
    LOCAL_PREFERRED = "LOCAL_PREFERRED"

    #: Try a remote CUDA worker first. Falling back to a local device is
    #: **not** automatic — see `PlacementRequest.allow_local_fallback`.
    REMOTE_CUDA_PREFERRED = "REMOTE_CUDA_PREFERRED"

    #: CUDA or nothing, wherever it is. No CUDA is a refusal.
    CUDA_REQUIRED = "CUDA_REQUIRED"

    #: Apple's Metal backend is an acceptable answer. Light fine-tuning,
    #: where the point is finding out whether the pipeline works.
    MPS_ALLOWED = "MPS_ALLOWED"

    #: No accelerator. For work that has no tensors in it at all.
    CPU_ONLY = "CPU_ONLY"


class PlacementOutcome(StrEnum):
    """What placement decided."""

    PLACED = "PLACED"
    #: The whole point of the phase: nothing can run this, and it is
    #: said rather than worked around.
    BLOCKED_NO_COMPATIBLE_EXECUTION_TARGET = "BLOCKED_NO_COMPATIBLE_EXECUTION_TARGET"
    #: Targets exist but the caller supplied none to consider.
    BLOCKED_NO_TARGETS = "BLOCKED_NO_TARGETS"


#: Devices each policy will accept.
POLICY_DEVICES: dict[str, tuple[str, ...]] = {
    PlacementPolicy.LOCAL_PREFERRED.value: (
        ComputeDevice.CUDA.value,
        ComputeDevice.MPS.value,
        ComputeDevice.CPU.value,
    ),
    PlacementPolicy.REMOTE_CUDA_PREFERRED.value: (
        ComputeDevice.CUDA.value,
        ComputeDevice.MPS.value,
        ComputeDevice.CPU.value,
    ),
    PlacementPolicy.CUDA_REQUIRED.value: (ComputeDevice.CUDA.value,),
    PlacementPolicy.MPS_ALLOWED.value: (
        ComputeDevice.CUDA.value,
        ComputeDevice.MPS.value,
        ComputeDevice.CPU.value,
    ),
    PlacementPolicy.CPU_ONLY.value: (ComputeDevice.CPU.value,),
}

#: The policy each workload gets when nobody names one.
#:
#: Every entry is a claim about this repository rather than a
#: preference. Preprocessing is numpy and ffmpeg; evaluation is HTTP and
#: arithmetic; heavy training is the only thing that wants a rented GPU.
DEFAULT_POLICY: dict[str, str] = {
    WorkloadClass.PREPROCESS.value: PlacementPolicy.CPU_ONLY.value,
    WorkloadClass.EVALUATION.value: PlacementPolicy.CPU_ONLY.value,
    WorkloadClass.INFERENCE.value: PlacementPolicy.LOCAL_PREFERRED.value,
    WorkloadClass.CHECKPOINT_VALIDATION.value: PlacementPolicy.LOCAL_PREFERRED.value,
    WorkloadClass.LIGHT_FINE_TUNE.value: PlacementPolicy.MPS_ALLOWED.value,
    WorkloadClass.HEAVY_TRAINING.value: PlacementPolicy.REMOTE_CUDA_PREFERRED.value,
}


@dataclass(frozen=True)
class ExecutionTarget:
    """One machine placement may choose, and where it sits."""

    name: str
    capability: MachineCapability
    location: str = ExecutionLocation.LOCAL.value
    #: Set for a remote worker, so a decision names the machine Phase 27
    #: will actually talk to.
    worker_id: str | None = None
    #: True when this machine is also running the API, the database and
    #: the queue. It raises the memory reservation.
    runs_control_plane: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "location": self.location,
            "worker_id": self.worker_id,
            "runs_control_plane": self.runs_control_plane,
            "planned": self.capability.planned,
            "capability_digest": self.capability.digest(),
        }


@dataclass(frozen=True)
class PlacementRequest:
    """A workload looking for somewhere to run."""

    workload: str = WorkloadClass.HEAVY_TRAINING.value
    policy: str | None = None
    preference: str = ComputePreference.AUTO.value
    precision: str = Precision.AUTO.value
    #: Operations the config asks for that some devices cannot do —
    #: `adamw8bit` being the live example.
    required_operations: tuple[str, ...] = ()
    #: Memory this workload is known to need. Almost always ``None``,
    #: because almost nothing has been measured, and ``None`` produces
    #: an UNKNOWN verdict rather than a pass.
    estimated_memory_mb: int | None = None
    #: Whether a remote-CUDA-preferring workload may run locally when no
    #: CUDA worker is reachable. **False by default.** This is the flag
    #: that stops a GPU job silently becoming a Mac job.
    allow_local_fallback: bool = False
    allow_unverified_precision: bool = False

    def resolved_policy(self) -> str:
        return self.policy or DEFAULT_POLICY.get(
            self.workload, PlacementPolicy.LOCAL_PREFERRED.value
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "policy": self.resolved_policy(),
            "preference": self.preference,
            "precision": self.precision,
            "required_operations": list(self.required_operations),
            "estimated_memory_mb": self.estimated_memory_mb,
            "allow_local_fallback": self.allow_local_fallback,
            "allow_unverified_precision": self.allow_unverified_precision,
        }


@dataclass(frozen=True)
class ExecutionPlacementDecision:
    """Where this workload runs, or why nowhere can take it."""

    outcome: str
    workload: str
    policy: str
    execution_location: str | None = None
    compute_device: str | None = None
    target_name: str | None = None
    worker_id: str | None = None
    precision: str | None = None
    torch_device: str | None = None
    reason: str = ""
    #: Targets looked at and why each was passed over. An operator
    #: asking "why not the GPU box" gets the answer without re-running.
    considered: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    fallback_used: bool = False
    memory: MemoryAssessment | None = None
    capability_digest: str | None = None
    #: True when the target is a planned profile rather than a probe.
    planned_target: bool = False
    policy_version: str = EXECUTION_PLACEMENT_POLICY_VERSION
    unknowns: tuple[str, ...] = field(default=())

    @property
    def placed(self) -> bool:
        return self.outcome == PlacementOutcome.PLACED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            **version_block(),
            "outcome": self.outcome,
            "workload": self.workload,
            "policy": self.policy,
            "execution_location": self.execution_location,
            "compute_device": self.compute_device,
            "target_name": self.target_name,
            "worker_id": self.worker_id,
            "precision": self.precision,
            "torch_device": self.torch_device,
            "reason": self.reason,
            "considered": list(self.considered),
            "limitations": list(self.limitations),
            "fallback_used": self.fallback_used,
            "memory": None if self.memory is None else self.memory.to_dict(),
            "capability_digest": self.capability_digest,
            "planned_target": self.planned_target,
            "unknowns": list(self.unknowns),
        }

    def render(self) -> str:
        if not self.placed:
            return f"{self.outcome}: {self.reason}"
        where = f"{self.execution_location} + {self.compute_device}"
        line = f"{self.workload} → {where} on {self.target_name} ({self.precision})"
        if self.planned_target:
            line += "  [PLANNED PROFILE]"
        return line


def _target_order(targets: Sequence[ExecutionTarget], policy: str) -> list[ExecutionTarget]:
    """Which machines to try, in which order.

    Remote-first for CUDA-preferring work, local-first otherwise.
    Stable within each group so the same inputs give the same answer:
    a placement that depended on dictionary ordering would be
    reproducible only by accident.
    """
    remote_first = policy in (
        PlacementPolicy.REMOTE_CUDA_PREFERRED.value,
        PlacementPolicy.CUDA_REQUIRED.value,
    )

    def rank(target: ExecutionTarget) -> tuple[int, int, str]:
        remote = target.location == ExecutionLocation.REMOTE.value
        primary = 0 if remote == remote_first else 1
        # A probed machine beats a planned one at equal rank. Planning
        # profiles exist to answer "could this work"; they must never
        # win a placement away from hardware that actually exists.
        return (primary, 1 if target.capability.planned else 0, target.name)

    return sorted(targets, key=rank)


def place(
    request: PlacementRequest,
    targets: Sequence[ExecutionTarget],
) -> ExecutionPlacementDecision:
    """Choose a target for this workload, or refuse and say why."""
    policy = request.resolved_policy()

    if not targets:
        return ExecutionPlacementDecision(
            outcome=PlacementOutcome.BLOCKED_NO_TARGETS.value,
            workload=request.workload,
            policy=policy,
            reason="no execution target was offered; nothing can be placed",
        )

    allowed = POLICY_DEVICES.get(policy, POLICY_DEVICES[PlacementPolicy.LOCAL_PREFERRED.value])
    considered: list[str] = []
    rejections: list[str] = []
    ordered = _target_order(targets, policy)

    for target in ordered:
        considered.append(target.name)

        if policy == PlacementPolicy.REMOTE_CUDA_PREFERRED.value and _is_local_fallback(
            target, request
        ):
            rejections.append(
                f"{target.name}: local execution is not permitted for this run "
                "(allow_local_fallback is false)"
            )
            continue

        decision = _try_target(request, target, allowed=allowed)
        if decision is not None:
            return decision
        rejections.append(f"{target.name}: {_last_reason(request, target, allowed)}")

    return ExecutionPlacementDecision(
        outcome=PlacementOutcome.BLOCKED_NO_COMPATIBLE_EXECUTION_TARGET.value,
        workload=request.workload,
        policy=policy,
        reason=(
            f"no target can run a {request.workload} workload under {policy}. "
            + "; ".join(rejections)
        ),
        considered=tuple(considered),
    )


def _is_local_fallback(target: ExecutionTarget, request: PlacementRequest) -> bool:
    """Would landing here be the quiet downgrade the policy forbids?

    Under REMOTE_CUDA_PREFERRED a local target is only acceptable when
    the caller said so. Without this the first Mac in the list would
    take every heavy training job the moment the GPU box went offline,
    and the run would complete looking exactly like the ones that did
    not.
    """
    return target.location == ExecutionLocation.LOCAL.value and not request.allow_local_fallback


def _device_request(request: PlacementRequest) -> DeviceRequest:
    return DeviceRequest(
        workload=request.workload,
        preference=request.preference,
        precision=request.precision,
        required_operations=request.required_operations,
        allow_unverified_precision=request.allow_unverified_precision,
    )


def _try_target(
    request: PlacementRequest,
    target: ExecutionTarget,
    *,
    allowed: tuple[str, ...],
) -> ExecutionPlacementDecision | None:
    """A decision if this target works, or None to keep looking."""
    resolver = DeviceResolver(target.capability, allowed=allowed)
    device: DeviceDecision = resolver.resolve(_device_request(request))
    if not device.resolved:
        return None

    budget = budget_for(
        target.capability.accelerator_memory_mb(device.device or ComputeDevice.CPU.value),
        shared_with_control_plane=target.runs_control_plane,
    )
    memory = assess(budget, request.estimated_memory_mb)
    if memory.blocks:
        return None

    unknowns: list[str] = []
    if memory.verdict == MemoryVerdict.UNKNOWN.value:
        unknowns.append(memory.reason)
    if device.precision_unverified:
        unknowns.append(
            f"{device.precision} on {device.device} has not been measured on this target"
        )
    if target.capability.planned:
        unknowns.append(
            f"{target.name} is a planned profile; no machine matching it has been probed"
        )
    if is_training(request.workload) and not uses_device(request.workload):
        unknowns.append("workload classification disagrees with itself")

    return ExecutionPlacementDecision(
        outcome=PlacementOutcome.PLACED.value,
        workload=request.workload,
        policy=request.resolved_policy(),
        execution_location=target.location,
        compute_device=device.device,
        target_name=target.name,
        worker_id=target.worker_id,
        precision=device.precision,
        torch_device=device.torch_device,
        reason=f"{target.name}: {device.reason}",
        considered=(target.name,),
        limitations=device.limitations,
        fallback_used=device.fallback_used,
        memory=memory,
        capability_digest=target.capability.digest(),
        planned_target=target.capability.planned,
        unknowns=tuple(unknowns),
    )


def _last_reason(
    request: PlacementRequest, target: ExecutionTarget, allowed: tuple[str, ...]
) -> str:
    """Why one target was passed over, in the operator's words."""
    resolver = DeviceResolver(target.capability, allowed=allowed)
    device = resolver.resolve(_device_request(request))
    if not device.resolved:
        return device.reason
    budget = budget_for(
        target.capability.accelerator_memory_mb(device.device or ComputeDevice.CPU.value),
        shared_with_control_plane=target.runs_control_plane,
    )
    return assess(budget, request.estimated_memory_mb).reason


__all__ = [
    "DEFAULT_POLICY",
    "POLICY_DEVICES",
    "ExecutionPlacementDecision",
    "ExecutionTarget",
    "PlacementOutcome",
    "PlacementPolicy",
    "PlacementRequest",
    "place",
]
