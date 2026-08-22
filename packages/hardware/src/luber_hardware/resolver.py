"""One place that decides which device, and it explains itself.

The alternative is what most projects have: `if torch.cuda.is_available()`
scattered through the training modules, each with slightly different
fallback behaviour, none of them recorded. Then a run that trained on
the CPU for nine hours looks identical to one that used the GPU, and the
only evidence is the wall clock.

So every device question in LUBER comes here, and the answer is a
`DeviceDecision` carrying its own reasoning. A refusal is a first-class
answer with a named outcome — never a quiet substitution.

Two rules give this module its shape.

**AUTO is deterministic.** Given the same capability report it returns
the same device every time, in a fixed order: CUDA, then MPS, then CPU.
That order matches what the installed trainer's own `resolve_gpu` does,
so `AUTO` on LUBER's side and `auto` on the trainer's side cannot
disagree.

**An explicit request is never downgraded.** Ask for CUDA on a Mac and
the answer is BLOCKED, not "CPU, hope that's fine". Somebody who named a
device had a reason, and quietly giving them a slower one is how a
training run silently becomes a different experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from luber_hardware.capability import MachineCapability
from luber_hardware.devices import (
    ComputeDevice,
    ComputePreference,
    MpsFallbackPolicy,
    Precision,
    torch_device_string,
)
from luber_hardware.precision import PrecisionDecision, resolve_precision
from luber_hardware.versions import version_block
from luber_hardware.workloads import WorkloadClass, is_training

#: The order AUTO considers devices, mirroring the installed trainer's
#: `gpu_utils.resolve_gpu`: cuda → mps → (xpu) → cpu. XPU is absent
#: because LUBER has no Intel target and inventing one would be a
#: capability nobody can test.
AUTO_ORDER: tuple[str, ...] = (
    ComputeDevice.CUDA.value,
    ComputeDevice.MPS.value,
    ComputeDevice.CPU.value,
)


class DeviceOutcome(StrEnum):
    """What the resolver decided."""

    SELECTED = "SELECTED"
    #: The caller named a device this machine cannot reach.
    BLOCKED_DEVICE_UNAVAILABLE = "BLOCKED_DEVICE_UNAVAILABLE"
    #: The device is here; the precision is not possible on it.
    BLOCKED_PRECISION_UNSUPPORTED = "BLOCKED_PRECISION_UNSUPPORTED"
    #: AUTO looked at everything and nothing qualified.
    BLOCKED_NO_COMPATIBLE_DEVICE = "BLOCKED_NO_COMPATIBLE_DEVICE"
    #: An operation this workload needs would fall back to the CPU, and
    #: the policy is strict.
    BLOCKED_MPS_FALLBACK_REQUIRED = "BLOCKED_MPS_FALLBACK_REQUIRED"


@dataclass(frozen=True)
class DeviceRequest:
    """What the caller wants, before anything checks whether it exists."""

    workload: str = WorkloadClass.HEAVY_TRAINING.value
    preference: str = ComputePreference.AUTO.value
    precision: str = Precision.AUTO.value
    mps_fallback: str = MpsFallbackPolicy.STRICT.value
    #: Operations this run is known to need that MPS cannot do. Named by
    #: the caller because only the caller knows what its config asks
    #: for — `adamw8bit` is the real example, and it is a bitsandbytes
    #: import rather than a torch operator.
    required_operations: tuple[str, ...] = ()
    #: Accept a precision nobody has measured on this target. False by
    #: default; a remote worker probed before this phase has no
    #: precision table and should be re-probed rather than assumed.
    allow_unverified_precision: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "preference": self.preference,
            "precision": self.precision,
            "mps_fallback": self.mps_fallback,
            "required_operations": list(self.required_operations),
            "allow_unverified_precision": self.allow_unverified_precision,
        }


@dataclass(frozen=True)
class DeviceDecision:
    """A device and a precision, or a refusal that names the obstacle."""

    outcome: str
    device: str | None = None
    precision: str | None = None
    fabric_precision: str | None = None
    torch_device: str | None = None
    reason: str = ""
    #: Devices AUTO looked at and why each was passed over. An operator
    #: asking "why not the GPU" gets the answer without re-running
    #: anything.
    considered: tuple[str, ...] = ()
    #: True when CPU fallback for an unsupported MPS operation is
    #: permitted *and* something in this run will use it.
    fallback_used: bool = False
    limitations: tuple[str, ...] = ()
    precision_unverified: bool = False

    @property
    def resolved(self) -> bool:
        return self.outcome == DeviceOutcome.SELECTED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            **version_block(),
            "outcome": self.outcome,
            "device": self.device,
            "precision": self.precision,
            "fabric_precision": self.fabric_precision,
            "torch_device": self.torch_device,
            "reason": self.reason,
            "considered": list(self.considered),
            "fallback_used": self.fallback_used,
            "limitations": list(self.limitations),
            "precision_unverified": self.precision_unverified,
        }


#: Operations known to be unavailable on Apple's Metal backend.
#:
#: Deliberately short, and every entry is a fact about a *dependency*
#: rather than a torch operator. Torch's own MPS operator coverage moves
#: with each release, so a list of unimplemented kernels here would be
#: wrong within a version — that question belongs to the runtime, which
#: is why the precision table is probed rather than tabulated.
#:
#: `adamw8bit` is the one that matters today. ACE-Step's `optim.py`
#: imports `bitsandbytes.optim.AdamW8bit` and, when the import fails,
#: **logs a warning and uses AdamW instead**. A run configured for 8-bit
#: Adam on a Mac would therefore train with a different optimizer than
#: its own plan records. Refusing before the run beats discovering it in
#: a log line nobody read.
MPS_UNSUPPORTED_OPERATIONS: dict[str, str] = {
    "adamw8bit": (
        "the 8-bit optimizer comes from bitsandbytes, which has no Metal build; the "
        "trainer would silently fall back to AdamW and train something other than "
        "what the plan says"
    ),
    "fused_adamw": (
        "torch's fused AdamW is a CUDA path; on MPS the optimizer runs unfused, which "
        "is slower but not wrong"
    ),
    "nccl": "NCCL is CUDA-only collective communication; there is no Metal equivalent",
}

#: Operations that need CUDA specifically, whatever the alternative.
CUDA_ONLY_OPERATIONS: frozenset[str] = frozenset({"adamw8bit", "nccl"})


@dataclass
class DeviceResolver:
    """Answers device questions for one capability report."""

    capability: MachineCapability
    #: Devices this resolver may return at all, for a caller that has
    #: already narrowed the field — a placement policy, usually.
    allowed: tuple[str, ...] = field(default=AUTO_ORDER)

    def resolve(self, request: DeviceRequest) -> DeviceDecision:
        """The device and precision this request gets, or why not."""
        # A precision nobody has heard of is a mistake in the request,
        # not a shortcoming of the hardware. Caught before any device is
        # considered, so the answer says "fp8 is not a precision" rather
        # than working through three devices and concluding that none of
        # them is compatible.
        known = {item.value for item in Precision}
        requested = (request.precision or Precision.AUTO.value).lower()
        if requested not in known:
            return DeviceDecision(
                outcome=DeviceOutcome.BLOCKED_PRECISION_UNSUPPORTED.value,
                reason=(
                    f"unknown precision {requested!r}; known: "
                    + ", ".join(sorted(known - {Precision.AUTO.value}))
                ),
            )

        preference = (request.preference or ComputePreference.AUTO.value).upper()
        if preference == ComputePreference.AUTO.value:
            return self._auto(request)
        return self._explicit(request, preference)

    # ── explicit ─────────────────────────────────────────────────────

    def _explicit(self, request: DeviceRequest, device: str) -> DeviceDecision:
        """A named device. Available or blocked; never substituted."""
        if device not in self.allowed:
            return DeviceDecision(
                outcome=DeviceOutcome.BLOCKED_DEVICE_UNAVAILABLE.value,
                reason=(
                    f"{device} is not permitted for a {request.workload} workload here; "
                    f"permitted: {', '.join(self.allowed)}"
                ),
                considered=(device,),
            )
        if not self.capability.has_device(device):
            return DeviceDecision(
                outcome=DeviceOutcome.BLOCKED_DEVICE_UNAVAILABLE.value,
                reason=self._why_absent(device),
                considered=(device,),
            )
        verdict = self._try(request, device)
        if verdict.resolved:
            return verdict
        # An explicit request that fails is a refusal. Trying the next
        # device down would answer a question nobody asked.
        return verdict

    def _why_absent(self, device: str) -> str:
        """Why this machine cannot reach a device, in the operator's terms."""
        if device == ComputeDevice.CUDA.value:
            if not self.capability.torch_installed:
                return (
                    "CUDA was requested, but the interpreter that answered has no torch, so "
                    "no GPU could be verified. Probe the interpreter that runs training."
                )
            return (
                "CUDA was requested and torch reports no CUDA device on this machine. "
                "This is not downgraded to CPU: a CUDA request is a statement about what "
                "the run needs"
            )
        if device == ComputeDevice.MPS.value:
            if self.capability.mps_built is False:
                return "MPS was requested, but this torch was built without the Metal backend"
            if not self.capability.torch_installed:
                return (
                    "MPS was requested, but the interpreter that answered has no torch. "
                    "Probe the interpreter that runs training."
                )
            return "MPS was requested and torch reports it unavailable on this machine"
        return f"{device} is not available on this machine"

    # ── auto ─────────────────────────────────────────────────────────

    def _auto(self, request: DeviceRequest) -> DeviceDecision:
        """Best available device, in a fixed order, with the rejections kept.

        Deterministic by construction: the order is a constant and the
        capability report is a value. The same report gives the same
        answer on every call and in every process.
        """
        considered: list[str] = []
        rejections: list[str] = []
        for device in AUTO_ORDER:
            if device not in self.allowed:
                continue
            considered.append(device)
            if not self.capability.has_device(device):
                rejections.append(f"{device}: not available on this machine")
                continue
            verdict = self._try(request, device)
            if verdict.resolved:
                return DeviceDecision(
                    outcome=verdict.outcome,
                    device=verdict.device,
                    precision=verdict.precision,
                    fabric_precision=verdict.fabric_precision,
                    torch_device=verdict.torch_device,
                    reason=(
                        f"AUTO selected {device}"
                        + (f" (passed over: {'; '.join(rejections)})" if rejections else "")
                        + f". {verdict.reason}"
                    ),
                    considered=tuple(considered),
                    fallback_used=verdict.fallback_used,
                    limitations=verdict.limitations,
                    precision_unverified=verdict.precision_unverified,
                )
            rejections.append(f"{device}: {verdict.reason}")

        return DeviceDecision(
            outcome=DeviceOutcome.BLOCKED_NO_COMPATIBLE_DEVICE.value,
            reason=(
                "no permitted device can run this workload. "
                + ("; ".join(rejections) if rejections else "no device was permitted")
            ),
            considered=tuple(considered),
        )

    # ── one candidate ────────────────────────────────────────────────

    def _try(self, request: DeviceRequest, device: str) -> DeviceDecision:
        """Whether one device can carry this request."""
        limitations: list[str] = []
        fallback_used = False

        if device != ComputeDevice.CUDA.value:
            blocking = [
                name for name in request.required_operations if name in CUDA_ONLY_OPERATIONS
            ]
            if blocking:
                details = "; ".join(
                    f"{name}: {MPS_UNSUPPORTED_OPERATIONS.get(name, 'requires CUDA')}"
                    for name in blocking
                )
                strict = request.mps_fallback == MpsFallbackPolicy.STRICT.value
                if strict or is_training(request.workload):
                    return DeviceDecision(
                        outcome=DeviceOutcome.BLOCKED_MPS_FALLBACK_REQUIRED.value,
                        reason=f"{device} cannot run {', '.join(blocking)} — {details}",
                        considered=(device,),
                    )
                # Fallback permitted, and recorded rather than assumed.
                fallback_used = True
                limitations.append(f"CPU fallback in use for {', '.join(blocking)} — {details}")

        precision: PrecisionDecision = resolve_precision(
            self.capability,
            device=device,
            requested=request.precision,
            allow_unverified=request.allow_unverified_precision,
        )
        if not precision.resolved:
            return DeviceDecision(
                outcome=DeviceOutcome.BLOCKED_PRECISION_UNSUPPORTED.value,
                device=device,
                reason=precision.reason,
                considered=(device,),
                precision_unverified=precision.unverified,
            )

        for name in request.required_operations:
            if device == ComputeDevice.MPS.value and name in MPS_UNSUPPORTED_OPERATIONS:
                if name not in CUDA_ONLY_OPERATIONS:
                    limitations.append(f"{name}: {MPS_UNSUPPORTED_OPERATIONS[name]}")

        return DeviceDecision(
            outcome=DeviceOutcome.SELECTED.value,
            device=device,
            precision=precision.precision,
            fabric_precision=precision.fabric_precision,
            torch_device=torch_device_string(device),
            reason=precision.reason,
            considered=(device,),
            fallback_used=fallback_used,
            limitations=tuple(limitations),
            precision_unverified=precision.unverified,
        )


__all__ = [
    "AUTO_ORDER",
    "CUDA_ONLY_OPERATIONS",
    "MPS_UNSUPPORTED_OPERATIONS",
    "DeviceDecision",
    "DeviceOutcome",
    "DeviceRequest",
    "DeviceResolver",
]
