"""Which dtype a device actually gets, and why.

Two sources of truth, and they answer different questions.

**What the trainer will do.** ACE-Step's `_select_compute_dtype` and
`_select_fabric_precision` (`training_v2/fixed_lora_module.py` at the
pinned commit) resolve `auto` per device: bf16 on CUDA and XPU, fp16 on
MPS, fp32 everywhere else. LUBER does not get to have an opinion about
this — it is what will happen — so `AUTO_BY_DEVICE` below mirrors it
exactly, and if upstream changes, this table is wrong and must be
re-read rather than reasoned about.

**What the hardware can do.** Upstream passes an *explicit* precision
straight through without checking. Ask for bf16 on any device and you
get bf16, or you get whatever failure that device produces halfway
through the first epoch. So an explicit request is validated here,
against the probe — which allocates a tensor of that dtype on that
device and adds it to itself, because allocation can succeed where the
arithmetic raises and it is the arithmetic that training does.

The two together are the point. Without the first, LUBER would report a
precision the trainer was never going to use. Without the second, it
would accept a precision the machine cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from luber_hardware.capability import MachineCapability
from luber_hardware.devices import ComputeDevice, Precision
from luber_hardware.versions import PRECISION_POLICY_VERSION

#: What `auto` resolves to per device, mirroring the installed trainer.
#:
#: Read from `training_v2/fixed_lora_module.py` at pinned commit
#: 6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0. Not a preference — a
#: reproduction. LUBER reports what will happen, and what will happen is
#: decided upstream.
AUTO_BY_DEVICE: dict[str, str] = {
    ComputeDevice.CUDA.value: Precision.BF16.value,
    ComputeDevice.MPS.value: Precision.FP16.value,
    ComputeDevice.CPU.value: Precision.FP32.value,
}

#: The Lightning Fabric precision string each choice becomes upstream.
#: Carried so an operator reading a placement can match it against a
#: trainer log line without translating in their head.
FABRIC_PRECISION: dict[str, str] = {
    Precision.FP32.value: "32-true",
    Precision.FP16.value: "16-mixed",
    Precision.BF16.value: "bf16-mixed",
}


@dataclass(frozen=True)
class PrecisionDecision:
    """A resolved precision, or a refusal with the reason."""

    resolved: bool
    precision: str | None
    #: What `auto` became, when the request was `auto`.
    from_auto: bool = False
    #: The Fabric string the trainer will log.
    fabric_precision: str | None = None
    reason: str = ""
    #: True when the probe never measured this dtype on this device.
    unverified: bool = False
    policy_version: str = PRECISION_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved,
            "precision": self.precision,
            "from_auto": self.from_auto,
            "fabric_precision": self.fabric_precision,
            "reason": self.reason,
            "unverified": self.unverified,
            "precision_policy_version": self.policy_version,
        }


def resolve_precision(
    capability: MachineCapability,
    *,
    device: str,
    requested: str = Precision.AUTO.value,
    allow_unverified: bool = False,
) -> PrecisionDecision:
    """The dtype this device will actually use, or why it cannot.

    ``allow_unverified`` decides what happens when nobody measured the
    dtype on this device — which is the normal case for a remote worker
    whose probe predates this phase, and for a planned profile that
    describes hardware nobody has. Refusing by default would block every
    placement onto a machine that has not been re-probed; accepting
    silently would be the fabrication this phase exists to avoid. So the
    default refuses for an explicit request and the decision always
    carries `unverified` either way.
    """
    requested = (requested or Precision.AUTO.value).lower()
    known = {item.value for item in Precision}
    if requested not in known:
        return PrecisionDecision(
            resolved=False,
            precision=None,
            reason=(
                f"unknown precision {requested!r}; known: " + ", ".join(sorted(known - {"auto"}))
            ),
        )

    from_auto = requested == Precision.AUTO.value
    chosen = AUTO_BY_DEVICE.get(device, Precision.FP32.value) if from_auto else requested

    support = capability.supports_precision(device, chosen)

    if support is False:
        return PrecisionDecision(
            resolved=False,
            precision=None,
            from_auto=from_auto,
            reason=(
                f"{device} cannot compute in {chosen}: the probe allocated a {chosen} tensor "
                "there and the arithmetic failed"
                + (
                    f". The trainer resolves 'auto' to {chosen} on {device}, so 'auto' is "
                    "refused here too rather than silently downgraded"
                    if from_auto
                    else ""
                )
            ),
        )

    if support is None:
        message = (
            f"no probe has measured {chosen} on {device} for this target, so support is UNKNOWN"
        )
        if from_auto or allow_unverified:
            # `auto` is what the trainer would pick anyway. Refusing it
            # for lack of a measurement would block every unprobed
            # target from running at all, which is a worse answer than
            # proceeding with the gap recorded.
            return PrecisionDecision(
                resolved=True,
                precision=chosen,
                from_auto=from_auto,
                fabric_precision=FABRIC_PRECISION.get(chosen),
                reason=f"{chosen} on {device}; {message}",
                unverified=True,
            )
        return PrecisionDecision(
            resolved=False,
            precision=None,
            from_auto=from_auto,
            reason=(
                f"{message}. Probe the interpreter that runs training, or request 'auto' to "
                "accept the trainer's own default for this device"
            ),
            unverified=True,
        )

    return PrecisionDecision(
        resolved=True,
        precision=chosen,
        from_auto=from_auto,
        fabric_precision=FABRIC_PRECISION.get(chosen),
        reason=(
            f"the trainer resolves 'auto' to {chosen} on {device}, and the probe confirms it"
            if from_auto
            else f"{chosen} is measured working on {device}"
        ),
    )


def supported_precisions(capability: MachineCapability, device: str) -> tuple[str, ...]:
    """Every precision measured working on *device*. Empty when unprobed."""
    support = capability.precision_support.get(device)
    if support is None:
        return ()
    return tuple(
        name
        for name in (Precision.FP32.value, Precision.FP16.value, Precision.BF16.value)
        if support.supports(name)
    )


__all__ = [
    "AUTO_BY_DEVICE",
    "FABRIC_PRECISION",
    "PrecisionDecision",
    "resolve_precision",
    "supported_precisions",
]
