"""What a machine can do, normalized — and nothing about whose it is.

`MachineCapability` is the one shape the rest of this package reads.
It is built from `_facts.collect()` (this machine, or another
interpreter's) or from a synthetic mapping in a test, and after that
nothing downstream asks the operating system anything.

Three properties are deliberate.

**`None` means nobody looked.** Never zero, never "probably fine". A
capability that was not measured does not satisfy a requirement — the
same rule Phase 27's preflight already applies, for the same reason.

**Nothing personal is representable.** There is no field for a
hostname, a username, a home directory, a serial number or a MAC
address, so no report can leak one. `label` is an operator-supplied
nickname for a *target*, not a machine identity, and it defaults to a
generic platform class rather than to anything the machine knows about
its owner.

**The digest covers capability only.** Free disk and utilisation move
minute to minute; a digest that included them would change constantly
and mean nothing. What is hashed is the set of facts that decide
whether a workload can run here at all.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from luber_hardware.devices import ComputeDevice
from luber_hardware.versions import CAPABILITY_SCHEMA_VERSION, version_block

#: What a value reads as when nothing measured it.
UNKNOWN = "UNKNOWN"

#: Facts that decide what can run here. The digest covers exactly these,
#: in this order, and nothing else.
DIGEST_FIELDS: tuple[str, ...] = (
    "architecture",
    "system",
    "cpu_model",
    "cpu_count",
    "memory_total_mb",
    "torch_version",
    "mps_built",
    "mps_available",
    "cuda_available",
    "cuda_version",
    "cuda_device_name",
    "cuda_device_count",
    "cuda_device_memory_mb",
    "precision_support",
)


@dataclass(frozen=True)
class DevicePrecisionSupport:
    """Which dtypes hold a tensor and survive an add, per device.

    Measured by the probe rather than derived from a version, because
    the answer is not derivable from a version. This is what stops the
    project advertising bf16 on a backend because a config file
    contains the string.
    """

    fp32: bool | None = None
    fp16: bool | None = None
    bf16: bool | None = None

    def supports(self, precision: str) -> bool | None:
        return {"fp32": self.fp32, "fp16": self.fp16, "bf16": self.bf16}.get(precision)

    def to_dict(self) -> dict[str, Any]:
        return {"fp32": self.fp32, "fp16": self.fp16, "bf16": self.bf16}


@dataclass(frozen=True)
class MachineCapability:
    """One machine's operational facts, as some interpreter saw them."""

    # ── identity of the *target*, not of the host ────────────────────
    #: A generic platform class by default — "Apple Silicon (arm64)",
    #: not a hostname and not a product somebody hopes to buy.
    label: str = UNKNOWN
    #: LOCAL or REMOTE. A capability report describes a machine; where
    #: that machine sits relative to the control plane is context the
    #: caller supplies.
    location: str = "LOCAL"

    # ── platform ─────────────────────────────────────────────────────
    platform: str | None = None
    system: str | None = None
    architecture: str | None = None
    kernel_release: str | None = None
    apple_silicon: bool | None = None

    # ── cpu and memory ───────────────────────────────────────────────
    cpu_model: str | None = None
    cpu_count: int | None = None
    performance_core_count: int | None = None
    efficiency_core_count: int | None = None
    memory_total_mb: int | None = None

    # ── the interpreter that answered ────────────────────────────────
    python_version: str | None = None
    torch_installed: bool = False
    torch_version: str | None = None

    # ── accelerators ─────────────────────────────────────────────────
    mps_built: bool | None = None
    mps_available: bool | None = None
    cuda_available: bool | None = None
    cuda_version: str | None = None
    cudnn_version: int | None = None
    cuda_device_name: str | None = None
    cuda_device_count: int | None = None
    cuda_device_memory_mb: int | None = None
    cuda_bf16_supported: bool | None = None

    # ── what torch can do ────────────────────────────────────────────
    distributed_available: bool | None = None
    torch_compile_available: bool | None = None
    precision_support: dict[str, DevicePrecisionSupport] = field(default_factory=dict)

    #: True when this report describes hardware nobody has, and exists
    #: for planning. Never true for a probe result.
    planned: bool = False
    #: Why a field is missing, when the reason is worth carrying.
    notes: tuple[str, ...] = ()
    schema_version: str = CAPABILITY_SCHEMA_VERSION

    # ── questions the rest of the package asks ───────────────────────

    def devices(self) -> tuple[str, ...]:
        """Compute devices this machine can actually reach, best first.

        CPU is always present: an interpreter that can run Python can
        run a tensor on its CPU, and reporting otherwise would leave a
        machine with no way to validate a checkpoint.
        """
        out = []
        if self.cuda_available:
            out.append(ComputeDevice.CUDA.value)
        if self.mps_available:
            out.append(ComputeDevice.MPS.value)
        out.append(ComputeDevice.CPU.value)
        return tuple(out)

    def has_device(self, device: str) -> bool:
        return device in self.devices()

    def supports_precision(self, device: str, precision: str) -> bool | None:
        """Whether *device* can do *precision*, or None if unmeasured."""
        support = self.precision_support.get(device)
        return None if support is None else support.supports(precision)

    def accelerator_memory_mb(self, device: str) -> int | None:
        """Memory available to a device, where the concept applies.

        MPS reports the machine's unified memory because that is what it
        allocates from — there is no separate pool. That is also why a
        Mac's number must never be compared with a GPU's VRAM as though
        they meant the same thing: the Mac's figure is shared with the
        operating system and everything else running.
        """
        if device == ComputeDevice.CUDA.value:
            return self.cuda_device_memory_mb
        if device in (ComputeDevice.MPS.value, ComputeDevice.CPU.value):
            return self.memory_total_mb
        return None

    def digest(self) -> str:
        """A stable fingerprint of what this machine can do.

        Traceability, not identity: two identically configured machines
        produce the same digest, which is correct — the digest answers
        "would this run the same way here", not "which box is this".
        """
        payload: dict[str, Any] = {}
        for name in DIGEST_FIELDS:
            value = getattr(self, name, None)
            if name == "precision_support":
                value = {
                    key: item.to_dict() for key, item in sorted(self.precision_support.items())
                }
            payload[name] = value
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            **version_block(),
            "label": self.label,
            "location": self.location,
            "planned": self.planned,
            "platform": self.platform,
            "system": self.system,
            "architecture": self.architecture,
            "kernel_release": self.kernel_release,
            "apple_silicon": self.apple_silicon,
            "cpu_model": self.cpu_model,
            "cpu_count": self.cpu_count,
            "performance_core_count": self.performance_core_count,
            "efficiency_core_count": self.efficiency_core_count,
            "memory_total_mb": self.memory_total_mb,
            "python_version": self.python_version,
            "torch_installed": self.torch_installed,
            "torch_version": self.torch_version,
            "mps_built": self.mps_built,
            "mps_available": self.mps_available,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "cuda_device_name": self.cuda_device_name,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_memory_mb": self.cuda_device_memory_mb,
            "cuda_bf16_supported": self.cuda_bf16_supported,
            "distributed_available": self.distributed_available,
            "torch_compile_available": self.torch_compile_available,
            "precision_support": {
                key: item.to_dict() for key, item in sorted(self.precision_support.items())
            },
            "devices": list(self.devices()),
            "notes": list(self.notes),
            "capability_digest": self.digest(),
        }

    def render(self) -> str:
        """The plain form, for a CLI or a log line."""
        lines = [f"{self.label} ({self.location})"]
        if self.planned:
            lines[0] += "  [PLANNED PROFILE — no such machine has been probed]"
        lines.append(f"  cpu: {self.cpu_model or UNKNOWN} x {self.cpu_count or UNKNOWN}")
        lines.append(f"  memory: {_mb(self.memory_total_mb)}")
        lines.append(
            f"  torch: {self.torch_version or 'NOT INSTALLED'}"
            f"  (python {self.python_version or UNKNOWN})"
        )
        lines.append(f"  mps: built={_tri(self.mps_built)} available={_tri(self.mps_available)}")
        cuda = f"  cuda: available={_tri(self.cuda_available)}"
        if self.cuda_available:
            cuda += f" {self.cuda_device_name or UNKNOWN} x {self.cuda_device_count or UNKNOWN}"
            cuda += f" ({_mb(self.cuda_device_memory_mb)})"
        lines.append(cuda)
        lines.append(f"  devices: {', '.join(self.devices())}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def _mb(value: int | None) -> str:
    if value is None:
        return UNKNOWN
    return f"{value / 1024:.1f} GiB" if value >= 1024 else f"{value} MiB"


def _tri(value: bool | None) -> str:
    return UNKNOWN if value is None else ("yes" if value else "no")


def _support_from(raw: Any) -> dict[str, DevicePrecisionSupport]:
    out: dict[str, DevicePrecisionSupport] = {}
    if not isinstance(raw, dict):
        return out
    # The probe reports torch's own device strings; the rest of this
    # package speaks `ComputeDevice`. Translated once, here, rather than
    # at every reader.
    names = {
        "cpu": ComputeDevice.CPU.value,
        "mps": ComputeDevice.MPS.value,
        "cuda": ComputeDevice.CUDA.value,
    }
    for key, answers in raw.items():
        device = names.get(str(key).lower())
        if device is None or not isinstance(answers, dict):
            continue
        out[device] = DevicePrecisionSupport(
            fp32=_bool(answers.get("fp32")),
            fp16=_bool(answers.get("fp16")),
            bf16=_bool(answers.get("bf16")),
        )
    return out


def _bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _str(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value.strip() else None


def capability_from_facts(
    facts: dict[str, Any], *, label: str | None = None, location: str = "LOCAL"
) -> MachineCapability:
    """A normalized capability from a raw fact mapping.

    Every field is read defensively. The facts may have come from
    another interpreter's stdout, and a probe that crashed on a
    surprising value would be worse than one that reports it unknown.
    """
    notes: list[str] = []
    if not facts.get("torch_installed"):
        notes.append(
            "torch is not installed in the interpreter that answered, so no accelerator "
            "could be verified from here. Probe the interpreter that runs training."
        )

    return MachineCapability(
        label=label or _default_label(facts),
        location=location,
        platform=_str(facts.get("platform")),
        system=_str(facts.get("system")),
        architecture=_str(facts.get("architecture")),
        kernel_release=_str(facts.get("kernel_release")),
        apple_silicon=_bool(facts.get("apple_silicon")),
        cpu_model=_str(facts.get("cpu_model")),
        cpu_count=_int(facts.get("cpu_count")),
        performance_core_count=_int(facts.get("performance_core_count")),
        efficiency_core_count=_int(facts.get("efficiency_core_count")),
        memory_total_mb=_int(facts.get("memory_total_mb")),
        python_version=_str(facts.get("python_version")),
        torch_installed=bool(facts.get("torch_installed")),
        torch_version=_str(facts.get("torch_version")),
        mps_built=_bool(facts.get("mps_built")),
        mps_available=_bool(facts.get("mps_available")),
        cuda_available=_bool(facts.get("cuda_available")),
        cuda_version=_str(facts.get("cuda_version")),
        cudnn_version=_int(facts.get("cudnn_version")),
        cuda_device_name=_str(facts.get("cuda_device_name")),
        cuda_device_count=_int(facts.get("cuda_device_count")),
        cuda_device_memory_mb=_int(facts.get("cuda_device_memory_mb")),
        cuda_bf16_supported=_bool(facts.get("cuda_bf16_supported")),
        distributed_available=_bool(facts.get("distributed_available")),
        torch_compile_available=_bool(facts.get("torch_compile_available")),
        precision_support=_support_from(facts.get("dtype_support")),
        notes=tuple(notes),
    )


def _default_label(facts: dict[str, Any]) -> str:
    """A generic platform class — never a product name, never a host.

    "Apple Silicon (arm64)" is true of the machine this runs on and of
    every machine like it. Calling it a Mac mini because somebody plans
    to buy one would put a guess in an operator's dashboard.
    """
    system = _str(facts.get("system")) or "Unknown system"
    architecture = _str(facts.get("architecture")) or "unknown arch"
    if facts.get("apple_silicon"):
        return f"Apple Silicon ({architecture})"
    if facts.get("cuda_available"):
        return f"{system} + NVIDIA ({architecture})"
    return f"{system} ({architecture})"


__all__ = [
    "DIGEST_FIELDS",
    "UNKNOWN",
    "DevicePrecisionSupport",
    "MachineCapability",
    "capability_from_facts",
]
