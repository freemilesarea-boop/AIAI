"""How much room there is, and how we know — kept as separate facts.

Every capacity number in this project is one of three things, and the
whole point of this module is that they never mix:

**MEASURED** — something on the machine reported it. `torch.mps` said
how much it had allocated; `shutil.disk_usage` said how many bytes were
free; the probe read the device's total memory. A measured figure names
what took it and when.

**ESTIMATED** — derived by stated arithmetic from a measured figure. The
derivation travels with the number, so a reader can disagree with the
multiplier instead of having to trust the result. An estimate is never
promoted to a measurement by being reasonable.

**UNKNOWN** — nobody looked, or nobody can. This is the common case for
real training workloads in this repository and it is not a soft pass:
`UNKNOWN` capacity makes a preflight UNVERIFIED, never READY.

Two properties are worth stating because they are the ones that would
be lost first.

**Apple unified memory is not VRAM.** A 24 GB Mac does not have 24 GB
of accelerator memory; it has 24 GB shared between the GPU, the CPU, the
operating system, Postgres, Redis and the browser somebody left open.
Evidence for MPS therefore carries `unified_memory=True`, and anything
rendering it must say so rather than putting the figure in a column
headed VRAM next to a GPU's.

**Nothing here extrapolates.** A peak allocation measured by a bounded
canary on four synthetic samples is a fact about that canary. It is
recorded as one, with its bounds attached, and no function in this
module turns it into a maximum model size.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from luber_hardware import ComputeDevice, MachineCapability

CAPACITY_SCHEMA_VERSION = "luber-training-capacity/1"


class EvidenceSource(StrEnum):
    """Where a capacity number came from. Never inferred from its value.

    ``DERIVED`` was added in Phase 34, for a figure that is arithmetic
    over measurements rather than a measurement or a guess — peak minus
    baseline, headroom remaining. It sits between MEASURED and ESTIMATED
    on purpose: every input to it was measured, and the subtraction is
    still somebody's choice of what to subtract.
    """

    MEASURED = "MEASURED"
    #: Arithmetic over measured values, with the arithmetic stated.
    DERIVED = "DERIVED"
    ESTIMATED = "ESTIMATED"
    UNKNOWN = "UNKNOWN"


#: Names this module uses for the things it can have evidence about.
#: Fixed strings so a reader and a test agree on what is missing.
DEVICE_MEMORY = "device_memory_mb"
HOST_MEMORY = "host_memory_mb"
FREE_DISK = "free_disk_mb"
REQUIRED_DISK = "required_disk_mb"
PEAK_DEVICE_ALLOCATION = "peak_device_allocation_mb"
TRAINING_MEMORY_REQUIREMENT = "training_memory_requirement_mb"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class CapacityEvidence:
    """One capacity fact, with its provenance attached.

    ``value_mb`` may be ``None`` only when the source is UNKNOWN. A
    MEASURED evidence with no value would be a measurement of nothing,
    and the constructor refuses it rather than letting it travel.
    """

    name: str
    source: str
    value_mb: int | None = None
    detail: str = ""
    #: What took the measurement — a probe, a runtime call, a canary.
    measured_by: str | None = None
    measured_at: str | None = None
    #: The arithmetic, for an ESTIMATED figure. Empty otherwise.
    derivation: str = ""
    #: True where the number is Apple unified memory shared with the
    #: operating system rather than dedicated accelerator memory.
    unified_memory: bool = False

    def __post_init__(self) -> None:
        if self.source != EvidenceSource.UNKNOWN.value and self.value_mb is None:
            raise ValueError(
                f"{self.name}: a {self.source} figure must have a value; "
                "an absent number is UNKNOWN"
            )
        if self.source == EvidenceSource.ESTIMATED.value and not self.derivation:
            raise ValueError(
                f"{self.name}: an ESTIMATED figure must state its derivation, or it is "
                "indistinguishable from a measurement"
            )

    @property
    def known(self) -> bool:
        return self.source != EvidenceSource.UNKNOWN.value

    @property
    def measured(self) -> bool:
        return self.source == EvidenceSource.MEASURED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "value_mb": self.value_mb,
            "detail": self.detail,
            "measured_by": self.measured_by,
            "measured_at": self.measured_at,
            "derivation": self.derivation,
            "unified_memory": self.unified_memory,
        }


@dataclass(frozen=True)
class CapacityReport:
    """Everything known about room to run, on one target.

    ``fits`` is deliberately absent. Whether a real training workload
    fits is the question nobody in this project can answer yet, and a
    boolean here would be the single most misleading field in the
    codebase.
    """

    device: str | None
    evidence: tuple[CapacityEvidence, ...] = ()
    schema_version: str = CAPACITY_SCHEMA_VERSION

    def by_name(self, name: str) -> CapacityEvidence | None:
        for item in self.evidence:
            if item.name == name:
                return item
        return None

    def sources(self) -> dict[str, list[str]]:
        """Evidence names grouped by how they were established."""
        grouped: dict[str, list[str]] = {source.value: [] for source in EvidenceSource}
        for item in self.evidence:
            grouped[item.source].append(item.name)
        return {key: sorted(value) for key, value in grouped.items()}

    def unknown_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(item.name for item in self.evidence if item.source == EvidenceSource.UNKNOWN)
        )

    @property
    def any_unknown(self) -> bool:
        return bool(self.unknown_names())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "device": self.device,
            "evidence": [item.to_dict() for item in self.evidence],
            "sources": self.sources(),
            "unknown": list(self.unknown_names()),
            "note": (
                "UNKNOWN capacity is not a pass. No LUBER configuration has a measured "
                "memory requirement, so whether a real training workload fits is unknown "
                "on every target."
            ),
        }


# ── the individual facts ─────────────────────────────────────────────


def device_memory_evidence(capability: MachineCapability, device: str) -> CapacityEvidence:
    """How much memory the selected device has, as the probe saw it.

    For CUDA this is the card's own memory and means what it usually
    means. For MPS and CPU it is the machine's total memory, which the
    accelerator shares with everything else — recorded with
    ``unified_memory`` set so no reader can mistake the two.
    """
    total = capability.accelerator_memory_mb(device)
    if total is None:
        return CapacityEvidence(
            name=DEVICE_MEMORY,
            source=EvidenceSource.UNKNOWN.value,
            detail=(f"no probe has reported how much memory {device} has on {capability.label}"),
        )
    unified = device in (ComputeDevice.MPS.value, ComputeDevice.CPU.value)
    return CapacityEvidence(
        name=DEVICE_MEMORY,
        source=EvidenceSource.MEASURED.value,
        value_mb=total,
        detail=(
            f"{total} MB of unified memory, shared with the operating system and "
            "everything else running on this machine; this is not dedicated VRAM and "
            "must not be compared with a GPU's figure"
            if unified
            else f"{total} MB of dedicated device memory reported by the probe"
        ),
        measured_by=f"hardware probe on {capability.label}",
        unified_memory=unified,
    )


def host_memory_evidence(capability: MachineCapability) -> CapacityEvidence:
    """The machine's total RAM, whatever the device is."""
    if capability.memory_total_mb is None:
        return CapacityEvidence(
            name=HOST_MEMORY,
            source=EvidenceSource.UNKNOWN.value,
            detail=f"no probe has reported system memory on {capability.label}",
        )
    return CapacityEvidence(
        name=HOST_MEMORY,
        source=EvidenceSource.MEASURED.value,
        value_mb=capability.memory_total_mb,
        detail=f"{capability.memory_total_mb} MB of system memory",
        measured_by=f"hardware probe on {capability.label}",
    )


def free_disk_evidence(free_mb: int | None, *, measured_by: str) -> CapacityEvidence:
    if free_mb is None:
        return CapacityEvidence(
            name=FREE_DISK,
            source=EvidenceSource.UNKNOWN.value,
            detail="free disk could not be measured on the execution target",
        )
    return CapacityEvidence(
        name=FREE_DISK,
        source=EvidenceSource.MEASURED.value,
        value_mb=free_mb,
        detail=f"{free_mb} MB free",
        measured_by=measured_by,
        measured_at=_now(),
    )


def training_memory_requirement() -> CapacityEvidence:
    """What a real LUBER training run needs. Nobody knows.

    A function rather than a constant so that the day somebody measures
    it, there is one place to put the number and one place that has to
    state how it was measured.
    """
    return CapacityEvidence(
        name=TRAINING_MEMORY_REQUIREMENT,
        source=EvidenceSource.UNKNOWN.value,
        detail=(
            "no LUBER configuration has a measured memory requirement on any device. "
            "Whether a production-scale run fits on a given accelerator is unknown, and "
            "a bounded canary does not answer it"
        ),
    )


def peak_allocation_evidence(
    peak_mb: int | None,
    *,
    device: str,
    measured_by: str,
    bounds: str,
) -> CapacityEvidence:
    """Peak device allocation observed during a bounded run.

    ``bounds`` is required and travels with the number, because the
    number is only meaningful beside them: "1 842 MB, at 4 synthetic
    samples for 1 epoch" is a fact, and "1 842 MB" on its own reads as a
    model's memory requirement, which it is not.
    """
    if peak_mb is None:
        return CapacityEvidence(
            name=PEAK_DEVICE_ALLOCATION,
            source=EvidenceSource.UNKNOWN.value,
            detail=f"the runtime reported no peak allocation for {device}",
        )
    return CapacityEvidence(
        name=PEAK_DEVICE_ALLOCATION,
        source=EvidenceSource.MEASURED.value,
        value_mb=peak_mb,
        detail=(
            f"{peak_mb} MB peak allocation on {device} under bounded conditions ({bounds}). "
            "This is a fact about that bounded run and may not be extrapolated to a "
            "production workload"
        ),
        measured_by=measured_by,
        measured_at=_now(),
    )


#: How much room a checkpoint write is given beyond the bytes it holds.
#:
#: Two, and it is a headroom factor rather than a size model: the
#: trainer writes an adapter and a training-state file, then rewrites
#: both at the next save while the previous one still exists. Doubling
#: the observed size is arithmetic anybody can check, which is the only
#: reason it is allowed to be here at all.
CHECKPOINT_DISK_HEADROOM = 2


def required_disk_evidence(
    checkpoint_bytes: int | None,
    *,
    checkpoints_expected: int = 1,
    headroom: int = CHECKPOINT_DISK_HEADROOM,
) -> CapacityEvidence:
    """Disk a bounded run needs, derived from a measured checkpoint.

    UNKNOWN until something has actually written a checkpoint. There is
    no formula here for "a LoRA of rank 16 is N megabytes": the only
    honest input is a size somebody observed, and until a canary has
    produced one this returns UNKNOWN rather than a plausible figure.
    """
    if checkpoint_bytes is None or checkpoint_bytes <= 0:
        return CapacityEvidence(
            name=REQUIRED_DISK,
            source=EvidenceSource.UNKNOWN.value,
            detail=(
                "no checkpoint size has been observed for this configuration, so the disk "
                "requirement cannot be derived. It is not guessed"
            ),
        )
    observed_mb = max(1, checkpoint_bytes // (1024 * 1024))
    total = observed_mb * max(1, checkpoints_expected) * max(1, headroom)
    return CapacityEvidence(
        name=REQUIRED_DISK,
        source=EvidenceSource.ESTIMATED.value,
        value_mb=total,
        detail=f"{total} MB for the bounded operation",
        derivation=(
            f"{observed_mb} MB observed for one checkpoint x {max(1, checkpoints_expected)} "
            f"checkpoint(s) x {max(1, headroom)} headroom (the trainer holds the previous "
            "checkpoint while writing the next)"
        ),
    )


def capacity_report(
    capability: MachineCapability,
    *,
    device: str,
    free_disk_mb: int | None = None,
    disk_measured_by: str = "control plane",
    checkpoint_bytes: int | None = None,
    checkpoints_expected: int = 1,
    peak_allocation_mb: int | None = None,
    peak_bounds: str = "",
    peak_measured_by: str = "",
) -> CapacityReport:
    """Everything this phase can say about room on one target."""
    evidence: list[CapacityEvidence] = [
        device_memory_evidence(capability, device),
        host_memory_evidence(capability),
        free_disk_evidence(free_disk_mb, measured_by=disk_measured_by),
        required_disk_evidence(checkpoint_bytes, checkpoints_expected=checkpoints_expected),
        training_memory_requirement(),
    ]
    if peak_allocation_mb is not None or peak_bounds:
        evidence.append(
            peak_allocation_evidence(
                peak_allocation_mb,
                device=device,
                measured_by=peak_measured_by or "bounded canary",
                bounds=peak_bounds or "bounded canary",
            )
        )
    return CapacityReport(device=device, evidence=tuple(evidence))


__all__ = [
    "CAPACITY_SCHEMA_VERSION",
    "CHECKPOINT_DISK_HEADROOM",
    "DEVICE_MEMORY",
    "FREE_DISK",
    "HOST_MEMORY",
    "PEAK_DEVICE_ALLOCATION",
    "REQUIRED_DISK",
    "TRAINING_MEMORY_REQUIREMENT",
    "CapacityEvidence",
    "CapacityReport",
    "EvidenceSource",
    "capacity_report",
    "device_memory_evidence",
    "free_disk_evidence",
    "host_memory_evidence",
    "peak_allocation_evidence",
    "required_disk_evidence",
    "training_memory_requirement",
]
