"""Machines nobody has probed, clearly labelled as such.

A planned profile exists so a compatibility question can be asked before
the hardware arrives: *if* the deployment had a 24 GB Apple Silicon Mac
mini, would a light fine-tune be placeable on it? That is a useful
question and the answer is derivable from what the trainer supports.

Everything else about a planned profile is a hazard, so the constraints
are strict:

- `planned=True`, and every renderer says **PLANNED PROFILE** out loud.
- No performance figure. Not throughput, not step time, not a
  comparison with anything. Nothing here has been run.
- No precision table. Precision support is *measured*, and nobody has
  measured this machine. It stays absent, which makes an explicit
  precision request against a planned profile come back UNVERIFIED
  rather than confidently wrong.
- `mps_available=True` is the one claim made, and it is a claim about
  Apple silicon plus a torch with the Metal backend — not about a
  specific model number. It is the premise of the question, not a
  measurement.

The machine this repository is developed on is an Apple M4 Pro with
24 GB. It is a fair *compatibility* proxy for a base-M4 Mac mini with
the same memory and a poor *performance* proxy, because the two have
different core and GPU configurations. Nothing in this module or
anywhere else may extrapolate a measurement from one to the other.
"""

from __future__ import annotations

from luber_hardware.capability import MachineCapability
from luber_hardware.devices import ExecutionLocation

#: The near-term purchase this project is planning around.
PLANNED_MAC_MINI_24GB_LABEL = "Apple Silicon Mac mini, 24 GB (PLANNED)"

#: A generic CUDA worker, for exercising placement before one exists.
#: Deliberately unnamed as to model: writing "H100" or "RTX 5090" into a
#: profile would put a GPU nobody has rented into an operator's console.
PLANNED_CUDA_WORKER_LABEL = "NVIDIA CUDA worker (PLANNED)"


def planned_mac_mini_24gb() -> MachineCapability:
    """A 24 GB Apple Silicon Mac mini that does not exist yet.

    For compatibility planning only. It answers "would this be
    placeable" and never "how fast would it be".
    """
    return MachineCapability(
        label=PLANNED_MAC_MINI_24GB_LABEL,
        location=ExecutionLocation.LOCAL.value,
        planned=True,
        platform="darwin",
        system="Darwin",
        architecture="arm64",
        apple_silicon=True,
        # 24 GiB, as the memory reports it. The chip is deliberately not
        # named: this profile is about Apple silicon with this much
        # memory, not about one product.
        cpu_model="Apple Silicon (model not specified)",
        memory_total_mb=24576,
        torch_installed=True,
        mps_built=True,
        mps_available=True,
        cuda_available=False,
        # No precision table on purpose. Nobody has run a tensor on this
        # machine, so every precision question about it is UNKNOWN and
        # the resolver should say so.
        precision_support={},
        notes=(
            "PLANNED PROFILE — no machine matching this has been probed. Present so "
            "compatibility can be checked before the hardware exists.",
            "No performance figure may be attributed to this profile. Development happens "
            "on an M4 Pro with the same memory, which is a compatibility proxy and not a "
            "performance one.",
            "Precision support is deliberately empty: it is measured, and this has not "
            "been measured.",
        ),
    )


def planned_cuda_worker(*, memory_mb: int | None = None) -> MachineCapability:
    """A CUDA worker with no model name and no benchmark.

    For exercising placement logic and operator views before a real
    machine is rented. `memory_mb` is whatever the caller states it is;
    left `None` it stays unknown, which is the correct default for
    hardware nobody has looked at.
    """
    return MachineCapability(
        label=PLANNED_CUDA_WORKER_LABEL,
        location=ExecutionLocation.REMOTE.value,
        planned=True,
        system="Linux",
        architecture="x86_64",
        apple_silicon=False,
        torch_installed=True,
        mps_built=False,
        mps_available=False,
        cuda_available=True,
        cuda_device_count=1,
        cuda_device_memory_mb=memory_mb,
        precision_support={},
        notes=(
            "PLANNED PROFILE — no NVIDIA hardware has been probed by this project.",
            "The GPU model is deliberately unspecified. A profile naming a specific card "
            "would put hardware nobody has rented into an operator's console.",
            "No VRAM figure is assumed. Where one appears, a caller stated it.",
        ),
    )


__all__ = [
    "PLANNED_CUDA_WORKER_LABEL",
    "PLANNED_MAC_MINI_24GB_LABEL",
    "planned_cuda_worker",
    "planned_mac_mini_24gb",
]
