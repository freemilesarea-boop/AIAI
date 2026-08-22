"""Machines that do not exist, and one that might.

Every capability below is a literal. That is the point: placement has to
be testable without owning an NVIDIA GPU, and the alternative — skipping
the CUDA tests until somebody rents a machine — would leave the logic
that decides where training runs unexercised for exactly as long as it
takes to forget about it.

What is *not* faked is measurement. These fixtures describe hypothetical
machines to drive decision logic; nothing here produces a benchmark, and
no number below is presented anywhere as something that was observed.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from luber_hardware import (
    DevicePrecisionSupport,
    ExecutionTarget,
    MachineCapability,
)

#: Every dtype working. The common case on real hardware, and the one
#: that keeps a precision test from passing for the wrong reason.
ALL_PRECISION = DevicePrecisionSupport(fp32=True, fp16=True, bf16=True)

#: A device that cannot do bfloat16 — an older CUDA card, or a backend
#: where the dtype allocates and the arithmetic raises.
NO_BF16 = DevicePrecisionSupport(fp32=True, fp16=True, bf16=False)


def apple_machine(
    *,
    mps: bool = True,
    memory_mb: int = 24576,
    precision: DevicePrecisionSupport | None = None,
    label: str = "test-mac",
) -> MachineCapability:
    """An Apple Silicon machine with a working torch."""
    support = precision or ALL_PRECISION
    return MachineCapability(
        label=label,
        platform="darwin",
        system="Darwin",
        architecture="arm64",
        apple_silicon=True,
        cpu_model="Apple Silicon (test)",
        cpu_count=10,
        memory_total_mb=memory_mb,
        python_version="3.12.11",
        torch_installed=True,
        torch_version="2.10.0",
        mps_built=True,
        mps_available=mps,
        cuda_available=False,
        precision_support=(
            {"CPU": ALL_PRECISION, "MPS": support} if mps else {"CPU": ALL_PRECISION}
        ),
    )


def cuda_machine(
    *,
    memory_mb: int = 81920,
    devices: int = 1,
    precision: DevicePrecisionSupport | None = None,
    label: str = "test-gpu",
) -> MachineCapability:
    """A Linux machine with an NVIDIA GPU that nobody owns.

    The GPU is unnamed on purpose. A fixture called "H100" ends up in a
    screenshot, and then in an expectation.
    """
    return MachineCapability(
        label=label,
        platform="linux",
        system="Linux",
        architecture="x86_64",
        apple_silicon=False,
        cpu_model="x86_64 (test)",
        cpu_count=32,
        memory_total_mb=131072,
        python_version="3.12.11",
        torch_installed=True,
        torch_version="2.10.0",
        mps_built=False,
        mps_available=False,
        cuda_available=True,
        cuda_version="12.4",
        cuda_device_name="NVIDIA (test fixture, not a real card)",
        cuda_device_count=devices,
        cuda_device_memory_mb=memory_mb,
        cuda_bf16_supported=True,
        precision_support={"CPU": ALL_PRECISION, "CUDA": precision or ALL_PRECISION},
    )


def torchless_machine(label: str = "control-plane") -> MachineCapability:
    """The control plane: a real machine whose Python has no torch.

    Not a degenerate case — it is what LUBER's own environment looks
    like, and the probe has to describe it without claiming the machine
    has no accelerator.
    """
    return MachineCapability(
        label=label,
        platform="darwin",
        system="Darwin",
        architecture="arm64",
        apple_silicon=True,
        cpu_model="Apple Silicon (test)",
        cpu_count=10,
        memory_total_mb=24576,
        python_version="3.12.11",
        torch_installed=False,
    )


def mac_target(**kwargs) -> ExecutionTarget:
    return ExecutionTarget("mac", apple_machine(**kwargs), runs_control_plane=True)


def gpu_target(**kwargs) -> ExecutionTarget:
    return ExecutionTarget("gpu-1", cuda_machine(**kwargs), location="REMOTE", worker_id="worker-1")


# ── real hardware, when there is any ─────────────────────────────────


def torch_interpreter() -> str | None:
    """An interpreter with torch, or None.

    Checked in a fixed order: an explicitly named one, then this
    process, then the ACE-Step trainer's virtualenv if it happens to be
    installed in the conventional place. Nothing is searched for and
    nothing is downloaded — on a machine without torch this returns
    None and the tests that need one skip with a reason.
    """
    named = os.environ.get("LUBER_TORCH_PYTHON")
    if named and shutil.which(named):
        return named

    candidates = []
    if named:
        candidates.append(Path(named))
    candidates.append(Path(__import__("sys").executable))
    candidates.append(Path.home() / "ace-step-1.5" / ".venv" / "bin" / "python")

    for candidate in candidates:
        if not candidate.is_file():
            continue
        if _has_torch(candidate):
            return str(candidate)
    return None


def _has_torch(python: Path) -> bool:
    import subprocess

    try:
        completed = subprocess.run(
            [str(python), "-c", "import torch"],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


__all__ = [
    "ALL_PRECISION",
    "NO_BF16",
    "apple_machine",
    "cuda_machine",
    "gpu_target",
    "mac_target",
    "torch_interpreter",
    "torchless_machine",
]
