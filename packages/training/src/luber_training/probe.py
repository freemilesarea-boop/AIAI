"""Worker capability probe: report what is here, invent nothing.

Run on a machine to record what it actually has. Every field is either a
measurement or ``None``, and ``None`` means nobody looked — never zero,
never "probably fine". An invented VRAM figure is how a run gets
scheduled onto hardware that cannot hold it, and the scheduler treats an
unreported capability as unsatisfied rather than assumed.

``nvidia-smi`` is only invoked when it is on PATH. Calling it on an
Apple Silicon Mac produces a confusing error in the log and no
information, so the GPU fields simply stay unknown — which is the true
answer for a machine with no NVIDIA hardware.

The classification is deliberately pessimistic. A machine becomes
``GPU_TRAINING_READY`` only by demonstrating CUDA through torch;
everything else is ``DEVELOPMENT_ONLY``. A Mac does not become a
training worker by having the field filled in optimistically.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from luber_training.entities import WorkerCapabilities, WorkerClass

#: How long a probe subprocess may take before it is abandoned.
PROBE_TIMEOUT_SECONDS = 30.0


def _run(command: list[str]) -> str | None:
    """Run a probe command, or return None if it is unavailable."""
    binary = shutil.which(command[0])
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, *command[1:]],
            capture_output=True,
            text=True,
            check=False,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _nvidia_smi() -> dict[str, object]:
    """GPU facts from nvidia-smi, or nothing at all.

    Not called unless the binary exists. A machine without NVIDIA
    hardware reports unknown GPU fields, which is accurate — it is not
    a machine with zero VRAM, it is a machine nobody can measure VRAM on.
    """
    if shutil.which("nvidia-smi") is None:
        return {}

    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not query:
        return {}

    lines = [line.strip() for line in query.splitlines() if line.strip()]
    if not lines:
        return {}

    first = [part.strip() for part in lines[0].split(",")]
    facts: dict[str, object] = {"gpu_count": len(lines)}
    if len(first) >= 1:
        facts["gpu_model"] = first[0]
        facts["gpu_vendor"] = "NVIDIA"
    if len(first) >= 2:
        try:
            facts["vram_total_mb"] = int(float(first[1]))
        except ValueError:
            pass
    if len(first) >= 3:
        facts["driver_version"] = first[2]
    return facts


def _torch_facts() -> dict[str, object]:
    """CUDA availability as torch sees it.

    This is the authority for `cuda_available`, not the presence of
    nvidia-smi: a driver can exist while torch was built without CUDA,
    and training runs through torch.
    """
    facts: dict[str, object] = {}
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return facts

    facts["torch_version"] = str(torch.__version__)
    try:
        available = bool(torch.cuda.is_available())
        facts["cuda_available"] = available
        facts["cuda_version"] = getattr(torch.version, "cuda", None)
        if available:
            facts["gpu_count"] = int(torch.cuda.device_count())
            properties = torch.cuda.get_device_properties(0)
            facts["gpu_model"] = str(properties.name)
            facts["vram_total_mb"] = int(properties.total_memory / (1024 * 1024))
            facts["gpu_vendor"] = "NVIDIA"
            try:
                facts["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            except Exception:
                facts["bf16_supported"] = None
    except Exception:
        # torch present but CUDA introspection failed. Leave the fields
        # unknown rather than recording a guess.
        facts.setdefault("cuda_available", None)
    return facts


def _system_facts() -> dict[str, object]:
    facts: dict[str, object] = {
        "cpu_count": os.cpu_count(),
        "python_version": sys.version.split()[0],
    }
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        facts["system_ram_mb"] = int(page_size * pages / (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        facts["system_ram_mb"] = None
    try:
        usage = shutil.disk_usage(Path.cwd())
        facts["free_disk_mb"] = int(usage.free / (1024 * 1024))
    except OSError:
        facts["free_disk_mb"] = None
    return facts


def _as_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _as_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_str(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value.strip() else None


def probe_worker() -> tuple[WorkerCapabilities, str]:
    """Measure this machine and classify what it may be used for."""
    facts: dict[str, object] = {}
    facts.update(_system_facts())
    facts.update(_nvidia_smi())
    # torch last: it is the authority on CUDA, and overriding nvidia-smi
    # is correct when the two disagree.
    facts.update(_torch_facts())

    capabilities = WorkerCapabilities(
        gpu_vendor=_as_str(facts.get("gpu_vendor")),
        gpu_model=_as_str(facts.get("gpu_model")),
        gpu_count=_as_int(facts.get("gpu_count")),
        vram_total_mb=_as_int(facts.get("vram_total_mb")),
        system_ram_mb=_as_int(facts.get("system_ram_mb")),
        cpu_count=_as_int(facts.get("cpu_count")),
        cuda_available=_as_bool(facts.get("cuda_available")),
        cuda_version=_as_str(facts.get("cuda_version")),
        driver_version=_as_str(facts.get("driver_version")),
        torch_version=_as_str(facts.get("torch_version")),
        python_version=_as_str(facts.get("python_version")),
        bf16_supported=_as_bool(facts.get("bf16_supported")),
        free_disk_mb=_as_int(facts.get("free_disk_mb")),
        reported_by=f"luber-training probe on {platform.system()} {platform.machine()}",
        reported_at=datetime.now(UTC).isoformat(),
    )

    classification = (
        WorkerClass.GPU_TRAINING_READY.value
        if capabilities.cuda_available and (capabilities.gpu_count or 0) >= 1
        else WorkerClass.DEVELOPMENT_ONLY.value
    )
    return capabilities, classification


def write_capabilities(destination: Path) -> Path:
    capabilities, classification = probe_worker()
    payload = {
        "worker_class": classification,
        "capabilities": capabilities.to_dict(),
        "note": (
            "null means unmeasured, never zero. A worker with unreported CUDA cannot "
            "satisfy a plan that requires it."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination
