"""Raw hardware facts, collected by whichever interpreter is asked.

This module has **no LUBER imports and no third-party requirements**,
and that constraint is the whole design. LUBER's own environment has no
`torch`; the ACE-Step trainer's environment does. A probe that could
only ask its own process would report "torch is not installed" on the
control plane and never learn what the machine that actually trains can
do.

So the same file serves two callers. In-process it is imported and
`collect()` is called directly. Out-of-process it is executed as a
script by another interpreter — `python /path/to/_facts.py` — and prints
one JSON document to stdout. Running the file rather than shipping a
copy of a probe script is what keeps the two paths from drifting: there
is one implementation and no string of source code to maintain beside
it.

Every value is a measurement or `None`. `None` means nobody looked. It
never means zero, and it is never rounded up into a default — an
invented memory figure is how a run gets scheduled onto hardware that
cannot hold it.

**Nothing personal is collected.** No username, no home directory, no
hostname, no serial number, no MAC address, no environment variables.
The fields below are the ones an operator needs to decide where a
workload runs, and there is no field here for anything else to occupy.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from typing import Any

#: Bump when the shape of the collected mapping changes.
FACTS_VERSION = "hardware-facts/1"

#: How long a probe subprocess may take. A hung `sysctl` must not hang
#: an operator's terminal.
PROBE_TIMEOUT_SECONDS = 30.0


def _sysctl(name: str) -> str | None:
    """One macOS sysctl value, or None anywhere else."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value or None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cpu_model() -> str | None:
    """The chip's marketing name — a model, never an identity.

    "Apple M4 Pro" says what the machine can do. It is not a serial
    number and cannot be traced to a person or a host.
    """
    if sys.platform == "darwin":
        return _sysctl("machdep.cpu.brand_string")
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip() or None
        except OSError:
            return None
    return None


def _memory_total_mb() -> int | None:
    if sys.platform == "darwin":
        raw = _int(_sysctl("hw.memsize"))
        return None if raw is None else raw // (1024 * 1024)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None
    return int(page_size * pages / (1024 * 1024))


def _system_facts() -> dict[str, Any]:
    return {
        "platform": sys.platform,
        "system": platform.system(),
        "architecture": platform.machine(),
        "kernel_release": platform.release(),
        "python_version": sys.version.split()[0],
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        # Apple silicon splits its cores. A count of 14 that is really
        # 10 performance plus 4 efficiency matters to anybody sizing a
        # dataloader.
        "performance_core_count": _int(_sysctl("hw.perflevel0.logicalcpu")),
        "efficiency_core_count": _int(_sysctl("hw.perflevel1.logicalcpu")),
        "memory_total_mb": _memory_total_mb(),
        "apple_silicon": sys.platform == "darwin" and platform.machine() == "arm64",
    }


def _torch_facts() -> dict[str, Any]:
    """What torch says, and only what it says.

    torch is the authority here for the same reason Phase 27 gave it
    the last word over `nvidia-smi`: a driver can be installed while
    torch was built without CUDA, and Darwin can be arm64 while this
    particular torch has no MPS backend compiled in. Training runs
    through torch, so torch decides.
    """
    facts: dict[str, Any] = {"torch_installed": False}
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        # Not an error. The control plane has no torch and the honest
        # report is that this interpreter cannot reach any accelerator.
        return facts

    facts["torch_installed"] = True
    facts["torch_version"] = str(torch.__version__)

    # ── MPS ──────────────────────────────────────────────────────────
    # `is_built` and `is_available` are different questions and both are
    # worth recording: built-but-unavailable is a real state (a torch
    # compiled with MPS running on Intel hardware or under Rosetta), and
    # it explains why a Mac cannot train when it looks like it should.
    try:
        backends = getattr(torch.backends, "mps", None)
        facts["mps_built"] = bool(backends.is_built()) if backends else False
        facts["mps_available"] = bool(backends.is_available()) if backends else False
    except Exception:
        facts["mps_built"] = None
        facts["mps_available"] = None

    # ── CUDA ─────────────────────────────────────────────────────────
    try:
        available = bool(torch.cuda.is_available())
        facts["cuda_available"] = available
        facts["cuda_version"] = getattr(torch.version, "cuda", None)
        facts["cudnn_version"] = _cudnn_version(torch)
        if available:
            facts["cuda_device_count"] = int(torch.cuda.device_count())
            properties = torch.cuda.get_device_properties(0)
            facts["cuda_device_name"] = str(properties.name)
            facts["cuda_device_memory_mb"] = int(properties.total_memory // (1024 * 1024))
            try:
                facts["cuda_bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            except Exception:
                facts["cuda_bf16_supported"] = None
    except Exception:
        # torch is here but CUDA introspection failed. Leave the fields
        # unknown rather than reporting a False that was never measured.
        facts.setdefault("cuda_available", None)

    facts["distributed_available"] = _distributed_available(torch)
    facts["torch_compile_available"] = bool(getattr(torch, "compile", None))
    facts["dtype_support"] = _dtype_support(torch, facts)
    return facts


def _cudnn_version(torch: Any) -> Any:
    try:
        version = torch.backends.cudnn.version()
    except Exception:
        return None
    return None if version is None else int(version)


def _distributed_available(torch: Any) -> Any:
    try:
        return bool(torch.distributed.is_available())
    except Exception:
        return None


def _dtype_support(torch: Any, facts: dict[str, Any]) -> dict[str, Any]:
    """Whether each dtype can actually hold a tensor on each device.

    Probed rather than asserted, because the answer is not derivable
    from a version string. It is the smallest possible allocation — one
    element — and it is wrapped, because "this raised" is itself the
    result rather than a failure of the probe.

    This is what stops the project advertising bf16 on Apple silicon
    because a config file contains the string.
    """
    support: dict[str, Any] = {}
    devices = ["cpu"]
    if facts.get("mps_available"):
        devices.append("mps")
    if facts.get("cuda_available"):
        devices.append("cuda")

    for device in devices:
        answers: dict[str, Any] = {}
        for name, dtype in (
            ("fp32", torch.float32),
            ("fp16", torch.float16),
            ("bf16", torch.bfloat16),
        ):
            try:
                tensor = torch.zeros(1, dtype=dtype, device=device)
                # Touching it matters: allocation can succeed where the
                # first arithmetic op raises, and it is the arithmetic
                # that training does.
                _ = (tensor + tensor).sum().item()
                answers[name] = True
            except Exception:
                answers[name] = False
        support[device] = answers
    return support


def collect() -> dict[str, Any]:
    """Every fact this interpreter can establish about its machine."""
    facts: dict[str, Any] = {"facts_version": FACTS_VERSION}
    facts.update(_system_facts())
    facts.update(_torch_facts())
    return facts


if __name__ == "__main__":
    # The out-of-process contract: one JSON document on stdout, nothing
    # else. Anything printed beside it would corrupt the parse.
    print(json.dumps(collect(), sort_keys=True))
