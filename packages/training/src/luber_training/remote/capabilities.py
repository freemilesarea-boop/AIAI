"""What a remote machine actually is, measured rather than assumed.

Phase 25's `probe_worker` answers the scheduler's question — can this
host take a CUDA job? This module answers the operator's: what is this
machine, exactly, and is it the same machine as last week?

Three things it adds.

**A machine-readable nvidia-smi parser.** `--query-gpu` with
`--format=csv` exists precisely so nobody has to parse the human table,
and every failure mode of the command — absent, non-zero, empty,
malformed, one row per GPU, `[N/A]` in a field — becomes UNKNOWN rather
than an exception or, far worse, a plausible number.

**A capability signature.** A digest over the properties that decide
whether a plan can run here: GPU model, count, VRAM, driver, CUDA,
torch, Python, architecture. Utilisation, free memory and temperature
are excluded — they change every second, and an identity that changed
every second could never establish that the machine is unchanged.

**Honest classification.** A host becomes CUDA_TRAINING only by
demonstrating CUDA through torch. Nothing here promotes a machine
because a field looked promising, and a Mac stays DEVELOPMENT_ONLY.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from luber_training.entities import WorkerCapabilities
from luber_training.remote.protocol import REMOTE_PROTOCOL_VERSION, now

#: How long any probe subprocess may take. Generous, because a busy GPU
#: box can be slow to answer, and short enough that a hung driver does
#: not hang the operator.
PROBE_TIMEOUT_SECONDS = 30.0

#: What nvidia-smi prints where it has no value.
_NOT_AVAILABLE = frozenset({"[n/a]", "n/a", "[not supported]", "not supported", "unknown", ""})

#: The fields queried, in order. Kept as one list so the query string
#: and the parser cannot drift apart.
GPU_QUERY_FIELDS: tuple[str, ...] = (
    "index",
    "name",
    "memory.total",
    "memory.used",
    "driver_version",
    "compute_cap",
    "uuid",
)

#: Telemetry fields, queried separately. They are volatile, so they are
#: never mixed into the identity query — a caller reading capabilities
#: should not accidentally receive something that changes every second.
TELEMETRY_QUERY_FIELDS: tuple[str, ...] = (
    "index",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)


class WorkerClassification(StrEnum):
    """What a worker may be used for, on the evidence.

    Wider than Phase 25's `WorkerClass` because remote execution
    distinguishes a host that can train from one that can only generate
    for evaluation — a smaller GPU may do the second and not the first.
    `to_worker_class` maps back for registry storage, so there is still
    one registry vocabulary.
    """

    DEVELOPMENT_ONLY = "DEVELOPMENT_ONLY"
    CUDA_TRAINING = "CUDA_TRAINING"
    CUDA_EVALUATION = "CUDA_EVALUATION"
    UNAVAILABLE = "UNAVAILABLE"


def to_worker_class(classification: str) -> str:
    """The Phase 25 class a remote classification corresponds to."""
    from luber_training.entities import WorkerClass

    if classification == WorkerClassification.CUDA_TRAINING.value:
        return WorkerClass.GPU_TRAINING_READY.value
    if classification == WorkerClassification.UNAVAILABLE.value:
        return WorkerClass.UNVERIFIED.value
    # CUDA_EVALUATION deliberately does not become GPU_TRAINING_READY. A
    # host that can run inference is not thereby a host that can train.
    return WorkerClass.DEVELOPMENT_ONLY.value


def _run(command: list[str], *, timeout: float = PROBE_TIMEOUT_SECONDS) -> str | None:
    """Run a probe command as argv, or return None if unusable.

    Never `shell=True`, and the binary is resolved through `which` so a
    missing tool is an absence rather than an exception.
    """
    binary = shutil.which(command[0])
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, *command[1:]],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _clean(value: str) -> str | None:
    """A field's value, or None where the driver had nothing to say."""
    text = value.strip()
    return None if text.lower() in _NOT_AVAILABLE else text


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value.replace("MiB", "").replace("W", "").strip()))
    except ValueError:
        return None


@dataclass
class GpuDevice:
    """One GPU, as the driver describes it."""

    index: int
    name: str | None = None
    memory_total_mb: int | None = None
    memory_used_mb: int | None = None
    driver_version: str | None = None
    compute_capability: str | None = None
    uuid: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_gpu_query(output: str | None) -> list[GpuDevice]:
    """Parse `--query-gpu ... --format=csv,noheader,nounits`.

    Written to survive everything a real driver does: blank lines, a
    trailing newline, fewer columns than requested on an old driver,
    `[N/A]` in any field, and a machine with several cards. A row that
    cannot even yield an index is dropped rather than guessed at,
    because a GPU whose identity is unknown is not a GPU this can
    schedule onto.
    """
    if not output:
        return []
    devices: list[GpuDevice] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        # A row shorter than the query means an older driver that does
        # not know some field. Pad rather than discard: the fields it
        # did answer are still true.
        while len(parts) < len(GPU_QUERY_FIELDS):
            parts.append("")
        index = _as_int(_clean(parts[0]))
        if index is None:
            continue
        devices.append(
            GpuDevice(
                index=index,
                name=_clean(parts[1]),
                memory_total_mb=_as_int(_clean(parts[2])),
                memory_used_mb=_as_int(_clean(parts[3])),
                driver_version=_clean(parts[4]),
                compute_capability=_clean(parts[5]),
                uuid=_clean(parts[6]),
            )
        )
    devices.sort(key=lambda device: device.index)
    return devices


def query_gpus() -> list[GpuDevice]:
    """Ask the driver about its GPUs. Empty where there are none."""
    output = _run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_gpu_query(output)


@dataclass
class GpuTelemetry:
    """A momentary reading. Never part of identity."""

    index: int
    utilization_pct: float | None = None
    memory_used_mb: int | None = None
    memory_total_mb: int | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    sampled_at: str = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace("%", "").replace("W", "").strip())
    except ValueError:
        return None


def parse_telemetry(output: str | None) -> list[GpuTelemetry]:
    if not output:
        return []
    readings: list[GpuTelemetry] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        while len(parts) < len(TELEMETRY_QUERY_FIELDS):
            parts.append("")
        index = _as_int(_clean(parts[0]))
        if index is None:
            continue
        readings.append(
            GpuTelemetry(
                index=index,
                utilization_pct=_as_float(_clean(parts[1])),
                memory_used_mb=_as_int(_clean(parts[2])),
                memory_total_mb=_as_int(_clean(parts[3])),
                temperature_c=_as_float(_clean(parts[4])),
                power_w=_as_float(_clean(parts[5])),
            )
        )
    return readings


def sample_telemetry() -> list[GpuTelemetry]:
    """Current GPU readings, or an empty list where there is no GPU.

    Empty rather than a list of nulls: "there is no GPU here" and "there
    is a GPU whose utilisation could not be read" are different facts,
    and the second is a device row with null fields.
    """
    output = _run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(TELEMETRY_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        timeout=10.0,
    )
    return parse_telemetry(output)


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _torch_facts() -> dict[str, Any]:
    """CUDA as torch sees it — the authority, over any driver tool.

    A driver can be installed while torch was built without CUDA, and
    training runs through torch. `nvidia-smi` describes the hardware;
    only torch can say whether this Python will reach it.
    """
    facts: dict[str, Any] = {}
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
            facts["fp16_supported"] = True
    except Exception:
        facts.setdefault("cuda_available", None)
    return facts


def _disk_facts(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        return {
            "free_disk_mb": int(usage.free / (1024 * 1024)),
            "total_disk_mb": int(usage.total / (1024 * 1024)),
        }
    except OSError:
        return {"free_disk_mb": None, "total_disk_mb": None}


def _ace_step_commit(trainer_root: Path | None) -> str | None:
    if trainer_root is None or not Path(trainer_root).is_dir():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(trainer_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


@dataclass
class CapabilityReport:
    """Everything a probe could establish about one machine.

    Every field is a measurement or ``None``. ``None`` means nobody
    could look — never zero, never a default. The scheduler treats an
    unreported capability as unsatisfied, which is the only reading that
    cannot put a run on hardware that will not hold it.
    """

    protocol_version: str = REMOTE_PROTOCOL_VERSION
    probed_at: str = field(default_factory=now)

    # ── platform ──
    os_name: str | None = None
    os_release: str | None = None
    architecture: str | None = None
    hostname: str | None = None
    python_version: str | None = None

    # ── compute ──
    cpu_count: int | None = None
    system_ram_mb: int | None = None
    free_disk_mb: int | None = None
    total_disk_mb: int | None = None
    filesystem: str | None = None

    # ── accelerator ──
    cuda_available: bool | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    gpu_vendor: str | None = None
    gpu_model: str | None = None
    gpu_count: int | None = None
    vram_total_mb: int | None = None
    bf16_supported: bool | None = None
    fp16_supported: bool | None = None
    gpus: list[GpuDevice] = field(default_factory=list)

    # ── software ──
    torch_version: str | None = None
    peft_version: str | None = None
    transformers_version: str | None = None
    ace_step_commit: str | None = None
    luber_commit: str | None = None

    #: Facts that could not be established, and why.
    unknown: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gpus"] = [device.to_dict() for device in self.gpus]
        return payload

    def classify(self) -> str:
        """What this machine may be used for, pessimistically.

        CUDA_TRAINING requires torch to demonstrate CUDA *and* a GPU to
        be visible. A host with a driver but no working torch is
        DEVELOPMENT_ONLY, because training would fail on it and finding
        that out after renting it is the expensive way to learn.
        """
        if self.cuda_available and (self.gpu_count or 0) >= 1:
            return WorkerClassification.CUDA_TRAINING.value
        return WorkerClassification.DEVELOPMENT_ONLY.value

    def to_worker_capabilities(self) -> WorkerCapabilities:
        """The Phase 25 shape, for the existing worker registry."""
        return WorkerCapabilities(
            gpu_vendor=self.gpu_vendor,
            gpu_model=self.gpu_model,
            gpu_count=self.gpu_count,
            vram_total_mb=self.vram_total_mb,
            system_ram_mb=self.system_ram_mb,
            cpu_count=self.cpu_count,
            cuda_available=self.cuda_available,
            cuda_version=self.cuda_version,
            driver_version=self.driver_version,
            torch_version=self.torch_version,
            python_version=self.python_version,
            bf16_supported=self.bf16_supported,
            free_disk_mb=self.free_disk_mb,
            reported_by=f"luber-remote probe on {self.os_name} {self.architecture}",
            reported_at=self.probed_at,
        )

    def signature(self) -> str:
        """A digest over what makes this machine this machine.

        Excludes everything volatile. Free disk, used VRAM, temperature
        and the probe timestamp all change between two probes of one
        unchanged host, and an identity that changed every probe could
        never answer the question it exists for: is this the same
        machine we verified?
        """
        stable = {
            "architecture": self.architecture,
            "os_name": self.os_name,
            "python_version": self.python_version,
            "cpu_count": self.cpu_count,
            "system_ram_mb": self.system_ram_mb,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "driver_version": self.driver_version,
            "gpu_vendor": self.gpu_vendor,
            "gpu_model": self.gpu_model,
            "gpu_count": self.gpu_count,
            "vram_total_mb": self.vram_total_mb,
            "bf16_supported": self.bf16_supported,
            "torch_version": self.torch_version,
            "ace_step_commit": self.ace_step_commit,
            "gpu_uuids": sorted(device.uuid for device in self.gpus if device.uuid),
        }
        return hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def probe(
    *,
    trainer_root: Path | None = None,
    repository_root: Path | None = None,
    disk_path: Path | None = None,
) -> CapabilityReport:
    """Measure this machine. Reports absence as absence.

    Safe to run on a Mac, a CPU-only Linux box or a GPU host. On a
    machine with no NVIDIA hardware the GPU fields stay ``None`` and
    `cuda_available` is False if torch is present to say so — which is
    the true answer, not a failure.
    """
    report = CapabilityReport()

    report.os_name = platform.system() or None
    report.os_release = platform.release() or None
    report.architecture = platform.machine() or None
    report.hostname = platform.node() or None
    report.python_version = sys.version.split()[0]
    report.cpu_count = os.cpu_count()

    try:
        report.system_ram_mb = int(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024)
        )
    except (ValueError, OSError, AttributeError):
        report.unknown.append("system_ram_mb: this platform does not expose physical memory")

    target = Path(disk_path or Path.cwd())
    disk = _disk_facts(target)
    report.free_disk_mb = disk["free_disk_mb"]
    report.total_disk_mb = disk["total_disk_mb"]
    if report.free_disk_mb is None:
        report.unknown.append(f"free_disk_mb: {target} could not be measured")

    report.gpus = query_gpus()
    if report.gpus:
        report.gpu_vendor = "NVIDIA"
        report.gpu_count = len(report.gpus)
        first = report.gpus[0]
        report.gpu_model = first.name
        report.vram_total_mb = first.memory_total_mb
        report.driver_version = first.driver_version
    elif shutil.which("nvidia-smi") is None:
        report.unknown.append(
            "GPU facts: nvidia-smi is not installed, so no NVIDIA hardware can be described"
        )
    else:
        report.unknown.append("GPU facts: nvidia-smi is present but produced no usable output")

    # torch last: where it disagrees with the driver about CUDA, torch
    # is right, because torch is what training runs through.
    torch_facts = _torch_facts()
    for key, value in torch_facts.items():
        setattr(report, key, value)
    if "torch_version" not in torch_facts:
        report.unknown.append(
            "cuda_available: torch is not installed here, so CUDA usability is unverified"
        )
        report.cuda_available = None

    report.peft_version = _package_version("peft")
    report.transformers_version = _package_version("transformers")
    report.ace_step_commit = _ace_step_commit(trainer_root)
    if report.ace_step_commit is None:
        report.unknown.append("ace_step_commit: no ACE-Step checkout was found to identify")

    if repository_root is not None:
        from luber_training.plan import capture_code_version

        code = capture_code_version(Path(repository_root))
        report.luber_commit = code.commit

    return report


__all__ = [
    "GPU_QUERY_FIELDS",
    "PROBE_TIMEOUT_SECONDS",
    "TELEMETRY_QUERY_FIELDS",
    "CapabilityReport",
    "GpuDevice",
    "GpuTelemetry",
    "WorkerClassification",
    "parse_gpu_query",
    "parse_telemetry",
    "probe",
    "query_gpus",
    "sample_telemetry",
    "to_worker_class",
]
