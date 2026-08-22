"""Turning registered workers into things placement can choose between.

The seam between Phase 27's worker registry and Phase 32's placement.
It exists here, in the training package, because it needs both — and
`luber_hardware` deliberately depends on nothing, so the translation
cannot live there without inverting that.

Two shapes are being reconciled.

`WorkerCapabilities` is Phase 27's, and it is CUDA-shaped: `gpu_vendor`,
`vram_total_mb`, `cuda_available`. That is right for the question it
answers — can this rented Linux box train — and it is a wire format the
remote protocol depends on, so Phase 32 does not touch it.

`MachineCapability` is Phase 32's, and it is device-shaped: it has to be
able to describe an Apple machine, a CPU-only control plane and an
unprobed host without any of them looking like a broken GPU.

Translating rather than replacing keeps both honest. Everything Phase 27
measured is carried across; everything it never asked about — MPS, the
precision table — arrives as `None`, which the resolver treats as
unmeasured rather than absent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from luber_hardware import (
    ExecutionLocation,
    ExecutionTarget,
    MachineCapability,
    probe_machine,
)
from luber_training.entities import TrainingWorker, WorkerClass

#: What a local target is called when nobody names it.
#:
#: A generic platform class, deliberately. Calling the development
#: machine "Mac mini" because one is planned would put a guess in an
#: operator's console; the probe supplies the real class.
LOCAL_TARGET_NAME = "control-plane"


def capability_from_worker(worker: TrainingWorker) -> MachineCapability:
    """A registered worker as Phase 32 sees it.

    The label is the worker's own name — operator-chosen, already in the
    console, and not a hostname this code invented. `host_identity` is
    deliberately not used: it can carry a user and a host, and a
    capability report has no business holding either.
    """
    return _capability(worker.name, worker.capabilities.to_dict())


def capability_from_worker_record(record: Mapping[str, Any]) -> MachineCapability:
    """The same translation, from the registry's own JSON.

    The operator console reads worker records as mappings rather than
    reconstructing entities, and rebuilding one here purely to read six
    fields back off it would be ceremony. Both paths call the same
    translation so they cannot answer differently.
    """
    return _capability(
        str(record.get("name") or record.get("worker_id") or "worker"),
        dict(record.get("capabilities") or {}),
    )


def _capability(name: str, reported: Mapping[str, Any]) -> MachineCapability:
    probed = reported.get("reported_by", "UNREPORTED") != "UNREPORTED"
    cuda_available = reported.get("cuda_available")

    return MachineCapability(
        label=name,
        location=ExecutionLocation.REMOTE.value,
        system="Linux" if cuda_available else None,
        cpu_count=_int(reported.get("cpu_count")),
        memory_total_mb=_int(reported.get("system_ram_mb")),
        python_version=_str(reported.get("python_version")),
        torch_installed=reported.get("torch_version") is not None,
        torch_version=_str(reported.get("torch_version")),
        # Phase 27 never asked about Metal, and a worker that was never
        # asked has not answered "no". A remote Linux GPU box does not
        # have it, but that is a conclusion for the probe to reach
        # rather than for this translation to assert.
        mps_built=None,
        mps_available=None,
        cuda_available=_bool(cuda_available),
        cuda_version=_str(reported.get("cuda_version")),
        cuda_device_name=_str(reported.get("gpu_model")),
        cuda_device_count=_int(reported.get("gpu_count")),
        cuda_device_memory_mb=_int(reported.get("vram_total_mb")),
        cuda_bf16_supported=_bool(reported.get("bf16_supported")),
        # Phase 27's probe predates the precision table, so it is empty
        # rather than assumed. An explicit precision request against
        # this worker comes back UNVERIFIED, which is the true state:
        # nobody has run a bf16 tensor on that machine.
        precision_support={},
        notes=(
            ()
            if probed
            else (
                f"{name} has never been probed; every capability below is "
                "unreported rather than absent",
            )
        ),
    )


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _str(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value.strip() else None


def target_from_worker(worker: TrainingWorker) -> ExecutionTarget:
    return ExecutionTarget(
        name=worker.name,
        capability=capability_from_worker(worker),
        location=ExecutionLocation.REMOTE.value,
        worker_id=worker.worker_id,
    )


def local_target(
    python_executable: str | None = None,
    *,
    name: str = LOCAL_TARGET_NAME,
    runs_control_plane: bool = True,
) -> ExecutionTarget:
    """This machine, probed through whichever interpreter is named.

    `runs_control_plane` defaults true because on the planned topology
    it is: the same Mac serves the API, holds Postgres and Redis, and
    runs the orchestrator. It raises the memory reservation, which is
    the point.
    """
    capability = probe_machine(python_executable or None, label=None)
    return ExecutionTarget(
        name=name,
        capability=capability,
        location=ExecutionLocation.LOCAL.value,
        runs_control_plane=runs_control_plane,
    )


def compute_targets(
    workers: Sequence[TrainingWorker],
    *,
    python_executable: str | None = None,
    include_local: bool = True,
) -> list[ExecutionTarget]:
    """Everywhere a workload could go on this deployment.

    Development-only workers are included rather than filtered. A
    machine registered as DEVELOPMENT_ONLY still shows in the console
    with that status, and placement refuses it on its capabilities —
    hiding it would make an operator wonder where it went.
    """
    targets: list[ExecutionTarget] = []
    if include_local:
        targets.append(local_target(python_executable))
    targets.extend(target_from_worker(worker) for worker in workers)
    return targets


def is_gpu_worker(worker: TrainingWorker) -> bool:
    return worker.worker_class == WorkerClass.GPU_TRAINING_READY.value


__all__ = [
    "LOCAL_TARGET_NAME",
    "capability_from_worker",
    "capability_from_worker_record",
    "compute_targets",
    "is_gpu_worker",
    "local_target",
    "target_from_worker",
]
