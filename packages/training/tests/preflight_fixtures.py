"""Machines, plans and evidence for the Phase 33 preflight tests.

Every capability here is a **fixture**: a literal describing a machine
that may or may not exist, built so the decision logic can be tested
without owning the hardware. That is stated in every docstring below
and in the names, because the one thing that must never happen is a
CUDA fixture being read later as a measurement.

Nothing here benchmarks anything, and no number below is presented
anywhere as observed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from luber_hardware import DevicePrecisionSupport, MachineCapability
from luber_training.config import TrainingConfig
from luber_training.entities import (
    TrainingDatasetRef,
    TrainingWorker,
    WorkerCapabilities,
    WorkerClass,
)
from luber_training.plan import TrainingPlan, default_requirements
from luber_training.preflight import (
    CanaryEvidence,
    DatasetEvidence,
    PreflightRequest,
    RemoteEvidence,
    StorageEvidence,
    TrainerEvidence,
)

#: A fixed instant, so an identical evidence set produces an identical
#: result. A preflight whose digest moved with the clock could not be
#: compared with itself.
MEASURED_AT = "2026-08-22T12:00:00+00:00"

ALL_PRECISION = DevicePrecisionSupport(fp32=True, fp16=True, bf16=True)
NO_BF16 = DevicePrecisionSupport(fp32=True, fp16=True, bf16=False)

#: The curated manifest digest every plan below cites.
MANIFEST_DIGEST = "c" * 64


def apple_capability(
    *, mps: bool = True, memory_mb: int = 24576, label: str = "fixture-mac"
) -> MachineCapability:
    """A fixture Apple machine. Nothing here was probed."""
    return MachineCapability(
        label=label,
        system="Darwin",
        architecture="arm64",
        apple_silicon=True,
        cpu_model="Apple Silicon (fixture)",
        cpu_count=10,
        memory_total_mb=memory_mb,
        python_version="3.12.11",
        torch_installed=True,
        torch_version="2.10.0",
        mps_built=True,
        mps_available=mps,
        cuda_available=False,
        precision_support=(
            {"CPU": ALL_PRECISION, "MPS": ALL_PRECISION} if mps else {"CPU": ALL_PRECISION}
        ),
    )


def cuda_capability(
    *,
    memory_mb: int = 81920,
    devices: int = 1,
    precision: DevicePrecisionSupport | None = None,
    label: str = "fixture-gpu",
) -> MachineCapability:
    """A fixture NVIDIA machine that nobody owns.

    The card is unnamed on purpose. A fixture called "H100" ends up in a
    screenshot and then in an expectation, and this project has never
    had an NVIDIA GPU attached to it.
    """
    return MachineCapability(
        label=label,
        system="Linux",
        architecture="x86_64",
        apple_silicon=False,
        cpu_model="x86_64 (fixture)",
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


def cpu_capability(label: str = "fixture-cpu") -> MachineCapability:
    return MachineCapability(
        label=label,
        system="Linux",
        architecture="x86_64",
        cpu_model="x86_64 (fixture)",
        cpu_count=8,
        memory_total_mb=16384,
        python_version="3.12.11",
        torch_installed=True,
        torch_version="2.10.0",
        mps_built=False,
        mps_available=False,
        cuda_available=False,
        precision_support={"CPU": ALL_PRECISION},
    )


def a_plan(
    *,
    device: str | None = None,
    config: TrainingConfig | None = None,
    schema_version: str | None = None,
    manifest_digest: str = MANIFEST_DIGEST,
) -> TrainingPlan:
    training_config = config or TrainingConfig(epochs=1, rank=4, alpha=8)
    plan = TrainingPlan(
        plan_id="plan_fixture",
        run_id="run_fixture",
        experiment_id="exp_fixture",
        base_model_id="model_fixture",
        base_model_upstream_commit=training_config.ace_step_commit,
        dataset_ref=TrainingDatasetRef(
            dataset_id="ds_fixture",
            dataset_lock_sha256="a" * 64,
            curation_id="cur_fixture",
            curation_lock_sha256="b" * 64,
            curated_manifest_sha256=manifest_digest,
            manifest_artifact_ref="curation://cur_fixture/curated_manifest",
            selected_track_count=4,
        ),
        config=training_config,
        execution_backend="remote-gpu",
        requirements=default_requirements(training_config, device=device),
    )
    if schema_version is not None:
        object.__setattr__(plan, "schema_version", schema_version)
    return plan


def a_worker(
    *,
    worker_class: str = WorkerClass.GPU_TRAINING_READY.value,
    cuda: bool | None = True,
    reported_at: str | None = MEASURED_AT,
) -> TrainingWorker:
    return TrainingWorker(
        worker_id="worker_fixture",
        name="fixture-gpu",
        backend_type="remote-gpu",
        host_identity="fixture-host",
        worker_class=worker_class,
        capabilities=WorkerCapabilities(
            gpu_vendor="NVIDIA" if cuda else None,
            gpu_model="NVIDIA (test fixture)" if cuda else None,
            gpu_count=1 if cuda else None,
            vram_total_mb=81920 if cuda else None,
            cuda_available=cuda,
            bf16_supported=True if cuda else None,
            free_disk_mb=500_000,
            reported_by="fixture probe",
            reported_at=reported_at,
        ),
    )


def good_dataset(**overrides: Any) -> DatasetEvidence:
    """Data that is present, permitted and unchanged."""
    base: dict[str, Any] = {
        "manifest_present": True,
        "locks_verified": True,
        "eligible_sample_count": 4,
        "evaluation_only_count": 0,
        "observed_manifest_sha256": MANIFEST_DIGEST,
        "referenced_files_present": True,
        "missing_file_count": 0,
        "detail": "4 selected of 4 curated record(s)",
    }
    base.update(overrides)
    return DatasetEvidence(**base)


def good_trainer(**overrides: Any) -> TrainerEvidence:
    """A trainer installation that would actually start."""
    base: dict[str, Any] = {
        "trainer_root_present": True,
        "entrypoint_present": True,
        "interpreter_present": True,
        "torch_importable": True,
        "observed_ace_step_commit": TrainingConfig().ace_step_commit,
        "command_accepted": True,
        "command_detail": "the installed trainer accepts every flag in the compiled command",
    }
    base.update(overrides)
    return TrainerEvidence(**base)


def good_storage(**overrides: Any) -> StorageEvidence:
    base: dict[str, Any] = {
        "dataset_readable": True,
        "output_writable": True,
        "checkpoint_writable": True,
        "free_disk_mb": 200_000,
        "dataset_within_trainer_root": True,
    }
    base.update(overrides)
    return StorageEvidence(**base)


def good_remote(**overrides: Any) -> RemoteEvidence:
    base: dict[str, Any] = {
        "capability_age_seconds": 60.0,
        "reachable": True,
        "transport": "local",
    }
    base.update(overrides)
    return RemoteEvidence(**base)


def a_request(
    *,
    plan: TrainingPlan | None = None,
    capability: MachineCapability | None = None,
    location: str = "LOCAL",
    intent: str = "CANARY",
    worker: TrainingWorker | None = None,
    gate_report: Any = None,
    dataset: DatasetEvidence | None = None,
    trainer: TrainerEvidence | None = None,
    storage: StorageEvidence | None = None,
    remote: RemoteEvidence | None = None,
    canary: CanaryEvidence | None = None,
    capacity: Any = None,
    with_gates: bool = True,
) -> PreflightRequest:
    """A request that is READY unless a test breaks one part of it.

    ``with_gates=False`` supplies no gate report at all, which is the
    situation of a run nobody has validated — distinct from a report
    that failed.
    """
    from luber_training.gates import GateReport, GateResult

    report = gate_report
    if report is None and with_gates:
        report = GateReport(
            results=[
                GateResult(name=name, passed=True, detail="fixture gate")
                for name in (
                    "dataset_lock",
                    "curation_lock",
                    "rights",
                    "evaluation_leakage",
                    "self_generated",
                )
            ]
        )
    return PreflightRequest(
        plan=plan or a_plan(device="CPU"),
        capability=capability or cpu_capability(),
        execution_location=location,
        intent=intent,
        worker=worker,
        gate_report=report,
        dataset=dataset or good_dataset(),
        trainer=trainer or good_trainer(),
        storage=storage or good_storage(),
        remote=remote or good_remote(),
        canary=canary or CanaryEvidence(),
        capacity=capacity,
        measured_at=MEASURED_AT,
    )


def write_curated_manifest(directory: Path, records: list[dict[str, Any]]) -> Path:
    import json

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "curated_manifest.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    return path


__all__ = [
    "ALL_PRECISION",
    "MANIFEST_DIGEST",
    "MEASURED_AT",
    "NO_BF16",
    "a_plan",
    "a_request",
    "a_worker",
    "apple_capability",
    "cpu_capability",
    "cuda_capability",
    "good_dataset",
    "good_remote",
    "good_storage",
    "good_trainer",
    "write_curated_manifest",
]
