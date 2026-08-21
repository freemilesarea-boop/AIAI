"""A synthetic operator registry, and the app that serves it.

Everything the console shows is built here from the *real* Phase 23-26
writers: the orchestrator creates the runs, the registry writes the
records, the dataset factory computes the lock digests. A hand-written
registry would let a read model pass against a shape nothing emits, and
the first place that would show up is production data rendering as
blanks.

No GPU, no SSH, no audio, no model weights, no network. Every hardware
figure is a value this file states; none of it is a measurement of
anything, and the checkpoints are ``MOCK`` or synthetic files of a few
bytes.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from luber_api.main import create_app
from luber_api.ops.context import OpsContext, context_from_settings
from luber_api.ops.readmodel import CONTROL_PREFLIGHT_NAME, GATE_REPORT_NAME, OPS_ARTIFACT_DIR
from luber_api.settings import get_settings
from luber_evaluation.registry import EvaluationRegistry
from luber_training.backends import DRY_RUN, REMOTE_GPU
from luber_training.config import preset
from luber_training.entities import (
    Checkpoint,
    CheckpointKind,
    CheckpointStatus,
    FailureCode,
    ModelBaseline,
    ModelStage,
    RunStatus,
    TrainingStrategySupport,
    TrainingWorker,
    WorkerCapabilities,
    WorkerClass,
    WorkerStatus,
)
from luber_training.ids import EntityKind, new_id
from luber_training.metrics import MetricEvent, MetricSource
from luber_training.orchestrator import Orchestrator
from luber_training.registry import Registry

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The Phase 25 test helpers already build a dataset and curation whose
#: locks the gates recompute and accept. Importing them rather than
#: writing a second set keeps one definition of what a valid build looks
#: like — two would drift, and the one that drifted would be the one
#: making a gate look like it passed.
_TRAINING_TESTS = REPO_ROOT / "packages" / "training" / "tests"
if str(_TRAINING_TESTS) not in sys.path:
    sys.path.insert(0, str(_TRAINING_TESTS))

from training_fixtures import build_locked_dataset, manifest_record  # noqa: E402

#: The operator token every test sends. A fixed literal so a test that
#: forgets to send it fails on the assertion rather than on entropy.
OPERATOR_TOKEN = "test-operator-token-0123456789"

ACE_STEP_COMMIT = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0"


def _stamp(seconds_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(seconds=seconds_ago)).isoformat()


@dataclass
class Scenario:
    """Identifiers a test needs in order to ask for something specific."""

    root: Path
    registry: Registry
    orchestrator: Orchestrator
    evaluations: EvaluationRegistry
    dataset_build_id: str
    curation_build_id: str
    model_id: str
    production_model_id: str
    experiment_id: str
    blocked_experiment_id: str
    run_ids: dict[str, str] = field(default_factory=dict)
    worker_ids: dict[str, str] = field(default_factory=dict)
    checkpoint_ids: dict[str, str] = field(default_factory=dict)
    evaluation_ids: dict[str, str] = field(default_factory=dict)
    candidate_ids: dict[str, str] = field(default_factory=dict)


def _write_metrics(directory: Path, run_id: str, *, simulated: bool, steps: int = 40) -> None:
    """Metric events in the shape the trainer and the worker emit.

    ``simulated`` is not cosmetic: a dry run's numbers carry
    ``SIMULATED`` and the console must never present them as
    measurements, so a test needs both kinds to tell the difference.
    """
    directory.mkdir(parents=True, exist_ok=True)
    source = MetricSource.SIMULATED.value if simulated else MetricSource.TRAINER.value
    lines: list[str] = []
    for step in range(1, steps + 1):
        loss = round(2.4 - (step * 0.02), 5)
        lines.append(
            json.dumps(
                MetricEvent(
                    run_id=run_id,
                    metric_name="train_loss",
                    value=loss,
                    source=source,
                    step=step,
                    # One epoch, because the SMOKE preset these runs cite
                    # declares one. A fixture whose metrics reported
                    # epoch 3 of 1 would be teaching the console to
                    # render a contradiction.
                    epoch=1,
                    timestamp=_stamp(steps - step),
                ).to_dict(),
                sort_keys=True,
            )
        )
        lines.append(
            json.dumps(
                MetricEvent(
                    run_id=run_id,
                    metric_name="learning_rate",
                    value=1e-4,
                    source=source,
                    step=step,
                    epoch=1,
                    unit="",
                    timestamp=_stamp(steps - step),
                ).to_dict(),
                sort_keys=True,
            )
        )
        if not simulated:
            lines.append(
                json.dumps(
                    MetricEvent(
                        run_id=run_id,
                        metric_name="gpu_memory_mb",
                        value=18_000.0 + step,
                        source=MetricSource.WORKER_TELEMETRY.value,
                        step=step,
                        unit="MB",
                        timestamp=_stamp(steps - step),
                    ).to_dict(),
                    sort_keys=True,
                )
            )
    (directory / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_logs(directory: Path, *, oom: bool = False) -> None:
    """Trainer output, including one line that must never reach a browser.

    The secret-shaped line is deliberate. Redaction is asserted against
    a log that actually contains something worth redacting, because a
    test over clean text proves only that the redactor did not corrupt
    it.
    """
    logs = directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = [
        "loading dataset manifest",
        "HF_TOKEN=hf_liveTokenValueThatMustNotLeak0000",
        "Authorization: Bearer sk-live-abcdefghijklmnop",
        "step 1 loss 2.38",
        "step 2 loss 2.36",
    ]
    stderr = ["warning: attention implementation falling back to sdpa"]
    if oom:
        stderr.append("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
    (logs / "trainer.stdout.log").write_text("\n".join(stdout) + "\n", encoding="utf-8")
    (logs / "trainer.stderr.log").write_text("\n".join(stderr) + "\n", encoding="utf-8")


def _gate_report(*, rights_passed: bool) -> dict[str, Any]:
    """A recorded Phase 25 gate report, passing or blocked on rights."""
    return {
        "passed": rights_passed,
        "failure_code": None if rights_passed else FailureCode.RIGHTS_GATE_FAILED.value,
        "gates": [
            {
                "name": "dataset_lock",
                "passed": True,
                "detail": "dataset ds-test-001 matches its lock (4 tracks)",
                "failure_code": None,
                "offending_ids": [],
                "offending_count": 0,
                "evidence": {},
            },
            {
                "name": "curation_lock",
                "passed": True,
                "detail": "curation cur-test-001 matches its lock",
                "failure_code": None,
                "offending_ids": [],
                "offending_count": 0,
                "evidence": {},
            },
            {
                "name": "rights",
                "passed": rights_passed,
                "detail": (
                    "every selected track is cleared for training"
                    if rights_passed
                    else "2 selected track(s) are not cleared for training"
                ),
                "failure_code": None if rights_passed else FailureCode.RIGHTS_GATE_FAILED.value,
                "offending_ids": [] if rights_passed else ["trk-0002", "trk-0003"],
                "offending_count": 0 if rights_passed else 2,
                "evidence": {},
            },
            {
                "name": "evaluation_leakage",
                "passed": True,
                "detail": "no evaluation-only material appears in the training selection",
                "failure_code": None,
                "offending_ids": [],
                "offending_count": 0,
                "evidence": {},
            },
            {
                "name": "self_generated",
                "passed": True,
                "detail": "no self-generated audio in the selection",
                "failure_code": None,
                "offending_ids": [],
                "offending_count": 0,
                "evidence": {},
            },
        ],
    }


def _control_preflight(run_id: str, *, ok: bool = True) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "ok": ok,
        "checks": {
            "gate:dataset_lock": "PASS",
            "gate:rights": "PASS" if ok else "FAIL",
            "code_version": "PASS",
            "worker_environment": "PASS",
            "worker_capacity": "PASS",
            "output_writable": "PASS",
            "disk_capacity": "UNKNOWN",
        },
        "problems": [] if ok else ["rights: 2 selected track(s) are not cleared for training"],
        "unknown": [
            "checkpoint size has never been measured for any LUBER configuration, so the "
            "disk requirement is unknown"
        ],
        "gates": {},
        "generated_at": _stamp(600),
    }


def _remote_preflight(run_id: str, worker_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "worker_id": worker_id,
        "protocol_version": "luber-remote/1",
        "schema_version": "luber-remote-preflight/1",
        "status": "PASS",
        "blocking_reasons": [],
        "checks": [
            {
                "name": "protocol_version",
                "status": "PASS",
                "detail": "worker and control plane speak luber-remote/1",
                "severity": "REQUIRED",
                "expected": "luber-remote/1",
                "observed": "luber-remote/1",
            },
            {
                "name": "artifact_digests",
                "status": "PASS",
                "detail": "every staged artifact rehashed to the digest it was sent with",
                "severity": "REQUIRED",
                "expected": None,
                "observed": None,
            },
            {
                "name": "disk_capacity",
                "status": "UNKNOWN",
                "detail": (
                    "no checkpoint size has ever been measured, so the requirement is unknown"
                ),
                "severity": "REQUIRED",
                "expected": None,
                "observed": None,
            },
        ],
        "created_at": _stamp(500),
    }


def _capabilities(*, cuda: bool) -> WorkerCapabilities:
    if not cuda:
        # The local Mac. Every GPU field stays None because nobody has
        # measured one, which is the distinction the console has to show.
        return WorkerCapabilities(
            cpu_count=12,
            system_ram_mb=36_864,
            python_version="3.11.9",
            free_disk_mb=220_000,
            reported_by="luber-remote probe on Darwin arm64",
            reported_at=_stamp(120),
        )
    return WorkerCapabilities(
        gpu_vendor="NVIDIA",
        gpu_model="NVIDIA A100-SXM4-40GB",
        gpu_count=1,
        vram_total_mb=40_960,
        system_ram_mb=131_072,
        cpu_count=32,
        cuda_available=True,
        cuda_version="12.4",
        driver_version="550.90.07",
        torch_version="2.5.1+cu124",
        python_version="3.11.9",
        bf16_supported=True,
        free_disk_mb=780_000,
        reported_by="luber-remote probe on Linux x86_64",
        reported_at=_stamp(60),
    )


def build_scenario(root: Path, *, bulk_runs: int = 0, bulk_workers: int = 0) -> Scenario:
    """Everything the console can show, in one registry.

    ``bulk_runs`` and ``bulk_workers`` pad the registry so a list page
    can be measured at a scale a real project reaches. They are plain
    records with no artifact directories: the point of them is the
    response size and the query cost, not another lifecycle.
    """
    registry = Registry(root / "registry")
    orchestrator = Orchestrator(registry, artifacts_root=root / "runs", repository_root=REPO_ROOT)
    evaluations = EvaluationRegistry(registry)

    # ── dataset and curation builds, locked by the real writers ──────
    staging = root / "_staging"
    records = [manifest_record(f"trk-{index:04d}") for index in range(1, 5)]
    dataset_dir, curation_dir = build_locked_dataset(staging, records)

    dataset_root = root / "builds" / "datasets"
    curation_root = root / "builds" / "curations"
    dataset_root.mkdir(parents=True, exist_ok=True)
    curation_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(dataset_dir), str(dataset_root / "primary"))
    shutil.move(str(curation_dir), str(curation_root / "primary"))

    # ── models ───────────────────────────────────────────────────────
    baseline = orchestrator.register_baseline(
        ModelBaseline(
            model_id=new_id(EntityKind.MODEL),
            provider="ACE-Step",
            model_family="acestep",
            model_name="acestep-v15-turbo",
            model_version="1.5",
            upstream_commit=ACE_STEP_COMMIT,
            architecture="DiT + VAE",
            training_strategy_support=[
                TrainingStrategySupport.LORA.value,
                TrainingStrategySupport.LOKR.value,
            ],
        )
    )
    production = orchestrator.register_baseline(
        ModelBaseline(
            model_id=new_id(EntityKind.MODEL),
            provider="ACE-Step",
            model_family="acestep",
            model_name="acestep-v15-turbo",
            model_version="1.5",
            upstream_commit=ACE_STEP_COMMIT,
            architecture="DiT + VAE",
            training_strategy_support=[TrainingStrategySupport.LORA.value],
            stage=ModelStage.PRODUCTION.value,
        )
    )

    # ── workers ──────────────────────────────────────────────────────
    workers: dict[str, str] = {}

    gpu = orchestrator.register_worker(
        TrainingWorker(
            worker_id=new_id(EntityKind.WORKER),
            name="rented-a100",
            backend_type=REMOTE_GPU,
            host_identity="fingerprint-a100-01",
            worker_class=WorkerClass.GPU_TRAINING_READY.value,
            status=WorkerStatus.BUSY.value,
            capabilities=_capabilities(cuda=True),
            software_environment={
                "remote_protocol_version": "luber-remote/1",
                "capability_signature": "cap" + "0" * 61,
                "remote_classification": "CUDA_TRAINING",
                "provider": "example-gpu-cloud",
                "instance_type": "1xA100-40GB",
                "hourly_rate": "1.29",
                "currency": "USD",
            },
            # A *reference*, never a key. The console reports that one
            # is configured and never what it is called.
            ssh_key_ref="operator-training-key",
        )
    )
    gpu.last_heartbeat = _stamp(4)
    registry.write("workers", gpu.worker_id, gpu.to_dict(), overwrite=True)
    workers["gpu"] = gpu.worker_id

    mac = orchestrator.register_worker(
        TrainingWorker(
            worker_id=new_id(EntityKind.WORKER),
            name="operator-mac",
            backend_type=DRY_RUN,
            host_identity="fingerprint-mac-01",
            worker_class=WorkerClass.DEVELOPMENT_ONLY.value,
            status=WorkerStatus.ONLINE.value,
            capabilities=_capabilities(cuda=False),
            software_environment={"remote_classification": "DEVELOPMENT_ONLY"},
        )
    )
    mac.last_heartbeat = _stamp(10)
    registry.write("workers", mac.worker_id, mac.to_dict(), overwrite=True)
    workers["mac"] = mac.worker_id

    stale = orchestrator.register_worker(
        TrainingWorker(
            worker_id=new_id(EntityKind.WORKER),
            name="rented-l40s",
            backend_type=REMOTE_GPU,
            host_identity="fingerprint-l40s-01",
            worker_class=WorkerClass.GPU_TRAINING_READY.value,
            status=WorkerStatus.ONLINE.value,
            capabilities=_capabilities(cuda=True),
            software_environment={"remote_protocol_version": "luber-remote/1"},
        )
    )
    # Past the stale threshold (300s) and short of offline (900s): the
    # registry still says ONLINE and the console must not.
    stale.last_heartbeat = _stamp(480)
    registry.write("workers", stale.worker_id, stale.to_dict(), overwrite=True)
    workers["stale"] = stale.worker_id

    # The box an operator rehearses orchestration on: a real GPU, probed,
    # and registered for the dry-run backend. It exists because every
    # compiled plan requires CUDA — Phase 25 will not let a dry run
    # launder a non-CUDA machine into a capable one — so a dry-run
    # dispatch still has to land somewhere a real run could.
    rehearsal = orchestrator.register_worker(
        TrainingWorker(
            worker_id=new_id(EntityKind.WORKER),
            name="rehearsal-box",
            backend_type=DRY_RUN,
            host_identity="fingerprint-rehearsal-01",
            worker_class=WorkerClass.GPU_TRAINING_READY.value,
            status=WorkerStatus.ONLINE.value,
            capabilities=_capabilities(cuda=True),
            software_environment={"remote_classification": "CUDA_TRAINING"},
        )
    )
    rehearsal.last_heartbeat = _stamp(8)
    registry.write("workers", rehearsal.worker_id, rehearsal.to_dict(), overwrite=True)
    workers["rehearsal"] = rehearsal.worker_id

    unverified = orchestrator.register_worker(
        TrainingWorker(
            worker_id=new_id(EntityKind.WORKER),
            name="unprobed-box",
            backend_type=REMOTE_GPU,
            host_identity="fingerprint-unknown-01",
            worker_class=WorkerClass.UNVERIFIED.value,
            capabilities=WorkerCapabilities(),
        )
    )
    workers["unverified"] = unverified.worker_id

    for index in range(bulk_workers):
        filler = orchestrator.register_worker(
            TrainingWorker(
                worker_id=new_id(EntityKind.WORKER),
                name=f"fleet-{index:03d}",
                backend_type=REMOTE_GPU,
                host_identity=f"fingerprint-fleet-{index:03d}",
                worker_class=WorkerClass.GPU_TRAINING_READY.value,
                capabilities=_capabilities(cuda=True),
                software_environment={"remote_protocol_version": "luber-remote/1"},
            )
        )
        filler.last_heartbeat = _stamp(30 + index)
        registry.write("workers", filler.worker_id, filler.to_dict(), overwrite=True)

    # ── experiments ──────────────────────────────────────────────────
    experiment = orchestrator.create_experiment(
        name="Korean vocal phrasing, reduced trot prior",
        hypothesis=(
            "Curating out trot-adjacent material improves modern Korean vocal phrasing "
            "without losing ballad warmth."
        ),
        base_model_id=baseline.model_id,
        description="First real LoRA experiment against the frozen P20 baseline.",
        operator="operator",
        tags=["korean", "vocal", "lora"],
    )
    blocked = orchestrator.create_experiment(
        name="English rap cadence",
        hypothesis="A rap-weighted curation improves English cadence.",
        base_model_id=baseline.model_id,
        operator="operator",
        tags=["english", "rap"],
        status="BLOCKED",
        blocked_reason="No rights-cleared English rap material exists in the library.",
    )

    dataset_lock = json.loads(
        (dataset_root / "primary" / "dataset_lock.json").read_text(encoding="utf-8")
    )
    curation_lock = json.loads(
        (curation_root / "primary" / "curation_lock.json").read_text(encoding="utf-8")
    )

    from luber_training.entities import TrainingDatasetRef

    dataset_ref = TrainingDatasetRef(
        dataset_id=str(dataset_lock["dataset_id"]),
        dataset_lock_sha256=str(curation_lock["source_dataset_lock_sha256"]),
        curation_id=str(curation_lock["curation_id"]),
        curation_lock_sha256=str(curation_lock["curated_manifest_sha256"]),
        curated_manifest_sha256=str(curation_lock["curated_manifest_sha256"]),
        manifest_artifact_ref=f"curation://{curation_lock['curation_id']}/curated_manifest",
        selected_track_count=int(curation_lock["selected_track_count"]),
        selected_hours=float(curation_lock["selected_hours"]),
    )

    runs: dict[str, str] = {}
    checkpoints: dict[str, str] = {}

    def _make_run(backend: str, worker_id: str | None) -> Any:
        return orchestrator.create_run(
            experiment_id=experiment.experiment_id,
            dataset_ref=dataset_ref,
            config=preset("SMOKE"),
            execution_backend=backend,
            worker_id=worker_id,
        )

    # 1. A draft, so validation and dispatch have something to refuse
    #    and then accept.
    draft = _make_run(DRY_RUN, rehearsal.worker_id)
    runs["draft"] = draft.run_id

    # 2. A completed dry run, with SIMULATED metrics and a MOCK
    #    checkpoint — the artifact that must never become a candidate.
    completed = _make_run(DRY_RUN, mac.worker_id)
    orchestrator.transition_run(completed.run_id, RunStatus.VALIDATING.value)
    orchestrator.transition_run(completed.run_id, RunStatus.QUEUED.value)
    orchestrator.transition_run(completed.run_id, RunStatus.STARTING.value)
    orchestrator.transition_run(completed.run_id, RunStatus.RUNNING.value)
    orchestrator.transition_run(completed.run_id, RunStatus.COMPLETED.value)
    completed_dir = root / "runs" / experiment.experiment_id / completed.run_id
    _write_metrics(completed_dir, completed.run_id, simulated=True, steps=10)
    (completed_dir / OPS_ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
    (completed_dir / OPS_ARTIFACT_DIR / GATE_REPORT_NAME).write_text(
        json.dumps(_gate_report(rights_passed=True), indent=2), encoding="utf-8"
    )
    mock_checkpoint = orchestrator.register_checkpoint(
        Checkpoint(
            checkpoint_id=new_id(EntityKind.CHECKPOINT),
            run_id=completed.run_id,
            kind=CheckpointKind.MOCK.value,
            step=10,
            epoch=1,
            status=CheckpointStatus.WRITING.value,
        )
    )
    orchestrator.finalize_checkpoint_record(
        mock_checkpoint.checkpoint_id,
        sha256="0" * 64,
        size_bytes=1024,
        reference=str(completed_dir / "checkpoints" / "mock"),
    )
    runs["completed"] = completed.run_id
    checkpoints["mock"] = mock_checkpoint.checkpoint_id

    # 3. A running remote job, with real-source metrics, telemetry,
    #    logs, a control preflight and a remote preflight.
    running = _make_run(REMOTE_GPU, gpu.worker_id)
    orchestrator.transition_run(running.run_id, RunStatus.VALIDATING.value)
    orchestrator.transition_run(running.run_id, RunStatus.QUEUED.value)
    orchestrator.transition_run(running.run_id, RunStatus.STARTING.value)
    orchestrator.transition_run(running.run_id, RunStatus.RUNNING.value)
    running_dir = root / "runs" / experiment.experiment_id / running.run_id
    _write_metrics(running_dir, running.run_id, simulated=False, steps=40)
    _write_logs(running_dir)
    ops_dir = running_dir / OPS_ARTIFACT_DIR
    ops_dir.mkdir(parents=True, exist_ok=True)
    (ops_dir / GATE_REPORT_NAME).write_text(
        json.dumps(_gate_report(rights_passed=True), indent=2), encoding="utf-8"
    )
    (ops_dir / CONTROL_PREFLIGHT_NAME).write_text(
        json.dumps(_control_preflight(running.run_id), indent=2), encoding="utf-8"
    )
    (running_dir / "environment_lock.json").write_text(
        json.dumps(
            {
                "digest": "env" + "1" * 61,
                "code": {"commit": "a6b4a7fafdd99f12e78fcda1d9096a6ac5bf0374", "dirty": False},
                "ace_step_commit": ACE_STEP_COMMIT,
                "python_version": "3.11.9",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    runs["running"] = running.run_id

    real_checkpoint = orchestrator.register_checkpoint(
        Checkpoint(
            checkpoint_id=new_id(EntityKind.CHECKPOINT),
            run_id=running.run_id,
            kind=CheckpointKind.ADAPTER.value,
            step=20,
            epoch=1,
            status=CheckpointStatus.WRITING.value,
            metrics_snapshot={"train_loss": 2.02},
        )
    )
    orchestrator.finalize_checkpoint_record(
        real_checkpoint.checkpoint_id,
        sha256="a" * 64,
        size_bytes=46_137_344,
        reference=str(running_dir / "checkpoints" / "step-20"),
    )
    checkpoints["adapter"] = real_checkpoint.checkpoint_id

    # 4. An OOM failure — the one classification that must stay
    #    definitive, and only where the trainer said so.
    oom = _make_run(REMOTE_GPU, gpu.worker_id)
    orchestrator.transition_run(oom.run_id, RunStatus.VALIDATING.value)
    orchestrator.transition_run(oom.run_id, RunStatus.QUEUED.value)
    orchestrator.transition_run(oom.run_id, RunStatus.STARTING.value)
    orchestrator.transition_run(oom.run_id, RunStatus.RUNNING.value)
    orchestrator.transition_run(
        oom.run_id,
        RunStatus.FAILED.value,
        error_code=FailureCode.OOM.value,
        error_message="CUDA out of memory. Tried to allocate 2.00 GiB",
    )
    oom_dir = root / "runs" / experiment.experiment_id / oom.run_id
    _write_metrics(oom_dir, oom.run_id, simulated=False, steps=6)
    _write_logs(oom_dir, oom=True)
    runs["oom"] = oom.run_id

    # 5. A rights refusal, recorded where the next operator will find it.
    rights = _make_run(REMOTE_GPU, gpu.worker_id)
    orchestrator.transition_run(rights.run_id, RunStatus.VALIDATING.value)
    orchestrator.transition_run(
        rights.run_id,
        RunStatus.FAILED.value,
        error_code=FailureCode.RIGHTS_GATE_FAILED.value,
        error_message="2 selected track(s) are not cleared for training",
    )
    rights_dir = root / "runs" / experiment.experiment_id / rights.run_id
    (rights_dir / OPS_ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
    (rights_dir / OPS_ARTIFACT_DIR / GATE_REPORT_NAME).write_text(
        json.dumps(_gate_report(rights_passed=False), indent=2), encoding="utf-8"
    )
    (rights_dir / OPS_ARTIFACT_DIR / CONTROL_PREFLIGHT_NAME).write_text(
        json.dumps(_control_preflight(rights.run_id, ok=False), indent=2), encoding="utf-8"
    )
    runs["rights_blocked"] = rights.run_id

    # 6. A lost worker. LOST, never FAILED: contact stopped, and nothing
    #    established that the trainer did.
    lost = _make_run(REMOTE_GPU, stale.worker_id)
    orchestrator.transition_run(lost.run_id, RunStatus.VALIDATING.value)
    orchestrator.transition_run(lost.run_id, RunStatus.QUEUED.value)
    orchestrator.transition_run(lost.run_id, RunStatus.STARTING.value)
    orchestrator.transition_run(lost.run_id, RunStatus.RUNNING.value)
    orchestrator.transition_run(
        lost.run_id,
        RunStatus.LOST.value,
        error_code=FailureCode.WORKER_LOST.value,
        error_message="the worker stopped reporting; remote state is unknown",
    )
    lost_dir = root / "runs" / experiment.experiment_id / lost.run_id
    _write_metrics(lost_dir, lost.run_id, simulated=False, steps=25)
    _write_logs(lost_dir)
    runs["worker_lost"] = lost.run_id

    # 7. A cancelled run, kept rather than erased.
    cancelled = _make_run(REMOTE_GPU, gpu.worker_id)
    orchestrator.transition_run(cancelled.run_id, RunStatus.VALIDATING.value)
    orchestrator.transition_run(cancelled.run_id, RunStatus.QUEUED.value)
    orchestrator.transition_run(
        cancelled.run_id,
        RunStatus.CANCELLED.value,
        error_code=FailureCode.CANCELLED_BY_OPERATOR.value,
        error_message="cancelled by operator",
    )
    runs["cancelled"] = cancelled.run_id

    # 8. A queued remote run, so dispatch has something to refuse for
    #    the right reason rather than because of its state.
    queued = _make_run(REMOTE_GPU, gpu.worker_id)
    orchestrator.transition_run(queued.run_id, RunStatus.VALIDATING.value)
    orchestrator.transition_run(queued.run_id, RunStatus.QUEUED.value)
    runs["queued_remote"] = queued.run_id

    for index in range(bulk_runs):
        filler = _make_run(DRY_RUN, mac.worker_id)
        orchestrator.transition_run(filler.run_id, RunStatus.VALIDATING.value)
        orchestrator.transition_run(filler.run_id, RunStatus.QUEUED.value)
        if index % 3 == 0:
            orchestrator.transition_run(filler.run_id, RunStatus.STARTING.value)
            orchestrator.transition_run(filler.run_id, RunStatus.RUNNING.value)
            orchestrator.transition_run(filler.run_id, RunStatus.COMPLETED.value)

    # ── candidates, evaluations, qualifications ──────────────────────
    candidate = orchestrator.create_candidate(
        real_checkpoint.checkpoint_id, notes="first real adapter"
    )
    candidates = {"adapter": candidate.candidate_id}

    evaluation_ids: dict[str, str] = {}
    for index, (label, outcome) in enumerate(
        (
            ("qualified", "QUALIFIED"),
            ("rejected", "REJECTED"),
            ("human_review", "HUMAN_REVIEW_REQUIRED"),
        ),
        start=1,
    ):
        # Hex, because Phase 26's id pattern is `eval_[0-9a-f]{16}` and a
        # fixture that used a shape the validator rejects would let a
        # boundary that checks ids pass here and fail on real data.
        evaluation_id = f"eval_{index:016x}"
        payload = {
            "schema_version": "luber-evaluation/1",
            "engine_version": "luber-evaluation/1.0.0",
            "evaluation_id": evaluation_id,
            "status": "COMPLETED",
            "mode": "RAW_MODEL",
            "suite_id": "p20-frozen",
            "suite_version": "1",
            "suite_digest": "suite" + "0" * 59,
            "policy_digest": "policy" + "0" * 58,
            "seeds": [11, 22, 33],
            "baseline": {
                "model_id": production.model_id,
                "upstream_commit": ACE_STEP_COMMIT,
                "checkpoint_id": None,
                "checkpoint_sha256": None,
                "label": "production baseline",
            },
            "candidate": {
                "model_id": baseline.model_id,
                "upstream_commit": ACE_STEP_COMMIT,
                "checkpoint_id": real_checkpoint.checkpoint_id,
                "checkpoint_sha256": "a" * 64,
                "label": f"candidate {label}",
            },
            "lineage": {
                "candidate_id": candidate.candidate_id,
                "checkpoint_id": real_checkpoint.checkpoint_id,
                "run_id": running.run_id,
                "experiment_id": experiment.experiment_id,
                "base_model_id": baseline.model_id,
                "dataset_id": dataset_ref.dataset_id,
                "dataset_lock_sha256": dataset_ref.dataset_lock_sha256,
                "curation_id": dataset_ref.curation_id,
                "curation_lock_sha256": dataset_ref.curation_lock_sha256,
                "training_config_sha256": running.config_sha256,
                "training_plan_sha256": "",
            },
            "experiment_hypothesis": experiment.hypothesis,
            "started_at": _stamp(3600),
            "completed_at": _stamp(3000),
            "cancelled_at": None,
            "failed_at": None,
            "error": None,
            "cost": {
                "wall_seconds": 600.0,
                "gpu_seconds": 540.0,
                "generated_audio_seconds": 720.0,
            },
        }
        evaluations.save_evaluation(payload)
        evaluations.save_qualification(
            {
                "schema_version": "luber-evaluation/1",
                "evaluation_id": evaluation_id,
                "candidate_id": candidate.candidate_id,
                "outcome": outcome,
                "policy_id": "NEUTRAL_CONSERVATIVE",
                "policy_version": "1",
                "policy_digest": "policy" + "0" * 58,
                "reasons": {
                    "QUALIFIED": ["every hard gate passed and no metric regressed"],
                    "REJECTED": ["lyric intelligibility regressed by a MAJOR margin"],
                    "HUMAN_REVIEW_REQUIRED": [
                        "the hypothesis is about vocal phrasing, which no technical metric measures"
                    ],
                }[outcome],
                "passed_gates": ["reliability", "silence_rate", "duration_accuracy"],
                "failed_gates": [] if outcome != "REJECTED" else ["lyric_intelligibility"],
                "inconclusive_gates": [] if outcome != "HUMAN_REVIEW_REQUIRED" else ["phrasing"],
                "gate_outcomes": [
                    {
                        "name": "reliability",
                        "passed": True,
                        "detail": "0 failures in 9 generations",
                        "severity": "NONE",
                        "inconclusive": False,
                    },
                    {
                        "name": "lyric_intelligibility",
                        "passed": outcome != "REJECTED",
                        "detail": (
                            "word error rate 0.41 against a baseline of 0.22"
                            if outcome == "REJECTED"
                            else "word error rate within tolerance"
                        ),
                        "severity": "MAJOR" if outcome == "REJECTED" else "NONE",
                        "inconclusive": False,
                    },
                ],
                "hypothesis_status": ("ADDRESSED" if outcome == "QUALIFIED" else "NOT_MEASURABLE"),
                "human_review_required_for": (
                    ["vocal_phrasing"] if outcome == "HUMAN_REVIEW_REQUIRED" else []
                ),
                "decided_at": _stamp(2900),
            }
        )
        artifacts = evaluations.artifacts(evaluation_id)
        artifacts.ensure()
        artifacts.comparisons_json.write_text(
            json.dumps(
                {
                    "comparisons": [
                        {
                            "metric_name": "lyric_intelligibility",
                            "verdict": "REGRESSED" if outcome == "REJECTED" else "UNCHANGED",
                            "baseline_value": 0.22,
                            "candidate_value": 0.41 if outcome == "REJECTED" else 0.23,
                            "delta": 0.19 if outcome == "REJECTED" else 0.01,
                            "severity": "MAJOR" if outcome == "REJECTED" else "NONE",
                            "detail": "word error rate over the frozen suite",
                        },
                        {
                            "metric_name": "generation_failure_rate",
                            "verdict": "UNCHANGED",
                            "baseline_value": 0.0,
                            "candidate_value": 0.0,
                            "delta": 0.0,
                            "severity": "NONE",
                            "detail": "no failures on either side",
                        },
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artifacts.report_md.write_text(
            f"# Evaluation {evaluation_id}\n\nOutcome: {outcome}\n",
            encoding="utf-8",
        )
        if outcome == "HUMAN_REVIEW_REQUIRED":
            artifacts.human_review_dir.mkdir(parents=True, exist_ok=True)
            (artifacts.human_review_dir / "request.json").write_text(
                json.dumps(
                    {
                        "request_id": "hrq_" + "0" * 16,
                        "evaluation_id": evaluation_id,
                        "mode": "LIGHT_AB",
                        "reason": "the hypothesis is about vocal phrasing",
                        "case_ids": ["case-01", "case-02", "case-03"],
                        "dimensions": ["vocal_phrasing"],
                        "status": "PENDING",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if outcome == "QUALIFIED":
            evaluations.save_promotion_review(
                {
                    "schema_version": "luber-evaluation/1",
                    "review_id": "rev_" + "0" * 16,
                    "candidate_id": candidate.candidate_id,
                    "evaluation_id": evaluation_id,
                    "qualification_outcome": outcome,
                    "decision": "HOLD",
                    "decided_by": "operator",
                    "rationale": "Qualified, but held until a second dataset confirms it.",
                    "decided_at": _stamp(2800),
                }
            )
        evaluation_ids[label] = evaluation_id

    return Scenario(
        root=root,
        registry=registry,
        orchestrator=orchestrator,
        evaluations=evaluations,
        dataset_build_id="primary",
        curation_build_id="primary",
        model_id=baseline.model_id,
        production_model_id=production.model_id,
        experiment_id=experiment.experiment_id,
        blocked_experiment_id=blocked.experiment_id,
        run_ids=runs,
        worker_ids=workers,
        checkpoint_ids=checkpoints,
        evaluation_ids=evaluation_ids,
        candidate_ids=candidates,
    )


def make_context(scenario: Scenario) -> OpsContext:
    """An `OpsContext` bound to a scenario, bypassing global settings.

    The registries are constructed from the scenario's own objects so a
    test never depends on process-wide configuration, and two tests
    running in one session cannot see each other's registry.
    """
    from luber_api.ops.context import CURATION_LOCK_NAME, DATASET_LOCK_NAME, BuildCatalogue

    settings = get_settings()
    return OpsContext(
        settings=settings,
        registry=scenario.registry,
        evaluations=scenario.evaluations,
        orchestrator=scenario.orchestrator,
        datasets=BuildCatalogue(
            root=scenario.root / "builds" / "datasets", lock_name=DATASET_LOCK_NAME
        ),
        curations=BuildCatalogue(
            root=scenario.root / "builds" / "curations", lock_name=CURATION_LOCK_NAME
        ),
        liveness=context_from_settings(settings).liveness,
    )


@pytest.fixture
def ops_environment(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Switch the console on for this test, and off again afterwards.

    Through the environment rather than by mutating a settings object,
    because that is how a deployment turns it on — and a test that
    reached past the switch would not prove the switch works.
    """
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("OPS_CONSOLE_ENABLED", "true")
    monkeypatch.setenv("OPS_OPERATOR_TOKEN", OPERATOR_TOKEN)
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def scenario(tmp_path: Path) -> Scenario:
    return build_scenario(tmp_path / "operator")


@pytest.fixture
def ops_app(ops_environment: Any, scenario: Scenario) -> FastAPI:
    application = create_app()
    application.state.ops_context = make_context(scenario)
    return application


@pytest.fixture
async def ops_client(ops_app: FastAPI) -> Any:
    """A client that sends the operator token on every request.

    Tests that need to prove the gate works build their own client
    without it — the point of this one is that every other test does not
    have to.
    """
    async with AsyncClient(
        transport=ASGITransport(app=ops_app),
        base_url="http://ops.test",
        headers={"X-Luber-Operator-Token": OPERATOR_TOKEN},
    ) as client:
        yield client
