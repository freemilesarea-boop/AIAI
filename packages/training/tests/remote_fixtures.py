"""A control plane and a worker, on one machine, in separate directories.

The point of this harness is that almost nothing about it is fake. The
worker is the real worker CLI in a real subprocess with its own root.
The transport performs real writes, hashes and renames. The trainer is a
real process in its own process group that really has to be signalled.
What is simulated is the network, and only the network.

That matters because the failures worth testing — a transfer cut off
halfway, an acknowledgement that never arrived, a worker that stops
answering — are all failures *of* the network, and a harness that
simulated the worker too would be testing agreement between two mocks.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from luber_training.config import TrainingConfig
from luber_training.entities import TrainingDatasetRef
from luber_training.plan import HardwareRequirements, TrainingPlan
from luber_training.remote.client import LocalWorkerClient
from luber_training.remote.manifest import ArtifactEntry, ArtifactRole, RemoteArtifactManifest
from luber_training.remote.transport import LocalArtifactTransport
from luber_training.remote.worker import RemoteWorker

SYNTHETIC_TRAINER = Path(__file__).parent / "synthetic_trainer.py"


@dataclass
class Harness:
    """One control plane, one worker, and the paths between them."""

    root: Path
    worker_root: Path
    trainer_root: Path
    staging_root: Path
    collect_root: Path
    worker: RemoteWorker
    client: LocalWorkerClient
    transport: LocalArtifactTransport

    @property
    def run_root(self) -> Path:
        return Path(self.worker.config().roots.run_root)

    def layout(self, run_id: str) -> Any:
        return self.worker.layout(run_id)

    def status(self, run_id: str) -> dict[str, Any]:
        return self.client.status(run_id)


def build_harness(tmp_path: Path, *, worker_name: str = "test-worker") -> Harness:
    """Two roots, one machine. The worker never sees the other side."""
    worker_root = tmp_path / "worker"
    control_root = tmp_path / "control"
    trainer_root = tmp_path / "trainer"
    staging_root = control_root / "remote_staging"
    collect_root = control_root / "collected_checkpoints"
    for path in (worker_root, control_root, staging_root, collect_root, trainer_root):
        path.mkdir(parents=True, exist_ok=True)

    # A stand-in ACE-Step installation. Preflight requires `train.py` to
    # exist in the trainer root, and it is right to: a worker without
    # the trainer installed cannot train, and discovering that after a
    # transfer is the expensive way to find out.
    (trainer_root / "train.py").write_text(
        "\n".join(
            [
                '"""SYNTHETIC_TEST_FIXTURE: stands in for the ACE-Step trainer."""',
                "import runpy",
                "import sys",
                "from pathlib import Path",
                "",
                f"sys.argv[0] = {str(SYNTHETIC_TRAINER)!r}",
                f"runpy.run_path({str(SYNTHETIC_TRAINER)!r}, run_name='__main__')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    worker = RemoteWorker(worker_root)
    worker.initialise(
        worker_name=worker_name,
        trainer_root=str(trainer_root),
        repository_root=None,
    )
    client = LocalWorkerClient(worker_root, python_executable=sys.executable)
    transport = LocalArtifactTransport(Path(worker.config().roots.run_root))
    return Harness(
        root=tmp_path,
        worker_root=worker_root,
        trainer_root=trainer_root,
        staging_root=staging_root,
        collect_root=collect_root,
        worker=worker,
        client=client,
        transport=transport,
    )


def make_plan(
    run_id: str = "run_" + "a" * 16,
    *,
    requires_cuda: bool = False,
    epochs: int = 1,
) -> TrainingPlan:
    """A plan that does not require CUDA, so a laptop can run the tests.

    `requires_cuda=True` is used by the mismatch test, where the point is
    that a development worker must be refused.
    """
    return TrainingPlan(
        plan_id="plan_" + "b" * 16,
        run_id=run_id,
        experiment_id="exp_" + "c" * 16,
        base_model_id="mdl_" + "d" * 16,
        base_model_upstream_commit="6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
        dataset_ref=TrainingDatasetRef(
            dataset_id="ds_test",
            dataset_lock_sha256="0" * 64,
            curation_id="cur_test",
            curation_lock_sha256="1" * 64,
            curated_manifest_sha256="2" * 64,
            manifest_artifact_ref="curated_manifest.jsonl",
            selected_track_count=2,
            selected_hours=0.1,
        ),
        config=TrainingConfig(epochs=epochs),
        execution_backend="remote-gpu",
        requirements=HardwareRequirements(
            requires_cuda=requires_cuda,
            minimum_vram_mb=None,
            minimum_gpu_count=1 if requires_cuda else 0,
        ),
    )


def synthetic_argv(
    layout: Any,
    run_id: str,
    *,
    steps: int = 3,
    step_seconds: float = 0.05,
    checkpoint_every: int = 0,
    fail_at_step: int | None = None,
    exit_code: int | None = None,
    simulate: str | None = None,
    invalid_checkpoint: bool = False,
    ignore_sigterm: bool = False,
) -> list[str]:
    """The synthetic trainer's invocation, as argv. Never a string."""
    argv = [
        sys.executable,
        str(SYNTHETIC_TRAINER),
        "--run-id",
        run_id,
        "--checkpoint-dir",
        str(layout.checkpoints_dir),
        "--metrics-file",
        str(layout.metrics_jsonl),
        "--steps",
        str(steps),
        "--step-seconds",
        str(step_seconds),
    ]
    if checkpoint_every:
        argv += ["--checkpoint-every", str(checkpoint_every)]
    if invalid_checkpoint:
        argv.append("--invalid-checkpoint")
    if fail_at_step:
        argv += ["--fail-at-step", str(fail_at_step)]
    if exit_code is not None:
        argv += ["--exit-code", str(exit_code)]
    if simulate:
        argv += ["--simulate", simulate]
    if ignore_sigterm:
        argv.append("--ignore-sigterm")
    return argv


def build_manifest(
    staging_dir: Path, plan: TrainingPlan, *, audio_files: int = 2
) -> RemoteArtifactManifest:
    """A small staged run: a plan, an environment lock, and some audio.

    Bypasses `build_staging` on purpose. These tests are about transfer
    and execution; the gates get their own tests where a real curated
    manifest is what is being checked.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    manifest = RemoteArtifactManifest(run_id=plan.run_id, training_plan_sha256=plan.digest())

    plan_path = staging_dir / "plan.json"
    plan_path.write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest.add(
        ArtifactEntry.from_file(plan_path, role=ArtifactRole.PLAN.value, target_path="plan.json")
    )

    environment = staging_dir / "environment_lock.json"
    environment.write_text(
        json.dumps(
            {
                "python_version": ".".join(str(part) for part in sys.version_info[:3]),
                "torch_version": None,
                "ace_step_commit": None,
                "luber_commit": None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.add(
        ArtifactEntry.from_file(
            environment,
            role=ArtifactRole.ENVIRONMENT_LOCK.value,
            target_path="environment_lock.json",
        )
    )

    for index in range(audio_files):
        audio = staging_dir / "dataset" / f"track-{index:03d}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        # Distinct contents per file, so deduplication and corruption
        # tests are measuring something real.
        audio.write_bytes(b"RIFF" + bytes([index]) * 4096)
        manifest.add(
            ArtifactEntry.from_file(
                audio,
                role=ArtifactRole.DATASET_AUDIO.value,
                target_path=f"dataset/track-{index:03d}.wav",
                track_id=f"trk_{index:03d}",
            )
        )

    manifest.write(staging_dir / "artifact_manifest.json")
    return manifest


def transfer(
    harness: Harness,
    plan: TrainingPlan,
    manifest: RemoteArtifactManifest,
    *,
    staging_dir: Path | None = None,
) -> Any:
    """Move a staged run to the worker, under its run id."""
    source_dir = Path(staging_dir or harness.staging_root / plan.run_id)
    scoped = RemoteArtifactManifest(
        run_id=manifest.run_id, training_plan_sha256=manifest.training_plan_sha256
    )
    for entry in manifest.entries:
        scoped.add(
            ArtifactEntry(
                artifact_id=entry.artifact_id,
                role=entry.role,
                target_path=f"{plan.run_id}/{entry.target_path}",
                sha256=entry.sha256,
                size_bytes=entry.size_bytes,
                required=entry.required,
                source_reference=str(source_dir / entry.target_path),
                track_id=entry.track_id,
            )
        )
    return harness.transport.upload_manifest(scoped)


def prepare_and_receive(
    harness: Harness,
    plan: TrainingPlan,
    manifest: RemoteArtifactManifest,
    *,
    staging_dir: Path | None = None,
) -> dict[str, Any]:
    """The full receive path: claim, transfer, record the manifest."""
    source_dir = Path(staging_dir or harness.staging_root / plan.run_id)
    harness.client.prepare_run(plan.run_id, plan.digest(), manifest.digest())
    result = transfer(harness, plan, manifest, staging_dir=source_dir)
    assert result.ok, result.failed

    staged_manifest = source_dir / "artifact_manifest.json"
    remote_manifest = harness.run_root / plan.run_id / "artifact_manifest.json"
    remote_manifest.parent.mkdir(parents=True, exist_ok=True)
    remote_manifest.write_text(staged_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    return harness.client.receive_manifest(plan.run_id, str(remote_manifest))
