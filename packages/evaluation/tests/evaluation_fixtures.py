"""Fixtures: a Phase 25 registry, and synthetic model profiles.

Everything here is built in a temporary directory, and nothing reads
the frozen benchmark unless a test explicitly asks for it. A test suite
that needed `benchmarks/` present would fail on a fresh clone for a
reason unrelated to what it was testing.

The synthetic profiles are numbers a test dictates, not a simulation of
music. They exist so the qualification machinery can be exercised
against known inputs — "this candidate clips 4% of its samples" is a
statement a test makes, and the point is checking that the gates react
to it correctly.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from luber_evaluation.runner import SyntheticProfile
from luber_training.config import TrainingConfig
from luber_training.entities import (
    Checkpoint,
    CheckpointKind,
    CheckpointStatus,
    EvaluationCandidate,
    Experiment,
    ModelBaseline,
    TrainingDatasetRef,
    TrainingRun,
)
from luber_training.orchestrator import Orchestrator
from luber_training.registry import Registry

#: Baseline behaviour every scenario is measured against. Reliable,
#: technically clean, unremarkable.
BASELINE_METRICS: dict[str, float] = {
    "clipping_sample_ratio": 0.001,
    "silence_ratio": 0.02,
    "peak_dbfs": -1.2,
    "true_peak_dbtp": -1.0,
    "integrated_lufs": -14.2,
    "crest_factor_db": 11.0,
    "spectral_centroid_hz": 2400.0,
    "high_frequency_energy_ratio": 0.16,
    "stereo_width": 0.55,
    "phase_correlation": 0.42,
    "sample_rate": 44100.0,
    "channels": 2.0,
    "duration_absolute_error_seconds": 0.4,
    "duration_relative_error": 0.007,
}


def profile(label: str, **overrides: Any) -> SyntheticProfile:
    """A synthetic model: the baseline, with named differences."""
    metrics = dict(BASELINE_METRICS)
    failing = tuple(overrides.pop("failing_cases", ()))
    failure_rate = float(overrides.pop("failure_rate", 0.0))
    metrics.update({key: float(value) for key, value in overrides.items()})
    return SyntheticProfile(
        label=label,
        metrics=metrics,
        failure_rate=failure_rate,
        failing_cases=failing,
    )


def write_profile(path: Path, model: SyntheticProfile) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    return path


@pytest.fixture
def registry_root(tmp_path: Path) -> Path:
    return tmp_path / "training-registry"


@pytest.fixture
def orchestrator(registry_root: Path) -> Orchestrator:
    return Orchestrator(Registry(registry_root))


def seed_registry(
    orchestrator: Orchestrator,
    *,
    checkpoint_kind: str = CheckpointKind.ADAPTER.value,
    checkpoint_status: str = CheckpointStatus.READY.value,
    hypothesis: str = "improve technical reliability on Korean vocal cases",
    checkpoints: int = 1,
    final_losses: tuple[float, ...] = (0.31,),
) -> dict[str, Any]:
    """A registry holding one trained run and its candidates.

    Written directly rather than through the training CLI: the gates
    that CLI enforces need a Phase 23 dataset build and a Phase 24
    curation lock on disk, and none of that is what these tests are
    about. What matters here is that the records have the shape Phase 25
    writes, which is why they are built from its own entity classes.
    """
    registry = orchestrator.registry

    baseline = ModelBaseline(
        model_id="mdl_" + "a" * 16,
        provider="ace-step",
        model_family="ace-step",
        model_name="acestep-v15-turbo",
        model_version="1.5",
        upstream_commit="6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
        architecture="dit",
        training_strategy_support=["LORA", "LOKR"],
    )
    registry.write("models", baseline.model_id, baseline.to_dict(), overwrite=True)

    experiment = Experiment(
        experiment_id="exp_" + "b" * 16,
        name="korean-vocal-reliability",
        hypothesis=hypothesis,
        base_model_id=baseline.model_id,
    )
    registry.write("experiments", experiment.experiment_id, experiment.to_dict(), overwrite=True)

    run = TrainingRun(
        run_id="run_" + "c" * 16,
        experiment_id=experiment.experiment_id,
        base_model_id=baseline.model_id,
        dataset_ref=TrainingDatasetRef(
            dataset_id="ds_test",
            dataset_lock_sha256="0" * 64,
            curation_id="cur_test",
            curation_lock_sha256="1" * 64,
            curated_manifest_sha256="2" * 64,
            manifest_artifact_ref="curated_manifest.jsonl",
            selected_track_count=120,
            selected_hours=6.5,
        ),
        config=TrainingConfig(),
        execution_backend="dry-run",
        status="COMPLETED",
        training_plan_sha256="3" * 64,
    )
    registry.write("runs", run.run_id, run.to_dict(), overwrite=True)

    made: list[dict[str, str]] = []
    for index in range(checkpoints):
        checkpoint = Checkpoint(
            checkpoint_id=f"ckpt_{chr(ord('d') + index) * 16}",
            run_id=run.run_id,
            kind=checkpoint_kind,
            step=1000 * (index + 1),
            epoch=5 * (index + 1),
            status=checkpoint_status,
            sha256=f"{index}" * 64,
            metrics_snapshot={
                "loss": final_losses[index] if index < len(final_losses) else 0.4,
            },
        )
        registry.write(
            "checkpoints", checkpoint.checkpoint_id, checkpoint.to_dict(), overwrite=True
        )

        candidate = EvaluationCandidate(
            candidate_id=f"cand_{chr(ord('h') + index) * 16}",
            run_id=run.run_id,
            checkpoint_id=checkpoint.checkpoint_id,
            experiment_id=experiment.experiment_id,
        )
        registry.write("candidates", candidate.candidate_id, candidate.to_dict(), overwrite=True)
        made.append(
            {"checkpoint_id": checkpoint.checkpoint_id, "candidate_id": candidate.candidate_id}
        )

    return {
        "model_id": baseline.model_id,
        "experiment_id": experiment.experiment_id,
        "run_id": run.run_id,
        "candidates": made,
        "candidate_id": made[0]["candidate_id"] if made else "",
        "checkpoint_id": made[0]["checkpoint_id"] if made else "",
    }


@pytest.fixture
def seeded(orchestrator: Orchestrator) -> dict[str, Any]:
    return seed_registry(orchestrator)


@pytest.fixture
def repository_root() -> Iterator[Path]:
    """The real repository, for tests that read the frozen benchmark."""
    yield Path(__file__).resolve().parents[3]
