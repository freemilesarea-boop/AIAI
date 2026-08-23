"""What stops an experiment, and in what order.

The order is the interesting part. If a run is blocked, the reason has
to be the *first* thing that was wrong with it — a contaminated split
reported as "the trainer crashed" sends somebody looking in the wrong
place, and a rights failure reported as anything else is worse than
that.
"""

import json
from pathlib import Path

import pytest
from preflight_fixtures import a_plan

from luber_dataset.splits import build_experiment_splits
from luber_training.capacity_policy import CapacityDecision
from luber_training.checkpoint_provenance import REQUIRED_FIELDS
from luber_training.config import TrainingConfig
from luber_training.experiment import ExperimentError, ExperimentFailure, ExperimentOutcome
from luber_training.experiment_runner import (
    ExperimentRequest,
    compose_provenance,
    run_experiment,
    split_digests,
)
from luber_training.gates import GateReport, GateResult
from luber_training.tensors import report_from_document


def _splits() -> dict:
    tracks, index = [], 0
    for group in ("A", "B"):
        for _ in range(64):
            index += 1
            tracks.append(
                {
                    "track_id": f"track-{index:03d}",
                    "audio_sha256": f"{index:064x}",
                    "source_group": group,
                    "duration_seconds": 120.0,
                    "training_allowed": True,
                }
            )
    return build_experiment_splits(
        {"dataset_id": "LIB", "content_hash": "c" * 64, "tracks": tracks},
        train_size=24,
        validation_size=4,
        evaluation_size=4,
        seed=36,
    ).to_dict()


def _tensors(count: int, **overrides):
    samples = []
    for index in range(count):
        sample = {
            "name": f"{index}.pt",
            "ok": True,
            "readable": True,
            "latent_length": 6000,
            "latent_channels": 64,
            "encoder_length": 769,
            "missing_fields": [],
            "non_finite_fields": [],
        }
        sample.update(overrides if index == 0 else {})
        samples.append(sample)
    return report_from_document({"dataset_dir": "/tensors", "samples": samples})


def _request(tmp_path: Path, **overrides) -> ExperimentRequest:
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    for directory, count in ((train_dir, 24), (validation_dir, 4)):
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (directory / f"{index}.pt").write_bytes(b"tensor")

    base = {
        "plan": a_plan(
            device="MPS", config=TrainingConfig(epochs=1, rank=16, alpha=32, precision="bf16")
        ),
        "train_dir": train_dir,
        "validation_dir": validation_dir,
        "trainer_root": tmp_path / "trainer",
        "python_executable": tmp_path / "python",
        "model_dir": tmp_path / "models",
        "workspace": tmp_path / "workspace",
        "splits": _splits(),
        "code_commit": "0" * 40,
        "gate_report": GateReport(results=(GateResult(name="rights", passed=True),)),
        "capacity": CapacityDecision(qualification="QUALIFIED", device="MPS"),
        "preflight_status": "READY",
        "tensor_report": _tensors(24),
        "validation_report": _tensors(4),
        "step_budget": 120,
    }
    base.update(overrides)
    return ExperimentRequest(**base)


class TestTheGateOrder:
    def test_a_failed_rights_gate_blocks_before_anything_else(self, tmp_path):
        """Even with a contaminated split, rights is the reason reported."""
        splits = _splits()
        splits["evaluation"]["tracks"].append(splits["train"]["tracks"][0])
        result = run_experiment(
            _request(
                tmp_path,
                gate_report=GateReport(
                    results=(GateResult(name="rights", passed=False, detail="not permitted"),)
                ),
                splits=splits,
            )
        )
        assert result.outcome == ExperimentOutcome.BLOCKED.value
        assert result.failure == ExperimentFailure.RIGHTS_GATE_FAILED.value

    def test_no_gate_report_is_a_refusal_not_a_default_pass(self, tmp_path):
        result = run_experiment(_request(tmp_path, gate_report=None))
        assert result.failure == ExperimentFailure.RIGHTS_GATE_FAILED.value
        assert "unestablished" in result.failure_detail

    def test_a_contaminated_split_blocks_before_capacity(self, tmp_path):
        splits = _splits()
        splits["evaluation"]["tracks"].append(splits["train"]["tracks"][0])
        result = run_experiment(
            _request(
                tmp_path,
                splits=splits,
                capacity=CapacityDecision(qualification="UNVERIFIED", device="MPS"),
            )
        )
        assert result.failure == ExperimentFailure.SPLIT_LEAKAGE.value

    def test_unusable_tensors_block_before_capacity(self, tmp_path):
        result = run_experiment(
            _request(
                tmp_path,
                tensor_report=_tensors(24, ok=False, non_finite_fields=["target_latents"]),
                capacity=CapacityDecision(qualification="UNVERIFIED", device="MPS"),
            )
        )
        assert result.failure == ExperimentFailure.DATASET_UNUSABLE.value
        assert "NON_FINITE" in result.failure_detail

    def test_unqualified_capacity_blocks_before_preflight(self, tmp_path):
        result = run_experiment(
            _request(
                tmp_path,
                capacity=CapacityDecision(qualification="UNVERIFIED", device="MPS"),
                preflight_status="BLOCKED",
            )
        )
        assert result.failure == ExperimentFailure.CAPACITY_NOT_QUALIFIED.value

    def test_a_preflight_that_is_not_ready_blocks(self, tmp_path):
        result = run_experiment(_request(tmp_path, preflight_status="UNVERIFIED"))
        assert result.failure == ExperimentFailure.PREFLIGHT_NOT_READY.value

    def test_a_blocked_run_reports_no_signal_and_no_steps(self, tmp_path):
        result = run_experiment(_request(tmp_path, preflight_status="BLOCKED"))
        assert result.completed_steps == 0
        assert result.training_signal == "INSUFFICIENT_EVIDENCE"
        assert result.generalization_signal == "INSUFFICIENT_EVIDENCE"
        assert result.listening_evaluation_required is False


class TestTheProvenanceItComposes:
    def test_every_required_field_is_filled_in(self, tmp_path):
        provenance = compose_provenance(_request(tmp_path)).to_dict()
        for field in REQUIRED_FIELDS:
            assert str(provenance[field]).strip(), field

    def test_it_carries_all_three_split_digests(self, tmp_path):
        request = _request(tmp_path)
        provenance = compose_provenance(request)
        train, validation, evaluation = split_digests(request.splits)
        assert provenance.train_split_digest == train
        assert provenance.validation_split_digest == validation
        assert provenance.evaluation_split_digest == evaluation
        assert len({train, validation, evaluation}) == 3

    def test_the_artifact_class_is_never_promotable(self, tmp_path):
        provenance = compose_provenance(_request(tmp_path))
        assert provenance.artifact_class == (
            "EXPERIMENTAL",
            "NON_PRODUCTION",
            "NEVER_AUTO_PROMOTE",
        )

    def test_an_incomplete_provenance_blocks_the_run_before_it_starts(self, tmp_path):
        """The template is validated before launch, not after."""
        request = _request(tmp_path)
        request.code_commit = ""
        result = run_experiment(request)
        assert result.failure == ExperimentFailure.PROVENANCE_INCOMPLETE.value
        assert "code_commit" in result.failure_detail


class TestTheSplitSummary:
    def test_it_carries_counts_and_digests_and_no_track_names(self, tmp_path):
        result = run_experiment(_request(tmp_path, preflight_status="BLOCKED"))
        rendered = json.dumps(result.splits)
        assert result.splits["train"]["track_count"] == 24
        assert result.splits["evaluation"]["track_count"] == 4
        assert "track-0" not in rendered
        assert "audio_sha256" not in rendered


class TestTheBudget:
    def test_the_step_budget_is_split_across_two_segments(self, tmp_path):
        result = run_experiment(_request(tmp_path, preflight_status="BLOCKED"))
        # 24 samples, batch 1, accumulation 4 -> 6 steps an epoch.
        assert result.identity.expected_steps == 60
        assert result.identity.epochs == 10

    @pytest.mark.parametrize("asked", [1_000, 10_000, 100_000])
    def test_no_requested_budget_raises_the_module_ceiling(self, tmp_path, asked):
        """The ceiling is the module's, whatever a caller asks for."""
        from luber_training.experiment import EXPERIMENT_MAX_OPTIMIZER_STEPS

        result = run_experiment(_request(tmp_path, step_budget=asked, preflight_status="BLOCKED"))
        # Two segments, so one segment may take at most half the ceiling.
        assert result.identity.expected_steps <= EXPERIMENT_MAX_OPTIMIZER_STEPS // 2
        assert result.step_ceiling == EXPERIMENT_MAX_OPTIMIZER_STEPS

    def test_the_phase_37_ceiling_is_six_hundred_steps(self):
        """Named by the phase brief, and no flag lifts it."""
        from luber_training.experiment import EXPERIMENT_MAX_OPTIMIZER_STEPS, StepBudget

        assert EXPERIMENT_MAX_OPTIMIZER_STEPS == 600
        with pytest.raises(ExperimentError, match="exceeds the experiment ceiling"):
            StepBudget(samples=128, micro_batch_size=1, gradient_accumulation=4, epochs=20)
