"""A bounded pilot, and everything it must refuse to become.

The step budget is the test that matters most. The installed trainer has
no `--max-steps`, so a pilot's length is arithmetic over the dataset
size, the micro batch, the accumulation factor and the epoch count — and
if that arithmetic is wrong, "tens of steps" quietly becomes hundreds on
a 2.4B model. So the budget is tested against the trainer's own formula,
including the edge cases where a naive division would under-count.

After that come the refusals: no rights, no capacity, no budget, no
resume across a different dataset. A pilot that ran when it should not
have is worse than one that did not run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from memory_fixtures import GIB, a_profile, an_identity, requested_from
from preflight_fixtures import a_plan

from luber_hardware import ComputeDevice
from luber_training.capacity_policy import CapacityQualification, qualify
from luber_training.config import TrainingConfig
from luber_training.gates import GateReport, GateResult
from luber_training.pilot import (
    ARTIFACT_CLASS,
    PILOT_MAX_OPTIMIZER_STEPS,
    PILOT_MAX_SEGMENT_STEPS,
    PILOT_MIN_TRACKS,
    GradientEvidence,
    LossPoint,
    LossSeries,
    ParameterUpdateEvidence,
    PilotBudgetError,
    PilotFailure,
    PilotFormatError,
    PilotIdentity,
    PilotOutcome,
    PilotStepBudget,
    PilotTrainingResult,
    TrainingSignal,
    classify_signal,
    outcome_for,
)
from luber_training.pilot_runner import (
    DatasetKind,
    PilotRequest,
    dataset_digest,
    identity_for,
    render_dataset_report,
    render_markdown,
    run_pilot,
    verify_pilot_dataset,
)

SYNTHETIC_MANIFEST = {"metadata": {"type": "synthetic_test_fixtures", "num_samples": 4}}


def _passing_gates() -> GateReport:
    return GateReport(
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


def _failing_gates(name: str = "rights") -> GateReport:
    return GateReport(
        results=[
            GateResult(name="dataset_lock", passed=True),
            GateResult(
                name=name,
                passed=False,
                detail="2 selected tracks are not permitted for training",
                failure_code="RIGHTS_GATE_FAILED",
            ),
        ]
    )


def _dataset(directory: Path, *, samples: int = 4, synthetic: bool = False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(samples):
        (directory / f"sample_{index:04d}.pt").write_bytes(f"sample {index}".encode())
    if synthetic:
        (directory / "manifest.json").write_text(json.dumps(SYNTHETIC_MANIFEST), encoding="utf-8")
    return directory


# ── 46. the step arithmetic ──────────────────────────────────────────


class TestStepBudget:
    @pytest.mark.parametrize(
        "samples,batch,accum,epochs,expected",
        [
            # The Phase 34 profile, observed: 2 samples, batch 1,
            # accumulation 4, one epoch produced exactly one step.
            (2, 1, 4, 1, 1),
            # A dataset smaller than one accumulation window still takes
            # a step per epoch: the loop flushes what it accumulated.
            (1, 1, 8, 3, 3),
            (3, 1, 4, 1, 1),
            (4, 1, 4, 1, 1),
            (5, 1, 4, 1, 2),
            (8, 1, 4, 2, 4),
            # drop_last=False, so a partial micro batch still counts.
            (5, 2, 1, 1, 3),
            (7, 2, 2, 1, 2),
            (10, 1, 1, 1, 10),
        ],
    )
    def test_it_reproduces_the_trainers_own_arithmetic(
        self, samples, batch, accum, epochs, expected
    ):
        budget = PilotStepBudget(
            samples=samples,
            micro_batch_size=batch,
            gradient_accumulation=accum,
            epochs=epochs,
        )
        assert budget.expected_steps == expected
        # The same formula, written the other way round.
        assert budget.expected_steps == epochs * max(
            1, math.ceil(math.ceil(samples / batch) / accum)
        )

    def test_a_partial_accumulation_window_is_a_step_not_a_rounding_error(self):
        """Five micro batches at accumulation four is two steps, not one."""
        assert (
            PilotStepBudget(
                samples=5, micro_batch_size=1, gradient_accumulation=4, epochs=1
            ).expected_steps
            == 2
        )

    def test_effective_batch_is_not_the_memory_batch(self):
        budget = PilotStepBudget(samples=4, micro_batch_size=1, gradient_accumulation=4, epochs=1)
        assert budget.micro_batch_size == 1
        assert budget.effective_batch_size == 4

    @pytest.mark.parametrize(
        "field", ["samples", "micro_batch_size", "gradient_accumulation", "epochs"]
    )
    def test_a_nonsense_dimension_is_refused(self, field):
        kwargs = {
            "samples": 4,
            "micro_batch_size": 1,
            "gradient_accumulation": 4,
            "epochs": 1,
            field: 0,
        }
        with pytest.raises(PilotBudgetError, match="at least 1"):
            PilotStepBudget(**kwargs)

    def test_a_distributed_run_is_refused_rather_than_guessed_at(self):
        with pytest.raises(PilotBudgetError, match="world_size"):
            PilotStepBudget(
                samples=4,
                micro_batch_size=1,
                gradient_accumulation=4,
                epochs=1,
                world_size=2,
            )

    # ── 47. the ceiling ──────────────────────────────────────────────

    def test_a_plan_over_the_ceiling_is_refused(self):
        budget = PilotStepBudget(samples=64, micro_batch_size=1, gradient_accumulation=1, epochs=4)
        assert budget.expected_steps > PILOT_MAX_OPTIMIZER_STEPS
        with pytest.raises(PilotBudgetError, match="the ceiling is not a parameter"):
            budget.validate()

    def test_a_plan_inside_the_ceiling_passes(self):
        PilotStepBudget(samples=8, micro_batch_size=1, gradient_accumulation=4, epochs=4).validate()

    def test_epochs_are_derived_from_the_ceiling_not_asked_for(self):
        budget = PilotStepBudget.for_ceiling(
            samples=4, micro_batch_size=1, gradient_accumulation=4, ceiling=8
        )
        assert budget.expected_steps <= 8
        assert budget.epochs == 8

    def test_a_dataset_too_large_for_one_epoch_is_refused(self):
        """The answer is a smaller dataset, not a larger bound."""
        with pytest.raises(PilotBudgetError, match="smaller dataset"):
            PilotStepBudget.for_ceiling(
                samples=1000, micro_batch_size=1, gradient_accumulation=1, ceiling=32
            )

    def test_the_ceiling_is_a_constant_not_an_argument(self):
        """There is no parameter anywhere that raises the module ceiling."""
        import inspect

        from luber_training import pilot_runner

        source = inspect.getsource(pilot_runner)
        assert "PILOT_MAX_OPTIMIZER_STEPS" in source
        # The runner compares against the constant; it never receives one.
        assert "max_optimizer_steps=" not in source
        assert "step_ceiling=PILOT_MAX_SEGMENT_STEPS" in source


# ── 51. pilot identity ───────────────────────────────────────────────


class TestPilotIdentity:
    def _identity(self, **overrides) -> PilotIdentity:
        base = {
            "plan_digest": "a" * 64,
            "dataset_manifest_digest": "b" * 64,
            "dataset_id": "pilot-1",
            "base_model_id": "model-1",
            "base_model_upstream_commit": "c" * 40,
            "ace_step_commit": "d" * 40,
            "device": ComputeDevice.MPS.value,
            "precision": "bf16",
            "optimizer": "adamw",
            "lora_rank": 32,
            "lora_alpha": 64,
            "micro_batch_size": 1,
            "gradient_accumulation": 4,
            "epochs": 8,
            "expected_steps": 8,
            "latent_length": 6000,
            "encoder_length": 256,
            "seed": 42,
        }
        base.update(overrides)
        return PilotIdentity(**base)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("lora_rank", 8),
            ("lora_alpha", 16),
            ("micro_batch_size", 4),
            ("precision", "fp32"),
            ("device", ComputeDevice.CUDA.value),
            ("latent_length", 128),
            ("encoder_length", 512),
            ("dataset_manifest_digest", "e" * 64),
            ("plan_digest", "f" * 64),
            ("seed", 7),
            ("expected_steps", 16),
        ],
    )
    def test_a_meaningful_change_changes_the_identity(self, field, value):
        assert self._identity().digest() != self._identity(**{field: value}).digest()

    def test_the_same_configuration_is_the_same_pilot(self):
        assert self._identity().digest() == self._identity().digest()
        assert self._identity().pilot_id() == self._identity().pilot_id()

    def test_the_pilot_id_carries_what_an_operator_scans_for(self):
        pilot_id = self._identity().pilot_id()
        assert "mps" in pilot_id and "bf16" in pilot_id and "r32" in pilot_id


# ── 52-53. the loss series ───────────────────────────────────────────


class TestLossSeries:
    def _series(self, losses: list[float]) -> LossSeries:
        return LossSeries(
            points=[
                LossPoint(step=index + 1, loss=value, learning_rate=1e-4, grad_norm=0.5)
                for index, value in enumerate(losses)
            ]
        )

    def test_finite_points_serialise_with_their_statistics(self):
        series = self._series([3.4, 3.2, 3.3, 3.0])
        payload = series.to_dict()
        assert len(payload["points"]) == 4
        statistics = payload["statistics"]
        assert statistics["first"] == 3.4
        assert statistics["last"] == 3.0
        assert statistics["minimum"] == 3.0
        assert statistics["maximum"] == 3.4
        assert statistics["finite_ratio"] == 1.0
        assert statistics["slope_source"] == "DERIVED"

    def test_a_round_trip_preserves_every_point(self):
        series = self._series([1.0, 2.0, 3.0])
        assert len(LossSeries.from_dict(series.to_dict()).points) == 3

    def test_a_nonfinite_loss_is_counted_not_dropped(self):
        series = self._series([3.0, float("nan"), 2.9])
        assert not series.all_finite
        assert series.finite_ratio == pytest.approx(2 / 3)
        assert series.statistics()["finite_count"] == 2

    def test_a_slope_needs_more_than_two_points(self):
        assert self._series([3.0, 2.0]).slope() is None
        assert self._series([3.0, 2.0, 1.0]).slope() is not None

    def test_a_rising_loss_is_reported_not_hidden(self):
        """Monotonic decrease is not required and not implied."""
        slope = self._series([1.0, 2.0, 3.0, 4.0]).slope()
        assert slope is not None and slope > 0

    def test_the_slope_carries_its_own_caveat(self):
        note = self._series([1.0, 2.0, 3.0]).statistics()["slope_note"].lower()
        assert "not a convergence claim" in note


# ── 53-54. classification ────────────────────────────────────────────


class TestSignalClassification:
    def _classify(self, **overrides):
        defaults = {
            "loss": LossSeries(
                points=[LossPoint(step=index + 1, loss=3.0 - index * 0.01) for index in range(8)]
            ),
            "parameters": ParameterUpdateEvidence(
                trainable_before_digest="a",
                trainable_after_digest="b",
                changed_tensor_count=384,
                trainable_tensor_count=384,
                base_model_digest_before="x",
                base_model_digest_after="x",
            ),
            "gradients": GradientEvidence(
                observed_steps=8, finite_steps=8, nonzero_steps=8, mean_grad_norm=0.4
            ),
            "expected_steps": 8,
            "completed_steps": 8,
        }
        defaults.update(overrides)
        return classify_signal(**defaults)

    def test_a_healthy_pilot_is_a_valid_signal_and_nothing_more(self):
        signal, detail = self._classify()
        assert signal == TrainingSignal.VALID_SIGNAL.value
        assert "nothing about convergence or quality" in detail

    def test_a_nonfinite_loss_is_numerically_unstable(self):
        signal, _ = self._classify(
            loss=LossSeries(
                points=[LossPoint(step=1, loss=3.0), LossPoint(step=2, loss=float("nan"))]
            )
        )
        assert signal == TrainingSignal.NUMERICALLY_UNSTABLE.value

    def test_an_infinite_loss_is_numerically_unstable(self):
        signal, _ = self._classify(loss=LossSeries(points=[LossPoint(step=1, loss=float("inf"))]))
        assert signal == TrainingSignal.NUMERICALLY_UNSTABLE.value

    def test_nonfinite_gradients_are_numerically_unstable(self):
        signal, _ = self._classify(
            gradients=GradientEvidence(observed_steps=8, finite_steps=5, nonzero_steps=5)
        )
        assert signal == TrainingSignal.NUMERICALLY_UNSTABLE.value

    def test_a_finite_loss_with_no_parameter_change_is_no_update(self):
        """The arithmetic ran and nothing was learned from it."""
        signal, detail = self._classify(
            parameters=ParameterUpdateEvidence(
                trainable_before_digest="a",
                trainable_after_digest="a",
                changed_tensor_count=0,
                trainable_tensor_count=384,
            )
        )
        assert signal == TrainingSignal.NO_UPDATE.value
        assert "nothing was learned" in detail

    def test_zero_gradients_everywhere_is_no_update(self):
        signal, _ = self._classify(
            gradients=GradientEvidence(observed_steps=8, finite_steps=8, nonzero_steps=0)
        )
        assert signal == TrainingSignal.NO_UPDATE.value

    def test_a_changed_base_model_is_a_failure_not_a_success(self):
        signal, detail = self._classify(
            parameters=ParameterUpdateEvidence(
                trainable_before_digest="a",
                trainable_after_digest="b",
                changed_tensor_count=384,
                base_model_digest_before="x",
                base_model_digest_after="y",
            )
        )
        assert signal == TrainingSignal.NUMERICALLY_UNSTABLE.value
        assert "must never do" in detail

    def test_too_few_steps_is_insufficient_evidence_not_a_verdict(self):
        signal, _ = self._classify(
            loss=LossSeries(points=[LossPoint(step=1, loss=3.0)]), completed_steps=1
        )
        assert signal == TrainingSignal.INSUFFICIENT_EVIDENCE.value

    def test_no_loss_at_all_is_insufficient_evidence(self):
        signal, _ = self._classify(loss=LossSeries(points=[]), completed_steps=0)
        assert signal == TrainingSignal.INSUFFICIENT_EVIDENCE.value

    def test_unestablished_parameter_evidence_is_insufficient(self):
        signal, _ = self._classify(parameters=ParameterUpdateEvidence())
        assert signal == TrainingSignal.INSUFFICIENT_EVIDENCE.value

    def test_there_is_no_vocabulary_for_convergence_or_quality(self):
        values = {item.value for item in TrainingSignal}
        assert values == {
            "VALID_SIGNAL",
            "NUMERICALLY_UNSTABLE",
            "NO_UPDATE",
            "INSUFFICIENT_EVIDENCE",
        }

    @pytest.mark.parametrize(
        "signal,expected",
        [
            (TrainingSignal.VALID_SIGNAL.value, PilotOutcome.COMPLETED_VALID_SIGNAL.value),
            (TrainingSignal.NUMERICALLY_UNSTABLE.value, PilotOutcome.FAILED_NUMERIC.value),
            (TrainingSignal.NO_UPDATE.value, PilotOutcome.FAILED_NUMERIC.value),
            (
                TrainingSignal.INSUFFICIENT_EVIDENCE.value,
                PilotOutcome.COMPLETED_INSUFFICIENT_SIGNAL.value,
            ),
        ],
    )
    def test_one_mapping_from_signal_to_outcome(self, signal, expected):
        assert outcome_for(signal, completed=True) == expected


# ── 48-50. the refusals ──────────────────────────────────────────────


class TestPilotRefusals:
    def _request(self, tmp_path: Path, **overrides) -> PilotRequest:
        trainer = tmp_path / "trainer"
        (trainer / "checkpoints" / "acestep-v15-turbo").mkdir(parents=True, exist_ok=True)
        (trainer / "train.py").write_text("", encoding="utf-8")
        interpreter = tmp_path / "python"
        interpreter.write_text("", encoding="utf-8")
        dataset = _dataset(trainer / "data", samples=4)
        base = {
            "plan": a_plan(
                device=ComputeDevice.MPS.value,
                config=TrainingConfig(epochs=1, rank=32, alpha=64, precision="bf16"),
            ),
            "dataset_dir": dataset,
            "trainer_root": trainer,
            "python_executable": interpreter,
            "model_dir": trainer / "checkpoints",
            "workspace": trainer / "workspace",
            "latent_length": 6000,
            "encoder_length": 256,
            "gate_report": _passing_gates(),
            "capacity": self._qualified(),
        }
        base.update(overrides)
        return PilotRequest(**base)

    def _qualified(self):
        return qualify(
            device=ComputeDevice.MPS.value,
            requested=requested_from(an_identity()),
            profiles=[a_profile(peak_bytes=6 * GIB)],
            host_total_bytes=24 * GIB,
            device_total_bytes=24 * GIB,
            runs_control_plane=False,
        )

    def test_without_rights_a_pilot_cannot_start(self, tmp_path: Path):
        result = run_pilot(self._request(tmp_path, gate_report=_failing_gates()))
        assert result.outcome == PilotOutcome.BLOCKED.value
        assert result.failure in (
            PilotFailure.NO_RIGHTS_CLEARED_DATA.value,
            PilotFailure.DATASET_INVALID.value,
        )

    def test_without_any_gate_report_a_pilot_cannot_start(self, tmp_path: Path):
        result = run_pilot(self._request(tmp_path, gate_report=None))
        assert result.outcome == PilotOutcome.BLOCKED.value
        assert "unestablished" in result.failure_detail

    def test_evaluation_only_material_cannot_enter_a_pilot(self, tmp_path: Path):
        """The leakage gate is a pilot gate, not just a full-run gate."""
        leaked = GateReport(
            results=[
                GateResult(name="dataset_lock", passed=True),
                GateResult(name="rights", passed=True),
                GateResult(
                    name="evaluation_leakage",
                    passed=False,
                    detail="1 selected track belongs to a non-training split",
                    failure_code="EVALUATION_LEAKAGE",
                ),
            ]
        )
        result = run_pilot(self._request(tmp_path, gate_report=leaked))
        assert result.outcome == PilotOutcome.BLOCKED.value
        assert "evaluation_leakage" in result.failure_detail

    def test_a_synthetic_fixture_is_refused_unless_asked_for_by_name(self, tmp_path: Path):
        trainer = tmp_path / "trainer"
        (trainer / "checkpoints" / "acestep-v15-turbo").mkdir(parents=True, exist_ok=True)
        (trainer / "train.py").write_text("", encoding="utf-8")
        dataset = _dataset(trainer / "synthetic", samples=4, synthetic=True)
        result = run_pilot(self._request(tmp_path, dataset_dir=dataset))
        assert result.outcome == PilotOutcome.BLOCKED.value
        assert result.dataset_kind == DatasetKind.SYNTHETIC_FIXTURE
        assert "never be evidence about real data" in result.failure_detail

    def test_too_few_tracks_is_refused(self, tmp_path: Path):
        trainer = tmp_path / "trainer"
        (trainer / "checkpoints" / "acestep-v15-turbo").mkdir(parents=True, exist_ok=True)
        (trainer / "train.py").write_text("", encoding="utf-8")
        dataset = _dataset(trainer / "small", samples=PILOT_MIN_TRACKS - 1)
        result = run_pilot(self._request(tmp_path, dataset_dir=dataset))
        assert result.outcome == PilotOutcome.BLOCKED.value
        assert "at least" in result.failure_detail

    def test_without_a_qualified_capacity_a_pilot_cannot_start(self, tmp_path: Path):
        result = run_pilot(self._request(tmp_path, capacity=None))
        assert result.outcome == PilotOutcome.BLOCKED.value
        assert result.failure == PilotFailure.CAPACITY_NOT_QUALIFIED.value

    def test_an_unverified_capacity_is_not_a_qualification(self, tmp_path: Path):
        unverified = qualify(
            device=ComputeDevice.MPS.value,
            requested=requested_from(an_identity()),
            profiles=[],
            host_total_bytes=24 * GIB,
        )
        assert unverified.qualification == CapacityQualification.UNVERIFIED.value
        result = run_pilot(self._request(tmp_path, capacity=unverified))
        assert result.failure == PilotFailure.CAPACITY_NOT_QUALIFIED.value

    def test_an_insufficient_capacity_blocks(self, tmp_path: Path):
        insufficient = qualify(
            device=ComputeDevice.MPS.value,
            requested=requested_from(an_identity()),
            profiles=[a_profile(peak_bytes=23 * GIB)],
            host_total_bytes=24 * GIB,
            device_total_bytes=24 * GIB,
            runs_control_plane=False,
        )
        assert insufficient.qualification == CapacityQualification.INSUFFICIENT.value
        result = run_pilot(self._request(tmp_path, capacity=insufficient))
        assert result.failure == PilotFailure.CAPACITY_NOT_QUALIFIED.value

    def test_a_blocked_preflight_blocks_the_pilot(self, tmp_path: Path):
        result = run_pilot(self._request(tmp_path, preflight_status="BLOCKED"))
        assert result.failure == PilotFailure.PREFLIGHT_BLOCKED.value

    def test_an_unverified_preflight_blocks_the_pilot(self, tmp_path: Path):
        result = run_pilot(self._request(tmp_path, preflight_status="UNVERIFIED"))
        assert result.failure == PilotFailure.PREFLIGHT_BLOCKED.value

    def test_a_dataset_outside_the_trainer_root_is_refused(self, tmp_path: Path):
        outside = _dataset(tmp_path / "elsewhere", samples=4)
        result = run_pilot(self._request(tmp_path, dataset_dir=outside))
        assert result.outcome == PilotOutcome.BLOCKED.value
        assert "outside the trainer's working directory" in result.failure_detail

    def test_a_missing_trainer_blocks(self, tmp_path: Path):
        result = run_pilot(self._request(tmp_path, trainer_root=tmp_path / "absent"))
        assert result.outcome == PilotOutcome.BLOCKED.value
        assert result.failure == PilotFailure.TRAINER_FAILED.value

    def test_a_blocked_pilot_still_records_what_would_have_run(self, tmp_path: Path):
        result = run_pilot(self._request(tmp_path, capacity=None))
        assert result.identity.expected_steps > 0
        assert result.expected_steps > 0
        assert result.pilot_id


# ── 45, 57. the dataset is frozen ────────────────────────────────────


class TestDatasetImmutability:
    def test_the_digest_covers_content_not_just_names(self, tmp_path: Path):
        first = _dataset(tmp_path / "a", samples=3)
        before = dataset_digest(first)
        (first / "sample_0000.pt").write_bytes(b"changed")
        assert dataset_digest(first) != before

    def test_the_digest_is_stable_across_reads(self, tmp_path: Path):
        directory = _dataset(tmp_path / "a", samples=3)
        assert dataset_digest(directory) == dataset_digest(directory)

    def test_changing_the_data_changes_the_pilot_identity(self, tmp_path: Path):
        trainer = tmp_path / "trainer"
        trainer.mkdir()
        directory = _dataset(trainer / "data", samples=3)
        request = PilotRequest(
            plan=a_plan(device=ComputeDevice.MPS.value),
            dataset_dir=directory,
            trainer_root=trainer,
            python_executable=tmp_path / "python",
            model_dir=trainer / "checkpoints",
            workspace=trainer / "workspace",
            latent_length=6000,
            encoder_length=256,
        )
        budget = PilotStepBudget(samples=3, micro_batch_size=1, gradient_accumulation=4, epochs=1)
        before = identity_for(request, budget, dataset_digest(directory)).digest()
        (directory / "sample_0000.pt").write_bytes(b"different")
        after = identity_for(request, budget, dataset_digest(directory)).digest()
        assert before != after


# ── 24, 40. what a pilot's artifacts are ─────────────────────────────


class TestArtifactClassification:
    def test_every_pilot_result_is_marked_non_production(self):
        result = PilotTrainingResult(
            pilot_id="pilot-1",
            identity=PilotIdentity(
                plan_digest="a" * 64,
                dataset_manifest_digest="b" * 64,
                dataset_id="ds",
                base_model_id="m",
                base_model_upstream_commit="c" * 40,
                ace_step_commit="d" * 40,
                device=ComputeDevice.MPS.value,
                precision="bf16",
                optimizer="adamw",
                lora_rank=32,
                lora_alpha=64,
                micro_batch_size=1,
                gradient_accumulation=4,
                epochs=8,
                expected_steps=8,
                latent_length=6000,
                encoder_length=256,
                seed=42,
            ),
        )
        payload = result.to_dict()
        assert payload["artifact_class"] == list(ARTIFACT_CLASS)
        assert "NEVER_AUTO_PROMOTE" in payload["artifact_class"]
        assert "makes no claim about convergence" in payload["note"]

    def test_a_record_from_another_schema_is_refused(self):
        with pytest.raises(PilotFormatError):
            PilotTrainingResult.from_dict({"schema_version": "something/9"})

    def test_a_record_without_an_identity_is_refused(self):
        with pytest.raises(PilotFormatError):
            PilotTrainingResult.from_dict(
                {"schema_version": "luber-training-pilot/1", "pilot_id": "x"}
            )


# ── the reports say nothing they should not ──────────────────────────


class TestReports:
    def _result(self) -> PilotTrainingResult:
        identity = PilotIdentity(
            plan_digest="a" * 64,
            dataset_manifest_digest="b" * 64,
            dataset_id="ds",
            base_model_id="m",
            base_model_upstream_commit="c" * 40,
            ace_step_commit="d" * 40,
            device=ComputeDevice.MPS.value,
            precision="bf16",
            optimizer="adamw",
            lora_rank=32,
            lora_alpha=64,
            micro_batch_size=1,
            gradient_accumulation=4,
            epochs=8,
            expected_steps=8,
            latent_length=6000,
            encoder_length=256,
            seed=42,
        )
        return PilotTrainingResult(
            pilot_id=identity.pilot_id(),
            identity=identity,
            outcome=PilotOutcome.COMPLETED_VALID_SIGNAL.value,
            signal=TrainingSignal.VALID_SIGNAL.value,
            completed_steps=8,
            expected_steps=8,
            loss=LossSeries(
                points=[LossPoint(step=index + 1, loss=3.0 - index * 0.01) for index in range(8)]
            ),
        )

    def test_the_loss_report_refuses_to_claim_convergence(self):
        rendered = render_markdown(self._result()).lower()
        assert "convergence" in rendered
        assert "it is not a convergence result and not a quality result" in rendered
        assert "converged" not in rendered

    def test_the_loss_report_labels_the_slope_as_derived(self):
        assert "slope (DERIVED)" in render_markdown(self._result())

    def test_the_dataset_report_carries_no_track_names(self, tmp_path: Path):
        directory = _dataset(tmp_path / "data", samples=3)
        verdict = verify_pilot_dataset(directory, gate_report=_passing_gates())
        report = render_dataset_report(verdict, self._result().identity)
        assert "sample_0000.pt" not in report
        assert "deliberately absent" in report
        assert str(verdict.sample_count) in report


def test_the_pilot_ceiling_is_tens_of_steps_not_hundreds():
    """The bound the whole phase rests on, asserted rather than assumed."""
    assert 16 <= PILOT_MAX_OPTIMIZER_STEPS <= 64
    assert PILOT_MAX_SEGMENT_STEPS <= PILOT_MAX_OPTIMIZER_STEPS
    assert PILOT_MAX_SEGMENT_STEPS * 2 <= PILOT_MAX_OPTIMIZER_STEPS
