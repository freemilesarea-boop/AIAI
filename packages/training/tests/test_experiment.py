"""The controlled experiment's budget, its verdicts, and their limits.

The tests that matter here are the refusals. Anyone can write a
classifier that says VALID_SIGNAL when the numbers look good; the work
is in making sure it says INSUFFICIENT_EVIDENCE when they merely look
good, and never says anything about quality at all.
"""

import pytest

from luber_training.experiment import (
    ARTIFACT_CLASS,
    EXPERIMENT_MAX_EPOCHS,
    EXPERIMENT_MAX_OPTIMIZER_STEPS,
    GENERALIZATION_IMPROVEMENT_THRESHOLD,
    MINIMUM_VALIDATION_POINTS,
    MINIMUM_VALIDATION_TRACKS,
    ExperimentError,
    GeneralizationSignal,
    GradientEvidence,
    LossPoint,
    LossSeries,
    ParameterUpdateEvidence,
    StepBudget,
    TrainingSignal,
    ValidationPoint,
    classify_generalization,
    classify_training_signal,
)


def _series(values, *, name="train") -> LossSeries:
    return LossSeries(
        name=name,
        points=tuple(
            LossPoint(step=index + 1, loss=value, epoch=index + 1)
            for index, value in enumerate(values)
        ),
    )


def _validation(values, *, tracks=4) -> tuple[ValidationPoint, ...]:
    return tuple(
        ValidationPoint(
            epoch=index + 1, step=index + 1, loss=value, sample_count=tracks, finite_count=tracks
        )
        for index, value in enumerate(values)
    )


class TestTheStepBudget:
    def test_it_computes_the_trainers_own_arithmetic(self):
        budget = StepBudget(samples=24, micro_batch_size=1, gradient_accumulation=4, epochs=10)
        assert budget.micro_batches_per_epoch == 24
        assert budget.steps_per_epoch == 6
        assert budget.expected_steps == 60
        assert budget.effective_batch_size == 4

    def test_a_partial_final_batch_still_counts(self):
        """`drop_last=False`, so 23 samples is 23 micro-batches."""
        budget = StepBudget(samples=23, micro_batch_size=1, gradient_accumulation=4, epochs=1)
        assert budget.micro_batches_per_epoch == 23
        assert budget.steps_per_epoch == 6

    def test_it_refuses_a_run_past_the_step_ceiling(self):
        with pytest.raises(ExperimentError, match="exceeds the experiment ceiling"):
            StepBudget(samples=24, micro_batch_size=1, gradient_accumulation=4, epochs=41)

    def test_it_refuses_a_run_past_the_epoch_ceiling(self):
        with pytest.raises(ExperimentError, match=str(EXPERIMENT_MAX_EPOCHS)):
            StepBudget(samples=1, micro_batch_size=1, gradient_accumulation=1, epochs=100)

    def test_for_ceiling_never_exceeds_what_it_was_given(self):
        budget = StepBudget.for_ceiling(
            samples=24, micro_batch_size=1, gradient_accumulation=4, ceiling=60
        )
        assert budget.expected_steps <= 60

    def test_for_ceiling_cannot_be_argued_past_the_module_ceiling(self):
        """No caller-supplied number raises the hard cap."""
        budget = StepBudget.for_ceiling(
            samples=24, micro_batch_size=1, gradient_accumulation=4, ceiling=10_000
        )
        assert budget.expected_steps <= EXPERIMENT_MAX_OPTIMIZER_STEPS

    def test_it_always_leaves_at_least_one_step(self):
        budget = StepBudget.for_ceiling(
            samples=200, micro_batch_size=1, gradient_accumulation=1, ceiling=1
        )
        assert budget.epochs == 1


class TestTheTrainingSignal:
    def _ok_parameters(self, **overrides) -> ParameterUpdateEvidence:
        base = {
            "changed_tensor_count": 384,
            "comparable_tensor_count": 384,
            "base_model_digest_before": "a" * 64,
            "base_model_digest_after": "a" * 64,
        }
        base.update(overrides)
        return ParameterUpdateEvidence(**base)

    def _ok_gradients(self, **overrides) -> GradientEvidence:
        base = {"observed_steps": 10, "finite_steps": 10, "nonzero_steps": 10}
        base.update(overrides)
        return GradientEvidence(**base)

    def test_a_healthy_run_is_a_valid_signal(self):
        signal, detail = classify_training_signal(
            loss=_series([1.4, 1.3, 1.35, 1.2]),
            gradients=self._ok_gradients(),
            parameters=self._ok_parameters(),
            expected_steps=4,
        )
        assert signal == TrainingSignal.VALID_SIGNAL.value
        assert "nothing about convergence or quality" in detail

    def test_a_non_finite_gradient_is_instability_not_success(self):
        signal, _ = classify_training_signal(
            loss=_series([1.4, 1.3, 1.2]),
            gradients=self._ok_gradients(finite_steps=9),
            parameters=self._ok_parameters(),
            expected_steps=3,
        )
        assert signal == TrainingSignal.NUMERICALLY_UNSTABLE.value

    def test_all_zero_gradients_are_no_update(self):
        signal, _ = classify_training_signal(
            loss=_series([1.4, 1.3, 1.2]),
            gradients=self._ok_gradients(nonzero_steps=0),
            parameters=self._ok_parameters(),
            expected_steps=3,
        )
        assert signal == TrainingSignal.NO_UPDATE.value

    def test_an_unmakeable_parameter_comparison_is_never_no_update(self):
        """The Phase 35 trap: unknown must not be reported as zero."""
        signal, detail = classify_training_signal(
            loss=_series([1.4, 1.3, 1.2]),
            gradients=self._ok_gradients(),
            parameters=self._ok_parameters(changed_tensor_count=None, detail="no comparable name"),
            expected_steps=3,
        )
        assert signal == TrainingSignal.INSUFFICIENT_EVIDENCE.value
        assert signal != TrainingSignal.NO_UPDATE.value
        assert "could not be established" in detail

    def test_a_moved_base_model_is_never_a_valid_signal(self):
        signal, detail = classify_training_signal(
            loss=_series([1.4, 1.3, 1.2]),
            gradients=self._ok_gradients(),
            parameters=self._ok_parameters(base_model_digest_after="b" * 64),
            expected_steps=3,
        )
        assert signal != TrainingSignal.VALID_SIGNAL.value
        assert "base model" in detail

    def test_nothing_measured_is_insufficient_evidence(self):
        signal, _ = classify_training_signal(
            loss=LossSeries(name="train"),
            gradients=GradientEvidence(),
            parameters=ParameterUpdateEvidence(),
            expected_steps=0,
        )
        assert signal == TrainingSignal.INSUFFICIENT_EVIDENCE.value


class TestTheGeneralizationSignal:
    def test_a_clear_fall_in_held_out_loss_is_reported_as_such(self):
        signal, detail = classify_generalization(
            validation=_validation([1.50, 1.42, 1.35, 1.30, 1.24]),
            training=_series([1.5, 1.4, 1.3, 1.2]),
            validation_track_count=4,
        )
        assert signal == GeneralizationSignal.HELD_OUT_LOSS_IMPROVED.value
        assert "not a quality claim" in detail
        assert "convergence" in detail

    def test_movement_inside_the_noise_band_is_not_a_result(self):
        signal, detail = classify_generalization(
            validation=_validation([1.500, 1.498, 1.502, 1.497, 1.499]),
            training=_series([1.5, 1.4, 1.3]),
            validation_track_count=4,
        )
        assert signal == GeneralizationSignal.NO_MEASURABLE_CHANGE.value
        assert "not evidence of no learning" in detail.lower()

    def test_a_rising_held_out_loss_against_a_falling_training_loss_is_named(self):
        signal, detail = classify_generalization(
            validation=_validation([1.20, 1.28, 1.35, 1.44, 1.52]),
            training=_series([1.5, 1.3, 1.1, 0.9]),
            validation_track_count=4,
        )
        assert signal == GeneralizationSignal.HELD_OUT_LOSS_DEGRADED.value
        assert "fitted the training split" in detail

    def test_too_few_tracks_is_insufficient_evidence(self):
        signal, detail = classify_generalization(
            validation=_validation([1.5, 1.3, 1.1, 0.9, 0.7], tracks=2),
            training=_series([1.5, 1.4, 1.3]),
            validation_track_count=MINIMUM_VALIDATION_TRACKS - 1,
        )
        assert signal == GeneralizationSignal.INSUFFICIENT_EVIDENCE.value
        assert "validation track" in detail

    def test_too_few_points_is_insufficient_evidence(self):
        signal, _ = classify_generalization(
            validation=_validation([1.5, 1.0]),
            training=_series([1.5, 1.4, 1.3]),
            validation_track_count=4,
        )
        assert signal == GeneralizationSignal.INSUFFICIENT_EVIDENCE.value

    def test_the_threshold_is_what_separates_the_two_verdicts(self):
        just_under = 1.0 - (GENERALIZATION_IMPROVEMENT_THRESHOLD / 2)
        just_over = 1.0 - (GENERALIZATION_IMPROVEMENT_THRESHOLD * 2)
        points = [1.0] * (MINIMUM_VALIDATION_POINTS - 1)
        under, _ = classify_generalization(
            validation=_validation([*points, just_under]),
            training=_series([1.5, 1.4, 1.3]),
            validation_track_count=4,
        )
        over, _ = classify_generalization(
            validation=_validation([*points, just_over]),
            training=_series([1.5, 1.4, 1.3]),
            validation_track_count=4,
        )
        assert under == GeneralizationSignal.NO_MEASURABLE_CHANGE.value
        assert over == GeneralizationSignal.HELD_OUT_LOSS_IMPROVED.value


class TestTheVocabularyItself:
    def test_there_is_no_value_that_claims_quality_or_convergence(self):
        forbidden = ("CONVERGED", "IMPROVED_QUALITY", "GOOD", "BETTER", "PRODUCTION", "READY")
        values = {item.value for item in GeneralizationSignal} | {
            item.value for item in TrainingSignal
        }
        for value in values:
            assert not any(word in value for word in forbidden), value

    def test_an_experiment_artifact_is_never_promotable(self):
        assert ARTIFACT_CLASS == (
            "EXPERIMENTAL",
            "NON_PRODUCTION",
            "NEVER_AUTO_PROMOTE",
        )


class TestTheLossSeries:
    def test_a_slope_needs_three_points(self):
        assert _series([1.0, 0.9]).slope() is None
        assert _series([1.0, 0.9, 0.8]).slope() is not None

    def test_the_slope_is_labelled_derived_and_disclaimed(self):
        stats = _series([1.0, 0.9, 0.8, 0.7]).statistics()
        assert stats["slope_source"] == "DERIVED"
        assert "not a convergence claim" in stats["slope_note"].lower()

    def test_a_non_finite_loss_lowers_the_finite_ratio(self):
        series = LossSeries(
            name="train",
            points=(
                LossPoint(step=1, loss=1.0),
                LossPoint(step=2, loss=float("nan")),
                LossPoint(step=3, loss=0.9),
            ),
        )
        assert series.finite_ratio == pytest.approx(2 / 3)
