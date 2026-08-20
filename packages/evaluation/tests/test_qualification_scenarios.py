"""The scenarios that decide whether this package is worth having.

Each one is a candidate the project could plausibly produce, and each
checks that the verdict is the honest one rather than the convenient
one. Two matter more than the rest:

*A candidate identical to the baseline must not appear improved.* If
noise can register as progress, every subsequent decision is built on
it.

*A hypothesis about something only a listener can judge cannot be
satisfied by technical measurement.* A candidate whose experiment
claimed better vocal naturalness must reach HUMAN_REVIEW_REQUIRED even
when every automatic gate passes, because nothing here has examined the
claim it was actually making.
"""

from __future__ import annotations

from typing import Any

import pytest
from evaluation_fixtures import BASELINE_METRICS, profile

from luber_evaluation.comparison import compare
from luber_evaluation.metrics import MetricStatus
from luber_evaluation.qualification import (
    Coverage,
    HypothesisTarget,
    QualificationPolicy,
    decide,
    policy_by_name,
)
from luber_evaluation.runner import (
    EvaluationRun,
    SyntheticGenerationBackend,
    SyntheticProfile,
    coverage_of,
    execute_side,
)
from luber_evaluation.schemas import (
    CandidateLineage,
    EvaluationRunStatus,
    ModelRef,
    QualificationOutcome,
)
from luber_evaluation.suite import smoke_suite

FULL_COVERAGE = Coverage(
    cases_expected=10,
    cases_with_results=10,
    metrics_expected=20,
    metrics_measured=20,
)


def _lineage() -> CandidateLineage:
    return CandidateLineage(
        candidate_id="cand_test",
        checkpoint_id="ckpt_test",
        run_id="run_test",
        experiment_id="exp_test",
        base_model_id="mdl_test",
    )


def run_scenario(
    tmp_path: Any,
    baseline: SyntheticProfile,
    candidate: SyntheticProfile,
    *,
    seeds: tuple[int, ...] = (11, 23),
) -> tuple[Any, Any, Coverage]:
    """Execute both sides synthetically and return what to decide on."""
    suite = smoke_suite(seeds=seeds)
    run = EvaluationRun(
        evaluation_id="eval_" + "0" * 16,
        suite=suite,
        baseline=ModelRef(model_id="mdl_base", upstream_commit="abc", label="baseline"),
        candidate=ModelRef(model_id="cand_test", upstream_commit="abc", checkpoint_id="ckpt_test"),
        lineage=_lineage(),
        suite_digest=suite.digest(),
        seeds=seeds,
        status=EvaluationRunStatus.RUNNING.value,
    )
    baseline_side = execute_side(
        run, run.baseline, SyntheticGenerationBackend(baseline), tmp_path / "baseline"
    )
    candidate_side = execute_side(
        run, run.candidate, SyntheticGenerationBackend(candidate), tmp_path / "candidate"
    )
    comparison = compare(
        run.evaluation_id, baseline_side.aggregates(suite), candidate_side.aggregates(suite)
    )
    cases, with_results, metrics, measured = coverage_of(suite, candidate_side)
    coverage = Coverage(
        cases_expected=cases,
        cases_with_results=with_results,
        metrics_expected=metrics,
        metrics_measured=measured,
    )
    return comparison, candidate_side.aggregates(suite), coverage


def test_identical_candidate_shows_no_improvement(tmp_path: Any) -> None:
    """A candidate identical to the baseline invents nothing.

    The most important negative result in the package: if this ever
    reports an improvement, every other verdict is noise.
    """
    comparison, aggregates, coverage = run_scenario(
        tmp_path, profile("baseline"), profile("candidate")
    )

    assert comparison.improvements == []
    assert comparison.regressions == []
    assert comparison.advisory_score()["score"] is None

    decision = decide(
        evaluation_id="eval_" + "0" * 16,
        candidate_id="cand_test",
        policy=QualificationPolicy(),
        comparison=comparison,
        candidate_aggregates=aggregates,
        coverage=coverage,
    )
    assert decision.outcome == QualificationOutcome.QUALIFIED.value
    assert decision.hypothesis_status == "NONE_STATED"


def test_technical_regression_is_rejected(tmp_path: Any) -> None:
    """Clipping past the hard ceiling rejects, whatever else improved."""
    comparison, aggregates, coverage = run_scenario(
        tmp_path,
        profile("baseline"),
        profile("candidate", clipping_sample_ratio=0.04, stereo_width=0.75),
    )

    decision = decide(
        evaluation_id="eval_" + "0" * 16,
        candidate_id="cand_test",
        policy=QualificationPolicy(),
        comparison=comparison,
        candidate_aggregates=aggregates,
        coverage=coverage,
    )
    assert decision.outcome == QualificationOutcome.REJECTED.value
    assert "clipping_sample_ratio" in " ".join(decision.failed_gates)


def test_failed_generations_reject_before_any_quality_claim(tmp_path: Any) -> None:
    """A model that cannot generate is unusable regardless of metrics."""
    comparison, aggregates, coverage = run_scenario(
        tmp_path,
        profile("baseline"),
        profile("candidate", failing_cases=("SYN-KO-01",)),
    )

    decision = decide(
        evaluation_id="eval_" + "0" * 16,
        candidate_id="cand_test",
        policy=QualificationPolicy(),
        comparison=comparison,
        candidate_aggregates=aggregates,
        coverage=coverage,
    )
    assert decision.outcome == QualificationOutcome.REJECTED.value
    assert any("generation" in reason for reason in decision.reasons)


def test_target_improvement_qualifies(tmp_path: Any) -> None:
    """A measurable hypothesis, met, with nothing else broken."""
    comparison, aggregates, coverage = run_scenario(
        tmp_path,
        profile("baseline", clipping_sample_ratio=0.006),
        profile("candidate", clipping_sample_ratio=0.001),
    )

    decision = decide(
        evaluation_id="eval_" + "0" * 16,
        candidate_id="cand_test",
        policy=QualificationPolicy(),
        comparison=comparison,
        candidate_aggregates=aggregates,
        coverage=coverage,
        hypothesis=HypothesisTarget(
            description="reduce clipping",
            metric_name="clipping_sample_ratio",
        ),
    )
    assert decision.outcome == QualificationOutcome.QUALIFIED.value
    assert decision.hypothesis_status == "SUPPORTED"


def test_human_only_hypothesis_requires_review(tmp_path: Any) -> None:
    """The branch the whole phase exists for.

    Every automatic gate passes and the candidate is technically clean.
    It still must not qualify: the claim it was making is about vocal
    naturalness, and no measurement in this project addresses that.
    """
    comparison, aggregates, coverage = run_scenario(
        tmp_path, profile("baseline"), profile("candidate", stereo_width=0.58)
    )

    decision = decide(
        evaluation_id="eval_" + "0" * 16,
        candidate_id="cand_test",
        policy=QualificationPolicy(),
        comparison=comparison,
        candidate_aggregates=aggregates,
        coverage=coverage,
        hypothesis=HypothesisTarget(
            description="the vocals should sound more like a person",
            metric_name="vocal_naturalness",
        ),
    )
    assert decision.outcome == QualificationOutcome.HUMAN_REVIEW_REQUIRED.value
    assert decision.human_review_required_for == ["vocal_naturalness"]
    assert decision.outcome != QualificationOutcome.QUALIFIED.value


def test_incomplete_evaluation_blocks_rather_than_rejects(tmp_path: Any) -> None:
    """Missing evidence is not evidence of failure.

    BLOCKED, not REJECTED: rejecting a candidate nobody finished
    measuring would record a conclusion the run never reached.
    """
    comparison, aggregates, _ = run_scenario(tmp_path, profile("baseline"), profile("candidate"))

    decision = decide(
        evaluation_id="eval_" + "0" * 16,
        candidate_id="cand_test",
        policy=QualificationPolicy(),
        comparison=comparison,
        candidate_aggregates=aggregates,
        coverage=Coverage(
            cases_expected=10,
            cases_with_results=4,
            metrics_expected=20,
            metrics_measured=6,
        ),
    )
    assert decision.outcome == QualificationOutcome.BLOCKED.value
    assert decision.outcome != QualificationOutcome.REJECTED.value


def test_integrity_problem_blocks_before_anything_else(tmp_path: Any) -> None:
    """A compromised benchmark makes every downstream number moot."""
    comparison, aggregates, coverage = run_scenario(
        tmp_path,
        profile("baseline"),
        profile("candidate", clipping_sample_ratio=0.09),
    )

    decision = decide(
        evaluation_id="eval_" + "0" * 16,
        candidate_id="cand_test",
        policy=QualificationPolicy(),
        comparison=comparison,
        candidate_aggregates=aggregates,
        coverage=coverage,
        blocking_problems=["benchmark_integrity: the frozen benchmark has changed"],
    )
    # Blocked even though the clipping alone would have rejected it: an
    # evaluation run against altered inputs cannot support any verdict.
    assert decision.outcome == QualificationOutcome.BLOCKED.value


def test_missing_hard_gate_metric_blocks(tmp_path: Any) -> None:
    """An unmeasured safety gate is not a passed safety gate."""
    comparison, aggregates, coverage = run_scenario(
        tmp_path, profile("baseline"), profile("candidate")
    )
    aggregates.pop("clipping_sample_ratio", None)

    decision = decide(
        evaluation_id="eval_" + "0" * 16,
        candidate_id="cand_test",
        policy=QualificationPolicy(),
        comparison=comparison,
        candidate_aggregates=aggregates,
        coverage=coverage,
    )
    assert decision.outcome == QualificationOutcome.BLOCKED.value


def test_never_regress_metric_rejects_at_any_severity(tmp_path: Any) -> None:
    """Reliability is not a dial that may be traded for anything."""
    comparison, aggregates, coverage = run_scenario(
        tmp_path,
        profile("baseline"),
        profile("candidate", failing_cases=("SYN-INST-01",)),
    )
    decision = decide(
        evaluation_id="eval_" + "0" * 16,
        candidate_id="cand_test",
        policy=QualificationPolicy(),
        comparison=comparison,
        candidate_aggregates=aggregates,
        coverage=coverage,
    )
    assert decision.outcome == QualificationOutcome.REJECTED.value


def test_strict_policy_is_stricter_about_coverage() -> None:
    strict = policy_by_name("STRICT")
    default = policy_by_name("NEUTRAL_CONSERVATIVE")
    assert strict.minimum_case_coverage > default.minimum_case_coverage
    assert strict.digest() != default.digest()


def test_synthetic_metrics_are_marked_simulated(tmp_path: Any) -> None:
    """A synthetic value can never be read as a measurement of a model."""
    suite = smoke_suite(seeds=(11,))
    run = EvaluationRun(
        evaluation_id="eval_" + "0" * 16,
        suite=suite,
        baseline=ModelRef(model_id="mdl_base", upstream_commit="abc"),
        candidate=ModelRef(model_id="cand_test", upstream_commit="abc"),
        lineage=_lineage(),
        seeds=(11,),
    )
    side = execute_side(
        run,
        run.candidate,
        SyntheticGenerationBackend(profile("candidate")),
        tmp_path / "candidate",
    )
    measured = [m for m in side.metrics if m.status == MetricStatus.MEASURED.value]
    assert measured
    assert all(m.source == "SIMULATED" for m in measured)
    assert all(sample.synthetic for sample in side.samples)
    assert all(sample.raw_sha256 is None for sample in side.samples)


def test_human_required_metrics_are_never_measured_automatically(tmp_path: Any) -> None:
    """Nothing synthetic supplies a value for a listening dimension."""
    suite = smoke_suite(seeds=(11,))
    run = EvaluationRun(
        evaluation_id="eval_" + "0" * 16,
        suite=suite,
        baseline=ModelRef(model_id="mdl_base", upstream_commit="abc"),
        candidate=ModelRef(model_id="cand_test", upstream_commit="abc"),
        lineage=_lineage(),
        seeds=(11,),
    )
    dishonest = SyntheticProfile(
        label="dishonest",
        metrics={**BASELINE_METRICS, "vocal_naturalness": 0.98, "korean_pronunciation": 0.95},
    )
    side = execute_side(
        run, run.candidate, SyntheticGenerationBackend(dishonest), tmp_path / "candidate"
    )
    for name in ("vocal_naturalness", "korean_pronunciation"):
        results = [m for m in side.metrics if m.metric_name == name]
        assert results, f"{name} should be recorded as unmeasurable, not omitted"
        assert all(m.status != MetricStatus.MEASURED.value for m in results)
        assert all(m.value is None for m in results)


@pytest.mark.parametrize("share", [0.0, 0.5, 1.0])
def test_coverage_thresholds_are_applied_not_averaged(tmp_path: Any, share: float) -> None:
    """Coverage is a floor per dimension, not a blended figure."""
    comparison, aggregates, _ = run_scenario(tmp_path, profile("baseline"), profile("candidate"))
    coverage = Coverage(
        cases_expected=10,
        cases_with_results=10,
        metrics_expected=20,
        metrics_measured=int(20 * share),
    )
    decision = decide(
        evaluation_id="eval_" + "0" * 16,
        candidate_id="cand_test",
        policy=QualificationPolicy(),
        comparison=comparison,
        candidate_aggregates=aggregates,
        coverage=coverage,
    )
    if share < QualificationPolicy().minimum_metric_coverage:
        assert decision.outcome == QualificationOutcome.BLOCKED.value
    else:
        assert decision.outcome != QualificationOutcome.BLOCKED.value
