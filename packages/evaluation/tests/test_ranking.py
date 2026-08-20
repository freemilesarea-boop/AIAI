"""Choosing between checkpoints, and refusing to choose badly.

The scenario that matters has three checkpoints from one run: one that
changed nothing, one that improved the experiment's target metric
cleanly, and one that improved the same target further while breaking
something technical. The third must not win. A weighted score would let
its larger target gain outweigh the regression, which is the trade this
project has decided not to make silently.
"""

from __future__ import annotations

from typing import Any

import pytest
from evaluation_fixtures import profile

from luber_evaluation.comparison import compare
from luber_evaluation.qualification import Coverage, HypothesisTarget, QualificationPolicy, decide
from luber_evaluation.ranking import CheckpointCandidate, RankingError, pareto_front, rank
from luber_evaluation.runner import (
    EvaluationRun,
    SyntheticGenerationBackend,
    SyntheticProfile,
    coverage_of,
    execute_side,
)
from luber_evaluation.schemas import CandidateLineage, ModelRef, QualificationOutcome
from luber_evaluation.suite import smoke_suite

TARGET = "silence_ratio"


def _evaluate(
    tmp_path: Any,
    label: str,
    baseline: SyntheticProfile,
    candidate: SyntheticProfile,
) -> tuple[Any, Any]:
    suite = smoke_suite(seeds=(11, 23))
    run = EvaluationRun(
        evaluation_id="eval_" + label.ljust(16, "0")[:16],
        suite=suite,
        baseline=ModelRef(model_id="mdl_base", upstream_commit="abc"),
        candidate=ModelRef(model_id=f"cand_{label}", upstream_commit="abc"),
        lineage=CandidateLineage(
            candidate_id=f"cand_{label}",
            checkpoint_id=f"ckpt_{label}",
            run_id="run_shared",
            experiment_id="exp_shared",
            base_model_id="mdl_base",
        ),
        seeds=(11, 23),
    )
    baseline_side = execute_side(
        run, run.baseline, SyntheticGenerationBackend(baseline), tmp_path / f"{label}-base"
    )
    candidate_side = execute_side(
        run, run.candidate, SyntheticGenerationBackend(candidate), tmp_path / f"{label}-cand"
    )
    comparison = compare(
        run.evaluation_id, baseline_side.aggregates(suite), candidate_side.aggregates(suite)
    )
    cases, with_results, metrics, measured = coverage_of(suite, candidate_side)
    decision = decide(
        evaluation_id=run.evaluation_id,
        candidate_id=run.lineage.candidate_id,
        policy=QualificationPolicy(),
        comparison=comparison,
        candidate_aggregates=candidate_side.aggregates(suite),
        coverage=Coverage(
            cases_expected=cases,
            cases_with_results=with_results,
            metrics_expected=metrics,
            metrics_measured=measured,
        ),
        hypothesis=HypothesisTarget(description="reduce dead air", metric_name=TARGET),
    )
    return comparison, decision


@pytest.fixture
def three_checkpoints(tmp_path: Any) -> list[CheckpointCandidate]:
    """A, B, C from one run — unchanged, clean gain, and a poisoned gain."""
    # Dead air the candidates are trying to reduce, far enough above
    # the rate noise floor that a real improvement can register.
    baseline = profile("baseline", silence_ratio=0.11)

    unchanged_cmp, unchanged_dec = _evaluate(
        tmp_path, "aaa", baseline, profile("a", silence_ratio=0.11)
    )
    clean_cmp, clean_dec = _evaluate(tmp_path, "bbb", baseline, profile("b", silence_ratio=0.05))
    poisoned_cmp, poisoned_dec = _evaluate(
        tmp_path,
        "ccc",
        baseline,
        profile("c", silence_ratio=0.02, clipping_sample_ratio=0.04),
    )

    return [
        CheckpointCandidate(
            checkpoint_id="ckpt_aaa",
            step=1000,
            decision=unchanged_dec,
            comparison=unchanged_cmp,
            final_train_loss=0.40,
        ),
        CheckpointCandidate(
            checkpoint_id="ckpt_bbb",
            step=2000,
            decision=clean_dec,
            comparison=clean_cmp,
            final_train_loss=0.35,
        ),
        CheckpointCandidate(
            checkpoint_id="ckpt_ccc",
            step=3000,
            decision=poisoned_dec,
            comparison=poisoned_cmp,
            # The lowest loss and the latest step, deliberately. Both
            # shortcuts would pick this one.
            final_train_loss=0.28,
        ),
    ]


def test_clean_target_gain_beats_poisoned_target_gain(
    three_checkpoints: list[CheckpointCandidate],
) -> None:
    ranked = rank(three_checkpoints, target_metric=TARGET)
    order = [item.checkpoint_id for item in ranked]

    assert order[0] == "ckpt_bbb"
    # Last despite the best training loss and the highest step.
    assert order[-1] == "ckpt_ccc"
    assert ranked[-1].outcome == QualificationOutcome.REJECTED.value


def test_ranking_states_that_loss_is_only_a_tie_break(
    three_checkpoints: list[CheckpointCandidate],
) -> None:
    ranked = rank(three_checkpoints, target_metric=TARGET)
    rationale = " ".join(ranked[0].rationale)
    assert "tie-break" in rationale
    assert "does not measure music quality" in rationale


def test_ranking_without_any_decision_refuses() -> None:
    """No evaluation means no ranking. Not a ranking by loss."""
    bare = [
        CheckpointCandidate(checkpoint_id="ckpt_x", step=9000, final_train_loss=0.05),
        CheckpointCandidate(checkpoint_id="ckpt_y", step=1000, final_train_loss=0.90),
    ]
    with pytest.raises(RankingError) as excinfo:
        rank(bare)
    assert "training loss" in str(excinfo.value)


def test_ranking_is_deterministic(three_checkpoints: list[CheckpointCandidate]) -> None:
    first = [item.checkpoint_id for item in rank(three_checkpoints, target_metric=TARGET)]
    shuffled = list(reversed(three_checkpoints))
    second = [item.checkpoint_id for item in rank(shuffled, target_metric=TARGET)]
    assert first == second


def test_pareto_front_does_not_invent_a_winner(
    three_checkpoints: list[CheckpointCandidate],
) -> None:
    """Where checkpoints genuinely trade off, several survive."""
    front = pareto_front(three_checkpoints, [TARGET, "clipping_sample_ratio"])
    assert "ckpt_bbb" in front
    # C is better on the target and worse on clipping: neither dominates
    # the other, so the front declines to pick between them.
    assert "ckpt_ccc" in front


def test_empty_ranking_is_empty_not_an_error() -> None:
    assert rank([]) == []
