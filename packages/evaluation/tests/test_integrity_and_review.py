"""Verification, benchmark integrity, blinded listening, and scale.

Four things that only fail quietly, which is why each is tested rather
than assumed:

*Verification* has to recompute claims, not re-read them. A check that
trusted the file it was checking would pass on a tampered evaluation.

*The frozen benchmark* is the fixed point every comparison rests on.
Its hash is asserted here against a literal, so a change to it fails a
test rather than silently redefining what "P20" means.

*Blinding* only works if the answer key lives somewhere the listener
does not.

*Scale* is checked because a decision that takes minutes gets skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from evaluation_fixtures import profile
from test_audio_backends import write_wav

from luber_evaluation.comparison import compare
from luber_evaluation.human import (
    LIGHT_AB_QUESTIONS,
    BlindMapping,
    HumanResponse,
    LightAbRubric,
    build_package,
    read_responses,
    record_responses,
    unblind,
    write_package,
)
from luber_evaluation.metrics import (
    CATALOGUE,
    MeasurementMode,
    MetricResult,
    MetricStatus,
    aggregate,
)
from luber_evaluation.qualification import Coverage, QualificationPolicy, decide
from luber_evaluation.reports import render_report, verify_evaluation, write_json, write_jsonl
from luber_evaluation.runner import (
    EvaluationRun,
    SyntheticGenerationBackend,
    coverage_of,
    execute_side,
)
from luber_evaluation.schemas import CandidateLineage, ModelRef, SampleProvenance
from luber_evaluation.serde import (
    DeserialisationError,
    comparison_from_dict,
    decision_from_dict,
    read_jsonl,
    read_metrics,
    run_from_dict,
    suite_from_dict,
)
from luber_evaluation.suite import (
    P20_EXPECTED_SHA256,
    BenchmarkIntegrityError,
    build_p20_suite,
    p20_identity,
    smoke_suite,
    verify_p20,
)


def _run(suite: Any) -> EvaluationRun:
    return EvaluationRun(
        evaluation_id="eval_" + "0" * 16,
        suite=suite,
        baseline=ModelRef(model_id="mdl_base", upstream_commit="abc"),
        candidate=ModelRef(model_id="cand_new", upstream_commit="abc", checkpoint_id="ckpt_new"),
        lineage=CandidateLineage(
            candidate_id="cand_new",
            checkpoint_id="ckpt_new",
            run_id="run_1",
            experiment_id="exp_1",
            base_model_id="mdl_base",
        ),
        suite_digest=suite.digest(),
        policy_digest=QualificationPolicy().digest(),
        seeds=suite.seeds,
    )


# ── benchmark integrity ──────────────────────────────────────────────


def test_the_frozen_benchmark_has_not_changed(repository_root: Path) -> None:
    assert verify_p20(repository_root) == P20_EXPECTED_SHA256


def test_a_changed_benchmark_is_refused(tmp_path: Path, repository_root: Path) -> None:
    """A suite built on altered benchmark content is not a P20 suite."""
    with pytest.raises(BenchmarkIntegrityError):
        verify_p20(repository_root, expected="0" * 64)


def test_the_benchmark_records_that_nobody_has_scored_it(repository_root: Path) -> None:
    """Phase 20H holds zero human scores. That is recorded, not filled in."""
    identity = p20_identity(repository_root)
    assert identity.human_scores_recorded == 0
    assert identity.human_score_store == "absent"
    assert identity.case_count == 28


def test_the_p20_suite_marks_human_dimensions_as_human(repository_root: Path) -> None:
    suite = build_p20_suite(repository_root, seeds=(11,))
    human = suite.human_required_metrics()
    assert "vocal_naturalness" in human
    for name in human:
        assert CATALOGUE[name].mode == MeasurementMode.HUMAN_REQUIRED.value
        assert name not in suite.required_metrics()


# ── verification ─────────────────────────────────────────────────────


def _completed(tmp_path: Path) -> tuple[EvaluationRun, Path, list[SampleProvenance]]:
    suite = smoke_suite(seeds=(11,))
    run = _run(suite)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    renders = tmp_path / "renders"
    from luber_evaluation.backends import RenderedAudioBackend, render_filename

    for case in suite.cases:
        write_wav(renders / render_filename(case.case_id, 11), 3.0)
    side = execute_side(
        run,
        run.candidate,
        RenderedAudioBackend(renders, serves_model_id="cand_new"),
        artifacts / "candidate",
    )

    write_json(artifacts / "evaluation.json", run.to_dict())
    write_json(artifacts / "suite.json", suite.to_dict())
    write_json(artifacts / "policy.json", QualificationPolicy().to_dict())
    write_jsonl(artifacts / "samples.jsonl", [s.to_dict() for s in side.samples])
    return run, artifacts, side.samples


def test_verification_passes_on_an_intact_evaluation(tmp_path: Path) -> None:
    run, artifacts, samples = _completed(tmp_path)
    problems = verify_evaluation(
        artifacts_dir=artifacts,
        suite=run.suite,
        policy=QualificationPolicy(),
        run=run,
        decision=None,
        samples=samples,
        checkpoint_status="READY",
        checkpoint_kind="ADAPTER",
    )
    assert problems == []


def test_verification_notices_swapped_audio(tmp_path: Path) -> None:
    """The audio a listener judges must be the audio that was measured."""
    run, artifacts, samples = _completed(tmp_path)
    swapped = Path(samples[0].artifact_ref or "")
    write_wav(swapped, 3.0, amplitude=0.9)

    problems = verify_evaluation(
        artifacts_dir=artifacts,
        suite=run.suite,
        policy=QualificationPolicy(),
        run=run,
        decision=None,
        samples=samples,
    )
    assert any(problem.check == "sample_digest" for problem in problems)


def test_verification_notices_a_missing_sample(tmp_path: Path) -> None:
    run, artifacts, samples = _completed(tmp_path)
    Path(samples[0].artifact_ref or "").unlink()
    problems = verify_evaluation(
        artifacts_dir=artifacts,
        suite=run.suite,
        policy=QualificationPolicy(),
        run=run,
        decision=None,
        samples=samples,
    )
    assert any(problem.check == "sample_present" for problem in problems)


def test_verification_notices_an_edited_suite(tmp_path: Path) -> None:
    run, artifacts, samples = _completed(tmp_path)
    edited = smoke_suite(seeds=(11,))
    edited.cases[0].spec.duration_seconds = 999.0

    problems = verify_evaluation(
        artifacts_dir=artifacts,
        suite=edited,
        policy=QualificationPolicy(),
        run=run,
        decision=None,
        samples=samples,
    )
    assert any(problem.check == "suite_digest" for problem in problems)


def test_verification_notices_a_mock_candidate(tmp_path: Path) -> None:
    run, artifacts, samples = _completed(tmp_path)
    problems = verify_evaluation(
        artifacts_dir=artifacts,
        suite=run.suite,
        policy=QualificationPolicy(),
        run=run,
        decision=None,
        samples=samples,
        checkpoint_status="READY",
        checkpoint_kind="MOCK",
    )
    assert any(problem.check == "candidate_readiness" for problem in problems)


def test_a_truncated_metrics_file_raises_rather_than_shrinking_the_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"metric_name": "peak_dbfs"}\n{"metric_name": "silence_ra\n', encoding="utf-8")
    with pytest.raises(DeserialisationError):
        read_jsonl(path)


def test_artifacts_round_trip_without_losing_anything(tmp_path: Path) -> None:
    """What was written is what comes back, and derived views are re-derived."""
    suite = smoke_suite(seeds=(11, 23))
    run = _run(suite)
    baseline = execute_side(
        run, run.baseline, SyntheticGenerationBackend(profile("b")), tmp_path / "b"
    )
    candidate = execute_side(
        run,
        run.candidate,
        SyntheticGenerationBackend(profile("c", silence_ratio=0.09)),
        tmp_path / "c",
    )
    comparison = compare(run.evaluation_id, baseline.aggregates(suite), candidate.aggregates(suite))
    cases, with_results, metrics, measured = coverage_of(suite, candidate)
    decision = decide(
        evaluation_id=run.evaluation_id,
        candidate_id="cand_new",
        policy=QualificationPolicy(),
        comparison=comparison,
        candidate_aggregates=candidate.aggregates(suite),
        coverage=Coverage(
            cases_expected=cases,
            cases_with_results=with_results,
            metrics_expected=metrics,
            metrics_measured=measured,
        ),
    )

    assert suite_from_dict(suite.to_dict()).digest() == suite.digest()
    assert run_from_dict(run.to_dict(), suite).to_dict() == run.to_dict()
    assert comparison_from_dict(comparison.to_dict()).to_dict() == comparison.to_dict()
    assert decision_from_dict(decision.to_dict()).to_dict() == decision.to_dict()

    rows = [{"side": "candidate", **m.to_dict()} for m in candidate.metrics]
    write_jsonl(tmp_path / "metrics.jsonl", rows)
    assert len(read_metrics(tmp_path / "metrics.jsonl", side="candidate")) == len(candidate.metrics)
    assert read_metrics(tmp_path / "metrics.jsonl", side="baseline") == []


def test_a_report_states_what_was_not_measured(tmp_path: Path) -> None:
    suite = smoke_suite(seeds=(11,))
    run = _run(suite)
    baseline = execute_side(
        run, run.baseline, SyntheticGenerationBackend(profile("b")), tmp_path / "b"
    )
    candidate = execute_side(
        run, run.candidate, SyntheticGenerationBackend(profile("c")), tmp_path / "c"
    )
    comparison = compare(run.evaluation_id, baseline.aggregates(suite), candidate.aggregates(suite))
    decision = decide(
        evaluation_id=run.evaluation_id,
        candidate_id="cand_new",
        policy=QualificationPolicy(),
        comparison=comparison,
        candidate_aggregates=candidate.aggregates(suite),
        coverage=Coverage(2, 2, 6, 6),
    )
    report = render_report(
        run=run,
        comparison=comparison,
        decision=decision,
        policy=QualificationPolicy(),
        baseline_aggregates=baseline.aggregates(suite),
        candidate_aggregates=candidate.aggregates(suite),
    )
    assert "vocal_naturalness" in report
    assert "not production" in report.lower() or "not mean production" in report.lower()


# ── blinded listening ────────────────────────────────────────────────


def _samples(suite: Any) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], str]]:
    baseline = {(case.case_id, 11): f"/audio/base/{case.case_id}.wav" for case in suite.cases}
    candidate = {(case.case_id, 11): f"/audio/cand/{case.case_id}.wav" for case in suite.cases}
    return baseline, candidate


def test_the_answer_key_is_not_in_the_package(tmp_path: Path) -> None:
    suite = smoke_suite(seeds=(11,))
    baseline, candidate = _samples(suite)
    package, mapping = build_package(
        evaluation_id="eval_" + "0" * 16,
        cases=suite.cases,
        baseline_samples=baseline,
        candidate_samples=candidate,
    )
    paths = write_package(tmp_path / "review", package, mapping)

    package_text = paths["package"].read_text(encoding="utf-8")
    assert "candidate" not in package_text
    assert "baseline" not in package_text
    assert "ckpt_" not in package_text

    mapping_payload = json.loads(paths["mapping"].read_text(encoding="utf-8"))
    assert mapping_payload["assignments"]
    assert "never give it to a listener" in mapping_payload["warning"]


def test_blinding_is_deterministic_but_not_all_one_way() -> None:
    """Stable across runs, and A is not always the candidate."""
    suite = smoke_suite(seeds=(11,))
    cases = [
        type(suite.cases[0])(
            case_id=f"CASE-{index:03d}",
            case_type=suite.cases[0].case_type,
            spec=suite.cases[0].spec,
            applicable_metrics=suite.cases[0].applicable_metrics,
        )
        for index in range(40)
    ]
    baseline = {(case.case_id, 11): f"/b/{case.case_id}.wav" for case in cases}
    candidate = {(case.case_id, 11): f"/c/{case.case_id}.wav" for case in cases}

    _first, first_map = build_package(
        evaluation_id="eval_" + "0" * 16,
        cases=cases,
        baseline_samples=baseline,
        candidate_samples=candidate,
    )
    _second, second_map = build_package(
        evaluation_id="eval_" + "0" * 16,
        cases=cases,
        baseline_samples=baseline,
        candidate_samples=candidate,
    )
    assert first_map.assignments == second_map.assignments

    a_is_candidate = sum(1 for value in first_map.assignments.values() if value["A"] == "candidate")
    assert 0 < a_is_candidate < len(first_map.assignments)


def test_unblinding_turns_a_b_answers_into_sides(tmp_path: Path) -> None:
    suite = smoke_suite(seeds=(11,))
    baseline, candidate = _samples(suite)
    package, mapping = build_package(
        evaluation_id="eval_" + "0" * 16,
        cases=suite.cases,
        baseline_samples=baseline,
        candidate_samples=candidate,
    )
    responses = []
    for pair in package.pairs:
        # Always pick whichever side is the candidate.
        choice = "A" if mapping.assignments[pair.pair_id]["A"] == "candidate" else "B"
        responses.append(
            HumanResponse(
                evaluation_id=package.evaluation_id,
                pair_id=pair.pair_id,
                case_id=pair.case_id,
                question_id="overall_preference",
                choice=choice,
                reviewer="listener-1",
                rubric_version=LightAbRubric().rubric_version,
            )
        )
    record_responses(tmp_path / "review", responses)
    evidence = unblind(read_responses(tmp_path / "review"), mapping)
    assert evidence.preference_share("overall_preference") == 1.0


def test_no_preference_is_not_counted_as_half_a_vote() -> None:
    mapping = BlindMapping(
        evaluation_id="eval_" + "0" * 16,
        assignments={"pair-1": {"A": "candidate", "B": "baseline"}},
    )
    responses = [
        HumanResponse(
            evaluation_id="eval_" + "0" * 16,
            pair_id="pair-1",
            case_id="CASE-1",
            question_id="overall_preference",
            choice="NO_PREFERENCE",
            reviewer="listener-1",
            rubric_version="1",
        )
    ]
    evidence = unblind(responses, mapping)
    assert evidence.preference_share("overall_preference") is None


def test_every_rubric_question_asks_about_something_unmeasurable() -> None:
    """A listener's attention is only spent where nothing else can look."""
    assert len(LIGHT_AB_QUESTIONS) == 5
    for question in LIGHT_AB_QUESTIONS:
        assert question.kind == "CHOICE"
    assert "NO_PREFERENCE" in LightAbRubric().choices


# ── scale ────────────────────────────────────────────────────────────


def test_ten_thousand_metric_entries_aggregate_quickly() -> None:
    results = [
        MetricResult(
            metric_name="peak_dbfs",
            status=MetricStatus.MEASURED.value,
            case_id=f"CASE-{index % 1000:04d}",
            seed=11 + index % 3,
            source="AUDIO_ANALYSIS",
            value=-1.0 - (index % 50) / 100.0,
        )
        for index in range(10_000)
    ]
    summary = aggregate("peak_dbfs", results)
    assert summary.count_measured == 10_000
    assert summary.median_value is not None
    assert summary.p10 is not None and summary.p90 is not None


def test_a_thousand_case_suite_compares_and_decides(tmp_path: Path) -> None:
    base = smoke_suite(seeds=(11,))
    template = base.cases[0]
    large = smoke_suite(seeds=(11,))
    large.cases = [
        type(template)(
            case_id=f"CASE-{index:04d}",
            case_type=template.case_type,
            spec=template.spec,
            applicable_metrics=template.applicable_metrics,
        )
        for index in range(1_000)
    ]
    run = _run(large)
    baseline = execute_side(
        run, run.baseline, SyntheticGenerationBackend(profile("b")), tmp_path / "b"
    )
    candidate = execute_side(
        run,
        run.candidate,
        SyntheticGenerationBackend(profile("c", silence_ratio=0.09)),
        tmp_path / "c",
    )
    comparison = compare(run.evaluation_id, baseline.aggregates(large), candidate.aggregates(large))
    cases, with_results, metrics, measured = coverage_of(large, candidate)
    decision = decide(
        evaluation_id=run.evaluation_id,
        candidate_id="cand_new",
        policy=QualificationPolicy(),
        comparison=comparison,
        candidate_aggregates=candidate.aggregates(large),
        coverage=Coverage(cases, with_results, metrics, measured),
    )
    assert with_results == 1_000
    assert decision.outcome in {"QUALIFIED", "REJECTED", "BLOCKED", "HUMAN_REVIEW_REQUIRED"}


def test_a_hundred_evaluations_list_and_read_back(tmp_path: Path) -> None:
    from luber_evaluation.registry import EvaluationRegistry
    from luber_training.registry import Registry

    registry = EvaluationRegistry(Registry(tmp_path / "registry"))
    for index in range(100):
        registry.save_evaluation(
            {
                "evaluation_id": f"eval_{index:016d}",
                "status": "COMPLETED",
                "lineage": {"candidate_id": f"cand_{index:016d}", "run_id": "run_1"},
            },
            overwrite=False,
        )
    listed = registry.list_all("evaluations")
    assert len(listed) == 100
    assert len(registry.find("evaluations", status="COMPLETED")) == 100
