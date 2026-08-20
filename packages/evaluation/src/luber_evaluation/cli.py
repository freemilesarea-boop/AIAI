"""Operator command line for model evaluation and qualification.

    python -m luber_evaluation --registry ./training-registry <command>

Operator-only and local, for the same reason Phase 25's CLI is: there
is no HTTP surface and no role. An ordinary LUBER account cannot start
an evaluation, read a qualification verdict or reach evaluation audio,
because no path to any of that exists outside this program.

`run start --backend synthetic` exercises the whole pipeline — suite,
generation lifecycle, aggregation, comparison, gates, ranking, report —
with a model that produces metric values and no audio at all. It is how
this package is tested without a trained checkpoint existing, and its
results are stamped SIMULATED at every layer so they can never be read
as evidence about a real model.

The verbs stop where the phase stops. `promote` records an operator
decision about a *qualified* candidate; nothing here activates a model
in production, and there is no flag that makes it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from luber_evaluation import backends as backend_module
from luber_evaluation.comparison import compare
from luber_evaluation.human import (
    HumanResponse,
    LightAbRubric,
    build_package,
    read_responses,
    record_responses,
    unblind,
)
from luber_evaluation.lineage import mark_evaluating, resolve_candidate
from luber_evaluation.metrics import CATALOGUE, MeasurementMode
from luber_evaluation.qualification import (
    POLICIES,
    Coverage,
    HypothesisTarget,
    QualificationPolicy,
    decide,
    policy_by_name,
)
from luber_evaluation.ranking import CheckpointCandidate, rank
from luber_evaluation.registry import (
    EVALUATION_COMPLETED,
    EVALUATION_CREATED,
    EVALUATION_FAILED,
    EVALUATION_STARTED,
    HUMAN_REVIEW_RECORDED,
    HUMAN_REVIEW_REQUESTED,
    EvaluationRegistry,
    transition,
)
from luber_evaluation.reports import (
    build_model_card_input,
    record_promotion_review,
    render_report,
    side_sample_rows,
    verify_evaluation,
    write_json,
    write_jsonl,
)
from luber_evaluation.runner import (
    EvaluationRun,
    SyntheticGenerationBackend,
    SyntheticProfile,
    coverage_of,
    execute_side,
)
from luber_evaluation.schemas import (
    EvaluationEntityKind,
    EvaluationMode,
    EvaluationRunStatus,
    is_valid_id,
    new_id,
    now,
)
from luber_evaluation.serde import (
    aggregates_from_dict,
    comparison_from_dict,
    coverage_from_dict,
    decision_from_dict,
    policy_from_dict,
    read_json,
    read_jsonl,
    run_from_dict,
    sample_from_dict,
    suite_from_dict,
)
from luber_evaluation.suite import EvaluationSuite, build_p20_suite, smoke_suite
from luber_training.orchestrator import Orchestrator
from luber_training.registry import Registry as TrainingRegistry

#: Suites this CLI can build. P20_FULL reads the frozen benchmark and
#: verifies its hash first; SMOKE never touches it.
SUITE_BUILDERS = ("P20_FULL", "SMOKE")

#: Where an evaluation's own files live inside its artifact directory.
AGGREGATES_FILE = "aggregates.json"
COVERAGE_FILE = "coverage.json"
MODEL_CARD_FILE = "model_card.json"


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def _fail(message: str) -> int:
    print(json.dumps({"ok": False, "error": message}, indent=2, ensure_ascii=False))
    return 1


def _registry(args: argparse.Namespace) -> EvaluationRegistry:
    training = TrainingRegistry(Path(args.registry).expanduser())
    root = Path(args.artifacts).expanduser() if args.artifacts else None
    return EvaluationRegistry(training, artifacts_root=root)


def _orchestrator(args: argparse.Namespace) -> Orchestrator:
    return Orchestrator(
        TrainingRegistry(Path(args.registry).expanduser()),
        repository_root=_repository(args),
    )


def _repository(args: argparse.Namespace) -> Path:
    return Path(args.repository).expanduser() if args.repository else Path.cwd()


def _evaluation_id(value: str) -> str:
    if not is_valid_id(value, EvaluationEntityKind.EVALUATION):
        raise ValueError(f"{value!r} is not a valid evaluation identifier")
    return value


def _build_suite(name: str, args: argparse.Namespace, seeds: tuple[int, ...]) -> EvaluationSuite:
    key = name.strip().upper()
    if key == "SMOKE":
        return smoke_suite(seeds=seeds)
    if key == "P20_FULL":
        return build_p20_suite(_repository(args), seeds=seeds, mode=args.mode)
    raise ValueError(f"unknown suite {name!r}. Available: {', '.join(SUITE_BUILDERS)}")


# ── suite ────────────────────────────────────────────────────────────


def cmd_suite_list(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for name in SUITE_BUILDERS:
        try:
            suite = _build_suite(name, args, (11,))
        except Exception as exc:
            rows.append({"suite_id": name, "available": False, "reason": str(exc)})
            continue
        rows.append(
            {
                "suite_id": suite.suite_id,
                "available": True,
                "suite_version": suite.suite_version,
                "cases": len(suite.cases),
                "automatic_metrics": len(suite.required_metrics()),
                "human_required_metrics": list(suite.human_required_metrics()),
                "benchmark": suite.benchmark.to_dict() if suite.benchmark else None,
            }
        )
    _print({"suites": rows})
    return 0


def cmd_suite_show(args: argparse.Namespace) -> int:
    suite = _build_suite(args.suite, args, tuple(args.seed or (11, 23, 37)))
    _print({"digest": suite.digest(), **suite.to_dict()})
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    """The catalogue, with what can and cannot be measured.

    Printed as a command because the honest answer to "what does this
    evaluate?" includes the dimensions it cannot, and those are easier
    to forget than the ones it reports on every run.
    """
    rows = []
    for name, spec in sorted(CATALOGUE.items()):
        rows.append(
            {
                "metric": name,
                "mode": spec.mode,
                "direction": spec.direction,
                "unit": spec.unit,
                "unavailability_reason": spec.unavailability_reason,
            }
        )
    _print(
        {
            "metrics": rows,
            "counts": {
                mode.value: sum(1 for spec in CATALOGUE.values() if spec.mode == mode.value)
                for mode in MeasurementMode
            },
        }
    )
    return 0


def cmd_policies(args: argparse.Namespace) -> int:
    _print(
        {
            name: {"digest": policy_by_name(name).digest(), **policy_by_name(name).to_dict()}
            for name in sorted(POLICIES)
        }
    )
    return 0


# ── run ──────────────────────────────────────────────────────────────


def cmd_run_create(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    registry = _registry(args)

    resolved = resolve_candidate(orchestrator, args.candidate_id)
    seeds = tuple(args.seed) if args.seed else None
    suite = _build_suite(args.suite, args, seeds or (11, 23, 37))
    if seeds:
        suite.seeds = seeds
    policy = policy_by_name(args.policy)

    evaluation_id = new_id(EvaluationEntityKind.EVALUATION)
    run = EvaluationRun(
        evaluation_id=evaluation_id,
        suite=suite,
        baseline=resolved.baseline,
        candidate=resolved.candidate,
        lineage=resolved.lineage,
        mode=args.mode,
        experiment_hypothesis=resolved.experiment_hypothesis,
        suite_digest=suite.digest(),
        policy_digest=policy.digest(),
        seeds=suite.seeds,
    )

    artifacts = registry.artifacts(evaluation_id)
    artifacts.ensure()
    write_json(artifacts.suite_json, suite.to_dict())
    write_json(artifacts.policy_json, policy.to_dict())
    payload = run.to_dict()
    payload["artifacts_directory"] = str(artifacts.directory)
    registry.save_evaluation(payload, overwrite=False)
    registry.audit(
        EVALUATION_CREATED,
        evaluation_id,
        candidate_id=resolved.lineage.candidate_id,
        suite_id=suite.suite_id,
        policy_id=policy.policy_id,
    )
    _print(payload)
    return 0


def _load(
    args: argparse.Namespace,
) -> tuple[
    EvaluationRegistry,
    dict[str, Any],
    EvaluationSuite,
    QualificationPolicy,
    EvaluationRun,
    Any,
]:
    registry = _registry(args)
    evaluation_id = _evaluation_id(args.evaluation_id)
    artifacts = registry.artifacts(evaluation_id)
    payload = registry.load_evaluation(evaluation_id)
    suite = suite_from_dict(read_json(artifacts.suite_json))
    policy = policy_from_dict(read_json(artifacts.policy_json))
    run = run_from_dict(payload, suite)
    return registry, payload, suite, policy, run, artifacts


def _backend_for(args: argparse.Namespace, side: str, model_id: str) -> Any:
    """The generation backend for one side.

    Each side gets its own, and a `rendered` or `ace-step` backend is
    told which model it serves. That is what stops one server or one
    directory answering for both sides of a comparison.
    """
    if args.backend == "synthetic":
        profile_path = getattr(args, f"{side}_profile")
        if not profile_path:
            raise ValueError(f"--{side}-profile is required for the synthetic backend")
        payload = read_json(Path(profile_path).expanduser())
        profile = SyntheticProfile(
            label=str(payload.get("label", side)),
            metrics={str(k): float(v) for k, v in payload.get("metrics", {}).items()},
            failure_rate=float(payload.get("failure_rate", 0.0)),
            failing_cases=tuple(payload.get("failing_cases", ())),
        )
        return SyntheticGenerationBackend(profile)

    if args.backend == "rendered":
        root = getattr(args, f"{side}_audio")
        if not root:
            raise ValueError(f"--{side}-audio is required for the rendered backend")
        return backend_module.RenderedAudioBackend(
            Path(root).expanduser(), serves_model_id=model_id
        )

    if args.backend == "ace-step":
        base_url = getattr(args, f"{side}_url")
        if not base_url:
            raise ValueError(f"--{side}-url is required for the ace-step backend")
        return backend_module.AceStepEvaluationBackend(
            backend_module.AceStepBackendConfig(
                base_url=base_url,
                serves_model_id=model_id,
                api_key_ref=args.api_key_ref,
            )
        )

    raise ValueError(f"unknown backend {args.backend!r}")


def cmd_run_start(args: argparse.Namespace) -> int:
    registry, payload, suite, policy, run, artifacts = _load(args)

    if run.status != EvaluationRunStatus.DRAFT.value:
        return _fail(
            f"evaluation {run.evaluation_id} is {run.status}; identity is frozen once a run "
            "starts, so re-running it means creating a new evaluation"
        )

    # VALIDATING is a real state, not a formality. Everything that can
    # be checked before anything is generated is checked here, so that a
    # misconfiguration costs nothing rather than an hour of GPU time and
    # a directory of audio nobody can use.
    run.status = transition(run.status, EvaluationRunStatus.VALIDATING.value)

    if suite.digest() != run.suite_digest:
        return _fail("the suite has changed since this evaluation was created")
    if policy.digest() != run.policy_digest:
        return _fail("the policy has changed since this evaluation was created")

    try:
        baseline_backend = _backend_for(args, "baseline", run.baseline.model_id)
        candidate_backend = _backend_for(args, "candidate", run.candidate.model_id)
    except (ValueError, OSError) as exc:
        run.status = transition(run.status, EvaluationRunStatus.FAILED.value)
        run.failed_at = now()
        run.error = str(exc)
        registry.save_evaluation(run.to_dict())
        registry.audit(EVALUATION_FAILED, run.evaluation_id, error=run.error)
        return _fail(str(exc))

    run.status = transition(run.status, EvaluationRunStatus.RUNNING.value)
    run.started_at = now()
    payload = run.to_dict()
    payload["artifacts_directory"] = str(artifacts.directory)
    registry.save_evaluation(payload)
    registry.audit(EVALUATION_STARTED, run.evaluation_id, backend=args.backend)

    try:
        baseline_side = execute_side(run, run.baseline, baseline_backend, artifacts.baseline_dir)
        candidate_side = execute_side(
            run, run.candidate, candidate_backend, artifacts.candidate_dir
        )
    except Exception as exc:
        run.status = transition(run.status, EvaluationRunStatus.FAILED.value)
        run.failed_at = now()
        run.error = f"{type(exc).__name__}: {exc}"
        registry.save_evaluation(run.to_dict())
        registry.audit(EVALUATION_FAILED, run.evaluation_id, error=run.error)
        return _fail(run.error)

    baseline_aggregates = baseline_side.aggregates(suite)
    candidate_aggregates = candidate_side.aggregates(suite)
    comparison = compare(run.evaluation_id, baseline_aggregates, candidate_aggregates)

    rows: list[dict[str, Any]] = []
    for side_name, side in (("baseline", baseline_side), ("candidate", candidate_side)):
        for result in [*side.metrics, *side.reliability_metrics(suite)]:
            rows.append({"side": side_name, **result.to_dict()})
    write_jsonl(artifacts.metrics_jsonl, rows)
    write_jsonl(
        artifacts.samples_jsonl,
        [*side_sample_rows(baseline_side), *side_sample_rows(candidate_side)],
    )
    write_json(
        artifacts.directory / AGGREGATES_FILE,
        {
            "baseline": {name: agg.to_dict() for name, agg in baseline_aggregates.items()},
            "candidate": {name: agg.to_dict() for name, agg in candidate_aggregates.items()},
        },
    )
    write_json(artifacts.comparisons_json, comparison.to_dict())

    expected_cases, with_results, expected_metrics, measured = coverage_of(suite, candidate_side)
    coverage = Coverage(
        cases_expected=expected_cases,
        cases_with_results=with_results,
        metrics_expected=expected_metrics,
        metrics_measured=measured,
    )
    write_json(artifacts.directory / COVERAGE_FILE, coverage.to_dict())

    run.status = transition(run.status, EvaluationRunStatus.COMPLETED.value)
    run.completed_at = now()
    payload = run.to_dict()
    payload["artifacts_directory"] = str(artifacts.directory)
    payload["coverage"] = coverage.to_dict()
    registry.save_evaluation(payload)
    registry.audit(
        EVALUATION_COMPLETED,
        run.evaluation_id,
        case_coverage=round(coverage.case_coverage, 4),
        metric_coverage=round(coverage.metric_coverage, 4),
    )
    mark_evaluating(_orchestrator(args), run.lineage.candidate_id)

    _print(
        {
            "evaluation_id": run.evaluation_id,
            "status": run.status,
            "backend": args.backend,
            "coverage": coverage.to_dict(),
            "pareto": comparison.pareto_summary(),
            "artifacts": str(artifacts.directory),
            "next": "qualify",
        }
    )
    return 0


def cmd_run_status(args: argparse.Namespace) -> int:
    registry = _registry(args)
    evaluation_id = _evaluation_id(args.evaluation_id)
    payload = registry.load_evaluation(evaluation_id)
    if registry.exists("qualifications", evaluation_id):
        payload["qualification"] = registry.read("qualifications", evaluation_id)
    payload["audit"] = registry.audit_events(evaluation_id)
    _print(payload)
    return 0


def cmd_run_list(args: argparse.Namespace) -> int:
    registry = _registry(args)
    rows = [
        {
            "evaluation_id": record["evaluation_id"],
            "status": record["status"],
            "candidate_id": record.get("lineage", {}).get("candidate_id"),
            "suite_id": record.get("suite_id"),
            "mode": record.get("mode"),
            "completed_at": record.get("completed_at"),
        }
        for record in registry.list_all("evaluations")
    ]
    _print({"evaluations": rows})
    return 0


# ── compare / qualify ────────────────────────────────────────────────


def cmd_compare(args: argparse.Namespace) -> int:
    _, _, _, _, run, artifacts = _load(args)
    aggregates = read_json(artifacts.directory / AGGREGATES_FILE)
    comparison = compare(
        run.evaluation_id,
        aggregates_from_dict(aggregates.get("baseline", {})),
        aggregates_from_dict(aggregates.get("candidate", {})),
    )
    write_json(artifacts.comparisons_json, comparison.to_dict())
    _print(comparison.to_dict())
    return 0


def _human_evidence(artifacts: Any) -> dict[str, Any] | None:
    """Human evidence, only if a review was actually completed."""
    from luber_evaluation.human import MAPPING_FILE_NAME

    mapping_path = artifacts.human_review_dir / MAPPING_FILE_NAME
    responses = read_responses(artifacts.human_review_dir)
    if not responses or not mapping_path.is_file():
        return None
    from luber_evaluation.human import BlindMapping

    payload = read_json(mapping_path)
    mapping = BlindMapping(
        evaluation_id=payload["evaluation_id"],
        assignments={k: dict(v) for k, v in payload.get("assignments", {}).items()},
        created_at=payload.get("created_at", ""),
    )
    return unblind(responses, mapping).to_dict()


def cmd_qualify(args: argparse.Namespace) -> int:
    registry, _record, suite, policy, run, artifacts = _load(args)

    if run.status != EvaluationRunStatus.COMPLETED.value:
        return _fail(
            f"evaluation {run.evaluation_id} is {run.status}; a verdict on an evaluation that "
            "did not complete would be a verdict on partial evidence"
        )

    comparison = comparison_from_dict(read_json(artifacts.comparisons_json))
    aggregates = read_json(artifacts.directory / AGGREGATES_FILE)
    candidate_aggregates = aggregates_from_dict(aggregates.get("candidate", {}))
    baseline_aggregates = aggregates_from_dict(aggregates.get("baseline", {}))
    coverage = coverage_from_dict(read_json(artifacts.directory / COVERAGE_FILE))

    hypothesis = None
    description = args.hypothesis or run.experiment_hypothesis
    if description or args.hypothesis_metric:
        hypothesis = HypothesisTarget(
            description=description,
            metric_name=args.hypothesis_metric,
            expect_improvement=not args.expect_decrease,
        )

    samples = [sample_from_dict(row) for row in read_jsonl(artifacts.samples_jsonl)]
    problems = verify_evaluation(
        artifacts_dir=artifacts.directory,
        suite=suite,
        policy=policy,
        run=run,
        decision=None,
        samples=samples,
        repository_root=_repository(args),
    )
    # Every problem verification can find at this point means the
    # recorded evidence does not reconstruct. None of them is
    # advisory, so all of them block.
    blocking = [f"{problem.check}: {problem.detail}" for problem in problems]

    decision = decide(
        evaluation_id=run.evaluation_id,
        candidate_id=run.lineage.candidate_id,
        policy=policy,
        comparison=comparison,
        candidate_aggregates=candidate_aggregates,
        coverage=coverage,
        hypothesis=hypothesis,
        human_evidence=_human_evidence(artifacts),
        blocking_problems=blocking or None,
    )

    registry.save_qualification(decision.to_dict())
    write_json(artifacts.qualification_json, decision.to_dict())

    report = render_report(
        run=run,
        comparison=comparison,
        decision=decision,
        policy=policy,
        baseline_aggregates=baseline_aggregates,
        candidate_aggregates=candidate_aggregates,
    )
    artifacts.report_md.write_text(report, encoding="utf-8")
    card = build_model_card_input(
        run=run,
        comparison=comparison,
        decision=decision,
        policy=policy,
        experiment_hypothesis=description,
        human_evidence_status="RECORDED" if _human_evidence(artifacts) else "NONE",
    )
    write_json(artifacts.directory / MODEL_CARD_FILE, card.to_dict())

    if decision.outcome == "HUMAN_REVIEW_REQUIRED":
        registry.audit(
            HUMAN_REVIEW_REQUESTED,
            run.evaluation_id,
            dimensions=decision.human_review_required_for,
        )

    summary: dict[str, Any] = {
        "outcome": decision.outcome,
        "hypothesis_status": decision.hypothesis_status,
        "reasons": decision.reasons,
        "failed_gates": sorted(decision.failed_gates),
        "human_review_required_for": sorted(decision.human_review_required_for),
        "report": str(artifacts.report_md),
    }
    # Only stated when it applies. Printing "QUALIFIED means…" under a
    # REJECTED verdict is the kind of line that gets quoted later
    # without the verdict above it.
    if decision.qualified:
        summary["note"] = (
            "QUALIFIED means the checkpoint may advance to promotion review. It is not production."
        )
    _print(summary)
    return 0


# ── checkpoint ranking ───────────────────────────────────────────────


def cmd_checkpoint_rank(args: argparse.Namespace) -> int:
    """Rank every evaluated checkpoint of one training run.

    Checkpoints without an evaluation are listed as unranked rather than
    ordered by step or loss. Ordering them would answer the question
    with the two things this package exists to say are not answers.
    """
    registry = _registry(args)
    orchestrator = _orchestrator(args)

    candidates: list[CheckpointCandidate] = []
    unranked: list[dict[str, Any]] = []
    for record in registry.list_all("evaluations"):
        lineage = record.get("lineage", {})
        if lineage.get("run_id") != args.run_id:
            continue
        evaluation_id = record["evaluation_id"]
        if not registry.exists("qualifications", evaluation_id):
            unranked.append(
                {
                    "checkpoint_id": lineage.get("checkpoint_id"),
                    "evaluation_id": evaluation_id,
                    "reason": "evaluated but not yet qualified",
                }
            )
            continue
        artifacts = registry.artifacts(evaluation_id)
        decision = decision_from_dict(registry.read("qualifications", evaluation_id))
        comparison = comparison_from_dict(read_json(artifacts.comparisons_json))
        checkpoint = orchestrator.get_checkpoint(lineage["checkpoint_id"])
        candidates.append(
            CheckpointCandidate(
                checkpoint_id=checkpoint.checkpoint_id,
                step=checkpoint.step,
                epoch=checkpoint.epoch,
                decision=decision,
                comparison=comparison,
                final_train_loss=checkpoint.metrics_snapshot.get("loss"),
                label=evaluation_id,
            )
        )

    for checkpoint in orchestrator.run_checkpoints(args.run_id):
        if any(c.checkpoint_id == checkpoint.checkpoint_id for c in candidates):
            continue
        if any(u["checkpoint_id"] == checkpoint.checkpoint_id for u in unranked):
            continue
        unranked.append(
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "evaluation_id": None,
                "reason": "never evaluated; step and training loss are not evidence of quality",
            }
        )

    if not candidates:
        _print(
            {
                "run_id": args.run_id,
                "ranked": [],
                "unranked": unranked,
                "note": (
                    "no checkpoint of this run has an evaluation decision, so none can be ranked"
                ),
            }
        )
        return 1

    ranked = rank(candidates, target_metric=args.target_metric)
    _print(
        {
            "run_id": args.run_id,
            "target_metric": args.target_metric,
            "ranked": [item.to_dict() for item in ranked],
            "unranked": unranked,
        }
    )
    return 0


# ── human review ─────────────────────────────────────────────────────


def cmd_human_package(args: argparse.Namespace) -> int:
    registry, _, suite, _, run, artifacts = _load(args)

    samples = [sample_from_dict(row) for row in read_jsonl(artifacts.samples_jsonl)]
    baseline_samples: dict[tuple[str, int], str] = {}
    candidate_samples: dict[tuple[str, int], str] = {}
    synthetic = 0
    for sample in samples:
        if sample.synthetic or not sample.artifact_ref:
            synthetic += 1
            continue
        target = (
            candidate_samples if sample.model_id == run.candidate.model_id else baseline_samples
        )
        target[(sample.case_id, sample.seed)] = sample.artifact_ref

    if not baseline_samples or not candidate_samples:
        return _fail(
            "this evaluation produced no audio to listen to"
            + (
                f" ({synthetic} synthetic samples); a synthetic run cannot be reviewed by a "
                "listener, and inventing files for one would be a fabricated review"
                if synthetic
                else ""
            )
        )

    package, mapping = build_package(
        evaluation_id=run.evaluation_id,
        cases=suite.cases,
        baseline_samples=baseline_samples,
        candidate_samples=candidate_samples,
        rubric=LightAbRubric(),
        max_pairs=args.max_pairs,
    )
    from luber_evaluation.human import write_package

    paths = write_package(artifacts.human_review_dir, package, mapping)
    registry.audit(
        HUMAN_REVIEW_REQUESTED,
        run.evaluation_id,
        pairs=len(package.pairs),
        mode=package.mode,
    )
    _print(
        {
            "evaluation_id": run.evaluation_id,
            "pairs": len(package.pairs),
            "package": str(paths["package"]),
            "mapping": str(paths["mapping"]),
            "warning": (
                "the mapping reveals which side is the candidate; never send it to a listener"
            ),
        }
    )
    return 0


def cmd_human_record(args: argparse.Namespace) -> int:
    registry, _, _, _, run, artifacts = _load(args)
    rows = read_jsonl(Path(args.responses).expanduser())
    responses = [
        HumanResponse(
            evaluation_id=run.evaluation_id,
            pair_id=str(row["pair_id"]),
            case_id=str(row.get("case_id", "")),
            question_id=str(row["question_id"]),
            choice=str(row["choice"]),
            reviewer=str(row["reviewer"]),
            rubric_version=str(row.get("rubric_version", LightAbRubric().rubric_version)),
        )
        for row in rows
    ]
    path = record_responses(artifacts.human_review_dir, responses)
    registry.audit(HUMAN_REVIEW_RECORDED, run.evaluation_id, responses=len(responses))
    evidence = _human_evidence(artifacts)
    _print({"recorded": len(responses), "path": str(path), "evidence": evidence})
    return 0


# ── verification and promotion ───────────────────────────────────────


def cmd_verify(args: argparse.Namespace) -> int:
    registry, _, suite, policy, run, artifacts = _load(args)
    orchestrator = _orchestrator(args)

    decision = None
    if registry.exists("qualifications", run.evaluation_id):
        decision = decision_from_dict(registry.read("qualifications", run.evaluation_id))

    checkpoint_status = None
    checkpoint_kind = None
    try:
        checkpoint = orchestrator.get_checkpoint(run.lineage.checkpoint_id)
        checkpoint_status = checkpoint.status
        checkpoint_kind = checkpoint.kind
    except Exception:
        checkpoint_status = "UNKNOWN"

    samples = [sample_from_dict(row) for row in read_jsonl(artifacts.samples_jsonl)]
    problems = verify_evaluation(
        artifacts_dir=artifacts.directory,
        suite=suite,
        policy=policy,
        run=run,
        decision=decision,
        samples=samples,
        repository_root=_repository(args),
        checkpoint_status=checkpoint_status,
        checkpoint_kind=checkpoint_kind,
    )
    _print(
        {
            "evaluation_id": run.evaluation_id,
            "ok": not problems,
            "problems": [problem.to_dict() for problem in problems],
        }
    )
    return 0 if not problems else 1


def cmd_promote(args: argparse.Namespace) -> int:
    registry, _, _, _, run, _ = _load(args)
    if not registry.exists("qualifications", run.evaluation_id):
        return _fail("this evaluation has no qualification decision; there is nothing to review")

    decision = decision_from_dict(registry.read("qualifications", run.evaluation_id))
    review = record_promotion_review(
        review_id=new_id(EvaluationEntityKind.REVIEW),
        candidate_id=run.lineage.candidate_id,
        evaluation_id=run.evaluation_id,
        decision=decision,
        operator_decision=args.decision,
        decided_by=args.by,
        rationale=args.rationale,
    )
    registry.save_promotion_review(review.to_dict())
    _print(
        {
            **review.to_dict(),
            "note": (
                "Approval for staging is not production activation. Serving a model to "
                "users is a deployment decision made elsewhere."
            ),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m luber_evaluation",
        description=(
            "Evaluate a trained checkpoint against an explicit baseline and decide whether "
            "it may advance to promotion review. Promotes nothing and serves nothing."
        ),
    )
    parser.add_argument("--registry", default="./training-registry", help="Phase 25 registry")
    parser.add_argument("--artifacts", help="evaluation artifact root")
    parser.add_argument("--repository", help="LUBER repository root, for the frozen benchmark")
    parser.add_argument(
        "--mode",
        default=EvaluationMode.RAW_MODEL.value,
        choices=[mode.value for mode in EvaluationMode],
        help="RAW_MODEL judges the model; DELIVERY judges what a listener receives",
    )
    sub = parser.add_subparsers(dest="group", required=True)

    # suite
    suite = sub.add_parser("suite", help="evaluation suites").add_subparsers(
        dest="action", required=True
    )
    suite.add_parser("list").set_defaults(func=cmd_suite_list)
    suite_show = suite.add_parser("show")
    suite_show.add_argument("--suite", default="P20_FULL", choices=SUITE_BUILDERS)
    suite_show.add_argument("--seed", type=int, action="append")
    suite_show.set_defaults(func=cmd_suite_show)

    # run
    run = sub.add_parser("run", help="evaluation runs").add_subparsers(dest="action", required=True)

    create = run.add_parser("create")
    create.add_argument("--candidate-id", required=True, help="a Phase 25 evaluation candidate")
    create.add_argument("--suite", default="P20_FULL", choices=SUITE_BUILDERS)
    create.add_argument("--policy", default="NEUTRAL_CONSERVATIVE", choices=sorted(POLICIES))
    create.add_argument("--seed", type=int, action="append", help="repeat for several seeds")
    create.set_defaults(func=cmd_run_create)

    start = run.add_parser("start")
    start.add_argument("--evaluation-id", required=True)
    start.add_argument(
        "--backend",
        default="synthetic",
        choices=["synthetic", "rendered", "ace-step"],
        help="synthetic produces metric values and no audio",
    )
    start.add_argument("--baseline-profile", help="synthetic backend: baseline profile JSON")
    start.add_argument("--candidate-profile", help="synthetic backend: candidate profile JSON")
    start.add_argument("--baseline-audio", help="rendered backend: baseline render directory")
    start.add_argument("--candidate-audio", help="rendered backend: candidate render directory")
    start.add_argument("--baseline-url", help="ace-step backend: server serving the baseline")
    start.add_argument("--candidate-url", help="ace-step backend: server serving the candidate")
    start.add_argument("--api-key-ref", help="a NAME, never a key")
    start.set_defaults(func=cmd_run_start)

    status = run.add_parser("status")
    status.add_argument("--evaluation-id", required=True)
    status.set_defaults(func=cmd_run_status)

    run.add_parser("list").set_defaults(func=cmd_run_list)

    # compare / qualify
    compare_parser = sub.add_parser("compare", help="recompute the baseline/candidate comparison")
    compare_parser.add_argument("--evaluation-id", required=True)
    compare_parser.set_defaults(func=cmd_compare)

    qualify = sub.add_parser("qualify", help="decide whether the candidate may advance")
    qualify.add_argument("--evaluation-id", required=True)
    qualify.add_argument("--hypothesis", help="what the experiment claimed")
    qualify.add_argument("--hypothesis-metric", help="the metric that would show it")
    qualify.add_argument(
        "--expect-decrease",
        action="store_true",
        help="the hypothesis expects the metric to fall",
    )
    qualify.set_defaults(func=cmd_qualify)

    # checkpoint
    checkpoint = sub.add_parser("checkpoint", help="checkpoint selection").add_subparsers(
        dest="action", required=True
    )
    rank_parser = checkpoint.add_parser("rank")
    rank_parser.add_argument("--run-id", required=True, help="a Phase 25 training run")
    rank_parser.add_argument("--target-metric", help="the experiment's target metric")
    rank_parser.set_defaults(func=cmd_checkpoint_rank)

    # human review
    package = sub.add_parser("human-package", help="build a blinded listening package")
    package.add_argument("--evaluation-id", required=True)
    package.add_argument("--max-pairs", type=int, help="cap the number of pairs")
    package.set_defaults(func=cmd_human_package)

    record = sub.add_parser("human-record", help="record listening responses")
    record.add_argument("--evaluation-id", required=True)
    record.add_argument("--responses", required=True, help="JSONL of responses")
    record.set_defaults(func=cmd_human_record)

    # verification / promotion
    verify = sub.add_parser("verify", help="recompute every claim an evaluation makes")
    verify.add_argument("--evaluation-id", required=True)
    verify.set_defaults(func=cmd_verify)

    promote = sub.add_parser("promote", help="record an operator promotion review")
    promote.add_argument("--evaluation-id", required=True)
    promote.add_argument(
        "--decision", required=True, choices=["APPROVE_FOR_STAGING", "REJECT", "HOLD"]
    )
    promote.add_argument("--by", required=True, help="who decided")
    promote.add_argument("--rationale", required=True, help="why")
    promote.set_defaults(func=cmd_promote)

    sub.add_parser(
        "metrics", help="the metric catalogue, including what cannot be measured"
    ).set_defaults(func=cmd_metrics)
    sub.add_parser("policies", help="available qualification policies").set_defaults(
        func=cmd_policies
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
