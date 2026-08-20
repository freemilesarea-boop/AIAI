"""Turning a Phase 25 candidate into the two sides of a comparison.

Phase 25 records what was trained; this resolves that record into a
baseline, a candidate, and the lineage that connects them to the data
they came from.

Two rules are enforced here rather than assumed.

*A placeholder cannot be evaluated.* Phase 25 already refuses to make a
MOCK checkpoint into a candidate. The check is repeated because the
consequence of it failing is an evaluation report full of confident
numbers about weights that were never trained — and a registry file
edited by hand would slip past the original guard.

*The two sides must have different identities.* The candidate's
`model_id` is its candidate id, not the base model's. A backend refuses
to generate for a model it does not serve, and that refusal only works
if baseline and candidate are distinguishable. Giving both sides the
base model's id would make a single misconfigured server able to answer
for both, and the comparison would silently be a model against itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from luber_evaluation.schemas import CandidateLineage, ModelRef
from luber_evaluation.serde import build
from luber_training.entities import (
    CandidateStatus,
    Checkpoint,
    CheckpointKind,
    CheckpointStatus,
    EvaluationCandidate,
)


class LineageError(RuntimeError):
    """Raised when a candidate cannot honestly be evaluated."""


@dataclass
class ResolvedCandidate:
    """Everything an evaluation needs about what it is judging."""

    baseline: ModelRef
    candidate: ModelRef
    lineage: CandidateLineage
    checkpoint: Checkpoint
    candidate_record: EvaluationCandidate
    experiment_hypothesis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "lineage": self.lineage.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "experiment_hypothesis": self.experiment_hypothesis,
        }


def resolve_candidate(orchestrator: Any, candidate_id: str) -> ResolvedCandidate:
    """Resolve a candidate through checkpoint, run, experiment and model.

    Everything is read from the training registry. Nothing is inferred:
    a missing dataset lock stays empty rather than being reconstructed
    from whatever build directory happens to be lying around, because a
    lineage that names the wrong dataset is worse than one that admits
    it does not know.
    """
    registry = orchestrator.registry
    record = registry.read("candidates", candidate_id)
    candidate_record = build(EvaluationCandidate, record)

    checkpoint = orchestrator.get_checkpoint(candidate_record.checkpoint_id)
    if checkpoint.kind == CheckpointKind.MOCK.value:
        raise LineageError(
            f"checkpoint {checkpoint.checkpoint_id} is a MOCK artifact and holds no trained "
            "weights; evaluating it would produce numbers about nothing"
        )
    if checkpoint.status != CheckpointStatus.READY.value:
        raise LineageError(
            f"checkpoint {checkpoint.checkpoint_id} is {checkpoint.status}, not READY; "
            "a checkpoint still being written cannot be evaluated"
        )

    run = orchestrator.get_run(checkpoint.run_id)
    experiment = orchestrator.get_experiment(run.experiment_id)
    baseline_model = orchestrator.get_baseline(run.base_model_id)

    dataset = run.dataset_ref
    lineage = CandidateLineage(
        candidate_id=candidate_record.candidate_id,
        checkpoint_id=checkpoint.checkpoint_id,
        run_id=run.run_id,
        experiment_id=experiment.experiment_id,
        base_model_id=baseline_model.model_id,
        dataset_id=dataset.dataset_id,
        dataset_lock_sha256=dataset.dataset_lock_sha256,
        curation_id=dataset.curation_id,
        curation_lock_sha256=dataset.curation_lock_sha256,
        training_config_sha256=run.config_sha256,
        training_plan_sha256=run.training_plan_sha256 or "",
    )

    baseline_ref = ModelRef(
        model_id=baseline_model.model_id,
        upstream_commit=baseline_model.upstream_commit,
        checkpoint_id=None,
        checkpoint_sha256=baseline_model.checkpoint_sha256,
        label="baseline",
    )
    candidate_ref = ModelRef(
        # The candidate id, not the base model id: the two sides must be
        # distinguishable for the backend's wrong-model refusal to work.
        model_id=candidate_record.candidate_id,
        upstream_commit=baseline_model.upstream_commit,
        checkpoint_id=checkpoint.checkpoint_id,
        checkpoint_sha256=checkpoint.sha256,
        label="candidate",
    )

    return ResolvedCandidate(
        baseline=baseline_ref,
        candidate=candidate_ref,
        lineage=lineage,
        checkpoint=checkpoint,
        candidate_record=candidate_record,
        experiment_hypothesis=getattr(experiment, "hypothesis", ""),
    )


def mark_evaluating(orchestrator: Any, candidate_id: str) -> None:
    """Record that a candidate is under evaluation.

    Best-effort and non-fatal: the evaluation's own record is the
    authority on what happened, and failing a run because a status
    field could not be updated would lose real results over bookkeeping.
    """
    try:
        record = orchestrator.registry.read("candidates", candidate_id)
        if record.get("status") == CandidateStatus.PENDING_EVALUATION.value:
            record["status"] = CandidateStatus.EVALUATING.value
            orchestrator.registry.write("candidates", candidate_id, record, overwrite=True)
    except Exception:
        return


__all__ = ["LineageError", "ResolvedCandidate", "mark_evaluating", "resolve_candidate"]
