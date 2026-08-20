"""Reports, verification, model-card input, and promotion review.

The report is written for someone deciding whether to spend money and
attention on a checkpoint, so it leads with what would stop them and
gives equal space to what could not be measured. A report that listed
only its findings would read as a complete picture and is not one.

Verification exists because an evaluation is a chain of claims —
this suite, this policy, this checkpoint, these samples — and any link
can rot. It recomputes rather than re-reads: a check that compared a
file against itself would pass after the thing it describes had changed.

Nothing here promotes a model. `PromotionReview` records an operator's
decision to move a checkpoint toward staging, and stops there.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luber_evaluation.comparison import CandidateComparison
from luber_evaluation.metrics import CATALOGUE, Aggregate, MeasurementMode, MetricStatus
from luber_evaluation.qualification import QualificationDecision, QualificationPolicy
from luber_evaluation.runner import EvaluationRun, SideResults
from luber_evaluation.schemas import (
    EVALUATION_SCHEMA_VERSION,
    ComparisonVerdict,
    PromotionDecisionValue,
    PromotionReview,
    QualificationOutcome,
    SampleProvenance,
    now,
)
from luber_evaluation.suite import EvaluationSuite, verify_p20


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── model card input ─────────────────────────────────────────────────


@dataclass
class ModelCardInput:
    """Facts a future model card can be built from. Not marketing.

    Deliberately includes ``known_regressions`` and ``limitations``
    alongside improvements. A card assembled only from what improved
    would be an advertisement, and the point of recording this at
    qualification time is that the limitations are known then and get
    forgotten later.
    """

    candidate_id: str
    checkpoint_id: str
    base_model_id: str
    base_model_upstream_commit: str
    training_run_id: str
    experiment_id: str
    experiment_hypothesis: str
    dataset_id: str
    dataset_lock_sha256: str
    curation_id: str
    curation_lock_sha256: str
    training_config_sha256: str
    evaluation_id: str
    suite_id: str
    suite_version: str
    suite_digest: str
    policy_id: str
    qualification_outcome: str
    known_improvements: list[str] = field(default_factory=list)
    known_regressions: list[str] = field(default_factory=list)
    not_measurable: list[str] = field(default_factory=list)
    human_evidence_status: str = "NONE"
    limitations: list[str] = field(default_factory=list)
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "checkpoint_id": self.checkpoint_id,
            "base_model": {
                "model_id": self.base_model_id,
                "upstream_commit": self.base_model_upstream_commit,
            },
            "training_lineage": {
                "run_id": self.training_run_id,
                "experiment_id": self.experiment_id,
                "hypothesis": self.experiment_hypothesis,
                "training_config_sha256": self.training_config_sha256,
            },
            "data_lineage": {
                "dataset_id": self.dataset_id,
                "dataset_lock_sha256": self.dataset_lock_sha256,
                "curation_id": self.curation_id,
                "curation_lock_sha256": self.curation_lock_sha256,
            },
            "evaluation": {
                "evaluation_id": self.evaluation_id,
                "suite_id": self.suite_id,
                "suite_version": self.suite_version,
                "suite_digest": self.suite_digest,
                "policy_id": self.policy_id,
                "outcome": self.qualification_outcome,
            },
            "known_improvements": sorted(self.known_improvements),
            "known_regressions": sorted(self.known_regressions),
            "not_measurable": sorted(self.not_measurable),
            "human_evidence_status": self.human_evidence_status,
            "limitations": self.limitations,
        }


DEFAULT_LIMITATIONS = [
    "Musical quality — melody, hook strength, emotional impact, instrument realism, "
    "vocal naturalness — has no validated automatic measure in this project and was "
    "not assessed automatically.",
    "Lyric completeness was not measured: no validated speech recogniser is configured.",
    "Vocal/instrumental adherence was not measured: no validated detector exists.",
    "Trot-like delivery cannot be detected automatically and remains human-judged.",
    "Evaluation compares raw model output; delivery-path behaviour is measured "
    "separately and does not qualify the model.",
]


def build_model_card_input(
    *,
    run: EvaluationRun,
    comparison: CandidateComparison,
    decision: QualificationDecision,
    policy: QualificationPolicy,
    experiment_hypothesis: str = "",
    human_evidence_status: str = "NONE",
) -> ModelCardInput:
    not_measurable = sorted(
        name
        for name, metric in comparison.metrics.items()
        if metric.verdict == ComparisonVerdict.NOT_MEASURABLE.value
    )
    return ModelCardInput(
        candidate_id=run.lineage.candidate_id,
        checkpoint_id=run.lineage.checkpoint_id,
        base_model_id=run.baseline.model_id,
        base_model_upstream_commit=run.baseline.upstream_commit,
        training_run_id=run.lineage.run_id,
        experiment_id=run.lineage.experiment_id,
        experiment_hypothesis=experiment_hypothesis,
        dataset_id=run.lineage.dataset_id,
        dataset_lock_sha256=run.lineage.dataset_lock_sha256,
        curation_id=run.lineage.curation_id,
        curation_lock_sha256=run.lineage.curation_lock_sha256,
        training_config_sha256=run.lineage.training_config_sha256,
        evaluation_id=run.evaluation_id,
        suite_id=run.suite.suite_id,
        suite_version=run.suite.suite_version,
        suite_digest=run.suite_digest,
        policy_id=policy.policy_id,
        qualification_outcome=decision.outcome,
        known_improvements=[m.metric_name for m in comparison.improvements],
        known_regressions=[f"{m.metric_name} ({m.severity})" for m in comparison.regressions],
        not_measurable=not_measurable,
        human_evidence_status=human_evidence_status,
        limitations=list(DEFAULT_LIMITATIONS),
    )


# ── human report ─────────────────────────────────────────────────────


def render_report(
    *,
    run: EvaluationRun,
    comparison: CandidateComparison,
    decision: QualificationDecision,
    policy: QualificationPolicy,
    baseline_aggregates: dict[str, Aggregate],
    candidate_aggregates: dict[str, Aggregate],
) -> str:
    lines: list[str] = []
    lines.append("# Evaluation report")
    lines.append("")
    lines.append(f"- Evaluation: `{run.evaluation_id}`")
    lines.append(f"- Mode: **{run.mode}**")
    lines.append(
        f"- Suite: `{run.suite.suite_id}` v{run.suite.suite_version} (`{run.suite_digest[:16]}…`)"
    )
    lines.append(f"- Policy: `{policy.policy_id}` v{policy.policy_version}")
    lines.append(f"- Baseline: `{run.baseline.model_id}` @ `{run.baseline.upstream_commit[:12]}`")
    lines.append(f"- Candidate checkpoint: `{run.lineage.checkpoint_id}`")
    lines.append(f"- Seeds: {list(run.seeds)}")
    lines.append("")

    lines.append(f"## Outcome: **{decision.outcome}**")
    lines.append("")
    for reason in decision.reasons:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append(
        "_QUALIFIED means the checkpoint may advance to promotion review. It does not "
        "mean production, and nothing in this pipeline activates a model._"
    )
    lines.append("")

    if decision.failed_gates:
        lines.append("## Failed gates")
        lines.append("")
        for outcome in decision.gate_outcomes:
            if not outcome.passed and not outcome.inconclusive:
                lines.append(f"- **{outcome.name}** ({outcome.severity}) — {outcome.detail}")
        lines.append("")

    if decision.inconclusive_gates:
        lines.append("## Gates that could not be evaluated")
        lines.append("")
        for outcome in decision.gate_outcomes:
            if outcome.inconclusive:
                lines.append(f"- **{outcome.name}** — {outcome.detail}")
        lines.append("")
        lines.append("_An unmeasured gate is not a passed gate._")
        lines.append("")

    lines.append("## What moved")
    lines.append("")
    pareto = comparison.pareto_summary()
    judged = [
        m
        for m in comparison.metrics.values()
        if m.verdict in (ComparisonVerdict.IMPROVED.value, ComparisonVerdict.REGRESSED.value)
    ]
    if judged:
        lines.append("| metric | baseline | candidate | delta | verdict | severity |")
        lines.append("|---|---|---|---|---|---|")
        for metric in sorted(judged, key=lambda m: m.metric_name):
            lines.append(
                f"| {metric.metric_name} | {metric.baseline_value} | {metric.candidate_value} | "
                f"{metric.absolute_delta:+.4f} | {metric.verdict} | {metric.severity} |"
            )
    else:
        lines.append("_Nothing moved beyond what the suite can resolve._")
    lines.append("")

    if pareto["inconclusive"]:
        lines.append(
            f"Inconclusive (movement inside the suite's resolution): "
            f"{', '.join(pareto['inconclusive'])}"
        )
        lines.append("")

    lines.append("## What could not be measured, and why")
    lines.append("")
    unmeasured = False
    for name in sorted(set(comparison.metrics) | set(candidate_aggregates)):
        spec = CATALOGUE.get(name)
        if spec is None:
            continue
        if spec.mode in (MeasurementMode.HUMAN_REQUIRED.value, MeasurementMode.NOT_AVAILABLE.value):
            unmeasured = True
            lines.append(f"- **{name}** ({spec.mode}) — {spec.unavailability_reason}")
    if not unmeasured:
        lines.append("_Every metric this suite carries was measurable._")
    lines.append("")

    lines.append("## Advisory composite")
    lines.append("")
    advisory = comparison.advisory_score()
    lines.append(f"```json\n{json.dumps(advisory, indent=2, sort_keys=True)}\n```")
    lines.append("")
    lines.append(
        "_Advisory only. No gate reads this number: a scalar cannot express "
        "'reliability improved and the stereo image collapsed'._"
    )
    lines.append("")

    lines.append("## Training context")
    lines.append("")
    loss = candidate_aggregates.get("final_train_loss")
    if loss is not None and loss.median_value is not None:
        lines.append(f"- Final training loss: {loss.median_value}")
    lines.append(
        "- Training loss is recorded as context. **A lower loss is not a better model**, "
        "and no qualification decision rests on it."
    )
    lines.append("")
    return "\n".join(lines)


# ── verification ─────────────────────────────────────────────────────


@dataclass
class VerificationProblem:
    check: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.check, "detail": self.detail}


def verify_evaluation(
    *,
    artifacts_dir: Path,
    suite: EvaluationSuite,
    policy: QualificationPolicy,
    run: EvaluationRun,
    decision: QualificationDecision | None,
    samples: list[SampleProvenance],
    repository_root: Path | None = None,
    checkpoint_status: str | None = None,
    checkpoint_kind: str | None = None,
) -> list[VerificationProblem]:
    """Recompute every claim an evaluation record makes.

    Recomputing rather than re-reading is the point throughout: the
    suite digest is derived from the suite object, not copied out of
    the file that claims it.
    """
    problems: list[VerificationProblem] = []

    if suite.digest() != run.suite_digest:
        problems.append(
            VerificationProblem(
                check="suite_digest",
                detail="the suite has changed since this evaluation ran",
            )
        )
    if policy.digest() != run.policy_digest:
        problems.append(
            VerificationProblem(
                check="policy_digest",
                detail="the qualification policy has changed since this evaluation ran",
            )
        )

    if repository_root is not None and suite.benchmark is not None:
        try:
            verify_p20(repository_root, expected=suite.benchmark.sha256)
        except Exception as exc:
            problems.append(VerificationProblem(check="benchmark_integrity", detail=str(exc)))

    if not run.baseline.model_id:
        problems.append(
            VerificationProblem(
                check="baseline_identity", detail="the evaluation records no baseline model"
            )
        )
    if not run.lineage.checkpoint_id:
        problems.append(
            VerificationProblem(
                check="candidate_identity", detail="the evaluation records no candidate checkpoint"
            )
        )

    if checkpoint_kind == "MOCK":
        problems.append(
            VerificationProblem(
                check="candidate_readiness",
                detail="the candidate references a MOCK artifact, holding no trained weights",
            )
        )
    if checkpoint_status is not None and checkpoint_status != "READY":
        problems.append(
            VerificationProblem(
                check="candidate_readiness",
                detail=f"the candidate checkpoint is {checkpoint_status}, not READY",
            )
        )

    # Sample digests: audio a human is about to judge must be the audio
    # the record describes.
    for sample in samples:
        if sample.synthetic:
            continue
        if sample.artifact_ref is None:
            continue
        path = Path(sample.artifact_ref)
        if not path.is_file():
            problems.append(
                VerificationProblem(
                    check="sample_present",
                    detail=f"{sample.case_id} seed {sample.seed}: artifact is missing",
                )
            )
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        if sample.raw_sha256 and digest.hexdigest() != sample.raw_sha256:
            problems.append(
                VerificationProblem(
                    check="sample_digest",
                    detail=(
                        f"{sample.case_id} seed {sample.seed}: the audio on disk is not the "
                        "audio this evaluation recorded"
                    ),
                )
            )

    if decision is not None:
        if decision.policy_digest != policy.digest():
            problems.append(
                VerificationProblem(
                    check="qualification_consistency",
                    detail="the decision cites a different policy than the one supplied",
                )
            )
        if decision.evaluation_id != run.evaluation_id:
            problems.append(
                VerificationProblem(
                    check="qualification_consistency",
                    detail="the decision belongs to a different evaluation",
                )
            )

    expected = {
        artifacts_dir / "evaluation.json",
        artifacts_dir / "suite.json",
        artifacts_dir / "policy.json",
    }
    for path in sorted(expected):
        if not path.is_file():
            problems.append(
                VerificationProblem(check="artifact_completeness", detail=f"{path.name} is missing")
            )
    return problems


# ── promotion review ─────────────────────────────────────────────────


class PromotionError(RuntimeError):
    """Raised when a promotion review is not permissible."""


def record_promotion_review(
    *,
    review_id: str,
    candidate_id: str,
    evaluation_id: str,
    decision: QualificationDecision,
    operator_decision: str,
    decided_by: str,
    rationale: str,
) -> PromotionReview:
    """Record an operator's decision about a qualified checkpoint.

    A checkpoint that did not qualify cannot be approved for staging.
    The review exists to add operator judgement *on top of* evidence,
    not to substitute for it — approving a rejected candidate would make
    every gate advisory.
    """
    permitted = {member.value for member in PromotionDecisionValue}
    if operator_decision not in permitted:
        raise PromotionError(
            f"{operator_decision!r} is not a promotion decision. Permitted: "
            f"{', '.join(sorted(permitted))}"
        )
    if (
        operator_decision == PromotionDecisionValue.APPROVE_FOR_STAGING.value
        and decision.outcome != QualificationOutcome.QUALIFIED.value
    ):
        raise PromotionError(
            f"the candidate is {decision.outcome}, not QUALIFIED; it cannot be approved "
            "for staging. Evidence first, judgement second."
        )
    if not rationale.strip():
        raise PromotionError("a promotion review must record why")

    return PromotionReview(
        review_id=review_id,
        candidate_id=candidate_id,
        evaluation_id=evaluation_id,
        qualification_outcome=decision.outcome,
        decision=operator_decision,
        decided_by=decided_by,
        rationale=rationale,
        decided_at=now(),
    )


def metric_status_summary(aggregates: dict[str, Aggregate]) -> dict[str, int]:
    counts = {status.value: 0 for status in MetricStatus}
    for summary in aggregates.values():
        counts[summary.status] = counts.get(summary.status, 0) + 1
    return counts


def side_sample_rows(side: SideResults) -> list[dict[str, Any]]:
    return [sample.to_dict() for sample in side.samples]
