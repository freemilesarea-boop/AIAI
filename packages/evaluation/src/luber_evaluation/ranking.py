"""Ranking several checkpoints from one run, without a magic number.

A training run produces many READY checkpoints. Picking one is a real
decision, and the two obvious shortcuts are both wrong:

*Latest step* assumes training monotonically improves the model, which
is exactly what nobody has established.

*Lowest training loss* assumes loss tracks musical quality. It does not.
A loss curve measures how well the model predicts its training
distribution; it says nothing about whether the singing sounds like a
person. Training loss is available here as context and can never be the
sole basis for a ranking — enforced, not merely advised.

Ranking is lexicographic on things that actually matter, in an order
that cannot be gamed by one metric: safety first, then regressions, then
the experiment's own target. A candidate that improves the target metric
while breaking reliability does not win, because the first comparison it
loses ends the matter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from luber_evaluation.comparison import CandidateComparison
from luber_evaluation.qualification import SEVERITY_RANK, QualificationDecision
from luber_evaluation.schemas import QualificationOutcome, RegressionSeverity

#: Outcome preference, best first. BLOCKED sits above REJECTED: not
#: knowing is better than knowing it failed, and a blocked checkpoint
#: can still be evaluated properly.
OUTCOME_RANK: dict[str, int] = {
    QualificationOutcome.QUALIFIED.value: 0,
    QualificationOutcome.HUMAN_REVIEW_REQUIRED.value: 1,
    QualificationOutcome.PENDING.value: 2,
    QualificationOutcome.BLOCKED.value: 3,
    QualificationOutcome.REJECTED.value: 4,
}


@dataclass
class CheckpointCandidate:
    """One checkpoint, with everything known about it."""

    checkpoint_id: str
    step: int | None = None
    epoch: int | None = None
    decision: QualificationDecision | None = None
    comparison: CandidateComparison | None = None
    #: Context only. Never the deciding factor.
    final_train_loss: float | None = None
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "step": self.step,
            "epoch": self.epoch,
            "label": self.label,
            "outcome": self.decision.outcome if self.decision else None,
            "final_train_loss": self.final_train_loss,
        }


@dataclass
class RankedCheckpoint:
    checkpoint_id: str
    rank: int
    outcome: str
    worst_regression: str
    target_verdict: str
    improvements: int
    regressions: int
    final_train_loss: float | None
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RankingError(ValueError):
    """Raised when a ranking would rest on something it must not."""


def rank(
    candidates: list[CheckpointCandidate],
    *,
    target_metric: str | None = None,
) -> list[RankedCheckpoint]:
    """Order checkpoints best-first on evidence, not on loss or recency.

    The sort key is deliberately lexicographic rather than weighted. A
    weighted score would let a large target-metric gain buy off a
    reliability regression, which is precisely the trade this project
    must not make silently.
    """
    if not candidates:
        return []

    if all(candidate.decision is None for candidate in candidates):
        raise RankingError(
            "no checkpoint has an evaluation decision; ranking on training loss or "
            "step alone is not permitted, because neither measures model quality"
        )

    scored: list[tuple[tuple[Any, ...], CheckpointCandidate, list[str]]] = []
    for candidate in candidates:
        rationale: list[str] = []

        outcome = (
            candidate.decision.outcome if candidate.decision else QualificationOutcome.PENDING.value
        )
        outcome_rank = OUTCOME_RANK.get(outcome, 5)
        rationale.append(f"qualification outcome {outcome}")

        worst = (
            candidate.comparison.worst_severity()
            if candidate.comparison
            else RegressionSeverity.NONE.value
        )
        severity_rank = SEVERITY_RANK.get(worst, 0)
        if severity_rank:
            rationale.append(f"worst regression {worst}")

        target_rank = 1
        target_verdict = "NOT_EVALUATED"
        if target_metric and candidate.comparison:
            metric = candidate.comparison.metrics.get(target_metric)
            if metric is not None:
                target_verdict = metric.verdict
                target_rank = 0 if metric.improved else 1 if not metric.regressed else 2
                rationale.append(f"target {target_metric} {metric.verdict}")

        improvements = len(candidate.comparison.improvements) if candidate.comparison else 0
        regressions = len(candidate.comparison.regressions) if candidate.comparison else 0

        # Loss enters only as the final tie-break, after outcome,
        # safety, target and the improvement balance have all tied. It
        # can separate two otherwise indistinguishable checkpoints and
        # can never overturn any of them.
        loss_key = (
            candidate.final_train_loss if candidate.final_train_loss is not None else float("inf")
        )
        if candidate.final_train_loss is not None:
            rationale.append(
                f"training loss {candidate.final_train_loss} used only as a final tie-break; "
                "it does not measure music quality"
            )

        key = (
            outcome_rank,
            severity_rank,
            target_rank,
            regressions,
            -improvements,
            loss_key,
            candidate.checkpoint_id,
        )
        scored.append((key, candidate, rationale))

    scored.sort(key=lambda item: item[0])
    ranked: list[RankedCheckpoint] = []
    for position, (_, candidate, rationale) in enumerate(scored, start=1):
        comparison = candidate.comparison
        target_verdict = "NOT_EVALUATED"
        if target_metric and comparison:
            metric = comparison.metrics.get(target_metric)
            if metric is not None:
                target_verdict = metric.verdict
        ranked.append(
            RankedCheckpoint(
                checkpoint_id=candidate.checkpoint_id,
                rank=position,
                outcome=(
                    candidate.decision.outcome
                    if candidate.decision
                    else QualificationOutcome.PENDING.value
                ),
                worst_regression=(
                    comparison.worst_severity() if comparison else RegressionSeverity.NONE.value
                ),
                target_verdict=target_verdict,
                improvements=len(comparison.improvements) if comparison else 0,
                regressions=len(comparison.regressions) if comparison else 0,
                final_train_loss=candidate.final_train_loss,
                rationale=rationale,
            )
        )
    return ranked


def pareto_front(candidates: list[CheckpointCandidate], metrics: list[str]) -> list[str]:
    """Checkpoints no other checkpoint dominates on every named metric.

    Offered instead of a composite where several checkpoints trade off
    genuinely — one better on reliability, another on the target. A
    scalar would declare a winner; the front says honestly that the
    choice has not been made by the data.
    """
    values: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        if candidate.comparison is None:
            continue
        row: dict[str, float] = {}
        for name in metrics:
            metric = candidate.comparison.metrics.get(name)
            if metric is None or metric.candidate_value is None:
                continue
            from luber_evaluation.metrics import CATALOGUE, MetricDirection

            spec = CATALOGUE.get(name)
            value = metric.candidate_value
            # Normalise so higher is always better within this function.
            if spec and spec.direction == MetricDirection.LOWER_BETTER.value:
                value = -value
            row[name] = value
        if row:
            values[candidate.checkpoint_id] = row

    front: list[str] = []
    for identifier, row in values.items():
        dominated = False
        for other, other_row in values.items():
            if other == identifier:
                continue
            shared = set(row) & set(other_row)
            if not shared:
                continue
            if all(other_row[name] >= row[name] for name in shared) and any(
                other_row[name] > row[name] for name in shared
            ):
                dominated = True
                break
        if not dominated:
            front.append(identifier)
    return sorted(front)
