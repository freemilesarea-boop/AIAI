"""Policy, and the deterministic decision it produces.

Policy lives in data, not in the runner. A threshold buried in control
flow cannot be versioned, cited by a result, or argued with — and the
first time someone wants a different standard for a different experiment
they edit the engine.

Four outcomes, and the distinctions between them are the point.

``REJECTED`` means the evidence says no: a hard gate failed, or a
regression exceeded what the policy tolerates.

``BLOCKED`` means we could not look properly — missing cases, missing
metrics, a checkpoint that will not load. **Not the same as rejected.**
Calling an unevaluated candidate "rejected" would record a verdict
nobody reached, and the fix for blocked is to run the evaluation
properly rather than to abandon the checkpoint.

``HUMAN_REVIEW_REQUIRED`` means every automatic gate passed and the
experiment's own hypothesis is about something no automatic metric can
measure. This is the outcome that stops the system quietly qualifying a
candidate on technical grounds while its actual claim — better vocal
naturalness, less trot-like delivery — went unexamined.

``QUALIFIED`` means the checkpoint may advance to promotion review. It
does not mean production, and nothing here activates a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from luber_evaluation.comparison import CandidateComparison
from luber_evaluation.metrics import CATALOGUE, MeasurementMode, MetricStatus
from luber_evaluation.schemas import (
    EVALUATION_SCHEMA_VERSION,
    ComparisonVerdict,
    HumanReviewMode,
    QualificationOutcome,
    RegressionSeverity,
    digest_of,
    now,
)

QUALIFICATION_POLICY_SCHEMA_VERSION = "luber-qualification-policy/1"

#: Severity ordering, worst first.
SEVERITY_RANK: dict[str, int] = {
    RegressionSeverity.CRITICAL.value: 4,
    RegressionSeverity.MAJOR.value: 3,
    RegressionSeverity.MINOR.value: 2,
    RegressionSeverity.INFO.value: 1,
    RegressionSeverity.NONE.value: 0,
}


@dataclass(frozen=True)
class HardGate:
    """A condition that rejects a candidate outright.

    Absolute ceilings rather than comparisons: a candidate that fails
    half its generations is unusable whatever the baseline did, and a
    purely relative gate would let both sides rot together.
    """

    metric_name: str
    maximum: float | None = None
    minimum: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualificationPolicy:
    """Versioned, hashable rules for deciding.

    The default is deliberately conservative about *safety* and silent
    about *musical quality*: it enforces reliability, technical validity
    and evaluation completeness, and sets no threshold on anything this
    project cannot measure. A policy that demanded "melody quality above
    0.8" would be demanding a number nobody can produce.
    """

    policy_id: str = "NEUTRAL_CONSERVATIVE"
    policy_version: str = "1"
    description: str = (
        "Enforces technical safety, generation reliability and evaluation "
        "completeness. Sets no threshold on musical quality, because no validated "
        "automatic measure of it exists in this project."
    )

    hard_gates: tuple[HardGate, ...] = (
        HardGate(
            metric_name="generation_success_rate",
            minimum=0.90,
            reason="a candidate that cannot reliably produce audio is unusable",
        ),
        HardGate(
            metric_name="invalid_audio_rate",
            maximum=0.02,
            reason="undecodable or NaN output is a defect, not a quality trade-off",
        ),
        HardGate(
            metric_name="silent_output_rate",
            maximum=0.02,
            reason="silence is not a generation",
        ),
        HardGate(
            metric_name="early_collapse_rate",
            maximum=0.10,
            reason="a model that stops early has failed the request",
        ),
        HardGate(
            metric_name="clipping_sample_ratio",
            maximum=0.01,
            reason="systematic clipping is damage the finishing engine cannot undo",
        ),
        HardGate(
            metric_name="wrong_duration_rate",
            maximum=0.20,
            reason="a duration the model ignores is a control it does not honour",
        ),
    )

    #: The worst regression severity that still permits qualification.
    max_tolerated_regression: str = RegressionSeverity.MINOR.value

    #: Metrics whose regression rejects regardless of severity tier.
    never_regress: tuple[str, ...] = (
        "generation_success_rate",
        "invalid_audio_rate",
    )

    #: Share of the suite's cases that must have produced results.
    minimum_case_coverage: float = 0.90
    #: Share of the suite's automatic metrics that must be measurable.
    minimum_metric_coverage: float = 0.80

    #: Whether a human-required hypothesis forces HUMAN_REVIEW_REQUIRED.
    require_human_for_human_dimensions: bool = True
    human_review_mode: str = HumanReviewMode.NONE.value

    #: Human evidence thresholds. Left unset on purpose — a preference
    #: share picked before any human has listened would be arbitrary,
    #: and would then be treated as a target.
    minimum_human_preference_share: float | None = None
    minimum_human_reviewed_cases: int | None = None

    schema_version: str = QUALIFICATION_POLICY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hard_gates"] = [gate.to_dict() for gate in self.hard_gates]
        payload["never_regress"] = list(self.never_regress)
        return payload

    def digest(self) -> str:
        return digest_of(self.to_dict())


@dataclass
class GateOutcome:
    name: str
    passed: bool
    detail: str
    severity: str = RegressionSeverity.NONE.value
    #: True when the gate could not be evaluated at all.
    inconclusive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualificationDecision:
    """The verdict, with every gate that produced it."""

    evaluation_id: str
    candidate_id: str
    outcome: str
    policy_id: str
    policy_version: str
    policy_digest: str
    reasons: list[str] = field(default_factory=list)
    passed_gates: list[str] = field(default_factory=list)
    failed_gates: list[str] = field(default_factory=list)
    inconclusive_gates: list[str] = field(default_factory=list)
    gate_outcomes: list[GateOutcome] = field(default_factory=list)
    hypothesis_status: str = ""
    human_review_required_for: list[str] = field(default_factory=list)
    decided_at: str = field(default_factory=now)
    schema_version: str = EVALUATION_SCHEMA_VERSION

    @property
    def qualified(self) -> bool:
        return self.outcome == QualificationOutcome.QUALIFIED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "candidate_id": self.candidate_id,
            "outcome": self.outcome,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_digest": self.policy_digest,
            "reasons": self.reasons,
            "passed_gates": sorted(self.passed_gates),
            "failed_gates": sorted(self.failed_gates),
            "inconclusive_gates": sorted(self.inconclusive_gates),
            "gate_outcomes": [outcome.to_dict() for outcome in self.gate_outcomes],
            "hypothesis_status": self.hypothesis_status,
            "human_review_required_for": sorted(self.human_review_required_for),
            "decided_at": self.decided_at,
            "note": (
                "QUALIFIED means the checkpoint may advance to promotion review. "
                "It does not mean production, and nothing here activates a model."
            ),
        }


@dataclass
class Coverage:
    """How much of the suite actually produced evidence."""

    cases_expected: int
    cases_with_results: int
    metrics_expected: int
    metrics_measured: int

    @property
    def case_coverage(self) -> float:
        return self.cases_with_results / self.cases_expected if self.cases_expected else 0.0

    @property
    def metric_coverage(self) -> float:
        return self.metrics_measured / self.metrics_expected if self.metrics_expected else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases_expected": self.cases_expected,
            "cases_with_results": self.cases_with_results,
            "case_coverage": round(self.case_coverage, 4),
            "metrics_expected": self.metrics_expected,
            "metrics_measured": self.metrics_measured,
            "metric_coverage": round(self.metric_coverage, 4),
        }


@dataclass
class HypothesisTarget:
    """What the experiment claimed, and whether it can be checked.

    ``metric_name`` names an automatic metric when one exists. When the
    hypothesis is about something only a listener can judge, it names a
    HUMAN_REQUIRED dimension instead, and that is what forces human
    review rather than an automatic pass.
    """

    description: str
    metric_name: str | None = None
    #: Which way the metric should move for the hypothesis to hold.
    expect_improvement: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hard_gate_outcomes(
    policy: QualificationPolicy, candidate_aggregates: dict[str, Any]
) -> list[GateOutcome]:
    outcomes: list[GateOutcome] = []
    for gate in policy.hard_gates:
        aggregate = candidate_aggregates.get(gate.metric_name)
        name = f"hard:{gate.metric_name}"
        if aggregate is None or aggregate.median_value is None:
            outcomes.append(
                GateOutcome(
                    name=name,
                    passed=False,
                    inconclusive=True,
                    detail=(
                        f"{gate.metric_name} was not measured, so this safety gate "
                        "could not be evaluated"
                    ),
                )
            )
            continue
        value = aggregate.median_value
        if gate.maximum is not None and value > gate.maximum:
            outcomes.append(
                GateOutcome(
                    name=name,
                    passed=False,
                    severity=RegressionSeverity.CRITICAL.value,
                    detail=(
                        f"{gate.metric_name} is {value:.4f}, above the {gate.maximum} "
                        f"ceiling — {gate.reason}"
                    ),
                )
            )
            continue
        if gate.minimum is not None and value < gate.minimum:
            outcomes.append(
                GateOutcome(
                    name=name,
                    passed=False,
                    severity=RegressionSeverity.CRITICAL.value,
                    detail=(
                        f"{gate.metric_name} is {value:.4f}, below the {gate.minimum} "
                        f"floor — {gate.reason}"
                    ),
                )
            )
            continue
        outcomes.append(
            GateOutcome(name=name, passed=True, detail=f"{gate.metric_name} = {value:.4f}")
        )
    return outcomes


def _regression_outcomes(
    policy: QualificationPolicy, comparison: CandidateComparison
) -> list[GateOutcome]:
    outcomes: list[GateOutcome] = []
    tolerated = SEVERITY_RANK.get(policy.max_tolerated_regression, 0)

    for name in policy.never_regress:
        metric = comparison.metrics.get(name)
        gate = f"no_regression:{name}"
        if metric is None or metric.verdict == ComparisonVerdict.NOT_MEASURABLE.value:
            outcomes.append(
                GateOutcome(
                    name=gate,
                    passed=False,
                    inconclusive=True,
                    detail=f"{name} was not comparable, so its no-regression gate did not run",
                )
            )
            continue
        if metric.regressed:
            outcomes.append(
                GateOutcome(
                    name=gate,
                    passed=False,
                    severity=RegressionSeverity.CRITICAL.value,
                    detail=(
                        f"{name} regressed from {metric.baseline_value} to "
                        f"{metric.candidate_value}; this metric may never regress"
                    ),
                )
            )
        else:
            outcomes.append(GateOutcome(name=gate, passed=True, detail=f"{name} did not regress"))

    worst = comparison.worst_severity()
    if SEVERITY_RANK.get(worst, 0) > tolerated:
        offenders = sorted(
            m.metric_name
            for m in comparison.regressions
            if SEVERITY_RANK.get(m.severity, 0) > tolerated
        )
        outcomes.append(
            GateOutcome(
                name="regression_severity",
                passed=False,
                severity=worst,
                detail=(
                    f"{worst} regression in {', '.join(offenders)}; the policy tolerates "
                    f"at most {policy.max_tolerated_regression}"
                ),
            )
        )
    else:
        outcomes.append(
            GateOutcome(
                name="regression_severity",
                passed=True,
                detail=f"worst regression severity is {worst}",
            )
        )
    return outcomes


def _coverage_outcomes(policy: QualificationPolicy, coverage: Coverage) -> list[GateOutcome]:
    outcomes: list[GateOutcome] = []
    if coverage.case_coverage < policy.minimum_case_coverage:
        outcomes.append(
            GateOutcome(
                name="coverage:cases",
                passed=False,
                inconclusive=True,
                detail=(
                    f"only {coverage.cases_with_results}/{coverage.cases_expected} cases "
                    f"produced results ({coverage.case_coverage:.0%}), below the "
                    f"{policy.minimum_case_coverage:.0%} the policy requires"
                ),
            )
        )
    else:
        outcomes.append(
            GateOutcome(
                name="coverage:cases",
                passed=True,
                detail=f"{coverage.cases_with_results}/{coverage.cases_expected} cases",
            )
        )

    if coverage.metric_coverage < policy.minimum_metric_coverage:
        outcomes.append(
            GateOutcome(
                name="coverage:metrics",
                passed=False,
                inconclusive=True,
                detail=(
                    f"only {coverage.metrics_measured}/{coverage.metrics_expected} "
                    f"automatic metrics were measurable ({coverage.metric_coverage:.0%})"
                ),
            )
        )
    else:
        outcomes.append(
            GateOutcome(
                name="coverage:metrics",
                passed=True,
                detail=f"{coverage.metrics_measured}/{coverage.metrics_expected} metrics",
            )
        )
    return outcomes


def _hypothesis_outcome(
    policy: QualificationPolicy,
    target: HypothesisTarget | None,
    comparison: CandidateComparison,
    human_evidence: dict[str, Any] | None,
) -> tuple[GateOutcome, list[str]]:
    """Did the candidate provide evidence for what the experiment claimed?

    The branch that matters is the last one: a hypothesis about a
    human-required dimension cannot be satisfied by measurement, so it
    returns the dimensions needing review rather than a pass.
    """
    if target is None:
        return (
            GateOutcome(
                name="hypothesis",
                passed=True,
                detail="the experiment states no measurable target",
            ),
            [],
        )

    if target.metric_name is None:
        return (
            GateOutcome(
                name="hypothesis",
                passed=False,
                inconclusive=True,
                detail=(
                    f"the hypothesis ({target.description}) names no metric, so no "
                    "evidence for it was gathered"
                ),
            ),
            [],
        )

    spec = CATALOGUE.get(target.metric_name)
    if spec is not None and spec.mode == MeasurementMode.HUMAN_REQUIRED.value:
        if not policy.require_human_for_human_dimensions:
            return (
                GateOutcome(
                    name="hypothesis",
                    passed=True,
                    detail=(
                        f"{target.metric_name} is human-required and the policy waives "
                        "human evidence"
                    ),
                ),
                [],
            )
        provided = (human_evidence or {}).get(target.metric_name)
        if provided is None:
            return (
                GateOutcome(
                    name="hypothesis",
                    passed=False,
                    inconclusive=True,
                    detail=(
                        f"the hypothesis rests on {target.metric_name}, which no automatic "
                        f"metric can measure: {spec.unavailability_reason}"
                    ),
                ),
                [target.metric_name],
            )
        return (
            GateOutcome(
                name="hypothesis",
                passed=bool(provided),
                detail=f"human evidence recorded for {target.metric_name}",
            ),
            [],
        )

    metric = comparison.metrics.get(target.metric_name)
    if metric is None or metric.verdict == ComparisonVerdict.NOT_MEASURABLE.value:
        return (
            GateOutcome(
                name="hypothesis",
                passed=False,
                inconclusive=True,
                detail=f"{target.metric_name} was not measurable, so the hypothesis is untested",
            ),
            [],
        )
    if metric.verdict == ComparisonVerdict.INCONCLUSIVE.value:
        return (
            GateOutcome(
                name="hypothesis",
                passed=False,
                inconclusive=True,
                detail=f"{target.metric_name} moved less than the suite can resolve",
            ),
            [],
        )

    moved_right_way = metric.improved if target.expect_improvement else metric.regressed
    return (
        GateOutcome(
            name="hypothesis",
            passed=moved_right_way,
            detail=(
                f"{target.metric_name} {metric.verdict.lower()}: "
                f"{metric.baseline_value} → {metric.candidate_value}"
            ),
        ),
        [],
    )


def decide(
    *,
    evaluation_id: str,
    candidate_id: str,
    policy: QualificationPolicy,
    comparison: CandidateComparison,
    candidate_aggregates: dict[str, Any],
    coverage: Coverage,
    hypothesis: HypothesisTarget | None = None,
    human_evidence: dict[str, Any] | None = None,
    blocking_problems: list[str] | None = None,
) -> QualificationDecision:
    """The whole decision, deterministic from its inputs.

    Order matters and encodes the priorities. Integrity problems block
    before anything else is considered — a leaked benchmark makes every
    number downstream meaningless. Then coverage, because a verdict on
    partial evidence is not a verdict. Then hard safety. Then
    regressions. Only then the hypothesis.
    """
    decision = QualificationDecision(
        evaluation_id=evaluation_id,
        candidate_id=candidate_id,
        outcome=QualificationOutcome.PENDING.value,
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        policy_digest=policy.digest(),
    )

    # ── integrity ────────────────────────────────────────────────────
    if blocking_problems:
        decision.outcome = QualificationOutcome.BLOCKED.value
        decision.reasons.extend(blocking_problems)
        decision.failed_gates.append("integrity")
        decision.gate_outcomes.append(
            GateOutcome(
                name="integrity",
                passed=False,
                detail="; ".join(blocking_problems),
                severity=RegressionSeverity.CRITICAL.value,
            )
        )
        return decision

    coverage_outcomes = _coverage_outcomes(policy, coverage)
    hard_outcomes = _hard_gate_outcomes(policy, candidate_aggregates)
    regression_outcomes = _regression_outcomes(policy, comparison)
    hypothesis_outcome, human_needed = _hypothesis_outcome(
        policy, hypothesis, comparison, human_evidence
    )

    decision.gate_outcomes = [
        *coverage_outcomes,
        *hard_outcomes,
        *regression_outcomes,
        hypothesis_outcome,
    ]
    for outcome in decision.gate_outcomes:
        if outcome.passed:
            decision.passed_gates.append(outcome.name)
        elif outcome.inconclusive:
            decision.inconclusive_gates.append(outcome.name)
        else:
            decision.failed_gates.append(outcome.name)

    # ── coverage: cannot judge what was not measured ─────────────────
    coverage_failed = [o for o in coverage_outcomes if not o.passed]
    if coverage_failed:
        decision.outcome = QualificationOutcome.BLOCKED.value
        decision.reasons.extend(o.detail for o in coverage_failed)
        decision.reasons.append(
            "BLOCKED rather than REJECTED: the evidence is incomplete, which is not the "
            "same as evidence of failure"
        )
        decision.hypothesis_status = "NOT_ASSESSED"
        decision.human_review_required_for = human_needed
        return decision

    # ── hard safety, then regressions ────────────────────────────────
    hard_failed = [o for o in hard_outcomes if not o.passed and not o.inconclusive]
    regression_failed = [o for o in regression_outcomes if not o.passed and not o.inconclusive]
    if hard_failed or regression_failed:
        decision.outcome = QualificationOutcome.REJECTED.value
        decision.reasons.extend(o.detail for o in (*hard_failed, *regression_failed))
        decision.hypothesis_status = "NOT_REACHED"
        return decision

    # A safety gate nobody could evaluate blocks rather than passes.
    hard_inconclusive = [o for o in (*hard_outcomes, *regression_outcomes) if o.inconclusive]
    if hard_inconclusive:
        decision.outcome = QualificationOutcome.BLOCKED.value
        decision.reasons.extend(o.detail for o in hard_inconclusive)
        decision.reasons.append(
            "a safety gate could not be evaluated; an unmeasured gate is not a passed gate"
        )
        decision.hypothesis_status = "NOT_ASSESSED"
        decision.human_review_required_for = human_needed
        return decision

    # ── hypothesis ───────────────────────────────────────────────────
    if human_needed:
        decision.outcome = QualificationOutcome.HUMAN_REVIEW_REQUIRED.value
        decision.human_review_required_for = human_needed
        decision.hypothesis_status = "HUMAN_REQUIRED"
        decision.reasons.append(hypothesis_outcome.detail)
        decision.reasons.append(
            "every automatic gate passed; the experiment's own claim is about a "
            "dimension only a listener can judge, so this candidate is not qualified "
            "on technical grounds alone"
        )
        return decision

    if hypothesis_outcome.inconclusive:
        decision.outcome = QualificationOutcome.BLOCKED.value
        decision.hypothesis_status = "INCONCLUSIVE"
        decision.reasons.append(hypothesis_outcome.detail)
        return decision

    if not hypothesis_outcome.passed:
        decision.outcome = QualificationOutcome.REJECTED.value
        decision.hypothesis_status = "NOT_SUPPORTED"
        decision.reasons.append(hypothesis_outcome.detail)
        return decision

    decision.outcome = QualificationOutcome.QUALIFIED.value
    decision.hypothesis_status = "SUPPORTED" if hypothesis else "NONE_STATED"
    decision.reasons.append(
        "all hard safety gates passed, no intolerable regression, evaluation coverage met"
    )
    if hypothesis:
        decision.reasons.append(hypothesis_outcome.detail)
    return decision


def strict_policy() -> QualificationPolicy:
    """A tighter policy for a milestone candidate."""
    return QualificationPolicy(
        policy_id="STRICT",
        policy_version="1",
        description="tighter reliability floors and no tolerated regression",
        max_tolerated_regression=RegressionSeverity.INFO.value,
        minimum_case_coverage=1.0,
        minimum_metric_coverage=0.90,
        human_review_mode=HumanReviewMode.LIGHT_AB.value,
    )


POLICIES: dict[str, Any] = {
    "NEUTRAL_CONSERVATIVE": QualificationPolicy,
    "STRICT": strict_policy,
}


def policy_by_name(name: str) -> QualificationPolicy:
    key = name.strip().upper()
    if key not in POLICIES:
        raise KeyError(f"unknown policy {name!r}. Available: {', '.join(sorted(POLICIES))}")
    factory = POLICIES[key]
    policy: QualificationPolicy = factory()
    return policy


__all__ = [
    "Coverage",
    "GateOutcome",
    "HardGate",
    "HypothesisTarget",
    "MetricStatus",
    "QualificationDecision",
    "QualificationPolicy",
    "decide",
    "policy_by_name",
    "strict_policy",
]
