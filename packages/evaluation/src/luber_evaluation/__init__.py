"""Model evaluation, checkpoint qualification, and promotion review.

Sits between "training completed" and "model accepted", and exists
because those are not the same thing. A finished run produces
checkpoints; whether any of them is better than what is already in
production is a separate question that needs evidence.

Four commitments shape the package:

*Nothing is judged in isolation.* Every evaluation names an explicit,
frozen baseline. Without one there is no way to tell an improvement
from the model having always done that.

*A missing measurement is never a zero.* Metrics carry a status, and
NOT_MEASURABLE is a real outcome. Substituting 0.0 for "no ASR exists"
would manufacture a regression that never happened.

*Some dimensions have no honest automatic metric.* Melody, vocal
naturalness, trot-like delivery: nothing here can measure them, so they
are declared HUMAN_REQUIRED and a hypothesis resting on one cannot be
satisfied by measurement. That outcome is HUMAN_REVIEW_REQUIRED, and it
is what stops a candidate qualifying on technical grounds while its
actual claim went unexamined.

*QUALIFIED is not PRODUCTION.* Qualification means a checkpoint may
advance to promotion review. Nothing in this package activates a model.
"""

from luber_evaluation.comparison import (
    CandidateComparison,
    MetricComparison,
    compare,
    compare_metric,
)
from luber_evaluation.human import (
    HumanEvidence,
    HumanResponse,
    LightAbRubric,
    build_package,
    unblind,
)
from luber_evaluation.metrics import (
    CATALOGUE,
    Aggregate,
    MeasurementMode,
    MetricDefinition,
    MetricDirection,
    MetricResult,
    MetricStatus,
    aggregate,
)
from luber_evaluation.qualification import (
    Coverage,
    HypothesisTarget,
    QualificationDecision,
    QualificationPolicy,
    decide,
    policy_by_name,
)
from luber_evaluation.ranking import CheckpointCandidate, RankedCheckpoint, rank
from luber_evaluation.registry import EvaluationRegistry
from luber_evaluation.runner import (
    EvaluationRun,
    GenerationBackend,
    SideResults,
    SyntheticGenerationBackend,
    SyntheticProfile,
    execute_side,
)
from luber_evaluation.schemas import (
    EVALUATION_ENGINE_VERSION,
    EVALUATION_SCHEMA_VERSION,
    CandidateLineage,
    CaseType,
    ComparisonVerdict,
    EvaluationCase,
    EvaluationMode,
    EvaluationRunStatus,
    GenerationSpec,
    HumanReviewMode,
    ModelRef,
    PromotionReview,
    QualificationOutcome,
    RegressionSeverity,
)
from luber_evaluation.suite import EvaluationSuite, build_p20_suite, smoke_suite, verify_p20

__all__ = [
    "CATALOGUE",
    "EVALUATION_ENGINE_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "Aggregate",
    "CandidateComparison",
    "CandidateLineage",
    "CaseType",
    "CheckpointCandidate",
    "ComparisonVerdict",
    "Coverage",
    "EvaluationCase",
    "EvaluationMode",
    "EvaluationRegistry",
    "EvaluationRun",
    "EvaluationRunStatus",
    "EvaluationSuite",
    "GenerationBackend",
    "GenerationSpec",
    "HumanEvidence",
    "HumanResponse",
    "HumanReviewMode",
    "HypothesisTarget",
    "LightAbRubric",
    "MeasurementMode",
    "MetricComparison",
    "MetricDefinition",
    "MetricDirection",
    "MetricResult",
    "MetricStatus",
    "ModelRef",
    "PromotionReview",
    "QualificationDecision",
    "QualificationOutcome",
    "QualificationPolicy",
    "RankedCheckpoint",
    "RegressionSeverity",
    "SideResults",
    "SyntheticGenerationBackend",
    "SyntheticProfile",
    "aggregate",
    "build_p20_suite",
    "build_package",
    "compare",
    "compare_metric",
    "decide",
    "execute_side",
    "policy_by_name",
    "rank",
    "smoke_suite",
    "unblind",
    "verify_p20",
]
