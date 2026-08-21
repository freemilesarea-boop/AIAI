"""Technical quality control for generated audio.

What this package does: measure a candidate, decide whether it can be
delivered, decide whether another attempt is worth an inference, and
rank the ones that survive.

What it does not do, and will not: judge whether a song is good. Every
name here is chosen to keep that line visible —
``technical_selection_score``, ``CONTROL_VOCAL_UNKNOWN``,
``NOT_MEASURABLE``. Where this repository has no validated detector, the
answer is that it has none, and the reason is recorded rather than
replaced with a heuristic.
"""

from luber_inference_qc.candidate import (
    CallAttribution,
    CandidateGeneration,
    CandidateStatus,
    SelectionStatus,
)
from luber_inference_qc.checks import RequestExpectation, run_checks
from luber_inference_qc.collapse import CollapseMeasurement, measure_collapse
from luber_inference_qc.detectors import (
    MINIMUM_VOCAL_CONFIDENCE,
    NullVocalDetector,
    VocalAssessment,
    VocalPresence,
    VocalPresenceDetector,
)
from luber_inference_qc.engine import judge
from luber_inference_qc.findings import (
    NON_RETRYABLE,
    RETRYABLE,
    Finding,
    QCFinding,
    Severity,
)
from luber_inference_qc.identity import derive_seed, request_digest
from luber_inference_qc.measurement import (
    CandidateMeasurement,
    MeasurementCache,
    MeasurementError,
    measure,
)
from luber_inference_qc.planner import AdaptiveRetryPlanner, RetryDecision, RetryPlan
from luber_inference_qc.policy import (
    Budget,
    CandidatePolicy,
    PolicyProfile,
    profile,
)
from luber_inference_qc.scoring import ScoreBreakdown, score
from luber_inference_qc.selector import (
    EligibilityVerdict,
    Selection,
    assess_eligibility,
    select,
)
from luber_inference_qc.trace import Outcome, QCTrace, summarise
from luber_inference_qc.versions import (
    CANDIDATE_SELECTION_VERSION,
    QC_ENGINE_VERSION,
    QC_SCHEMA_VERSION,
    RETRY_POLICY_VERSION,
    version_block,
)

__all__ = [
    "CANDIDATE_SELECTION_VERSION",
    "MINIMUM_VOCAL_CONFIDENCE",
    "NON_RETRYABLE",
    "QC_ENGINE_VERSION",
    "QC_SCHEMA_VERSION",
    "RETRYABLE",
    "RETRY_POLICY_VERSION",
    "AdaptiveRetryPlanner",
    "Budget",
    "CallAttribution",
    "CandidateGeneration",
    "CandidateMeasurement",
    "CandidatePolicy",
    "CandidateStatus",
    "CollapseMeasurement",
    "EligibilityVerdict",
    "Finding",
    "MeasurementCache",
    "MeasurementError",
    "NullVocalDetector",
    "Outcome",
    "PolicyProfile",
    "QCFinding",
    "QCTrace",
    "RequestExpectation",
    "RetryDecision",
    "RetryPlan",
    "ScoreBreakdown",
    "Selection",
    "SelectionStatus",
    "Severity",
    "VocalAssessment",
    "VocalPresence",
    "VocalPresenceDetector",
    "assess_eligibility",
    "derive_seed",
    "judge",
    "measure",
    "measure_collapse",
    "profile",
    "request_digest",
    "run_checks",
    "score",
    "select",
    "summarise",
    "version_block",
]
