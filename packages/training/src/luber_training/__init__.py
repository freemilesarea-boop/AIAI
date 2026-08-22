"""Training orchestration, experiment registry, and checkpoint lifecycle.

Everything needed to launch training on a rented GPU *before* one is
rented, so that GPU day is provisioning and a launch rather than
architecture work.

This package performs no training. It decides what may be trained, on
what data, with what configuration, and records what happened. The
execution backend owns everything vendor-specific, which is why nothing
here imports torch, CUDA or a provider SDK.

Three properties it exists to guarantee:

*Nothing untrainable reaches a GPU.* Dataset lock, curation lock,
rights, evaluation leakage and self-generated data are hard gates
applied before a run can be queued. No override parameter exists.

*A finished run is explicable.* The plan hash names one combination of
model, data, config and code; the environment and repository revision
are captured; the bundle ties them together.

*Training completion is not model success.* A completed run yields an
EvaluationCandidate and stops. Nothing here can promote anything.
"""

from luber_training.canary import (
    CANARY_MAX_OPTIMIZER_STEPS,
    CANARY_MAX_SAMPLES,
    CanaryBoundsError,
    CanaryEnvelope,
    CanaryMode,
    CanaryResult,
    CanaryStatus,
    ace_step_canary,
    orchestration_canary,
)
from luber_training.capacity import (
    CapacityEvidence,
    CapacityReport,
    EvidenceSource,
    capacity_report,
)
from luber_training.capacity_policy import (
    CAPACITY_POLICY_VERSION,
    Applicability,
    CapacityDecision,
    CapacityPolicy,
    CapacityQualification,
    qualify,
)
from luber_training.config import (
    PRESETS,
    TrainingConfig,
    TrainingStrategy,
    preset,
)
from luber_training.config import (
    validate as validate_config,
)
from luber_training.entities import (
    Checkpoint,
    CheckpointKind,
    CheckpointStatus,
    EvaluationCandidate,
    Experiment,
    ExperimentStatus,
    FailureCode,
    ModelBaseline,
    RunStatus,
    TrainingDatasetRef,
    TrainingRun,
    TrainingWorker,
    WorkerCapabilities,
    WorkerClass,
)
from luber_training.gates import GateInputs, GateReport, run_all
from luber_training.ids import EntityKind, new_id
from luber_training.memory import (
    MemoryDomain,
    MemoryProfileIdentity,
    MemorySnapshot,
    PeakKind,
    ProfileOutcome,
    ProfileStage,
    Representativeness,
    TrainingMemoryProfile,
    classify_memory_failure,
)
from luber_training.memory_profiler import ProbeShape, ProfileRequest, profile_memory
from luber_training.orchestrator import OrchestrationError, Orchestrator, PreflightReport
from luber_training.plan import TrainingPlan
from luber_training.preflight import (
    BlockingReason,
    PreflightIntent,
    PreflightRequest,
    PreflightStatus,
    TrainingPreflightResult,
)
from luber_training.preflight import (
    evaluate as evaluate_preflight,
)
from luber_training.registry import Registry

__all__ = [
    "CANARY_MAX_OPTIMIZER_STEPS",
    "CANARY_MAX_SAMPLES",
    "CAPACITY_POLICY_VERSION",
    "PRESETS",
    "Applicability",
    "BlockingReason",
    "CanaryBoundsError",
    "CanaryEnvelope",
    "CanaryMode",
    "CanaryResult",
    "CanaryStatus",
    "CapacityDecision",
    "CapacityEvidence",
    "CapacityPolicy",
    "CapacityQualification",
    "CapacityReport",
    "Checkpoint",
    "CheckpointKind",
    "CheckpointStatus",
    "EntityKind",
    "EvaluationCandidate",
    "EvidenceSource",
    "Experiment",
    "ExperimentStatus",
    "FailureCode",
    "GateInputs",
    "GateReport",
    "MemoryDomain",
    "MemoryProfileIdentity",
    "MemorySnapshot",
    "ModelBaseline",
    "OrchestrationError",
    "Orchestrator",
    "PeakKind",
    "PreflightIntent",
    "PreflightReport",
    "PreflightRequest",
    "PreflightStatus",
    "ProbeShape",
    "ProfileOutcome",
    "ProfileRequest",
    "ProfileStage",
    "Registry",
    "Representativeness",
    "RunStatus",
    "TrainingConfig",
    "TrainingDatasetRef",
    "TrainingMemoryProfile",
    "TrainingPlan",
    "TrainingPreflightResult",
    "TrainingRun",
    "TrainingStrategy",
    "TrainingWorker",
    "WorkerCapabilities",
    "WorkerClass",
    "ace_step_canary",
    "capacity_report",
    "classify_memory_failure",
    "evaluate_preflight",
    "new_id",
    "orchestration_canary",
    "preset",
    "profile_memory",
    "qualify",
    "run_all",
    "validate_config",
]
