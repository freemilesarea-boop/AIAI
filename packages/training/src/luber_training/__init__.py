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
from luber_training.orchestrator import OrchestrationError, Orchestrator, PreflightReport
from luber_training.plan import TrainingPlan
from luber_training.registry import Registry

__all__ = [
    "PRESETS",
    "Checkpoint",
    "CheckpointKind",
    "CheckpointStatus",
    "EntityKind",
    "EvaluationCandidate",
    "Experiment",
    "ExperimentStatus",
    "FailureCode",
    "GateInputs",
    "GateReport",
    "ModelBaseline",
    "OrchestrationError",
    "Orchestrator",
    "PreflightReport",
    "Registry",
    "RunStatus",
    "TrainingConfig",
    "TrainingDatasetRef",
    "TrainingPlan",
    "TrainingRun",
    "TrainingStrategy",
    "TrainingWorker",
    "WorkerCapabilities",
    "WorkerClass",
    "new_id",
    "preset",
    "run_all",
    "validate_config",
]
