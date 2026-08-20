"""The durable entities, and the state machines that govern them.

Five things worth stating up front.

**Experiment and TrainingRun are not the same thing.** An experiment is
a hypothesis — "reducing the trot prior improves modern Korean vocal
phrasing" — and it outlives the runs that test it. A run is one concrete
execution with one config on one worker. Conflating them means a failed
execution reads as a disproved hypothesis, and a retry reads as
rewriting history.

**A retry is a new run.** Failed runs are never overwritten or reused.
Lineage is recorded (`parent_run_id`), so the third attempt can be traced
back to the first without the first being edited.

**Training completion is not model success.** A `COMPLETED` run produces
checkpoints; a checkpoint becomes an `EvaluationCandidate` and stops
there. Nothing in this module can promote anything to production, and
the promotion states exist only so a later phase has somewhere to put a
decision it has actually earned.

**Secrets are referenced, never stored.** Every entity that might touch
credentials holds a *name* — `ssh_key_ref`, `credential_ref` — that a
future backend resolves out of band. No entity, plan, log or checkpoint
record ever carries a secret value.

**States that do not exist are not modelled.** There is no `PAUSED`,
because the installed trainer offers no pause and a state nothing can
enter is a lie in a diagram.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from luber_training.config import TrainingConfig

ENTITY_SCHEMA_VERSION = "luber-training/1"


def now() -> str:
    return datetime.now(UTC).isoformat()


# ── model baseline ───────────────────────────────────────────────────


class TrainingStrategySupport(StrEnum):
    LORA = "LORA"
    LOKR = "LOKR"
    FULL_FINETUNE = "FULL_FINETUNE"


class ModelStage(StrEnum):
    """Where a model sits in its lifecycle.

    ``PRODUCTION`` exists in the vocabulary and nothing in Phase 25 can
    move anything into it. The current ACE-Step baseline holds it, and
    only evidence from a real evaluation may change that.
    """

    BASELINE = "BASELINE"
    EXPERIMENT = "EXPERIMENT"
    CANDIDATE = "CANDIDATE"
    ACCEPTED = "ACCEPTED"
    PRODUCTION = "PRODUCTION"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"


@dataclass
class ModelBaseline:
    """An immutable model identity a run can be built on.

    ``checkpoint_sha256`` is optional and usually absent. Hashing a
    multi-gigabyte weight tree on every registry read is not reasonable,
    so identity rests on the upstream commit and the declared variant —
    which is reproducible from a public source — and the limitation is
    recorded in ``identity_basis`` rather than hidden behind a null.
    """

    model_id: str
    provider: str
    model_family: str
    model_name: str
    model_version: str
    upstream_commit: str
    architecture: str
    training_strategy_support: list[str] = field(default_factory=list)
    checkpoint_reference: str | None = None
    checkpoint_sha256: str | None = None
    #: How this model's identity is established, in words.
    identity_basis: str = "upstream commit and declared model variant"
    stage: str = ModelStage.BASELINE.value
    created_at: str = field(default_factory=now)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = ENTITY_SCHEMA_VERSION

    def supports(self, strategy: str) -> bool:
        return strategy in self.training_strategy_support

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── experiment ───────────────────────────────────────────────────────


class ExperimentStatus(StrEnum):
    DRAFT = "DRAFT"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    ARCHIVED = "ARCHIVED"


@dataclass
class Experiment:
    """A hypothesis, with the runs that test it kept separate."""

    experiment_id: str
    name: str
    hypothesis: str
    base_model_id: str
    description: str = ""
    status: str = ExperimentStatus.DRAFT.value
    #: Why the experiment cannot proceed, when BLOCKED.
    blocked_reason: str = ""
    dataset_lock_ref: str | None = None
    curation_lock_ref: str | None = None
    operator: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=now)
    schema_version: str = ENTITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── training run ─────────────────────────────────────────────────────


class RunStatus(StrEnum):
    """The run state machine.

    No PAUSING/PAUSED: the installed trainer cannot pause, and a state
    nothing can enter would misrepresent what the system can do.
    """

    DRAFT = "DRAFT"
    VALIDATING = "VALIDATING"
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    #: The worker stopped reporting. Deliberately distinct from FAILED:
    #: we do not know that training stopped, only that we lost contact.
    LOST = "LOST"


#: Legal transitions. Everything else is rejected, which is what makes
#: an idempotent launch possible — starting a RUNNING run is not an
#: error to swallow, it is a transition that does not exist.
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    RunStatus.DRAFT.value: frozenset({RunStatus.VALIDATING.value, RunStatus.CANCELLED.value}),
    RunStatus.VALIDATING.value: frozenset(
        {RunStatus.QUEUED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
    ),
    RunStatus.QUEUED.value: frozenset(
        {RunStatus.STARTING.value, RunStatus.CANCELLED.value, RunStatus.FAILED.value}
    ),
    RunStatus.STARTING.value: frozenset(
        {RunStatus.RUNNING.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
    ),
    RunStatus.RUNNING.value: frozenset(
        {
            RunStatus.COMPLETED.value,
            RunStatus.FAILED.value,
            RunStatus.CANCELLED.value,
            RunStatus.LOST.value,
        }
    ),
    # Terminal. A retry is a new run, so nothing leaves these.
    RunStatus.COMPLETED.value: frozenset(),
    RunStatus.FAILED.value: frozenset(),
    RunStatus.CANCELLED.value: frozenset(),
    RunStatus.LOST.value: frozenset({RunStatus.FAILED.value, RunStatus.COMPLETED.value}),
}

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
)


class FailureCode(StrEnum):
    """Why a run failed, in a closed vocabulary.

    Closed on purpose. Parsing arbitrary exception text into ever more
    specific codes produces fiction; the raw sanitised diagnostic is
    stored separately in ``error_detail`` and the code stays honest,
    including ``UNKNOWN``.
    """

    DATASET_LOCK_INVALID = "DATASET_LOCK_INVALID"
    CURATION_LOCK_INVALID = "CURATION_LOCK_INVALID"
    RIGHTS_GATE_FAILED = "RIGHTS_GATE_FAILED"
    EVALUATION_LEAKAGE = "EVALUATION_LEAKAGE"
    SELF_GENERATED_BLOCKED = "SELF_GENERATED_BLOCKED"
    ENVIRONMENT_INVALID = "ENVIRONMENT_INVALID"
    INSUFFICIENT_HARDWARE = "INSUFFICIENT_HARDWARE"
    CODE_VERSION_DIRTY = "CODE_VERSION_DIRTY"
    WORKER_LOST = "WORKER_LOST"
    TRAINER_CRASH = "TRAINER_CRASH"
    OOM = "OOM"
    CHECKPOINT_WRITE_FAILED = "CHECKPOINT_WRITE_FAILED"
    CANCELLED_BY_OPERATOR = "CANCELLED_BY_OPERATOR"
    UNKNOWN = "UNKNOWN"


@dataclass
class TrainingDatasetRef:
    """A portable reference to training data.

    Deliberately not a filesystem path. A run created on a Mac has to be
    executable on a rented Linux box, so the plan names *identities and
    digests* and the execution backend resolves them to whatever paths
    exist on the machine it runs on.
    """

    dataset_id: str
    dataset_lock_sha256: str
    curation_id: str
    curation_lock_sha256: str
    curated_manifest_sha256: str
    manifest_artifact_ref: str
    sampling_weights_sha256: str | None = None
    selected_track_count: int = 0
    selected_hours: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingRun:
    """One concrete execution attempt."""

    run_id: str
    experiment_id: str
    base_model_id: str
    dataset_ref: TrainingDatasetRef
    config: TrainingConfig
    execution_backend: str
    status: str = RunStatus.DRAFT.value
    worker_id: str | None = None
    training_plan_sha256: str | None = None
    #: Lineage. A retry cites its predecessor rather than replacing it.
    parent_run_id: str | None = None
    resume_from_checkpoint_id: str | None = None
    output_directory: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: str = field(default_factory=now)
    queued_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    failed_at: str | None = None
    cancelled_at: str | None = None
    schema_version: str = ENTITY_SCHEMA_VERSION

    @property
    def config_sha256(self) -> str:
        return self.config.digest()

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def has_started(self) -> bool:
        """Whether the immutable fields are frozen.

        From QUEUED onward the plan has been compiled and cited, so the
        inputs that produced it may no longer change.
        """
        return self.status not in (RunStatus.DRAFT.value, RunStatus.VALIDATING.value)

    def can_transition_to(self, status: str) -> bool:
        return status in ALLOWED_TRANSITIONS.get(self.status, frozenset())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        payload["dataset_ref"] = self.dataset_ref.to_dict()
        payload["config_sha256"] = self.config_sha256
        return payload


#: Fields frozen once a run has started. Changing any of them changes
#: what was trained, so a change means a new run, not an edit.
IMMUTABLE_AFTER_START: frozenset[str] = frozenset(
    {
        "base_model_id",
        "dataset_ref",
        "config",
        "training_plan_sha256",
        "experiment_id",
        "resume_from_checkpoint_id",
    }
)


# ── worker ───────────────────────────────────────────────────────────


class WorkerStatus(StrEnum):
    REGISTERED = "REGISTERED"
    ONLINE = "ONLINE"
    BUSY = "BUSY"
    OFFLINE = "OFFLINE"
    LOST = "LOST"
    RETIRED = "RETIRED"


class WorkerClass(StrEnum):
    """What a worker is allowed to be used for.

    The local Mac is ``DEVELOPMENT_ONLY`` until a probe proves
    otherwise. It is not promoted by having a GPU-shaped field filled
    in — every hardware fact must come from a worker report.
    """

    DEVELOPMENT_ONLY = "DEVELOPMENT_ONLY"
    GPU_TRAINING_READY = "GPU_TRAINING_READY"
    UNVERIFIED = "UNVERIFIED"


@dataclass
class WorkerCapabilities:
    """Hardware facts, every one of them reported rather than assumed.

    ``None`` means nobody has measured it. It never means zero, and it
    must never be turned into a default — an invented VRAM figure is
    how a run gets scheduled onto hardware that cannot hold it.
    """

    gpu_vendor: str | None = None
    gpu_model: str | None = None
    gpu_count: int | None = None
    vram_total_mb: int | None = None
    system_ram_mb: int | None = None
    cpu_count: int | None = None
    cuda_available: bool | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    torch_version: str | None = None
    python_version: str | None = None
    bf16_supported: bool | None = None
    free_disk_mb: int | None = None
    #: Where these facts came from: a probe, or nothing.
    reported_by: str = "UNREPORTED"
    reported_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingWorker:
    worker_id: str
    name: str
    backend_type: str
    host_identity: str
    worker_class: str = WorkerClass.UNVERIFIED.value
    status: str = WorkerStatus.REGISTERED.value
    capabilities: WorkerCapabilities = field(default_factory=WorkerCapabilities)
    #: One model at a time unless deliberately configured otherwise.
    #: Multiple GPUs on a host means one bigger run, not several runs.
    max_concurrent_runs: int = 1
    software_environment: dict[str, str] = field(default_factory=dict)
    #: Names only. Never a key, token or password.
    ssh_key_ref: str | None = None
    credential_ref: str | None = None
    last_heartbeat: str | None = None
    created_at: str = field(default_factory=now)
    schema_version: str = ENTITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capabilities"] = self.capabilities.to_dict()
        return payload


# ── checkpoint ───────────────────────────────────────────────────────


class CheckpointStatus(StrEnum):
    #: Being written. Never eligible for anything.
    WRITING = "WRITING"
    READY = "READY"
    CORRUPT = "CORRUPT"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"


class CheckpointKind(StrEnum):
    """What kind of artifact this is.

    ``ADAPTER`` is what the installed trainer produces. ``MOCK`` exists
    so tests and dry runs have something to register, and it is a
    distinct kind rather than a flag so that no query for a real
    checkpoint can return one by accident.
    """

    ADAPTER = "ADAPTER"
    FULL_MODEL = "FULL_MODEL"
    MOCK = "MOCK"


@dataclass
class Checkpoint:
    checkpoint_id: str
    run_id: str
    kind: str
    step: int | None = None
    epoch: int | None = None
    reference: str | None = None
    status: str = CheckpointStatus.WRITING.value
    size_bytes: int | None = None
    sha256: str | None = None
    checkpoint_format: str = "peft-adapter-safetensors"
    metrics_snapshot: dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=now)
    finalized_at: str | None = None
    schema_version: str = ENTITY_SCHEMA_VERSION

    @property
    def is_real_model(self) -> bool:
        """Whether this is trained weights rather than a placeholder."""
        return self.kind in (CheckpointKind.ADAPTER.value, CheckpointKind.FULL_MODEL.value)

    @property
    def promotable(self) -> bool:
        return self.status == CheckpointStatus.READY.value and self.is_real_model

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── evaluation candidate ─────────────────────────────────────────────


class CandidateStatus(StrEnum):
    PENDING_EVALUATION = "PENDING_EVALUATION"
    EVALUATING = "EVALUATING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


@dataclass
class EvaluationCandidate:
    """A checkpoint nominated for evaluation. Nothing more.

    Creating one is the furthest Phase 25 goes. It carries no quality
    claim, and the evaluation that could support one does not exist
    yet.
    """

    candidate_id: str
    run_id: str
    checkpoint_id: str
    experiment_id: str
    status: str = CandidateStatus.PENDING_EVALUATION.value
    created_at: str = field(default_factory=now)
    notes: str = ""
    schema_version: str = ENTITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionDecision:
    """A recorded decision about a candidate.

    Phase 25 can express one and never makes one. Promotion requires
    evaluation evidence, which is the next phase's job.
    """

    candidate_id: str
    decision: str
    decided_by: str
    rationale: str
    evidence_ref: str | None = None
    decided_at: str = field(default_factory=now)
    schema_version: str = ENTITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactRef:
    """A pointer to something on a worker or in storage.

    A reference, not a path: ``scheme`` says how to resolve it, and the
    execution backend does the resolving. This is what lets a plan built
    on a Mac execute on a rented Linux host.
    """

    scheme: str
    locator: str
    sha256: str | None = None
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_experiment_status(current: str, run_statuses: list[str]) -> str:
    """Experiment status from the runs beneath it.

    A failed run does not fail the experiment. A hypothesis is not
    disproved by a crashed process, and marking it so would push people
    toward editing run history to make an experiment look clean.
    """
    if current in (ExperimentStatus.ARCHIVED.value, ExperimentStatus.BLOCKED.value):
        return current
    if not run_statuses:
        return current if current == ExperimentStatus.DRAFT.value else ExperimentStatus.READY.value
    active = {
        RunStatus.VALIDATING.value,
        RunStatus.QUEUED.value,
        RunStatus.STARTING.value,
        RunStatus.RUNNING.value,
    }
    if any(status in active for status in run_statuses):
        return ExperimentStatus.RUNNING.value
    if RunStatus.COMPLETED.value in run_statuses:
        return ExperimentStatus.COMPLETED.value
    return ExperimentStatus.READY.value
