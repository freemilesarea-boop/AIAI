"""The operator wire contract.

Concrete models rather than loose dictionaries, for the reason Step 64
gives: a console whose entities are `dict[str, Any]` cannot tell an
operator that a field it is showing no longer exists, and the first
place that shows up is a blank panel where a plan hash used to be.

Two rules run through the shapes here.

**Unknown is a value, not an absence.** ``None`` in these models means
nobody measured it, and the UI renders it as UNKNOWN. It is never
defaulted to zero, never omitted, and never allowed to look like a
pass. That is Phase 25's rule about hardware facts and Phase 27's rule
about preflight, carried across the wire instead of being softened at
the edge.

**Nothing secret has a field.** There is no model here with a place to
put an SSH key path, a token, a credential value or a resolved secret.
A field that does not exist cannot be populated by a later change, and
that is a stronger guarantee than remembering to strip one.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ── shared ───────────────────────────────────────────────────────────


class Page(BaseModel):
    """Where a list response sits in a larger collection."""

    total: int
    limit: int
    offset: int
    returned: int


class SystemCheck(BaseModel):
    """One infrastructure fact the console can state.

    ``status`` is deliberately not a boolean. "We could not look" and
    "we looked and it is broken" are different things to an operator
    deciding whether to rent a GPU.
    """

    name: str
    status: Literal["OK", "DEGRADED", "UNAVAILABLE", "UNKNOWN"]
    detail: str = ""


class CountBreakdown(BaseModel):
    """A total, and how it splits by state.

    ``by_state`` rather than a field per state: the state vocabularies
    belong to Phase 25 and 26, and mirroring them into fixed fields here
    would mean a new state silently disappearing from the dashboard.
    """

    total: int = 0
    by_state: dict[str, int] = Field(default_factory=dict)


class OverviewResponse(BaseModel):
    generated_at: str
    registry_present: bool
    experiments: CountBreakdown
    runs: CountBreakdown
    workers: CountBreakdown
    worker_classes: CountBreakdown
    checkpoints: CountBreakdown
    checkpoint_kinds: CountBreakdown
    evaluations: CountBreakdown
    qualifications: CountBreakdown
    system: list[SystemCheck] = Field(default_factory=list)
    #: Present when the registry holds nothing at all, so the UI can say
    #: which of "nothing has happened yet" and "we are pointed at the
    #: wrong directory" it is looking at.
    empty_reason: str | None = None


# ── experiments ──────────────────────────────────────────────────────


class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    hypothesis: str
    description: str = ""
    base_model_id: str
    status: str
    blocked_reason: str = ""
    dataset_lock_ref: str | None = None
    curation_lock_ref: str | None = None
    operator: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: str
    run_count: int = 0
    latest_run_id: str | None = None
    latest_run_status: str | None = None


class ExperimentListResponse(BaseModel):
    items: list[ExperimentSummary]
    page: Page
    #: Every distinct value present, so the filter controls offer what
    #: the registry actually contains rather than a hard-coded list.
    available_statuses: list[str] = Field(default_factory=list)
    available_base_models: list[str] = Field(default_factory=list)
    available_tags: list[str] = Field(default_factory=list)


class CandidateSummary(BaseModel):
    candidate_id: str
    run_id: str
    checkpoint_id: str
    experiment_id: str
    status: str
    created_at: str
    notes: str = ""


class ExperimentDetail(BaseModel):
    experiment: ExperimentSummary
    base_model: ModelBaselineView | None = None
    runs: list[RunSummary] = Field(default_factory=list)
    candidates: list[CandidateSummary] = Field(default_factory=list)
    evaluations: list[EvaluationSummary] = Field(default_factory=list)
    qualifications: list[QualificationSummary] = Field(default_factory=list)
    audit_events: list[AuditEvent] = Field(default_factory=list)


class ExperimentCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    hypothesis: str = Field(min_length=1, max_length=2000)
    base_model_id: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=8000)
    operator: str = Field(default="", max_length=120)
    tags: list[str] = Field(default_factory=list, max_length=32)


# ── models ───────────────────────────────────────────────────────────


class ModelBaselineView(BaseModel):
    model_id: str
    provider: str
    model_family: str
    model_name: str
    model_version: str
    upstream_commit: str
    architecture: str
    training_strategy_support: list[str] = Field(default_factory=list)
    checkpoint_sha256: str | None = None
    identity_basis: str
    stage: str
    created_at: str


class BaselineResponse(BaseModel):
    """Which model the product is actually serving, if any is marked.

    Read-only in every sense: Phase 28 has no path that changes a stage,
    and promotion review stops at staging by design.
    """

    production: list[ModelBaselineView] = Field(default_factory=list)
    all_models: list[ModelBaselineView] = Field(default_factory=list)
    note: str


# ── runs ─────────────────────────────────────────────────────────────


class FailureView(BaseModel):
    """A failure, in the operator's language and the system's.

    Both, always. The humanised line is what stops an operator hunting
    through logs for a rights failure; the raw code is what they search
    for when they do go to the logs.
    """

    code: str
    headline: str
    guidance: str
    raw_message: str | None = None
    #: True only where the classification is definitive. An OOM claimed
    #: from a SIGKILL would send the next experiment chasing a memory
    #: problem that never existed.
    confident: bool = True


class MetricPoint(BaseModel):
    step: int | None = None
    epoch: int | None = None
    value: float
    timestamp: str


class MetricSeries(BaseModel):
    metric_name: str
    unit: str = ""
    #: Where the numbers came from. SIMULATED never becomes evidence.
    sources: list[str] = Field(default_factory=list)
    points: list[MetricPoint] = Field(default_factory=list)
    total_points: int = 0
    #: True when the series was thinned to fit a response. The chart
    #: says so rather than implying it drew every step.
    sampled: bool = False
    last_value: float | None = None


class RunSummary(BaseModel):
    run_id: str
    experiment_id: str
    experiment_name: str = ""
    base_model_id: str
    status: str
    execution_backend: str
    worker_id: str | None = None
    worker_name: str | None = None
    created_at: str
    queued_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    failed_at: str | None = None
    cancelled_at: str | None = None
    #: Wall time from start to a terminal stamp, or to now while
    #: running. None when the run never started.
    duration_seconds: float | None = None
    parent_run_id: str | None = None
    checkpoint_count: int = 0
    latest_metric: MetricPoint | None = None
    latest_metric_name: str | None = None
    failure: FailureView | None = None
    #: A cancellation the console requested but nothing has confirmed.
    cancel_requested_at: str | None = None


class RunListResponse(BaseModel):
    items: list[RunSummary]
    page: Page
    available_statuses: list[str] = Field(default_factory=list)
    available_backends: list[str] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    """One step of the Phase 25 run state machine, and whether it happened."""

    state: str
    reached: bool
    at: str | None = None
    current: bool = False
    #: Terminal states that this run did not take. Shown greyed rather
    #: than hidden, so the shape of the machine stays visible.
    terminal: bool = False


class TrainingConfigView(BaseModel):
    """Exactly the fields the installed trainer accepts.

    Mirrors ``TrainingConfig``. Nothing aspirational: a setting the
    trainer ignores would read here as a knob that did something.
    """

    strategy: str
    learning_rate: float
    batch_size: int
    gradient_accumulation: int
    epochs: int
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float
    seed: int
    optimizer_type: str
    scheduler_type: str
    gradient_checkpointing: bool
    offload_encoder: bool
    shift: float
    num_inference_steps: int
    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]
    bias: str
    attention_type: str
    precision: str
    num_devices: int
    checkpoint_every_epochs: int
    log_every_steps: int
    log_heavy_every_steps: int
    sample_every_n_epochs: int
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    persistent_workers: bool
    ace_step_commit: str


class DatasetRefView(BaseModel):
    """Identity and digests. Never a path, never a track listing."""

    dataset_id: str
    dataset_lock_sha256: str
    curation_id: str
    curation_lock_sha256: str
    curated_manifest_sha256: str
    manifest_artifact_ref: str
    sampling_weights_sha256: str | None = None
    selected_track_count: int = 0
    selected_hours: float = 0.0


class GateView(BaseModel):
    """One Phase 25 gate result, with no override anywhere near it."""

    name: str
    status: Literal["PASS", "FAIL", "NOT_EVALUATED"]
    detail: str = ""
    failure_code: str | None = None
    offending_count: int = 0
    #: Ids of offending tracks, capped by the gate itself. Identifiers
    #: only — never a filename, a title or a path.
    offending_ids: list[str] = Field(default_factory=list)


class PreflightCheckView(BaseModel):
    name: str
    status: Literal["PASS", "FAIL", "UNKNOWN"]
    detail: str = ""
    severity: str = "REQUIRED"
    expected: str | None = None
    observed: str | None = None


class PreflightView(BaseModel):
    """One side of preflight — control plane or worker.

    ``available`` distinguishes "this has not been run" from "this ran
    and everything passed", which look identical if you only carry a
    list of failures.
    """

    available: bool
    unavailable_reason: str | None = None
    status: Literal["PASS", "BLOCKED", "FAIL", "UNKNOWN"] = "UNKNOWN"
    checks: list[PreflightCheckView] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    unknown: list[str] = Field(default_factory=list)
    generated_at: str | None = None


class TrainingPreflightCheckView(BaseModel):
    """One Phase 33 check, with its taxonomy entry attached."""

    name: str
    group: str = "plan"
    status: Literal["PASS", "FAIL", "UNKNOWN", "NOT_APPLICABLE"]
    detail: str = ""
    reason: str | None = None
    mandatory: bool = True


class CapacityEvidenceView(BaseModel):
    """One capacity fact and how it was established.

    ``source`` is never dropped. A number without it is a number a
    reader would assume was measured, and almost nothing here is.
    """

    name: str
    source: Literal["MEASURED", "ESTIMATED", "UNKNOWN"]
    value_mb: int | None = None
    detail: str = ""
    derivation: str = ""
    #: True where the figure is Apple unified memory shared with the
    #: operating system rather than dedicated accelerator memory.
    unified_memory: bool = False


class CanaryView(BaseModel):
    """What a bounded canary established, if one has run."""

    available: bool = False
    unavailable_reason: str | None = None
    status: Literal["PASSED", "FAILED", "BLOCKED", "NOT_RUN"] = "NOT_RUN"
    mode: str | None = None
    detail: str = ""
    steps: int | None = None
    max_optimizer_steps: int | None = None
    max_samples: int | None = None
    max_epochs: int | None = None
    dataset_kind: str | None = None
    exit_code: int | None = None
    seconds: float | None = None
    checkpoint_ok: bool | None = None
    checkpoint_step: int | None = None
    checkpoint_provenance_plan_digest: str | None = None
    checkpoint_problems: list[str] = Field(default_factory=list)
    resume_ok: bool | None = None
    resume_detail: str = ""


class TrainingPreflightView(BaseModel):
    """Phase 33's verdict: can this machine execute this plan?

    ``UNVERIFIED`` is a first-class status here rather than a shade of
    pass. The console renders it distinctly for the same reason the
    model carries it: a check nobody could perform is not a check that
    passed, and an operator about to rent a GPU is the person a
    reassuring colour would mislead.
    """

    available: bool = False
    unavailable_reason: str | None = None
    status: Literal["READY", "BLOCKED", "UNVERIFIED"] = "UNVERIFIED"
    intent: str = "CANARY"
    plan_digest: str | None = None
    execution_location: str | None = None
    execution_device: str | None = None
    torch_device: str | None = None
    resolved_precision: str | None = None
    optimizer: str | None = None
    worker_identity: str | None = None
    target_label: str | None = None
    capability_digest: str | None = None
    dataset_status: str = "UNKNOWN"
    dependency_status: str = "UNKNOWN"
    storage_status: str = "UNKNOWN"
    checkpoint_status: str = "NOT_APPLICABLE"
    canary_status: str = "NOT_RUN"
    capacity_status: str = "UNKNOWN"
    checks: list[TrainingPreflightCheckView] = Field(default_factory=list)
    capacity: list[CapacityEvidenceView] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    unverified: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    hardware: dict[str, Any] = Field(default_factory=dict)
    measured_at: str | None = None
    policy_version: str | None = None


class ReproducibilityView(BaseModel):
    """Everything needed to rebuild this run, in one place."""

    luber_commit: str | None = None
    luber_dirty: bool | None = None
    ace_step_commit: str | None = None
    base_model_id: str
    base_model_upstream_commit: str | None = None
    dataset_lock_sha256: str
    curation_lock_sha256: str
    curated_manifest_sha256: str
    training_config_sha256: str
    training_plan_sha256: str | None = None
    environment_lock_digest: str | None = None
    worker_capability_signature: str | None = None
    python_version: str | None = None
    torch_version: str | None = None


class CostView(BaseModel):
    """Cost metadata, shown only where it is known.

    No pricing is fetched from a provider and none is inferred from an
    instance name. Every field here is either recorded or None.
    """

    provider: str | None = None
    instance_type: str | None = None
    hourly_rate: float | None = None
    currency: str | None = None
    wall_seconds: float | None = None
    gpu_seconds: float | None = None
    estimated_cost: float | None = None
    actual_cost: float | None = None
    unknown: list[str] = Field(default_factory=list)


class StagingView(BaseModel):
    """What was transferred to the worker, summarised.

    A summary and a count, never a list of thousands of files by
    default — Step 52. The entries a caller can expand are artifact
    roles and digests, not audio.
    """

    available: bool
    unavailable_reason: str | None = None
    total_entries: int = 0
    unique_contents: int = 0
    total_bytes: int = 0
    #: Whether the console could look at the worker's filesystem at all.
    #: Without it, present/missing are both zero and mean nothing — a
    #: distinction the UI has to be able to make.
    presence_checked: bool = False
    #: Entries whose bytes exist on the worker. Presence, not integrity:
    #: re-hashing a staged dataset on every page view is not something a
    #: console gets to do, and claiming "verified" from an `is_file`
    #: check would be exactly the kind of unearned green this phase is
    #: meant to avoid. The digests were checked on arrival by Phase 27.
    present_entries: int = 0
    missing_entries: int = 0
    roles: dict[str, int] = Field(default_factory=dict)
    built_at: str | None = None


class RemoteStateView(BaseModel):
    """The worker's own view, kept separate from the run's status.

    Deliberately never merged with ``RunSummary.status``. The control
    plane saying LOST while the worker says RUNNING is not a
    contradiction to be resolved by picking one — it is the exact
    situation reconciliation exists for, and collapsing the two would
    delete the information that makes recovery possible.
    """

    available: bool
    unavailable_reason: str | None = None
    worker_state: str | None = None
    implied_run_status: str | None = None
    detail: str = ""
    exit_code: int | None = None
    failure_code: str | None = None
    lease_id: str | None = None
    process_alive: bool | None = None
    updated_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    cancel_requested_at: str | None = None
    protocol_version: str | None = None
    plan_sha256: str | None = None


class HeartbeatView(BaseModel):
    """Liveness, with the exact instant preserved beside the friendly age."""

    available: bool
    unavailable_reason: str | None = None
    timestamp: str | None = None
    age_seconds: float | None = None
    liveness: Literal["ONLINE", "STALE", "OFFLINE", "UNKNOWN"] = "UNKNOWN"
    worker_state: str | None = None
    active_run_id: str | None = None
    health: str | None = None
    uptime_seconds: float | None = None
    free_disk_mb: int | None = None
    gpu: list[GpuTelemetryView] = Field(default_factory=list)
    detail: str = ""


class GpuTelemetryView(BaseModel):
    index: int
    utilization_pct: float | None = None
    memory_used_mb: int | None = None
    memory_total_mb: int | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    sampled_at: str | None = None


class CheckpointSummary(BaseModel):
    checkpoint_id: str
    run_id: str
    experiment_id: str = ""
    kind: str
    #: True only for ADAPTER and FULL_MODEL. A MOCK artifact is marked
    #: everywhere it appears, because the whole point of it being a
    #: distinct kind is that no query returns it by accident.
    is_real_model: bool
    status: str
    step: int | None = None
    epoch: int | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    checkpoint_format: str
    created_at: str
    finalized_at: str | None = None
    metrics_snapshot: dict[str, float] = Field(default_factory=dict)
    #: Whether this checkpoint could be nominated for evaluation right
    #: now, and if not, why. Computed server-side: a UI that decided
    #: this itself would eventually disagree with the API.
    can_evaluate: bool = False
    evaluate_blocked_reason: str | None = None
    candidate_id: str | None = None
    #: Where the bytes are, as a scheme and a locator — never a
    #: filesystem path from this machine.
    location_scheme: str | None = None
    location_present: bool | None = None


class CheckpointListResponse(BaseModel):
    items: list[CheckpointSummary]
    page: Page
    available_statuses: list[str] = Field(default_factory=list)
    available_kinds: list[str] = Field(default_factory=list)


class CheckpointDetail(BaseModel):
    checkpoint: CheckpointSummary
    run: RunSummary | None = None
    experiment: ExperimentSummary | None = None
    evaluations: list[EvaluationSummary] = Field(default_factory=list)
    qualifications: list[QualificationSummary] = Field(default_factory=list)
    audit_events: list[AuditEvent] = Field(default_factory=list)


class AuditEvent(BaseModel):
    timestamp: str
    event: str
    entity_id: str
    entity_kind: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LogView(BaseModel):
    """One incremental read of one stream.

    ``next_offset`` is the whole protocol: the browser sends back what
    it was given and receives only what arrived since. Refetching the
    file each poll would make a long run's log unusable, and Phase 27
    already solved this — the cursor semantics here are its own.
    """

    available: bool
    unavailable_reason: str | None = None
    stream: Literal["stdout", "stderr"] = "stdout"
    offset: int = 0
    next_offset: int = 0
    size_bytes: int = 0
    eof: bool = True
    truncated: bool = False
    text: str = ""
    #: True when the caller asked for the tail of a large file and the
    #: beginning was skipped. The UI offers "load older" rather than
    #: implying it has shown everything.
    from_tail: bool = False


class RunDetail(BaseModel):
    run: RunSummary
    experiment: ExperimentSummary | None = None
    base_model: ModelBaselineView | None = None
    timeline: list[TimelineEntry] = Field(default_factory=list)
    config: TrainingConfigView
    config_sha256: str
    training_plan_sha256: str | None = None
    dataset: DatasetRefView
    dataset_available: bool = False
    worker: WorkerSummary | None = None
    heartbeat: HeartbeatView
    remote: RemoteStateView
    staging: StagingView
    control_preflight: PreflightView
    remote_preflight: PreflightView
    training_preflight: TrainingPreflightView
    canary: CanaryView
    gates: list[GateView] = Field(default_factory=list)
    gates_available: bool = False
    gates_unavailable_reason: str | None = None
    metrics: list[MetricSeries] = Field(default_factory=list)
    telemetry: list[MetricSeries] = Field(default_factory=list)
    progress: RunProgress
    checkpoints: list[CheckpointSummary] = Field(default_factory=list)
    evaluations: list[EvaluationSummary] = Field(default_factory=list)
    reproducibility: ReproducibilityView
    cost: CostView
    audit_events: list[AuditEvent] = Field(default_factory=list)
    actions: list[ActionAvailability] = Field(default_factory=list)
    environment_lock: dict[str, Any] = Field(default_factory=dict)


class RunProgress(BaseModel):
    """Where a run has got to, with no invented arithmetic.

    ``eta_seconds`` is None unless total steps are known *and* recent
    step times are stable. Everywhere else it is None and the UI says
    UNKNOWN, because a made-up ETA is the number an operator plans a
    day around.
    """

    latest_step: int | None = None
    latest_epoch: int | None = None
    total_epochs: int | None = None
    elapsed_seconds: float | None = None
    latest_train_loss: float | None = None
    latest_learning_rate: float | None = None
    latest_checkpoint_id: str | None = None
    eta_seconds: float | None = None
    eta_reason: str = "not calculable from what has been measured"


class ActionAvailability(BaseModel):
    """Whether an action is offered, and if not, the honest reason.

    Computed on the server for the reason Step 66 gives: a disabled
    button is a courtesy, not a control. Every action endpoint
    re-validates independently, and this exists so the UI does not have
    to guess what the endpoint will say.
    """

    action: str
    label: str
    available: bool
    reason: str = ""
    #: What the operator is about to cause, in one sentence, for the
    #: confirmation dialog. Never "Are you sure?".
    confirmation: str = ""
    destructive: bool = False


# ── workers ──────────────────────────────────────────────────────────


class WorkerCapabilitiesView(BaseModel):
    """Reported hardware facts. ``None`` means nobody measured it."""

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
    reported_by: str = "UNREPORTED"
    reported_at: str | None = None


class WorkerSummary(BaseModel):
    worker_id: str
    name: str
    backend_type: str
    #: A digest of stable machine facts, not a hostname. Shown because
    #: an operator needs to know whether the box was rebuilt.
    host_identity: str
    worker_class: str
    #: The wider Phase 27 vocabulary when a probe recorded one:
    #: CUDA_TRAINING / CUDA_EVALUATION / DEVELOPMENT_ONLY / UNAVAILABLE.
    remote_classification: str | None = None
    #: Registry status, which is not liveness. A worker recorded ONLINE
    #: that has not spoken for an hour is STALE, and both are shown.
    status: str
    liveness: Literal["ONLINE", "STALE", "OFFLINE", "UNKNOWN"] = "UNKNOWN"
    last_heartbeat: str | None = None
    heartbeat_age_seconds: float | None = None
    max_concurrent_runs: int = 1
    active_run_ids: list[str] = Field(default_factory=list)
    capabilities: WorkerCapabilitiesView
    protocol_version: str | None = None
    capability_signature: str | None = None
    created_at: str
    #: True when this worker holds a credential *reference*. The
    #: reference name itself is never sent: an operator needs to know a
    #: key is configured, not what it is called.
    has_credentials: bool = False


class WorkerListResponse(BaseModel):
    items: list[WorkerSummary]
    page: Page
    available_classes: list[str] = Field(default_factory=list)
    available_liveness: list[str] = Field(default_factory=list)


class WorkerDetail(BaseModel):
    worker: WorkerSummary
    heartbeat: HeartbeatView
    software_environment: dict[str, str] = Field(default_factory=dict)
    recent_runs: list[RunSummary] = Field(default_factory=list)
    audit_events: list[AuditEvent] = Field(default_factory=list)
    #: Every capability the console could not establish, named. An
    #: operator reading UNKNOWN should be able to see the list of them
    #: rather than hunting field by field.
    unknown_capabilities: list[str] = Field(default_factory=list)


class WorkerCompatibility(BaseModel):
    """Whether a worker may be chosen for a run, and why not.

    Reasons are computed from reported capabilities only. A worker whose
    VRAM was never measured is not assumed to be large enough and is not
    assumed to be too small — it is reported as unverified, because
    guessing a requirement is how a run gets scheduled onto hardware
    that cannot hold it.
    """

    worker: WorkerSummary
    compatible: bool
    reasons: list[str] = Field(default_factory=list)


# ── compute targets (Phase 32) ───────────────────────────────────────


class ComputeTargetView(BaseModel):
    """One place a workload could run, and what it can take.

    No field here can hold a hostname, a username or a path. A compute
    target is a location, a device and a set of measurements — the
    worker's *name* is operator-chosen and already in the console, and
    `host_identity` is deliberately not carried across.
    """

    name: str
    #: LOCAL or REMOTE. Separate from `device` on purpose: this
    #: deployment has a local target that is not CUDA and will have a
    #: remote one that is, and neither implies the other.
    location: str
    #: CPU, MPS or CUDA.
    device: str
    #: READY, NOT_AVAILABLE, NOT_CONNECTED or UNPROBED.
    status: str
    detail: str = ""
    memory_mb: int | None = None
    #: Precisions measured working on this device. Empty means nobody
    #: measured, never "none work".
    precisions: list[str] = Field(default_factory=list)
    #: Workload classes this target could actually be asked to take,
    #: computed by the same policy the scheduler uses.
    workloads: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    #: True for a profile describing hardware nobody has probed.
    planned: bool = False
    capability_digest: str | None = None


class ComputeTargetsResponse(BaseModel):
    """The compute-targets panel: every target, and what is missing."""

    at: str
    summary: str
    targets: list[ComputeTargetView] = Field(default_factory=list)
    #: Concurrent local training jobs permitted. One, so the machine
    #: that runs the control plane stays a control plane.
    local_training_concurrency: int = 1
    capability_schema_version: str = ""
    execution_placement_policy_version: str = ""


# ── evaluations ──────────────────────────────────────────────────────


class EvaluationSummary(BaseModel):
    evaluation_id: str
    status: str
    mode: str
    suite_id: str = ""
    suite_version: str = ""
    suite_digest: str = ""
    policy_digest: str = ""
    candidate_id: str = ""
    checkpoint_id: str = ""
    run_id: str = ""
    experiment_id: str = ""
    baseline_label: str = ""
    candidate_label: str = ""
    experiment_hypothesis: str = ""
    seeds: list[int] = Field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None
    failed_at: str | None = None
    cancelled_at: str | None = None
    error: str | None = None
    qualification_outcome: str | None = None
    wall_seconds: float | None = None
    gpu_seconds: float | None = None


class EvaluationListResponse(BaseModel):
    items: list[EvaluationSummary]
    page: Page
    available_statuses: list[str] = Field(default_factory=list)
    available_outcomes: list[str] = Field(default_factory=list)


class GateOutcomeView(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    severity: str = "NONE"
    inconclusive: bool = False


class QualificationSummary(BaseModel):
    """A verdict, never reduced to a score.

    Step 40: an operator deciding whether to promote a checkpoint needs
    to know *which* gates failed. A single number would make a
    borderline pass and a catastrophic regression look adjacent.
    """

    evaluation_id: str
    candidate_id: str
    outcome: str
    policy_id: str
    policy_version: str
    policy_digest: str
    reasons: list[str] = Field(default_factory=list)
    passed_gates: list[str] = Field(default_factory=list)
    failed_gates: list[str] = Field(default_factory=list)
    inconclusive_gates: list[str] = Field(default_factory=list)
    gate_outcomes: list[GateOutcomeView] = Field(default_factory=list)
    hypothesis_status: str = ""
    human_review_required_for: list[str] = Field(default_factory=list)
    decided_at: str


class ComparisonView(BaseModel):
    metric_name: str
    verdict: str
    baseline_value: float | None = None
    candidate_value: float | None = None
    delta: float | None = None
    severity: str = "NONE"
    detail: str = ""


class PromotionReviewView(BaseModel):
    review_id: str
    candidate_id: str
    evaluation_id: str
    qualification_outcome: str
    decision: str
    decided_by: str
    rationale: str
    decided_at: str


class EvaluationDetail(BaseModel):
    evaluation: EvaluationSummary
    lineage: dict[str, str] = Field(default_factory=dict)
    qualification: QualificationSummary | None = None
    comparisons: list[ComparisonView] = Field(default_factory=list)
    regressions: list[ComparisonView] = Field(default_factory=list)
    promotion_reviews: list[PromotionReviewView] = Field(default_factory=list)
    human_review: HumanReviewView | None = None
    checkpoint: CheckpointSummary | None = None
    run: RunSummary | None = None
    experiment: ExperimentSummary | None = None
    audit_events: list[AuditEvent] = Field(default_factory=list)
    report_available: bool = False


class HumanReviewView(BaseModel):
    """What listening is being asked for, and whether it has happened.

    Phase 28 shows the state and does not run the session. Reviving the
    41-dimension Phase 20H rubric here would make the pipeline
    unusable, which in practice means bypassed.
    """

    required: bool
    mode: str
    reason: str = ""
    case_count: int = 0
    dimensions: list[str] = Field(default_factory=list)
    status: str = "PENDING"
    package_available: bool = False


class ComparisonRequest(BaseModel):
    """Which checkpoints to place side by side."""

    checkpoint_ids: list[str] = Field(min_length=2, max_length=8)


class CheckpointComparisonRow(BaseModel):
    checkpoint: CheckpointSummary
    evaluation: EvaluationSummary | None = None
    qualification: QualificationSummary | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    #: Training loss is carried separately and labelled as context. A
    #: lower training loss is not evidence that a model is better, and
    #: putting it in the same table as evaluation metrics invites
    #: exactly that reading.
    training_context: dict[str, float] = Field(default_factory=dict)


class CheckpointComparisonResponse(BaseModel):
    rows: list[CheckpointComparisonRow]
    metric_names: list[str] = Field(default_factory=list)
    note: str


# ── catalogues and creation ──────────────────────────────────────────


class BuildOption(BaseModel):
    """One dataset or curation build an operator may select.

    Identified by ``build_id`` and described by digests. No path: the
    console offers what the deployment configured, and where it lives
    is not the operator's input.
    """

    build_id: str
    identity: str = ""
    lock_sha256: str = ""
    manifest_sha256: str = ""
    track_count: int | None = None
    hours: float | None = None
    created_at: str | None = None
    #: For a curation: the dataset lock it was built from, so the UI can
    #: refuse a mismatched pairing before the gates do.
    source_dataset_lock_sha256: str | None = None


class CatalogueResponse(BaseModel):
    datasets: list[BuildOption] = Field(default_factory=list)
    curations: list[BuildOption] = Field(default_factory=list)
    dataset_problems: list[str] = Field(default_factory=list)
    curation_problems: list[str] = Field(default_factory=list)
    presets: list[PresetOption] = Field(default_factory=list)
    backends: list[str] = Field(default_factory=list)
    base_models: list[ModelBaselineView] = Field(default_factory=list)


class PresetOption(BaseModel):
    name: str
    intent: str
    config: TrainingConfigView


class RunCreateRequest(BaseModel):
    experiment_id: str = Field(min_length=1, max_length=120)
    dataset_build_id: str = Field(min_length=1, max_length=200)
    curation_build_id: str = Field(min_length=1, max_length=200)
    preset: str = Field(min_length=1, max_length=64)
    execution_backend: str = Field(min_length=1, max_length=64)
    worker_id: str | None = Field(default=None, max_length=120)
    parent_run_id: str | None = Field(default=None, max_length=120)
    resume_from_checkpoint_id: str | None = Field(default=None, max_length=120)


class ActionResponse(BaseModel):
    """What an action actually did, rather than what was asked for.

    ``performed`` is False when the console recorded intent it cannot
    itself deliver — a cancellation for a worker it has no transport to,
    say. Reporting that as success is how an operator comes to believe a
    rented GPU has been released.
    """

    action: str
    performed: bool
    run_status: str | None = None
    detail: str
    outcome: str | None = None
    created_id: str | None = None


# Deferred annotations: several models above reference ones defined
# further down, which is the natural reading order for a human and
# requires one rebuild pass for pydantic.
for _model in (
    ExperimentDetail,
    RunDetail,
    CheckpointDetail,
    EvaluationDetail,
    HeartbeatView,
    WorkerDetail,
    WorkerCompatibility,
    CheckpointComparisonRow,
    CatalogueResponse,
):
    _model.model_rebuild()
