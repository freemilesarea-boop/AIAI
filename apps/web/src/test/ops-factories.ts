/**
 * Operator fixtures, built from the wire types.
 *
 * One builder per entity so a field added to the contract updates every
 * test at once instead of leaving half of them constructing an object
 * the type no longer describes.
 *
 * The values are deliberately opinionated where the phase has a rule:
 * an unprobed worker has `null` capabilities rather than zeros, a dry
 * run's metrics carry `SIMULATED`, and the MOCK checkpoint is not
 * `is_real_model`. A fixture that softened any of those would let a
 * component pass a test it should fail.
 */

import type {
  ActionAvailability,
  CanaryRun,
  Capacity,
  CheckpointSummary,
  EvaluationDetail,
  GateView,
  Heartbeat,
  MemoryProfile,
  MetricSeries,
  Overview,
  Pilot,
  Preflight,
  Qualification,
  RunDetail,
  RunSummary,
  TimelineEntry,
  TrainingConfig,
  TrainingPreflight,
  WorkerSummary,
} from "@/lib/ops/types";

const NOW = "2026-08-21T04:00:00.000Z";

export function metricSeries(overrides: Partial<MetricSeries> = {}): MetricSeries {
  return {
    metric_name: "train_loss",
    unit: "",
    sources: ["TRAINER"],
    points: Array.from({ length: 12 }, (_, index) => ({
      step: index + 1,
      epoch: 1,
      value: 2.4 - index * 0.05,
      timestamp: NOW,
    })),
    total_points: 12,
    sampled: false,
    last_value: 1.85,
    ...overrides,
  };
}

export function heartbeat(overrides: Partial<Heartbeat> = {}): Heartbeat {
  return {
    available: true,
    unavailable_reason: null,
    timestamp: NOW,
    age_seconds: 4,
    liveness: "ONLINE",
    worker_state: "RUNNING",
    active_run_id: "run_1",
    health: "OK",
    uptime_seconds: 7200,
    free_disk_mb: 780_000,
    gpu: [],
    detail: "",
    ...overrides,
  };
}

export function trainingConfig(overrides: Partial<TrainingConfig> = {}): TrainingConfig {
  return {
    strategy: "LORA",
    learning_rate: 0.0001,
    batch_size: 1,
    gradient_accumulation: 4,
    epochs: 30,
    warmup_steps: 100,
    weight_decay: 0.01,
    max_grad_norm: 1,
    seed: 42,
    optimizer_type: "adamw",
    scheduler_type: "cosine",
    gradient_checkpointing: true,
    offload_encoder: false,
    shift: 3,
    num_inference_steps: 8,
    rank: 32,
    alpha: 64,
    dropout: 0.1,
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"],
    bias: "none",
    attention_type: "both",
    precision: "auto",
    num_devices: 1,
    checkpoint_every_epochs: 5,
    log_every_steps: 10,
    log_heavy_every_steps: 50,
    sample_every_n_epochs: 0,
    num_workers: 2,
    pin_memory: true,
    prefetch_factor: 2,
    persistent_workers: true,
    ace_step_commit: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
    ...overrides,
  };
}

export function runSummary(overrides: Partial<RunSummary> = {}): RunSummary {
  return {
    run_id: "run_1",
    experiment_id: "exp_1",
    experiment_name: "Korean vocal phrasing",
    base_model_id: "model_1",
    status: "RUNNING",
    execution_backend: "remote-gpu",
    worker_id: "wrk_1",
    worker_name: "rented-a100",
    created_at: NOW,
    queued_at: NOW,
    started_at: NOW,
    completed_at: null,
    failed_at: null,
    cancelled_at: null,
    duration_seconds: 3600,
    parent_run_id: null,
    checkpoint_count: 1,
    latest_metric: { step: 12, epoch: 1, value: 1.85, timestamp: NOW },
    latest_metric_name: "train_loss",
    failure: null,
    cancel_requested_at: null,
    ...overrides,
  };
}

export function timeline(status: string): TimelineEntry[] {
  const linear = ["DRAFT", "VALIDATING", "QUEUED", "STARTING", "RUNNING"];
  const terminal = ["COMPLETED", "FAILED", "CANCELLED", "LOST"];
  const reachedThrough = linear.indexOf(status) >= 0 ? linear.indexOf(status) : linear.length - 1;
  return [
    ...linear.map((state, index) => ({
      state: state as TimelineEntry["state"],
      reached: index <= reachedThrough,
      at: index === 0 ? NOW : null,
      current: state === status,
      terminal: false,
    })),
    ...terminal.map((state) => ({
      state: state as TimelineEntry["state"],
      reached: state === status,
      at: state === status ? NOW : null,
      current: state === status,
      terminal: true,
    })),
  ];
}

export function actions(overrides: Partial<ActionAvailability>[] = []): ActionAvailability[] {
  const base: ActionAvailability[] = [
    {
      action: "validate",
      label: "Validate",
      available: false,
      reason: "Only a DRAFT run can be validated; this run is RUNNING.",
      confirmation: "Run every Phase 25 gate against run run_1.",
      destructive: false,
    },
    {
      action: "dispatch",
      label: "Dispatch",
      available: false,
      reason: "Only a QUEUED run can be dispatched; this run is RUNNING.",
      confirmation: "Start run run_1 on the remote-gpu backend.",
      destructive: true,
    },
    {
      action: "cancel",
      label: "Cancel",
      available: true,
      reason: "",
      confirmation:
        "Request graceful cancellation of run run_1. The run stays as it is until a worker " +
        "confirms it stopped — this console records the request, it does not deliver the signal.",
      destructive: true,
    },
    {
      action: "reconcile",
      label: "Reconcile remote state",
      available: false,
      reason: "This console holds no transport to a worker.",
      confirmation: "Ask the worker what is actually happening to run run_1.",
      destructive: false,
    },
    {
      action: "create_retry_run",
      label: "Create retry run",
      available: false,
      reason: "A retry is only offered once a run has stopped.",
      confirmation: "Create a new DRAFT run citing run_1 as its parent.",
      destructive: false,
    },
  ];
  return base.map((action) => {
    const override = overrides.find((item) => item.action === action.action);
    return override ? { ...action, ...override } : action;
  });
}

export function unavailablePreflight(reason: string): Preflight {
  return {
    available: false,
    unavailable_reason: reason,
    status: "UNKNOWN",
    checks: [],
    problems: [],
    unknown: [],
    generated_at: null,
  };
}

/**
 * A Phase 33 preflight that reached UNVERIFIED — the honest default for
 * this project, since nothing has measured a memory requirement.
 */
export function trainingPreflight(
  overrides: Partial<TrainingPreflight> = {},
): TrainingPreflight {
  return {
    available: true,
    unavailable_reason: null,
    status: "UNVERIFIED",
    intent: "FULL_TRAINING",
    plan_digest: "a".repeat(64),
    execution_location: "REMOTE",
    execution_device: "CUDA",
    torch_device: "cuda",
    resolved_precision: "bf16",
    optimizer: "adamw",
    worker_identity: "worker_1",
    target_label: "rented-gpu",
    capability_digest: "b".repeat(64),
    dataset_status: "PASS",
    dependency_status: "UNKNOWN",
    storage_status: "PASS",
    checkpoint_status: "NOT_APPLICABLE",
    canary_status: "NOT_RUN",
    capacity_status: "UNKNOWN",
    checks: [
      {
        name: "plan.execution_device",
        group: "plan",
        status: "PASS",
        detail: "CUDA (torch: cuda)",
        reason: null,
        mandatory: true,
      },
      {
        name: "capacity.training_requirement",
        group: "capacity",
        status: "UNKNOWN",
        detail: "no memory requirement has been measured for this configuration",
        reason: "CAPACITY_UNVERIFIED",
        mandatory: true,
      },
    ],
    capacity: [
      {
        name: "device_memory_mb",
        source: "MEASURED",
        value_mb: 81920,
        detail: "81920 MB of dedicated device memory reported by the probe",
        derivation: "",
        unified_memory: false,
      },
      {
        name: "training_memory_requirement_mb",
        source: "UNKNOWN",
        value_mb: null,
        detail: "no LUBER configuration has a measured memory requirement on any device",
        derivation: "",
        unified_memory: false,
      },
    ],
    blocking_reasons: [],
    unverified: [
      "CAPACITY_UNVERIFIED: capacity.training_requirement: no memory requirement has been measured",
    ],
    warnings: [],
    hardware: { selected_device: "CUDA" },
    measured_at: "2026-08-22T10:00:00+00:00",
    policy_version: "training-preflight-v1",
    ...overrides,
  };
}

/**
 * A capacity decision built on a real-shaped Apple measurement: a
 * production-length sequence, a sampled peak, and unified memory that
 * is never called VRAM.
 */
export function capacity(overrides: Partial<Capacity> = {}): Capacity {
  return {
    available: true,
    unavailable_reason: null,
    qualification: "QUALIFIED",
    device: "MPS",
    policy_version: "capacity-policy-v1",
    policy: {},
    applicability: "APPLICABLE",
    applicability_detail: "every memory-relevant field matches",
    reasons: ["12189 MiB required against a 20889 MiB budget"],
    domains: [
      {
        domain: "APPLE_UNIFIED",
        qualification: "QUALIFIED",
        peak_mib: 9751,
        peak_kind: "SAMPLED_PEAK",
        required_mib: 12189,
        reserved_mib: 3686,
        budget_mib: 20889,
        total_mib: 24576,
        detail: "12189 MiB required against a 20889 MiB budget",
      },
      {
        domain: "HOST",
        qualification: "QUALIFIED",
        peak_mib: 486,
        peak_kind: "SAMPLED_PEAK",
        required_mib: 607,
        reserved_mib: 9011,
        budget_mib: 15564,
        total_mib: 24576,
        detail: "607 MiB required against a 15564 MiB budget",
      },
    ],
    evidence: [
      {
        name: "measured_peak_apple_unified_mb",
        source: "MEASURED",
        value_mb: 9751,
        detail: "sampled peak",
        derivation: "",
        unified_memory: true,
      },
      {
        name: "required_with_margin_apple_unified_mb",
        source: "DERIVED",
        value_mb: 12189,
        detail: "with the sampled-peak margin applied",
        derivation: "9751 MiB SAMPLED_PEAK x 1.25 safety margin",
        unified_memory: true,
      },
    ],
    profile: memoryProfile(),
    measured_at: "2026-08-22T21:00:00+00:00",
    ...overrides,
  };
}

export function memoryProfile(overrides: Partial<MemoryProfile> = {}): MemoryProfile {
  return {
    profile_id: "mps-bf16-b1-r32-t6000-7f9be176c11d",
    outcome: "COMPLETED",
    representativeness: "REPRESENTATIVE",
    representativeness_detail:
      "6000 latent frames ≈ 240s of audio, against a production maximum of 6000 frames",
    identity_digest: "7f9be176c11d".padEnd(64, "0"),
    device: "MPS",
    precision: "bf16",
    optimizer: "adamw",
    micro_batch_size: 1,
    gradient_accumulation: 4,
    effective_batch_size: 4,
    lora_rank: 32,
    gradient_checkpointing: true,
    latent_length: 6000,
    latent_seconds: 240,
    encoder_length: 256,
    peaks: [
      {
        domain: "APPLE_UNIFIED",
        kind: "SAMPLED_PEAK",
        source: "MEASURED",
        peak_mib: 9855,
        baseline_mib: 0,
        growth_mib: 9855,
        total_mib: 24576,
        sample_count: 113,
        detail: "the largest value a sampler observed. This is a lower bound",
      },
      {
        domain: "HOST",
        kind: "SAMPLED_PEAK",
        source: "MEASURED",
        peak_mib: 612,
        baseline_mib: 20,
        growth_mib: 592,
        total_mib: 24576,
        sample_count: 114,
        detail: "process resident set",
      },
    ],
    checkpoint_peak_mib: 9393,
    resume_peak_mib: 5464,
    optimizer_steps: 1,
    not_observed: {},
    torch_version: "2.10.0",
    ace_step_commit: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
    measured_at: "2026-08-22T21:00:00+00:00",
    failure_reason: "",
    failure_kind: "NOT_A_MEMORY_FAILURE",
    ...overrides,
  };
}

/**
 * A pilot that produced a valid signal on real rights-cleared material.
 * The default is deliberately the strongest honest case, so a test that
 * asserts restraint has to work to break it.
 */
export function pilot(overrides: Partial<Pilot> = {}): Pilot {
  return {
    available: true,
    unavailable_reason: null,
    pilot_id: "pilot-mps-bf16-r32-s24-5504d57229c5",
    outcome: "COMPLETED_VALID_SIGNAL",
    signal: "VALID_SIGNAL",
    signal_detail:
      "48 finite optimizer step(s), finite non-zero gradients, and 384 trainable tensor(s) " +
      "changed. This says the training path works. It says nothing about convergence or quality",
    failure: null,
    failure_detail: "",
    dataset_kind: "REAL_RIGHTS_CLEARED",
    expected_steps: 48,
    completed_steps: 48,
    step_ceiling: 48,
    within_budget: true,
    device: "MPS",
    precision: "bf16",
    lora_rank: 32,
    micro_batch_size: 1,
    gradient_accumulation: 4,
    latent_length: 6000,
    seed: 42,
    plan_digest: "a".repeat(64),
    dataset_manifest_digest: "b".repeat(64),
    capacity_qualification: "QUALIFIED",
    capacity_profile_id: "mps-bf16-b1-r32-t6000-7f9be176c11d",
    preflight_status: "READY",
    loss: [
      {
        step: 1,
        loss: 3.41,
        epoch: 1,
        learning_rate: 0.0001,
        grad_norm: 0.42,
        elapsed_seconds: 12.1,
        segment: "A",
        finite: true,
      },
      {
        step: 2,
        loss: 3.38,
        epoch: 2,
        learning_rate: 0.0001,
        grad_norm: 0.39,
        elapsed_seconds: 18.4,
        segment: "A",
        finite: true,
      },
    ],
    loss_statistics: {
      count: 48,
      finite_count: 48,
      finite_ratio: 1,
      first: 3.41,
      last: 3.29,
      minimum: 3.24,
      maximum: 3.44,
      mean: 3.35,
      median: 3.35,
      slope: -0.0021,
      slope_source: "DERIVED",
      slope_note:
        "least squares over the finite losses. Not a convergence claim: over tens of steps " +
        "this is a line through noise",
    },
    parameters: {
      changed_tensor_count: 384,
      trainable_tensor_count: 384,
      parameters_changed: true,
      base_model_preserved: true,
      max_absolute_delta: 0.0043,
    },
    gradients: {
      observed_steps: 48,
      finite_steps: 48,
      nonzero_steps: 48,
      mean_grad_norm: 0.41,
      all_finite: true,
      any_nonzero: true,
    },
    segments: [
      {
        name: "A",
        completed_steps: 24,
        first_step: 1,
        last_step: 24,
        checkpoint_id: "epoch_24_loss_3.3100",
        resumed_from: null,
        exit_code: 0,
        wall_seconds: 210.4,
        detail: "",
      },
      {
        name: "B",
        completed_steps: 24,
        first_step: 25,
        last_step: 48,
        checkpoint_id: "epoch_48_loss_3.2900",
        resumed_from: "epoch_24_loss_3.3100",
        exit_code: 0,
        wall_seconds: 205.2,
        detail: "",
      },
    ],
    checkpoint: { ok: true, step: 24, reopened: true },
    resume: {
      performed: true,
      resumed_from: "epoch_24_loss_3.3100",
      source_step: 24,
      final_step: 48,
      advanced: true,
    },
    artifact_class: ["EXPERIMENTAL", "NON_PRODUCTION", "NEVER_AUTO_PROMOTE"],
    started_at: "2026-08-22T23:00:00+00:00",
    finished_at: "2026-08-22T23:07:00+00:00",
    wall_seconds: 420,
    ...overrides,
  };
}

export function canaryRun(overrides: Partial<CanaryRun> = {}): CanaryRun {
  return {
    available: true,
    unavailable_reason: null,
    status: "PASSED",
    mode: "ACE_STEP",
    detail: "the installed trainer took 1 optimizer step and wrote a checkpoint that reopens",
    steps: 1,
    max_optimizer_steps: 8,
    max_samples: 2,
    max_epochs: 1,
    dataset_kind: "SYNTHETIC",
    exit_code: 0,
    seconds: 12.9,
    checkpoint_ok: true,
    checkpoint_step: 1,
    checkpoint_provenance_plan_digest: "c".repeat(64),
    checkpoint_problems: [],
    resume_ok: true,
    resume_detail: "training continued from step 1 to 2 with optimizer state restored",
    ...overrides,
  };
}

export function preflight(overrides: Partial<Preflight> = {}): Preflight {
  return {
    available: true,
    unavailable_reason: null,
    status: "BLOCKED",
    checks: [
      {
        name: "code_version",
        status: "PASS",
        detail: "",
        severity: "REQUIRED",
        expected: null,
        observed: null,
      },
      {
        name: "disk_capacity",
        status: "UNKNOWN",
        detail: "no checkpoint size has ever been measured",
        severity: "REQUIRED",
        expected: null,
        observed: null,
      },
    ],
    problems: [],
    unknown: ["checkpoint size has never been measured for any LUBER configuration"],
    generated_at: NOW,
    ...overrides,
  };
}

export function gates(rightsPassed = true): GateView[] {
  return [
    {
      name: "dataset_lock",
      status: "PASS",
      detail: "dataset ds-001 matches its lock",
      failure_code: null,
      offending_count: 0,
      offending_ids: [],
    },
    {
      name: "rights",
      status: rightsPassed ? "PASS" : "FAIL",
      detail: rightsPassed
        ? "every selected track is cleared for training"
        : "2 selected track(s) are not cleared for training",
      failure_code: rightsPassed ? null : "RIGHTS_GATE_FAILED",
      offending_count: rightsPassed ? 0 : 2,
      offending_ids: rightsPassed ? [] : ["trk-0002", "trk-0003"],
    },
    {
      name: "evaluation_leakage",
      status: "PASS",
      detail: "no evaluation-only material appears in the training selection",
      failure_code: null,
      offending_count: 0,
      offending_ids: [],
    },
  ];
}

export function checkpoint(overrides: Partial<CheckpointSummary> = {}): CheckpointSummary {
  return {
    checkpoint_id: "ckpt_1",
    run_id: "run_1",
    experiment_id: "exp_1",
    kind: "ADAPTER",
    is_real_model: true,
    status: "READY",
    step: 20,
    epoch: 1,
    size_bytes: 46_137_344,
    sha256: "a".repeat(64),
    checkpoint_format: "peft-adapter-safetensors",
    created_at: NOW,
    finalized_at: NOW,
    metrics_snapshot: { train_loss: 2.02 },
    can_evaluate: true,
    evaluate_blocked_reason: null,
    candidate_id: "cand_1",
    location_scheme: "file",
    location_present: true,
    ...overrides,
  };
}

export function mockCheckpoint(): CheckpointSummary {
  return checkpoint({
    checkpoint_id: "ckpt_mock",
    kind: "MOCK",
    is_real_model: false,
    can_evaluate: false,
    evaluate_blocked_reason:
      "This is a MOCK artifact from a dry run. It contains no trained weights and can never " +
      "become an evaluation candidate.",
    candidate_id: null,
    metrics_snapshot: {},
  });
}

export function worker(overrides: Partial<WorkerSummary> = {}): WorkerSummary {
  return {
    worker_id: "wrk_1",
    name: "rented-a100",
    backend_type: "remote-gpu",
    host_identity: "fingerprint-a100-01",
    worker_class: "GPU_TRAINING_READY",
    remote_classification: "CUDA_TRAINING",
    status: "BUSY",
    liveness: "ONLINE",
    last_heartbeat: NOW,
    heartbeat_age_seconds: 4,
    max_concurrent_runs: 1,
    active_run_ids: ["run_1"],
    capabilities: {
      gpu_vendor: "NVIDIA",
      gpu_model: "NVIDIA A100-SXM4-40GB",
      gpu_count: 1,
      vram_total_mb: 40_960,
      system_ram_mb: 131_072,
      cpu_count: 32,
      cuda_available: true,
      cuda_version: "12.4",
      driver_version: "550.90.07",
      torch_version: "2.5.1+cu124",
      python_version: "3.11.9",
      bf16_supported: true,
      free_disk_mb: 780_000,
      reported_by: "luber-remote probe on Linux x86_64",
      reported_at: NOW,
    },
    protocol_version: "luber-remote/1",
    capability_signature: "cap" + "0".repeat(61),
    created_at: NOW,
    has_credentials: true,
    ...overrides,
  };
}

/** The local Mac: nothing about a GPU has ever been measured on it. */
export function unprobedWorker(): WorkerSummary {
  return worker({
    worker_id: "wrk_mac",
    name: "operator-mac",
    backend_type: "dry-run",
    worker_class: "DEVELOPMENT_ONLY",
    remote_classification: "DEVELOPMENT_ONLY",
    status: "ONLINE",
    liveness: "UNKNOWN",
    last_heartbeat: null,
    heartbeat_age_seconds: null,
    active_run_ids: [],
    has_credentials: false,
    protocol_version: null,
    capability_signature: null,
    capabilities: {
      gpu_vendor: null,
      gpu_model: null,
      gpu_count: null,
      vram_total_mb: null,
      system_ram_mb: 36_864,
      cpu_count: 12,
      cuda_available: null,
      cuda_version: null,
      driver_version: null,
      torch_version: null,
      python_version: "3.11.9",
      bf16_supported: null,
      free_disk_mb: 220_000,
      reported_by: "luber-remote probe on Darwin arm64",
      reported_at: NOW,
    },
  });
}

export function runDetail(overrides: Partial<RunDetail> = {}): RunDetail {
  const run = overrides.run ?? runSummary();
  return {
    run,
    experiment: null,
    base_model: null,
    timeline: timeline(run.status),
    config: trainingConfig(),
    config_sha256: "c".repeat(64),
    training_plan_sha256: "p".repeat(64),
    dataset: {
      dataset_id: "ds-001",
      dataset_lock_sha256: "d".repeat(64),
      curation_id: "cur-001",
      curation_lock_sha256: "e".repeat(64),
      curated_manifest_sha256: "e".repeat(64),
      manifest_artifact_ref: "curation://cur-001/curated_manifest",
      sampling_weights_sha256: null,
      selected_track_count: 240,
      selected_hours: 13.5,
    },
    dataset_available: true,
    worker: worker(),
    heartbeat: heartbeat(),
    remote: {
      available: false,
      unavailable_reason:
        "This console holds no transport to a worker. Remote credentials belong to the " +
        "operator CLI, so reconciliation and cancellation delivery run there.",
      worker_state: null,
      implied_run_status: null,
      detail: "",
      exit_code: null,
      failure_code: null,
      lease_id: null,
      process_alive: null,
      updated_at: null,
      started_at: null,
      completed_at: null,
      cancel_requested_at: null,
      protocol_version: null,
      plan_sha256: null,
    },
    staging: {
      available: false,
      unavailable_reason: "No artifact manifest has been received for this run.",
      total_entries: 0,
      unique_contents: 0,
      total_bytes: 0,
      presence_checked: false,
      present_entries: 0,
      missing_entries: 0,
      roles: {},
      built_at: null,
    },
    control_preflight: preflight(),
    training_preflight: trainingPreflight(),
    canary: canaryRun(),
    capacity: capacity(),
    pilot: pilot(),
    remote_preflight: unavailablePreflight("The worker has not recorded a preflight."),
    gates: gates(),
    gates_available: true,
    gates_unavailable_reason: null,
    metrics: [metricSeries()],
    telemetry: [],
    progress: {
      latest_step: 12,
      latest_epoch: 1,
      total_epochs: 30,
      elapsed_seconds: 3600,
      latest_train_loss: 1.85,
      latest_learning_rate: 0.0001,
      latest_checkpoint_id: "ckpt_1",
      eta_seconds: null,
      eta_reason:
        "The installed trainer measures length in epochs and records no step total, so " +
        "remaining time cannot be derived from what has been measured.",
    },
    checkpoints: [checkpoint()],
    evaluations: [],
    reproducibility: {
      luber_commit: "a6b4a7fafdd99f12e78fcda1d9096a6ac5bf0374",
      luber_dirty: false,
      ace_step_commit: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
      base_model_id: "model_1",
      base_model_upstream_commit: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
      dataset_lock_sha256: "d".repeat(64),
      curation_lock_sha256: "e".repeat(64),
      curated_manifest_sha256: "e".repeat(64),
      training_config_sha256: "c".repeat(64),
      training_plan_sha256: "p".repeat(64),
      environment_lock_digest: "f".repeat(64),
      worker_capability_signature: "cap" + "0".repeat(61),
      python_version: "3.11.9",
      torch_version: "2.5.1+cu124",
    },
    cost: {
      provider: "example-gpu-cloud",
      instance_type: "1xA100-40GB",
      hourly_rate: 1.29,
      currency: "USD",
      wall_seconds: 3600,
      gpu_seconds: null,
      estimated_cost: 1.29,
      actual_cost: null,
      unknown: [],
    },
    audit_events: [
      {
        timestamp: NOW,
        event: "RUN_CREATED",
        entity_id: run.run_id,
        entity_kind: "run",
        metadata: { backend: run.execution_backend },
      },
    ],
    actions: actions(),
    environment_lock: {},
    ...overrides,
  };
}

export function overview(overrides: Partial<Overview> = {}): Overview {
  return {
    generated_at: NOW,
    registry_present: true,
    experiments: { total: 2, by_state: { DRAFT: 1, BLOCKED: 1 } },
    runs: { total: 8, by_state: { RUNNING: 1, FAILED: 2, LOST: 1, COMPLETED: 4 } },
    workers: { total: 4, by_state: { ONLINE: 2, STALE: 1, UNKNOWN: 1 } },
    worker_classes: {
      total: 4,
      by_state: { GPU_TRAINING_READY: 2, DEVELOPMENT_ONLY: 1, UNVERIFIED: 1 },
    },
    checkpoints: { total: 2, by_state: { READY: 2 } },
    checkpoint_kinds: { total: 2, by_state: { ADAPTER: 1, MOCK: 1 } },
    evaluations: { total: 3, by_state: { COMPLETED: 3 } },
    qualifications: {
      total: 3,
      by_state: { QUALIFIED: 1, REJECTED: 1, HUMAN_REVIEW_REQUIRED: 1 },
    },
    system: [
      { name: "training registry", status: "OK", detail: "readable" },
      {
        name: "remote worker transport",
        status: "UNAVAILABLE",
        detail: "This console holds no transport to a worker. Remote credentials belong to the operator CLI.",
      },
      {
        name: "training capability",
        status: "DEGRADED",
        detail: "1 of 2 probe-verified worker(s) reporting within the liveness window",
      },
    ],
    empty_reason: null,
    ...overrides,
  };
}

export function qualification(outcome: string): Qualification {
  return {
    evaluation_id: "eval_1",
    candidate_id: "cand_1",
    outcome,
    policy_id: "NEUTRAL_CONSERVATIVE",
    policy_version: "1",
    policy_digest: "q".repeat(64),
    reasons:
      outcome === "REJECTED"
        ? ["lyric intelligibility regressed by a MAJOR margin"]
        : outcome === "HUMAN_REVIEW_REQUIRED"
          ? ["the hypothesis is about vocal phrasing, which no technical metric measures"]
          : ["every hard gate passed and no metric regressed"],
    passed_gates: ["reliability", "silence_rate"],
    failed_gates: outcome === "REJECTED" ? ["lyric_intelligibility"] : [],
    inconclusive_gates: outcome === "HUMAN_REVIEW_REQUIRED" ? ["phrasing"] : [],
    gate_outcomes: [
      {
        name: "reliability",
        passed: true,
        detail: "0 failures in 9 generations",
        severity: "NONE",
        inconclusive: false,
      },
    ],
    hypothesis_status: outcome === "QUALIFIED" ? "ADDRESSED" : "NOT_MEASURABLE",
    human_review_required_for: outcome === "HUMAN_REVIEW_REQUIRED" ? ["vocal_phrasing"] : [],
    decided_at: NOW,
  };
}

export function evaluationDetail(outcome: string): EvaluationDetail {
  return {
    evaluation: {
      evaluation_id: "eval_1",
      status: "COMPLETED",
      mode: "RAW_MODEL",
      suite_id: "p20-frozen",
      suite_version: "1",
      suite_digest: "s".repeat(64),
      policy_digest: "q".repeat(64),
      candidate_id: "cand_1",
      checkpoint_id: "ckpt_1",
      run_id: "run_1",
      experiment_id: "exp_1",
      baseline_label: "production baseline",
      candidate_label: "candidate",
      experiment_hypothesis: "Curating out trot-adjacent material improves phrasing.",
      seeds: [11, 22, 33],
      started_at: NOW,
      completed_at: NOW,
      failed_at: null,
      cancelled_at: null,
      error: null,
      qualification_outcome: outcome,
      wall_seconds: 600,
      gpu_seconds: 540,
    },
    lineage: {
      candidate_id: "cand_1",
      checkpoint_id: "ckpt_1",
      run_id: "run_1",
      experiment_id: "exp_1",
    },
    qualification: qualification(outcome),
    comparisons: [
      {
        metric_name: "lyric_intelligibility",
        verdict: outcome === "REJECTED" ? "REGRESSED" : "UNCHANGED",
        baseline_value: 0.22,
        candidate_value: outcome === "REJECTED" ? 0.41 : 0.23,
        delta: outcome === "REJECTED" ? 0.19 : 0.01,
        severity: outcome === "REJECTED" ? "MAJOR" : "NONE",
        detail: "word error rate over the frozen suite",
      },
    ],
    regressions:
      outcome === "REJECTED"
        ? [
            {
              metric_name: "lyric_intelligibility",
              verdict: "REGRESSED",
              baseline_value: 0.22,
              candidate_value: 0.41,
              delta: 0.19,
              severity: "MAJOR",
              detail: "word error rate over the frozen suite",
            },
          ]
        : [],
    promotion_reviews:
      outcome === "QUALIFIED"
        ? [
            {
              review_id: "rev_1",
              candidate_id: "cand_1",
              evaluation_id: "eval_1",
              qualification_outcome: outcome,
              decision: "HOLD",
              decided_by: "operator",
              rationale: "Qualified, but held until a second dataset confirms it.",
              decided_at: NOW,
            },
          ]
        : [],
    human_review:
      outcome === "HUMAN_REVIEW_REQUIRED"
        ? {
            required: true,
            mode: "LIGHT_AB",
            reason: "the hypothesis is about vocal phrasing",
            case_count: 3,
            dimensions: ["vocal_phrasing"],
            status: "PENDING",
            package_available: true,
          }
        : null,
    checkpoint: checkpoint(),
    run: runSummary(),
    experiment: null,
    audit_events: [],
    report_available: true,
  };
}
