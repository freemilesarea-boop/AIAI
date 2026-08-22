/**
 * The operator wire contract, as TypeScript.
 *
 * Mirrors `luber_api.ops.schemas`. Written out rather than inferred from
 * `unknown` for the reason Step 64 gives: a console whose entities are
 * `Record<string, any>` cannot tell anyone that a field it renders has
 * gone, and the first place that shows up is a blank panel where a plan
 * hash used to be.
 *
 * Two conventions carry across from the Python side and matter here:
 *
 * `null` means nobody measured it. It is never rendered as zero, never
 * as a dash that could be read as "none", and never as a pass — the UI
 * renders UNKNOWN and says so.
 *
 * Local and remote state are separate types on purpose. `RunSummary.status`
 * is the control plane's record; `RemoteState.workerState` is what the
 * worker believes. They can legitimately disagree, and the disagreement
 * is the information.
 */

export type RunStatus =
  | "DRAFT"
  | "VALIDATING"
  | "QUEUED"
  | "STARTING"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED"
  | "LOST";

export type Liveness = "ONLINE" | "STALE" | "OFFLINE" | "UNKNOWN";
export type CheckStatus = "PASS" | "FAIL" | "UNKNOWN";
export type PreflightStatus = "PASS" | "BLOCKED" | "FAIL" | "UNKNOWN";
export type GateStatus = "PASS" | "FAIL" | "NOT_EVALUATED";
export type SystemStatus = "OK" | "DEGRADED" | "UNAVAILABLE" | "UNKNOWN";

export interface Page {
  total: number;
  limit: number;
  offset: number;
  returned: number;
}

export interface SystemCheck {
  name: string;
  status: SystemStatus;
  detail: string;
}

export interface CountBreakdown {
  total: number;
  by_state: Record<string, number>;
}

export interface Overview {
  generated_at: string;
  registry_present: boolean;
  experiments: CountBreakdown;
  runs: CountBreakdown;
  workers: CountBreakdown;
  worker_classes: CountBreakdown;
  checkpoints: CountBreakdown;
  checkpoint_kinds: CountBreakdown;
  evaluations: CountBreakdown;
  qualifications: CountBreakdown;
  system: SystemCheck[];
  empty_reason: string | null;
}

export interface ExperimentSummary {
  experiment_id: string;
  name: string;
  hypothesis: string;
  description: string;
  base_model_id: string;
  status: string;
  blocked_reason: string;
  dataset_lock_ref: string | null;
  curation_lock_ref: string | null;
  operator: string;
  tags: string[];
  created_at: string;
  run_count: number;
  latest_run_id: string | null;
  latest_run_status: string | null;
}

export interface ExperimentList {
  items: ExperimentSummary[];
  page: Page;
  available_statuses: string[];
  available_base_models: string[];
  available_tags: string[];
}

export interface ModelBaseline {
  model_id: string;
  provider: string;
  model_family: string;
  model_name: string;
  model_version: string;
  upstream_commit: string;
  architecture: string;
  training_strategy_support: string[];
  checkpoint_sha256: string | null;
  identity_basis: string;
  stage: string;
  created_at: string;
}

export interface BaselineResponse {
  production: ModelBaseline[];
  all_models: ModelBaseline[];
  note: string;
}

export interface CandidateSummary {
  candidate_id: string;
  run_id: string;
  checkpoint_id: string;
  experiment_id: string;
  status: string;
  created_at: string;
  notes: string;
}

export interface AuditEvent {
  timestamp: string;
  event: string;
  entity_id: string;
  entity_kind: string;
  metadata: Record<string, unknown>;
}

export interface ExperimentDetail {
  experiment: ExperimentSummary;
  base_model: ModelBaseline | null;
  runs: RunSummary[];
  candidates: CandidateSummary[];
  evaluations: EvaluationSummary[];
  qualifications: Qualification[];
  audit_events: AuditEvent[];
}

export interface FailureView {
  code: string;
  headline: string;
  guidance: string;
  raw_message: string | null;
  confident: boolean;
}

export interface MetricPoint {
  step: number | null;
  epoch: number | null;
  value: number;
  timestamp: string;
}

export interface MetricSeries {
  metric_name: string;
  unit: string;
  sources: string[];
  points: MetricPoint[];
  total_points: number;
  sampled: boolean;
  last_value: number | null;
}

export interface RunSummary {
  run_id: string;
  experiment_id: string;
  experiment_name: string;
  base_model_id: string;
  status: RunStatus;
  execution_backend: string;
  worker_id: string | null;
  worker_name: string | null;
  created_at: string;
  queued_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  failed_at: string | null;
  cancelled_at: string | null;
  duration_seconds: number | null;
  parent_run_id: string | null;
  checkpoint_count: number;
  latest_metric: MetricPoint | null;
  latest_metric_name: string | null;
  failure: FailureView | null;
  cancel_requested_at: string | null;
}

export interface RunList {
  items: RunSummary[];
  page: Page;
  available_statuses: string[];
  available_backends: string[];
}

export interface TimelineEntry {
  state: RunStatus;
  reached: boolean;
  at: string | null;
  current: boolean;
  terminal: boolean;
}

export interface TrainingConfig {
  strategy: string;
  learning_rate: number;
  batch_size: number;
  gradient_accumulation: number;
  epochs: number;
  warmup_steps: number;
  weight_decay: number;
  max_grad_norm: number;
  seed: number;
  optimizer_type: string;
  scheduler_type: string;
  gradient_checkpointing: boolean;
  offload_encoder: boolean;
  shift: number;
  num_inference_steps: number;
  rank: number;
  alpha: number;
  dropout: number;
  target_modules: string[];
  bias: string;
  attention_type: string;
  precision: string;
  num_devices: number;
  checkpoint_every_epochs: number;
  log_every_steps: number;
  log_heavy_every_steps: number;
  sample_every_n_epochs: number;
  num_workers: number;
  pin_memory: boolean;
  prefetch_factor: number;
  persistent_workers: boolean;
  ace_step_commit: string;
}

export interface DatasetRef {
  dataset_id: string;
  dataset_lock_sha256: string;
  curation_id: string;
  curation_lock_sha256: string;
  curated_manifest_sha256: string;
  manifest_artifact_ref: string;
  sampling_weights_sha256: string | null;
  selected_track_count: number;
  selected_hours: number;
}

export interface GateView {
  name: string;
  status: GateStatus;
  detail: string;
  failure_code: string | null;
  offending_count: number;
  offending_ids: string[];
}

export interface PreflightCheck {
  name: string;
  status: CheckStatus;
  detail: string;
  severity: string;
  expected: string | null;
  observed: string | null;
}

export interface Preflight {
  available: boolean;
  unavailable_reason: string | null;
  status: PreflightStatus;
  checks: PreflightCheck[];
  problems: string[];
  unknown: string[];
  generated_at: string | null;
}

/**
 * Phase 33 statuses. `UNVERIFIED` is not a shade of ready — a check
 * nobody could perform is not a check that passed, and the console
 * renders it as its own state rather than as a softer green.
 */
export type TrainingPreflightStatus = "READY" | "BLOCKED" | "UNVERIFIED";
export type TrainingCheckStatus = "PASS" | "FAIL" | "UNKNOWN" | "NOT_APPLICABLE";
export type CapacitySource = "MEASURED" | "DERIVED" | "ESTIMATED" | "UNKNOWN";
export type CanaryStatus = "PASSED" | "FAILED" | "BLOCKED" | "NOT_RUN";

export interface TrainingPreflightCheck {
  name: string;
  group: string;
  status: TrainingCheckStatus;
  detail: string;
  reason: string | null;
  mandatory: boolean;
}

export interface CapacityEvidence {
  name: string;
  source: CapacitySource;
  value_mb: number | null;
  detail: string;
  derivation: string;
  /** Apple unified memory, shared with the OS. Never dedicated VRAM. */
  unified_memory: boolean;
}

export interface TrainingPreflight {
  available: boolean;
  unavailable_reason: string | null;
  status: TrainingPreflightStatus;
  intent: string;
  plan_digest: string | null;
  execution_location: string | null;
  execution_device: string | null;
  torch_device: string | null;
  resolved_precision: string | null;
  optimizer: string | null;
  worker_identity: string | null;
  target_label: string | null;
  capability_digest: string | null;
  dataset_status: string;
  dependency_status: string;
  storage_status: string;
  checkpoint_status: string;
  canary_status: string;
  capacity_status: string;
  checks: TrainingPreflightCheck[];
  capacity: CapacityEvidence[];
  blocking_reasons: string[];
  unverified: string[];
  warnings: string[];
  hardware: Record<string, unknown>;
  measured_at: string | null;
  policy_version: string | null;
}

export interface CanaryRun {
  available: boolean;
  unavailable_reason: string | null;
  status: CanaryStatus;
  mode: string | null;
  detail: string;
  steps: number | null;
  max_optimizer_steps: number | null;
  max_samples: number | null;
  max_epochs: number | null;
  dataset_kind: string | null;
  exit_code: number | null;
  seconds: number | null;
  checkpoint_ok: boolean | null;
  checkpoint_step: number | null;
  checkpoint_provenance_plan_digest: string | null;
  checkpoint_problems: string[];
  resume_ok: boolean | null;
  resume_detail: string;
}

/**
 * Phase 34. `UNVERIFIED` means nobody has an applicable measurement —
 * a gap in evidence, not a fault in the machine — and it is rendered as
 * its own state rather than as a softer pass.
 */
export type CapacityQualification = "QUALIFIED" | "MARGIN_LOW" | "INSUFFICIENT" | "UNVERIFIED";
export type MemoryDomain = "HOST" | "APPLE_UNIFIED" | "CUDA_DEVICE";
/**
 * A sampled peak is the largest value a sampler observed — a lower
 * bound. A runtime peak is a high-water mark the runtime kept. The
 * console must not render them identically.
 */
export type PeakKind = "RUNTIME_PEAK" | "SAMPLED_PEAK" | "NOT_AVAILABLE";
export type ProfileOutcome =
  | "COMPLETED"
  | "FAILED"
  | "PROFILE_TIMEOUT"
  | "BLOCKED"
  | "NOT_RUN";
export type Representativeness =
  | "REPRESENTATIVE"
  | "PARTIALLY_REPRESENTATIVE"
  | "NOT_REPRESENTATIVE"
  | "UNKNOWN";

export interface MemoryPeak {
  domain: MemoryDomain;
  kind: PeakKind;
  source: CapacitySource;
  peak_mib: number | null;
  baseline_mib: number | null;
  growth_mib: number | null;
  total_mib: number | null;
  sample_count: number;
  detail: string;
}

export interface CapacityDomainVerdict {
  domain: MemoryDomain;
  qualification: CapacityQualification;
  peak_mib: number | null;
  peak_kind: string;
  required_mib: number | null;
  reserved_mib: number | null;
  budget_mib: number | null;
  total_mib: number | null;
  detail: string;
}

export interface MemoryProfile {
  profile_id: string;
  outcome: ProfileOutcome;
  representativeness: Representativeness;
  representativeness_detail: string;
  identity_digest: string | null;
  device: string | null;
  precision: string | null;
  optimizer: string | null;
  micro_batch_size: number | null;
  gradient_accumulation: number | null;
  effective_batch_size: number | null;
  lora_rank: number | null;
  gradient_checkpointing: boolean | null;
  latent_length: number | null;
  latent_seconds: number | null;
  encoder_length: number | null;
  peaks: MemoryPeak[];
  checkpoint_peak_mib: number | null;
  resume_peak_mib: number | null;
  optimizer_steps: number | null;
  not_observed: Record<string, string>;
  torch_version: string | null;
  ace_step_commit: string | null;
  measured_at: string | null;
  failure_reason: string;
  failure_kind: string;
}

export interface Capacity {
  available: boolean;
  unavailable_reason: string | null;
  qualification: CapacityQualification;
  device: string | null;
  policy_version: string | null;
  policy: Record<string, unknown>;
  applicability: string | null;
  applicability_detail: string;
  reasons: string[];
  domains: CapacityDomainVerdict[];
  evidence: CapacityEvidence[];
  profile: MemoryProfile | null;
  measured_at: string | null;
}

/**
 * Phase 35. The signal vocabulary tops out at `VALID_SIGNAL`: tens of
 * steps cannot support a convergence or quality claim, so there is no
 * word here for one.
 */
export type PilotOutcome =
  | "COMPLETED_VALID_SIGNAL"
  | "COMPLETED_INSUFFICIENT_SIGNAL"
  | "BLOCKED"
  | "FAILED_NUMERIC"
  | "FAILED_RUNTIME"
  | "CANCELLED"
  | "TIMEOUT"
  | "NOT_RUN";
export type TrainingSignal =
  | "VALID_SIGNAL"
  | "NUMERICALLY_UNSTABLE"
  | "NO_UPDATE"
  | "INSUFFICIENT_EVIDENCE";
/** A synthetic fixture validates the mechanism and is never real-data evidence. */
export type PilotDatasetKind = "REAL_RIGHTS_CLEARED" | "SYNTHETIC_FIXTURE" | "UNKNOWN";

export interface LossPoint {
  step: number;
  loss: number;
  epoch: number | null;
  learning_rate: number | null;
  grad_norm: number | null;
  elapsed_seconds: number | null;
  segment: string;
  finite: boolean;
}

export interface PilotSegment {
  name: string;
  completed_steps: number;
  first_step: number | null;
  last_step: number | null;
  checkpoint_id: string | null;
  resumed_from: string | null;
  exit_code: number | null;
  wall_seconds: number | null;
  detail: string;
}

export interface Pilot {
  available: boolean;
  unavailable_reason: string | null;
  pilot_id: string | null;
  outcome: PilotOutcome;
  signal: TrainingSignal;
  signal_detail: string;
  failure: string | null;
  failure_detail: string;
  dataset_kind: PilotDatasetKind;
  expected_steps: number;
  completed_steps: number;
  step_ceiling: number;
  within_budget: boolean;
  device: string | null;
  precision: string | null;
  lora_rank: number | null;
  micro_batch_size: number | null;
  gradient_accumulation: number | null;
  latent_length: number | null;
  seed: number | null;
  plan_digest: string | null;
  dataset_manifest_digest: string | null;
  capacity_qualification: string | null;
  capacity_profile_id: string | null;
  preflight_status: string | null;
  loss: LossPoint[];
  loss_statistics: Record<string, unknown>;
  parameters: Record<string, unknown>;
  gradients: Record<string, unknown>;
  segments: PilotSegment[];
  checkpoint: Record<string, unknown> | null;
  resume: Record<string, unknown> | null;
  artifact_class: string[];
  started_at: string | null;
  finished_at: string | null;
  wall_seconds: number | null;
}

export interface Reproducibility {
  luber_commit: string | null;
  luber_dirty: boolean | null;
  ace_step_commit: string | null;
  base_model_id: string;
  base_model_upstream_commit: string | null;
  dataset_lock_sha256: string;
  curation_lock_sha256: string;
  curated_manifest_sha256: string;
  training_config_sha256: string;
  training_plan_sha256: string | null;
  environment_lock_digest: string | null;
  worker_capability_signature: string | null;
  python_version: string | null;
  torch_version: string | null;
}

export interface Cost {
  provider: string | null;
  instance_type: string | null;
  hourly_rate: number | null;
  currency: string | null;
  wall_seconds: number | null;
  gpu_seconds: number | null;
  estimated_cost: number | null;
  actual_cost: number | null;
  unknown: string[];
}

export interface Staging {
  available: boolean;
  unavailable_reason: string | null;
  total_entries: number;
  unique_contents: number;
  total_bytes: number;
  presence_checked: boolean;
  present_entries: number;
  missing_entries: number;
  roles: Record<string, number>;
  built_at: string | null;
}

export interface RemoteState {
  available: boolean;
  unavailable_reason: string | null;
  worker_state: string | null;
  implied_run_status: string | null;
  detail: string;
  exit_code: number | null;
  failure_code: string | null;
  lease_id: string | null;
  process_alive: boolean | null;
  updated_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  cancel_requested_at: string | null;
  protocol_version: string | null;
  plan_sha256: string | null;
}

export interface GpuTelemetry {
  index: number;
  utilization_pct: number | null;
  memory_used_mb: number | null;
  memory_total_mb: number | null;
  temperature_c: number | null;
  power_w: number | null;
  sampled_at: string | null;
}

export interface Heartbeat {
  available: boolean;
  unavailable_reason: string | null;
  timestamp: string | null;
  age_seconds: number | null;
  liveness: Liveness;
  worker_state: string | null;
  active_run_id: string | null;
  health: string | null;
  uptime_seconds: number | null;
  free_disk_mb: number | null;
  gpu: GpuTelemetry[];
  detail: string;
}

export interface CheckpointSummary {
  checkpoint_id: string;
  run_id: string;
  experiment_id: string;
  kind: string;
  is_real_model: boolean;
  status: string;
  step: number | null;
  epoch: number | null;
  size_bytes: number | null;
  sha256: string | null;
  checkpoint_format: string;
  created_at: string;
  finalized_at: string | null;
  metrics_snapshot: Record<string, number>;
  can_evaluate: boolean;
  evaluate_blocked_reason: string | null;
  candidate_id: string | null;
  location_scheme: string | null;
  location_present: boolean | null;
}

export interface CheckpointList {
  items: CheckpointSummary[];
  page: Page;
  available_statuses: string[];
  available_kinds: string[];
}

export interface CheckpointDetail {
  checkpoint: CheckpointSummary;
  run: RunSummary | null;
  experiment: ExperimentSummary | null;
  evaluations: EvaluationSummary[];
  qualifications: Qualification[];
  audit_events: AuditEvent[];
}

export interface LogView {
  available: boolean;
  unavailable_reason: string | null;
  stream: "stdout" | "stderr";
  offset: number;
  next_offset: number;
  size_bytes: number;
  eof: boolean;
  truncated: boolean;
  text: string;
  from_tail: boolean;
}

export interface RunProgress {
  latest_step: number | null;
  latest_epoch: number | null;
  total_epochs: number | null;
  elapsed_seconds: number | null;
  latest_train_loss: number | null;
  latest_learning_rate: number | null;
  latest_checkpoint_id: string | null;
  eta_seconds: number | null;
  eta_reason: string;
}

export interface ActionAvailability {
  action: string;
  label: string;
  available: boolean;
  reason: string;
  confirmation: string;
  destructive: boolean;
}

export interface WorkerCapabilities {
  gpu_vendor: string | null;
  gpu_model: string | null;
  gpu_count: number | null;
  vram_total_mb: number | null;
  system_ram_mb: number | null;
  cpu_count: number | null;
  cuda_available: boolean | null;
  cuda_version: string | null;
  driver_version: string | null;
  torch_version: string | null;
  python_version: string | null;
  bf16_supported: boolean | null;
  free_disk_mb: number | null;
  reported_by: string;
  reported_at: string | null;
}

export interface WorkerSummary {
  worker_id: string;
  name: string;
  backend_type: string;
  host_identity: string;
  worker_class: string;
  remote_classification: string | null;
  status: string;
  liveness: Liveness;
  last_heartbeat: string | null;
  heartbeat_age_seconds: number | null;
  max_concurrent_runs: number;
  active_run_ids: string[];
  capabilities: WorkerCapabilities;
  protocol_version: string | null;
  capability_signature: string | null;
  created_at: string;
  has_credentials: boolean;
}

export interface WorkerList {
  items: WorkerSummary[];
  page: Page;
  available_classes: string[];
  available_liveness: string[];
}

export interface WorkerDetail {
  worker: WorkerSummary;
  heartbeat: Heartbeat;
  software_environment: Record<string, string>;
  recent_runs: RunSummary[];
  audit_events: AuditEvent[];
  unknown_capabilities: string[];
}

export interface WorkerCompatibility {
  worker: WorkerSummary;
  compatible: boolean;
  reasons: string[];
}

export interface RunDetail {
  run: RunSummary;
  experiment: ExperimentSummary | null;
  base_model: ModelBaseline | null;
  timeline: TimelineEntry[];
  config: TrainingConfig;
  config_sha256: string;
  training_plan_sha256: string | null;
  dataset: DatasetRef;
  dataset_available: boolean;
  worker: WorkerSummary | null;
  heartbeat: Heartbeat;
  remote: RemoteState;
  staging: Staging;
  control_preflight: Preflight;
  remote_preflight: Preflight;
  training_preflight: TrainingPreflight;
  canary: CanaryRun;
  capacity: Capacity;
  pilot: Pilot;
  gates: GateView[];
  gates_available: boolean;
  gates_unavailable_reason: string | null;
  metrics: MetricSeries[];
  telemetry: MetricSeries[];
  progress: RunProgress;
  checkpoints: CheckpointSummary[];
  evaluations: EvaluationSummary[];
  reproducibility: Reproducibility;
  cost: Cost;
  audit_events: AuditEvent[];
  actions: ActionAvailability[];
  environment_lock: Record<string, unknown>;
}

export interface EvaluationSummary {
  evaluation_id: string;
  status: string;
  mode: string;
  suite_id: string;
  suite_version: string;
  suite_digest: string;
  policy_digest: string;
  candidate_id: string;
  checkpoint_id: string;
  run_id: string;
  experiment_id: string;
  baseline_label: string;
  candidate_label: string;
  experiment_hypothesis: string;
  seeds: number[];
  started_at: string | null;
  completed_at: string | null;
  failed_at: string | null;
  cancelled_at: string | null;
  error: string | null;
  qualification_outcome: string | null;
  wall_seconds: number | null;
  gpu_seconds: number | null;
}

export interface EvaluationList {
  items: EvaluationSummary[];
  page: Page;
  available_statuses: string[];
  available_outcomes: string[];
}

export interface GateOutcome {
  name: string;
  passed: boolean;
  detail: string;
  severity: string;
  inconclusive: boolean;
}

export interface Qualification {
  evaluation_id: string;
  candidate_id: string;
  outcome: string;
  policy_id: string;
  policy_version: string;
  policy_digest: string;
  reasons: string[];
  passed_gates: string[];
  failed_gates: string[];
  inconclusive_gates: string[];
  gate_outcomes: GateOutcome[];
  hypothesis_status: string;
  human_review_required_for: string[];
  decided_at: string;
}

export interface Comparison {
  metric_name: string;
  verdict: string;
  baseline_value: number | null;
  candidate_value: number | null;
  delta: number | null;
  severity: string;
  detail: string;
}

export interface PromotionReview {
  review_id: string;
  candidate_id: string;
  evaluation_id: string;
  qualification_outcome: string;
  decision: string;
  decided_by: string;
  rationale: string;
  decided_at: string;
}

export interface HumanReview {
  required: boolean;
  mode: string;
  reason: string;
  case_count: number;
  dimensions: string[];
  status: string;
  package_available: boolean;
}

export interface EvaluationDetail {
  evaluation: EvaluationSummary;
  lineage: Record<string, string>;
  qualification: Qualification | null;
  comparisons: Comparison[];
  regressions: Comparison[];
  promotion_reviews: PromotionReview[];
  human_review: HumanReview | null;
  checkpoint: CheckpointSummary | null;
  run: RunSummary | null;
  experiment: ExperimentSummary | null;
  audit_events: AuditEvent[];
  report_available: boolean;
}

export interface BuildOption {
  build_id: string;
  identity: string;
  lock_sha256: string;
  manifest_sha256: string;
  track_count: number | null;
  hours: number | null;
  created_at: string | null;
  source_dataset_lock_sha256: string | null;
}

export interface PresetOption {
  name: string;
  intent: string;
  config: TrainingConfig;
}

export interface Catalogue {
  datasets: BuildOption[];
  curations: BuildOption[];
  dataset_problems: string[];
  curation_problems: string[];
  presets: PresetOption[];
  backends: string[];
  base_models: ModelBaseline[];
}

export interface ActionResult {
  action: string;
  performed: boolean;
  run_status: string | null;
  detail: string;
  outcome: string | null;
  created_id: string | null;
}

export interface ComparisonRow {
  checkpoint: CheckpointSummary;
  evaluation: EvaluationSummary | null;
  qualification: Qualification | null;
  metrics: Record<string, number>;
  training_context: Record<string, number>;
}

export interface CheckpointComparison {
  rows: ComparisonRow[];
  metric_names: string[];
  note: string;
}

/* ── compute targets (Phase 32) ─────────────────────────────────────── */

/**
 * One place a workload could run.
 *
 * `location` and `device` are separate fields because they are separate
 * questions: this deployment has a local target that is not CUDA, and
 * will have a remote one that is. Neither implies the other.
 *
 * `precisions: []` means nobody measured — never "none work".
 */
export interface ComputeTarget {
  name: string;
  /** LOCAL | REMOTE */
  location: string;
  /** CPU | MPS | CUDA */
  device: string;
  /** READY | NOT_AVAILABLE | NOT_CONNECTED | UNPROBED */
  status: string;
  detail: string;
  memory_mb: number | null;
  precisions: string[];
  workloads: string[];
  limitations: string[];
  planned: boolean;
  capability_digest: string | null;
}

export interface ComputeTargets {
  at: string;
  summary: string;
  targets: ComputeTarget[];
  local_training_concurrency: number;
  capability_schema_version: string;
  execution_placement_policy_version: string;
}
