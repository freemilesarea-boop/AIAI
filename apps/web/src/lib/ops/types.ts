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
