/**
 * What the inference console receives.
 *
 * Mirrors `luber_api.ops.inference_schemas`. Two properties of that
 * module survive into these types on purpose.
 *
 * **No field can hold user content.** There is no `prompt`, no `lyrics`,
 * no `title`, no user identity — not because the UI avoids rendering
 * them, but because nothing that reaches the browser has anywhere to
 * put them.
 *
 * **A rate is never a bare number.** `Rate` carries its numerator and
 * denominator and a status, so a component cannot render "4%" without
 * being able to render "12 of 300", and an empty window is `NO_DATA`
 * rather than zero.
 */

export interface Versions {
  observability_schema_version: string;
  aggregation_version: string;
  regression_engine_version: string;
  incident_policy_version: string;
}

export interface Window {
  start: string;
  end: string;
  duration_seconds: number;
}

/** OK when there is something to divide; NO_DATA when the denominator is zero. */
export type MetricStatus = "OK" | "NO_DATA" | "INSUFFICIENT_DATA";

export interface Rate {
  name: string;
  numerator: number;
  denominator: number;
  excluded: number;
  value: number | null;
  percent: number | null;
  status: MetricStatus;
  render: string;
}

export interface Average {
  name: string;
  value: number | null;
  count: number;
  status: MetricStatus;
}

export interface Distribution {
  name: string;
  count: number;
  p50: number | null;
  p90: number | null;
  p95: number | null;
  p99: number | null;
  max: number | null;
  mean: number | null;
  status: MetricStatus;
}

export interface Coverage {
  observations: number;
  with_qc_data: number;
  without_qc_data: number;
  complete: boolean;
  partial: boolean;
  boundary_commit: string;
  note: string | null;
}

export interface Counters {
  generation_requests: number;
  completed_generations: number;
  failed_generations: number;
  cancelled_generations: number;
  without_qc_data: number;
  provider_calls: number;
  candidates_generated: number;
  quality_retries: number;
  retry_exhaustions: number;
  candidate_rejections: number;
  first_candidate_accepted: number;
  qc_observed: number;
  finding_counts: Record<string, number>;
  soft_finding_counts: Record<string, number>;
  failure_code_counts: Record<string, number>;
  data_quality_counts: Record<string, number>;
}

export interface Summary extends Versions {
  window: Window;
  segment: Record<string, string>;
  sample_count: number;
  counters: Counters;
  overview: Record<string, Rate>;
  rates: Record<string, Rate>;
  averages: Record<string, Average>;
  latency: Record<string, Distribution>;
  /** Rejections and advisories, deliberately in separate maps. */
  findings: { critical: Record<string, number>; soft: Record<string, number> };
  data_quality: Record<string, number>;
  coverage: Coverage;
}

export interface Marker {
  kind: string;
  occurred_at: string;
  label: string;
  detail: Record<string, unknown>;
  caveat: string;
}

export interface Overview extends Versions {
  window: Window;
  summary: Summary;
  open_incidents: number;
  critical_incidents: number;
  markers: Marker[];
  automatic_remediation: string;
}

export interface TrendPoint {
  start: string;
  end: string;
  sample_count: number;
  /** `null` where a bucket had no samples. Never drawn as zero. */
  values: Record<string, number | null>;
}

export interface Trend extends Versions {
  window: Window;
  segment: Record<string, string>;
  metrics: string[];
  bucket_seconds: number;
  points: TrendPoint[];
  has_data: boolean;
}

export type Severity = "INFO" | "MINOR" | "MAJOR" | "CRITICAL";
export type IncidentStatus = "OPEN" | "ACKNOWLEDGED" | "RESOLVED" | "DISMISSED";
export type Category = "AVAILABILITY" | "QUALITY" | "EFFICIENCY";

export interface Regression extends Versions {
  finding_type: string;
  category: Category;
  metric: string;
  segment: Record<string, string>;
  segment_label: string;
  status: string;
  severity: Severity;
  baseline_value: number | null;
  current_value: number | null;
  absolute_delta: number | null;
  relative_delta: number | null;
  baseline_numerator: number | null;
  baseline_denominator: number | null;
  current_numerator: number | null;
  current_denominator: number | null;
  baseline_sample_count: number;
  current_sample_count: number;
  baseline_window: Record<string, unknown>;
  current_window: Record<string, unknown>;
  quantile_fraction: number | null;
  thresholds: Record<string, unknown>;
  threshold_crossed: string | null;
  reason: string;
  explanation: string;
  recommendations: string[];
  partial_history: boolean;
}

export interface IncidentEvidence {
  observed_at: string;
  status: string;
  severity: Severity;
  baseline_value: number | null;
  current_value: number | null;
  absolute_delta: number | null;
  relative_delta: number | null;
  current_sample_count: number;
  baseline_sample_count: number;
  explanation: string;
}

export interface Incident extends Versions {
  incident_id: string;
  created_at: string;
  status: IncidentStatus;
  severity: Severity;
  peak_severity: Severity;
  finding_type: string;
  category: Category;
  metric: string;
  provider: string | null;
  provider_version: string | null;
  affected_dimensions: Record<string, string>;
  segment_label: string;
  baseline_window: Record<string, unknown>;
  current_window: Record<string, unknown>;
  first_seen: string | null;
  last_seen: string | null;
  occurrence_count: number;
  consecutive_clean: number;
  evidence: IncidentEvidence[];
  evidence_total: number;
  recommendations: string[];
  summary: string;
  acknowledged_at: string | null;
  acknowledged_by: string | null;
  resolved_at: string | null;
  dismissed_at: string | null;
  dismissed_by: string | null;
  dismissal_reason: string | null;
}

export interface IncidentList {
  total: number;
  limit: number;
  offset: number;
  items: Incident[];
}

export interface ProviderSummary {
  provider: string;
  provider_revision: string;
  model_name: string;
  model_version: string;
  sample_count: number;
  first_seen: string | null;
  last_seen: string | null;
  baseline_status: "READY" | "BASELINE_BUILDING";
  rates: Record<string, Rate>;
}

export interface Providers extends Versions {
  window: Window;
  providers: ProviderSummary[];
}

export interface SegmentRank {
  segment: Record<string, string>;
  segment_label: string;
  metric: string;
  value: number | null;
  numerator: number;
  denominator: number;
  render: string;
  sample_count: number;
}

export interface Segments extends Versions {
  window: Window;
  grouped_by: string[];
  metric: string;
  minimum_samples: number;
  segments: SegmentRank[];
  segments_considered: number;
  segments_below_minimum: number;
}

export interface GenerationListItem {
  generation_id: string;
  occurred_at: string;
  provider_revision: string;
  task_type: string;
  duration_bucket: string;
  generation_status: string;
  quality_retry_count: number | null;
  first_candidate_accepted: boolean | null;
  critical_findings: string[];
  total_latency_seconds: number | null;
}

export interface GenerationList {
  total: number;
  limit: number;
  offset: number;
  items: GenerationListItem[];
}

export interface GenerationAttempt {
  attempt_index: number;
  candidate_id: string;
  status: string;
  selection_status: string;
  attribution: string;
  seed: number | null;
  retry_reason: string | null;
  not_selected_reason: string | null;
  duration_seconds: number | null;
  critical_findings: string[];
  soft_findings: string[];
  provider_seconds: number | null;
  qc_seconds: number | null;
}

export interface GenerationTrace {
  generation_id: string;
  occurred_at: string;
  provider: string;
  provider_revision: string;
  task_type: string;
  duration_bucket: string;
  requested_duration_seconds: number | null;
  language: string;
  instrumental: string;
  generation_status: string;
  generation_failure_code: string | null;
  qc_policy: string;
  qc_data_available: boolean;
  qc_outcome: string | null;
  finishing_outcome: string | null;
  candidate_count: number | null;
  provider_call_count: number | null;
  quality_retry_count: number | null;
  selected_on_attempt: number | null;
  first_candidate_accepted: boolean | null;
  retry_exhausted: boolean | null;
  provider_latency_seconds: number | null;
  qc_latency_seconds: number | null;
  delivery_latency_seconds: number | null;
  total_latency_seconds: number | null;
  critical_findings: string[];
  soft_findings: string[];
  data_quality_issues: string[];
  attempts: GenerationAttempt[];
  explanation: string[];
}

export interface IngestStatus {
  observations: number;
  latest_observation_at: string | null;
  seconds_behind: number | null;
  stale: boolean;
  note: string | null;
}

export type WindowSize = "1h" | "24h" | "7d";

export interface InferenceFilters {
  window: WindowSize;
  provider?: string;
  revision?: string;
  task?: string;
  duration_bucket?: string;
}
