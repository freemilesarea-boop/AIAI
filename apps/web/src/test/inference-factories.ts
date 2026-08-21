/**
 * Synthetic console payloads, shaped exactly like the API's.
 *
 * Built as helpers rather than as one frozen blob so a test can say what
 * it is about — "a window with nothing in it", "a metric with four
 * samples" — instead of mutating a fixture and hoping.
 *
 * None of these contains a prompt, lyrics or a title, because no
 * response model has a field for one. A factory that could produce one
 * would be testing a shape the API cannot emit.
 */

import type {
  Coverage,
  GenerationList,
  GenerationTrace,
  Incident,
  IncidentList,
  IngestStatus,
  Overview,
  Providers,
  Rate,
  Regression,
  Segments,
  Summary,
  Trend,
  TrendPoint,
} from "@/lib/ops/inference-types";

const VERSIONS = {
  observability_schema_version: "luber-inference-observability/1",
  aggregation_version: "agg-v1",
  regression_engine_version: "regress-v1",
  incident_policy_version: "incident-v1",
};

const WINDOW = {
  start: "2026-08-21T11:00:00+00:00",
  end: "2026-08-21T12:00:00+00:00",
  duration_seconds: 3600,
};

export function rate(name: string, numerator: number, denominator: number): Rate {
  if (denominator === 0) {
    return {
      name,
      numerator: 0,
      denominator: 0,
      excluded: 0,
      value: null,
      percent: null,
      status: "NO_DATA",
      render: `${name}: NO_DATA (0 samples)`,
    };
  }
  const value = numerator / denominator;
  return {
    name,
    numerator,
    denominator,
    excluded: 0,
    value,
    percent: value * 100,
    status: "OK",
    render: `${name}: ${numerator}/${denominator} (${(value * 100).toFixed(2)}%)`,
  };
}

export function coverage(partial = false): Coverage {
  return {
    observations: 120,
    with_qc_data: partial ? 90 : 120,
    without_qc_data: partial ? 30 : 0,
    complete: !partial,
    partial,
    boundary_commit: "460642e",
    note: partial
      ? "30 of 120 generations in this window predate the candidate trace (460642e). Their retry counts are unknown, not zero."
      : null,
  };
}

export function summary(overrides: Partial<Summary> = {}): Summary {
  return {
    ...VERSIONS,
    window: WINDOW,
    segment: {},
    sample_count: 120,
    counters: {
      generation_requests: 120,
      completed_generations: 110,
      failed_generations: 8,
      cancelled_generations: 2,
      without_qc_data: 0,
      provider_calls: 150,
      candidates_generated: 150,
      quality_retries: 30,
      retry_exhaustions: 4,
      candidate_rejections: 30,
      first_candidate_accepted: 88,
      qc_observed: 118,
      finding_counts: { EARLY_COLLAPSE: 24, DURATION_SHORT: 6 },
      soft_finding_counts: { NARROW_STEREO: 40, HIGH_HARSHNESS_PROXY: 12 },
      failure_code_counts: { QUALITY_RETRY_EXHAUSTED: 4 },
      data_quality_counts: {},
    },
    overview: {
      generation_success_rate: rate("generation_success_rate", 110, 118),
      first_candidate_accept_rate: rate("first_candidate_accept_rate", 88, 118),
      quality_retry_rate: rate("quality_retry_rate", 30, 118),
      retry_exhaustion_rate: rate("retry_exhaustion_rate", 4, 118),
      provider_failure_rate: rate("provider_failure_rate", 1, 118),
      early_collapse_rate: rate("early_collapse_rate", 24, 118),
    },
    rates: {},
    averages: {
      average_provider_calls_per_generation: {
        name: "average_provider_calls_per_generation",
        value: 1.27,
        count: 118,
        status: "OK",
      },
    },
    latency: {
      total_latency_seconds: {
        name: "total_latency_seconds",
        count: 118,
        p50: 62.5,
        p90: 88.1,
        p95: 95.4,
        p99: 120.2,
        max: 130.0,
        mean: 66.2,
        status: "OK",
      },
    },
    findings: {
      critical: { EARLY_COLLAPSE: 24, DURATION_SHORT: 6 },
      soft: { NARROW_STEREO: 40, HIGH_HARSHNESS_PROXY: 12 },
    },
    data_quality: {},
    coverage: coverage(),
    ...overrides,
  };
}

export function emptySummary(): Summary {
  return summary({
    sample_count: 0,
    counters: {
      ...summary().counters,
      generation_requests: 0,
      completed_generations: 0,
      failed_generations: 0,
      cancelled_generations: 0,
      qc_observed: 0,
      finding_counts: {},
      soft_finding_counts: {},
      failure_code_counts: {},
    },
    overview: {
      generation_success_rate: rate("generation_success_rate", 0, 0),
      first_candidate_accept_rate: rate("first_candidate_accept_rate", 0, 0),
      quality_retry_rate: rate("quality_retry_rate", 0, 0),
      retry_exhaustion_rate: rate("retry_exhaustion_rate", 0, 0),
      provider_failure_rate: rate("provider_failure_rate", 0, 0),
      early_collapse_rate: rate("early_collapse_rate", 0, 0),
    },
    latency: {
      total_latency_seconds: {
        name: "total_latency_seconds",
        count: 0,
        p50: null,
        p90: null,
        p95: null,
        p99: null,
        max: null,
        mean: null,
        status: "NO_DATA",
      },
    },
    findings: { critical: {}, soft: {} },
    coverage: { ...coverage(), observations: 0, with_qc_data: 0, complete: true, partial: false },
  });
}

export function overview(overrides: Partial<Overview> = {}): Overview {
  return {
    ...VERSIONS,
    window: WINDOW,
    summary: summary(),
    open_incidents: 2,
    critical_incidents: 1,
    markers: [],
    automatic_remediation:
      "none — this console detects and explains; every action is an operator's",
    ...overrides,
  };
}

/** A trend with a deliberate hole in it, so the gap behaviour is tested. */
export function trend(metrics: string[], withGap = true): Trend {
  const points: TrendPoint[] = [];
  for (let index = 0; index < 12; index += 1) {
    const empty = withGap && index >= 4 && index <= 6;
    const values: Record<string, number | null> = {};
    for (const metric of metrics) {
      values[metric] = empty ? null : 0.05 + index * 0.01;
    }
    points.push({
      start: `2026-08-21T11:${String(index * 5).padStart(2, "0")}:00+00:00`,
      end: `2026-08-21T11:${String(index * 5 + 5).padStart(2, "0")}:00+00:00`,
      sample_count: empty ? 0 : index === 1 ? 3 : 40,
      values,
    });
  }
  return {
    ...VERSIONS,
    window: WINDOW,
    segment: {},
    metrics,
    bucket_seconds: 300,
    points,
    has_data: true,
  };
}

export function emptyTrend(metrics: string[]): Trend {
  return { ...trend(metrics), points: [], has_data: false };
}

export function regression(overrides: Partial<Regression> = {}): Regression {
  return {
    ...VERSIONS,
    finding_type: "EARLY_COLLAPSE_INCREASE",
    category: "QUALITY",
    metric: "early_collapse_rate",
    segment: { duration_bucket: "181_240" },
    segment_label: "duration_bucket=181_240",
    status: "REGRESSED",
    severity: "CRITICAL",
    baseline_value: 0.0073,
    current_value: 0.1146,
    absolute_delta: 0.1073,
    relative_delta: 14.7,
    baseline_numerator: 3,
    baseline_denominator: 412,
    current_numerator: 11,
    current_denominator: 96,
    baseline_sample_count: 412,
    current_sample_count: 96,
    baseline_window: WINDOW,
    current_window: WINDOW,
    quantile_fraction: null,
    thresholds: { minimum_absolute_delta: 0.02 },
    threshold_crossed: "absolute ≥ 0.02 and relative ≥ 0.5",
    reason: "moved 0.1073 absolute and 1470.0% relative, crossing both minimums",
    explanation:
      "early_collapse_rate moved from 3/412 (0.73%) to 11/96 (11.46%) for duration_bucket=181_240.",
    recommendations: ["CHECK_SAMPLE_GENERATIONS", "INSPECT_DURATION_SEGMENT"],
    partial_history: false,
    ...overrides,
  };
}

export function incident(overrides: Partial<Incident> = {}): Incident {
  return {
    ...VERSIONS,
    incident_id: "abc123",
    created_at: "2026-08-21T11:05:00+00:00",
    status: "OPEN",
    severity: "CRITICAL",
    peak_severity: "CRITICAL",
    finding_type: "EARLY_COLLAPSE_INCREASE",
    category: "QUALITY",
    metric: "early_collapse_rate",
    provider: "ace_step",
    provider_version: "acestep@v2",
    affected_dimensions: { duration_bucket: "181_240" },
    segment_label: "duration_bucket=181_240",
    baseline_window: WINDOW,
    current_window: WINDOW,
    first_seen: "2026-08-21T11:05:00+00:00",
    last_seen: "2026-08-21T11:55:00+00:00",
    occurrence_count: 11,
    consecutive_clean: 0,
    evidence: [
      {
        observed_at: "2026-08-21T11:55:00+00:00",
        status: "REGRESSED",
        severity: "CRITICAL",
        baseline_value: 0.0073,
        current_value: 0.1146,
        absolute_delta: 0.1073,
        relative_delta: 14.7,
        current_sample_count: 96,
        baseline_sample_count: 412,
        explanation:
          "early_collapse_rate moved from 3/412 (0.73%) to 11/96 (11.46%) for duration_bucket=181_240.",
      },
    ],
    evidence_total: 11,
    recommendations: ["CHECK_SAMPLE_GENERATIONS", "INSPECT_DURATION_SEGMENT"],
    summary:
      "[CRITICAL] EARLY_COLLAPSE_INCREASE — duration_bucket=181_240. early_collapse_rate moved from 3/412 (0.73%) to 11/96 (11.46%).",
    acknowledged_at: null,
    acknowledged_by: null,
    resolved_at: null,
    dismissed_at: null,
    dismissed_by: null,
    dismissal_reason: null,
    ...overrides,
  };
}

export function incidentList(items: Incident[] = [incident()]): IncidentList {
  return { total: items.length, limit: 25, offset: 0, items };
}

export function providers(): Providers {
  return {
    ...VERSIONS,
    window: WINDOW,
    providers: [
      {
        provider: "ace_step",
        provider_revision: "acestep@v1",
        model_name: "acestep",
        model_version: "v1",
        sample_count: 400,
        first_seen: "2026-08-14T00:00:00+00:00",
        last_seen: "2026-08-21T11:55:00+00:00",
        baseline_status: "READY",
        rates: {
          generation_success_rate: rate("generation_success_rate", 396, 400),
          first_candidate_accept_rate: rate("first_candidate_accept_rate", 380, 400),
          quality_retry_rate: rate("quality_retry_rate", 20, 400),
          early_collapse_rate: rate("early_collapse_rate", 2, 400),
        },
      },
      {
        provider: "ace_step",
        provider_revision: "acestep@v2",
        model_name: "acestep",
        model_version: "v2",
        sample_count: 20,
        first_seen: "2026-08-21T11:00:00+00:00",
        last_seen: "2026-08-21T11:55:00+00:00",
        baseline_status: "BASELINE_BUILDING",
        rates: {
          generation_success_rate: rate("generation_success_rate", 14, 20),
          first_candidate_accept_rate: rate("first_candidate_accept_rate", 10, 20),
          quality_retry_rate: rate("quality_retry_rate", 10, 20),
          early_collapse_rate: rate("early_collapse_rate", 8, 20),
        },
      },
    ],
  };
}

export function segments(withRows = true): Segments {
  return {
    ...VERSIONS,
    window: WINDOW,
    grouped_by: ["provider", "duration_bucket"],
    metric: "generation_failure_rate",
    minimum_samples: 30,
    segments: withRows
      ? [
          {
            segment: { provider: "ace_step", duration_bucket: "181_240" },
            segment_label: "duration_bucket=181_240, provider=ace_step",
            metric: "generation_failure_rate",
            value: 0.1146,
            numerator: 11,
            denominator: 96,
            render: "generation_failure_rate: 11/96 (11.46%)",
            sample_count: 96,
          },
        ]
      : [],
    segments_considered: 6,
    segments_below_minimum: withRows ? 4 : 6,
  };
}

export function generationList(): GenerationList {
  return {
    total: 2,
    limit: 25,
    offset: 0,
    items: [
      {
        generation_id: "11111111-1111-4111-8111-111111111111",
        occurred_at: "2026-08-21T11:50:00+00:00",
        provider_revision: "acestep@v1",
        task_type: "TEXT_TO_MUSIC",
        duration_bucket: "181_240",
        generation_status: "COMPLETED",
        quality_retry_count: 1,
        first_candidate_accepted: false,
        critical_findings: ["EARLY_COLLAPSE"],
        total_latency_seconds: 96.4,
      },
      {
        generation_id: "22222222-2222-4222-8222-222222222222",
        occurred_at: "2026-08-21T11:40:00+00:00",
        provider_revision: "acestep@v1",
        task_type: "COVER",
        duration_bucket: "121_180",
        generation_status: "FAILED",
        quality_retry_count: 2,
        first_candidate_accepted: false,
        critical_findings: ["EARLY_COLLAPSE"],
        total_latency_seconds: 140.2,
      },
    ],
  };
}

export function generationTrace(): GenerationTrace {
  return {
    generation_id: "11111111-1111-4111-8111-111111111111",
    occurred_at: "2026-08-21T11:50:00+00:00",
    provider: "ace_step",
    provider_revision: "acestep@v1",
    task_type: "TEXT_TO_MUSIC",
    duration_bucket: "181_240",
    requested_duration_seconds: 200,
    language: "ko",
    instrumental: "NO",
    generation_status: "COMPLETED",
    generation_failure_code: null,
    qc_policy: "STANDARD",
    qc_data_available: true,
    qc_outcome: "SELECTED",
    finishing_outcome: "FINISHED",
    candidate_count: 2,
    provider_call_count: 2,
    quality_retry_count: 1,
    selected_on_attempt: 1,
    first_candidate_accepted: false,
    retry_exhausted: false,
    provider_latency_seconds: 80.2,
    qc_latency_seconds: 2.4,
    delivery_latency_seconds: 13.8,
    total_latency_seconds: 96.4,
    critical_findings: ["EARLY_COLLAPSE"],
    soft_findings: ["NARROW_STEREO"],
    data_quality_issues: [],
    attempts: [
      {
        attempt_index: 0,
        candidate_id: "cand_00",
        status: "REJECTED",
        selection_status: "NOT_SELECTED",
        attribution: "USER_REQUEST",
        seed: 1000,
        retry_reason: null,
        not_selected_reason: "not eligible: EARLY_COLLAPSE",
        duration_seconds: 200,
        critical_findings: ["EARLY_COLLAPSE"],
        soft_findings: [],
        provider_seconds: 40.1,
        qc_seconds: 1.2,
      },
      {
        attempt_index: 1,
        candidate_id: "cand_01",
        status: "ELIGIBLE",
        selection_status: "SELECTED",
        attribution: "QUALITY_RETRY",
        seed: 8842,
        retry_reason: "EARLY_COLLAPSE: retried with a different seed",
        not_selected_reason: null,
        duration_seconds: 200,
        critical_findings: [],
        soft_findings: ["NARROW_STEREO"],
        provider_seconds: 40.1,
        qc_seconds: 1.2,
      },
    ],
    explanation: [
      "Attempt 1 rejected: EARLY_COLLAPSE",
      "Attempt 2 selected.",
      "Provider calls: 2. Quality retries: 1.",
    ],
  };
}

export function ingestStatus(stale = false): IngestStatus {
  return {
    observations: stale ? 400 : 420,
    latest_observation_at: "2026-08-21T11:55:00+00:00",
    seconds_behind: stale ? 26000 : 120,
    stale,
    note: stale
      ? "The newest observation is 7.2 hours old. Charts may be behind reality; check that ingestion is running."
      : null,
  };
}
