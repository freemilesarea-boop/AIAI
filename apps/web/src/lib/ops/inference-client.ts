/**
 * How the inference console talks to the operator API.
 *
 * Same proxy, different namespace. The token still lives only in the
 * Next server's environment; nothing in this module holds a credential
 * and there is no place in it for one to be put.
 *
 * Errors keep the server's sentence. A refused segment grouping answers
 * 409 with a reason written for an operator — "at most 2 grouping
 * dimensions are supported; a wider split divides the samples until no
 * bucket can support a finding" — and that is far more useful than
 * "Request failed".
 */

import { OpsError } from "@/lib/ops/client";

import type {
  GenerationList,
  GenerationTrace,
  Incident,
  IncidentList,
  InferenceFilters,
  IngestStatus,
  Overview,
  Providers,
  Regression,
  Segments,
  Summary,
  Trend,
} from "@/lib/ops/inference-types";

export const INFERENCE_API_BASE = "/ops/api/inference";

async function readDetail(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") return payload.detail;
    return JSON.stringify(payload.detail ?? payload);
  } catch {
    return response.statusText || `Request failed with ${response.status}`;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${INFERENCE_API_BASE}${path}`, {
      ...init,
      cache: "no-store",
      headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new OpsError(0, "The console could not reach the server.");
  }
  if (!response.ok) throw new OpsError(response.status, await readDetail(response));
  return (await response.json()) as T;
}

/** Drops empty filters, so a cleared control does not become `?provider=`. */
function query(params: Record<string, string | number | boolean | undefined | null>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    search.set(key, String(value));
  }
  const rendered = search.toString();
  return rendered ? `?${rendered}` : "";
}

function filterParams(filters: InferenceFilters): Record<string, string | undefined> {
  return {
    window: filters.window,
    provider: filters.provider,
    revision: filters.revision,
    task: filters.task,
    duration_bucket: filters.duration_bucket,
  };
}

export const inference = {
  overview: (filters: InferenceFilters) =>
    request<Overview>(`/overview${query(filterParams(filters))}`),

  summary: (filters: InferenceFilters) =>
    request<Summary>(`/summary${query(filterParams(filters))}`),

  trend: (chart: "retry" | "failure" | "latency", filters: InferenceFilters) =>
    request<Trend>(`/trend${query({ chart, ...filterParams(filters) })}`),

  providers: (window: string) => request<Providers>(`/providers${query({ window })}`),

  compareRevisions: (params: {
    left: string;
    right: string;
    window?: string;
    minimum_samples?: number;
  }) => request<Record<string, unknown>>(`/providers/compare${query(params)}`),

  segments: (params: {
    window: string;
    group_by?: string;
    metric?: string;
    minimum_samples?: number;
    limit?: number;
  }) => request<Segments>(`/segments${query(params)}`),

  regressions: (params: { window: string; group_by?: string }) =>
    request<Regression[]>(`/regressions${query(params)}`),

  incidents: (params: { include_closed?: boolean; limit?: number; offset?: number }) =>
    request<IncidentList>(`/incidents${query(params)}`),

  incident: (id: string) => request<Incident>(`/incidents/${id}`),

  acknowledge: (id: string, operator: string) =>
    request<{ ok: boolean; incident: Incident }>(
      `/incidents/${id}/acknowledge${query({ operator })}`,
      { method: "POST" },
    ),

  dismiss: (id: string, operator: string, reason: string) =>
    request<{ ok: boolean; incident: Incident }>(
      `/incidents/${id}/dismiss${query({ operator, reason })}`,
      { method: "POST" },
    ),

  generations: (params: InferenceFilters & { limit?: number; offset?: number; only_failures?: boolean }) =>
    request<GenerationList>(
      `/generations${query({
        ...filterParams(params),
        limit: params.limit,
        offset: params.offset,
        only_failures: params.only_failures,
      })}`,
    ),

  generation: (id: string) => request<GenerationTrace>(`/generations/${id}`),

  ingestStatus: () => request<IngestStatus>("/ingest-status"),
};

export { OpsError };
