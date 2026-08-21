/**
 * How the console talks to the operator API.
 *
 * Every request goes to the same-origin `/ops/api/training/...` proxy, which
 * attaches the operator token server-side. Nothing here holds a
 * credential, and there is no place in this module for one to be put.
 *
 * Errors carry the server's sentence. The API answers a refused action
 * with 409 and a reason written for an operator — "only a QUEUED run can
 * be dispatched", "rights are not clear for every selected track" — and
 * that reason is far more useful than "Request failed". `OpsError` keeps
 * both the status and the text so a panel can distinguish "this is not
 * allowed" from "the API is down".
 */

import type {
  ActionResult,
  BaselineResponse,
  Catalogue,
  CheckpointComparison,
  CheckpointDetail,
  CheckpointList,
  EvaluationDetail,
  EvaluationList,
  ExperimentDetail,
  ExperimentList,
  ExperimentSummary,
  LogView,
  Overview,
  RunDetail,
  RunList,
  WorkerCompatibility,
  WorkerDetail,
  WorkerList,
} from "@/lib/ops/types";

/**
 * The training namespace of the operator proxy.
 *
 * The namespace is in the base rather than in every call site: the
 * proxy routes on the first segment, and repeating it thirty times
 * would be thirty chances to route a training call at the inference
 * console.
 */
export const OPS_API_BASE = "/ops/api/training";

export class OpsError extends Error {
  readonly status: number;
  /** True when the world said no, rather than the request being wrong. */
  readonly refused: boolean;

  constructor(status: number, message: string) {
    super(message);
    this.name = "OpsError";
    this.status = status;
    this.refused = status === 409;
  }
}

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
    response = await fetch(`${OPS_API_BASE}${path}`, {
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

/** Drops empty filters so a cleared control does not become `?status=`. */
function query(params: Record<string, string | number | boolean | undefined | null>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    search.set(key, String(value));
  }
  const rendered = search.toString();
  return rendered ? `?${rendered}` : "";
}

export const ops = {
  overview: () => request<Overview>("/overview"),
  baseline: () => request<BaselineResponse>("/baseline"),
  catalogue: () => request<Catalogue>("/catalogue"),

  experiments: (params: {
    status?: string;
    base_model_id?: string;
    tag?: string;
    q?: string;
    limit?: number;
    offset?: number;
  }) => request<ExperimentList>(`/experiments${query(params)}`),

  experiment: (id: string) => request<ExperimentDetail>(`/experiments/${id}`),

  createExperiment: (payload: {
    name: string;
    hypothesis: string;
    base_model_id: string;
    description: string;
    operator: string;
    tags: string[];
  }) =>
    request<ExperimentSummary>("/experiments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  runs: (params: {
    status?: string;
    experiment_id?: string;
    worker_id?: string;
    backend?: string;
    limit?: number;
    offset?: number;
    with_metrics?: boolean;
  }) => request<RunList>(`/runs${query(params)}`),

  run: (id: string) => request<RunDetail>(`/runs/${id}`),

  createRun: (payload: {
    experiment_id: string;
    dataset_build_id: string;
    curation_build_id: string;
    preset: string;
    execution_backend: string;
    worker_id: string | null;
  }) =>
    request<RunDetail>("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  /**
   * Continue a log from where the last read stopped.
   *
   * `offset` omitted means "wherever an operator would want to start" —
   * the tail of a long file. Passing back `next_offset` is what makes a
   * poll incremental instead of a re-download.
   */
  logs: (id: string, params: { stream?: "stdout" | "stderr"; offset?: number }) =>
    request<LogView>(`/runs/${id}/logs${query(params)}`),

  diagnostics: (id: string) => request<string[]>(`/runs/${id}/diagnostics`),

  runAction: (id: string, action: string) =>
    request<ActionResult>(`/runs/${id}/actions/${action}`, { method: "POST" }),

  workers: (params: { worker_class?: string; liveness?: string; limit?: number; offset?: number }) =>
    request<WorkerList>(`/workers${query(params)}`),

  worker: (id: string) => request<WorkerDetail>(`/workers/${id}`),

  workerCompatibility: (backend: string) =>
    request<WorkerCompatibility[]>(
      `/workers/compatibility${query({ execution_backend: backend })}`,
    ),

  checkpoints: (params: {
    status?: string;
    kind?: string;
    run_id?: string;
    experiment_id?: string;
    limit?: number;
    offset?: number;
  }) => request<CheckpointList>(`/checkpoints${query(params)}`),

  checkpoint: (id: string) => request<CheckpointDetail>(`/checkpoints/${id}`),

  compareCheckpoints: (ids: string[]) =>
    request<CheckpointComparison>("/checkpoints/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ checkpoint_ids: ids }),
    }),

  evaluations: (params: {
    status?: string;
    outcome?: string;
    experiment_id?: string;
    limit?: number;
    offset?: number;
  }) => request<EvaluationList>(`/evaluations${query(params)}`),

  evaluation: (id: string) => request<EvaluationDetail>(`/evaluations/${id}`),
};

/** Where a downloadable artifact lives, for an anchor's `href`. */
export const opsDownload = {
  runBundle: (id: string) => `${OPS_API_BASE}/runs/${id}/bundle`,
  evaluationReport: (id: string) => `${OPS_API_BASE}/evaluations/${id}/report`,
};
