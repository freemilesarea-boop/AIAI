/**
 * How the circuit view talks to the operator API.
 *
 * Same proxy, third namespace, and every call is a `GET` — there is no
 * mutating verb in this module because there is no mutating route
 * behind it. Forcing a circuit open is a CLI, for the reason given in
 * `apps/api/src/luber_api/routes/ops_resilience.py`: this console does
 * not exist in production, which is where the incident is.
 */

import { OpsError } from "@/lib/ops/client";

import type { CircuitList, Policy, Readiness, TransitionList } from "@/lib/ops/resilience-types";

export const RESILIENCE_API_BASE = "/ops/api/resilience";

async function readDetail(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") return payload.detail;
    return JSON.stringify(payload.detail ?? payload);
  } catch {
    return response.statusText || `Request failed with ${response.status}`;
  }
}

async function request<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${RESILIENCE_API_BASE}${path}`, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
  } catch {
    throw new OpsError(0, "The console could not reach the server.");
  }
  if (!response.ok) throw new OpsError(response.status, await readDetail(response));
  return (await response.json()) as T;
}

export const resilience = {
  circuits: () => request<CircuitList>("/circuits"),
  readiness: () => request<Readiness>("/readiness"),
  policy: () => request<Policy>("/policy"),
  transitions: (limit = 50) => request<TransitionList>(`/transitions?limit=${limit}`),
};
