/**
 * Plans and allowance, read from the server.
 *
 * This file used to hold three tier names with every number `null`,
 * because there was no policy to describe. There is one now, and it
 * lives in `packages/schemas/src/luber_schemas/plans.py` — the same
 * definition the API enforces.
 *
 * So nothing here states a price, a limit or an entitlement. Every
 * figure is fetched. A number hardcoded in this file would be a number
 * that disagrees with the server the first time it changes, and the
 * disagreement would be in the customer's favour or ours, never neither.
 *
 * There is still no payment provider. `checkoutAvailable` comes from the
 * server and is false, and the page renders an honest unavailable state
 * rather than a subscribe button that cannot subscribe.
 */

import { API_BASE_URL, ApiError } from "@/lib/api";

export type PlanId = "free" | "basic" | "pro" | "creator";

/** One tier, exactly as `/v1/plans` serves it. */
export interface Plan {
  plan_id: PlanId;
  display_name: string;
  monthly_price_krw: number;
  monthly_generation_limit: number;
  download_mp3: boolean;
  download_wav: boolean;
  commercial_use: boolean;
  priority_level: number;
  lab_access: boolean;
  /** Highlights a column. Presentation only — it grants nothing. */
  recommended: boolean;
}

export interface PlanCatalogue {
  plans: Plan[];
  /** False while no payment provider is connected. */
  checkout_available: boolean;
}

/**
 * What the signed-in account may do right now.
 *
 * The server's own words. The UI never computes remaining from a limit
 * and a count it kept itself — a client-side tally drifts, and the one
 * that matters is the one the server will enforce at generate time.
 */
export interface Entitlement {
  plan: Plan;
  period_start: string;
  period_end: string;
  generation_limit: number;
  generation_used: number;
  generation_remaining: number;
  download_mp3: boolean;
  download_wav: boolean;
  commercial_use: boolean;
}

async function readJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    cache: "no-store",
    credentials: "include",
    signal,
  });
  if (!res.ok) {
    throw new ApiError(`${path} failed: ${res.status}`, res.status);
  }
  return (await res.json()) as T;
}

export async function fetchPlans(signal?: AbortSignal): Promise<PlanCatalogue> {
  return readJson<PlanCatalogue>("/v1/plans", signal);
}

export async function fetchEntitlement(signal?: AbortSignal): Promise<Entitlement> {
  return readJson<Entitlement>("/v1/account/entitlement", signal);
}

// ── formatting ───────────────────────────────────────────────────────

export function formatPriceKrw(price: number): string {
  return price === 0 ? "무료" : `₩${price.toLocaleString("ko-KR")}`;
}

/** "20곡" — the unit users think in. Never "credits". */
export function formatSongs(count: number): string {
  return `${count.toLocaleString("ko-KR")}곡`;
}

/** The allowance period as a Korean date range. */
export function formatPeriod(entitlement: Entitlement): string {
  const format = (iso: string) => {
    const at = new Date(iso);
    return Number.isNaN(at.getTime())
      ? iso
      : `${at.getFullYear()}. ${at.getMonth() + 1}. ${at.getDate()}.`;
  };
  return `${format(entitlement.period_start)} – ${format(entitlement.period_end)}`;
}

/**
 * How much of the allowance is gone, 0–1.
 *
 * Guards a zero limit rather than dividing by it: a plan configured with
 * no allowance should render an empty bar, not `NaN%`.
 */
export function usageRatio(entitlement: Entitlement): number {
  if (entitlement.generation_limit <= 0) return 0;
  return Math.min(1, entitlement.generation_used / entitlement.generation_limit);
}

/** The threshold at which the UI starts warning. */
export const USAGE_WARNING_RATIO = 0.9;

export function isNearlyExhausted(entitlement: Entitlement): boolean {
  return !isExhausted(entitlement) && usageRatio(entitlement) >= USAGE_WARNING_RATIO;
}

export function isExhausted(entitlement: Entitlement): boolean {
  return entitlement.generation_remaining <= 0;
}

/**
 * Whether this account can download at all.
 *
 * Advisory only. The server refuses a download the plan does not cover
 * whatever this returns — this exists so the UI can explain the refusal
 * before the user meets it, not so it can be the thing that enforces it.
 */
export function canDownload(entitlement: Entitlement | null): boolean {
  return entitlement !== null && (entitlement.download_mp3 || entitlement.download_wav);
}
