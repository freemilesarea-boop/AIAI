/**
 * Plan and entitlement payloads for tests.
 *
 * Shaped exactly like `/v1/plans` and `/v1/account/entitlement`, with
 * the figures the server actually publishes. Tests that need a specific
 * situation — nearly exhausted, spent, Free — override the two or three
 * fields that matter and inherit the rest, so a change to the response
 * shape breaks in one place rather than in every file.
 */

import type { Entitlement, Plan, PlanId } from "@/lib/plans";

const TIERS: Record<PlanId, Omit<Plan, "recommended">> = {
  free: {
    plan_id: "free",
    display_name: "Free",
    monthly_price_krw: 0,
    monthly_generation_limit: 20,
    download_mp3: false,
    download_wav: false,
    commercial_use: false,
    priority_level: 0,
    lab_access: false,
  },
  basic: {
    plan_id: "basic",
    display_name: "Basic",
    monthly_price_krw: 19900,
    monthly_generation_limit: 200,
    download_mp3: true,
    download_wav: true,
    commercial_use: true,
    priority_level: 1,
    lab_access: false,
  },
  pro: {
    plan_id: "pro",
    display_name: "Pro",
    monthly_price_krw: 29900,
    monthly_generation_limit: 500,
    download_mp3: true,
    download_wav: true,
    commercial_use: true,
    priority_level: 2,
    lab_access: true,
  },
  creator: {
    plan_id: "creator",
    display_name: "Creator",
    monthly_price_krw: 49900,
    monthly_generation_limit: 1000,
    download_mp3: true,
    download_wav: true,
    commercial_use: true,
    priority_level: 3,
    lab_access: true,
  },
};

export function planFixture(id: PlanId): Plan {
  return { ...TIERS[id], recommended: id === "pro" };
}

export function planCatalogueFixture() {
  return {
    plans: (Object.keys(TIERS) as PlanId[]).map(planFixture),
    checkout_available: false,
  };
}

/** An entitlement on *id*, with *used* songs spent this period. */
export function entitlementFixture(id: PlanId = "basic", used = 0): Entitlement {
  const plan = planFixture(id);
  return {
    plan,
    period_start: "2026-08-01T00:00:00+00:00",
    period_end: "2026-09-01T00:00:00+00:00",
    generation_limit: plan.monthly_generation_limit,
    generation_used: used,
    generation_remaining: Math.max(0, plan.monthly_generation_limit - used),
    download_mp3: plan.download_mp3,
    download_wav: plan.download_wav,
    commercial_use: plan.commercial_use,
  };
}

/** An account with nothing left this period. */
export function exhaustedFixture(id: PlanId = "free"): Entitlement {
  return entitlementFixture(id, TIERS[id].monthly_generation_limit);
}
