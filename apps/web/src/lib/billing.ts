/**
 * Talking to BOORDA's billing API. Never to PayApp.
 *
 * The browser's whole part in a subscription is: name a plan, hand over
 * a phone number, and follow the URL the server returns. It never sees a
 * PayApp credential, never sends a price, and never learns a `rebill_no`
 * — the server resolves all three, and this file has no field for any of
 * them.
 *
 * The most important function here is the one that does the least.
 * `fetchBillingStatus` is what the post-payment page reads, and it reads
 * the server's records. The return URL's query parameters are not
 * consulted anywhere in this file, because reaching that URL proves
 * nothing: a user gets there by paying, by closing the PayApp window, or
 * by typing it.
 */

import { API_BASE_URL, ApiError } from "@/lib/api";
import type { PlanId } from "@/lib/plans";

/**
 * Where a subscription is, in the server's words.
 *
 * `NONE` is this file's own value for "no subscription row at all"; the
 * rest come from the backend state machine.
 */
export type SubscriptionStatus =
  | "NONE"
  | "PENDING_INITIAL_PAYMENT"
  | "ACTIVE"
  | "PAST_DUE"
  | "CANCEL_PENDING"
  | "CANCELED"
  | "EXPIRED";

export interface BillingStatus {
  plan_id: string;
  display_name: string;
  status: SubscriptionStatus;
  auto_renew: boolean;
  period_start: string | null;
  period_end: string | null;
  next_renewal_at: string | null;
  last_payment_at: string | null;
  /** True while a payment is expected but unconfirmed. */
  awaiting_payment: boolean;
  /** False when the deployment has no PayApp credentials configured. */
  checkout_available: boolean;
}

export interface CheckoutResult {
  payurl: string;
  plan_id: string;
  display_name: string;
  amount_krw: number;
  /**
   * Always `PENDING_INITIAL_PAYMENT`. The server says so in the response
   * body so no client can read a 201 as "subscribed".
   */
  status: string;
  correlation_id: string;
}

export interface PaymentRecord {
  paid_at: string;
  plan_id: string;
  amount_krw: number;
  status: "SUCCEEDED" | "FAILED";
  failure_reason: string | null;
}

async function readJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    cache: "no-store",
    credentials: "include",
    ...init,
  });
  if (!res.ok) {
    let code: string | undefined;
    try {
      code = (await res.json())?.detail;
    } catch {
      code = undefined;
    }
    throw new ApiError(`${path} failed: ${res.status}`, res.status, code);
  }
  return (await res.json()) as T;
}

/**
 * Start a subscription.
 *
 * Sends a plan name and a phone number. Not a price — the server reads
 * that from its own table, and there is deliberately no parameter here
 * through which one could be suggested.
 */
export async function createCheckout(
  planId: PlanId,
  phone: string,
  signal?: AbortSignal,
): Promise<CheckoutResult> {
  return readJson<CheckoutResult>("/v1/billing/checkout", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan_id: planId, phone }),
    signal,
  });
}

export async function fetchBillingStatus(signal?: AbortSignal): Promise<BillingStatus> {
  return readJson<BillingStatus>("/v1/billing/status", { signal });
}

/**
 * Cancel this account's subscription.
 *
 * Takes no arguments, which is the point: the server knows which
 * subscription belongs to the session, so there is no identifier here
 * that could be pointed at somebody else's.
 */
export async function cancelSubscription(signal?: AbortSignal): Promise<BillingStatus> {
  return readJson<BillingStatus>("/v1/billing/cancel", { method: "POST", signal });
}

export async function fetchPayments(signal?: AbortSignal): Promise<PaymentRecord[]> {
  const body = await readJson<{ items: PaymentRecord[] }>("/v1/billing/payments", { signal });
  return body.items;
}

// ── presentation ─────────────────────────────────────────────────────

/** What Settings shows beside the plan name. */
export const STATUS_LABELS: Record<SubscriptionStatus, string> = {
  NONE: "구독 없음",
  PENDING_INITIAL_PAYMENT: "결제 확인 중",
  ACTIVE: "이용 중",
  // Named for what the user experiences, not for the database value: a
  // failed renewal is a payment problem, and "PAST_DUE" is our word.
  PAST_DUE: "결제 실패",
  CANCEL_PENDING: "해지 예정",
  CANCELED: "해지됨",
  EXPIRED: "만료됨",
};

export function formatBillingDate(iso: string | null): string {
  if (!iso) return "—";
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return "—";
  return `${at.getFullYear()}년 ${at.getMonth() + 1}월 ${at.getDate()}일`;
}

/**
 * Whether the account currently has paid access.
 *
 * Presentation only. Every entitlement the product actually enforces is
 * decided server-side — this exists so a page can say the right thing,
 * not so it can decide anything.
 */
export function hasPaidAccess(status: BillingStatus | null): boolean {
  return status !== null && (status.status === "ACTIVE" || status.status === "CANCEL_PENDING");
}

/** Korean mobile format, mirroring what the server will accept. */
export function isValidKoreanMobile(raw: string): boolean {
  return /^01[016789]-?\d{3,4}-?\d{4}$/.test(raw.trim().replace(/\s/g, ""));
}
