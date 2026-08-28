/**
 * The operator console's client.
 *
 * Everything here talks to `/v1/admin/*`, and everything there is
 * checked server-side against the session's own row. Nothing in this
 * module authorises anything: `isAdmin` below decides whether to draw a
 * link, and a browser that lies about it reaches endpoints that answer
 * 403. That separation is worth stating because the tempting shortcut —
 * gating the console in the router and trusting it — is exactly the bug
 * an admin console cannot afford.
 *
 * The numbers are computed in the database and arrive already
 * aggregated. The console never fetches rows to add them up in the
 * browser, which is why a growing customer base does not make the
 * dashboard slower.
 */

import { API_BASE_URL, ApiError } from "@/lib/api";
import type { SupportCategory, SupportStatus } from "@/lib/support";

export type UserRole = "USER" | "ADMIN" | "SUPER_ADMIN";

/** Whether to render operator navigation. Presentation only. */
export function isAdmin(role: string | undefined): boolean {
  return role === "ADMIN" || role === "SUPER_ADMIN";
}

/** Whether to render the "administrators" section. Presentation only. */
export function isSuperAdmin(role: string | undefined): boolean {
  return role === "SUPER_ADMIN";
}

export type Granularity = "day" | "week" | "month" | "year";

export const GRANULARITIES: { id: Granularity; label: string }[] = [
  { id: "day", label: "오늘" },
  { id: "week", label: "이번 주" },
  { id: "month", label: "이번 달" },
  { id: "year", label: "올해" },
];

export interface Bucket {
  day: string;
  value: number;
  secondary: number;
}

export interface PlanShare {
  plan_id: string;
  count: number;
  share: number;
}

export interface Dashboard {
  range: { start: string; end: string };
  generated_at: string;
  revenue_krw: number;
  revenue_today_krw: number;
  payment_count: number;
  users: { total: number; paid: number; free: number; new_in_range: number };
  generations: {
    requested: number;
    completed: number;
    failed: number;
    creators: number;
    average_per_creator: number;
  };
  downloads: number;
  support: Record<string, number>;
  plans: PlanShare[];
  revenue_series: Bucket[];
  generation_series: Bucket[];
}

export interface RevenueReport {
  range: { start: string; end: string };
  total_krw: number;
  payment_count: number;
  new_krw: number;
  new_count: number;
  renewal_krw: number;
  renewal_count: number;
  series: Bucket[];
}

export interface AdminUser {
  id: string;
  email: string;
  display_name: string | null;
  role: UserRole;
  created_at: string;
  deleted_at: string | null;
  /**
   * Null where the endpoint did not resolve one — the role endpoints
   * answer null rather than defaulting to Free, so a paying
   * administrator is never described as being on a plan they are not.
   */
  plan_id: string | null;
  subscription_status: string | null;
}

/** A tier name, or a dash where the endpoint did not resolve one. */
export function planLabel(planId: string | null): string {
  if (!planId) return "—";
  return PLAN_LABELS[planId] ?? planId;
}

export interface AdminUserDetail {
  user: AdminUser;
  activity: { generations: number; completed: number; downloads: number; payments: number };
}

export interface AdminTicketSummary {
  reference: string;
  user_email: string;
  category: SupportCategory;
  subject: string;
  status: SupportStatus;
  created_at: string;
}

export interface AdminTicket extends AdminTicketSummary {
  message: string;
  context_url: string | null;
  admin_note: string | null;
  updated_at: string;
  resolved_at: string | null;
}

export type AudienceType = "ALL" | "PLAN" | "USERS";

export interface Campaign {
  id: string;
  subject: string;
  body: string;
  audience_type: AudienceType;
  audience_plan_id: string | null;
  recipient_count: number;
  status: string;
  created_by_email: string;
  created_at: string;
  sent_at: string | null;
  /** Why nothing was sent. Present on every campaign today. */
  delivery_note: string | null;
}

export interface AuditEntry {
  id: string;
  action: string;
  actor_email: string;
  target_email: string | null;
  metadata: Record<string, string>;
  created_at: string;
}

/** What an operator reads, which is not what the database stores. */
export const ACTION_LABELS: Record<string, string> = {
  ADMIN_GRANTED: "관리자 권한 부여",
  ADMIN_REVOKED: "관리자 권한 해제",
  ADMIN_ROLE_CHANGED: "관리자 등급 변경",
  SUPPORT_STATUS_CHANGED: "문의 상태 변경",
  SUPPORT_NOTE_ADDED: "문의 메모 작성",
  EMAIL_CAMPAIGN_CREATED: "이메일 초안 작성",
};

export const ROLE_LABELS: Record<UserRole, string> = {
  USER: "일반 회원",
  ADMIN: "관리자",
  SUPER_ADMIN: "최고 관리자",
};

/**
 * Tier names for the console.
 *
 * The catalogue at `/v1/plans` is the authority on price and limits and
 * this is not a second copy of it — these are labels for aggregate rows,
 * which arrive as plan ids with no display name attached.
 */
export const PLAN_LABELS: Record<string, string> = {
  free: "Free",
  basic: "Basic",
  pro: "Pro",
  creator: "Creator",
};

async function readJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    cache: "no-store",
    credentials: "include",
    ...init,
  });
  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = (await res.clone().json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      detail = undefined;
    }
    throw new ApiError(detail ?? `${path} failed: ${res.status}`, res.status, detail);
  }
  return (await res.json()) as T;
}

function query(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") search.set(key, String(value));
  }
  const rendered = search.toString();
  return rendered ? `?${rendered}` : "";
}

export async function fetchDashboard(
  granularity: Granularity,
  signal?: AbortSignal,
): Promise<Dashboard> {
  return readJson<Dashboard>(`/v1/admin/dashboard${query({ granularity })}`, { signal });
}

export async function fetchRevenue(
  granularity: Granularity,
  signal?: AbortSignal,
): Promise<RevenueReport> {
  return readJson<RevenueReport>(`/v1/admin/analytics/revenue${query({ granularity })}`, {
    signal,
  });
}

export async function fetchUsers(
  params: { search?: string; plan?: string; limit?: number; offset?: number },
  signal?: AbortSignal,
): Promise<{ items: AdminUser[]; total: number }> {
  return readJson(`/v1/admin/users${query(params)}`, { signal });
}

export async function fetchUser(id: string, signal?: AbortSignal): Promise<AdminUserDetail> {
  return readJson(`/v1/admin/users/${encodeURIComponent(id)}`, { signal });
}

export async function fetchTickets(
  params: { status?: string; limit?: number; offset?: number },
  signal?: AbortSignal,
): Promise<{ items: AdminTicketSummary[]; total: number }> {
  return readJson(`/v1/admin/support${query(params)}`, { signal });
}

export async function fetchTicket(
  reference: string,
  signal?: AbortSignal,
): Promise<AdminTicket> {
  return readJson(`/v1/admin/support/${encodeURIComponent(reference)}`, { signal });
}

export async function updateTicket(
  reference: string,
  patch: { status?: SupportStatus; admin_note?: string },
): Promise<AdminTicket> {
  return readJson(`/v1/admin/support/${encodeURIComponent(reference)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
}

export interface CampaignDraft {
  subject: string;
  body: string;
  audience_type: AudienceType;
  plan_id?: string | null;
  user_ids?: string[];
}

/** Just the audience, so the count can be asked before anything is written. */
export interface AudienceQuery {
  audience_type: AudienceType;
  plan_id?: string | null;
  user_ids?: string[];
}

export async function previewAudience(
  audience: AudienceQuery,
): Promise<{ recipient_count: number }> {
  return readJson("/v1/admin/email/audience", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(audience),
  });
}

export async function createCampaign(draft: CampaignDraft): Promise<Campaign> {
  return readJson("/v1/admin/email/campaigns", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(draft),
  });
}

export async function fetchCampaigns(signal?: AbortSignal): Promise<{ items: Campaign[] }> {
  return readJson("/v1/admin/email/campaigns", { signal });
}

export async function fetchAdmins(signal?: AbortSignal): Promise<AdminUser[]> {
  return readJson("/v1/admin/admins", { signal });
}

export async function grantAdmin(email: string, role: UserRole): Promise<AdminUser> {
  return readJson("/v1/admin/admins", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, role }),
  });
}

export async function changeRole(userId: string, role: UserRole): Promise<AdminUser> {
  return readJson("/v1/admin/admins", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, role }),
  });
}

export async function revokeAdmin(userId: string): Promise<AdminUser> {
  return readJson(`/v1/admin/admins/${encodeURIComponent(userId)}`, { method: "DELETE" });
}

export async function fetchAudit(
  params: { action?: string; limit?: number; offset?: number },
  signal?: AbortSignal,
): Promise<{ items: AuditEntry[]; total: number }> {
  return readJson(`/v1/admin/audit${query(params)}`, { signal });
}

/** ₩1,234,500 — the unit an operator reconciles against. */
export function formatWon(amount: number): string {
  return new Intl.NumberFormat("ko-KR", {
    style: "currency",
    currency: "KRW",
    maximumFractionDigits: 0,
  }).format(amount);
}

export function formatCount(value: number): string {
  return new Intl.NumberFormat("ko-KR").format(value);
}

export function formatDateTime(iso: string): string {
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return "—";
  return new Intl.DateTimeFormat("ko-KR", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "Asia/Seoul",
  }).format(at);
}

export function formatDay(day: string): string {
  const at = new Date(`${day}T00:00:00+09:00`);
  if (Number.isNaN(at.getTime())) return day;
  return new Intl.DateTimeFormat("ko-KR", {
    month: "short",
    day: "numeric",
    timeZone: "Asia/Seoul",
  }).format(at);
}
