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

/** How the server bucketed a series. Mirrors `Bucketing` in the API. */
export type Bucketing = "day" | "week" | "month";

/**
 * A reporting window, as two Korean calendar days.
 *
 * Inclusive at both ends — `from` and `to` equal means one day, which is
 * what "오늘" means to a person. The strings are `YYYY-MM-DD` and go to
 * the API verbatim; the server converts them to UTC instants.
 */
export interface DateRange {
  from: string;
  to: string;
}

export type PresetId = "today" | "7d" | "30d" | "month" | "year";

export const PRESETS: { id: PresetId; label: string }[] = [
  { id: "today", label: "오늘" },
  { id: "7d", label: "7일" },
  { id: "30d", label: "30일" },
  { id: "month", label: "이번 달" },
  { id: "year", label: "올해" },
];

/** The default window when nothing valid was asked for. */
export const DEFAULT_PRESET: PresetId = "month";

/**
 * The server refuses anything longer. Mirrored so the picker can say so
 * before a request rather than after a 422.
 */
export const MAX_RANGE_DAYS = 366;

/**
 * Today's date in Seoul, whatever the browser's own timezone is.
 *
 * An operator in another timezone must still see Korean days, because
 * that is what the figures are bucketed by. `en-CA` formats as
 * `YYYY-MM-DD`, which is the shape the API takes.
 */
export function kstToday(now: Date = new Date()): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(now);
}

/**
 * Date arithmetic on `YYYY-MM-DD`, done at UTC midnight.
 *
 * Anchoring to UTC keeps the arithmetic away from the browser's local
 * timezone and its daylight-saving jumps — adding a day to a local-time
 * Date can land on the same day twice a year.
 */
export function addDays(iso: string, days: number): string {
  const at = new Date(`${iso}T00:00:00Z`);
  at.setUTCDate(at.getUTCDate() + days);
  return at.toISOString().slice(0, 10);
}

/** Inclusive length in days. One day is 1, not 0. */
export function rangeLength(range: DateRange): number {
  const from = Date.parse(`${range.from}T00:00:00Z`);
  const to = Date.parse(`${range.to}T00:00:00Z`);
  return Math.floor((to - from) / 86_400_000) + 1;
}

/** The window a preset means, resolved against Korean today. */
export function presetRange(preset: PresetId, now: Date = new Date()): DateRange {
  const today = kstToday(now);
  switch (preset) {
    case "today":
      return { from: today, to: today };
    case "7d":
      // Seven days *including* today, which is what "7일" reads as.
      return { from: addDays(today, -6), to: today };
    case "30d":
      return { from: addDays(today, -29), to: today };
    case "year":
      return { from: `${today.slice(0, 4)}-01-01`, to: today };
    case "month":
    default:
      return { from: `${today.slice(0, 7)}-01`, to: today };
  }
}

const ISO_DAY = /^\d{4}-\d{2}-\d{2}$/;

/** Whether a string is a real calendar day, not merely date-shaped. */
export function isValidDay(value: string | null | undefined): value is string {
  if (!value || !ISO_DAY.test(value)) return false;
  // `2026-02-31` is date-shaped and not a date. Round-tripping catches
  // it: Date normalises the overflow to March and the strings differ.
  const at = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(at.getTime()) && at.toISOString().slice(0, 10) === value;
}

/** Whether a range is one the API will accept. */
export function isValidRange(range: DateRange): boolean {
  if (!isValidDay(range.from) || !isValidDay(range.to)) return false;
  if (range.to < range.from) return false;
  return rangeLength(range) <= MAX_RANGE_DAYS;
}

/**
 * The window a URL asks for, or the default when it asks for nonsense.
 *
 * Never throws and never propagates a bad range to the API: a
 * bookmarked link with a typo shows this month rather than an error.
 */
export function rangeFromParams(
  params: { get(key: string): string | null },
  now: Date = new Date(),
): DateRange {
  const candidate = { from: params.get("from") ?? "", to: params.get("to") ?? "" };
  return isValidRange(candidate) ? candidate : presetRange(DEFAULT_PRESET, now);
}

/** The preset this range corresponds to, or null when it is custom. */
export function matchingPreset(range: DateRange, now: Date = new Date()): PresetId | null {
  for (const { id } of PRESETS) {
    const candidate = presetRange(id, now);
    if (candidate.from === range.from && candidate.to === range.to) return id;
  }
  return null;
}

export type AttributionMode = "first_touch" | "last_touch";

export const ATTRIBUTION_MODES: { id: AttributionMode; label: string; hint: string }[] = [
  {
    id: "first_touch",
    label: "최초 유입",
    hint: "전환을 처음 데려온 경로에 귀속합니다.",
  },
  {
    id: "last_touch",
    label: "최종 유입",
    hint: "전환 직전의 마지막 비직접 경로에 귀속합니다.",
  },
];

export interface AcquisitionSummary {
  range: RangeMeta;
  mode: AttributionMode;
  visitors: number;
  signups: number;
  conversions: number;
  revenue_krw: number;
  signup_rate: number | null;
  conversion_rate: number | null;
  /** Accounts predating attribution. Shown as 기존 회원, never as direct. */
  unattributed_users: number;
}

export interface ChannelRow {
  key: string;
  label: string;
  source: string;
  medium: string;
  visitors: number;
  signups: number;
  conversions: number;
  revenue_krw: number;
  signup_rate: number | null;
  conversion_rate: number | null;
}

export interface CampaignRow {
  source: string;
  medium: string;
  campaign: string | null;
  visitors: number;
  signups: number;
  conversions: number;
  revenue_krw: number;
}

export async function fetchAcquisitionSummary(
  range: DateRange,
  mode: AttributionMode,
  signal?: AbortSignal,
): Promise<AcquisitionSummary> {
  return readJson(`/v1/admin/acquisition/summary${query({ ...rangeQuery(range), mode })}`, {
    signal,
  });
}

export async function fetchAcquisitionChannels(
  range: DateRange,
  mode: AttributionMode,
  signal?: AbortSignal,
): Promise<ChannelRow[]> {
  return readJson(`/v1/admin/acquisition/channels${query({ ...rangeQuery(range), mode })}`, {
    signal,
  });
}

export async function fetchAcquisitionCampaigns(
  range: DateRange,
  mode: AttributionMode,
  signal?: AbortSignal,
): Promise<CampaignRow[]> {
  return readJson(`/v1/admin/acquisition/campaigns${query({ ...rangeQuery(range), mode })}`, {
    signal,
  });
}

/** `12.4%`, or a dash when there is no denominator to divide by. */
export function formatRate(rate: number | null): string {
  return rate === null ? "—" : `${(rate * 100).toFixed(1)}%`;
}

/** `2026.08.01 ~ 2026.08.28`, the way the range is shown to an operator. */
export function formatRange(range: DateRange): string {
  const dotted = (iso: string) => iso.replaceAll("-", ".");
  return range.from === range.to
    ? dotted(range.from)
    : `${dotted(range.from)} ~ ${dotted(range.to)}`;
}

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

/** A window as the API reports it, with how the series was bucketed. */
export interface RangeMeta {
  start: string;
  end: string;
  days: number;
  bucketing: Bucketing;
}

/**
 * The equal-length window immediately before the selected one.
 *
 * A `*_delta_pct` of null means the previous period was zero. There is
 * no honest percentage from a zero base, so the console shows "신규"
 * rather than a number — never Infinity, never a bare 100%.
 */
export interface Comparison {
  start: string;
  end: string;
  revenue_krw: number;
  payment_count: number;
  new_users: number;
  generations: number;
  revenue_delta_pct: number | null;
  payment_delta_pct: number | null;
  user_delta_pct: number | null;
  generation_delta_pct: number | null;
}

export interface Dashboard {
  range: RangeMeta;
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
  comparison: Comparison;
}

export interface RevenueReport {
  range: RangeMeta;
  total_krw: number;
  payment_count: number;
  comparison: Comparison | null;
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

/** The query the API takes for a window. One place, so every call agrees. */
export function rangeQuery(range: DateRange): { start: string; end: string } {
  return { start: range.from, end: range.to };
}

export async function fetchDashboard(
  range: DateRange,
  signal?: AbortSignal,
): Promise<Dashboard> {
  return readJson<Dashboard>(`/v1/admin/dashboard${query(rangeQuery(range))}`, { signal });
}

export async function fetchRevenue(
  range: DateRange,
  signal?: AbortSignal,
): Promise<RevenueReport> {
  return readJson<RevenueReport>(`/v1/admin/analytics/revenue${query(rangeQuery(range))}`, {
    signal,
  });
}

/**
 * A delta as the console prints it: `+12.4%`, `-3.1%`, or null.
 *
 * Null stays null all the way to the component, which renders "신규".
 * Turning it into a number anywhere in between is how a zero base
 * becomes a misleading percentage.
 */
export function formatDelta(pct: number | null): string | null {
  if (pct === null) return null;
  const sign = pct > 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)}%`;
}

/** Which way a delta points, for colour. Zero is neither. */
export function deltaTone(pct: number | null): "up" | "down" | "flat" {
  if (pct === null || pct === 0) return "flat";
  return pct > 0 ? "up" : "down";
}

/** The axis label for a bucket: a day, the week it starts, or a month. */
export function formatBucket(day: string, bucketing: Bucketing): string {
  if (bucketing === "month") {
    const at = new Date(`${day}T00:00:00+09:00`);
    if (Number.isNaN(at.getTime())) return day;
    return new Intl.DateTimeFormat("ko-KR", {
      year: "numeric",
      month: "short",
      timeZone: "Asia/Seoul",
    }).format(at);
  }
  const label = formatDay(day);
  return bucketing === "week" ? `${label} 주` : label;
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
