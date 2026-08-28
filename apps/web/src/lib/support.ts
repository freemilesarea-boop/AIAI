/**
 * Support: filing an inquiry and reading your own.
 *
 * The client sends a category, a subject and a message. It does not send
 * a user id, a status or a reference — those are server-owned, and the
 * request schema has nowhere to put them.
 *
 * Tickets are addressed by `reference` (`SUP-XXXXXXXX`), never by the
 * database id. That is not what protects them — every query is scoped to
 * the owner server-side — but it keeps the primary key off the wire.
 */

import { API_BASE_URL, ApiError } from "@/lib/api";

export type SupportCategory =
  | "BILLING"
  | "GENERATION"
  | "DOWNLOAD"
  | "ACCOUNT"
  | "BUG"
  | "FEATURE"
  | "OTHER";

export type SupportStatus = "OPEN" | "IN_PROGRESS" | "RESOLVED" | "CLOSED";

export interface TicketSummary {
  reference: string;
  category: SupportCategory;
  subject: string;
  status: SupportStatus;
  created_at: string;
}

export interface Ticket extends TicketSummary {
  message: string;
  context_url: string | null;
  updated_at: string;
  resolved_at: string | null;
}

/** The order the form offers them, most common first. */
export const CATEGORIES: { id: SupportCategory; label: string }[] = [
  { id: "BILLING", label: "결제 및 구독" },
  { id: "GENERATION", label: "음악 생성" },
  { id: "DOWNLOAD", label: "다운로드" },
  { id: "ACCOUNT", label: "계정" },
  { id: "BUG", label: "오류 신고" },
  { id: "FEATURE", label: "기능 제안" },
  { id: "OTHER", label: "기타" },
];

export const CATEGORY_LABELS: Record<SupportCategory, string> = Object.fromEntries(
  CATEGORIES.map((c) => [c.id, c.label]),
) as Record<SupportCategory, string>;

/**
 * What the customer reads, which is not what the database stores.
 *
 * Kept separate on purpose: renaming a status in the UI should not be a
 * migration, and "IN_PROGRESS" is not a thing to show a person.
 */
export const STATUS_LABELS: Record<SupportStatus, string> = {
  OPEN: "접수됨",
  IN_PROGRESS: "처리중",
  RESOLVED: "답변완료",
  CLOSED: "종료",
};

/** Server-enforced too. Mirrored so the form can say so before submitting. */
export const SUBJECT_MAX_LENGTH = 200;
export const MESSAGE_MAX_LENGTH = 5000;

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

export async function createInquiry(input: {
  category: SupportCategory;
  subject: string;
  message: string;
  context_url?: string | null;
}): Promise<Ticket> {
  return readJson<Ticket>("/v1/support/inquiries", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

export async function fetchInquiries(
  signal?: AbortSignal,
): Promise<{ items: TicketSummary[]; total: number }> {
  return readJson<{ items: TicketSummary[]; total: number }>("/v1/support/inquiries", { signal });
}

export async function fetchInquiry(reference: string, signal?: AbortSignal): Promise<Ticket> {
  return readJson<Ticket>(`/v1/support/inquiries/${encodeURIComponent(reference)}`, { signal });
}

export function formatFiled(iso: string): string {
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return "—";
  return new Intl.DateTimeFormat("ko-KR", { dateStyle: "long", timeStyle: "short" }).format(at);
}
