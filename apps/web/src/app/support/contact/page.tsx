"use client";

/**
 * Filing an inquiry.
 *
 * The form sends a category, a subject and a message. It does not send
 * an email address: the account's own is used, and is shown read-only so
 * the customer knows where the reply will go. Letting the form carry an
 * address would mean a ticket could claim to be from someone else.
 */

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import { useAuth } from "@/components/auth/AuthProvider";
import { Button, ButtonLink, Card } from "@/components/ui";
import { ApiError } from "@/lib/api";
import {
  CATEGORIES,
  MESSAGE_MAX_LENGTH,
  SUBJECT_MAX_LENGTH,
  createInquiry,
  type SupportCategory,
} from "@/lib/support";

export default function ContactPage() {
  const { user } = useAuth();
  const router = useRouter();
  const [category, setCategory] = useState<SupportCategory>("BILLING");
  const [subject, setSubject] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Survives re-render, unlike state: two clicks in one tick must not
  // both get past the guard and file two tickets.
  const submitting = useRef(false);

  const ready = subject.trim().length > 0 && message.trim().length > 0;

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (submitting.current || !ready) return;
    submitting.current = true;
    setBusy(true);
    setError(null);
    try {
      const ticket = await createInquiry({
        category,
        subject: subject.trim(),
        message: message.trim(),
        // Where they were when it happened. A clue for whoever reads the
        // ticket; nothing navigates to it.
        context_url: typeof window === "undefined" ? null : window.location.origin,
      });
      router.push(`/support/inquiries/${ticket.reference}?filed=1`);
    } catch (caught) {
      const status = caught instanceof ApiError ? caught.status : undefined;
      setError(
        status === 429
          ? "문의가 너무 많이 접수되었습니다. 잠시 후 다시 시도해 주세요."
          : status === 401
            ? "로그인이 필요합니다."
            : caught instanceof ApiError && caught.message
              ? caught.message
              : "문의를 접수하지 못했습니다. 잠시 후 다시 시도해 주세요.",
      );
      submitting.current = false;
      setBusy(false);
    }
  }

  const field =
    "rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-sunken)] px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--brand)]";

  return (
    <div className="flex max-w-2xl flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)] sm:text-3xl">
          문의하기
        </h1>
        <p className="text-sm text-[var(--text-secondary)]">
          접수하시면 문의번호가 발급되고, 내 문의내역에서 진행 상태를 확인할 수 있습니다.
        </p>
      </header>

      <Card className="p-5">
        <form onSubmit={submit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="category" className="text-sm text-[var(--text-secondary)]">
              문의 유형
            </label>
            <select
              id="category"
              value={category}
              onChange={(event) => setCategory(event.target.value as SupportCategory)}
              disabled={busy}
              className={field}
            >
              {CATEGORIES.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="subject" className="text-sm text-[var(--text-secondary)]">
              제목
            </label>
            <input
              id="subject"
              value={subject}
              onChange={(event) => setSubject(event.target.value)}
              maxLength={SUBJECT_MAX_LENGTH}
              disabled={busy}
              className={field}
            />
            <p className="text-right text-xs text-[var(--text-muted)]">
              {subject.length} / {SUBJECT_MAX_LENGTH}
            </p>
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="message" className="text-sm text-[var(--text-secondary)]">
              내용
            </label>
            <textarea
              id="message"
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              maxLength={MESSAGE_MAX_LENGTH}
              rows={9}
              disabled={busy}
              className={`${field} resize-y`}
            />
            <p className="text-right text-xs text-[var(--text-muted)]">
              {message.length} / {MESSAGE_MAX_LENGTH}
            </p>
          </div>

          {/*
            Read-only, and from the session. The form has no email field
            because a ticket must not be able to claim it came from
            somebody else.
          */}
          <div className="rounded-[var(--radius-md)] bg-[var(--surface-sunken)] px-3 py-2">
            <p className="text-xs text-[var(--text-muted)]">답변받을 이메일</p>
            <p className="text-sm text-[var(--text-primary)]">{user?.email ?? "—"}</p>
          </div>

          {error ? (
            <p role="alert" className="text-sm text-[var(--danger)]">
              {error}
            </p>
          ) : null}

          <div className="flex items-center gap-2">
            <Button type="submit" variant="primary" disabled={busy || !ready}>
              {busy ? "접수 중…" : "문의 접수"}
            </Button>
            <ButtonLink href="/support" variant="ghost">
              취소
            </ButtonLink>
          </div>
        </form>
      </Card>
    </div>
  );
}
