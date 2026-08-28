"use client";

/** The customer's own inquiries, newest first. */

import Link from "next/link";
import { useEffect, useState } from "react";

import { ButtonLink, Card, EmptyState } from "@/components/ui";
import {
  CATEGORY_LABELS,
  STATUS_LABELS,
  fetchInquiries,
  formatFiled,
  type TicketSummary,
} from "@/lib/support";

function StatusPill({ status }: { status: TicketSummary["status"] }) {
  const tone =
    status === "RESOLVED"
      ? "bg-[var(--brand-muted)] text-[var(--brand-text)]"
      : status === "CLOSED"
        ? "bg-[var(--surface-sunken)] text-[var(--text-muted)]"
        : "bg-[var(--accent-muted)] text-[var(--accent)]";
  return (
    <span
      className={`rounded-[var(--radius-full)] px-2 py-0.5 text-[11px] font-medium ${tone}`}
    >
      {STATUS_LABELS[status]}
    </span>
  );
}

export default function InquiriesPage() {
  const [items, setItems] = useState<TicketSummary[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        setItems((await fetchInquiries(controller.signal)).items);
      } catch {
        // An abort is not a failure — React's development double-effect
        // tears the first attempt down immediately.
        if (controller.signal.aborted) return;
        setFailed(true);
        setItems([]);
      }
    })();
    return () => controller.abort();
  }, []);

  return (
    <div className="flex max-w-3xl flex-col gap-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-col gap-1">
          <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)] sm:text-3xl">
            내 문의내역
          </h1>
          <p className="text-sm text-[var(--text-secondary)]">
            접수하신 문의와 진행 상태입니다.
          </p>
        </div>
        <ButtonLink href="/support/contact" variant="primary">
          문의하기
        </ButtonLink>
      </header>

      {items === null ? (
        <p className="text-sm text-[var(--text-muted)]">불러오는 중…</p>
      ) : failed ? (
        <Card className="p-5">
          <p className="text-sm text-[var(--text-secondary)]">
            문의내역을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.
          </p>
        </Card>
      ) : items.length === 0 ? (
        <EmptyState
          title="아직 접수한 문의가 없습니다"
          description="이용 중 문제가 있거나 궁금한 점이 있으면 언제든 문의해 주세요."
          action={
            <ButtonLink href="/support/contact" variant="primary">
              문의하기
            </ButtonLink>
          }
        />
      ) : (
        <ul className="flex flex-col gap-2">
          {items.map((ticket) => (
            <li key={ticket.reference}>
              <Link
                href={`/support/inquiries/${ticket.reference}`}
                className="block rounded-[var(--radius-md)] border border-[var(--border-subtle)] bg-[var(--surface-raised)] px-5 py-4 transition-colors hover:border-[var(--border-default)]"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-xs text-[var(--text-muted)]">
                    {ticket.reference}
                  </span>
                  <span className="text-xs text-[var(--text-secondary)]">
                    {CATEGORY_LABELS[ticket.category]}
                  </span>
                  <span className="ml-auto">
                    <StatusPill status={ticket.status} />
                  </span>
                </div>
                <p className="mt-1.5 truncate text-sm font-medium text-[var(--text-primary)]">
                  {ticket.subject}
                </p>
                <p className="mt-0.5 text-xs text-[var(--text-muted)]">
                  {formatFiled(ticket.created_at)}
                </p>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
