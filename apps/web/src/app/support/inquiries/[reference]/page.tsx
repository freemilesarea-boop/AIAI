"use client";

/**
 * One inquiry.
 *
 * A reference belonging to another account answers 404 server-side, so
 * this page's "not found" covers both cases without having to know which
 * it is — and without being able to reveal which it is.
 *
 * The message is rendered as text. Nothing here uses
 * `dangerouslySetInnerHTML`: a customer describing a bug by pasting
 * markup is a bug report, not markup.
 */

import { useParams, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

import { ButtonLink, Card } from "@/components/ui";
import { ApiError } from "@/lib/api";
import {
  CATEGORY_LABELS,
  STATUS_LABELS,
  fetchInquiry,
  formatFiled,
  type Ticket,
} from "@/lib/support";

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1 border-b border-[var(--border-subtle)] px-5 py-3 last:border-b-0 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
      <dt className="text-sm text-[var(--text-secondary)]">{label}</dt>
      <dd className="text-sm font-medium text-[var(--text-primary)]">{value}</dd>
    </div>
  );
}

export default function InquiryDetailPage() {
  const params = useParams<{ reference: string }>();
  const search = useSearchParams();
  const reference = params?.reference ?? "";
  const justFiled = search?.get("filed") === "1";

  const [ticket, setTicket] = useState<Ticket | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "missing" | "error">("loading");

  useEffect(() => {
    if (!reference) return;
    const controller = new AbortController();
    void (async () => {
      try {
        setTicket(await fetchInquiry(reference, controller.signal));
        setState("ready");
      } catch (caught) {
        if (controller.signal.aborted) return;
        setState(caught instanceof ApiError && caught.status === 404 ? "missing" : "error");
      }
    })();
    return () => controller.abort();
  }, [reference]);

  return (
    <div className="flex max-w-2xl flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)] sm:text-3xl">
          문의 상세
        </h1>
        <ButtonLink href="/support/inquiries" variant="ghost" size="sm" className="w-fit px-0">
          ← 내 문의내역
        </ButtonLink>
      </header>

      {justFiled && state === "ready" ? (
        <div
          role="status"
          className="rounded-[var(--radius-md)] border border-[var(--brand)] bg-[var(--brand-muted)] px-4 py-3 text-sm text-[var(--text-primary)]"
        >
          문의가 정상적으로 접수되었습니다.
        </div>
      ) : null}

      {state === "loading" ? (
        <p className="text-sm text-[var(--text-muted)]">불러오는 중…</p>
      ) : state === "missing" ? (
        <Card className="p-5">
          <p className="text-sm text-[var(--text-secondary)]">
            문의를 찾을 수 없습니다. 문의번호를 다시 확인해 주세요.
          </p>
        </Card>
      ) : state === "error" ? (
        <Card className="p-5">
          <p className="text-sm text-[var(--text-secondary)]">
            문의를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.
          </p>
        </Card>
      ) : ticket ? (
        <>
          <Card className="p-0">
            <dl className="flex flex-col">
              <Row
                label="문의번호"
                value={<span className="font-mono">{ticket.reference}</span>}
              />
              <Row label="문의 유형" value={CATEGORY_LABELS[ticket.category]} />
              <Row label="상태" value={STATUS_LABELS[ticket.status]} />
              <Row label="접수일" value={formatFiled(ticket.created_at)} />
            </dl>
          </Card>

          <Card className="p-5">
            <h2 className="text-sm font-semibold text-[var(--text-primary)]">{ticket.subject}</h2>
            {/*
              Text, deliberately. `whitespace-pre-line` keeps the line
              breaks the customer typed; React escapes the rest.
            */}
            <p className="mt-3 whitespace-pre-line text-sm leading-relaxed text-[var(--text-secondary)]">
              {ticket.message}
            </p>
          </Card>

          <p className="text-xs text-[var(--text-muted)]">
            답변이 등록되면 이 페이지에서 확인하실 수 있습니다.
          </p>
        </>
      ) : null}
    </div>
  );
}
