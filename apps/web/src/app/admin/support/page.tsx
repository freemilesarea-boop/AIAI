"use client";

/**
 * The support queue.
 *
 * An operator may move a ticket's status and attach an internal note.
 * They may not edit what the customer wrote — a support record the
 * operator can rewrite is not a record, and the API refuses the attempt
 * as well as the button being absent.
 *
 * The internal note is never shown to the customer. It lives on the
 * ticket rather than in the reply thread precisely because the reply
 * thread is what the customer will eventually be shown, and the two
 * being one field is how a private remark ends up in someone's inbox.
 *
 * Replying to the customer is not implemented: BOORDA has no mail
 * provider configured, and a compose box that quietly saved a draft
 * nobody sends would be worse than not offering one.
 */

import { useCallback, useEffect, useState } from "react";

import { Button, Card, EmptyState, Skeleton, Tabs, inputClass } from "@/components/ui";
import { useToast } from "@/components/ui/Toast";
import {
  fetchTicket,
  fetchTickets,
  formatDateTime,
  updateTicket,
  type AdminTicket,
  type AdminTicketSummary,
} from "@/lib/admin";
import { CATEGORY_LABELS, STATUS_LABELS, type SupportStatus } from "@/lib/support";

const FILTERS: { value: string; label: string }[] = [
  { value: "", label: "전체" },
  { value: "OPEN", label: "접수됨" },
  { value: "IN_PROGRESS", label: "처리중" },
  { value: "RESOLVED", label: "답변완료" },
  { value: "CLOSED", label: "종료" },
];

const NEXT_STATUS: SupportStatus[] = ["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"];

function TicketDetail({
  reference,
  onChanged,
  onClose,
}: {
  reference: string;
  onChanged: () => void;
  onClose: () => void;
}) {
  const [ticket, setTicket] = useState<AdminTicket | null>(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const { notify, notifyError } = useToast();

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        const loaded = await fetchTicket(reference, controller.signal);
        setTicket(loaded);
        setNote(loaded.admin_note ?? "");
      } catch {
        if (controller.signal.aborted) return;
        setFailed(true);
      }
    })();
    return () => controller.abort();
  }, [reference]);

  const apply = useCallback(
    async (patch: { status?: SupportStatus; admin_note?: string }) => {
      setBusy(true);
      try {
        setTicket(await updateTicket(reference, patch));
        onChanged();
        notify("문의를 업데이트했습니다.");
      } catch {
        notifyError("업데이트에 실패했습니다.");
      } finally {
        setBusy(false);
      }
    },
    [reference, onChanged, notify, notifyError],
  );

  return (
    <Card className="flex flex-col gap-4 p-5">
      <div className="flex items-start justify-between gap-3">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">{reference}</h2>
        <Button variant="ghost" onClick={onClose}>
          닫기
        </Button>
      </div>

      {failed ? (
        <p className="text-sm text-[var(--danger)]">문의를 불러오지 못했습니다.</p>
      ) : ticket === null ? (
        <Skeleton className="h-32 w-full" />
      ) : (
        <>
          <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <div>
              <dt className="text-xs text-[var(--text-muted)]">회원</dt>
              <dd className="truncate text-[var(--text-primary)]">{ticket.user_email}</dd>
            </div>
            <div>
              <dt className="text-xs text-[var(--text-muted)]">유형</dt>
              <dd className="text-[var(--text-primary)]">
                {CATEGORY_LABELS[ticket.category] ?? ticket.category}
              </dd>
            </div>
            <div>
              <dt className="text-xs text-[var(--text-muted)]">상태</dt>
              <dd className="text-[var(--text-primary)]">{STATUS_LABELS[ticket.status]}</dd>
            </div>
            <div>
              <dt className="text-xs text-[var(--text-muted)]">접수</dt>
              <dd className="text-[var(--text-primary)]">{formatDateTime(ticket.created_at)}</dd>
            </div>
          </dl>

          <div className="flex flex-col gap-1">
            <h3 className="text-sm font-medium text-[var(--text-primary)]">{ticket.subject}</h3>
            {/* Plain text, rendered escaped. A ticket saying `<script>`
                is a customer describing a bug. */}
            <p className="whitespace-pre-wrap text-sm text-[var(--text-secondary)]">
              {ticket.message}
            </p>
            {ticket.context_url ? (
              <p className="text-xs text-[var(--text-muted)]">발생 위치: {ticket.context_url}</p>
            ) : null}
          </div>

          <div className="flex flex-wrap gap-2">
            {NEXT_STATUS.filter((s) => s !== ticket.status).map((status) => (
              <Button
                key={status}
                size="sm"
                busy={busy}
                onClick={() => void apply({ status })}
              >
                {STATUS_LABELS[status]}(으)로 변경
              </Button>
            ))}
          </div>

          <form
            className="flex flex-col gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              void apply({ admin_note: note });
            }}
          >
            <label className="text-sm font-medium text-[var(--text-primary)]" htmlFor="admin-note">
              내부 메모
            </label>
            <textarea
              id="admin-note"
              className={inputClass}
              rows={3}
              value={note}
              onChange={(event) => setNote(event.target.value)}
              placeholder="운영자만 볼 수 있는 메모입니다."
            />
            <p className="text-xs text-[var(--text-muted)]">
              고객에게는 표시되지 않습니다. 고객 회신 기능은 아직 제공되지 않습니다.
            </p>
            <div>
              <Button type="submit" variant="primary" busy={busy}>
                메모 저장
              </Button>
            </div>
          </form>
        </>
      )}
    </Card>
  );
}

export default function AdminSupportPage() {
  const [filter, setFilter] = useState("");
  const [data, setData] = useState<{ items: AdminTicketSummary[]; total: number } | null>(null);
  const [failed, setFailed] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setFailed(false);
    void (async () => {
      try {
        setData(await fetchTickets({ status: filter || undefined }, controller.signal));
      } catch {
        if (controller.signal.aborted) return;
        setFailed(true);
      }
    })();
    return () => controller.abort();
  }, [filter, reloadKey]);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-col gap-1">
          <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">
            고객문의
          </h1>
          <p className="text-sm text-[var(--text-secondary)]">
            {data ? `${data.total}건` : "불러오는 중…"}
          </p>
        </div>
        <Tabs label="상태" value={filter} onChange={setFilter} options={FILTERS} />
      </header>

      {selected ? (
        <TicketDetail
          reference={selected}
          onChanged={() => setReloadKey((k) => k + 1)}
          onClose={() => setSelected(null)}
        />
      ) : null}

      {failed ? (
        <EmptyState title="문의를 불러오지 못했습니다" description="잠시 후 다시 시도해 주세요." />
      ) : data === null ? (
        <Skeleton className="h-64 w-full" />
      ) : data.items.length === 0 ? (
        <EmptyState title="문의가 없습니다" description="이 조건에 해당하는 문의가 없습니다." />
      ) : (
        <Card className="overflow-x-auto">
          <table className="w-full min-w-[36rem] text-sm">
            <caption className="sr-only">고객문의 목록</caption>
            <thead>
              <tr className="border-b border-[var(--border-default)] text-left text-xs text-[var(--text-muted)]">
                <th scope="col" className="p-3 font-medium">
                  번호
                </th>
                <th scope="col" className="p-3 font-medium">
                  제목
                </th>
                <th scope="col" className="p-3 font-medium">
                  회원
                </th>
                <th scope="col" className="p-3 font-medium">
                  상태
                </th>
                <th scope="col" className="p-3 font-medium">
                  접수
                </th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((ticket) => (
                <tr
                  key={ticket.reference}
                  className="cursor-pointer border-b border-[var(--border-default)] last:border-0 hover:bg-[var(--surface-sunken)]"
                  onClick={() => setSelected(ticket.reference)}
                >
                  <td className="p-3 font-mono text-xs text-[var(--text-muted)]">
                    {ticket.reference}
                  </td>
                  <td className="max-w-xs truncate p-3 text-[var(--text-primary)]">
                    {ticket.subject}
                  </td>
                  <td className="p-3 text-[var(--text-secondary)]">{ticket.user_email}</td>
                  <td className="p-3 text-[var(--text-secondary)]">
                    {STATUS_LABELS[ticket.status]}
                  </td>
                  <td className="p-3 text-[var(--text-secondary)]">
                    {formatDateTime(ticket.created_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}
