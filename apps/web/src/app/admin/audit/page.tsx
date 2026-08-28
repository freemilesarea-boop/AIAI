"use client";

/**
 * What operators have done.
 *
 * Readable by any administrator rather than only super administrators: a
 * log that only some operators can see is a weaker deterrent than one
 * they can all see. There is no control here that edits or deletes an
 * entry, and no route behind one — an audit trail the audited can edit
 * is decoration.
 *
 * Entries record the action, not the content it touched. A note written
 * on a support ticket appears here as "a note was written", never as the
 * note's text.
 */

import { useEffect, useState } from "react";

import { Card, EmptyState, Skeleton, Tabs } from "@/components/ui";
import {
  ACTION_LABELS,
  fetchAudit,
  formatDateTime,
  type AuditEntry,
} from "@/lib/admin";

const FILTERS = [
  { value: "", label: "전체" },
  { value: "ADMIN_GRANTED", label: "권한 부여" },
  { value: "ADMIN_REVOKED", label: "권한 해제" },
  { value: "SUPPORT_STATUS_CHANGED", label: "문의 상태" },
  { value: "EMAIL_CAMPAIGN_CREATED", label: "이메일" },
];

export default function AdminAuditPage() {
  const [action, setAction] = useState("");
  const [data, setData] = useState<{ items: AuditEntry[]; total: number } | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setFailed(false);
    void (async () => {
      try {
        setData(await fetchAudit({ action: action || undefined }, controller.signal));
      } catch {
        if (controller.signal.aborted) return;
        setFailed(true);
      }
    })();
    return () => controller.abort();
  }, [action]);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-col gap-1">
          <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">
            활동 기록
          </h1>
          <p className="text-sm text-[var(--text-secondary)]">
            {data ? `${data.total}건` : "불러오는 중…"}
          </p>
        </div>
        <Tabs label="유형" value={action} onChange={setAction} options={FILTERS} />
      </header>

      {failed ? (
        <EmptyState title="기록을 불러오지 못했습니다" description="잠시 후 다시 시도해 주세요." />
      ) : data === null ? (
        <Skeleton className="h-64 w-full" />
      ) : data.items.length === 0 ? (
        <EmptyState title="기록이 없습니다" description="해당 유형의 활동이 아직 없습니다." />
      ) : (
        <Card className="overflow-x-auto">
          <table className="w-full min-w-[38rem] text-sm">
            <caption className="sr-only">운영자 활동 기록</caption>
            <thead>
              <tr className="border-b border-[var(--border-default)] text-left text-xs text-[var(--text-muted)]">
                <th scope="col" className="p-3 font-medium">
                  시각
                </th>
                <th scope="col" className="p-3 font-medium">
                  작업
                </th>
                <th scope="col" className="p-3 font-medium">
                  수행자
                </th>
                <th scope="col" className="p-3 font-medium">
                  대상
                </th>
                <th scope="col" className="p-3 font-medium">
                  상세
                </th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((entry) => (
                <tr key={entry.id} className="border-b border-[var(--border-default)] last:border-0">
                  <td className="whitespace-nowrap p-3 text-[var(--text-secondary)]">
                    {formatDateTime(entry.created_at)}
                  </td>
                  <td className="p-3 text-[var(--text-primary)]">
                    {ACTION_LABELS[entry.action] ?? entry.action}
                  </td>
                  <td className="p-3 text-[var(--text-secondary)]">{entry.actor_email}</td>
                  <td className="p-3 text-[var(--text-secondary)]">{entry.target_email ?? "—"}</td>
                  <td className="p-3 text-xs text-[var(--text-muted)]">
                    {Object.entries(entry.metadata)
                      .map(([key, value]) => `${key}: ${value}`)
                      .join(", ") || "—"}
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
