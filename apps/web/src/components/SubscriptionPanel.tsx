"use client";

/**
 * The subscription section of Settings, and the cancel flow.
 *
 * Everything rendered here is the server's answer to
 * `/v1/billing/status`. Nothing is computed locally from a plan name or
 * a date — a UI that worked out "you are subscribed" for itself would
 * eventually disagree with the server about whether someone had paid.
 *
 * The cancel confirmation is the part worth being careful about. PayApp
 * cancellation stops the *next* charge and does not refund the last one,
 * so the dialog says exactly that. Telling someone their access ends
 * immediately would be wrong; implying they get a refund would be worse.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { useEntitlement } from "@/components/EntitlementProvider";
import { Button, Card } from "@/components/ui";
import {
  STATUS_LABELS,
  cancelSubscription,
  fetchBillingStatus,
  fetchPayments,
  formatBillingDate,
  hasPaidAccess,
  type BillingStatus,
  type PaymentRecord,
} from "@/lib/billing";
import { formatPriceKrw } from "@/lib/plans";

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-[var(--border-subtle)] px-5 py-3 last:border-b-0">
      <dt className="text-sm text-[var(--text-secondary)]">{label}</dt>
      <dd className="text-sm font-medium text-[var(--text-primary)]">{value}</dd>
    </div>
  );
}

function CancelDialog({
  status,
  onClose,
  onCancelled,
}: {
  status: BillingStatus;
  onClose: () => void;
  onCancelled: (next: BillingStatus) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submitting = useRef(false);

  async function confirm() {
    if (submitting.current) return;
    submitting.current = true;
    setBusy(true);
    setError(null);
    try {
      onCancelled(await cancelSubscription());
    } catch {
      // The server leaves the subscription untouched when PayApp
      // refuses, so it is genuinely still active and saying so is
      // accurate rather than reassuring.
      setError("해지를 처리하지 못했습니다. 구독은 그대로 유지됩니다. 잠시 후 다시 시도해 주세요.");
      submitting.current = false;
      setBusy(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <button
        type="button"
        aria-label="닫기"
        onClick={() => !busy && onClose()}
        className="absolute inset-0 bg-black/60"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="cancel-title"
        className="relative w-full max-w-md rounded-[var(--radius-lg)] border border-[var(--border-default)] bg-[var(--surface-raised)] p-6 shadow-xl"
      >
        <h2 id="cancel-title" className="text-lg font-semibold text-[var(--text-primary)]">
          구독을 해지할까요?
        </h2>
        {/*
          The exact PayApp semantics, in the user's words. Cancellation
          prevents the next charge; it does not reverse the last one.
        */}
        <p className="mt-2 text-sm text-[var(--text-secondary)]">
          현재 결제 기간이 끝날 때까지 이용할 수 있으며 다음 결제부터 청구되지 않습니다.
          {status.period_end ? ` ${formatBillingDate(status.period_end)}까지 ${status.display_name} 플랜을 그대로 사용할 수 있습니다.` : ""}
        </p>
        <p className="mt-2 text-sm text-[var(--text-secondary)]">
          만든 음악은 해지 후에도 라이브러리에 그대로 남습니다.
        </p>

        {error ? (
          <p role="alert" className="mt-3 text-sm text-[var(--danger)]">
            {error}
          </p>
        ) : null}

        <div className="mt-5 flex items-center justify-end gap-2">
          <Button type="button" variant="secondary" onClick={onClose} disabled={busy}>
            유지하기
          </Button>
          <Button type="button" variant="danger" onClick={confirm} disabled={busy}>
            {busy ? "해지 중…" : "구독 해지"}
          </Button>
        </div>
      </div>
    </div>
  );
}

export function SubscriptionPanel() {
  const [status, setStatus] = useState<BillingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [confirming, setConfirming] = useState(false);
  const { refresh: refreshEntitlement } = useEntitlement();

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      setStatus(await fetchBillingStatus(signal));
    } catch {
      setStatus(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  if (loading) {
    return <p className="text-sm text-[var(--text-muted)]">구독 정보 불러오는 중…</p>;
  }

  const subscribed = hasPaidAccess(status);

  return (
    <>
      <Card className="p-0">
        <dl className="flex flex-col">
          <Row label="현재 플랜" value={status ? status.display_name : "Free"} />
          <Row
            label="구독 상태"
            value={status ? STATUS_LABELS[status.status] : STATUS_LABELS.NONE}
          />
          <Row
            label="이용 기간"
            value={
              status?.period_end && subscribed
                ? `${formatBillingDate(status.period_end)}까지`
                : "—"
            }
          />
          <Row
            label="다음 결제일"
            value={
              // Null whenever auto-renew is off — which is exactly the
              // state a cancelling user is in, and showing them a next
              // billing date then would be a lie.
              status?.next_renewal_at ? formatBillingDate(status.next_renewal_at) : "없음"
            }
          />
          <Row
            label="자동 갱신"
            value={status?.auto_renew ? "켜짐" : "꺼짐"}
          />
          <Row
            label="마지막 결제"
            value={status?.last_payment_at ? formatBillingDate(status.last_payment_at) : "—"}
          />
        </dl>
      </Card>

      {status?.status === "PAST_DUE" ? (
        <div
          role="status"
          className="rounded-[var(--radius-md)] border border-[var(--danger)] bg-[var(--danger-muted)] px-4 py-3 text-sm text-[var(--text-secondary)]"
        >
          <p className="font-medium text-[var(--text-primary)]">결제가 실패했습니다</p>
          <p className="mt-1">
            카드사에서 정기 결제가 승인되지 않아 유료 기능을 사용할 수 없습니다. 결제 수단을
            확인한 뒤 다시 구독해 주세요.
          </p>
        </div>
      ) : null}

      {status?.status === "CANCEL_PENDING" ? (
        <div
          role="status"
          className="rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-sunken)] px-4 py-3 text-sm text-[var(--text-secondary)]"
        >
          해지가 예약되었습니다. {formatBillingDate(status.period_end)}까지 이용할 수 있으며
          다음 결제부터 청구되지 않습니다.
        </div>
      ) : null}

      {subscribed && status?.auto_renew ? (
        <div>
          <Button type="button" variant="secondary" onClick={() => setConfirming(true)}>
            구독 해지
          </Button>
        </div>
      ) : null}

      {confirming && status ? (
        <CancelDialog
          status={status}
          onClose={() => setConfirming(false)}
          onCancelled={(next) => {
            setStatus(next);
            setConfirming(false);
            refreshEntitlement();
          }}
        />
      ) : null}
    </>
  );
}

/**
 * The account's own payment history.
 *
 * Successful and failed alike — a failed renewal is something the user
 * needs to see, and hiding it would leave them wondering why their plan
 * stopped working.
 *
 * No provider identifiers appear here. The server does not send them.
 */
export function PaymentHistory() {
  const [rows, setRows] = useState<PaymentRecord[] | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        setRows(await fetchPayments(controller.signal));
      } catch {
        setRows([]);
      }
    })();
    return () => controller.abort();
  }, []);

  if (rows === null) {
    return <p className="text-sm text-[var(--text-muted)]">결제 내역 불러오는 중…</p>;
  }

  if (rows.length === 0) {
    return (
      <Card className="px-5 py-4">
        <p className="text-sm text-[var(--text-secondary)]">아직 결제 내역이 없습니다.</p>
      </Card>
    );
  }

  return (
    <Card className="p-0">
      <ul className="flex flex-col">
        {rows.map((row, index) => (
          <li
            key={`${row.paid_at}-${index}`}
            className="flex items-center justify-between gap-4 border-b border-[var(--border-subtle)] px-5 py-3 last:border-b-0"
          >
            <div className="flex flex-col gap-0.5">
              <span className="text-sm text-[var(--text-primary)]">
                {formatBillingDate(row.paid_at)}
              </span>
              <span className="text-xs text-[var(--text-muted)]">{row.plan_id.toUpperCase()}</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium text-[var(--text-primary)]">
                {formatPriceKrw(row.amount_krw)}
              </span>
              <span
                className={
                  "rounded-[var(--radius-full)] px-2 py-0.5 text-[11px] font-medium " +
                  (row.status === "SUCCEEDED"
                    ? "bg-[var(--brand-muted)] text-[var(--brand-text)]"
                    : "bg-[var(--danger-muted)] text-[var(--danger)]")
                }
              >
                {row.status === "SUCCEEDED" ? "결제 완료" : "결제 실패"}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </Card>
  );
}
