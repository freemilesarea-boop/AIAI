"use client";

/**
 * The one thing we have to ask for before sending someone to PayApp.
 *
 * PayApp's recurring registration requires `recvphone`, and BOORDA has
 * no phone number on an account. Rather than adding one to the profile —
 * where it would be collected from everyone, including the people who
 * never subscribe — it is asked for here, at the moment it is actually
 * needed, and stored against the billing record only.
 *
 * The dialog is deliberately small. It states the plan and the price it
 * is about to start charging, takes a phone number, and hands off. It
 * does not take a price: the amount shown comes from the server's plan
 * catalogue and the amount charged comes from the server's plan table,
 * and this component is not in that path at all.
 *
 * Double-submission is guarded here *and* on the server. The button
 * disables, which handles the honest double-click; the server's partial
 * unique index handles the two requests that arrive in the same
 * millisecond, which a disabled button cannot.
 */

import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui";
import { createCheckout, isValidKoreanMobile } from "@/lib/billing";
import { ApiError } from "@/lib/api";
import { formatPriceKrw, type Plan } from "@/lib/plans";

interface Props {
  plan: Plan;
  onClose: () => void;
}

/** Server error codes translated into something a person can act on. */
function messageFor(error: unknown): string {
  const code = error instanceof ApiError ? error.code : undefined;
  switch (code) {
    case "CHECKOUT_ALREADY_OPEN":
      return "이미 진행 중인 결제가 있습니다. 잠시 후 다시 시도하거나 결제 창을 확인해 주세요.";
    case "SUBSCRIPTION_ALREADY_ACTIVE":
      return "이미 구독 중입니다. 플랜을 바꾸려면 먼저 현재 구독을 해지해 주세요.";
    case "PAYMENT_PROVIDER_UNAVAILABLE":
      return "결제사에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.";
    case "BILLING_NOT_CONFIGURED":
      return "결제가 아직 준비되지 않았습니다.";
    default:
      return "결제를 시작하지 못했습니다. 잠시 후 다시 시도해 주세요.";
  }
}

export function CheckoutDialog({ plan, onClose }: Props) {
  const [phone, setPhone] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  // Survives re-render, unlike state: two clicks in one tick must not
  // both get past the guard.
  const submitting = useRef(false);

  useEffect(() => inputRef.current?.focus(), []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [busy, onClose]);

  const valid = isValidKoreanMobile(phone);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (submitting.current || !valid) return;
    submitting.current = true;
    setBusy(true);
    setError(null);
    try {
      const result = await createCheckout(plan.plan_id, phone);
      // PayApp's own hosted page. Replacing rather than pushing, so the
      // back button does not land on a checkout form for a payment that
      // may already have happened.
      window.location.assign(result.payurl);
    } catch (caught) {
      setError(messageFor(caught));
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
        aria-labelledby="checkout-title"
        className="relative w-full max-w-md rounded-[var(--radius-lg)] border border-[var(--border-default)] bg-[var(--surface-raised)] p-6 shadow-xl"
      >
        <h2 id="checkout-title" className="text-lg font-semibold text-[var(--text-primary)]">
          {plan.display_name} 구독 시작
        </h2>
        <p className="mt-1 text-sm text-[var(--text-secondary)]">
          매월 {formatPriceKrw(plan.monthly_price_krw)}이 청구됩니다. 결제는 PayApp에서
          진행되며, 카드 정보는 부르다에 저장되지 않습니다.
        </p>

        <form onSubmit={submit} className="mt-5 flex flex-col gap-2">
          <label htmlFor="checkout-phone" className="text-sm text-[var(--text-secondary)]">
            휴대폰 번호
          </label>
          <input
            id="checkout-phone"
            ref={inputRef}
            value={phone}
            onChange={(event) => setPhone(event.target.value)}
            inputMode="tel"
            autoComplete="tel"
            placeholder="010-1234-5678"
            disabled={busy}
            className="rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-sunken)] px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--brand)]"
          />
          <p className="text-xs text-[var(--text-muted)]">
            결제사에서 결제 요청을 보낼 때 사용합니다. 프로필에는 표시되지 않습니다.
          </p>

          {error ? (
            <p role="alert" className="mt-1 text-sm text-[var(--danger)]">
              {error}
            </p>
          ) : null}

          <div className="mt-4 flex items-center justify-end gap-2">
            <Button type="button" variant="secondary" onClick={onClose} disabled={busy}>
              취소
            </Button>
            <Button type="submit" variant="primary" disabled={busy || !valid}>
              {busy ? "결제창으로 이동 중…" : "결제 진행"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
