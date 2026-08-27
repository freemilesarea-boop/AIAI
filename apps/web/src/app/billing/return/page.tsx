"use client";

/**
 * Where PayApp sends the browser back to. Proof of nothing.
 *
 * This is the page most likely to be got wrong in a payment
 * integration, and the mistake is always the same: reading the URL. A
 * return URL is reached by a user who paid, by a user who closed the
 * PayApp window, and by anyone who types it — and PayApp puts whatever
 * it likes in the query string. Treating arrival here as evidence of
 * payment would mean a subscription costs one bookmark.
 *
 * So this page does not read `useSearchParams` at all. It asks BOORDA's
 * own server what happened and renders that.
 *
 * It also has to handle the ordinary race. The user's browser is
 * redirected the instant PayApp finishes; the notification that confirms
 * the payment travels server-to-server and may arrive a moment later.
 * So the page polls for a short while, says 결제 확인 중 in the
 * meantime, and — this is the part that matters — never resolves the
 * uncertainty in the customer's favour by guessing. If the confirmation
 * has not arrived by the time polling stops, it says so honestly and
 * points at Settings, where the same server-side truth will appear
 * whenever it does.
 */

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { useEntitlement } from "@/components/EntitlementProvider";
import { ButtonLink, Card } from "@/components/ui";
import {
  fetchBillingStatus,
  formatBillingDate,
  hasPaidAccess,
  type BillingStatus,
} from "@/lib/billing";

/**
 * How long to wait for the notification before saying so.
 *
 * PayApp's server-to-server callback normally lands within seconds. Two
 * minutes is generous enough that a slow one is not reported as a
 * failure, and short enough that nobody watches a spinner wondering
 * whether the page is broken.
 */
const POLL_LIMIT_MS = 120_000;
const POLL_INTERVAL_MS = 3_000;

type Phase = "checking" | "active" | "unconfirmed" | "failed";

export default function BillingReturnPage() {
  const [status, setStatus] = useState<BillingStatus | null>(null);
  const [phase, setPhase] = useState<Phase>("checking");
  const { refresh: refreshEntitlement } = useEntitlement();
  const startedAt = useRef<number | null>(null);

  const classify = useCallback((next: BillingStatus, elapsedOut: boolean): Phase => {
    if (hasPaidAccess(next)) return "active";
    if (next.status === "PAST_DUE") return "failed";
    // Still pending. Only call it unconfirmed once we have waited.
    return elapsedOut ? "unconfirmed" : "checking";
  }, []);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      // `performance.now` rather than Date: a clock adjustment mid-poll
      // should not decide whether we keep waiting.
      startedAt.current ??= performance.now();
      const elapsedOut = performance.now() - startedAt.current > POLL_LIMIT_MS;
      try {
        const next = await fetchBillingStatus(controller.signal);
        if (cancelled) return;
        setStatus(next);
        const resolved = classify(next, elapsedOut);
        setPhase(resolved);
        if (resolved === "active") {
          // The sidebar and the meters read the entitlement, which the
          // payment has just changed.
          refreshEntitlement();
          return;
        }
        if (resolved === "checking") timer = setTimeout(tick, POLL_INTERVAL_MS);
      } catch {
        if (cancelled) return;
        setPhase(elapsedOut ? "unconfirmed" : "checking");
        if (!elapsedOut) timer = setTimeout(tick, POLL_INTERVAL_MS);
      }
    };

    void tick();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [classify, refreshEntitlement]);

  return (
    <div className="mx-auto flex max-w-lg flex-col gap-6 py-8">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">결제</h1>
        <p className="text-sm text-[var(--text-secondary)]">
          결제 결과는 부르다 서버에서 확인합니다.
        </p>
      </header>

      {phase === "checking" ? (
        <Card className="flex flex-col gap-2 p-6" data-testid="billing-checking">
          <p className="text-base font-medium text-[var(--text-primary)]">결제 확인 중…</p>
          <p className="text-sm text-[var(--text-secondary)]">
            결제사에서 결과가 도착하는 데 잠시 걸릴 수 있습니다. 이 페이지를 열어 두세요.
          </p>
        </Card>
      ) : null}

      {phase === "active" && status ? (
        <Card className="flex flex-col gap-3 p-6" data-testid="billing-active">
          <p className="text-base font-medium text-[var(--text-primary)]">
            구독이 활성화되었습니다
          </p>
          <dl className="flex flex-col gap-1 text-sm">
            <div className="flex justify-between gap-3">
              <dt className="text-[var(--text-secondary)]">플랜</dt>
              <dd className="font-medium text-[var(--text-primary)]">{status.display_name}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-[var(--text-secondary)]">이용 기간</dt>
              <dd className="text-[var(--text-primary)]">
                {formatBillingDate(status.period_end)}까지
              </dd>
            </div>
          </dl>
          <div className="mt-2 flex gap-2">
            <ButtonLink href="/create" variant="primary">
              음악 만들기
            </ButtonLink>
            <ButtonLink href="/settings#subscription" variant="secondary">
              구독 정보
            </ButtonLink>
          </div>
        </Card>
      ) : null}

      {phase === "failed" ? (
        <Card className="flex flex-col gap-3 p-6" data-testid="billing-failed">
          <p className="text-base font-medium text-[var(--text-primary)]">
            결제를 완료하지 못했습니다
          </p>
          <p className="text-sm text-[var(--text-secondary)]">
            카드사에서 결제가 승인되지 않았습니다. 다른 카드로 다시 시도해 주세요.
          </p>
          <Link
            href="/plans"
            className="w-fit text-sm font-medium text-[var(--brand-text)] underline underline-offset-4"
          >
            플랜으로 돌아가기
          </Link>
        </Card>
      ) : null}

      {phase === "unconfirmed" ? (
        <Card className="flex flex-col gap-3 p-6" data-testid="billing-unconfirmed">
          <p className="text-base font-medium text-[var(--text-primary)]">
            아직 결제 확인을 받지 못했습니다
          </p>
          {/*
            Deliberately not "결제에 실패했습니다". We do not know that.
            The confirmation may still arrive, and telling someone their
            payment failed when it did not is worse than telling them we
            are not sure yet.
          */}
          <p className="text-sm text-[var(--text-secondary)]">
            결제가 완료되었다면 잠시 후 설정에 반영됩니다. 계속 확인되지 않으면 고객센터로
            문의해 주세요. 중복 결제를 막기 위해 다시 결제하지 마세요.
          </p>
          <ButtonLink href="/settings#subscription" variant="secondary">
            구독 정보 확인
          </ButtonLink>
        </Card>
      ) : null}
    </div>
  );
}
