"use client";

/**
 * Plans — four tiers with real prices, and no checkout.
 *
 * The prices and allowances here are the ones the server enforces: they
 * are fetched from `/v1/plans`, which serves the same definition the
 * generation and download routes read. Nothing on this page is a
 * number typed into a component.
 *
 * There is still no payment provider, and the page says so plainly
 * rather than showing a subscribe button that opens nothing. A control
 * that cannot do what it says is worse than no control — the user
 * presses it, nothing happens, and they conclude the product is broken
 * rather than unfinished.
 */

import { useEffect, useState } from "react";

import { useEntitlement } from "@/components/EntitlementProvider";
import { Card } from "@/components/ui";
import { UsageBar } from "@/components/UsageMeter";
import { fetchPlans, formatPriceKrw, formatSongs, type Plan } from "@/lib/plans";

function Mark({ included }: { included: boolean }) {
  return included ? (
    <span aria-label="포함" className="text-[var(--brand)]">
      ✓
    </span>
  ) : (
    <span aria-label="미포함" className="text-[var(--text-muted)]">
      ✕
    </span>
  );
}

const TAGLINES: Record<string, string> = {
  free: "부르다를 처음 써보는 분께",
  basic: "꾸준히 만드는 분께",
  pro: "작업량이 많은 분께",
  creator: "매일 작업하는 분께",
};

function PlanCard({
  plan,
  current,
  checkoutAvailable,
}: {
  plan: Plan;
  current: boolean;
  checkoutAvailable: boolean;
}) {
  return (
    <Card
      className={
        "flex flex-col gap-4 p-6" +
        (plan.recommended ? " border-[var(--brand)] ring-1 ring-[var(--brand)]" : "")
      }
      data-testid={`plan-${plan.plan_id}`}
    >
      <div className="flex flex-col gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-lg font-semibold text-[var(--text-primary)]">{plan.display_name}</h2>
          {plan.recommended ? (
            <span className="rounded-[var(--radius-full)] bg-[var(--brand-muted)] px-2 py-0.5 text-[11px] font-medium text-[var(--brand-text)]">
              추천
            </span>
          ) : null}
          {current ? (
            <span className="rounded-[var(--radius-full)] border border-[var(--border-default)] px-2 py-0.5 text-[11px] font-medium text-[var(--text-secondary)]">
              현재 플랜
            </span>
          ) : null}
        </div>
        <p className="text-sm text-[var(--text-secondary)]">{TAGLINES[plan.plan_id] ?? ""}</p>
      </div>

      <div className="flex flex-col gap-0.5">
        <p className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">
          {formatPriceKrw(plan.monthly_price_krw)}
        </p>
        <p className="text-xs text-[var(--text-muted)]">
          {plan.monthly_price_krw === 0 ? "가입 시 바로 사용" : "월 결제"}
        </p>
      </div>

      <dl className="flex flex-col gap-2 border-t border-[var(--border-subtle)] pt-4 text-sm">
        <div className="flex items-center justify-between gap-3">
          <dt className="text-[var(--text-secondary)]">월 생성</dt>
          <dd className="font-medium text-[var(--text-primary)]">
            {formatSongs(plan.monthly_generation_limit)}
          </dd>
        </div>
      </dl>

      <ul className="flex flex-col gap-2 border-t border-[var(--border-subtle)] pt-4 text-sm">
        <li className="flex items-center justify-between gap-3">
          <span className="text-[var(--text-secondary)]">MP3 다운로드</span>
          <Mark included={plan.download_mp3} />
        </li>
        <li className="flex items-center justify-between gap-3">
          <span className="text-[var(--text-secondary)]">WAV 다운로드</span>
          <Mark included={plan.download_wav} />
        </li>
        <li className="flex items-center justify-between gap-3">
          <span className="text-[var(--text-secondary)]">상업적 이용</span>
          <Mark included={plan.commercial_use} />
        </li>
      </ul>

      {/*
        No subscribe button while `checkout_available` is false. The flag
        comes from the server, so the day a provider is connected this
        page starts offering checkout without being edited — and until
        then it cannot accidentally offer one.
      */}
      <p className="mt-auto rounded-[var(--radius-md)] bg-[var(--surface-sunken)] px-3 py-2 text-xs text-[var(--text-muted)]">
        {checkoutAvailable
          ? "결제 페이지로 이동합니다."
          : current
            ? "현재 사용 중인 플랜입니다."
            : "결제는 아직 준비 중입니다."}
      </p>
    </Card>
  );
}

export default function PlansPage() {
  const [plans, setPlans] = useState<Plan[]>([]);
  const [checkoutAvailable, setCheckoutAvailable] = useState(false);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const { entitlement } = useEntitlement();

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        const catalogue = await fetchPlans(controller.signal);
        setPlans(catalogue.plans);
        setCheckoutAvailable(catalogue.checkout_available);
      } catch {
        // An abort is not a failure. React's development double-effect
        // tears the first attempt down immediately, and treating that
        // rejection as an error would leave the page showing "불러오지
        // 못했습니다" over a catalogue that loaded fine on the retry.
        if (controller.signal.aborted) return;
        setFailed(true);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    })();
    return () => controller.abort();
  }, []);

  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)] sm:text-3xl">
          플랜
        </h1>
        <p className="max-w-prose text-sm text-[var(--text-secondary)]">
          부르다의 요금제입니다. 생성 한도는 매달 초기화되고, 실패한 생성은 한도에서 차감되지
          않습니다.
        </p>
      </header>

      {entitlement ? (
        <Card className="flex flex-col gap-3 p-5">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-sm font-semibold text-[var(--text-primary)]">현재 사용량</h2>
            <span className="rounded-[var(--radius-full)] bg-[var(--brand-muted)] px-2 py-0.5 text-[11px] font-medium text-[var(--brand-text)]">
              {entitlement.plan.display_name}
            </span>
          </div>
          <UsageBar entitlement={entitlement} />
        </Card>
      ) : null}

      {loading ? (
        <p className="text-sm text-[var(--text-muted)]">플랜 불러오는 중…</p>
      ) : failed ? (
        <Card className="p-5">
          <p className="text-sm text-[var(--text-secondary)]">
            요금제를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.
          </p>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {plans.map((plan) => (
            <PlanCard
              key={plan.plan_id}
              plan={plan}
              current={entitlement?.plan.plan_id === plan.plan_id}
              checkoutAvailable={checkoutAvailable}
            />
          ))}
        </div>
      )}

      <Card className="p-5">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">알아두실 점</h2>
        <ul className="mt-2 flex list-disc flex-col gap-1 pl-5 text-sm text-[var(--text-secondary)]">
          <li>생성 한도는 완성된 곡 기준입니다. 실패한 생성은 차감되지 않습니다.</li>
          <li>플랜을 바꿔도 이번 기간의 사용량은 초기화되지 않습니다.</li>
          <li>Free 플랜은 만든 곡을 들을 수 있지만 내려받을 수는 없습니다.</li>
          <li>결제 수단은 아직 연결되지 않았습니다.</li>
        </ul>
      </Card>
    </div>
  );
}
