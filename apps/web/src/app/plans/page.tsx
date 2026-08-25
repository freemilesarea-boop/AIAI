"use client";

/**
 * Plans — a shell with three named tiers and no prices in it.
 *
 * Pricing and credit policy are undecided, so this page renders the
 * *structure* of a plan comparison and says "미정" wherever a number
 * would go. There is no checkout, no provider SDK and no subscribe
 * action, because there is no billing backend to subscribe to.
 *
 * The tiers come from `lib/plans.ts`. When pricing is settled, that file
 * gains values and this page shows them; nothing here changes.
 */

import { Card } from "@/components/ui";
import { PLANS, formatCredits, formatPriceKrw, type Plan } from "@/lib/plans";

function FeatureMark({ included }: { included: boolean | null }) {
  if (included === null) {
    return (
      <span aria-label="미정" className="text-[var(--text-muted)]">
        —
      </span>
    );
  }
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

function PlanCard({ plan }: { plan: Plan }) {
  return (
    <Card
      className={
        "flex flex-col gap-4 p-6" +
        (plan.highlighted ? " border-[var(--brand)] ring-1 ring-[var(--brand)]" : "")
      }
    >
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold text-[var(--text-primary)]">{plan.name}</h2>
          {plan.highlighted ? (
            <span className="rounded-[var(--radius-full)] bg-[var(--brand-muted)] px-2 py-0.5 text-[11px] font-medium text-[var(--brand-text)]">
              추천
            </span>
          ) : null}
        </div>
        <p className="text-sm text-[var(--text-secondary)]">{plan.tagline}</p>
      </div>

      <div className="flex flex-col gap-0.5">
        <p className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">
          {formatPriceKrw(plan.monthlyPriceKrw)}
        </p>
        <p className="text-xs text-[var(--text-muted)]">
          {plan.monthlyPriceKrw === null ? "가격이 아직 정해지지 않았습니다" : "월 결제"}
        </p>
      </div>

      <dl className="flex flex-col gap-2 border-t border-[var(--border-subtle)] pt-4 text-sm">
        <div className="flex items-center justify-between gap-3">
          <dt className="text-[var(--text-secondary)]">월 크레딧</dt>
          <dd className="font-medium text-[var(--text-primary)]">
            {formatCredits(plan.monthlyCredits)}
          </dd>
        </div>
        <div className="flex items-center justify-between gap-3">
          <dt className="text-[var(--text-secondary)]">최대 길이</dt>
          <dd className="font-medium text-[var(--text-primary)]">
            {plan.maxDurationSeconds === null ? "미정" : `${plan.maxDurationSeconds}초`}
          </dd>
        </div>
      </dl>

      <ul className="flex flex-col gap-2 border-t border-[var(--border-subtle)] pt-4 text-sm">
        {plan.features.map((feature) => (
          <li key={feature.label} className="flex items-center justify-between gap-3">
            <span className="text-[var(--text-secondary)]">{feature.label}</span>
            <FeatureMark included={feature.included} />
          </li>
        ))}
      </ul>

      {/*
        No subscribe button. A control that cannot do what it says is
        worse than no control: there is no payment provider, no
        subscription record and no price to charge.
      */}
      <p className="rounded-[var(--radius-md)] bg-[var(--surface-sunken)] px-3 py-2 text-xs text-[var(--text-muted)]">
        결제는 아직 준비 중입니다.
      </p>
    </Card>
  );
}

export default function PlansPage() {
  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)] sm:text-3xl">
          플랜
        </h1>
        <p className="max-w-prose text-sm text-[var(--text-secondary)]">
          부르다의 요금제입니다. 가격과 크레딧 정책은 아직 확정되지 않았고, 정해지지 않은
          항목은 &ldquo;미정&rdquo;으로 표시됩니다.
        </p>
      </header>

      <div className="grid gap-4 lg:grid-cols-3">
        {PLANS.map((plan) => (
          <PlanCard key={plan.id} plan={plan} />
        ))}
      </div>

      <Card className="p-5">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">아직 정해지지 않은 것</h2>
        <ul className="mt-2 flex list-disc flex-col gap-1 pl-5 text-sm text-[var(--text-secondary)]">
          <li>각 플랜의 가격</li>
          <li>크레딧 지급량과 차감 방식</li>
          <li>다운로드 및 상업적 이용 범위</li>
          <li>결제 수단</li>
        </ul>
      </Card>
    </div>
  );
}
