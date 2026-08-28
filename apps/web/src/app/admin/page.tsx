"use client";

/**
 * The overview an operator opens first.
 *
 * One request fills the whole page. That is a deliberate shape: six
 * parallel fetches would give six independent spinners and six
 * independent ways to half-fail, and an operator reading a dashboard
 * where two panels loaded and one silently did not is worse off than one
 * looking at an error.
 *
 * The selected window lives in the URL, not in component state. So a
 * refresh keeps it, Back returns to the previous window, and a link
 * pasted into a chat opens on the same numbers the sender was looking
 * at — which is most of what makes a dashboard shareable.
 *
 * Zero renders as zero everywhere. BOORDA has generation switched off in
 * production, so an empty generation chart is the truth — a panel that
 * hid itself when a number was zero would teach its reader to distrust
 * the ones that remain.
 */

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";

import { DateRangePicker } from "@/components/admin/DateRangePicker";
import { GenerationChart, Kpi, PlanDonut, RevenueChart } from "@/components/admin/Charts";
import { Card, EmptyState, SkeletonCard } from "@/components/ui";
import {
  PLAN_LABELS,
  type DateRange,
  type Dashboard,
  fetchDashboard,
  formatCount,
  formatWon,
  rangeFromParams,
} from "@/lib/admin";
import { STATUS_LABELS } from "@/lib/support";

function AdminDashboard() {
  const router = useRouter();
  const params = useSearchParams();
  // The URL is the single source of truth for the window. Deriving it on
  // every render rather than mirroring it into state is what makes Back
  // and a pasted link behave without a second code path.
  const range = rangeFromParams(params ?? new URLSearchParams());

  const [data, setData] = useState<Dashboard | null>(null);
  const [failed, setFailed] = useState(false);

  const select = useCallback(
    (next: DateRange) => {
      // `push`, not `replace`: choosing a different window is a step an
      // operator should be able to walk back.
      router.push(`/admin?from=${next.from}&to=${next.to}`);
    },
    [router],
  );

  useEffect(() => {
    const controller = new AbortController();
    setFailed(false);
    void (async () => {
      try {
        setData(await fetchDashboard(range, controller.signal));
      } catch {
        // An abort is not a failure — React's development double-effect
        // tears the first attempt down immediately.
        if (controller.signal.aborted) return;
        setFailed(true);
      }
    })();
    return () => controller.abort();
  }, [range.from, range.to]); // eslint-disable-line react-hooks/exhaustive-deps

  const comparison = data?.comparison ?? null;

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">
            운영 대시보드
          </h1>
          <p className="text-sm text-[var(--text-secondary)]">
            모든 수치는 한국 시간(KST) 기준입니다.
          </p>
        </div>
        <DateRangePicker range={range} onChange={select} />
      </header>

      {failed ? (
        <EmptyState
          title="대시보드를 불러오지 못했습니다"
          description="잠시 후 다시 시도해 주세요."
        />
      ) : data === null ? (
        <SkeletonCard />
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Kpi
              label="기간 매출"
              value={formatWon(data.revenue_krw)}
              hint={`결제 ${formatCount(data.payment_count)}건`}
              delta={comparison?.revenue_delta_pct ?? null}
            />
            <Kpi
              label="결제 건수"
              value={`${formatCount(data.payment_count)}건`}
              delta={comparison?.payment_delta_pct ?? null}
            />
            <Kpi
              label="신규 회원"
              value={`${formatCount(data.users.new_in_range)}명`}
              hint={`전체 ${formatCount(data.users.total)}명 · 유료 ${formatCount(
                data.users.paid,
              )}명`}
              delta={comparison?.user_delta_pct ?? null}
            />
            <Kpi
              label="생성 요청"
              value={formatCount(data.generations.requested)}
              hint={`완료 ${formatCount(data.generations.completed)} · 실패 ${formatCount(
                data.generations.failed,
              )}`}
              delta={comparison?.generation_delta_pct ?? null}
            />
          </div>

          {comparison ? (
            <p className="text-xs text-[var(--text-muted)]">
              증감은 직전 동일 기간({comparison.start} ~ {comparison.end}) 대비입니다.
            </p>
          ) : null}

          <Card className="p-5">
            <RevenueChart data={data.revenue_series} bucketing={data.range.bucketing} />
          </Card>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card className="p-5">
              <PlanDonut data={data.plans} labels={PLAN_LABELS} />
            </Card>
            <Card className="p-5">
              <GenerationChart
                data={data.generation_series}
                bucketing={data.range.bucketing}
              />
            </Card>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card className="flex flex-col gap-3 p-5">
              <h2 className="text-sm font-semibold text-[var(--text-primary)]">고객문의</h2>
              <ul className="flex flex-col gap-1.5 text-sm">
                {Object.entries(data.support).map(([status, count]) => (
                  <li key={status} className="flex justify-between">
                    <span className="text-[var(--text-secondary)]">
                      {STATUS_LABELS[status as keyof typeof STATUS_LABELS] ?? status}
                    </span>
                    <span className="tabular-nums text-[var(--text-primary)]">
                      {formatCount(count)}건
                    </span>
                  </li>
                ))}
              </ul>
            </Card>
            <Card className="flex flex-col gap-3 p-5">
              <h2 className="text-sm font-semibold text-[var(--text-primary)]">다운로드</h2>
              <p className="text-xl font-semibold tabular-nums text-[var(--text-primary)]">
                {formatCount(data.downloads)}회
              </p>
              <p className="text-xs text-[var(--text-muted)]">
                저장(다운로드)만 집계합니다. 플레이어 재생은 포함되지 않습니다.
              </p>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

export default function AdminDashboardPage() {
  // `useSearchParams` needs a Suspense boundary in the app router.
  return (
    <Suspense fallback={<SkeletonCard />}>
      <AdminDashboard />
    </Suspense>
  );
}
