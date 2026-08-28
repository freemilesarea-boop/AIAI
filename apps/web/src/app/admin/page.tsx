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
 * Zero renders as zero everywhere. BOORDA has generation switched off in
 * production, so an empty generation chart is the truth — a panel that
 * hid itself when a number was zero would teach its reader to distrust
 * the ones that remain.
 */

import { useCallback, useEffect, useState } from "react";

import { BarChart, Kpi, PlanDonut, RevenueChart } from "@/components/admin/Charts";
import { Card, EmptyState, SkeletonCard, Tabs } from "@/components/ui";
import {
  GRANULARITIES,
  PLAN_LABELS,
  formatCount,
  formatWon,
  fetchDashboard,
  type Dashboard,
  type Granularity,
} from "@/lib/admin";
import { STATUS_LABELS } from "@/lib/support";

export default function AdminDashboardPage() {
  const [granularity, setGranularity] = useState<Granularity>("month");
  const [data, setData] = useState<Dashboard | null>(null);
  const [failed, setFailed] = useState(false);

  const load = useCallback(
    (signal: AbortSignal) => {
      setFailed(false);
      void (async () => {
        try {
          setData(await fetchDashboard(granularity, signal));
        } catch {
          // An abort is not a failure — React's development double-effect
          // tears the first attempt down immediately.
          if (signal.aborted) return;
          setFailed(true);
        }
      })();
    },
    [granularity],
  );

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal);
    return () => controller.abort();
  }, [load]);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-col gap-1">
          <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">
            운영 대시보드
          </h1>
          <p className="text-sm text-[var(--text-secondary)]">
            모든 수치는 한국 시간(KST) 기준입니다.
          </p>
        </div>
        <Tabs
          label="기간"
          value={granularity}
          onChange={setGranularity}
          options={GRANULARITIES.map((g) => ({ value: g.id, label: g.label }))}
        />
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
            />
            <Kpi label="오늘 매출" value={formatWon(data.revenue_today_krw)} />
            <Kpi
              label="전체 회원"
              value={`${formatCount(data.users.total)}명`}
              hint={`유료 ${formatCount(data.users.paid)}명 · 신규 ${formatCount(
                data.users.new_in_range,
              )}명`}
            />
            <Kpi
              label="생성 요청"
              value={formatCount(data.generations.requested)}
              hint={`완료 ${formatCount(data.generations.completed)} · 실패 ${formatCount(
                data.generations.failed,
              )}`}
            />
          </div>

          <Card className="p-5">
            <RevenueChart data={data.revenue_series} />
          </Card>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card className="p-5">
              <PlanDonut data={data.plans} labels={PLAN_LABELS} />
            </Card>
            <Card className="p-5">
              <BarChart
                title="일별 생성"
                caption="현재 프로덕션에서는 생성이 비활성화되어 있습니다."
                data={data.generation_series}
                emptyLabel="이 기간에는 생성 요청이 없습니다"
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
