"use client";

/**
 * Revenue, split into the two questions an operator actually asks:
 * how much came in, and how much of it was somebody new.
 *
 * The split is computed from payment history rather than read off a
 * flag, because nothing in the billing path records which a payment was
 * — see `revenue_split` in `admin_analytics`. Only successful payments
 * count: a checkout is not revenue and a failed charge is not revenue.
 *
 * The window lives in the URL, exactly as it does on the dashboard, so
 * the two pages share one notion of "the period being looked at".
 */

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";

import { DateRangePicker } from "@/components/admin/DateRangePicker";
import { Kpi, RevenueChart } from "@/components/admin/Charts";
import { Card, EmptyState, SkeletonCard } from "@/components/ui";
import {
  type DateRange,
  type RevenueReport,
  fetchRevenue,
  formatCount,
  formatRange,
  formatWon,
  rangeFromParams,
} from "@/lib/admin";

function AdminRevenue() {
  const router = useRouter();
  const params = useSearchParams();
  const range = rangeFromParams(params ?? new URLSearchParams());

  const [report, setReport] = useState<RevenueReport | null>(null);
  const [failed, setFailed] = useState(false);

  const select = useCallback(
    (next: DateRange) => router.push(`/admin/revenue?from=${next.from}&to=${next.to}`),
    [router],
  );

  useEffect(() => {
    const controller = new AbortController();
    setFailed(false);
    void (async () => {
      try {
        setReport(await fetchRevenue(range, controller.signal));
      } catch {
        if (controller.signal.aborted) return;
        setFailed(true);
      }
    })();
    return () => controller.abort();
  }, [range.from, range.to]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">매출</h1>
          <p className="text-sm text-[var(--text-secondary)]">
            결제가 완료된 건만 집계합니다. 한국 시간(KST) 기준입니다.
          </p>
        </div>
        <DateRangePicker range={range} onChange={select} />
      </header>

      {failed ? (
        <EmptyState title="매출을 불러오지 못했습니다" description="잠시 후 다시 시도해 주세요." />
      ) : report === null ? (
        <SkeletonCard />
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-3">
            <Kpi
              label="총 매출"
              value={formatWon(report.total_krw)}
              hint={`결제 ${formatCount(report.payment_count)}건`}
              delta={report.comparison?.revenue_delta_pct ?? null}
            />
            <Kpi
              label="신규 결제"
              value={formatWon(report.new_krw)}
              hint={`${formatCount(report.new_count)}건`}
            />
            <Kpi
              label="갱신 결제"
              value={formatWon(report.renewal_krw)}
              hint={`${formatCount(report.renewal_count)}건`}
            />
          </div>
          <Card className="p-5">
            <RevenueChart data={report.series} bucketing={report.range.bucketing} />
          </Card>
          <p className="text-xs text-[var(--text-muted)]">
            기간 {formatRange({ from: report.range.start, to: report.range.end })} ·{" "}
            {report.range.days}일
            {report.comparison
              ? ` · 직전 동일 기간 ${report.comparison.start} ~ ${report.comparison.end} 대비`
              : ""}
          </p>
        </>
      )}
    </div>
  );
}

export default function AdminRevenuePage() {
  return (
    <Suspense fallback={<SkeletonCard />}>
      <AdminRevenue />
    </Suspense>
  );
}
