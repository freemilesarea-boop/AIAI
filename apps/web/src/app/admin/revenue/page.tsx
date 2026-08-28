"use client";

/**
 * Revenue, split into the two questions an operator actually asks:
 * how much came in, and how much of it was somebody new.
 *
 * The split is computed from payment history rather than read off a
 * flag, because nothing in the billing path records which a payment was
 * — see `revenue_split` in `admin_analytics`. Only successful payments
 * count: a checkout is not revenue and a failed charge is not revenue.
 */

import { useEffect, useState } from "react";

import { Kpi, RevenueChart } from "@/components/admin/Charts";
import { Card, EmptyState, SkeletonCard, Tabs } from "@/components/ui";
import {
  GRANULARITIES,
  fetchRevenue,
  formatCount,
  formatWon,
  type Granularity,
  type RevenueReport,
} from "@/lib/admin";

export default function AdminRevenuePage() {
  const [granularity, setGranularity] = useState<Granularity>("month");
  const [report, setReport] = useState<RevenueReport | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setFailed(false);
    void (async () => {
      try {
        setReport(await fetchRevenue(granularity, controller.signal));
      } catch {
        if (controller.signal.aborted) return;
        setFailed(true);
      }
    })();
    return () => controller.abort();
  }, [granularity]);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-col gap-1">
          <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">매출</h1>
          <p className="text-sm text-[var(--text-secondary)]">
            결제가 완료된 건만 집계합니다. 한국 시간(KST) 기준입니다.
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
            <RevenueChart data={report.series} />
          </Card>
          <p className="text-xs text-[var(--text-muted)]">
            기간: {report.range.start} ~ {report.range.end}
          </p>
        </>
      )}
    </div>
  );
}
