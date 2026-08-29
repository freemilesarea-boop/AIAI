"use client";

/**
 * Where customers came from, and which of those sources paid off.
 *
 * The two questions this answers are not the same, and the toggle at
 * the top is which one is being asked. 최초 유입 credits the source that
 * originally found somebody; 최종 유입 credits the last campaign that
 * brought them back before they converted. A channel can look excellent
 * under one and unremarkable under the other, and that difference is
 * the useful part rather than a discrepancy to reconcile.
 *
 * Every figure is event-period, not cohort: a visitor acquired in July
 * who pays in August is in July's visitors and August's conversions.
 * The page says so in words, because a reader who assumes cohort
 * reporting would draw the opposite conclusion from the same numbers.
 */

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";

import { CampaignLinkBuilder } from "@/components/admin/CampaignLinkBuilder";
import { DateRangePicker } from "@/components/admin/DateRangePicker";
import { Kpi } from "@/components/admin/Charts";
import { Card, EmptyState, Skeleton, SkeletonCard, Tabs } from "@/components/ui";
import {
  ATTRIBUTION_MODES,
  type AcquisitionSummary,
  type AttributionMode,
  type CampaignRow,
  type ChannelRow,
  type DateRange,
  fetchAcquisitionCampaigns,
  fetchAcquisitionChannels,
  fetchAcquisitionSummary,
  formatCount,
  formatRate,
  formatWon,
  rangeFromParams,
} from "@/lib/admin";

type SortKey = "visitors" | "signups" | "conversions" | "revenue_krw";

function isMode(value: string | null): value is AttributionMode {
  return value === "first_touch" || value === "last_touch";
}

function AcquisitionDashboard() {
  const router = useRouter();
  const params = useSearchParams() ?? new URLSearchParams();
  const range = rangeFromParams(params);
  const modeParam = params.get("mode");
  const mode: AttributionMode = isMode(modeParam) ? modeParam : "first_touch";

  const [summary, setSummary] = useState<AcquisitionSummary | null>(null);
  const [channels, setChannels] = useState<ChannelRow[] | null>(null);
  const [campaigns, setCampaigns] = useState<CampaignRow[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [sort, setSort] = useState<SortKey>("visitors");

  const go = useCallback(
    (next: DateRange, nextMode: AttributionMode) =>
      router.push(`/admin/acquisition?from=${next.from}&to=${next.to}&mode=${nextMode}`),
    [router],
  );

  useEffect(() => {
    const controller = new AbortController();
    setFailed(false);
    void (async () => {
      try {
        const [s, c, k] = await Promise.all([
          fetchAcquisitionSummary(range, mode, controller.signal),
          fetchAcquisitionChannels(range, mode, controller.signal),
          fetchAcquisitionCampaigns(range, mode, controller.signal),
        ]);
        setSummary(s);
        setChannels(c);
        setCampaigns(k);
      } catch {
        if (controller.signal.aborted) return;
        setFailed(true);
      }
    })();
    return () => controller.abort();
  }, [range.from, range.to, mode]); // eslint-disable-line react-hooks/exhaustive-deps

  const sorted = [...(campaigns ?? [])].sort((a, b) => b[sort] - a[sort]);
  const nothingYet =
    summary !== null && summary.visitors === 0 && summary.signups === 0 && channels?.length === 0;

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">
            유입 분석
          </h1>
          <p className="text-sm text-[var(--text-secondary)]">
            방문 · 가입 · 첫 결제가 해당 기간에 발생한 건을 유입 경로별로 집계합니다. 한국
            시간(KST) 기준입니다.
          </p>
        </div>
        <div className="flex flex-col items-start gap-2 lg:items-end">
          <DateRangePicker range={range} onChange={(next) => go(next, mode)} />
          <Tabs
            label="귀속 기준"
            value={mode}
            onChange={(next) => go(range, next)}
            options={ATTRIBUTION_MODES.map((m) => ({ value: m.id, label: m.label }))}
          />
        </div>
      </header>

      <p className="text-xs text-[var(--text-muted)]">
        {ATTRIBUTION_MODES.find((m) => m.id === mode)?.hint}
      </p>

      {failed ? (
        <EmptyState
          title="유입 데이터를 불러오지 못했습니다"
          description="잠시 후 다시 시도해 주세요."
        />
      ) : summary === null ? (
        <SkeletonCard />
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
            <Kpi label="방문자" value={`${formatCount(summary.visitors)}명`} />
            <Kpi label="가입자" value={`${formatCount(summary.signups)}명`} />
            <Kpi label="유료 전환" value={`${formatCount(summary.conversions)}건`} />
            <Kpi label="매출" value={formatWon(summary.revenue_krw)} />
            <Kpi
              label="가입 전환율"
              value={formatRate(summary.signup_rate)}
              hint="가입 ÷ 방문자"
            />
            <Kpi
              label="결제 전환율"
              value={formatRate(summary.conversion_rate)}
              hint="첫 결제 ÷ 방문자"
            />
          </div>

          {summary.unattributed_users > 0 ? (
            <p className="text-xs text-[var(--text-muted)]">
              유입 경로가 기록되지 않은 회원 {formatCount(summary.unattributed_users)}명은 집계에
              포함되지 않습니다. 유입 분석 도입 이전에 가입한 기존 회원이며, 근거가 없으므로
              직접 유입으로 분류하지 않습니다.
            </p>
          ) : null}

          {nothingYet ? (
            <EmptyState
              title="아직 수집된 유입 데이터가 없습니다"
              description="UTM 링크를 사용하거나 새로운 방문이 발생하면 여기에 표시됩니다."
            />
          ) : (
            <>
              <Card className="overflow-x-auto">
                <table className="w-full min-w-[46rem] text-sm">
                  <caption className="sr-only">유입 채널별 성과</caption>
                  <thead>
                    <tr className="border-b border-[var(--border-default)] text-left text-xs text-[var(--text-muted)]">
                      <th scope="col" className="p-3 font-medium">유입 채널</th>
                      <th scope="col" className="p-3 text-right font-medium">방문자</th>
                      <th scope="col" className="p-3 text-right font-medium">가입</th>
                      <th scope="col" className="p-3 text-right font-medium">유료 전환</th>
                      <th scope="col" className="p-3 text-right font-medium">매출</th>
                      <th scope="col" className="p-3 text-right font-medium">가입 전환율</th>
                      <th scope="col" className="p-3 text-right font-medium">결제 전환율</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(channels ?? []).map((row) => (
                      <tr key={row.key} className="border-b border-[var(--border-default)] last:border-0">
                        <td className="p-3 text-[var(--text-primary)]">{row.label}</td>
                        <td className="p-3 text-right tabular-nums text-[var(--text-secondary)]">
                          {formatCount(row.visitors)}
                        </td>
                        <td className="p-3 text-right tabular-nums text-[var(--text-secondary)]">
                          {formatCount(row.signups)}
                        </td>
                        <td className="p-3 text-right tabular-nums text-[var(--text-secondary)]">
                          {formatCount(row.conversions)}
                        </td>
                        <td className="p-3 text-right tabular-nums text-[var(--text-primary)]">
                          {formatWon(row.revenue_krw)}
                        </td>
                        <td className="p-3 text-right tabular-nums text-[var(--text-muted)]">
                          {formatRate(row.signup_rate)}
                        </td>
                        <td className="p-3 text-right tabular-nums text-[var(--text-muted)]">
                          {formatRate(row.conversion_rate)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>

              <section className="flex flex-col gap-3">
                <div className="flex flex-wrap items-end justify-between gap-2">
                  <h2 className="text-sm font-semibold text-[var(--text-primary)]">
                    소스 · 매체 · 캠페인
                  </h2>
                  <Tabs
                    label="정렬"
                    value={sort}
                    onChange={setSort}
                    options={[
                      { value: "visitors" as SortKey, label: "방문자" },
                      { value: "signups" as SortKey, label: "가입" },
                      { value: "conversions" as SortKey, label: "전환" },
                      { value: "revenue_krw" as SortKey, label: "매출" },
                    ]}
                  />
                </div>
                {campaigns === null ? (
                  <Skeleton className="h-32 w-full" />
                ) : sorted.length === 0 ? (
                  <p className="text-sm text-[var(--text-muted)]">
                    캠페인 데이터가 아직 없습니다.
                  </p>
                ) : (
                  <Card className="overflow-x-auto">
                    <table className="w-full min-w-[42rem] text-sm">
                      <caption className="sr-only">캠페인별 성과</caption>
                      <thead>
                        <tr className="border-b border-[var(--border-default)] text-left text-xs text-[var(--text-muted)]">
                          <th scope="col" className="p-3 font-medium">소스</th>
                          <th scope="col" className="p-3 font-medium">매체</th>
                          <th scope="col" className="p-3 font-medium">캠페인</th>
                          <th scope="col" className="p-3 text-right font-medium">방문자</th>
                          <th scope="col" className="p-3 text-right font-medium">가입</th>
                          <th scope="col" className="p-3 text-right font-medium">유료 전환</th>
                          <th scope="col" className="p-3 text-right font-medium">매출</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sorted.map((row) => (
                          <tr
                            key={`${row.source}/${row.medium}/${row.campaign ?? ""}`}
                            className="border-b border-[var(--border-default)] last:border-0"
                          >
                            <td className="p-3 text-[var(--text-primary)]">{row.source}</td>
                            <td className="p-3 text-[var(--text-secondary)]">{row.medium}</td>
                            <td className="p-3 text-[var(--text-secondary)]">
                              {row.campaign ?? "—"}
                            </td>
                            <td className="p-3 text-right tabular-nums text-[var(--text-secondary)]">
                              {formatCount(row.visitors)}
                            </td>
                            <td className="p-3 text-right tabular-nums text-[var(--text-secondary)]">
                              {formatCount(row.signups)}
                            </td>
                            <td className="p-3 text-right tabular-nums text-[var(--text-secondary)]">
                              {formatCount(row.conversions)}
                            </td>
                            <td className="p-3 text-right tabular-nums text-[var(--text-primary)]">
                              {formatWon(row.revenue_krw)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </Card>
                )}
              </section>
            </>
          )}

          <CampaignLinkBuilder />
        </>
      )}
    </div>
  );
}

export default function AcquisitionPage() {
  return (
    <Suspense fallback={<SkeletonCard />}>
      <AcquisitionDashboard />
    </Suspense>
  );
}
