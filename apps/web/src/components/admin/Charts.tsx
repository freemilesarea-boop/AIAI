"use client";

/**
 * Charts, drawn as SVG by hand.
 *
 * No charting library. Three shapes are needed — bars over days, a line,
 * a donut — and a dependency for that would add more bundle than the
 * whole console. Hand-drawn SVG also makes the accessibility work
 * possible rather than fought against: each chart carries a table of the
 * same numbers for screen readers, because a trend nobody can read is
 * not a trend anybody can act on.
 *
 * Every one of these renders an empty state rather than an empty box. A
 * dashboard whose charts vanish when a figure is zero teaches its
 * operator to distrust it, and zero is a correct answer here — BOORDA
 * has generation switched off in production today.
 */

import { useState, type ReactNode } from "react";

import { cx } from "@/components/ui";
import {
  type Bucket,
  type Bucketing,
  deltaTone,
  formatBucket,
  formatCount,
  formatDelta,
  formatWon,
} from "@/lib/admin";

/**
 * How wide a single bar may get.
 *
 * Without a cap, one data point stretches to the full plot width and
 * reads as a filled rectangle rather than a measurement — which is
 * exactly what production looked like with one day of revenue. Bars
 * still shrink below this when there are many.
 */
const MAX_BAR_WIDTH = "3.5rem";

function Frame({
  title,
  caption,
  children,
}: {
  title: string;
  caption?: string;
  children: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex flex-col gap-0.5">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">{title}</h2>
        {caption ? <p className="text-xs text-[var(--text-muted)]">{caption}</p> : null}
      </div>
      {children}
    </section>
  );
}

function NoData({ label }: { label: string }) {
  return (
    <div className="flex h-32 items-center justify-center rounded-[var(--radius-md)] border border-dashed border-[var(--border-default)] text-xs text-[var(--text-muted)]">
      {label}
    </div>
  );
}

/**
 * The same numbers as a table, for anyone not reading the picture.
 *
 * Visually hidden rather than omitted: a sighted operator gets the
 * shape, everyone else gets the figures, and both are the same data.
 */
function DataTable({
  caption,
  rows,
  format,
  bucketing,
  secondaryLabel,
  formatSecondary,
}: {
  caption: string;
  rows: Bucket[];
  format: (value: number) => string;
  bucketing: Bucketing;
  secondaryLabel?: string;
  formatSecondary?: (value: number) => string;
}) {
  return (
    <table className="sr-only">
      <caption>{caption}</caption>
      <thead>
        <tr>
          <th scope="col">기간</th>
          <th scope="col">값</th>
          {secondaryLabel ? <th scope="col">{secondaryLabel}</th> : null}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.day}>
            <th scope="row">{formatBucket(row.day, bucketing)}</th>
            <td>{format(row.value)}</td>
            {secondaryLabel ? (
              <td>{(formatSecondary ?? formatCount)(row.secondary)}</td>
            ) : null}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function BarChart({
  title,
  caption,
  data,
  format = formatCount,
  emptyLabel = "이 기간에는 데이터가 없습니다",
  bucketing = "day",
  secondaryLabel,
  formatSecondary,
}: {
  title: string;
  caption?: string;
  data: Bucket[];
  format?: (value: number) => string;
  emptyLabel?: string;
  bucketing?: Bucketing;
  /** Name of the second figure, shown in the tooltip and the table. */
  secondaryLabel?: string;
  formatSecondary?: (value: number) => string;
}) {
  const peak = Math.max(1, ...data.map((d) => d.value));
  // Which bar the pointer or keyboard is on. One index, so hover and
  // focus cannot both claim the tooltip at once.
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const active = activeIndex === null ? null : (data[activeIndex] ?? null);

  return (
    <Frame title={title} caption={caption}>
      {data.length === 0 ? (
        <NoData label={emptyLabel} />
      ) : (
        <>
          <div className="relative">
            {/* The tooltip is supplementary. Every figure in it is also
                in the table below, so nothing here is reachable only by
                hovering — which would put the analytics out of reach of
                a keyboard and a screen reader both. */}
            {active ? (
              <div
                role="status"
                className="pointer-events-none absolute inset-x-0 top-0 z-10 mx-auto w-fit rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-overlay)] px-3 py-2 text-xs shadow-lg"
              >
                <p className="font-medium text-[var(--text-primary)]">
                  {formatBucket(active.day, bucketing)}
                </p>
                <p className="text-[var(--text-secondary)]">{format(active.value)}</p>
                {secondaryLabel ? (
                  <p className="text-[var(--text-muted)]">
                    {secondaryLabel} {(formatSecondary ?? formatCount)(active.secondary)}
                  </p>
                ) : null}
              </div>
            ) : null}

            {/* Each bar is a *direct* child of the track, and the track is
                the element with the definite height. A percentage height
                resolves against the parent's height, so a wrapper sized by
                its content (the default under `items-end`) would resolve
                every bar to zero and draw an empty chart holding correct
                numbers — which is worse than an obviously broken one. */}
            <div className="flex h-32 items-end justify-center gap-1">
              {data.map((point, index) => (
                <button
                  key={point.day}
                  type="button"
                  // Focusable so the tooltip is reachable by keyboard,
                  // and labelled so a screen reader gets the whole
                  // figure rather than an unnamed button.
                  aria-label={`${formatBucket(point.day, bucketing)} ${format(point.value)}`}
                  className="min-w-0 flex-1 rounded-t-[3px] bg-[var(--accent)] transition-opacity hover:opacity-80 focus-visible:opacity-80 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)]"
                  style={{
                    height: `${Math.max(2, (point.value / peak) * 100)}%`,
                    maxWidth: MAX_BAR_WIDTH,
                  }}
                  onMouseEnter={() => setActiveIndex(index)}
                  onMouseLeave={() => setActiveIndex((i) => (i === index ? null : i))}
                  onFocus={() => setActiveIndex(index)}
                  onBlur={() => setActiveIndex((i) => (i === index ? null : i))}
                />
              ))}
            </div>
          </div>

          <div
            className="flex justify-between text-[11px] text-[var(--text-muted)]"
            aria-hidden="true"
          >
            <span>{formatBucket(data[0].day, bucketing)}</span>
            {data.length > 1 ? (
              <span>{formatBucket(data[data.length - 1].day, bucketing)}</span>
            ) : null}
          </div>

          <DataTable
            caption={title}
            rows={data}
            format={format}
            bucketing={bucketing}
            secondaryLabel={secondaryLabel}
            formatSecondary={formatSecondary}
          />
        </>
      )}
    </Frame>
  );
}

/** How the chart describes its own bucket size, so the axis is not a guess. */
const BUCKET_CAPTION: Record<Bucketing, string> = {
  day: "한국 시간 기준 하루 단위입니다.",
  week: "한국 시간 기준 주 단위입니다. 주는 월요일에 시작합니다.",
  month: "한국 시간 기준 월 단위입니다.",
};

export function RevenueChart({
  data,
  bucketing = "day",
}: {
  data: Bucket[];
  bucketing?: Bucketing;
}) {
  return (
    <BarChart
      title="매출 추이"
      caption={BUCKET_CAPTION[bucketing]}
      data={data}
      format={formatWon}
      bucketing={bucketing}
      secondaryLabel="결제"
      formatSecondary={(n) => `${formatCount(n)}건`}
      emptyLabel="이 기간에는 결제가 없습니다"
    />
  );
}

export function GenerationChart({
  data,
  bucketing = "day",
}: {
  data: Bucket[];
  bucketing?: Bucketing;
}) {
  return (
    <BarChart
      title="생성 추이"
      caption={BUCKET_CAPTION[bucketing]}
      data={data}
      format={(n) => `생성 요청 ${formatCount(n)}건`}
      bucketing={bucketing}
      secondaryLabel="완료"
      formatSecondary={(n) => `${formatCount(n)}건`}
      emptyLabel="이 기간에는 생성 요청이 없습니다"
    />
  );
}

const DONUT_TONES = [
  "var(--text-muted)",
  "var(--accent)",
  "var(--brand-text)",
  "var(--success, #16a34a)",
];

export function PlanDonut({
  data,
  labels,
}: {
  data: { plan_id: string; count: number; share: number }[];
  labels: Record<string, string>;
}) {
  const total = data.reduce((sum, row) => sum + row.count, 0);

  if (total === 0) {
    return (
      <Frame title="요금제 분포">
        <NoData label="아직 회원이 없습니다" />
      </Frame>
    );
  }

  // A stroke-dasharray ring: each slice is an arc length on one circle,
  // which avoids computing arc paths and their edge cases at 0% and
  // 100%.
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;

  return (
    <Frame title="요금제 분포">
      <div className="flex items-center gap-5">
        <svg viewBox="0 0 100 100" className="h-28 w-28 shrink-0 -rotate-90" aria-hidden="true">
          {data.map((row, index) => {
            const length = (row.count / total) * circumference;
            const dash = `${length} ${circumference - length}`;
            const element = (
              <circle
                key={row.plan_id}
                cx="50"
                cy="50"
                r={radius}
                fill="none"
                stroke={DONUT_TONES[index % DONUT_TONES.length]}
                strokeWidth="14"
                strokeDasharray={dash}
                strokeDashoffset={-offset}
              />
            );
            offset += length;
            return element;
          })}
        </svg>
        <ul className="flex min-w-0 flex-col gap-1.5 text-sm">
          {data.map((row, index) => (
            <li key={row.plan_id} className="flex items-center gap-2">
              <span
                aria-hidden="true"
                className="h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ background: DONUT_TONES[index % DONUT_TONES.length] }}
              />
              <span className="text-[var(--text-secondary)]">
                {labels[row.plan_id] ?? row.plan_id}
              </span>
              <span className="ml-auto tabular-nums text-[var(--text-primary)]">
                {formatCount(row.count)}명
              </span>
              <span className="w-12 text-right tabular-nums text-[var(--text-muted)]">
                {(row.share * 100).toFixed(1)}%
              </span>
            </li>
          ))}
        </ul>
      </div>
    </Frame>
  );
}

export function Kpi({
  label,
  value,
  hint,
  delta,
}: {
  label: string;
  value: string;
  hint?: string;
  /**
   * Change against the previous equal-length period, or null when that
   * period was zero. Null is rendered "신규" rather than a percentage:
   * there is no honest one from a zero base, and a number here is one
   * an operator might act on.
   */
  delta?: number | null;
}) {
  const formatted = delta === undefined ? null : formatDelta(delta);
  const tone = deltaTone(delta ?? null);

  return (
    <div className="flex flex-col gap-1 rounded-[var(--radius-lg)] border border-[var(--border-default)] bg-[var(--surface-raised)] p-4">
      <span className="text-xs text-[var(--text-muted)]">{label}</span>
      <span className="text-xl font-semibold tabular-nums text-[var(--text-primary)]">
        {value}
      </span>
      {delta !== undefined ? (
        <span
          className={cx(
            "text-xs tabular-nums",
            tone === "up" && "text-[var(--success,#16a34a)]",
            tone === "down" && "text-[var(--danger)]",
            tone === "flat" && "text-[var(--text-muted)]",
          )}
        >
          {formatted ?? "신규"}
        </span>
      ) : null}
      {hint ? <span className="text-xs text-[var(--text-muted)]">{hint}</span> : null}
    </div>
  );
}
