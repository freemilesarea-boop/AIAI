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

import type { ReactNode } from "react";

import { Bucket, formatCount, formatDay, formatWon } from "@/lib/admin";

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
}: {
  caption: string;
  rows: Bucket[];
  format: (value: number) => string;
}) {
  return (
    <table className="sr-only">
      <caption>{caption}</caption>
      <thead>
        <tr>
          <th scope="col">날짜</th>
          <th scope="col">값</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.day}>
            <th scope="row">{formatDay(row.day)}</th>
            <td>{format(row.value)}</td>
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
}: {
  title: string;
  caption?: string;
  data: Bucket[];
  format?: (value: number) => string;
  emptyLabel?: string;
}) {
  const peak = Math.max(1, ...data.map((d) => d.value));

  return (
    <Frame title={title} caption={caption}>
      {data.length === 0 ? (
        <NoData label={emptyLabel} />
      ) : (
        <>
          {/* Each bar is a *direct* child of the track, and the track is
              the element with the definite height. A percentage height
              resolves against the parent's height, so a wrapper sized by
              its content (the default under `items-end`) would resolve
              every bar to zero and draw an empty chart holding correct
              numbers — which is worse than an obviously broken one. */}
          <div className="flex h-32 items-end gap-1" aria-hidden="true">
            {data.map((point) => (
              <div
                key={point.day}
                className="min-w-0 flex-1 rounded-t-[3px] bg-[var(--accent)]"
                style={{ height: `${Math.max(2, (point.value / peak) * 100)}%` }}
                title={`${formatDay(point.day)} · ${format(point.value)}`}
              />
            ))}
          </div>
          <div className="flex justify-between text-[11px] text-[var(--text-muted)]" aria-hidden="true">
            <span>{formatDay(data[0].day)}</span>
            <span>{formatDay(data[data.length - 1].day)}</span>
          </div>
          <DataTable caption={title} rows={data} format={format} />
        </>
      )}
    </Frame>
  );
}

export function RevenueChart({ data }: { data: Bucket[] }) {
  return (
    <BarChart
      title="일별 매출"
      caption="한국 시간 기준 하루 단위입니다."
      data={data}
      format={formatWon}
      emptyLabel="이 기간에는 결제가 없습니다"
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
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="flex flex-col gap-1 rounded-[var(--radius-lg)] border border-[var(--border-default)] bg-[var(--surface-raised)] p-4">
      <span className="text-xs text-[var(--text-muted)]">{label}</span>
      <span className="text-xl font-semibold tabular-nums text-[var(--text-primary)]">
        {value}
      </span>
      {hint ? <span className="text-xs text-[var(--text-muted)]">{hint}</span> : null}
    </div>
  );
}
