"use client";

/**
 * Several metrics over time, in SVG, with no chart library.
 *
 * The training console's `MetricChart` draws one series against a step
 * number. This draws several against wall-clock time, which changes two
 * things that matter more than the axis does.
 *
 * **A gap is a gap.** A bucket with no samples carries `null`, and the
 * line breaks there rather than being interpolated across. Drawing a
 * straight line through a quiet night would show a recovery that never
 * happened — and the quiet is usually why the numbers either side look
 * strange.
 *
 * **Thin buckets are marked.** A point computed from three requests is
 * drawn hollow. Without that, a spike from three requests and a spike
 * from three hundred look identical, and an operator chases the wrong
 * one.
 *
 * The summary beneath is not decoration either: it carries the range and
 * the sample count in text, so a screen reader gets the finding rather
 * than the word "graphic".
 */

import { useId, useMemo } from "react";

import type { TrendPoint } from "@/lib/ops/inference-types";

const WIDTH = 720;
const HEIGHT = 200;
const PADDING = { top: 12, right: 16, bottom: 28, left: 56 };

/** Below this a bucket is drawn hollow: too few samples to read as a level. */
const THIN_SAMPLE_COUNT = 10;

const SERIES_COLOURS = [
  "var(--brand)",
  "var(--accent)",
  "var(--danger, #dc2626)",
  "var(--text-secondary)",
];

export interface SeriesSpec {
  key: string;
  label: string;
  /** "rate" renders as a percentage; "seconds" renders as a duration. */
  unit: "rate" | "seconds" | "count";
}

function formatValue(value: number, unit: SeriesSpec["unit"]): string {
  if (unit === "rate") return `${(value * 100).toFixed(1)}%`;
  if (unit === "seconds") return `${value.toFixed(1)}s`;
  return value.toFixed(2);
}

function axisLabel(value: number, unit: SeriesSpec["unit"]): string {
  if (unit === "rate") return `${Math.round(value * 100)}%`;
  if (unit === "seconds") return value >= 60 ? `${Math.round(value / 60)}m` : `${Math.round(value)}s`;
  return value >= 1000 ? Math.round(value).toLocaleString() : value.toFixed(1);
}

function timeLabel(iso: string): string {
  const date = new Date(iso);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

interface Placed {
  x: number;
  y: number;
  value: number;
  sampleCount: number;
  start: string;
}

export function TimeSeriesChart({
  title,
  points,
  series,
  unit,
  caption,
}: {
  title: string;
  points: TrendPoint[];
  series: SeriesSpec[];
  unit: SeriesSpec["unit"];
  caption?: string;
}) {
  const titleId = useId();

  const { lines, maximum, hasAnyValue, firstLabel, lastLabel } = useMemo(() => {
    const values: number[] = [];
    for (const point of points) {
      for (const spec of series) {
        const value = point.values[spec.key];
        if (value !== null && value !== undefined) values.push(value);
      }
    }
    const highest = values.length ? Math.max(...values) : 0;
    // A little headroom so the top of a line is not on the frame, and a
    // floor so an all-zero chart still has a scale rather than dividing
    // by nothing.
    const ceiling = highest > 0 ? highest * 1.15 : unit === "rate" ? 0.1 : 1;

    const plotWidth = WIDTH - PADDING.left - PADDING.right;
    const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;
    const step = points.length > 1 ? plotWidth / (points.length - 1) : 0;

    const built = series.map((spec, index) => {
      // Segments rather than one path: a `null` bucket ends the current
      // segment and the next value starts a new one, so the line breaks
      // at a gap instead of jumping it.
      const segments: Placed[][] = [];
      let current: Placed[] = [];
      points.forEach((point, position) => {
        const value = point.values[spec.key];
        if (value === null || value === undefined) {
          if (current.length) segments.push(current);
          current = [];
          return;
        }
        current.push({
          x: PADDING.left + step * position,
          y: PADDING.top + plotHeight - (value / ceiling) * plotHeight,
          value,
          sampleCount: point.sample_count,
          start: point.start,
        });
      });
      if (current.length) segments.push(current);
      return { spec, colour: SERIES_COLOURS[index % SERIES_COLOURS.length], segments };
    });

    return {
      lines: built,
      maximum: ceiling,
      hasAnyValue: values.length > 0,
      firstLabel: points.length ? timeLabel(points[0].start) : "",
      lastLabel: points.length ? timeLabel(points[points.length - 1].start) : "",
    };
  }, [points, series, unit]);

  if (!points.length || !hasAnyValue) {
    return (
      <div className="rounded-[var(--radius-md)] border border-[var(--border-subtle)] bg-[var(--surface-overlay)] p-4">
        <h3 className="text-xs font-medium text-[var(--text-secondary)]">{title}</h3>
        <p className="mt-3 text-[11px] text-[var(--text-muted)]">
          No data in this window. Nothing is drawn rather than a flat line at zero — an empty
          chart and a healthy one must not look the same.
        </p>
      </div>
    );
  }

  const summary = lines
    .map((line) => {
      const last = line.segments.at(-1)?.at(-1);
      return last ? `${line.spec.label} ${formatValue(last.value, unit)}` : null;
    })
    .filter(Boolean)
    .join(", ");

  return (
    <figure className="rounded-[var(--radius-md)] border border-[var(--border-subtle)] bg-[var(--surface-overlay)] p-4">
      <figcaption className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <h3 id={titleId} className="text-xs font-medium text-[var(--text-secondary)]">
          {title}
        </h3>
        <ul className="flex flex-wrap gap-3">
          {lines.map((line) => (
            <li key={line.spec.key} className="flex items-center gap-1.5 text-[11px]">
              <span
                aria-hidden="true"
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: line.colour }}
              />
              <span className="text-[var(--text-muted)]">{line.spec.label}</span>
            </li>
          ))}
        </ul>
      </figcaption>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-labelledby={titleId}
        className="w-full"
      >
        <line
          x1={PADDING.left}
          y1={HEIGHT - PADDING.bottom}
          x2={WIDTH - PADDING.right}
          y2={HEIGHT - PADDING.bottom}
          stroke="var(--border-subtle)"
        />
        <text x={4} y={PADDING.top + 4} className="text-[10px]" fill="var(--text-muted)">
          {axisLabel(maximum, unit)}
        </text>
        <text
          x={4}
          y={HEIGHT - PADDING.bottom}
          className="text-[10px]"
          fill="var(--text-muted)"
        >
          {axisLabel(0, unit)}
        </text>
        <text
          x={PADDING.left}
          y={HEIGHT - 8}
          className="text-[10px]"
          fill="var(--text-muted)"
        >
          {firstLabel}
        </text>
        <text
          x={WIDTH - PADDING.right}
          y={HEIGHT - 8}
          textAnchor="end"
          className="text-[10px]"
          fill="var(--text-muted)"
        >
          {lastLabel}
        </text>

        {lines.map((line) =>
          line.segments.map((segment, index) => (
            <polyline
              key={`${line.spec.key}-${index}`}
              fill="none"
              stroke={line.colour}
              strokeWidth={1.5}
              points={segment.map((point) => `${point.x},${point.y}`).join(" ")}
            />
          )),
        )}

        {lines.map((line) =>
          line.segments.flat().map((point) => (
            <circle
              key={`${line.spec.key}-${point.start}`}
              cx={point.x}
              cy={point.y}
              r={2}
              // Hollow when the bucket is thin, so a spike from three
              // requests cannot be read as a spike from three hundred.
              fill={point.sampleCount < THIN_SAMPLE_COUNT ? "var(--surface-overlay)" : line.colour}
              stroke={line.colour}
              strokeWidth={1}
            >
              <title>
                {`${timeLabel(point.start)} — ${formatValue(point.value, unit)} (${point.sampleCount} samples)`}
              </title>
            </circle>
          )),
        )}
      </svg>

      <p className="mt-2 text-[11px] text-[var(--text-muted)]">
        {summary ? `Latest: ${summary}. ` : ""}
        {caption ?? ""} Hollow points are buckets with fewer than {THIN_SAMPLE_COUNT} samples.
      </p>
    </figure>
  );
}
