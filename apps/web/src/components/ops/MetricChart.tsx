"use client";

/**
 * A line chart, in SVG, with no chart library.
 *
 * Step 70 asks for the smallest thing that works, and this is a single
 * series against step number. Recharts, visx and d3 all bring a
 * dependency measured in hundreds of kilobytes to draw a polyline, and
 * the console renders at most a handful of these.
 *
 * What the drawing has to get right:
 *
 * **Only metrics that exist.** There is no placeholder chart. A trainer
 * that computes no validation loss produces no validation panel, because
 * an empty axis labelled "validation loss" reads as a number that has
 * not arrived yet rather than one that is never coming.
 *
 * **Sampling is disclosed.** A long run's series arrives thinned, and
 * the caption says so. A chart that quietly drew 600 of 400,000 points
 * while implying completeness is a chart that can hide a spike.
 *
 * **Simulated numbers stay labelled.** A dry run's chart is a real chart
 * of numbers nothing measured, and it says SIMULATED on it.
 *
 * **It is readable without seeing it.** The summary text under the chart
 * carries the range and the last value, so a screen reader gets the
 * finding rather than "graphic".
 */

import { useId, useMemo, useState } from "react";

import { decimal, num } from "@/lib/ops/format";

import type { MetricSeries } from "@/lib/ops/types";

/**
 * An axis label at a width that fits.
 *
 * `decimal` is right for a value an operator reads carefully; an axis
 * needs the magnitude at a glance, and "18040.000" is three characters
 * of noise where "18,040" says the same thing.
 */
function axisLabel(value: number): string {
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude < 0.001) return value.toExponential(1);
  if (magnitude >= 1000) return Math.round(value).toLocaleString();
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(4);
}

const WIDTH = 640;
const HEIGHT = 180;
const PADDING = { top: 12, right: 12, bottom: 26, left: 52 };

interface Placed {
  x: number;
  y: number;
  step: number;
  value: number;
}

export function MetricChart({ series }: { series: MetricSeries }) {
  const titleId = useId();
  const [hover, setHover] = useState<Placed | null>(null);

  const { path, points, min, max, firstStep, lastStep } = useMemo(() => {
    const values = series.points.map((point) => point.value);
    const lowest = Math.min(...values);
    const highest = Math.max(...values);
    const flat = highest === lowest;
    // A flat series has no range to scale against. Drawing it against a
    // fabricated one would put a constant learning rate at the bottom of
    // the chart as though it had fallen there; a flat line through the
    // middle is what actually happened.
    const span = flat ? 1 : highest - lowest;
    const steps = series.points.map((point, index) => point.step ?? index);
    const firstX = steps[0] ?? 0;
    const lastX = steps[steps.length - 1] ?? 1;
    const rangeX = lastX - firstX || 1;

    const plotWidth = WIDTH - PADDING.left - PADDING.right;
    const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;

    const placed: Placed[] = series.points.map((point, index) => {
      const step = steps[index];
      return {
        x: PADDING.left + ((step - firstX) / rangeX) * plotWidth,
        y: flat
          ? PADDING.top + plotHeight / 2
          : PADDING.top + plotHeight - ((point.value - lowest) / span) * plotHeight,
        step,
        value: point.value,
      };
    });

    return {
      path: placed.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" "),
      points: placed,
      min: lowest,
      max: highest,
      firstStep: firstX,
      lastStep: lastX,
    };
  }, [series]);

  if (series.points.length === 0) return null;

  const simulated = series.sources.includes("SIMULATED");
  const summary =
    `${series.metric_name}: ${series.total_points} point(s) from step ${num(firstStep)} to ` +
    `${num(lastStep)}. Lowest ${decimal(min)}, highest ${decimal(max)}, latest ` +
    `${decimal(series.last_value)}.` +
    (simulated ? " These values are SIMULATED and measure nothing." : "");

  return (
    <figure className="min-w-0">
      <figcaption className="mb-1 flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-xs font-medium text-[var(--text-primary)]">
          {series.metric_name}
          {series.unit && (
            <span className="ml-1 text-[var(--text-muted)]">({series.unit})</span>
          )}
        </span>
        <span className="text-[11px] text-[var(--text-muted)]">
          latest {series.last_value === null ? "UNKNOWN" : axisLabel(series.last_value)}
          {simulated && (
            <span className="ml-2 rounded-[var(--radius-sm)] bg-[var(--accent-muted)] px-1.5 py-0.5 text-[var(--accent)]">
              SIMULATED
            </span>
          )}
        </span>
      </figcaption>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-labelledby={titleId}
        className="h-[180px] w-full"
        preserveAspectRatio="none"
        onMouseLeave={() => setHover(null)}
      >
        <title id={titleId}>{summary}</title>

        <line
          x1={PADDING.left}
          y1={HEIGHT - PADDING.bottom}
          x2={WIDTH - PADDING.right}
          y2={HEIGHT - PADDING.bottom}
          stroke="var(--border-default)"
        />
        <line
          x1={PADDING.left}
          y1={PADDING.top}
          x2={PADDING.left}
          y2={HEIGHT - PADDING.bottom}
          stroke="var(--border-default)"
        />

        <text x={4} y={PADDING.top + 4} className="fill-[var(--text-muted)] text-[9px]">
          {axisLabel(max)}
        </text>
        <text x={4} y={HEIGHT - PADDING.bottom} className="fill-[var(--text-muted)] text-[9px]">
          {axisLabel(min)}
        </text>
        <text
          x={PADDING.left}
          y={HEIGHT - 8}
          className="fill-[var(--text-muted)] text-[9px]"
        >
          step {num(firstStep)}
        </text>
        <text
          x={WIDTH - PADDING.right}
          y={HEIGHT - 8}
          textAnchor="end"
          className="fill-[var(--text-muted)] text-[9px]"
        >
          step {num(lastStep)}
        </text>

        <path
          d={path}
          fill="none"
          stroke={simulated ? "var(--accent)" : "var(--brand)"}
          strokeWidth={1.5}
          strokeDasharray={simulated ? "4 3" : undefined}
          vectorEffect="non-scaling-stroke"
        />

        {hover && (
          <circle cx={hover.x} cy={hover.y} r={3} fill="var(--text-primary)" />
        )}

        {/* An invisible hit target per point, so a tooltip works without
            a library and without hijacking the whole surface. */}
        {points.map((point) => (
          <rect
            key={`${point.step}-${point.x}`}
            x={point.x - 3}
            y={PADDING.top}
            width={6}
            height={HEIGHT - PADDING.top - PADDING.bottom}
            fill="transparent"
            onMouseEnter={() => setHover(point)}
          />
        ))}
      </svg>

      <p className="mt-1 text-[11px] text-[var(--text-muted)]">
        {hover ? (
          <span className="text-[var(--text-secondary)]">
            step {num(hover.step)} · {decimal(hover.value)}
          </span>
        ) : series.sampled ? (
          <>
            Showing {series.points.length} of {num(series.total_points)} points — the series was
            sampled to fit.
          </>
        ) : (
          <>{num(series.total_points)} points</>
        )}
      </p>
    </figure>
  );
}
