"use client";

/**
 * The console's shared surfaces.
 *
 * Built on the product's design tokens so the operator console looks
 * like the same system, and kept separate from `components/ui` because
 * the vocabulary is different: a customer never sees BLOCKED, a
 * capability signature, or a gate that was not evaluated.
 *
 * Three rules the components enforce rather than leave to callers.
 *
 * **Status is never colour alone.** Every pill carries a word, and the
 * ones that matter carry a shape too. A red dot means nothing to a
 * colour-blind operator at 2am, and the states here are the ones that
 * decide whether somebody stops a rented GPU.
 *
 * **UNKNOWN is visually distinct from PASS.** Not a lighter green: a
 * different treatment entirely, because the whole point of Phase 25's
 * preflight is that an unmeasured thing is not a satisfied one.
 *
 * **Dense tables scroll inside themselves.** The page never scrolls
 * sideways; a wide table gets its own overflow container, so a run list
 * with fifteen columns does not break the layout around it.
 */

import Link from "next/link";
import { useCallback, useState, type ReactNode } from "react";

import { cx } from "@/components/ui";
import { UNKNOWN } from "@/lib/ops/format";

/* ── panels ─────────────────────────────────────────────────────────── */

export function Panel({
  title,
  subtitle,
  actions,
  children,
  id,
  tone = "default",
}: {
  title: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  id?: string;
  tone?: "default" | "danger" | "warning";
}) {
  const border =
    tone === "danger"
      ? "border-[var(--danger)]/40"
      : tone === "warning"
        ? "border-[var(--accent)]/40"
        : "border-[var(--border-subtle)]";
  return (
    <section
      id={id}
      aria-labelledby={id ? `${id}-heading` : undefined}
      className={cx(
        "rounded-[var(--radius-lg)] border bg-[var(--surface-raised)]",
        border,
      )}
    >
      <header className="flex flex-wrap items-start justify-between gap-3 border-b border-[var(--border-subtle)] px-4 py-3">
        <div className="min-w-0">
          <h2
            id={id ? `${id}-heading` : undefined}
            className="text-sm font-semibold text-[var(--text-primary)]"
          >
            {title}
          </h2>
          {subtitle && (
            <div className="mt-0.5 text-xs text-[var(--text-muted)]">{subtitle}</div>
          )}
        </div>
        {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
      </header>
      <div className="px-4 py-4">{children}</div>
    </section>
  );
}

/**
 * A panel that failed on its own.
 *
 * Step 62: one section failing must not blank the console. Metrics
 * unavailable and logs still readable is a normal state of the world,
 * and a page that renders one error for everything hides which part is
 * actually broken.
 */
export function PanelError({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      role="alert"
      className="rounded-[var(--radius-md)] border border-[var(--danger)]/40 bg-[var(--danger-muted)]/40 px-3 py-2.5 text-xs text-[var(--danger)]"
    >
      <p>{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 underline underline-offset-2 hover:text-[var(--text-primary)]"
        >
          Try again
        </button>
      )}
    </div>
  );
}

export function Unavailable({ reason }: { reason: string | null | undefined }) {
  return (
    <p className="text-xs leading-relaxed text-[var(--text-muted)]">
      {reason ?? "Not available on this deployment."}
    </p>
  );
}

/* ── status ─────────────────────────────────────────────────────────── */

type Tone = "neutral" | "live" | "good" | "bad" | "warn" | "unknown";

const TONE_CLASS: Record<Tone, string> = {
  neutral: "bg-[var(--surface-overlay)] text-[var(--text-secondary)]",
  live: "bg-[var(--accent-muted)] text-[var(--accent)]",
  good: "bg-[var(--brand-muted)] text-[var(--brand-text)]",
  bad: "bg-[var(--danger-muted)] text-[var(--danger)]",
  warn: "bg-[var(--accent-muted)] text-[var(--accent)]",
  // Deliberately not a washed-out green. An unmeasured thing must not
  // read as a quiet pass.
  unknown:
    "bg-transparent text-[var(--text-muted)] border border-dashed border-[var(--border-strong)]",
};

/** Symbols, so status survives greyscale and colour blindness. */
const TONE_MARK: Record<Tone, string> = {
  neutral: "·",
  live: "●",
  good: "✓",
  bad: "✕",
  warn: "!",
  unknown: "?",
};

const STATUS_TONE: Record<string, Tone> = {
  // Run states
  DRAFT: "neutral",
  VALIDATING: "live",
  QUEUED: "neutral",
  STARTING: "live",
  RUNNING: "live",
  COMPLETED: "good",
  FAILED: "bad",
  CANCELLED: "neutral",
  LOST: "warn",
  // Experiments
  READY: "neutral",
  BLOCKED: "bad",
  ARCHIVED: "neutral",
  // Gates and preflight
  PASS: "good",
  FAIL: "bad",
  NOT_EVALUATED: "unknown",
  UNKNOWN: "unknown",
  // Liveness
  ONLINE: "good",
  STALE: "warn",
  OFFLINE: "bad",
  // Worker classes
  GPU_TRAINING_READY: "good",
  CUDA_TRAINING: "good",
  CUDA_EVALUATION: "neutral",
  DEVELOPMENT_ONLY: "neutral",
  UNVERIFIED: "unknown",
  UNAVAILABLE: "bad",
  // Checkpoints
  WRITING: "live",
  READY_CHECKPOINT: "good",
  CORRUPT: "bad",
  REJECTED: "bad",
  ADAPTER: "good",
  FULL_MODEL: "good",
  MOCK: "warn",
  // Qualification
  QUALIFIED: "good",
  HUMAN_REVIEW_REQUIRED: "warn",
  PENDING: "neutral",
  // Worker states
  IDLE: "neutral",
  RECEIVING: "live",
  PREFLIGHT: "live",
  CANCELLING: "warn",
  // System checks
  OK: "good",
  DEGRADED: "warn",
  // Provider circuits. CLOSED is the healthy one: a closed circuit
  // conducts. Worth stating, because "closed" reads as "shut" to
  // everyone who has not met a circuit breaker before.
  CLOSED: "good",
  OPEN: "bad",
  HALF_OPEN: "warn",
  AVAILABLE: "good",
  NOT_CONFIGURED: "unknown",
  AUTOMATIC: "neutral",
  MANUAL: "warn",
  // Phase 33. READY is absent on purpose: it already means "ready to
  // be run" for an experiment, where neutral is right, and a training
  // preflight's READY is a pass. The caller overrides the tone rather
  // than one word carrying two meanings.
  NOT_APPLICABLE: "neutral",
  PASSED: "good",
  NOT_RUN: "unknown",
  MEASURED: "good",
  // Arithmetic over measurements. Not a measurement, not a guess.
  DERIVED: "neutral",
  // An estimate is not a measurement and must not look like one.
  ESTIMATED: "warn",
  // Phase 34 capacity. QUALIFIED is absent for the same reason READY
  // is: the caller decides the tone, because the word means different
  // things in different places.
  INSUFFICIENT: "bad",
  MARGIN_LOW: "warn",
  REPRESENTATIVE: "good",
  PARTIALLY_REPRESENTATIVE: "warn",
  NOT_REPRESENTATIVE: "bad",
  COMPLETED_PROFILE: "good",
  PROFILE_TIMEOUT: "bad",
  // Phase 35. COMPLETED_VALID_SIGNAL and VALID_SIGNAL are absent for the
  // same reason READY and QUALIFIED are: the caller decides the tone.
  COMPLETED_INSUFFICIENT_SIGNAL: "warn",
  FAILED_NUMERIC: "bad",
  FAILED_RUNTIME: "bad",
  NUMERICALLY_UNSTABLE: "bad",
  NO_UPDATE: "bad",
  INSUFFICIENT_EVIDENCE: "unknown",
  // A synthetic fixture is never a failure and never a real-data pass.
  SYNTHETIC_FIXTURE: "warn",
  REAL_OPERATOR_AUTHORIZED: "good",
  REAL_RIGHTS_CLEARED: "good",
};

export function OpsStatus({
  status,
  label,
  title,
  tone: toneOverride,
}: {
  status: string;
  label?: string;
  title?: string;
  /**
   * Force a tone for a word that means different things in different
   * places. `READY` is the live case: an experiment that is READY has
   * not started, and a training preflight that is READY has passed.
   */
  tone?: Tone;
}) {
  const tone = toneOverride ?? STATUS_TONE[status] ?? "neutral";
  return (
    <span
      title={title}
      className={cx(
        "inline-flex items-center gap-1.5 rounded-[var(--radius-full)] px-2 py-0.5",
        "text-[11px] font-medium whitespace-nowrap",
        TONE_CLASS[tone],
      )}
    >
      <span aria-hidden="true" className="text-[9px] leading-none">
        {TONE_MARK[tone]}
      </span>
      {label ?? status}
    </span>
  );
}

/* ── key/value ──────────────────────────────────────────────────────── */

export function KeyValue({
  items,
  columns = 2,
}: {
  items: { label: string; value: ReactNode; hint?: string }[];
  columns?: 1 | 2 | 3;
}) {
  const grid =
    columns === 1
      ? "grid-cols-1"
      : columns === 3
        ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"
        : "grid-cols-1 sm:grid-cols-2";
  return (
    <dl className={cx("grid gap-x-6 gap-y-3", grid)}>
      {items.map((item) => (
        <div key={item.label} className="min-w-0">
          <dt className="text-[11px] font-medium tracking-wide text-[var(--text-muted)] uppercase">
            {item.label}
          </dt>
          <dd className="mt-0.5 text-sm break-words text-[var(--text-primary)]">{item.value}</dd>
          {item.hint && <p className="mt-0.5 text-[11px] text-[var(--text-muted)]">{item.hint}</p>}
        </div>
      ))}
    </dl>
  );
}

/** A value that reads as absent when nothing measured it. */
export function Maybe({ value }: { value: string | number | null | undefined }) {
  if (value === null || value === undefined || value === "" || value === UNKNOWN) {
    return (
      <span className="text-[var(--text-muted)] italic" title="Nobody has measured this">
        {UNKNOWN}
      </span>
    );
  }
  return <>{value}</>;
}

/* ── copy ───────────────────────────────────────────────────────────── */

export function CopyValue({
  value,
  display,
  label,
}: {
  value: string | null | undefined;
  display?: string;
  label: string;
}) {
  const [copied, setCopied] = useState(false);
  const copy = useCallback(() => {
    if (!value) return;
    void navigator.clipboard?.writeText(value).then(
      () => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1600);
      },
      () => setCopied(false),
    );
  }, [value]);

  if (!value) return <Maybe value={null} />;
  return (
    <span className="inline-flex max-w-full items-center gap-1.5">
      <code className="truncate font-mono text-xs text-[var(--text-primary)]" title={value}>
        {display ?? value}
      </code>
      <button
        type="button"
        onClick={copy}
        aria-label={`Copy ${label}`}
        className="shrink-0 rounded-[var(--radius-sm)] px-1 py-0.5 text-[10px] text-[var(--text-muted)] hover:bg-[var(--surface-hover)] hover:text-[var(--text-primary)]"
      >
        {copied ? "copied" : "copy"}
      </button>
    </span>
  );
}

/* ── tables ─────────────────────────────────────────────────────────── */

export interface Column<T> {
  key: string;
  header: string;
  render: (row: T) => ReactNode;
  /** Right-aligned for numbers, so digits line up down the column. */
  numeric?: boolean;
}

export function DataTable<T>({
  rows,
  columns,
  rowKey,
  href,
  caption,
  empty,
}: {
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T) => string;
  href?: (row: T) => string;
  caption: string;
  empty: ReactNode;
}) {
  if (rows.length === 0) return <>{empty}</>;
  return (
    // The table scrolls inside this box. The page around it does not.
    <div className="-mx-4 overflow-x-auto px-4">
      <table className="w-full min-w-[720px] border-collapse text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-[var(--border-default)] text-left">
            {columns.map((column) => (
              <th
                key={column.key}
                scope="col"
                className={cx(
                  "px-3 py-2 text-[11px] font-semibold tracking-wide text-[var(--text-muted)] uppercase",
                  column.numeric && "text-right",
                )}
              >
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const target = href?.(row);
            return (
              <tr
                key={rowKey(row)}
                className="border-b border-[var(--border-subtle)] last:border-0 hover:bg-[var(--surface-overlay)]"
              >
                {columns.map((column, index) => (
                  <td
                    key={column.key}
                    className={cx(
                      "px-3 py-2.5 align-top text-[var(--text-secondary)]",
                      column.numeric && "text-right tabular-nums",
                    )}
                  >
                    {target && index === 0 ? (
                      <Link
                        href={target}
                        className="font-medium text-[var(--text-primary)] hover:text-[var(--brand-text)]"
                      >
                        {column.render(row)}
                      </Link>
                    ) : (
                      column.render(row)
                    )}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── pagination ─────────────────────────────────────────────────────── */

export function Pagination({
  page,
  onOffset,
}: {
  page: { total: number; limit: number; offset: number; returned: number };
  onOffset: (offset: number) => void;
}) {
  const first = page.total === 0 ? 0 : page.offset + 1;
  const last = page.offset + page.returned;
  const hasPrevious = page.offset > 0;
  const hasNext = last < page.total;
  return (
    <nav
      aria-label="Pagination"
      className="flex flex-wrap items-center justify-between gap-3 pt-3 text-xs text-[var(--text-muted)]"
    >
      <p>
        {first}–{last} of {page.total.toLocaleString()}
      </p>
      <div className="flex gap-2">
        <button
          type="button"
          disabled={!hasPrevious}
          onClick={() => onOffset(Math.max(0, page.offset - page.limit))}
          className="rounded-[var(--radius-sm)] border border-[var(--border-default)] px-2.5 py-1 disabled:opacity-40"
        >
          Previous
        </button>
        <button
          type="button"
          disabled={!hasNext}
          onClick={() => onOffset(page.offset + page.limit)}
          className="rounded-[var(--radius-sm)] border border-[var(--border-default)] px-2.5 py-1 disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </nav>
  );
}

/* ── loading and empty ──────────────────────────────────────────────── */

export function SectionSkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div role="status" aria-label="Loading" className="space-y-2">
      {Array.from({ length: rows }).map((_, index) => (
        <div key={index} className="luber-skeleton h-8 w-full" />
      ))}
    </div>
  );
}

export function OpsEmpty({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="rounded-[var(--radius-md)] border border-dashed border-[var(--border-default)] px-4 py-8 text-center">
      <p className="text-sm font-medium text-[var(--text-primary)]">{title}</p>
      <p className="mx-auto mt-1 max-w-md text-xs leading-relaxed text-[var(--text-muted)]">
        {description}
      </p>
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

/* ── filters ────────────────────────────────────────────────────────── */

export function FilterSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
      <span>{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="rounded-[var(--radius-sm)] border border-[var(--border-default)] bg-[var(--surface-overlay)] px-2 py-1 text-xs text-[var(--text-primary)]"
      >
        <option value="">All</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}
