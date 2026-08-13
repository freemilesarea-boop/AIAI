"use client";

/**
 * LUBER UI primitives.
 *
 * Every component here reads from the design tokens in `globals.css`.
 * Nothing in the product should hand-roll a surface colour, a radius or
 * a focus ring — if a style is needed twice, it belongs here.
 *
 * Kept deliberately small and dependency-free: the existing stack
 * (React + Tailwind) covers this design cleanly, and pulling in a
 * component framework would add weight without adding capability.
 */

import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode, Ref } from "react";

function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

/* ── Button ─────────────────────────────────────────────────────────── */

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md" | "lg";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** Renders a busy label and blocks interaction. */
  busy?: boolean;
  /**
   * React 19 passes `ref` to function components as an ordinary prop, so
   * no `forwardRef` wrapper is needed. Declared explicitly because
   * `ButtonHTMLAttributes` does not include it.
   */
  ref?: Ref<HTMLButtonElement>;
}

const BUTTON_VARIANT: Record<ButtonVariant, string> = {
  primary:
    "bg-[var(--brand)] text-white hover:bg-[var(--brand-hover)] " +
    "disabled:bg-[var(--brand-muted)] disabled:text-[var(--text-muted)]",
  secondary:
    "bg-[var(--surface-overlay)] text-[var(--text-primary)] border border-[var(--border-default)] " +
    "hover:bg-[var(--surface-hover)] disabled:text-[var(--text-muted)]",
  ghost:
    "bg-transparent text-[var(--text-secondary)] hover:bg-[var(--surface-raised)] " +
    "hover:text-[var(--text-primary)]",
  danger:
    "bg-[var(--danger-muted)] text-[var(--danger)] border border-[var(--danger)]/40 " +
    "hover:bg-[var(--danger)]/20",
};

const BUTTON_SIZE: Record<ButtonSize, string> = {
  sm: "h-8 px-3 text-xs",
  md: "h-10 px-4 text-sm",
  lg: "h-12 px-6 text-base",
};

export function Button({
  variant = "secondary",
  size = "md",
  busy = false,
  disabled,
  className,
  children,
  ref,
  ...rest
}: ButtonProps) {
  return (
    <button
      {...rest}
      ref={ref}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-[var(--radius-md)] font-medium",
        "transition-colors disabled:cursor-not-allowed",
        BUTTON_VARIANT[variant],
        BUTTON_SIZE[size],
        className,
      )}
    >
      {children}
    </button>
  );
}

/* ── Card ───────────────────────────────────────────────────────────── */

export function Card({ className, children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      {...rest}
      className={cx(
        "rounded-[var(--radius-lg)] border border-[var(--border-subtle)]",
        "bg-[var(--surface-raised)] shadow-[var(--shadow-card)]",
        className,
      )}
    >
      {children}
    </div>
  );
}

/* ── Chip ───────────────────────────────────────────────────────────── */

export interface ChipProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  selected?: boolean;
}

export function Chip({ selected = false, className, children, ...rest }: ChipProps) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      {...rest}
      className={cx(
        // 36px min height: 30px was below a comfortable thumb target
        // on a phone, measured at 390px.
        "inline-flex min-h-9 items-center rounded-[var(--radius-full)] border px-3.5",
        "text-xs font-medium transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-50",
        selected
          ? "border-[var(--brand)] bg-[var(--brand-muted)] text-[var(--brand-text)]"
          : "border-[var(--border-default)] text-[var(--text-secondary)] " +
              "hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]",
        className,
      )}
    >
      {children}
    </button>
  );
}

/* ── Tabs ───────────────────────────────────────────────────────────── */

export interface TabsProps<T extends string> {
  value: T;
  onChange: (value: T) => void;
  options: { value: T; label: string }[];
  /** Accessible name for the tablist. */
  label: string;
}

export function Tabs<T extends string>({ value, onChange, options, label }: TabsProps<T>) {
  return (
    // Scrolls inside its own box rather than widening the page. Five
    // filters do not fit across 390px, and a tab strip that pushes the
    // whole document sideways breaks every other layout on the screen.
    <div className="max-w-full overflow-x-auto">
      <div
        role="tablist"
        aria-label={label}
        className="inline-flex rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-sunken)] p-1"
      >
        {options.map((option) => {
          const active = option.value === value;
          return (
            <button
              key={option.value}
              role="tab"
              type="button"
              aria-selected={active}
              onClick={() => onChange(option.value)}
              className={cx(
                "shrink-0 rounded-[var(--radius-sm)] px-2.5 py-1.5 text-sm font-medium",
                "transition-colors sm:px-4",
                active
                  ? "bg-[var(--surface-overlay)] text-[var(--text-primary)]"
                  : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]",
              )}
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* ── Field wrappers ─────────────────────────────────────────────────── */

export const inputClass =
  "w-full rounded-[var(--radius-md)] border border-[var(--border-default)] " +
  "bg-[var(--surface-sunken)] px-3 py-2.5 text-sm text-[var(--text-primary)] " +
  "placeholder:text-[var(--text-muted)] transition-colors " +
  "hover:border-[var(--border-strong)] focus:border-[var(--brand)] focus:outline-none " +
  "disabled:cursor-not-allowed disabled:opacity-60";

export const labelClass = "block text-sm font-medium text-[var(--text-primary)]";
export const hintClass = "mt-1.5 text-xs text-[var(--text-muted)]";
export const errorClass = "mt-1.5 text-xs text-[var(--danger)]";

/* ── Skeleton ───────────────────────────────────────────────────────── */

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden="true" className={cx("luber-skeleton", className)} />;
}

export function SkeletonCard() {
  return (
    <Card className="p-4">
      <div className="flex gap-4">
        <Skeleton className="h-16 w-16 shrink-0" />
        <div className="flex-1 space-y-2 py-1">
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-3 w-1/3" />
        </div>
      </div>
    </Card>
  );
}

/* ── Empty state ────────────────────────────────────────────────────── */

export interface EmptyStateProps {
  title: string;
  description: string;
  action?: ReactNode;
  icon?: ReactNode;
}

export function EmptyState({ title, description, action, icon }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center rounded-[var(--radius-lg)] border border-dashed border-[var(--border-default)] px-6 py-14 text-center">
      {icon && <div className="mb-4 text-[var(--text-muted)]">{icon}</div>}
      <h3 className="text-base font-semibold text-[var(--text-primary)]">{title}</h3>
      <p className="mt-1.5 max-w-sm text-sm text-[var(--text-secondary)]">{description}</p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

/* ── Status pill ────────────────────────────────────────────────────── */

/**
 * Factual, stage-based status. There is no percentage here on purpose:
 * the provider cannot report progress, and inventing a number would be
 * a lie the user would reasonably act on.
 */
export function StatusPill({ status }: { status: string }) {
  const map: Record<string, { label: string; className: string; live?: boolean }> = {
    QUEUED: {
      label: "Queued",
      className: "bg-[var(--surface-overlay)] text-[var(--text-secondary)]",
      live: true,
    },
    STARTING: {
      label: "Starting",
      className: "bg-[var(--accent-muted)] text-[var(--accent)]",
      live: true,
    },
    GENERATING: {
      label: "Generating",
      className: "bg-[var(--accent-muted)] text-[var(--accent)]",
      live: true,
    },
    POST_PROCESSING: {
      label: "Finishing",
      className: "bg-[var(--accent-muted)] text-[var(--accent)]",
      live: true,
    },
    UPLOADING: {
      label: "Saving",
      className: "bg-[var(--accent-muted)] text-[var(--accent)]",
      live: true,
    },
    COMPLETED: { label: "Ready", className: "bg-[var(--brand-muted)] text-[var(--brand-text)]" },
    FAILED: { label: "Failed", className: "bg-[var(--danger-muted)] text-[var(--danger)]" },
    CANCELLED: {
      label: "Cancelled",
      className: "bg-[var(--surface-overlay)] text-[var(--text-muted)]",
    },
  };
  const entry = map[status] ?? {
    label: status,
    className: "bg-[var(--surface-overlay)] text-[var(--text-secondary)]",
  };
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-[var(--radius-full)] px-2.5 py-1",
        "text-[11px] font-medium",
        entry.className,
      )}
    >
      {entry.live && (
        <span aria-hidden="true" className="luber-pulse h-1.5 w-1.5 rounded-full bg-current" />
      )}
      {entry.label}
    </span>
  );
}

export { cx };
