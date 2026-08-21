"use client";

/**
 * The console's own chrome.
 *
 * Deliberately not the product shell. A training console next to a
 * "Create" button invites somebody to reach one from the other, and the
 * two have different audiences, different vocabularies and different
 * consequences for a misclick. Nothing here appears in the customer's
 * navigation, and nothing in the customer's navigation appears here.
 *
 * The banner is not decoration. An operator with several tabs open needs
 * to know at a glance which one can cancel a run that costs money.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { cx } from "@/components/ui";

const NAV = [
  { href: "/ops/training", label: "Overview", exact: true },
  { href: "/ops/training/experiments", label: "Experiments" },
  { href: "/ops/training/runs", label: "Runs" },
  { href: "/ops/training/workers", label: "Workers" },
  { href: "/ops/training/checkpoints", label: "Checkpoints" },
  { href: "/ops/training/evaluations", label: "Evaluations" },
];

export function OpsShell({ children }: { children: ReactNode }) {
  const pathname = usePathname() ?? "";

  return (
    <div className="min-h-screen bg-[var(--surface-base)]">
      <div className="border-b border-[var(--accent)]/30 bg-[var(--accent-muted)] px-4 py-1.5 text-center text-[11px] text-[var(--accent)]">
        Operator console — training internals and actions that spend money on rented hardware.
        Not part of the customer product.
      </div>

      <header className="border-b border-[var(--border-subtle)] bg-[var(--surface-sunken)]">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
          <Link href="/ops/training" className="flex items-center gap-2.5">
            <span
              aria-hidden="true"
              className="flex h-6 w-6 items-center justify-center rounded-[var(--radius-sm)] bg-[var(--brand)] text-[11px] font-black text-white"
            >
              L
            </span>
            <span className="text-sm font-bold tracking-tight text-[var(--text-primary)]">
              Training console
            </span>
          </Link>

          <nav aria-label="Operator sections" className="flex flex-wrap gap-1">
            {NAV.map((item) => {
              const active = item.exact
                ? pathname === item.href
                : pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={active ? "page" : undefined}
                  className={cx(
                    "rounded-[var(--radius-md)] px-3 py-1.5 text-xs font-medium transition-colors",
                    active
                      ? "bg-[var(--surface-overlay)] text-[var(--text-primary)]"
                      : "text-[var(--text-secondary)] hover:bg-[var(--surface-raised)] hover:text-[var(--text-primary)]",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-[1500px] px-4 py-6 sm:px-6">{children}</main>
    </div>
  );
}

/**
 * A page heading with a manual refresh.
 *
 * Step 61: an operator must never have to reload the browser to see
 * whether something changed. The button says when the data was last
 * read, because "refresh" without that is a button whose effect you
 * cannot observe.
 */
export function OpsHeader({
  title,
  description,
  breadcrumb,
  onRefresh,
  refreshing,
  children,
}: {
  title: string;
  description?: ReactNode;
  breadcrumb?: { href: string; label: string }[];
  onRefresh?: () => void;
  refreshing?: boolean;
  children?: ReactNode;
}) {
  return (
    <div className="mb-5">
      {breadcrumb && breadcrumb.length > 0 && (
        <nav aria-label="Breadcrumb" className="mb-2 flex flex-wrap gap-1 text-xs">
          {breadcrumb.map((crumb, index) => (
            <span key={crumb.href} className="flex items-center gap-1">
              {index > 0 && (
                <span aria-hidden="true" className="text-[var(--text-muted)]">
                  /
                </span>
              )}
              <Link href={crumb.href} className="text-[var(--text-muted)] hover:text-[var(--brand-text)]">
                {crumb.label}
              </Link>
            </span>
          ))}
        </nav>
      )}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-lg font-semibold break-words text-[var(--text-primary)]">{title}</h1>
          {description && (
            <div className="mt-1 text-xs leading-relaxed text-[var(--text-secondary)]">
              {description}
            </div>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {children}
          {onRefresh && (
            <button
              type="button"
              onClick={onRefresh}
              aria-busy={refreshing || undefined}
              className="rounded-[var(--radius-md)] border border-[var(--border-default)] px-3 py-1.5 text-xs text-[var(--text-secondary)] hover:bg-[var(--surface-raised)] hover:text-[var(--text-primary)]"
            >
              {refreshing ? "Refreshing…" : "Refresh"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
