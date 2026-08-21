"use client";

/**
 * The inference console's chrome.
 *
 * Its own shell rather than a tab inside the training console, for the
 * same reason the training console is not a tab inside the product:
 * they answer different questions for different people. Training spends
 * money on rented hardware; this one spends nothing and changes nothing.
 * Putting them behind one nav would invite an operator investigating a
 * retry spike to land one click away from a button that starts a GPU.
 *
 * The banner says what this console can do, which is nothing. An
 * operator who has just read a CRITICAL incident should know, without
 * hunting, that nothing was done about it automatically.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { cx } from "@/components/ui";

const NAV = [
  { href: "/ops/inference", label: "Health", exact: true },
  { href: "/ops/inference/incidents", label: "Incidents" },
  { href: "/ops/inference/providers", label: "Providers" },
  { href: "/ops/inference/generations", label: "Generations" },
];

export function InferenceShell({ children }: { children: ReactNode }) {
  const pathname = usePathname() ?? "";

  return (
    <div className="min-h-screen bg-[var(--surface-base)]">
      <div className="border-b border-[var(--accent)]/30 bg-[var(--accent-muted)] px-4 py-1.5 text-center text-[11px] text-[var(--accent)]">
        Operator console — inference observability. This console detects and explains. It
        disables nothing, changes no threshold and starts nothing.
      </div>

      <header className="border-b border-[var(--border-subtle)] bg-[var(--surface-sunken)]">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
          <Link href="/ops/inference" className="flex items-center gap-2.5">
            <span
              aria-hidden="true"
              className="flex h-6 w-6 items-center justify-center rounded-[var(--radius-sm)] bg-[var(--brand)] text-[11px] font-black text-white"
            >
              L
            </span>
            <span className="text-sm font-bold tracking-tight text-[var(--text-primary)]">
              Inference console
            </span>
          </Link>

          <nav aria-label="Inference sections" className="flex flex-wrap gap-1">
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

          <Link
            href="/ops/training"
            className="ml-auto text-[11px] text-[var(--text-muted)] hover:text-[var(--brand-text)]"
          >
            Training console →
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-[1500px] px-4 py-6 sm:px-6">{children}</main>
    </div>
  );
}
