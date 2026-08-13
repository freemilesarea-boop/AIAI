"use client";

/**
 * The application shell: sidebar, workspace, persistent player.
 *
 * On desktop the sidebar is fixed and the workspace scrolls beside it.
 * Below `lg` the sidebar collapses into a slide-over and a bottom tab
 * bar takes over primary navigation, because a music tool is used
 * one-handed on a phone as often as at a desk.
 *
 * The workspace reserves space for the player only while a track is
 * loaded, so an empty player never eats 84px of a small screen.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import { PlayerBar } from "@/components/player/PlayerBar";
import { usePlayer } from "@/components/player/PlayerProvider";
import { cx } from "@/components/ui";

interface NavItem {
  href: string;
  label: string;
  icon: ReactNode;
}

const icon = (path: string) => (
  <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" fill="currentColor" aria-hidden="true">
    <path d={path} />
  </svg>
);

const NAV: NavItem[] = [
  {
    href: "/create",
    label: "Create",
    icon: icon("M12 4a1 1 0 0 1 1 1v6h6a1 1 0 1 1 0 2h-6v6a1 1 0 1 1-2 0v-6H5a1 1 0 1 1 0-2h6V5a1 1 0 0 1 1-1Z"),
  },
  {
    href: "/library",
    label: "Library",
    icon: icon("M12 3v10.55A4 4 0 1 0 14 17V7h4V3h-6Z"),
  },
  {
    href: "/projects",
    label: "Projects",
    icon: icon("M4 5a2 2 0 0 1 2-2h3.6l2 2H18a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V5Z"),
  },
];

function Brand() {
  return (
    <Link
      href="/create"
      className="flex items-center gap-2.5 rounded-[var(--radius-md)] px-2 py-1.5"
      aria-label="LUBER home"
    >
      <span
        aria-hidden="true"
        className="flex h-7 w-7 items-center justify-center rounded-[var(--radius-sm)] bg-[var(--brand)] text-sm font-black text-white"
      >
        L
      </span>
      <span className="text-[15px] font-bold tracking-tight text-[var(--text-primary)]">
        LUBER
      </span>
    </Link>
  );
}

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <nav aria-label="Main" className="flex flex-col gap-1">
      {NAV.map((item) => {
        const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
        return (
          <Link
            key={item.href}
            href={item.href}
            onClick={onNavigate}
            aria-current={active ? "page" : undefined}
            className={cx(
              "flex items-center gap-3 rounded-[var(--radius-md)] px-3 py-2 text-sm font-medium transition-colors",
              active
                ? "bg-[var(--surface-overlay)] text-[var(--text-primary)]"
                : "text-[var(--text-secondary)] hover:bg-[var(--surface-raised)] hover:text-[var(--text-primary)]",
            )}
          >
            {item.icon}
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const pathname = usePathname();
  const { track } = usePlayer();

  // Close the slide-over on navigation, so a tap never leaves it hanging.
  useEffect(() => setMenuOpen(false), [pathname]);

  return (
    <div className="min-h-screen bg-[var(--surface-base)]">
      {/* Desktop sidebar */}
      <aside
        className="fixed inset-y-0 left-0 z-30 hidden w-[var(--sidebar-width)] flex-col border-r border-[var(--border-subtle)] bg-[var(--surface-sunken)] px-3 py-4 lg:flex"
      >
        <Brand />
        <div className="mt-6">
          <NavLinks />
        </div>
        <p className="mt-auto px-3 text-[11px] leading-relaxed text-[var(--text-muted)]">
          Generated music. Quality varies by prompt.
        </p>
      </aside>

      {/* Mobile top bar */}
      <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-[var(--border-subtle)] bg-[var(--surface-base)]/95 px-4 py-3 backdrop-blur lg:hidden">
        <button
          type="button"
          onClick={() => setMenuOpen(true)}
          aria-label="Open menu"
          aria-expanded={menuOpen}
          className="rounded-[var(--radius-sm)] p-1.5 text-[var(--text-secondary)] hover:bg-[var(--surface-raised)]"
        >
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor" aria-hidden="true">
            <path d="M4 6h16v2H4V6Zm0 5h16v2H4v-2Zm0 5h16v2H4v-2Z" />
          </svg>
        </button>
        <Brand />
      </header>

      {menuOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button
            type="button"
            aria-label="Close menu"
            onClick={() => setMenuOpen(false)}
            className="absolute inset-0 bg-black/60"
          />
          <div className="absolute inset-y-0 left-0 w-64 border-r border-[var(--border-default)] bg-[var(--surface-sunken)] px-3 py-4">
            <Brand />
            <div className="mt-6">
              <NavLinks onNavigate={() => setMenuOpen(false)} />
            </div>
          </div>
        </div>
      )}

      <main
        className="lg:pl-[var(--sidebar-width)]"
        style={{ paddingBottom: track ? "calc(var(--player-height) + 16px)" : undefined }}
      >
        <div className="mx-auto max-w-[1400px] px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          {children}
        </div>
      </main>

      <PlayerBar />
    </div>
  );
}
