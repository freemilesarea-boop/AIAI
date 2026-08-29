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

import { AccountMenu } from "@/components/auth/AccountMenu";
import { CompactFooter, Footer } from "@/components/Footer";
import { RequireAuth } from "@/components/auth/RequireAuth";
import { useEffect, useState, type ReactNode } from "react";

import { PlayerBar } from "@/components/player/PlayerBar";
import { usePlayer } from "@/components/player/PlayerProvider";
import { cx } from "@/components/ui";
import { useEntitlement } from "@/components/EntitlementProvider";
import { formatSongs, isExhausted } from "@/lib/plans";

interface NavItem {
  href: string;
  label: string;
  icon: ReactNode;
  /** Small marker beside the label. LAB carries one; nothing else does. */
  badge?: string;
}

const icon = (path: string) => (
  <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" fill="currentColor" aria-hidden="true">
    <path d={path} />
  </svg>
);

const NAV: NavItem[] = [
  {
    href: "/",
    label: "Home",
    icon: icon("M12 3.2 3 10.1V21h6v-6h6v6h6V10.1L12 3.2Z"),
  },
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
    href: "/lab",
    label: "LAB",
    icon: icon("M9 2v6.2L3.6 17.4A3 3 0 0 0 6.2 22h11.6a3 3 0 0 0 2.6-4.6L15 8.2V2H9Zm2 2h2v4.8l1.6 2.7H9.4L11 8.8V4Z"),
    badge: "BETA",
  },
  {
    href: "/plans",
    label: "Plans",
    icon: icon("M4 6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v2H4V6Zm0 4h16v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-8Zm3 5h5v2H7v-2Z"),
  },
  {
    href: "/settings",
    label: "Settings",
    icon: icon("M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm9.4 4a7.6 7.6 0 0 0-.13-1.4l2-1.55-2-3.46-2.35.95a7.5 7.5 0 0 0-2.42-1.4L16.1 2h-4l-.4 2.54a7.5 7.5 0 0 0-2.42 1.4L6.93 5 4.93 8.45l2 1.55a7.7 7.7 0 0 0 0 2.8l-2 1.55 2 3.46 2.35-.95a7.5 7.5 0 0 0 2.42 1.4L12.1 22h4l.4-2.54a7.5 7.5 0 0 0 2.42-1.4l2.35.95 2-3.46-2-1.55c.09-.46.13-.93.13-1.4Z"),
  },
];

function Brand() {
  return (
    <Link
      href="/"
      className="flex items-center gap-2.5 rounded-[var(--radius-md)] px-2 py-1.5"
      aria-label="BOORDA 홈"
    >
      <span
        aria-hidden="true"
        className="flex h-7 w-7 items-center justify-center rounded-[var(--radius-sm)] bg-[var(--brand)] text-sm font-black text-white"
      >
        B
      </span>
      <span className="text-[15px] font-bold tracking-tight text-[var(--text-primary)]">
        BOORDA
      </span>
    </Link>
  );
}

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <nav aria-label="Main" className="flex flex-col gap-1">
      {NAV.map((item) => {
        // `/` is a prefix of everything, so Home matches exactly and
        // never lights up while the user is somewhere else.
        const active =
          item.href === "/"
            ? pathname === "/"
            : pathname === item.href || pathname.startsWith(`${item.href}/`);
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
            <span className="flex-1">{item.label}</span>
            {item.badge ? (
              <span className="rounded-[var(--radius-full)] bg-[var(--surface-overlay)] px-1.5 py-0.5 text-[9px] font-semibold tracking-wide text-[var(--text-muted)]">
                {item.badge}
              </span>
            ) : null}
          </Link>
        );
      })}
    </nav>
  );
}

//: Pages that exist precisely so a guest can reach them. Everything
//: else in the product is private.
const PUBLIC_ROUTES = new Set(["/login", "/signup"]);

//: The operator training console. It is not part of the product: it has
//: its own navigation, its own vocabulary and its own audience, and it
//: must not appear in a customer's sidebar or sit behind a customer's
//: session. Rendered bare here so `/ops` brings its own shell.
const OPERATOR_PREFIX = "/ops";

//: Legal notices, readable by anyone.
//:
//: Not behind `RequireAuth`: a privacy policy a visitor must create an
//: account to read is not published, and the people most likely to need
//: these pages — someone deciding whether to sign up, or a regulator —
//: have no account by definition.
const LEGAL_ROUTES = new Set(["/privacy", "/terms", "/refund-policy"]);

//: The operator console. It renders inside the product shell — it needs
//: the sidebar and the session — but it is not the consumer site, and a
//: marketing footer with business registration details has no place in a
//: back office. Excluded explicitly rather than by omission, so the next
//: person to touch the shell can see the decision.
const ADMIN_PREFIX = "/admin";


/**
 * Plan and songs left, above the account menu.
 *
 * The one number that changes as the user works, kept where they can see
 * it without opening a page. Renders nothing at all when the entitlement
 * has not loaded or failed to: a sidebar slot that says "—" every time
 * the network hiccups trains people to ignore it.
 */
function PlanSummary() {
  const { entitlement } = useEntitlement();
  if (!entitlement) return null;

  const spent = isExhausted(entitlement);
  return (
    <Link
      href="/plans"
      className="mx-1 mb-2 flex flex-col gap-1 rounded-[var(--radius-md)] border border-[var(--border-subtle)] px-3 py-2.5 hover:bg-[var(--surface-raised)]"
    >
      <span className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-[var(--text-primary)]">
          {entitlement.plan.display_name}
        </span>
        <span className="text-[11px] text-[var(--brand-text)]">플랜</span>
      </span>
      <span
        className={cx(
          "text-[11px]",
          spent ? "text-[var(--danger)]" : "text-[var(--text-muted)]",
        )}
      >
        {spent
          ? "이번 달 한도 소진"
          : `이번 달 ${formatSongs(entitlement.generation_remaining)} 남음`}
      </span>
    </Link>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const pathname = usePathname() ?? "/";
  const { track } = usePlayer();

  // Close the slide-over on navigation, so a tap never leaves it hanging.
  useEffect(() => setMenuOpen(false), [pathname]);

  // Auth pages get no sidebar, no player and no navigation: a signed-out
  // visitor should not be looking at the furniture of a product they
  // cannot use. Placed after every hook — an early return above one
  // makes it conditional, which breaks the rules of hooks.
  // The operator console renders with no product chrome at all: no
  // sidebar, no player, and — importantly — no RequireAuth, because it
  // is gated by the deployment rather than by a customer session.
  if (pathname === OPERATOR_PREFIX || pathname.startsWith(`${OPERATOR_PREFIX}/`)) {
    return <>{children}</>;
  }

  const isAdmin = pathname === ADMIN_PREFIX || pathname.startsWith(`${ADMIN_PREFIX}/`);

  if (LEGAL_ROUTES.has(pathname)) {
    return (
      <div className="flex min-h-screen flex-col bg-[var(--surface-base)]">
        <main className="flex-1 px-5 py-10 sm:px-8">{children}</main>
        <Footer />
      </div>
    );
  }

  if (PUBLIC_ROUTES.has(pathname)) {
    return (
      <div className="min-h-screen bg-[var(--surface-base)]">
        <main className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-4 py-10">
          <div className="w-full flex-1 content-center">{children}</div>
          <CompactFooter />
        </main>
      </div>
    );
  }

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
        <div className="mt-auto">
          <PlanSummary />
          <AccountMenu />
          <p className="px-3 pt-3 text-[11px] leading-relaxed text-[var(--text-muted)]">
            Generated music. Quality varies by prompt.
          </p>
        </div>
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
            <div className="mt-6">
              <PlanSummary />
              <AccountMenu />
            </div>
          </div>
        </div>
      )}

      <main
        className="flex min-h-screen flex-col lg:pl-[var(--sidebar-width)]"
        style={{ paddingBottom: track ? "calc(var(--player-height) + 16px)" : undefined }}
      >
        {/* `flex-1` pushes the footer to the bottom on a short page
            without positioning it there: on a long page it simply
            follows the content, and it never floats over the player. */}
        <div className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          <RequireAuth pathname={pathname}>{children}</RequireAuth>
        </div>
        {/* Outside RequireAuth on purpose: the terms must be reachable
            while a session is still resolving, and by someone who is
            being redirected to sign in. */}
        {isAdmin ? null : <Footer />}
      </main>

      <PlayerBar />
    </div>
  );
}
