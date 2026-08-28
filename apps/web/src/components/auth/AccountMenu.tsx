"use client";

/**
 * Who you are, and the way out.
 *
 * Deliberately small and at the bottom of the navigation. Account
 * controls are used rarely and the sidebar's job is to get someone to
 * their music; a prominent identity block would take space from the
 * thing the product is for.
 */

import Link from "next/link";
import { useState } from "react";

import { useAuth } from "@/components/auth/AuthProvider";
import { isAdmin } from "@/lib/admin";

export function AccountMenu() {
  const { status, user, signOut } = useAuth();
  const [leaving, setLeaving] = useState(false);

  // Nothing to show while the session is still being resolved: a
  // flicker of "Sign in" for an already-signed-in user reads as a bug.
  if (status !== "authenticated" || user === null) return null;

  const label = user.display_name?.trim() || user.email;

  return (
    <div className="border-t border-[var(--border-subtle)] px-3 pt-3">
      <p className="truncate text-[11px] text-[var(--text-muted)]" title={user.email}>
        Signed in as
      </p>
      <p className="truncate text-xs font-medium text-[var(--text-secondary)]" title={user.email}>
        {label}
      </p>
      {/* Shown to operators only, and shown is all it is: the console's
          endpoints check the session's own row, so a browser that draws
          this link without the role reaches nothing behind it. */}
      {isAdmin(user.role) ? (
        <Link
          href="/admin"
          className="mt-2 block rounded-[var(--radius-sm)] px-2 py-1.5 text-xs text-[var(--text-secondary)] hover:bg-[var(--surface-raised)]"
        >
          운영 관리
        </Link>
      ) : null}
      <button
        type="button"
        onClick={() => {
          setLeaving(true);
          // signOut navigates; no need to reset the flag on success.
          void signOut().catch(() => setLeaving(false));
        }}
        disabled={leaving}
        className="mt-2 w-full rounded-[var(--radius-sm)] px-2 py-1.5 text-left text-xs text-[var(--text-secondary)] hover:bg-[var(--surface-raised)] disabled:opacity-60"
      >
        {leaving ? "Signing out…" : "Sign out"}
      </button>
    </div>
  );
}
