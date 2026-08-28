"use client";

/**
 * The console's layout: a session, then a role, then the page.
 *
 * `RequireAuth` handles the first and `AdminShell` the second. Neither
 * authorises anything — the API checks every request against the
 * session's own row — but rendering in that order keeps a customer who
 * lands here from firing a page of requests that all answer 403.
 */

import type { ReactNode } from "react";

import { AdminShell } from "@/components/admin/AdminShell";
import { RequireAuth } from "@/components/auth/RequireAuth";

export default function AdminLayout({ children }: { children: ReactNode }) {
  return (
    <RequireAuth pathname="/admin">
      <AdminShell>{children}</AdminShell>
    </RequireAuth>
  );
}
