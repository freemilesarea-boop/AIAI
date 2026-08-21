/**
 * Where the console stops existing.
 *
 * A server component, so the check happens before any operator markup
 * is sent. `notFound()` renders the application's 404 — the same page a
 * mistyped URL produces — because a deployment that has not enabled the
 * console should look like one that has never heard of it.
 *
 * This is the browser-side half of a boundary the API enforces
 * independently: even if this page rendered, every request it made would
 * be refused by a backend that is not serving the operator router. Two
 * checks, because the cost of one of them being wrong is a public
 * training console.
 */

import type { Metadata } from "next";
import { notFound } from "next/navigation";
import type { ReactNode } from "react";

import { OpsShell } from "@/components/ops/OpsShell";

export const metadata: Metadata = {
  title: "LUBER — training console",
  // Internal, and not something to find in a search result.
  robots: { index: false, follow: false },
};

/** Nothing here is cacheable: it is a view of state that is changing. */
export const dynamic = "force-dynamic";

export default function OpsTrainingLayout({ children }: { children: ReactNode }) {
  if (process.env.OPS_CONSOLE_ENABLED !== "true") notFound();
  return <OpsShell>{children}</OpsShell>;
}
