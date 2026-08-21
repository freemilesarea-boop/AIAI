/**
 * Where the inference console stops existing.
 *
 * A server component, so the check happens before any operator markup
 * is sent. `notFound()` renders the application's ordinary 404, because
 * a deployment that has not enabled the console should look like one
 * that has never heard of it.
 *
 * The browser-side half of a boundary the API enforces independently:
 * even if this page rendered, every request it made would be refused by
 * a backend not serving the operator router, and the proxy that would
 * carry those requests 404s under the same condition. Three checks,
 * because the cost of one being wrong is an operator console on the
 * public internet.
 */

import type { Metadata } from "next";
import { notFound } from "next/navigation";
import type { ReactNode } from "react";

import { InferenceShell } from "@/components/ops/InferenceShell";

export const metadata: Metadata = {
  title: "LUBER — inference console",
  robots: { index: false, follow: false },
};

/** A view of state that is changing; nothing here is cacheable. */
export const dynamic = "force-dynamic";

export default function OpsInferenceLayout({ children }: { children: ReactNode }) {
  if (process.env.OPS_CONSOLE_ENABLED !== "true") notFound();
  return <InferenceShell>{children}</InferenceShell>;
}
