/**
 * Telling BOORDA where a visit came from.
 *
 * One call, once per browser tab session, carrying only what the page
 * already knows: its own path, its referrer, and the campaign
 * parameters in its own URL. No identifier is generated here — the
 * server mints the visitor id and keeps it in a cookie this code cannot
 * read.
 *
 * Everything is best-effort. A blocked request, a rejected cookie, a
 * network failure: all of them end with the visitor reading the page
 * normally and us knowing nothing, which is the correct trade. Analytics
 * that can break the product it measures is not worth having.
 */

import { API_BASE_URL } from "@/lib/api";

/** Parameters worth forwarding. Mirrors the server's allowlist. */
export const TRACKED_PARAMS = [
  "utm_source",
  "utm_medium",
  "utm_campaign",
  "utm_content",
  "utm_term",
  "gclid",
  "gbraid",
  "wbraid",
  "fbclid",
] as const;

/**
 * Paths that are not acquisition.
 *
 * The console and the operator tools are navigation by people already
 * here; counting an administrator opening a dashboard as a visit would
 * put our own traffic in the marketing report.
 */
const IGNORED_PREFIXES = ["/admin", "/ops"];

/** Marks this tab as counted, so a client-side route change is not a new visit. */
const SESSION_FLAG = "boorda.acquisitionSent";

export function shouldTrack(pathname: string): boolean {
  return !IGNORED_PREFIXES.some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`),
  );
}

/** The campaign parameters present in a query string, and nothing else. */
export function trackedParams(search: string): Record<string, string> {
  const params = new URLSearchParams(search);
  const kept: Record<string, string> = {};
  for (const name of TRACKED_PARAMS) {
    const value = params.get(name);
    if (value) kept[name] = value.slice(0, 200);
  }
  return kept;
}

/**
 * Whether this tab has already reported a visit.
 *
 * `sessionStorage` rather than a cookie: it is per-tab and disappears
 * when the tab does, which is exactly the lifetime of "this visit". A
 * browser that refuses it simply reports again — a duplicate session
 * row is a much smaller problem than a crash.
 */
function alreadySent(): boolean {
  try {
    if (typeof window === "undefined" || !window.sessionStorage) return false;
    return window.sessionStorage.getItem(SESSION_FLAG) === "1";
  } catch {
    return false;
  }
}

function markSent(): void {
  try {
    window.sessionStorage?.setItem(SESSION_FLAG, "1");
  } catch {
    // A browser refusing session storage still gets to read the page.
  }
}

export async function reportVisit(
  location: { pathname: string; search: string },
  referrer: string,
): Promise<void> {
  if (!shouldTrack(location.pathname) || alreadySent()) return;
  markSent();

  try {
    await fetch(`${API_BASE_URL}/v1/acquisition/visit`, {
      method: "POST",
      // The cookie is the whole point: without it the server cannot
      // recognise this browser next time.
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path: location.pathname,
        referrer: referrer || null,
        params: trackedParams(location.search),
      }),
      keepalive: true,
    });
  } catch {
    // Silent by design. See the module docstring.
  }
}
