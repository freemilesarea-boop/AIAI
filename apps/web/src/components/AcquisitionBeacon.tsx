"use client";

/**
 * Reports one visit, once, and renders nothing.
 *
 * Mounted above the router so it sees the landing URL before any
 * client-side navigation rewrites it — the campaign parameters that
 * brought somebody here are in the *first* URL, and a component mounted
 * inside a page would often miss them.
 */

import { useEffect } from "react";

import { reportVisit } from "@/lib/acquisition";

export function AcquisitionBeacon() {
  useEffect(() => {
    void reportVisit(
      { pathname: window.location.pathname, search: window.location.search },
      document.referrer,
    );
    // Deliberately once per mount. A route change inside the app is the
    // same visit, and the per-tab flag in `reportVisit` enforces that
    // even across React's development double-effect.
  }, []);

  return null;
}
