"use client";

/**
 * Fetching, refreshing, and knowing when to stop.
 *
 * A training console is a page somebody leaves open for eight hours, so
 * the polling rules matter more than the fetching does. Four of them are
 * enforced here rather than left to each page:
 *
 * **Terminal states stop the poll.** A COMPLETED run does not change
 * again. Continuing to ask is a request every few seconds, for hours,
 * about a fact that is settled.
 *
 * **A hidden tab stops the poll.** A background tab that keeps polling
 * is invisible load on the operator's own machine and on the API, and
 * the moment the tab becomes visible again it refreshes, so nothing is
 * stale when anybody is actually looking.
 *
 * **One request at a time.** A slow response must not let a second
 * interval fire on top of it; that turns a struggling API into a
 * struggling API with a queue.
 *
 * **Unmounting cancels.** Navigating away mid-request must not set state
 * on a component that is gone, and must not leave an interval running.
 *
 * Errors do not clear the previous data. A poll that fails once should
 * show the last good reading with a warning, not blank the panel an
 * operator was reading.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { OpsError } from "@/lib/ops/client";

export interface OpsResource<T> {
  data: T | null;
  error: string | null;
  /** True only for the first load, so a refresh does not blank the page. */
  loading: boolean;
  refreshing: boolean;
  refresh: () => void;
  setData: (value: T) => void;
}

export function useOpsResource<T>(
  load: () => Promise<T>,
  {
    deps = [],
    intervalMs = 0,
    enabled = true,
  }: { deps?: unknown[]; intervalMs?: number; enabled?: boolean } = {},
): OpsResource<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadRef = useRef(load);
  loadRef.current = load;
  const inFlight = useRef(false);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const run = useCallback(async (initial: boolean) => {
    if (inFlight.current) return;
    inFlight.current = true;
    if (!initial) setRefreshing(true);
    try {
      const result = await loadRef.current();
      if (!mounted.current) return;
      setData(result);
      setError(null);
    } catch (caught) {
      if (!mounted.current) return;
      // The previous reading stays on screen. A transient failure
      // should not erase what an operator was looking at.
      setError(
        caught instanceof OpsError ? caught.message : "The console could not reach the server.",
      );
    } finally {
      inFlight.current = false;
      if (mounted.current) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, []);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    setLoading(true);
    void run(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, ...deps]);

  useEffect(() => {
    if (!enabled || intervalMs <= 0) return;

    let timer: number | undefined;
    const start = () => {
      if (timer !== undefined) return;
      timer = window.setInterval(() => void run(false), intervalMs);
    };
    const stop = () => {
      if (timer === undefined) return;
      window.clearInterval(timer);
      timer = undefined;
    };
    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        void run(false);
        start();
      } else {
        stop();
      }
    };

    if (document.visibilityState === "visible") start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, intervalMs, ...deps]);

  const refresh = useCallback(() => void run(false), [run]);

  return { data, error, loading, refreshing, refresh, setData };
}

/** Run states that will never change again, so polling can stop. */
const TERMINAL = new Set(["COMPLETED", "FAILED", "CANCELLED"]);

/**
 * How often to poll a run, given what it is doing.
 *
 * LOST keeps a slow poll rather than none: a worker that comes back
 * changes the picture, and an operator watching a lost run wants to see
 * that happen without reloading. Terminal states poll not at all.
 */
export function runPollInterval(status: string | undefined): number {
  if (!status) return 0;
  if (TERMINAL.has(status)) return 0;
  if (status === "LOST") return 15_000;
  if (status === "RUNNING" || status === "STARTING") return 5_000;
  return 10_000;
}
