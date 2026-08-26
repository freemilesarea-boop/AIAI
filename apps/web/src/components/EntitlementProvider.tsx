"use client";

/**
 * One entitlement read, shared by every screen that shows it.
 *
 * The sidebar, Create, Home, Library and Settings all need to know the
 * same three things: which plan, how many songs are left, and whether
 * downloads are included. Fetching that five times would be five
 * requests and — worse — five answers that can disagree while a
 * generation is in flight.
 *
 * So it is fetched once here and refreshed deliberately: after a
 * generation is submitted, and when the tab is brought back to the
 * front. `refresh()` is exposed for the first case.
 *
 * This is a display cache and nothing else. Every rule it describes is
 * enforced by the server on the request that matters, and a stale value
 * here can only make the UI briefly wrong, never make a refusal into a
 * permission.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { fetchEntitlement, type Entitlement } from "@/lib/plans";

interface EntitlementState {
  entitlement: Entitlement | null;
  /** True until the first answer arrives, so callers can avoid flashing. */
  loading: boolean;
  /**
   * Set when the read failed. The UI degrades to showing nothing rather
   * than to showing a guess — an invented allowance is worse than an
   * absent one.
   */
  error: boolean;
  refresh: () => void;
}

const EntitlementContext = createContext<EntitlementState>({
  entitlement: null,
  loading: false,
  error: false,
  refresh: () => {},
});

export function EntitlementProvider({ children }: { children: ReactNode }) {
  const [entitlement, setEntitlement] = useState<Entitlement | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [nonce, setNonce] = useState(0);
  // Survives re-render so a slow response cannot overwrite a newer one.
  const latest = useRef(0);

  const refresh = useCallback(() => setNonce((value) => value + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    const ticket = latest.current + 1;
    latest.current = ticket;

    void (async () => {
      try {
        const next = await fetchEntitlement(controller.signal);
        if (latest.current !== ticket) return;
        setEntitlement(next);
        setError(false);
      } catch {
        // Includes 401 on a signed-out visitor, which is not an error
        // worth surfacing: the shell is already sending them to login.
        if (latest.current !== ticket) return;
        setError(true);
      } finally {
        if (latest.current === ticket) setLoading(false);
      }
    })();

    return () => controller.abort();
  }, [nonce]);

  // Coming back to a tab left open overnight should not show yesterday's
  // count — the period may have rolled while it sat there.
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === "visible") refresh();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, [refresh]);

  const value = useMemo(
    () => ({ entitlement, loading, error, refresh }),
    [entitlement, loading, error, refresh],
  );

  return <EntitlementContext.Provider value={value}>{children}</EntitlementContext.Provider>;
}

export function useEntitlement(): EntitlementState {
  return useContext(EntitlementContext);
}
