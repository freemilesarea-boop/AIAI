"use client";

/**
 * The one place the app knows who is signed in.
 *
 * State is derived from the server on every load: the session lives in
 * an HttpOnly cookie the browser sends automatically and JavaScript
 * cannot read, so the only way to learn the current user is to ask.
 * Nothing about the session is kept in localStorage — there is no token
 * to keep, which is the point of the design rather than an omission.
 *
 * Three states, and the distinction between two of them matters:
 * `loading` is "we have not asked yet", `unauthenticated` is "we asked
 * and there is nobody". Collapsing them makes every page flash its
 * signed-out state on first paint.
 */

import { usePathname, useRouter } from "next/navigation";
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

import { clearPrivateGenerationCache } from "@/lib/generationStorage";
import { loginUrlFor } from "@/lib/redirect";
import { emitSessionEnded } from "@/lib/session-events";
import {
  type AuthUser,
  fetchCurrentUser,
  login as loginRequest,
  setSessionExpiredHandler,
  logout as logoutRequest,
  signup as signupRequest,
} from "@/lib/api";

export type AuthStatus = "loading" | "authenticated" | "unauthenticated";

export interface AuthContextValue {
  status: AuthStatus;
  user: AuthUser | null;
  signIn: (email: string, password: string) => Promise<AuthUser>;
  register: (email: string, password: string) => Promise<AuthUser>;
  signOut: () => Promise<void>;
  /**
   * Replace the cached user after the server returns an updated one —
   * a profile edit, for instance. Avoids a second `/me` round-trip and
   * keeps every consumer of `user` in step.
   */
  adopt: (user: AuthUser) => void;
  /** Called when a product request 401s: the session ended mid-session. */
  sessionExpired: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext);
  if (value === null) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return value;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [user, setUser] = useState<AuthUser | null>(null);
  const router = useRouter();
  const pathname = usePathname() ?? "/";
  // Guards against a burst of 401s from parallel requests each trying
  // to tear the session down and navigate.
  const expiring = useRef(false);
  // Held in a ref so the handler registration does not need to be torn
  // down and rebuilt every time the callback identity changes.
  const sessionExpiredRef = useRef<() => void>(() => {});

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        const current = await fetchCurrentUser(controller.signal);
        setUser(current);
        setStatus(current ? "authenticated" : "unauthenticated");
      } catch {
        // A failed bootstrap is not proof of a guest, but it is the only
        // safe assumption: showing the product to someone we could not
        // identify is worse than making them sign in again.
        if (!controller.signal.aborted) {
          setUser(null);
          setStatus("unauthenticated");
        }
      }
    })();
    return () => controller.abort();
  }, []);

  // Registered once so any product request that 401s tears the session
  // down through the same path a manual sign-out uses.
  useEffect(() => {
    setSessionExpiredHandler(() => sessionExpiredRef.current());
    return () => setSessionExpiredHandler(null);
  }, []);

  const adopt = useCallback((next: AuthUser) => {
    expiring.current = false;
    setUser(next);
    setStatus("authenticated");
  }, []);

  const signIn = useCallback(
    async (email: string, password: string) => {
      const next = await loginRequest(email, password);
      adopt(next);
      return next;
    },
    [adopt],
  );

  const register = useCallback(
    async (email: string, password: string) => {
      // Signup issues a session, so the user is in the product without
      // a second round trip. Verified against the real backend rather
      // than assumed.
      const next = await signupRequest(email, password);
      adopt(next);
      return next;
    },
    [adopt],
  );

  const signOut = useCallback(async () => {
    // The server destroys the session first. Clearing only local state
    // would leave a live session any retained cookie could still use.
    // The local session is discarded whether or not the server call
    // succeeds. A failed request most often means the network is gone,
    // and leaving someone apparently signed in on a shared machine is
    // the worse outcome — the cookie is cleared by the server when it
    // can be reached, and the session expires on its own regardless.
    try {
      await logoutRequest();
    } catch {
      /* deliberate: local state is cleared either way */
    }
    // Private client cache and in-memory state go with the session. The
    // API already refuses to serve another user's songs, but a cached
    // title is readable by whoever signs in next on this browser.
    clearPrivateGenerationCache();
    emitSessionEnded();
    setUser(null);
    setStatus("unauthenticated");
    // replace, not push: Back must not return to a private page.
    router.replace("/login");
  }, [router]);

  const sessionExpired = useCallback(() => {
    if (expiring.current) return;
    expiring.current = true;
    clearPrivateGenerationCache();
    emitSessionEnded();
    setUser(null);
    setStatus("unauthenticated");
    // An expiry mid-session should not also lose the user's place.
    // loginUrlFor applies the same open-redirect rules as every other
    // destination and refuses to bounce back to an auth page.
    router.replace(loginUrlFor(pathname));
  }, [router, pathname]);

  sessionExpiredRef.current = sessionExpired;

  const value = useMemo(
    () => ({ status, user, signIn, register, signOut, adopt, sessionExpired }),
    [status, user, signIn, register, signOut, adopt, sessionExpired],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
