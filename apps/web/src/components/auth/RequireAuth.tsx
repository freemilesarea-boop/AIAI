"use client";

/**
 * The gate every product page sits behind.
 *
 * While the session is still being resolved it renders a placeholder,
 * **not** the page. That ordering is the point: rendering children
 * first would fire their private data requests on behalf of a guest,
 * producing a burst of 401s and, briefly, a page shaped like somebody's
 * library. The user sees a moment of nothing instead of a moment of
 * someone else's product.
 *
 * This is UX and request hygiene, not authorization. The real boundary
 * is the API, which refuses anonymous callers regardless of what the
 * browser chooses to render.
 */

import { useRouter } from "next/navigation";
import { useEffect, type ReactNode } from "react";

import { useAuth } from "@/components/auth/AuthProvider";
import { loginUrlFor } from "@/lib/redirect";

export function RequireAuth({
  children,
  pathname,
}: {
  children: ReactNode;
  pathname: string;
}) {
  const { status } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (status === "unauthenticated") {
      // replace so Back does not bounce between the gate and login.
      router.replace(loginUrlFor(pathname));
    }
  }, [status, router, pathname]);

  if (status === "authenticated") return <>{children}</>;

  return (
    <div
      role="status"
      aria-live="polite"
      aria-label={status === "loading" ? "Checking your session" : "Redirecting to sign in"}
      className="flex min-h-[50vh] items-center justify-center"
    >
      <span className="text-sm text-[var(--text-muted)]">
        {status === "loading" ? "Loading…" : "Redirecting to sign in…"}
      </span>
    </div>
  );
}
