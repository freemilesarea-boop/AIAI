"use client";

import { Suspense } from "react";

import { AuthForm, AuthLink } from "@/components/auth/AuthForm";

/**
 * No "forgot password" link. Password reset does not exist yet, and a
 * dead link that leads nowhere is worse than its absence — it promises
 * a way back in that the product cannot honour.
 */
export default function LoginPage() {
  return (
    <Suspense fallback={null}>
      <AuthForm
        mode="login"
        title="Sign in"
        subtitle="Your songs, your library, your projects."
        submitLabel="Sign in"
        busyLabel="Signing in…"
        footer={
          <>
            New here? <AuthLink href="/signup">Create an account</AuthLink>
          </>
        }
      />
    </Suspense>
  );
}
