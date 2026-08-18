"use client";

import { Suspense } from "react";

import { AuthForm, AuthLink } from "@/components/auth/AuthForm";

/**
 * Signup issues a session, so a new account lands in the product
 * immediately rather than being sent to sign in again with the password
 * it just chose.
 */
export default function SignupPage() {
  return (
    <Suspense fallback={null}>
      <AuthForm
        mode="signup"
        title="Create your account"
        subtitle="Start making music in a private workspace."
        submitLabel="Create account"
        busyLabel="Creating account…"
        footer={
          <>
            Already have an account? <AuthLink href="/login">Sign in</AuthLink>
          </>
        }
      />
    </Suspense>
  );
}
