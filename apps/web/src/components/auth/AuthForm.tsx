"use client";

/**
 * The shared shell for signing in and signing up.
 *
 * One component because the two forms differ in three strings and one
 * request. Two near-identical files would drift, and the half that
 * drifts is usually the one with the accessibility attributes.
 */

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState, type FormEvent, type ReactNode } from "react";

import { useAuth } from "@/components/auth/AuthProvider";
import { safeDestination } from "@/lib/redirect";

//: Mirrors the backend policy (luber_api.security). Kept in step with
//: it deliberately: a stricter frontend rule would reject passwords the
//: server accepts, and a looser one would promise what it cannot keep.
export const MIN_PASSWORD_LENGTH = 10;

export interface AuthFormProps {
  mode: "login" | "signup";
  title: string;
  subtitle: string;
  submitLabel: string;
  busyLabel: string;
  footer: ReactNode;
}

export function AuthForm({
  mode,
  title,
  subtitle,
  submitLabel,
  busyLabel,
  footer,
}: AuthFormProps) {
  const { status, signIn, register } = useAuth();
  const router = useRouter();
  const params = useSearchParams();
  const destination = safeDestination(params?.get("next"));

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Someone already signed in has no business on this page; send them
  // where they were going instead of showing a form they cannot use.
  useEffect(() => {
    if (status === "authenticated") router.replace(destination);
  }, [status, router, destination]);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);

    const trimmed = email.trim();
    if (!trimmed || !password) {
      setError("Enter your email and password.");
      return;
    }
    if (mode === "signup") {
      if (password.length < MIN_PASSWORD_LENGTH) {
        setError(`Password must be at least ${MIN_PASSWORD_LENGTH} characters.`);
        return;
      }
      if (password !== confirm) {
        setError("Those passwords do not match.");
        return;
      }
    }

    setBusy(true);
    try {
      await (mode === "login" ? signIn(trimmed, password) : register(trimmed, password));
      router.replace(destination);
    } catch (caught) {
      // The backend writes these for humans and carries no internals.
      setError(caught instanceof Error ? caught.message : "That did not work. Please try again.");
      setBusy(false);
    }
  }

  return (
    <div className="w-full">
      <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">{title}</h1>
      <p className="mt-1.5 text-sm text-[var(--text-secondary)]">{subtitle}</p>

      {/*
        method="post" is a safety net, not a feature. The submit handler
        calls preventDefault, so this method is never actually used — but
        a form with no method defaults to GET, and a submit that happens
        before React hydrates (or with JS broken) would then put the
        password in the URL, the browser history and every access log in
        between. One attribute removes that entire failure mode.
      */}
      <form
        onSubmit={onSubmit}
        method="post"
        className="mt-7 flex flex-col gap-4"
        noValidate
      >
        <div className="flex flex-col gap-1.5">
          <label htmlFor="email" className="text-xs font-medium text-[var(--text-secondary)]">
            Email
          </label>
          <input
            id="email"
            name="email"
            type="email"
            autoComplete="email"
            autoCapitalize="none"
            spellCheck={false}
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-sunken)] px-3 py-2.5 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--brand)] focus:ring-2 focus:ring-[var(--brand-muted)]"
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <label htmlFor="password" className="text-xs font-medium text-[var(--text-secondary)]">
            Password
          </label>
          <input
            id="password"
            name="password"
            type="password"
            // new-password on signup so a manager offers to generate one;
            // current-password on login so it offers the saved one.
            autoComplete={mode === "signup" ? "new-password" : "current-password"}
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            aria-describedby={mode === "signup" ? "password-hint" : undefined}
            className="rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-sunken)] px-3 py-2.5 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--brand)] focus:ring-2 focus:ring-[var(--brand-muted)]"
          />
          {mode === "signup" && (
            <p id="password-hint" className="text-[11px] text-[var(--text-muted)]">
              At least {MIN_PASSWORD_LENGTH} characters. A phrase you can remember beats a
              short jumble.
            </p>
          )}
        </div>

        {mode === "signup" && (
          <div className="flex flex-col gap-1.5">
            <label htmlFor="confirm" className="text-xs font-medium text-[var(--text-secondary)]">
              Confirm password
            </label>
            <input
              id="confirm"
              name="confirm"
              type="password"
              autoComplete="new-password"
              required
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              className="rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-sunken)] px-3 py-2.5 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--brand)] focus:ring-2 focus:ring-[var(--brand-muted)]"
            />
          </div>
        )}

        {error && (
          // Tied to the form so a screen reader hears it, and announced
          // rather than only appearing.
          <p role="alert" className="text-sm text-[var(--danger)]">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy}
          aria-busy={busy}
          className="mt-1 rounded-[var(--radius-md)] bg-[var(--brand)] px-4 py-2.5 text-sm font-medium text-white hover:opacity-95 disabled:opacity-60"
        >
          {busy ? busyLabel : submitLabel}
        </button>
      </form>

      <p className="mt-6 text-sm text-[var(--text-secondary)]">{footer}</p>
    </div>
  );
}

export function AuthLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <Link href={href} className="font-medium text-[var(--brand-text)] hover:underline">
      {children}
    </Link>
  );
}
