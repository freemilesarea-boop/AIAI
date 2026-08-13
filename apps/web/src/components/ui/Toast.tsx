"use client";

/**
 * Transient action feedback.
 *
 * Every mutating action in the product confirms itself here — renamed,
 * favourited, added to a project, deleted, download started. The
 * alternative people reach for is `alert()`, which blocks the page,
 * cannot be styled, and (in this codebase specifically) freezes browser
 * automation.
 *
 * Deliberately not a notification centre: a toast says one short thing
 * and leaves. Anything that needs a decision belongs in a
 * `ConfirmDialog`, and anything that needs to persist belongs on the
 * page itself.
 *
 * The region is `aria-live="polite"`, so a screen reader hears the same
 * confirmation a sighted user sees rather than nothing at all.
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

import { cx } from "@/components/ui";

export type ToastTone = "success" | "error";

export interface Toast {
  id: number;
  message: string;
  tone: ToastTone;
}

export interface ToastApi {
  /** Confirm that something happened. */
  notify: (message: string) => void;
  /** Report that something did not. Errors linger longer. */
  notifyError: (message: string) => void;
  toasts: Toast[];
  dismiss: (id: number) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

export const TOAST_DURATION_MS = 3200;
/** Failures stay longer: a user who missed one has lost information. */
export const TOAST_ERROR_DURATION_MS = 6000;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);
  const timers = useRef(new Map<number, number>());

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
    const timer = timers.current.get(id);
    if (timer !== undefined) {
      window.clearTimeout(timer);
      timers.current.delete(id);
    }
  }, []);

  const push = useCallback(
    (message: string, tone: ToastTone) => {
      const id = nextId.current++;
      setToasts((current) => [...current, { id, message, tone }]);
      const timer = window.setTimeout(
        () => dismiss(id),
        tone === "error" ? TOAST_ERROR_DURATION_MS : TOAST_DURATION_MS,
      );
      timers.current.set(id, timer);
    },
    [dismiss],
  );

  // Clearing timers on unmount keeps a test environment (and a fast
  // navigation) from calling setState on a gone component.
  const pending = timers.current;
  useEffect(() => () => pending.forEach((timer) => window.clearTimeout(timer)), [pending]);

  const value = useMemo<ToastApi>(
    () => ({
      notify: (message: string) => push(message, "success"),
      notifyError: (message: string) => push(message, "error"),
      toasts,
      dismiss,
    }),
    [push, toasts, dismiss],
  );

  return (
    <ToastContext.Provider value={value}>
      {children}
      {/* `aria-live` without `role="status"`: the role would add a
          second, permanent status region to every page, which makes an
          in-flight generation's own status region ambiguous to both
          assistive tech and tests. Announcements are identical. */}
      <div
        aria-live="polite"
        className="pointer-events-none fixed inset-x-0 bottom-0 z-[60] flex flex-col items-center gap-2 px-4 pb-4 sm:items-end sm:px-6"
        style={{ paddingBottom: "calc(var(--player-height) + 16px)" }}
      >
        {toasts.map((toast) => (
          <button
            key={toast.id}
            type="button"
            onClick={() => dismiss(toast.id)}
            className={cx(
              "pointer-events-auto w-full max-w-sm rounded-[var(--radius-md)] border px-4 py-3",
              "text-left text-sm shadow-[var(--shadow-card)] backdrop-blur transition-colors",
              toast.tone === "error"
                ? "border-[var(--danger)]/50 bg-[var(--danger-muted)] text-[var(--danger)]"
                : "border-[var(--border-default)] bg-[var(--surface-overlay)] text-[var(--text-primary)]",
            )}
          >
            {toast.message}
          </button>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

/**
 * Feedback for the current surface.
 *
 * Falls back to a no-op outside a provider so an isolated component test
 * does not have to build the whole app shell to render a button.
 */
export function useToast(): ToastApi {
  const context = useContext(ToastContext);
  return context ?? NOOP_TOASTS;
}

const NOOP_TOASTS: ToastApi = {
  notify: () => {},
  notifyError: () => {},
  toasts: [],
  dismiss: () => {},
};
