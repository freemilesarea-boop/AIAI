"use client";

/**
 * The product's confirmation surface for destructive actions.
 *
 * `window.confirm` is not used anywhere in BOORDA: it cannot say *how
 * many* songs are about to go, cannot mark the destructive choice as
 * destructive, cannot be styled, and blocks the page (and browser
 * automation) while it is open.
 *
 * Keyboard behaviour is the whole point of writing this by hand:
 *
 * - Escape closes without acting.
 * - Focus moves into the dialog on open and returns to whatever opened
 *   it on close, so a keyboard user is never dropped at the top of the
 *   page.
 * - Tab cycles inside the dialog rather than wandering into the page
 *   behind it.
 * - The **cancel** button takes initial focus, not the destructive one.
 *   A stray Enter should not delete anything.
 */

import { useCallback, useEffect, useRef, type ReactNode } from "react";

import { Button } from "@/components/ui";

export interface ConfirmDialogProps {
  open: boolean;
  title: string;
  /** Say precisely what will happen, including counts. */
  description: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  /** Styles the confirm button as destructive. */
  destructive?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

const FOCUSABLE =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  destructive = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);

  // Remember the opener so focus can go back where it came from.
  useEffect(() => {
    if (!open) return;
    restoreRef.current = document.activeElement as HTMLElement | null;
    cancelRef.current?.focus();
    return () => restoreRef.current?.focus?.();
  }, [open]);

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onCancel();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = Array.from(
        panelRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? [],
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    },
    [onCancel],
  );

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[70] flex items-end justify-center p-4 sm:items-center">
      {/* Clicking away cancels — the same as Escape, never the action.
          Deliberately not a button: it would be a second control named
          "Cancel" for assistive tech, and Escape already covers the
          keyboard path this element would otherwise duplicate. */}
      <div aria-hidden="true" onClick={onCancel} className="absolute inset-0 bg-black/65" />
      <div
        ref={panelRef}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        aria-describedby="confirm-dialog-description"
        onKeyDown={handleKeyDown}
        className="relative w-full max-w-md rounded-[var(--radius-lg)] border border-[var(--border-default)] bg-[var(--surface-raised)] p-5 shadow-[var(--shadow-card)]"
      >
        <h2 id="confirm-dialog-title" className="text-base font-semibold">
          {title}
        </h2>
        <div
          id="confirm-dialog-description"
          className="mt-2 text-sm leading-relaxed text-[var(--text-secondary)]"
        >
          {description}
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <Button ref={cancelRef} onClick={onCancel}>
            {cancelLabel}
          </Button>
          <Button variant={destructive ? "danger" : "primary"} onClick={onConfirm}>
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}
