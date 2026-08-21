"use client";

/**
 * The buttons that change something, and how they ask.
 *
 * Every one of these is confirmed with a sentence describing what will
 * actually happen — Step 23. "Are you sure?" is not a question anybody
 * can answer well: it asks the operator to remember what they clicked
 * rather than telling them what it does. The sentence comes from the
 * server, so the dialog cannot describe an action differently from the
 * endpoint that performs it.
 *
 * A disabled button carries its reason. An operator who cannot dispatch
 * a run should be able to read why — "this run has not been validated",
 * "remote dispatch needs SSH credentials this console does not hold" —
 * without opening a document.
 *
 * The result is shown as the server phrased it, including when the
 * server did *less* than the button implied. A remote cancellation is
 * recorded rather than delivered, and saying "Cancelled" there would be
 * how an operator comes to believe a rented GPU has been released.
 */

import { useState } from "react";

import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { Button } from "@/components/ui";
import { OpsError, ops } from "@/lib/ops/client";
import type { ActionAvailability, ActionResult } from "@/lib/ops/types";

export function RunActions({
  runId,
  actions,
  onCompleted,
}: {
  runId: string;
  actions: ActionAvailability[];
  onCompleted: (result: ActionResult) => void;
}) {
  const [pending, setPending] = useState<ActionAvailability | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [result, setResult] = useState<ActionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const perform = async (action: ActionAvailability) => {
    setPending(null);
    // Guards a double submit at the source as well as at the server:
    // the API is idempotent by state, and this keeps the second request
    // from being sent at all.
    if (busy) return;
    setBusy(action.action);
    setError(null);
    try {
      const outcome = await ops.runAction(runId, apiAction(action.action));
      setResult(outcome);
      onCompleted(outcome);
    } catch (caught) {
      setError(
        caught instanceof OpsError ? caught.message : "The action could not be performed.",
      );
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        {actions.map((action) => (
          <span key={action.action} className="inline-flex flex-col gap-1">
            <Button
              size="sm"
              // A disabled destructive button styled as destructive is
              // louder than the enabled action beside it, which is
              // exactly backwards: the eye goes to the thing that
              // cannot be done.
              variant={action.destructive && action.available ? "danger" : "secondary"}
              disabled={!action.available}
              busy={busy === action.action}
              title={action.reason || undefined}
              onClick={() => setPending(action)}
            >
              {action.label}
            </Button>
            {!action.available && action.reason && (
              <span className="max-w-[16rem] text-[10px] leading-snug text-[var(--text-muted)]">
                {action.reason}
              </span>
            )}
          </span>
        ))}
      </div>

      {error && (
        <p role="alert" className="text-xs text-[var(--danger)]">
          {error}
        </p>
      )}

      {result && (
        <div
          role="status"
          className="rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-overlay)] px-3 py-2 text-xs text-[var(--text-secondary)]"
        >
          <p className="font-medium text-[var(--text-primary)]">
            {result.performed ? "Done" : "Recorded, not performed"}
            {result.outcome && (
              <span className="ml-2 font-mono text-[10px] text-[var(--text-muted)]">
                {result.outcome}
              </span>
            )}
          </p>
          <p className="mt-0.5 leading-relaxed">{result.detail}</p>
        </div>
      )}

      <ConfirmDialog
        open={pending !== null}
        title={pending ? pending.label : ""}
        description={pending?.confirmation ?? ""}
        confirmLabel={pending ? CONFIRM_LABEL[pending.action] ?? pending.label : "Confirm"}
        // Not "Cancel": one of the actions *is* a cancellation, and a
        // dialog with two buttons reading "Cancel" is a dialog nobody
        // can answer.
        cancelLabel="Go back"
        destructive={pending?.destructive ?? false}
        onConfirm={() => pending && void perform(pending)}
        onCancel={() => setPending(null)}
      />
    </div>
  );
}

/**
 * What the confirming button says.
 *
 * Named for the effect rather than the control, so the destructive
 * button in a cancellation dialog does not read "Cancel" beside a
 * dismiss button that also reads "Cancel".
 */
const CONFIRM_LABEL: Record<string, string> = {
  validate: "Run the gates",
  dispatch: "Dispatch the run",
  cancel: "Cancel the run",
  reconcile: "Ask the worker",
  create_retry_run: "Create retry run",
};

/** The console's action names map onto the API's route segments. */
function apiAction(name: string): string {
  return name === "create_retry_run" ? "retry" : name;
}
