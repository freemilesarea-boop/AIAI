"use client";

/**
 * Optional musical controls: BPM, key/scale, time signature.
 *
 * Every control here is optional and starts empty, and empty is a real
 * answer — it means "the model decides", which is exactly what happened
 * before these controls existed. Leaving the whole panel untouched
 * therefore produces the same request as Phase 7 did.
 *
 * The offered values are the pinned ACE-Step build's own accepted
 * values (see `@/lib/songcraft`), not a house selection. Nothing here
 * invents a default, and no control is shown for an engine parameter we
 * have not verified.
 */

import { useId } from "react";

import {
  BPM_MAX,
  BPM_MIN,
  TIME_SIGNATURE_OPTIONS,
  VALID_KEY_SCALES,
} from "@/lib/songcraft";

export interface AdvancedControlsProps {
  bpm: string;
  keyScale: string;
  timeSignature: string;
  onBpmChange: (value: string) => void;
  onKeyScaleChange: (value: string) => void;
  onTimeSignatureChange: (value: string) => void;
  onClear: () => void;
  bpmError?: string;
  disabled?: boolean;
  /** Open on first render — used when "Generate again" prefilled them. */
  defaultOpen?: boolean;
}

const AUTO_LABEL = "Let the model decide";

export function AdvancedControls({
  bpm,
  keyScale,
  timeSignature,
  onBpmChange,
  onKeyScaleChange,
  onTimeSignatureChange,
  onClear,
  bpmError,
  disabled = false,
  defaultOpen = false,
}: AdvancedControlsProps) {
  const ids = {
    bpm: useId(),
    keyScale: useId(),
    timeSignature: useId(),
  };

  const anySet = Boolean(bpm || keyScale || timeSignature);

  const fieldClass =
    "w-full rounded-lg border border-[var(--border-strong)] bg-[var(--surface-raised)] px-3 py-2 text-[var(--text-primary)] " +
    "placeholder:text-[var(--text-muted)] focus:border-[var(--brand)] focus:outline-none " +
    "focus-visible:ring-2 focus-visible:ring-[var(--brand)] disabled:opacity-60";
  const labelClass = "block text-sm font-medium text-[var(--text-primary)]";
  const hintClass = "mt-1 text-xs text-[var(--text-muted)]";

  return (
    <details
      open={defaultOpen || anySet}
      className="rounded-xl border border-[var(--border-default)] bg-[var(--surface-raised)] px-4 py-3"
    >
      <summary className="cursor-pointer select-none text-sm font-medium text-[var(--text-primary)] marker:text-[var(--text-muted)]">
        Advanced controls{" "}
        <span className="font-normal text-[var(--text-muted)]">
          — optional{anySet ? " · in use" : ""}
        </span>
      </summary>

      <p className="mt-2 text-xs text-[var(--text-muted)]">
        Leave any of these empty and the model chooses for you, exactly as it does today.
        Only values the pinned ACE-Step engine accepts are offered.
      </p>

      <div className="mt-4 grid gap-4 sm:grid-cols-3">
        <div>
          <label htmlFor={ids.bpm} className={labelClass}>
            BPM
          </label>
          <input
            id={ids.bpm}
            type="number"
            inputMode="numeric"
            value={bpm}
            onChange={(e) => onBpmChange(e.target.value)}
            placeholder="Auto"
            min={BPM_MIN}
            max={BPM_MAX}
            disabled={disabled}
            aria-invalid={Boolean(bpmError)}
            aria-describedby={bpmError ? `${ids.bpm}-error` : `${ids.bpm}-hint`}
            className={`mt-1.5 ${fieldClass}`}
          />
          {bpmError ? (
            <p id={`${ids.bpm}-error`} className="mt-1 text-sm text-[var(--danger)]">
              {bpmError}
            </p>
          ) : (
            <p id={`${ids.bpm}-hint`} className={hintClass}>
              {BPM_MIN}–{BPM_MAX}, or empty for auto.
            </p>
          )}
        </div>

        <div>
          <label htmlFor={ids.keyScale} className={labelClass}>
            Key / Scale
          </label>
          <select
            id={ids.keyScale}
            value={keyScale}
            onChange={(e) => onKeyScaleChange(e.target.value)}
            disabled={disabled}
            className={`mt-1.5 ${fieldClass}`}
          >
            <option value="">{AUTO_LABEL}</option>
            {VALID_KEY_SCALES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
          <p className={hintClass}>Optional.</p>
        </div>

        <div>
          <label htmlFor={ids.timeSignature} className={labelClass}>
            Time Signature
          </label>
          <select
            id={ids.timeSignature}
            value={timeSignature}
            onChange={(e) => onTimeSignatureChange(e.target.value)}
            disabled={disabled}
            className={`mt-1.5 ${fieldClass}`}
          >
            <option value="">{AUTO_LABEL}</option>
            {TIME_SIGNATURE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <p className={hintClass}>Optional.</p>
        </div>
      </div>

      {anySet && (
        <button
          type="button"
          onClick={onClear}
          disabled={disabled}
          className="mt-3 rounded-lg border border-[var(--border-strong)] px-3 py-1.5 text-xs font-medium
            text-[var(--text-secondary)] transition-colors hover:bg-[var(--surface-overlay)] focus-visible:outline-none
            focus-visible:ring-2 focus-visible:ring-[var(--border-strong)]"
        >
          Clear advanced controls
        </button>
      )}
    </details>
  );
}
