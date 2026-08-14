"use client";

/**
 * Create Cover: a new performance of a song in a different style.
 *
 * Called a cover rather than a remix on purpose. Calibration showed the
 * engine regenerates the whole performance — it is steered by the source
 * but keeps none of the recording, so nothing here may promise that the
 * original vocal, take or arrangement survives.
 *
 * The style description is the real control; calibration moved similarity
 * to the source far more by changing the target than by moving the
 * strength dial. Strength offers two levels because only two engine
 * settings were validated, and a third would be a control nobody measured.
 */

import { useState } from "react";

import { Button, Card, cx, inputClass } from "@/components/ui";
import { useToast } from "@/components/ui/Toast";
import {
  COVER_STRENGTHS,
  coverGeneration,
  findMasterAsset,
  type CoverStrength,
  type Generation,
} from "@/lib/api";

export interface CreateCoverProps {
  generation: Generation;
  onCovered?: (generationId: string) => void;
}

/** Neutral, concrete examples. No artist names, living or otherwise. */
export const STYLE_EXAMPLES = [
  "modern synth pop with glossy production",
  "warm contemporary R&B with live-feeling drums",
  "dreamy indie pop with spacious guitars",
];

export function CreateCover({ generation, onCovered }: CreateCoverProps) {
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [style, setStyle] = useState("");
  const [strength, setStrength] = useState<CoverStrength>("subtle");
  const [touched, setTouched] = useState(false);

  const ready = generation.status === "COMPLETED" && findMasterAsset(generation) !== null;
  if (!ready) return null;

  const trimmed = style.trim();
  const error = trimmed ? null : "Describe the style you want.";

  const submit = async () => {
    setTouched(true);
    if (error) return;
    setBusy(true);
    try {
      const created = await coverGeneration(generation.id, { prompt: trimmed, strength });
      toast.notify("Creating your cover");
      setOpen(false);
      onCovered?.(created.generation_id);
    } catch {
      toast.notifyError("Could not create a cover of this song.");
    } finally {
      setBusy(false);
    }
  };

  if (!open) {
    return (
      <Button onClick={() => setOpen(true)} aria-expanded={false}>
        Create cover
      </Button>
    );
  }

  return (
    <Card className="w-full p-4">
      <h3 className="text-sm font-semibold">Create a cover</h3>
      <p className="mt-1 text-xs text-[var(--text-secondary)]">
        {/* Truthful framing. The engine keeps the musical structure as
            conditioning and regenerates the performance; it does not
            preserve the recording, the take or the voice. */}
        Creates a new performance inspired by this song&rsquo;s musical structure and your
        target style. The original recording and vocal are not kept — you get a new
        recording, and this song stays as it is.
      </p>

      <div className="mt-3">
        <label htmlFor="cover-style" className="block text-xs font-medium">
          Target style
        </label>
        <textarea
          id="cover-style"
          value={style}
          onChange={(e) => {
            setStyle(e.target.value);
            setTouched(true);
          }}
          rows={2}
          placeholder={STYLE_EXAMPLES[0]}
          className={cx(inputClass, "mt-1 resize-y py-1.5 text-sm")}
        />
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {STYLE_EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => {
                setStyle(example);
                setTouched(true);
              }}
              className="inline-flex min-h-8 items-center rounded-[var(--radius-full)]
                border border-[var(--border-default)] px-3 text-[11px]
                text-[var(--text-secondary)] transition-colors
                hover:border-[var(--brand)] hover:text-[var(--brand-text)]"
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      <fieldset className="mt-3">
        <legend className="text-xs font-medium">How much should it change?</legend>
        <div className="mt-1.5 flex flex-wrap gap-2">
          {COVER_STRENGTHS.map((option) => (
            <label
              key={option.value}
              className={cx(
                "inline-flex min-h-9 cursor-pointer items-center gap-2 rounded-[var(--radius-md)]",
                "border px-3 text-xs transition-colors",
                strength === option.value
                  ? "border-[var(--brand)] bg-[var(--brand-muted)] text-[var(--brand-text)]"
                  : "border-[var(--border-default)] text-[var(--text-secondary)]",
              )}
              title={option.hint}
            >
              <input
                type="radio"
                name="cover-strength"
                value={option.value}
                checked={strength === option.value}
                onChange={() => setStrength(option.value)}
                className="h-3.5 w-3.5 accent-[var(--brand)]"
              />
              {option.label}
            </label>
          ))}
        </div>
      </fieldset>

      {touched && error && (
        <p role="alert" className="mt-2 text-xs text-[var(--danger)]">
          {error}
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button variant="primary" size="sm" busy={busy} onClick={() => void submit()}>
          Create cover
        </Button>
        <Button size="sm" disabled={busy} onClick={() => setOpen(false)}>
          Cancel
        </Button>
      </div>
    </Card>
  );
}
