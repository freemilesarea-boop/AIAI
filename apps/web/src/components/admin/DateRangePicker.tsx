"use client";

/**
 * The window every figure on the page is computed over.
 *
 * Presets for the questions asked daily, and a calendar for everything
 * else. The presets are not a separate mode: each one resolves to a
 * concrete `from`/`to` pair, and the chip highlights when the current
 * range happens to equal it. So a custom range that lands on this
 * month's boundaries lights up "이번 달" — the same state, described
 * once, rather than a preset flag that can disagree with the dates
 * beside it.
 *
 * Dates are Korean calendar days. The picker computes them in
 * Asia/Seoul regardless of where the operator's laptop thinks it is,
 * because the figures underneath are bucketed by Korean days and a
 * range that meant something else would quietly shift every number.
 *
 * `<input type="date">` rather than a bespoke calendar: it is a real
 * date picker on every platform, it is keyboard-navigable and
 * screen-reader-labelled without any work from us, and it costs no
 * bundle. A hand-rolled one would be worse at all three.
 */

import { useEffect, useState } from "react";

import { Button, cx, inputClass } from "@/components/ui";
import {
  MAX_RANGE_DAYS,
  PRESETS,
  type DateRange,
  type PresetId,
  formatRange,
  isValidRange,
  kstToday,
  matchingPreset,
  presetRange,
  rangeLength,
} from "@/lib/admin";

export function DateRangePicker({
  range,
  onChange,
}: {
  range: DateRange;
  onChange: (next: DateRange) => void;
}) {
  const active: PresetId | null = matchingPreset(range);
  const [custom, setCustom] = useState(active === null);
  const [draft, setDraft] = useState<DateRange>(range);

  // Follow the range when it changes elsewhere — a back navigation, or
  // a preset chip. Without this the open calendar keeps showing the
  // dates from before the change.
  useEffect(() => {
    setDraft(range);
    if (matchingPreset(range) === null) setCustom(true);
  }, [range]);

  const draftValid = isValidRange(draft);
  const tooLong = draft.to >= draft.from && rangeLength(draft) > MAX_RANGE_DAYS;
  const backwards = Boolean(draft.from && draft.to && draft.to < draft.from);

  return (
    <div className="flex flex-col items-start gap-2 lg:items-end">
      <div className="flex max-w-full flex-wrap gap-1.5">
        {PRESETS.map((preset) => (
          <button
            key={preset.id}
            type="button"
            aria-pressed={active === preset.id}
            onClick={() => {
              setCustom(false);
              onChange(presetRange(preset.id));
            }}
            className={cx(
              "rounded-[var(--radius-md)] px-3 py-1.5 text-sm font-medium transition-colors",
              active === preset.id && !custom
                ? "bg-[var(--accent-muted)] text-[var(--accent)]"
                : "text-[var(--text-secondary)] hover:bg-[var(--surface-sunken)]",
            )}
          >
            {preset.label}
          </button>
        ))}
        <button
          type="button"
          aria-pressed={custom || active === null}
          aria-expanded={custom}
          onClick={() => setCustom((open) => !open)}
          className={cx(
            "rounded-[var(--radius-md)] px-3 py-1.5 text-sm font-medium transition-colors",
            custom || active === null
              ? "bg-[var(--accent-muted)] text-[var(--accent)]"
              : "text-[var(--text-secondary)] hover:bg-[var(--surface-sunken)]",
          )}
        >
          직접 선택
        </button>
      </div>

      {custom ? (
        <form
          className="flex max-w-full flex-wrap items-end gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (draftValid) onChange(draft);
          }}
        >
          <div className="flex flex-col gap-1">
            <label className="text-[11px] text-[var(--text-muted)]" htmlFor="range-from">
              시작 날짜
            </label>
            <input
              id="range-from"
              type="date"
              className={cx(inputClass, "w-[9.5rem]")}
              value={draft.from}
              max={kstToday()}
              onChange={(event) => setDraft((d) => ({ ...d, from: event.target.value }))}
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-[11px] text-[var(--text-muted)]" htmlFor="range-to">
              종료 날짜
            </label>
            <input
              id="range-to"
              type="date"
              className={cx(inputClass, "w-[9.5rem]")}
              value={draft.to}
              onChange={(event) => setDraft((d) => ({ ...d, to: event.target.value }))}
            />
          </div>
          <Button type="submit" variant="primary" disabled={!draftValid}>
            적용
          </Button>
        </form>
      ) : null}

      {/* Said before the request rather than after a 422. */}
      {custom && backwards ? (
        <p role="alert" className="text-xs text-[var(--danger)]">
          종료 날짜가 시작 날짜보다 앞설 수 없습니다.
        </p>
      ) : null}
      {custom && tooLong ? (
        <p role="alert" className="text-xs text-[var(--danger)]">
          최대 {MAX_RANGE_DAYS}일까지 조회할 수 있습니다.
        </p>
      ) : null}

      <p className="text-xs text-[var(--text-muted)]" aria-live="polite">
        조회 기간 <span className="text-[var(--text-secondary)]">{formatRange(range)}</span> (KST)
      </p>
    </div>
  );
}
