/**
 * Rendering values without inventing any.
 *
 * One rule runs through all of it: `null` and `undefined` become the
 * word UNKNOWN, never a zero, never an em dash, never a blank cell.
 * A blank reads as "nothing there" and a zero reads as a measurement,
 * and the difference between "this GPU has no VRAM" and "nobody has
 * measured this GPU" is the difference between a run that will not fit
 * and a run nobody checked.
 */

export const UNKNOWN = "UNKNOWN";

export function isKnown<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}

/** A number, or UNKNOWN. Never a silent zero. */
export function num(value: number | null | undefined, digits = 0): string {
  if (!isKnown(value)) return UNKNOWN;
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function decimal(value: number | null | undefined, digits = 4): string {
  if (!isKnown(value)) return UNKNOWN;
  // Small losses need precision; large learning rates read better in
  // exponent form than as a run of zeroes.
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(digits);
}

export function bool(value: boolean | null | undefined): string {
  if (!isKnown(value)) return UNKNOWN;
  return value ? "Yes" : "No";
}

export function text(value: string | null | undefined): string {
  return value && value.trim() ? value : UNKNOWN;
}

/** Bytes at human scale. Binary units, because that is what disks report. */
export function bytes(value: number | null | undefined): string {
  if (!isKnown(value)) return UNKNOWN;
  if (value < 1024) return `${value} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let size = value / 1024;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(size >= 100 ? 0 : 1)} ${units[index]}`;
}

export function megabytes(value: number | null | undefined): string {
  if (!isKnown(value)) return UNKNOWN;
  return bytes(value * 1024 * 1024);
}

/** A duration as an operator would say it. */
export function duration(seconds: number | null | undefined): string {
  if (!isKnown(seconds)) return UNKNOWN;
  const total = Math.max(0, Math.floor(seconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes}m ${total % 60}s`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ${minutes % 60}m`;
  return `${Math.floor(hours / 24)}d ${hours % 24}h`;
}

/**
 * How long a *run* has taken, or that it has not begun.
 *
 * Separate from `duration` because the null means something different
 * here. A run has no elapsed time exactly when it has not started, and
 * that is a fact we hold rather than a measurement nobody took —
 * rendering it UNKNOWN would send an operator looking for a number that
 * does not exist yet. Everywhere else (an evaluation's wall time, a
 * worker's uptime) null really does mean unmeasured, so those keep
 * `duration`.
 */
export function runDuration(
  seconds: number | null | undefined,
  startedAt: string | null | undefined,
): string {
  if (!isKnown(seconds)) return startedAt ? UNKNOWN : "not started";
  return duration(seconds);
}

/** A timestamp for something that may simply not have happened yet. */
export function timestampOrNotYet(value: string | null | undefined, label: string): string {
  return value ? timestamp(value) : label;
}

/**
 * How long ago, in words.
 *
 * The exact timestamp is always shown beside this — Step 32 — because
 * "5s ago" is what an operator reads at a glance and the ISO instant is
 * what they paste into a bug report.
 */
export function age(seconds: number | null | undefined): string {
  if (!isKnown(seconds)) return "never";
  if (seconds < 2) return "just now";
  return `${duration(seconds)} ago`;
}

/** A local-time rendering that keeps the seconds. */
export function timestamp(value: string | null | undefined): string {
  if (!value) return UNKNOWN;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/**
 * A digest, shortened for a table but never silently.
 *
 * The full value is always available to copy; this is the reading form.
 */
export function shortDigest(value: string | null | undefined, length = 12): string {
  if (!value) return UNKNOWN;
  return value.length <= length ? value : `${value.slice(0, length)}…`;
}

/** Turns `EVALUATION_LEAKAGE` into `Evaluation leakage` for a heading. */
export function humanise(value: string): string {
  if (!value) return "";
  const spaced = value.replace(/[_-]+/g, " ").toLowerCase();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

export function currency(value: number | null | undefined, code: string | null): string {
  if (!isKnown(value)) return UNKNOWN;
  try {
    return value.toLocaleString(undefined, {
      style: "currency",
      currency: code ?? "USD",
      maximumFractionDigits: 2,
    });
  } catch {
    return `${value.toFixed(2)} ${code ?? ""}`.trim();
  }
}
