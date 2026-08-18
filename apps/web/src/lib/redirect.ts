/**
 * Where it is safe to send someone after they sign in.
 *
 * A `?next=` parameter is attacker-controlled: anyone can send a link
 * that logs a user in and then bounces them to a site that looks like
 * LUBER and asks for their password again. The defence is not to
 * sanitise the value but to refuse anything that is not a plain
 * in-app path.
 */

/** Where a signed-in user goes when there is no specific destination. */
export const DEFAULT_DESTINATION = "/library";

/**
 * The supplied path, if it is safe; otherwise the default.
 *
 * Only a single-slash-rooted relative path is accepted. Rejected:
 * absolute URLs (`https://evil.example`), scheme-relative ones
 * (`//evil.example` — a browser treats that as another origin),
 * anything with a scheme (`javascript:`), backslash variants that some
 * parsers normalise into slashes, and control characters that can be
 * used to smuggle one past a naive check.
 */
export function safeDestination(next: string | null | undefined): string {
  if (!next) return DEFAULT_DESTINATION;

  // Strip nothing — evaluate exactly what was supplied. A value that
  // needs cleaning to be safe is a value to reject.
  if (!next.startsWith("/")) return DEFAULT_DESTINATION;
  if (next.startsWith("//")) return DEFAULT_DESTINATION;
  if (next.includes("\\")) return DEFAULT_DESTINATION;
  // Control characters, written as escapes rather than literal bytes
  // so the check stays readable and cannot be lost to a copy-paste.
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001f\u007f]/.test(next)) return DEFAULT_DESTINATION;
  // "/\evil.example" and "/	/x" style tricks are covered above; a colon
  // before the first slash would mean a scheme crept in.
  const firstSegment = next.slice(1).split("/", 1)[0] ?? "";
  if (firstSegment.includes(":")) return DEFAULT_DESTINATION;

  return next;
}

/** The login URL that will return the user to where they were headed. */
export function loginUrlFor(pathname: string): string {
  if (pathname === "/login" || pathname === "/signup") return "/login";
  const destination = safeDestination(pathname);
  return destination === DEFAULT_DESTINATION
    ? "/login"
    : `/login?next=${encodeURIComponent(destination)}`;
}
