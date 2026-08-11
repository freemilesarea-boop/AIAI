/**
 * Browser-local persistence for generation continuity.
 *
 * Two concerns, both deliberately small:
 *  - the active generation id, so a refresh mid-generation reconnects
 *    to the job instead of losing it;
 *  - a short list of recently created ids for the session's history.
 *
 * The backend remains the source of truth: nothing here is trusted
 * beyond "an id worth asking the API about". Ownership and durable
 * history belong to a later phase.
 */

const ACTIVE_KEY = "luber.activeGenerationId";
const RECENT_KEY = "luber.recentGenerations";
const RECENT_LIMIT = 8;

/** Shape of a locally remembered generation (display hints only). */
export interface RecentGenerationEntry {
  id: string;
  title: string;
  createdAt: string;
}

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Guards against corrupt or hand-edited localStorage values. */
export function isValidGenerationId(value: unknown): value is string {
  return typeof value === "string" && UUID_RE.test(value);
}

function safeLocalStorage(): Storage | null {
  try {
    if (typeof window === "undefined" || !window.localStorage) return null;
    return window.localStorage;
  } catch {
    // Private mode / blocked storage — the app still works, just without recovery.
    return null;
  }
}

export function loadActiveGenerationId(): string | null {
  const store = safeLocalStorage();
  if (!store) return null;
  const raw = store.getItem(ACTIVE_KEY);
  if (!isValidGenerationId(raw)) {
    if (raw !== null) store.removeItem(ACTIVE_KEY);
    return null;
  }
  return raw;
}

export function saveActiveGenerationId(id: string): void {
  if (!isValidGenerationId(id)) return;
  safeLocalStorage()?.setItem(ACTIVE_KEY, id);
}

export function clearActiveGenerationId(): void {
  safeLocalStorage()?.removeItem(ACTIVE_KEY);
}

export function loadRecentGenerations(): RecentGenerationEntry[] {
  const store = safeLocalStorage();
  if (!store) return [];
  const raw = store.getItem(RECENT_KEY);
  if (!raw) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (e): e is RecentGenerationEntry =>
        typeof e === "object" &&
        e !== null &&
        isValidGenerationId((e as RecentGenerationEntry).id) &&
        typeof (e as RecentGenerationEntry).title === "string",
    );
  } catch {
    store.removeItem(RECENT_KEY);
    return [];
  }
}

export function rememberGeneration(entry: RecentGenerationEntry): RecentGenerationEntry[] {
  const store = safeLocalStorage();
  if (!store || !isValidGenerationId(entry.id)) return loadRecentGenerations();
  const next = [entry, ...loadRecentGenerations().filter((e) => e.id !== entry.id)].slice(
    0,
    RECENT_LIMIT,
  );
  store.setItem(RECENT_KEY, JSON.stringify(next));
  return next;
}
