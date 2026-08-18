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
  return loadActiveGenerationIds()[0] ?? null;
}

/**
 * Every generation this browser had running when it was last here.
 *
 * A list rather than a single id since Phase 12: one CREATE can start
 * two songs, and a user can start a third while those run. Reconnecting
 * to only the most recent would strand the others in a state the page
 * never shows again.
 *
 * Tolerates the Phase 11 format — a bare id string — so a refresh across
 * the upgrade reconnects instead of silently dropping the job.
 */
export function loadActiveGenerationIds(): string[] {
  const store = safeLocalStorage();
  if (!store) return [];
  const raw = store.getItem(ACTIVE_KEY);
  if (raw === null) return [];
  if (isValidGenerationId(raw)) return [raw];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed.filter(isValidGenerationId);
  } catch {
    // Corrupt or hand-edited: drop it rather than carry it forward.
  }
  store.removeItem(ACTIVE_KEY);
  return [];
}

export function setActiveGenerationIds(ids: string[]): void {
  const store = safeLocalStorage();
  if (!store) return;
  const valid = ids.filter(isValidGenerationId);
  if (valid.length === 0) store.removeItem(ACTIVE_KEY);
  else store.setItem(ACTIVE_KEY, JSON.stringify(valid));
}

export function saveActiveGenerationId(id: string): void {
  if (!isValidGenerationId(id)) return;
  setActiveGenerationIds([...loadActiveGenerationIds().filter((x) => x !== id), id]);
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

/**
 * Forget everything this module remembers about the signed-in user.
 *
 * Called on sign-out and on session loss. These keys hold song ids and
 * titles — private data — and a browser is shared: without this, the
 * next person to sign in on the same machine sees the previous user's
 * songs listed before any request is made. The API would refuse to
 * serve them, but the titles alone are the leak.
 */
export function clearPrivateGenerationCache(): void {
  const store = safeLocalStorage();
  if (!store) return;
  store.removeItem(ACTIVE_KEY);
  store.removeItem(RECENT_KEY);
}
