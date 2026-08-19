/**
 * A broadcast for "this session has ended".
 *
 * Exists because of a nesting problem with a real consequence.
 * `AuthProvider` sits *above* `PlayerProvider` in the layout, so it
 * cannot call `usePlayer()` to clear playback — and the player holds a
 * track title, which is private data belonging to whoever was signed
 * in. Inverting the providers would only move the problem, since the
 * player would then be unable to see auth state.
 *
 * So the two communicate through this instead: auth announces the end
 * of a session, and anything holding private state listens and drops
 * it. Subscribers are plain callbacks rather than React context, which
 * keeps the direction of the dependency one-way.
 */

type Listener = () => void;

const listeners = new Set<Listener>();

/**
 * Register something to clear when the session ends.
 * Returns the unsubscribe function, for effect cleanup.
 */
export function onSessionEnded(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/**
 * Tell every holder of private state to drop it.
 *
 * Called on sign-out and on session expiry. A listener that throws must
 * not stop the others: failing to clear the player is not a reason to
 * leave the library populated as well.
 */
export function emitSessionEnded(): void {
  for (const listener of [...listeners]) {
    try {
      listener();
    } catch {
      /* one listener's failure must not block the rest */
    }
  }
}
