import "@testing-library/jest-dom/vitest";

/**
 * localStorage polyfill for the test environment.
 *
 * Node 26 exposes a native `localStorage` global that is inert unless
 * the process is started with `--localstorage-file`, and its presence
 * stops jsdom from installing its own implementation — so `window`
 * ends up with `sessionStorage` but no `localStorage`. The app already
 * degrades gracefully when storage is missing, but refresh-recovery
 * tests need a working store, so provide a minimal in-memory one.
 */
if (typeof window !== "undefined" && !window.localStorage) {
  const createMemoryStorage = (): Storage => {
    let entries = new Map<string, string>();
    return {
      get length() {
        return entries.size;
      },
      clear() {
        entries = new Map();
      },
      getItem(key: string) {
        return entries.has(key) ? (entries.get(key) as string) : null;
      },
      key(index: number) {
        return Array.from(entries.keys())[index] ?? null;
      },
      removeItem(key: string) {
        entries.delete(key);
      },
      setItem(key: string, value: string) {
        entries.set(key, String(value));
      },
    };
  };

  Object.defineProperty(window, "localStorage", {
    value: createMemoryStorage(),
    configurable: true,
    writable: true,
  });
}
