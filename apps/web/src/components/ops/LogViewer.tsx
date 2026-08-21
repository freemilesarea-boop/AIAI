"use client";

/**
 * An incremental log viewer that does not try to hold the whole log.
 *
 * Four properties, each of which is a way a naive version fails.
 *
 * **Incremental.** Every poll sends the offset it was last given and
 * receives only what has arrived since. Refetching the file each time
 * would move hundreds of megabytes over a long run, and the browser
 * would spend the run re-parsing its own history.
 *
 * **Bounded in the DOM.** Only the most recent lines are rendered.
 * Putting a 500MB log into the DOM is not slow, it is a dead tab, and an
 * operator scrolling a dead tab has lost the thing they came for. Older
 * lines are dropped from view with the count kept visible so nothing
 * disappears silently.
 *
 * **Auto-scroll that can be stopped.** Following the tail is right until
 * somebody is reading, at which point being yanked to the bottom every
 * three seconds makes reading impossible. Scrolling up pauses it; the
 * button resumes.
 *
 * **Already redacted.** The text arrives with credentials removed by the
 * server. Nothing here hides anything, because anything this component
 * could hide it would already have received.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { Button, cx } from "@/components/ui";
import { OpsError, ops } from "@/lib/ops/client";
import { bytes } from "@/lib/ops/format";
import type { LogView } from "@/lib/ops/types";

/** Lines kept in the DOM. Well past a screenful, far short of a problem. */
const MAX_LINES = 2000;

const POLL_MS = 4000;

export function LogViewer({ runId, live }: { runId: string; live: boolean }) {
  const [stream, setStream] = useState<"stdout" | "stderr">("stdout");
  const [lines, setLines] = useState<string[]>([]);
  const [meta, setMeta] = useState<LogView | null>(null);
  const [dropped, setDropped] = useState(0);
  const [follow, setFollow] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(true);

  const offset = useRef<number | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const append = useCallback((view: LogView) => {
    setMeta(view);
    offset.current = view.next_offset;
    if (!view.text) return;
    setLines((previous) => {
      const incoming = view.text.split("\n");
      // A chunk boundary can split a line; stitch it back onto the last
      // one rather than showing half a line twice.
      const merged =
        previous.length > 0 && incoming.length > 0
          ? [...previous.slice(0, -1), previous[previous.length - 1] + incoming[0], ...incoming.slice(1)]
          : incoming;
      if (merged.length <= MAX_LINES) return merged;
      setDropped((count) => count + (merged.length - MAX_LINES));
      return merged.slice(merged.length - MAX_LINES);
    });
  }, []);

  const read = useCallback(
    async (fromStart: boolean) => {
      try {
        const view = await ops.logs(runId, {
          stream,
          offset: fromStart ? undefined : (offset.current ?? undefined),
        });
        if (!mounted.current) return;
        setError(null);
        if (fromStart) {
          setLines(view.text ? view.text.split("\n") : []);
          setDropped(0);
          setMeta(view);
          offset.current = view.next_offset;
        } else {
          append(view);
        }
      } catch (caught) {
        if (!mounted.current) return;
        setError(caught instanceof OpsError ? caught.message : "The log could not be read.");
      } finally {
        if (mounted.current) setLoading(false);
      }
    },
    [append, runId, stream],
  );

  // Switching stream starts a new read from the tail.
  useEffect(() => {
    offset.current = null;
    setLoading(true);
    void read(true);
  }, [read]);

  useEffect(() => {
    if (!live) return;
    let timer: number | undefined;
    const start = () => {
      if (timer === undefined) timer = window.setInterval(() => void read(false), POLL_MS);
    };
    const stop = () => {
      if (timer !== undefined) {
        window.clearInterval(timer);
        timer = undefined;
      }
    };
    const onVisibility = () =>
      document.visibilityState === "visible" ? start() : stop();

    if (document.visibilityState === "visible") start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [live, read]);

  useEffect(() => {
    if (!follow || !box.current) return;
    box.current.scrollTop = box.current.scrollHeight;
  }, [lines, follow]);

  const onScroll = useCallback(() => {
    const element = box.current;
    if (!element) return;
    const atBottom =
      element.scrollHeight - element.scrollTop - element.clientHeight < 24;
    setFollow(atBottom);
  }, []);

  const visible = filter
    ? lines.filter((line) => line.toLowerCase().includes(filter.toLowerCase()))
    : lines;

  if (loading && lines.length === 0) {
    return <div role="status" aria-label="Loading log" className="luber-skeleton h-48 w-full" />;
  }

  if (meta && !meta.available) {
    return (
      <p className="text-xs leading-relaxed text-[var(--text-muted)]">
        {meta.unavailable_reason ?? "No log has been written for this run."}
      </p>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <div role="group" aria-label="Log stream" className="flex gap-1">
          {(["stdout", "stderr"] as const).map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => setStream(name)}
              aria-pressed={stream === name}
              className={cx(
                "rounded-[var(--radius-sm)] px-2.5 py-1 font-mono text-[11px]",
                stream === name
                  ? "bg-[var(--surface-overlay)] text-[var(--text-primary)]"
                  : "text-[var(--text-muted)] hover:bg-[var(--surface-raised)]",
              )}
            >
              {name}
            </button>
          ))}
        </div>

        <label className="flex items-center gap-1.5 text-[11px] text-[var(--text-muted)]">
          <span className="sr-only">Filter log lines</span>
          <input
            type="search"
            value={filter}
            placeholder="filter"
            onChange={(event) => setFilter(event.target.value)}
            className="w-32 rounded-[var(--radius-sm)] border border-[var(--border-default)] bg-[var(--surface-overlay)] px-2 py-1 text-[11px] text-[var(--text-primary)]"
          />
        </label>

        <label className="flex items-center gap-1.5 text-[11px] text-[var(--text-muted)]">
          <input
            type="checkbox"
            checked={follow}
            onChange={(event) => setFollow(event.target.checked)}
          />
          Follow
        </label>

        <Button size="sm" variant="ghost" onClick={() => void read(false)}>
          Fetch new
        </Button>
        <Button size="sm" variant="ghost" onClick={() => void read(true)}>
          Reload tail
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => void navigator.clipboard?.writeText(visible.join("\n"))}
        >
          Copy
        </Button>

        {meta && (
          <span className="ml-auto text-[11px] text-[var(--text-muted)]">
            {bytes(meta.size_bytes)}
            {meta.from_tail && " · showing the tail"}
          </span>
        )}
      </div>

      {error && (
        <p role="alert" className="text-[11px] text-[var(--danger)]">
          {error} The lines below are the last that were read.
        </p>
      )}

      {dropped > 0 && (
        <p className="text-[11px] text-[var(--text-muted)]">
          {dropped.toLocaleString()} earlier line(s) are no longer held in the page. Use{" "}
          <span className="font-medium">Reload tail</span> to re-read from the end of the file.
        </p>
      )}

      <div
        ref={box}
        onScroll={onScroll}
        role="log"
        aria-label={`Trainer ${stream}`}
        aria-live="off"
        className="h-72 overflow-auto rounded-[var(--radius-md)] border border-[var(--border-subtle)] bg-[var(--surface-sunken)] p-3"
      >
        <pre className="font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-[var(--text-secondary)]">
          {visible.length > 0 ? visible.join("\n") : "(no output yet)"}
        </pre>
      </div>
    </div>
  );
}
