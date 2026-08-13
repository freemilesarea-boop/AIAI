"use client";

/**
 * Many generations in flight at once.
 *
 * Phase 11 tracked exactly one job, and the Create form was disabled for
 * its whole duration — which, at 240 seconds of real inference, meant
 * the page was unusable for minutes at a time. This hook holds a list
 * instead: every submission becomes an entry that polls itself to a
 * terminal state, and submitting again adds to the list rather than
 * replacing what is there.
 *
 * Polling rules, unchanged from Phase 3 and still enforced per entry:
 * one request in flight at a time, a fixed interval, stop on
 * COMPLETED/FAILED/CANCELLED, stop on unmount, bounded deadline. A
 * client-side timeout is *not* backend failure — the entry goes to
 * `stalled` and the job keeps running.
 *
 * Active ids are mirrored into `localStorage` so a refresh mid-generation
 * reattaches to everything that was running, not just the last one.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  type CreateGenerationInput,
  type Generation,
  createGeneration,
  getGeneration,
  isTerminalStatus,
  newIdempotencyKey,
} from "@/lib/api";
import { describeApiError, type UserFacingError } from "@/lib/errors";
import {
  loadActiveGenerationIds,
  rememberGeneration,
  setActiveGenerationIds,
} from "@/lib/generationStorage";

export const POLL_INTERVAL_MS = 2000;
/** Generous ceiling: real MLX inference plus queueing, not a hard failure. */
export const POLL_TIMEOUT_MS = 20 * 60 * 1000;

export interface QueueEntry {
  id: string;
  /** Shared by the results of a single CREATE. */
  groupId: string | null;
  /** Known title before the first poll returns. */
  title: string;
  generation: Generation | null;
  /** True once the entry reaches a terminal status. */
  done: boolean;
  /** Set when the client deadline passed with the job still running. */
  stalled: boolean;
  startedAt: number;
}

export interface GenerationQueueState {
  entries: QueueEntry[];
  /** True only while a POST is in flight — never during inference. */
  submitting: boolean;
  /** A submission that never reached the backend. */
  submitError: UserFacingError | null;
  submit: (input: CreateGenerationInput) => Promise<void>;
  dismiss: (id: string) => void;
  clearFinished: () => void;
  clearSubmitError: () => void;
  /** Re-attach to a generation this browser knows about. */
  track: (id: string, title?: string) => void;
}

export function useGenerationQueue(): GenerationQueueState {
  const [entries, setEntries] = useState<QueueEntry[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<UserFacingError | null>(null);

  const mountedRef = useRef(true);
  const submittingRef = useRef(false);
  const inFlight = useRef(new Set<string>());

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const addEntry = useCallback((id: string, title: string, groupId: string | null) => {
    setEntries((current) => {
      if (current.some((entry) => entry.id === id)) return current;
      return [
        ...current,
        { id, groupId, title, generation: null, done: false, stalled: false, startedAt: Date.now() },
      ];
    });
  }, []);

  // Refresh recovery: adopt every id left behind by a previous load.
  useEffect(() => {
    for (const id of loadActiveGenerationIds()) addEntry(id, "Generating…", null);
  }, [addEntry]);

  // The set of entries still worth polling, as a stable string. The
  // polling effect keys off *this* rather than `entries`: every poll
  // response replaces the generation object, and depending on the array
  // itself would tear down and restart the interval on every response,
  // firing an immediate new request each time — a tight request loop
  // wearing an interval's clothes.
  const activeKey = entries
    .filter((entry) => !entry.done && !entry.stalled)
    .map((entry) => entry.id)
    .join(",");

  const entriesRef = useRef(entries);
  entriesRef.current = entries;

  // Keep stored ids in step with what is still running, so a refresh
  // never reattaches to a job that already finished.
  useEffect(() => {
    setActiveGenerationIds(activeKey ? activeKey.split(",") : []);
  }, [activeKey]);

  // One interval for the whole queue: each tick polls every unfinished
  // entry that does not already have a request outstanding. A timer per
  // entry would multiply requests without making anything more current.
  useEffect(() => {
    if (!activeKey) return;

    let cancelled = false;
    const controller = new AbortController();

    const pollOne = async (entry: QueueEntry) => {
      if (cancelled || inFlight.current.has(entry.id)) return;
      if (Date.now() - entry.startedAt > POLL_TIMEOUT_MS) {
        // The backend may still be working; do not claim failure.
        setEntries((current) =>
          current.map((e) => (e.id === entry.id ? { ...e, stalled: true } : e)),
        );
        return;
      }
      inFlight.current.add(entry.id);
      try {
        const next = await getGeneration(entry.id, controller.signal);
        if (cancelled || !mountedRef.current) return;
        setEntries((current) =>
          current.map((e) =>
            e.id === entry.id
              ? {
                  ...e,
                  generation: next,
                  title: next.title,
                  done: isTerminalStatus(next.status),
                }
              : e,
          ),
        );
      } catch (err) {
        if (cancelled || !mountedRef.current) return;
        // A 404 means the id is unusable — stop asking. Anything
        // transient is left alone; the next tick may succeed.
        if (!describeApiError(err).retryable) {
          setEntries((current) => current.filter((e) => e.id !== entry.id));
        }
      } finally {
        inFlight.current.delete(entry.id);
      }
    };

    const tick = () => {
      for (const entry of entriesRef.current) {
        if (!entry.done && !entry.stalled) void pollOne(entry);
      }
    };

    tick();
    const interval = window.setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      controller.abort();
      window.clearInterval(interval);
    };
  }, [activeKey]);

  const submit = useCallback(
    async (input: CreateGenerationInput) => {
      // Ref guard: two clicks in one tick must produce one POST. This is
      // *not* a lock for the generation's duration — a second, deliberate
      // submission while one is running is now supported.
      if (submittingRef.current) return;
      submittingRef.current = true;
      setSubmitting(true);
      setSubmitError(null);
      try {
        const created = await createGeneration(input, newIdempotencyKey());
        if (!mountedRef.current) return;
        // Fall back to the top-level id when `generations` is absent or
        // empty. A current backend always sends it, but a single missing
        // field must not cost the user the generation they just started.
        const results = created.generations?.length
          ? created.generations
          : [{ generation_id: created.generation_id, status: created.status, seed: null }];
        for (const result of results) {
          addEntry(result.generation_id, input.title, created.generation_group_id);
          rememberGeneration({
            id: result.generation_id,
            title: input.title,
            createdAt: new Date().toISOString(),
          });
        }
      } catch (err) {
        if (!mountedRef.current) return;
        setSubmitError(describeApiError(err));
      } finally {
        submittingRef.current = false;
        if (mountedRef.current) setSubmitting(false);
      }
    },
    [addEntry],
  );

  const dismiss = useCallback((id: string) => {
    setEntries((current) => current.filter((entry) => entry.id !== id));
  }, []);

  const clearFinished = useCallback(() => {
    setEntries((current) => current.filter((entry) => !entry.done));
  }, []);

  const track = useCallback(
    (id: string, title = "Generating…") => addEntry(id, title, null),
    [addEntry],
  );

  return {
    entries,
    submitting,
    submitError,
    submit,
    dismiss,
    clearFinished,
    clearSubmitError: () => setSubmitError(null),
    track,
  };
}
