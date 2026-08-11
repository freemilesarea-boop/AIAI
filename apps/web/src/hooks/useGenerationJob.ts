"use client";

/**
 * Tracks one generation from submission to a terminal state.
 *
 * Polling rules (Step 15): one request in flight at a time, a fixed
 * interval, stop on COMPLETED/FAILED/CANCELLED, stop on unmount, and a
 * bounded client deadline. A client timeout is explicitly *not* treated
 * as backend failure — the job keeps running and the user can reconnect.
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
  clearActiveGenerationId,
  loadActiveGenerationId,
  rememberGeneration,
  saveActiveGenerationId,
} from "@/lib/generationStorage";

export const POLL_INTERVAL_MS = 2000;
/** Generous ceiling: real MLX inference plus queueing, not a hard failure. */
export const POLL_TIMEOUT_MS = 20 * 60 * 1000;

export type JobPhase = "idle" | "submitting" | "tracking" | "done" | "error" | "stalled";

export interface GenerationJobState {
  phase: JobPhase;
  generation: Generation | null;
  generationId: string | null;
  error: UserFacingError | null;
  elapsedSeconds: number;
  submit: (input: CreateGenerationInput) => Promise<void>;
  reset: () => void;
  reconnect: (id: string) => void;
}

export function useGenerationJob(): GenerationJobState {
  const [phase, setPhase] = useState<JobPhase>("idle");
  const [generation, setGeneration] = useState<Generation | null>(null);
  const [generationId, setGenerationId] = useState<string | null>(null);
  const [error, setError] = useState<UserFacingError | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const mountedRef = useRef(true);
  const pollingRef = useRef(false);
  const submittingRef = useRef(false);
  const startedAtRef = useRef<number | null>(null);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Refresh recovery: adopt any active id left behind by a prior load.
  useEffect(() => {
    const stored = loadActiveGenerationId();
    if (stored) {
      setGenerationId(stored);
      setPhase("tracking");
      startedAtRef.current = Date.now();
    }
  }, []);

  // Elapsed-time ticker, independent of poll results.
  useEffect(() => {
    if (phase !== "tracking" && phase !== "submitting") return;
    if (startedAtRef.current === null) startedAtRef.current = Date.now();
    const started = startedAtRef.current;
    const id = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - started) / 1000));
    }, 1000);
    return () => window.clearInterval(id);
  }, [phase]);

  // Bounded, single-flight polling.
  useEffect(() => {
    if (phase !== "tracking" || !generationId) return;

    let cancelled = false;
    const controller = new AbortController();
    const deadline = Date.now() + POLL_TIMEOUT_MS;

    const tick = async () => {
      if (cancelled || pollingRef.current) return;
      if (Date.now() > deadline) {
        // The backend may still be working; do not claim failure.
        if (!cancelled && mountedRef.current) setPhase("stalled");
        return;
      }
      pollingRef.current = true;
      try {
        const next = await getGeneration(generationId, controller.signal);
        if (cancelled || !mountedRef.current) return;
        setGeneration(next);
        setError(null);
        if (isTerminalStatus(next.status)) {
          setPhase("done");
          clearActiveGenerationId();
        }
      } catch (err) {
        if (cancelled || !mountedRef.current) return;
        const described = describeApiError(err);
        if (!described.retryable) {
          // e.g. 404 — the id is not usable; stop and clear it.
          setError(described);
          setPhase("error");
          clearActiveGenerationId();
        }
        // Transient errors: keep polling, the next tick may succeed.
      } finally {
        pollingRef.current = false;
      }
    };

    void tick();
    const interval = window.setInterval(() => void tick(), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      controller.abort();
      window.clearInterval(interval);
      pollingRef.current = false;
    };
  }, [phase, generationId]);

  const submit = useCallback(async (input: CreateGenerationInput) => {
    // Duplicate-submit guard: ref, so concurrent calls in one tick lose.
    if (submittingRef.current) return;
    submittingRef.current = true;
    setPhase("submitting");
    setError(null);
    setGeneration(null);
    setElapsedSeconds(0);
    startedAtRef.current = Date.now();
    try {
      const created = await createGeneration(input, newIdempotencyKey());
      if (!mountedRef.current) return;
      setGenerationId(created.generation_id);
      saveActiveGenerationId(created.generation_id);
      rememberGeneration({
        id: created.generation_id,
        title: input.title,
        createdAt: new Date().toISOString(),
      });
      setPhase("tracking");
    } catch (err) {
      if (!mountedRef.current) return;
      setError(describeApiError(err));
      setPhase("error");
    } finally {
      submittingRef.current = false;
    }
  }, []);

  const reset = useCallback(() => {
    clearActiveGenerationId();
    setPhase("idle");
    setGeneration(null);
    setGenerationId(null);
    setError(null);
    setElapsedSeconds(0);
    startedAtRef.current = null;
  }, []);

  const reconnect = useCallback((id: string) => {
    setError(null);
    setGeneration(null);
    setGenerationId(id);
    setElapsedSeconds(0);
    startedAtRef.current = Date.now();
    setPhase("tracking");
  }, []);

  return {
    phase,
    generation,
    generationId,
    error,
    elapsedSeconds,
    submit,
    reset,
    reconnect,
  };
}
