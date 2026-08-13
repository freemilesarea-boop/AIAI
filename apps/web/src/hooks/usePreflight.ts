"use client";

/**
 * Live pre-flight advisories for the lyrics being edited.
 *
 * The heuristics live in the backend and are *not* reimplemented here:
 * this debounces the draft and asks the API, so what the editor shows is
 * by construction what submitting would record.
 *
 * Three rules shape the behaviour:
 *
 * - Advisories never block. A failed or in-flight pre-flight must never
 *   stop the user from generating, so failures clear the panel silently
 *   rather than surfacing an error.
 * - No lyrics, no request. Every advisory family reads lyric text, so a
 *   blank draft has nothing to say and costs no round-trip.
 * - One request at a time. Each keystroke aborts the previous call.
 */

import { useEffect, useRef, useState } from "react";

import { preflightGeneration } from "@/lib/api";
import type { Advisory, SectionSummary } from "@/lib/songcraft";

export const PREFLIGHT_DEBOUNCE_MS = 400;

export interface PreflightInput {
  lyrics: string;
  duration: number;
  language: string | null;
  instrumental: boolean;
}

export interface PreflightState {
  advisories: Advisory[];
  sections: SectionSummary[];
  preambleLineCount: number;
  estimatedSyllables: number;
  checking: boolean;
}

const EMPTY: PreflightState = {
  advisories: [],
  sections: [],
  preambleLineCount: 0,
  estimatedSyllables: 0,
  checking: false,
};

export function usePreflight(
  { lyrics, duration, language, instrumental }: PreflightInput,
  debounceMs = PREFLIGHT_DEBOUNCE_MS,
): PreflightState {
  const [state, setState] = useState<PreflightState>(EMPTY);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!lyrics.trim()) {
      setState(EMPTY);
      return;
    }

    const controller = new AbortController();
    setState((previous) => ({ ...previous, checking: true }));

    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const result = await preflightGeneration(
            { lyrics, duration, language, instrumental },
            controller.signal,
          );
          if (!mountedRef.current || controller.signal.aborted) return;
          setState({
            advisories: result.advisories,
            sections: result.sections,
            preambleLineCount: result.preamble_line_count,
            estimatedSyllables: result.estimated_syllables,
            checking: false,
          });
        } catch {
          // Diagnostics are best-effort: say nothing rather than
          // alarm the user about a check they did not ask for.
          if (!mountedRef.current || controller.signal.aborted) return;
          setState(EMPTY);
        }
      })();
    }, debounceMs);

    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [lyrics, duration, language, instrumental, debounceMs]);

  return state;
}
