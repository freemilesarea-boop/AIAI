/**
 * Failure translation for the generation UI.
 *
 * Users see a plain explanation and, where useful, what to do next.
 * Stack traces, exception strings, environment values, filesystem
 * paths, and database/queue internals are never rendered — the backend
 * already restricts clients to machine-readable `ErrorCode` values, and
 * this module maps those to human sentences.
 */

import { ApiError } from "./api";

export type ErrorCodeLike =
  | "GENERATION_TIMEOUT"
  | "MODEL_LOAD_FAILED"
  | "OUT_OF_MEMORY"
  | "INVALID_AUDIO"
  | "UPLOAD_FAILED"
  | "ENCODING_FAILED"
  | "QUEUE_FAILED"
  | "UNKNOWN_GENERATION_ERROR";

const CODE_MESSAGES: Record<string, string> = {
  GENERATION_TIMEOUT:
    "The model took too long to finish this track. Try a shorter duration or generate again.",
  MODEL_LOAD_FAILED:
    "The music model is not available right now. Please try again in a moment.",
  OUT_OF_MEMORY:
    "The server ran out of memory for this track. Try a shorter duration.",
  INVALID_AUDIO:
    "The generated audio could not be read. Please try generating again.",
  UPLOAD_FAILED: "Your track was created but could not be saved. Please try again.",
  ENCODING_FAILED: "The track could not be prepared for playback. Please try again.",
  QUEUE_FAILED:
    "The generation service is busy and could not accept your request. Please try again.",
  UNKNOWN_GENERATION_ERROR:
    "Something went wrong while creating your track. Please try again.",
};

export interface UserFacingError {
  message: string;
  /** Whether offering a Retry button is safe for this failure. */
  retryable: boolean;
}

/** Message for a generation that reached FAILED, from its error_code. */
export function describeGenerationFailure(errorCode: string | null): UserFacingError {
  const message =
    (errorCode && CODE_MESSAGES[errorCode]) ?? CODE_MESSAGES.UNKNOWN_GENERATION_ERROR;
  return { message, retryable: true };
}

/** Message for a transport/HTTP failure raised while calling the API. */
export function describeApiError(error: unknown): UserFacingError {
  if (error instanceof ApiError) {
    if (error.status === 404) {
      return {
        message: "That generation could not be found. It may have been removed.",
        retryable: false,
      };
    }
    if (error.status === 422) {
      return {
        message: "Some of the details in the form were not accepted. Please review and try again.",
        retryable: false,
      };
    }
    if (error.status === 503) {
      return {
        message:
          "The generation service is unavailable right now. Please try again in a moment.",
        retryable: true,
      };
    }
    if (error.code && CODE_MESSAGES[error.code]) {
      return { message: CODE_MESSAGES[error.code], retryable: true };
    }
    return {
      message: "The server could not complete that request. Please try again.",
      retryable: true,
    };
  }
  // Network failure, DNS, offline, CORS — never surface the raw text.
  return {
    message:
      "Could not reach the LUBER service. Check your connection and try again.",
    retryable: true,
  };
}
