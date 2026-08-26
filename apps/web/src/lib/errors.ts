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
  | "PROVIDER_BUSY"
  | "GENERATION_INTERRUPTED"
  | "UPLOAD_FAILED"
  | "ENCODING_FAILED"
  | "QUEUE_FAILED"
  | "QUALITY_CHECK_FAILED"
  | "QUALITY_RETRY_EXHAUSTED"
  | "UNKNOWN_GENERATION_ERROR";

const CODE_MESSAGES: Record<string, string> = {
  GENERATION_TIMEOUT:
    "음악을 완성하는 데 시간이 너무 오래 걸렸습니다. 길이를 줄이거나 다시 만들어 주세요.",
  MODEL_LOAD_FAILED:
    "음악 모델을 지금 이용할 수 없습니다. 잠시 후 다시 시도해 주세요.",
  OUT_OF_MEMORY:
    "이 곡을 만들 메모리가 부족했습니다. 길이를 줄여 보세요.",
  INVALID_AUDIO:
    "만들어진 오디오를 읽지 못했습니다. 다시 만들어 주세요.",
  // The engine's own queue was full. Nothing is wrong with the song, so
  // the copy says "busy", not "failed" — the identical request works
  // once a slot frees.
  PROVIDER_BUSY:
    "음악 엔진이 지금 바쁩니다. 설정에는 문제가 없으니 잠시 후 다시 시도해 주세요.",
  // The worker stopped mid-run. Usually the queue retries and this is
  // never seen; when it is, retrying is exactly the right move.
  GENERATION_INTERRUPTED:
    "곡이 완성되기 전에 중단되었습니다. 다시 만들어 주세요.",
  UPLOAD_FAILED: "곡은 만들어졌지만 저장하지 못했습니다. 다시 시도해 주세요.",
  ENCODING_FAILED: "재생 준비에 실패했습니다. 다시 시도해 주세요.",
  QUEUE_FAILED:
    "생성 서비스가 바빠 요청을 받지 못했습니다. 다시 시도해 주세요.",
  // Phase 29. Every attempt failed a technical check — silent audio, a
  // track that stopped early, the wrong length. The copy stays general
  // on purpose: "candidate 2 was rejected for early collapse" is an
  // implementation detail, and the operator trace already holds it.
  QUALITY_CHECK_FAILED:
    "쓸 만한 품질로 만들어지지 않았습니다. 다시 만들어 주세요.",
  QUALITY_RETRY_EXHAUSTED:
    "쓸 만한 품질로 만들어지지 않았습니다. 다시 만들어 주세요.",
  UNKNOWN_GENERATION_ERROR:
    "음악을 만드는 중 문제가 생겼습니다. 다시 시도해 주세요.",
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
        message: "해당 음악을 찾을 수 없습니다. 삭제되었을 수 있습니다.",
        retryable: false,
      };
    }
    if (error.status === 422) {
      return {
        message: "입력한 내용 중 일부가 받아들여지지 않았습니다. 확인 후 다시 시도해 주세요.",
        retryable: false,
      };
    }
    if (error.status === 503) {
      return {
        message:
          "생성 서비스를 지금 이용할 수 없습니다. 잠시 후 다시 시도해 주세요.",
        retryable: true,
      };
    }
    if (error.code && CODE_MESSAGES[error.code]) {
      return { message: CODE_MESSAGES[error.code], retryable: true };
    }
    return {
      message: "서버가 요청을 처리하지 못했습니다. 다시 시도해 주세요.",
      retryable: true,
    };
  }
  // Network failure, DNS, offline, CORS — never surface the raw text.
  return {
    message:
      "BOORDA 서버에 연결하지 못했습니다. 연결 상태를 확인하고 다시 시도해 주세요.",
    retryable: true,
  };
}
