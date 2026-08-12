/**
 * Typed client for the LUBER backend API.
 *
 * Every network call the browser makes goes through this module. The
 * browser talks only to the LUBER API — it never contacts ACE-Step or
 * any model runtime directly, and it never sees a storage key or a
 * filesystem path. Audio is addressed by generation id.
 *
 * Values mirror `packages/schemas` (`luber_schemas.enums`), which owns
 * the persisted contract.
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export interface HealthResponse {
  status: string;
  service: string;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status}`);
  }
  return (await res.json()) as HealthResponse;
}

/** Lifecycle of a generation job — mirrors `GenerationStatus`. */
export const GENERATION_STATUSES = [
  "QUEUED",
  "STARTING",
  "GENERATING",
  "POST_PROCESSING",
  "UPLOADING",
  "COMPLETED",
  "FAILED",
  "CANCELLED",
] as const;

export type GenerationStatus = (typeof GENERATION_STATUSES)[number];

const TERMINAL_STATUSES: ReadonlySet<GenerationStatus> = new Set([
  "COMPLETED",
  "FAILED",
  "CANCELLED",
]);

export function isTerminalStatus(status: GenerationStatus): boolean {
  return TERMINAL_STATUSES.has(status);
}

/** Mirrors `VocalGender`. */
export type VocalGender = "female" | "male" | "instrumental";

/** Mirrors `AssetType`. */
export type AssetType = "MASTER" | "PREVIEW" | "STEM";

export interface AudioAsset {
  id: string;
  asset_type: AssetType;
  format: string;
  mime_type: string;
  file_extension: string;
  sample_rate: number;
  bit_depth: number | null;
  bitrate: number | null;
  channels: number;
  duration: number;
  storage_key: string;
  sha256: string;
  file_size: number;
  created_at: string;
}

export interface Generation {
  id: string;
  title: string;
  prompt: string;
  lyrics: string;
  vocal_gender: string;
  duration_requested: number;
  duration_actual: number | null;
  seed: number | null;
  language: string | null;
  instrumental: boolean;
  status: GenerationStatus;
  provider: string | null;
  model_name: string | null;
  model_version: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error_code: string | null;
  error_message: string | null;
  audio_assets: AudioAsset[];
}

export interface GenerationListResponse {
  items: Generation[];
  total: number;
  limit: number;
  offset: number;
}

export interface CreateGenerationInput {
  title: string;
  prompt: string;
  lyrics: string;
  vocal_gender: VocalGender;
  language: string;
  duration: number;
}

export interface CreateGenerationResponse {
  generation_id: string;
  status: GenerationStatus;
}

/**
 * An API failure the UI can present safely.
 *
 * `code` is the backend's machine-readable `ErrorCode` when available.
 * Raw server text is deliberately not surfaced to users.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string | null;

  constructor(message: string, status: number, code: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

/** A fresh Idempotency-Key so a retry never collides with a prior submit. */
export function newIdempotencyKey(): string {
  const cryptoObj = globalThis.crypto;
  if (cryptoObj && typeof cryptoObj.randomUUID === "function") {
    return cryptoObj.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

async function readErrorCode(res: Response): Promise<string | null> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    return typeof body.detail === "string" ? body.detail : null;
  } catch {
    return null;
  }
}

export async function createGeneration(
  input: CreateGenerationInput,
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<CreateGenerationResponse> {
  const res = await fetch(`${API_BASE_URL}/v1/generations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Idempotency-Key": idempotencyKey,
    },
    body: JSON.stringify(input),
    signal,
  });
  if (!res.ok) {
    throw new ApiError(
      `Create generation failed: ${res.status}`,
      res.status,
      await readErrorCode(res),
    );
  }
  return (await res.json()) as CreateGenerationResponse;
}

export async function getGeneration(
  generationId: string,
  signal?: AbortSignal,
): Promise<Generation> {
  const res = await fetch(
    `${API_BASE_URL}/v1/generations/${encodeURIComponent(generationId)}`,
    { cache: "no-store", signal },
  );
  if (!res.ok) {
    throw new ApiError(
      `Get generation failed: ${res.status}`,
      res.status,
      await readErrorCode(res),
    );
  }
  return (await res.json()) as Generation;
}

export async function listGenerations(
  limit = 20,
  offset = 0,
  signal?: AbortSignal,
): Promise<GenerationListResponse> {
  const res = await fetch(
    `${API_BASE_URL}/v1/generations?limit=${limit}&offset=${offset}`,
    { cache: "no-store", signal },
  );
  if (!res.ok) {
    throw new ApiError(
      `List generations failed: ${res.status}`,
      res.status,
      await readErrorCode(res),
    );
  }
  return (await res.json()) as GenerationListResponse;
}

/** Which delivery asset to fetch. */
export type AudioAssetKind = "master" | "preview";

/**
 * URL the browser uses to fetch one delivery asset.
 *
 * Audio is addressed by generation id and asset role; the client never
 * handles a storage key, bucket name, or filesystem path. In production
 * the backend may answer with a redirect to a short-lived signed URL,
 * which the browser follows transparently.
 */
export function getAudioAssetUrl(
  generationId: string,
  asset: AudioAssetKind = "master",
  download = false,
): string {
  const params = new URLSearchParams({ asset });
  if (download) params.set("download", "true");
  return `${API_BASE_URL}/v1/generations/${encodeURIComponent(generationId)}/audio?${params}`;
}

export function findMasterAsset(generation: Generation): AudioAsset | null {
  return generation.audio_assets.find((a) => a.asset_type === "MASTER") ?? null;
}

export function findPreviewAsset(generation: Generation): AudioAsset | null {
  return generation.audio_assets.find((a) => a.asset_type === "PREVIEW") ?? null;
}
