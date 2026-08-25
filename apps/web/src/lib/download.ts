/**
 * Download naming and availability, in one place.
 *
 * The **server** decides the filename that actually lands on disk — it
 * sets `Content-Disposition`, and a client-side `download` attribute
 * cannot override a server-supplied filename anyway. What lives here is
 * the client's side of the contract: the same name, so the UI can tell
 * the user what they are about to get, and the rule for which formats
 * are offered.
 *
 * Keep `downloadFilename` in step with `build_download_filename` in
 * `apps/api/src/luber_api/routes/generations.py`.
 */

import { findMasterAsset, findPreviewAsset, type Generation } from "@/lib/api";

export const DOWNLOAD_FILENAME_PREFIX = "BOORDA - ";
export const DOWNLOAD_TITLE_MAX_LENGTH = 60;

/** Characters no mainstream filesystem accepts, plus control characters. */
const UNSAFE = /[\\/:*?"<>|\x00-\x1f\x7f]+/g;

/**
 * `BOORDA - Midnight Window.wav`.
 *
 * Unicode is kept: this product's main audience writes Korean titles,
 * and a Korean track that downloads as `luber-track-1a2b3c4d.wav` is not
 * recognisable in a downloads folder.
 */
export function downloadFilename(
  title: string,
  generationId: string,
  extension = "wav",
): string {
  const cleaned = title
    .replace(UNSAFE, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\.{2,}/g, ".")
    .replace(/^[.\s]+|[.\s]+$/g, "")
    .slice(0, DOWNLOAD_TITLE_MAX_LENGTH)
    .replace(/^[.\s]+|[.\s]+$/g, "");
  const stem = cleaned || `track-${generationId.replace(/-/g, "").slice(0, 8)}`;
  return `${DOWNLOAD_FILENAME_PREFIX}${stem}.${extension}`;
}

export interface DownloadOption {
  kind: "master" | "preview";
  label: string;
  /** Shown beside the label so nothing is mistaken for the master. */
  hint: string;
  filename: string;
}

/**
 * The formats this generation can actually be downloaded as.
 *
 * Driven entirely by the assets the backend produced. Nothing is offered
 * speculatively, and the MP3 is never described as a master — it is a
 * lossy preview that happens to already exist, which is the only reason
 * it is offered at all.
 */
export function downloadOptions(generation: Generation): DownloadOption[] {
  const options: DownloadOption[] = [];
  if (findMasterAsset(generation)) {
    options.push({
      kind: "master",
      label: "Download WAV",
      hint: "Master · lossless",
      filename: downloadFilename(generation.title, generation.id, "wav"),
    });
  }
  if (findPreviewAsset(generation)) {
    options.push({
      kind: "preview",
      label: "Download MP3",
      hint: "Preview · compressed",
      filename: downloadFilename(generation.title, generation.id, "mp3"),
    });
  }
  return options;
}
