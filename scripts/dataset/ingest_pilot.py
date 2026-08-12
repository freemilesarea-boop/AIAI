#!/usr/bin/env python3
"""Phase 7 Step 1 — real-data ingestion smoke test.

Runs supplied audio through the full gate stack and reports, per track,
exactly why it was accepted or excluded. Nothing is auto-repaired: a
track that fails is recorded with its reason, not quietly fixed.

This script never acquires audio. It reads a directory the operator
has already populated with material they hold rights to.

    uv run python scripts/dataset/ingest_pilot.py --input /path/to/drop --limit 5

Expected drop layout (basename-matched, mirroring upstream's trainer):

    drop/
    ├── track001.wav          # decoded WAV; convert lossy sources first
    ├── track001.lyrics.txt   # exact lyrics, section tags, line breaks
    ├── track001.json         # annotations + rights record
    └── …

`track001.json`:

    {
      "caption": "contemporary Korean R&B, warm Rhodes, restrained vocal",
      "bpm": 88, "keyscale": "F# minor", "timesignature": "4",
      "language": "ko", "genre": "RNB", "subgenre": "k-r&b",
      "vocal_gender": "female",
      "vocal": {
        "vocal_style": "contemporary_krnb", "delivery": "breathy",
        "vibrato_amount": "subtle", "vibrato_character": "natural",
        "pronunciation_style": "modern_standard", "timbre": "airy",
        "genre_vocal_identity": "contemporary korean r&b"
      },
      "rights": {
        "origin_type": "HUMAN_RECORDED",
        "training_rights_status": "CONFIRMED",
        "basis": "ORIGINAL_WORK",
        "source": "commissioned session, LUBER studio",
        "rights_holder": "…", "document_reference": "contract-…",
        "confirmed_on": "2026-08-12",
        "audio_use_confirmed": true, "lyrics_rights_confirmed": true,
        "performer_rights_confirmed": true,
        "commercial_training_allowed": true
      }
    }

Vocal annotations must describe what is actually there. Do not guess a
label to fill a field — an unlabelled track is better than a wrongly
labelled one, because the whole point of the pilot is to move vocal
style in a measured direction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "dataset" / "src"))

from luber_dataset import (  # noqa: E402
    Delivery,
    OriginType,
    PronunciationStyle,
    RightsBasis,
    RightsError,
    RightsRecord,
    TrainingRightsStatus,
    TrainingTrack,
    VibratoAmount,
    VibratoCharacter,
    VocalAnnotation,
    VocalStyle,
    VocalTimbre,
    build_manifest,
    inspect_lyrics,
    inspect_training_audio,
    summarize_quality_grade,
    validate_rights,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rights_from(meta: dict[str, Any]) -> RightsRecord:
    raw = meta.get("rights") or {}
    return RightsRecord(
        origin_type=OriginType(raw.get("origin_type", "UNKNOWN")),
        training_rights_status=TrainingRightsStatus(
            raw.get("training_rights_status", "UNVERIFIED")
        ),
        basis=RightsBasis(raw.get("basis", "NONE")),
        source=str(raw.get("source", "")),
        rights_holder=str(raw.get("rights_holder", "")),
        document_reference=str(raw.get("document_reference", "")),
        confirmed_on=str(raw.get("confirmed_on", "")),
        audio_use_confirmed=bool(raw.get("audio_use_confirmed", False)),
        lyrics_rights_confirmed=bool(raw.get("lyrics_rights_confirmed", False)),
        performer_rights_confirmed=bool(raw.get("performer_rights_confirmed", False)),
        commercial_training_allowed=bool(raw.get("commercial_training_allowed", False)),
        notes=str(raw.get("notes", "")),
    )


def _vocal_from(meta: dict[str, Any]) -> VocalAnnotation | None:
    raw = meta.get("vocal")
    if not raw:
        return None
    return VocalAnnotation(
        vocal_style=VocalStyle(raw["vocal_style"]),
        delivery=Delivery(raw["delivery"]),
        vibrato_amount=VibratoAmount(raw["vibrato_amount"]),
        vibrato_character=VibratoCharacter(raw["vibrato_character"]),
        pronunciation_style=PronunciationStyle(raw["pronunciation_style"]),
        genre_vocal_identity=str(raw.get("genre_vocal_identity", "")),
        timbre=VocalTimbre(raw.get("timbre", "neutral")),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory of supplied audio")
    parser.add_argument("--dataset-id", default="LUBER_TRAINSET_PILOT_V1")
    parser.add_argument("--limit", type=int, default=5, help="Smoke-test a few tracks first")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data" / "trainset" / "pilot_manifest.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.is_dir():
        print(f"ABORT: {args.input} is not a directory", file=sys.stderr)
        return 2

    audio_files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".wav", ".flac"})
    if not audio_files:
        print(f"ABORT: no .wav/.flac found in {args.input}", file=sys.stderr)
        print("Lossy sources must be decoded to WAV before ingestion.", file=sys.stderr)
        return 2
    audio_files = audio_files[: args.limit]

    print(f"ingesting {len(audio_files)} track(s) from {args.input}\n")
    candidates: list[TrainingTrack] = []
    hard_failures = 0

    for audio in audio_files:
        stem = audio.stem
        print(f"── {stem}")
        meta_path = audio.with_suffix(".json")
        lyrics_path = audio.with_name(f"{stem}.lyrics.txt")

        if not meta_path.is_file():
            print("   EXCLUDED: no annotation/rights JSON\n")
            hard_failures += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        rights = _rights_from(meta)
        lyrics_text = lyrics_path.read_text(encoding="utf-8") if lyrics_path.is_file() else ""
        vocal_gender = str(meta.get("vocal_gender", "instrumental"))
        has_vocals = vocal_gender != "instrumental"

        # Rights first: no amount of audio quality makes unlicensed
        # material usable.
        try:
            validate_rights(rights, has_lyrics=bool(lyrics_text.strip()), has_vocals=has_vocals)
            print(f"   rights   : OK ({rights.origin_type} / {rights.basis})")
        except RightsError as exc:
            print(f"   EXCLUDED: rights — {exc}\n")
            hard_failures += 1
            continue

        audio_q = inspect_training_audio(audio)
        print(
            f"   audio    : {audio_q.sample_rate} Hz {audio_q.channels}ch "
            f"{audio_q.duration_seconds:.1f}s peak={audio_q.peak:.3f} "
            f"crest={audio_q.crest_factor_db} dB"
        )
        if audio_q.flags:
            print(f"              flags: {', '.join(audio_q.flags)}")

        lyrics_q = inspect_lyrics(lyrics_text, language=str(meta.get("language", "ko")))
        if lyrics_text.strip():
            print(
                f"   lyrics   : {lyrics_q.line_count} lines, "
                f"{lyrics_q.section_count} sections {lyrics_q.sections}"
            )
            if lyrics_q.flags:
                print(f"              flags: {', '.join(lyrics_q.flags)}")
        elif has_vocals:
            print("              flags: vocal track supplied without lyrics")

        candidates.append(
            TrainingTrack(
                track_id=stem,
                source=rights.source,
                rights=rights,
                audio_sha256=sha256_file(audio),
                duration_seconds=audio_q.duration_seconds,
                sample_rate=audio_q.sample_rate,
                channels=audio_q.channels,
                language=str(meta.get("language", "ko")),
                genre=str(meta.get("genre", "")),
                subgenre=str(meta.get("subgenre", "")),
                vocal_gender=vocal_gender,
                lyrics_available=bool(lyrics_text.strip()),
                bpm=meta.get("bpm"),
                key_scale=meta.get("keyscale"),
                time_signature=str(meta.get("timesignature", "")) or None,
                production_style=str(meta.get("production_style", "")),
                instrumentation=list(meta.get("instrumentation", [])),
                vocal=_vocal_from(meta),
                quality_grade=summarize_quality_grade(audio_q.flags, lyrics_q.flags)
                if lyrics_text.strip()
                else summarize_quality_grade(audio_q.flags, []),
                audio_quality_flags=audio_q.flags,
                lyrics_qa_flags=lyrics_q.flags if lyrics_text.strip() else [],
                caption=str(meta.get("caption", "")),
            )
        )
        print()

    manifest = build_manifest(
        args.dataset_id,
        candidates,
        notes="Phase 7 pilot ingestion smoke test. Real supplied audio only.",
    )

    print("=" * 62)
    print(f"accepted : {manifest.track_count}")
    print(f"excluded : {len(manifest.exclusions) + hard_failures}")
    for exclusion in manifest.exclusions:
        print(f"   {exclusion.track_id}: {exclusion.reason} — {exclusion.detail}")
    if manifest.track_count:
        print(f"hash     : {manifest.content_hash()}")
        print(f"styles   : {manifest.style_distribution()}")
        print(f"languages: {manifest.language_distribution()}")
        print(f"duration : {manifest.total_duration_seconds / 60:.1f} min")
        manifest.write(args.out)
        print(f"manifest : {args.out}")
    else:
        print("\nNo track passed. REAL_DATA_PIPELINE_PASS not achieved.")
        return 1

    print("\nREAL_DATA_PIPELINE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
