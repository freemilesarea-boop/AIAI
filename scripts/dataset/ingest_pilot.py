#!/usr/bin/env python3
"""Phase 7 Step 1 — real-data ingestion smoke test.

Runs supplied audio through the full gate stack and reports, per track,
exactly why it was accepted or excluded. Nothing is auto-repaired: a
track that fails is recorded with its reason, not quietly fixed.

This script never acquires audio. It reads a directory the operator
has already populated with material they hold rights to.

    uv run python scripts/dataset/ingest_pilot.py --input /path/to/drop --limit 5

Phase 35B added a second, narrower mode for material an operator has
authorised as a whole directory rather than annotated track by track:

    uv run python scripts/dataset/ingest_pilot.py \
        --input ~/Desktop/LUBER_TRAINING_DATA --recursive \
        --operator-authorized-scope '~/Desktop/LUBER_TRAINING_DATA/**' \
        --operator 'the operator' --limit 0

In that mode the rights record is synthesised from the authorisation
itself — basis `OPERATOR_AUTHORIZED_SCOPE`, with the source, the scope
and the date recorded — and it claims nothing else. No contract, no
licence and no performer agreement is asserted, because none was shown.
A per-track JSON, where one exists, still wins: an operator who
annotated a track meant what they wrote.

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
from datetime import UTC, datetime
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


#: What an operator-authorised record says, in one place so the wording
#: cannot drift between the manifest and the report.
OPERATOR_AUTHORIZATION_SOURCE = "OPERATOR_EXPLICIT_AUTHORIZATION"

OPERATOR_AUTHORIZATION_NOTES = (
    "The operator explicitly authorised this source directory for LUBER model training. "
    "That authorisation is the entire evidence: no contract, licence, publisher clearance "
    "or performer agreement was produced or verified, and none is claimed here."
)


def _operator_rights(*, scope: str, operator: str, recorded_at: str, group: str) -> RightsRecord:
    """A rights record whose only evidence is the operator's own decision.

    Every field says what it means. `origin_type` stays UNKNOWN because
    nobody established how the audio was made — a directory of files
    does not say whether a human or a model produced them, and guessing
    would be the fabrication this whole module exists to prevent.

    The confirmation booleans are set from the operator's authorisation
    and from nothing else. They record that the operator authorised the
    material for training; they are not a claim that a third party
    signed anything, and the basis and notes say so on every record.
    """
    return RightsRecord(
        origin_type=OriginType.UNKNOWN,
        training_rights_status=TrainingRightsStatus.CONFIRMED,
        basis=RightsBasis.OPERATOR_AUTHORIZED_SCOPE,
        source=f"operator-authorised directory, group {group!r}",
        rights_holder=operator,
        document_reference=f"operator authorisation of {scope}",
        confirmed_on=recorded_at,
        audio_use_confirmed=True,
        # False because nobody produced a performer agreement or a
        # publisher clearance. The operator authorised the works; that
        # is a different, weaker thing, and the record says which one
        # it has rather than rounding it up to the stronger claim.
        lyrics_rights_confirmed=False,
        performer_rights_confirmed=False,
        commercial_training_allowed=True,
        notes=OPERATOR_AUTHORIZATION_NOTES,
        authorization_source=OPERATOR_AUTHORIZATION_SOURCE,
        authorization_scope=scope,
        authorization_recorded_at=recorded_at,
    )


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
    parser.add_argument(
        "--limit", type=int, default=5, help="Smoke-test a few tracks first; 0 means all"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="descend into subdirectories, keeping the first level as a source group",
    )
    parser.add_argument(
        "--operator-authorized-scope",
        help=(
            "record an operator authorisation covering this scope, and synthesise a rights "
            "record from it for any track without its own annotation JSON"
        ),
    )
    parser.add_argument(
        "--operator", default="", help="who authorised the scope; recorded as rights_holder"
    )
    parser.add_argument(
        "--authorized-on",
        default="",
        help=(
            "ISO date the authorisation was recorded; defaults to today. Pass the original "
            "date to reproduce an earlier manifest digest byte for byte"
        ),
    )
    parser.add_argument(
        "--extensions",
        default=".wav",
        help="comma-separated audio extensions to ingest (default: .wav)",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data" / "trainset" / "pilot_manifest.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.is_dir():
        print(f"ABORT: {args.input} is not a directory", file=sys.stderr)
        return 2

    wanted = {item.strip().lower() for item in args.extensions.split(",") if item.strip()}
    walker = args.input.rglob("*") if args.recursive else args.input.iterdir()
    audio_files = sorted(
        path
        for path in walker
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in wanted
    )
    if not audio_files:
        print(f"ABORT: no {'/'.join(sorted(wanted))} found in {args.input}", file=sys.stderr)
        print("Lossy sources must be decoded to WAV before ingestion.", file=sys.stderr)
        return 2
    if args.limit:
        audio_files = audio_files[: args.limit]

    authorization_recorded_at = args.authorized_on or datetime.now(UTC).date().isoformat()
    if args.operator_authorized_scope:
        print(f"operator authorisation: {args.operator_authorized_scope}")
        print(f"recorded on           : {authorization_recorded_at}")
        print(
            "evidence              : the operator's own authorisation. No contract, "
            "licence or performer agreement is claimed.\n"
        )

    # Byte-identical files are one piece of audio however many paths
    # point at it. Ingesting both would weight it twice in training and
    # make the manifest digest depend on how the operator arranged
    # folders.
    seen_digests: dict[str, str] = {}

    print(f"ingesting {len(audio_files)} track(s) from {args.input}\n")
    candidates: list[TrainingTrack] = []
    source_paths: dict[str, str] = {}
    hard_failures = 0

    for audio in audio_files:
        stem = audio.stem
        relative = audio.relative_to(args.input)
        group = relative.parts[0] if len(relative.parts) > 1 else args.input.name
        # A stable, path-independent id. Filenames are operator-chosen
        # text that ends up in reports; a digest prefix does not.
        digest = sha256_file(audio)
        track_id = f"{digest[:16]}"
        print(f"── {track_id}  (group {group})")

        if digest in seen_digests:
            print(f"   EXCLUDED: byte-identical to {seen_digests[digest]}\n")
            hard_failures += 1
            continue
        seen_digests[digest] = track_id

        meta_path = audio.with_suffix(".json")
        lyrics_path = audio.with_name(f"{stem}.lyrics.txt")
        meta: dict[str, Any] = {}

        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rights = _rights_from(meta)
        elif args.operator_authorized_scope:
            rights = _operator_rights(
                scope=args.operator_authorized_scope,
                operator=args.operator or "operator",
                recorded_at=authorization_recorded_at,
                group=group,
            )
        else:
            print("   EXCLUDED: no annotation/rights JSON\n")
            hard_failures += 1
            continue
        lyrics_text = lyrics_path.read_text(encoding="utf-8") if lyrics_path.is_file() else ""
        # Absent metadata means unknown. Defaulting to "instrumental"
        # would assert a musical fact nobody established and would
        # silently waive the performer-rights check.
        vocal_gender = str(meta.get("vocal_gender", "unknown"))
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
                track_id=track_id,
                source=rights.source,
                rights=rights,
                audio_sha256=digest,
                duration_seconds=audio_q.duration_seconds,
                sample_rate=audio_q.sample_rate,
                channels=audio_q.channels,
                # "ko" is the drop-format default for annotated tracks.
                # An unannotated track has no stated language, and
                # guessing one would put a fact in the manifest that
                # nobody established.
                language=str(meta.get("language", "ko" if meta else "unknown")),
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
                source_group=group,
            )
        )
        source_paths[track_id] = str(audio)
        print()

    manifest = build_manifest(
        args.dataset_id,
        candidates,
        notes=(
            "Operator-authorised ingestion. Real supplied audio only."
            if args.operator_authorized_scope
            else "Phase 7 pilot ingestion smoke test. Real supplied audio only."
        ),
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
        # The manifest deliberately carries no filesystem paths, so a
        # later stage still needs to find the audio. That map is
        # machine-local and private: it lives beside the manifest under
        # the gitignored data root and is never committed.
        path_map = args.out.with_suffix(".paths.json")
        path_map.write_text(
            json.dumps(
                {
                    "note": "machine-local source paths; never commit",
                    "authorization_scope": args.operator_authorized_scope or "",
                    "tracks": source_paths,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"path map : {path_map} (local only)")
    else:
        print("\nNo track passed. REAL_DATA_PIPELINE_PASS not achieved.")
        return 1

    print("\nREAL_DATA_PIPELINE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
