#!/usr/bin/env python3
"""Ingest approved candidates straight from the approval manifest.

Unlike `ingest_pilot.py`, this expects no sidecar files next to the
audio: the rights record lives in the approval manifest and the audio is
read where it already sits. That matters because the operator's source
folders must not be written to — no `.json`, no `.lyrics.txt`, nothing.

Only candidates whose recorded decision is CONFIRM_TRAINING_RIGHTS are
considered. Everything else is skipped with a reason. Nothing here can
promote a track: the rights gate re-validates every record, so a
manifest edited by hand still cannot smuggle an unconfirmed track
through.

    uv run python scripts/dataset/ingest_from_manifest.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "dataset" / "src"))

from luber_dataset import (  # noqa: E402
    OriginType,
    QualityGrade,
    RightsBasis,
    RightsError,
    RightsRecord,
    TrainingRightsStatus,
    TrainingTrack,
    build_manifest,
    inspect_training_audio,
    validate_rights,
)
from luber_dataset.discovery import sha256_file  # noqa: E402


def _rights_from(raw: dict[str, Any]) -> RightsRecord:
    return RightsRecord(
        origin_type=OriginType(raw.get("origin_type", "UNKNOWN")),
        training_rights_status=TrainingRightsStatus(
            raw.get("training_rights_status", "UNVERIFIED")
        ),
        basis=RightsBasis(raw.get("basis", "NONE")),
        source=str(raw.get("source") or raw.get("document_reference", "")),
        rights_holder=str(raw.get("rights_holder", "")),
        document_reference=str(raw.get("document_reference", "")),
        confirmed_on=str(raw.get("confirmed_on", "")),
        audio_use_confirmed=bool(raw.get("audio_use_confirmed", False)),
        lyrics_rights_confirmed=bool(raw.get("lyrics_rights_confirmed", False)),
        performer_rights_confirmed=bool(raw.get("performer_rights_confirmed", False)),
        commercial_training_allowed=bool(raw.get("commercial_training_allowed", False)),
        notes=str(raw.get("notes", "")),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path.home() / ".luber" / "rights_approval_manifest.json"
    )
    parser.add_argument(
        "--catalog", type=Path, default=Path.home() / ".luber" / "discovery_catalog.json"
    )
    parser.add_argument("--group", default=None, help="Restrict to one directory group")
    parser.add_argument("--dataset-id", default="LUBER_TRAINSET_PILOT_V1")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path.home() / ".luber" / "pilot_manifest.json")
    parser.add_argument(
        "--summary", type=Path, default=REPO_ROOT / "data" / "pilot_manifest_summary.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.manifest, args.catalog):
        if not path.is_file():
            print(f"ABORT: missing {path}", file=sys.stderr)
            return 2

    approval = json.loads(args.manifest.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    # The approval manifest holds sanitized names; the catalog holds the
    # real locations. Join on the content hash, which cannot drift.
    path_by_hash = {entry["sha256"]: entry["absolute_path"] for entry in catalog}

    approved = [c for c in approval["candidates"] if c.get("decision") == "CONFIRM_TRAINING_RIGHTS"]
    if args.group:
        target = unicodedata.normalize("NFC", args.group)
        approved = [
            c
            for c in approved
            if unicodedata.normalize("NFC", str(Path(c["sanitized_relative_name"]).parent))
            == target
        ]
    approved.sort(key=lambda c: c["sanitized_relative_name"])
    selected = approved[: args.limit]

    print(f"approved candidates : {len(approved)}")
    print(f"ingesting           : {len(selected)}\n")
    if not selected:
        print("Nothing approved to ingest.", file=sys.stderr)
        return 1

    candidates: list[TrainingTrack] = []
    failures = 0

    for candidate in selected:
        name = candidate["sanitized_relative_name"]
        print(f"── {name}")
        source_path = path_by_hash.get(candidate["audio_sha256"])
        if source_path is None or not Path(source_path).is_file():
            print("   EXCLUDED: source file not found for this hash\n")
            failures += 1
            continue

        audio = Path(source_path)
        rights = _rights_from(candidate.get("rights_record") or {})
        # Lyrics were discovered, never invented. None here means none.
        has_lyrics = bool(candidate.get("adjacent_lyrics"))

        try:
            validate_rights(rights, has_lyrics=has_lyrics, has_vocals=True)
            print(f"   rights   : CONFIRMED ({rights.origin_type} / {rights.basis})")
        except RightsError as exc:
            print(f"   EXCLUDED: rights — {exc}\n")
            failures += 1
            continue

        # Re-measure from the file itself rather than trusting the
        # catalog: the gate should see current bytes.
        quality = inspect_training_audio(audio)
        print(
            f"   audio    : {quality.sample_rate} Hz {quality.channels}ch "
            f"{quality.duration_seconds:.1f}s peak={quality.peak:.3f} "
            f"crest={quality.crest_factor_db} dB"
        )
        if quality.flags:
            print(f"              flags: {', '.join(quality.flags)}")

        digest = sha256_file(audio)
        if digest != candidate["audio_sha256"]:
            print("   EXCLUDED: file changed since discovery (hash mismatch)\n")
            failures += 1
            continue
        print(f"   sha256   : {digest[:16]}… verified against catalog")
        print(f"   lyrics   : {'present' if has_lyrics else 'LYRICS_MISSING'}")

        candidates.append(
            TrainingTrack(
                track_id=Path(name).stem,
                source=rights.document_reference,
                rights=rights,
                audio_sha256=digest,
                duration_seconds=quality.duration_seconds,
                sample_rate=quality.sample_rate,
                channels=quality.channels,
                # Not asserted: no metadata was supplied and guessing
                # language or vocal gender from a filename would be
                # inventing training labels.
                language="unknown",
                genre="",
                subgenre="",
                vocal_gender="unknown",
                lyrics_available=has_lyrics,
                bpm=None,
                key_scale=None,
                time_signature=None,
                vocal=None,
                quality_grade=(QualityGrade.REJECTED if quality.flags else QualityGrade.GOOD),
                audio_quality_flags=quality.flags,
                lyrics_qa_flags=[],
                caption="",
                notes="ingested from approval manifest; source file read in place",
            )
        )
        print()

    manifest = build_manifest(
        args.dataset_id,
        candidates,
        notes=(
            "Phase 7 pilot. Real operator-supplied audio, rights confirmed by operator "
            "attestation. Source files were read in place and never modified."
        ),
    )

    print("=" * 64)
    print(f"accepted : {manifest.track_count}")
    print(f"excluded : {len(manifest.exclusions) + failures}")
    for exclusion in manifest.exclusions:
        print(f"   {exclusion.track_id}: {exclusion.reason} — {exclusion.detail}")

    if not manifest.track_count:
        print("\nNo track passed. REAL_DATA_PIPELINE_PASS not achieved.")
        return 1

    print(f"hash     : {manifest.content_hash()}")
    print(f"duration : {manifest.total_duration_seconds / 60:.1f} min")
    manifest.write(args.out)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(
            {
                "dataset_id": manifest.dataset_id,
                "track_count": manifest.track_count,
                "content_hash": manifest.content_hash(),
                "total_duration_seconds": round(manifest.total_duration_seconds, 2),
                "exclusions": [e.to_dict() for e in manifest.exclusions],
                "lyrics_available": sum(1 for t in manifest.tracks if t.lyrics_available),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"manifest : {args.out}")
    print(f"summary  : {args.summary}")
    print("\nREAL_DATA_PIPELINE_PASS")
    print("No source audio was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
