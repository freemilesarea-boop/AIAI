#!/usr/bin/env python
"""Build a candidate training manifest from an explicitly named directory.

Three rules shape this tool, and all three are about not doing something
clever:

* **It only looks where it is told.** There is no default directory and
  no recursive sweep of a home folder. Scanning somebody's music library
  because it happened to be reachable is exactly the failure this design
  exists to prevent, so the path is a required argument.
* **It never writes to the source.** Files are opened for reading and
  hashed. Nothing is moved, renamed, converted or deleted, and there is
  no flag that changes that.
* **It cannot make anything trainable.** Rights are supplied by the
  operator on the command line, default to ``UNKNOWN``, and ``UNKNOWN``
  never reaches TRAIN. Everything it produces is a *candidate* manifest:
  a human decides tier and split afterwards.

    ingest.py --dir data/benchmark-audio --name p20_smoke            # dry run
    ingest.py --dir data/benchmark-audio --name p20_smoke \
              --rights OWNED --source-type LUBER_GENERATED --write out.json

Dry run is the default. ``--write`` is what makes it produce a file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "schemas" / "src"))

from luber_schemas.dataset import (  # noqa: E402
    DataRights,
    DatasetItem,
    DatasetManifest,
    DataSplit,
    QualityTier,
    SourceType,
)

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}
CHUNK = 1024 * 1024


@dataclass
class ScanReport:
    discovered: int = 0
    valid: int = 0
    invalid: list[tuple[str, str]] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)
    rights_unknown: int = 0
    eligible: int = 0
    quarantined: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "discovered": self.discovered,
            "valid": self.valid,
            "invalid": len(self.invalid),
            "duplicates": len(self.duplicates),
            "rights_unknown": self.rights_unknown,
            "eligible_for_training": self.eligible,
            "quarantined": self.quarantined,
        }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def pcm_sha256_of(path: Path) -> str | None:
    """Hash of the decoded samples, so a re-encode is still a duplicate.

    Only attempted for WAV, where it costs a read and no dependency.
    Anything else returns ``None`` rather than a guess — see the note on
    near-duplicates in the docstring of :func:`find_duplicates`.
    """
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            digest = hashlib.sha256()
            while frames := handle.readframes(65536):
                digest.update(frames)
            return digest.hexdigest()
    except (wave.Error, OSError):
        return None


def probe(path: Path) -> dict[str, float | int] | None:
    """Technical metadata, or ``None`` if the file will not decode.

    ffprobe ships with the ffmpeg this project already requires, so this
    adds no dependency. A file that cannot be probed is reported as
    invalid rather than guessed at.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        return {
            "sample_rate": int(stream["sample_rate"]),
            "channels": int(stream["channels"]),
            "duration_seconds": float(payload["format"]["duration"]),
        }
    except (KeyError, IndexError, ValueError, json.JSONDecodeError):
        return None


def find_duplicates(items: list[DatasetItem]) -> list[tuple[str, str]]:
    """Exact duplicates only — by file bytes, and by decoded samples.

    This detects the same file under two names, and a WAV re-encoded or
    re-containered without altering the samples. It does **not** detect
    a trimmed, resampled or lossily re-encoded near-duplicate: that
    needs perceptual fingerprinting, which is not implemented here and
    is not claimed anywhere in the output.
    """
    seen: dict[str, str] = {}
    duplicates: list[tuple[str, str]] = []
    for item in items:
        for digest in (item.sha256, item.pcm_sha256):
            if not digest:
                continue
            if digest in seen and seen[digest] != item.item_id:
                duplicates.append((item.item_id, seen[digest]))
                break
            seen.setdefault(digest, item.item_id)
    return duplicates


def scan(
    directory: Path,
    *,
    rights: DataRights,
    source_type: SourceType,
    rights_note: str,
) -> tuple[list[DatasetItem], ScanReport]:
    report = ScanReport()
    items: list[DatasetItem] = []
    now = datetime.now(UTC).isoformat()

    for path in sorted(p for p in directory.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES):
        report.discovered += 1
        relative = path.relative_to(directory).as_posix()
        metadata = probe(path)
        if metadata is None:
            report.invalid.append((relative, "would not decode"))
            continue
        if metadata["channels"] not in (1, 2):
            report.invalid.append((relative, f"{metadata['channels']} channels"))
            continue
        report.valid += 1
        items.append(
            DatasetItem(
                item_id=relative,
                audio_path=relative,
                sha256=sha256_of(path),
                pcm_sha256=pcm_sha256_of(path),
                source_type=source_type,
                source_identifier=relative,
                rights=rights,
                rights_note=rights_note,
                ingested_at=now,
                duration_seconds=float(metadata["duration_seconds"]),
                sample_rate=int(metadata["sample_rate"]),
                channels=int(metadata["channels"]),
                # Tier and split are human decisions. Everything arrives
                # as REJECT / EVALUATION_ONLY and is promoted on purpose.
                quality_tier=QualityTier.REJECT,
                split=DataSplit.EVALUATION_ONLY,
                notes="candidate; tier and split not yet reviewed",
            )
        )

    report.duplicates = find_duplicates(items)
    report.rights_unknown = sum(1 for i in items if i.rights is DataRights.UNKNOWN)
    report.eligible = sum(
        1
        for i in items
        if i.rights
        in {
            DataRights.OWNED,
            DataRights.LICENSED_FOR_TRAINING,
            DataRights.PUBLIC_DOMAIN,
            DataRights.AI_GENERATED_ALLOWED,
        }
    )
    report.quarantined = len(items) - report.eligible
    return items, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="directory to scan (required; no default)")
    parser.add_argument("--name", required=True, help="dataset name")
    parser.add_argument(
        "--rights",
        default=DataRights.UNKNOWN.value,
        choices=[r.value for r in DataRights],
        help="provenance of everything in this directory (default: UNKNOWN)",
    )
    parser.add_argument(
        "--source-type",
        default=SourceType.OTHER.value,
        choices=[s.value for s in SourceType],
    )
    parser.add_argument("--rights-note", default="", help="where the permission comes from")
    parser.add_argument("--write", help="write the manifest here (otherwise dry run)")
    args = parser.parse_args()

    directory = Path(args.dir).expanduser().resolve()
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        return 1

    rights = DataRights(args.rights)
    items, report = scan(
        directory,
        rights=rights,
        source_type=SourceType(args.source_type),
        rights_note=args.rights_note,
    )

    print(f"scanned {directory}")
    for key, value in report.as_dict().items():
        print(f"  {key:<22} {value}")
    for name, reason in report.invalid[:10]:
        print(f"  invalid: {name} — {reason}")
    for dupe, original in report.duplicates[:10]:
        print(f"  duplicate: {dupe} == {original}")
    if rights is DataRights.UNKNOWN and items:
        print("\n  Rights are UNKNOWN, so nothing here can enter a training manifest.")
        print("  Pass --rights only when the provenance is actually established.")
    print("\n  Near-duplicate (trimmed / re-encoded) detection is NOT implemented;")
    print("  only exact file and decoded-PCM matches are reported.")

    if not args.write:
        print("\ndry run — no manifest written (pass --write PATH to produce one)")
        return 0

    manifest = DatasetManifest(
        dataset_name=args.name,
        created_at=datetime.now(UTC).isoformat(),
        dataset_root_note="paths are relative to the scanned directory",
        items=items,
    )
    target = Path(args.write)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.model_dump_json(indent=2))
    print(
        f"\nwrote {target} — {len(items)} candidate items, "
        f"{len(manifest.trainable_items())} trainable"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
