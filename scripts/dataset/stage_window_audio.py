#!/usr/bin/env python3
"""Phase 37 — cut each planned window out of its source track.

Windows are materialised as audio, not as tensor slices, so the real
ACE-Step preprocessing pipeline runs over each one exactly as it would
over a whole track. Every clip is the same length, so every tensor comes
out the same shape and the Phase 36 allocator problem cannot return.

Reading and writing is done with the standard library's `wave` module on
frame boundaries. The authorised source is 16-bit PCM at 48 kHz, so a
window is an exact copy of a byte range — no decode, no resample, no
re-encode, and nothing about the audio changes on the way through.

The source root is opened read-only and re-hashed afterwards.

    uv run python scripts/dataset/stage_window_audio.py \
        --experiment data/trainset/exp37 --output data/trainset/exp37/audio
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "dataset" / "src"))

from luber_dataset.factory.provenance import (  # noqa: E402
    RightsStatus,
    SourceType,
    TrainingPermission,
)
from luber_dataset.windows import LATENT_FRAMES_PER_SECOND  # noqa: E402

OPERATOR_AUTHORIZATION_NOTES = (
    "The operator explicitly authorised this source directory for LUBER model training. "
    "That authorisation is the entire evidence: no contract, licence, publisher clearance "
    "or performer agreement was produced or verified, and none is claimed here."
)


def sidecar_for(window: dict[str, Any], *, scope: str) -> dict[str, Any]:
    """Operator metadata for one window.

    A window inherits its track's authorisation and adds nothing to it —
    slicing a recording does not create rights it did not have, and the
    record names which part of which track this is so the inheritance is
    checkable rather than assumed.
    """
    group = str(window["source_group"])
    return {
        "source": f"operator-authorised directory, group {group!r}",
        "source_type": SourceType.UNKNOWN.value,
        "rights_status": RightsStatus.OPERATOR_AUTHORIZED.value,
        "commercial_training_allowed": TrainingPermission.TRUE.value.lower(),
        "genre": group.lower(),
        "notes": (
            f"{OPERATOR_AUTHORIZATION_NOTES} Authorised scope: {scope}. "
            f"Phase 37 window {window['window_index'] + 1} of {window['window_count']} "
            f"({window['position']}) from source track {window['track_id']}, "
            f"{window['start_seconds']:.1f}s to {window['end_seconds']:.1f}s."
        ),
    }


#: The trainer's VAE consumes 48 kHz. One latent frame is this many
#: audio samples, which is what makes a window an exact frame range.
SAMPLES_PER_LATENT_FRAME = 1920


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cut_window(source: Path, target: Path, *, start_frame: int, frames: int) -> int:
    """Copy one frame range into its own WAV. Returns samples written."""
    with wave.open(str(source), "rb") as reader:
        start_sample = start_frame * SAMPLES_PER_LATENT_FRAME
        want = frames * SAMPLES_PER_LATENT_FRAME
        available = reader.getnframes() - start_sample
        if available < want:
            raise ValueError(
                f"{source.name}: window at frame {start_frame} needs {want} samples and "
                f"only {available} remain; a window must never run past its source"
            )
        reader.setpos(start_sample)
        payload = reader.readframes(want)
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as writer:
            writer.setnchannels(reader.getnchannels())
            writer.setsampwidth(reader.getsampwidth())
            writer.setframerate(reader.getframerate())
            writer.writeframes(payload)
    return want


def _sample_count(path: Path) -> int:
    """Frames already in a clip, so an existing correct cut is not redone."""
    try:
        with wave.open(str(path), "rb") as reader:
            return reader.getnframes()
    except (OSError, wave.Error):
        return -1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="train,validation,evaluation")
    parser.add_argument("--scope", default="", help="authorised scope, for the sidecars")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources: dict[str, str] = json.loads(
        (args.experiment / "splits.paths.json").read_text(encoding="utf-8")
    )["tracks"]

    scope = args.scope
    if not scope:
        library_paths = Path("data/trainset/authorized_library_manifest.paths.json")
        if library_paths.is_file():
            scope = json.loads(library_paths.read_text(encoding="utf-8")).get(
                "authorization_scope", ""
            )

    unchanged = True
    written: dict[str, int] = {}
    for split in (name.strip() for name in args.splits.split(",") if name.strip()):
        manifest = json.loads(
            (args.experiment / f"windows_{split}.json").read_text(encoding="utf-8")
        )
        destination = args.output / split
        destination.mkdir(parents=True, exist_ok=True)
        count = 0
        for window in manifest["windows"]:
            origin = Path(sources[window["track_id"]])
            before = sha256_file(origin)
            if before != window["audio_sha256"]:
                print(
                    f"ABORT: {window['track_id']} no longer matches the manifest digest",
                    file=sys.stderr,
                )
                return 2
            clip = destination / f"{window['window_id']}.wav"
            expected = window["latent_frames"] * SAMPLES_PER_LATENT_FRAME
            if not (clip.is_file() and _sample_count(clip) == expected):
                cut_window(
                    origin,
                    clip,
                    start_frame=window["start_frame"],
                    frames=window["latent_frames"],
                )
            clip.with_suffix(".json").write_text(
                json.dumps(sidecar_for(window, scope=scope), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            if sha256_file(origin) != before:
                unchanged = False
                print(f"   !! source changed while reading: {window['track_id']}")
            count += 1
        written[split] = count
        print(
            f"{manifest['split']:<11}: {count} window(s) cut from "
            f"{manifest['track_count']} track(s) -> {destination}"
        )

    seconds = manifest["window_frames"] / LATENT_FRAMES_PER_SECOND
    print(f"\nwindow length: {manifest['window_frames']} frames = {seconds:.1f} s")
    print(f"sources      : {'unchanged' if unchanged else 'CHANGED — investigate'}")
    (args.output / "staging_report.json").write_text(
        json.dumps(
            {"written": written, "source_unchanged": unchanged, "window_seconds": seconds},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
