#!/usr/bin/env python3
"""Phase 38 — measure what the authorised library actually sounds like.

The library carries two folder names and nothing else. Which tracks have
live high end, a steady pulse or a busy arrangement — and where inside
each track those qualities sit — is not recorded anywhere, so it is
measured here and written down.

Per track and per candidate window: high-frequency energy share,
spectral centroid, high-band RMS, transient and onset density, beat
stability, tempo and its consistency, a drum/bass alignment proxy, and a
layer-density proxy. Onset times are kept too, because a window chooser
that wants to start on a beat needs them.

Read-only over the source. Nothing is judged here; tiering is a separate
step so a measurement and an opinion never get confused.

    uv run python scripts/dataset/analyse_library_audio.py \
        --manifest data/trainset/authorized_library_manifest.json \
        --output data/trainset/exp38/library_features.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "dataset" / "src"))

from luber_dataset.audio_features import analyse_track  # noqa: E402
from luber_dataset.windows import (  # noqa: E402
    DEFAULT_WINDOW_FRAMES,
    LATENT_FRAMES_PER_SECOND,
)

#: Candidate window starts are placed on this grid before any
#: beat-aware refinement. Fine enough to find a busy section, coarse
#: enough that a four-minute track has a dozen candidates rather than a
#: thousand.
CANDIDATE_GRID_SECONDS = 5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES)
    parser.add_argument("--grid", type=float, default=CANDIDATE_GRID_SECONDS)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    library = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths_file = args.paths or args.manifest.with_suffix(".paths.json")
    sources: dict[str, str] = json.loads(paths_file.read_text(encoding="utf-8"))["tracks"]

    window_seconds = args.window_frames / LATENT_FRAMES_PER_SECOND
    records = []
    unchanged = True
    started = time.perf_counter()

    for index, track in enumerate(library["tracks"], start=1):
        track_id = track["track_id"]
        origin = Path(sources[track_id])
        before = sha256_file(origin)
        if before != track["audio_sha256"]:
            print(f"ABORT: {track_id} no longer matches the manifest", file=sys.stderr)
            return 2

        duration = float(track["duration_seconds"])
        last_start = duration - window_seconds
        starts = (
            tuple(
                float(round(value * args.grid, 3))
                for value in range(int(last_start // args.grid) + 1)
            )
            if last_start >= 0
            else ()
        )
        analysis = analyse_track(
            origin,
            track_id=track_id,
            audio_sha256=track["audio_sha256"],
            source_group=str(track.get("source_group", "")),
            window_seconds=window_seconds if starts else None,
            window_starts=starts,
        )
        # Onset times are what a beat-aware chooser snaps to; they are
        # the one bulky field, so they are kept and the report says so.
        payload = analysis.to_dict()
        payload["onset_times"] = [round(value, 4) for value in analysis.onset_times]
        payload["candidate_window_count"] = len(starts)
        records.append(payload)
        if sha256_file(origin) != before:
            unchanged = False
            print(f"   !! source changed while reading: {track_id}")
        if index % 20 == 0 or index == len(library["tracks"]):
            print(
                f"  analysed {index}/{len(library['tracks'])} "
                f"({time.perf_counter() - started:.0f}s)"
            )

    document = {
        "schema_version": "luber-library-features/1",
        "library_content_hash": library.get("content_hash", ""),
        "window_frames": args.window_frames,
        "window_seconds": window_seconds,
        "candidate_grid_seconds": args.grid,
        "track_count": len(records),
        "source_unchanged": unchanged,
        "note": (
            "Measurements, not judgements. Onset density is spectral-flux peaks per "
            "second and not a transcription; drum_bass_alignment is a correlation "
            "between two band-limited onset envelopes and is a proxy; layer_density is "
            "spectral entropy and counts no instruments."
        ),
        "tracks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nwritten : {args.output}")
    print(f"sources : {'unchanged' if unchanged else 'CHANGED — investigate'}")
    return 0 if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
