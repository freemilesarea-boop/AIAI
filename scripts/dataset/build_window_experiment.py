#!/usr/bin/env python3
"""Phase 37 — split tracks, then window inside each split.

The order is the whole point. Splitting tracks first and windowing
afterwards makes it impossible for two views of one recording to land on
opposite sides of a held-out boundary. Windowing a library and splitting
the windows would look identical in a summary table and be worthless.

Writes a split manifest, one window manifest per split, per-window
sampling weights that equalise *track* influence, and a machine-local
path map. Runs the leakage gate on the result before anything is
written.

    uv run python scripts/dataset/build_window_experiment.py \
        --manifest data/trainset/authorized_library_manifest.json \
        --train 64 --validation 8 --evaluation 8 --seed 37 \
        --output data/trainset/exp37
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "dataset" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "training" / "src"))

from luber_dataset.splits import build_experiment_splits, leakage_report  # noqa: E402
from luber_dataset.windows import (  # noqa: E402
    DEFAULT_WINDOW_FRAMES,
    eligible_tracks,
    plan_windows,
    sampling_weights,
)
from luber_training.gates import split_leakage_gate  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--validation", type=int, default=8)
    parser.add_argument("--evaluation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    library = json.loads(args.manifest.read_text(encoding="utf-8"))

    # Eligibility is a track-level property and is applied before the
    # split, so a track too short for one window is never counted into a
    # split it cannot contribute to.
    keep, drop = eligible_tracks(library["tracks"], window_frames=args.window_frames)
    print(f"library    : {len(library['tracks'])} track(s)")
    print(f"eligible   : {len(keep)} (>= {args.window_frames} frames)")
    print(f"too short  : {len(drop)} — {sorted(round(t['duration_seconds'], 1) for t in drop)}")

    splits = build_experiment_splits(
        {**library, "tracks": keep},
        train_size=args.train,
        validation_size=args.validation,
        evaluation_size=args.evaluation,
        seed=args.seed,
    )
    payload = splits.to_dict()

    report = leakage_report(splits)
    gate = split_leakage_gate(payload)
    print(f"\nleakage report: {'PASS' if report.passed else 'FAIL'} — {report.detail}")
    print(f"leakage gate  : {'PASS' if gate.passed else 'FAIL'} — {gate.detail}")
    if not (report.passed and gate.passed):
        print("ABORT: contaminated split, nothing written", file=sys.stderr)
        return 1

    by_id = {track["track_id"]: track for track in keep}
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for key in ("train", "validation", "evaluation"):
        split = payload[key]
        tracks = [by_id[item["track_id"]] for item in split["tracks"]]
        manifest = plan_windows(
            tracks, split=split["name"], seed=args.seed, window_frames=args.window_frames
        )
        manifests[key] = manifest
        document = manifest.to_dict()
        document["sampling_weights"] = sampling_weights(manifest)
        (args.output / f"windows_{key}.json").write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"\n{split['name']:<11}: {manifest.track_count} track(s) -> "
            f"{len(manifest.windows)} window(s)"
        )
        print(f"  groups         : {split['group_distribution']}")
        print(
            f"  windows/track  : avg {document['average_windows_per_track']}, "
            f"max {document['maximum_windows_per_track']}"
        )
        print(f"  unique frames  : {manifest.unique_latent_frames}")
        print(f"  manifest digest: {manifest.digest()[:16]}...")

    # Every window in every split must be one shape, or MPS is back to
    # the Phase 36 failure. Checked across splits, not within one.
    shapes = sorted({n for m in manifests.values() for n in m.unique_latent_frames})
    print(f"\nunique latent shapes across all splits: {shapes}")
    if len(shapes) != 1:
        print("ABORT: more than one tensor shape; this is the Phase 36 OOM", file=sys.stderr)
        return 1

    used = sorted({w.track_id for m in manifests.values() for w in m.windows})
    reserve = sorted(t["track_id"] for t in library["tracks"] if t["track_id"] not in used)
    payload["window_frames"] = args.window_frames
    payload["unique_latent_shapes"] = shapes
    payload["eligible_track_count"] = len(keep)
    payload["ineligible_track_count"] = len(drop)
    payload["used_track_count"] = len(used)
    payload["reserved_track_count"] = len(reserve)
    (args.output / "splits.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "reserve.json").write_text(
        json.dumps(
            {
                "note": "authorised tracks deliberately left unseen by Phase 37",
                "reserved_track_count": len(reserve),
                "track_ids": reserve,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nused     : {len(used)} track(s)")
    print(f"reserved : {len(reserve)} track(s) held back unseen")

    paths_file = args.paths or args.manifest.with_suffix(".paths.json")
    if paths_file.is_file():
        sources = json.loads(paths_file.read_text(encoding="utf-8"))["tracks"]
        (args.output / "splits.paths.json").write_text(
            json.dumps(
                {
                    "note": "machine-local source paths; never commit",
                    "tracks": {k: sources[k] for k in used if k in sources},
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"path map : {args.output / 'splits.paths.json'} (local only)")
    print(f"\nwritten  : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
