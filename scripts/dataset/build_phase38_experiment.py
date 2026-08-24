#!/usr/bin/env python3
"""Phase 38 — tier the library, then window it beat-aware and arrangement-first.

Four things happen here, in an order that matters:

1. Tracks are tiered on the measured axes (HIGH_END, RHYTHM,
   ARRANGEMENT). VOCAL is never scored — see `quality_tiers`.
2. Tracks are **split** — before any windowing, so no recording can
   appear on two sides of a held-out boundary.
3. Windows are planned inside each split, with starts nudged onto
   nearby onsets so a window does not open halfway through a hit.
4. Per-window weights tilt toward the busier windows of each track,
   while every *track* still contributes the same total.

Only Phase 37's data representation changes. Rank, optimizer, precision
and batch geometry are all left exactly where Phase 37 had them.

    uv run python scripts/dataset/build_phase38_experiment.py \
        --features data/trainset/exp38/library_features.json \
        --manifest data/trainset/authorized_library_manifest.json \
        --output data/trainset/exp38
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "dataset" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "training" / "src"))

from luber_dataset.audio_features import AudioFeatures  # noqa: E402
from luber_dataset.quality_tiers import (  # noqa: E402
    classify_population,
    score_population,
    tier_summary,
)
from luber_dataset.splits import build_experiment_splits, leakage_report  # noqa: E402
from luber_dataset.windows import (  # noqa: E402
    DEFAULT_WINDOW_FRAMES,
    LATENT_FRAMES_PER_SECOND,
    Window,
    WindowManifest,
    arrangement_weighted_sampling,
    beat_aware_offsets,
    eligible_tracks,
    window_count_for,
)


def _features(payload: dict[str, Any]) -> AudioFeatures:
    known = set(AudioFeatures.__dataclass_fields__)
    return AudioFeatures(**{k: v for k, v in payload.items() if k in known})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=None)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--validation", type=int, default=8)
    parser.add_argument("--evaluation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES)
    parser.add_argument(
        "--tiers",
        default="TIER_A,TIER_B",
        help="tiers eligible for the experiment; the rest are reserved",
    )
    parser.add_argument(
        "--groups",
        default="",
        help=(
            "restrict every split to these source groups. Phase 38 is a POP/R&B "
            "experiment, and a held-out set drawn from material the training split "
            "barely contains measures the wrong thing"
        ),
    )
    parser.add_argument(
        "--emphasis",
        type=float,
        default=1.0,
        help="how much per-window weight tilts toward busier windows (0 = flat)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    library = json.loads(args.manifest.read_text(encoding="utf-8"))
    analysis = json.loads(args.features.read_text(encoding="utf-8"))
    by_id = {t["track_id"]: t for t in library["tracks"]}
    measured = {t["track_id"]: t for t in analysis["tracks"]}

    # ── 1. tier ──────────────────────────────────────────────────────
    population = [
        (track_id, _features(record["features"])) for track_id, record in sorted(measured.items())
    ]
    groups = {tid: str(by_id[tid].get("source_group", "")) for tid, _ in population}
    assignments = classify_population(population, groups=groups)
    tiers = {item.item_id: item for item in assignments}
    summary = tier_summary(assignments)
    print("── tiers over the whole library")
    print(f"  by tier : {summary['by_tier']}")
    for group, counts in sorted(summary["by_group"].items()):
        print(f"  {group:<6}: {dict(sorted(counts.items()))}")

    wanted_tiers = {name.strip() for name in args.tiers.split(",") if name.strip()}
    eligible, too_short = eligible_tracks(library["tracks"], window_frames=args.window_frames)
    candidates = [t for t in eligible if tiers[t["track_id"]].tier in wanted_tiers]
    print(f"\n  eligible by length : {len(eligible)} (too short: {len(too_short)})")
    print(f"  eligible by tier   : {len(candidates)} in {sorted(wanted_tiers)}")
    wanted_groups = {name.strip() for name in args.groups.split(",") if name.strip()}
    if wanted_groups:
        candidates = [t for t in candidates if str(t.get("source_group", "")) in wanted_groups]
        print(f"  eligible by group  : {len(candidates)} in {sorted(wanted_groups)}")

    # ── 2. split tracks, before any windowing ────────────────────────
    splits = build_experiment_splits(
        {**library, "tracks": candidates},
        train_size=args.train,
        validation_size=args.validation,
        evaluation_size=args.evaluation,
        seed=args.seed,
    )
    payload = splits.to_dict()
    report = leakage_report(splits)
    print(f"\n  leakage: {'PASS' if report.passed else 'FAIL'} — {report.detail}")
    if not report.passed:
        print("ABORT: contaminated split", file=sys.stderr)
        return 1

    # ── 3 & 4. window inside each split, beat-aware ──────────────────
    args.output.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, WindowManifest] = {}
    shapes: set[int] = set()
    snap_reasons: dict[str, int] = {}

    for key in ("train", "validation", "evaluation"):
        split = payload[key]
        windows: list[Window] = []
        window_scores: dict[str, float] = {}
        for member in split["tracks"]:
            track_id = member["track_id"]
            record = measured[track_id]
            frames = int(float(record["features"]["duration_seconds"]) * LATENT_FRAMES_PER_SECOND)
            count = window_count_for(frames, window_frames=args.window_frames)
            if count == 0:
                continue
            onset_frames = [
                round(value * LATENT_FRAMES_PER_SECOND) for value in record["onset_times"]
            ]
            placed = beat_aware_offsets(
                frames, count, onset_frames=onset_frames, window_frames=args.window_frames
            )
            # Window-level arrangement scores come from the analysis
            # grid: the nearest measured candidate to where the window
            # actually starts.
            grid = {float(k): _features(v) for k, v in record["window_features"].items()}
            grid_scores = (
                score_population([(str(k), v) for k, v in sorted(grid.items())]) if grid else {}
            )
            for index, (start, reason) in enumerate(placed):
                snap_reasons[reason] = snap_reasons.get(reason, 0) + 1
                window = Window(
                    window_id=(f"{member['audio_sha256'][:16]}-w{index}-b{start}"),
                    track_id=track_id,
                    audio_sha256=member["audio_sha256"],
                    source_group=member.get("source_group", ""),
                    window_index=index,
                    window_count=len(placed),
                    position=f"BEAT_AWARE_{index}",
                    start_frame=start,
                    end_frame=start + args.window_frames,
                    latent_frames=args.window_frames,
                    track_frames=frames,
                    experiment_seed=args.seed,
                    authorization_basis="OPERATOR_AUTHORIZED_SCOPE",
                )
                windows.append(window)
                seconds = start / LATENT_FRAMES_PER_SECOND
                nearest = min(grid, key=lambda g: abs(g - seconds)) if grid else None
                window_scores[window.window_id] = (
                    grid_scores[str(nearest)].arrangement
                    if nearest is not None and str(nearest) in grid_scores
                    else 0.5
                )

        windows.sort(key=lambda item: (item.audio_sha256, item.window_index))
        manifest = WindowManifest(
            split=split["name"],
            window_frames=args.window_frames,
            experiment_seed=args.seed,
            windows=tuple(windows),
        )
        manifests[key] = manifest
        shapes.update(manifest.unique_latent_frames)

        document = manifest.to_dict()
        document["arrangement_scores"] = {k: round(v, 4) for k, v in window_scores.items()}
        # Written at full precision. Rounded to six places, a
        # three-window track's shares sum to 1.000001, and these are
        # evidence rather than display.
        document["sampling_weights"] = arrangement_weighted_sampling(
            manifest, window_scores, emphasis=args.emphasis
        )
        document["tier_distribution"] = {}
        for member in split["tracks"]:
            tier = tiers[member["track_id"]].tier
            document["tier_distribution"][tier] = document["tier_distribution"].get(tier, 0) + 1
        (args.output / f"windows_{key}.json").write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"\n  {split['name']:<11}: {manifest.track_count} track(s) -> "
            f"{len(manifest.windows)} window(s), tiers "
            f"{dict(sorted(document['tier_distribution'].items()))}"
        )
        print(f"    groups        : {split['group_distribution']}")
        print(f"    unique frames : {manifest.unique_latent_frames}")

    print(f"\n  beat snapping : {dict(sorted(snap_reasons.items()))}")
    print(f"  unique latent shapes across all splits: {sorted(shapes)}")
    if len(shapes) != 1:
        print("ABORT: more than one tensor shape", file=sys.stderr)
        return 1

    used = sorted({w.track_id for m in manifests.values() for w in m.windows})
    reserve = sorted(t["track_id"] for t in library["tracks"] if t["track_id"] not in used)
    payload.update(
        window_frames=args.window_frames,
        unique_latent_shapes=sorted(shapes),
        window_start_policy="BEAT_AWARE",
        arrangement_emphasis=args.emphasis,
        eligible_tiers=sorted(wanted_tiers),
        eligible_groups=sorted(wanted_groups) if wanted_groups else "ALL",
        used_track_count=len(used),
        reserved_track_count=len(reserve),
    )
    (args.output / "splits.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "tiers.json").write_text(
        json.dumps(
            {
                "note": (
                    "Ranks inside this library, not judgements of the music. VOCAL is "
                    "never scored — nothing here can measure it."
                ),
                "summary": summary,
                "tier_a_percentile": 0.70,
                "tier_b_percentile": 0.40,
                "assignments": [item.to_dict() for item in assignments],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output / "reserve.json").write_text(
        json.dumps(
            {
                "note": "authorised tracks Phase 38 deliberately never saw",
                "reserved_track_count": len(reserve),
                "track_ids": reserve,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

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
            )
            + "\n",
            encoding="utf-8",
        )

    print(f"\n  used     : {len(used)} track(s)")
    print(f"  reserved : {len(reserve)} track(s)")
    print(f"  written  : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
