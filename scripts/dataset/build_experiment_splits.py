#!/usr/bin/env python3
"""Phase 36 — build the controlled train / validation / evaluation split.

Reads the authorised library manifest, allocates three exclusive sets
from it, re-checks the result with the leakage gate, and writes the
split manifest plus a machine-local path map.

    uv run python scripts/dataset/build_experiment_splits.py \
        --manifest data/trainset/authorized_library_manifest.json \
        --train 24 --validation 4 --evaluation 4 --seed 36 \
        --output data/trainset/experiment_splits.json

The gate runs here rather than only at training time on purpose: a
contaminated split should never reach disk in the first place, and a
script that writes one and leaves the checking to somebody else is how
it eventually does.
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
from luber_training.gates import split_leakage_gate  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--paths", type=Path, default=None, help="defaults to <manifest>.paths.json"
    )
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--validation", type=int, default=4)
    parser.add_argument("--evaluation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    splits = build_experiment_splits(
        manifest,
        train_size=args.train,
        validation_size=args.validation,
        evaluation_size=args.evaluation,
        seed=args.seed,
    )
    payload = splits.to_dict()

    print(f"library : {splits.dataset_id} ({splits.library_content_hash[:16]}...)")
    print(f"seed    : {splits.seed}")
    for split in splits.splits:
        print(
            f"{split.name:<11}: {len(split.members):>2} track(s)  "
            f"{split.total_duration_seconds / 60:6.1f} min  "
            f"{split.group_distribution}  {split.digest()[:16]}..."
        )
    print(f"splits digest: {splits.digest()}")

    # Two checks, deliberately: the library-side report, and the gate
    # the trainer will run against the file. They must agree.
    report = leakage_report(splits)
    gate = split_leakage_gate(payload)
    print(f"\nleakage report: {'PASS' if report.passed else 'FAIL'} — {report.detail}")
    print(f"leakage gate  : {'PASS' if gate.passed else 'FAIL'} — {gate.detail}")
    if not (report.passed and gate.passed):
        print("ABORT: the split is contaminated and was not written", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nwritten : {args.output}")

    paths_file = args.paths or args.manifest.with_suffix(".paths.json")
    if paths_file.is_file():
        sources = json.loads(paths_file.read_text(encoding="utf-8"))["tracks"]
        selected = {
            member.track_id: sources[member.track_id]
            for split in splits.splits
            for member in split.members
            if member.track_id in sources
        }
        local = args.output.with_suffix(".paths.json")
        local.write_text(
            json.dumps(
                {"note": "machine-local source paths; never commit", "tracks": selected},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"path map: {local} (local only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
