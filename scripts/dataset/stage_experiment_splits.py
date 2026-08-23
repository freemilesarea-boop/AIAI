#!/usr/bin/env python3
"""Phase 36 — stage the split tracks out of the read-only authorised root.

One directory per split, so the three sets are separated on disk and not
merely by a field in a manifest. A training run pointed at
``staging/train`` cannot reach an evaluation track by accident, which is
a stronger guarantee than remembering to filter.

Copies only. Nothing in the authorised root is renamed, moved, edited or
transcoded, and every source file is re-hashed afterwards to show it.

    uv run python scripts/dataset/stage_experiment_splits.py \
        --splits data/trainset/experiment_splits.json \
        --output data/trainset/experiment_staging
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "dataset" / "src"))

from luber_dataset.factory.provenance import (  # noqa: E402
    RightsStatus,
    SourceType,
    TrainingPermission,
)

OPERATOR_AUTHORIZATION_NOTES = (
    "The operator explicitly authorised this source directory for LUBER model training. "
    "That authorisation is the entire evidence: no contract, licence, publisher clearance "
    "or performer agreement was produced or verified, and none is claimed here."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_for(*, group: str, scope: str, split: str) -> dict[str, Any]:
    """Operator metadata for one staged track.

    ``rights_status`` is OPERATOR_AUTHORIZED, never USER_OWNED or
    VERIFIED: an authorisation is what exists, and an ownership document
    is not. ``genre`` carries the operator's own folder label, which is
    a claim about music a folder name can support — and which is kept
    well away from the rights fields, which it cannot.
    """
    return {
        "source": f"operator-authorised directory, group {group!r}",
        "source_type": SourceType.UNKNOWN.value,
        "rights_status": RightsStatus.OPERATOR_AUTHORIZED.value,
        "commercial_training_allowed": TrainingPermission.TRUE.value.lower(),
        "genre": group.lower(),
        "notes": (
            f"{OPERATOR_AUTHORIZATION_NOTES} Authorised scope: {scope}. Phase 36 split: {split}."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=None, help="defaults to <splits>.paths.json")
    parser.add_argument("--scope", default="", help="authorised scope, for the sidecars")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.splits.read_text(encoding="utf-8"))
    paths_file = args.paths or args.splits.with_suffix(".paths.json")
    sources: dict[str, str] = json.loads(paths_file.read_text(encoding="utf-8"))["tracks"]

    scope = args.scope
    if not scope:
        library_paths = Path("data/trainset/authorized_library_manifest.paths.json")
        if library_paths.is_file():
            scope = json.loads(library_paths.read_text(encoding="utf-8")).get(
                "authorization_scope", ""
            )

    unchanged = True
    staged_counts: dict[str, int] = {}
    for key in ("train", "validation", "evaluation"):
        split = payload[key]
        destination = args.output / key
        destination.mkdir(parents=True, exist_ok=True)
        print(f"\n── {split['name']}  ({split['track_count']} track(s)) → {destination}")

        for track in split["tracks"]:
            track_id = track["track_id"]
            origin = Path(sources[track_id])
            if not origin.is_file():
                print(f"ABORT: {track_id} is no longer at its recorded path", file=sys.stderr)
                return 2

            before = sha256_file(origin)
            if before != track["audio_sha256"]:
                print(
                    f"ABORT: {track_id} no longer hashes to the split manifest; the "
                    "source changed and this is not the split that was gated",
                    file=sys.stderr,
                )
                return 2

            target = destination / f"{track_id}.wav"
            if not (target.is_file() and sha256_file(target) == before):
                shutil.copy2(origin, target)
            if sha256_file(target) != before:
                print(f"ABORT: staged copy of {track_id} does not match", file=sys.stderr)
                return 2

            target.with_suffix(".json").write_text(
                json.dumps(
                    sidecar_for(group=track["source_group"], scope=scope, split=split["name"]),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            if sha256_file(origin) != before:
                unchanged = False
                print(f"   !! source changed during staging: {track_id}")

        staged_counts[key] = len(list(destination.glob("*.wav")))
        print(f"   staged {staged_counts[key]} file(s)")

    record = {
        "splits_digest": payload["splits_digest"],
        "staging_root": str(args.output),
        "staged_counts": staged_counts,
        "source_unchanged": unchanged,
    }
    (args.output / "staging_report.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nsources : {'unchanged' if unchanged else 'CHANGED — investigate before training'}")
    return 0 if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
