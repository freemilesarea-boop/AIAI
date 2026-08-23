#!/usr/bin/env python3
"""Phase 35B — stage a bounded pilot subset out of an authorised library.

The authorised source root is read-only material. Nothing here renames,
moves, edits or transcodes anything in it: the subset is *copied* to a
LUBER-managed staging directory, and every source file is re-hashed
afterwards to show it came out the way it went in.

Staging exists because the dataset factory reads operator metadata from
a sidecar beside each audio file, and writing sidecars into the source
root would be exactly the in-place modification the phase forbids. So
the copies get the sidecars, and the originals are left alone.

    uv run python scripts/dataset/stage_pilot_subset.py \
        --manifest data/trainset/authorized_library_manifest.json \
        --size 4 --output data/trainset/pilot_staging

The subset is chosen by :func:`luber_dataset.select_pilot_subset`, which
is a pure function of the manifest — so re-running this reproduces the
same four tracks rather than whichever four the filesystem offered
first.
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

from luber_dataset import select_pilot_subset  # noqa: E402
from luber_dataset.factory.provenance import (  # noqa: E402
    RightsStatus,
    SourceType,
    TrainingPermission,
)

#: Repeated verbatim from the ingestion script so a reader of a staged
#: sidecar sees the same sentence as a reader of the library manifest.
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


def sidecar_for(*, group: str, scope: str) -> dict[str, Any]:
    """Operator metadata for one staged track.

    ``rights_status`` is OPERATOR_AUTHORIZED rather than USER_OWNED or
    VERIFIED: the operator authorised the scope, and nobody produced an
    ownership document, so the record says the weaker of the two things
    rather than the more useful one.

    ``genre`` carries the operator's own folder label. That is a claim
    about music, which a folder name can support; it is deliberately
    not allowed anywhere near the rights fields, which it cannot.
    """
    return {
        "source": f"operator-authorised directory, group {group!r}",
        "source_type": SourceType.UNKNOWN.value,
        "rights_status": RightsStatus.OPERATOR_AUTHORIZED.value,
        "commercial_training_allowed": TrainingPermission.TRUE.value.lower(),
        "genre": group.lower(),
        "notes": f"{OPERATOR_AUTHORIZATION_NOTES} Authorised scope: {scope}",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--paths",
        type=Path,
        default=None,
        help="source path map; defaults to <manifest>.paths.json",
    )
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--groups", default="", help="comma-separated source groups to draw from")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths_file = args.paths or args.manifest.with_suffix(".paths.json")
    path_map = json.loads(paths_file.read_text(encoding="utf-8"))
    sources: dict[str, str] = path_map["tracks"]
    scope = str(path_map.get("authorization_scope", ""))

    groups = tuple(g.strip() for g in args.groups.split(",") if g.strip()) or None
    subset = select_pilot_subset(manifest, size=args.size, groups=groups)

    print(f"library  : {subset.dataset_id} ({subset.library_content_hash[:16]}…)")
    print(f"subset   : {len(subset.members)} track(s), digest {subset.digest()[:16]}…")
    print(f"groups   : {subset.group_distribution}")
    print(f"duration : {subset.total_duration_seconds / 60:.1f} min\n")

    args.output.mkdir(parents=True, exist_ok=True)
    unchanged = True
    for member in subset.members:
        origin = Path(sources[member.track_id])
        target = args.output / f"{member.track_id}.wav"
        if not origin.is_file():
            print(f"ABORT: {member.track_id} is no longer at its recorded path", file=sys.stderr)
            return 2

        before = sha256_file(origin)
        if before != member.audio_sha256:
            print(
                f"ABORT: {member.track_id} no longer hashes to the manifest digest; "
                "the source changed and the subset is not the one that was locked",
                file=sys.stderr,
            )
            return 2

        if not (target.is_file() and sha256_file(target) == member.audio_sha256):
            shutil.copy2(origin, target)
        staged = sha256_file(target)
        if staged != member.audio_sha256:
            print(f"ABORT: staged copy of {member.track_id} does not match", file=sys.stderr)
            return 2

        sidecar = sidecar_for(group=member.source_group, scope=scope)
        target.with_suffix(".json").write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        # Re-read the source after the copy. Reading a file should not
        # change it, and the phase asked for evidence rather than the
        # assumption.
        after = sha256_file(origin)
        if after != before:
            unchanged = False
            print(f"   !! source changed during staging: {member.track_id}")
        print(f"── {member.track_id}  {member.source_group}  {member.duration_seconds:.1f}s  ok")

    record = subset.to_dict()
    record["staging_dir"] = str(args.output)
    record["source_unchanged"] = unchanged
    (args.output / "pilot_subset.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nstaged   : {args.output}")
    print(f"sources  : {'unchanged' if unchanged else 'CHANGED — investigate before training'}")
    return 0 if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
