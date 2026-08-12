#!/usr/bin/env python3
"""Build the candidate rights-approval manifest.

Turns the graded catalog into one reviewable list so rights can be
decided in batches, rather than asking the operator to hand-write JSON
for every track.

Nothing here confers rights. Every candidate starts UNVERIFIED, and the
manifest is an input to a human decision — the operator edits the
`decision` field, and only then can ingestion accept a track.

    uv run python scripts/dataset/build_approval_manifest.py

Decisions the operator may set per candidate or per group:

    CONFIRM_TRAINING_RIGHTS   may enter a training manifest
    REFERENCE_ONLY            listening/analysis target only
    EXCLUDE                   not used at all
    (unset)                   stays UNVERIFIED, cannot train

Commercial recordings are never offered a CONFIRM default, and this
script will not write one for them.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "dataset" / "src"))

from luber_dataset.discovery import sanitize  # noqa: E402

#: Folders whose content is closest to the Phase 6 target domain
#: (modern Korean pop / K-R&B vocal). Ordering is a review priority
#: hint only; it grants nothing.
PRIORITY_HINTS: tuple[str, ...] = ("ai 음원", "발매음원", "제작 음원", "장현", "daylist")


def priority_of(relative_dir: str) -> int:
    lowered = unicodedata.normalize("NFC", relative_dir).lower()
    for index, hint in enumerate(PRIORITY_HINTS):
        if hint in lowered:
            return index
    return len(PRIORITY_HINTS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grades", type=Path, default=Path.home() / ".luber" / "candidate_grades.json"
    )
    parser.add_argument(
        "--catalog", type=Path, default=Path.home() / ".luber" / "discovery_catalog.json"
    )
    parser.add_argument(
        "--out", type=Path, default=Path.home() / ".luber" / "rights_approval_manifest.json"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "data" / "rights_approval_summary.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.grades.is_file():
        print(f"ABORT: no grades at {args.grades}", file=sys.stderr)
        return 2

    desktop = Path.home() / "Desktop"
    graded = json.loads(args.grades.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8")) if args.catalog.is_file() else []

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in graded:
        relative = sanitize(str(Path(item["absolute_path"]).parent), root=desktop)
        groups[relative].append(item)

    # A "마스터링" subfolder usually holds mastered versions of its
    # parent's tracks. Training on both is redundant, so flag the overlap
    # rather than silently shipping duplicates of the same song.
    overlaps: dict[str, list[str]] = {}
    for relative, items in groups.items():
        parent = str(Path(relative).parent)
        if parent in groups and parent != relative:
            stems = {Path(i["filename"]).stem for i in items}
            parent_stems = {Path(i["filename"]).stem for i in groups[parent]}
            shared = sorted(stems & parent_stems)
            if shared:
                overlaps[relative] = shared

    candidates: list[dict[str, Any]] = []
    for relative in sorted(groups, key=lambda r: (priority_of(r), r)):
        for index, item in enumerate(sorted(groups[relative], key=lambda i: i["filename"]), 1):
            measurement = item["measurement"]
            candidates.append(
                {
                    "candidate_id": f"{relative.replace('/', '_')}__{index:03d}",
                    "sanitized_relative_name": f"{relative}/{item['filename']}",
                    "audio_sha256": item["sha256"],
                    "origin_hypothesis": item["origin_hypothesis"],
                    "commercial_reference_hypothesis": item["commercial_reference_hypothesis"],
                    # Everything starts unverified. Only the operator
                    # changes this, and only via `decision`.
                    "training_rights_status": "UNVERIFIED",
                    "decision": None,
                    "grade": item["grade"],
                    "duration_seconds": measurement.get("duration_seconds"),
                    "sample_rate": measurement.get("sample_rate"),
                    "channels": measurement.get("channels"),
                    "bit_depth": measurement.get("bit_depth"),
                    "spectral_centroid_hz": measurement.get("spectral_centroid_hz"),
                    "high_frequency_ratio": measurement.get("high_frequency_ratio"),
                    "adjacent_lyrics": item.get("adjacent_lyrics") or [],
                    "lyrics_status": "PRESENT" if item.get("adjacent_lyrics") else "LYRICS_MISSING",
                    "review_priority": priority_of(relative),
                }
            )

    reference_only = [
        {
            "sanitized_relative_name": sanitize(
                f"{Path(e['absolute_path']).parent}/{e['filename']}", root=desktop
            ),
            "audio_sha256": e["sha256"],
            "decision": "REFERENCE_ONLY",
            "reason": (
                "commercial recording — indexed for reference, "
                "never trainable without explicit rights"
            ),
        }
        for e in catalog
        if e["commercial_reference_hypothesis"]
    ]

    group_summary: list[dict[str, Any]] = [
        {
            "directory": relative,
            "count": len(items),
            "origin_hypothesis": items[0]["origin_hypothesis"],
            "commercial_reference": items[0]["commercial_reference_hypothesis"],
            "grades": {
                g: sum(1 for i in items if i["grade"] == g)
                for g in sorted({i["grade"] for i in items})
            },
            "review_priority": priority_of(relative),
            "likely_duplicate_of_parent": overlaps.get(relative, []),
            "group_decision": None,
        }
        for relative, items in sorted(groups.items(), key=lambda kv: (priority_of(kv[0]), kv[0]))
    ]

    manifest: dict[str, Any] = {
        "manifest_version": "phase7-approval-v1",
        "instructions": (
            "Set `decision` on each candidate (or on a whole group) to one of "
            "CONFIRM_TRAINING_RIGHTS / REFERENCE_ONLY / EXCLUDE. Leaving it null keeps the "
            "track UNVERIFIED and unusable for training. Confirming means you hold "
            "commercial ML training rights to that audio, including performer and lyric "
            "rights where applicable."
        ),
        "group_summary": group_summary,
        "candidates": candidates,
        "reference_only": reference_only,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(
            {
                "manifest_version": manifest["manifest_version"],
                "candidates_awaiting_decision": len(candidates),
                "reference_only_indexed": len(reference_only),
                "groups": manifest["group_summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"candidates awaiting a rights decision: {len(candidates)}")
    print(f"reference-only (commercial, indexed) : {len(reference_only)}\n")
    print(f"{'priority':>8}  {'count':>5}  {'grades':<12} directory")
    for group in group_summary:
        dup = (
            f"   ⚠ shares {len(group['likely_duplicate_of_parent'])} filenames with parent"
            if group["likely_duplicate_of_parent"]
            else ""
        )
        print(
            f"{group['review_priority']:>8}  {group['count']:>5}  "
            f"{group['grades']!s:<12} {group['directory']}{dup}"
        )
    print(f"\napproval manifest (personal paths): {args.out}")
    print(f"sanitized summary                 : {args.summary}")
    print("\nAll candidates are UNVERIFIED. Nothing may be trained on until you decide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
