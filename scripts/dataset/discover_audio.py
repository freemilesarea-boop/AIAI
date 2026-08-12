#!/usr/bin/env python3
"""Read-only discovery of candidate training audio.

Walks a directory the operator has authorized, catalogs every audio
file, and forms an *origin hypothesis* from path context.

Non-destructive by construction: files are opened for reading only.
Nothing is written, renamed, moved, converted, tagged, or deleted under
the scanned root, and no audio is copied into the repository.

A hypothesis is not a licence. Nothing this script produces confers
training rights — it produces a list for the operator to decide on.

    uv run python scripts/dataset/discover_audio.py --root ~/Desktop

The full catalog contains absolute personal paths and is written
outside the repository by default. Only the sanitized summary is safe
to commit.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "dataset" / "src"))

from luber_dataset.discovery import sanitize, scan, summarize  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory to scan")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path.home() / ".luber" / "discovery_catalog.json",
        help="Where the full catalog (with absolute paths) is written",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "data" / "discovery_summary.json",
        help="Sanitized summary, safe to inspect and share",
    )
    parser.add_argument("--no-hash", action="store_true", help="Skip SHA256 (faster)")
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ABORT: {root} is not a directory", file=sys.stderr)
        return 2

    print(f"scanning {root} (read-only)…")
    files = scan(
        root,
        # Never propose our own generated output as training material.
        exclude_roots=(REPO_ROOT / "data",),
        hash_files=not args.no_hash,
        max_files=args.max_files,
    )
    stats = summarize(files)

    print(f"\nfound {stats['total_files']} audio files")
    print(f"  unique hashes : {stats['unique_hashes']}")
    print(f"  duplicates    : {stats['duplicate_files']}")
    print(f"  measured hours: {stats['measured_hours']} (WAV headers only)")
    print(f"  total size    : {stats['total_bytes'] / 1024**3:.1f} GB")
    print(f"  by extension  : {stats['by_extension']}")
    print(f"  by hypothesis : {stats['by_origin_hypothesis']}")
    print(f"  with adjacent lyrics/metadata: {stats['with_adjacent_lyrics']}")

    counts: Counter[str] = Counter()
    hypotheses: dict[str, Counter[str]] = defaultdict(Counter)
    extensions: dict[str, Counter[str]] = defaultdict(Counter)
    for item in files:
        relative = sanitize(str(Path(item.absolute_path).parent), root=root)
        counts[relative] += 1
        hypotheses[relative][item.origin_hypothesis] += 1
        extensions[relative][item.extension] += 1

    print(f"\ndirectories containing audio: {len(counts)}")
    ranked = counts.most_common()
    for relative, count in ranked[:30]:
        print(f"  {count:>4}  {relative}   {dict(hypotheses[relative])}")

    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    args.catalog.write_text(
        json.dumps([f.to_dict() for f in files], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(
            {
                "scanned_root": sanitize(str(root)),
                "summary": stats,
                "directories": {
                    rel: {
                        "count": count,
                        "hypotheses": dict(hypotheses[rel]),
                        "extensions": dict(extensions[rel]),
                    }
                    for rel, count in ranked
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nfull catalog (personal paths): {args.catalog}")
    print(f"sanitized summary            : {args.summary}")
    print("\nNo file under the scanned root was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
