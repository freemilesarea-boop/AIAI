#!/usr/bin/env python3
"""Phase 38 — build a listening set organised around four named axes.

The operator is asked four separate questions about each generation —
HIGH-END, RHYTHM, ARRANGEMENT, VOCAL — so the prompts are chosen to give
each question something to bite on rather than being eight variations of
the same request.

Prompts come from the evaluation split and from nowhere else. Seeds are
read out of each track's audio digest, so the set reproduces without a
random number anywhere.

    uv run python scripts/dataset/build_axis_eval_spec.py \
        --splits data/trainset/exp38/splits.json --out spec.json ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: What each axis asks the ear to attend to. The caption is the same
#: operator group label the model trained on, extended with the words
#: that make one axis salient — so the comparison stays a comparison of
#: models, not of prompts.
AXIS_PROMPTS: dict[str, str] = {
    "HIGH_END": (
        "modern pop R&B with bright open top end, crisp hi-hats and airy cymbals, "
        "detailed high frequency sparkle, clear polished mix"
    ),
    "RHYTHM": (
        "modern pop R&B with a tight locked groove, punchy drums and a steady kick "
        "and snare pattern, deep bass locked to the beat, unwavering tempo"
    ),
    "ARRANGEMENT": (
        "modern pop R&B with a full layered arrangement, warm synth pads, clean "
        "electric guitar, counter-melodies and stacked backing textures"
    ),
    "VOCAL": (
        "modern pop R&B with a clear expressive lead vocal front and centre, natural "
        "vocal timbre, layered background harmonies"
    ),
}

#: Seeds are 31-bit so a signed round-trip cannot change them.
SEED_MASK = 0x7FFFFFFF


def seed_for(digest: str, index: int) -> int:
    return int(digest[index * 8 : (index + 1) * 8], 16) & SEED_MASK


def build_items(splits: dict[str, Any], *, duration: float) -> list[dict[str, Any]]:
    """One item per axis per evaluation track, capped at one seed each."""
    tracks = (splits.get("evaluation") or {}).get("tracks") or []
    items: list[dict[str, Any]] = []
    for position, track in enumerate(tracks):
        axis = list(AXIS_PROMPTS)[position % len(AXIS_PROMPTS)]
        items.append(
            {
                "id": f"{axis.lower()}_{position + 1:02d}",
                "axis": axis,
                "caption": AXIS_PROMPTS[axis],
                "lyrics": "",
                "instrumental": False,
                "seed": seed_for(str(track["audio_sha256"]), 0),
                "duration": duration,
                "source_track": track["track_id"],
                "source_group": track.get("source_group", ""),
            }
        )
    return items


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument(
        "--side",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a side to generate; an empty PATH means the untouched base model",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    sides = []
    for entry in args.side:
        name, _, path = entry.partition("=")
        sides.append({"name": name, "lora_path": path or None})

    spec = {
        "project_root": args.project_root,
        "model_variant": "acestep-v15-turbo",
        "device": "mps",
        "duration": args.duration,
        "inference_steps": args.inference_steps,
        "shift": 3.0,
        "output_root": args.output_root,
        "manifest": args.manifest,
        "sides": sides,
        "items": build_items(splits, duration=args.duration),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{len(spec['items'])} item(s) x {len(sides)} side(s) = "
        f"{len(spec['items']) * len(sides)} generations"
    )
    for item in spec["items"]:
        print(
            f"  {item['id']:<16} axis={item['axis']:<12} seed={item['seed']:<11} "
            f"from {item['source_track']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
