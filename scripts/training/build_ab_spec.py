#!/usr/bin/env python3
"""Phase 36 — derive a deterministic A/B listening set from the evaluation split.

Prompts and seeds come from the evaluation tracks and from nowhere else:
the split the optimizer never saw is the only place an honest comparison
can draw its questions from.

Each evaluation track contributes two pairs. The caption is the
operator's own group label — the same text the training conditioning
used, so the two sides are asked the question the run was trained on —
and the seeds are read out of the track's audio digest, which makes the
set reproducible without a random number anywhere.

    uv run python scripts/training/build_ab_spec.py \
        --splits data/trainset/experiment_splits.json \
        --lora <adapter dir> --project-root ~/ace-step-1.5 \
        --base-dir <out>/base --lora-dir <out>/lora \
        --manifest <out>/manifest.json --out <out>/spec.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: Two per track keeps the set inside the 5-10 pairs the phase asks for
#: while giving each track more than a single draw.
SEEDS_PER_TRACK = 2


def seeds_for(digest: str, count: int = SEEDS_PER_TRACK) -> list[int]:
    """Seeds read out of the audio digest, so the set is reproducible.

    Masked to 31 bits because a seed is passed through as a signed
    integer and a value that wraps would silently stop being the seed
    the manifest claims.
    """
    return [int(digest[index * 8 : (index + 1) * 8], 16) & 0x7FFFFFFF for index in range(count)]


def build_spec(splits: dict[str, Any], **fields: Any) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for track in (splits.get("evaluation") or {}).get("tracks") or []:
        digest = str(track["audio_sha256"])
        group = str(track.get("source_group", ""))
        for index, seed in enumerate(seeds_for(digest)):
            pairs.append(
                {
                    "id": f"{track['track_id']}-{index}",
                    "caption": group.lower(),
                    "seed": seed,
                    "source_group": group,
                }
            )
    return {"pairs": pairs, **fields}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-variant", default="acestep-v15-turbo")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    spec = build_spec(
        splits,
        project_root=args.project_root,
        model_variant=args.model_variant,
        device=args.device,
        duration=args.duration,
        inference_steps=args.inference_steps,
        lora_path=args.lora,
        base_dir=args.base_dir,
        lora_dir=args.lora_dir,
        manifest=args.manifest,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(spec['pairs'])} A/B pair(s) from the evaluation split -> {args.out}")
    for pair in spec["pairs"]:
        print(f"  {pair['id']}  caption={pair['caption']!r}  seed={pair['seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
